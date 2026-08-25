from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Self, Union

import numpy as np

from .loss_fn_inputs import LossFnInputs
from .provenance_spans import PromptProvenanceSpan, SampledProvenanceSpan
from .tensor_data import TensorData

try:
    import torch  # type: ignore[import-not-found]

    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

if TYPE_CHECKING:
    import torch  # noqa: TC004

from ._pydantic_types.model_input import ModelInput

__all__ = ["Datum"]

# Field-name → wire dtype for raw Python lists with no inferable numpy dtype.
_KEY_TO_TYPE = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
}

_SPARSE_ELIGIBLE_KEYS = {"target_tokens", "weights"}


ProvenanceSpans = List[Union[PromptProvenanceSpan, SampledProvenanceSpan]]


@dataclass(frozen=True)
class Datum:
    model_input: ModelInput
    loss_fn_inputs: LossFnInputs = field(default_factory=dict)

    model_input_spans: Optional[ProvenanceSpans] = None
    """Provenance of the input tokens as assembled: consecutive runs tiling
    ``model_input`` (lengths sum to its token length; position comes from
    order). Attach via ``with_provenance``."""

    loss_fn_input_spans: Optional[ProvenanceSpans] = None
    """Attribution of the loss rows: consecutive runs tiling
    ``loss_fn_inputs["target_tokens"]`` (lengths sum to its length). Attach
    via ``with_provenance``."""

    def __post_init__(self) -> None:
        coerced: Dict[str, TensorData] = {}
        for key, value in self.loss_fn_inputs.items():
            coerced[key] = _maybe_convert_array(key, value)
        object.__setattr__(self, "loss_fn_inputs", coerced)
        _check_span_types("model_input_spans", self.model_input_spans)
        _check_span_types("loss_fn_input_spans", self.loss_fn_input_spans)
        assert (self.model_input_spans is not None) == (self.loss_fn_input_spans is not None), (
            "provenance comes as a pair: set both span lists or neither"
        )

    def with_provenance(
        self,
        model_input: ProvenanceSpans,
        loss_fn_inputs: ProvenanceSpans,
    ) -> Self:
        """Return a copy carrying provenance for both span-annotated fields.

        Each keyword names the Datum field whose positions its runs tile —
        ``model_input`` spans tile the input tokens, ``loss_fn_inputs``
        spans tile ``loss_fn_inputs["target_tokens"]``. Tiling is validated
        here, so a miscounted partition fails at the call site rather than
        as a request error.
        """
        _check_tiling("model_input", model_input, self.model_input.length)
        target_tokens = self.loss_fn_inputs.get("target_tokens")
        if (
            target_tokens is not None
            and target_tokens.shape is not None
            and len(target_tokens.shape) == 1
        ):
            _check_tiling("loss_fn_inputs", loss_fn_inputs, target_tokens.shape[0])
        return replace(
            self,
            model_input_spans=model_input,
            loss_fn_input_spans=loss_fn_inputs,
        )


def _check_span_types(field_name: str, spans: Optional[ProvenanceSpans]) -> None:
    if spans is None:
        return
    for span in spans:
        if not isinstance(span, (PromptProvenanceSpan, SampledProvenanceSpan)):
            raise TypeError(
                f"{field_name} entries must be PromptProvenanceSpan or "
                f"SampledProvenanceSpan, got {type(span).__name__}"
            )


def _check_tiling(kwarg: str, spans: ProvenanceSpans, expected_length: int) -> None:
    total = sum(span.length for span in spans)
    if total != expected_length:
        raise ValueError(
            f"with_provenance {kwarg} spans cover {total} positions but the "
            f"field has {expected_length}; the runs must tile it exactly"
        )


def _maybe_convert_array(
    key: str, value: Union[TensorData, "torch.Tensor", np.ndarray, list]
) -> TensorData:
    if isinstance(value, TensorData):
        return value
    if _HAVE_TORCH and isinstance(value, torch.Tensor):
        if key in _SPARSE_ELIGIBLE_KEYS and value.ndim == 2:
            return TensorData.from_torch_sparse(value)
        return TensorData.from_torch(value)
    if isinstance(value, np.ndarray):
        return TensorData.from_numpy(value)
    if isinstance(value, list):
        try:
            array = np.asarray(value)
        except ValueError as exc:
            if any(isinstance(item, list) for item in value):
                raise ValueError(
                    f"{key} must be a rectangular numeric array; ragged nested lists are not supported"
                ) from exc
            raise
        if array.dtype.kind in ("f", "i", "u"):
            target_dtype = _KEY_TO_TYPE[key]
            if target_dtype == "int64":
                array = array.astype(np.int64)
            else:
                array = array.astype(np.float32)
            return TensorData.from_numpy(array)
        if any(isinstance(item, list) for item in value):
            raise ValueError(
                f"{key} must be a rectangular numeric array; ragged nested lists are not supported"
            )
        return TensorData(data=value, dtype=_KEY_TO_TYPE[key], shape=[len(value)])
    raise TypeError(f"Unsupported loss_fn_inputs value for {key!r}: {type(value).__name__}")
