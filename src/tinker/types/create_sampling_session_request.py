from __future__ import annotations

from typing import Optional

from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase

__all__ = ["CreateSamplingSessionRequest"]


class CreateSamplingSessionRequest(StrictBase):
    session_id: str
    """The session ID to create the sampling session within"""

    sampling_session_seq_id: int
    """Sequence ID for the sampling session within the session"""

    base_model: Optional[str] = None
    """Optional base model name to sample from.

    Is inferred from model_path, if provided. If sampling against a base model, this
    is required.
    """

    model_path: Optional[str] = None
    """Optional tinker:// path to your model weights or LoRA weights.

    If not provided, samples against the base model.
    """

    type: Literal["create_sampling_session"] = "create_sampling_session"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
