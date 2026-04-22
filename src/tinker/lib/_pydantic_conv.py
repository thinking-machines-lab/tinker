"""Conversion from internal Pydantic types to public dataclass types.

Each migrated type has a converter registered here. The JSON deserialization
path in api_future_impl uses deserialize_json_response() to look up and
apply the appropriate converter.

Non-migrated types fall through to Pydantic's model_validate directly.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import pydantic

from tinker.types._pydantic_types.sample_response import SampleResponse as _PydanticSampleResponse
from tinker.types.sample_response import SampleResponse
from tinker.types.sampled_sequence import SampledSequence

# Registry: public dataclass type -> (pydantic_cls, converter_fn)
_CONVERTERS: dict[type, tuple[type, Callable[[Any], Any]]] = {}


def _convert_sample_response(pydantic_obj: _PydanticSampleResponse) -> SampleResponse:
    return SampleResponse(
        sequences=[
            SampledSequence(
                stop_reason=s.stop_reason,
                _tokens_list=s.tokens,
                _logprobs_list=s.logprobs,
            )
            for s in pydantic_obj.sequences
        ],
        _prompt_logprobs_list=pydantic_obj.prompt_logprobs,
        _topk_prompt_logprobs_list=pydantic_obj.topk_prompt_logprobs,
    )


_CONVERTERS[SampleResponse] = (_PydanticSampleResponse, _convert_sample_response)


def deserialize_json_response(result_dict: dict[str, Any], model_cls: type) -> Any:
    """Deserialize a JSON response dict into the expected type.

    Migrated types are deserialized via their Pydantic counterpart, then
    converted to the public dataclass. Non-migrated types use Pydantic directly.
    """
    entry = _CONVERTERS.get(model_cls)
    if entry is not None:
        pydantic_cls, converter = entry
        pydantic_obj = pydantic_cls.model_validate(result_dict)
        return converter(pydantic_obj)
    if inspect.isclass(model_cls) and issubclass(model_cls, pydantic.BaseModel):
        return model_cls.model_validate(result_dict)
    return result_dict
