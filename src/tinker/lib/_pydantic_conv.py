"""Deserialization of JSON responses into public SDK types.

Proto-capable response types (``SampleResponse``, ``ForwardBackwardOutput``)
arrive as proto bytes and decode via ``proto/response_conv.py``; they never
pass through here. Every remaining response type is a Pydantic model and
validates directly.
"""

from __future__ import annotations

import inspect
from typing import Any

import pydantic


def deserialize_json_response(result_dict: dict[str, Any], model_cls: type) -> Any:
    """Deserialize a JSON response dict into the expected type."""
    if inspect.isclass(model_cls) and issubclass(model_cls, pydantic.BaseModel):
        return model_cls.model_validate(result_dict)
    return result_dict
