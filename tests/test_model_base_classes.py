"""Request types must be strict, response types must not.

`_response.py` refuses to parse a response whose model is not a `BaseModel`
subclass, so a response declared as `StrictBase` raises TypeError on the first
real call -- a mocked transport never reaches that code. Responses also have to
tolerate extra fields, or adding one server-side breaks every older client.
"""

from __future__ import annotations

import inspect

import pydantic
import pytest

import tinker.types as types
from tinker._models import BaseModel, StrictBase


def _models(suffix: str) -> list[type[pydantic.BaseModel]]:
    return [
        obj
        for name in dir(types)
        if name.endswith(suffix)
        if inspect.isclass(obj := getattr(types, name)) and issubclass(obj, pydantic.BaseModel)
    ]


@pytest.mark.parametrize("model", _models("Request"), ids=lambda m: m.__name__)
def test_request_types_are_strict(model: type[pydantic.BaseModel]) -> None:
    assert issubclass(model, StrictBase), (
        f"{model.__name__} must subclass StrictBase so unknown fields are rejected early."
    )


@pytest.mark.parametrize("model", _models("Response"), ids=lambda m: m.__name__)
def test_response_types_accept_extra_fields(model: type[pydantic.BaseModel]) -> None:
    assert issubclass(model, BaseModel), (
        f"{model.__name__} must subclass BaseModel: _response.py cannot parse anything else, "
        "and responses must tolerate fields added by a newer server."
    )
