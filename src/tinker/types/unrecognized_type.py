from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["UnrecognizedType"]


class UnrecognizedType(BaseModel):
    """An object, in a union the API tags by `type`, that this SDK version cannot parse."""

    type: Literal["unrecognized"] = "unrecognized"

    raw_object: dict[str, object]
    """The object as the API reported it"""
