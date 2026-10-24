from typing import Union

from pydantic import Field
from typing_extensions import Literal, TypeAlias

from .._models import StrictBase

__all__ = ["OwnSearchOwner", "AccessibleSearchOwner", "UserSearchOwner", "SearchOwner"]


class OwnSearchOwner(StrictBase):
    """The caller's own sessions."""

    type: Literal["own"] = "own"


class AccessibleSearchOwner(StrictBase):
    """Every owner's sessions."""

    type: Literal["accessible"] = "accessible"


class UserSearchOwner(StrictBase):
    """The listed users' sessions."""

    type: Literal["user"] = "user"

    owner_urns: list[str] = Field(min_length=1, max_length=10)
    """Owner URNs to match (1 to 10)"""


SearchOwner: TypeAlias = Union[OwnSearchOwner, AccessibleSearchOwner, UserSearchOwner]
