from __future__ import annotations

from typing import Any

from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase

__all__ = ["CreateSessionRequest"]


class CreateSessionRequest(StrictBase):
    tags: list[str]
    user_metadata: dict[str, Any] | None
    sdk_version: str
    project_id: str | None = None
    display_name: str | None = None
    """Optional human-readable display name for the session. Does not need to be unique. Only used for display and search purposes."""

    type: Literal["create_session"] = "create_session"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
