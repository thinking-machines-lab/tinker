from __future__ import annotations

from .._models import BaseModel

__all__ = ["AuthTokenResponse"]


class AuthTokenResponse(BaseModel):
    jwt: str
