from pydantic import AwareDatetime

from .._models import BaseModel

__all__ = ["SessionSearchCursor"]


class SessionSearchCursor(BaseModel):
    """Where the next page starts; pass a response's `next_cursor` back unchanged."""

    last_created_at: AwareDatetime

    last_session_id: str
