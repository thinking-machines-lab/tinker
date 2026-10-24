from pydantic import Field

from .._models import StrictBase
from .search_owner import OwnSearchOwner, SearchOwner
from .session_search_cursor import SessionSearchCursor
from .session_status_query import SessionStatusQuery

__all__ = ["SessionSearchRequest"]


class SessionSearchRequest(StrictBase):
    """A session search. Results are restricted to projects the caller may list."""

    session_id: str | None = None
    """Exact session ID to match; None searches every session. With
    `display_name`, a session matching either one matches"""

    project_ids: list[str] | None = None
    """Projects to search; None searches every listable project. Naming a
    project the caller cannot list is an error"""

    user_metadata: dict[str, str] = {}
    """Exact user_metadata key/value matches; every pair must hold"""

    display_name: str | None = Field(default=None, min_length=1)
    """Case-insensitive substring match on the display name; with several
    words, each must appear somewhere in the name. With `session_id`, a session
    matching either one matches"""

    owner: SearchOwner = OwnSearchOwner()
    """Whose sessions to search; defaults to the caller's own"""

    statuses: list[SessionStatusQuery] | None = Field(default=None, min_length=1)
    """Match any listed status; None matches every status"""

    limit: int = Field(default=50, ge=1, le=200)
    """Page size, 1 to 200"""

    cursor: SessionSearchCursor | None = None
    """`next_cursor` of the previous page; None for the first page"""
