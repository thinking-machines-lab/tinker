from .._models import BaseModel
from .session_search_cursor import SessionSearchCursor
from .session_summary import SessionSummary

__all__ = ["SessionSearchResponse"]


class SessionSearchResponse(BaseModel):
    """One page of search results, newest first."""

    sessions: list[SessionSummary]

    next_cursor: SessionSearchCursor | None = None
    """Pass as the next request's `cursor`; None once the results are exhausted"""
