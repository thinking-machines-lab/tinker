from .._models import BaseModel
from .checkpoint import Checkpoint
from .checkpoint_search_cursor import CheckpointSearchCursor

__all__ = ["CheckpointSearchResponse"]


class CheckpointSearchResponse(BaseModel):
    """One page of search results, newest first."""

    checkpoints: list[Checkpoint]

    next_cursor: CheckpointSearchCursor | None = None
    """Pass as the next request's `cursor`; None once the results are exhausted"""
