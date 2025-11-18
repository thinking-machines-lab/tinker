from .._models import BaseModel
from .checkpoint import Checkpoint
from .cursor import Cursor

__all__ = ["CheckpointsListResponse"]


class CheckpointsListResponse(BaseModel):
    checkpoints: list[Checkpoint]
    """List of available model checkpoints for the model"""

    cursor: Cursor | None = None
    """Pagination cursor information (None for unpaginated responses)"""
