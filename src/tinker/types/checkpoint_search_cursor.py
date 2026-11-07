from pydantic import AwareDatetime

from .._models import BaseModel

__all__ = ["CheckpointSearchCursor"]


class CheckpointSearchCursor(BaseModel):
    """Where the next page starts; pass a response's `next_cursor` back unchanged."""

    last_created_at: AwareDatetime

    last_checkpoint_uuid: str
