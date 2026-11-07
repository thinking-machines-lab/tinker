from pydantic import Field

from .._models import StrictBase
from .checkpoint import CheckpointType
from .checkpoint_search_cursor import CheckpointSearchCursor
from .search_owner import OwnSearchOwner, SearchOwner

__all__ = ["CheckpointSearchRequest"]


class CheckpointSearchRequest(StrictBase):
    """A checkpoint search. Results are restricted to projects the caller may list."""

    project_ids: list[str] | None = None
    """Projects to search; None searches every listable project. Naming a
    project the caller cannot list is an error"""

    user_metadata: dict[str, str] = {}
    """Exact user_metadata key/value matches; every pair must hold"""

    checkpoint_type: CheckpointType | None = None
    """Checkpoint type to match; None matches every type"""

    owner: SearchOwner = OwnSearchOwner()
    """Whose checkpoints to search; defaults to the caller's own"""

    base_models: list[str] | None = None
    """Match any listed base model; None matches every base model. An empty
    list matches no checkpoints"""

    limit: int = Field(default=50, ge=1, le=200)
    """Page size, 1 to 200"""

    cursor: CheckpointSearchCursor | None = None
    """`next_cursor` of the previous page; None for the first page"""
