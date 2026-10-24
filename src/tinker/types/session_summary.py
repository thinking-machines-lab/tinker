from datetime import datetime

from .._models import BaseModel
from .session_status import SessionStatus

__all__ = ["SessionSummary"]


class SessionSummary(BaseModel):
    session_id: str

    created_at: datetime

    owner: str

    project_id: str

    status: SessionStatus | None = None
    """None when the session neither finished nor is known to be running"""

    checkpoints_count: int

    display_name: str | None = None

    user_metadata: dict[str, str] | None = None

    checkpoint_storage_bytes: int | None = None
