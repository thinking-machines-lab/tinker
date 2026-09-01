from __future__ import annotations

from datetime import datetime

from .._models import BaseModel

__all__ = ["AuditLogEntry"]


class AuditLogEntry(BaseModel):
    """A single entry in the audit log."""

    timestamp: datetime
    """When the event occurred."""

    event: str
    """`<resource>_<action>`, e.g. `checkpoint_read`, `project_grant_set`."""

    event_details: dict[str, object] = {}
    """Who did it, what it was about, and whatever else this event records."""

    model_id: str | None = None
    """Deprecated: read `event_details`. Set on checkpoint events only."""

    tinker_path: str | None = None
    """Deprecated: read `event_details`. Set on checkpoint events only."""

    purpose: str | None = None
    """Deprecated: read `event_details`. Set on checkpoint reads only."""
