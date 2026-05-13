from __future__ import annotations

from datetime import datetime

from .._models import BaseModel

__all__ = ["AuditLogEntry"]


class AuditLogEntry(BaseModel):
    """A single entry in the audit log."""

    timestamp: datetime
    """When the event occurred."""

    event: str
    """The event type identifier."""

    model_id: str | None = None
    """The model ID associated with the event, if any."""

    tinker_path: str | None = None
    """The tinker path associated with the event, if any."""

    purpose: str | None = None
    """The purpose of the event, if any."""
