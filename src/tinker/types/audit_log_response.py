from __future__ import annotations

from .._models import BaseModel
from .audit_log_entry import AuditLogEntry

__all__ = ["AuditLogResponse"]


class AuditLogResponse(BaseModel):
    """Audit log response containing a list of entries."""

    entries: list[AuditLogEntry]
    """List of audit log entries, sorted by timestamp."""
