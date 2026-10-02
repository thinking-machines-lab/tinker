from .._models import BaseModel

__all__ = [
    "CurrentCheckpointStorageUsageItem",
    "CurrentCheckpointStorageUsageResponse",
]


class CurrentCheckpointStorageUsageItem(BaseModel):
    """HTTP/JSON response item returned by the API server to SDK and CLI consumers.

    Contains current active checkpoint storage attributed through its session.
    """

    project_id: str | None = None
    """Project currently associated with the checkpoint's session"""

    org_user_urn: str | None = None
    """Organization-user URN of the checkpoint's session owner"""

    user_email: str | None = None
    """Current email of the checkpoint's session owner"""

    user_name: str | None = None
    """Current display name of the checkpoint's session owner"""

    checkpoint_count: int
    """Number of active checkpoints"""

    size_bytes: int
    """Current stored bytes"""

    size_gigabytes: float
    """Current stored bytes converted to the storage billing unit"""

    estimated_monthly_cost_usd: float | None = None
    """Projected gross 720-hour cost at the current storage rate"""


class CurrentCheckpointStorageUsageResponse(BaseModel):
    """HTTP/JSON response returned by the API server to SDK and CLI consumers.

    Contains current checkpoint storage usage for the authenticated organization.
    Data can lag by 1-2 hours.
    """

    effective_rate_usd_per_gigabyte_month: float | None = None
    """Current gross storage rate used for projected costs"""

    data: list[CurrentCheckpointStorageUsageItem]
    """Storage grouped by project and session owner"""
