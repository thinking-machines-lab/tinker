from .._models import StrictBase

__all__ = ["GetCurrentCheckpointStorageUsageRequest"]


class GetCurrentCheckpointStorageUsageRequest(StrictBase):
    """HTTP query parameters sent by the SDK and CLI to the API server.

    Used by GET /api/v1/billing/usage/checkpoints/current; no JSON request body.
    With no parameters, returns checkpoint storage usage for the entire
    authenticated organization.
    """

    project_id: str | None = None
    """Only return checkpoints attributed to this project; omit for the entire org"""
