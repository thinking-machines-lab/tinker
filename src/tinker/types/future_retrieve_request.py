from .._models import StrictBase
from .request_id import RequestID

__all__ = ["FutureRetrieveRequest"]


class FutureRetrieveRequest(StrictBase):
    request_id: RequestID
    """The ID of the request to retrieve"""

    allow_metadata_only: bool = False
    """When True, the server may return only response metadata (status and size)
    instead of the full payload if the response exceeds the server's inline size limit."""
