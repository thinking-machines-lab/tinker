from .._models import StrictBase
from .request_id import RequestID

__all__ = ["FutureRetrieveRequest"]


class FutureRetrieveRequest(StrictBase):
    request_id: RequestID
    """The ID of the request to retrieve"""
