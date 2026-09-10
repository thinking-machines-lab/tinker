import datetime

from .._models import BaseModel

__all__ = ["ExternalWeightsUrlsResponse"]


class ExternalWeightsUrlsResponse(BaseModel):
    urls: dict[str, str]
    """Signed download URL per file, keyed by the file path relative to the checkpoint root"""

    expires: datetime.datetime
    """When the signed URLs expire"""
