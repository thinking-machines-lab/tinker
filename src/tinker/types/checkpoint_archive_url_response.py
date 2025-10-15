import datetime
from .._models import BaseModel

__all__ = ["CheckpointArchiveUrlResponse"]


class CheckpointArchiveUrlResponse(BaseModel):
    url: str
    """Signed URL to download the checkpoint archive"""

    expires: datetime.datetime
    """Unix timestamp when the signed URL expires, if available"""
