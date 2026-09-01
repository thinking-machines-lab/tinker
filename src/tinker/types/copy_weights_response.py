from .._models import BaseModel

__all__ = ["CopyWeightsResponse"]


class CopyWeightsResponse(BaseModel):
    tinker_path: str
    """Path of the newly created copy."""
