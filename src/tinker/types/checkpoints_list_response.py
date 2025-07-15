from .._models import BaseModel
from .checkpoint import Checkpoint

__all__ = ["CheckpointsListResponse"]


class CheckpointsListResponse(BaseModel):
    checkpoints: list[Checkpoint]
    """List of available model checkpoints for the model"""
