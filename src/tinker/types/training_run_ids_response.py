from .._models import BaseModel

__all__ = ["TrainingRunIdsResponse"]


class TrainingRunIdsResponse(BaseModel):
    training_run_ids: list[str]
    """List of training run IDs"""

    has_more: bool
    """Whether there are more results available for pagination"""
