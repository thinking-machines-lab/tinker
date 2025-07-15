from .._models import BaseModel
from .cursor import Cursor
from .training_run import TrainingRun

__all__ = ["TrainingRunsResponse"]


class TrainingRunsResponse(BaseModel):
    training_runs: list[TrainingRun]
    """List of training runs"""

    cursor: Cursor
    """Pagination cursor information"""
