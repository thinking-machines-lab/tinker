from typing import Dict, Optional

from .._models import BaseModel

__all__ = ["OptimStepResponse"]


class OptimStepResponse(BaseModel):
    metrics: Optional[Dict[str, float]] = None
    """Optimization step metrics as key-value pairs"""
