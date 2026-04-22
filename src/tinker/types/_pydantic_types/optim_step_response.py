from typing import Dict, Optional

from ..._models import BaseModel


class OptimStepResponse(BaseModel):
    metrics: Optional[Dict[str, float]] = None
    """Optimization step metrics as key-value pairs"""
