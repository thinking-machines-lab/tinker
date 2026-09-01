from __future__ import annotations

from typing import List

from .._models import BaseModel
from .future_completion import FutureCompletion

__all__ = ["FuturesRetrieveResponse"]


class FuturesRetrieveResponse(BaseModel):
    completions: List[FutureCompletion]
    cursor: int
    """Opaque cursor to pass back as ``prev_cursor`` on the next poll."""
