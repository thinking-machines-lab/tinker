from .._models import BaseModel

__all__ = ["Cursor"]


class Cursor(BaseModel):
    offset: int
    """The offset used for pagination"""

    limit: int
    """The maximum number of items requested"""

    total_count: int
    """The total number of items available"""
