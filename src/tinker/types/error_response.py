# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, Optional

from .._models import BaseModel

__all__ = ["ErrorResponse"]


class ErrorResponse(BaseModel):
    error: str
    """Error code"""

    message: str
    """Human-readable error message"""

    details: Optional[Dict[str, object]] = None
    """Additional error details"""
