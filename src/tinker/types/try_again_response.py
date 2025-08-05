# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["TryAgainResponse"]


class TryAgainResponse(BaseModel):
    request_id: str
    """Request ID that is still pending"""

    type: Literal["try_again"] = "try_again"
