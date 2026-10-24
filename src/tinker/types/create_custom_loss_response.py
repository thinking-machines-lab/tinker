from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["CreateCustomLossResponse"]


class CreateCustomLossResponse(BaseModel):
    loss_id: str
    """Names the stored loss in ``forward_backward`` requests of the same session."""

    type: Literal["create_custom_loss"] = "create_custom_loss"
