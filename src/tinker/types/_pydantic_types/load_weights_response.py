from typing import Optional

from typing_extensions import Literal

from ..._models import BaseModel


class LoadWeightsResponse(BaseModel):
    path: Optional[str] = None
    """A tinker URI for model weights at a specific step"""

    model_id: Optional[str] = None
    """Canonical id of the model the weights were loaded onto."""

    type: Optional[Literal["load_weights"]] = None
