from .._models import BaseModel
from .model_id import ModelID

__all__ = ["GetSessionResponse"]


class GetSessionResponse(BaseModel):
    # List of training run IDs (model IDs) associated with this session
    training_run_ids: list[ModelID]

    # List of sampler IDs associated with this session
    sampler_ids: list[str]

    user_metadata: dict[str, str] | None = None
    """Optional metadata attached to this session by the user."""
