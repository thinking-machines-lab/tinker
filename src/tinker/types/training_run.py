from datetime import datetime

from .._models import BaseModel
from .checkpoint import Checkpoint

__all__ = ["TrainingRun"]


class TrainingRun(BaseModel):
    training_run_id: str
    """The unique identifier for the training run"""

    base_model: str
    """The base model name this model is derived from"""

    model_owner: str
    """The owner/creator of this model"""

    is_lora: bool
    """Whether this model uses LoRA (Low-Rank Adaptation)"""

    corrupted: bool = False
    """Whether the model is in a corrupted state"""

    lora_rank: int | None = None
    """The LoRA rank if this is a LoRA model, null otherwise"""

    last_request_time: datetime
    """The timestamp of the last request made to this model"""

    last_checkpoint: Checkpoint | None = None
    """The most recent training checkpoint, if available"""

    last_sampler_checkpoint: Checkpoint | None = None
    """The most recent sampler checkpoint, if available"""

    user_metadata: dict[str, str] | None = None
    """Optional metadata about this training run, set by the end-user"""
