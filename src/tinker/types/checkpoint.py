from datetime import datetime
from typing import Literal

from .._models import BaseModel

__all__ = ["Checkpoint", "CheckpointType"]

CheckpointType = Literal["training", "sampler", "external"]


class Checkpoint(BaseModel):
    checkpoint_id: str
    """The checkpoint ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training, sampler, or external)"""

    time: datetime
    """The time when the checkpoint was created"""

    tinker_path: str
    """The tinker path to the checkpoint"""

    size_bytes: int | None = None
    """The size of the checkpoint in bytes"""

    public: bool = False
    """Whether the checkpoint is publicly accessible"""

    user_metadata: dict[str, str] | None = None
    """Optional user-provided metadata attached to the checkpoint"""

    expires_at: datetime | None = None
    """When this checkpoint expires (None = never expires)"""


class ParsedCheckpointTinkerPath(BaseModel):
    tinker_path: str
    """The tinker path to the checkpoint"""

    training_run_id: str
    """The training run ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training, sampler, or external)"""

    checkpoint_id: str
    """The checkpoint ID"""

    @classmethod
    def from_tinker_path(cls, tinker_path: str) -> "ParsedCheckpointTinkerPath":
        """Parse a tinker path to an instance of ParsedCheckpointTinkerPath"""
        if not tinker_path.startswith("tinker://"):
            raise ValueError(f"Invalid tinker path: {tinker_path}")
        parts = tinker_path[9:].split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid tinker path: {tinker_path}")
        segment_to_type: dict[str, CheckpointType] = {
            "weights": "training",
            "sampler_weights": "sampler",
            "external_weights": "external",
        }
        if parts[1] not in segment_to_type:
            raise ValueError(f"Invalid tinker path: {tinker_path}")
        checkpoint_type = segment_to_type[parts[1]]
        return cls(
            tinker_path=tinker_path,
            training_run_id=parts[0],
            checkpoint_type=checkpoint_type,
            checkpoint_id="/".join(parts[1:]),
        )
