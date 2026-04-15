from datetime import datetime
from typing import Literal

from .._models import BaseModel

__all__ = ["Checkpoint", "CheckpointType"]

CheckpointType = Literal["training", "sampler"]


class Checkpoint(BaseModel):
    checkpoint_id: str
    """The checkpoint ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training or sampler)"""

    time: datetime
    """The time when the checkpoint was created"""

    tinker_path: str
    """The tinker path to the checkpoint"""

    size_bytes: int | None = None
    """The size of the checkpoint in bytes"""

    public: bool = False
    """Whether the checkpoint is publicly accessible"""

    expires_at: datetime | None = None
    """When this checkpoint expires (None = never expires)"""


class ParsedCheckpointTinkerPath(BaseModel):
    tinker_path: str
    """The tinker path to the checkpoint"""

    training_run_id: str
    """The training run ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training or sampler)"""

    checkpoint_id: str
    """The checkpoint ID"""

    @property
    def api_checkpoint_id(self) -> str:
        """Return the checkpoint ID formatted for API calls.
        
        For training checkpoints: returns just the checkpoint number (e.g., '0001').
        For sampler checkpoints: returns prefixed ID (e.g., 'sampler_weights/0001').
        """
        if self.checkpoint_type == "training":
            # Training checkpoints use just the number
            return self.checkpoint_id.split("/")[-1]
        else:
            # Sampler checkpoints include the prefix
            return self.checkpoint_id

    @classmethod
    def from_tinker_path(cls, tinker_path: str) -> "ParsedCheckpointTinkerPath":
        """Parse a tinker path to an instance of ParsedCheckpointTinkerPath"""
        if not tinker_path.startswith("tinker://"):
            raise ValueError(f"Invalid tinker path: {tinker_path}")
        parts = tinker_path[9:].split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid tinker path: {tinker_path}")
        if parts[1] not in ["weights", "sampler_weights"]:
            raise ValueError(f"Invalid tinker path: {tinker_path}")
        checkpoint_type = "training" if parts[1] == "weights" else "sampler"
        return cls(
            tinker_path=tinker_path,
            training_run_id=parts[0],
            checkpoint_type=checkpoint_type,
            checkpoint_id="/".join(parts[1:]),
        )
