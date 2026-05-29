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

    @classmethod
    def from_tinker_path(cls, tinker_path: str) -> "ParsedCheckpointTinkerPath":
        """Parse a tinker path to an instance of ParsedCheckpointTinkerPath.

        Supports two formats:
        - Standard: tinker://run-id/weights/0001
        - With suffix: tinker://run-id:suffix/weights/0001
          (e.g., tinker://run-id:train:0/weights/0001)
        """
        if not tinker_path.startswith("tinker://"):
            raise ValueError(f"Invalid tinker path: {tinker_path}")

        # Remove the tinker:// prefix
        path_parts = tinker_path[9:]

        # Split into segments
        # Format: run_id_with_type/checkpoint_type/checkpoint_id
        segments = path_parts.split("/")
        if len(segments) != 3:
            raise ValueError(
                f"Invalid tinker path: {tinker_path}. "
                f"Expected: tinker://run-id/weights/0001 or tinker://run-id:train:0/weights/0001"
            )

        run_id_with_type = segments[0]
        checkpoint_type_segment = segments[1]
        checkpoint_id = segments[2]

        # Validate checkpoint type
        if checkpoint_type_segment not in ["weights", "sampler_weights"]:
            raise ValueError(
                f"Invalid checkpoint type: {checkpoint_type_segment}. "
                f"Expected: weights or sampler_weights"
            )


        checkpoint_type = "training" if checkpoint_type_segment == "weights" else "sampler"

        return cls(
            tinker_path=tinker_path,
            training_run_id=run_id_with_type,
            checkpoint_type=checkpoint_type,
            checkpoint_id="/".join([checkpoint_type_segment, checkpoint_id]),
        )
