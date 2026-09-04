from typing import Optional

from typing_extensions import Literal

from .._models import BaseModel
from ..lib.console_urls import checkpoint_console_url
from .checkpoint import ParsedCheckpointTinkerPath

__all__ = ["SaveWeightsResponse"]


class SaveWeightsResponse(BaseModel):
    path: str
    """A tinker URI for model weights at a specific step"""

    type: Optional[Literal["save_weights"]] = None

    def get_console_url(self) -> str:
        """Return the Tinker Console URL for this checkpoint."""
        parsed_path = ParsedCheckpointTinkerPath.from_tinker_path(self.path)
        return checkpoint_console_url(parsed_path.training_run_id, parsed_path.checkpoint_id)
