from typing import Optional

from typing_extensions import Literal

from ..._models import BaseModel
from ...lib.console_urls import checkpoint_playground_url, sampler_checkpoint_console_url
from ..checkpoint import ParsedCheckpointTinkerPath

__all__: list[str] = ["SaveWeightsForSamplerResponse"]


class SaveWeightsForSamplerResponseInternal(BaseModel):
    path: str | None = None
    """A tinker URI for model weights for sampling at a specific step"""
    sampling_session_id: str | None = None
    """The generated sampling session ID"""

    type: Optional[Literal["save_weights_for_sampler"]] = None


class SaveWeightsForSamplerResponse(BaseModel):
    path: str
    """A tinker URI for model weights for sampling at a specific step"""

    type: Optional[Literal["save_weights_for_sampler"]] = None

    def get_console_url(self) -> str:
        """Return the Tinker Console URL for this sampler checkpoint."""
        parsed_path = ParsedCheckpointTinkerPath.from_tinker_path(self.path)
        return sampler_checkpoint_console_url(parsed_path.training_run_id)

    def get_playground_url(self) -> str:
        """Return a Tinker Playground URL configured with this checkpoint."""
        return checkpoint_playground_url(self.path)
