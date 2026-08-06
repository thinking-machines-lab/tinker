from typing import Any, Optional

from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID

__all__ = ["LoadWeightsRequest"]


class LoadWeightsRequest(StrictBase):
    model_id: Optional[ModelID] = None
    """Legacy addressing: a load right after create_model (with seq_id 1).
    Set together with seq_id, and mutually exclusive with session_id."""

    seq_id: Optional[int] = None

    session_id: Optional[str] = None
    """Create-via-load addressing: the load is the model's first request
    (seq id 0, no create_model). Set together with model_seq_id; mirrors
    CreateModelRequest minus lora_config, which the server derives from the
    checkpoint."""

    model_seq_id: Optional[int] = None

    base_model: Optional[str] = None
    """Optional base model override for create-via-load; must be compatible
    with the checkpoint's base model (e.g. a different context length)."""

    user_metadata: Optional[dict[str, Any]] = None
    """Optional metadata about this model/training run, set by the end-user."""

    path: str
    """A tinker URI for model weights at a specific step"""

    optimizer: bool
    """Whether to load optimizer state along with model weights"""

    weights_access_token: Optional[str] = None
    """Optional access token for loading checkpoints under a different account."""

    type: Literal["load_weights"] = "load_weights"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
