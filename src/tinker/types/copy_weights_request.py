from typing import Optional

from .._models import StrictBase

__all__ = ["CopyWeightsRequest"]


class CopyWeightsRequest(StrictBase):
    session_id: str
    """Session that will own the copy; the copy inherits its project."""

    model_seq_id: int
    """Training run slot within the session, as CreateModelRequest and LoadWeightsRequest use."""

    source_path: str
    """A tinker URI for the weights to copy. Either kind is accepted."""

    ttl_seconds: Optional[int] = None
    """Seconds until the copy expires. The source's expiry is not inherited."""

    weights_access_token: Optional[str] = None
    """Optional access token for copying weights under a different account."""
