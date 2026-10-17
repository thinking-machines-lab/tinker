from typing import Optional

from .._models import StrictBase

__all__ = ["CopyWeightsRequest"]


class CopyWeightsRequest(StrictBase):
    """Provide a destination project, or both a session ID and its model sequence ID."""

    session_id: Optional[str] = None
    """Session that will own the copy; the copy inherits its project."""

    model_seq_id: Optional[int] = None
    """Training run slot within the session, as CreateModelRequest and LoadWeightsRequest use."""

    destination_project_id: Optional[str] = None
    """Project that will own the copy when no session_id is provided."""

    source_path: str
    """A tinker URI for the weights to copy. Either kind is accepted."""

    ttl_seconds: Optional[int] = None
    """Seconds until the copy expires, 1 hour (3600) to 10 years. Source expiry is not inherited."""

    weights_access_token: Optional[str] = None
    """Optional access token for copying weights under a different account."""
