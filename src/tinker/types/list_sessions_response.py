from .._models import BaseModel

__all__ = ["ListSessionsResponse"]


class ListSessionsResponse(BaseModel):
    # A list of session IDs
    sessions: list[str]
