from .._models import BaseModel

__all__ = ["WhoamiResponse"]


class WhoamiResponse(BaseModel):
    # URN of the calling principal
    user_urn: str

    # URN of the organization the credential acts in
    org_urn: str

    # Email of the calling user, if the principal is user-backed
    email: str | None = None
