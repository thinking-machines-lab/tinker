"""The Tinker auth endpoints `tinker auth login` and `tinker auth logout` use.

- GET /api/v1/auth/apikey/me, which verifies the API key presented in the
  X-API-Key header and returns its public metadata. Login uses it to check the
  key the user pasted before storing it.
- DELETE /api/v1/auth/apikey/me, which deletes the API key presented in the
  X-API-Key header (and nothing else — it deliberately cannot delete other
  keys). Logout uses it to revoke the stored key it is about to discard.

Neither authenticates with a resolved SDK credential — they run before one is
stored, or with only the raw key in hand — so they go over a plain httpx
client rather than the SDK client (which resolves a credential up front).
"""

from __future__ import annotations

from typing import Optional, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

# The base URL resolution is shared with the SDK client (see tinker.lib.base_url)
# so `tinker auth login` always talks to the same deployment the SDK will.
from tinker.lib.base_url import resolve_base_url
from tinker.lib.credentials import ApiKeyDetails

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class AuthApiError(Exception):
    """A call to the Tinker auth API failed."""


class SelfApiKeyResponse(BaseModel):
    """The response of the current API key endpoint."""

    key_id: int
    name: str
    note: str
    details: ApiKeyDetails


class TinkerAuthApi:
    """Client for the Tinker API endpoints used to log in and out."""

    def __init__(self, http_client: httpx.Client, *, base_url: Optional[str] = None) -> None:
        self._http = http_client
        self._base_url = resolve_base_url(base_url)

    @property
    def base_url(self) -> str:
        return self._base_url

    def get_self_api_key(self, api_key: str) -> SelfApiKeyResponse:
        """Verify `api_key` and return its public metadata."""
        response = self._request("GET", "/api/v1/auth/apikey/me", headers={"X-API-Key": api_key})
        return self._parse(response, SelfApiKeyResponse)

    def delete_self_api_key(self, api_key: str) -> None:
        """Delete `api_key` itself on the server, revoking it everywhere."""
        self._request("DELETE", "/api/v1/auth/apikey/me", headers={"X-API-Key": api_key})

    def _request(
        self, method: str, path: str, *, headers: Optional[dict[str, str]] = None
    ) -> httpx.Response:
        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            response = self._http.request(method, url, headers=headers)
        except httpx.HTTPError as e:
            raise AuthApiError(f"Could not reach the Tinker API at {self._base_url}: {e}") from e
        if not response.is_success:
            raise AuthApiError(f"{_detail(response)} (HTTP {response.status_code} from {path})")
        return response

    @staticmethod
    def _parse(response: httpx.Response, model: type[_ModelT]) -> _ModelT:
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as e:
            raise AuthApiError(f"The Tinker API returned an unexpected response: {e}") from e


class _ErrorResponse(BaseModel):
    detail: Optional[str] = None


def _detail(response: httpx.Response) -> str:
    """The error message the API sent back, or a generic one."""
    try:
        return _ErrorResponse.model_validate(response.json()).detail or "Tinker API request failed"
    except (ValueError, ValidationError):
        return "Tinker API request failed"
