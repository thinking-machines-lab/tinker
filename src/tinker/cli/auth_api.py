"""The Tinker auth endpoints `tinker auth login` and `tinker auth logout` use.

- GET /api/unauthed/auth-config, which hands out the WorkOS client id the
  device-auth flow needs and requires no credential at all.
- POST /api/v1/auth/apikey, which mints an API key for the caller. It accepts
  exactly one kind of credential: a WorkOS access token, sent as a bearer
  token — API keys deliberately cannot mint further API keys.
- GET /api/v1/auth/apikey/me, which verifies the API key presented in the
  X-API-Key header and returns its public metadata.
- DELETE /api/v1/auth/apikey/me, which deletes the API key presented in the
  X-API-Key header (and nothing else — it deliberately cannot delete other
  keys). Logout uses it to revoke the stored key it is about to discard.

None of these authenticate with a resolved SDK credential — they run before
one is stored, or with only the raw key in hand — so they go over a plain
httpx client rather than the SDK client (which resolves a credential up front).
"""

from __future__ import annotations

from typing import Optional, TypeVar

import httpx
from pydantic import AliasChoices, BaseModel, Field, ValidationError

# The base URL resolution is shared with the SDK client (see tinker.lib.base_url)
# so `tinker auth login` always talks to the same deployment the SDK will.
from tinker.lib.base_url import resolve_base_url
from tinker.lib.credentials import ApiKeyDetails

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class AuthApiError(Exception):
    """A call to the Tinker auth API failed."""


class CreatedApiKey(BaseModel):
    """The response of the API key creation endpoint."""

    api_key: str
    # Public identifier of the minted key; not sensitive, unlike api_key.
    id: int
    details: ApiKeyDetails


class SelfApiKeyResponse(BaseModel):
    """The response of the current API key endpoint."""

    key_id: int
    name: str
    note: str
    details: ApiKeyDetails


class _AuthConfigResponse(BaseModel):
    # The server serializes this field as `clientId`; accept either spelling.
    client_id: str = Field(validation_alias=AliasChoices("clientId", "client_id"))


class TinkerAuthApi:
    """Client for the Tinker API endpoints used while logging in."""

    def __init__(self, http_client: httpx.Client, *, base_url: Optional[str] = None) -> None:
        self._http = http_client
        self._base_url = resolve_base_url(base_url)

    @property
    def base_url(self) -> str:
        return self._base_url

    def fetch_workos_client_id(self) -> str:
        """The WorkOS client id this Tinker deployment authenticates against."""
        response = self._request("GET", "/api/unauthed/auth-config")
        client_id = self._parse(response, _AuthConfigResponse).client_id
        if not client_id.strip():
            raise AuthApiError("The Tinker API returned an empty WorkOS client ID")
        return client_id

    def create_api_key(self, access_token: str, *, name: str, note: str) -> CreatedApiKey:
        """Mint an API key for the user the WorkOS access token belongs to."""
        response = self._request(
            "POST",
            "/api/v1/auth/apikey",
            json={"name": name, "note": note},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        return self._parse(response, CreatedApiKey)

    def get_self_api_key(self, api_key: str) -> SelfApiKeyResponse:
        """Verify `api_key` and return its public metadata."""
        response = self._request("GET", "/api/v1/auth/apikey/me", headers={"X-API-Key": api_key})
        return self._parse(response, SelfApiKeyResponse)

    def delete_self_api_key(self, api_key: str) -> None:
        """Delete `api_key` itself on the server, revoking it everywhere."""
        self._request("DELETE", "/api/v1/auth/apikey/me", headers={"X-API-Key": api_key})

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            response = self._http.request(method, url, json=json, headers=headers)
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
