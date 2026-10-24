"""What the REST-only clients, `RestClient` and `AdminClient`, share."""

from __future__ import annotations

from tinker._version import __version__ as tinker_sdk_version

from ..internal_client_holder import InternalClientHolder, sessionless_holder_cache


def sdk_headers(sdk_method: str) -> dict[str, str]:
    """Headers attributing a REST request to the SDK version and public method that sent it.

    The server uses these to see who still calls an endpoint, and through which method,
    before deprecating it. `sdk_method` is the public method name without its `_async`
    suffix, since both variants are the same API.
    """
    return {"x-sdk-version": tinker_sdk_version, "x-sdk-method": sdk_method}


def rest_client_holder(
    client_name: str,
    _holder: InternalClientHolder | None,
    *,
    base_url: str | None,
    api_key: str | None,
) -> InternalClientHolder:
    """The holder a REST-only client runs over: `_holder` when one is handed over, else
    the process-wide session-less holder for `base_url` and `api_key`.

    A client constructed from credentials talks to the REST API only: no session is
    created on the server, so no project is needed and nothing shows up in the sessions
    list. Its connections are shared by every client constructed this way with the same
    base URL and API key, stay open for the life of the process, and need no closing.
    """
    if _holder is None:
        return sessionless_holder_cache.get_or_create(base_url, api_key)
    if base_url is not None or api_key is not None:
        raise ValueError(
            f"{client_name} takes either a holder or base_url/api_key, not both: "
            "a holder already carries its own base URL and credentials"
        )
    return _holder
