"""Helpers for inspecting the Tinker SDK's authentication state."""

from __future__ import annotations

from ._exceptions import TinkerError
from .lib._auth_token_provider import ApiKeyAuthProvider, resolve_auth_provider

__all__ = [
    "get_tinker_token",
    "raise_if_tinker_not_accessible",
    "tinker_has_credentials",
]


def get_tinker_token() -> str | None:
    """Return the configured static Tinker token, or None if unavailable.

    This is a local-only lookup and does not execute ``TINKER_CREDENTIAL_CMD``.
    """
    try:
        provider = ApiKeyAuthProvider.create_or_env() or ApiKeyAuthProvider.create_from_stored()
    except (OSError, TinkerError, ValueError):
        return None
    return provider.api_key if provider is not None else None


def tinker_has_credentials() -> bool:
    """Return whether the SDK can resolve a configured Tinker credential.

    This is a local-only check. In particular, a configured
    ``TINKER_CREDENTIAL_CMD`` counts as available without running the command,
    since the command itself may contact an external credential service.
    """
    try:
        resolve_auth_provider(api_key=None, enforce_cmd=False)
    except (OSError, TinkerError, ValueError):
        return False
    return True


def raise_if_tinker_not_accessible() -> None:
    """Raise unless Tinker is usable with the configured credential.

    This is a blocking check that creates a :class:`~tinker.ServiceClient` and
    makes a single request to the server capabilities endpoint, which requires
    billing to be set up. It returns normally when the service accepted the
    request, and raises a :class:`~tinker.TinkerError` subclass otherwise —
    :class:`~tinker.TinkerError` when no credential is configured,
    :class:`~tinker.AuthenticationError` when the credential is rejected,
    :class:`~tinker.BillingError` when the account is not fully set up (a 402
    when billing is missing), or :class:`~tinker.APIConnectionError` when the
    service is unreachable. The request is not retried, so the failure
    surfaces as an exception rather than as a wait.
    """
    # Lazy import to avoid a circular import: this module is imported from
    # ``tinker.__init__`` before the public client interfaces are.
    from .lib.public_interfaces import ServiceClient

    try:
        ServiceClient()._check_accessible()
    except TinkerError:
        raise
    except Exception as e:
        # Credential resolution can surface e.g. OSError or ValueError; fold
        # those into the documented contract of raising TinkerError subclasses.
        raise TinkerError(str(e)) from e
