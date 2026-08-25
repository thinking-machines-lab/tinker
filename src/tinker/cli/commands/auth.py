"""Commands for managing authentication credentials.

This module implements the 'tinker auth' commands:
- login: store an API key in ~/.tinker/credentials.json for the SDK and CLI,
  either by logging in through the browser (the default) or by entering an
  existing key with --api-key
- logout: remove the default credential from ~/.tinker/credentials.json, also
  deleting its API key on the server if the browser login minted it
- status: report whether credentials are available and Tinker is accessible

Only the user-facing wiring lives here: the browser flow itself is in
cli/login.py, which composes the WorkOS device-auth client (cli/device_auth.py)
with the Tinker auth API (cli/auth_api.py).
"""

import click

from ..client import handle_api_errors
from ..exceptions import TinkerCliError

_HTTP_TIMEOUT_SECONDS = 30.0


@click.group()
def cli():
    """Manage authentication credentials."""
    pass


@cli.command()
@click.option(
    "--api-key",
    "api_key",
    is_flag=True,
    help="Manually enter an API key instead of logging in via the browser",
)
def login(api_key: bool) -> None:
    """Store a credential for the SDK and CLI to use.

    By default this logs in through the browser: it shows a confirmation code
    and a URL, waits for you to approve the login, and mints an API key for
    your account. With --api-key it prompts for an existing key instead.

    Either way the key is stored in ~/.tinker/credentials.json as the default
    credential, which the SDK picks up when neither TINKER_API_KEY nor
    TINKER_CREDENTIAL_CMD is set.
    """
    # Lazy import to keep CLI startup fast.
    from tinker.lib.credentials import JsonCredentialStore, default_credentials_path

    store = JsonCredentialStore(default_credentials_path())
    if store.get_default_key() is not None:
        raise TinkerCliError(
            "Already logged in",
            "Run 'tinker auth logout' before logging in again.",
        )

    if api_key:
        _login_with_api_key()
    else:
        _login_with_device_auth()


def _login_with_api_key() -> None:
    """Prompt for an existing API key and store it."""
    # Lazy import to keep CLI startup fast.
    from tinker.lib.credentials import JsonCredentialStore, ManualKey, default_credentials_path

    key = click.prompt("API key", hide_input=True).strip()
    if not key:
        raise TinkerCliError("The API key must not be empty")

    store = JsonCredentialStore(default_credentials_path())
    store.add_key("manual", ManualKey(key=key, name="Manually added api key"))
    store.set_default("manual")
    click.echo("Successfully set API key")


def _login_with_device_auth() -> None:
    """Run the WorkOS device-auth flow and store the API key it mints."""
    # Lazy import to keep CLI startup fast.
    from ..auth_api import AuthApiError
    from ..device_auth import DeviceAuthError
    from ..login import device_login

    try:
        record = device_login(click.echo)
    except (DeviceAuthError, AuthApiError) as e:
        raise TinkerCliError(
            "Login failed",
            f"{e}\nYou can still enter a key manually with 'tinker auth login --api-key'.",
        ) from e

    details = record.details
    click.echo(
        f"Logged in as {details.user_details.email} ({details.org_details.name})."
        f" Stored API key '{record.name}' as the default credential."
    )


@cli.command()
def logout() -> None:
    """Remove the default credential, deleting browser-minted keys on the server.

    Removes the default credential from ~/.tinker/credentials.json (other
    stored credentials are kept). A key minted by the browser login is also
    deleted on the server, so it stops working everywhere it may have been
    copied. A manually entered key is only removed locally, since it may be
    shared with other machines or tools; delete it on the Tinker console if
    you want it revoked.

    Only stored credentials are affected: a key set via TINKER_API_KEY or
    TINKER_CREDENTIAL_CMD is untouched.
    """
    # Lazy import to keep CLI startup fast.
    from tinker.lib.credentials import GeneratedKey, JsonCredentialStore, default_credentials_path

    store = JsonCredentialStore(default_credentials_path())
    key_id = store.get_default_key_id()
    record = store.get_default_key()
    if key_id is None or record is None:
        raise TinkerCliError(
            "No default credential is stored",
            "There is nothing to log out from. Run 'tinker auth login' to log in.",
        )

    delete = isinstance(record, GeneratedKey)
    if delete:
        _delete_key_on_server(record.key)
    # Removing the key also clears the default, which pointed at it.
    store.delete_key(key_id)

    if delete:
        click.echo(f"Removed credential '{record.name}' and deleted its API key on the server.")
    else:
        click.echo(f"Removed credential '{record.name}'. The API key is still active.")


def _delete_key_on_server(key: str) -> None:
    """Revoke `key` via the Tinker API, using the key itself to authenticate."""
    # Lazy import to keep CLI startup fast.
    import httpx

    from ..auth_api import AuthApiError, TinkerAuthApi

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            TinkerAuthApi(client).delete_self_api_key(key)
    except AuthApiError as e:
        raise TinkerCliError(
            "Could not delete the API key on the server",
            f"{e}\nThe stored credential was left in place, so you can retry the logout.",
        ) from e


@cli.command()
@handle_api_errors
def status() -> None:
    """Check local credential availability and live access to Tinker."""
    # Lazy import to keep CLI startup fast.
    from tinker.auth import raise_if_tinker_not_accessible, tinker_has_credentials

    has_credentials = tinker_has_credentials()
    click.echo(f"Credentials available: {'yes' if has_credentials else 'no'}")
    if not has_credentials:
        raise TinkerCliError(
            "No Tinker credentials are available",
            "Run 'tinker auth login --api-key' or configure a Tinker credential.",
        )

    try:
        raise_if_tinker_not_accessible()
    except Exception:
        click.echo("Tinker accessible: no")
        raise
    click.echo("Tinker accessible: yes")
