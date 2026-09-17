"""Commands for managing authentication credentials.

This module implements the 'tinker auth' commands:
- login: store an API key in ~/.tinker/credentials.json for the SDK and CLI,
  by printing the console's New API Key URL and prompting for the key they
  create there
- logout: remove the default credential from ~/.tinker/credentials.json, also
  deleting its API key on the server if the CLI minted it for itself
- status: report whether credentials are available and Tinker is accessible
"""

import click

from ..client import handle_api_errors
from ..exceptions import TinkerCliError

_HTTP_TIMEOUT_SECONDS = 30.0
_OSC = "\x1b]"
_ST = "\x1b\\"


@click.group()
def cli():
    """Manage authentication credentials."""
    pass


@cli.command()
def login() -> None:
    """Store a credential for the SDK and CLI to use.

    Prints the URL for the Tinker console's New API Key dialog with a name for
    this machine filled in, then prompts you to paste the key you create there
    back in.

    The key is stored in ~/.tinker/credentials.json as the default credential,
    which the SDK picks up when neither TINKER_API_KEY nor
    TINKER_CREDENTIAL_CMD is set.
    """
    # Lazy import to keep CLI startup fast.
    import httpx

    from tinker.lib.console_urls import new_api_key_console_url
    from tinker.lib.credentials import JsonCredentialStore, ManualKey, default_credentials_path

    from ..auth_api import AuthApiError, TinkerAuthApi
    from ..login import api_key_name, prompt_api_key

    store = JsonCredentialStore(default_credentials_path())
    if store.get_default_key() is not None:
        raise TinkerCliError(
            "Already logged in",
            "Run 'tinker auth logout' before logging in again.",
        )

    url = new_api_key_console_url(api_key_name())
    click.echo(f"Create an API Key: {_terminal_hyperlink(url)}")
    click.echo()

    key = prompt_api_key().strip()
    if not key:
        raise TinkerCliError("The API key must not be empty")

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS) as client:
            api_key = TinkerAuthApi(client).get_self_api_key(key)
    except AuthApiError as e:
        raise TinkerCliError("Could not validate the API key", str(e)) from e

    key_id = str(api_key.key_id)
    record = ManualKey(
        key=key,
        name=api_key.name,
        note=api_key.note,
        details=api_key.details,
    )
    store.add_key(key_id, record)
    store.set_default(key_id)
    details = api_key.details
    click.echo(
        f"Logged in as {details.user_details.email} ({details.org_details.name})."
        f" Stored API key '{api_key.name}' as the default credential."
    )


@cli.command()
def logout() -> None:
    """Remove the default credential, deleting CLI-minted keys on the server.

    Removes the default credential from ~/.tinker/credentials.json (other
    stored credentials are kept). A key the CLI minted for itself is also
    deleted on the server, so it stops working everywhere it may have been
    copied. A pasted key is only removed locally, since it may be shared with
    other machines or tools; delete it on the Tinker console if you want it
    revoked.

    Only stored credentials are affected: a key set via TINKER_API_KEY or
    TINKER_CREDENTIAL_CMD is untouched.
    """
    # Lazy import to keep CLI startup fast.
    from tinker.lib.console_urls import api_key_console_url
    from tinker.lib.credentials import GeneratedKey, JsonCredentialStore, default_credentials_path

    from ..auth_api import AuthApiError

    store = JsonCredentialStore(default_credentials_path())
    key_id = store.get_default_key_id()
    record = store.get_default_key()
    if key_id is None or record is None:
        raise TinkerCliError(
            "No default credential is stored",
            "There is nothing to log out from. Run 'tinker auth login' to log in.",
        )

    organization = (
        record.details.org_details.name
        if record.details is not None
        else "unknown (not stored with this credential)"
    )
    delete = isinstance(record, GeneratedKey)
    delete_error: AuthApiError | None = None
    if delete:
        try:
            _delete_key_on_server(record.key)
        except AuthApiError as e:
            delete_error = e
    # Removing the key also clears the default, which pointed at it.
    store.delete_key(key_id)

    if delete_error is not None:
        url = api_key_console_url(key_id)
        raise TinkerCliError(
            "Could not delete the API key on the server",
            f"Removed the local credential.\n"
            f"API key name: {record.name}\n"
            f"API key ID: {key_id}\n"
            f"{_api_key_deletion_hint(url, organization)}",
        ) from delete_error
    if delete:
        click.echo(f"Removed credential '{record.name}' and deleted its API key on the server.")
    else:
        click.echo(f"Removed credential '{record.name}'. The API key is still active.")
        url = api_key_console_url(key_id)
        click.echo(_api_key_deletion_hint(url, organization))


def _terminal_hyperlink(url: str) -> str:
    """Display `url` as itself while making it clickable in OSC-8-aware terminals."""
    return f"{_OSC}8;;{url}{_ST}{url}{_OSC}8;;{_ST}"


def _api_key_deletion_hint(url: str, organization: str) -> str:
    return f"API key organization: {organization}\nDelete api key here: {_terminal_hyperlink(url)}"


def _delete_key_on_server(key: str) -> None:
    """Revoke `key` via the Tinker API, using the key itself to authenticate."""
    # Lazy import to keep CLI startup fast.
    import httpx

    from ..auth_api import TinkerAuthApi

    with httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS) as client:
        TinkerAuthApi(client).delete_self_api_key(key)


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
            "Run 'tinker auth login' or configure a Tinker credential.",
        )

    try:
        raise_if_tinker_not_accessible()
    except Exception:
        click.echo("Tinker accessible: no")
        raise
    click.echo("Tinker accessible: yes")
