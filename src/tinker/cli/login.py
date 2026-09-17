"""Helpers for `tinker auth login`."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Callable

import click


def api_key_name() -> str:
    """The name a key for this machine gets: `tinker-cli-<machine name>`."""
    hostname = socket.gethostname().split(".", 1)[0].strip()
    return f"tinker-cli-{hostname or 'unknown'}"


def prompt_api_key() -> str:
    """Prompt for an API key, masking input when a terminal is available."""
    if not sys.stdin.isatty():
        return click.prompt("Paste your API key", hide_input=True)

    try:
        if sys.platform == "win32":
            key = _prompt_masked_windows()
        else:
            key = _prompt_masked_posix()
    except (KeyboardInterrupt, EOFError):
        click.echo()
        raise click.Abort() from None

    click.echo()
    return key


def _prompt_masked_posix() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return _read_masked_input(lambda: os.read(fd, 32).decode(sys.stdin.encoding or "utf-8"))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)
        sys.stdout.flush()


def _prompt_masked_windows() -> str:
    import msvcrt

    return _read_masked_input(msvcrt.getwch)


def _read_masked_input(read: Callable[[], str]) -> str:
    click.echo("Paste your API key: ", nl=False)
    key: list[str] = []
    while True:
        entered = read()
        if entered.startswith("\x1b"):
            continue
        for character in entered:
            if character in ("\r", "\n"):
                return "".join(key)
            if character in ("\b", "\x7f"):
                if key:
                    key.pop()
                    click.echo("\b \b", nl=False)
            elif character == "\x03":
                raise KeyboardInterrupt
            elif character == "\x04":
                raise EOFError
            elif character.isprintable():
                key.append(character)
                click.echo("*", nl=False)
