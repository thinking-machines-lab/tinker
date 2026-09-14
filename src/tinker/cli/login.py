"""The pieces `tinker auth login` needs beyond the flow itself.

The command lives in cli/commands/auth.py; the name it gives a key and the
best-effort browser opener live here, away from the command's lazy imports.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser


def api_key_name() -> str:
    """The name a key for this machine gets: `tinker-cli-<machine name>`."""
    hostname = socket.gethostname().split(".", 1)[0].strip()
    return f"tinker-cli-{hostname or 'unknown'}"


def open_url(url: str) -> bool:
    """Ask a browser to open `url`, reporting whether there was one to ask.

    The open runs on a daemon thread and its result is ignored: `webbrowser`
    runs a custom $BROWSER command in the foreground, which would otherwise
    block the login for as long as that browser stays open.
    """
    if not _browser_is_available():
        return False
    threading.Thread(target=_open_quietly, args=(url,), daemon=True).start()
    return True


def _open_quietly(url: str) -> None:
    """Open `url`, swallowing the failure of a browser that won't start."""
    try:
        webbrowser.open(url)
    except (webbrowser.Error, OSError):
        pass


def _browser_is_available() -> bool:
    """Whether launching a browser is worth attempting.

    On a headless Linux box or over SSH, `webbrowser` happily falls back to a
    console browser like lynx, which would take over the terminal the login is
    running in — so require a graphical session there.
    """
    if os.environ.get("BROWSER"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
