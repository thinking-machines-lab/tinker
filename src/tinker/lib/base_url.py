"""Where the Tinker API lives, and how deployments other than prod are chosen.

Both the SDK client (tinker._client.AsyncTinker) and the CLI's login flow
(tinker.cli.auth_api, which runs before any credential exists) resolve the
base URL here, so they can never disagree about which deployment a
TINKER_BASE_URL override points at.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

DEFAULT_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod"


def resolve_base_url(base_url: str | httpx.URL | None = None) -> str:
    """The Tinker API base URL: the argument, else $TINKER_BASE_URL, else prod.

    Empty values count as unset, and any trailing slash is dropped so callers
    can join paths onto the result.
    """
    resolved = str(base_url) if base_url is not None else ""
    resolved = resolved or os.environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL
    return resolved.rstrip("/")
