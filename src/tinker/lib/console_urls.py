"""URL builders for opening Tinker resources in the console."""

from urllib.parse import quote

_CONSOLE_BASE_URL = "https://tinker.thinkingmachines.ai"


def sessions_console_url() -> str:
    return f"{_CONSOLE_BASE_URL}/sessions"


def session_console_url(session_id: str) -> str:
    return f"{_CONSOLE_BASE_URL}/sessions/{session_id}"


def training_run_console_url(training_run_id: str) -> str:
    return f"{_CONSOLE_BASE_URL}/training_runs/{training_run_id}"


def checkpoint_console_url(training_run_id: str, checkpoint_id: str) -> str:
    return (
        f"{_CONSOLE_BASE_URL}/checkpoints/{quote(training_run_id, safe='')}/"
        f"{quote(checkpoint_id, safe='')}"
    )


def sampler_checkpoint_console_url(training_run_id: str) -> str:
    return f"{_CONSOLE_BASE_URL}/checkpoints/{quote(training_run_id, safe='')}/sampler_weights"


def checkpoint_playground_url(tinker_path: str) -> str:
    return (
        f"{_CONSOLE_BASE_URL}/playground?mode=checkpoint&checkpoint={quote(tinker_path, safe='')}"
    )
