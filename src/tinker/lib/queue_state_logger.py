"""Queue state logging utilities."""

import logging
import time

from .api_future_impl import QueueState, QueueStateObserver

logger = logging.getLogger(__name__)

# It can take a while for Tinker to make room for a request, so a pause is
# only reported once it has lasted this long.
PAUSE_GRACE_PERIOD = 15 * 60


class QueueStateLogger(QueueStateObserver):
    """Observer that logs queue state changes with throttling.

    Args:
        identifier: A string identifying what is being queued (e.g., model name or model ID)
        message_prefix: Prefix for the log message (e.g., "Model creation" or "Training")
    """

    def __init__(self, identifier: str, message_prefix: str = "Operation"):
        self._identifier = identifier
        self._message_prefix = message_prefix
        self._last_queue_state_logged: float = 0
        self._last_state: QueueState | None = None
        self._last_state_update_time: float = 0

    def on_queue_state_change(
        self, queue_state: QueueState, queue_state_reason: str | None
    ) -> None:
        QUEUE_STATE_LOG_INTERVAL = 60
        now = time.monotonic()
        if queue_state != self._last_state:
            self._last_state = queue_state
            self._last_state_update_time = now
        if (
            queue_state == QueueState.ACTIVE
            or now - self._last_state_update_time < PAUSE_GRACE_PERIOD
        ):
            return
        if time.time() - self._last_queue_state_logged < QUEUE_STATE_LOG_INTERVAL:
            return
        self._last_queue_state_logged = time.time()

        if not queue_state_reason:
            if queue_state == QueueState.PAUSED_RATE_LIMIT:
                queue_state_reason = "concurrent training clients rate limit hit"
            elif queue_state == QueueState.PAUSED_CAPACITY:
                queue_state_reason = "Tinker backend is running short on capacity, please wait"
            else:
                queue_state_reason = "unknown"

        logger.warning(
            f"{self._message_prefix} for {self._identifier} is paused. "
            f"Reason: {queue_state_reason}."
        )
