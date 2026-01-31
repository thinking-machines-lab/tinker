"""Queue state logging utilities."""

import logging
import time

from .api_future_impl import QueueState, QueueStateObserver

logger = logging.getLogger(__name__)


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

    def on_queue_state_change(
        self, queue_state: QueueState, queue_state_reason: str | None
    ) -> None:
        QUEUE_STATE_LOG_INTERVAL = 60
        if queue_state == QueueState.ACTIVE:
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
