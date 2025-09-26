from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .telemetry import Telemetry


@runtime_checkable
class TelemetryProvider(Protocol):
    def get_telemetry(self) -> Telemetry | None: ...
