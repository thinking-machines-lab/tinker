from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .forward_backward_input import ForwardBackwardInput
from .model_id import ModelID

__all__ = ["ForwardBackwardRequest"]


@dataclass(frozen=True)
class ForwardBackwardRequest:
    forward_backward_input: ForwardBackwardInput

    model_id: ModelID

    seq_id: Optional[int] = None
