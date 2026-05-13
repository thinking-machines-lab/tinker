from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .loss_fn_output import LossFnOutput

__all__ = ["ForwardBackwardOutput"]


@dataclass(frozen=True)
class ForwardBackwardOutput:
    loss_fn_output_type: str
    """The class name of the loss function output records (e.g., 'TorchLossReturn', 'ArrayRecord')."""

    loss_fn_outputs: List[LossFnOutput]
    """List of per-datum dicts mapping field names to ``TensorData``."""

    metrics: Dict[str, float] = field(default_factory=dict)
    """Training metrics as key-value pairs.

    The following metrics are recorded only during MoE (Mixture of Experts) training.

    - ``e_frac_with_tokens:mean``: Fraction of experts that received at least one token,
      averaged across layers. A value of 1.0 means every expert got work; 0.5 means half
      were idle. Decreasing over time is concerning (routing collapse).

    - ``e_frac_oversubscribed:mean``: Fraction of experts receiving more tokens than
      perfect balance, averaged across layers. Increasing over time is concerning.

    - ``e_max_violation:mean``: How much the most overloaded expert exceeds perfect
      balance, as a fraction of perfect balance, averaged across layers. Computed as
      ``(max_tokens - perfect_balance) / perfect_balance``. A value of 2.0 means the
      busiest expert got 3x the fair share. Increasing over time is concerning.

    - ``e_max_violation:max``: Same as ``e_max_violation:mean`` but takes the max
      across layers instead of the mean.

    - ``e_min_violation:mean``: How much the least loaded expert is below perfect
      balance, as a fraction of perfect balance, averaged across layers. Typically
      negative; decreasing (more negative) is concerning.
    """
