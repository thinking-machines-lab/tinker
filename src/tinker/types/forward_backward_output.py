from typing import Dict, List

from .._models import BaseModel
from .loss_fn_output import LossFnOutput

__all__ = ["ForwardBackwardOutput"]


class ForwardBackwardOutput(BaseModel):
    loss_fn_output_type: str
    """The class name of the loss function output records (e.g., 'TorchLossReturn', 'ArrayRecord')."""

    loss_fn_outputs: List[LossFnOutput]
    """Dictionary mapping field names to tensor data"""

    metrics: Dict[str, float]
    """Training metrics as key-value pairs.

    The following metrics are recorded only during MoE (Mixture of Experts) training.
    Note: Don't fixate on the exact values of these metrics at the start of training.
    Different models on different data will have different initial values. How these
    metrics evolve over training is what matters.

    In the definitions below, *perfect balance* means ``total_tokens / num_experts``
    — the number of tokens each expert would receive if routing were perfectly uniform.

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
      across layers instead of the mean. Shows the worst-case load imbalance in any
      single layer.

    - ``e_min_violation:mean``: How much the least loaded expert is below perfect
      balance, as a fraction of perfect balance, averaged across layers. Computed as
      ``(min_tokens - perfect_balance) / perfect_balance``. A value of -0.5 means the
      least-used expert got half the fair share; -1.0 means it got nothing. Typically
      negative. Decreasing over time (more negative) is concerning.
    """
