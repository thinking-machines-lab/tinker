import logging
from typing import Any, Dict, List, Sequence, Set, cast

import numpy as np
from tinker.types import ForwardBackwardOutput, LossFnOutput

logger = logging.getLogger(__name__)

Metrics = Dict[str, float]


def combine_fwd_bwd_output_results(
    results: Sequence[ForwardBackwardOutput],
) -> ForwardBackwardOutput:
    if not results:
        return ForwardBackwardOutput(loss_fn_output_type="", metrics={}, loss_fn_outputs=[])

    combined_metrics = _metrics_reduction(results)
    combined_outputs = _combine_loss_fn_outputs(results)

    return ForwardBackwardOutput(
        loss_fn_output_type=results[0].loss_fn_output_type,
        metrics=combined_metrics,
        loss_fn_outputs=combined_outputs,
    )


def _combine_loss_fn_outputs(results: Sequence[ForwardBackwardOutput]) -> List[LossFnOutput]:
    return [output for result in results for output in result.loss_fn_outputs]


def _order_insensitive_hash(xs: Sequence[Set[Any]] | Sequence[float]) -> int:
    """Combine hash values in an order-insensitive way.

    Args:
        xs: Either a sequence of sets (original data) or a sequence of already-computed hash values
    """
    # If we have sets, flatten and hash them (original behavior)
    if xs and isinstance(xs[0], set):
        return hash(tuple(sorted([y for x in xs for y in cast(Set[Any], x)])))

    # If we have already-computed hash values, combine them deterministically
    # Sort them to ensure order-insensitive combination
    return hash(tuple(sorted(int(cast(float, x)) for x in xs)))


def _mean(xs: Sequence[float | int], weights: Sequence[float] | None = None) -> float:
    if weights is None:
        return np.mean(xs).item()
    return np.average(xs, weights=weights).item()


def _sum(xs: Sequence[float | int]) -> float:
    return np.sum(xs).item()


def _min(xs: Sequence[float | int]) -> float:
    return np.min(xs).item()


def _max(xs: Sequence[float | int]) -> float:
    return np.max(xs).item()


def _slack(xs: Sequence[float | int], weights: Sequence[float] | None = None) -> float:
    if weights is None:
        return (np.max(xs) - np.mean(xs)).item()
    return (np.max(xs) - np.average(xs, weights=weights)).item()


def _unique(xs: Sequence[float | int]) -> Sequence[float | int]:
    """
    A unique metric can't actually fold. But in order to work around the fact that
    it's a str:float dict, we just insert unique keys with some suffix for each
    unique value. It's a hack, that's for sure.

    This is a dummy identity function just for documentation and consistency purposes.
    """
    return xs


REDUCE_MAP = {
    "mean": _mean,
    "sum": _sum,
    "min": _min,
    "max": _max,
    "slack": _slack,
    "hash_unordered": _order_insensitive_hash,
    "unique": _unique,
}


def _metrics_reduction(results: Sequence[ForwardBackwardOutput]) -> Metrics:
    """Reduce metrics from all actors.
    every metric must indicate a reduction_type in its name for example "mfu:mean"

    Metrics are weighted by the number of loss_fn_outputs (data points) each actor processed.
    """
    if not results:
        return {}
    keys = results[0].metrics.keys()

    weights = [len(m.loss_fn_outputs) for m in results]

    res = {}
    for key in keys:
        name, reduction = key.split(":")
        if reduction not in REDUCE_MAP:
            # Can happen when a new reduction type is added
            logger.debug(
                f"Invalid {reduction=} for metric {name=}. Expecting one of {REDUCE_MAP.keys()}"
            )
            continue
        if not all(key in m.metrics for m in results):
            continue
        reduce_fn = REDUCE_MAP[reduction]
        values = [m.metrics[key] for m in results]

        if reduction in ["mean", "slack"]:
            res[key] = reduce_fn(values, weights)
        elif reduction in ["unique"]:
            res[key] = values[0]
            res.update({f"{key}_{i + 1}": v for i, v in enumerate(values[1:])})
        else:
            res[key] = reduce_fn(values)
    return res
