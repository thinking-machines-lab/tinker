from __future__ import annotations

from typing import Any, Dict, List

from typing_extensions import Literal, TypeAlias

from .._models import StrictBase

__all__ = ["CustomLossGraph", "CustomLossNode", "CustomLossNodeOp"]

CustomLossNodeOp: TypeAlias = Literal[
    "placeholder", "get_attr", "call_function", "call_method", "call_module", "output"
]


class CustomLossNode(StrictBase):
    """One node of a traced custom loss, as torch.fx records it."""

    op: CustomLossNodeOp

    name: str
    """Unique within the graph; later nodes refer to this node by it."""

    target: str
    """The parameter name (placeholder), the function's qualified name (call_function,
    e.g. ``torch.exp`` or ``operator.mul``), the method or attribute name, or ``output``."""

    args: List[Any] = []
    """Encoded arguments; an output node's one argument is what the loss returns.
    JSON scalars and strings stand for themselves and lists for lists. Anything else
    is a one-key object: ``{"ref": name}`` for a node, ``{"tuple": [...]}``,
    ``{"dict": {...}}``, ``{"slice": [start, stop, step]}``, ``{"ellipsis": true}``,
    ``{"float": "inf"}`` (also ``"-inf"`` and ``"nan"``), ``{"dtype": "float32"}``,
    ``{"device": "cpu"}``, ``{"layout": "strided"}`` or
    ``{"memory_format": "contiguous_format"}``."""

    kwargs: Dict[str, Any] = {}
    """Encoded keyword arguments, in the same form as ``args``."""


class CustomLossGraph(StrictBase):
    """A traced custom loss as plain data: its nodes in execution order. Nothing in it
    is executable until the server rebuilds it from its own list of allowed
    operations."""

    format: Literal[1] = 1

    nodes: List[CustomLossNode]
