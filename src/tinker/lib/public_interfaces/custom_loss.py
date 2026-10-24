"""Custom loss functions created with ``ServiceClient.create_custom_loss`` and run by handle."""

from __future__ import annotations

import builtins
import hashlib
import inspect
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Concatenate, Dict, List, Mapping, Tuple, cast

from tinker import types
from tinker.types.custom_loss_graph import CustomLossNodeOp

if TYPE_CHECKING:
    import torch
    import torch.fx

__all__ = ["LossHandle", "PerDatumLossFn"]

# A per-datum custom loss: (logprobs, *the loss_fn_inputs it reads, each a tensor)
#   -> (scalar loss, scalar metrics keyed custom/<name>:sum or custom/<name>:mean).
PerDatumLossFn = Callable[
    Concatenate["torch.Tensor", ...],
    Tuple["torch.Tensor", Mapping[str, "torch.Tensor"]],
]

# Modules whose functions a custom loss may call. ``_operator`` is where the
# ``operator`` functions fx records for Python arithmetic and indexing live.
_ALLOWED_FUNCTION_MODULES = ("torch", "math", "operator", "_operator")
# The custom/ prefix keeps a loss's metrics from shadowing the trainer's own, like loss:sum.
_METRIC_KEY = re.compile(r"custom/[a-z0-9_./-]{1,64}:(sum|mean)")
_INPUT_NAME = re.compile(r"[a-z_][a-z0-9_]{0,63}")
_MAX_INPUTS = 16
# Tensor methods that turn a tensor into Python or numpy values.
_HOST_METHODS = frozenset({"item", "numpy", "tolist"})


@dataclass(frozen=True)
class LossHandle:
    """A loss function created with ``ServiceClient.create_custom_loss``.

    Pass it to ``TrainingClient.forward_backward_custom`` in place of a callable.
    """

    name: str
    inputs: Tuple[types.CustomLossInput, ...]
    """``logprobs``, then the ``loss_fn_inputs`` keys the loss reads, in the order it
    takes them, each with its declared rank."""

    session_id: str
    """The session that created the loss; only its clients can use it."""

    graph: "torch.fx.GraphModule" = field(repr=False, compare=False)
    """The traced loss; this, not the original function, is what runs."""

    loss_id: str | None = None
    """The server's id for the uploaded loss, or None when it runs client-side."""

    @property
    def key(self) -> str:
        """Identifies the traced loss, so a session creates each loss once."""
        digest = hashlib.sha256(self.graph.code.encode())
        digest.update(repr(self.inputs).encode())
        return digest.hexdigest()

    @property
    def input_names(self) -> Tuple[str, ...]:
        """The ``loss_fn_inputs`` keys the loss reads, in the order it takes them."""
        return tuple(loss_input.name for loss_input in self.inputs[1:])


def trace_loss(
    fn: PerDatumLossFn, input_ranks: Mapping[str, types.CustomLossInputRank], session_id: str
) -> LossHandle:
    """Trace ``fn`` into a ``LossHandle``, rejecting anything torch.fx cannot represent
    with only torch and math."""
    try:
        import torch.fx
    except ImportError as e:
        raise ImportError("PyTorch is not installed. Cannot create a custom loss.") from e

    inputs = _inputs(fn, input_ranks)
    try:
        graph = torch.fx.symbolic_trace(fn)
    except Exception as e:
        raise ValueError(
            f"create_custom_loss: {_fn_name(fn)} cannot be traced by torch.fx: {e}. Custom losses "
            "cannot branch on tensor values or call libraries other than torch and math."
        ) from e
    _check_graph(fn, graph)
    return LossHandle(name=_fn_name(fn), inputs=inputs, session_id=session_id, graph=graph)


def check_loss_inputs(handle: LossHandle, data: List[types.Datum]) -> None:
    """Raise unless every datum has each input the loss reads, at its declared rank.

    ``logprobs`` has the shape of the datum's ``target_tokens``.
    """
    logprobs, *inputs = handle.inputs
    for i, datum in enumerate(data):
        for loss_input, key in [(logprobs, "target_tokens"), *((x, x.name) for x in inputs)]:
            if key not in datum.loss_fn_inputs:
                raise ValueError(
                    f"datum[{i}] has no loss_fn_inputs[{key!r}], which loss {handle.name!r} takes"
                )
            shape = datum.loss_fn_inputs[key].shape
            rank = 1 if shape is None else len(shape)
            if rank != loss_input.rank:
                raise ValueError(
                    f"datum[{i}] loss_fn_inputs[{key!r}] has rank {rank}, but loss "
                    f"{handle.name!r} declared {loss_input.name} with rank {loss_input.rank}"
                )


def encode_graph(handle: LossHandle) -> types.CustomLossGraph:
    """The handle's graph as plain data, node for node; the server decides what it allows."""
    from torch.fx.node import _get_qualified_name

    nodes = []
    for node in handle.graph.graph.nodes:
        if node.op == "call_function":
            assert callable(node.target)
            # Python operators live in CPython's private _operator module; the wire
            # names them by their public operator module.
            target = _get_qualified_name(node.target)
            if target.startswith("_operator."):
                target = "operator." + target.removeprefix("_operator.")
        else:
            target = str(node.target)
        nodes.append(
            types.CustomLossNode(
                op=cast(CustomLossNodeOp, node.op),
                name=node.name,
                target=target,
                args=[_encode_arg(handle.name, arg) for arg in node.args],
                kwargs={key: _encode_arg(handle.name, arg) for key, arg in node.kwargs.items()},
            )
        )
    return types.CustomLossGraph(nodes=nodes)


def _encode_arg(name: str, arg: Any) -> Any:
    import torch
    import torch.fx

    if isinstance(arg, torch.fx.Node):
        return {"ref": arg.name}
    if isinstance(arg, float) and not math.isfinite(arg):
        return {"float": str(arg)}
    if arg is None or isinstance(arg, (bool, int, float, str)):
        return arg
    if isinstance(arg, list):
        return [_encode_arg(name, a) for a in arg]
    if isinstance(arg, tuple):
        return {"tuple": [_encode_arg(name, a) for a in arg]}
    if isinstance(arg, dict) and all(isinstance(key, str) for key in arg):
        return {"dict": {key: _encode_arg(name, a) for key, a in arg.items()}}
    if isinstance(arg, slice):
        return {"slice": [_encode_arg(name, a) for a in (arg.start, arg.stop, arg.step)]}
    if arg is Ellipsis:
        return {"ellipsis": True}
    torch_constants = {
        torch.dtype: "dtype",
        torch.device: "device",
        torch.layout: "layout",
        torch.memory_format: "memory_format",
    }
    if (kind := torch_constants.get(type(arg))) is not None:
        return {kind: str(arg).removeprefix("torch.")}
    raise ValueError(
        f"create_custom_loss: {name} passes a {type(arg).__name__} argument, which a "
        "custom loss cannot send to the server"
    )


def run_loss_handle(
    handle: LossHandle, data: List[types.Datum], logprobs_list: List["torch.Tensor"]
) -> Tuple["torch.Tensor", Dict[str, float]]:
    """Run ``handle`` on each datum and sum the losses; metrics reduce by their suffix.

    ``data`` has passed ``check_loss_inputs``.
    """
    import torch

    total: Any = None
    metric_sums: Dict[str, float] = {}
    for datum, logprobs in zip(data, logprobs_list, strict=True):
        inputs = [datum.loss_fn_inputs[name].to_torch() for name in handle.input_names]
        loss, metrics = handle.graph(logprobs, *inputs)
        if loss.ndim != 0:
            raise ValueError(
                f"loss {handle.name!r} must return a scalar loss, got shape {tuple(loss.shape)}"
            )
        total = loss if total is None else total + loss
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
    metrics_out = {
        key: value / len(data) if key.endswith(":mean") else value
        for key, value in metric_sums.items()
    }
    return total, metrics_out


def _fn_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


def _inputs(
    fn: Callable[..., Any], input_ranks: Mapping[str, types.CustomLossInputRank]
) -> Tuple[types.CustomLossInput, ...]:
    params = list(inspect.signature(fn).parameters.values())
    if not params or params[0].name != "logprobs":
        raise ValueError(
            f"create_custom_loss: {_fn_name(fn)} must take logprobs as its first parameter, followed "
            "by the loss_fn_inputs it reads, e.g. def loss(logprobs, advantages)"
        )
    if len(params) > _MAX_INPUTS:
        raise ValueError(
            f"create_custom_loss: {_fn_name(fn)} takes {len(params)} parameters; a custom loss "
            f"takes at most {_MAX_INPUTS}, logprobs included"
        )
    for param in params:
        if not _INPUT_NAME.fullmatch(param.name):
            raise ValueError(
                f"create_custom_loss: {_fn_name(fn)} parameter {param.name!r} must be a lowercase "
                "name of letters, digits and underscores"
            )
        if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            raise ValueError(
                f"create_custom_loss: {_fn_name(fn)} parameter {param.name!r} must be a plain "
                "positional parameter; *args, **kwargs and keyword-only parameters are not "
                "supported"
            )
        if param.default is not param.empty:
            raise ValueError(
                f"create_custom_loss: {_fn_name(fn)} parameter {param.name!r} has a default; every "
                "parameter after logprobs names a required loss_fn_inputs key"
            )
    names = [param.name for param in params]
    if set(input_ranks) != set(names):
        raise ValueError(
            f"create_custom_loss: input_ranks must give a rank for each parameter of "
            f"{_fn_name(fn)} ({', '.join(names)}), got {sorted(input_ranks)}"
        )
    if input_ranks["logprobs"] == 0:
        raise ValueError("create_custom_loss: logprobs has rank 1 ([T]) or 2 ([T, K])")
    return tuple(types.CustomLossInput(name=name, rank=input_ranks[name]) for name in names)


def _check_graph(fn: Callable[..., Any], graph: "torch.fx.GraphModule") -> None:
    import torch

    name = _fn_name(fn)
    for node in graph.graph.nodes:
        if node.op == "call_function":
            target = node.target
            module = getattr(target, "__module__", None) or ""
            if target is not builtins.getattr and module.split(".")[0] not in (
                _ALLOWED_FUNCTION_MODULES
            ):
                raise ValueError(
                    f"create_custom_loss: {name} calls {getattr(target, '__qualname__', target)!r} "
                    f"from {module or 'an unknown module'!r}; custom losses may only call "
                    "torch and math"
                )
        elif node.op == "call_method":
            assert isinstance(node.target, str)
            if not hasattr(torch.Tensor, node.target):
                raise ValueError(
                    f"create_custom_loss: {name} calls method {node.target!r}, which is not a "
                    "torch.Tensor method"
                )
            if node.target == "backward":
                raise ValueError(
                    f"create_custom_loss: {name} calls backward(); return the loss and "
                    "forward_backward_custom runs the backward pass"
                )
            if node.target in _HOST_METHODS:
                raise ValueError(
                    f"create_custom_loss: {name} calls {node.target}(), which leaves torch; "
                    "keep every value, metrics included, as a tensor"
                )
        elif node.op == "call_module":
            raise ValueError(
                f"create_custom_loss: {name} calls module {node.target!r}; custom losses cannot "
                "use torch.nn modules, only functions such as torch.nn.functional"
            )
        elif node.op == "get_attr":
            raise ValueError(
                f"create_custom_loss: {name} captures a tensor ({node.target!r}); a custom loss "
                "may only read logprobs and the loss_fn_inputs it names"
            )
        elif node.op == "output":
            _check_output(name, node.args[0])


def _check_output(name: str, output: Any) -> None:
    import torch.fx

    if not (isinstance(output, (tuple, list)) and len(output) == 2):
        raise ValueError(f"create_custom_loss: {name} must return (loss, metrics)")
    # A traced tensor is a graph node; anything else is a Python value.
    if not isinstance(output[0], torch.fx.Node):
        raise ValueError(
            f"create_custom_loss: {name} must return its loss as a torch.Tensor, got {output[0]!r}"
        )
    metrics = output[1]
    if not isinstance(metrics, dict):
        raise ValueError(f"create_custom_loss: {name} must return its metrics as a dict")
    for key, value in metrics.items():
        if not (isinstance(key, str) and _METRIC_KEY.fullmatch(key)):
            raise ValueError(
                f"create_custom_loss: {name} metric {key!r} must be named custom/<name>:sum or "
                "custom/<name>:mean in lowercase; the suffix says how it combines across datums"
            )
        if not isinstance(value, torch.fx.Node):
            raise ValueError(
                f"create_custom_loss: {name} metric {key!r} must be a torch.Tensor, got {value!r}"
            )
