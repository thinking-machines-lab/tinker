from __future__ import annotations

from dataclasses import dataclass

from .tensor_data import TensorData

__all__ = ["PromptAltTokens"]


@dataclass(frozen=True)
class PromptAltTokens:
    """Independent draws from the model's next-token distribution at every prompt
    position after the first, taken in the prefill that served the request
    (``SampleRequest.prompt_alt_tokens_k``).

    Both tensors are dense ``[len(prompt) - 1, k]``. Row ``i`` holds ``k`` draws,
    with replacement and at the request's temperature, from the distribution
    over prompt token ``i + 1`` (position 0 has no preceding context), so
    ``tokens[i]`` are alternatives to ``prompt[i + 1]``.
    """

    tokens: TensorData
    """int64 token ids, shape ``[len(prompt) - 1, k]``."""

    logprobs: TensorData
    """float32 logprobs, shape ``[len(prompt) - 1, k]``: ``logprobs[i][j]`` is the
    model's logprob of ``tokens[i][j]`` at prompt position ``i + 1``."""
