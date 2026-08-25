"""Sampling from a deployed model, the way a user does it: text in, text out."""

import math

import pytest

import tinker

PROMPT = "List three prime numbers, separated by commas.\nAnswer:"
MAX_TOKENS = 16


def _sample(service_client, tokenizer, model: str, *, num_samples: int = 1, **overrides):
    params = {"max_tokens": MAX_TOKENS, "temperature": 0.0, "seed": 0} | overrides
    return (
        service_client.create_sampling_client(base_model=model)
        .sample(
            prompt=tinker.ModelInput.from_ints(tokenizer.encode(PROMPT)),
            num_samples=num_samples,
            sampling_params=tinker.SamplingParams(**params),
        )
        .result()
    )


@pytest.mark.gpu
def test_sampling_returns_decodable_text(service_client, tokenizers, model: str) -> None:
    tokenizer = tokenizers(model)
    sequence = _sample(service_client, tokenizer, model).sequences[0]

    assert sequence.tokens, "the model returned an empty completion"
    assert len(sequence.tokens) <= MAX_TOKENS

    completion = tokenizer.decode(sequence.tokens)
    assert completion.strip(), f"completion decoded to nothing: {sequence.tokens}"

    assert len(sequence.logprobs) == len(sequence.tokens)
    assert all(math.isfinite(lp) and lp <= 0 for lp in sequence.logprobs), sequence.logprobs

    # Batching needs its own sample call, but not its own test.
    batch = _sample(service_client, tokenizer, model, num_samples=3)
    assert len(batch.sequences) == 3
    assert all(sequence.tokens for sequence in batch.sequences)


@pytest.mark.gpu
def test_sampling_stops_at_the_token_budget(service_client, tokenizers, model: str) -> None:
    """A budget too small to finish on must be respected, and reported as such."""
    sequence = _sample(service_client, tokenizers(model), model, max_tokens=2).sequences[0]

    assert len(sequence.tokens) <= 2
    # Unconditional, so the check cannot evaporate if the model stops early.
    assert sequence.stop_reason in {"length", "stop"}, sequence.stop_reason
    if len(sequence.tokens) == 2:
        assert sequence.stop_reason == "length"


@pytest.mark.gpu
def test_sampling_client_reports_its_base_model(service_client, model: str) -> None:
    assert service_client.create_sampling_client(base_model=model).get_base_model() == model
