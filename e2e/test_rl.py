"""The RL round trip: sample an answer, score it, train on the score.

The only test crossing the sampling and training boundary, where an integration
breaks without either half looking broken alone.
"""

import pytest

import tinker

PROMPT = "Compute 17 + 25. Reply with the number only.\nAnswer:"
EXPECTED = "42"
MAX_TOKENS = 8

# All four take the same inputs: see loss_function_registry.DEFAULT_FIELD_SHAPES.
RL_LOSS_FNS = ["importance_sampling", "ppo", "cispo", "dro"]


def _reward(completion: str) -> float:
    """Verifiable: the arithmetic is either right or it is not."""
    return 1.0 if EXPECTED in completion else -1.0


@pytest.mark.gpu
@pytest.mark.parametrize("loss_fn", RL_LOSS_FNS)
def test_train_on_a_scored_answer(
    service_client, training_clients, tokenizers, model: str, loss_fn: str
) -> None:
    tokenizer = tokenizers(model)
    prompt_ids = tokenizer.encode(PROMPT)

    response = (
        service_client.create_sampling_client(base_model=model)
        .sample(
            prompt=tinker.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0, seed=0),
        )
        .result()
    )

    sequence = response.sequences[0]
    completion = tokenizer.decode(sequence.tokens)
    assert completion.strip(), "nothing was sampled to train on"
    assert len(sequence.logprobs) == len(sequence.tokens)

    # Score only the tokens the model produced.
    advantage = _reward(completion)
    tokens = prompt_ids + sequence.tokens
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "logprobs": [0.0] * (len(prompt_ids) - 1) + list(sequence.logprobs),
            "advantages": [0.0] * (len(prompt_ids) - 1) + [advantage] * len(sequence.tokens),
        },
    )

    training_client = training_clients(model)
    output = training_client.forward_backward([datum], loss_fn=loss_fn).result()
    assert output.loss_fn_outputs

    training_client.optim_step(tinker.AdamParams(learning_rate=1e-5)).result()
