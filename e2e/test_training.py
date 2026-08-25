"""Supervised fine-tuning: one datum, trained until the loss comes down, then
sampled from the trained weights.

Asserts the path works, not that the model learned well: what it produces is the
model's business, and asserting on that would fail while the service is healthy.
"""

import pytest
from conftest import LORA_RANK

import tinker

PROMPT = "Which environment variable does the Tinker SDK run to fetch dynamic credentials?\nAnswer:"
ANSWER = " TINKER_CREDENTIAL_CMD"
MAX_STEPS = 12
LEARNING_RATE = 1e-3
LOSS_TARGET = 0.5


def _sft_datum(tokenizer) -> tinker.Datum:
    prompt_ids = tokenizer.encode(PROMPT)
    answer_ids = tokenizer.encode(ANSWER, add_special_tokens=False)
    tokens = prompt_ids + answer_ids

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            # Score the answer only, not the question.
            "weights": [0.0] * (len(prompt_ids) - 1) + [1.0] * len(answer_ids),
        },
    )


@pytest.mark.gpu
def test_training_lowers_the_loss_and_serves_the_result(
    training_clients, tokenizers, model: str
) -> None:
    tokenizer = tokenizers(model)
    training_client = training_clients(model)
    datum = _sft_datum(tokenizer)

    losses = []
    for _ in range(MAX_STEPS):
        output = training_client.forward_backward([datum], loss_fn="cross_entropy").result()
        losses.append(output.metrics["loss:sum"])
        if losses[-1] < LOSS_TARGET:
            break
        training_client.optim_step(tinker.AdamParams(learning_rate=LEARNING_RATE)).result()

    assert losses[-1] < LOSS_TARGET, f"loss never reached {LOSS_TARGET} in {MAX_STEPS}: {losses}"

    # Ephemeral weights: no path, never listed, reclaimed by the server.
    trained = training_client.save_weights_and_get_sampling_client()
    response = trained.sample(
        prompt=tinker.ModelInput.from_ints(tokenizer.encode(PROMPT)),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=12, temperature=0.0, seed=0),
    ).result()

    assert response.sequences[0].tokens, "the trained weights produced nothing"


def test_training_client_describes_its_model(training_clients, model: str) -> None:
    info = training_clients(model).get_info()
    assert info.model_id
    assert info.is_lora
    assert info.lora_rank == LORA_RANK
    assert info.model_data.tokenizer_id


@pytest.mark.gpu
def test_forward_reports_per_token_loss(training_clients, tokenizers, model: str) -> None:
    """`forward` skips the backward pass, so it needs no optimizer step."""
    datum = _sft_datum(tokenizers(model))
    output = training_clients(model).forward([datum], loss_fn="cross_entropy").result()

    elementwise = output.loss_fn_outputs[0]["elementwise_loss"].data
    assert len(elementwise) == len(datum.loss_fn_inputs["target_tokens"].data)
    assert all(loss >= 0 for loss in elementwise)
    assert output.metrics["loss:sum"] > 0
