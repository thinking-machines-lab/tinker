"""Whether the weights coming back are the ones we asked for.

The rest of the suite asserts the calls succeed. These catch the failure that
keeps shipping: training that does not load, sampling that falls back to base.
Ported from personal/andrii/tinker_tests.
"""

import pytest

import tinker

TEXT = "Tinker training is working!"
OVERFIT_STEPS = 30
OVERFIT_LR = 1e-3
# Low enough that the model can only be reciting the sequence back.
OVERFIT_LOSS_TARGET = 0.05
TTL_SECONDS = 3600

LOGPROB_TEXT = (
    "The mitochondrion is often called the powerhouse of the cell because it "
    "generates most of the cell's supply of adenosine triphosphate, which is used "
    "as a source of chemical energy. Beyond supplying cellular energy, mitochondria "
    "are involved in signaling, cellular differentiation, and the control of the "
    "cell cycle and cell growth."
)
# Kernels differ, but the same weights land far closer than different ones do.
LOGPROB_TOLERANCE = 0.1


@pytest.mark.gpu
def test_saved_weights_are_what_the_sampler_serves(
    service_client, rest_client, new_training_client, tokenizers, model: str
) -> None:
    """Overfit one sequence, save it, load it back by path, and make it recite.

    Base weights cannot continue TEXT from its first token, so a pass means the
    sampler loaded our checkpoint rather than the model underneath it.
    """
    tokenizer = tokenizers(model)
    tokens = tokenizer.encode(TEXT)
    training_client = new_training_client(model)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * (len(tokens) - 1),
        },
    )

    losses = []
    for _ in range(OVERFIT_STEPS):
        output = training_client.forward_backward([datum], loss_fn="cross_entropy").result()
        losses.append(output.metrics["loss:sum"] / (len(tokens) - 1))
        if losses[-1] < OVERFIT_LOSS_TARGET:
            break
        training_client.optim_step(tinker.AdamParams(learning_rate=OVERFIT_LR)).result()

    assert losses[-1] < OVERFIT_LOSS_TARGET, (
        f"never overfit in {OVERFIT_STEPS} steps, so the sampling half proves nothing: {losses}"
    )

    # Through an explicit path, which is where the weights went missing before.
    saved = training_client.save_weights_for_sampler(
        name="e2e-overfit", ttl_seconds=TTL_SECONDS
    ).result()

    try:
        response = (
            service_client.create_sampling_client(model_path=saved.path)
            .sample(
                prompt=tinker.ModelInput.from_ints(tokens[:1]),
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=len(tokens) + 4, temperature=0.0, seed=0
                ),
            )
            .result()
        )
    finally:
        rest_client.delete_checkpoint_from_tinker_path(saved.path).result()

    sampled = tokenizer.decode(response.sequences[0].tokens)
    expected = tokenizer.decode(tokens[1:])
    assert sampled.startswith(expected), (
        f"sampler did not serve the trained weights: expected {expected!r}, got {sampled!r}"
    )


@pytest.mark.gpu
def test_trainer_and_sampler_agree_on_logprobs(
    service_client, forward_only_clients, tokenizers, model: str
) -> None:
    """A zero init LoRA is a no-op, so both halves should be reading base weights."""
    tokenizer = tokenizers(model)
    tokens = tokenizer.encode(LOGPROB_TEXT)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * (len(tokens) - 1),
        },
    )
    forward = forward_only_clients(model).forward([datum], loss_fn="cross_entropy").result()
    trainer = forward.loss_fn_outputs[0]["logprobs"].tolist()

    # Index 0 has no preceding token, so drop it to line up with tokens[1:].
    sampler = (
        service_client.create_sampling_client(base_model=model)
        .compute_logprobs(tinker.ModelInput.from_ints(tokens))
        .result()[1:]
    )
    assert len(trainer) == len(sampler), (
        f"the two halves scored different token counts: {len(trainer)} vs {len(sampler)}"
    )

    pairs = [
        (a, b) for a, b in zip(trainer, sampler, strict=True) if a is not None and b is not None
    ]
    assert len(pairs) > len(tokens) // 2, "too few comparable tokens to conclude anything"

    drift = sum(abs(a - b) for a, b in pairs) / len(pairs)
    assert drift < LOGPROB_TOLERANCE, (
        f"trainer and sampler disagree by {drift:.4f} per token, so they are not "
        f"reading the same weights"
    )
