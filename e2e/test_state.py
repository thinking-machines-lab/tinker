"""Checkpoint a training run and pick it up again.

A checkpoint that loads but does not restore looks exactly like one that works,
right up until the loss curve is wrong.
"""

import pytest

import tinker

PROMPT = "The Tinker end to end suite checkpoints its state.\nAnswer:"
STEPS = 4
LEARNING_RATE = 1e-3
TTL_SECONDS = 3600
# Trainer noise between two forward passes on identical weights.
LOSS_TOLERANCE = 0.05


def _datum(tokenizer) -> tinker.Datum:
    tokens = tokenizer.encode(PROMPT)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * (len(tokens) - 1),
        },
    )


def _loss(training_client, datum) -> float:
    return training_client.forward([datum], loss_fn="cross_entropy").result().metrics["loss:sum"]


def _train(training_client, datum, steps: int = STEPS) -> None:
    for _ in range(steps):
        training_client.forward_backward([datum], loss_fn="cross_entropy").result()
        training_client.optim_step(tinker.AdamParams(learning_rate=LEARNING_RATE)).result()


@pytest.mark.gpu
def test_saved_state_reloads_the_same_weights(
    service_client, rest_client, new_training_client, tokenizers, model: str
) -> None:
    """Train away from base, save, then pick the checkpoint up both ways.

    `load_state` is only accepted as a client's first call: the server rejects it
    once the sequence has moved, and even a forward pass counts. So it gets a
    fresh client rather than the one that saved.
    """
    datum = _datum(tokenizers(model))
    training_client = new_training_client(model)

    base_loss = _loss(training_client, datum)
    _train(training_client, datum)
    trained_loss = _loss(training_client, datum)
    assert base_loss - trained_loss > LOSS_TOLERANCE, (
        f"training moved nothing, so a reload proves nothing: {base_loss} -> {trained_loss}"
    )

    saved = training_client.save_state(name="e2e-state", ttl_seconds=TTL_SECONDS).result()
    assert saved.path.startswith("tinker://"), saved.path

    try:
        info = rest_client.get_weights_info_by_tinker_path(saved.path).result()
        assert info.base_model == model
        assert info.is_lora

        built_loss = _loss(service_client.create_training_client_from_state(saved.path), datum)
        assert abs(built_loss - trained_loss) < LOSS_TOLERANCE, (
            f"create_training_client_from_state gave {built_loss}, saved at {trained_loss}, "
            f"tolerance {LOSS_TOLERANCE}"
        )

        loaded = new_training_client(model)
        loaded.load_state(saved.path).result()
        loaded_loss = _loss(loaded, datum)
        assert abs(loaded_loss - trained_loss) < LOSS_TOLERANCE, (
            f"load_state gave {loaded_loss}, saved at {trained_loss}, tolerance {LOSS_TOLERANCE}"
        )
    finally:
        rest_client.delete_checkpoint_from_tinker_path(saved.path).result()


@pytest.mark.gpu
def test_resuming_with_the_optimizer_continues_the_descent(
    service_client, rest_client, new_training_client, tokenizers, model: str
) -> None:
    """Adam's moments are state too: without them the next step lands elsewhere."""
    datum = _datum(tokenizers(model))
    training_client = new_training_client(model)

    _train(training_client, datum)
    saved = training_client.save_state(name="e2e-optim-state", ttl_seconds=TTL_SECONDS).result()

    try:
        # The reference: one more step on the client that never left.
        _train(training_client, datum, steps=1)
        continued = _loss(training_client, datum)

        resumed = service_client.create_training_client_from_state_with_optimizer(saved.path)
        _train(resumed, datum, steps=1)

        resumed_loss = _loss(resumed, datum)
        assert abs(resumed_loss - continued) < LOSS_TOLERANCE, (
            f"resumed to {resumed_loss}, the client that never left reached {continued}, "
            f"tolerance {LOSS_TOLERANCE}"
        )
    finally:
        rest_client.delete_checkpoint_from_tinker_path(saved.path).result()
