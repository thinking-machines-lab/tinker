"""Save a checkpoint, sample from it, delete it."""

import httpx
import pytest

import tinker

PROMPT = "Name one prime number.\nAnswer:"
# TTL is a backstop: a run that dies before teardown should not keep the weights.
TTL_SECONDS = 3600


@pytest.mark.gpu
def test_saved_weights_can_be_sampled_from_and_deleted(
    service_client, rest_client, training_clients, tokenizers, model: str
) -> None:
    training_client = training_clients(model)

    saved = training_client.save_weights_for_sampler(
        name="e2e-checkpoint", ttl_seconds=TTL_SECONDS
    ).result()
    path = saved.path
    assert path.startswith("tinker://"), path

    try:
        run_id = tinker.ParsedCheckpointTinkerPath.from_tinker_path(path).training_run_id
        listed = rest_client.list_checkpoints(run_id).result()
        assert any(c.tinker_path == path for c in listed.checkpoints), (
            "saved checkpoint is not listed"
        )

        # The account wide listing has to show it too, while it is still alive.
        user_checkpoints = rest_client.list_user_checkpoints(limit=100).result().checkpoints
        assert any(c.tinker_path == path for c in user_checkpoints), (
            "saved checkpoint is missing from the account wide listing"
        )

        # Fetched, not pattern matched: a signed url that 403s is not a download.
        archive = rest_client.get_checkpoint_archive_url_from_tinker_path(path).result()
        assert archive.url.startswith("https://"), archive.url
        head = httpx.head(archive.url, follow_redirects=True, timeout=60)
        assert head.status_code == 200, f"archive url answered {head.status_code}"
        rest_client.set_checkpoint_ttl_from_tinker_path(path, TTL_SECONDS * 2).result()

        sampling_client = service_client.create_sampling_client(model_path=path)
        response = sampling_client.sample(
            prompt=tinker.ModelInput.from_ints(tokenizers(model).encode(PROMPT)),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=4, temperature=0.0, seed=0),
        ).result()
        assert response.sequences[0].tokens
    finally:
        rest_client.delete_checkpoint_from_tinker_path(path).result()

    remaining = rest_client.list_checkpoints(run_id).result()
    assert not any(c.tinker_path == path for c in remaining.checkpoints), (
        "checkpoint outlived the test"
    )
