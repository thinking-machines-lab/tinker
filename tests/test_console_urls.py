from tinker.types import SaveWeightsForSamplerResponse, SaveWeightsResponse


def test_save_weights_console_url() -> None:
    response = SaveWeightsResponse(path="tinker://session-123:train:1/weights/0001")

    assert (
        response.get_console_url() == "https://tinker.thinkingmachines.ai/checkpoints/"
        "session-123%3Atrain%3A1/weights%2F0001"
    )


def test_save_weights_for_sampler_console_url_includes_full_checkpoint_id() -> None:
    response = SaveWeightsForSamplerResponse(
        path="tinker://session-123:train:1/sampler_weights/0001"
    )

    assert (
        response.get_console_url() == "https://tinker.thinkingmachines.ai/checkpoints/"
        "session-123%3Atrain%3A1/sampler_weights%2F0001"
    )
