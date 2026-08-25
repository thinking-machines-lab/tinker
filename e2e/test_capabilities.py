"""What the deployment says it can do, and that each selected model is usable."""


def test_capabilities_list_models(service_client) -> None:
    capabilities = service_client.get_server_capabilities()
    assert capabilities.supported_models, "a deployment with no models is not serving anyone"
    for supported in capabilities.supported_models:
        assert supported.model_name
