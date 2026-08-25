"""The async half, which has its own plumbing and no other coverage.

One cheap call proves the path is wired; the sync tests carry the behaviour.
Driven by `asyncio.run` so the CI venv stays the wheel plus pytest.
"""

import asyncio


def test_async_calls_reach_the_service(service_client) -> None:
    rest_client = service_client.create_rest_client()

    async def call_both():
        # whoami has no _async twin: its APIFuture is awaitable directly.
        who = await rest_client.whoami()
        capabilities = await service_client.get_server_capabilities_async()
        return who, capabilities

    who, capabilities = asyncio.run(call_both())

    assert who.user_urn.startswith("tml:organization_user:"), who.user_urn
    assert capabilities.supported_models
