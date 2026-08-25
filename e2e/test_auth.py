"""Authentication against a deployed Tinker."""

import pytest
from conftest import resolve_env

import tinker


def test_whoami_identifies_the_caller(rest_client) -> None:
    who = rest_client.whoami().result()
    assert who.user_urn.startswith("tml:organization_user:"), who.user_urn
    # Only user backed principals carry an email: a service key would drop it.
    assert who.email


def test_a_rejected_key_raises_authentication_error() -> None:
    # Well-formed but not a real key: the SDK checks the prefix before sending.
    client = tinker.ServiceClient(api_key="tml-" + "0" * 40, base_url=resolve_env().base_url)
    with pytest.raises(tinker.AuthenticationError):
        client.create_rest_client().whoami().result()
