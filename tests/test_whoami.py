"""Tests for RestClient.whoami's JWT-claims-based identity resolution."""

import base64
import json

import pytest

from tinker.lib._jwt_auth import jwt_claims
from tinker.lib.public_interfaces.rest_client import _whoami_response_from_jwt


def _make_jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def test_jwt_claims_roundtrip():
    claims = {"sub": "tml:organization_user:abc", "email": "user@example.com", "exp": 123}
    assert jwt_claims(_make_jwt(claims)) == claims


def test_jwt_claims_invalid_token_raises():
    with pytest.raises(ValueError):
        jwt_claims("not-a-jwt")


def test_whoami_response_from_jwt_with_email():
    jwt = _make_jwt({"sub": "tml:organization_user:abc", "email": "user@example.com"})
    response = _whoami_response_from_jwt(jwt)
    assert response.user_urn == "tml:organization_user:abc"
    assert response.email == "user@example.com"


def test_whoami_response_from_jwt_coerces_empty_email_to_none():
    # Non-user-backed principals carry an empty email claim.
    jwt = _make_jwt({"sub": "tml:organization_user:abc", "email": ""})
    assert _whoami_response_from_jwt(jwt).email is None


def test_whoami_response_from_jwt_missing_email_claim():
    jwt = _make_jwt({"sub": "tml:organization_user:abc"})
    assert _whoami_response_from_jwt(jwt).email is None
