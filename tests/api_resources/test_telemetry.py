# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tinker import Tinker, AsyncTinker
from tests.utils import assert_matches_type
from tinker.types import TelemetryResponse
from tinker._utils import parse_datetime

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestTelemetry:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_send(self, client: Tinker) -> None:
        telemetry = client.telemetry.send(
            events=[
                {
                    "event": "SESSION_START",
                    "event_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "event_session_index": 0,
                    "severity": "DEBUG",
                    "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                }
            ],
            platform="Linux",
            sdk_version="1.2.3",
            session_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(TelemetryResponse, telemetry, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_send(self, client: Tinker) -> None:
        response = client.telemetry.with_raw_response.send(
            events=[
                {
                    "event": "SESSION_START",
                    "event_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "event_session_index": 0,
                    "severity": "DEBUG",
                    "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                }
            ],
            platform="Linux",
            sdk_version="1.2.3",
            session_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        telemetry = response.parse()
        assert_matches_type(TelemetryResponse, telemetry, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_send(self, client: Tinker) -> None:
        with client.telemetry.with_streaming_response.send(
            events=[
                {
                    "event": "SESSION_START",
                    "event_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "event_session_index": 0,
                    "severity": "DEBUG",
                    "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                }
            ],
            platform="Linux",
            sdk_version="1.2.3",
            session_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            telemetry = response.parse()
            assert_matches_type(TelemetryResponse, telemetry, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncTelemetry:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_send(self, async_client: AsyncTinker) -> None:
        telemetry = await async_client.telemetry.send(
            events=[
                {
                    "event": "SESSION_START",
                    "event_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "event_session_index": 0,
                    "severity": "DEBUG",
                    "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                }
            ],
            platform="Linux",
            sdk_version="1.2.3",
            session_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )
        assert_matches_type(TelemetryResponse, telemetry, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_send(self, async_client: AsyncTinker) -> None:
        response = await async_client.telemetry.with_raw_response.send(
            events=[
                {
                    "event": "SESSION_START",
                    "event_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "event_session_index": 0,
                    "severity": "DEBUG",
                    "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                }
            ],
            platform="Linux",
            sdk_version="1.2.3",
            session_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        telemetry = await response.parse()
        assert_matches_type(TelemetryResponse, telemetry, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_send(self, async_client: AsyncTinker) -> None:
        async with async_client.telemetry.with_streaming_response.send(
            events=[
                {
                    "event": "SESSION_START",
                    "event_id": "182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
                    "event_session_index": 0,
                    "severity": "DEBUG",
                    "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                }
            ],
            platform="Linux",
            sdk_version="1.2.3",
            session_id="182bd5e5-6e1a-4fe4-a799-aa6d9a6ab26e",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            telemetry = await response.parse()
            assert_matches_type(TelemetryResponse, telemetry, path=["response"])

        assert cast(Any, response.is_closed) is True
