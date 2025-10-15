import asyncio
import contextlib
import os
import platform
import threading
from collections.abc import Coroutine
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, TypeVar, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from tinker._exceptions import (
    APIStatusError,
    BadRequestError,
    ConflictError,
    UnprocessableEntityError,
)
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.telemetry import (
    MAX_BATCH_SIZE,
    MAX_QUEUE_SIZE,
    Telemetry,
    _is_telemetry_enabled,
    capture_exceptions,
    init_telemetry,
)
from tinker.lib.telemetry_provider import TelemetryProvider
from tinker.types.generic_event import GenericEvent
from tinker.types.session_end_event import SessionEndEvent
from tinker.types.session_start_event import SessionStartEvent
from tinker.types.telemetry_batch import TelemetryBatch
from tinker.types.telemetry_event import TelemetryEvent
from tinker.types.telemetry_response import TelemetryResponse
from tinker.types.unhandled_exception_event import UnhandledExceptionEvent

# pyright: reportMissingParameterType=false
# pyright: reportOptionalMemberAccess=false

T = TypeVar("T")


class MockEventLoopProvider:
    def __init__(self):
        self.loop = Mock()
        self.loop.call_soon_threadsafe = Mock()

    def get_loop(self):
        return self.loop

    def run_coroutine_threadsafe(self, coro: Coroutine[Any, Any, T]):
        future = Mock()
        future.result = Mock(return_value=asyncio.run(coro))
        return future


class MockAsyncTinkerProvider:
    def __init__(self):
        self.loop = Mock()
        self.callbacks = []
        self.loop.call_soon_threadsafe = Mock(side_effect=lambda cb: self.callbacks.append(cb))
        self._client = MagicMock()
        self.telemetry_send_mock = AsyncMock(return_value=TelemetryResponse(status="accepted"))
        self._client.telemetry.send = self.telemetry_send_mock
        self._client_cm = MagicMock()
        self._client_cm.__enter__ = Mock(return_value=self._client)
        self._client_cm.__exit__ = Mock(return_value=None)

    def execute_callbacks(self):
        for cb in self.callbacks:
            cb()
        self.callbacks.clear()

    def get_loop(self):
        return self.loop

    def run_coroutine_threadsafe(self, coro: Coroutine[Any, Any, T]):
        fut: ConcurrentFuture[Any] = ConcurrentFuture()

        async def _runner():
            try:
                result = await coro
                fut.set_result(result)
            except Exception as e:
                fut.set_exception(e)

        _ = asyncio.get_event_loop_policy().get_event_loop().create_task(_runner())
        return AwaitableConcurrentFuture(fut)

    def aclient(self, client_pool_type: ClientConnectionPoolType):
        _ = client_pool_type
        return self._client_cm


class TestTelemetryClass:
    def setup_method(self):
        self.tinker_provider = MockAsyncTinkerProvider()
        self.telemetry = Telemetry(self.tinker_provider, session_id="test-session-id")

    def teardown_method(self):
        if hasattr(self, "telemetry"):
            self.telemetry.stop()

    @pytest.mark.asyncio
    async def test_initialization(self):
        self.tinker_provider.execute_callbacks()
        assert self.telemetry._session_index == 1
        assert len(self.telemetry._queue) == 1
        assert isinstance(self.telemetry._queue[0], SessionStartEvent)
        assert isinstance(self.telemetry._flush_event, asyncio.Event)

    def test_log_single_event(self):
        event = self.telemetry._session_end_event()
        result = self.telemetry._log(event)
        assert result is True
        assert len(self.telemetry._queue) == 2
        assert self.telemetry._queue[-1] == event

    def test_log_multiple_events(self):
        event1 = self.telemetry._session_end_event()
        event2 = self.telemetry._exception_event(ValueError("test"), "ERROR")
        result = self.telemetry._log(event1, event2)
        assert result is True
        assert len(self.telemetry._queue) == 3
        assert self.telemetry._queue[-2] == event1
        assert self.telemetry._queue[-1] == event2

    def test_log_generic_event_default(self):
        idx_before = self.telemetry._session_index
        result = self.telemetry.log("test-event")
        assert result is True
        assert len(self.telemetry._queue) == 2
        event = self.telemetry._queue[-1]
        assert isinstance(event, GenericEvent)
        assert event.event == "GENERIC_EVENT"
        assert event.event_name == "test-event"
        assert event.severity == "INFO"
        assert event.event_data == {}
        assert event.event_session_index == idx_before
        assert isinstance(event.event_id, str) and event.event_id

    def test_log_generic_event_custom(self):
        idx_before = self.telemetry._session_index
        payload: dict[str, object] = {"a": 1, "b": "x"}
        result = self.telemetry.log("custom-event", event_data=payload, severity="WARNING")
        assert result is True
        assert len(self.telemetry._queue) == 2
        event = self.telemetry._queue[-1]
        assert isinstance(event, GenericEvent)
        assert event.event == "GENERIC_EVENT"
        assert event.event_name == "custom-event"
        assert event.severity == "WARNING"
        assert event.event_data == payload
        assert event.event_session_index == idx_before

    def test_log_queue_full(self):
        initial_size = len(self.telemetry._queue)
        events_to_add = MAX_QUEUE_SIZE - initial_size - 1
        for _ in range(events_to_add):
            self.telemetry._queue.append(self.telemetry._session_end_event())
        assert len(self.telemetry._queue) == MAX_QUEUE_SIZE - 1
        event1 = self.telemetry._session_end_event()
        event2 = self.telemetry._session_end_event()
        with patch("tinker.lib.telemetry.logger") as mock_logger:
            result = self.telemetry._log(event1, event2)
        assert result is False
        assert len(self.telemetry._queue) == MAX_QUEUE_SIZE - 1
        mock_logger.warning.assert_called_once_with("Telemetry queue full, dropping events")

    def test_batch_creation(self):
        events = [
            self.telemetry._session_start_event(),
            self.telemetry._session_end_event(),
        ]
        batch = self.telemetry._batch(cast(list[TelemetryEvent], events))
        assert isinstance(batch, TelemetryBatch)
        assert batch.platform == platform.system()
        assert batch.session_id == str(self.telemetry._session_id)
        assert batch.events == events
        assert batch.sdk_version is not None

    def test_log_exception_sync(self):
        try:
            raise RuntimeError("Test exception")
        except RuntimeError as e:
            with patch.object(self.telemetry, "_trigger_flush") as mock_trigger:
                result = self.telemetry.log_exception_sync(e, "ERROR")

        assert result is True
        assert len(self.telemetry._queue) == 2
        mock_trigger.assert_called_once()
        exception_event = self.telemetry._queue[-1]
        assert isinstance(exception_event, UnhandledExceptionEvent)
        assert exception_event.error_type == "RuntimeError"
        assert exception_event.error_message == "Test exception"

    def test_log_exception_sync_user_error(self):
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(400, request=request)
        error = BadRequestError(
            "Invalid request payload",
            response=response,
            body={"detail": "bad request"},
        )

        with patch.object(self.telemetry, "_trigger_flush") as mock_trigger:
            result = self.telemetry.log_exception_sync(error, "ERROR")

        assert result is True
        assert len(self.telemetry._queue) == 2
        mock_trigger.assert_called_once()
        generic_event = self.telemetry._queue[-1]
        assert isinstance(generic_event, GenericEvent)
        assert generic_event.event_name == "user_error"
        assert generic_event.severity == "WARNING"
        assert generic_event.event_data["error_type"] == "BadRequestError"
        assert generic_event.event_data["status_code"] == 400
        assert generic_event.event_data["message"] == "Invalid request payload"
        assert generic_event.event_data["body"] == {"detail": "bad request"}

    @pytest.mark.asyncio
    async def test_log_exception_async(self):
        try:
            raise RuntimeError("Test exception")
        except RuntimeError as e:
            with patch.object(self.telemetry, "_trigger_flush") as mock_trigger:
                result = await self.telemetry.log_exception(e, "ERROR")

        assert result is True
        assert len(self.telemetry._queue) == 2
        mock_trigger.assert_called_once()
        exception_event = self.telemetry._queue[-1]
        assert isinstance(exception_event, UnhandledExceptionEvent)
        assert exception_event.error_type == "RuntimeError"
        assert exception_event.error_message == "Test exception"

    def test_log_fatal_exception_sync(self):
        try:
            raise RuntimeError("Fatal error")
        except RuntimeError as e:
            with patch.object(self.telemetry, "_trigger_flush") as mock_trigger:
                with patch.object(
                    self.telemetry, "_wait_until_drained_sync", return_value=True
                ) as mock_wait:
                    result = self.telemetry.log_fatal_exception_sync(e, "CRITICAL")

        assert result is True
        assert len(self.telemetry._queue) == 3
        mock_trigger.assert_called_once()
        mock_wait.assert_called_once()
        exception_event = self.telemetry._queue[-2]
        assert isinstance(exception_event, UnhandledExceptionEvent)
        assert exception_event.severity == "CRITICAL"
        end_event = self.telemetry._queue[-1]
        assert isinstance(end_event, SessionEndEvent)

    def test_log_fatal_exception_sync_user_error(self):
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(422, request=request)
        error = UnprocessableEntityError(
            "Payload is invalid",
            response=response,
            body={"errors": ["invalid field"]},
        )

        with patch.object(self.telemetry, "_trigger_flush") as mock_trigger:
            with patch.object(
                self.telemetry, "_wait_until_drained_sync", return_value=True
            ) as mock_wait:
                result = self.telemetry.log_fatal_exception_sync(error, "ERROR")

        assert result is True
        assert len(self.telemetry._queue) == 3
        mock_trigger.assert_called_once()
        mock_wait.assert_called_once()
        generic_event = self.telemetry._queue[-2]
        assert isinstance(generic_event, GenericEvent)
        assert generic_event.event_name == "user_error"
        assert generic_event.severity == "WARNING"
        assert generic_event.event_data["error_type"] == "UnprocessableEntityError"
        assert generic_event.event_data["status_code"] == 422
        assert generic_event.event_data["message"] == "Payload is invalid"
        assert generic_event.event_data["body"] == {"errors": ["invalid field"]}
        end_event = self.telemetry._queue[-1]
        assert isinstance(end_event, SessionEndEvent)

    def test_log_exception_sync_timeout_not_user_error(self):
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(408, request=request)
        error = APIStatusError(
            "Request timed out",
            response=response,
            body={"error": "Request timed out"},
        )

        with patch.object(self.telemetry, "_trigger_flush") as mock_trigger:
            result = self.telemetry.log_exception_sync(error, "ERROR")

        assert result is True
        assert len(self.telemetry._queue) == 2
        mock_trigger.assert_called_once()
        exception_event = self.telemetry._queue[-1]
        assert isinstance(exception_event, UnhandledExceptionEvent)
        assert exception_event.error_type == "APIStatusError"

    def test_log_exception_sync_value_error_with_api_status_cause(self):
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(409, request=request)
        conflict_error = ConflictError(
            "Resource already exists",
            response=response,
            body={"detail": "Resource already exists"},
        )

        try:
            raise conflict_error
        except ConflictError as exc:
            try:
                raise ValueError("Wrapped user error") from exc
            except ValueError as outer:
                wrapped_error = outer

        with patch.object(self.telemetry, "_trigger_flush") as mock_trigger:
            result = self.telemetry.log_exception_sync(wrapped_error, "ERROR")

        assert result is True
        assert len(self.telemetry._queue) == 2
        mock_trigger.assert_called_once()
        generic_event = self.telemetry._queue[-1]
        assert isinstance(generic_event, GenericEvent)
        assert generic_event.event_name == "user_error"
        assert generic_event.event_data["status_code"] == 409
        assert generic_event.event_data["body"] == {"detail": "Resource already exists"}

    @pytest.mark.asyncio
    async def test_log_fatal_exception_async(self):
        try:
            raise RuntimeError("Fatal error")
        except RuntimeError as e:
            with patch.object(self.telemetry, "_trigger_flush") as mock_trigger:
                with patch.object(
                    self.telemetry, "_wait_until_drained", new_callable=AsyncMock, return_value=True
                ) as mock_wait:
                    result = await self.telemetry.log_fatal_exception(e, "CRITICAL")

        assert result is True
        assert len(self.telemetry._queue) == 3
        mock_trigger.assert_called_once()
        mock_wait.assert_called_once()
        exception_event = self.telemetry._queue[-2]
        assert isinstance(exception_event, UnhandledExceptionEvent)
        assert exception_event.severity == "CRITICAL"
        end_event = self.telemetry._queue[-1]
        assert isinstance(end_event, SessionEndEvent)

    def test_capture_exceptions_context_manager(self):
        with patch.object(self.telemetry, "_trigger_flush"):
            with pytest.raises(ValueError):
                with self.telemetry.capture_exceptions():
                    raise ValueError("Test error")

            assert len(self.telemetry._queue) == 2
            exception_event = self.telemetry._queue[-1]
            assert isinstance(exception_event, UnhandledExceptionEvent)
            assert exception_event.error_type == "ValueError"

    def test_capture_exceptions_context_manager_fatal(self):
        with patch.object(self.telemetry, "_trigger_flush"):
            with patch.object(self.telemetry, "_wait_until_drained_sync", return_value=True):
                with pytest.raises(ValueError):
                    with self.telemetry.capture_exceptions(fatal=True, severity="CRITICAL"):
                        raise ValueError("Fatal error")

                assert len(self.telemetry._queue) == 3
                exception_event = self.telemetry._queue[-2]
                assert isinstance(exception_event, UnhandledExceptionEvent)
                assert exception_event.severity == "CRITICAL"
                end_event = self.telemetry._queue[-1]
                assert isinstance(end_event, SessionEndEvent)

    @pytest.mark.asyncio
    async def test_acapture_exceptions_context_manager(self):
        with patch.object(self.telemetry, "_trigger_flush"):
            with pytest.raises(ValueError):
                async with self.telemetry.acapture_exceptions():
                    raise ValueError("Async test error")

            assert len(self.telemetry._queue) == 2
            exception_event = self.telemetry._queue[-1]
            assert isinstance(exception_event, UnhandledExceptionEvent)
            assert exception_event.error_type == "ValueError"

    @pytest.mark.asyncio
    async def test_acapture_exceptions_context_manager_fatal(self):
        with patch.object(self.telemetry, "_trigger_flush"):
            with patch.object(
                self.telemetry, "_wait_until_drained", new_callable=AsyncMock, return_value=True
            ):
                with pytest.raises(ValueError):
                    async with self.telemetry.acapture_exceptions(fatal=True):
                        raise ValueError("Async fatal error")
                assert len(self.telemetry._queue) == 3
                exception_event = self.telemetry._queue[-2]
                assert isinstance(exception_event, UnhandledExceptionEvent)
                end_event = self.telemetry._queue[-1]
                assert isinstance(end_event, SessionEndEvent)


class TestTelemetryEnvironment:
    @pytest.mark.parametrize(
        "env_value,expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("Yes", True),
            ("on", True),
            ("ON", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
            ("", False),
            ("random", False),
        ],
    )
    def test_is_telemetry_enabled(self, env_value, expected):
        with patch.dict(os.environ, {"TINKER_TELEMETRY": env_value}):
            assert _is_telemetry_enabled() == expected

    def test_is_telemetry_enabled_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _is_telemetry_enabled() is True

    def test_init_telemetry_enabled(self):
        with patch.dict(os.environ, {"TINKER_TELEMETRY": "1"}):
            tinker_provider = MockAsyncTinkerProvider()
            telemetry = init_telemetry(tinker_provider, session_id="test-session-id")
            assert telemetry is not None
            assert isinstance(telemetry, Telemetry)
            telemetry.stop()

    def test_init_telemetry_disabled(self):
        with patch.dict(os.environ, {"TINKER_TELEMETRY": "0"}):
            tinker_provider = MockAsyncTinkerProvider()
            telemetry = init_telemetry(tinker_provider, session_id="test-session-id")
            assert telemetry is None

    def test_init_telemetry_with_exception(self):
        with patch.dict(os.environ, {"TINKER_TELEMETRY": "1"}):
            tinker_provider = Mock()
            tinker_provider.get_loop.side_effect = Exception("Init error")
            with patch("tinker.lib.telemetry.logger") as mock_logger:
                telemetry = init_telemetry(tinker_provider, session_id="test-session-id")
            assert telemetry is None
            mock_logger.warning.assert_called_once()
            assert "Error initializing telemetry" in str(mock_logger.warning.call_args)


class TestCaptureExceptionsDecorator:
    class MockTelemetryProvider:
        def __init__(self):
            self.telemetry = Mock()
            self.telemetry.capture_exceptions = Mock()
            self.telemetry.acapture_exceptions = Mock()

        def get_telemetry(self):
            return self.telemetry

    def test_decorator_on_sync_function(self):
        provider = self.MockTelemetryProvider()

        @capture_exceptions
        def test_func(self):
            return "success"

        provider.telemetry.capture_exceptions.return_value.__enter__ = Mock()
        provider.telemetry.capture_exceptions.return_value.__exit__ = Mock(return_value=False)
        result = test_func(provider)
        assert result == "success"
        provider.telemetry.capture_exceptions.assert_called_once_with(fatal=False, severity="ERROR")

    def test_decorator_on_sync_function_with_exception(self):
        provider = self.MockTelemetryProvider()

        @capture_exceptions(fatal=True, severity="CRITICAL")
        def test_func(self):
            raise ValueError("Test error")

        provider.telemetry.capture_exceptions.return_value.__enter__ = Mock()
        provider.telemetry.capture_exceptions.return_value.__exit__ = Mock(return_value=False)
        with pytest.raises(ValueError, match="Test error"):
            test_func(provider)
        provider.telemetry.capture_exceptions.assert_called_once_with(
            fatal=True, severity="CRITICAL"
        )

    @pytest.mark.asyncio
    async def test_decorator_on_async_function(self):
        provider = self.MockTelemetryProvider()

        @capture_exceptions
        async def test_func(self):
            return "async success"

        async_cm = AsyncMock()
        async_cm.__aenter__ = AsyncMock(return_value=None)
        async_cm.__aexit__ = AsyncMock(return_value=False)
        provider.telemetry.acapture_exceptions.return_value = async_cm
        result = await test_func(provider)
        assert result == "async success"
        provider.telemetry.acapture_exceptions.assert_called_once_with(
            fatal=False, severity="ERROR"
        )

    @pytest.mark.asyncio
    async def test_decorator_on_async_function_with_exception(self):
        provider = self.MockTelemetryProvider()

        @capture_exceptions(severity="WARNING")
        async def test_func(self):
            raise RuntimeError("Async error")

        async_cm = AsyncMock()
        async_cm.__aenter__ = AsyncMock(return_value=None)
        async_cm.__aexit__ = AsyncMock(return_value=False)
        provider.telemetry.acapture_exceptions.return_value = async_cm
        with pytest.raises(RuntimeError, match="Async error"):
            await test_func(provider)
        provider.telemetry.acapture_exceptions.assert_called_once_with(
            fatal=False, severity="WARNING"
        )

    def test_decorator_without_telemetry_provider(self):
        @capture_exceptions
        def test_func():
            return "no provider"

        result = test_func()
        assert result == "no provider"

    def test_decorator_with_non_provider_self(self):
        class NonProvider:
            @capture_exceptions
            def test_method(self):
                return "not a provider"

        obj = NonProvider()
        result = obj.test_method()
        assert result == "not a provider"

    def test_decorator_as_plain_decorator(self):
        @capture_exceptions
        def test_func():
            return "plain decorator"

        result = test_func()
        assert result == "plain decorator"

    def test_decorator_with_parentheses(self):
        @capture_exceptions()
        def test_func():
            return "decorator with parens"

        result = test_func()
        assert result == "decorator with parens"

    def test_decorator_on_inner_sync_function_closing_over_self(self):
        provider = self.MockTelemetryProvider()

        class Wrapper:
            def __init__(self, p: Any):
                self.p: Any = p

            @capture_exceptions
            def outer(self) -> str:
                @capture_exceptions
                def inner() -> str:
                    # reference `self` so it is captured in the closure
                    return "ok" if self else "bad"

                return inner()

            def get_telemetry(self) -> Telemetry | None:
                return self.p.get_telemetry()

        wrapper = Wrapper(provider)
        provider.telemetry.capture_exceptions.return_value.__enter__ = Mock()
        provider.telemetry.capture_exceptions.return_value.__exit__ = Mock(return_value=False)
        result = wrapper.outer()
        assert result == "ok"
        # Called twice: once for outer, once for inner via closure lookup
        assert provider.telemetry.capture_exceptions.call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_on_inner_async_function_closing_over_self(self):
        provider = self.MockTelemetryProvider()

        class Wrapper:
            def __init__(self, p: Any):
                self.p: Any = p

            @capture_exceptions
            async def outer(self) -> str:
                @capture_exceptions
                async def inner() -> str:
                    # reference `self` so it is captured in the closure
                    return "ok-async" if self else "bad"

                return await inner()

            def get_telemetry(self) -> Telemetry | None:
                return self.p.get_telemetry()

        wrapper = Wrapper(provider)
        async_cm = AsyncMock()
        async_cm.__aenter__ = AsyncMock(return_value=None)
        async_cm.__aexit__ = AsyncMock(return_value=False)
        provider.telemetry.acapture_exceptions.return_value = async_cm
        result = await wrapper.outer()
        assert result == "ok-async"
        # Called twice: once for outer, once for inner via closure lookup
        assert provider.telemetry.acapture_exceptions.call_count == 2


class TestTelemetryFlush:
    def setup_method(self):
        self.tinker_provider = MockAsyncTinkerProvider()
        self.telemetry = Telemetry(self.tinker_provider, session_id="test-session-id")

    def teardown_method(self):
        if hasattr(self, "telemetry"):
            self.telemetry.stop()

    def test_flush_empty_queue(self):
        self.telemetry._queue.clear()
        with patch.object(
            self.telemetry, "_send_batch_with_retry", new_callable=AsyncMock
        ) as mock_send:
            asyncio.run(self.telemetry._flush())
        mock_send.assert_not_called()

    def test_flush_small_batch(self):
        for _ in range(5):
            _ = self.telemetry._log(self.telemetry._session_end_event())
        with patch.object(
            self.telemetry, "_send_batch_with_retry", new_callable=AsyncMock
        ) as mock_send:
            asyncio.run(self.telemetry._flush())
        assert len(self.telemetry._queue) == 0
        assert mock_send.call_count == 1

    def test_flush_large_batch(self):
        for _ in range(MAX_BATCH_SIZE + 10):
            _ = self.telemetry._log(self.telemetry._session_end_event())
        with patch.object(
            self.telemetry, "_send_batch_with_retry", new_callable=AsyncMock
        ) as mock_send:
            asyncio.run(self.telemetry._flush())
        assert len(self.telemetry._queue) == 0
        assert mock_send.call_count == 2

    def test_counters_and_wait(self):
        initial_push = self.telemetry._push_counter
        for _ in range(3):
            _ = self.telemetry._log(self.telemetry._session_end_event())
        assert self.telemetry._push_counter == initial_push + 3
        with patch.object(
            self.telemetry,
            "_send_batch_with_retry",
            new_callable=AsyncMock,
            return_value=TelemetryResponse(status="accepted"),
        ):
            asyncio.run(self.telemetry._flush())
        assert self.telemetry._flush_counter >= initial_push + 3

    @pytest.mark.asyncio
    async def test_log_exception_sync_from_event_loop_protection(self):
        with patch("tinker.lib.telemetry._current_loop") as mock_current_loop:
            mock_current_loop.return_value = Mock()
            try:
                raise ValueError("Test error")
            except ValueError as e:
                result = self.telemetry.log_exception_sync(e)
            assert result is True


class TestTelemetryProviderProtocol:
    def test_telemetry_provider_protocol(self):
        class ValidProvider:
            def get_telemetry(self):
                return None

        class InvalidProvider:
            pass

        assert isinstance(ValidProvider(), TelemetryProvider)
        assert not isinstance(InvalidProvider(), TelemetryProvider)


class TestSyncContextManager:
    @pytest.mark.asyncio
    async def test_send_batch_uses_sync_context_manager(self):
        tinker_provider = MockAsyncTinkerProvider()
        telemetry = Telemetry(tinker_provider, session_id="test-session-id")
        events = [telemetry._session_start_event()]
        batch = telemetry._batch(cast(list[TelemetryEvent], events))
        result = await telemetry._send_batch(batch)
        tinker_provider._client_cm.__enter__.assert_called_once()
        tinker_provider._client_cm.__exit__.assert_called_once()
        tinker_provider.telemetry_send_mock.assert_called_once()
        assert result.status == "accepted"
        telemetry.stop()


class TestCrossLoopSafety:
    @pytest.mark.asyncio
    async def test_trigger_flush_from_different_loop(self):
        tinker_provider = MockAsyncTinkerProvider()
        telemetry = Telemetry(tinker_provider, session_id="test-session-id")
        tinker_provider.execute_callbacks()

        def trigger_from_thread():
            telemetry._trigger_flush()

        thread = threading.Thread(target=trigger_from_thread)
        thread.start()
        thread.join()
        tinker_provider.execute_callbacks()
        assert telemetry._flush_event.is_set()
        telemetry.stop()

    @pytest.mark.asyncio
    async def test_periodic_flush_with_asyncio_event(self):
        tinker_provider = MockAsyncTinkerProvider()
        telemetry = Telemetry(tinker_provider, session_id="test-session-id")
        tinker_provider.execute_callbacks()
        telemetry._log(telemetry._session_end_event())
        with patch.object(telemetry, "_flush", new_callable=AsyncMock) as mock_flush:
            telemetry._flush_event.set()
            task = asyncio.create_task(telemetry._periodic_flush())
            await asyncio.sleep(0.1)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            mock_flush.assert_called()
        telemetry.stop()
