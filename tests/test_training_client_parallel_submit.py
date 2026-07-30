from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import tinker.lib.public_interfaces.training_client as training_client_module
from tinker import APIConnectionError, types
from tinker.lib.public_interfaces.training_client import TrainingClient


def _datum(token: int) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints([token, token + 1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=[token + 1],
                dtype="int64",
                shape=[1],
            )
        },
    )


async def test_parallel_submit_attempts_gate_when_later_chunk_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[int] = []

    class Holder:
        _client_config = SimpleNamespace(
            parallel_fwdbwd_chunks=True,
            proto_write_fwdbwd=False,
        )

        async def execute_with_retries(
            self,
            _send: Any,
            request_id: int,
            _data: list[types.Datum],
        ) -> object:
            seq_id = request_id + 1
            submitted.append(seq_id)
            await asyncio.sleep(0)
            if seq_id == 2:
                raise APIConnectionError(
                    request=httpx.Request(
                        "POST",
                        "https://trainer.test/api/v1/forward_backward",
                    )
                )
            return object()

        def run_coroutine_threadsafe(self, coroutine: Any) -> asyncio.Task[Any]:
            return asyncio.ensure_future(coroutine)

        def get_telemetry(self) -> None:
            return None

    class FakeAPIFuture:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    client = TrainingClient(
        holder=Holder(),  # type: ignore[arg-type]
        model_seq_id=1,
        model_id="model",
    )

    def chunked_requests(
        _data: list[types.Datum],
    ) -> list[tuple[int, list[types.Datum]]]:
        return [
            (0, [_datum(1)]),
            (1, [_datum(10)]),
            (2, [_datum(20)]),
        ]

    monkeypatch.setattr(
        client,
        "_chunked_requests",
        chunked_requests,
    )
    monkeypatch.setattr(
        training_client_module,
        "_APIFuture",
        FakeAPIFuture,
    )

    with pytest.raises(APIConnectionError):
        await client._run_fwd_bwd(
            [_datum(1)],
            "cross_entropy",
            None,
            forward_only=False,
        )

    assert 2 in submitted
    assert 3 in submitted
    assert 1 in submitted
