"""SDK-side tests for proto request serialization (the send path).

The bulk encode-then-parse round-trip is property-tested with hypothesis
(varying tensor dtype × shape × chunk mix); edge cases that exercise specific
code paths (unsupported dtype, sparse CSR, the wire-level Content-Type
integration) stay hand-crafted.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from tinker import types
from tinker.proto import tinker_public_pb2 as public_pb
from tinker.proto.request_conv import forward_backward_request_to_proto


def _roundtrip(msg: public_pb.ForwardBackwardRequest) -> public_pb.ForwardBackwardRequest:
    parsed = public_pb.ForwardBackwardRequest()
    parsed.ParseFromString(msg.SerializeToString())
    return parsed


def _make_request(
    *,
    data: list[types.Datum],
    loss_fn: types.LossFnType = "cross_entropy",
    loss_fn_config: dict[str, float] | None = None,
    model_id: str = "m-test",
    seq_id: int | None = 7,
) -> types.ForwardBackwardRequest:
    return types.ForwardBackwardRequest(
        forward_backward_input=types.ForwardBackwardInput(
            data=data, loss_fn=loss_fn, loss_fn_config=loss_fn_config
        ),
        model_id=model_id,
        seq_id=seq_id,
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


_FINITE_FLOAT32 = st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)
_LOSS_FN: st.SearchStrategy[types.LossFnType] = st.sampled_from(
    ["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]
)


@st.composite
def _datum(draw: DrawFn) -> types.Datum:
    """Random Datum: 1-2 EncodedTextChunks (optionally + image), dense tensors."""
    n_chunks = draw(st.integers(min_value=1, max_value=2))
    chunks: list[types.ModelInputChunk] = []
    for _ in range(n_chunks):
        t = draw(st.integers(min_value=1, max_value=4))
        tokens = draw(
            st.lists(
                st.integers(min_value=0, max_value=32000),
                min_size=t,
                max_size=t,
            )
        )
        chunks.append(types.EncodedTextChunk(tokens=tokens))
    if draw(st.booleans()):
        img_size = draw(st.integers(min_value=4, max_value=64))
        img_bytes = draw(st.binary(min_size=img_size, max_size=img_size))
        # Cover both expected_tokens=None and expected_tokens=int paths.
        expected_tokens: int | None = (
            draw(st.integers(min_value=1, max_value=64)) if draw(st.booleans()) else None
        )
        chunks.append(
            types.ImageChunk(
                data=img_bytes,
                format=draw(st.sampled_from(["png", "jpeg"])),
                expected_tokens=expected_tokens,
            )
        )

    n = draw(st.integers(min_value=1, max_value=4))
    target_tokens = draw(
        st.lists(st.integers(min_value=0, max_value=32000), min_size=n, max_size=n)
    )
    weights = draw(st.lists(_FINITE_FLOAT32, min_size=n, max_size=n))
    return types.Datum(
        model_input=types.ModelInput(chunks=chunks),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": weights,
        },
    )


@st.composite
def _request(draw: DrawFn) -> types.ForwardBackwardRequest:
    n_datums = draw(st.integers(min_value=1, max_value=3))
    data = [draw(_datum()) for _ in range(n_datums)]
    include_seq_id = draw(st.booleans())
    include_config = draw(st.booleans())
    loss_fn = draw(_LOSS_FN)
    return _make_request(
        data=data,
        loss_fn=loss_fn,
        loss_fn_config={"clip_low": 0.8, "clip_high": 1.2} if include_config else None,
        seq_id=draw(st.integers(min_value=0, max_value=999)) if include_seq_id else None,
    )


# ---------------------------------------------------------------------------
# Round-trip property
# ---------------------------------------------------------------------------


@given(_request())
def test_request_to_proto_preserves_envelope_and_data(
    request: types.ForwardBackwardRequest,
) -> None:
    """forward_backward_request_to_proto + ParseFromString preserves model_id,
    seq_id, loss_fn name + config, datum count, chunk types, and the
    dense weights/target_tokens tensors."""
    msg = _roundtrip(forward_backward_request_to_proto(request))

    assert msg.model_id == request.model_id
    # seq_id is non-optional on the public proto; SDK always sends it. A None
    # on the dataclass decodes as 0 (proto3 default) by `request_conv`.
    assert msg.seq_id == (request.seq_id if request.seq_id is not None else 0)
    fbi = request.forward_backward_input
    assert msg.loss_fn == fbi.loss_fn
    if fbi.loss_fn_config is None:
        assert dict(msg.loss_fn_config) == {}
    else:
        assert dict(msg.loss_fn_config) == fbi.loss_fn_config
    assert len(msg.data) == len(fbi.data)

    for orig_datum, datum_msg in zip(fbi.data, msg.data, strict=True):
        # Chunks: same count and type ordering.
        assert len(datum_msg.model_input) == len(orig_datum.model_input.chunks)
        for orig_chunk, chunk_msg in zip(
            orig_datum.model_input.chunks, datum_msg.model_input, strict=True
        ):
            if isinstance(orig_chunk, types.EncodedTextChunk):
                assert chunk_msg.WhichOneof("chunk") == "encoded_text"
                tokens = np.frombuffer(chunk_msg.encoded_text.tokens, dtype=np.int32).tolist()
                assert tokens == list(orig_chunk.tokens)
            else:
                assert isinstance(orig_chunk, types.ImageChunk)
                assert chunk_msg.WhichOneof("chunk") == "image"
                assert chunk_msg.image.data == orig_chunk.data
                assert chunk_msg.image.format == orig_chunk.format
                if orig_chunk.expected_tokens is None:
                    assert not chunk_msg.image.HasField("expected_tokens")
                else:
                    assert chunk_msg.image.HasField("expected_tokens")
                    assert chunk_msg.image.expected_tokens == orig_chunk.expected_tokens

        # Dense tensors.
        for fname, orig_td in orig_datum.loss_fn_inputs.items():
            t_msg = datum_msg.loss_fn_inputs[fname]
            if orig_td.dtype == "float32":
                assert t_msg.dtype == public_pb.DTYPE_FLOAT32
                arr = np.frombuffer(t_msg.dense, dtype=np.float32).tolist()
                assert arr == pytest.approx(list(orig_td.data))
            else:
                assert t_msg.dtype == public_pb.DTYPE_INT64
                arr = np.frombuffer(t_msg.dense, dtype=np.int64).tolist()
                assert arr == list(orig_td.data)


# ---------------------------------------------------------------------------
# Hand-crafted edge cases
# ---------------------------------------------------------------------------


def test_sparse_tensor_round_trip() -> None:
    """Sparse CSR TensorData → proto sparse_csr branch."""
    sparse_td = types.TensorData(
        data=[1.0, 2.0],  # values
        dtype="float32",
        shape=[3, 4],
        sparse_crow_indices=[0, 1, 1, 2],
        sparse_col_indices=[1, 3],
    )
    request = _make_request(
        data=[
            types.Datum(
                model_input=types.ModelInput.from_ints([1]),
                loss_fn_inputs={
                    "target_tokens": [0],
                    "weights": sparse_td,
                },
            )
        ]
    )
    msg = _roundtrip(forward_backward_request_to_proto(request))
    weights_msg = msg.data[0].loss_fn_inputs["weights"]
    assert weights_msg.HasField("sparse_csr")
    assert not weights_msg.HasField("dense")
    assert list(weights_msg.shape) == [3, 4]
    assert np.frombuffer(weights_msg.sparse_csr.values, dtype=np.float32).tolist() == pytest.approx(
        [1.0, 2.0]
    )
    assert np.frombuffer(weights_msg.sparse_csr.crow_indices, dtype=np.int64).tolist() == [
        0,
        1,
        1,
        2,
    ]
    assert np.frombuffer(weights_msg.sparse_csr.col_indices, dtype=np.int64).tolist() == [1, 3]


def test_unsupported_dtype_raises() -> None:
    """Future-proofing: a TensorData dtype outside {float32, int64} raises early."""
    bad_td = types.TensorData(data=[1, 2, 3], dtype="float32", shape=[3])
    # Mutate dtype to an unsupported value to bypass Pydantic Literal validation.
    object.__setattr__(bad_td, "dtype", "float16")  # type: ignore[arg-type]
    request = _make_request(
        data=[
            types.Datum(
                model_input=types.ModelInput.from_ints([1]),
                loss_fn_inputs={"target_tokens": [0], "weights": bad_td},
            )
        ]
    )
    with pytest.raises(ValueError, match="dtype"):
        forward_backward_request_to_proto(request)


@pytest.mark.asyncio
async def test_send_forward_backward_proto_uses_protobuf_content_type() -> None:
    """The send helper POSTs raw proto bytes with Content-Type: application/x-protobuf."""
    from unittest.mock import AsyncMock

    from tinker.lib.public_interfaces.training_client import _send_forward_backward_proto

    request = _make_request(
        data=[
            types.Datum(
                model_input=types.ModelInput.from_ints([1, 2]),
                loss_fn_inputs={"target_tokens": [3, 4], "weights": [1.0, 1.0]},
            )
        ]
    )

    fake_client = AsyncMock()
    fake_client.post = AsyncMock(return_value="sentinel-future")

    result = await _send_forward_backward_proto(fake_client, request)
    assert result == "sentinel-future"

    fake_client.post.assert_awaited_once()
    call = fake_client.post.await_args
    assert call.args[0] == "/api/v1/forward_backward"
    body = call.kwargs["body"]
    assert isinstance(body, bytes)
    headers = call.kwargs["options"].get("headers")
    assert headers == {"Content-Type": "application/x-protobuf"}

    # Body parses back to the same proto.
    parsed = public_pb.ForwardBackwardRequest()
    parsed.ParseFromString(body)
    assert parsed.model_id == "m-test"
    assert parsed.loss_fn == "cross_entropy"


@pytest.mark.asyncio
async def test_send_forward_backward_proto_writes_bytes_to_wire() -> None:
    """End-to-end: helper drives a real httpx.AsyncClient via MockTransport so
    the Content-Type header and bytes body are observed on the actual outgoing
    HTTP request line — not just on the RequestOptions dict structure."""
    import httpx

    from tinker._client import AsyncTinker

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(202, json={"request_id": "rid"})

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    try:
        async_tinker = AsyncTinker(
            base_url="http://test",
            api_key="tml-test-api-key",
            http_client=http_client,
        )
        from tinker.lib.public_interfaces.training_client import _send_forward_backward_proto

        request = _make_request(
            data=[
                types.Datum(
                    model_input=types.ModelInput.from_ints([1, 2]),
                    loss_fn_inputs={"target_tokens": [3, 4], "weights": [1.0, 1.0]},
                )
            ]
        )
        # The SDK's response parser triggers a Pydantic v2 deprecation warning
        # (calls deprecated ``construct`` instead of ``model_construct``); not
        # our concern in this test, suppress to keep the wire-shape check focused.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            await _send_forward_backward_proto(async_tinker, request)
    finally:
        await http_client.aclose()

    assert len(captured) == 1
    sent = captured[0]
    # Content-Type lands on the outgoing httpx.Request — not just options dict.
    assert sent.headers["content-type"] == "application/x-protobuf"
    body = sent.read()
    assert isinstance(body, bytes) and len(body) > 0
    parsed = public_pb.ForwardBackwardRequest()
    parsed.ParseFromString(body)
    assert parsed.model_id == "m-test"


@pytest.mark.asyncio
async def test_send_forward_backward_proto_zstd_compresses_body_when_enabled() -> None:
    """With ``compress=True`` the helper zstd-compresses the proto body and
    sets ``Content-Encoding: zstd``; the body decompresses back to the same
    proto bytes."""
    from unittest.mock import AsyncMock

    import zstandard as zstd

    from tinker.lib.public_interfaces.training_client import _send_forward_backward_proto

    request = _make_request(
        data=[
            types.Datum(
                model_input=types.ModelInput.from_ints([1, 2]),
                loss_fn_inputs={"target_tokens": [3, 4], "weights": [1.0, 1.0]},
            )
        ]
    )

    fake_client = AsyncMock()
    fake_client.post = AsyncMock(return_value="sentinel-future")

    await _send_forward_backward_proto(fake_client, request, compress=True)

    fake_client.post.assert_awaited_once()
    call = fake_client.post.await_args
    headers = call.kwargs["options"].get("headers")
    assert headers == {
        "Content-Type": "application/x-protobuf",
        "Content-Encoding": "zstd",
    }
    body = call.kwargs["body"]
    assert isinstance(body, bytes)
    # The wire body is zstd-compressed; decompressing yields a valid proto.
    decompressed = zstd.ZstdDecompressor().decompress(body)
    parsed = public_pb.ForwardBackwardRequest()
    parsed.ParseFromString(decompressed)
    assert parsed.model_id == "m-test"
    assert parsed.loss_fn == "cross_entropy"
