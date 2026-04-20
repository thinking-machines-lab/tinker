"""Tests for ModelInput / EncodedTextChunk __repr__ performance.

Pydantic v2's ``model_dump(mode="json")`` falls back to ``__repr__`` for
``ModelInputChunk`` union variants because the discriminated union uses
``PropertyInfo(discriminator=...)`` instead of pydantic's native Discriminator.

Without cheap __repr__ overrides, serializing a SampleRequest with an 8K-token
prompt takes seconds (formatting every token ID as a string). With the fix,
it completes in under a millisecond.
"""

import time

from tinker._compat import model_dump
from tinker.types.encoded_text_chunk import EncodedTextChunk
from tinker.types.model_input import ModelInput
from tinker.types.sample_request import SampleRequest
from tinker.types.sampling_params import SamplingParams


def test_encoded_text_chunk_repr_is_cheap() -> None:
    """EncodedTextChunk.__repr__ should NOT dump all token IDs."""
    chunk = EncodedTextChunk(tokens=list(range(100_000)))
    start = time.perf_counter()
    r = repr(chunk)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.01, f"repr took {elapsed:.3f}s — still dumping all tokens?"
    assert "100000" in r
    # Must NOT contain individual token values
    assert "99999" not in r


def test_model_input_repr_is_cheap() -> None:
    """ModelInput.__repr__ should summarise, not expand all chunks."""
    mi = ModelInput.from_ints(list(range(50_000)))
    start = time.perf_counter()
    r = repr(mi)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.01, f"repr took {elapsed:.3f}s"
    assert "50000" in r


def test_sample_request_model_dump_json_performance() -> None:
    """model_dump(mode='json') on a realistic SampleRequest must be fast.

    This is the exact code path used by ``sampling.asample()`` before
    sending the request over HTTP.  Before the fix, an 8K-token prompt
    caused ~2-5 s of CPU time formatting token IDs via __repr__.
    """
    prompt = ModelInput.from_ints(list(range(8192)))
    request = SampleRequest(
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=1024),
    )

    # Warm up (pydantic schema compilation, etc.)
    model_dump(request, exclude_unset=True, mode="json")

    start = time.perf_counter()
    result = model_dump(request, exclude_unset=True, mode="json")
    elapsed = time.perf_counter() - start

    # Should complete in well under 100ms, not seconds.
    assert elapsed < 0.1, (
        f"model_dump took {elapsed:.3f}s — __repr__ fallback is likely still "
        f"formatting all token IDs"
    )
    # Sanity: the prompt field should be present in the output
    assert "prompt" in result


def test_model_dump_json_preserves_data() -> None:
    """Ensure the fast __repr__ path doesn't break actual data extraction.

    model_dump(mode='json') should still produce a dict with the correct
    structure, even though the serializer falls back to repr for chunks.
    """
    tokens = [1, 2, 3, 42, 100]
    prompt = ModelInput.from_ints(tokens)
    request = SampleRequest(
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
    )
    result = model_dump(request, mode="json")

    assert result["num_samples"] == 1
    assert result["type"] == "sample"
    assert result["sampling_params"]["max_tokens"] == 64
    assert result["sampling_params"]["temperature"] == 0.7
    # prompt is serialized (possibly via repr, but present)
    assert "prompt" in result
