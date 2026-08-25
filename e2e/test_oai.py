"""The OpenAI compatible surface, which has its own server and its own bugs.

Spoken over plain HTTP so the suite keeps installing only the wheel a user does.
"""

import json

import pytest

MESSAGES = [{"role": "user", "content": "Reply with the single word OK."}]
# The served models reason first, so a small budget returns only reasoning_content.
MAX_TOKENS = 64


def _post(oai_client, path: str, payload: dict):
    """One retry on a 5xx. Intern sits behind Cloudflare, which answers 524 on a
    slow completion, and unlike the SDK this client has no retries of its own."""
    for attempt in range(2):
        response = oai_client.post(path, json=payload)
        if response.status_code < 500 or attempt:
            return response
    raise AssertionError("unreachable")


def _completion_text(choice: dict) -> str:
    message = choice["message"]
    return (message.get("content") or "") + (message.get("reasoning_content") or "")


def test_models_endpoint_is_a_well_formed_listing(oai_client) -> None:
    """Lists the caller's sampler checkpoints, so it is legitimately empty."""
    response = oai_client.get("/models")
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["object"] == "list"
    for entry in body["data"]:
        assert entry["id"]
        assert entry["object"] == "model"


@pytest.mark.gpu
def test_chat_completion_answers(oai_client, model: str) -> None:
    response = _post(
        oai_client,
        "/chat/completions",
        {"model": model, "messages": MESSAGES, "max_tokens": MAX_TOKENS, "temperature": 0},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == model

    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert _completion_text(choice).strip(), "the model returned nothing at all"
    assert choice["finish_reason"] in {"stop", "length"}
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0


@pytest.mark.gpu
def test_chat_completion_streams_chunks(oai_client, model: str) -> None:
    payload = {
        "model": model,
        "messages": MESSAGES,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
    }
    with oai_client.stream("POST", "/chat/completions", json=payload) as response:
        # Read first: the body is unavailable for the failure message otherwise.
        if response.status_code != 200:
            response.read()
            raise AssertionError(f"stream returned {response.status_code}: {response.text}")

        chunks, saw_done = [], False
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(data))

    assert saw_done, "the stream ended without [DONE]"
    assert chunks, "the stream carried no chunks"
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)

    # Iterating choices rather than indexing: a usage chunk carries none.
    streamed = "".join(
        (delta.get("content") or "") + (delta.get("reasoning_content") or "")
        for chunk in chunks
        for choice in chunk["choices"]
        for delta in [choice.get("delta") or {}]
    )
    assert streamed.strip(), "the stream carried chunks but no text"


@pytest.mark.gpu
def test_legacy_completions_returns_text(oai_client, model: str) -> None:
    response = _post(
        oai_client,
        "/completions",
        {"model": model, "prompt": "1 2 3", "max_tokens": 8, "temperature": 0},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"].strip(), "the completion came back empty"
