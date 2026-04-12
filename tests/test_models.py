import pytest
from mlx_serve.api.models import (
    ChatCompletionRequest,
    ChatCompletionChunk,
    ChatCompletionResponse,
    CompletionRequest,
    ModelList,
    Message,
)


def test_chat_request_parses_messages():
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "hello"


def test_chat_request_stream_defaults_false():
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.stream is False


def test_chat_request_accepts_generation_params():
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        top_p=0.8,
        max_tokens=256,
    )
    assert req.temperature == 0.5
    assert req.top_p == 0.8
    assert req.max_tokens == 256


def test_completion_request_parses():
    req = CompletionRequest(model="test-model", prompt="once upon a time")
    assert req.prompt == "once upon a time"
    assert req.stream is False


def test_chat_completion_chunk_shape():
    chunk = ChatCompletionChunk(
        id="chatcmpl-abc",
        created=1234567890,
        model="test-model",
        choices=[{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
    )
    assert chunk.choices[0].delta.content == "hello"
    assert chunk.object == "chat.completion.chunk"


def test_model_list_shape():
    ml = ModelList(data=[{"id": "my-model"}])
    assert ml.data[0].id == "my-model"
    assert ml.object == "list"
