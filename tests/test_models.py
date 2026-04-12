import pytest
from pydantic import ValidationError
from mlx_serve.api.models import (
    ChatCompletionRequest,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ModelInfo,
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


def test_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="dragon", content="x")


def test_chat_completion_response_shape():
    resp = ChatCompletionResponse(
        id="chatcmpl-abc",
        created=1234567890,
        model="test-model",
        choices=[{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }],
    )
    assert resp.object == "chat.completion"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.content == "hello"


def test_model_info_defaults():
    info = ModelInfo(id="my-model")
    assert info.object == "model"
    assert info.created == 0
    assert info.owned_by == "local"


