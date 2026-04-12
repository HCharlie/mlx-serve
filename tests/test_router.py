import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from mlx_serve.api.router import router


async def mock_generate(*args, **kwargs):
    """Async generator that yields two tokens."""
    yield "Hello"
    yield " world"


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)

    mock_engine = MagicMock()
    mock_engine.model_path = "test-model"
    mock_engine.tokenizer = MagicMock()
    mock_engine.tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_engine.generate = mock_generate
    mock_engine._job_queue = MagicMock()
    mock_engine._job_queue.qsize.return_value = 0

    app.state.engine = mock_engine
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "queue_depth" in data


def test_chat_completions_non_streaming(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello world"


def test_chat_completions_streaming(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [l for l in resp.text.splitlines() if l.startswith("data:")]
    # Last line should be [DONE]
    assert lines[-1] == "data: [DONE]"
    # Second-to-last is the stop chunk (empty delta, finish_reason="stop")
    stop_chunk = json.loads(lines[-2][len("data: "):])
    assert stop_chunk["choices"][0]["finish_reason"] == "stop"
    # Content lines: everything before the stop chunk
    content_lines = lines[:-2]
    assert len(content_lines) == 2
    for line in content_lines:
        chunk = json.loads(line[len("data: "):])
        assert chunk["object"] == "chat.completion.chunk"
        assert "content" in chunk["choices"][0]["delta"]


def test_chat_completions_no_template_returns_400(client, app):
    app.state.engine.tokenizer.apply_chat_template.side_effect = Exception("no template")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 400


def test_completions_non_streaming(client):
    resp = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "once upon", "stream": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["text"] == "Hello world"


def test_generation_error_returns_openai_error_format(client, app):
    async def failing_generate(*args, **kwargs):
        raise RuntimeError("GPU out of memory")
        yield  # make it an async generator

    app.state.engine.generate = failing_generate
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data
    assert "message" in data["error"]
