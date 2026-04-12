import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from mlx_serve.api.router import router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)

    mock_engine = MagicMock()
    mock_engine.model_path = "test-model"
    mock_engine.tokenizer = MagicMock()
    mock_engine.tokenizer.apply_chat_template.return_value = "<prompt>"

    async def mock_generate(*args, **kwargs):
        return "Hello world"

    mock_engine.generate = mock_generate

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
    assert data["data"][0]["id"] == "test-model"


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_completions(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello world"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_stream_returns_501(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == 501


def test_chat_completions_no_template_returns_400(client, app):
    app.state.engine.tokenizer.apply_chat_template.side_effect = Exception("no template")
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == 400


def test_generation_error_returns_500(client, app):
    async def failing_generate(*args, **kwargs):
        raise RuntimeError("GPU out of memory")

    app.state.engine.generate = failing_generate
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 500
    assert "message" in resp.json()["error"]
