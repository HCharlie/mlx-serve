import pytest
from unittest.mock import MagicMock
from mlx_serve.engine import Engine


def make_mock_token(text: str):
    t = MagicMock()
    t.text = text
    return t


@pytest.fixture
def mock_mlx(monkeypatch):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"

    monkeypatch.setattr("mlx_serve.engine.load", lambda path: (mock_model, mock_tokenizer))
    monkeypatch.setattr(
        "mlx_serve.engine.stream_generate",
        lambda model, tokenizer, prompt, max_tokens, sampler: iter(
            [make_mock_token("Hello"), make_mock_token(" world")]
        ),
    )
    monkeypatch.setattr("mlx_serve.engine.make_sampler", lambda temp, top_p: None)
    return mock_model, mock_tokenizer


@pytest.mark.asyncio
async def test_engine_start_loads_model(mock_mlx):
    engine = Engine("fake/model")
    await engine.start()
    assert engine.model is not None
    assert engine.tokenizer is not None
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_generate_returns_text(mock_mlx):
    engine = Engine("fake/model")
    await engine.start()

    result = await engine.generate("<prompt>", max_tokens=10)
    assert result == "Hello world"

    await engine.stop()


@pytest.mark.asyncio
async def test_engine_generation_error_raises(mock_mlx, monkeypatch):
    def failing_generate(model, tokenizer, prompt, max_tokens, sampler):
        raise RuntimeError("GPU out of memory")
        yield  # make it a generator

    monkeypatch.setattr("mlx_serve.engine.stream_generate", failing_generate)

    engine = Engine("fake/model")
    await engine.start()

    with pytest.raises(RuntimeError, match="GPU out of memory"):
        await engine.generate("<prompt>")

    await engine.stop()
