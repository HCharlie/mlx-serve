import asyncio
import pytest
from unittest.mock import MagicMock
from mlx_serve.engine import Engine


def make_mock_token(text: str):
    """mlx_lm.stream_generate yields objects with a .text attribute."""
    t = MagicMock()
    t.text = text
    return t


@pytest.fixture
def mock_mlx(monkeypatch):
    """
    Patch mlx_lm.load, stream_generate, and make_sampler so tests run
    without Apple Silicon or an actual model.
    """
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
    # make_sampler would import mlx on non-Apple-Silicon — patch it too
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
async def test_engine_generate_yields_tokens(mock_mlx):
    engine = Engine("fake/model")
    await engine.start()

    tokens = []
    async for token in engine.generate("<prompt>", max_tokens=10):
        tokens.append(token)

    assert tokens == ["Hello", " world"]
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_serializes_requests(mock_mlx, monkeypatch):
    """
    Two concurrent generate() calls must both complete.
    Serialization is verified by tracking which request the worker is
    processing — the second should not start until the first finishes.
    """
    order = []

    def tracking_generate(model, tokenizer, prompt, max_tokens, sampler):
        order.append(f"start:{prompt}")
        yield make_mock_token("tok")
        order.append(f"end:{prompt}")

    monkeypatch.setattr("mlx_serve.engine.stream_generate", tracking_generate)

    engine = Engine("fake/model")
    await engine.start()

    async def collect(prompt):
        tokens = []
        async for t in engine.generate(prompt, max_tokens=10):
            tokens.append(t)
        return tokens

    results = await asyncio.gather(collect("p1"), collect("p2"))
    assert all(len(r) > 0 for r in results)

    # Serialization: end of first request must appear before start of second
    all_starts = [i for i, e in enumerate(order) if e.startswith("start:")]
    all_ends = [i for i, e in enumerate(order) if e.startswith("end:")]
    assert all_ends[0] < all_starts[1]

    await engine.stop()


@pytest.mark.asyncio
async def test_engine_generation_error_raises(mock_mlx, monkeypatch):
    """If stream_generate raises, generate() should propagate the error."""
    def failing_generate(model, tokenizer, prompt, max_tokens, sampler):
        raise RuntimeError("GPU out of memory")
        yield  # make it a generator

    monkeypatch.setattr("mlx_serve.engine.stream_generate", failing_generate)

    engine = Engine("fake/model")
    await engine.start()

    with pytest.raises(RuntimeError, match="GPU out of memory"):
        async for _ in engine.generate("<prompt>"):
            pass

    await engine.stop()
