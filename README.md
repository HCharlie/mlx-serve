# mlx-serve

A minimal OpenAI-compatible LLM server for macOS Apple Silicon, built on [mlx-lm](https://github.com/ml-explore/mlx-lm).

Point any OpenAI-compatible tool at `http://localhost:8080/v1` and use a locally-hosted model instead of a cloud API.

---

## Requirements

- macOS with Apple Silicon (M1 or later; M3/M4 with 32 GB+ recommended for larger models)
- [uv](https://docs.astral.sh/uv/) — install once, never think about Python environments again:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Install

```bash
uv tool install git+https://github.com/HCharlie/mlx-serve
```

---

## Start the server

```bash
# Default model (Llama 3.2 3B Instruct, ~2 GB)
mlx-serve

# Pick a different model from mlx-community on HuggingFace
mlx-serve --model mlx-community/Llama-3.2-1B-Instruct-4bit    # ~1 GB, fastest
mlx-serve --model mlx-community/Mistral-7B-Instruct-v0.3-4bit # ~4 GB
mlx-serve --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit # ~5 GB

# Custom host/port
mlx-serve --host 0.0.0.0 --port 9090
```

The model is downloaded automatically from HuggingFace on first use and cached locally.

---

## Connect your tools

### Any OpenAI-compatible tool

Set these two environment variables — most tools (Cursor, Continue.dev, Codex CLI, etc.) pick them up automatically:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=ignored
```

### Claude Code

Claude Code uses Anthropic's API as its primary model and cannot be redirected to a local server. However, you can use mlx-serve from within Claude Code via the OpenAI Python SDK in scripts or tools.

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="ignored")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### curl

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## API reference

All endpoints follow the OpenAI REST API format.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion |
| `GET` | `/v1/models` | List loaded model |
| `GET` | `/health` | Server status |

### Request parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | — | Model name (must match the running model) |
| `messages` | array | — | Chat messages with `role` and `content` |
| `max_tokens` | integer | `512` | Maximum tokens to generate |
| `temperature` | float | `0.7` | Sampling temperature (0 = deterministic) |
| `top_p` | float | `0.9` | Nucleus sampling threshold |

---

## Model recommendations

All models below are 4-bit quantized and available on HuggingFace under [mlx-community](https://huggingface.co/mlx-community).

| Model | Size | Good for |
|-------|------|----------|
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | ~1 GB | Fast responses, simple tasks |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~2 GB | **Default.** Good balance |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | ~4 GB | Better reasoning |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | ~5 GB | Strong all-rounder |
| `mlx-community/Llama-3.3-70B-Instruct-4bit` | ~40 GB | Best quality, needs 48 GB+ |

---

## How it works

One process. Three components:

1. **FastAPI** handles HTTP requests
2. **Engine** serializes inference via an asyncio semaphore — one request runs at a time
3. **mlx-lm** does the actual inference on Apple Silicon via Metal GPU

The semaphore means multiple concurrent clients all get served without errors — they just wait their turn. On an M4 Pro at ~100 tok/s, the wait is rarely noticeable.

---

## Development

```bash
git clone https://github.com/HCharlie/mlx-serve
cd mlx-serve
uv sync --dev
uv run pytest
```
