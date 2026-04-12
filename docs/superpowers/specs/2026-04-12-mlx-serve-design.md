# mlx-serve Design Spec

**Date:** 2026-04-12
**Status:** Approved

---

## Goal

A minimal, educational OpenAI-compatible LLM server for macOS (Apple Silicon) built on Apple's MLX framework. Designed to be simple enough to explain in a Medium post: three dependencies, four files, full local LLM serving.

Secondary goal: let tools like Claude Code and Codex point at `http://localhost:8080` and use a locally-hosted open-source model instead of a cloud API.

---

## Scope

**In scope:**
- OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`)
- Streaming responses via SSE
- Async request queue to handle multiple concurrent tool sessions
- CLI entry point with flags for model, host, port
- Distribution via `uv tool install`

**Out of scope (for now):**
- MCP server
- Frontend UI
- Multi-model support / model hot-swapping
- launchd service installer
- Metrics, tracing, dashboards
- Non-Apple-Silicon platforms

---

## Target User

A macOS developer on Apple Silicon (M1+, recommended M3/M4 with 32GB+ unified memory) who wants to run open-source models locally and connect them to existing AI coding tools.

---

## Tech Stack

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | mlx-lm is Python-native; fastest path |
| Package manager | uv | Solves Python packaging pain; clean install story |
| Inference | mlx-lm | Best MLX model support; Apple-maintained |
| HTTP server | FastAPI + uvicorn | Async, widely known, minimal boilerplate |
| Schemas | Pydantic | Ships with FastAPI, no extra dependency |
| Protocol | OpenAI-compatible REST | Universal support across Claude Code, Codex, Cursor, etc. |
| Streaming | SSE (Server-Sent Events) | Standard OpenAI streaming format |

**Direct dependencies (3):**
```toml
dependencies = ["mlx-lm", "fastapi", "uvicorn[standard]"]
```

Note: `mlx-lm` brings in transitive dependencies (huggingface-hub, transformers, etc.) and `uvicorn[standard]` adds uvloop and httptools for proper async operation. The "3 dependencies" framing refers to direct dependencies — `uv.lock` will contain more.

---

## Architecture

```
┌────────────────────────────────────────┐
│  Claude Code / Codex / Any OAI client │
└───────────────┬────────────────────────┘
                │ HTTP (OpenAI-compatible)
                ▼
┌────────────────────────────────────────┐
│           FastAPI Process              │
│  ┌─────────────────────────────────┐  │
│  │  /v1/chat/completions           │  │
│  │  /v1/completions                │  │
│  │  /v1/models                     │  │
│  │  /health                        │  │
│  └──────────────┬──────────────────┘  │
│           ┌─────▼──────┐              │
│           │ Async Queue│              │
│           │ (in engine)│              │
│           └─────┬──────┘              │
│           ┌─────▼──────┐              │
│           │  mlx-lm    │              │
│           │  Engine    │              │
│           └────────────┘              │
└────────────────────────────────────────┘
```

One process. FastAPI handles HTTP. The engine owns the asyncio queue and the mlx-lm model. The queue serializes inference requests — only one runs at a time. mlx-lm runs in a thread pool so it does not block the event loop.

---

## Project Structure

```
mlx_serve/
├── pyproject.toml
└── src/
    └── mlx_serve/
        ├── __init__.py
        ├── main.py         # CLI entry point, FastAPI app setup, lifespan
        ├── engine.py       # mlx-lm model loading, asyncio queue, generate()
        └── api/
            ├── router.py   # /v1/* route handlers
            └── models.py   # Pydantic request/response schemas (OpenAI format)
```

### File responsibilities

**`main.py`**
- Parses CLI flags: `--model`, `--host`, `--port`
- FastAPI app with lifespan context (loads model on startup, cleans up on shutdown)
- Mounts the API router
- Runs uvicorn

**`engine.py`**
- Loads the model and tokenizer via `mlx_lm.load()` once at startup
- Owns an `asyncio.Queue` for incoming inference requests
- Runs a worker coroutine that pulls requests one at a time
- For each request, spawns a `threading.Thread` that runs `mlx_lm.stream_generate()` and pushes tokens into a per-request `queue.Queue` (stdlib, not asyncio) — this is necessary because `stream_generate` is a synchronous generator and cannot be used with `asyncio.to_thread()` incrementally
- The worker reads from the stdlib queue via `run_in_executor` to bridge sync→async without blocking the event loop
- Generation parameters (`temperature`, `top_p`) are passed via `mlx_lm.sample_utils.make_sampler`, not as bare kwargs to `stream_generate`
- Exposes a single `generate()` async generator that the API layer consumes token by token

**`api/models.py`**
- Pydantic models matching the OpenAI API schema
- `ChatCompletionRequest`, `ChatCompletionResponse`, `ChatCompletionChunk`
- `CompletionRequest`, `CompletionResponse`

**`api/router.py`**
- `POST /v1/chat/completions` — converts messages to prompt, calls engine, streams SSE or returns full response
- `POST /v1/completions` — legacy completions endpoint
- `GET /v1/models` — returns the currently loaded model in OpenAI format
- `GET /health` — server status and queue depth
- SSE streaming is handled inline here via FastAPI's `StreamingResponse`. When `stream=true`, the route handler returns a `StreamingResponse` that yields formatted SSE chunks as tokens arrive from the engine's async generator. No separate file is needed — it is ~10 lines of formatting logic.

---

## Threading & Queue Design

mlx-lm's `stream_generate()` is a **synchronous Python generator** — it blocks the thread it runs on and yields tokens one at a time. FastAPI runs on an **asyncio event loop**. These two cannot communicate directly. Three options were considered:

| Option | Mechanism | Why rejected / chosen |
|---|---|---|
| `asyncio.to_thread()` | Runs a callable in a thread pool | Cannot stream — wraps the entire generator call, returns only after all tokens are generated |
| `threading.Thread` + `stdlib queue.Queue` + `run_in_executor` | Thread pushes tokens into a thread-safe stdlib queue; async worker reads via `run_in_executor` | **Chosen.** Correct, conventional, used by mlx-lm's own server internally |
| `threading.Thread` + `asyncio.Queue` + `call_soon_threadsafe` | Thread calls back into the event loop to push to an asyncio queue | Works but requires passing the event loop reference into the generation thread — fragile |

**How the chosen approach works:**

```
asyncio event loop                    generation thread
─────────────────                     ─────────────────
worker coroutine                      threading.Thread
  awaits run_in_executor  ←─tokens─── stream_generate() yields tokens
  (reads stdlib queue)                pushes to stdlib queue.Queue
  yields token to SSE
```

`run_in_executor(None, token_queue.get)` turns the blocking stdlib `queue.get()` into an awaitable, bridging the sync generator world and the async FastAPI world without blocking the event loop.

---

## Request Flow

```
POST /v1/chat/completions
    │
    ▼
Validate request (Pydantic)
    │
    ▼
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
→ prompt string. If model has no chat template, return 400 with clear message.
    │
    ▼
Build sampler: make_sampler(temp=temperature, top_p=top_p)
    │
    ▼
Push (prompt, sampler, max_tokens, stdlib_token_queue) onto engine asyncio.Queue
    │
    ▼
Engine worker picks up request (serialized, one at a time)
    │
    ▼
Spawn threading.Thread → runs stream_generate(model, tokenizer, prompt,
    max_tokens=max_tokens, sampler=sampler)
    → each token pushed to per-request stdlib queue.Queue
    │
    ▼
Worker reads tokens via run_in_executor (sync queue → async bridge)
    │
    ├─ stream=true  → SSE: each token sent immediately as data: {...}
    └─ stream=false → buffer all tokens, return single JSON response
```

---

## Configuration

CLI flags only — no config file:

```bash
mlx-serve --model mlx-community/Llama-3.2-3B-Instruct-4bit
mlx-serve --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 9090 --host 0.0.0.0
```

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace repo ID or local path |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Port |

Generation parameters (`temperature`, `top_p`, `max_tokens`) come from the request body per the OpenAI spec. Reasonable defaults applied if not provided.

---

## Error Handling

Minimal — enough to be useful, no more:

| Situation | Behavior |
|---|---|
| Model fails to load | Server exits with clear error message |
| Model has no chat template | 400 response with message explaining the model requires a prompt-style request or a chat template |
| Generation error | 500 response in OpenAI error format; server stays alive |
| Client disconnects mid-stream | Generation cancelled; queue slot freed |
| Invalid request | 422 response via Pydantic validation |

All errors returned in OpenAI error format:
```json
{"error": {"message": "...", "type": "...", "code": 500}}
```

---

## Distribution

```bash
uv tool install mlx-serve        # install
mlx-serve --model <model>        # run
```

`pyproject.toml` entry point:
```toml
[project.scripts]
mlx-serve = "mlx_serve.main:cli"
```

Users need only `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`). No system Python, no venv management, no pip.

---

## Connecting to Tools

For any OpenAI-compatible client or tool:
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="ignored")
```

For tools that support a custom OpenAI base URL (e.g. Cursor, Codex CLI):
```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_API_KEY=ignored
```

Note: Claude Code connects to Anthropic's API and does not natively route to a local OpenAI-compatible server as its primary model. The local server is used by other tools (Cursor, Codex, Continue.dev, custom scripts) that support OpenAI-compatible endpoints.

---

## What This Is Not

This is not a production server. It is an educational starting point — a clear, minimal implementation that shows how the pieces fit together. Features like continuous batching, paged KV cache, multi-model routing, and authentication are intentional omissions, not oversights.

---

## Future Work (not in scope now)

- MCP server for model management tools
- Frontend UI (web or native macOS app)
- Multiple model support
- launchd service for background running
- Pluggable inference backends (llama.cpp)
