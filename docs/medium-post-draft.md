# I Built a Local LLM Server in 4 Python Files. Here's Exactly How It Works.

Cloud AI bills are real. I wanted to run open-source models on my MacBook — and have them actually work with the tools I use every day, like Cursor and Codex CLI. So I built `mlx-serve`: a minimal OpenAI-compatible server for Apple Silicon, in four files, with three dependencies.

This isn't a "look what I built" post. It's a walkthrough of every decision — including the ones that almost tripped me up.

---

## The Goal

Point any OpenAI-compatible tool at `http://localhost:8080/v1`, and have it talk to a local model instead of a cloud API. That's it.

The constraint I gave myself: keep it small enough to explain in a single blog post. No feature flags, no plugin systems, no "we might need this later" abstractions. Just the minimum that actually works.

---

## The Stack

Three direct dependencies:

```toml
dependencies = ["mlx-lm", "fastapi", "uvicorn[standard]"]
```

- **mlx-lm** — Apple's inference library, runs models on the Metal GPU
- **FastAPI** — async HTTP server, handles request validation for free via Pydantic
- **uvicorn** — the ASGI server that runs FastAPI

That's it. No Redis, no message queues, no Docker.

---

## The Project Structure

```
src/mlx_serve/
├── main.py       # CLI + FastAPI app setup
├── engine.py     # model loading + inference
└── api/
    ├── models.py # request/response schemas
    └── router.py # HTTP route handlers
```

Four files. Let's go through them.

---

## Step 1: Define the Contract (`models.py`)

Before writing a single route, I defined exactly what the API accepts and returns. OpenAI's format is the standard — every tool that talks to an LLM already speaks it.

```python
from typing import Literal
from pydantic import BaseModel

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoiceFull]
```

Pydantic does the validation automatically. If a client sends `role: "dragon"`, it gets a 422 before the request even reaches my code.

---

## Step 2: The Tricky Part — The Engine (`engine.py`)

This is where it gets interesting.

mlx-lm's `stream_generate()` is a **synchronous Python generator**. It runs on a thread, blocks that thread, and yields tokens one at a time. FastAPI, on the other hand, runs on an **asyncio event loop** — a single-threaded loop that handles many requests by switching between them at `await` points.

If you call `stream_generate()` directly from a FastAPI route handler, you block the entire event loop. Every other request freezes until generation finishes. That's not great when you have multiple tools talking to the same server.

The naive solution — `asyncio.to_thread()` — wraps a callable in a thread so it doesn't block the event loop. Here's the thing: that works fine *if you don't need incremental results*. Since I decided to keep things simple and return the full response at once, `asyncio.to_thread()` is actually the perfect fit.

```python
class Engine:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._gpu_lock = asyncio.Semaphore(1)  # one generation at a time

    async def start(self) -> None:
        self.model, self.tokenizer = load(self.model_path)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        def _run() -> str:
            sampler = make_sampler(temp=temperature, top_p=top_p)
            tokens = []
            for response in stream_generate(
                self.model, self.tokenizer, prompt,
                max_tokens=max_tokens, sampler=sampler,
            ):
                tokens.append(response.text)
            return "".join(tokens)

        # stream_generate() is synchronous and blocks its thread.
        # asyncio.to_thread() offloads it to a thread pool so the event loop
        # stays free to accept other requests while this one runs.
        # The lock ensures only one generation runs at a time (single GPU).
        async with self._gpu_lock:
            return await asyncio.to_thread(_run)
```

Two things worth explaining here:

**`asyncio.to_thread(_run)`** — runs `_run` on a thread pool thread. The event loop is free while the GPU is doing its thing. When it finishes, control returns to the async world with the full response string.

**`_gpu_lock = asyncio.Semaphore(1)`** — the GPU is a shared resource. If three requests came in simultaneously and all called `asyncio.to_thread(_run)` at once, they'd fight over memory and likely crash. The semaphore ensures they take turns. One runs, others wait. On an M4 Pro at ~100 tokens/second, the wait is barely noticeable for personal use.

That's the entire engine. 40 lines.

---

## Step 3: Wire Up the Routes (`router.py`)

The router is straightforward. One real endpoint — `POST /v1/chat/completions` — plus two utility endpoints.

```python
@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    engine = _get_engine(request)

    try:
        prompt = engine.tokenizer.apply_chat_template(
            [m.model_dump() for m in body.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return JSONResponse(status_code=400, content={"error": {
            "message": "This model has no chat template.",
            "type": "invalid_request_error", "code": 400,
        }})

    try:
        content = await engine.generate(
            prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {
            "message": str(exc), "type": "generation_error", "code": 500,
        }})

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=body.model,
        choices=[ChatChoiceFull(message=Message(role="assistant", content=content))],
    )
```

`apply_chat_template` converts the messages list into the prompt format the model expects — things like `<|user|>Hello<|assistant|>`. If the model doesn't have a chat template, we return a clear 400 rather than generating garbage.

---

## Step 4: Tie It Together (`main.py`)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: code before yield runs on startup, after yield on shutdown."""
    engine: Engine = app.state.engine
    print(f"Loading model: {engine.model_path} ...")
    await engine.start()
    print("Model loaded. Server ready.")
    yield
    await engine.stop()

def build_app(model_path: str) -> FastAPI:
    app = FastAPI(title="mlx-serve", lifespan=lifespan)
    app.state.engine = Engine(model_path)
    app.include_router(router)
    return app

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(build_app(args.model), host=args.host, port=args.port)
```

The `lifespan` context manager is FastAPI's way of running setup/teardown code. Everything before `yield` runs on startup (load the model), everything after runs on shutdown. The engine is stored on `app.state` so every route handler can access it.

---

## Running It

Install with a single command using [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/HCharlie/mlx-serve
mlx-serve
```

The model downloads automatically on first run and gets cached locally. Then just point any OpenAI-compatible tool at `http://127.0.0.1:8080/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="ignored")
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## What I Left Out (On Purpose)

No streaming, no multi-model support, no authentication, no metrics. Each of those is real work and adds real complexity — and none of them are needed for my use case of a personal development server.

The interesting extension if you want to add streaming: `asyncio.to_thread()` can't do it — it wraps the entire call and returns only when done. You'd need to bring back a `threading.Thread` + `stdlib queue.Queue` bridge to push tokens incrementally across the sync/async boundary. That's a whole other post.

---

## Final Thoughts

The most interesting part of this project wasn't the FastAPI wiring or the CLI — it was figuring out how to connect a synchronous GPU-bound library to an async HTTP server without either blocking the event loop or overcomplicating the code. `asyncio.to_thread()` + a semaphore turned out to be the simplest answer that actually works.

Four files. Three dependencies. One GPU. Full OpenAI compatibility.

The full code is on GitHub: [github.com/HCharlie/mlx-serve](https://github.com/HCharlie/mlx-serve)
