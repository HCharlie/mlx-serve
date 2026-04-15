import argparse
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from mlx_serve.api.router import router
from mlx_serve.engine import Engine


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
    parser = argparse.ArgumentParser(
        description="Serve a local LLM via an OpenAI-compatible API using MLX."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="HuggingFace repo ID or local path (default: mlx-community/Llama-3.2-3B-Instruct-4bit)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    app = build_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
