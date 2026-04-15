import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from mlx_serve.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoiceFull,
    Message,
    ModelInfo,
    ModelList,
)

router = APIRouter()


def _get_engine(request: Request):
    # FastAPI stores shared state on app.state — set once at startup in main.py
    return request.app.state.engine


@router.get("/v1/models")
async def list_models(request: Request):
    engine = _get_engine(request)
    return ModelList(data=[ModelInfo(id=engine.model_path)])


@router.get("/health")
async def health(request: Request):
    engine = _get_engine(request)
    return {"status": "ok", "model": engine.model_path}


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    engine = _get_engine(request)

    if body.stream:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Streaming is not yet implemented.",
                    "type": "not_implemented",
                    "code": 501,
                }
            },
        )

    try:
        prompt = engine.tokenizer.apply_chat_template(
            [m.model_dump() for m in body.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "This model has no chat template. Use a model that supports chat.",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
        )

    try:
        content = await engine.generate(
            prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "generation_error", "code": 500}},
        )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=body.model,
        choices=[ChatChoiceFull(message=Message(role="assistant", content=content))],
    )
