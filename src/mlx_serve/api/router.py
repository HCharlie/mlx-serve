import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_serve.api.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoiceFull,
    ChatChoiceChunk,
    ChatDelta,
    Message,
    ModelInfo,
    ModelList,
)

router = APIRouter()


def _get_engine(request: Request):
    return request.app.state.engine


async def _token_stream_to_sse(
    token_gen: AsyncGenerator[str, None],
    model: str,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Format tokens as OpenAI SSE chunks."""
    async for token in token_gen:
        chunk = ChatCompletionChunk(
            id=req_id,
            created=int(time.time()),
            model=model,
            choices=[ChatChoiceChunk(delta=ChatDelta(content=token), finish_reason=None)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
    # Final stop chunk — required by OpenAI protocol for clients that check finish_reason
    stop_chunk = ChatCompletionChunk(
        id=req_id,
        created=int(time.time()),
        model=model,
        choices=[ChatChoiceChunk(delta=ChatDelta(), finish_reason="stop")],
    )
    yield f"data: {stop_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.get("/v1/models")
async def list_models(request: Request):
    engine = _get_engine(request)
    return ModelList(data=[ModelInfo(id=engine.model_path)])


@router.get("/health")
async def health(request: Request):
    engine = _get_engine(request)
    return {
        "status": "ok",
        "model": engine.model_path,
        "queue_depth": engine.queue_depth,
    }


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
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "This model has no chat template. Use /v1/completions with a raw prompt instead.",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
        )

    token_gen = engine.generate(
        prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
    )
    req_id = f"chatcmpl-{uuid.uuid4().hex}"

    if body.stream:
        return StreamingResponse(
            _token_stream_to_sse(token_gen, body.model, req_id),
            media_type="text/event-stream",
        )

    try:
        tokens = [t async for t in token_gen]
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "generation_error", "code": 500}},
        )

    content = "".join(tokens)
    return ChatCompletionResponse(
        id=req_id,
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatChoiceFull(message=Message(role="assistant", content=content))
        ],
    )
