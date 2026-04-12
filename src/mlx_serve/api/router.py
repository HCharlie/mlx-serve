import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_serve.api.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoiceFull,
    ChatChoiceChunk,
    ChatDelta,
    CompletionRequest,
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
        "queue_depth": engine._job_queue.qsize(),
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
        raise HTTPException(
            status_code=400,
            detail=(
                "This model has no chat template. "
                "Use /v1/completions with a raw prompt instead."
            ),
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


@router.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request):
    engine = _get_engine(request)
    token_gen = engine.generate(
        body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
    )
    req_id = f"cmpl-{uuid.uuid4().hex}"

    if body.stream:
        async def sse():
            async for token in token_gen:
                yield f"data: {json.dumps({'id': req_id, 'object': 'text_completion', 'choices': [{'text': token, 'index': 0}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    tokens = [t async for t in token_gen]
    return {
        "id": req_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{"text": "".join(tokens), "index": 0, "finish_reason": "stop"}],
    }
