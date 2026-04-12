import asyncio
import queue as stdlib_queue
import threading
from dataclasses import dataclass
from typing import AsyncGenerator

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


@dataclass
class _InferenceRequest:
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    result_queue: asyncio.Queue  # constructed explicitly in generate() on the running loop


class Engine:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._job_queue: asyncio.Queue[_InferenceRequest] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Load the model and start the background worker. Call once at startup."""
        self.model, self.tokenizer = load(self.model_path)
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Cancel the worker. Call on shutdown."""
        if self._worker_task:
            self._worker_task.cancel()

    @property
    def queue_depth(self) -> int:
        """Number of requests currently waiting in the job queue."""
        return self._job_queue.qsize()

    async def _worker(self) -> None:
        """
        Pulls one inference request at a time from the job queue.
        Runs stream_generate in a background thread (it is a synchronous
        generator), bridges tokens back to asyncio via a stdlib queue and
        run_in_executor, then forwards each token (or a raised exception)
        to the per-request result queue.
        """
        loop = asyncio.get_running_loop()
        while True:
            request = await self._job_queue.get()
            token_bridge: stdlib_queue.Queue = stdlib_queue.Queue()

            def _run_generation() -> None:
                try:
                    sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
                    for response in stream_generate(
                        self.model,
                        self.tokenizer,
                        request.prompt,
                        max_tokens=request.max_tokens,
                        sampler=sampler,
                    ):
                        token_bridge.put(response.text)
                    token_bridge.put(None)  # sentinel — generation complete
                except Exception as exc:
                    token_bridge.put(exc)  # forward exception to async side

            thread = threading.Thread(target=_run_generation, daemon=True)
            thread.start()

            # Bridge: blocking stdlib queue → non-blocking asyncio result queue
            while True:
                item = await loop.run_in_executor(None, token_bridge.get)
                await request.result_queue.put(item)
                if item is None or isinstance(item, BaseException):
                    break

            self._job_queue.task_done()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """
        Submit a generation request and yield tokens as they are produced.
        Requests are serialized — only one runs at a time.
        Raises if generation fails inside the worker thread.
        """
        # Construct result_queue here, inside the running event loop
        result_queue: asyncio.Queue = asyncio.Queue()
        request = _InferenceRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            result_queue=result_queue,
        )
        await self._job_queue.put(request)

        while True:
            item = await result_queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
