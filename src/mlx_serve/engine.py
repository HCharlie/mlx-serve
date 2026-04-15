import asyncio

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


class Engine:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._gpu_lock = asyncio.Semaphore(1)  # one generation at a time

    async def start(self) -> None:
        self.model, self.tokenizer = load(self.model_path)

    async def stop(self) -> None:
        pass

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
