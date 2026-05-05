import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, Union

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.sampling_params import SamplingParams


@dataclass
class CompletionOutput:
    index: int = 0
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: Optional[str] = None


@dataclass
class RequestOutput:
    request_id: str
    prompt: Optional[str] = None
    prompt_token_ids: list[int] = field(default_factory=list)
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = False

    @classmethod
    def new(cls, request_id, prompt, prompt_token_ids):
        return cls(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=list(prompt_token_ids),
            outputs=[CompletionOutput()],
        )


class AsyncLLM(LLMEngine):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self._seq_id_to_request_id: dict[int, str] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._prompts: dict[int, tuple] = {}
        self._sent_count: dict[int, int] = {}
        self._loop_task: Optional[asyncio.Task] = None
        self._wakeup: Optional[asyncio.Event] = None

    def _start_loop(self):
        if self._loop_task is None:
            self._wakeup = asyncio.Event()
            self._loop_task = asyncio.get_running_loop().create_task(self._run_loop())

    async def _run_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            if not self._queues and self.is_finished():
                self._wakeup.clear()
                await self._wakeup.wait()
                continue

            if self.is_finished():
                await asyncio.sleep(0)
                continue

            finished_outputs, _ = await loop.run_in_executor(None, self.step)

            for seq in list(self.scheduler.running):
                seq_id = seq.seq_id
                request_id = self._seq_id_to_request_id.get(seq_id)
                if request_id is None or request_id not in self._queues:
                    continue
                tokens = seq.completion_token_ids
                prev = self._sent_count.get(seq_id, 0)
                if len(tokens) > prev:
                    new_tokens = tokens[prev:]
                    self._sent_count[seq_id] = len(tokens)
                    prompt_str, prompt_ids = self._prompts.get(seq_id, ("", []))
                    out = RequestOutput.new(request_id, prompt_str, prompt_ids)
                    out.outputs[0].token_ids = list(new_tokens)
                    self._queues[request_id].put_nowait(out)

            for seq_id, token_ids in finished_outputs:
                request_id = self._seq_id_to_request_id.pop(seq_id, None)
                if request_id is None or request_id not in self._queues:
                    continue
                self._sent_count.pop(seq_id, None)
                prompt_str, prompt_ids = self._prompts.pop(seq_id, ("", []))
                out = RequestOutput.new(request_id, prompt_str, prompt_ids)
                out.outputs[0].token_ids = list(token_ids)
                out.outputs[0].finish_reason = "stop"
                out.finished = True
                self._queues[request_id].put_nowait(out)

            await asyncio.sleep(0)

    async def generate(
        self,
        prompt: Union[str, list[int]],
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncGenerator[RequestOutput, None]:
        self._start_loop()

        self._queues[request_id] = asyncio.Queue()
        if self._wakeup is not None:
            self._wakeup.set()

        if isinstance(prompt, str):
            prompt_str = prompt
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_str = ""
            prompt_token_ids = list(prompt)

        self.add_request(prompt_token_ids, sampling_params)
        seq_id = self.scheduler.waiting[-1].seq_id
        self._seq_id_to_request_id[seq_id] = request_id
        self._prompts[seq_id] = (prompt_str, prompt_token_ids)
        self._sent_count[seq_id] = 0

        try:
            while True:
                out = await self._queues[request_id].get()
                yield out
                if out.finished:
                    return
        finally:
            self._queues.pop(request_id, None)

    def abort(self, request_id: str):
        queue = self._queues.get(request_id)
        if queue is None:
            return
        out = RequestOutput(request_id=request_id, finished=True)
        out.outputs = [CompletionOutput(finish_reason="abort")]
        queue.put_nowait(out)
        self._queues.pop(request_id, None)
        seq_id = next((sid for sid, rid in self._seq_id_to_request_id.items()
                      if rid == request_id), None)
        if seq_id is None:
            return
        self._seq_id_to_request_id.pop(seq_id, None)
        self._sent_count.pop(seq_id, None)
        self._prompts.pop(seq_id, None)
        for seq in list(self.scheduler.running):
            if seq.seq_id == seq_id:
                seq.status = SequenceStatus.FINISHED
                self.scheduler.block_manager.deallocate(seq)
                self.scheduler.running.remove(seq)
                return
        for seq in list(self.scheduler.waiting):
            if seq.seq_id == seq_id:
                self.scheduler.waiting.remove(seq)
                return
