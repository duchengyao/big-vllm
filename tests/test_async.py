import asyncio
import os

import pytest

from nanovllm import AsyncLLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


@pytest.fixture(scope="module")
def llm():
    engine = AsyncLLM(MODEL)
    yield engine
    engine.exit()


@pytest.mark.asyncio
async def test_single_streaming(llm):
    partial_count = 0
    async for out in llm.generate(
        "Hello, my name is",
        SamplingParams(max_tokens=10),
        request_id="test-single",
    ):
        partial_count += 1
        if out.finished:
            assert out.outputs[0].finish_reason == "stop"
            assert len(out.outputs[0].token_ids) == 10
    assert partial_count >= 2


@pytest.mark.asyncio
async def test_multiple_concurrent(llm):
    async def run_one(request_id, prompt, max_tokens):
        results = []
        async for out in llm.generate(
            prompt, SamplingParams(max_tokens=max_tokens), request_id=request_id,
        ):
            results.append(out)
        return request_id, results

    tasks = [
        run_one("concur-a", "The capital of France is", 10),
        run_one("concur-b", "Python is a programming", 10),
        run_one("concur-c", "The meaning of life is", 10),
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    for req_id, outputs in results:
        assert outputs[-1].finished
        tokens = outputs[-1].outputs[0].token_ids
        assert len(tokens) == 10, f"{req_id}: expected 10 tokens, got {len(tokens)}"


@pytest.mark.asyncio
async def test_abort(llm):
    async for out in llm.generate(
        "Write a very long story about",
        SamplingParams(max_tokens=200),
        request_id="test-abort",
    ):
        if not out.finished and len(out.outputs[0].token_ids) >= 5:
            llm.abort("test-abort")
            break
    else:
        llm.abort("test-abort")


@pytest.mark.asyncio
async def test_request_output_dataclass(llm):
    async for out in llm.generate(
        [1, 2, 3],
        SamplingParams(max_tokens=5),
        request_id="test-dataclass",
    ):
        assert out.request_id == "test-dataclass"
        assert isinstance(out.prompt_token_ids, list)
        assert len(out.outputs) == 1
        assert out.outputs[0].index == 0
        if out.finished:
            assert out.outputs[0].finish_reason is not None
            break


@pytest.mark.asyncio
async def test_token_ids_streaming(llm):
    token_counts = []
    async for out in llm.generate(
        [1, 2, 3],
        SamplingParams(max_tokens=8),
        request_id="test-tokens",
    ):
        token_counts.append(len(out.outputs[0].token_ids))
        if out.finished:
            break
    assert token_counts[0] >= 1
    assert token_counts[-1] == 8
