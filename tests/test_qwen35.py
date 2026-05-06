import os

import pytest

MODEL_08B = os.path.expanduser("~/huggingface/Qwen3.5-0.8B")
MODEL_4B = os.path.expanduser("~/huggingface/Qwen3.5-4B")

TOKENS = [9419, 11, 821, 803, 369]  # "Hello, my name is"


@pytest.fixture(scope="module")
def llm_08b():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    from nanovllm import LLM, SamplingParams
    engine = LLM(MODEL_08B, enforce_eager=True, max_model_len=256, gpu_memory_utilization=0.85)
    yield engine
    engine.model_runner.call("exit")


@pytest.fixture(scope="module")
def llm_4b():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    from nanovllm import LLM, SamplingParams
    engine = LLM(MODEL_4B, enforce_eager=True, max_model_len=256, gpu_memory_utilization=0.85)
    yield engine
    engine.model_runner.call("exit")


class TestQwen3508B:
    def test_generates_coherent_text(self, llm_08b):
        from nanovllm import SamplingParams
        out = llm_08b.generate(["Hello, my name is"], SamplingParams(temperature=0.6, max_tokens=16))
        text = out[0]["text"].strip()
        assert len(text) > 0
        ch = [c for c in text[:12] if c.strip()]
        assert len(set(ch[:4])) > 1, f"repeating: {text!r}"

    def test_prefill_token_valid(self, llm_08b):
        from nanovllm import SamplingParams
        out = llm_08b.generate([TOKENS], SamplingParams(max_tokens=1))
        tok = out[0]["token_ids"][0]
        assert 0 <= tok < 248320, f"bad token: {tok}"

    def test_multiple_prompts(self, llm_08b):
        from nanovllm import SamplingParams
        sp = SamplingParams(temperature=0.6, max_tokens=8)
        out = llm_08b.generate(["Hello, my name is", "The capital of France is", "Python is a"], sp)
        assert len(out) == 3
        for o in out:
            assert len(o["text"].strip()) > 0

    def test_prefill_not_repeating(self, llm_08b):
        from nanovllm import SamplingParams
        out = llm_08b.generate([TOKENS], SamplingParams(max_tokens=1))
        tok = out[0]["token_ids"][0]
        assert tok != TOKENS[-1], f"model output repeats last input token"


class TestQwen354B:
    def test_generates_coherent_text(self, llm_4b):
        from nanovllm import SamplingParams
        out = llm_4b.generate(["Hello, my name is"], SamplingParams(temperature=0.6, max_tokens=8))
        text = out[0]["text"].strip()
        assert len(text) > 0
        ch = [c for c in text[:12] if c.strip()]
        assert len(set(ch[:4])) > 1, f"repeating: {text!r}"

    def test_prefill_runs(self, llm_4b):
        from nanovllm import SamplingParams
        out = llm_4b.generate([TOKENS], SamplingParams(max_tokens=1))
        tok = out[0]["token_ids"][0]
        assert 0 <= tok < 248320, f"bad token: {tok}"
