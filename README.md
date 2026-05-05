# Big-vLLM

A high-performance LLM inference engine forked from [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), with native support for hybrid-attention models like Qwen3.5.

Supported model families: `Qwen2` (including `Qwen2.5`), `Qwen3`, and `Qwen3.5`.

## Key Features

- **Fast offline inference** — Competitive with vLLM across Qwen3 and Qwen3.5
- **Native hybrid-attention** — Hand-written GatedDeltaNet for Qwen3.5, no HuggingFace model dependency
- **Async streaming API** — `AsyncLLM` with `generate()` async generator, supports concurrent requests and abort
- **CUDA graph** — Zero-overhead kernel replay for decode
- **Paged KV cache** — Efficient memory management with prefix caching
- **Readable codebase** — ~1,500 lines of Python

## Installation

```bash
pip install git+https://github.com/duchengyao/big-vllm.git
```

## Model Download

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False

huggingface-cli download --resume-download Qwen/Qwen3.5-0.8B \
  --local-dir ~/huggingface/Qwen3.5-0.8B/ \
  --local-dir-use-symlinks False
```

## Quick Start

### Synchronous

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3.5-0.8B", enforce_eager=False, max_model_len=1024)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, my name is"], sampling_params)
print(outputs[0]["text"])
```

### Async streaming

```python
import asyncio
from nanovllm import AsyncLLM, SamplingParams

async def main():
    llm = AsyncLLM("~/huggingface/Qwen3-0.6B")
    async for out in llm.generate(
        "Hello, my name is",
        SamplingParams(max_tokens=50),
        request_id="req-1",
    ):
        text = llm.tokenizer.decode(out.outputs[0].token_ids)
        print(text, end="", flush=True)

asyncio.run(main())
```

Supports concurrent requests, abort, and per-token streaming. Run `pytest tests/test_async.py` to verify.

For Qwen3.5, set `TORCH_COMPILE_DISABLE=1` to avoid PyTorch recompilation overhead with variable-length inputs:

```bash
TORCH_COMPILE_DISABLE=1 python your_script.py
```

## Benchmark

See `benchmarks/run_bench.sh`. Hardware: NVIDIA RTX 3090 (24GB).

### Qwen3-0.6B (128 seqs, 100-512 in, 100-512 out)

| Engine | Throughput |
|--------|-----------|
| big-vLLM | **6,515 tok/s** |
| vLLM | 6,347 tok/s |

### Qwen3.5-0.8B (8 seqs, 100-200 in, 100-200 out)

| Engine | Throughput |
|--------|-----------|
| vLLM | 1,789 tok/s |
| big-vLLM | 1,018 tok/s |

## Why TORCH_COMPILE_DISABLE?

PyTorch's `torch.compile` (`@torch.compile` decorator) is used on several nano-vLLM kernels (RoPE, RMSNorm, Attention). Under variable batch sizes — especially in Qwen3.5 where prefill can be hundreds of tokens and decode is a single token — the compiler hits `recompile_limit` and recompiles the same functions repeatedly. This adds more overhead than eager execution, causing a net slowdown.

Disabling `torch.compile` avoids this recompilation thrash and results in faster inference for Qwen3.5.

## Acknowledgments

Forked from [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) by Xingkai Yu. Built with [flash-attn](https://github.com/Dao-AILab/flash-attention), [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d), and [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
