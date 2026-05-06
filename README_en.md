# Big-vLLM

> 中文文档: [README.md](README.md)

A high-performance LLM inference engine forked from [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), with native support for hybrid-attention models like Qwen3.5.

Supported model families: `Qwen2` (including `Qwen2.5`), `Qwen3`, and `Qwen3.5`.

## Key Features

- **Fast offline inference** — Competitive with vLLM across Qwen3 and Qwen3.5
- **Native hybrid-attention** — Hand-written GatedDeltaNet for Qwen3.5, no HuggingFace model dependency
- **Async streaming API** — `AsyncLLM` with `generate()` async generator, supports concurrent requests and abort
- **CUDA graph** — Zero-overhead kernel replay for decode
- **Paged KV cache** — Efficient memory management with prefix caching
- **Quantized models** — Supports FP8 (rtn) and 4-bit W4A16-G128 via `compressed-tensors` format, with automatic dequantization on load
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

Quantized models can be produced from FP16 checkpoints using [llm-compressor](https://github.com/vllm-project/llm-compressor):

```bash
llm-compressor compress --model ~/huggingface/Qwen3-8B \
  --scheme w4a16_g128 \
  --output ~/huggingface/Qwen3-8B-W4A16-G128
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

### Qwen3-8B (4 seqs, 50-100 in, 50-100 out)

| Model | big-VLLM | vLLM |
|-------|----------|------|
| FP16 | 75 tok/s | 77 tok/s |
| W4A16-G128 | 75 tok/s | 88 tok/s |

W4A16-G128 uses ~5 GB vs ~16 GB for FP16 — 3x memory reduction with negligible quality loss.

## Quantization

big-VLLM supports [llm-compressor](https://github.com/vllm-project/llm-compressor) quantized models (`compressed-tensors` format) with automatic dequantization during weight loading. No special flags needed — just pass the model path:

```python
llm = LLM("~/huggingface/Qwen3-8B-W4A16-G128", enforce_eager=False)
```

Supported formats:

| Format | Bit-width | Example | Dequant |
|--------|-----------|---------|---------|
| W4A16-G128 | 4-bit weights, group 128 | `weight_packed` + `weight_scale` | `u4 → s4 → scale` |
| FP8 (rtn) | 8-bit float, block [128,128] | `float8_e4m3fn` + `weight_scale` | `fp8 → scale` |

The dequantization happens in `load_model()` — packed weights are unpacked, scaled, and converted to float16 before copying into model parameters. Output quality is near-identical to FP16 (cos similarity > 0.99). Run `python tests/test_quant.py` to verify.

## Why TORCH_COMPILE_DISABLE?

PyTorch's `torch.compile` (`@torch.compile` decorator) is used on several nano-vLLM kernels (RoPE, RMSNorm, Attention). Under variable batch sizes — especially in Qwen3.5 where prefill can be hundreds of tokens and decode is a single token — the compiler hits `recompile_limit` and recompiles the same functions repeatedly. This adds more overhead than eager execution, causing a net slowdown.

Disabling `torch.compile` avoids this recompilation thrash and results in faster inference for Qwen3.5.

## Tests

```bash
# Quick regression (core models, ~2 min)
bash tests/regression.sh --quick

# Full regression (includes 4B models, ~5 min)
bash tests/regression.sh

# Unit tests
python -m pytest tests/test_async.py tests/test_quant.py -v
```

## Development

- Always branch from `main`, never commit directly
- Run `bash tests/regression.sh --quick` before pushing
- Compare with upstream: `git diff upstream..main`

## Acknowledgments

Forked from [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) by Xingkai Yu. Built with [flash-attn](https://github.com/Dao-AILab/flash-attention), [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d), and [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
