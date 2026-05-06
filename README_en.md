# Big-vLLM

> 中文文档: [README.md](README.md)

A high-performance LLM inference engine forked from [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), with native support for hybrid-attention models like Qwen3.5.

Supported model families: `Qwen2` (including `Qwen2.5`), `Qwen3`, and `Qwen3.5`.

## Key Features

- **Fast offline inference** — Qwen3 graph mode matches or beats vLLM; native Qwen3.5 GatedDeltaNet
- **Native hybrid-attention** — Hand-written GatedDeltaNet for Qwen3.5, no HuggingFace model dependency
- **Async streaming API** — `AsyncLLM` with async generator, concurrent requests, and abort
- **CUDA graph** — Zero-overhead kernel replay for decode, including GDN state management
- **Quantized models** — Supports FP8 (rtn) and 4-bit W4A16-G128 via `compressed-tensors` format, auto-dequant on load
- **Readable codebase** — ~1,500 lines of Python

## Features Checklist

- [x] Native Qwen3.5 GatedDeltaNet support (`qwen35.py`)
- [x] CUDA Graph for GDN models (state save/restore, skip warmup capture)
- [x] W4A16-G128 / FP8 (rtn) quantization (`compressed-tensors` format)
- [x] Async streaming API (`AsyncLLM.generate()`, concurrent + abort)
- [x] GDN kernel optimizations (gate fusion into Triton, GVA native handling, RMSNorm refactor)
- [x] Regression tests (12 model × mode combos) + unit tests
- [x] Benchmark system (`benchmarks/bench_all.sh`)
- [x] Chinese + English bilingual docs
- [x] Synced with upstream nano-vllm (forked from `bb823b3`)

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

Test environment: NVIDIA GeForce RTX 3090 (24G). Full script: `bash benchmarks/bench_all.sh`.

### Qwen3-0.6B (128 seqs, in 100-512, out 100-512)

| Mode | big-VLLM | vLLM | vs |
|------|----------|------|-----|
| graph | 6,337 tok/s | 6,268 tok/s | **101%** |
| eager | 2,472 tok/s | 3,352 tok/s | 74% |

### Qwen3.5-0.8B (8 seqs, in 100-200, out 100-200)

| Mode | big-VLLM | vLLM | vs |
|------|----------|------|-----|
| graph | 986 tok/s | 1,784 tok/s | 55% |
| eager | 122 tok/s | 215 tok/s | 57% |
| W4A16 eager | 117 tok/s | N/A | - |

### Qwen3.5-4B (4 seqs, in 100-200, out 100-200)

| Mode | big-VLLM | vLLM | vs |
|------|----------|------|-----|
| graph | 187 tok/s | 243 tok/s | 77% |
| eager | 45 tok/s | 83 tok/s | 54% |

> vLLM cannot load Qwen3.5 W4A16 models (missing `preprocessor_config.json` + multimodal config compatibility issue).

## Quantization

big-VLLM supports [llm-compressor](https://github.com/vllm-project/llm-compressor) quantized models (`compressed-tensors` format) with automatic dequantization on load:

```python
llm = LLM("~/huggingface/Qwen3-8B-W4A16-G128", enforce_eager=False)
```

| Format | Bit-width | Storage | Dequant |
|--------|-----------|---------|---------|
| W4A16-G128 | 4-bit weights, group 128 | `weight_packed` + `weight_scale` | `int32 → u4 → s4 → float × scale` |
| FP8 (rtn) | 8-bit float | `float8_e4m3fn` + `weight_scale` | `fp8 → bf16 × scale` |

Dequantization happens in `load_model()`. Qwen3.5 0.8B W4A16 prefill cosine similarity > 0.998, output quality near FP16. Run `python tests/test_quant.py` to verify.

## Why TORCH_COMPILE_DISABLE?

Qwen3.5 prefill sequences vary widely in length (hundreds of tokens) while decode is fixed at 1 token. PyTorch's `@torch.compile` recompiles on new shapes, hitting `recompile_limit` and causing more overhead than eager execution. Disabling compile avoids this thrashing.

## CUDA Graph & GDN State Management

Qwen3.5's GatedDeltaNet has per-layer recurrent states (conv_state, recurrent_state). big-VLLM handles this by:

1. **Skip init capture** — GDN models don't capture graph at init
2. **Skip warmup recapture** — Warmup phase temporarily avoids capture
3. **Capture after prefill** — Real prefill initializes state, then graph is captured; replay saves/restores state

## TODO / Known Issues

- [ ] Qwen3.5 ~2× slower than vLLM (no full Triton fusion kernel for GDN, uses B-FLA per-op calls)
- [ ] Qwen3.5 eager slower than vLLM eager (`torch.compile` blanket-disabled, missing selective enable)
- [ ] Qwen3.5 W4A16 4B decode degradation (GDN recurrent structure quantization error accumulation, no community fix yet)
- [ ] Multi-seq batch GDN state conflict (Qwen3.5 serialized scheduling, throughput doesn't scale with concurrency)
- [ ] No tensor parallelism (`assert tp_size == 1` in qwen35.py)
- [ ] vLLM cannot load Qwen3.5 W4A16 models (needs preprocessor_config files)
- [ ] No Qwen3.5 4B FP8 model yet

## Tests

```bash
# Quick regression (core models, ~2 min)
bash tests/regression.sh --quick

# Full regression (includes 4B models, ~5 min)
bash tests/regression.sh

# Unit tests
python -m pytest tests/test_qwen35.py tests/test_async.py -v

# Quantization test
python tests/test_quant.py
```

## Acknowledgments

Forked from [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) by Xingkai Yu. Built with [flash-attn](https://github.com/Dao-AILab/flash-attention), [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d), and [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
