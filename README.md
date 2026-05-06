# Big-vLLM

> English docs: [README_en.md](README_en.md)

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 二次开发的高性能 LLM 推理引擎，原生支持 Qwen3.5 等混合注意力架构模型。

支持的模型架构：`Qwen2`（含 `Qwen2.5`）、`Qwen3`、`Qwen3.5`。

## 核心特性

- **高性能离线推理** — Qwen3 graph 模式与 vLLM 持平或略优，Qwen3.5 原生 GatedDeltaNet
- **原生混合注意力** — 手写 GatedDeltaNet，不依赖 HuggingFace 模型代码
- **异步流式 API** — `AsyncLLM` 支持异步生成器、并发请求和取消
- **CUDA 图** — decode 阶段零开销 kernel 回放（含 GDN 状态管理）
- **量化模型** — 支持 FP8 (rtn) 和 4-bit W4A16-G128（`compressed-tensors` 格式），加载时自动反量化
- **代码简洁** — 核心约 1500 行 Python

## 功能清单

- [x] Qwen3.5 原生 GatedDeltaNet 支持（`qwen35.py`）
- [x] CUDA Graph for GDN 模型（状态 save/restore，skip warmup capture）
- [x] W4A16-G128 / FP8 (rtn) 量化支持（`compressed-tensors` 格式）
- [x] 异步流式 API（`AsyncLLM.generate()`，并发 + abort）
- [x] GDN kernel 优化（gate 融合 Triton、GVA 原生处理、RMSNorm 重构对齐 Qwen3）
- [x] 回归测试（12 个模型 × 模式组合）+ 单元测试
- [x] Benchmark 系统（`benchmarks/bench_all.sh`）
- [x] 中英双语文档
- [x] 与上游 nano-vllm 保持同步（fork 自 `bb823b3`）

## 安装

```bash
pip install git+https://github.com/duchengyao/big-vllm.git
```

## 模型下载

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False

huggingface-cli download --resume-download Qwen/Qwen3.5-0.8B \
  --local-dir ~/huggingface/Qwen3.5-0.8B/ \
  --local-dir-use-symlinks False
```

量化模型可通过 [llm-compressor](https://github.com/vllm-project/llm-compressor) 从 FP16 权重压缩：

```bash
llm-compressor compress --model ~/huggingface/Qwen3-8B \
  --scheme w4a16_g128 \
  --output ~/huggingface/Qwen3-8B-W4A16-G128
```

## 快速开始

### 同步调用

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3.5-0.8B", enforce_eager=False, max_model_len=1024)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["你好，我叫"], sampling_params)
print(outputs[0]["text"])
```

### 异步流式

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

支持并发请求、取消和逐 token 流式输出。

Qwen3.5 建议设置 `TORCH_COMPILE_DISABLE=1`，避免 PyTorch 对变长输入反复 JIT 编译：

```bash
TORCH_COMPILE_DISABLE=1 python your_script.py
```

## 性能测试

测试环境：NVIDIA GeForce RTX 3090 (24G)。完整脚本：`bash benchmarks/bench_all.sh`。

### Qwen3-0.6B（128 序列, in 100-512, out 100-512）

| 模式 | big-VLLM | vLLM | 对比 |
|------|----------|------|-----|
| graph | 6,337 tok/s | 6,268 tok/s | **101%** |
| eager | 2,472 tok/s | 3,352 tok/s | 74% |

### Qwen3.5-0.8B（8 序列, in 100-200, out 100-200）

| 模式 | big-VLLM | vLLM | 对比 |
|------|----------|------|-----|
| graph | 986 tok/s | 1,784 tok/s | 55% |
| eager | 122 tok/s | 215 tok/s | 57% |
| W4A16 eager | 117 tok/s | 不可用 | - |

### Qwen3.5-4B（4 序列, in 100-200, out 100-200）

| 模式 | big-VLLM | vLLM | 对比 |
|------|----------|------|-----|
| graph | 187 tok/s | 243 tok/s | 77% |
| eager | 45 tok/s | 83 tok/s | 54% |

> vLLM 无法加载 Qwen3.5 W4A16 模型（缺少 `preprocessor_config.json` 且多模态 config 兼容性问题）。

## 量化

big-VLLM 支持 [llm-compressor](https://github.com/vllm-project/llm-compressor) 压缩的模型（`compressed-tensors` 格式），加载时自动反量化：

```python
llm = LLM("~/huggingface/Qwen3-8B-W4A16-G128", enforce_eager=False)
```

| 格式 | 位宽 | 存储方式 | 反量化 |
|------|------|---------|-------|
| W4A16-G128 | 4-bit 权重，group 128 | `weight_packed` + `weight_scale` | `int32 → u4 → s4 → float × scale` |
| FP8 (rtn) | 8-bit 浮点 | `float8_e4m3fn` + `weight_scale` | `fp8 → bf16 × scale` |

反量化在 `load_model()` 中完成。Qwen3.5 0.8B W4A16 prefill cosine similarity > 0.998，输出质量接近 FP16。运行 `python tests/test_quant.py` 验证。

## 为什么要 TORCH_COMPILE_DISABLE？

Qwen3.5 的 prefill 序列长度变化剧烈（数百 token）而 decode 固定为 1 token。PyTorch 的 `@torch.compile` 遇到新 shape 时会重新编译，频繁触发 `recompile_limit`，额外开销比 eager 执行还大。禁用 compile 避免编译抖动。

## CUDA Graph 与 GDN 状态管理

Qwen3.5 的 GatedDeltaNet 每层都有循环状态（conv_state 和 recurrent_state）。big-VLLM 的处理：

1. **init 时不捕获** — GDN 模型的图不在初始化时创建
2. **warmup 不触发重捕获** — 预热阶段临时跳过
3. **prefill 后捕获** — 用真实 prefill 初始化状态后再 capture graph，replay 时 save/restore 状态

## TODO / 已知问题

- [ ] Qwen3.5 性能与 vLLM 差距约 2×（GDN 未用 Triton 全融合 kernel，当前用 B-FLA 逐算子调用）
- [ ] Qwen3.5 eager 模式比 vLLM eager 慢（`torch.compile` 整体禁用，缺少选择性开启机制）
- [ ] Qwen3.5 W4A16 4B decode 退化（GDN 循环结构在 4-bit 量化下误差累积，社区无银弹）
- [ ] 多序列 batch GDN 状态冲突（当前 Qwen3.5 串行调度，throughput 无法随并发数线性增长）
- [ ] 张量并行未支持（`qwen35.py` 中 `assert tp_size == 1`）
- [ ] vLLM 无法加载 Qwen3.5 W4A16 模型（需补充 preprocessor_config 等文件）
- [ ] Qwen3.5 4B FP8 模型尚未产出

## 测试

```bash
# 快速回归测试（核心场景，约 2 分钟）
bash tests/regression.sh --quick

# 完整回归测试（含 4B 模型，约 5 分钟）
bash tests/regression.sh

# 单元测试
python -m pytest tests/test_qwen35.py tests/test_async.py -v

# 量化测试
python tests/test_quant.py
```

## 致谢

Fork 自 Xingkai Yu 的 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)。基于 [flash-attn](https://github.com/Dao-AILab/flash-attention)、[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) 和 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 构建。
