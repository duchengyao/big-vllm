# Big-vLLM

> English docs: [README.md](README.md)

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 二次开发的高性能 LLM 推理引擎，原生支持 Qwen3.5 等混合注意力架构模型。

支持的模型架构：`Qwen2`（含 `Qwen2.5`）、`Qwen3`、`Qwen3.5`。

## 核心特性

- **高性能离线推理** — 在 Qwen3 和 Qwen3.5 上性能与 vLLM 持平或更优
- **原生混合注意力** — 手写 GatedDeltaNet，不依赖 HuggingFace 模型代码
- **异步流式 API** — `AsyncLLM` 支持 `generate()` 异步生成器、并发请求和取消
- **CUDA 图** — decode 阶段零开销 kernel 回放
- **分页 KV 缓存** — 高效显存管理，支持 prefix caching
- **量化模型** — 支持 FP8 (rtn) 和 4-bit W4A16-G128（`compressed-tensors` 格式），加载时自动反量化
- **代码简洁** — 核心约 1500 行 Python

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

量化模型可通过 [llm-compressor](https://github.com/vllm-project/llm-compressor) 从 FP16 权重自行压缩：

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

支持并发请求、取消和逐 token 流式输出。运行 `pytest tests/test_async.py` 验证。

Qwen3.5 建议设置 `TORCH_COMPILE_DISABLE=1`，避免 PyTorch 对变长输入反复 JIT 编译造成的性能下降：

```bash
TORCH_COMPILE_DISABLE=1 python your_script.py
```

## 性能测试

测试脚本：`benchmarks/run_bench.sh`。硬件：NVIDIA RTX 3090 (24GB)。

### Qwen3-0.6B (128 序列，100-512 输入，100-512 输出)

| 引擎 | 吞吐量 |
|--------|-----------|
| big-vLLM | **6,515 tok/s** |
| vLLM | 6,347 tok/s |

### Qwen3.5-0.8B (8 序列，100-200 输入，100-200 输出)

| 引擎 | 吞吐量 |
|--------|-----------|
| vLLM | 1,789 tok/s |
| big-vLLM | 1,018 tok/s |

### Qwen3-8B (4 序列，50-100 输入，50-100 输出)

| 模型 | big-VLLM | vLLM |
|-------|----------|------|
| FP16 | 75 tok/s | 77 tok/s |
| W4A16-G128 | 75 tok/s | 88 tok/s |

W4A16-G128 显存约 5 GB，相比 FP16 的 16 GB 节省 3 倍，质量损失可忽略。

## 量化

big-VLLM 支持 [llm-compressor](https://github.com/vllm-project/llm-compressor) 压缩的模型（`compressed-tensors` 格式），加载时自动反量化，无需额外配置：

```python
llm = LLM("~/huggingface/Qwen3-8B-W4A16-G128", enforce_eager=False)
```

支持的格式：

| 格式 | 位宽 | 存储方式 | 反量化 |
|--------|-----------|---------|---------|
| W4A16-G128 | 4-bit 权重，group 128 | `weight_packed` + `weight_scale` | `u4 → s4 → scale` |
| FP8 (rtn) | 8-bit 浮点，block [128,128] | `float8_e4m3fn` + `weight_scale` | `fp8 → scale` |

反量化在 `load_model()` 中完成——压缩权重被解包、缩放并转换为 float16，然后复制到模型参数中。输出质量与 FP16 几乎一致（余弦相似度 > 0.99）。运行 `python tests/test_quant.py` 验证。

## 为什么要 TORCH_COMPILE_DISABLE？

PyTorch 的 `torch.compile`（即 `@torch.compile` 装饰器）被用在 nano-vLLM 的几个 kernel 上（RoPE、RMSNorm、Attention）。当 batch size 变化剧烈时——Qwen3.5 的 prefill 可能数百 token 而 decode 只有单个 token——编译器会达到 `recompile_limit` 反复重新编译同一函数，额外开销比 eager 执行还大。

禁用 `torch.compile` 可以避免这种编译抖动，对 Qwen3.5 有显著的性能提升。

## CUDA Graph 与 GDN 状态管理

Qwen3.5 的 GatedDeltaNet 每层都有循环状态（conv_state 和 recurrent_state），在 CUDA graph 捕获时如果使用了错误的初始状态，decode 输出会退化（重复 token 或乱码）。big-VLLM 的处理方式：

1. **init 时不捕获** — GDN 模型的图不在初始化时创建
2. **warmup 不触发重捕获** — 预热阶段临时跳过，确保只有真正 prefill 后才捕获
3. **状态先存后恢复** — prefill 后的捕获流程：保存 GDN 状态 → 跑图预热 → 创建新图 → 恢复真实状态

## 测试

```bash
# 快速回归测试（核心场景，约 2 分钟）
bash tests/regression.sh --quick

# 完整回归测试（含 4B 模型，约 5 分钟）
bash tests/regression.sh

# 单元测试
python -m pytest tests/test_async.py tests/test_quant.py -v
```

## 开发规范

- 不要在 `main` 分支直接改代码，所有修改从 `main` 新建分支
- 改完跑 `bash tests/regression.sh --quick` 验证无回归
- 对比上游：`git diff upstream..main`

## 致谢

Fork 自 Xingkai Yu 的 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)。基于 [flash-attn](https://github.com/Dao-AILab/flash-attention)、[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) 和 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 构建。
