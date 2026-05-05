Nano-vLLM vs vLLM Benchmark Results
=====================================
Date: 2026-05-02
Hardware: NVIDIA GeForce RTX 3090 (24GB)
Model: Qwen3-0.6B
Config: max_model_len=4096, enforce_eager=True (CUDA graph disabled)

1. CUDA Graph Enabled (original bench.py)
   Both engines use CUDA graph replay. Difference is small because GPU
   compute dominates, and both use the same FlashAttention kernels.

   | Engine     | Output Tokens | Time (s) | Throughput (tok/s) |
   |------------|---------------|----------|--------------------|
   | Nano-vLLM  |       133,966 |    28.80 |           4,651.68 |
   | vLLM       |       133,966 |    27.84 |           4,811.63 |
   | Gap        |               |          |        vLLM +3.4%  |

2. CUDA Graph Disabled (enforce_eager=True)
   vLLM's scheduler and memory management advantages become visible.

   Scenario: decode-heavy (128 seqs, input 64-256 tok, output 1024-2048 tok)
   Script: bench_decode_heavy.py

   | Engine     | Output Tokens | Time (s) | Throughput (tok/s) |
   |------------|---------------|----------|--------------------|
   | Nano-vLLM  |       194,902 |    57.04 |           3,416.66 |
   | vLLM       |       194,902 |    45.21 |           4,310.57 |
   | Gap        |               |          |       vLLM +26.2%  |

   Scenario: balanced (256 seqs, input 100-1024 tok, output 100-1024 tok)
   Script: bench_balanced.py

   | Engine     | Output Tokens | Time (s) | Throughput (tok/s) |
   |------------|---------------|----------|--------------------|
   | Nano-vLLM  |       137,221 |    42.32 |           3,242.17 |
   | vLLM       |       137,221 |    31.37 |           4,373.88 |
   | Gap        |               |          |       vLLM +34.9%  |

3. vLLM Key Advantages (from code analysis)

   a) More efficient scheduler — can interleave prefill and decode
      more aggressively within a single step.

   b) Full multi-sequence chunked prefill — vLLM can chunk-prefill
      many sequences simultaneously; nano-vllm only chunks the first
      sequence (nanovllm/engine/scheduler.py line 42).

   c) Preemption with CPU swap — when KV cache is over-subscribed,
      vLLM swaps evicted blocks to CPU memory; nano-vLLM discards
      all KV cache and recomputes from scratch.

   d) Better batch efficiency — vLLM packs more tokens into each
      forward pass, especially important for decode-heavy workloads.

Usage (reproduction):
  # With CUDA graph (default):
  python bench_compare.py --scenario long-prefill --engine nanovllm
  python bench_compare.py --scenario long-prefill --engine vllm
  python bench_compare.py --scenario kv-pressure --engine nanovllm
  python bench_compare.py --scenario kv-pressure --engine vllm
  # Without CUDA graph:
  python bench_compare.py --scenario decode-heavy --engine nanovllm --no-cuda-graph
  python bench_compare.py --scenario decode-heavy --engine vllm --no-cuda-graph
  python bench_compare.py --scenario balanced --engine nanovllm --no-cuda-graph
  python bench_compare.py --scenario balanced --engine vllm --no-cuda-graph
