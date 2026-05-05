import os
import time
import argparse
from random import randint, seed


def main():
    parser = argparse.ArgumentParser(description="均衡负载场景 — 原始 bench.py 配置 + eager 模式")
    parser.add_argument("--engine", choices=["nanovllm", "vllm"], default="nanovllm")
    args = parser.parse_args()

    if args.engine == "nanovllm":
        from nanovllm import LLM, SamplingParams
    else:
        from vllm import LLM, SamplingParams

    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/models/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=True, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    if args.engine == "vllm":
        prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Warmup"], SamplingParams() if args.engine == "nanovllm" else SamplingParams(max_tokens=1))
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Scenario: balanced | Engine: {args.engine}")
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
