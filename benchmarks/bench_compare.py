"""Unified benchmark: nano-VLLM vs vLLM on Qwen3 / Qwen3.5."""

import argparse, os, sys, time
from random import randint, seed


SCENARIOS = {
    "qwen3": {
        "model": os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        "num_seqs": 128,
        "min_in": 100, "max_in": 512,
        "min_out": 100, "max_out": 512,
        "max_model_len": 4096,
        "enforce_eager": False,
    },
    "qwen35": {
        "model": os.path.expanduser("~/huggingface/Qwen3.5-0.8B"),
        "num_seqs": 8,
        "min_in": 100, "max_in": 200,
        "min_out": 100, "max_out": 200,
        "max_model_len": 1024,
        "enforce_eager": False,
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["nanovllm", "vllm"], required=True)
    p.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    args = p.parse_args()

    cfg = SCENARIOS[args.scenario]
    model = cfg["model"]
    if not os.path.isdir(model):
        print(f"ERROR: model not found: {model}", flush=True)
        sys.exit(1)

    seed(0)
    if args.engine == "nanovllm":
        if "qwen35" in args.scenario:
            os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        from nanovllm import LLM, SamplingParams
        llm = LLM(model, enforce_eager=cfg["enforce_eager"],
                  max_model_len=cfg["max_model_len"], gpu_memory_utilization=0.85)
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model, enforce_eager=False,
                  max_model_len=cfg["max_model_len"], gpu_memory_utilization=0.85,
                  disable_log_stats=True)

    # Warmup
    llm.generate(["Warmup"], SamplingParams(max_tokens=1))

    # Generate random requests
    prompts, params = [], []
    for _ in range(cfg["num_seqs"]):
        ilen = randint(cfg["min_in"], cfg["max_in"])
        olen = randint(cfg["min_out"], cfg["max_out"])
        if args.engine == "vllm":
            prompts.append(dict(prompt_token_ids=[randint(0, 10000) for _ in range(ilen)]))
        else:
            prompts.append([randint(0, 10000) for _ in range(ilen)])
        params.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=olen))

    t0 = time.time()
    llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.time() - t0

    total_out = sum(p.max_tokens for p in params)
    tp = total_out / elapsed
    print(f"RESULT {args.scenario} {args.engine} {total_out} {elapsed:.2f} {tp:.0f}")


if __name__ == "__main__":
    main()
