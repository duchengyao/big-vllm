"""Benchmark: nano-VLLM vs vLLM. All params from CLI, driven by run_bench.sh."""

import argparse, os, time
from random import randint, seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["nanovllm", "vllm"], required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--num-seqs", type=int, required=True)
    p.add_argument("--min-in", type=int, required=True)
    p.add_argument("--max-in", type=int, required=True)
    p.add_argument("--min-out", type=int, required=True)
    p.add_argument("--max-out", type=int, required=True)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--torch-compile-disable", action="store_true")
    args = p.parse_args()

    model = os.path.expanduser(args.model)
    if not os.path.isdir(model):
        p.error(f"model not found: {model}")

    seed(0)
    if args.torch_compile_disable and args.engine == "nanovllm":
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    if args.engine == "nanovllm":
        from nanovllm import LLM, SamplingParams
    else:
        from vllm import LLM, SamplingParams

    llm = LLM(model, enforce_eager=args.enforce_eager,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
              **({"disable_log_stats": True} if args.engine == "vllm" else {}))

    llm.generate(["Warmup"], SamplingParams(max_tokens=1))

    prompts, params = [], []
    for _ in range(args.num_seqs):
        ilen = randint(args.min_in, args.max_in)
        olen = randint(args.min_out, args.max_out)
        if args.engine == "vllm":
            prompts.append(dict(prompt_token_ids=[randint(0, 10000) for _ in range(ilen)]))
        else:
            prompts.append([randint(0, 10000) for _ in range(ilen)])
        params.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=olen))

    t0 = time.time()
    llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.time() - t0
    total_out = sum(p.max_tokens for p in params)
    print(f"RESULT {args.engine} {total_out} {elapsed:.2f} {total_out / elapsed:.0f}")


if __name__ == "__main__":
    main()
