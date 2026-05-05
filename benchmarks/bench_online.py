import asyncio
import os
import time
import argparse
from dataclasses import dataclass
from random import randint, seed


SCENARIOS = {
    "online": dict(
        desc="Continuous arrival (online serving)",
        initial_burst=32,
        inject_every=0.5,
        inject_count=4,
        total_requests=256,
        min_input=100,
        max_input=1024,
        min_output=100,
        max_output=1024,
    ),
    "long-prefill": dict(
        desc="Burst of long prompts to stress chunked prefill",
        initial_burst=32,
        inject_every=999,
        inject_count=0,
        total_requests=32,
        min_input=2048,
        max_input=3072,
        min_output=64,
        max_output=128,
    ),
}


@dataclass
class RequestStats:
    request_id: str
    prompt_len: int
    output_len: int
    submit_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0
    token_count: int = 0

    @property
    def ttft(self):
        return self.first_token_time - self.submit_time

    @property
    def latency(self):
        return self.finish_time - self.submit_time


def make_engine(args):
    path = os.path.expanduser("~/huggingface/models/Qwen3-0.6B/")
    if args.engine == "nanovllm":
        from nanovllm import AsyncLLM
        return AsyncLLM(path)
    else:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM as VllmAsyncLLM
        engine_args = AsyncEngineArgs(
            model=path, enforce_eager=False, disable_log_stats=True,
            max_model_len=4096,
        )
        return VllmAsyncLLM.from_engine_args(engine_args)


async def run_benchmark(engine, args, cfg):
    if args.engine == "nanovllm":
        from nanovllm import SamplingParams
    else:
        from vllm import SamplingParams

    seed(0)
    stats: dict[str, RequestStats] = {}

    async def submit_one(req_id, input_len, output_len):
        prompt = [randint(0, 10000) for _ in range(input_len)]
        sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        st = RequestStats(req_id, input_len, output_len)
        st.submit_time = time.time()
        stats[req_id] = st

        async for out in engine.generate(prompt, sp, request_id=req_id):
            if not st.first_token_time:
                st.first_token_time = time.time()
            st.token_count += len(out.outputs[0].token_ids)
            if out.finished:
                st.finish_time = time.time()
                return

    warmup = SamplingParams() if args.engine == "nanovllm" else SamplingParams(max_tokens=1)
    async for _ in engine.generate("Warmup", warmup, request_id="__warmup__"):
        pass
    t0 = time.time()

    tasks = []
    for i in range(cfg["initial_burst"]):
        inp = randint(cfg["min_input"], cfg["max_input"])
        out = randint(cfg["min_output"], cfg["max_output"])
        tasks.append(asyncio.create_task(submit_one(f"req-{i}", inp, out)))

    submitted = cfg["initial_burst"]
    while submitted < cfg["total_requests"]:
        await asyncio.sleep(cfg["inject_every"])
        for _ in range(cfg["inject_count"]):
            if submitted >= cfg["total_requests"]:
                break
            inp = randint(cfg["min_input"], cfg["max_input"])
            out = randint(cfg["min_output"], cfg["max_output"])
            tasks.append(asyncio.create_task(submit_one(f"req-{submitted}", inp, out)))
            submitted += 1

    await asyncio.wait(tasks)
    total_time = time.time() - t0
    total_time = time.time() - t0
    total_output = sum(s.output_len for s in stats.values())

    ttfts = [s.ttft for s in stats.values() if s.first_token_time]
    latencies = [s.latency for s in stats.values() if s.finish_time]
    ttfts.sort()
    latencies.sort()

    def pct(data, p):
        if not data:
            return 0
        return data[int(len(data) * p / 100)]

    print(f"  Scenario:           {args.scenario} ({cfg['desc']})")
    print(f"  Total requests:     {len(stats)}")
    print(f"  Total output tokens:{total_output}")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Throughput:         {total_output / total_time:.2f} tok/s")
    print(f"  TTFT avg:           {sum(ttfts)/len(ttfts):.3f}s" if ttfts else "  TTFT avg: N/A")
    print(f"  TTFT p50/p99:       {pct(ttfts, 50):.3f}s / {pct(ttfts, 99):.3f}s")
    print(f"  Latency avg:        {sum(latencies)/len(latencies):.3f}s" if latencies else "  Latency avg: N/A")
    print(f"  Latency p50/p99:    {pct(latencies, 50):.3f}s / {pct(latencies, 99):.3f}s")

    print(f"RESULT {args.scenario} {args.engine} {total_output} {total_time:.2f} {total_output/total_time:.2f} "
          f"{pct(ttfts, 50):.3f} {pct(ttfts, 99):.3f} "
          f"{pct(latencies, 50):.3f} {pct(latencies, 99):.3f}",
          flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="online")
    parser.add_argument("--engine", choices=["nanovllm", "vllm"], required=True)
    args = parser.parse_args()
    cfg = SCENARIOS[args.scenario]
    print(f"Scenario: {args.scenario} — {cfg['desc']}")
    engine = make_engine(args)
    try:
        await run_benchmark(engine, args, cfg)
    finally:
        if args.engine == "nanovllm":
            engine.exit()
        else:
            engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
