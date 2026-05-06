"""Regression tests for all supported models and modes.

Usage:
    python tests/test_regression.py [--quick]
    TORCH_COMPILE_DISABLE=1 python tests/test_regression.py

Each test runs in a subprocess to ensure clean GPU state.
"""

import os, subprocess, sys, argparse

PY = sys.executable


def make_script(model, enforce_eager, test_name, port):
    return f"""
import os, torch, sys
sys.path.insert(0, '.')

MODEL = {model!r}
ENFORCE_EAGER = {enforce_eager!r}

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ["MASTER_PORT"] = {port!r}
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch.distributed as dist
dist.init_process_group("nccl", world_size=1, rank=0)
torch.cuda.set_device(0)
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")

from nanovllm import LLM, SamplingParams

path = os.path.expanduser(MODEL)
try:
    llm = LLM(path, enforce_eager=ENFORCE_EAGER, max_model_len=512, gpu_memory_utilization=0.85)
except Exception as e:
    print("FAIL {test_name}: load error: " + str(e))
    sys.exit(1)

sp = SamplingParams(temperature=0.6, max_tokens=24)
out = llm.generate(
    ['Hello, my name is', 'The capital of France is'],
    sp
)
texts = [o["text"] if isinstance(o, dict) else o.outputs[0].text for o in out]

for i, text in enumerate(texts):
    stripped = text.strip()
    if not stripped:
        print("FAIL {test_name}: output " + str(i) + " empty")
        sys.exit(1)
    chars = [c for c in stripped[:16] if c.strip()]
    if len(chars) >= 4 and len(set(chars[:4])) == 1:
        print("FAIL {test_name}: output " + str(i) + " repeating: " + repr(text))
        sys.exit(1)

print("PASS {test_name}")
import torch; torch.cuda.empty_cache()
dist.destroy_process_group()
"""


TESTS = [
    ("qwen3-graph",        "~/huggingface/Qwen3-0.6B",         False),
    ("qwen30.6-eager",     "~/huggingface/Qwen3-0.6B",         True),
    ("qwen35-0.8-eager",   "~/huggingface/Qwen3.5-0.8B",       True),
    ("qwen35-0.8-graph",   "~/huggingface/Qwen3.5-0.8B",       False),
    ("qwen35-4-eager",     "~/huggingface/Qwen3.5-4B",         True),
    ("qwen35-4-graph",     "~/huggingface/Qwen3.5-4B",         False),
    ("qwen35-8B-rtn-eager","~/huggingface/Qwen3.5-0.8B-rtn",   True),
    ("qwen3-8B-g128-eager", "~/huggingface/Qwen3-8B-W4A16-G128", True),
]

QUICK_TESTS = [
    ("qwen3-graph",        "~/huggingface/Qwen3-0.6B",         False),
    ("qwen35-0.8-eager",   "~/huggingface/Qwen3.5-0.8B",       True),
    ("qwen35-0.8-graph",   "~/huggingface/Qwen3.5-0.8B",       False),
    ("qwen3-8B-g128-eager", "~/huggingface/Qwen3-8B-W4A16-G128", True),
]


def run_one(name, model, eager, port):
    code = make_script(model, eager, name, port)
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=180)
    out = (r.stdout + r.stderr).strip()
    for line in out.split("\n"):
        if line.startswith("PASS") or line.startswith("FAIL"):
            print(f"  {line}")
            return line.startswith("PASS")
    print(f"  FAIL {name}: no output")
    print(f"  stderr: {r.stderr[-300:]}")
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    tests = QUICK_TESTS if args.quick else TESTS
    base_port = 30300
    passed, failed = 0, 0

    print(f"Running {len(tests)} tests...\n")
    for i, (name, model, eager) in enumerate(tests):
        if run_one(name, model, eager, str(base_port + i)):
            passed += 1
        else:
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
