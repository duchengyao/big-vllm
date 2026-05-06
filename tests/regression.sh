#!/bin/bash
# Regression test: run all model/mode combinations after any change.
# Each test is a separate python process.
# Usage: bash tests/regression.sh [--quick]

run_test() {
    local name=$1 model=$2 eager=$3 port=$4
    printf "  %-30s " "$name"
    REG_PORT=$port timeout 180 python -u -c "
import os, sys; sys.path.insert(0, '.')
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ['MASTER_PORT'] = os.environ['REG_PORT']
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
from nanovllm import LLM, SamplingParams
llm = LLM(os.path.expanduser('$model'), enforce_eager=$eager, max_model_len=512, gpu_memory_utilization=0.85)
sp = SamplingParams(temperature=0.6, max_tokens=24)
out = llm.generate(['Hello, my name is', 'The capital of France is'], sp)
texts = [o['text'] if isinstance(o, dict) else o.outputs[0].text for o in out]
for i, t in enumerate(texts):
    s = t.strip()
    if not s:
        print('FAIL: empty output'); raise SystemExit(1)
    ch = [c for c in s[:16] if c.strip()]
    if len(ch) >= 4 and len(set(ch[:4])) == 1:
        print('FAIL: repeating char'); raise SystemExit(1)
print('PASS')
" 2>&1 | grep -E '^PASS|^FAIL|^Traceback|Error' | head -1 || echo "CRASH"
}

QUICK=0
[ "$1" = "--quick" ] && QUICK=1

echo "=== Regression Tests ==="
echo ""

run_test "qwen3-0.6B (graph)"     "~/huggingface/Qwen3-0.6B"          False 30321
run_test "qwen3-0.6B (eager)"     "~/huggingface/Qwen3-0.6B"          True  30322
run_test "qwen35-0.8B (eager)"    "~/huggingface/Qwen3.5-0.8B"        True  30323
run_test "qwen35-0.8B (graph)"    "~/huggingface/Qwen3.5-0.8B"        False 30324
run_test "qwen3-8B-W4A16 (eager)" "~/huggingface/Qwen3-8B-W4A16-G128" True  30325

if [ "$QUICK" = "0" ]; then
    run_test "qwen35-4B (eager)"  "~/huggingface/Qwen3.5-4B"          True  30326
    run_test "qwen35-4B (graph)"  "~/huggingface/Qwen3.5-4B"          False 30327
    run_test "qwen35-0.8B-rtn"    "~/huggingface/Qwen3.5-0.8B-rtn"    True  30328
fi

echo ""
echo "Done."
