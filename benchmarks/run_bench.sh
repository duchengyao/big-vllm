#!/bin/bash
# Benchmark nano-VLLM vs vLLM on Qwen3 and Qwen3.5

SCRIPT="benchmarks/bench_compare.py"

echo "========================================="
echo "  nano-VLLM vs vLLM Benchmark"
echo "========================================="

for scenario in qwen3 qwen35; do
    echo ""
    echo "--- $scenario ---"
    for engine in nanovllm vllm; do
        echo -n "  $engine: "
        python "$SCRIPT" --engine "$engine" --scenario "$scenario" 2>&1 | grep RESULT || echo "FAILED"
    done
done

echo ""
echo "Done."
