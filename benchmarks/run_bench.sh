#!/bin/bash
# nano-VLLM vs vLLM benchmark — all params here
SCRIPT="benchmarks/bench.py"

echo "===== Qwen3-0.6B ====="
for engine in nanovllm vllm; do
    echo -n "  $engine: "
    python "$SCRIPT" \
        --engine "$engine" \
        --model ~/huggingface/Qwen3-0.6B/ \
        --num-seqs 128 \
        --min-in 100 --max-in 512 \
        --min-out 100 --max-out 512 \
        --max-model-len 4096 \
        2>&1 | grep RESULT || echo "FAILED"
done

echo ""
echo "===== Qwen3.5-0.8B ====="
for engine in nanovllm vllm; do
    extra=""
    [ "$engine" = "nanovllm" ] && extra="--torch-compile-disable"
    echo -n "  $engine: "
    python "$SCRIPT" \
        --engine "$engine" \
        --model ~/huggingface/Qwen3.5-0.8B \
        --num-seqs 8 \
        --min-in 100 --max-in 200 \
        --min-out 100 --max-out 200 \
        --max-model-len 1024 \
        $extra \
        2>&1 | grep RESULT || echo "FAILED"
done

echo ""
echo "Done."
