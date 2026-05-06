#!/bin/bash
# Comprehensive benchmark: big-VLLM vs vLLM across all models.
# Usage: bash benchmarks/bench_all.sh

VLLM_VENV=~/project/vllm/.venv/bin/python
BENCH=benchmarks/bench.py
TMPFILE=$(mktemp)

bench_one() {
    local engine=$1 model=$2 eager=$3 extra=$4 label=$5
    local py eager_flag=""
    [ "$engine" = "nanovllm" ] && py=python || py=$VLLM_VENV
    [ "$eager" = "true" ] && eager_flag="--enforce-eager"
    printf "  %-35s " "$label ($engine)"
    local out=$($py $BENCH --engine $engine --model "$model" --num-seqs $num_seqs \
        --min-in $min_in --max-in $max_in --min-out $min_out --max-out $max_out \
        --max-model-len $max_len $eager_flag $extra 2>/dev/null | grep RESULT)
    if [ -n "$out" ]; then
        local tok_s=$(echo "$out" | awk '{printf "%d", $5}')
        printf "%7s tok/s\n" "$tok_s"
        echo "$label|$engine|$tok_s" >> $TMPFILE
    else
        echo "  FAILED"
        echo "$label|$engine|0" >> $TMPFILE
    fi
}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
GPU_MEM_G=$((GPU_MEM / 1024))
echo "=============================================="
echo "  big-VLLM vs vLLM — Full Benchmark"
echo "  $(date)"
echo "  GPU: $GPU_NAME (${GPU_MEM_G}G)"
echo "=============================================="
echo ""

# --- Qwen3-0.6B ---
num_seqs=128; min_in=100; max_in=512; min_out=100; max_out=512; max_len=4096
echo "--- Qwen3-0.6B (128 seqs, in 100-512, out 100-512) ---"
bench_one nanovllm ~/huggingface/Qwen3-0.6B false ""       "qwen3-0.6B graph"
bench_one vllm     ~/huggingface/Qwen3-0.6B false ""       "qwen3-0.6B graph"
bench_one nanovllm ~/huggingface/Qwen3-0.6B true  ""       "qwen3-0.6B eager"
bench_one vllm     ~/huggingface/Qwen3-0.6B true  ""       "qwen3-0.6B eager"

echo ""

# --- Qwen3.5-0.8B ---
num_seqs=8; min_in=100; max_in=200; min_out=100; max_out=200; max_len=1024
echo "--- Qwen3.5-0.8B (8 seqs, in 100-200, out 100-200) ---"
bench_one nanovllm ~/huggingface/Qwen3.5-0.8B false "--torch-compile-disable" "qwen35-0.8B graph"
bench_one vllm     ~/huggingface/Qwen3.5-0.8B false ""                           "qwen35-0.8B graph"
bench_one nanovllm ~/huggingface/Qwen3.5-0.8B true  "--torch-compile-disable" "qwen35-0.8B eager"
bench_one vllm     ~/huggingface/Qwen3.5-0.8B true  ""                           "qwen35-0.8B eager"

echo ""

# --- Qwen3.5-4B ---
num_seqs=4; min_in=100; max_in=200; min_out=100; max_out=200; max_len=2048
echo "--- Qwen3.5-4B (4 seqs, in 100-200, out 100-200) ---"
bench_one nanovllm ~/huggingface/Qwen3.5-4B false "--torch-compile-disable" "qwen35-4B graph"
bench_one vllm     ~/huggingface/Qwen3.5-4B false ""                           "qwen35-4B graph"
bench_one nanovllm ~/huggingface/Qwen3.5-4B true  "--torch-compile-disable" "qwen35-4B eager"
bench_one vllm     ~/huggingface/Qwen3.5-4B true  ""                           "qwen35-4B eager"

echo ""

# --- Qwen3.5-0.8B W4A16 ---
num_seqs=8; min_in=100; max_in=200; min_out=100; max_out=200; max_len=1024
echo "--- Qwen3.5-0.8B W4A16 (8 seqs) ---"
bench_one nanovllm ~/huggingface/Qwen3.5-0.8B-W4A16-G128 true "--torch-compile-disable" "qwen35-0.8B-W4A16 eager"
bench_one vllm     ~/huggingface/Qwen3.5-0.8B-W4A16-G128 true ""                           "qwen35-0.8B-W4A16 eager"

echo ""
echo "=============================================="
echo "  Summary"
echo "=============================================="
declare -A nano_val vllm_val
while IFS='|' read -r label engine tok_s; do
    if [ "$engine" = "nanovllm" ]; then
        nano_val["$label"]=$tok_s
    else
        vllm_val["$label"]=$tok_s
    fi
done < $TMPFILE

printf "\n%-35s %10s %10s %8s\n" "Test" "big-VLLM" "vLLM" "vs"
printf "%-35s %10s %10s %8s\n" "----" "--------" "----" "--"
for label in "${!nano_val[@]}"; do
    nv="${nano_val[$label]}"
    vv="${vllm_val[$label]}"
    if [ "$nv" != "0" ] && [ "$vv" != "0" ]; then
        pct=$(awk "BEGIN {printf \"%.0f%%\", ($nv/$vv)*100}")
    else
        pct="N/A"
    fi
    printf "%-35s %8s %8s %8s\n" "$label" "$nv" "$vv" "$pct"
done | sort -k2 -rn

echo ""
rm -f $TMPFILE
echo "Done."
