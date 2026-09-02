#!/usr/bin/env bash
# Sweep --n-cpu-moe. For each value it reports tokens/s and the bytes actually
# read from the SSD per generated token: the second number tells you whether the
# bottleneck is the disk or the CPU, which is the only way to know what to tune
# next. This is the first thing to re-run on a machine with different VRAM.
#
#   VALUES="48 47 46 45" ./bench-ncpumoe.sh
set -uo pipefail
export LC_ALL=C   # jq prints a decimal point; printf under some locales wants a comma

. "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

PORT="${PORT:-8099}"; export PORT
N_PREDICT="${N_PREDICT:-96}"
N_WARM="${N_WARM:-128}"          # warm the page cache up to steady state first
DEV="${DEV:-$(disk_of "$MODEL")}"
VALUES="${VALUES:-46 45 44 43}"
PROMPT="${PROMPT:-Explain in detail how an AVL tree works and why rebalancing is O(log n).}"

require_exec "$LLAMA_SERVER"
require_file "$MODEL"
[ -n "$DEV" ] || { log_error "could not determine the block device holding $MODEL; set DEV="; exit 1; }

ask() {   # $1 = n_predict -> the response JSON on stdout
    curl -sf --max-time 900 "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$PROMPT" --argjson n "$1" \
              '{prompt: $p, n_predict: $n, cache_prompt: false}')"
}

printf '%-7s %-9s %-12s %-11s %-11s %s\n' ncmoe tok/s prefill_t/s MB_from_SSD MB/token outcome
for n in $VALUES; do
    log="$LOG_DIR/ncpumoe-$n.log"
    NCMOE="$n" "$(dirname -- "${BASH_SOURCE[0]}")/serve.sh" >"$log" 2>&1 &
    srv=$!
    trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null' EXIT

    if ! wait_for_server "$srv" 240; then
        why=$(grep -oiE 'out of memory|cudaMalloc failed|failed to allocate' "$log" | head -1)
        printf '%-7s %-9s %-12s %-11s %-11s %s\n' "$n" - - - - "FAILED: ${why:-see $log}"
        kill $srv 2>/dev/null; wait $srv 2>/dev/null; trap - EXIT; continue
    fi

    ask "$N_WARM" >/dev/null 2>&1        # steady state: hot experts in the page cache
    b0=$(bytes_read "$DEV")
    out=$(ask "$N_PREDICT")
    b1=$(bytes_read "$DEV")

    ntok=$(jq -r '.timings.predicted_n // 0' <<<"$out" 2>/dev/null)
    if [ -z "$out" ] || [ "${ntok:-0}" -eq 0 ]; then
        why=$(grep -oiE 'out of memory|CUDA error|ggml_abort' "$log" | head -1)
        printf '%-7s %-9s %-12s %-11s %-11s %s\n' "$n" - - - - "FAILED: ${why:-no token generated}"
        kill $srv 2>/dev/null; wait $srv 2>/dev/null; trap - EXIT; continue
    fi
    tg=$(jq -r '.timings.predicted_per_second // 0' <<<"$out")
    pp=$(jq -r '.timings.prompt_per_second // 0'    <<<"$out")
    mb=$(awk -v a="$b0" -v b="$b1" 'BEGIN { printf "%.0f", (b - a) / 1048576 }')
    printf '%-7s %-9.2f %-12.1f %-11s %-11.1f %s\n' \
        "$n" "$tg" "$pp" "$mb" "$(awk -v m="$mb" -v t="$ntok" 'BEGIN { print m / t }')" ok

    kill $srv 2>/dev/null; wait $srv 2>/dev/null
    trap - EXIT
done
echo
echo "Reading it: MB/token high (>100) = IO-bound, act on RAM / VRAM / filesystem."
echo "            MB/token low  (<20)  = CPU-bound, lowering -ncmoe stops helping."
