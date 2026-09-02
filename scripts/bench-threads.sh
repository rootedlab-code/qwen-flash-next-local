#!/usr/bin/env bash
# With --n-cpu-moe fixed at its optimum, check whether more threads deepen the
# NVMe queue or just fight each other. Run it *after* applying the reclaim
# sysctls (sysctl/95-large-model.conf): without them the extra threads stall in
# synchronous reclaim and 12 is indistinguishable from 8.
#
#   VALUES="8 12 16" ./bench-threads.sh
set -uo pipefail
export LC_ALL=C

. "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

PORT="${PORT:-8099}"; export PORT
DEV="${DEV:-$(disk_of "$MODEL")}"
PROMPT="${PROMPT:-Explain in detail how an AVL tree works and why rebalancing is O(log n).}"

require_exec "$LLAMA_SERVER"
require_file "$MODEL"
[ -n "$DEV" ] || { log_error "could not determine the block device holding $MODEL; set DEV="; exit 1; }

ask() {
    curl -sf --max-time 900 "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$PROMPT" --argjson n "$1" \
              '{prompt: $p, n_predict: $n, cache_prompt: false}')"
}

printf '%-9s %-9s %-12s %-11s %s\n' threads tok/s prefill_t/s MB/token outcome
for th in ${VALUES:-8 12 16}; do
    log="$LOG_DIR/threads-$th.log"
    THREADS="$th" "$(dirname -- "${BASH_SOURCE[0]}")/serve.sh" >"$log" 2>&1 &
    srv=$!
    trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null' EXIT

    if ! wait_for_server "$srv" 240; then
        printf '%-9s %-9s %-12s %-11s %s\n' "$th" - - - "FAILED (see $log)"
        kill $srv 2>/dev/null; wait $srv 2>/dev/null; trap - EXIT; continue
    fi

    ask 128 >/dev/null 2>&1
    b0=$(bytes_read "$DEV"); out=$(ask 96); b1=$(bytes_read "$DEV")
    nt=$(jq -r '.timings.predicted_n // 0' <<<"$out")
    if [ -z "$out" ] || [ "${nt:-0}" -eq 0 ]; then
        why=$(grep -oiE 'out of memory|CUDA error|ggml_abort' "$log" | head -1)
        printf '%-9s %-9s %-12s %-11s %s\n' "$th" - - - "FAILED: ${why:-no token generated}"
        kill $srv 2>/dev/null; wait $srv 2>/dev/null; trap - EXIT; continue
    fi
    tg=$(jq -r '.timings.predicted_per_second // 0' <<<"$out")
    pp=$(jq -r '.timings.prompt_per_second // 0'    <<<"$out")
    printf '%-9s %-9.2f %-12.1f %-11.1f %s\n' "$th" "$tg" "$pp" \
        "$(awk -v a="$b0" -v b="$b1" -v t="$nt" 'BEGIN { print (b - a) / 1048576 / t }')" ok

    kill $srv 2>/dev/null; wait $srv 2>/dev/null
    trap - EXIT
done
