#!/usr/bin/env bash
# How much of the throughput comes from speculative decoding? Compares one
# --spec-type against another on the same prompt in the same warm state.
#
# The measured answer on this machine was "less than zero" for ngram-mod:
# 25.8% draft acceptance is not enough for the accepted tokens to pay for the
# rejected ones, because every rejected token still costs expert reads from SSD.
#
#   VALUES="ngram-mod none" ./bench-spec.sh
set -uo pipefail
export LC_ALL=C

. "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

PORT="${PORT:-8095}"; export PORT
PROMPT="${PROMPT:-Explain in detail how the TCP three-way handshake works.}"
MAX_TOKENS="${MAX_TOKENS:-400}"

require_exec "$LLAMA_SERVER"
require_file "$MODEL"

ask() {
    curl -sf --max-time 400 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$PROMPT" --argjson n "$MAX_TOKENS" --argjson t "$TEMP" \
              '{messages: [{role: "user", content: $p}], max_tokens: $n, temperature: $t}')"
}

printf '%-14s %-9s %-12s %s\n' 'spec-type' 'tok/s' 'acceptance' 'note'
for sp in ${VALUES:-ngram-mod none}; do
    log="$LOG_DIR/spec-$sp.log"
    SPEC="$sp" "$(dirname -- "${BASH_SOURCE[0]}")/serve.sh" >"$log" 2>&1 &
    srv=$!
    trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null' EXIT

    if ! wait_for_server "$srv" 300; then
        printf '%-14s %-9s %-12s %s\n' "$sp" - - "FAILED (see $log)"
        kill $srv 2>/dev/null; wait $srv 2>/dev/null; trap - EXIT; continue
    fi

    ask >/dev/null 2>&1          # warm-up, discarded
    out=$(ask)
    tps=$(jq -r '.timings.predicted_per_second // 0' <<<"$out")
    acc=$(grep -oE 'draft acceptance = [0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')
    printf '%-14s %-9.2f %-12s %s\n' "$sp" "$tps" "${acc:-—}" \
        "$([ "$sp" = none ] && echo 'plain decode' || echo 'with speculation')"

    kill $srv 2>/dev/null; wait $srv 2>/dev/null
    trap - EXIT
done
