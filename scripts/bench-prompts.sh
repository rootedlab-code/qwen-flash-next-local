#!/usr/bin/env bash
# Honest throughput: eight prompts on eight *different* topics, so no request
# reuses the experts the previous one warmed. Measuring the same prompt twice on
# this setup inflates the number by 2-3x. VRAM is sampled after every reply
# because the failure mode that matters here is a CUDA OOM on request N, not on
# load: CUDA graphs get reallocated and the headroom disappears.
set -uo pipefail
export LC_ALL=C

. "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

PORT="${PORT:-8099}"; export PORT
N_PREDICT="${N_PREDICT:-64}"

require_exec "$LLAMA_SERVER"
require_file "$MODEL"

readonly -a PROMPT=(
"Explain how the TCP three-way handshake works."
"Write a short poem about the sea in winter."
"What is the difference between mitosis and meiosis in eukaryotic cells?"
"Implement a Python function that reverses a linked list."
"Summarize the economic causes of the 1929 crash."
"How do you compute the determinant of a 3x3 matrix? Show the steps."
"Describe the life cycle of a monarch butterfly."
"What are the main security risks of a public REST API?"
)

vram_used()  { nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits 2>/dev/null || echo 0; }
vram_total() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0; }

TOTAL=$(vram_total)
log="$LOG_DIR/bench-prompts.log"

echo "config: ncmoe=$NCMOE ubatch=$UBATCH ctx=$CTX threads=$THREADS"
"$(dirname -- "${BASH_SOURCE[0]}")/serve.sh" >"$log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null' EXIT

wait_for_server "$srv" 300 || { echo "server died during load, see $log"; exit 1; }

loaded=$(vram_used)
echo "VRAM after load: ${loaded} MiB / ${TOTAL} (headroom $((TOTAL - loaded)))"
echo
printf '%-3s %-8s %-10s %s\n' '#' 'tok/s' 'VRAM_MiB' 'topic'

peak=0
for i in "${!PROMPT[@]}"; do
    out=$(curl -sf --max-time 600 "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "${PROMPT[$i]}" --argjson n "$N_PREDICT" \
              '{prompt: $p, n_predict: $n, cache_prompt: false}')")
    v=$(vram_used)
    (( v > peak )) && peak=$v
    nt=$(jq -r '.timings.predicted_n // 0' <<<"$out" 2>/dev/null)
    if [ -z "$out" ] || [ "${nt:-0}" -eq 0 ]; then
        printf '%-3s %-8s %-10s %s\n' "$((i + 1))" CRASH "$v" "${PROMPT[$i]:0:42}"
        grep -oiE 'out of memory|CUDA error' "$log" | head -1
        break
    fi
    printf '%-3s %-8.2f %-10s %s\n' "$((i + 1))" \
        "$(jq -r '.timings.predicted_per_second' <<<"$out")" "$v" "${PROMPT[$i]:0:42}"
done
echo
echo "peak VRAM: ${peak} MiB -> headroom $((TOTAL - peak)) MiB"
