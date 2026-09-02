#!/usr/bin/env bash
# Measure one llama-server configuration in greedy mode on two fixed prompts
# (code, prose) and record everything on disk.
#
#   measure-mtp.sh <label> <shard 1 of the model> [extra llama-server flags...]
#
# Writes into $LOG_DIR: G-<label>-server.log and G-<label>-<prompt>.json/.txt,
# and prints tok/s, draft_n, draft_n_accepted and the sha256 of the generated
# text. The hash is the point: with temperature 0 and a fixed seed, two
# configurations that decode the same distribution must produce the same bytes.
set -euo pipefail
export LC_ALL=C

MTP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$MTP_DIR/../scripts/config.sh"

[ $# -ge 2 ] || { echo "usage: measure-mtp.sh <label> <model shard 1> [flags...]" >&2; exit 2; }
LABEL="$1"; TARGET="$2"; shift 2

PORT="${MTP_PORT:-8095}"
N_PREDICT="${N_PREDICT:-300}"
WAIT_S="${WAIT_S:-1800}"
SEED="${SEED:-42}"
readonly PROMPTS=(code prose)

require_exec "$LLAMA_SERVER"
require_file "$TARGET"
for p in "${PROMPTS[@]}"; do require_file "$MTP_DIR/prompt-$p.txt"; done
if pgrep -x llama-server >/dev/null; then
    log_error "another llama-server is running (PID $(pgrep -d, -x llama-server))"; exit 3
fi

LOG="$LOG_DIR/G-$LABEL-server.log"
"$LLAMA_SERVER" --model "$TARGET" \
    --n-gpu-layers 99 --n-cpu-moe "$NCMOE" --override-tensor "$OT" \
    --load-mode mmap --ctx-size "$CTX" --parallel 1 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 --threads "$THREADS" --threads-batch "$THREADS_BATCH" \
    --batch-size "$BATCH" --ubatch-size "$UBATCH" --no-warmup --jinja \
    --host 127.0.0.1 --port "$PORT" "$@" >"$LOG" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null' EXIT

for ((i = 0; i < WAIT_S; i++)); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $srv 2>/dev/null || { log_error "server died, see $LOG"; exit 1; }
    sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { log_error "server not ready after ${WAIT_S}s"; exit 1; }
echo "[$LABEL] server ready after ${i}s (PID $srv)"

request() {  # $1 = prompt file, $2 = n_predict
    jq -nc --rawfile p "$1" --argjson n "$2" --argjson s "$SEED" \
        '{prompt: $p, n_predict: $n, temperature: 0, seed: $s, cache_prompt: false}' \
    | curl -sf --max-time 1200 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d @-
}

request "$MTP_DIR/prompt-code.txt" 32 >/dev/null   # warm-up: first experts into cache
printf '%-8s %-7s %-7s %-8s %-8s %-8s %s\n' prompt tok gen_t/s pp_t/s draft_n accepted sha256
for p in "${PROMPTS[@]}"; do
    out="$LOG_DIR/G-$LABEL-$p.json"
    request "$MTP_DIR/prompt-$p.txt" "$N_PREDICT" >"$out"
    jq -r '.content' "$out" >"$LOG_DIR/G-$LABEL-$p.txt"
    printf '%-8s %-7s %-7.2f %-8.1f %-8s %-8s %s\n' "$p" \
        "$(jq -r '.timings.predicted_n' "$out")" \
        "$(jq -r '.timings.predicted_per_second' "$out")" \
        "$(jq -r '.timings.prompt_per_second' "$out")" \
        "$(jq -r '.timings.draft_n // "-"' "$out")" \
        "$(jq -r '.timings.draft_n_accepted // "-"' "$out")" \
        "$(sha256sum "$LOG_DIR/G-$LABEL-$p.txt" | cut -c1-16)"
done

kill $srv; wait $srv 2>/dev/null || true
trap - EXIT
echo "--- 'draft acceptance' lines from the server log:"
grep -a "draft acceptance" "$LOG" || echo "(none)"
