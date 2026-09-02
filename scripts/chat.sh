#!/usr/bin/env bash
# Interactive chat with the same tuned configuration as serve.sh, but through
# llama-cli instead of the server. Run it from a real terminal.
set -euo pipefail

. "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

require_exec "$LLAMA_CLI"
require_file "$MODEL"

if pgrep -x llama-server >/dev/null; then
    echo "[WARNING] a llama-server is running and contends for the page cache." >&2
    echo "          stop it with: pkill -x llama-server" >&2
fi

cat <<'NOTE'
--------------------------------------------------------------------
The first replies are slow: the page cache is empty and every expert
comes from the SSD. Throughput climbs over the next few exchanges as
the hot experts settle in RAM. Long reasoning warms itself up: the
answer is faster than the first tokens.

Ctrl-C interrupts generation, Ctrl-D exits.
--------------------------------------------------------------------
NOTE

exec "$LLAMA_CLI" \
    --model "$MODEL" \
    --n-gpu-layers 99 \
    --n-cpu-moe "$NCMOE" \
    --override-tensor "$OT" \
    --load-mode mmap \
    --ctx-size "$CTX" \
    --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads "$THREADS" --threads-batch "$THREADS_BATCH" \
    --batch-size "$BATCH" --ubatch-size "$UBATCH" \
    --spec-type "$SPEC" \
    --reasoning-budget "${THINK_CLI:-1024}" \
    --no-warmup \
    --temp "$TEMP" --top-p 0.95 --top-k 20 --min-p 0.05 \
    "$@"
