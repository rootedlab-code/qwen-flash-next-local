#!/usr/bin/env bash
# Start llama-server on a 125B-A6B MoE that does not fit in RAM.
#
# Split: non-expert weights on the GPU, as many expert layers as fit in the page
# cache, and the 28.8 GB PLE n-gram table left on the SSD through mmap. Every
# knob comes from config.sh and can be overridden from the environment.
set -euo pipefail

. "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

# Injected when --reasoning-budget truncates the thinking block: without it
# llama.cpp cuts the thought off and the model never transitions to an answer.
: "${THINK_MSG:=\n\nI have reasoned enough. I will now write the final answer, complete and structured step by step, with concrete numeric examples where they help.}"

require_exec "$LLAMA_SERVER"
require_file "$MODEL"

# Another server contends for page cache *and* VRAM. VRAM is the fatal one: the
# load runs all the way to the end and then dies with a CUDA OOM after ~40 s,
# so stopping now is more useful than warning. FORCE=1 to proceed anyway.
if pgrep -x llama-server >/dev/null; then
    others=$(pgrep -d, -x llama-server)
    log_error "another llama-server is running (PID $others)."
    log_error "  it holds page cache and VRAM: this load would fail with"
    log_error "  'cudaMalloc failed: out of memory'."
    log_error "  stop it with:  kill $others"
    log_error "  or re-run with FORCE=1 to proceed anyway."
    [ "${FORCE:-0}" = 1 ] || exit 3
fi

# --override-tensor keeps the PLE table out of the GPU *and* out of the resident
# set: with mmap the kernel faults in the ~8 rows of ~90 bytes a token actually
# needs and evicts them again. --load-mode mmap is not optional here; llama.cpp
# suggests --load-mode none in the log when it sees -ot ...=CPU, and following
# that suggestion tries to pull the whole file into RAM.
exec "$LLAMA_SERVER" \
    --model "$MODEL" \
    --n-gpu-layers 99 \
    --n-cpu-moe "$NCMOE" \
    --override-tensor "$OT" \
    --load-mode mmap \
    --ctx-size "$CTX" \
    --parallel 1 \
    --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads "$THREADS" --threads-batch "$THREADS_BATCH" \
    --batch-size "$BATCH" --ubatch-size "$UBATCH" \
    --spec-type "$SPEC" \
    --reasoning-budget "$THINK" \
    --reasoning-budget-message "$THINK_MSG" \
    --no-warmup \
    --temp "$TEMP" --top-p 0.95 --top-k 20 --min-p 0.05 \
    --dry-multiplier 0.4 --dry-base 1.75 --dry-allowed-length 6 --dry-penalty-last-n 4096 \
    --repeat-penalty 1.05 --repeat-last-n 512 \
    --host 127.0.0.1 --port "$PORT" \
    --jinja \
    "$@"
