#!/usr/bin/env bash
# Shared configuration and helpers. Sourced by every script in this directory.
#
# Nothing here is hardcoded to one machine: every value below can be overridden
# from the environment, or once and for all in a `.env` file at the repo root
# (copy `env.example` to `.env`). No path points inside /tmp, /dev/shm or /run:
# those are tmpfs on most desktop distributions, and every byte written there is
# a byte taken away from the page cache that is holding the experts.

REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

# shellcheck disable=SC1091
[ -f "$REPO_ROOT/.env" ] && . "$REPO_ROOT/.env"

# Where llama.cpp is checked out and built. The build must carry PR #27836
# (qwen4exp MTP) if you want --spec-type draft-mtp; see docs/mtp-graft.md.
: "${LLAMA_CPP_DIR:=$HOME/llama.cpp}"
: "${LLAMA_BIN_DIR:=$LLAMA_CPP_DIR/build/bin}"
: "${LLAMA_SERVER:=$LLAMA_BIN_DIR/llama-server}"
: "${LLAMA_CLI:=$LLAMA_BIN_DIR/llama-cli}"
: "${LLAMA_TOKENIZE:=$LLAMA_BIN_DIR/llama-tokenize}"
: "${LLAMA_QUANTIZE:=$LLAMA_BIN_DIR/llama-quantize}"

# First shard of the GGUF set. Put the model on the fastest *unencrypted*
# filesystem you have: see docs/measurements.md, it is worth a factor of two.
: "${MODEL_DIR:=$HOME/models}"
: "${MODEL:=$MODEL_DIR/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-MTP-00001-of-00004.gguf}"

# Scratch space for logs and benchmark output. Never under /tmp.
: "${WORK_DIR:=$REPO_ROOT/work}"
: "${LOG_DIR:=$WORK_DIR/log}"

: "${PORT:=8081}"

# --- tuned defaults, all measured; see README.md -----------------------------
# MoE layers whose experts stay on the CPU. 48 = all of them. Lowering it moves
# a whole expert layer into VRAM (~862 MB each on this quant); 43 does not fit
# in 8 GB. Re-run bench-ncpumoe.sh on your own card.
: "${NCMOE:=46}"
: "${CTX:=32768}"
# 1.0 is the Qwen recommendation for the full-precision model. At 4 bits and
# below the distribution is flat enough that 1.0 samples quantization noise.
: "${TEMP:=0.3}"
# Measured on IQ4_XS after the reclaim sysctls: 8 -> 8.16, 12 -> 9.64, 16 -> 5.20.
: "${THREADS:=12}"
: "${THREADS_BATCH:=16}"
# A smaller micro-batch means a smaller CUDA compute buffer, i.e. VRAM headroom.
# 512 works but leaves ~299 MiB free and OOMs when CUDA graphs get reallocated.
: "${UBATCH:=256}"
: "${BATCH:=2048}"
# Speculative decoding. "none" is the measured optimum for plain generation;
# "draft-mtp" needs a grafted model and PR #27836. See README.md.
: "${SPEC:=none}"
: "${THINK:=256}"

# Tensor placement overrides, as ONE comma-separated value. This must stay a
# single --override-tensor argument: passing the flag twice makes llama.cpp keep
# only the last one, with nothing but a deprecation warning in the log to say so.
# Assigned the long way because `: "${OT:=...}"` would eat the backslash.
if [ -z "${OT+x}" ]; then
    OT='per_layer_token_embd\.weight=CPU'
fi

log_error() { printf '[ERROR] %s\n' "$*" >&2; }

require_file() { [ -f "$1" ] || { log_error "file not found: $1"; exit 1; }; }
require_exec() { [ -x "$1" ] || { log_error "binary not found or not executable: $1"; exit 1; }; }

# The block device holding a path, as it is named in /proc/diskstats. Never
# hardcode an NVMe name: the kernel can swap nvme0/nvme1 between boots, and the
# model may move.
#
# For a plain partition this is the parent disk (nvme0n1), whose counters
# aggregate the partition's. For a device-mapper volume (LUKS) there is no
# parent, so we fall back to its own kernel name (dm-0) — which counts the
# decrypted reads the filesystem asked for, not the physical ones.
disk_of() {
    local src dev
    src=$(findmnt -no SOURCE -T "$1" 2>/dev/null | sed 's/\[.*//') || return 1
    [ -n "$src" ] || return 1
    dev=$(lsblk -no PKNAME "$src" 2>/dev/null | head -1)
    [ -n "$dev" ] || dev=$(lsblk -no KNAME "$src" 2>/dev/null | head -1)
    printf '%s\n' "$dev"
}

# Bytes read from a block device since boot, from /proc/diskstats (field 6 is
# sectors read, 512 bytes each). Used to get real per-token SSD traffic.
bytes_read() { awk -v d="$1" '$3 == d { print $6 * 512 }' /proc/diskstats; }

# Poll /health until the server answers or the process dies. $1 = pid, $2 = seconds.
wait_for_server() {
    local pid="$1" limit="${2:-600}" i
    for ((i = 0; i < limit; i++)); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

mkdir -p "$LOG_DIR"
