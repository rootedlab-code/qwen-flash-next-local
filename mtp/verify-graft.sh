#!/usr/bin/env bash
# Validate a grafted model and check what the MTP head does to the output.
#
#   TRUNK=<original shard 1> GRAFTED=<grafted shard 1> ./verify-graft.sh
#
# Two steps:
#   1. header validation with llama-tokenize. It loads with vocab_only, so it
#      reads no tensor data at all (seconds, zero page cache), but llama.cpp
#      still runs the split validation: split.no, duplicate tensor names,
#      split.tensors.count, per-layer array lengths.
#   2. three greedy runs at temperature 0 with a fixed seed, and a byte-for-byte
#      comparison of the generated text:
#        original   = the untouched trunk                      -> reference
#        plain      = grafted model, MTP not enabled           -> must be IDENTICAL:
#                     proves the graft did not disturb the trunk
#        mtp-n3     = grafted model + draft-mtp, n_max 3       -> see below
#
# What to expect for mtp-n3: in theory greedy speculative decoding is lossless
# and the text should be identical. On this setup it is NOT — the same prompt
# gives a different sha256, reproducibly, on two different machines. Verifying
# k+1 tokens in one batch is not the same sequence of floating-point reductions
# as decoding them one at a time, and MoE routing amplifies a tie-break. That is
# an explanation, not a proof: proving it needs a logit-level comparison, which
# this script does not do. Treat a divergence as "expected but unproven".
set -uo pipefail
export LC_ALL=C

MTP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$MTP_DIR/../scripts/config.sh"

: "${TRUNK:?set TRUNK to shard 1 of the original (un-grafted) GGUF}"
: "${GRAFTED:?set GRAFTED to shard 1 of the grafted GGUF}"
# Where the head's experts go. Without this they follow --n-cpu-moe like the
# trunk's; pinning them to the CPU keeps VRAM free for the KV cache.
#
# Appended to $OT as ONE comma-separated value, never as a second
# --override-tensor: llama.cpp keeps only the last occurrence of the flag and
# says so with nothing louder than a deprecation warning. The original runs of
# this experiment hit exactly that, and silently lost the PLE override.
if [ -z "${OT_HEAD+x}" ]; then
    OT_HEAD='blk\.48\.ffn_.*_exps=CPU'
fi
OT="$OT,$OT_HEAD"
export OT

require_exec "$LLAMA_TOKENIZE"
require_file "$TRUNK"
require_file "$GRAFTED"

echo "=== header validation (vocab_only, no tensor data read) ==="
if "$LLAMA_TOKENIZE" -m "$GRAFTED" -p "hello" >"$LOG_DIR/F-tokenize.log" 2>&1; then
    echo "  OK: architecture, split completeness and consistency, vocabulary"
    echo "  (note: with vocab_only load_hparams returns early, so hyperparameters are NOT checked here)"
else
    echo "  FAILED — the graft produced an inconsistent header:"
    tail -20 "$LOG_DIR/F-tokenize.log"; exit 1
fi

echo; echo "=== greedy runs, temperature 0, seed 42 ==="
"$MTP_DIR/measure-mtp.sh" original "$TRUNK"   || echo "  [configuration failed]"
echo
"$MTP_DIR/measure-mtp.sh" plain "$GRAFTED" || echo "  [configuration failed]"
echo
"$MTP_DIR/measure-mtp.sh" mtp-n3 "$GRAFTED" \
    --spec-type draft-mtp --spec-draft-n-max 3 || echo "  [configuration failed]"

echo; echo "=== where block 48 landed ==="
grep -aiE 'blk\.48\.|CUDA0 model buffer|CPU model buffer' "$LOG_DIR/G-mtp-n3-server.log" 2>/dev/null | head -8 \
    || echo "(log missing)"

echo; echo "=== byte-for-byte comparison ==="
for p in code prose; do
    echo "  --- prompt $p ---"
    ref="$LOG_DIR/G-original-$p.txt"
    for e in original plain mtp-n3; do
        f="$LOG_DIR/G-$e-$p.txt"
        [ -f "$f" ] && printf '    %-10s %s  (%s bytes)\n' "$e" "$(sha256sum "$f" | cut -c1-16)" "$(stat -c%s "$f")"
    done
    for e in plain mtp-n3; do
        f="$LOG_DIR/G-$e-$p.txt"
        [ -f "$ref" ] && [ -f "$f" ] && {
            if cmp -s "$ref" "$f"; then echo "    $e == original  IDENTICAL"
            else echo "    $e != original  DIVERGES"; fi; }
    done
done
