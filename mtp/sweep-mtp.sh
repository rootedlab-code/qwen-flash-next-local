#!/usr/bin/env bash
# Is --spec-draft-n-max the binding constraint, or is the per-step cost the
# problem?
#
# At n_max=3 the mean accepted draft length was 3.88 out of a maximum of 4: the
# head is saturating the limit, not brushing against it. If the limit is what
# binds, raising it must pay. If instead the cost per step grows with the number
# of tokens to verify — more distinct experts read from SSD — raising it makes
# things worse. This sweep separates the two.
#
#   GRAFTED=<grafted shard 1> ./sweep-mtp.sh
set -uo pipefail
export LC_ALL=C

MTP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$MTP_DIR/../scripts/config.sh"

: "${GRAFTED:?set GRAFTED to shard 1 of the grafted GGUF}"
# One comma-separated --override-tensor; see verify-graft.sh for why.
if [ -z "${OT_HEAD+x}" ]; then
    OT_HEAD='blk\.48\.ffn_.*_exps=CPU'
fi
OT="$OT,$OT_HEAD"
export OT

require_file "$GRAFTED"

echo "reference, this machine, no MTP:  code 8.91 t/s   prose 9.53 t/s"
echo "already measured at n_max 3:      code 6.99 t/s   prose 5.70 t/s   (acceptance 96.1% / 47.4%)"
echo

try() {   # $1 label, rest = flags
    local label="$1"; shift
    echo "--- $label ---"
    "$MTP_DIR/measure-mtp.sh" "$label" "$GRAFTED" "$@" || echo "  [failed]"
    echo
}

# Is the limit what binds? Raise it.
try mtp-n6  --spec-type draft-mtp --spec-draft-n-max 6
try mtp-n10 --spec-type draft-mtp --spec-draft-n-max 10

# On prose acceptance collapses to 47%: every rejected draft costs expert reads
# for nothing. p-min stops the draft as soon as the head is unsure.
try mtp-n6-p60 --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.6

# --spec-type takes a list: MTP head and n-grams together.
try mtp-ngram --spec-type draft-mtp,ngram-mod --spec-draft-n-max 6

echo "=== SUMMARY ==="
printf '%-14s %-9s %-9s %-9s %s\n' config prompt tok/s draft_n accepted
for e in mtp-n6 mtp-n10 mtp-n6-p60 mtp-ngram; do
    for p in code prose; do
        f="$LOG_DIR/G-$e-$p.json"
        [ -f "$f" ] && printf '%-14s %-9s %-9.2f %-9s %s\n' "$e" "$p" \
            "$(jq -r '.timings.predicted_per_second' "$f")" \
            "$(jq -r '.timings.draft_n // "-"' "$f")" \
            "$(jq -r '.timings.draft_n_accepted // "-"' "$f")"
    done
done
