#!/usr/bin/env bash
# The defect: one and the same answer contains several *different* numeric
# arrays, as if the model forgot which example it had picked. A one-sample-per-
# config test came back clean everywhere, which means the defect is INTERMITTENT:
# without a base rate you cannot tell a better sampler from noise. This measures
# the RATE over repetitions, with a single model load — the configurations are
# per-request parameters, not restarts.
#
# The prompt and the metric are Italian on purpose: they are what was actually
# measured. coherence.py parses Italian phrasings ("array[i] è ..."), so port
# both together if you change language.
#
#   REPS=4 ./bench-coherence.sh
set -uo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"

PORT="${PORT:-8093}"; export PORT
REPS="${REPS:-4}"
STAMP="$(date +%Y%m%d-%H%M)"
OUT="${OUT:-$WORK_DIR/coherence-$STAMP.json}"
LOG="$LOG_DIR/coherence-$STAMP.log"
PROMPT="${PROMPT:-Spiega passo per passo come si esegue una ricerca binaria su un array ordinato di 8 elementi, mostrando ogni iterazione con numeri concreti.}"

require_exec "$LLAMA_SERVER"
require_file "$MODEL"
command -v python3 >/dev/null || { log_error "python3 not found"; exit 1; }

# name | temp | top_p | top_k | min_p | dry_mult | dry_len
readonly -a CONFIG=(
  "dry-weak|0.3|0.95|20|0.05|0.4|6"      # what serve.sh ships: 0/4 incoherent, 14.1% repetition
  "dry-aggressive|0.3|0.95|20|0.05|0.8|2" # the obvious setting, and the worst: 2/4, 17.3%
  "dry-off|0.3|0.95|20|0.05|0|2"          # 0/4, 15.2%
  "temp-0.15|0.15|0.95|20|0.05|0.8|2"     # 2/4, 29.4% -- looked best at n=1, worst at n=4
  "top_k-40|0.3|0.95|40|0.05|0.8|2"       # 1/4, 13.1%
  "temp-0.6|0.6|0.95|20|0.05|0.8|2"       # 2/2 before the run was cut short
)

mkdir -p "$(dirname -- "$OUT")"
"$SCRIPT_DIR/serve.sh" >"$LOG" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null' EXIT
wait_for_server "$srv" 300 || { echo "server died, see $LOG"; exit 1; }

echo "server ready  |  $REPS repetitions x ${#CONFIG[@]} configs = $((REPS * ${#CONFIG[@]})) answers"
echo

echo "[" > "$OUT"; first=1
printf '%-12s %-6s %-9s %-10s %-9s %-8s %s\n' config rep distinct claims wrong tok/s outcome
for row in "${CONFIG[@]}"; do
    IFS='|' read -r name temp topp topk minp dmul dlen <<<"$row"
    incoherent=0
    for r in $(seq "$REPS"); do
        out=$(curl -sf --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "$(jq -nc --arg p "$PROMPT" \
              --argjson t "$temp" --argjson tp "$topp" --argjson tk "$topk" \
              --argjson mp "$minp" --argjson dm "$dmul" --argjson dl "$dlen" \
              '{messages: [{role: "user", content: $p}], max_tokens: 3000, temperature: $t,
                top_p: $tp, top_k: $tk, min_p: $mp, dry_multiplier: $dm, dry_base: 1.75,
                dry_allowed_length: $dl, dry_penalty_last_n: 4096, repeat_penalty: 1.05}')")
        txt=$(jq -r '.choices[0].message.content // ""' <<<"$out")
        tps=$(jq -r '.timings.predicted_per_second // 0' <<<"$out")
        n=$(jq -r '.timings.predicted_n // 0' <<<"$out")
        if [ "${n:-0}" -eq 0 ]; then
            printf '%-12s %-6s %s\n' "$name" "$r" "REQUEST FAILED"
            continue
        fi
        met=$(printf '%s' "$txt" | python3 "$SCRIPT_DIR/coherence.py")
        nd=$(jq -r '.array_distinti' <<<"$met")
        na=$(jq -r '.asserzioni'     <<<"$met")
        ne=$(jq -r '.errate'         <<<"$met")
        # trigram repetition rate: the other way this defect shows up
        rep=$(printf '%s' "$txt" | tr -cs '[:alnum:]' '\n' | awk '
            { w[NR] = $0 }
            END { for (i = 1; i <= NR - 2; i++) { k = w[i] " " w[i+1] " " w[i+2]; c[k]++; n++ }
                  d = 0; for (k in c) if (c[k] > 1) d += c[k] - 1
                  printf "%.1f", (n ? d / n * 100 : 0) }')
        good=0
        [ "$nd" -eq 1 ] && [ "$ne" -eq 0 ] && good=1
        (( good )) || incoherent=$((incoherent + 1))
        printf '%-12s %-6s %-9s %-10s %-9s %-8.2f %s\n' "$name" "$r" "$nd" "$na" "$ne" "$tps" \
            "$( (( good )) && echo ok || echo INCOHERENT )"
        (( first )) || echo "," >> "$OUT"; first=0
        jq -n --arg c "$name" --argjson rp "$r" --argjson m "$met" --arg t "$txt" \
              --argjson tps "$tps" --argjson rep "$rep" \
              '{config: $c, rep: $rp, metric: $m, tok_s: $tps, repetition_pct: $rep, answer: $t}' >> "$OUT"
    done
    printf '%-12s -> incoherent %d/%d\n\n' "$name" "$incoherent" "$REPS"
done
echo "]" >> "$OUT"

echo "=== INCOHERENCE RATE ==="
jq -r 'group_by(.config)[]
       | "\(.[0].config)|\(length)|\([.[] | select(.metric.array_distinti != 1 or .metric.errate > 0)] | length)|\([.[] | .tok_s] | add / length)|\([.[] | .repetition_pct] | add / length)"' "$OUT" \
 | awk -F'|' 'BEGIN { printf "%-12s %-10s %-12s %-9s %s\n", "config", "samples", "incoherent", "tok/s", "repeats" }
              { printf "%-12s %-10s %-12s %-9.2f %.1f%%\n", $1, $2, $3 "/" $2, $4, $5 }'
echo
echo "saved to $OUT"
