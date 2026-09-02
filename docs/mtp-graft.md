# Grafting the MTP head onto a quantized qwen4exp GGUF

How the multi-token-prediction (MTP) draft head of Qwen3.8-Flash-Next was taken
out of a 360 GB Hugging Face checkpoint and added to an already-quantized,
already-split IQ4_XS GGUF, **without re-quantizing and without copying 97 GB**.

Everything here was run end to end. The commands assume `LLAMA_CPP_DIR` points at
a llama.cpp checkout with PR #27836 applied and built, and that its `.venv` (or
your Python environment) has `requests`, `huggingface_hub`, `safetensors`,
`torch` (CPU is enough) and `gguf`.

---

## 0. The build

PR #27836 adds the `qwen4exp` MTP path: the `--mtp` mode of
`convert_hf_to_gguf.py`, the `blk.N.nextn.hc_head_{norm,down,up}` tensors, and
`--spec-type draft-mtp` in the server.

```bash
cd "$LLAMA_CPP_DIR"
git fetch origin pull/27836/head:mtp-27836
git checkout mtp-27836        # or cherry-pick onto your own branch
cmake --build build --target llama-server llama-quantize llama-tokenize -j 6
```

Non-regression checks that cost nothing: `llama-server --help` lists `draft-mtp`,
`--version` starts, and `strings build/bin/libllama.so | grep nextn.hc_head`
finds the new tensor names.

Keep a copy of the pre-patch binaries (`llama-server`, `libllama.so.*`) if you
want to roll back without rebuilding.

---

## 1. Extract the head — 7.24 GiB instead of 180

`mtp/extract-mtp-head.py`. The head is 36 tensors scattered across 31 of the
checkpoint's 131 safetensors shards. Downloading those 31 shards whole would be
~40 GB; downloading the checkpoint would be 360 GB. The script instead reads
each shard's **safetensors header** (an 8-byte little-endian length followed by a
JSON blob with every tensor's `data_offsets`), computes absolute byte offsets,
merges tensors that happen to be adjacent into one request, and issues a single
HTTP `Range` request per run.

| group | tensors | why |
|---|---|---|
| `mtp.*` | 31 (asserted with `--expect-count 31`) | the head itself: `fc_embedding`, `fc_hidden`, `hyper_connection_mixer.*`, `layers.0.*` |
| `model.embed_tokens.weight`, `lm_head.weight` | 2 | needed only by the detached sidecar path (`-md`): the loader wants `token_embd`, and if `lm_head` is not tied it would otherwise silently reuse `token_embd` as the LM head and produce wrong logits. `--head-only` skips them; that is enough for grafting into a full trunk |
| `model.hyper_connection_mixer.*` | 3 (kilobytes) | the trunk's head mixer (`output_hc_*`); the loader asks for it even in draft-only mode and it costs nothing |

Safety properties worth knowing about, because they are what makes this usable
over a flaky link: it refuses an HTTP 200 (a server that ignored `Range` and is
about to hand you a whole shard), checks `Content-Range`, checks that
`shape * itemsize == data_offsets` for every tensor, checks free space up front,
refuses to write under `/tmp`, `/dev/shm` or `/run`, and resumes both across runs
(`.extract-mtp-head.progress.json`) and inside an interrupted run (from the last
byte written).

```bash
export MTP_OUT=/path/on/a/real/disk/mtp-head
python mtp/extract-mtp-head.py --dry-run -v   # ~60 MB of headers; prints the exact plan
python mtp/extract-mtp-head.py                # 7.24 GiB
```

The repository is gated, so a token is needed —
`huggingface_hub.get_token()` picks up whatever `huggingface-cli login` stored,
or `HF_TOKEN` from the environment.

Measured dry-run plan: `36 tensors in 31 Range request(s) over 31 of 131 shards;
7.24 GiB to fetch`.

The output directory holds one `model-mtp-head.safetensors`, a consistent
`model.safetensors.index.json`, and the small config/tokenizer files, so it can
be handed straight to the converter.

---

## 2. Convert and quantize

```bash
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" --mtp --outtype bf16 \
    --outfile "$MTP_OUT/Qwen3.8-Flash-Next-MTP-BF16.gguf" "$MTP_OUT"

"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$MTP_OUT/Qwen3.8-Flash-Next-MTP-BF16.gguf" \
    "$MTP_OUT/Qwen3.8-Flash-Next-MTP-Q8_0.gguf" Q8_0
```

`--mtp` (`mtp_only`) writes `block_count=49`, `nextn_predict_layers=1`,
`attention.compress_ratios` with 49 entries (the last one `0`, meaning dense
attention), the `blk.48.*` tensors, and `token_embd`/`output` if they are in the
safetensors. `eh_proj` comes out as `[2 * n_embd, n_embd]` — the fusion of
`fc_embedding` and `fc_hidden`, embedding first.

Measured: 34 tensors, 7400 MiB bf16 -> 3935 MiB Q8_0.

**Do not match the trunk's quantization.** The head is ~4B parameters and its
drafts are verified by the trunk, so its precision only moves the acceptance
rate, never correctness. Q8_0 costs 0.4 GB more than Q6_K and buys acceptance.

Inspect the result without loading it:

```bash
python "$LLAMA_CPP_DIR/gguf-py/gguf/scripts/gguf_dump.py" --no-tensors <gguf>
python scripts/gguf-header.py <gguf> | grep 'blk\.48'
```

---

## 3. Graft it in — the loader invariants

This is the part worth reading even if you never run any of it. `mtp/graft-mtp-head.py`
adds the head as **one extra shard at the end** of the trunk's split set, leaving
shards 2..N untouched. It works because of six facts about
`src/llama-model-loader.cpp` and `src/llama.cpp`:

1. Every model KV (`general.*`, `<arch>.*`, tokenizer) is read from **shard 1 only**.
2. Of every other shard, only `split.no == the file's index` (0-based) is checked.
   `split.count` and `split.tensors.count` in shards >= 2 are never read.
3. `split.tensors.count` in shard 1 must be **exactly** the total tensor count
   across all shards, or you get
   `corrupted model: N tensors expected but M found`.
4. A tensor name present in two shards is an error (`tensor 'x' is duplicated`).
5. File names are derived from shard 1's own name:
   `<prefix>-000ii-of-000NN.gguf`, with `NN = split.count`.
6. Per-layer arrays such as `<arch>.attention.compress_ratios` are read with
   `get_key_or_arr(..., n_layer_all)`. A length other than `block_count` is a
   hard error.

So the recipe is:

| where | key | change | why |
|---|---|---|---|
| shard 1 | `<arch>.block_count` | 48 -> 49 | this is `n_layer_all`; `n_layer()` becomes 49 − nextn = 48 |
| shard 1 | `<arch>.nextn_predict_layers` (new) | — -> 1 | without it block 48 would be treated as a trunk layer |
| shard 1 | `<arch>.attention.compress_ratios` | 48 -> 49 entries, last one `0` | invariant 6; `0` = dense attention for the MTP block |
| shard 1 | any other `<arch>.*` array of length exactly `block_count` | one more entry | invariant 6 — enumerate them from a dump first |
| shard 1 | `split.count` | N -> N+1 | invariant 5 |
| shard 1 | `split.tensors.count` | T -> T + head tensors | invariant 3 |
| new shard N+1 | `split.no`=N, `split.count`=N+1, `split.tensors.count`=T+head | | invariant 2: only `split.no` is actually checked |
| new shard N+1 | **only** `blk.48.*` from the quantized head | | invariant 4: `token_embd`, `output` and `output_hc_*` already exist in the trunk |
| shards 2..N | nothing | | their `split.no` is already right; their stale `split.count` is never read |

Leave alone: `general.*`, the tokenizer, `general.alignment` (each file carries
its own, read by `gguf_init_from_file`), and every tensor's type.

### Why shard 1 has to be copied

Adding `nextn_predict_layers` (+45 bytes) and one `compress_ratios` entry
(+4 bytes) pushes the start of the data section past the alignment padding
(at most 31 bytes), and **every tensor offset is relative to that start**. If
shard 1 contains tensor data — it does unless the set was produced with
`llama-gguf-split --no-tensor-first-split` — the whole shard has to be rewritten.
On this model that was 41.7 GiB copied once, about four minutes, and it evicts
the page cache: do it with no server running.

Measured on the IQ4_XS set: 1224 tensors in 3 shards -> 1256 tensors in 4 shards,
with the new shard 4 at 2.78 GB.

### Layout on disk

The output directory gets the rewritten shard 1, **symlinks** to the trunk's
shards 2..N (hard links do not cross filesystems; the loader uses `fopen`, so
symlinks are fine), and the new shard N+1. Nothing in the trunk's directory is
touched, so the original set stays servable without MTP in parallel.

```bash
python mtp/graft-mtp-head.py \
    --trunk  /path/to/Qwen3.8-...-IQ4_XS-00001-of-00003.gguf \
    --head   "$MTP_OUT/Qwen3.8-Flash-Next-MTP-Q8_0.gguf" \
    --out    /path/to/grafted \
    --name   Qwen3.8-Flash-Next-Uncensored-IQ4_XS-MTP \
    --dry-run          # prints the whole plan and writes nothing
```

`mtp/test_graft_mtp_head.py` builds a synthetic two-shard trunk and a synthetic
head of a few KiB, grafts them, re-reads every output with `GGUFReader`, and
compares metadata and bytes — including that exactly one key was added, that
arrays of other lengths were left alone, and that shard 2 is a symlink. It also
checks two refusals: a head whose block index does not match, and an output
directory that already exists. It needs no model and runs in a second:

```bash
LLAMA_CPP_DIR=... python mtp/test_graft_mtp_head.py
```

### Validating the result without reading 97 GB

```bash
"$LLAMA_BIN_DIR/llama-tokenize" -m <new shard 1> -p "hello"
```

`llama-tokenize` loads with `vocab_only`, so no tensor data is read at all —
seconds, and no page cache disturbed. But `llama_model_load` still runs the split
validation: `split.no`, duplicate names, `split.tensors.count`, array lengths.
It is the cheapest possible check that the graft is coherent. (Note that with
`vocab_only` the hyperparameter load returns early, so hyperparameters are *not*
validated at this step.)

Then `mtp/verify-graft.sh` does the real proof: the same greedy prompt through
the original trunk, through the grafted model with MTP off, and through the
grafted model with MTP on, comparing generated bytes.

---

## 4. Running with the head

```bash
"$LLAMA_BIN_DIR/llama-server" -m <grafted shard 1> \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    --override-tensor 'blk\.48\.ffn_.*_exps=CPU' \
    <the same production flags: -ngl, --n-cpu-moe, -ot, -c, -fa, ...>
```

`--spec-type draft-mtp` sets `load_mtp=true`, so block 48 is loaded and the MTP
context shares the target's weights — there is no second model in memory. Where
block 48 lands is decided by the same `-ot` / `--n-cpu-moe` rules as the trunk,
so pin its experts explicitly and check the load log.

What to look at:

- **acceptance**: `timings.draft_n` and `timings.draft_n_accepted` in the
  response JSON; in the server log, per slot,
  `draft acceptance = 0.xxxxx (a accepted / g generated), mean len = L`.
- **speed**: tokens/s against the plain model, on a code prompt *and* a prose
  prompt — the gap between them is large.
- **correctness**: identical greedy output is the theoretical expectation. See
  the honesty section of the README for what actually happened.

---

## 5. A known gap: loading the head as a sidecar (`-md`)

The head can also be kept as a separate small GGUF and passed with `-md`, which
avoids rewriting a 41 GiB shard entirely. On the PR as it stands that path does
not work, for two reasons:

1. `load_arch_tensors` in `src/models/qwen4exp.cpp` marks the trunk's
   `blk.0..47` and the model-level `hc_head_*` as required (flags 0) even when
   the file being loaded is only the head.
2. The converter in `--mtp` mode drops `model.hyper_connection_mixer.*`: the
   keep-list in `_QwenMtpMixin.filter_tensors` retains only
   `embed_tokens`/`norm`/`lm_head` in `mtp_only`.

The shape of a fix, following how `qwen3next.cpp` and deepseek4 handle the same
situation:

- Detect head-only files —
  `const bool mtp_only = n_layer_nextn > 0 && ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr;`
  — and use `TENSOR_NOT_REQUIRED` for every tensor with `il < n_layer`, for the
  model-level `hc_head_*`, and for `per_layer_tok_embd` (PLE is a trunk-only
  structure; skip the `ple_n_heads > 0` block entirely in `mtp_only`).
- Keep `tok_embd` required — the head has no `nextn.embed_tokens`. Make `output`
  optional with a fallback to `tok_embd` **only if** the checkpoint is tied;
  otherwise keep it required, because the alternative is silently wrong logits.
- `graph_mtp` needs no change: it already uses the head's own
  `layer.nextn.hc_head_*`.
- In the converter, keep `model.hyper_connection_mixer.*` (which becomes
  `output_hc_*`) in `mtp_only`, and leave `embed_tokens`/`lm_head` in the
  sidecar.

The demonstration would be: `-m <plain IQ4_XS, no block 48> -md <sidecar Q8_0>
--spec-type draft-mtp` producing the same greedy output as the grafted model at
comparable acceptance, with no missing tensors in the loader log.

Before writing that patch, check whether it has landed upstream already — at the
time of writing there were at least two open attempts at the same fix.

---

## Open risks

- The head attends densely (no QSA). That is a numerical superset and the drafts
  are verified anyway, but acceptance may fall at long context.
- `GGML_ASSERT(ubatch.token)`: no drafting on batches carrying embeddings
  (images).
- The recurrent state is checkpointed and restored on every rejection. The PR
  reports roughly 600 ms per round on discrete GPUs; there is an upstream
  follow-up for it.
- GGUFs produced by other MTP experiments are not interchangeable with #27836:
  the `nextn.hc_head_*` tensor names differ, or the architecture string does.
