# A 125B-A6B MoE at ~9 tokens/s on 30 GiB of RAM and 8 GB of VRAM

Qwen3.8-Flash-Next is a 125B-parameter mixture-of-experts model with 6B active
parameters, 180B parameters on disk once you count its n-gram table and its
multi-token-prediction head. The quantization used here (IQ4_XS) is **97 GB in
three GGUF shards**. The published guidance asks for at least 75 GB of RAM.

It runs on a laptop with **30 GiB of RAM and 8 GB of VRAM**, at **8.9 to 9.6
tokens/s** in steady state, because 28.8 GB of that file — 30% of it — never
enters RAM at all and is read from the SSD roughly eight times per token, 90
bytes at a time.

This repository is the configuration, the tooling, and the measurements. It also
records grafting the model's MTP draft head onto the quantized GGUF, and the
result of that: **96.1% draft acceptance on code, and still slower than not
using it.** The explanation is the interesting part, and it is the part that may
come out differently on your machine.

---

## Read the numbers with this attached

Every figure in this README was measured on **one machine**, and several of them
are properties of that machine rather than of the model:

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS — 8 cores / 16 threads, AVX-512, VNNI, BF16 |
| RAM | 30 GiB usable, DDR5-5600 dual channel (~89.6 GB/s theoretical) |
| GPU | NVIDIA RTX 4060 Laptop, 8188 MiB VRAM — the display runs on the iGPU, so nearly all of it is free |
| Storage A | WD SN530 NVMe, PCIe Gen3, **LUKS + btrfs** |
| Storage B | Micron NVMe, PCIe Gen4, **plain ext4** |
| Software | llama.cpp `d08c787` + PR #27836, CUDA 12.4, `CMAKE_CUDA_ARCHITECTURES=89` |
| Model | Qwen3.8-Flash-Next-Uncensored, IQ4_XS, 97.47 GB |

What that means in practice:

- **9 t/s is a steady-state number on storage B.** On storage A, the same file
  and the same flags give 4.3 t/s. From a cold page cache, both are 2-3x lower
  for the first prompts. There is no single throughput number for this setup;
  there is a throughput number per state.
- **The thread and `--n-cpu-moe` optima are specific to 16 threads and 8 GB of
  VRAM.** Re-derive them with the scripts here rather than copying them.
- **The MTP verdict is specific to a machine where the model does not fit in
  RAM.** With more RAM it could plausibly reverse. See below.
- The measurements and their method are in
  [`docs/measurements.md`](docs/measurements.md); the raw response JSON and the
  generated text are in [`results/`](results/).

---

## Why it works

### 1. The PLE n-gram table is never held in RAM

This is the whole trick. Read straight from the GGUF tensor table
(`scripts/gguf-header.py`, which parses the header without loading anything):

```
28,800,138,240  IQ4_NL     160x320001536     per_layer_token_embd.weight
```

**28.80 GB in one tensor**, 320,001,536 rows of 160 values. That is the
per-layer-embedding n-gram lookup table, and it is 30% of the file.

It is a **lookup table**, not a matrix that participates in a matmul. With
`ple.ngram_size=3` and `heads_per_ngram=8`, generating one token needs about
**8 random reads of ~90 bytes each**. On an NVMe drive that is well under a
millisecond. So 30% of the model can live on disk essentially for free — if you
stop the runtime from trying to load it.

```
--override-tensor 'per_layer_token_embd\.weight=CPU'
--load-mode mmap
```

The override keeps the table off the GPU; mmap keeps it out of the resident set.
The kernel works the rest out by itself: `mincore(2)` on the mapping reports
**0% of the PLE pages resident** while **72% of the expert pages** are. Nobody
told it to do that. The access patterns are simply different enough — a table
touched 8 rows at a time versus experts touched by the megabyte — that plain LRU
gets it right.

One qualification, because the logs are more honest than the intent. In the
grafted-model runs the override was passed twice (see the trap below) and
llama.cpp kept only the second one, so those runs had **no explicit PLE pin** —
and they were no slower than the run that did (8.91 against 8.86 t/s). The
inference is that on this build `--n-cpu-moe` already keeps the table off the
GPU, and the explicit override is belt-and-braces rather than the load-bearing
flag. It is kept because it states the intent, it does not cost anything, and
nothing guarantees the implicit behaviour stays that way. What is *not* in doubt
is the mechanism: 28.8 GB is not resident, and the model runs in 30 GiB of RAM.

### 2. `--load-mode mmap` is mandatory, and llama.cpp will tell you otherwise

When llama.cpp sees `-ot ...=CPU` it prints:

```
tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance
```

**Do not follow that suggestion here.** `--load-mode none` disables mmap and
tries to read the whole 97 GB into 30 GiB of RAM. The advice is right for the
case it was written for — overriding a few tensors onto the CPU on a machine
where everything fits — and catastrophic for this one. It is a log line, not an
error, and it is easy to act on it without thinking.

### 3. `--n-cpu-moe 46` with `--n-gpu-layers 99`

`--n-gpu-layers 99` means "put everything on the GPU"; `--n-cpu-moe 46` then
claws back the MoE expert weights of the first 46 of 48 layers and leaves them
on the CPU. What ends up in 8 GB of VRAM is: all the attention, norms, indexer
and shared-expert weights of all 48 layers, plus the *routed experts of 2 layers*,
plus the KV cache.

Each expert layer moved into VRAM is ~862 MB on this quant, so the useful range
is three or four values wide and the bottom of it is a cliff: at 43 the model
loads all the way to the end and dies with `cudaMalloc failed: out of memory`
after about 40 seconds. `scripts/bench-ncpumoe.sh` sweeps the range and treats
that as a result rather than a crash.

Related and not optional: **`--parallel 1`**. Without it llama.cpp sizes the KV
cache for four slots, and reclaiming that ~1 GB is exactly what buys one more
expert layer in VRAM.

### 4. Move reclaim off the fault path

[`sysctl/95-large-model.conf`](sysctl/95-large-model.conf):

```
vm.swappiness = 10
vm.watermark_scale_factor = 200
vm.min_free_kbytes = 262144
```

With the defaults, the thread that takes a page fault does the memory reclaim
itself (`pgscan_direct` in `/proc/vmstat`) and blocks every other worker thread
on a futex while it does. Raising the watermarks moves that work to the
background kswapd thread: **CPU utilisation went from 381% to 498%**, and that
is what made 12 threads worth using instead of 8.

The low `swappiness` is worth a sentence, because the usual advice is the
opposite for memory-hungry workloads. It is right *here* because the two kinds
of memory in play are asymmetric: the model weights are **file-backed**, so
evicting them costs a re-read that the design already budgets for, while the KV
cache is **anonymous** memory, and pushing that to swap is ruinous. Low
swappiness biases the kernel toward exactly the eviction you want.

### 5. Threads: 12, and 16 is a cliff

| threads | t/s |
|---|---|
| 8 | 8.16 |
| **12** | **9.64** |
| 16 | 5.20 |

The collapse at 16 is real and repeatable. 16 is the hardware thread count, and
at that point the reclaim and I/O work has nowhere left to run. Note that this
only holds *after* the sysctls above: before them, 12 was within noise of 8,
because the extra threads were stalling in synchronous reclaim rather than
finding work. The sysctl file and this table are a package.

### 6. The filesystem is worth a factor of two

Same file, same bytes, same model, same flags, byte-identical output:

| storage | code | prose |
|---|---|---|
| NVMe Gen3, LUKS + btrfs | 4.34 t/s | 6.10 t/s |
| NVMe Gen4, plain ext4 | **8.91 t/s** | **9.53 t/s** |

This was the single largest factor after the base configuration, and it was the
last one found. An instrumented run of the same comparison measured where it
comes from: **7x less disk traffic at identical RAM** (520 MB/token against 75.5
MB/token, an implied page-cache hit rate of 29.8% against 89.8%). The likely
mechanism is double caching in `dm-crypt` — encrypted and decrypted pages both
held — plus btrfs copy-on-write metadata, roughly halving the usable page cache.
That has not been proven at the `dm-crypt` level; it is an inference from the
traffic counters.

The generalizable part: for a page-fault-driven access pattern, the cost of
volume encryption is **per fault**, and there are on the order of 1350 expert
faults per token. A sequential-throughput benchmark of the same encrypted volume
does not predict it.

**If you take one operational thing from this document, take this one:** put the
weights on an unencrypted filesystem.

---

## Quick start

You need llama.cpp built with CUDA, and — only for the MTP part — PR #27836.

```bash
git clone <this repo> && cd qwen-flash-next-local
cp env.example .env && $EDITOR .env        # LLAMA_CPP_DIR, MODEL_DIR

sudo cp sysctl/95-large-model.conf /etc/sysctl.d/ && sudo sysctl --system

./scripts/serve.sh                          # llama-server on 127.0.0.1:8081
./scripts/chat.sh                           # or llama-cli in a terminal
```

Every knob lives in [`scripts/config.sh`](scripts/config.sh) and can be
overridden from the environment or from `.env`. Nothing is hardcoded to one
machine: the block device used for I/O accounting, for instance, is derived from
the model path at run time, because NVMe names swap between boots.

Then re-derive the two numbers that are yours, not mine:

```bash
VALUES="48 47 46 45 44" ./scripts/bench-ncpumoe.sh   # find your VRAM cliff
VALUES="8 12 16"        ./scripts/bench-threads.sh   # find your thread optimum
./scripts/bench-prompts.sh                           # honest throughput + VRAM headroom
```

### A trap worth knowing about

`--override-tensor` **cannot be passed twice**. llama.cpp keeps only the last
occurrence and says so with nothing louder than a deprecation warning in the
log. The original runs of this experiment hit exactly that and silently lost the
PLE override. The scripts here compose one comma-separated value in `$OT`
instead; if you add your own overrides, append to it rather than adding a second
flag.

---

## Adapting it to your hardware

| you have | change |
|---|---|
| **more VRAM** | lower `NCMOE`. Each step is ~862 MB of VRAM and moves one layer's routed experts off the CPU. Sweep it — the bottom of the range is a hard OOM, not a gradual slowdown |
| **less VRAM** | raise `NCMOE` toward 48 (all experts on the CPU), and lower `UBATCH` to 128. 48 still works, at about a third of the speed |
| **more RAM** | nothing to change, but re-run the MTP comparison — the verdict below may flip for you |
| **fewer cores** | re-run `bench-threads.sh`. The optimum here was 0.75x the hardware thread count |
| **a slower SSD** | the PLE trick still works (8 reads of 90 bytes per token is nothing), but expert streaming will dominate. Watch MB/token in `bench-ncpumoe.sh`, not tokens/s |
| **an encrypted volume** | see above. Moving the weights was worth more than any flag |

### Space you need

| | |
|---|---|
| the model | **97 GB** (IQ4_XS, 3 shards) |
| working space for the MTP graft | **~50 GB** — 7.24 GB extracted + 7.4 GB bf16 + 3.9 GB Q8_0 + a 41.7 GiB rewrite of shard 1 |
| the grafted model | +2.78 GB (the new shard 4) and a rewritten shard 1; shards 2 and 3 are symlinked, not copied |

If you only want to run the model, you need the 97 GB and nothing else.

---

## The MTP head

Qwen3.8-Flash-Next ships a multi-token-prediction head: a small extra block
trained to guess the next few tokens, so the big model can verify k+1 tokens in
one batch instead of decoding them one at a time. The checkpoint that has it is
the 180 GB safetensors repository, not the GGUF.

### Getting the head without downloading 180 GB

[`mtp/extract-mtp-head.py`](mtp/extract-mtp-head.py) downloads **7.24 GiB**.

The head is 36 tensors scattered across 31 of the checkpoint's 131 shards. The
script reads each of those 31 shards' **safetensors header** — an 8-byte length
followed by a JSON blob giving every tensor's byte range — computes absolute
offsets, merges adjacent tensors into single requests, and issues **31 HTTP
`Range` requests**, one per shard. It never fetches a shard whole; it refuses an
HTTP 200 response precisely because that means the server ignored `Range` and is
about to send one.

```
plan: 36 tensors in 31 Range request(s) over 31 of 131 shards; 7.24 GiB to fetch
```

It resumes across runs and inside an interrupted run, verifies `Content-Range`,
dtypes and `shape * itemsize == data_offsets` for every tensor, and checks free
space before it starts.

### Grafting it in without copying 97 GB

[`mtp/graft-mtp-head.py`](mtp/graft-mtp-head.py) adds the quantized head as one
extra shard at the end of the existing split set. Shards 2 and 3 are symlinked,
not copied; only shard 1's header changes, and only because six loader
invariants say it must. Those invariants — which keys are read from which shard,
what is validated and what is silently ignored — are written up in
[`docs/mtp-graft.md`](docs/mtp-graft.md), and they are reusable for any
"add a block to an existing split GGUF" problem.

`mtp/test_graft_mtp_head.py` exercises the whole thing on synthetic multi-KiB
GGUFs, no model required, in about a second.

### The result: 96.1% acceptance, and slower

| | plain | MTP, `n_max 3` |
|---|---|---|
| code prompt | **8.91 t/s** | 6.99 t/s — 222/231 drafts accepted (**96.1%**), mean accepted length 3.88 of 4 |
| prose prompt | **9.53 t/s** | 5.70 t/s — 175/369 accepted (**47.4%**), mean length 2.42 |

96.1% is at the top of what the upstream PR reports, and a mean accepted length
of 3.88 out of a maximum of 4 says the head is *saturating* the draft limit, not
brushing against it. The head is working. It is still a 22% slowdown.

**Why, structurally:** verifying k+1 tokens in one batch is not free on this
machine. Each token routes to its own set of experts, so a batch of k+1 tokens
activates up to (k+1) x 10 *distinct* experts — and on a model that does not fit
in RAM, "activates an expert" means "reads it from the SSD". Speculative decoding
trades compute for sequential steps. That trade is a win when the extra compute
is free, which is what happens when the weights are in memory. Here the extra
compute is extra I/O, and the batch that was supposed to be cheap is the most
expensive thing in the loop.

**This is the number most likely to be different for you.** If your machine
holds the experts in RAM, the verification batch costs compute you already have
idle, and MTP should pay — the PR's author measures +38-52% on a 3090 with
offload. The crossover is "do the experts fit". If you have more RAM than 30 GiB,
run [`mtp/sweep-mtp.sh`](mtp/sweep-mtp.sh) before concluding anything from the
table above; it is built to separate "the draft limit binds" from "the per-step
cost grows with batch size", which is the question that decides it.

### MTP does not fit at the optimal `--n-cpu-moe`

Raising `--spec-draft-n-max` above 3 kills the server at load time on 8 GB of
VRAM:

```
common_speculative_init_result: creating MTP draft context
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 64.00 MiB on device 0:
  cudaMalloc failed: out of memory
failed to allocate buffer for kv cache
```

The MTP draft context wants its **own** KV cache on the GPU. At `--n-cpu-moe 46`
the model already sits at 7133 MiB of 8188, and 64 MiB is not there. Testing a
longer draft limit requires `--n-cpu-moe 47`, which pushes one more layer's
experts onto the CPU and off the SSD-backed fast path.

That cost belongs in the ledger for this technique, not hidden in a command
line: on this class of machine, **using MTP means giving up a layer of experts on
the GPU**. Any comparison must re-measure the plain baseline at the same
`--n-cpu-moe`, or it attributes to MTP a slowdown that is really the moved
layer.

---

## Honest limits

**What was not demonstrated.**

- **MTP output diverges from plain output at temperature 0, and it should not.**
  Greedy speculative decoding is supposed to be lossless: the target model
  verifies every drafted token, so the accepted sequence should be identical to
  what plain decoding would have produced. It is not. Same prompt, `temperature:
  0`, `seed: 42`:

  | run | code sha256 | prose sha256 |
  |---|---|---|
  | original trunk | `9689cdb784ec1244` | `7a7d0fcc65dc6045` |
  | grafted, MTP off | `9689cdb784ec1244` | `7a7d0fcc65dc6045` |
  | grafted, MTP `n_max 3` | `606ada9e5d04a35d` | `ed52448d4c5e4f8f` |
  | grafted, MTP `n_max 2` | `16e0bbb94ed294bb` | `82ebff60489609e9` |

  The graft itself is clean — MTP off reproduces the original byte for byte. But
  the MTP runs differ, and differ again between `n_max 3` and `n_max 2`. The
  divergence is **deterministic and reproducible**: the `n_max 3` hashes are
  identical across two different storage devices and two separate server runs.

  The compatible explanation is numerical: verifying k+1 tokens in a batch is a
  different order of floating-point reductions than decoding them one at a time,
  and a near-tie in expert routing or in the final logits resolves the other way.
  **That is an explanation, not a proof.** Proving it needs a logit-level
  comparison of the two paths on the diverging token, which was not done. Until
  someone does that, "MTP is lossless here" is unverified, and a real bug in the
  verification path is not excluded. Hashes and full outputs are in
  [`results/`](results/) so the claim can be checked rather than believed.

- **The `dm-crypt` double-caching hypothesis** is inferred from traffic
  counters, not measured at the device-mapper level.

- **`mincore` residency was sampled, not tracked over time.** The 0%/72% split
  is a snapshot in steady state.

- **The PLE override was not isolated as a variable.** No run deliberately
  compared "with the pin" against "without it"; the comparison above is a
  by-product of the duplicate-flag bug, across two model files. Treat "the
  override is what keeps the PLE off the GPU" as intent, not as a measured
  claim.

- **The `-md` sidecar path does not work** on PR #27836 as it stands — the
  loader marks trunk tensors as required even for a head-only file. The analysis
  and the shape of a fix are in [`docs/mtp-graft.md`](docs/mtp-graft.md), but no
  patch here has been tested.

**Known rough edges.**

- The head attends densely (no QSA). Numerically that is a superset and the
  drafts are verified anyway, but acceptance may fall at long context; it was
  measured at 300 tokens, not at 32k.
- No drafting on batches carrying embeddings (images): `GGML_ASSERT(ubatch.token)`.
- The recurrent state is checkpointed and restored on every rejection, which
  upstream reports as ~600 ms per round on discrete GPUs.
- Low-bit quantizations of this model can reason forever and emit an empty
  answer. The three settings that fix it — `--temp 0.3`, DRY, and a reasoning
  budget — are in `scripts/serve.sh` with the measurements behind each in
  [`docs/measurements.md`](docs/measurements.md).
- **The obvious DRY setting is the wrong one.** `--dry-multiplier 0.8
  --dry-allowed-length 2` penalizes every repeated 3-token sequence. A
  step-by-step explanation *has* to repeat: the array, the table, the running
  values. Over 4 repetitions per configuration it produced a self-inconsistent
  answer **2 times out of 4** — numbers silently changing mid-answer — and it
  had the **highest** repetition rate of the six configurations tested (17.3%),
  which is the one thing it exists to prevent. `serve.sh` ships `0.4` with
  `allowed_length 6`: 0/4 inconsistent, 14.1% repetition.
- **One sample per configuration is noise, and it lies in both directions.** The
  same sweep at n=1 had cleared the aggressive DRY setting and crowned
  `temp 0.15` (9% repetition, the lowest). At n=4, `temp 0.15` is the *worst*
  row in the table: 2/4 inconsistent, 29.4% repetition. `bench-coherence.sh`
  defaults to 4 repetitions for that reason; treat any single run of it as a
  hypothesis, not a result.
- `scripts/coherence.py` and `bench-coherence.sh` score an Italian prompt with
  Italian-aware patterns, because that is what was measured. Port both together
  if you change language.

---

## Layout

```
scripts/
  config.sh            all paths and tuned knobs; sourced by everything, overridable via .env
  serve.sh             llama-server with the working configuration
  chat.sh              the same through llama-cli
  bench-ncpumoe.sh     sweep --n-cpu-moe; reports t/s AND MB read per token
  bench-threads.sh     sweep --threads at a fixed --n-cpu-moe
  bench-prompts.sh     eight unrelated prompts, honest throughput, VRAM headroom watch
  bench-spec.sh        compare --spec-type variants
  bench-coherence.sh   measure the rate of self-inconsistent answers over repetitions
  coherence.py         the metric behind it
  gguf-header.py       print a GGUF's tensor table without loading the model
mtp/
  extract-mtp-head.py  fetch 36 tensors out of a 180 GB checkpoint over HTTP Range
  graft-mtp-head.py    add the head to a split GGUF as one extra shard
  test_graft_mtp_head.py   end-to-end test on synthetic GGUFs, no model needed
  measure-mtp.sh       one greedy configuration, two prompts, hashes and draft stats
  verify-graft.sh      prove the graft is clean and compare MTP output byte for byte
  sweep-mtp.sh         is the draft limit binding, or is the per-step cost growing?
docs/
  measurements.md      every number, and how it was taken
  mtp-graft.md         the loader invariants and the full graft procedure
results/               raw response JSON, generated text, sha256 manifest
sysctl/                the reclaim tuning
```

## Credits

The MTP support this builds on is llama.cpp PR #27836; the "extra shard at the
end" graft strategy follows the approach published by dzannotti in that
discussion, which is what made grafting the head possible without rewriting the
whole 97 GB model. The model is Qwen3.8-Flash-Next; the quantization used here is a
third-party abliterated IQ4_XS build. None of those are redistributed here —
this repository contains only configuration, tooling and measurements.

## License

MIT. See [LICENSE](LICENSE).
