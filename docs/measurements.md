# Measurements and how they were taken

Every number in the README comes from here. The raw response JSON and the
generated text for the greedy runs are in [`../results/`](../results/).

## Reference machine

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS, 8C/16T, AVX-512 + VNNI + BF16 |
| RAM | 30 GiB usable, DDR5-5600 dual channel (~89.6 GB/s) |
| GPU | RTX 4060 Laptop, 8188 MiB VRAM (the display runs on the iGPU, so it is nearly all free) |
| Storage A | NVMe PCIe Gen3, ~2.2 GB/s sequential, **LUKS + btrfs** |
| Storage B | NVMe PCIe Gen4, **plain ext4** |
| Software | llama.cpp `d08c787` + PR #27836, CUDA 12.4, `CMAKE_CUDA_ARCHITECTURES=89` |
| Model | Qwen3.8-Flash-Next-Uncensored, IQ4_XS, 97.47 GB in 3 shards (4 after the graft) |

## Method

- **Paired prompts.** Two fixed prompts, one code
  ([`prompt-code.txt`](../mtp/prompt-code.txt): the opening of a thread-safe LRU
  cache in Python) and one prose
  ([`prompt-prose.txt`](../mtp/prompt-prose.txt): a paragraph of Italian
  history), 300 tokens each, `temperature: 0`, `seed: 42`, `cache_prompt: false`.
  Greedy is what makes the byte-for-byte comparison meaningful.
- **Warm-up before every measurement.** One 32-token request is issued and
  discarded, so the first experts are already in the page cache. Numbers taken
  from a cold cache on this setup are 2-3x lower and say more about the cache
  than about the configuration.
- **Different topics when measuring throughput honestly.**
  `scripts/bench-prompts.sh` uses eight prompts on eight unrelated subjects.
  Asking the same question twice keeps the same experts hot and inflates the
  result to ~11.6 t/s, which is not a number anyone will see in use.
- **Real disk traffic.** `scripts/bench-ncpumoe.sh` and `bench-threads.sh` read
  sectors from `/proc/diskstats` before and after the measured request and
  report MB per generated token. That is the number that tells you whether you
  are IO-bound or CPU-bound; tokens/s alone does not.
- **The device is derived, never hardcoded.** NVMe names swap between boots and
  the model moves; `disk_of()` in `scripts/config.sh` resolves it from the model
  path each time.
- **`drop_caches` between arms** for the A/B comparisons that involve the page
  cache (filesystem, readahead, quantization). It needs root; the benchmark then
  runs as a normal user.

## Result 1 — filesystem, the single largest factor after the base config

Same grafted model, byte-identical file, same llama.cpp, same flags, same
prompts. The only difference is which volume the weights are read from.

| storage | code t/s | prose t/s |
|---|---|---|
| NVMe Gen3, LUKS + btrfs | 4.34 | 6.10 |
| NVMe Gen4, plain ext4 | **8.91** | **9.53** |

The generated text is identical between the two (sha256
`9689cdb784ec1244...` / `7a7d0fcc65dc6045...`, see
[`../results/sha256.txt`](../results/sha256.txt)), so this is purely a
throughput difference.

An earlier, more instrumented run of the same comparison on a different quant of
the same model measured where it comes from:

| | MB read per token | implied page-cache hit rate | bandwidth used |
|---|---|---|---|
| Gen3 + LUKS + btrfs | 520.0 | 29.8% | 1.38 GB/s (saturated) |
| Gen4 + ext4 | 75.5 | 89.8% | 0.65 GB/s |

**7x less disk traffic at identical RAM.** The most likely cause is double
caching in `dm-crypt` — encrypted and decrypted pages both held — plus btrfs CoW
metadata, roughly halving the usable page cache. That has not been proven at the
`dm-crypt` level here; it is an inference from the traffic.

The practical lesson generalizes past this model: for a page-fault-driven access
pattern the cost of volume encryption is **per fault**, and there are on the
order of 1350 expert-tensor faults per token. Sequential-throughput benchmarks
of the same encrypted volume do not predict it.

## Result 2 — threads

At `-ncmoe 46`, on ext4, after applying `sysctl/95-large-model.conf`:

| threads | t/s |
|---|---|
| 8 | 8.16 |
| **12** | **9.64** |
| 16 | 5.20 |

The collapse at 16 is real and repeatable — 16 is exactly the number of hardware
threads, and the reclaim and I/O work has nowhere left to run.

Before the sysctls, 12 threads were within noise of 8: the extra threads were
stalling in synchronous reclaim (`pgscan_direct`) rather than finding work.
Applying them moved reclaim to kswapd and took CPU utilisation from 381% to
498%. The sysctl file and the thread sweep are a package: neither is worth much
without the other.

## Result 3 — `--n-cpu-moe`

The sweep on the earlier UD-IQ1_S quant, which shows the shape of the curve and
the cliff:

| `-ncmoe` | t/s | MB/token |
|---|---|---|
| 48 | 2.58 | 564.9 |
| 46 | 2.76 | 425.9 |
| **45** | **4.32** | **269.0** |
| 44 | `cudaMalloc failed` | |

On the IQ4_XS the optimum is 46 and 43 does not fit. Each layer moved off the
CPU is ~862 MB of VRAM on this quant, so the useful range is narrow and the
failure at the bottom is abrupt: the model loads all the way to the end and then
dies with a CUDA OOM after ~40 seconds. `scripts/bench-ncpumoe.sh` treats that
as a data point rather than a crash.

## Result 4 — MTP

Grafted model, `--spec-type draft-mtp`, experts of block 48 pinned to the CPU.

| configuration | prompt | t/s | drafted | accepted | acceptance | mean accepted length |
|---|---|---|---|---|---|---|
| plain (no MTP) | code | 8.91 | | | | |
| plain (no MTP) | prose | 9.53 | | | | |
| `--spec-draft-n-max 3` | code | 6.99 | 231 | 222 | **96.1%** | 3.88 / 4 |
| `--spec-draft-n-max 3` | prose | 5.70 | 369 | 175 | **47.4%** | 2.42 |
| `--spec-draft-n-max 2` (btrfs run) | code | 2.52 | 206 | 195 | 94.7% | 2.89 |
| `--spec-draft-n-max 2` (btrfs run) | prose | 2.24 | 279 | 158 | 56.6% | 2.13 |

The acceptance rate on code is at the top of what the PR reports (85-89% with
head and trunk aligned; here they are, both come from the same abliterated
checkpoint) and the mean accepted length of 3.88 out of a maximum of 4 says the
head is *saturating* the draft limit, not brushing against it.

And it is still slower. See the README for why, and `mtp/sweep-mtp.sh` for the
experiment that separates "the limit binds" from "the per-step cost grows".

## Result 5 — things that did not help, measured rather than assumed

- **`--spec-type ngram-mod`**: 25.8% acceptance, not enough for the accepted
  tokens to pay for the rejected ones. 14.70 t/s with it against 18.51 without,
  same prompt, same warm state.
- **Readahead**: the 128 KB default wins. 512 KB is indistinguishable from noise
  (4/8 prompts better, 4/8 worse); 1024 KB costs **−32%** (7/8 prompts worse).
  The likely mechanism is read amplification: over-reading does not merely waste
  bandwidth, it evicts hot experts from a cache that is already too small.
- **`--ubatch-size 512`**: works, but leaves 299 MiB of VRAM headroom and
  reliably OOMs on the *second* request, when CUDA graphs are reallocated. 256
  leaves 377 MiB, stays flat across eight requests, and costs nothing in decode
  speed.
- **`--parallel 1` is mandatory**: without it llama.cpp allocates KV cache for
  four slots. Reclaiming that ~1 GB is exactly what lets one more expert layer
  fit in VRAM.

## Result 6 — the bandwidth view

Both of these saturate whatever medium holds their weights:

| | a 35B-A3B model that fits in RAM | this 125B-A6B streaming from SSD |
|---|---|---|
| bytes per token | 2117 MB | 520 MB |
| t/s | 39 | 2.9 |
| **achieved bandwidth** | **83 GB/s** (92% of DDR5) | **1.51 GB/s** (69% of NVMe) |

A 41x gap in media speed becomes a 13x gap in tokens/s, because the ultra-sparse
MoE moves 4x fewer bytes per token. This is the whole reason the exercise works.

(The 2.9 t/s figure is the encrypted-btrfs arm; on ext4 the same model is at
8.9-9.5 t/s and no longer IO-bound at all — at 0.65 GB/s out of ~5 available the
bottleneck has moved to the CPU.)

## Result 7 — low-bit reasoning models may never emit an answer

Not a performance issue, but it costs time to diagnose. At IQ2_XXS the model
would reason for 10,000+ characters and then end the turn with an **empty**
`content` field: `finish_reason: stop`, nothing after the reasoning. Three
changes, all needed:

- **`--temp 0.3`** instead of the 1.0 on the model card. That recommendation is
  for full precision; at ~2 bits per weight the distribution is flat enough that
  1.0 samples the quantization noise. It also **halved** the reasoning length.
- **DRY** (`--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2`). This
  is what actually fixes the empty answers: the model was looping *inside* the
  reasoning block until it ran out of budget. Low temperature *increases* loop
  risk, so temp 0.3 and DRY have to be adopted together.
- **`--reasoning-budget 256`** plus **`--reasoning-budget-message`**. The message
  turned out not to be what makes the answer appear — a controlled run shows the
  answer appears without it too — but without it the repeated-trigram rate goes
  from 2% to 22% and the reply balloons from 1337 to 5686 characters. It buys
  conciseness and loop suppression, not existence.

`scripts/bench-coherence.sh` and `scripts/coherence.py` are the tooling for the
related defect where one answer contains several mutually inconsistent numeric
examples: the metric counts distinct arrays and self-contradicting claims, over
repetitions, because a single sample per configuration cannot distinguish a
better sampler from noise.
