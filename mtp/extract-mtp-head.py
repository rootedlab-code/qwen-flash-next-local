#!/usr/bin/env python3
"""Extract the Qwen3.8-Flash-Next MTP draft head from a Hugging Face safetensors checkpoint.

Only the byte ranges of the wanted tensors are fetched (HTTP Range), so the shards that
hold them are never downloaded whole. The output is one safetensors file, a matching
model.safetensors.index.json and the small config/tokenizer files, so the directory can
be fed to convert_hf_to_gguf.py --mtp.

Nothing is written under /tmp, /dev/shm or /run: those are usually tmpfs (RAM), and
the output is several GiB. Set MTP_OUT, or pass --out, to choose the directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

try:
    import requests
    from huggingface_hub import HfApi, get_token, hf_hub_download, hf_hub_url
except ImportError as exc:
    sys.exit(f"missing python package '{exc.name}': pip install requests huggingface_hub")

logger = logging.getLogger("extract-mtp-head")

T = TypeVar("T")

DEFAULT_REPO     = "orcarouter/Qwen3.8-Flash-Next-Uncensored"
DEFAULT_REVISION = "main"
DEFAULT_OUT      = Path(os.environ.get("MTP_OUT", "mtp-head"))
TMPFS_PREFIXES   = ("/tmp", "/dev/shm", "/run")

HEAD_PREFIX       = "mtp."
EXPECT_HEAD_COUNT = 31
# the detached-head (-md) path needs the token table and the LM head; the trunk mixer is tiny.
# Each group lists the names one checkpoint layout or another uses for the same tensor; the
# converter strips "language_model." (conversion/base.py), so either spelling converts alike.
EXTRA_TENSORS  = (("model.embed_tokens.weight", "model.language_model.embed_tokens.weight"),
                  ("lm_head.weight",))
EXTRA_PREFIXES = ("model.hyper_connection_mixer.", "model.language_model.hyper_connection_mixer.")

INDEX_FILE    = "model.safetensors.index.json"
OUT_WEIGHTS   = "model-mtp-head.safetensors"  # convert_hf_to_gguf.py lists the parts as model*.safetensors
PROGRESS_FILE = ".extract-mtp-head.progress.json"

HEADER_LEN_BYTES   = 8          # safetensors: little-endian u64, then the JSON header
HEADER_PROBE_BYTES = 2 << 20
HEADER_MAX_BYTES   = 100 << 20  # the reference implementation refuses larger headers
HEADER_ALIGN       = 8
CHUNK_BYTES        = 8 << 20
AUX_MAX_BYTES      = 64 << 20   # tokenizer.json is ~10 MiB; anything larger is a weight
FREE_SPACE_MARGIN  = 1 << 30
MAX_ATTEMPTS       = 6
BACKOFF_BASE_S     = 2.0
TIMEOUT_S          = (30, 120)

DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


@dataclass(frozen=True)
class TensorRange:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    begin: int  # absolute byte offset inside the shard
    end: int    # exclusive

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class Run:
    """Contiguous tensors of one shard, fetched with a single Range request."""
    shard: str
    begin: int
    end: int
    tensors: tuple[TensorRange, ...]

    @property
    def key(self) -> str:
        return f"{self.shard}:{self.begin}-{self.end}"


@dataclass(frozen=True)
class Layout:
    header_bytes: bytes
    offsets: dict[str, int]  # tensor name -> absolute offset in the output file
    data_bytes: int

    @property
    def total_bytes(self) -> int:
        return len(self.header_bytes) + self.data_bytes

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.header_bytes).hexdigest()


def refuse_tmpfs(dest: Path) -> None:
    for prefix in TMPFS_PREFIXES:
        if dest == Path(prefix) or Path(prefix) in dest.parents:
            sys.exit(f"{dest} is under {prefix}, which is tmpfs (RAM) on this machine: choose a path on disk")


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def make_session(token: str | None) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "extract-mtp-head/1.0"
    if token:
        # requests drops Authorization on the cross-host redirect to the CDN, which is what we want
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def with_retries(action: Callable[[], T], what: str) -> T:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return action()
        except (requests.RequestException, OSError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"{what}: giving up after {attempt} attempts") from exc
            delay = BACKOFF_BASE_S ** attempt
            logger.warning("%s: attempt %d failed (%s); retrying in %.0fs", what, attempt, exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def parse_content_range(value: str) -> tuple[int, int]:
    """'bytes 10-19/1234' -> (10, 20)."""
    try:
        unit, rest = value.split(" ", 1)
        span, _total = rest.split("/", 1)
        first, last = (int(x) for x in span.split("-", 1))
    except ValueError as exc:
        raise RuntimeError(f"malformed Content-Range {value!r}") from exc
    if unit != "bytes" or last < first:
        raise RuntimeError(f"malformed Content-Range {value!r}")
    return first, last + 1


def open_range(session: requests.Session, url: str, begin: int, end: int,
               allow_short: bool = False) -> requests.Response:
    """Streaming 206 response for [begin, end); a server that ignores Range is a hard error."""
    resp = session.get(url, headers={"Range": f"bytes={begin}-{end - 1}"}, stream=True, timeout=TIMEOUT_S)
    try:
        resp.raise_for_status()
        if resp.status_code != 206:
            raise RuntimeError(f"server ignored Range (HTTP {resp.status_code}) for {url}")
        got_begin, got_end = parse_content_range(resp.headers.get("Content-Range", ""))
        if got_begin != begin or got_end > end or (got_end != end and not allow_short):
            raise RuntimeError(f"Content-Range {resp.headers.get('Content-Range')!r} does not cover [{begin}, {end})")
    except Exception:
        resp.close()
        raise
    return resp


def read_range(session: requests.Session, url: str, begin: int, end: int, allow_short: bool = False) -> bytes:
    def fetch() -> bytes:
        with open_range(session, url, begin, end, allow_short) as resp:
            return resp.content

    data = with_retries(fetch, f"GET [{begin}, {end}) of {url}")
    if len(data) > end - begin or (len(data) != end - begin and not allow_short):
        raise RuntimeError(f"got {len(data)} bytes for [{begin}, {end}) of {url}")
    return data


def fetch_json(session: requests.Session, url: str) -> dict:
    def fetch() -> dict:
        resp = session.get(url, timeout=TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()

    return with_retries(fetch, f"GET {url}")


def fetch_header(session: requests.Session, url: str) -> tuple[dict, int]:
    """Return (tensor entries, data_start) of a safetensors shard, reading only its header."""
    probe = read_range(session, url, 0, HEADER_PROBE_BYTES, allow_short=True)
    if len(probe) < HEADER_LEN_BYTES:
        raise RuntimeError(f"{url}: shorter than a safetensors header")
    (header_len,) = struct.unpack("<Q", probe[:HEADER_LEN_BYTES])
    if header_len == 0 or header_len > HEADER_MAX_BYTES:
        raise RuntimeError(f"{url}: implausible header length {header_len}")
    data_start = HEADER_LEN_BYTES + header_len
    raw = probe[HEADER_LEN_BYTES:data_start]
    if len(raw) < header_len:
        raw += read_range(session, url, len(probe), data_start)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header, data_start


def wanted_names(weight_map: dict[str, str], head_only: bool) -> list[str]:
    names = sorted(n for n in weight_map if n.startswith(HEAD_PREFIX))
    if head_only:
        return names
    for group in EXTRA_TENSORS:
        present = [n for n in group if n in weight_map]
        if not present:
            logger.info("%s is not in the checkpoint (tied or absent); skipping", " / ".join(group))
        names += present
    names += sorted(n for n in weight_map if n.startswith(EXTRA_PREFIXES))
    return names


def check_head_count(names: list[str], expect: int) -> None:
    n_head = sum(1 for n in names if n.startswith(HEAD_PREFIX))
    if n_head == 0:
        sys.exit(f"no '{HEAD_PREFIX}*' tensors in {INDEX_FILE}: this checkpoint carries no MTP head")
    if expect and n_head != expect:
        sys.exit(f"found {n_head} '{HEAD_PREFIX}*' tensors, expected {expect}: the checkpoint layout changed, "
                 "re-check it before extracting (or pass --expect-count 0)")


def tensor_range(name: str, shard: str, header: dict, data_start: int) -> TensorRange:
    entry = header.get(name)
    if entry is None:
        raise KeyError(f"{name}: listed under {shard} in {INDEX_FILE} but missing from its header")
    dtype = entry["dtype"]
    shape = tuple(int(d) for d in entry["shape"])
    begin, end = (int(x) for x in entry["data_offsets"])
    if dtype not in DTYPE_SIZE:
        raise ValueError(f"{name}: unsupported dtype {dtype}")
    expected = math.prod(shape) * DTYPE_SIZE[dtype]
    if begin < 0 or end < begin or end - begin != expected:
        raise ValueError(f"{name}: data_offsets {entry['data_offsets']} do not match {shape} {dtype} ({expected} bytes)")
    return TensorRange(name, shard, dtype, shape, data_start + begin, data_start + end)


def resolve_ranges(session: requests.Session, repo: str, revision: str,
                   weight_map: dict[str, str], names: list[str]) -> list[TensorRange]:
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)

    ranges: list[TensorRange] = []
    for shard in sorted(by_shard):
        header, data_start = fetch_header(session, hf_hub_url(repo, shard, revision=revision))
        found = [tensor_range(name, shard, header, data_start) for name in by_shard[shard]]
        ranges += found
        logger.info("%s: %d tensor(s), %.1f MiB", shard, len(found), sum(t.nbytes for t in found) / 2**20)
    return ranges


def coalesce(ranges: list[TensorRange]) -> list[Run]:
    runs: list[Run] = []
    current: list[TensorRange] = []

    def flush() -> None:
        if current:
            runs.append(Run(current[0].shard, current[0].begin, current[-1].end, tuple(current)))

    for t in sorted(ranges, key=lambda t: (t.shard, t.begin)):
        if current and current[-1].shard == t.shard:
            if t.begin < current[-1].end:
                raise RuntimeError(f"{t.name} overlaps {current[-1].name} inside {t.shard}: corrupt header")
            if t.begin == current[-1].end:
                current.append(t)
                continue
        flush()
        current = [t]
    flush()
    return runs


def build_layout(runs: list[Run]) -> Layout:
    """Lay the tensors out in run order, so every source run is one contiguous write."""
    header: dict = {"__metadata__": {"format": "pt"}}
    relative: dict[str, int] = {}
    cursor = 0
    for run in runs:
        for t in run.tensors:
            header[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [cursor, cursor + t.nbytes]}
            relative[t.name] = cursor
            cursor += t.nbytes
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body += b" " * (-len(body) % HEADER_ALIGN)
    header_bytes = struct.pack("<Q", len(body)) + body
    offsets = {name: len(header_bytes) + rel for name, rel in relative.items()}
    return Layout(header_bytes, offsets, cursor)


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_progress(out_dir: Path, digest: str) -> set[str]:
    path = out_dir / PROGRESS_FILE
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    if saved.get("layout_sha256") != digest:
        logger.warning("%s belongs to a different layout; starting over", path)
        return set()
    return set(saved.get("done", []))


def save_progress(out_dir: Path, digest: str, done: set[str]) -> None:
    atomic_write_json(out_dir / PROGRESS_FILE, {"layout_sha256": digest, "done": sorted(done)})


def download_run(session: requests.Session, url: str, run: Run, out, out_offset: int) -> None:
    """Stream one run into the output file; a dropped connection resumes from the bytes already written."""
    total = run.end - run.begin
    done = 0
    attempt = 0
    while done < total:
        try:
            with open_range(session, url, run.begin + done, run.end) as resp:
                out.seek(out_offset + done)
                for chunk in resp.iter_content(CHUNK_BYTES):
                    out.write(chunk)
                    done += len(chunk)
        except (requests.RequestException, OSError) as exc:
            attempt += 1
            if attempt >= MAX_ATTEMPTS:
                raise RuntimeError(f"{run.key}: giving up after {attempt} attempts") from exc
            delay = BACKOFF_BASE_S ** attempt
            logger.warning("%s: attempt %d failed at %d/%d bytes (%s); retrying in %.0fs",
                           run.key, attempt, done, total, exc, delay)
            time.sleep(delay)
    if done != total:
        raise RuntimeError(f"{run.key}: wrote {done} bytes, expected {total}")


def write_weights(session: requests.Session, repo: str, revision: str, runs: list[Run],
                  layout: Layout, out_dir: Path) -> None:
    part = out_dir / (OUT_WEIGHTS + ".part")
    done = load_progress(out_dir, layout.digest) if part.exists() else set()

    with open(part, "r+b" if part.exists() else "w+b") as out:
        out.truncate(layout.total_bytes)
        out.seek(0)
        out.write(layout.header_bytes)
        for i, run in enumerate(runs, 1):
            if run.key in done:
                logger.info("[%d/%d] %s: already done", i, len(runs), run.key)
                continue
            logger.info("[%d/%d] %s: %.1f MiB, %d tensor(s)", i, len(runs), run.key,
                        (run.end - run.begin) / 2**20, len(run.tensors))
            download_run(session, hf_hub_url(repo, run.shard, revision=revision), run, out,
                         layout.offsets[run.tensors[0].name])
            out.flush()
            done.add(run.key)
            save_progress(out_dir, layout.digest, done)

    os.replace(part, out_dir / OUT_WEIGHTS)
    (out_dir / PROGRESS_FILE).unlink(missing_ok=True)


def write_index(out_dir: Path, layout: Layout) -> None:
    index = {
        "metadata": {"total_size": layout.data_bytes},
        "weight_map": {name: OUT_WEIGHTS for name in layout.offsets},
    }
    atomic_write_json(out_dir / INDEX_FILE, index)


def download_aux_files(api: HfApi, repo: str, revision: str, out_dir: Path) -> None:
    """config.json, tokenizer files and the like: top-level, small, not weights."""
    for entry in api.list_repo_tree(repo, revision=revision):
        path = entry.path
        size = getattr(entry, "size", None)
        if "/" in path or path.endswith(".safetensors") or path == INDEX_FILE:
            continue
        if size is None or size > AUX_MAX_BYTES:
            continue
        logger.info("aux: %s (%d bytes)", path, size)
        hf_hub_download(repo, path, revision=revision, local_dir=str(out_dir))


def report_plan(runs: list[Run], layout: Layout, weight_map: dict[str, str]) -> None:
    shards = {run.shard for run in runs}
    logger.info("plan: %d tensors in %d Range request(s) over %d of %d shards; %.2f GiB to fetch, %.2f GiB on disk",
                len(layout.offsets), len(runs), len(shards), len(set(weight_map.values())),
                layout.data_bytes / 2**30, layout.total_bytes / 2**30)
    for run in runs:
        for t in run.tensors:
            logger.debug("  %-60s %-5s %-24s %10.1f MiB  %s", t.name, t.dtype, list(t.shape), t.nbytes / 2**20, t.shard)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face repo (default: {DEFAULT_REPO})")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="branch, tag or commit (default: main)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output directory, on a real disk (default: {DEFAULT_OUT})")
    parser.add_argument("--expect-count", type=int, default=EXPECT_HEAD_COUNT,
                        help=f"abort unless exactly this many '{HEAD_PREFIX}*' tensors exist; 0 disables "
                             f"(default: {EXPECT_HEAD_COUNT})")
    parser.add_argument("--head-only", action="store_true",
                        help=f"only the '{HEAD_PREFIX}*' tensors; skip {', '.join(g[0] for g in EXTRA_TENSORS)} and "
                             f"{', '.join(p + '*' for p in EXTRA_PREFIXES)} (enough for grafting into a full trunk GGUF, "
                             "not for a detached -md draft)")
    parser.add_argument("--skip-aux", action="store_true", help="do not download config/tokenizer files")
    parser.add_argument("--dry-run", action="store_true",
                        help="read the index and the shard headers, print the plan, write nothing")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every tensor in the plan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    out_dir = args.out.expanduser().resolve()
    refuse_tmpfs(out_dir)

    token = get_token()
    if token is None:
        logger.warning("no Hugging Face token found (run `huggingface-cli login`); a gated repo will answer 401")
    session = make_session(token)

    index = fetch_json(session, hf_hub_url(args.repo, INDEX_FILE, revision=args.revision))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        sys.exit(f"{INDEX_FILE} of {args.repo} has no weight_map")

    names = wanted_names(weight_map, args.head_only)
    check_head_count(names, args.expect_count)

    runs = coalesce(resolve_ranges(session, args.repo, args.revision, weight_map, names))
    layout = build_layout(runs)
    report_plan(runs, layout, weight_map)
    if args.dry_run:
        logger.info("dry run: nothing written")
        return 0

    needed = layout.total_bytes + FREE_SPACE_MARGIN
    if free_bytes(out_dir) < needed:
        sys.exit(f"{out_dir}: need {needed / 2**30:.1f} GiB free, have {free_bytes(out_dir) / 2**30:.1f} GiB")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_weights(session, args.repo, args.revision, runs, layout, out_dir)
    write_index(out_dir, layout)
    if not args.skip_aux:
        download_aux_files(HfApi(token=token), args.repo, args.revision, out_dir)
    logger.info("done: %s (%.2f GiB)", out_dir / OUT_WEIGHTS, layout.total_bytes / 2**30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
