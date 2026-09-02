#!/usr/bin/env python3
"""End-to-end check of graft-mtp-head.py on a synthetic two-shard trunk and a synthetic head.

  test_graft_mtp_head.py [--workdir DIR]   (DIR must be on a real disk, default ~/.cache)

Builds files of a few KiB, grafts the head, re-reads every output with GGUFReader and
compares metadata and bytes; then checks two refusals (wrong block index, existing output).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _use_gguf_py() -> None:
    """Same lookup as graft-mtp-head.py: $LLAMA_CPP_DIR/gguf-py, then ../gguf-py."""
    roots = []
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        roots.append(Path(env).expanduser() / "gguf-py")
    roots.append(HERE.parent / "gguf-py")
    for root in roots:
        if (root / "gguf").is_dir():
            sys.path.insert(0, str(root))
            return


_use_gguf_py()

import numpy as np  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter  # noqa: E402

SCRIPT   = HERE / "graft-mtp-head.py"
ARCH     = "qwen4exp"
N_EMBD   = 64
N_LAYER  = 2
PREFIX   = "tiny"
Q8_ROW_BYTES = N_EMBD // 32 * 34  # Q8_0: 32 elements per block, 34 bytes per block


def q8_rows(rng: np.random.Generator, rows: int) -> np.ndarray:
    return rng.integers(0, 256, size=(rows, Q8_ROW_BYTES), dtype=np.uint8)


def add_model_keys(writer: GGUFWriter, block_count: int, ratios: list[int]) -> None:
    writer.add_name("tiny trunk")
    writer.add_block_count(block_count)
    writer.add_embedding_length(N_EMBD)
    writer.add_array(f"{ARCH}.rope.dimension_sections", [1, 1, 1, 0])
    writer.add_attention_compress_ratios(ratios)
    writer.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    writer.add_float32(f"{ARCH}.rope.freq_base", 10000.0)


def finish(writer: GGUFWriter) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def build_trunk(dir_: Path, rng: np.random.Generator) -> Path:
    first = dir_ / f"{PREFIX}-00001-of-00002.gguf"
    w = GGUFWriter(first, ARCH)
    add_model_keys(w, N_LAYER, [0, 4])
    w.add_uint16("split.no", 0)
    w.add_uint16("split.count", 2)
    w.add_int32("split.tensors.count", 4)
    w.add_tensor("token_embd.weight", rng.standard_normal((8, N_EMBD), dtype=np.float32))
    w.add_tensor("blk.0.attn_q.weight", q8_rows(rng, 4), raw_dtype=GGMLQuantizationType.Q8_0)
    w.add_tensor("blk.0.attn_norm.weight", rng.standard_normal(N_EMBD, dtype=np.float32))
    finish(w)

    w = GGUFWriter(dir_ / f"{PREFIX}-00002-of-00002.gguf", ARCH)
    w.remove_key("general.architecture")
    w.add_uint16("split.no", 1)
    w.add_uint16("split.count", 2)
    w.add_int32("split.tensors.count", 4)
    w.add_tensor("blk.1.attn_q.weight", q8_rows(rng, 4), raw_dtype=GGMLQuantizationType.Q8_0)
    finish(w)
    return first


def build_head(path: Path, rng: np.random.Generator, block: int) -> Path:
    w = GGUFWriter(path, ARCH)
    add_model_keys(w, N_LAYER + 1, [0, 4, 0])
    w.add_nextn_predict_layers(1)
    w.add_tensor("token_embd.weight", rng.standard_normal((8, N_EMBD), dtype=np.float32))
    w.add_tensor(f"blk.{block}.attn_q.weight", q8_rows(rng, 4), raw_dtype=GGMLQuantizationType.Q8_0)
    w.add_tensor(f"blk.{block}.nextn.enorm.weight", rng.standard_normal(N_EMBD, dtype=np.float32))
    finish(w)
    return path


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def field(reader: GGUFReader, key: str):
    f = reader.get_field(key)
    return None if f is None else f.contents()


def check(cond: bool, what: str) -> None:
    if not cond:
        sys.exit(f"FAIL: {what}")
    print(f"ok: {what}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path.home() / ".cache")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    with tempfile.TemporaryDirectory(dir=args.workdir, prefix="graft-mtp-test-") as tmp:
        base = Path(tmp)
        trunk_first = build_trunk(base / "trunk", rng) if (base / "trunk").mkdir() is None else None
        head = build_head(base / "head.gguf", rng, N_LAYER)
        out = base / "out"

        dry = run("--trunk", str(trunk_first), "--head", str(head), "--out", str(out), "--dry-run")
        check(dry.returncode == 0 and not out.exists(), f"dry run writes nothing\n{dry.stderr}")

        real = run("--trunk", str(trunk_first), "--head", str(head), "--out", str(out), "-v")
        check(real.returncode == 0, f"graft succeeds\n{real.stderr}")

        old = GGUFReader(trunk_first)
        new = GGUFReader(out / f"{PREFIX}-00001-of-00003.gguf")
        check(field(new, f"{ARCH}.block_count") == 3 and field(new, f"{ARCH}.nextn_predict_layers") == 1, "block_count 3, nextn 1")
        check(list(field(new, f"{ARCH}.attention.compress_ratios")) == [0, 4, 0], "compress_ratios extended with 0")
        check(list(field(new, f"{ARCH}.rope.dimension_sections")) == [1, 1, 1, 0], "arrays of other lengths untouched")
        check(field(new, "split.count") == 3 and field(new, "split.tensors.count") == 6 and field(new, "split.no") == 0, "split keys of shard 1")
        check(field(new, "tokenizer.ggml.tokens") == ["a", "b", "c"] and field(new, "general.name") == "tiny trunk", "string keys copied")
        check(field(new, "tokenizer.ggml.add_bos_token") is not None and field(new, "tokenizer.ggml.add_bos_token") == field(old, "tokenizer.ggml.add_bos_token"), "bool key copied")
        check(len(new.fields) == len(old.fields) + 1, "exactly one key added")
        check([t.name for t in new.tensors] == [t.name for t in old.tensors], "shard 1 tensor names and order")
        for a, b in zip(old.tensors, new.tensors):
            check(a.tensor_type == b.tensor_type and a.shape.tolist() == b.shape.tolist()
                  and np.array_equal(a.data.reshape(-1).view(np.uint8), b.data.reshape(-1).view(np.uint8)), f"shard 1 bytes of {a.name}")

        link = out / f"{PREFIX}-00002-of-00003.gguf"
        check(link.is_symlink() and os.readlink(link) == str(base / "trunk" / f"{PREFIX}-00002-of-00002.gguf"), "shard 2 is a symlink to the trunk")

        last = GGUFReader(out / f"{PREFIX}-00003-of-00003.gguf")
        head_r = GGUFReader(head)
        check(field(last, "split.no") == 2 and field(last, "split.count") == 3 and field(last, "split.tensors.count") == 6, "split keys of the head shard")
        check(field(last, "general.architecture") is None, "head shard carries no model keys")
        check([t.name for t in last.tensors] == ["blk.2.attn_q.weight", "blk.2.nextn.enorm.weight"], "head shard holds only blk.2.*")
        for want in head_r.tensors:
            got = next((t for t in last.tensors if t.name == want.name), None)
            if got is None:
                continue
            check(want.tensor_type == got.tensor_type and want.shape.tolist() == got.shape.tolist()
                  and np.array_equal(want.data.reshape(-1).view(np.uint8), got.data.reshape(-1).view(np.uint8)), f"head bytes of {want.name}")

        again = run("--trunk", str(trunk_first), "--head", str(head), "--out", str(out))
        check(again.returncode != 0 and "already exists" in again.stderr, "refuses to overwrite an existing output")

        bad = build_head(base / "bad-head.gguf", rng, N_LAYER + 3)
        wrong = run("--trunk", str(trunk_first), "--head", str(bad), "--out", str(base / "out2"))
        check(wrong.returncode != 0 and not (base / "out2").exists(), "refuses a head with the wrong block index")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
