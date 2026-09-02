#!/usr/bin/env python3
"""Graft a quantized qwen4exp MTP head GGUF onto a split trunk GGUF as one extra shard.

  graft-mtp-head.py --trunk <trunk shard 1> --head <head gguf> --out <dir> [--name PREFIX] [--dry-run]

Nothing under the trunk's directory is touched. Under --out:
  <prefix>-00001-of-000MM.gguf  shard 1 rewritten: block_count + n_head_layers, nextn_predict_layers,
                                every <arch>.* array of length block_count extended, split.count,
                                split.tensors.count; tensor data copied byte for byte
  <prefix>-000ii-of-000MM.gguf  i in 2..N: symlinks to the trunk's own shards
  <prefix>-000MM-of-000MM.gguf  the head's blk.<block_count>.* tensors, raw bytes (MM = N + 1)

Loader facts this relies on (src/llama-model-loader.cpp, src/models/qwen4exp.cpp load_arch_hparams):
  1. every model KV is read from shard 1 only;
  2. of the other shards only split.no == file index is checked;
  3. split.tensors.count of shard 1 must equal the tensor total over all shards;
  4. a tensor name present in two shards is an error;
  5. file names derive from shard 1: <prefix>-000ii-of-000MM.gguf with MM = split.count;
  6. per-layer arrays are read with length block_count.

Every file is written under a temporary name and renamed into place once complete.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any



def _use_gguf_py() -> None:
    """Put llama.cpp's gguf-py on sys.path, unless an installed `gguf` already works.

    Checked in order: $LLAMA_CPP_DIR/gguf-py, then ../gguf-py (this file living inside a
    llama.cpp checkout). If neither exists the plain `import gguf` below has to succeed,
    which it does when the package is pip-installed.
    """
    roots = []
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        roots.append(Path(env).expanduser() / "gguf-py")
    roots.append(Path(__file__).resolve().parent.parent / "gguf-py")
    for root in roots:
        if (root / "gguf").is_dir():
            sys.path.insert(0, str(root))
            return


_use_gguf_py()

import numpy as np  # noqa: E402
from gguf import GGUFReader, GGUFValueType, GGUFWriter, Keys  # noqa: E402
from gguf.gguf_reader import ReaderTensor  # noqa: E402

logger = logging.getLogger("graft-mtp-head")

TMPFS_PREFIXES = ("/tmp", "/dev/shm", "/run")
SPLIT_NAME_RE  = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{5})-of-(?P<count>\d{5})\.gguf$")

KEY_ARCH          = Keys.General.ARCHITECTURE
KEY_SPLIT_NO      = Keys.Split.LLM_KV_SPLIT_NO
KEY_SPLIT_COUNT   = Keys.Split.LLM_KV_SPLIT_COUNT
KEY_SPLIT_TENSORS = Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT
READER_VIRTUAL_PREFIX = "GGUF."

COMPRESS_RATIOS_SUFFIX = ".attention.compress_ratios"
DENSE_COMPRESS_RATIO   = 0  # the MTP block attends densely
# hparams the head must share with the trunk: a head from another checkpoint would load and draft noise
SHAPE_KEYS = (
    "embedding_length", "attention.head_count", "attention.head_count_kv", "attention.key_length",
    "attention.value_length", "expert_count", "expert_used_count", "expert_feed_forward_length",
    "expert_shared_feed_forward_length", "hyper_connection.count", "hyper_connection.low_rank",
    "attention.indexer.head_count", "attention.indexer.key_length",
)

COPY_CHUNK_BYTES  = 64 << 20
FLUSH_EVERY_BYTES = 1 << 30   # bound dirty pages: 30 GiB of RAM would otherwise stall on writeback
COMPARE_CHUNK     = 256 << 20
FREE_SPACE_MARGIN = 1 << 30


@dataclass(frozen=True)
class Plan:
    arch: str
    trunk_dir: Path
    trunk_prefix: str
    out_prefix: str
    n_split_old: int
    n_tensors_old: int
    n_layer: int
    n_nextn: int
    extended_arrays: dict[str, tuple[list[Any], GGUFValueType]]
    head_tensors: tuple[ReaderTensor, ...]
    skipped_head_tensors: tuple[str, ...]
    trunk_size: int
    trunk_data_len: int

    @property
    def n_split_new(self) -> int:
        return self.n_split_old + 1

    @property
    def n_tensors_new(self) -> int:
        return self.n_tensors_old + len(self.head_tensors)

    @property
    def head_bytes(self) -> int:
        return sum(t.n_bytes for t in self.head_tensors)

    def out_name(self, idx: int) -> str:
        return split_name(self.out_prefix, idx, self.n_split_new)

    def trunk_name(self, idx: int) -> str:
        return split_name(self.trunk_prefix, idx, self.n_split_old)


def refuse_tmpfs(dest: Path) -> None:
    for prefix in TMPFS_PREFIXES:
        if dest == Path(prefix) or Path(prefix) in dest.parents:
            sys.exit(f"{dest} is under {prefix}, which is tmpfs (RAM) on this machine: choose a path on disk")


def split_name(prefix: str, idx: int, count: int) -> str:
    return f"{prefix}-{idx:05d}-of-{count:05d}.gguf"


def parse_split_name(path: Path) -> tuple[str, int, int]:
    match = SPLIT_NAME_RE.match(path.name)
    if match is None:
        sys.exit(f"{path.name}: not a split file name (<prefix>-000ii-of-000NN.gguf); single-file trunks are not supported")
    return match["prefix"], int(match["idx"]), int(match["count"])


def scalar(reader: GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    return None if field is None else field.contents()


def require_int(reader: GGUFReader, key: str, what: str) -> int:
    value = scalar(reader, key)
    if value is None:
        sys.exit(f"{what}: missing key {key}")
    return int(value)


def open_gguf(path: Path) -> GGUFReader:
    """Header-only access: the reader maps the file, but touches nothing past the tensor infos."""
    try:
        return GGUFReader(path)
    except (OSError, ValueError) as exc:
        sys.exit(f"{path}: {exc}")


def pad(n: int, align: int) -> int:
    return GGUFWriter.ggml_pad(n, align)


def check_head_matches_trunk(trunk: GGUFReader, head: GGUFReader, arch: str) -> None:
    for suffix in SHAPE_KEYS:
        key = f"{arch}.{suffix}"
        trunk_val, head_val = scalar(trunk, key), scalar(head, key)
        if trunk_val is None or head_val is None:
            logger.warning("%s: absent in %s, not compared", key, "trunk" if trunk_val is None else "head")
            continue
        if trunk_val != head_val:
            sys.exit(f"{key}: trunk has {trunk_val!r}, head has {head_val!r}: the head belongs to another model")


def select_head_tensors(head: GGUFReader, n_layer: int, n_nextn: int) -> tuple[list[ReaderTensor], list[str]]:
    wanted_layers = set(range(n_layer, n_layer + n_nextn))
    blocks, skipped = [], []
    for t in head.tensors:
        parts = t.name.split(".")
        if parts[0] != "blk":
            skipped.append(t.name)  # token_embd, output, ...: the trunk already has them (fact 4)
            continue
        layer = int(parts[1])
        if layer not in wanted_layers:
            sys.exit(f"{t.name}: head block index outside {sorted(wanted_layers)}: was the head converted "
                     f"against a trunk of {n_layer} layers?")
        blocks.append(t)
    found_layers = {int(t.name.split(".")[1]) for t in blocks}
    if found_layers != wanted_layers:
        sys.exit(f"head carries blocks {sorted(found_layers)}, expected {sorted(wanted_layers)}")
    return blocks, skipped


def per_layer_arrays(trunk: GGUFReader, arch: str, n_layer: int, n_nextn: int) -> dict[str, tuple[list[Any], GGUFValueType]]:
    """Fact 6: every <arch>.* array of exactly n_layer entries is read with length block_count."""
    extended: dict[str, tuple[list[Any], GGUFValueType]] = {}
    for field in trunk.fields.values():
        if not field.name.startswith(arch + ".") or field.types[0] != GGUFValueType.ARRAY:
            continue
        values = list(field.contents())
        if len(values) != n_layer:
            continue
        if field.name.endswith(COMPRESS_RATIOS_SUFFIX):
            tail = [DENSE_COMPRESS_RATIO] * n_nextn
        else:
            tail = [values[-1]] * n_nextn
            logger.warning("%s: per-layer array of unknown meaning, extended with its last value %r", field.name, values[-1])
        extended[field.name] = (values + tail, field.types[-1])
    return extended


def check_trunk_layout(trunk: GGUFReader, trunk_size: int) -> int:
    """The data section is copied as one blob, so the writer's offsets must reproduce the file's."""
    offset = 0
    for t in trunk.tensors:
        relative = t.data_offset - trunk.data_offset
        if relative != offset:
            sys.exit(f"{t.name}: data at {relative}, expected {offset}: tensors are not laid out in header order")
        offset += pad(t.n_bytes, trunk.alignment)
    data_len = trunk_size - trunk.data_offset
    if data_len != offset:
        sys.exit(f"trunk data section is {data_len} bytes, tensor infos account for {offset}")
    return data_len


def collect_trunk_tensor_names(plan_dir: Path, prefix: str, n_split: int, first: GGUFReader) -> set[str]:
    names = {t.name for t in first.tensors}
    for idx in range(2, n_split + 1):
        path = plan_dir / split_name(prefix, idx, n_split)
        if not path.is_file():
            sys.exit(f"{path}: trunk shard missing")
        reader = open_gguf(path)
        split_no = require_int(reader, KEY_SPLIT_NO, path.name)
        if split_no != idx - 1:
            sys.exit(f"{path.name}: split.no is {split_no}, expected {idx - 1}")
        for t in reader.tensors:
            if t.name in names:
                sys.exit(f"{t.name}: duplicated inside the trunk itself")
            names.add(t.name)
    return names


def build_plan(trunk_path: Path, head_path: Path, out_prefix: str | None) -> tuple[Plan, GGUFReader, GGUFReader]:
    trunk_prefix, idx, n_split_old = parse_split_name(trunk_path)
    if idx != 1:
        sys.exit(f"{trunk_path.name}: pass the first shard (-00001-of-...)")
    trunk, head = open_gguf(trunk_path), open_gguf(head_path)

    arch = scalar(trunk, KEY_ARCH)
    if scalar(head, KEY_ARCH) != arch:
        sys.exit(f"architecture mismatch: trunk {arch!r}, head {scalar(head, KEY_ARCH)!r}")
    n_layer = require_int(trunk, f"{arch}.block_count", "trunk")
    if int(scalar(trunk, f"{arch}.nextn_predict_layers") or 0) != 0:
        sys.exit("trunk already declares nextn_predict_layers")
    n_nextn = int(scalar(head, f"{arch}.nextn_predict_layers") or 0)
    if n_nextn < 1:
        sys.exit("head declares no nextn_predict_layers: not an MTP export")
    head_block_count = require_int(head, f"{arch}.block_count", "head")
    if head_block_count != n_layer + n_nextn:
        sys.exit(f"head block_count {head_block_count} != trunk {n_layer} + nextn {n_nextn}")
    check_head_matches_trunk(trunk, head, arch)

    if require_int(trunk, KEY_SPLIT_NO, "trunk") != 0 or require_int(trunk, KEY_SPLIT_COUNT, "trunk") != n_split_old:
        sys.exit("trunk split.no/split.count disagree with its file name")
    n_tensors_old = require_int(trunk, KEY_SPLIT_TENSORS, "trunk")
    trunk_names = collect_trunk_tensor_names(trunk_path.parent, trunk_prefix, n_split_old, trunk)
    if len(trunk_names) != n_tensors_old:
        sys.exit(f"trunk declares {n_tensors_old} tensors, its shards hold {len(trunk_names)}")

    head_tensors, skipped = select_head_tensors(head, n_layer, n_nextn)
    clash = sorted(t.name for t in head_tensors if t.name in trunk_names)
    if clash:
        sys.exit(f"head tensors already in the trunk (fact 4): {clash[:5]}")

    trunk_size = trunk_path.stat().st_size
    plan = Plan(
        arch=arch, trunk_dir=trunk_path.parent, trunk_prefix=trunk_prefix,
        out_prefix=out_prefix or trunk_prefix, n_split_old=n_split_old, n_tensors_old=n_tensors_old,
        n_layer=n_layer, n_nextn=n_nextn,
        extended_arrays=per_layer_arrays(trunk, arch, n_layer, n_nextn),
        head_tensors=tuple(head_tensors), skipped_head_tensors=tuple(skipped),
        trunk_size=trunk_size, trunk_data_len=check_trunk_layout(trunk, trunk_size),
    )
    return plan, trunk, head


def report_plan(plan: Plan, out_dir: Path) -> None:
    arch = plan.arch
    logger.info("shard 1 -> %s", out_dir / plan.out_name(1))
    logger.info("  %s.block_count: %d -> %d", arch, plan.n_layer, plan.n_layer + plan.n_nextn)
    logger.info("  %s.nextn_predict_layers: (absent) -> %d", arch, plan.n_nextn)
    for key, (values, sub_type) in plan.extended_arrays.items():
        logger.info("  %s: %d -> %d elements (%s), tail %r", key, plan.n_layer, len(values), sub_type.name, values[-plan.n_nextn:])
    logger.info("  %s: %d -> %d", KEY_SPLIT_COUNT, plan.n_split_old, plan.n_split_new)
    logger.info("  %s: %d -> %d", KEY_SPLIT_TENSORS, plan.n_tensors_old, plan.n_tensors_new)
    logger.info("  tensor data: %.2f GiB copied byte for byte", plan.trunk_data_len / 2**30)
    for idx in range(2, plan.n_split_old + 1):
        logger.info("shard %d -> symlink %s -> %s", idx, plan.out_name(idx), plan.trunk_dir / plan.trunk_name(idx))
    logger.info("shard %d -> %s: %d tensors blk.%d.*, %.2f GiB", plan.n_split_new, plan.out_name(plan.n_split_new),
                len(plan.head_tensors), plan.n_layer, plan.head_bytes / 2**30)
    if plan.skipped_head_tensors:
        logger.info("  head tensors left out (the trunk has them): %s", ", ".join(plan.skipped_head_tensors))
    for t in plan.head_tensors:
        logger.debug("  %-40s %-8s %-24s %12d bytes", t.name, t.tensor_type.name, list(t.shape.tolist()), t.n_bytes)


def copy_fields(reader: GGUFReader, writer: GGUFWriter, overrides: dict[str, tuple[Any, GGUFValueType, GGUFValueType | None]],
                insert_after: dict[str, tuple[str, Any, GGUFValueType]]) -> None:
    for field in reader.fields.values():
        if field.name.startswith(READER_VIRTUAL_PREFIX) or field.name == KEY_ARCH:
            continue  # the writer's constructor already emitted general.architecture
        if field.name in overrides:
            value, vtype, sub_type = overrides[field.name]
        else:
            vtype = field.types[0]
            sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
            value = field.contents()
        writer.add_key_value(field.name, value, vtype, sub_type=sub_type)
        if field.name in insert_after:
            key, value, vtype = insert_after[field.name]
            writer.add_key_value(key, value, vtype)


def add_tensor_infos(writer: GGUFWriter, tensors: tuple[ReaderTensor, ...] | list[ReaderTensor]) -> None:
    for t in tensors:
        # data.shape is numpy order (byte shape for quantized types); the writer reverses it back
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, raw_dtype=t.tensor_type)


def drop_written_pages(fh) -> None:
    fh.flush()
    os.fdatasync(fh.fileno())
    os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def append_data_section(src_path: Path, src_offset: int, length: int, dst_path: Path, align: int) -> str:
    """Append [src_offset, src_offset + length) of src to dst, aligned; return the blob's blake2b."""
    digest = hashlib.blake2b()
    with open(src_path, "rb") as src, open(dst_path, "r+b") as dst:
        dst.seek(0, os.SEEK_END)
        end = dst.tell()
        dst.write(bytes(pad(end, align) - end))
        src.seek(src_offset)
        done = since_flush = 0
        while done < length:
            chunk = src.read(min(COPY_CHUNK_BYTES, length - done))
            if not chunk:
                raise RuntimeError(f"{src_path}: short read at {src_offset + done}")
            digest.update(chunk)
            dst.write(chunk)
            done += len(chunk)
            since_flush += len(chunk)
            if since_flush >= FLUSH_EVERY_BYTES:
                drop_written_pages(dst)
                since_flush = 0
                logger.info("  copied %.1f / %.1f GiB", done / 2**30, length / 2**30)
        drop_written_pages(dst)
    return digest.hexdigest()


def hash_file_from(path: Path, offset: int) -> str:
    digest = hashlib.blake2b()
    with open(path, "rb") as fh:
        fh.seek(offset)
        while chunk := fh.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
        os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def write_first_shard(trunk: GGUFReader, trunk_path: Path, plan: Plan, out_path: Path) -> str:
    tmp = out_path.with_name(out_path.name + ".tmp")
    arch = plan.arch
    overrides: dict[str, tuple[Any, GGUFValueType, GGUFValueType | None]] = {
        f"{arch}.block_count": (plan.n_layer + plan.n_nextn, GGUFValueType.UINT32, None),
        KEY_SPLIT_COUNT:       (plan.n_split_new, GGUFValueType.UINT16, None),
        KEY_SPLIT_TENSORS:     (plan.n_tensors_new, GGUFValueType.INT32, None),
    }
    for key, (values, sub_type) in plan.extended_arrays.items():
        overrides[key] = (values, GGUFValueType.ARRAY, sub_type)
    insert_after = {f"{arch}.block_count": (f"{arch}.nextn_predict_layers", plan.n_nextn, GGUFValueType.UINT32)}

    writer = GGUFWriter(tmp, arch=arch, endianess=trunk.endianess)
    writer.data_alignment = trunk.alignment
    copy_fields(trunk, writer, overrides, insert_after)
    add_tensor_infos(writer, trunk.tensors)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()

    logger.info("copying %.2f GiB of tensor data into %s", plan.trunk_data_len / 2**30, tmp.name)
    digest = append_data_section(trunk_path, trunk.data_offset, plan.trunk_data_len, tmp, trunk.alignment)
    os.replace(tmp, out_path)
    return digest


def write_head_shard(head: GGUFReader, plan: Plan, out_path: Path) -> None:
    tmp = out_path.with_name(out_path.name + ".tmp")
    writer = GGUFWriter(tmp, arch=plan.arch, endianess=head.endianess)
    writer.remove_key(KEY_ARCH)  # like llama-gguf-split, shards past the first carry only the split keys
    writer.data_alignment = head.alignment
    writer.add_uint16(KEY_SPLIT_NO, plan.n_split_new - 1)
    writer.add_uint16(KEY_SPLIT_COUNT, plan.n_split_new)
    writer.add_int32(KEY_SPLIT_TENSORS, plan.n_tensors_new)
    add_tensor_infos(writer, plan.head_tensors)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in plan.head_tensors:
        writer.write_tensor_data(t.data, tensor_endianess=head.endianess)
    writer.close()
    with open(tmp, "rb+") as fh:
        drop_written_pages(fh)
    os.replace(tmp, out_path)


def link_middle_shards(plan: Plan, out_dir: Path) -> None:
    for idx in range(2, plan.n_split_old + 1):
        target = plan.trunk_dir / plan.trunk_name(idx)
        link = out_dir / plan.out_name(idx)
        tmp = link.with_name(link.name + ".tmp")
        if tmp.is_symlink():
            tmp.unlink()
        os.symlink(target, tmp)
        os.replace(tmp, link)


def same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    av, bv = a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8)
    if av.size != bv.size:
        return False
    return all(np.array_equal(av[i:i + COMPARE_CHUNK], bv[i:i + COMPARE_CHUNK]) for i in range(0, av.size, COMPARE_CHUNK))


def same_tensor_info(a: ReaderTensor, b: ReaderTensor) -> bool:
    return (a.name == b.name and a.tensor_type == b.tensor_type and a.n_bytes == b.n_bytes
            and a.shape.tolist() == b.shape.tolist())


def verify_first_shard(plan: Plan, trunk: GGUFReader, path: Path, data_digest: str, check_data: bool) -> list[str]:
    errors: list[str] = []
    new = open_gguf(path)
    arch = plan.arch
    expected = {
        KEY_ARCH: arch,
        f"{arch}.block_count": plan.n_layer + plan.n_nextn,
        f"{arch}.nextn_predict_layers": plan.n_nextn,
        KEY_SPLIT_NO: 0, KEY_SPLIT_COUNT: plan.n_split_new, KEY_SPLIT_TENSORS: plan.n_tensors_new,
    }
    for key, want in expected.items():
        if scalar(new, key) != want:
            errors.append(f"{path.name}: {key} = {scalar(new, key)!r}, expected {want!r}")
    for key, (values, _) in plan.extended_arrays.items():
        if list(new.get_field(key).contents()) != values:
            errors.append(f"{path.name}: {key} differs from the plan")
    if len(new.fields) != len(trunk.fields) + 1:
        errors.append(f"{path.name}: {len(new.fields)} fields, expected {len(trunk.fields) + 1}")
    if new.alignment != trunk.alignment or len(new.tensors) != len(trunk.tensors):
        errors.append(f"{path.name}: alignment/tensor count differ from the trunk")
    for old_t, new_t in zip(trunk.tensors, new.tensors):
        if not same_tensor_info(old_t, new_t) or (new_t.data_offset - new.data_offset) != (old_t.data_offset - trunk.data_offset):
            errors.append(f"{path.name}: tensor info of {old_t.name} differs")
            break
    if path.stat().st_size - new.data_offset != plan.trunk_data_len:
        errors.append(f"{path.name}: data section length differs")
    if check_data and not errors:
        logger.info("re-reading %s to check the data section", path.name)
        if hash_file_from(path, new.data_offset) != data_digest:
            errors.append(f"{path.name}: data section hash differs from the bytes copied")
    return errors


def verify_other_shards(plan: Plan, out_dir: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    n_tensors = 0
    for idx in range(2, plan.n_split_old + 1):
        link = out_dir / plan.out_name(idx)
        if not link.is_symlink() or Path(os.readlink(link)) != plan.trunk_dir / plan.trunk_name(idx):
            errors.append(f"{link.name}: not a symlink to the trunk shard")
            continue
        reader = open_gguf(link)
        if scalar(reader, KEY_SPLIT_NO) != idx - 1:
            errors.append(f"{link.name}: split.no != {idx - 1}")
        n_tensors += len(reader.tensors)

    last = out_dir / plan.out_name(plan.n_split_new)
    reader = open_gguf(last)
    if scalar(reader, KEY_SPLIT_NO) != plan.n_split_new - 1:
        errors.append(f"{last.name}: split.no != {plan.n_split_new - 1}")
    if len(reader.tensors) != len(plan.head_tensors):
        errors.append(f"{last.name}: {len(reader.tensors)} tensors, expected {len(plan.head_tensors)}")
    else:
        for want, got in zip(plan.head_tensors, reader.tensors):
            if not same_tensor_info(want, got) or not same_bytes(want.data, got.data):
                errors.append(f"{last.name}: {want.name} differs from the head file")
    n_tensors += len(reader.tensors)
    return errors, n_tensors


def verify(plan: Plan, trunk: GGUFReader, out_dir: Path, data_digest: str, check_data: bool) -> None:
    errors = verify_first_shard(plan, trunk, out_dir / plan.out_name(1), data_digest, check_data)
    other_errors, n_other = verify_other_shards(plan, out_dir)
    errors += other_errors
    if not errors and len(trunk.tensors) + n_other != plan.n_tensors_new:
        errors.append(f"{len(trunk.tensors) + n_other} tensors across the shards, split.tensors.count says {plan.n_tensors_new}")
    if errors:
        for err in errors:
            logger.error(err)
        sys.exit("verification failed: the output must not be used")
    logger.info("verified: %d tensors in %d shards", plan.n_tensors_new, plan.n_split_new)


def check_output_dir(plan: Plan, out_dir: Path) -> None:
    refuse_tmpfs(out_dir)
    if out_dir.resolve() == plan.trunk_dir.resolve():
        sys.exit(f"{out_dir}: refusing to write next to the trunk")
    for idx in range(1, plan.n_split_new + 1):
        existing = out_dir / plan.out_name(idx)
        if existing.exists() or existing.is_symlink():
            sys.exit(f"{existing}: already exists; remove it yourself or choose another --out/--name")
    needed = plan.trunk_size + plan.head_bytes + FREE_SPACE_MARGIN
    probe = out_dir
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed:
        sys.exit(f"{out_dir}: need {needed / 2**30:.1f} GiB free, have {free / 2**30:.1f} GiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trunk", type=Path, required=True, help="first shard of the trunk GGUF (-00001-of-...)")
    parser.add_argument("--head", type=Path, required=True, help="quantized MTP head GGUF (convert_hf_to_gguf.py --mtp)")
    parser.add_argument("--out", type=Path, required=True, help="output directory, on a real disk")
    parser.add_argument("--name", help="file name prefix of the grafted set (default: the trunk's)")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="do not re-read the copied data section to compare its hash")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every head tensor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("gguf").setLevel(logging.WARNING)

    plan, trunk, head = build_plan(args.trunk.resolve(), args.head.resolve(), args.name)
    out_dir = args.out.expanduser().resolve()
    report_plan(plan, out_dir)
    if args.dry_run:
        logger.info("dry run: nothing written")
        return 0
    check_output_dir(plan, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    link_middle_shards(plan, out_dir)
    write_head_shard(head, plan, out_dir / plan.out_name(plan.n_split_new))
    logger.info("wrote %s", plan.out_name(plan.n_split_new))
    data_digest = write_first_shard(trunk, args.trunk.resolve(), plan, out_dir / plan.out_name(1))
    logger.info("wrote %s", plan.out_name(1))
    verify(plan, trunk, out_dir, data_digest, check_data=not args.skip_data_check)
    logger.info("done: load with -m %s", out_dir / plan.out_name(1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
