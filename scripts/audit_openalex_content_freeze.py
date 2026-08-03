#!/usr/bin/env python3
"""Create a resumable per-file SHA-256 baseline for the current OpenAlex root.

The full command is a sequential CPU/I/O component.  It creates the first
content baseline for the bytes read during this run; it cannot prove that the
same bytes existed during an earlier footer-only inventory or component run.
Any ``--max-files`` invocation is a pilot and can never receive the full-pass
status.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Iterator, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/openalex_g1_content_freeze_v0.yaml"
CONFIG_SCHEMA_VERSION = "openalex_g1_content_freeze_config.v1"
PLAN_SCHEMA_VERSION = "openalex_g1_content_freeze_plan.v1"
CHECKPOINT_SCHEMA_VERSION = "openalex_g1_content_freeze_checkpoint.v1"
STATE_SCHEMA_VERSION = "openalex_g1_content_freeze_state.v1"
REPORT_SCHEMA_VERSION = "openalex_g1_content_freeze_report.v1"
OUTPUTS_SCHEMA_VERSION = "openalex_g1_content_freeze_outputs.v1"
FULL_SUCCESS_STATUS = "PASS_CURRENT_CONTENT_BASELINE_COMPONENT"
PILOT_STATUS = "PILOT_CONTENT_FREEZE_ONLY"
PAUSED_STATUS = "PAUSED_CURRENT_CONTENT_FREEZE"
FINALIZING_STATUS = "FINALIZING_CURRENT_CONTENT_FREEZE"
EXPECTED_STAT_FIELDS = ("device", "inode", "mode", "bytes", "mtime_ns", "ctime_ns")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_CLAIM_BOUNDARY = (
    "A full pass records the first per-file SHA-256 baseline for bytes read from "
    "the currently declared source after exact parent-inventory and pre/post stat "
    "validation. It does not prove that these bytes equal the earlier footer-only "
    "inventory-era bytes, does not retroactively bind prior component runs, does "
    "not validate raw entity semantics, establish G-1 or G0, run empirical G1, or "
    "establish a scientific effect. A --max-files run is "
    "PILOT_CONTENT_FREEZE_ONLY and cannot be promoted to the full component status."
)
EXPECTED_MUTATION_LIMITATION = (
    "Resume reuses a checkpoint only after its schema, task, parent source "
    "identity, checkpoint digest, and current device, inode, mode, size, mtime, "
    "and ctime match. It does not rehash completed files. A privileged or "
    "storage-layer mutation that restores every checked stat field, or a mutation "
    "after the final stat snapshot, can evade this check; therefore the source "
    "must remain operationally immutable, and this first baseline cannot "
    "establish historical byte identity."
)
EXPECTED_CHECKPOINT_TRUST_LIMITATION = (
    "Checkpoint SHA-256 is an unkeyed self-digest for accidental-corruption "
    "detection, not an authentication signature. A writer who can modify the "
    "output directory can replace a checkpoint hash and recompute its digest; "
    "resume cannot detect that without rehashing the source or an external "
    "trusted or WORM anchor. The output directory must therefore remain "
    "operator-controlled and immutable to untrusted writers."
)
EXPECTED_FULL_GATES = {
    "current_content_baseline_component": "PASS",
    "historical_inventory_era_byte_identity": "UNVERIFIED",
    "prior_component_run_content_identity": "UNVERIFIED",
    "raw_source_id_validity": "BLOCKED",
    "w_zero_entity_validity": "UNRESOLVED",
    "g_minus_1_data_readiness": "PENDING",
    "g0_measurement_freeze": "PENDING",
    "empirical_g1_signature": "NOT_RUN",
    "pass_g1": False,
    "scientific_effect": "NOT_ESTABLISHED",
}
ALLOWED_TOP_LEVEL = {
    "audit.lock",
    "audit_plan.json",
    "run_state.json",
    "checkpoints",
    "content_freeze_audit.json",
    "content_freeze_audit.md",
    "content_freeze_audit.outputs.json",
}


class ContentFreezeError(RuntimeError):
    """Raised when the frozen current-content contract cannot be honored."""


@dataclass(frozen=True)
class FileTask:
    index: int
    table_name: str
    relative_path: str
    bytes: int
    mtime_ns: int
    source_stat: dict[str, int]


@dataclass(frozen=True)
class InputSnapshot:
    value: dict[str, Any]
    sha256: str
    source_stat: dict[str, int]


@dataclass(frozen=True)
class ValidatedCheckpoint:
    value: dict[str, Any]
    file_sha256: str
    file_bytes: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_absolute_regular_no_follow(path: Path) -> int:
    require(path.is_absolute(), "regular-file open requires an absolute path")
    parent_descriptor = _open_absolute_directory_no_follow(path.parent)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        return os.open(path.name, flags, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)


def read_regular_file_bytes(
    path: Path,
    *,
    field: str,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> tuple[bytes, str, dict[str, int]]:
    """Read and hash one immutable descriptor, rejecting path-component links."""

    require(chunk_bytes > 0, "input read chunk must be positive")
    try:
        descriptor = _open_absolute_regular_no_follow(path)
    except OSError as exc:
        detail = os.strerror(exc.errno) if exc.errno is not None else type(exc).__name__
        raise ContentFreezeError(f"cannot open {field}: {detail}") from exc
    try:
        before_info = os.fstat(descriptor)
        require(stat.S_ISREG(before_info.st_mode), f"{field} is not a regular file")
        before = _stat_identity_from_result(before_info)
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, chunk_bytes)
            if not block:
                break
            blocks.append(block)
            digest.update(block)
            total += len(block)
        after = _stat_identity_from_result(os.fstat(descriptor))
    except OSError as exc:
        detail = os.strerror(exc.errno) if exc.errno is not None else type(exc).__name__
        raise ContentFreezeError(f"cannot read {field}: {detail}") from exc
    finally:
        os.close(descriptor)
    require(before == after, f"{field} changed while being read")
    require(total == before["bytes"], f"{field} byte count changed while being read")
    return b"".join(blocks), digest.hexdigest(), before


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContentFreezeError(message)


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be a mapping")
    return value


def require_list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be a list")
    return value


def require_int(value: Any, field: str) -> int:
    require(type(value) is int, f"{field} must be an integer")
    return value


def require_sha256(value: Any, field: str) -> str:
    text = str(value)
    require(SHA256_PATTERN.fullmatch(text) is not None, f"{field} is not SHA-256")
    return text


def safe_relative_path(value: Any, field: str) -> Path:
    path = Path(str(value))
    require(not path.is_absolute(), f"{field} must be relative")
    require(path.parts and path != Path("."), f"{field} may not be empty")
    require(".." not in path.parts, f"{field} may not contain '..'")
    return path


def path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _assert_absolute_components_without_symlinks(path: Path, *, field: str) -> None:
    require(path.is_absolute(), f"{field} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            continue
        try:
            info = current.lstat()
        except OSError as exc:
            raise ContentFreezeError(
                f"cannot stat {field} component: {type(exc).__name__}"
            ) from exc
        require(not stat.S_ISLNK(info.st_mode), f"{field} component is a symlink")


def validate_source_root(path: Path, *, expected_id: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    _assert_absolute_components_without_symlinks(absolute, field="source root")
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise ContentFreezeError(f"cannot stat source root: {type(exc).__name__}") from exc
    require(stat.S_ISDIR(info.st_mode), "source root is not a directory")
    require(absolute.name == expected_id, "source root ID mismatch")
    return absolute


def validate_output_location(source_root: Path, output_dir: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(output_dir.expanduser())))
    _assert_absolute_components_without_symlinks(absolute, field="output path")
    require(absolute != source_root, "output path equals source root")
    require(
        not absolute.is_relative_to(source_root),
        "output path may not be inside source root",
    )
    require(
        not source_root.is_relative_to(absolute),
        "output path may not contain source root",
    )
    return absolute


def _stat_identity_from_result(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(stat.S_IMODE(info.st_mode)),
        "bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def regular_identity(path: Path, *, field: str) -> dict[str, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ContentFreezeError(f"cannot stat {field}: {type(exc).__name__}") from exc
    require(not stat.S_ISLNK(info.st_mode), f"{field} may not be a symlink")
    require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
    return _stat_identity_from_result(info)


def secure_join_regular(root: Path, relative: Path, *, field: str) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise ContentFreezeError(
                f"cannot stat {field} {relative.as_posix()}: {type(exc).__name__}"
            ) from exc
        require(
            not stat.S_ISLNK(info.st_mode),
            f"{field} path component is a symlink: {relative.as_posix()}",
        )
    regular_identity(current, field=f"{field} {relative.as_posix()}")
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ContentFreezeError(
            f"{field} escapes source root: {relative.as_posix()}"
        ) from exc
    return current


def read_json_regular(path: Path, *, field: str) -> dict[str, Any]:
    return read_json_snapshot(path, field=field).value


def read_json_snapshot(path: Path, *, field: str) -> InputSnapshot:
    try:
        raw, digest, identity = read_regular_file_bytes(path, field=field)
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentFreezeError(f"cannot read {field}: {type(exc).__name__}") from exc
    return InputSnapshot(
        value=require_mapping(value, field),
        sha256=digest,
        source_stat=identity,
    )


def read_config(path: Path) -> dict[str, Any]:
    return read_config_snapshot(path).value


def read_config_snapshot(path: Path) -> InputSnapshot:
    try:
        raw_bytes, digest, identity = read_regular_file_bytes(path, field="config")
        raw = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ContentFreezeError(f"cannot read config: {type(exc).__name__}") from exc
    config = require_mapping(raw, "config")
    require(config.get("schema_version") == CONFIG_SCHEMA_VERSION, "bad config schema")
    execution = require_mapping(config.get("execution"), "execution")
    require(execution.get("hash_algorithm") == "sha256", "hash algorithm drift")
    require(execution.get("workers") == 1, "content freeze must use one worker")
    require(
        tuple(execution.get("checkpoint_current_stat_fields", ()))
        == EXPECTED_STAT_FIELDS,
        "checkpoint stat fields drift",
    )
    require(
        execution.get("checkpoint_reuse")
        == "validate_schema_task_parent_and_current_stat_without_content_rehash",
        "checkpoint reuse contract drift",
    )
    require(
        execution.get("checkpoint_integrity")
        == "unkeyed_sha256_self_digest_accidental_corruption_only",
        "checkpoint integrity contract drift",
    )
    require(
        execution.get("trusted_output_directory_required") is True,
        "trusted output-directory contract drift",
    )
    require(require_int(execution.get("read_chunk_bytes"), "read chunk") > 0, "bad read chunk")
    require(
        require_int(execution.get("progress_every_files"), "progress every files")
        > 0,
        "bad file progress interval",
    )
    require(
        require_int(
            execution.get("progress_every_seconds"), "progress every seconds"
        )
        > 0,
        "bad time progress interval",
    )
    success = require_mapping(config.get("success_contract"), "success contract")
    require(success.get("full_success_status") == FULL_SUCCESS_STATUS, "full status drift")
    require(success.get("pilot_status") == PILOT_STATUS, "pilot status drift")
    require(success.get("paused_status") == PAUSED_STATUS, "paused status drift")
    require(
        require_mapping(
            success.get("required_gate_status_after_full_success"), "full gates"
        )
        == EXPECTED_FULL_GATES,
        "full gate contract drift",
    )
    require(config.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY, "claim boundary drift")
    require(
        config.get("same_size_restored_mtime_limitation")
        == EXPECTED_MUTATION_LIMITATION,
        "mutation limitation drift",
    )
    require(
        config.get("checkpoint_trust_limitation")
        == EXPECTED_CHECKPOINT_TRUST_LIMITATION,
        "checkpoint trust limitation drift",
    )
    return InputSnapshot(value=config, sha256=digest, source_stat=identity)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_absolute_components_without_symlinks(path.parent, field="output parent")
    if path.exists() or path.is_symlink():
        regular_identity(path, field="managed output")
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    require(not partial.exists() and not partial.is_symlink(), "temporary output exists")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(partial, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
        _fsync_parent(path)
    finally:
        partial.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_absolute_components_without_symlinks(path.parent, field="output parent")
    if path.exists() or path.is_symlink():
        regular_identity(path, field="managed output")
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    require(not partial.exists() and not partial.is_symlink(), "temporary output exists")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(partial, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
        _fsync_parent(path)
    finally:
        partial.unlink(missing_ok=True)


def _validate_trusted_output_chain(output: Path) -> dict[str, Any]:
    """Verify that another unprivileged user cannot replace managed paths."""

    require(output.is_absolute(), "output trust check requires an absolute path")
    euid = os.geteuid()
    current = Path(output.anchor)
    checked = 0
    for part in output.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise ContentFreezeError(
                f"cannot stat output trust-chain component: {type(exc).__name__}"
            ) from exc
        require(
            stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            "output trust-chain component is not a regular directory",
        )
        require(
            info.st_uid in {0, euid},
            "output trust-chain component has an untrusted owner",
        )
        shared_write = bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
        if current == output:
            require(info.st_uid == euid, "output directory is not operator-owned")
            require(not shared_write, "output directory is group/world writable")
        elif shared_write:
            require(
                bool(info.st_mode & stat.S_ISVTX),
                "writable output ancestor lacks sticky protection",
            )
        checked += 1
    return {
        "status": "VERIFIED_OWNER_AND_ANCESTOR_MODE_CHAIN",
        "operator_uid": euid,
        "components_checked": checked,
        "output_group_or_world_writable": False,
        "path_redacted": True,
    }


@contextmanager
def output_lock(source_root: Path, output_dir: Path) -> Iterator[Path]:
    output = validate_output_location(source_root, output_dir)
    missing: list[Path] = []
    current = output
    while not current.exists() and not current.is_symlink():
        missing.append(current)
        current = current.parent
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    for created in reversed(missing):
        info = created.lstat()
        require(
            stat.S_ISDIR(info.st_mode)
            and not stat.S_ISLNK(info.st_mode)
            and info.st_uid == os.geteuid(),
            "new output path component is not operator-owned",
        )
        os.chmod(created, 0o700, follow_symlinks=False)
    _assert_absolute_components_without_symlinks(output, field="output directory")
    require(output.is_dir() and not output.is_symlink(), "output is not a directory")
    _validate_trusted_output_chain(output)
    lock_path = output / "audit.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    lock_info = os.fstat(descriptor)
    if not stat.S_ISREG(lock_info.st_mode):
        os.close(descriptor)
        raise ContentFreezeError("output lock is not a regular file")
    handle = os.fdopen(descriptor, "r+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContentFreezeError("another content freeze holds the output lock") from exc
        yield output
    finally:
        handle.close()


def snapshot_fingerprint(
    rows: Iterable[tuple[str, int, int]], *, include_mtime: bool
) -> str:
    digest = hashlib.sha256()
    for relative_path, size, mtime_ns in sorted(rows):
        parts = [relative_path, str(size)]
        if include_mtime:
            parts.append(str(mtime_ns))
        digest.update("\0".join(parts).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_snapshot(tasks: Iterable[FileTask]) -> dict[str, Any]:
    ordered = sorted(tasks, key=lambda item: item.relative_path)
    rows = [(task.relative_path, task.bytes, task.mtime_ns) for task in ordered]
    stat_rows = [
        {"relative_path": task.relative_path, "source_stat": task.source_stat}
        for task in ordered
    ]
    return {
        "file_count": len(ordered),
        "bytes": sum(task.bytes for task in ordered),
        "inventory_fingerprint_sha256": snapshot_fingerprint(rows, include_mtime=False),
        "stability_fingerprint_sha256": snapshot_fingerprint(rows, include_mtime=True),
        "rich_stat_tree_sha256": sha256_bytes(canonical_json_bytes(stat_rows)),
    }


def _validate_source_tree_entries(
    root: Path,
    expected_by_table: Mapping[str, set[str]],
) -> None:
    root_entries = sorted(root.iterdir(), key=lambda item: item.name)
    observed_dirs: set[str] = set()
    root_files: list[str] = []
    for entry in root_entries:
        info = entry.lstat()
        require(not stat.S_ISLNK(info.st_mode), "source root entry is a symlink")
        if stat.S_ISDIR(info.st_mode):
            observed_dirs.add(entry.name)
        else:
            root_files.append(entry.name)
    require(not root_files, "unexpected source root files")
    require(observed_dirs == set(expected_by_table), "source table directory set differs")
    for table_name, expected_names in expected_by_table.items():
        table_dir = root / table_name
        info = table_dir.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), "invalid table directory")
        observed_names: set[str] = set()
        for entry in table_dir.iterdir():
            entry_info = entry.lstat()
            require(not stat.S_ISLNK(entry_info.st_mode), "source table entry is a symlink")
            require(stat.S_ISREG(entry_info.st_mode), "source table entry is not a file")
            observed_names.add(entry.name)
        require(observed_names == expected_names, f"source table entries differ: {table_name}")


def validate_parent_inventory(
    *,
    config: Mapping[str, Any],
    manifest_path: Path,
    source_root: Path,
) -> tuple[dict[str, Any], list[FileTask], dict[str, Any], InputSnapshot]:
    parent = require_mapping(config.get("parent_inventory"), "parent inventory")
    expected_manifest_sha = require_sha256(parent.get("manifest_sha256"), "manifest sha256")
    manifest_snapshot = read_json_snapshot(manifest_path, field="parent manifest")
    require(
        manifest_snapshot.sha256 == expected_manifest_sha,
        "parent manifest SHA-256 mismatch",
    )
    manifest = manifest_snapshot.value
    require(manifest.get("schema_version") == parent.get("required_schema_version"), "manifest schema mismatch")
    require(manifest.get("manifest_id") == parent.get("required_manifest_id"), "manifest ID mismatch")
    require(manifest.get("status") == parent.get("required_status"), "manifest status mismatch")
    require(manifest.get("pass_metadata") is False, "parent already claims metadata pass")
    require(manifest.get("pass_g1") is False, "parent unexpectedly claims G1")
    execution = require_mapping(manifest.get("execution"), "manifest execution")
    require(execution.get("content_sha256") is parent.get("required_content_sha256"), "parent hash state mismatch")
    require(execution.get("path_redacted") is True, "parent path redaction differs")
    root_record = require_mapping(manifest.get("root"), "manifest root")
    scalar_expectations = {
        "id": parent.get("source_root_id"),
        "table_count": parent.get("expected_table_count"),
        "parquet_file_count": parent.get("expected_parquet_file_count"),
        "bytes": parent.get("expected_bytes"),
        "inventory_fingerprint_sha256": parent.get("inventory_fingerprint_sha256"),
        "stability_fingerprint_sha256": parent.get("stability_fingerprint_sha256"),
    }
    for field, expected in scalar_expectations.items():
        require(root_record.get(field) == expected, f"parent root {field} mismatch")

    raw_tables = require_list(manifest.get("tables"), "manifest tables")
    require(len(raw_tables) == parent.get("expected_table_count"), "manifest table count mismatch")
    records: list[tuple[str, str, int, int]] = []
    expected_by_table: dict[str, set[str]] = {}
    for raw_table in raw_tables:
        table = require_mapping(raw_table, "manifest table")
        table_name = str(table.get("name", ""))
        require(table_name and table_name not in expected_by_table, "duplicate manifest table")
        files = require_list(table.get("files"), f"manifest files {table_name}")
        expected_by_table[table_name] = set()
        require(len(files) == table.get("file_count"), f"manifest file count mismatch: {table_name}")
        for raw_file in files:
            record = require_mapping(raw_file, "manifest file")
            relative = safe_relative_path(record.get("relative_path"), "source relative path")
            require(relative.parts[0] == table_name and len(relative.parts) == 2, "source path/table mismatch")
            require(relative.suffix == ".parquet", "source file is not Parquet")
            require(record.get("content_sha256") is None, "parent already contains content hashes")
            size = require_int(record.get("bytes"), "source bytes")
            mtime_ns = require_int(record.get("mtime_ns"), "source mtime")
            require(size > 0 and mtime_ns > 0, "invalid source identity")
            require(relative.name not in expected_by_table[table_name], "duplicate source path")
            expected_by_table[table_name].add(relative.name)
            records.append((table_name, relative.as_posix(), size, mtime_ns))
    require(len(records) == parent.get("expected_parquet_file_count"), "source file total mismatch")
    require(sum(item[2] for item in records) == parent.get("expected_bytes"), "source byte total mismatch")
    require(len({item[1] for item in records}) == len(records), "duplicate source relative path")
    physical_rows = [(item[1], item[2], item[3]) for item in records]
    require(
        snapshot_fingerprint(physical_rows, include_mtime=False)
        == parent.get("inventory_fingerprint_sha256"),
        "parent path-size fingerprint mismatch",
    )
    require(
        snapshot_fingerprint(physical_rows, include_mtime=True)
        == parent.get("stability_fingerprint_sha256"),
        "parent path-size-mtime fingerprint mismatch",
    )
    _validate_source_tree_entries(source_root, expected_by_table)

    tasks: list[FileTask] = []
    for index, (table_name, relative_text, size, mtime_ns) in enumerate(
        sorted(records, key=lambda item: item[1])
    ):
        relative = Path(relative_text)
        secure_join_regular(source_root, relative, field="source file")
        identity = anchored_regular_identity(
            source_root, relative, field=f"source file {relative_text}"
        )
        require(identity["bytes"] == size, f"source size differs: {relative_text}")
        require(identity["mtime_ns"] == mtime_ns, f"source mtime differs: {relative_text}")
        tasks.append(
            FileTask(
                index=index,
                table_name=table_name,
                relative_path=relative_text,
                bytes=size,
                mtime_ns=mtime_ns,
                source_stat=identity,
            )
        )
    snapshot = source_snapshot(tasks)
    require(snapshot["inventory_fingerprint_sha256"] == parent.get("inventory_fingerprint_sha256"), "current inventory fingerprint differs")
    require(snapshot["stability_fingerprint_sha256"] == parent.get("stability_fingerprint_sha256"), "current stability fingerprint differs")
    return manifest, tasks, snapshot, manifest_snapshot


def build_plan(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    manifest: Mapping[str, Any],
    all_tasks: list[FileTask],
    snapshot: Mapping[str, Any],
    max_files: int | None,
    tool_sha256: str,
) -> dict[str, Any]:
    require(max_files is None or max_files > 0, "--max-files must be positive")
    selected = all_tasks if max_files is None else all_tasks[:max_files]
    require(selected, "content-freeze plan selects no files")
    execution = require_mapping(config.get("execution"), "execution")
    core = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "label": config.get("label"),
        "full_scope": max_files is None,
        "max_files": max_files,
        "source": {
            "root_id": manifest["root"]["id"],
            "root_env": require_mapping(config.get("source"), "source").get("root_env"),
            "path_redacted": True,
        },
        "parent_inventory": {
            "manifest_label": path_label(manifest_path),
            "manifest_sha256": require_sha256(
                manifest_sha256, "carried parent manifest SHA-256"
            ),
            "inventory_fingerprint_sha256": snapshot["inventory_fingerprint_sha256"],
            "stability_fingerprint_sha256": snapshot["stability_fingerprint_sha256"],
        },
        "source_snapshot": dict(snapshot),
        "execution": {
            "hash_algorithm": "sha256",
            "read_chunk_bytes": require_int(execution.get("read_chunk_bytes"), "read chunk"),
            "workers": 1,
            "checkpoint_reuse": execution.get("checkpoint_reuse"),
            "checkpoint_content_rehash": False,
            "checkpoint_integrity": execution.get("checkpoint_integrity"),
            "trusted_output_directory_required": True,
            "output_trust_precondition": (
                "VERIFIED_OWNER_AND_ANCESTOR_MODE_CHAIN"
            ),
        },
        "provenance": {
            "tool": "scripts/audit_openalex_content_freeze.py",
            "tool_sha256": require_sha256(tool_sha256, "carried tool SHA-256"),
            "tool_attestation_scope": (
                "CURRENT_TOOL_FILE_SNAPSHOT_NOT_LOADED_CODE_ATTESTATION"
            ),
            "config": path_label(config_path),
            "config_sha256": require_sha256(
                config_sha256, "carried config SHA-256"
            ),
            "input_snapshot_method": (
                "config_and_manifest_single_descriptor_bytes_hash_and_parse;"
                "tool_single_descriptor_bytes_hash"
            ),
        },
        "declared_file_count": len(all_tasks),
        "declared_bytes": sum(task.bytes for task in all_tasks),
        "selected_file_count": len(selected),
        "selected_bytes": sum(task.bytes for task in selected),
        "tasks": [asdict(task) for task in selected],
        "claim_boundary": config.get("claim_boundary"),
        "same_size_restored_mtime_limitation": config.get(
            "same_size_restored_mtime_limitation"
        ),
        "checkpoint_trust_limitation": config.get("checkpoint_trust_limitation"),
    }
    task_fingerprint = sha256_bytes(canonical_json_bytes(core))
    plan = {**core, "task_fingerprint_sha256": task_fingerprint}
    plan["plan_sha256"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


def task_from_mapping(value: Mapping[str, Any]) -> FileTask:
    identity = require_mapping(value.get("source_stat"), "task source stat")
    require(set(identity) == set(EXPECTED_STAT_FIELDS), "task stat fields differ")
    return FileTask(
        index=require_int(value.get("index"), "task index"),
        table_name=str(value.get("table_name", "")),
        relative_path=str(value.get("relative_path", "")),
        bytes=require_int(value.get("bytes"), "task bytes"),
        mtime_ns=require_int(value.get("mtime_ns"), "task mtime"),
        source_stat={field: require_int(identity[field], field) for field in EXPECTED_STAT_FIELDS},
    )


def checkpoint_path(output_dir: Path, task: FileTask) -> Path:
    token = sha256_bytes(task.relative_path.encode("utf-8"))
    return output_dir / "checkpoints" / token[:2] / f"{token}.json"


def _open_absolute_directory_no_follow(path: Path) -> int:
    require(path.is_absolute(), "directory open requires an absolute path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_source_no_follow(root: Path, relative: Path) -> int:
    directory_descriptor = _open_absolute_directory_no_follow(root)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        for part in relative.parts[:-1]:
            next_descriptor = os.open(
                part, directory_flags, dir_fd=directory_descriptor
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        return os.open(relative.name, file_flags, dir_fd=directory_descriptor)
    finally:
        os.close(directory_descriptor)


def anchored_regular_identity(root: Path, relative: Path, *, field: str) -> dict[str, int]:
    try:
        descriptor = _open_source_no_follow(root, relative)
    except OSError as exc:
        detail = os.strerror(exc.errno) if exc.errno is not None else type(exc).__name__
        raise ContentFreezeError(
            f"cannot open {field} {relative.as_posix()}: {detail}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
        return _stat_identity_from_result(info)
    finally:
        os.close(descriptor)


def hash_task(
    task: FileTask,
    *,
    source_root: Path,
    task_fingerprint_sha256: str,
    plan_sha256: str,
    chunk_bytes: int,
    after_hash_hook: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    relative = safe_relative_path(task.relative_path, "task relative path")
    path = secure_join_regular(source_root, relative, field="source file")
    before = anchored_regular_identity(
        source_root, relative, field=f"source file {task.relative_path}"
    )
    require(before == task.source_stat, f"source stat differs before hash: {task.relative_path}")
    started_at = utc_now()
    started = time.monotonic()
    digest = hashlib.sha256()
    bytes_hashed = 0
    descriptor: int | None = None
    try:
        descriptor = _open_source_no_follow(source_root, relative)
        opened = _stat_identity_from_result(os.fstat(descriptor))
        require(opened == before, f"opened source identity differs: {task.relative_path}")
        while True:
            block = os.read(descriptor, chunk_bytes)
            if not block:
                break
            digest.update(block)
            bytes_hashed += len(block)
        descriptor_after = _stat_identity_from_result(os.fstat(descriptor))
    except OSError as exc:
        detail = os.strerror(exc.errno) if exc.errno is not None else type(exc).__name__
        raise ContentFreezeError(
            f"source read failed: {task.relative_path}: {detail}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if after_hash_hook is not None:
        after_hash_hook(path)
    after = anchored_regular_identity(
        source_root, relative, field=f"source file {task.relative_path}"
    )
    require(bytes_hashed == task.bytes, f"hashed byte count differs: {task.relative_path}")
    require(descriptor_after == before, f"source changed through open descriptor: {task.relative_path}")
    require(after == before, f"source changed during hash: {task.relative_path}")
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task_fingerprint_sha256": task_fingerprint_sha256,
        "plan_sha256": plan_sha256,
        "task_index": task.index,
        "table_name": task.table_name,
        "relative_path": task.relative_path,
        "expected_parent_identity": {"bytes": task.bytes, "mtime_ns": task.mtime_ns},
        "source_identity_before": before,
        "source_identity_after": after,
        "hash_algorithm": "sha256",
        "content_sha256": digest.hexdigest(),
        "bytes_hashed": bytes_hashed,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "path_redacted": True,
    }
    checkpoint["checkpoint_sha256"] = sha256_bytes(canonical_json_bytes(checkpoint))
    return checkpoint


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    task: FileTask,
    *,
    source_root: Path,
    task_fingerprint_sha256: str,
    plan_sha256: str,
) -> dict[str, Any]:
    require(checkpoint.get("schema_version") == CHECKPOINT_SCHEMA_VERSION, "checkpoint schema mismatch")
    require(checkpoint.get("task_fingerprint_sha256") == task_fingerprint_sha256, "checkpoint task fingerprint mismatch")
    require(checkpoint.get("plan_sha256") == plan_sha256, "checkpoint plan mismatch")
    require(checkpoint.get("task_index") == task.index, "checkpoint task index mismatch")
    require(checkpoint.get("table_name") == task.table_name, "checkpoint table mismatch")
    require(checkpoint.get("relative_path") == task.relative_path, "checkpoint path mismatch")
    require(checkpoint.get("hash_algorithm") == "sha256", "checkpoint algorithm mismatch")
    require(checkpoint.get("path_redacted") is True, "checkpoint path redaction differs")
    expected_parent = {"bytes": task.bytes, "mtime_ns": task.mtime_ns}
    require(checkpoint.get("expected_parent_identity") == expected_parent, "checkpoint parent identity mismatch")
    require(checkpoint.get("source_identity_before") == task.source_stat, "checkpoint source identity mismatch")
    require(checkpoint.get("source_identity_after") == task.source_stat, "checkpoint post-stat mismatch")
    require(checkpoint.get("bytes_hashed") == task.bytes, "checkpoint byte count mismatch")
    require_sha256(checkpoint.get("content_sha256"), "checkpoint content hash")
    claimed = require_sha256(checkpoint.get("checkpoint_sha256"), "checkpoint digest")
    core = dict(checkpoint)
    del core["checkpoint_sha256"]
    require(claimed == sha256_bytes(canonical_json_bytes(core)), "checkpoint digest mismatch")
    relative = safe_relative_path(task.relative_path, "checkpoint source path")
    secure_join_regular(source_root, relative, field="checkpoint source file")
    current = anchored_regular_identity(
        source_root,
        relative,
        field=f"checkpoint source file {task.relative_path}",
    )
    require(current == task.source_stat, f"source stat changed since checkpoint: {task.relative_path}")
    return dict(checkpoint)


def _load_validated_checkpoints(
    output_dir: Path,
    tasks: list[FileTask],
    *,
    source_root: Path,
    task_fingerprint_sha256: str,
    plan_sha256: str,
) -> list[ValidatedCheckpoint]:
    """Re-read every checkpoint from disk and validate the exact bytes read."""

    _validate_checkpoint_entries(output_dir, tasks, require_complete=True)
    validated: list[ValidatedCheckpoint] = []
    for task in tasks:
        snapshot = read_json_snapshot(
            checkpoint_path(output_dir, task),
            field=f"final checkpoint {task.index}",
        )
        value = validate_checkpoint(
            snapshot.value,
            task,
            source_root=source_root,
            task_fingerprint_sha256=task_fingerprint_sha256,
            plan_sha256=plan_sha256,
        )
        validated.append(
            ValidatedCheckpoint(
                value=value,
                file_sha256=snapshot.sha256,
                file_bytes=snapshot.source_stat["bytes"],
            )
        )
    return validated


def _checkpoint_tree(
    output_dir: Path,
    tasks: Iterable[FileTask],
    checkpoints: Iterable[ValidatedCheckpoint],
) -> dict[str, Any]:
    commitments: list[dict[str, Any]] = []
    for task, checkpoint in zip(tasks, checkpoints, strict=True):
        path = checkpoint_path(output_dir, task)
        commitments.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": checkpoint.file_bytes,
                "sha256": checkpoint.file_sha256,
                "source_relative_path": task.relative_path,
                "content_sha256": checkpoint.value["content_sha256"],
                "source_bytes": task.bytes,
            }
        )
    return {
        "files": len(commitments),
        "bytes": sum(item["bytes"] for item in commitments),
        "tree_sha256": sha256_bytes(canonical_json_bytes(commitments)),
        "commitments": commitments,
    }


def _content_tree(
    tasks: Iterable[FileTask], checkpoints: Iterable[ValidatedCheckpoint]
) -> dict[str, Any]:
    commitments = [
        {
            "relative_path": task.relative_path,
            "bytes": task.bytes,
            "sha256": checkpoint.value["content_sha256"],
        }
        for task, checkpoint in zip(tasks, checkpoints, strict=True)
    ]
    return {
        "files": len(commitments),
        "bytes": sum(item["bytes"] for item in commitments),
        "tree_sha256": sha256_bytes(canonical_json_bytes(commitments)),
    }


def git_metadata() -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True
        )

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _state_payload(
    *,
    plan: Mapping[str, Any],
    status: str,
    completed_files: int,
    completed_bytes: int,
    resumed_files: int,
    latest_relative_path: str | None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "status": status,
        "completed_files": completed_files,
        "completed_bytes": completed_bytes,
        "total_files": plan["selected_file_count"],
        "total_bytes": plan["selected_bytes"],
        "resumed_files": resumed_files,
        "latest_relative_path": latest_relative_path,
        "updated_at_utc": utc_now(),
        "path_redacted": True,
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__[:80],
            "message": (
                "audit paused; exception detail omitted to preserve path redaction"
            ),
            "detail_omitted": True,
        }
    return payload


def _validate_output_entries(output_dir: Path, *, resume: bool) -> None:
    entries = {path.name: path for path in output_dir.iterdir()}
    unexpected = sorted(set(entries) - ALLOWED_TOP_LEVEL)
    require(not unexpected, f"unexpected output entries: {unexpected}")
    for name, path in entries.items():
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"output entry is a symlink: {name}")
        if name == "checkpoints":
            require(stat.S_ISDIR(info.st_mode), "checkpoints output is not a directory")
        else:
            require(stat.S_ISREG(info.st_mode), f"output entry is not a file: {name}")
    managed = set(entries) - {"audit.lock"}
    if resume:
        require("audit_plan.json" in entries, "resume output lacks audit plan")
    else:
        require(not managed, "nonempty output requires --resume")


def _validate_checkpoint_entries(
    output_dir: Path,
    tasks: Iterable[FileTask],
    *,
    require_complete: bool,
) -> None:
    expected = {checkpoint_path(output_dir, task) for task in tasks}
    root = output_dir / "checkpoints"
    expected_directories = {root, *(path.parent for path in expected)}
    if not root.exists() and not root.is_symlink():
        require(not require_complete, "checkpoint tree is missing")
        return
    root_info = root.lstat()
    require(
        stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
        "checkpoint tree is not a regular directory",
    )
    observed: set[Path] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in directory_names:
            path = parent / name
            info = path.lstat()
            require(
                stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                "checkpoint directory is not a regular directory",
            )
            require(path in expected_directories, f"unexpected checkpoint directory: {name}")
        for name in file_names:
            path = parent / name
            info = path.lstat()
            require(
                stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                "checkpoint entry is not a regular file",
            )
            require(path in expected, f"unexpected checkpoint entry: {name}")
            observed.add(path)
    require(observed <= expected, "checkpoint path set exceeds frozen plan")
    if require_complete:
        require(observed == expected, "checkpoint tree is incomplete")


def _revalidate_final_source(
    source_root: Path, tasks: Iterable[FileTask], expected_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    frozen_tasks = list(tasks)
    expected_by_table: dict[str, set[str]] = {}
    for task in frozen_tasks:
        expected_by_table.setdefault(task.table_name, set()).add(
            Path(task.relative_path).name
        )
    _validate_source_tree_entries(source_root, expected_by_table)
    current_tasks: list[FileTask] = []
    for task in frozen_tasks:
        secure_join_regular(
            source_root,
            safe_relative_path(task.relative_path, "final source path"),
            field="final source file",
        )
        current = anchored_regular_identity(
            source_root,
            safe_relative_path(task.relative_path, "final source path"),
            field=f"final source file {task.relative_path}",
        )
        require(current == task.source_stat, f"final source stat differs: {task.relative_path}")
        current_tasks.append(
            FileTask(
                index=task.index,
                table_name=task.table_name,
                relative_path=task.relative_path,
                bytes=task.bytes,
                mtime_ns=task.mtime_ns,
                source_stat=current,
            )
        )
    observed = source_snapshot(current_tasks)
    require(observed == expected_snapshot, "final source snapshot differs from frozen plan")
    return observed


def build_report(
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    tasks: list[FileTask],
    checkpoints: list[Mapping[str, Any]],
    checkpoint_tree: Mapping[str, Any],
    content_tree: Mapping[str, Any],
    final_snapshot: Mapping[str, Any],
    started_at: str,
    elapsed_seconds: float,
    resumed_files: int,
) -> dict[str, Any]:
    full_scope = bool(plan["full_scope"])
    status = FULL_SUCCESS_STATUS if full_scope else PILOT_STATUS
    gates = dict(EXPECTED_FULL_GATES)
    if not full_scope:
        gates["current_content_baseline_component"] = "NOT_ESTABLISHED_PILOT"
    deterministic_core = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "full_scope": full_scope,
        "max_files": plan["max_files"],
        "parent_manifest_sha256": plan["parent_inventory"]["manifest_sha256"],
        "files": content_tree["files"],
        "bytes": content_tree["bytes"],
        "content_tree_sha256": content_tree["tree_sha256"],
        "gate_status": gates,
        "claim_boundary": config["claim_boundary"],
        "same_size_restored_mtime_limitation": config[
            "same_size_restored_mtime_limitation"
        ],
        "checkpoint_trust_limitation": config["checkpoint_trust_limitation"],
        "publication_completion_contract": (
            "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
        ),
    }
    result_fingerprint = sha256_bytes(canonical_json_bytes(deterministic_core))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": config.get("label"),
        "status": status,
        "pass_current_content_baseline_component": full_scope,
        "pass_g1": False,
        "scope": {
            "full_scope": full_scope,
            "max_files": plan["max_files"],
            "declared_files": plan["declared_file_count"],
            "declared_bytes": plan["declared_bytes"],
            "selected_files": len(tasks),
            "selected_bytes": sum(task.bytes for task in tasks),
            "source_root_id": plan["source"]["root_id"],
            "gpu_used": False,
            "scientific_effect_estimation_performed": False,
        },
        "parent_inventory": plan["parent_inventory"],
        "source_snapshot": {
            **dict(final_snapshot),
            "exactly_equal_to_plan_start": True,
        },
        "content_baseline": {
            "algorithm": "sha256",
            "files": content_tree["files"],
            "bytes": content_tree["bytes"],
            "content_tree_sha256": content_tree["tree_sha256"],
            "historical_identity_established": False,
            "prior_component_run_identity_established": False,
        },
        "checkpoint_evidence": {
            "checkpoint_files": checkpoint_tree["files"],
            "checkpoint_bytes": checkpoint_tree["bytes"],
            "checkpoint_tree_sha256": checkpoint_tree["tree_sha256"],
            "resumed_files": resumed_files,
            "newly_hashed_files": len(tasks) - resumed_files,
            "resume_validates_current_stat": True,
            "resume_rehashes_completed_content": False,
            "checkpoint_digest_authenticated": False,
            "checkpoint_integrity_scope": "ACCIDENTAL_CORRUPTION_ONLY",
            "trusted_output_directory_required": True,
            "output_trust_precondition": plan["execution"][
                "output_trust_precondition"
            ],
            "disk_revalidated_before_report": True,
            "disk_revalidated_in_final_output_validation": True,
            "stat_fields": list(EXPECTED_STAT_FIELDS),
        },
        "publication": {
            "completion_contract": (
                "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
            ),
            "complete_state_written_last": True,
            "run_state_bound_separately_from_outputs_artifacts": True,
            "failure_removes_pass_report_markdown_and_manifest": True,
        },
        "gate_status": gates,
        "execution": {
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": elapsed_seconds,
            "workers": 1,
            "read_chunk_bytes": plan["execution"]["read_chunk_bytes"],
            "workload_class": "CPU_IO_SEQUENTIAL_SHA256",
            "path_redacted": True,
        },
        "provenance": {
            **plan["provenance"],
            **git_metadata(),
            "parent_manifest_sha256": plan["parent_inventory"]["manifest_sha256"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "result_fingerprint_sha256": result_fingerprint,
            "task_plan_fingerprint_scope": "HOST_FILESYSTEM_IDENTITY_BOUND",
            "result_fingerprint_scope": "PATH_SIZE_CONTENT_AND_CONTRACT",
            "path_redacted": True,
        },
        "claim_boundary": config["claim_boundary"],
        "same_size_restored_mtime_limitation": config[
            "same_size_restored_mtime_limitation"
        ],
        "checkpoint_trust_limitation": config["checkpoint_trust_limitation"],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    scope = report["scope"]
    checkpoints = report["checkpoint_evidence"]
    baseline = report["content_baseline"]
    lines = [
        "# OpenAlex G-1 Current-Content Freeze",
        "",
        f"- Status: `{report['status']}`",
        "- Current-content component pass: "
        f"`{str(report['pass_current_content_baseline_component']).lower()}`",
        f"- G-1 pass: `{str(report['pass_g1']).lower()}`",
        f"- Full scope: `{str(scope['full_scope']).lower()}`",
        f"- Files / bytes: `{scope['selected_files']} / {scope['selected_bytes']}`",
        f"- Content tree SHA-256: `{baseline['content_tree_sha256']}`",
        f"- Resumed / newly hashed: `{checkpoints['resumed_files']} / "
        f"{checkpoints['newly_hashed_files']}`",
        "- Completed checkpoints are rehashed on resume: "
        f"`{str(checkpoints['resume_rehashes_completed_content']).lower()}`",
        "",
        "## Gate status",
        "",
        "| Gate | Status |",
        "| --- | --- |",
    ]
    for key, value in sorted(report["gate_status"].items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
            "## Resume and mutation limitation",
            "",
            str(report["same_size_restored_mtime_limitation"]),
            "",
            "## Checkpoint trust limitation",
            "",
            str(report["checkpoint_trust_limitation"]),
            "",
            "## Tool provenance scope",
            "",
            str(report["provenance"]["tool_attestation_scope"]),
            "",
            "## Publication completion contract",
            "",
            str(report["publication"]["completion_contract"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_final_outputs(
    output_dir: Path,
    *,
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoint_tree: Mapping[str, Any],
    tasks: list[FileTask],
    source_root: Path,
) -> dict[str, Any]:
    report_path = output_dir / "content_freeze_audit.json"
    markdown_path = output_dir / "content_freeze_audit.md"
    outputs_path = output_dir / "content_freeze_audit.outputs.json"
    try:
        atomic_write_json(report_path, report)
        atomic_write_text(markdown_path, render_markdown(report))
        artifacts = []
        for path in (
            output_dir / "audit_plan.json",
            report_path,
            markdown_path,
        ):
            artifacts.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        outputs = {
            "schema_version": OUTPUTS_SCHEMA_VERSION,
            "status": report["status"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "result_fingerprint_sha256": report["provenance"][
                "result_fingerprint_sha256"
            ],
            "content_tree_sha256": report["content_baseline"][
                "content_tree_sha256"
            ],
            "checkpoint_tree": {
                key: checkpoint_tree[key]
                for key in ("files", "bytes", "tree_sha256")
            },
            "artifacts": artifacts,
            "completion_contract": (
                "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
            ),
            "run_state_bound_separately": True,
            "path_redacted": True,
        }
        atomic_write_json(outputs_path, outputs)
        return validate_final_outputs(
            output_dir,
            expected_report=report,
            expected_outputs=outputs,
            tasks=tasks,
            source_root=source_root,
            plan=plan,
            expected_state_status=FINALIZING_STATUS,
        )
    except BaseException:
        _remove_final_publication(output_dir)
        raise


def _remove_final_publication(output_dir: Path) -> None:
    """Fail closed: a paused run may not retain standalone PASS artifacts."""

    for name in (
        "content_freeze_audit.outputs.json",
        "content_freeze_audit.md",
        "content_freeze_audit.json",
    ):
        path = output_dir / name
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        require(
            stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode),
            f"cannot remove non-file final publication: {name}",
        )
        path.unlink()
    _fsync_parent(output_dir / "content_freeze_audit.json")


def validate_final_outputs(
    output_dir: Path,
    *,
    expected_report: Mapping[str, Any],
    expected_outputs: Mapping[str, Any],
    tasks: list[FileTask],
    source_root: Path,
    plan: Mapping[str, Any],
    expected_state_status: str,
) -> dict[str, Any]:
    entries = {path.name: path for path in output_dir.iterdir()}
    require(set(entries) == ALLOWED_TOP_LEVEL, "final output entry set differs")
    _validate_output_entries(output_dir, resume=True)
    final_checkpoints = _load_validated_checkpoints(
        output_dir,
        tasks,
        source_root=source_root,
        task_fingerprint_sha256=str(plan["task_fingerprint_sha256"]),
        plan_sha256=str(plan["plan_sha256"]),
    )
    final_checkpoint_tree = _checkpoint_tree(output_dir, tasks, final_checkpoints)
    final_content_tree = _content_tree(tasks, final_checkpoints)
    report = read_json_regular(
        output_dir / "content_freeze_audit.json", field="final report"
    )
    require(report == expected_report, "final report differs after write")
    markdown_path = output_dir / "content_freeze_audit.md"
    regular_identity(markdown_path, field="final Markdown")
    require(
        markdown_path.read_text(encoding="utf-8") == render_markdown(report),
        "final Markdown differs from JSON rendering",
    )
    outputs = read_json_regular(
        output_dir / "content_freeze_audit.outputs.json", field="outputs manifest"
    )
    require(outputs == expected_outputs, "outputs manifest differs after write")
    require(
        outputs["checkpoint_tree"]
        == {
            key: final_checkpoint_tree[key]
            for key in ("files", "bytes", "tree_sha256")
        },
        "final checkpoint tree differs after write",
    )
    require(
        outputs["content_tree_sha256"] == final_content_tree["tree_sha256"],
        "final content tree differs after write",
    )
    state = read_json_regular(output_dir / "run_state.json", field="final state")
    require(state.get("schema_version") == STATE_SCHEMA_VERSION, "final state schema differs")
    require(
        state.get("status") == expected_state_status,
        "final state status differs",
    )
    require(
        state.get("task_fingerprint_sha256") == plan["task_fingerprint_sha256"],
        "final state task fingerprint differs",
    )
    require(state.get("plan_sha256") == plan["plan_sha256"], "final state plan differs")
    require(
        state.get("completed_files") == plan["selected_file_count"],
        "final state completed file count differs",
    )
    require(
        state.get("completed_bytes") == plan["selected_bytes"],
        "final state completed byte count differs",
    )
    require(state.get("total_files") == plan["selected_file_count"], "final state file total differs")
    require(state.get("total_bytes") == plan["selected_bytes"], "final state byte total differs")
    require(state.get("path_redacted") is True, "final state path redaction differs")
    require("error" not in state, "final state unexpectedly contains an error")
    for raw_artifact in require_list(outputs.get("artifacts"), "output artifacts"):
        artifact = require_mapping(raw_artifact, "output artifact")
        relative = safe_relative_path(
            artifact.get("relative_path"), "output artifact relative path"
        )
        path = output_dir / relative
        regular_identity(path, field="bound output artifact")
        require(path.stat().st_size == artifact.get("bytes"), "output artifact size differs")
        require(sha256_file(path) == artifact.get("sha256"), "output artifact SHA-256 differs")
    return outputs


def run_audit(
    *,
    config_path: Path,
    inventory_manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    max_files: int | None = None,
    resume: bool = False,
    progress: Callable[[int, int, int, FileTask, bool], None] | None = None,
    config_snapshot: InputSnapshot | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = Path(os.path.abspath(os.fspath(config_path.expanduser())))
    inventory_manifest_path = Path(
        os.path.abspath(os.fspath(inventory_manifest_path.expanduser()))
    )
    frozen_config = config_snapshot or read_config_snapshot(config_path)
    config = frozen_config.value
    tool_path = Path(os.path.abspath(os.fspath(Path(__file__))))
    _, tool_sha256, _ = read_regular_file_bytes(tool_path, field="audit tool")
    source_contract = require_mapping(config.get("source"), "source")
    root = validate_source_root(
        source_root, expected_id=str(source_contract.get("expected_root_id", ""))
    )
    with output_lock(root, output_dir) as output:
        _validate_output_entries(output, resume=resume)
        manifest, all_tasks, snapshot, manifest_snapshot = validate_parent_inventory(
            config=config,
            manifest_path=inventory_manifest_path,
            source_root=root,
        )
        plan = build_plan(
            config=config,
            config_path=config_path,
            config_sha256=frozen_config.sha256,
            manifest_path=inventory_manifest_path,
            manifest_sha256=manifest_snapshot.sha256,
            manifest=manifest,
            all_tasks=all_tasks,
            snapshot=snapshot,
            max_files=max_files,
            tool_sha256=tool_sha256,
        )
        plan_path = output / "audit_plan.json"
        if resume:
            prior = read_json_regular(plan_path, field="audit plan")
            require(prior == plan, "resume plan differs from current frozen plan")
        else:
            atomic_write_json(plan_path, plan)
        tasks = [task_from_mapping(value) for value in plan["tasks"]]
        _validate_checkpoint_entries(output, tasks, require_complete=False)
        state_path = output / "run_state.json"
        atomic_write_json(
            state_path,
            _state_payload(
                plan=plan,
                status="RUNNING",
                completed_files=0,
                completed_bytes=0,
                resumed_files=0,
                latest_relative_path=None,
            ),
        )
        started_at = utc_now()
        started = time.monotonic()
        checkpoints: list[Mapping[str, Any]] = []
        completed_bytes = 0
        resumed_files = 0
        latest: str | None = None
        try:
            for completed, task in enumerate(tasks, start=1):
                path = checkpoint_path(output, task)
                if path.exists() or path.is_symlink():
                    require(resume, "checkpoint exists without --resume")
                    checkpoint = validate_checkpoint(
                        read_json_regular(path, field="checkpoint"),
                        task,
                        source_root=root,
                        task_fingerprint_sha256=plan["task_fingerprint_sha256"],
                        plan_sha256=plan["plan_sha256"],
                    )
                    resumed = True
                    resumed_files += 1
                else:
                    checkpoint = hash_task(
                        task,
                        source_root=root,
                        task_fingerprint_sha256=plan["task_fingerprint_sha256"],
                        plan_sha256=plan["plan_sha256"],
                        chunk_bytes=plan["execution"]["read_chunk_bytes"],
                    )
                    atomic_write_json(path, checkpoint)
                    resumed = False
                checkpoints.append(checkpoint)
                completed_bytes += task.bytes
                latest = task.relative_path
                atomic_write_json(
                    state_path,
                    _state_payload(
                        plan=plan,
                        status="RUNNING",
                        completed_files=completed,
                        completed_bytes=completed_bytes,
                        resumed_files=resumed_files,
                        latest_relative_path=latest,
                    ),
                )
                if progress is not None:
                    progress(completed, len(tasks), completed_bytes, task, resumed)
            final_snapshot = _revalidate_final_source(root, all_tasks, snapshot)
            _, final_config_sha256, _ = read_regular_file_bytes(
                config_path, field="final config"
            )
            _, final_tool_sha256, _ = read_regular_file_bytes(
                tool_path, field="final audit tool"
            )
            _, final_manifest_sha256, _ = read_regular_file_bytes(
                inventory_manifest_path, field="final parent manifest"
            )
            require(
                final_config_sha256 == plan["provenance"]["config_sha256"],
                "config changed during audit",
            )
            require(
                final_tool_sha256 == plan["provenance"]["tool_sha256"],
                "tool changed during audit",
            )
            require(
                final_manifest_sha256
                == plan["parent_inventory"]["manifest_sha256"],
                "parent manifest changed during audit",
            )
            validated_checkpoints = _load_validated_checkpoints(
                output,
                tasks,
                source_root=root,
                task_fingerprint_sha256=plan["task_fingerprint_sha256"],
                plan_sha256=plan["plan_sha256"],
            )
            checkpoint_tree = _checkpoint_tree(
                output, tasks, validated_checkpoints
            )
            content_tree = _content_tree(tasks, validated_checkpoints)
            report = build_report(
                config=config,
                plan=plan,
                tasks=tasks,
                checkpoints=[item.value for item in validated_checkpoints],
                checkpoint_tree=checkpoint_tree,
                content_tree=content_tree,
                final_snapshot=final_snapshot,
                started_at=started_at,
                elapsed_seconds=time.monotonic() - started,
                resumed_files=resumed_files,
            )
            atomic_write_json(
                state_path,
                _state_payload(
                    plan=plan,
                    status=FINALIZING_STATUS,
                    completed_files=len(tasks),
                    completed_bytes=completed_bytes,
                    resumed_files=resumed_files,
                    latest_relative_path=latest,
                ),
            )
            outputs = write_final_outputs(
                output,
                report=report,
                plan=plan,
                checkpoint_tree=checkpoint_tree,
                tasks=tasks,
                source_root=root,
            )
            atomic_write_json(
                state_path,
                _state_payload(
                    plan=plan,
                    status="COMPLETE",
                    completed_files=len(tasks),
                    completed_bytes=completed_bytes,
                    resumed_files=resumed_files,
                    latest_relative_path=latest,
                ),
            )
            outputs = validate_final_outputs(
                output,
                expected_report=report,
                expected_outputs=outputs,
                tasks=tasks,
                source_root=root,
                plan=plan,
                expected_state_status="COMPLETE",
            )
            return report, outputs
        except BaseException as exc:
            _remove_final_publication(output)
            atomic_write_json(
                state_path,
                _state_payload(
                    plan=plan,
                    status=PAUSED_STATUS,
                    completed_files=len(checkpoints),
                    completed_bytes=completed_bytes,
                    resumed_files=resumed_files,
                    latest_relative_path=latest,
                    error=exc,
                ),
            )
            raise


def resolve_source_root(config: Mapping[str, Any], override: Path | None) -> Path:
    source = require_mapping(config.get("source"), "source")
    if override is not None:
        return override
    env_name = str(source.get("root_env", ""))
    raw = os.environ.get(env_name)
    require(raw is not None and raw != "", f"set {env_name} or pass --source-root")
    return Path(raw)


def resolve_manifest_path(config: Mapping[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override
    parent = require_mapping(config.get("parent_inventory"), "parent inventory")
    relative = safe_relative_path(
        parent.get("manifest_relative_path"), "parent manifest relative path"
    )
    return REPO_ROOT / relative


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inventory-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_path = Path(os.path.abspath(os.fspath(args.config.expanduser())))
        config_snapshot = read_config_snapshot(config_path)
        config = config_snapshot.value
        source_root = resolve_source_root(config, args.source_root)
        manifest_path = resolve_manifest_path(config, args.inventory_manifest)
        execution = require_mapping(config.get("execution"), "execution")
        progress_every_files = require_int(
            execution.get("progress_every_files"), "progress every files"
        )
        progress_every_seconds = require_int(
            execution.get("progress_every_seconds"), "progress every seconds"
        )
        last_print = 0.0

        def progress(
            completed: int,
            total: int,
            completed_bytes: int,
            task: FileTask,
            resumed: bool,
        ) -> None:
            nonlocal last_print
            now = time.monotonic()
            if (
                completed == total
                or completed % progress_every_files == 0
                or now - last_print >= progress_every_seconds
            ):
                action = "reused" if resumed else "hashed"
                print(
                    f"content freeze: {completed:,}/{total:,} files; "
                    f"{completed_bytes:,} bytes; {action}={task.relative_path}",
                    file=sys.stderr,
                    flush=True,
                )
                last_print = now

        report, outputs = run_audit(
            config_path=config_path,
            inventory_manifest_path=manifest_path,
            source_root=source_root,
            output_dir=args.output_dir,
            max_files=args.max_files,
            resume=args.resume,
            progress=progress,
            config_snapshot=config_snapshot,
        )
    except KeyboardInterrupt:
        print("PAUSED: interrupted; resume with --resume", file=sys.stderr)
        return 130
    except (ContentFreezeError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "content_tree_sha256": report["content_baseline"][
                    "content_tree_sha256"
                ],
                "result_fingerprint_sha256": report["provenance"][
                    "result_fingerprint_sha256"
                ],
                "outputs_manifest_sha256": sha256_file(
                    Path(os.path.abspath(os.fspath(args.output_dir.expanduser())))
                    / "content_freeze_audit.outputs.json"
                ),
            },
            sort_keys=True,
        )
    )
    if args.require_pass and report["status"] != FULL_SUCCESS_STATUS:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
