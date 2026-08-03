#!/usr/bin/env python3
"""Audit OpenAlex work-topic relation multiplicity and primary inclusion.

The audit projects only ``(work ID, primary-topic ID)`` from the main works
table and ``(work ID, topic ID, score)`` from the exploded topics table.  It
materializes deterministic file-checkpointed event segments, globally sorts
the combined stream, measures raw edge multiplicity, and promotes a set view
only when duplicate scores agree exactly after canonical ``-0.0`` handling.

This is a CPU/I/O evidence component.  A max-files run or a deterministic
hash-bucket run is always pilot-only and can never receive the full status.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_work_topic_relation_v0.yaml"
)

CONFIG_SCHEMA = "openalex_g1_work_topic_relation_config.v1"
PLAN_SCHEMA = "openalex_g1_work_topic_relation_plan.v1"
CHECKPOINT_SCHEMA = "openalex_g1_work_topic_relation_checkpoint.v1"
SORT_SCHEMA = "openalex_g1_work_topic_relation_sort.v1"
REPORT_SCHEMA = "openalex_g1_work_topic_relation_report.v1"
OUTPUTS_SCHEMA = "openalex_g1_work_topic_relation_outputs.v1"
PRODUCTION_CONFIG_SHA256 = (
    "7e7d50f212e9738dfce5b437174cfecbfa76679e532a15e23d84ca88c5e0b7fc"
)
# Tests replace this binding with the digest of a closed synthetic contract.
# The CLI never exposes an override and therefore accepts only the frozen v0.
FROZEN_CONFIG_SHA256 = PRODUCTION_CONFIG_SHA256

FULL_STATUS = (
    "PASS_CURRENT_SNAPSHOT_WORK_TOPIC_RELATION_COMPONENT_WITH_FROZEN_EXACT_SCORE_DEDUP"
)
PILOT_STATUS = "PILOT_WORK_TOPIC_RELATION_ONLY"
BLOCKED_STATUS = "BLOCKED_WORK_TOPIC_RELATION_COMPONENT"

MAIN_KIND = np.uint8(0)
TOPIC_KIND = np.uint8(1)
LOCATOR_ROW_BITS = 20
LOCATOR_ROW_MASK = (1 << LOCATOR_ROW_BITS) - 1
LOCATOR_FILE_LIMIT = 1 << (32 - LOCATOR_ROW_BITS)
SAMPLE_LIMIT = 20

# Explicit padding makes the on-disk ABI stable and naturally aligned.
EVENT_DTYPE = np.dtype(
    [
        ("key", "<u8"),
        ("score_bits", "<u8"),
        ("locator", "<u4"),
        ("kind", "u1"),
        ("padding", "V3"),
    ],
    align=False,
)
DEDUP_DTYPE = np.dtype([("key", "<u8"), ("score_bits", "<u8")], align=False)
assert EVENT_DTYPE.itemsize == 24
assert DEDUP_DTYPE.itemsize == 16


class RelationAuditError(RuntimeError):
    """Raised when an evidence or execution contract is violated."""


@dataclass(frozen=True)
class FileTask:
    table_key: str
    table_name: str
    file_index: int
    relative_path: str
    bytes: int
    mtime_ns: int
    footer_rows: int
    projected_columns: tuple[str, ...]
    projected_schema: dict[str, dict[str, Any]]
    projected_schema_sha256: str
    source_rich_identity: dict[str, int]
    task_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_freeze_published_json_bytes(value: Any) -> bytes:
    """Return the exact JSON byte format published by the content-freeze tool."""

    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RelationAuditError(
            f"cannot open regular file for hashing: {exc}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), "hash input is not a regular file")
        handle = os.fdopen(descriptor, "rb", closefd=False)
        while block := handle.read(chunk_bytes):
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_regular_bytes_snapshot(
    path: Path, *, field: str
) -> tuple[bytes, str, dict[str, int]]:
    _assert_no_symlink_components(path, field=field)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RelationAuditError(f"cannot open {field}") from exc
    try:
        start = _rich_identity_from_stat(os.fstat(descriptor))
        require(stat.S_ISREG(start["mode"]), f"{field} is not a regular file")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        end = _rich_identity_from_stat(os.fstat(descriptor))
        require(start == end, f"{field} changed during single-FD snapshot")
        payload = b"".join(chunks)
        require(len(payload) == start["bytes"], f"{field} snapshot byte count mismatch")
        return payload, sha256_bytes(payload), start
    except OSError as exc:
        raise RelationAuditError(f"cannot read {field}") from exc
    finally:
        os.close(descriptor)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RelationAuditError(message)


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be a mapping")
    return dict(value)


def require_list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be a list")
    return list(value)


def require_int(value: Any, field: str) -> int:
    require(type(value) is int, f"{field} must be an integer")
    return int(value)


def require_sha256(value: Any, field: str) -> str:
    text = str(value)
    require(re.fullmatch(r"[0-9a-f]{64}", text) is not None, f"{field} is not SHA-256")
    return text


def safe_relative_path(value: Any, field: str) -> Path:
    path = Path(str(value))
    require(not path.is_absolute(), f"{field} must be relative")
    require(path.parts and path != Path("."), f"{field} may not be empty")
    require(".." not in path.parts, f"{field} may not contain '..'")
    return path


def read_json(path: Path, field: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{field} is not a regular file")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            return require_mapping(json.load(handle), field)
    except (OSError, json.JSONDecodeError) as exc:
        raise RelationAuditError(f"cannot read {field}: {exc}") from exc


def _parse_content_freeze_json(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = require_mapping(json.loads(payload), field)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelationAuditError(f"cannot parse {field}") from exc
    require(
        payload == content_freeze_published_json_bytes(value),
        f"{field} is not exact content-freeze published JSON bytes",
    )
    return value


def _validate_trusted_content_output_chain(output: Path) -> dict[str, Any]:
    output = _absolute(output)
    euid = os.geteuid()
    current = Path(output.anchor)
    checked = 0
    for part in output.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            raise RelationAuditError("cannot inspect content-freeze output chain") from exc
        require(
            stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            "content-freeze output chain is not a directory",
        )
        require(
            info.st_uid in {0, euid},
            "content-freeze output chain has an untrusted owner",
        )
        shared_write = bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
        if current == output:
            require(
                info.st_uid == euid,
                "content-freeze output directory is not operator-owned",
            )
            require(
                stat.S_IMODE(info.st_mode) == 0o700,
                "content-freeze output directory must be mode 0700",
            )
        elif shared_write:
            require(
                bool(info.st_mode & stat.S_ISVTX),
                "writable content-freeze output ancestor lacks sticky protection",
            )
        checked += 1
    return {
        "status": "VERIFIED_PRIVATE_0700_OWNER_CHAIN",
        "components_checked": checked,
        "operator_paths_redacted": True,
    }


def _trusted_directory_fd(path: Path, *, field: str) -> int:
    _assert_no_symlink_components(path, field=field)
    try:
        info = path.lstat()
    except OSError as exc:
        raise RelationAuditError(f"cannot stat {field}: {exc}") from exc
    require(stat.S_ISDIR(info.st_mode), f"{field} is not a directory")
    require(info.st_uid == os.geteuid(), f"{field} is not owned by the current user")
    require(
        stat.S_IMODE(info.st_mode) & 0o022 == 0,
        f"{field} is group/world writable",
    )
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RelationAuditError(f"cannot open {field}: {exc}") from exc


def _validate_regular_output_entry(directory_fd: int, name: str, *, field: str) -> None:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RelationAuditError(f"cannot stat {field}: {exc}") from exc
    require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
    require(info.st_uid == os.geteuid(), f"{field} is not owned by the current user")
    require(stat.S_IMODE(info.st_mode) == 0o600, f"{field} mode is not 0600")


def _fsync_parent(path: Path) -> None:
    descriptor = _trusted_directory_fd(path.parent, field="artifact parent")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, value: bytes) -> None:
    directory_fd = _trusted_directory_fd(path.parent, field="artifact parent")
    temporary = f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    try:
        _validate_regular_output_entry(
            directory_fd, path.name, field="existing output artifact"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except OSError as exc:
        raise RelationAuditError(
            f"cannot atomically publish output artifact: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        finally:
            os.close(directory_fd)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def _open_trusted_output(path: Path, flags: int, *, create: bool = False) -> int:
    directory_fd = _trusted_directory_fd(path.parent, field="artifact parent")
    try:
        _validate_regular_output_entry(directory_fd, path.name, field="output artifact")
        effective = flags | os.O_NOFOLLOW
        if create:
            effective |= os.O_CREAT
        descriptor = os.open(path.name, effective, 0o600, dir_fd=directory_fd)
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), "output artifact is not a regular file")
        require(info.st_uid == os.geteuid(), "output artifact owner mismatch")
        require(stat.S_IMODE(info.st_mode) == 0o600, "output artifact mode is not 0600")
        return descriptor
    except OSError as exc:
        raise RelationAuditError(f"cannot open trusted output artifact: {exc}") from exc
    finally:
        os.close(directory_fd)


def _begin_atomic_binary(path: Path) -> tuple[int, str, Any]:
    directory_fd = _trusted_directory_fd(path.parent, field="artifact parent")
    temporary = f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    try:
        _validate_regular_output_entry(
            directory_fd, path.name, field="existing output artifact"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        return directory_fd, temporary, os.fdopen(descriptor, "wb")
    except OSError as exc:
        os.close(directory_fd)
        raise RelationAuditError(
            f"cannot create atomic output temporary: {exc}"
        ) from exc


def _finish_atomic_binary(
    path: Path,
    *,
    directory_fd: int,
    temporary: str,
    publish: bool,
) -> None:
    try:
        if publish:
            os.replace(
                temporary,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
        else:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    except OSError as exc:
        raise RelationAuditError(
            f"cannot finalize atomic output artifact: {exc}"
        ) from exc
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _unlink_trusted_output(path: Path, *, missing_ok: bool = True) -> None:
    directory_fd = _trusted_directory_fd(path.parent, field="artifact parent")
    try:
        try:
            info = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return
            raise RelationAuditError("required output artifact is missing")
        require(stat.S_ISREG(info.st_mode), "output artifact is not a regular file")
        require(info.st_uid == os.geteuid(), "output artifact owner mismatch")
        os.unlink(path.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as exc:
        raise RelationAuditError(f"cannot remove output artifact: {exc}") from exc
    finally:
        os.close(directory_fd)


def _regular_identity(path: Path, *, field: str) -> dict[str, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RelationAuditError(f"cannot stat {field}: {exc}") from exc
    require(not stat.S_ISLNK(info.st_mode), f"{field} may not be a symlink")
    require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
    return {"bytes": int(info.st_size), "mtime_ns": int(info.st_mtime_ns)}


def _rich_identity_from_stat(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "links": int(info.st_nlink),
        "bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _rich_regular_identity(path: Path, *, field: str) -> dict[str, int]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RelationAuditError(f"cannot open {field}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
        return _rich_identity_from_stat(info)
    finally:
        os.close(descriptor)


def _assert_no_symlink_components(path: Path, *, field: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            continue
        try:
            info = current.lstat()
        except OSError as exc:
            raise RelationAuditError(f"cannot stat {field} component: {exc}") from exc
        require(not stat.S_ISLNK(info.st_mode), f"{field} path component is a symlink")


def secure_join_regular(root: Path, relative: Path, *, field: str) -> Path:
    require(root.is_dir() and not root.is_symlink(), f"{field} root is not a directory")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise RelationAuditError(f"cannot stat {field}: {exc}") from exc
        require(not stat.S_ISLNK(info.st_mode), f"{field} path component is a symlink")
    _regular_identity(current, field=field)
    try:
        current.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RelationAuditError(f"{field} escapes its declared root") from exc
    return current


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def resolve_root(explicit: Path | None, env_name: str, *, field: str) -> Path:
    raw = explicit if explicit is not None else Path(os.environ.get(env_name, ""))
    require(str(raw) != "." and str(raw) != "", f"{field} is not configured")
    require(raw.is_absolute(), f"{field} must be absolute")
    _assert_no_symlink_components(raw, field=field)
    require(raw.is_dir() and not raw.is_symlink(), f"{field} is not a directory")
    return raw.resolve()


def _validate_output_tree(output_dir: Path) -> None:
    root_fd = _trusted_directory_fd(output_dir, field="output directory")
    os.close(root_fd)
    for current, directory_names, file_names in os.walk(
        output_dir, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        directory_fd = _trusted_directory_fd(
            current_path, field="output tree directory"
        )
        try:
            for name in sorted(directory_names):
                relative = (current_path / name).relative_to(output_dir).as_posix()
                require(
                    relative
                    in {"checkpoints", "checkpoints/main", "checkpoints/topics"},
                    "output tree contains an unknown directory",
                )
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                require(
                    stat.S_ISDIR(info.st_mode),
                    "output tree contains a symlink or non-directory entry",
                )
                require(
                    info.st_uid == os.geteuid(),
                    "output tree directory is not owned by the current user",
                )
                require(
                    stat.S_IMODE(info.st_mode) & 0o022 == 0,
                    "output tree directory is group/world writable",
                )
            for name in sorted(file_names):
                relative = (current_path / name).relative_to(output_dir).as_posix()
                root_files = {
                    "audit.lock",
                    "audit_plan.json",
                    "run_state.json",
                    "sort_manifest.json",
                    "work_topic_relation.events.raw.bin",
                    "work_topic_relation.events.sorted.bin",
                    "work_topic_edges.deduplicated.sorted.bin",
                    "work_topic_relation_audit.json",
                    "work_topic_relation_audit.md",
                    "work_topic_relation_audit.outputs.json",
                }
                checkpoint_file = re.fullmatch(
                    r"checkpoints/(main|topics)/[0-9a-f]{24}[.]json", relative
                )
                require(
                    relative in root_files or checkpoint_file is not None,
                    "output tree contains an unknown or stale managed artifact",
                )
                _validate_regular_output_entry(
                    directory_fd, name, field="output tree artifact"
                )
        finally:
            os.close(directory_fd)


def _secure_output_subdir(output_dir: Path, *parts: str) -> Path:
    descriptor = _trusted_directory_fd(output_dir, field="output directory")
    current = output_dir
    try:
        for part in parts:
            require(
                part not in {"", ".", ".."} and "/" not in part,
                "invalid output subdirectory component",
            )
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            info = os.fstat(child)
            require(
                stat.S_ISDIR(info.st_mode), "output subdirectory is not a directory"
            )
            require(
                info.st_uid == os.geteuid(),
                "output subdirectory is not owned by the current user",
            )
            require(
                stat.S_IMODE(info.st_mode) & 0o022 == 0,
                "output subdirectory is group/world writable",
            )
            os.close(descriptor)
            descriptor = child
            current /= part
        return current
    except OSError as exc:
        raise RelationAuditError(
            f"cannot create trusted output subdirectory: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def prepare_output_dir(
    output_dir: Path,
    *,
    source_root: Path,
    taxonomy_root: Path,
    resume: bool,
) -> Path:
    require(output_dir.is_absolute(), "output directory must be absolute")
    if output_dir.is_symlink():
        raise RelationAuditError("output directory may not be a symlink")
    if output_dir.exists():
        require(output_dir.is_dir(), "output path is not a directory")
        entries = [path.name for path in output_dir.iterdir()]
        if entries and not resume:
            raise RelationAuditError("non-empty output directory requires --resume")
    else:
        output_dir.mkdir(parents=True, mode=0o700)
        output_dir.chmod(0o700)
    _assert_no_symlink_components(output_dir, field="output directory")
    output = output_dir.resolve()
    require(
        not _paths_overlap(output, source_root), "output directory overlaps source root"
    )
    require(
        not _paths_overlap(output, taxonomy_root),
        "output directory overlaps taxonomy root",
    )
    _validate_output_tree(output)
    _secure_output_subdir(output, "checkpoints", "main")
    _secure_output_subdir(output, "checkpoints", "topics")
    _validate_output_tree(output)
    return output


@contextmanager
def output_lock(output_dir: Path) -> Iterator[None]:
    directory_fd = _trusted_directory_fd(output_dir, field="output directory")
    descriptor: int | None = None
    try:
        _validate_regular_output_entry(directory_fd, "audit.lock", field="audit lock")
        descriptor = os.open(
            "audit.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), "audit lock is not a regular file")
        require(info.st_uid == os.geteuid(), "audit lock owner mismatch")
        require(stat.S_IMODE(info.st_mode) == 0o600, "audit lock mode is not 0600")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RelationAuditError(
                "another relation audit holds the output lock"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError as exc:
        raise RelationAuditError(f"cannot open audit lock: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _validate_content_baseline_contract(config: Mapping[str, Any]) -> None:
    parents = require_mapping(config.get("parents"), "parents")
    contract = require_mapping(
        parents.get("current_content_baseline"), "current content baseline parent"
    )
    require(
        str(contract.get("receipt_path_env", "")),
        "content receipt environment variable is empty",
    )
    require(
        contract.get("expected_basename") == "content_freeze_audit.outputs.json",
        "content receipt basename drift",
    )
    require(
        contract.get("schema_version") == "openalex_g1_content_freeze_outputs.v1",
        "content receipt schema drift",
    )
    require(
        contract.get("required_status") == "PASS_CURRENT_CONTENT_BASELINE_COMPONENT",
        "content receipt status drift",
    )
    require_sha256(contract.get("required_plan_sha256"), "content plan anchor")
    require_sha256(
        contract.get("required_task_fingerprint_sha256"), "content task anchor"
    )
    required_file_count = require_int(
        contract.get("required_file_count"), "content file count"
    )
    required_bytes = require_int(contract.get("required_bytes"), "content bytes")
    require(required_file_count > 0, "content file count must be positive")
    require(required_bytes > 0, "content bytes must be positive")
    required_artifacts = require_list(
        contract.get("required_receipt_artifacts"),
        "required content receipt artifacts",
    )
    require(
        len(required_artifacts) == 3
        and set(required_artifacts)
        == {"audit_plan.json", "content_freeze_audit.json", "content_freeze_audit.md"},
        "content receipt artifact ABI drift",
    )
    require(
        contract.get("separately_bound_run_state") == "run_state.json",
        "content state path drift",
    )
    require(
        contract.get("required_run_state") == "COMPLETE",
        "content run-state drift",
    )
    require(
        contract.get("full_scope_required") is True,
        "content full-scope requirement drift",
    )
    require(
        contract.get("pilot_may_omit_only_because_unpromotable") is True,
        "content pilot omission drift",
    )

    subset = require_mapping(
        contract.get("relation_source_subset"), "relation source subset"
    )
    table_keys = require_list(
        subset.get("required_table_keys"), "relation source table keys"
    )
    require(
        table_keys == ["main", "topics"], "relation source table-key order drift"
    )
    fields = require_list(
        subset.get("required_source_stat_fields"), "content source-stat fields"
    )
    require(
        fields == ["device", "inode", "mode", "bytes", "mtime_ns", "ctime_ns"],
        "content source-stat ABI drift",
    )
    require(
        subset.get("full_scope_policy") == "exact_relation_table_task_set",
        "full relation content bridge policy drift",
    )
    require(
        subset.get("pilot_policy")
        == "selected_relation_tasks_may_be_a_strict_subset",
        "pilot relation content bridge policy drift",
    )
    tables = require_mapping(config.get("tables"), "tables")
    expected_relation_files = sum(
        require_int(
            require_mapping(tables.get(key), f"tables.{key}").get("expected_files"),
            f"tables.{key}.expected_files",
        )
        for key in table_keys
    )
    expected_relation_bytes = sum(
        require_int(
            require_mapping(tables.get(key), f"tables.{key}").get("expected_bytes"),
            f"tables.{key}.expected_bytes",
        )
        for key in table_keys
    )
    require(
        subset.get("required_files") == expected_relation_files,
        "relation content subset file total drift",
    )
    require(
        subset.get("required_bytes") == expected_relation_bytes,
        "relation content subset byte total drift",
    )
    require(
        expected_relation_files <= required_file_count
        and expected_relation_bytes <= required_bytes,
        "relation content subset exceeds the complete content baseline",
    )


def _read_config_snapshot(path: Path) -> tuple[dict[str, Any], str, int]:
    _assert_no_symlink_components(path, field="config")
    require(path.is_file() and not path.is_symlink(), "config is not a regular file")
    payload, observed_config_sha256, _ = _read_regular_bytes_snapshot(
        path, field="config"
    )
    require(
        observed_config_sha256 == FROZEN_CONFIG_SHA256,
        "config SHA-256 differs from the frozen v0 contract",
    )
    try:
        config = require_mapping(yaml.safe_load(payload.decode("utf-8")), "config")
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RelationAuditError(f"cannot read config: {exc}") from exc
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    _validate_content_baseline_contract(config)
    codec = require_mapping(config.get("identifier_codec"), "identifier_codec")
    require(codec.get("topic_bits") == 14, "topic codec must remain 14-bit")
    require(codec.get("null_primary_topic_code") == 0, "null topic code drift")
    policy = require_mapping(config.get("relation_policy"), "relation_policy")
    require(
        policy.get("duplicate_policy") == "collapse_only_same_key_same_canonical_score",
        "duplicate policy drift",
    )
    require(
        policy.get("score_conflict_policy")
        == "block_without_choosing_first_max_mean_or_other_value",
        "score-conflict policy drift",
    )
    execution = require_mapping(config.get("execution"), "execution")
    require(
        execution.get("resume_verification")
        == "rehash_current_projected_payload_and_spool_segment",
        "resume verification drift",
    )
    return config, observed_config_sha256, len(payload)


def read_config(path: Path) -> dict[str, Any]:
    config, _, _ = _read_config_snapshot(path)
    return config


def _repo_commitment(
    contract: Mapping[str, Any], *, field: str
) -> tuple[Path, dict[str, Any], str]:
    relative = safe_relative_path(
        contract.get("relative_path"), f"{field}.relative_path"
    )
    path = secure_join_regular(REPO_ROOT, relative, field=field)
    payload, observed, _ = _read_regular_bytes_snapshot(path, field=field)
    require(
        observed == require_sha256(contract.get("sha256"), f"{field}.sha256"),
        f"{field} SHA-256 mismatch",
    )
    try:
        value = require_mapping(json.loads(payload.decode("utf-8")), field)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelationAuditError(f"cannot parse {field}") from exc
    require(
        value.get("schema_version") == contract.get("schema_version"),
        f"{field} schema mismatch",
    )
    return path, value, observed


def _require_gate_subset(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, field: str
) -> None:
    for key, value in expected.items():
        require(observed.get(key) == value, f"{field}.{key} mismatch")


def validate_parents(config: Mapping[str, Any]) -> dict[str, Any]:
    parents = require_mapping(config.get("parents"), "parents")
    _, inventory, inventory_hash = _repo_commitment(
        require_mapping(parents.get("inventory"), "parents.inventory"),
        field="inventory parent",
    )
    inventory_contract = require_mapping(parents.get("inventory"), "inventory contract")
    root = require_mapping(inventory.get("root"), "inventory.root")
    source_contract = require_mapping(config.get("source"), "source")
    require(root.get("id") == source_contract.get("root_id"), "source root ID mismatch")
    require(
        root.get("snapshot_date") == source_contract.get("snapshot_date"),
        "snapshot date mismatch",
    )
    require(
        root.get("inventory_fingerprint_sha256")
        == inventory_contract.get("inventory_fingerprint_sha256"),
        "inventory fingerprint mismatch",
    )
    require(
        root.get("stability_fingerprint_sha256")
        == inventory_contract.get("stability_fingerprint_sha256"),
        "inventory stability fingerprint mismatch",
    )
    require(
        inventory_contract.get("content_sha256_complete") is False,
        "parent identity boundary must remain incomplete",
    )

    _, grammar, grammar_hash = _repo_commitment(
        require_mapping(parents.get("grammar_valid_main_recount"), "grammar parent"),
        field="grammar-valid recount parent",
    )
    grammar_contract = require_mapping(
        parents.get("grammar_valid_main_recount"), "grammar contract"
    )
    require(
        grammar.get("status") == grammar_contract.get("required_status"),
        "grammar parent status mismatch",
    )
    _require_gate_subset(
        require_mapping(grammar.get("gate_status"), "grammar gates"),
        require_mapping(
            grammar_contract.get("required_gate_states"), "required grammar gates"
        ),
        field="grammar gate",
    )
    exact = require_mapping(grammar.get("exact_counts"), "grammar exact counts")
    _require_gate_subset(
        exact,
        require_mapping(
            grammar_contract.get("required_exact_counts"), "required exact counts"
        ),
        field="grammar exact count",
    )

    _, work_source, work_source_hash = _repo_commitment(
        require_mapping(parents.get("work_source_membership"), "work-source parent"),
        field="work-source parent",
    )
    work_contract = require_mapping(
        parents.get("work_source_membership"), "work-source contract"
    )
    require(
        work_source.get("status") == work_contract.get("required_status"),
        "work-source status mismatch",
    )
    require(
        work_source.get("provenance", {}).get("result_fingerprint_sha256")
        == work_contract.get("result_fingerprint_sha256"),
        "work-source fingerprint mismatch",
    )
    _require_gate_subset(
        require_mapping(work_source.get("gate_status"), "work-source gates"),
        require_mapping(
            work_contract.get("required_gate_states"), "required work-source gates"
        ),
        field="work-source gate",
    )

    _, taxonomy_report, taxonomy_hash = _repo_commitment(
        require_mapping(parents.get("taxonomy_reference"), "taxonomy parent"),
        field="taxonomy-reference parent",
    )
    taxonomy_contract = require_mapping(
        parents.get("taxonomy_reference"), "taxonomy contract"
    )
    require(
        taxonomy_report.get("status") == taxonomy_contract.get("required_status"),
        "taxonomy status mismatch",
    )
    require(
        taxonomy_report.get("provenance", {}).get("result_fingerprint_sha256")
        == taxonomy_contract.get("result_fingerprint_sha256"),
        "taxonomy fingerprint mismatch",
    )
    require(
        taxonomy_report.get("taxonomy", {}).get("canonical_sha256")
        == taxonomy_contract.get("taxonomy_canonical_sha256"),
        "taxonomy canonical hash mismatch",
    )
    _require_gate_subset(
        require_mapping(taxonomy_report.get("gate_status"), "taxonomy gates"),
        require_mapping(
            taxonomy_contract.get("required_gate_states"), "required taxonomy gates"
        ),
        field="taxonomy gate",
    )
    return {
        "inventory": inventory,
        "grammar": grammar,
        "work_source": work_source,
        "taxonomy_report": taxonomy_report,
        "hashes": {
            "inventory_manifest_sha256": inventory_hash,
            "grammar_valid_main_recount_sha256": grammar_hash,
            "work_source_membership_sha256": work_source_hash,
            "taxonomy_reference_sha256": taxonomy_hash,
            "work_source_result_fingerprint_sha256": str(
                work_contract.get("result_fingerprint_sha256")
            ),
            "taxonomy_reference_result_fingerprint_sha256": str(
                taxonomy_contract.get("result_fingerprint_sha256")
            ),
            "taxonomy_canonical_sha256": str(
                taxonomy_contract.get("taxonomy_canonical_sha256")
            ),
        },
    }


def _validate_content_freeze_parent(
    config: Mapping[str, Any], receipt_path: Path | None, *, required: bool
) -> dict[str, Any]:
    contract = require_mapping(
        config.get("parents", {}).get("current_content_baseline"),
        "content-freeze parent",
    )
    if receipt_path is None:
        require(not required, "full scope requires current-content outputs receipt")
        return {
            "commitment": {
                "status": "OMITTED_UNPROMOTABLE_PILOT",
                "current_content_baseline_component": (
                    "NOT_REQUIRED_UNPROMOTABLE_PILOT"
                ),
                "historical_content_identity": "UNVERIFIED",
                "operator_paths_redacted": True,
                "relation_source_rich_stat_bridge": "OMITTED_PILOT",
            },
            "runtime": None,
        }

    receipt_path = _absolute(receipt_path)
    require(
        receipt_path.name == contract.get("expected_basename"),
        "content-freeze receipt basename mismatch",
    )
    _validate_trusted_content_output_chain(receipt_path.parent)
    receipt_payload, receipt_sha, receipt_identity = _read_regular_bytes_snapshot(
        receipt_path, field="content-freeze receipt"
    )
    receipt = _parse_content_freeze_json(
        receipt_payload, field="content-freeze receipt"
    )
    require(
        set(receipt)
        == {
            "schema_version",
            "status",
            "task_fingerprint_sha256",
            "plan_sha256",
            "result_fingerprint_sha256",
            "content_tree_sha256",
            "checkpoint_tree",
            "artifacts",
            "completion_contract",
            "run_state_bound_separately",
            "path_redacted",
        },
        "content-freeze receipt key set differs",
    )
    require(
        receipt.get("schema_version") == contract.get("schema_version"),
        "content-freeze receipt schema mismatch",
    )
    require(
        receipt.get("status") == contract.get("required_status"),
        "content-freeze status mismatch",
    )
    require(
        receipt.get("plan_sha256") == contract.get("required_plan_sha256"),
        "content-freeze plan anchor mismatch",
    )
    require(
        receipt.get("task_fingerprint_sha256")
        == contract.get("required_task_fingerprint_sha256"),
        "content-freeze task anchor mismatch",
    )
    checkpoint_tree = require_mapping(
        receipt.get("checkpoint_tree"), "content-freeze checkpoint tree"
    )
    require(
        set(checkpoint_tree) == {"files", "bytes", "tree_sha256"},
        "content-freeze checkpoint tree key set differs",
    )
    require_sha256(
        checkpoint_tree.get("tree_sha256"), "content-freeze checkpoint tree SHA-256"
    )
    checkpoint_bytes = require_int(
        checkpoint_tree.get("bytes"), "content-freeze checkpoint tree bytes"
    )
    require(
        checkpoint_bytes > 0,
        "content-freeze checkpoint tree bytes must be positive",
    )
    require(
        checkpoint_tree.get("files") == contract.get("required_file_count"),
        "content-freeze file count mismatch",
    )
    require(
        receipt.get("run_state_bound_separately") is True,
        "content-freeze state binding mismatch",
    )
    require(
        receipt.get("completion_contract")
        == "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST",
        "content-freeze completion contract mismatch",
    )
    require(
        receipt.get("path_redacted") is True,
        "content-freeze receipt path redaction mismatch",
    )

    artifact_items = [
        require_mapping(item, "content-freeze artifact")
        for item in require_list(receipt.get("artifacts"), "content-freeze artifacts")
    ]
    artifact_paths = [str(item.get("relative_path")) for item in artifact_items]
    require(
        len(artifact_paths) == len(set(artifact_paths)),
        "content-freeze receipt has duplicate artifacts",
    )
    artifacts = dict(zip(artifact_paths, artifact_items, strict=True))
    required_artifacts = set(
        str(item)
        for item in require_list(
            contract.get("required_receipt_artifacts"),
            "required content-freeze artifacts",
        )
    )
    require(
        required_artifacts == set(artifacts),
        "content-freeze receipt artifact set mismatch",
    )

    runtime_files: dict[str, dict[str, Any]] = {}
    decoded: dict[str, Any] = {}
    for relative_text in sorted(required_artifacts):
        relative = safe_relative_path(
            relative_text, "content-freeze artifact path"
        )
        artifact_path = secure_join_regular(
            receipt_path.parent, relative, field="content-freeze artifact"
        )
        payload, observed, identity = _read_regular_bytes_snapshot(
            artifact_path, field="content-freeze artifact"
        )
        artifact = artifacts[relative_text]
        require(
            set(artifact) == {"relative_path", "bytes", "sha256"},
            "content-freeze artifact key set differs",
        )
        require_sha256(
            artifact.get("sha256"), "content-freeze artifact SHA-256"
        )
        require(
            require_int(artifact.get("bytes"), "content-freeze artifact bytes") >= 0,
            "content-freeze artifact bytes are negative",
        )
        require(
            observed == artifact.get("sha256"),
            "content-freeze artifact hash mismatch",
        )
        require(
            len(payload) == artifact.get("bytes"),
            "content-freeze artifact byte mismatch",
        )
        runtime_files[relative_text] = {
            "path": artifact_path,
            "sha256": observed,
            "rich_identity": identity,
        }
        if relative.suffix == ".json":
            decoded[relative_text] = _parse_content_freeze_json(
                payload, field="content-freeze artifact JSON"
            )

    plan = require_mapping(decoded.get("audit_plan.json"), "content-freeze plan")
    report = require_mapping(
        decoded.get("content_freeze_audit.json"), "content-freeze report"
    )
    require(
        plan.get("schema_version") == "openalex_g1_content_freeze_plan.v1",
        "content-freeze plan schema mismatch",
    )
    require(plan.get("full_scope") is True, "content-freeze plan is not full scope")
    require(
        plan.get("max_files") is None, "content-freeze full plan has max-files"
    )
    require(
        plan.get("plan_sha256") == contract.get("required_plan_sha256"),
        "content-freeze plan digest mismatch",
    )
    plan_payload = dict(plan)
    plan_digest = plan_payload.pop("plan_sha256", None)
    require(
        plan_digest == sha256_bytes(canonical_json_bytes(plan_payload)),
        "content-freeze plan self-digest mismatch",
    )
    require(
        plan.get("task_fingerprint_sha256")
        == contract.get("required_task_fingerprint_sha256"),
        "content-freeze plan task mismatch",
    )
    require(
        plan.get("declared_file_count") == contract.get("required_file_count"),
        "content-freeze declared files mismatch",
    )
    require(
        plan.get("declared_bytes") == contract.get("required_bytes"),
        "content-freeze declared bytes mismatch",
    )
    require(
        plan.get("selected_file_count") == contract.get("required_file_count"),
        "content-freeze selected files mismatch",
    )
    require(
        plan.get("selected_bytes") == contract.get("required_bytes"),
        "content-freeze selected bytes mismatch",
    )
    content_tasks = [
        require_mapping(item, "content-freeze task")
        for item in require_list(plan.get("tasks"), "content-freeze tasks")
    ]
    require(
        len(content_tasks) == contract.get("required_file_count"),
        "content-freeze task count mismatch",
    )

    require(
        report.get("schema_version") == "openalex_g1_content_freeze_report.v1",
        "content-freeze report schema mismatch",
    )
    require(
        report.get("status") == contract.get("required_status"),
        "content-freeze report status mismatch",
    )
    require(
        report.get("scope", {}).get("full_scope") is True,
        "content-freeze report is not full scope",
    )
    require(
        report.get("pass_current_content_baseline_component") is True,
        "content-freeze report did not pass component",
    )
    baseline = require_mapping(report.get("content_baseline"), "content baseline")
    require(
        baseline.get("files") == contract.get("required_file_count"),
        "content baseline files mismatch",
    )
    require(
        baseline.get("bytes") == contract.get("required_bytes"),
        "content baseline bytes mismatch",
    )
    require(
        baseline.get("historical_identity_established") is False,
        "historical identity overclaim",
    )
    require(
        baseline.get("prior_component_run_identity_established") is False,
        "prior-run identity overclaim",
    )
    required_content_gates = {
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
    require(
        report.get("gate_status") == required_content_gates,
        "content baseline gate mismatch",
    )
    scope = require_mapping(report.get("scope"), "content-freeze scope")
    require(
        scope.get("full_scope") == plan.get("full_scope"),
        "content report/plan full-scope mismatch",
    )
    require(
        scope.get("max_files") == plan.get("max_files"),
        "content report/plan max-files mismatch",
    )
    require(
        scope.get("declared_files")
        == plan.get("declared_file_count")
        == contract.get("required_file_count"),
        "content report declared files mismatch",
    )
    require(
        scope.get("declared_bytes")
        == plan.get("declared_bytes")
        == contract.get("required_bytes"),
        "content report declared bytes mismatch",
    )
    require(
        scope.get("selected_files")
        == plan.get("selected_file_count")
        == contract.get("required_file_count"),
        "content report selected files mismatch",
    )
    require(
        scope.get("selected_bytes")
        == plan.get("selected_bytes")
        == contract.get("required_bytes"),
        "content report selected bytes mismatch",
    )
    require(scope.get("gpu_used") is False, "content report GPU boundary mismatch")
    require(
        report.get("parent_inventory") == plan.get("parent_inventory"),
        "content report/plan parent inventory mismatch",
    )
    require(
        report.get("execution", {}).get("path_redacted") is True,
        "content report execution path redaction mismatch",
    )
    require(
        report.get("provenance", {}).get("path_redacted") is True,
        "content report provenance path redaction mismatch",
    )
    content_provenance = require_mapping(
        report.get("provenance"), "content-freeze report provenance"
    )
    parent_manifest_sha256 = require_sha256(
        plan.get("parent_inventory", {}).get("manifest_sha256"),
        "content-freeze parent manifest SHA-256",
    )
    require(
        content_provenance.get("parent_manifest_sha256")
        == parent_manifest_sha256,
        "content-freeze report parent manifest provenance mismatch",
    )
    require(
        content_provenance.get("task_fingerprint_sha256")
        == plan.get("task_fingerprint_sha256")
        == contract.get("required_task_fingerprint_sha256"),
        "content-freeze report task provenance mismatch",
    )
    require(
        content_provenance.get("plan_sha256")
        == plan.get("plan_sha256")
        == contract.get("required_plan_sha256"),
        "content-freeze report plan provenance mismatch",
    )
    require(
        receipt.get("content_tree_sha256") == baseline.get("content_tree_sha256"),
        "content-freeze content tree mismatch",
    )
    require(
        receipt.get("result_fingerprint_sha256")
        == report.get("provenance", {}).get("result_fingerprint_sha256"),
        "content-freeze result fingerprint mismatch",
    )
    deterministic_core = {
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "full_scope": plan.get("full_scope"),
        "max_files": plan.get("max_files"),
        "parent_manifest_sha256": plan.get("parent_inventory", {}).get(
            "manifest_sha256"
        ),
        "files": baseline.get("files"),
        "bytes": baseline.get("bytes"),
        "content_tree_sha256": baseline.get("content_tree_sha256"),
        "gate_status": report.get("gate_status"),
        "claim_boundary": report.get("claim_boundary"),
        "same_size_restored_mtime_limitation": report.get(
            "same_size_restored_mtime_limitation"
        ),
        "checkpoint_trust_limitation": report.get("checkpoint_trust_limitation"),
        "publication_completion_contract": report.get("publication", {}).get(
            "completion_contract"
        ),
    }
    computed_content_result = sha256_bytes(canonical_json_bytes(deterministic_core))
    require(
        report.get("provenance", {}).get("result_fingerprint_sha256")
        == computed_content_result,
        "content-freeze report result fingerprint mismatch",
    )
    require(
        receipt.get("result_fingerprint_sha256") == computed_content_result,
        "content-freeze receipt result fingerprint mismatch",
    )
    report_checkpoints = require_mapping(
        report.get("checkpoint_evidence"), "content-freeze checkpoint evidence"
    )
    report_checkpoint_bytes = require_int(
        report_checkpoints.get("checkpoint_bytes"),
        "content-freeze report checkpoint bytes",
    )
    require(
        report_checkpoint_bytes > 0,
        "content-freeze report checkpoint bytes must be positive",
    )
    require(
        report_checkpoint_bytes == checkpoint_bytes,
        "content-freeze report/receipt checkpoint byte count mismatch",
    )
    require(
        checkpoint_tree
        == {
            "files": report_checkpoints.get("checkpoint_files"),
            "bytes": report_checkpoints.get("checkpoint_bytes"),
            "tree_sha256": report_checkpoints.get("checkpoint_tree_sha256"),
        },
        "content-freeze checkpoint tree mismatch",
    )

    state_relative = safe_relative_path(
        contract.get("separately_bound_run_state"), "content-freeze state path"
    )
    state_path = secure_join_regular(
        receipt_path.parent, state_relative, field="content-freeze run state"
    )
    state_payload, state_sha, state_identity = _read_regular_bytes_snapshot(
        state_path, field="content-freeze run state"
    )
    state = _parse_content_freeze_json(
        state_payload, field="content-freeze run state"
    )
    require(
        state.get("schema_version") == "openalex_g1_content_freeze_state.v1",
        "content-freeze state schema mismatch",
    )
    require(
        state.get("status") == contract.get("required_run_state"),
        "content-freeze state is not COMPLETE",
    )
    require(
        state.get("plan_sha256") == contract.get("required_plan_sha256"),
        "content-freeze state plan mismatch",
    )
    require(
        state.get("task_fingerprint_sha256")
        == contract.get("required_task_fingerprint_sha256"),
        "content-freeze state task mismatch",
    )
    require(
        state.get("completed_files") == contract.get("required_file_count"),
        "content-freeze state files mismatch",
    )
    require(
        state.get("completed_bytes") == contract.get("required_bytes"),
        "content-freeze state bytes mismatch",
    )
    require(
        state.get("total_files") == contract.get("required_file_count"),
        "content-freeze state total files mismatch",
    )
    require(
        state.get("total_bytes") == contract.get("required_bytes"),
        "content-freeze state total bytes mismatch",
    )
    require(
        state.get("path_redacted") is True,
        "content-freeze state path redaction mismatch",
    )
    require("error" not in state, "content-freeze COMPLETE state contains error")
    runtime_files[state_relative.as_posix()] = {
        "path": state_path,
        "sha256": state_sha,
        "rich_identity": state_identity,
    }

    tables = require_mapping(config.get("tables"), "tables")
    subset_contract = require_mapping(
        contract.get("relation_source_subset"), "relation source subset"
    )
    table_keys = [
        str(item)
        for item in require_list(
            subset_contract.get("required_table_keys"), "relation table keys"
        )
    ]
    table_names = {
        str(require_mapping(tables.get(key), f"tables.{key}").get("name")): key
        for key in table_keys
    }
    require(
        len(table_names) == len(table_keys),
        "relation source table names are not unique",
    )
    source_stat_fields = tuple(
        str(item)
        for item in require_list(
            subset_contract.get("required_source_stat_fields"),
            "relation content source-stat fields",
        )
    )
    relation_source_tasks: dict[str, dict[str, Any]] = {}
    per_table_counts = {key: 0 for key in table_keys}
    per_table_bytes = {key: 0 for key in table_keys}
    for item in content_tasks:
        table_name = str(item.get("table_name"))
        if table_name not in table_names:
            continue
        table_key = table_names[table_name]
        relative = safe_relative_path(
            item.get("relative_path"), "content-freeze relation task path"
        ).as_posix()
        require(
            relative not in relation_source_tasks,
            "content-freeze relation task paths are duplicated",
        )
        raw_stat = require_mapping(
            item.get("source_stat"), "content-freeze relation task source stat"
        )
        require(
            set(raw_stat) == set(source_stat_fields),
            "content-freeze source-stat ABI mismatch",
        )
        source_stat = {
            field: require_int(raw_stat.get(field), f"content source stat {field}")
            for field in source_stat_fields
        }
        relation_source_tasks[relative] = {
            "table_key": table_key,
            "table_name": table_name,
            "source_stat": source_stat,
        }
        per_table_counts[table_key] += 1
        per_table_bytes[table_key] += source_stat["bytes"]

    for table_key in table_keys:
        table_contract = require_mapping(tables.get(table_key), f"tables.{table_key}")
        require(
            per_table_counts[table_key] == table_contract.get("expected_files"),
            f"content-freeze {table_key} relation file count mismatch",
        )
        require(
            per_table_bytes[table_key] == table_contract.get("expected_bytes"),
            f"content-freeze {table_key} relation byte count mismatch",
        )
    require(
        len(relation_source_tasks) == subset_contract.get("required_files"),
        "content-freeze relation source subset file count mismatch",
    )
    require(
        sum(per_table_bytes.values()) == subset_contract.get("required_bytes"),
        "content-freeze relation source subset byte count mismatch",
    )

    commitment = {
        "status": "PASS_BOUND_CURRENT_CONTENT_BASELINE_RECEIPT",
        "receipt_sha256": receipt_sha,
        "plan_sha256": receipt["plan_sha256"],
        "task_fingerprint_sha256": receipt["task_fingerprint_sha256"],
        "content_tree_sha256": receipt.get("content_tree_sha256"),
        "files": checkpoint_tree["files"],
        "bytes": baseline["bytes"],
        "current_content_baseline_component": "PASS",
        "historical_content_identity": "UNVERIFIED",
        "operator_paths_redacted": True,
    }
    return {
        "commitment": commitment,
        "runtime": {
            "receipt": {
                "path": receipt_path,
                "sha256": receipt_sha,
                "rich_identity": receipt_identity,
            },
            "files": runtime_files,
            "relation_source_tasks": relation_source_tasks,
        },
    }


def _resolve_content_receipt(
    config: Mapping[str, Any], explicit: Path | None
) -> Path | None:
    if explicit is not None:
        return explicit
    contract = require_mapping(
        config.get("parents", {}).get("current_content_baseline"),
        "content-freeze parent",
    )
    raw = os.environ.get(str(contract.get("receipt_path_env")), "")
    return None if not raw else Path(raw)


def load_topic_ids(
    taxonomy_root: Path, contract: Mapping[str, Any], *, topic_bits: int
) -> tuple[np.ndarray, np.ndarray, str]:
    pattern = re.compile(r"^https://openalex[.]org/T([1-9][0-9]*)$")
    numeric_ids: list[int] = []
    commitments: list[dict[str, Any]] = []
    for raw in require_list(contract.get("files"), "taxonomy.files"):
        record = require_mapping(raw, "taxonomy file")
        relative = safe_relative_path(record.get("relative_path"), "taxonomy file path")
        path = secure_join_regular(taxonomy_root, relative, field="taxonomy topic file")
        compressed, observed_hash, _ = _read_regular_bytes_snapshot(
            path, field="taxonomy topic file"
        )
        require(
            observed_hash
            == require_sha256(record.get("sha256"), "taxonomy file sha256"),
            f"taxonomy file SHA-256 mismatch: {relative}",
        )
        rows = 0
        try:
            text = gzip.decompress(compressed).decode("utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                value = require_mapping(json.loads(line), "taxonomy topic row")
                match = pattern.fullmatch(str(value.get("id", "")))
                require(match is not None, "taxonomy topic ID grammar mismatch")
                numeric_ids.append(int(match.group(1)))
                rows += 1
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RelationAuditError(f"cannot read taxonomy topic file: {exc}") from exc
        require(
            rows == require_int(record.get("expected_rows"), "taxonomy expected rows"),
            "taxonomy topic row count mismatch",
        )
        commitments.append(
            {
                "relative_path": str(relative),
                "sha256": observed_hash,
                "rows": rows,
            }
        )
    numeric = np.asarray(sorted(numeric_ids), dtype="<u8")
    require(
        len(numeric)
        == require_int(contract.get("expected_topic_count"), "topic count"),
        "taxonomy topic count mismatch",
    )
    require(
        len(np.unique(numeric)) == len(numeric), "duplicate taxonomy topic numeric ID"
    )
    require(
        int(numeric[0])
        == require_int(contract.get("expected_min_numeric_id"), "topic min"),
        "taxonomy topic minimum mismatch",
    )
    require(
        int(numeric[-1])
        == require_int(contract.get("expected_max_numeric_id"), "topic max"),
        "taxonomy topic maximum mismatch",
    )
    mask_limit = (1 << topic_bits) - 1
    require(int(numeric[-1]) <= mask_limit, "taxonomy topic ID exceeds codec")
    allowed = np.zeros(mask_limit + 1, dtype=bool)
    allowed[numeric.astype(np.int64)] = True
    commitment_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "numeric_ids": [int(value) for value in numeric],
                "files": commitments,
                "topic_bits": topic_bits,
            }
        )
    )
    return numeric, allowed, commitment_hash


def _projected_schema_from_record(
    record: Mapping[str, Any], columns: Sequence[str]
) -> dict[str, dict[str, Any]]:
    schema = require_mapping(record.get("schema_fields"), "manifest schema_fields")
    projected: dict[str, dict[str, Any]] = {}
    for column in columns:
        require(column in schema, f"manifest schema missing {column}")
        projected[column] = dict(require_mapping(schema[column], f"schema {column}"))
    return projected


def _task_from_mapping(value: Mapping[str, Any]) -> FileTask:
    return FileTask(
        table_key=str(value["table_key"]),
        table_name=str(value["table_name"]),
        file_index=require_int(value["file_index"], "file index"),
        relative_path=str(value["relative_path"]),
        bytes=require_int(value["bytes"], "file bytes"),
        mtime_ns=require_int(value["mtime_ns"], "file mtime"),
        footer_rows=require_int(value["footer_rows"], "footer rows"),
        projected_columns=tuple(str(item) for item in value["projected_columns"]),
        projected_schema={
            str(key): dict(item) for key, item in value["projected_schema"].items()
        },
        projected_schema_sha256=str(value["projected_schema_sha256"]),
        source_rich_identity={
            str(key): require_int(item, f"source rich identity {key}")
            for key, item in require_mapping(
                value["source_rich_identity"], "source rich identity"
            ).items()
        },
        task_sha256=str(value["task_sha256"]),
    )


def _source_tree_evidence(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = [
        {
            "table_key": str(value["table_key"]),
            "relative_path": str(value["relative_path"]),
            "rich_identity": dict(value["source_rich_identity"]),
        }
        for value in tasks
    ]
    return {
        "file_count": len(records),
        "rich_stat_fields": [
            "device",
            "inode",
            "mode",
            "uid",
            "gid",
            "links",
            "bytes",
            "mtime_ns",
            "ctime_ns",
        ],
        "tree_sha256": sha256_bytes(canonical_json_bytes(records)),
        "records": records,
    }


def _resource_estimate(
    config: Mapping[str, Any], table_plans: Mapping[str, Any]
) -> dict[str, Any]:
    execution = require_mapping(config.get("execution"), "execution")
    contract = require_mapping(
        execution.get("resource_preflight"), "execution.resource_preflight"
    )
    event_bytes = require_int(contract.get("event_bytes_per_row"), "event bytes")
    dedup_bytes = require_int(
        contract.get("deduplicated_edge_bytes_per_row"), "dedup bytes"
    )
    require(event_bytes == EVENT_DTYPE.itemsize, "event resource ABI drift")
    require(dedup_bytes == DEDUP_DTYPE.itemsize, "dedup resource ABI drift")
    main_rows = require_int(
        table_plans["main"]["selected_footer_rows"], "selected main rows"
    )
    topic_rows = require_int(
        table_plans["topics"]["selected_footer_rows"], "selected topic rows"
    )
    event_upper_bound = main_rows + topic_rows
    raw_bytes = event_upper_bound * event_bytes
    sorted_bytes = event_upper_bound * event_bytes
    dedup_upper_bytes = topic_rows * dedup_bytes
    artifact_upper_bytes = raw_bytes + sorted_bytes + dedup_upper_bytes
    disk_multiplier = require_int(
        contract.get("disk_safety_multiplier_milli"), "disk multiplier"
    )
    memory_multiplier = require_int(
        contract.get("memory_safety_multiplier_milli"), "memory multiplier"
    )
    require(disk_multiplier >= 1000, "disk multiplier must be at least 1000")
    require(memory_multiplier >= 1000, "memory multiplier must be at least 1000")
    disk_reserve = require_int(contract.get("disk_reserve_bytes"), "disk reserve")
    memory_reserve = require_int(contract.get("memory_reserve_bytes"), "memory reserve")
    require(disk_reserve >= 0 and memory_reserve >= 0, "resource reserve is negative")
    required_disk = (
        artifact_upper_bytes * disk_multiplier + 999
    ) // 1000 + disk_reserve
    required_memory = (raw_bytes * memory_multiplier + 999) // 1000 + memory_reserve
    return {
        "selected_main_footer_rows": main_rows,
        "selected_topic_footer_rows": topic_rows,
        "event_row_upper_bound": event_upper_bound,
        "raw_spool_bytes": raw_bytes,
        "sorted_event_bytes": sorted_bytes,
        "deduplicated_edge_bytes_upper_bound": dedup_upper_bytes,
        "artifact_bytes_upper_bound": artifact_upper_bytes,
        "peak_in_memory_sort_array_bytes": raw_bytes,
        "disk_safety_multiplier_milli": disk_multiplier,
        "disk_reserve_bytes": disk_reserve,
        "required_free_disk_bytes": required_disk,
        "memory_safety_multiplier_milli": memory_multiplier,
        "memory_reserve_bytes": memory_reserve,
        "required_available_memory_bytes": required_memory,
        "full_scope_only": contract.get("full_scope_only") is True,
        "pilot_promotion_allowed": contract.get("pilot_promotion_allowed") is True,
    }


def _available_disk_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _available_memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError) as exc:
        raise RelationAuditError("cannot determine available memory") from exc
    require(pages >= 0 and page_size > 0, "available memory probe is invalid")
    return pages * page_size


def enforce_resource_preflight(plan: Mapping[str, Any], *, output_dir: Path) -> None:
    estimate = require_mapping(plan.get("resource_estimate"), "resource estimate")
    require(
        estimate.get("pilot_promotion_allowed") is False,
        "pilot resource contract may not allow promotion",
    )
    if not bool(plan.get("full_scope")):
        return
    required_disk = require_int(
        estimate.get("required_free_disk_bytes"), "required free disk"
    )
    required_memory = require_int(
        estimate.get("required_available_memory_bytes"),
        "required available memory",
    )
    require(
        _available_disk_bytes(output_dir) >= required_disk,
        "BLOCKED_FULL_EXECUTION_RESOURCE_PREFLIGHT: insufficient free disk",
    )
    require(
        _available_memory_bytes() >= required_memory,
        "BLOCKED_FULL_EXECUTION_RESOURCE_PREFLIGHT: insufficient available memory",
    )


def _full_scope_from_arguments(
    *,
    max_files_per_table: int | None,
    hash_bucket_count: int | None,
    hash_bucket_index: int | None,
) -> bool:
    require(
        max_files_per_table is None or max_files_per_table > 0,
        "max files must be positive",
    )
    require(
        (hash_bucket_count is None) == (hash_bucket_index is None),
        "hash bucket count and index must be supplied together",
    )
    if hash_bucket_count is not None and hash_bucket_index is not None:
        require(
            max_files_per_table is None,
            "max-files and hash-bucket pilots may not be combined",
        )
        require(hash_bucket_count > 1, "hash bucket count must exceed one")
        require(
            hash_bucket_count & (hash_bucket_count - 1) == 0,
            "hash bucket count must be a power of two",
        )
        require(
            0 <= hash_bucket_index < hash_bucket_count,
            "hash bucket index is out of range",
        )
    return max_files_per_table is None and hash_bucket_count is None


def build_plan(
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    verified_config_sha256: str,
    verified_config_bytes: int,
    source_root: Path,
    max_files_per_table: int | None,
    hash_bucket_count: int | None,
    hash_bucket_index: int | None,
    batch_size: int,
    aggregation_chunk_rows: int,
    taxonomy_topic_set_sha256: str,
) -> dict[str, Any]:
    require(
        verified_config_sha256 == FROZEN_CONFIG_SHA256,
        "plan config digest differs from the verified frozen snapshot",
    )
    require(verified_config_bytes > 0, "verified config snapshot is empty")
    require(
        max_files_per_table is None or max_files_per_table > 0,
        "max files must be positive",
    )
    require(batch_size > 0, "batch size must be positive")
    require(aggregation_chunk_rows > 0, "aggregation chunk rows must be positive")
    if hash_bucket_count is None or hash_bucket_index is None:
        require(
            hash_bucket_count is None and hash_bucket_index is None,
            "hash bucket count and index must be supplied together",
        )
    else:
        require(
            max_files_per_table is None,
            "max-files and hash-bucket pilots may not be combined",
        )
        require(hash_bucket_count > 1, "hash bucket count must exceed one")
        require(
            hash_bucket_count & (hash_bucket_count - 1) == 0,
            "hash bucket count must be a power of two",
        )
        require(
            0 <= hash_bucket_index < hash_bucket_count,
            "hash bucket index is out of range",
        )

    manifest_tables = {
        str(value["name"]): value
        for value in require_list(
            context["inventory"].get("tables"), "inventory tables"
        )
    }
    tables = require_mapping(config.get("tables"), "tables")
    tasks: list[dict[str, Any]] = []
    table_plans: dict[str, Any] = {}
    for table_key in ("main", "topics"):
        contract = require_mapping(tables.get(table_key), f"tables.{table_key}")
        name = str(contract.get("name"))
        require(name in manifest_tables, f"inventory missing {name}")
        manifest_table = require_mapping(
            manifest_tables[name], f"inventory table {name}"
        )
        require(
            manifest_table.get("file_count") == contract.get("expected_files"),
            f"{name} file-count mismatch",
        )
        require(
            manifest_table.get("footer_rows") == contract.get("expected_footer_rows"),
            f"{name} footer-row mismatch",
        )
        require(
            manifest_table.get("bytes") == contract.get("expected_bytes"),
            f"{name} byte-count mismatch",
        )
        files = sorted(
            [
                require_mapping(value, "inventory file")
                for value in require_list(
                    manifest_table.get("files"), "inventory files"
                )
            ],
            key=lambda value: str(value.get("relative_path")),
        )
        selected = files if max_files_per_table is None else files[:max_files_per_table]
        require(
            len(selected) < LOCATOR_FILE_LIMIT,
            f"{name} exceeds locator file-index capacity",
        )
        columns = tuple(
            str(item)
            for item in require_list(
                contract.get("projected_columns"), "projected columns"
            )
        )
        types = require_mapping(contract.get("projected_types"), "projected types")
        selected_rows = 0
        selected_bytes = 0
        for file_index, record in enumerate(selected):
            footer_rows = require_int(record.get("footer_rows"), "file footer rows")
            require(
                footer_rows <= LOCATOR_ROW_MASK + 1, "file exceeds locator row capacity"
            )
            projected_schema = _projected_schema_from_record(record, columns)
            for column in columns:
                require(
                    projected_schema[column].get("type") == types.get(column),
                    f"projected type mismatch: {column}",
                )
            schema_hash = sha256_bytes(canonical_json_bytes(projected_schema))
            relative = safe_relative_path(
                record.get("relative_path"), "inventory source file"
            )
            source_path = secure_join_regular(
                source_root, relative, field="source Parquet"
            )
            rich_identity = _rich_regular_identity(source_path, field="source Parquet")
            require(
                rich_identity["bytes"] == require_int(record.get("bytes"), "file bytes")
                and rich_identity["mtime_ns"]
                == require_int(record.get("mtime_ns"), "file mtime"),
                "source rich identity differs from inventory",
            )
            task_core = {
                "table_key": table_key,
                "table_name": name,
                "file_index": file_index,
                "relative_path": str(record.get("relative_path")),
                "bytes": require_int(record.get("bytes"), "file bytes"),
                "mtime_ns": require_int(record.get("mtime_ns"), "file mtime"),
                "footer_rows": footer_rows,
                "projected_columns": list(columns),
                "projected_schema": projected_schema,
                "projected_schema_sha256": schema_hash,
                "source_rich_identity": rich_identity,
            }
            task_core["task_sha256"] = sha256_bytes(canonical_json_bytes(task_core))
            tasks.append(task_core)
            selected_rows += footer_rows
            selected_bytes += int(record["bytes"])
        table_plans[table_key] = {
            "name": name,
            "parent_files": len(files),
            "selected_files": len(selected),
            "parent_footer_rows": int(manifest_table["footer_rows"]),
            "selected_footer_rows": selected_rows,
            "parent_bytes": int(manifest_table["bytes"]),
            "selected_bytes": selected_bytes,
        }

    full_scope = max_files_per_table is None and hash_bucket_count is None
    pilot_mode = None
    if max_files_per_table is not None:
        pilot_mode = "max_files_parser_smoke"
    elif hash_bucket_count is not None:
        pilot_mode = "full_file_scope_hash_bucket"
    plan = {
        "schema_version": PLAN_SCHEMA,
        "label": config.get("label"),
        "full_scope": full_scope,
        "pilot_mode": pilot_mode,
        "max_files_per_table": max_files_per_table,
        "hash_bucket": (
            None
            if hash_bucket_count is None
            else {
                "algorithm": "splitmix64-v1",
                "bucket_count": hash_bucket_count,
                "bucket_index": hash_bucket_index,
                "equality_key_remains_full_packed_key": True,
            }
        ),
        "batch_size": batch_size,
        "aggregation_chunk_rows": aggregation_chunk_rows,
        # Keep the in-memory plan JSON-shaped so a decoded persisted plan is
        # exactly comparable during resume (NumPy exposes tuple descriptors).
        "event_dtype": [list(item) for item in EVENT_DTYPE.descr],
        "dedup_dtype": [list(item) for item in DEDUP_DTYPE.descr],
        "codec": config.get("identifier_codec"),
        "parent_hashes": context["hashes"],
        "taxonomy_topic_set_sha256": taxonomy_topic_set_sha256,
        "config_sha256": verified_config_sha256,
        "config_snapshot_bytes": verified_config_bytes,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "tables": table_plans,
        "tasks": tasks,
        "source_tree_start": _source_tree_evidence(tasks),
        "resource_estimate": _resource_estimate(config, table_plans),
    }
    plan["task_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


def _plan_fingerprint(plan: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {key: value for key, value in plan.items() if key != "task_fingerprint_sha256"}
        )
    )


def _relation_source_bridge(
    plan: Mapping[str, Any], content_context: Mapping[str, Any]
) -> dict[str, Any] | str:
    runtime = content_context.get("runtime")
    if runtime is None:
        return "OMITTED_PILOT"
    frozen = require_mapping(
        runtime.get("relation_source_tasks"), "frozen relation source tasks"
    )
    common_fields = ("device", "inode", "mode", "bytes", "mtime_ns", "ctime_ns")
    current: dict[str, dict[str, Any]] = {}
    for raw in require_list(plan.get("tasks"), "relation plan tasks"):
        item = require_mapping(raw, "relation plan task")
        relative = safe_relative_path(
            item.get("relative_path"), "relation plan task path"
        ).as_posix()
        require(relative not in current, "relation plan task paths are duplicated")
        rich = require_mapping(
            item.get("source_rich_identity"), "relation plan source identity"
        )
        identity = {
            field: (
                stat.S_IMODE(require_int(rich.get(field), f"source identity {field}"))
                if field == "mode"
                else require_int(rich.get(field), f"source identity {field}")
            )
            for field in common_fields
        }
        current[relative] = {
            "table_key": str(item.get("table_key")),
            "rich_identity": identity,
        }

    if bool(plan.get("full_scope")):
        require(
            set(current) == set(frozen),
            "content-freeze relation task set differs from full relation plan",
        )
        status = "PASS_EXACT_RELATION_TASK_SET_AND_RICH_STAT_EQUALITY"
        task_set_policy = "EXACT_FULL_RELATION_TASK_SET"
    else:
        require(
            set(current).issubset(set(frozen)),
            "pilot relation tasks are not a subset of the content-freeze plan",
        )
        status = "PASS_SELECTED_PILOT_SUBSET_AND_RICH_STAT_EQUALITY"
        task_set_policy = "SELECTED_UNPROMOTABLE_PILOT_SUBSET"

    table_counts: dict[str, int] = {}
    table_bytes: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for relative in sorted(current):
        frozen_item = require_mapping(
            frozen.get(relative), "content-freeze relation source task"
        )
        require(
            current[relative]["table_key"] == frozen_item.get("table_key"),
            "content-freeze relation task table mismatch",
        )
        frozen_identity = require_mapping(
            frozen_item.get("source_stat"), "content-freeze relation source stat"
        )
        require(
            set(frozen_identity) == set(common_fields),
            "content-freeze source-stat ABI mismatch",
        )
        require(
            current[relative]["rich_identity"] == frozen_identity,
            "current source rich identity differs from content-freeze plan",
        )
        table_key = current[relative]["table_key"]
        table_counts[table_key] = table_counts.get(table_key, 0) + 1
        table_bytes[table_key] = table_bytes.get(table_key, 0) + int(
            frozen_identity["bytes"]
        )
        records.append(
            {
                "table_key": table_key,
                "relative_path": relative,
                "rich_identity": current[relative]["rich_identity"],
            }
        )
    return {
        "status": status,
        "task_set_policy": task_set_policy,
        "files": len(records),
        "bytes": sum(table_bytes.values()),
        "table_file_counts": dict(sorted(table_counts.items())),
        "table_bytes": dict(sorted(table_bytes.items())),
        "tree_sha256": sha256_bytes(canonical_json_bytes(records)),
        "source_stat_fields": list(common_fields),
        "mode_representation": (
            "permission_bits_matching_content_freeze_source_stat"
        ),
    }


def _bind_current_source_to_content_baseline(
    plan: Mapping[str, Any], content_context: dict[str, Any]
) -> None:
    content_context["commitment"]["relation_source_rich_stat_bridge"] = (
        _relation_source_bridge(plan, content_context)
    )


def _revalidate_snapshot(
    path: Path, expected: Mapping[str, Any], *, field: str
) -> None:
    _, observed, identity = _read_regular_bytes_snapshot(path, field=field)
    require(observed == expected.get("sha256"), f"{field} content changed")
    require(identity == expected.get("rich_identity"), f"{field} identity changed")


def _revalidate_content_freeze_parent(
    *, plan: Mapping[str, Any], content_context: Mapping[str, Any]
) -> None:
    runtime = content_context.get("runtime")
    if runtime is None:
        require(
            not bool(plan.get("full_scope")),
            "full relation plan lost its content-freeze runtime binding",
        )
        require(
            content_context.get("commitment", {}).get(
                "relation_source_rich_stat_bridge"
            )
            == "OMITTED_PILOT",
            "pilot content-freeze omission commitment drift",
        )
        return
    runtime = require_mapping(runtime, "content-freeze runtime")
    receipt = require_mapping(runtime.get("receipt"), "content-freeze receipt snapshot")
    _revalidate_snapshot(
        Path(receipt["path"]), receipt, field="final content-freeze receipt"
    )
    for name, raw in require_mapping(
        runtime.get("files"), "content-freeze runtime files"
    ).items():
        snapshot = require_mapping(raw, f"content-freeze snapshot {name}")
        _revalidate_snapshot(
            Path(snapshot["path"]),
            snapshot,
            field=f"final content-freeze artifact {name}",
        )
    observed_bridge = _relation_source_bridge(plan, content_context)
    require(
        observed_bridge
        == content_context.get("commitment", {}).get(
            "relation_source_rich_stat_bridge"
        ),
        "final content-freeze relation source bridge differs from plan",
    )


def _actual_projected_schema(
    parquet: pq.ParquetFile, columns: Sequence[str]
) -> dict[str, dict[str, Any]]:
    schema = parquet.schema_arrow
    result: dict[str, dict[str, Any]] = {}
    for column in columns:
        require(column in schema.names, f"Parquet missing projected column: {column}")
        field = schema.field(column)
        result[column] = {"nullable": bool(field.nullable), "type": str(field.type)}
    return result


def _bool_array(value: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return (
        pc.fill_null(value, False)
        .to_numpy(zero_copy_only=False)
        .astype(bool, copy=False)
    )


def _regex_mask(value: pa.Array, pattern: str) -> np.ndarray:
    return _bool_array(pc.match_substring_regex(value, pattern=pattern))


def _equal_mask(value: pa.Array, scalar: str) -> np.ndarray:
    return _bool_array(pc.equal(value, pa.scalar(scalar, type=pa.string())))


def _parse_numeric_strings(
    values: pa.Array,
    *,
    active_indices: np.ndarray,
    prefix_length: int,
    field: str,
) -> np.ndarray:
    if len(active_indices) == 0:
        return np.empty(0, dtype="<u8")
    selected = values.take(pa.array(active_indices, type=pa.int64()))
    suffix = pc.utf8_slice_codeunits(selected, start=prefix_length)
    try:
        numeric = pc.cast(suffix, pa.uint64(), safe=True).to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise RelationAuditError(f"{field} does not fit uint64") from exc
    return np.asarray(numeric, dtype="<u8")


def splitmix64(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        result += np.uint64(0x9E3779B97F4A7C15)
        result = (result ^ (result >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        result = (result ^ (result >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        result ^= result >> np.uint64(31)
    return result


def _selection_mask(
    values: np.ndarray, hash_bucket: Mapping[str, Any] | None
) -> np.ndarray:
    if hash_bucket is None:
        return np.ones(len(values), dtype=bool)
    count = int(hash_bucket["bucket_count"])
    index = int(hash_bucket["bucket_index"])
    return (splitmix64(values) & np.uint64(count - 1)) == np.uint64(index)


def _pack_locator(file_index: int, rows: np.ndarray) -> np.ndarray:
    require(0 <= file_index < LOCATOR_FILE_LIMIT, "file index exceeds locator codec")
    require(
        len(rows) == 0 or int(np.max(rows)) <= LOCATOR_ROW_MASK,
        "file row exceeds locator codec",
    )
    return (np.uint32(file_index) << np.uint32(LOCATOR_ROW_BITS)) | rows.astype(
        np.uint32, copy=False
    )


def _canonical_score_bits(scores: pa.Array) -> np.ndarray:
    require(scores.null_count == 0, "topic score may not be null")
    numeric = np.array(scores.to_numpy(zero_copy_only=False), dtype="<f8", copy=True)
    require(np.all(np.isfinite(numeric)), "topic score must be finite")
    require(np.all((numeric >= 0.0) & (numeric <= 1.0)), "topic score outside [0,1]")
    numeric[numeric == 0.0] = 0.0
    return np.ascontiguousarray(numeric).view("<u8")


def checkpoint_path(output_dir: Path, task: FileTask) -> Path:
    token = sha256_bytes(task.relative_path.encode("utf-8"))[:24]
    return output_dir / "checkpoints" / task.table_key / f"{token}.json"


def _source_file(task: FileTask, source_root: Path) -> Path:
    return secure_join_regular(
        source_root,
        safe_relative_path(task.relative_path, "task.relative_path"),
        field="source Parquet",
    )


def _open_validate_parquet(
    task: FileTask, source_root: Path
) -> tuple[Path, Any, dict[str, int], pq.ParquetFile]:
    path = _source_file(task, source_root)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RelationAuditError(f"cannot open source Parquet: {exc}") from exc
    handle = os.fdopen(descriptor, "rb")
    identity = _rich_identity_from_stat(os.fstat(handle.fileno()))
    require(stat.S_ISREG(identity["mode"]), "source Parquet is not a regular file")
    require(
        identity == task.source_rich_identity,
        "source rich identity differs from frozen plan",
    )
    require(
        identity["bytes"] == task.bytes and identity["mtime_ns"] == task.mtime_ns,
        "source file identity differs from inventory",
    )
    try:
        parquet = pq.ParquetFile(handle)
    except (OSError, pa.ArrowException) as exc:
        handle.close()
        raise RelationAuditError(f"cannot open source Parquet: {exc}") from exc
    require(
        parquet.metadata.num_rows == task.footer_rows, "source footer row count differs"
    )
    schema = _actual_projected_schema(parquet, task.projected_columns)
    require(
        schema == task.projected_schema,
        "source projected schema differs from inventory",
    )
    require(
        sha256_bytes(canonical_json_bytes(schema)) == task.projected_schema_sha256,
        "source projected schema hash mismatch",
    )
    return path, handle, identity, parquet


def _projected_payload_sha256(
    task: FileTask, *, source_root: Path, batch_size: int
) -> tuple[str, int]:
    path, handle, identity, parquet = _open_validate_parquet(task, source_root)
    digest = hashlib.sha256()
    rows = 0
    try:
        for row_group in range(parquet.metadata.num_row_groups):
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group],
                columns=list(task.projected_columns),
                use_threads=False,
            ):
                rows += batch.num_rows
                serialized = batch.serialize().to_pybytes()
                digest.update(len(serialized).to_bytes(8, "little"))
                digest.update(serialized)
        require(
            _rich_identity_from_stat(os.fstat(handle.fileno())) == identity,
            "open source file changed during projected-payload verification",
        )
        require(
            _rich_regular_identity(path, field="source Parquet") == identity,
            "source path changed during projected-payload verification",
        )
    except (OSError, pa.ArrowException) as exc:
        raise RelationAuditError(f"cannot rehash source projection: {exc}") from exc
    finally:
        handle.close()
    require(rows == task.footer_rows, "projected payload row count differs from footer")
    return digest.hexdigest(), rows


def _segment_sha256(path: Path, *, start_record: int, rows: int) -> str:
    digest = hashlib.sha256()
    remaining = rows * EVENT_DTYPE.itemsize
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RelationAuditError(f"cannot open event spool: {exc}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        require(
            stat.S_ISREG(os.fstat(handle.fileno()).st_mode),
            "event spool is not a regular file",
        )
        handle.seek(start_record * EVENT_DTYPE.itemsize)
        while remaining:
            block = handle.read(min(16 * 1024 * 1024, remaining))
            if not block:
                raise RelationAuditError("event spool segment is truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _checkpoint_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    task: FileTask,
    *,
    expected_start_record: int,
    source_root: Path,
    spool_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        checkpoint.get("schema_version") == CHECKPOINT_SCHEMA,
        "checkpoint schema mismatch",
    )
    require(
        checkpoint.get("checkpoint_sha256")
        == sha256_bytes(canonical_json_bytes(_checkpoint_payload(checkpoint))),
        "checkpoint payload hash mismatch",
    )
    require(
        checkpoint.get("task_fingerprint_sha256")
        == plan.get("task_fingerprint_sha256"),
        "checkpoint plan fingerprint mismatch",
    )
    require(
        checkpoint.get("task_sha256") == task.task_sha256, "checkpoint task mismatch"
    )
    require(checkpoint.get("table_key") == task.table_key, "checkpoint table mismatch")
    require(
        checkpoint.get("relative_path") == task.relative_path,
        "checkpoint path mismatch",
    )
    require(
        checkpoint.get("source_identity")
        == {
            **task.source_rich_identity,
            "footer_rows": task.footer_rows,
            "projected_schema_sha256": task.projected_schema_sha256,
        },
        "checkpoint source identity mismatch",
    )
    segment = require_mapping(
        checkpoint.get("spool_segment"), "checkpoint spool segment"
    )
    require(
        segment.get("start_record") == expected_start_record,
        "checkpoint spool offset mismatch",
    )
    rows = require_int(segment.get("rows"), "checkpoint spool rows")
    require(rows >= 0, "checkpoint spool rows must be non-negative")
    require(
        _rich_regular_identity(spool_path, field="event spool")["bytes"]
        >= (expected_start_record + rows) * EVENT_DTYPE.itemsize,
        "event spool does not contain checkpoint segment",
    )
    require(
        _segment_sha256(spool_path, start_record=expected_start_record, rows=rows)
        == segment.get("sha256"),
        "event spool segment hash mismatch",
    )
    current_projection, current_rows = _projected_payload_sha256(
        task,
        source_root=source_root,
        batch_size=require_int(plan.get("batch_size"), "plan batch size"),
    )
    require(current_rows == task.footer_rows, "checkpoint current row count mismatch")
    require(
        current_projection == checkpoint.get("projected_payload_sha256"),
        "current projected payload differs from checkpoint",
    )
    return dict(checkpoint)


def _quarantine_observation(
    *, task: FileTask, row_group: int, file_row: int, raw_value: str
) -> dict[str, Any]:
    return {
        "table_key": task.table_key,
        "relative_path": task.relative_path,
        "row_group": row_group,
        "file_row": file_row,
        "raw_value": raw_value,
    }


def scan_task(
    task: FileTask,
    *,
    source_root: Path,
    config: Mapping[str, Any],
    allowed_topics: np.ndarray,
    plan: Mapping[str, Any],
    spool_handle: Any,
    spool_start_record: int,
) -> dict[str, Any]:
    path, source_handle, start_identity, parquet = _open_validate_parquet(
        task, source_root
    )
    codec = require_mapping(config.get("identifier_codec"), "identifier codec")
    quarantine = require_mapping(config.get("quarantine"), "quarantine")
    w_minus = str(
        require_mapping(quarantine.get("w_minus_1"), "W-1 contract").get("raw_value")
    )
    w_zero = str(
        require_mapping(quarantine.get("w_zero"), "W0 contract").get("raw_value")
    )
    work_pattern = str(codec.get("work_pattern"))
    topic_pattern = str(codec.get("topic_pattern"))
    work_prefix = str(codec.get("work_prefix"))
    topic_prefix = str(codec.get("topic_prefix"))
    topic_bits = require_int(codec.get("topic_bits"), "topic bits")
    topic_mask = (1 << topic_bits) - 1
    max_work = np.uint64(np.iinfo(np.uint64).max >> topic_bits)
    hash_bucket = plan.get("hash_bucket")
    batch_size = require_int(plan.get("batch_size"), "batch size")

    metrics: dict[str, Any] = {
        "raw_rows": 0,
        "null_work_id_rows": 0,
        "grammar_valid_work_rows": 0,
        "quarantined_w_minus_1_rows": 0,
        "unexpected_malformed_work_rows": 0,
        "w_zero_rows": 0,
        "selected_event_rows": 0,
        "main_primary_null_rows": 0,
        "main_primary_nonnull_rows": 0,
        "topic_rows": 0,
    }
    quarantine_observations: list[dict[str, Any]] = []
    projected_digest = hashlib.sha256()
    segment_digest = hashlib.sha256()
    file_row = 0
    segment_rows = 0

    try:
        for row_group in range(parquet.metadata.num_row_groups):
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group],
                columns=list(task.projected_columns),
                use_threads=False,
            ):
                batch_rows = batch.num_rows
                serialized = batch.serialize().to_pybytes()
                projected_digest.update(len(serialized).to_bytes(8, "little"))
                projected_digest.update(serialized)
                columns = {
                    name: batch.column(index)
                    for index, name in enumerate(batch.schema.names)
                }
                work_values = columns[task.projected_columns[0]]
                work_null = pc.is_null(work_values).to_numpy(zero_copy_only=False)
                work_valid = _regex_mask(work_values, work_pattern)
                work_w_minus = _equal_mask(work_values, w_minus)
                work_w_zero = _equal_mask(work_values, w_zero)
                malformed = ~work_null & ~work_valid
                unexpected = malformed & ~work_w_minus
                require(not np.any(work_null), "source work ID may not be null")
                require(not np.any(unexpected), "unexpected malformed source work ID")
                invalid_indices = np.flatnonzero(work_w_minus)
                if task.table_key == "topics":
                    require(
                        len(invalid_indices) == 0,
                        "topics may not contain W-1 source rows",
                    )
                for local_index in invalid_indices:
                    quarantine_observations.append(
                        _quarantine_observation(
                            task=task,
                            row_group=row_group,
                            file_row=file_row + int(local_index),
                            raw_value=w_minus,
                        )
                    )

                valid_indices = np.flatnonzero(work_valid)
                work_numeric = _parse_numeric_strings(
                    work_values,
                    active_indices=valid_indices,
                    prefix_length=len(work_prefix),
                    field="work ID",
                )
                require(
                    len(work_numeric) == 0 or np.all(work_numeric <= max_work),
                    "work ID overflows the 14-bit packed relation codec",
                )
                selected = _selection_mask(work_numeric, hash_bucket)
                selected_indices = valid_indices[selected]
                selected_work = work_numeric[selected]
                events = np.zeros(len(selected_work), dtype=EVENT_DTYPE)
                if task.table_key == "main":
                    primary_values = columns[task.projected_columns[1]].take(
                        pa.array(valid_indices, type=pa.int64())
                    )
                    primary_null = pc.is_null(primary_values).to_numpy(
                        zero_copy_only=False
                    )
                    primary_valid = _regex_mask(primary_values, topic_pattern)
                    require(
                        np.all(primary_null | primary_valid),
                        "main primary topic ID grammar mismatch",
                    )
                    primary_codes = np.zeros(len(valid_indices), dtype="<u8")
                    primary_nonnull_indices = np.flatnonzero(~primary_null)
                    parsed_primary = _parse_numeric_strings(
                        primary_values,
                        active_indices=primary_nonnull_indices,
                        prefix_length=len(topic_prefix),
                        field="primary topic ID",
                    )
                    require(
                        len(parsed_primary) == 0
                        or np.all(parsed_primary <= np.uint64(topic_mask)),
                        "primary topic ID exceeds 14-bit codec",
                    )
                    if len(parsed_primary):
                        allowed = allowed_topics[parsed_primary.astype(np.int64)]
                        require(
                            np.all(allowed),
                            "primary topic ID is absent from frozen taxonomy",
                        )
                        primary_codes[primary_nonnull_indices] = parsed_primary
                    selected_primary = primary_codes[selected]
                    events["key"] = (
                        selected_work << np.uint64(topic_bits)
                    ) | selected_primary
                    events["kind"] = MAIN_KIND
                    metrics["main_primary_null_rows"] += int(
                        np.count_nonzero(primary_null)
                    )
                    metrics["main_primary_nonnull_rows"] += int(
                        np.count_nonzero(~primary_null)
                    )
                else:
                    topic_values = columns[task.projected_columns[1]]
                    topic_null = pc.is_null(topic_values).to_numpy(zero_copy_only=False)
                    topic_valid = _regex_mask(topic_values, topic_pattern)
                    require(not np.any(topic_null), "topics topic ID may not be null")
                    require(np.all(topic_valid), "topics topic ID grammar mismatch")
                    all_indices = np.arange(batch_rows, dtype=np.int64)
                    topic_numeric = _parse_numeric_strings(
                        topic_values,
                        active_indices=all_indices,
                        prefix_length=len(topic_prefix),
                        field="topic ID",
                    )
                    require(
                        np.all(topic_numeric <= np.uint64(topic_mask)),
                        "topic ID exceeds 14-bit codec",
                    )
                    require(
                        np.all(allowed_topics[topic_numeric.astype(np.int64)]),
                        "topic ID is absent from frozen taxonomy",
                    )
                    scores = _canonical_score_bits(columns[task.projected_columns[2]])
                    events["key"] = (
                        selected_work << np.uint64(topic_bits)
                    ) | topic_numeric[selected]
                    events["score_bits"] = scores[selected]
                    events["kind"] = TOPIC_KIND
                    metrics["topic_rows"] += batch_rows

                local_rows = selected_indices.astype(np.uint64) + np.uint64(file_row)
                events["locator"] = _pack_locator(task.file_index, local_rows)
                event_bytes = events.tobytes(order="C")
                spool_handle.write(event_bytes)
                segment_digest.update(event_bytes)
                segment_rows += len(events)

                metrics["raw_rows"] += batch_rows
                metrics["null_work_id_rows"] += int(np.count_nonzero(work_null))
                metrics["grammar_valid_work_rows"] += len(valid_indices)
                metrics["quarantined_w_minus_1_rows"] += len(invalid_indices)
                metrics["unexpected_malformed_work_rows"] += int(
                    np.count_nonzero(unexpected)
                )
                metrics["w_zero_rows"] += int(np.count_nonzero(work_w_zero))
                metrics["selected_event_rows"] += len(events)
                file_row += batch_rows
    except RelationAuditError:
        source_handle.close()
        raise
    except (OSError, pa.ArrowException, ValueError, IndexError) as exc:
        source_handle.close()
        raise RelationAuditError(f"cannot scan {task.relative_path}: {exc}") from exc

    require(file_row == task.footer_rows, "scanned data-page rows differ from footer")
    require(
        metrics["raw_rows"]
        == metrics["grammar_valid_work_rows"]
        + metrics["quarantined_w_minus_1_rows"]
        + metrics["unexpected_malformed_work_rows"]
        + metrics["null_work_id_rows"],
        "file work-ID row conservation failed",
    )
    require(
        segment_rows == metrics["selected_event_rows"], "event segment count mismatch"
    )
    spool_handle.flush()
    os.fsync(spool_handle.fileno())
    try:
        require(
            _rich_identity_from_stat(os.fstat(source_handle.fileno()))
            == start_identity,
            "open source file changed during scan",
        )
        require(
            _rich_regular_identity(path, field="source Parquet") == start_identity,
            "source path changed during scan",
        )
    finally:
        source_handle.close()
    checkpoint: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "task_sha256": task.task_sha256,
        "table_key": task.table_key,
        "table_name": task.table_name,
        "file_index": task.file_index,
        "relative_path": task.relative_path,
        "source_identity": {
            **task.source_rich_identity,
            "footer_rows": task.footer_rows,
            "projected_schema_sha256": task.projected_schema_sha256,
        },
        "projected_payload_sha256": projected_digest.hexdigest(),
        "spool_segment": {
            "start_record": spool_start_record,
            "rows": segment_rows,
            "sha256": segment_digest.hexdigest(),
        },
        "metrics": metrics,
        "quarantine_observations": sorted(
            quarantine_observations,
            key=lambda value: (
                value["relative_path"],
                value["row_group"],
                value["file_row"],
            ),
        ),
    }
    checkpoint["checkpoint_sha256"] = sha256_bytes(
        canonical_json_bytes(_checkpoint_payload(checkpoint))
    )
    return checkpoint


def _write_array_atomic(path: Path, values: np.ndarray) -> None:
    directory_fd, temporary, handle = _begin_atomic_binary(path)
    publish = False
    try:
        values.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        publish = True
    finally:
        if not handle.closed:
            handle.close()
        _finish_atomic_binary(
            path,
            directory_fd=directory_fd,
            temporary=temporary,
            publish=publish,
        )


def prepare_sorted_events(
    output_dir: Path,
    *,
    plan: Mapping[str, Any],
    event_rows: int,
    resume: bool,
) -> dict[str, Any]:
    raw_path = output_dir / "work_topic_relation.events.raw.bin"
    sorted_path = output_dir / "work_topic_relation.events.sorted.bin"
    manifest_path = output_dir / "sort_manifest.json"
    expected_bytes = event_rows * EVENT_DTYPE.itemsize
    require(
        _rich_regular_identity(raw_path, field="raw event spool")["bytes"]
        == expected_bytes,
        "raw event spool size mismatch",
    )
    raw_hash = sha256_file(raw_path)
    if resume and manifest_path.is_file():
        manifest = read_json(manifest_path, "sort manifest")
        require(
            manifest.get("schema_version") == SORT_SCHEMA,
            "sort manifest schema mismatch",
        )
        require(
            manifest.get("task_fingerprint_sha256")
            == plan.get("task_fingerprint_sha256"),
            "sort manifest plan mismatch",
        )
        require(manifest.get("event_rows") == event_rows, "sort manifest row mismatch")
        require(
            manifest.get("raw_sha256") == raw_hash, "sort manifest raw hash mismatch"
        )
        require(
            _rich_regular_identity(sorted_path, field="sorted event artifact")["bytes"]
            == expected_bytes,
            "sorted event artifact size mismatch",
        )
        require(
            sha256_file(sorted_path) == manifest.get("sorted_sha256"),
            "sorted event artifact hash mismatch",
        )
        return manifest

    if event_rows:
        descriptor = _open_trusted_output(raw_path, os.O_RDONLY)
        with os.fdopen(descriptor, "rb") as handle:
            raw = np.memmap(handle, dtype=EVENT_DTYPE, mode="r", shape=(event_rows,))
            values = np.array(raw, dtype=EVENT_DTYPE, copy=True)
            del raw
        values.sort(order=["key", "kind", "score_bits", "locator"], kind="quicksort")
    else:
        values = np.empty(0, dtype=EVENT_DTYPE)
    _write_array_atomic(sorted_path, values)
    del values
    sorted_hash = sha256_file(sorted_path)
    manifest = {
        "schema_version": SORT_SCHEMA,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "event_rows": event_rows,
        "event_bytes": expected_bytes,
        "raw_sha256": raw_hash,
        "sorted_sha256": sorted_hash,
        "sort_order": ["key", "kind", "score_bits", "locator"],
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _group_blocks(
    events: np.ndarray,
    *,
    chunk_rows: int,
    key_mode: str,
    topic_bits: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    require(key_mode in {"edge", "work"}, "unknown grouping mode")
    total = len(events)
    carry = np.empty(0, dtype=EVENT_DTYPE)
    offset = 0
    while offset < total:
        end = min(total, offset + chunk_rows)
        block = np.array(events[offset:end], dtype=EVENT_DTYPE, copy=True)
        if len(carry):
            block = np.concatenate([carry, block])
        keys = (
            block["key"]
            if key_mode == "edge"
            else block["key"] >> np.uint64(topic_bits)
        )
        boundaries = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        if end < total:
            if len(boundaries) == 0:
                carry = block
                offset = end
                continue
            cut = int(boundaries[-1])
            complete = block[:cut]
            complete_keys = keys[:cut]
            carry = block[cut:]
        else:
            complete = block
            complete_keys = keys
            carry = np.empty(0, dtype=EVENT_DTYPE)
        if len(complete):
            starts = np.r_[
                0, np.flatnonzero(complete_keys[1:] != complete_keys[:-1]) + 1
            ]
            ends = np.r_[starts[1:], len(complete)]
            yield (
                complete,
                complete_keys[starts],
                starts.astype(np.int64),
                ends.astype(np.int64),
            )
        offset = end
    if len(carry):
        keys = (
            carry["key"]
            if key_mode == "edge"
            else carry["key"] >> np.uint64(topic_bits)
        )
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        ends = np.r_[starts[1:], len(carry)]
        yield carry, keys[starts], starts.astype(np.int64), ends.astype(np.int64)


def _group_sum(mask: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    prefix = np.empty(len(mask) + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(mask, dtype=np.int64, out=prefix[1:])
    return prefix[ends] - prefix[starts]


def _score_from_bits(value: int) -> float:
    return float(np.asarray([value], dtype="<u8").view("<f8")[0])


def _decode_locator(
    locator: int, *, table_key: str, task_paths: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    file_index = locator >> LOCATOR_ROW_BITS
    file_row = locator & LOCATOR_ROW_MASK
    paths = task_paths[table_key]
    require(0 <= file_index < len(paths), "artifact locator file index is invalid")
    return {"relative_path": paths[file_index], "file_row": file_row}


def _empty_edge_view() -> dict[str, int]:
    return {
        "topic_raw_rows": 0,
        "unique_work_topic_edges": 0,
        "duplicate_row_excess": 0,
        "duplicate_edge_groups": 0,
        "score_conflict_edge_groups": 0,
        "main_primary_nonnull_rows": 0,
        "primary_child_present_rows": 0,
        "primary_child_missing_rows": 0,
        "main_primary_null_rows": 0,
    }


def aggregate_edge_groups(
    sorted_path: Path,
    *,
    event_rows: int,
    output_dir: Path,
    plan: Mapping[str, Any],
    topic_bits: int,
    chunk_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sorted_handle: Any | None = None
    if event_rows:
        sorted_descriptor = _open_trusted_output(sorted_path, os.O_RDONLY)
        sorted_handle = os.fdopen(sorted_descriptor, "rb")
        events = np.memmap(
            sorted_handle, dtype=EVENT_DTYPE, mode="r", shape=(event_rows,)
        )
    else:
        events = np.empty(0, dtype=EVENT_DTYPE)
    task_paths: dict[str, list[str]] = {"main": [], "topics": []}
    for raw in plan["tasks"]:
        task_paths[str(raw["table_key"])].append(str(raw["relative_path"]))
    dedup_path = output_dir / "work_topic_edges.deduplicated.sorted.bin"
    _unlink_trusted_output(dedup_path, missing_ok=True)
    directory_fd, temporary, dedup_handle = _begin_atomic_binary(dedup_path)

    retained = _empty_edge_view()
    positive = _empty_edge_view()
    duplicate_samples: list[dict[str, Any]] = []
    conflict_samples: list[dict[str, Any]] = []
    max_multiplicity = 0
    main_marker_duplicate_groups = 0
    topic_code_zero_rows = 0
    dedup_rows = 0

    try:
        for block, group_keys, starts, ends in _group_blocks(
            events,
            chunk_rows=chunk_rows,
            key_mode="edge",
            topic_bits=topic_bits,
        ):
            lengths = ends - starts
            is_main = block["kind"] == MAIN_KIND
            main_counts = _group_sum(is_main, starts, ends)
            topic_counts = lengths - main_counts
            main_marker_duplicate_groups += int(np.count_nonzero(main_counts > 1))
            codes = group_keys & np.uint64((1 << topic_bits) - 1)
            works = group_keys >> np.uint64(topic_bits)
            topic_code_zero_rows += int(
                np.sum(topic_counts[codes == 0], dtype=np.int64)
            )

            topic_group = topic_counts > 0
            primary_group = (main_counts == 1) & (codes > 0)
            null_primary_group = (main_counts == 1) & (codes == 0)
            if np.any(topic_group):
                first_topic = starts[topic_group] + main_counts[topic_group]
                last_topic = ends[topic_group] - 1
                first_scores = block["score_bits"][first_topic]
                last_scores = block["score_bits"][last_topic]
                conflicts = first_scores != last_scores
                topic_keys = group_keys[topic_group]
                records = np.empty(len(topic_keys), dtype=DEDUP_DTYPE)
                records["key"] = topic_keys
                records["score_bits"] = first_scores
                records.tofile(dedup_handle)
                dedup_rows += len(records)

                duplicate_local = topic_counts[topic_group] > 1
                max_multiplicity = max(
                    max_multiplicity,
                    int(np.max(topic_counts[topic_group], initial=0)),
                )
                remaining = SAMPLE_LIMIT - len(duplicate_samples)
                if remaining > 0:
                    candidates = np.flatnonzero(duplicate_local)[:remaining]
                    topic_group_indices = np.flatnonzero(topic_group)
                    for local in candidates:
                        group_index = int(topic_group_indices[local])
                        first_index = int(
                            starts[group_index] + main_counts[group_index]
                        )
                        last_index = int(ends[group_index] - 1)
                        key = int(group_keys[group_index])
                        duplicate_samples.append(
                            {
                                "work_numeric_id": key >> topic_bits,
                                "topic_numeric_id": key & ((1 << topic_bits) - 1),
                                "multiplicity": int(topic_counts[group_index]),
                                "first_score": _score_from_bits(
                                    int(block["score_bits"][first_index])
                                ),
                                "last_score": _score_from_bits(
                                    int(block["score_bits"][last_index])
                                ),
                                "first_locator": _decode_locator(
                                    int(block["locator"][first_index]),
                                    table_key="topics",
                                    task_paths=task_paths,
                                ),
                                "last_locator": _decode_locator(
                                    int(block["locator"][last_index]),
                                    table_key="topics",
                                    task_paths=task_paths,
                                ),
                            }
                        )
                remaining_conflicts = SAMPLE_LIMIT - len(conflict_samples)
                if remaining_conflicts > 0 and np.any(conflicts):
                    candidates = np.flatnonzero(conflicts)[:remaining_conflicts]
                    topic_group_indices = np.flatnonzero(topic_group)
                    for local in candidates:
                        group_index = int(topic_group_indices[local])
                        key = int(group_keys[group_index])
                        first_index = int(
                            starts[group_index] + main_counts[group_index]
                        )
                        last_index = int(ends[group_index] - 1)
                        conflict_samples.append(
                            {
                                "work_numeric_id": key >> topic_bits,
                                "topic_numeric_id": key & ((1 << topic_bits) - 1),
                                "first_score": _score_from_bits(
                                    int(block["score_bits"][first_index])
                                ),
                                "last_score": _score_from_bits(
                                    int(block["score_bits"][last_index])
                                ),
                            }
                        )

            for view, selector in (
                (retained, np.ones(len(group_keys), dtype=bool)),
                (positive, works > 0),
            ):
                view["topic_raw_rows"] += int(
                    np.sum(topic_counts[selector], dtype=np.int64)
                )
                view["unique_work_topic_edges"] += int(
                    np.count_nonzero(topic_group & selector)
                )
                excess = np.maximum(topic_counts - 1, 0)
                view["duplicate_row_excess"] += int(
                    np.sum(excess[selector], dtype=np.int64)
                )
                view["duplicate_edge_groups"] += int(
                    np.count_nonzero((topic_counts > 1) & selector)
                )
                if np.any(topic_group):
                    first_all = starts[topic_group] + main_counts[topic_group]
                    last_all = ends[topic_group] - 1
                    conflicts_all = np.zeros(len(group_keys), dtype=bool)
                    conflicts_all[topic_group] = (
                        block["score_bits"][first_all] != block["score_bits"][last_all]
                    )
                    view["score_conflict_edge_groups"] += int(
                        np.count_nonzero(conflicts_all & selector)
                    )
                view["main_primary_nonnull_rows"] += int(
                    np.sum(main_counts[primary_group & selector], dtype=np.int64)
                )
                view["primary_child_present_rows"] += int(
                    np.sum(
                        main_counts[primary_group & topic_group & selector],
                        dtype=np.int64,
                    )
                )
                view["primary_child_missing_rows"] += int(
                    np.sum(
                        main_counts[primary_group & ~topic_group & selector],
                        dtype=np.int64,
                    )
                )
                view["main_primary_null_rows"] += int(
                    np.sum(main_counts[null_primary_group & selector], dtype=np.int64)
                )
        dedup_handle.flush()
        os.fsync(dedup_handle.fileno())
        dedup_handle.close()
    except BaseException:
        if not dedup_handle.closed:
            dedup_handle.close()
        _finish_atomic_binary(
            dedup_path,
            directory_fd=directory_fd,
            temporary=temporary,
            publish=False,
        )
        del events
        if sorted_handle is not None:
            sorted_handle.close()
        raise
    del events
    if sorted_handle is not None:
        sorted_handle.close()

    conflict_count = retained["score_conflict_edge_groups"]
    _finish_atomic_binary(
        dedup_path,
        directory_fd=directory_fd,
        temporary=temporary,
        publish=conflict_count == 0,
    )
    if conflict_count == 0:
        dedup_artifact = {
            "status": (
                "PROMOTED_FROZEN_EXACT_SCORE_SET_VIEW"
                if bool(plan.get("full_scope"))
                else "MATERIALIZED_PILOT_ONLY_EXACT_SCORE_SET_VIEW"
            ),
            "relative_path": dedup_path.name,
            "rows": dedup_rows,
            "bytes": _rich_regular_identity(
                dedup_path, field="deduplicated relation artifact"
            )["bytes"],
            "sha256": sha256_file(dedup_path),
        }
    else:
        dedup_artifact = {
            "status": "NOT_MATERIALIZED_SCORE_CONFLICT",
            "relative_path": None,
            "rows": None,
            "bytes": None,
            "sha256": None,
        }
    require(
        retained["topic_raw_rows"]
        == retained["unique_work_topic_edges"] + retained["duplicate_row_excess"],
        "raw/unique/duplicate edge conservation failed",
    )
    require(
        dedup_rows == retained["unique_work_topic_edges"], "dedup row count mismatch"
    )
    return (
        {
            "grammar_valid_including_w0": retained,
            "positive_work_ids_only": positive,
            "max_raw_edge_multiplicity": max_multiplicity,
            "main_marker_duplicate_edge_groups": main_marker_duplicate_groups,
            "topic_code_zero_rows": topic_code_zero_rows,
            "duplicate_samples": duplicate_samples,
            "score_conflict_samples": conflict_samples,
        },
        dedup_artifact,
    )


def _empty_work_view() -> dict[str, int]:
    return {
        "work_groups": 0,
        "main_work_rows": 0,
        "topic_rows": 0,
        "child_source_work_groups": 0,
        "orphan_topic_work_groups": 0,
        "duplicate_main_work_groups": 0,
        "main_works_with_child_topics": 0,
        "main_works_without_child_topics": 0,
        "null_primary_with_child_topics": 0,
        "null_primary_without_child_topics": 0,
        "nonnull_primary_with_child_topics": 0,
        "nonnull_primary_without_child_topics": 0,
    }


def aggregate_work_groups(
    sorted_path: Path,
    *,
    event_rows: int,
    topic_bits: int,
    chunk_rows: int,
) -> dict[str, Any]:
    sorted_handle: Any | None = None
    if event_rows:
        sorted_descriptor = _open_trusted_output(sorted_path, os.O_RDONLY)
        sorted_handle = os.fdopen(sorted_descriptor, "rb")
        events = np.memmap(
            sorted_handle, dtype=EVENT_DTYPE, mode="r", shape=(event_rows,)
        )
    else:
        events = np.empty(0, dtype=EVENT_DTYPE)
    retained = _empty_work_view()
    positive = _empty_work_view()
    w0_group: dict[str, int] | None = None
    for block, work_keys, starts, ends in _group_blocks(
        events,
        chunk_rows=chunk_rows,
        key_mode="work",
        topic_bits=topic_bits,
    ):
        is_main = block["kind"] == MAIN_KIND
        main_counts = _group_sum(is_main, starts, ends)
        topic_counts = (ends - starts) - main_counts
        codes = block["key"] & np.uint64((1 << topic_bits) - 1)
        main_null_counts = _group_sum(is_main & (codes == 0), starts, ends)
        main_nonnull_counts = main_counts - main_null_counts
        if np.any(work_keys == 0):
            index = int(np.flatnonzero(work_keys == 0)[0])
            w0_group = {
                "main_work_rows": int(main_counts[index]),
                "topic_rows": int(topic_counts[index]),
                "null_primary_rows": int(main_null_counts[index]),
                "nonnull_primary_rows": int(main_nonnull_counts[index]),
            }
        for view, selector in (
            (retained, np.ones(len(work_keys), dtype=bool)),
            (positive, work_keys > 0),
        ):
            main_present = main_counts > 0
            child_present = topic_counts > 0
            view["work_groups"] += int(np.count_nonzero(selector))
            view["main_work_rows"] += int(np.sum(main_counts[selector], dtype=np.int64))
            view["topic_rows"] += int(np.sum(topic_counts[selector], dtype=np.int64))
            view["child_source_work_groups"] += int(
                np.count_nonzero(child_present & selector)
            )
            view["orphan_topic_work_groups"] += int(
                np.count_nonzero(child_present & ~main_present & selector)
            )
            view["duplicate_main_work_groups"] += int(
                np.count_nonzero((main_counts > 1) & selector)
            )
            view["main_works_with_child_topics"] += int(
                np.sum(
                    main_counts[main_present & child_present & selector], dtype=np.int64
                )
            )
            view["main_works_without_child_topics"] += int(
                np.sum(
                    main_counts[main_present & ~child_present & selector],
                    dtype=np.int64,
                )
            )
            view["null_primary_with_child_topics"] += int(
                np.sum(main_null_counts[child_present & selector], dtype=np.int64)
            )
            view["null_primary_without_child_topics"] += int(
                np.sum(main_null_counts[~child_present & selector], dtype=np.int64)
            )
            view["nonnull_primary_with_child_topics"] += int(
                np.sum(main_nonnull_counts[child_present & selector], dtype=np.int64)
            )
            view["nonnull_primary_without_child_topics"] += int(
                np.sum(main_nonnull_counts[~child_present & selector], dtype=np.int64)
            )
    del events
    if sorted_handle is not None:
        sorted_handle.close()
    return {
        "grammar_valid_including_w0": retained,
        "positive_work_ids_only": positive,
        "w0_work_group": w0_group,
    }


def _checkpoint_tree(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    commitments = sorted(
        [
            {
                "table_key": value["table_key"],
                "relative_path": value["relative_path"],
                "checkpoint_sha256": value["checkpoint_sha256"],
                "projected_payload_sha256": value["projected_payload_sha256"],
                "spool_segment_sha256": value["spool_segment"]["sha256"],
            }
            for value in checkpoints
        ],
        key=lambda value: (value["table_key"], value["relative_path"]),
    )
    return {
        "checkpoint_count": len(commitments),
        "checkpoint_tree_sha256": sha256_bytes(canonical_json_bytes(commitments)),
        "resume_verification": "REHASH_CURRENT_PROJECTED_PAYLOAD_AND_SPOOL_SEGMENT",
        "metrics_reused_only_after_source_and_segment_rehash": True,
    }


def execute_scan(
    output_dir: Path,
    *,
    source_root: Path,
    config: Mapping[str, Any],
    allowed_topics: np.ndarray,
    plan: Mapping[str, Any],
    resume: bool,
    pause_after_new_files: int | None,
) -> tuple[list[dict[str, Any]], int, bool]:
    require(
        pause_after_new_files is None or pause_after_new_files > 0,
        "pause-after-new-files must be positive",
    )
    tasks = [_task_from_mapping(value) for value in plan["tasks"]]
    raw_path = output_dir / "work_topic_relation.events.raw.bin"
    descriptor = _open_trusted_output(raw_path, os.O_RDWR, create=True)
    os.close(descriptor)

    checkpoints: list[dict[str, Any]] = []
    expected_start = 0
    gap_seen = False
    for task in tasks:
        path = checkpoint_path(output_dir, task)
        if path.exists():
            require(resume, "existing checkpoints require --resume")
            require(not gap_seen, "checkpoint tree is not a contiguous task prefix")
            checkpoint = validate_checkpoint(
                read_json(path, "relation checkpoint"),
                task,
                expected_start_record=expected_start,
                source_root=source_root,
                spool_path=raw_path,
                plan=plan,
            )
            checkpoints.append(checkpoint)
            expected_start += int(checkpoint["spool_segment"]["rows"])
        else:
            gap_seen = True
    expected_bytes = expected_start * EVENT_DTYPE.itemsize
    raw_bytes = _rich_regular_identity(raw_path, field="raw event spool")["bytes"]
    require(
        raw_bytes >= expected_bytes,
        "raw event spool is shorter than checkpoint prefix",
    )
    if raw_bytes != expected_bytes:
        descriptor = _open_trusted_output(raw_path, os.O_RDWR)
        with os.fdopen(descriptor, "r+b") as handle:
            handle.truncate(expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())

    new_files = 0
    descriptor = _open_trusted_output(raw_path, os.O_RDWR)
    with os.fdopen(descriptor, "r+b") as spool:
        spool.seek(expected_bytes)
        for task in tasks[len(checkpoints) :]:
            checkpoint = scan_task(
                task,
                source_root=source_root,
                config=config,
                allowed_topics=allowed_topics,
                plan=plan,
                spool_handle=spool,
                spool_start_record=expected_start,
            )
            atomic_write_json(checkpoint_path(output_dir, task), checkpoint)
            checkpoints.append(checkpoint)
            expected_start += int(checkpoint["spool_segment"]["rows"])
            new_files += 1
            atomic_write_json(
                output_dir / "run_state.json",
                {
                    "schema_version": "openalex_g1_work_topic_relation_state.v1",
                    "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
                    "status": "RUNNING",
                    "completed_files": len(checkpoints),
                    "total_files": len(tasks),
                    "event_rows": expected_start,
                },
            )
            if (
                pause_after_new_files is not None
                and new_files >= pause_after_new_files
                and len(checkpoints) < len(tasks)
            ):
                atomic_write_json(
                    output_dir / "run_state.json",
                    {
                        "schema_version": "openalex_g1_work_topic_relation_state.v1",
                        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
                        "status": "PAUSED",
                        "completed_files": len(checkpoints),
                        "total_files": len(tasks),
                        "event_rows": expected_start,
                    },
                )
                return checkpoints, expected_start, True
        spool.truncate(expected_start * EVENT_DTYPE.itemsize)
        spool.flush()
        os.fsync(spool.fileno())
    return checkpoints, expected_start, False


def _validate_source_tree_end(
    *,
    source_root: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    current_tasks: list[dict[str, Any]] = []
    for raw in plan["tasks"]:
        task = _task_from_mapping(raw)
        path = _source_file(task, source_root)
        current_identity = _rich_regular_identity(path, field="source Parquet")
        require(
            current_identity == task.source_rich_identity,
            "final source rich-stat tree differs from plan start",
        )
        current = dict(raw)
        current["source_rich_identity"] = current_identity
        current_tasks.append(current)
    evidence = _source_tree_evidence(current_tasks)
    require(
        evidence == plan.get("source_tree_start"),
        "final source rich-stat tree commitment differs from plan start",
    )
    return {
        "status": "PASS_RICH_STAT_TREE_START_END_EQUAL",
        "file_count": evidence["file_count"],
        "tree_sha256": evidence["tree_sha256"],
        "open_fd_pre_post_fields": evidence["rich_stat_fields"],
        "projected_payload_rehashed_from_disk_checkpoints": True,
    }


def revalidate_checkpoints_from_disk(
    output_dir: Path,
    *,
    source_root: Path,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    _validate_output_tree(output_dir)
    tasks = [_task_from_mapping(value) for value in plan["tasks"]]
    expected_paths = [checkpoint_path(output_dir, task) for task in tasks]
    observed_paths = sorted(
        (output_dir / "checkpoints").rglob("*.json"),
        key=lambda value: value.relative_to(output_dir).as_posix(),
    )
    require(
        {path.relative_to(output_dir).as_posix() for path in observed_paths}
        == {path.relative_to(output_dir).as_posix() for path in expected_paths},
        "checkpoint files differ from the exact task set",
    )
    raw_path = output_dir / "work_topic_relation.events.raw.bin"
    verified: list[dict[str, Any]] = []
    expected_start = 0
    for task, path in zip(tasks, expected_paths, strict=True):
        checkpoint = validate_checkpoint(
            read_json(path, "final relation checkpoint"),
            task,
            expected_start_record=expected_start,
            source_root=source_root,
            spool_path=raw_path,
            plan=plan,
        )
        verified.append(checkpoint)
        expected_start += require_int(
            checkpoint["spool_segment"]["rows"], "checkpoint segment rows"
        )
    require(
        _rich_regular_identity(raw_path, field="raw event spool")["bytes"]
        == expected_start * EVENT_DTYPE.itemsize,
        "final raw event spool has uncommitted bytes",
    )
    _validate_output_tree(output_dir)
    return verified, expected_start


def _sum_checkpoint_metrics(
    checkpoints: Sequence[Mapping[str, Any]], table_key: str
) -> dict[str, int]:
    keys = {
        "raw_rows",
        "null_work_id_rows",
        "grammar_valid_work_rows",
        "quarantined_w_minus_1_rows",
        "unexpected_malformed_work_rows",
        "w_zero_rows",
        "selected_event_rows",
        "main_primary_null_rows",
        "main_primary_nonnull_rows",
        "topic_rows",
    }
    result = {key: 0 for key in keys}
    for checkpoint in checkpoints:
        if checkpoint["table_key"] != table_key:
            continue
        metrics = require_mapping(checkpoint.get("metrics"), "checkpoint metrics")
        require(set(metrics) == keys, "checkpoint metric set mismatch")
        for key in keys:
            result[key] += require_int(metrics[key], key)
    return result


def _expected_quarantine(config: Mapping[str, Any]) -> dict[str, Any]:
    value = require_mapping(
        require_mapping(config.get("quarantine"), "quarantine").get("w_minus_1"),
        "W-1 quarantine",
    )
    return {
        "table_key": str(value.get("table_key")),
        "relative_path": str(value.get("relative_path")),
        "row_group": require_int(value.get("row_group"), "W-1 row group"),
        "file_row": require_int(value.get("file_row"), "W-1 file row"),
        "raw_value": str(value.get("raw_value")),
    }


def _observed_quarantine(
    checkpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            dict(item)
            for checkpoint in checkpoints
            for item in checkpoint.get("quarantine_observations", [])
        ],
        key=lambda value: (
            value["relative_path"],
            value["row_group"],
            value["file_row"],
        ),
    )


def _subtract_metrics(
    retained: Mapping[str, int], positive: Mapping[str, int]
) -> dict[str, int]:
    require(set(retained) == set(positive), "sensitivity metric sets differ")
    return {key: int(retained[key]) - int(positive[key]) for key in sorted(retained)}


def build_report(
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    sort_manifest: Mapping[str, Any],
    edge: Mapping[str, Any],
    work: Mapping[str, Any],
    dedup_artifact: Mapping[str, Any],
    source_identity_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    full_scope = bool(plan.get("full_scope"))
    main = _sum_checkpoint_metrics(checkpoints, "main")
    topics = _sum_checkpoint_metrics(checkpoints, "topics")
    observed_quarantine = _observed_quarantine(checkpoints)
    expected_quarantine = _expected_quarantine(config)
    grammar_parent = context["grammar"]
    grammar_exact = require_mapping(grammar_parent.get("exact_counts"), "grammar exact")
    taxonomy_parent = context["taxonomy_report"]
    taxonomy_main = require_mapping(
        taxonomy_parent.get("accounting", {})
        .get("tables", {})
        .get("main", {})
        .get("metrics"),
        "taxonomy main metrics",
    )
    edge_retained = require_mapping(
        edge.get("grammar_valid_including_w0"), "retained edge metrics"
    )
    edge_positive = require_mapping(
        edge.get("positive_work_ids_only"), "positive edge metrics"
    )
    work_retained = require_mapping(
        work.get("grammar_valid_including_w0"), "retained work metrics"
    )
    work_positive = require_mapping(
        work.get("positive_work_ids_only"), "positive work metrics"
    )
    w0_group = work.get("w0_work_group")
    w0_contract = require_mapping(
        require_mapping(config.get("quarantine"), "quarantine").get("w_zero"),
        "W0 contract",
    )
    content_commitment = require_mapping(
        plan.get("current_content_baseline_parent"),
        "current content baseline parent",
    )
    content_bridge = content_commitment.get("relation_source_rich_stat_bridge")
    full_content_bound = bool(
        content_commitment.get("status")
        == "PASS_BOUND_CURRENT_CONTENT_BASELINE_RECEIPT"
        and content_commitment.get("current_content_baseline_component") == "PASS"
        and isinstance(content_bridge, dict)
        and content_bridge.get("status")
        == "PASS_EXACT_RELATION_TASK_SET_AND_RICH_STAT_EQUALITY"
        and content_bridge.get("files") == len(plan.get("tasks", []))
    )

    edge_delta = _subtract_metrics(edge_retained, edge_positive)
    work_delta = _subtract_metrics(work_retained, work_positive)
    w0_ledger_conserved = (
        isinstance(w0_group, dict)
        and work_delta["main_work_rows"] == int(w0_group["main_work_rows"])
        and work_delta["topic_rows"] == int(w0_group["topic_rows"])
        and edge_delta["main_primary_null_rows"] == int(w0_group["null_primary_rows"])
        and edge_delta["main_primary_nonnull_rows"]
        == int(w0_group["nonnull_primary_rows"])
        and edge_delta["topic_raw_rows"] == int(w0_group["topic_rows"])
        and edge_delta["primary_child_present_rows"]
        + edge_delta["primary_child_missing_rows"]
        == int(w0_group["nonnull_primary_rows"])
    )

    local_checks: dict[str, bool] = {
        "event_rows_equal_checkpoint_segments": int(sort_manifest["event_rows"])
        == main["selected_event_rows"] + topics["selected_event_rows"],
        "edge_topic_rows_equal_selected_topic_events": edge_retained["topic_raw_rows"]
        == topics["selected_event_rows"],
        "work_topic_rows_equal_edge_topic_rows": work_retained["topic_rows"]
        == edge_retained["topic_raw_rows"],
        "raw_unique_duplicate_conservation": edge_retained["topic_raw_rows"]
        == edge_retained["unique_work_topic_edges"]
        + edge_retained["duplicate_row_excess"],
        "primary_denominator_conserved": edge_retained["main_primary_nonnull_rows"]
        == edge_retained["primary_child_present_rows"]
        + edge_retained["primary_child_missing_rows"],
        "main_marker_edge_groups_unique": edge["main_marker_duplicate_edge_groups"]
        == 0,
        "topic_code_zero_absent": edge["topic_code_zero_rows"] == 0,
        "w0_structural_delta_conserved": bool(w0_ledger_conserved),
    }
    if full_scope:
        main_contract = require_mapping(
            config.get("tables", {}).get("main"), "main contract"
        )
        topics_contract = require_mapping(
            config.get("tables", {}).get("topics"), "topics contract"
        )
        local_checks.update(
            {
                "main_raw_rows_exact": main["raw_rows"]
                == require_int(main_contract.get("expected_footer_rows"), "main rows"),
                "topics_raw_rows_exact": topics["raw_rows"]
                == require_int(
                    topics_contract.get("expected_footer_rows"), "topics rows"
                ),
                "quarantine_exact": observed_quarantine == [expected_quarantine],
                "main_grammar_rows_equal_parent": main["grammar_valid_work_rows"]
                == require_int(grammar_exact.get("grammar_valid_rows"), "grammar rows"),
                "main_w0_exact": main["w_zero_rows"]
                == require_int(w0_contract.get("expected_main_rows"), "main W0 rows"),
                "topics_w0_exact": topics["w_zero_rows"]
                == require_int(
                    w0_contract.get("expected_topic_rows"), "topics W0 rows"
                ),
                "main_null_primary_equal_taxonomy_parent": main[
                    "main_primary_null_rows"
                ]
                == require_int(
                    taxonomy_main.get("hierarchy_all_null_rows"),
                    "parent null primary rows",
                ),
                "main_nonnull_primary_equal_taxonomy_parent": main[
                    "main_primary_nonnull_rows"
                ]
                == require_int(
                    taxonomy_main.get("hierarchy_complete_rows"),
                    "parent nonnull primary rows",
                ),
                "main_work_rows_unique_and_complete": work_retained["main_work_rows"]
                == require_int(
                    grammar_exact.get("grammar_valid_rows"), "parent retained works"
                )
                and work_retained["duplicate_main_work_groups"] == 0,
                "topic_source_work_orphans_zero": work_retained[
                    "orphan_topic_work_groups"
                ]
                == 0,
                "primary_child_missing_zero": edge_retained[
                    "primary_child_missing_rows"
                ]
                == 0,
                "score_conflict_groups_zero": edge_retained[
                    "score_conflict_edge_groups"
                ]
                == 0,
                "dedup_view_promoted": dedup_artifact.get("status")
                == "PROMOTED_FROZEN_EXACT_SCORE_SET_VIEW",
                "current_content_baseline_exactly_bound": full_content_bound,
            }
        )
    all_checks_pass = all(local_checks.values())
    status = (
        FULL_STATUS
        if full_scope and all_checks_pass
        else BLOCKED_STATUS
        if full_scope
        else PILOT_STATUS
    )

    raw_duplicates = int(edge_retained["duplicate_row_excess"])
    score_conflicts = int(edge_retained["score_conflict_edge_groups"])
    w0_prerequisites_pass = bool(
        full_scope
        and w0_ledger_conserved
        and local_checks.get("main_w0_exact") is True
        and local_checks.get("topics_w0_exact") is True
        and local_checks.get("quarantine_exact") is True
        and local_checks.get("main_grammar_rows_equal_parent") is True
    )
    parent_boundary = dict(
        require_mapping(
            config.get("success_contract", {}).get("required_full_parent_boundary"),
            "parent boundary",
        )
    )
    gate_status: dict[str, Any] = {
        **parent_boundary,
        "current_content_baseline_component": (
            "PASS_BOUND_TO_EXACT_RELATION_SOURCE_TASK_SET"
            if full_scope and full_content_bound
            else "BLOCKED"
            if full_scope
            else "NOT_REQUIRED_UNPROMOTABLE_PILOT"
            if content_commitment.get("status") == "OMITTED_UNPROMOTABLE_PILOT"
            else "BOUND_PILOT_ONLY_NOT_PROMOTABLE"
        ),
        "historical_content_identity": "UNVERIFIED",
        "work_topic_relation_component": (
            "PASS"
            if status == FULL_STATUS
            else "BLOCKED"
            if status == BLOCKED_STATUS
            else "NOT_ESTABLISHED_PILOT"
        ),
        "duplicate_work_topic_audit": "PASS" if full_scope else "NOT_ESTABLISHED_PILOT",
        "raw_work_topic_edge_uniqueness": (
            "NOT_ESTABLISHED_PILOT"
            if not full_scope
            else "PASS_RAW_UNIQUE"
            if raw_duplicates == 0
            else "FAIL_DUPLICATES_PRESENT"
        ),
        "frozen_exact_score_dedup_view": (
            "PASS"
            if full_scope and score_conflicts == 0
            else "BLOCKED_SCORE_CONFLICT"
            if full_scope
            else "NOT_ESTABLISHED_PILOT"
        ),
        "primary_topic_child_inclusion": (
            "PASS"
            if full_scope and edge_retained["primary_child_missing_rows"] == 0
            else "BLOCKED_MISSING_CHILD"
            if full_scope
            else "NOT_ESTABLISHED_PILOT"
        ),
        "w_zero_positive_id_structural_sensitivity": (
            "PASS_COMPONENT_ONLY"
            if w0_prerequisites_pass
            else "BLOCKED"
            if full_scope
            else "NOT_ESTABLISHED_PILOT"
        ),
    }
    scope = {
        "full_scope": full_scope,
        "pilot_mode": plan.get("pilot_mode"),
        "max_files_per_table": plan.get("max_files_per_table"),
        "hash_bucket": plan.get("hash_bucket"),
        "selected_files": len(plan["tasks"]),
        "selected_footer_rows": sum(
            int(value["selected_footer_rows"]) for value in plan["tables"].values()
        ),
        "event_rows": int(sort_manifest["event_rows"]),
        "gpu_used": False,
        "scientific_effect_estimation_performed": False,
        "resource_estimate": dict(plan["resource_estimate"]),
    }
    accounting = {
        "source_tables": {"main": main, "topics": topics},
        "edge_groups": edge,
        "work_groups": work,
        "local_checks": local_checks,
        "all_local_checks_pass": all_checks_pass,
    }
    sensitivity = {
        "status": (
            "PASS_WORK_TOPIC_STRUCTURAL_SENSITIVITY_ONLY"
            if w0_prerequisites_pass
            else "NOT_ESTABLISHED_PILOT"
            if not full_scope
            else "BLOCKED"
        ),
        "grammar_valid_including_w0_minus_positive_work_ids_only": {
            "edge_metrics": edge_delta,
            "work_metrics": work_delta,
        },
        "w0_work_group": w0_group,
        "ledger_conserved": bool(w0_ledger_conserved),
        "full_prerequisites_pass": w0_prerequisites_pass,
        "required_full_checks": {
            key: local_checks.get(key)
            for key in (
                "main_w0_exact",
                "topics_w0_exact",
                "quarantine_exact",
                "main_grammar_rows_equal_parent",
            )
        },
        "w_zero_entity_validity": "UNRESOLVED",
        "scientific_or_panel_sensitivity_established": False,
    }
    checkpoint_evidence = _checkpoint_tree(checkpoints)
    fingerprint_payload = {
        "status": status,
        "gate_status": gate_status,
        "scope": scope,
        "accounting": accounting,
        "w0_structural_sensitivity": sensitivity,
        "deduplicated_relation_artifact": dedup_artifact,
        "checkpoint_evidence": checkpoint_evidence,
        "sort_evidence": dict(sort_manifest),
        "source_identity_evidence": dict(source_identity_evidence),
        "content_baseline_parent": content_commitment,
        "parent_hashes": context["hashes"],
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "claim_boundary": config.get("claim_boundary"),
    }
    result_fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_payload))
    report = {
        "schema_version": REPORT_SCHEMA,
        "label": config.get("label"),
        "status": status,
        "gate_status": gate_status,
        "scope": scope,
        "accounting": accounting,
        "w0_structural_sensitivity": sensitivity,
        "deduplicated_relation_artifact": dedup_artifact,
        "checkpoint_evidence": checkpoint_evidence,
        "sort_evidence": dict(sort_manifest),
        "source_identity_evidence": dict(source_identity_evidence),
        "content_baseline_parent": content_commitment,
        "claim_boundary": config.get("claim_boundary"),
        "provenance": {
            **context["hashes"],
            "config_sha256": plan["config_sha256"],
            "config_snapshot_bytes": plan["config_snapshot_bytes"],
            "tool_sha256": plan["tool_sha256"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "result_fingerprint_sha256": result_fingerprint,
            "tool_relative_path": "scripts/audit_openalex_work_topic_relation.py",
            "path_redacted": True,
        },
    }
    serialized = json.dumps(report)
    require(
        all(prefix not in serialized for prefix in ("/home/", "/hdd/", "/ssd/")),
        "report leaked an operator path",
    )
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    scope = report["scope"]
    edge = report["accounting"]["edge_groups"]["grammar_valid_including_w0"]
    work = report["accounting"]["work_groups"]["grammar_valid_including_w0"]
    gates = report["gate_status"]
    sensitivity = report["w0_structural_sensitivity"]
    content = report["content_baseline_parent"]
    bridge = content["relation_source_rich_stat_bridge"]
    bridge_status = bridge["status"] if isinstance(bridge, dict) else str(bridge)
    lines = [
        "# OpenAlex Work-Topic Relation Audit",
        "",
        "## Status and boundary",
        "",
        f"- Status: `{report['status']}`",
        f"- Full scope: `{str(scope['full_scope']).lower()}`",
        f"- Pilot mode: `{scope['pilot_mode']}`",
        f"- Files: `{scope['selected_files']:,}`",
        f"- Event rows: `{scope['event_rows']:,}`",
        "- Workload: `local_cpu_io`; GPU used: `false`",
        "- Scientific effect estimation: `false`",
        "",
        "Raw edge uniqueness, deduplicated-view validity, primary-child inclusion,",
        "and the W0 structural sensitivity are separate states. A pilot run cannot",
        "receive the full component status.",
        "",
        "## Current-content baseline binding",
        "",
        f"- Parent receipt: `{content['status']}`",
        f"- Relation source bridge: `{bridge_status}`",
        "- Historical inventory-era byte identity remains: `UNVERIFIED`",
        "",
        "## Relation accounting",
        "",
        "| Metric | Exact count |",
        "| --- | ---: |",
        f"| Raw topic rows | {edge['topic_raw_rows']:,} |",
        f"| Unique work-topic edges | {edge['unique_work_topic_edges']:,} |",
        f"| Duplicate row excess | {edge['duplicate_row_excess']:,} |",
        f"| Duplicate edge groups | {edge['duplicate_edge_groups']:,} |",
        f"| Score-conflict edge groups | {edge['score_conflict_edge_groups']:,} |",
        f"| Main non-null primary rows | {edge['main_primary_nonnull_rows']:,} |",
        f"| Primary child present | {edge['primary_child_present_rows']:,} |",
        f"| Primary child missing | {edge['primary_child_missing_rows']:,} |",
        f"| Main null primary rows | {edge['main_primary_null_rows']:,} |",
        f"| Main works | {work['main_work_rows']:,} |",
        f"| Orphan topic-source work groups | {work['orphan_topic_work_groups']:,} |",
        "",
        "## W0 structural sensitivity",
        "",
        f"- Status: `{sensitivity['status']}`",
        f"- Exact delta ledger conserved: `{str(sensitivity['ledger_conserved']).lower()}`",
        "- W0 entity validity remains: `UNRESOLVED`",
        "- Scientific or final-panel sensitivity established: `false`",
        "",
        "## Gate status",
        "",
        "| Gate | State |",
        "| --- | --- |",
    ]
    for key in sorted(gates):
        value = gates[key]
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def _report_fingerprint_payload_from_disk(
    report: Mapping[str, Any], *, parent_hashes: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "gate_status": report.get("gate_status"),
        "scope": report.get("scope"),
        "accounting": report.get("accounting"),
        "w0_structural_sensitivity": report.get("w0_structural_sensitivity"),
        "deduplicated_relation_artifact": report.get("deduplicated_relation_artifact"),
        "checkpoint_evidence": report.get("checkpoint_evidence"),
        "sort_evidence": report.get("sort_evidence"),
        "source_identity_evidence": report.get("source_identity_evidence"),
        "content_baseline_parent": report.get("content_baseline_parent"),
        "parent_hashes": dict(parent_hashes),
        "task_fingerprint_sha256": report.get("provenance", {}).get(
            "task_fingerprint_sha256"
        ),
        "claim_boundary": report.get("claim_boundary"),
    }


def _read_text_nofollow(path: Path, *, field: str) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            require(
                stat.S_ISREG(os.fstat(handle.fileno()).st_mode),
                f"{field} is not a regular file",
            )
            return handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise RelationAuditError(f"cannot read {field}") from exc


def _terminal_checkpoint_evidence(
    output_dir: Path, plan: Mapping[str, Any]
) -> tuple[dict[str, Any], int, set[str]]:
    checkpoints: list[dict[str, Any]] = []
    checkpoint_paths: set[str] = set()
    expected_start_record = 0
    for raw in require_list(plan.get("tasks"), "terminal plan tasks"):
        task = _task_from_mapping(require_mapping(raw, "terminal plan task"))
        path = checkpoint_path(output_dir, task)
        relative = path.relative_to(output_dir).as_posix()
        require(
            relative not in checkpoint_paths,
            "terminal checkpoint paths are duplicated",
        )
        checkpoint_paths.add(relative)
        checkpoint = read_json(path, "terminal checkpoint")
        require(
            checkpoint.get("schema_version") == CHECKPOINT_SCHEMA,
            "terminal checkpoint schema mismatch",
        )
        require(
            checkpoint.get("checkpoint_sha256")
            == sha256_bytes(canonical_json_bytes(_checkpoint_payload(checkpoint))),
            "terminal checkpoint payload hash mismatch",
        )
        require(
            checkpoint.get("task_fingerprint_sha256")
            == plan.get("task_fingerprint_sha256"),
            "terminal checkpoint plan fingerprint mismatch",
        )
        require(
            checkpoint.get("task_sha256") == task.task_sha256,
            "terminal checkpoint task mismatch",
        )
        require(
            checkpoint.get("table_key") == task.table_key,
            "terminal checkpoint table mismatch",
        )
        require(
            checkpoint.get("relative_path") == task.relative_path,
            "terminal checkpoint source path mismatch",
        )
        require(
            checkpoint.get("source_identity")
            == {
                **task.source_rich_identity,
                "footer_rows": task.footer_rows,
                "projected_schema_sha256": task.projected_schema_sha256,
            },
            "terminal checkpoint source identity mismatch",
        )
        require_sha256(
            checkpoint.get("projected_payload_sha256"),
            "terminal checkpoint projected payload SHA-256",
        )
        segment = require_mapping(
            checkpoint.get("spool_segment"), "terminal checkpoint spool segment"
        )
        require(
            segment.get("start_record") == expected_start_record,
            "terminal checkpoint spool offset mismatch",
        )
        rows = require_int(segment.get("rows"), "terminal checkpoint spool rows")
        require(rows >= 0, "terminal checkpoint spool rows are negative")
        require_sha256(
            segment.get("sha256"), "terminal checkpoint spool segment SHA-256"
        )
        expected_start_record += rows
        checkpoints.append(checkpoint)
    return _checkpoint_tree(checkpoints), expected_start_record, checkpoint_paths


def _terminal_published_artifact_paths(output_dir: Path) -> set[str]:
    paths: set[str] = set()
    for current, _, file_names in os.walk(output_dir, followlinks=False):
        current_path = Path(current)
        for name in file_names:
            relative = (current_path / name).relative_to(output_dir).as_posix()
            if relative in {
                "audit.lock",
                "work_topic_relation_audit.outputs.json",
            }:
                continue
            paths.add(relative)
    return paths


def _has_valid_terminal_completion_receipt(output_dir: Path) -> bool:
    require(output_dir.is_absolute(), "output directory must be absolute")
    if not output_dir.exists() and not output_dir.is_symlink():
        return False
    require(
        output_dir.is_dir() and not output_dir.is_symlink(),
        "resume output path is not a trusted directory",
    )
    _assert_no_symlink_components(output_dir, field="resume output directory")
    output = output_dir.resolve()
    receipt_path = output / "work_topic_relation_audit.outputs.json"
    if not receipt_path.exists() and not receipt_path.is_symlink():
        return False
    _validate_output_tree(output)
    plan = read_json(output / "audit_plan.json", "terminal audit plan")
    report = read_json(
        output / "work_topic_relation_audit.json", "terminal work-topic report"
    )
    receipt = read_json(receipt_path, "terminal work-topic receipt")
    state = read_json(output / "run_state.json", "terminal run state")
    require(plan.get("schema_version") == PLAN_SCHEMA, "terminal plan schema mismatch")
    require(
        plan.get("task_fingerprint_sha256") == _plan_fingerprint(plan),
        "terminal plan fingerprint mismatch",
    )
    require(
        report.get("schema_version") == REPORT_SCHEMA,
        "terminal report schema mismatch",
    )
    require(
        receipt.get("schema_version") == OUTPUTS_SCHEMA,
        "terminal receipt schema mismatch",
    )
    require(
        set(receipt)
        == {
            "schema_version",
            "status",
            "result_fingerprint_sha256",
            "artifacts",
            "checkpoint_evidence",
        },
        "terminal receipt keys mismatch",
    )
    require(
        report.get("status") in {FULL_STATUS, PILOT_STATUS, BLOCKED_STATUS}
        and receipt.get("status") == report.get("status"),
        "terminal report/receipt status mismatch",
    )
    task_fingerprint = plan.get("task_fingerprint_sha256")
    result_fingerprint = report.get("provenance", {}).get(
        "result_fingerprint_sha256"
    )
    require_sha256(task_fingerprint, "terminal task fingerprint")
    require_sha256(result_fingerprint, "terminal result fingerprint")
    require(
        report.get("provenance", {}).get("task_fingerprint_sha256")
        == task_fingerprint,
        "terminal report task fingerprint mismatch",
    )
    parent_hashes = require_mapping(
        plan.get("parent_hashes"), "terminal plan parent hashes"
    )
    require(
        sha256_bytes(
            canonical_json_bytes(
                _report_fingerprint_payload_from_disk(
                    report, parent_hashes=parent_hashes
                )
            )
        )
        == result_fingerprint,
        "terminal report result fingerprint mismatch",
    )
    require(
        receipt.get("result_fingerprint_sha256") == result_fingerprint,
        "terminal receipt result fingerprint mismatch",
    )
    require(
        state.get("status") == "COMPLETE_PENDING_RECEIPT"
        and state.get("complete_only_if_receipt_exists") is True
        and state.get("completion_receipt_relative_path") == receipt_path.name
        and state.get("task_fingerprint_sha256") == task_fingerprint
        and state.get("result_fingerprint_sha256") == result_fingerprint,
        "terminal state/receipt lifecycle mismatch",
    )
    require(
        _read_text_nofollow(
            output / "work_topic_relation_audit.md",
            field="terminal work-topic Markdown",
        )
        == render_markdown(report),
        "terminal Markdown differs from report",
    )
    artifact_contracts = require_mapping(
        receipt.get("artifacts"), "terminal receipt artifacts"
    )
    fixed_artifact_paths = {
        "audit_plan.json",
        "run_state.json",
        "sort_manifest.json",
        "work_topic_relation.events.raw.bin",
        "work_topic_relation.events.sorted.bin",
        "work_topic_relation_audit.json",
        "work_topic_relation_audit.md",
    }
    plan_tasks = [
        _task_from_mapping(require_mapping(raw, "terminal plan task"))
        for raw in require_list(plan.get("tasks"), "terminal plan tasks")
    ]
    expected_checkpoint_paths: set[str] = set()
    for task in plan_tasks:
        require(
            task.table_key in {"main", "topics"},
            "terminal checkpoint table is invalid",
        )
        relative = checkpoint_path(output, task).relative_to(output).as_posix()
        require(
            relative not in expected_checkpoint_paths,
            "terminal checkpoint paths are duplicated",
        )
        expected_checkpoint_paths.add(relative)
    require(
        len(expected_checkpoint_paths) == len(plan_tasks),
        "terminal checkpoint path count mismatch",
    )

    dedup_evidence = require_mapping(
        report.get("deduplicated_relation_artifact"),
        "terminal deduplicated relation evidence",
    )
    dedup_relative: str | None = None
    if dedup_evidence.get("sha256") is None:
        require(
            dedup_evidence.get("status") == "NOT_MATERIALIZED_SCORE_CONFLICT"
            and dedup_evidence.get("relative_path") is None
            and dedup_evidence.get("rows") is None
            and dedup_evidence.get("bytes") is None,
            "terminal absent deduplicated artifact evidence mismatch",
        )
    else:
        dedup_relative_value = dedup_evidence.get("relative_path")
        dedup_relative = safe_relative_path(
            dedup_relative_value,
            "terminal deduplicated artifact path",
        ).as_posix()
        require(
            dedup_relative_value == "work_topic_edges.deduplicated.sorted.bin"
            and dedup_relative == "work_topic_edges.deduplicated.sorted.bin",
            "terminal deduplicated artifact path mismatch",
        )
        dedup_rows = require_int(
            dedup_evidence.get("rows"), "terminal deduplicated artifact rows"
        )
        dedup_bytes = require_int(
            dedup_evidence.get("bytes"), "terminal deduplicated artifact bytes"
        )
        require(
            dedup_rows >= 0 and dedup_bytes == dedup_rows * DEDUP_DTYPE.itemsize,
            "terminal deduplicated artifact size mismatch",
        )
        require_sha256(
            dedup_evidence.get("sha256"),
            "terminal deduplicated artifact SHA-256",
        )

    expected_artifact_paths = fixed_artifact_paths | expected_checkpoint_paths
    if dedup_relative is not None:
        expected_artifact_paths.add(dedup_relative)
    require(
        set(artifact_contracts) == expected_artifact_paths,
        "terminal receipt artifact set mismatch",
    )
    require(
        _terminal_published_artifact_paths(output) == expected_artifact_paths,
        "terminal published artifact set mismatch",
    )

    verified_artifacts: dict[str, dict[str, Any]] = {}
    for relative_text in sorted(expected_artifact_paths):
        contract = require_mapping(
            artifact_contracts.get(relative_text),
            f"terminal receipt artifact {relative_text}",
        )
        require(
            set(contract) == {"relative_path", "bytes", "sha256"},
            "terminal receipt artifact keys mismatch",
        )
        relative_value = contract.get("relative_path")
        relative = safe_relative_path(relative_value, "terminal receipt artifact path")
        require(
            relative_value == relative_text and relative.as_posix() == relative_text,
            "terminal receipt artifact path mismatch",
        )
        path = secure_join_regular(
            output, relative, field="terminal receipt-bound artifact"
        )
        artifact_bytes = require_int(
            contract.get("bytes"), "terminal artifact bytes"
        )
        require(artifact_bytes >= 0, "terminal artifact bytes are negative")
        require(
            _rich_regular_identity(path, field="terminal receipt-bound artifact")[
                "bytes"
            ]
            == artifact_bytes,
            "terminal receipt artifact byte mismatch",
        )
        require(
            sha256_file(path)
            == require_sha256(contract.get("sha256"), "terminal artifact SHA-256"),
            "terminal receipt artifact SHA-256 mismatch",
        )
        verified_artifacts[relative_text] = contract

    checkpoint_evidence, checkpoint_event_rows, checkpoint_paths = (
        _terminal_checkpoint_evidence(output, plan)
    )
    require(
        checkpoint_paths == expected_checkpoint_paths,
        "terminal checkpoint artifact set mismatch",
    )
    require(
        checkpoint_evidence == report.get("checkpoint_evidence")
        and checkpoint_evidence == receipt.get("checkpoint_evidence"),
        "terminal checkpoint evidence mismatch",
    )

    sort_manifest = read_json(output / "sort_manifest.json", "terminal sort manifest")
    require(
        set(sort_manifest)
        == {
            "schema_version",
            "task_fingerprint_sha256",
            "event_rows",
            "event_bytes",
            "raw_sha256",
            "sorted_sha256",
            "sort_order",
        },
        "terminal sort manifest keys mismatch",
    )
    require(
        sort_manifest.get("schema_version") == SORT_SCHEMA,
        "terminal sort manifest schema mismatch",
    )
    require(
        sort_manifest == report.get("sort_evidence"),
        "terminal sort manifest/report mismatch",
    )
    require(
        sort_manifest.get("task_fingerprint_sha256") == task_fingerprint,
        "terminal sort manifest plan mismatch",
    )
    event_rows = require_int(sort_manifest.get("event_rows"), "terminal event rows")
    event_bytes = require_int(
        sort_manifest.get("event_bytes"), "terminal event bytes"
    )
    require(event_rows >= 0, "terminal event rows are negative")
    require(
        event_bytes == event_rows * EVENT_DTYPE.itemsize,
        "terminal event byte count mismatch",
    )
    report_scope = require_mapping(report.get("scope"), "terminal report scope")
    require(
        event_rows == checkpoint_event_rows
        and event_rows
        == require_int(report_scope.get("event_rows"), "terminal report event rows"),
        "terminal event row cross-binding mismatch",
    )
    raw_contract = verified_artifacts["work_topic_relation.events.raw.bin"]
    sorted_contract = verified_artifacts["work_topic_relation.events.sorted.bin"]
    require(
        raw_contract["bytes"] == event_bytes
        and sorted_contract["bytes"] == event_bytes,
        "terminal event artifact byte cross-binding mismatch",
    )
    require(
        sort_manifest.get("raw_sha256") == raw_contract["sha256"]
        and sort_manifest.get("sorted_sha256") == sorted_contract["sha256"],
        "terminal event artifact SHA-256 cross-binding mismatch",
    )
    require(
        sort_manifest.get("sort_order")
        == ["key", "kind", "score_bits", "locator"],
        "terminal sort order mismatch",
    )
    require(
        state.get("schema_version")
        == "openalex_g1_work_topic_relation_state.v1"
        and state.get("completed_files") == len(plan_tasks)
        and state.get("total_files") == len(plan_tasks)
        and state.get("event_rows") == event_rows,
        "terminal state completion accounting mismatch",
    )
    if dedup_relative is not None:
        dedup_contract = verified_artifacts[dedup_relative]
        require(
            dedup_contract["bytes"] == dedup_evidence["bytes"]
            and dedup_contract["sha256"] == dedup_evidence["sha256"],
            "terminal deduplicated artifact cross-binding mismatch",
        )
    return True


def validate_final_publication_from_disk(
    output_dir: Path,
    *,
    report: Mapping[str, Any],
    outputs: Mapping[str, Any],
    plan: Mapping[str, Any],
    parent_hashes: Mapping[str, Any],
    source_root: Path,
    event_rows: int,
    content_context: Mapping[str, Any],
) -> None:
    _validate_output_tree(output_dir)
    report_path = output_dir / "work_topic_relation_audit.json"
    markdown_path = output_dir / "work_topic_relation_audit.md"
    receipt_path = output_dir / "work_topic_relation_audit.outputs.json"
    disk_report = read_json(report_path, "final work-topic report")
    disk_receipt = read_json(receipt_path, "final work-topic receipt")
    require(
        disk_report.get("schema_version") == REPORT_SCHEMA,
        "final report schema mismatch",
    )
    require(
        disk_receipt.get("schema_version") == OUTPUTS_SCHEMA,
        "final receipt schema mismatch",
    )
    require(
        disk_report == report, "final report differs from the verified in-memory report"
    )
    require(
        disk_receipt == outputs,
        "final receipt differs from the verified in-memory receipt",
    )
    provenance = require_mapping(
        disk_report.get("provenance"), "final report provenance"
    )
    require(
        provenance.get("config_sha256") == plan.get("config_sha256")
        and provenance.get("config_snapshot_bytes") == plan.get("config_snapshot_bytes")
        and provenance.get("tool_sha256") == plan.get("tool_sha256")
        and provenance.get("task_fingerprint_sha256")
        == plan.get("task_fingerprint_sha256"),
        "final report provenance differs from the frozen plan",
    )
    require(
        disk_report.get("content_baseline_parent")
        == plan.get("current_content_baseline_parent"),
        "final report content baseline commitment differs from the frozen plan",
    )
    expected_fingerprint = sha256_bytes(
        canonical_json_bytes(
            _report_fingerprint_payload_from_disk(
                disk_report, parent_hashes=parent_hashes
            )
        )
    )
    require(
        provenance.get("result_fingerprint_sha256") == expected_fingerprint,
        "final report result fingerprint mismatch",
    )
    require(
        disk_receipt.get("result_fingerprint_sha256") == expected_fingerprint,
        "final receipt result fingerprint mismatch",
    )
    require(
        _read_text_nofollow(markdown_path, field="final work-topic Markdown")
        == render_markdown(disk_report),
        "final Markdown differs from the verified report",
    )

    verified_checkpoints, verified_event_rows = revalidate_checkpoints_from_disk(
        output_dir,
        source_root=source_root,
        plan=plan,
    )
    require(
        verified_event_rows == event_rows,
        "post-publication checkpoint event count mismatch",
    )
    require(
        _checkpoint_tree(verified_checkpoints)
        == disk_report.get("checkpoint_evidence"),
        "post-publication checkpoint evidence mismatch",
    )
    final_source_evidence = _validate_source_tree_end(
        source_root=source_root,
        plan=plan,
    )
    require(
        final_source_evidence == disk_report.get("source_identity_evidence"),
        "post-publication source evidence mismatch",
    )
    _revalidate_content_freeze_parent(
        plan=plan, content_context=content_context
    )

    artifact_contracts = require_mapping(
        disk_receipt.get("artifacts"), "final receipt artifacts"
    )
    for key, raw in artifact_contracts.items():
        contract = require_mapping(raw, f"receipt artifact {key}")
        relative = safe_relative_path(contract.get("relative_path"), "receipt path")
        require(relative.as_posix() == key, "receipt artifact key/path mismatch")
        path = secure_join_regular(
            output_dir, relative, field="receipt-bound output artifact"
        )
        require(
            _rich_regular_identity(path, field="receipt-bound output artifact")["bytes"]
            == require_int(contract.get("bytes"), "receipt artifact bytes"),
            "receipt artifact byte count mismatch",
        )
        require(
            sha256_file(path)
            == require_sha256(contract.get("sha256"), "receipt artifact sha256"),
            "receipt artifact SHA-256 mismatch",
        )
    state = read_json(output_dir / "run_state.json", "final run state")
    require(
        state.get("status") == "COMPLETE_PENDING_RECEIPT"
        and state.get("complete_only_if_receipt_exists") is True
        and state.get("completion_receipt_relative_path") == receipt_path.name
        and state.get("task_fingerprint_sha256") == plan.get("task_fingerprint_sha256")
        and state.get("result_fingerprint_sha256") == expected_fingerprint,
        "final state/receipt lifecycle mismatch",
    )
    _validate_output_tree(output_dir)


def _output_manifest(
    output_dir: Path,
    *,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    paths = [
        output_dir / "audit_plan.json",
        output_dir / "run_state.json",
        output_dir / "sort_manifest.json",
        output_dir / "work_topic_relation.events.raw.bin",
        output_dir / "work_topic_relation.events.sorted.bin",
        output_dir / "work_topic_relation_audit.json",
        output_dir / "work_topic_relation_audit.md",
    ]
    dedup = output_dir / "work_topic_edges.deduplicated.sorted.bin"
    if dedup.exists():
        paths.append(dedup)
    paths.extend(
        sorted(
            (output_dir / "checkpoints").rglob("*.json"),
            key=lambda value: value.relative_to(output_dir).as_posix(),
        )
    )
    artifacts = {
        path.relative_to(output_dir).as_posix(): {
            "relative_path": path.relative_to(output_dir).as_posix(),
            "bytes": _rich_regular_identity(path, field="output artifact")["bytes"],
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    return {
        "schema_version": OUTPUTS_SCHEMA,
        "status": report["status"],
        "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
        "artifacts": artifacts,
        "checkpoint_evidence": report["checkpoint_evidence"],
    }


def _invalidate_failed_final_publication(
    output_dir: Path,
    *,
    plan: Mapping[str, Any],
    completed_files: int,
    event_rows: int,
) -> None:
    errors: list[tuple[str, BaseException]] = []
    receipt_path = output_dir / "work_topic_relation_audit.outputs.json"
    try:
        # A completion receipt is the authoritative positive surface and must be
        # invalidated before any best-effort removal of the descriptive files.
        _unlink_trusted_output(receipt_path, missing_ok=True)
    except BaseException as exc:  # pragma: no branch - exercised by fault injection
        errors.append(("completion_receipt", exc))

    try:
        atomic_write_json(
            output_dir / "run_state.json",
            {
                "schema_version": "openalex_g1_work_topic_relation_state.v1",
                "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
                "status": "FINALIZATION_FAILED_NO_VALID_RECEIPT",
                "completion_receipt_invalidation_attempted_first": True,
                "completed_files": completed_files,
                "total_files": len(plan["tasks"]),
                "event_rows": event_rows,
                "operator_paths_redacted": True,
            },
        )
    except BaseException as exc:  # pragma: no branch - exercised by fault injection
        errors.append(("failure_state", exc))

    for label, path in (
        ("report", output_dir / "work_topic_relation_audit.json"),
        ("markdown", output_dir / "work_topic_relation_audit.md"),
    ):
        try:
            _unlink_trusted_output(path, missing_ok=True)
        except BaseException as exc:  # pragma: no branch - fault-injection boundary
            errors.append((label, exc))

    if errors:
        labels = ",".join(label for label, _ in errors)
        raise RelationAuditError(
            "finalization cleanup failed after completion invalidation attempts: "
            f"{labels}"
        ) from errors[0][1]


def run_audit(
    *,
    config_path: Path,
    source_root: Path | None,
    taxonomy_root: Path | None,
    output_dir: Path,
    content_freeze_receipt: Path | None = None,
    max_files_per_table: int | None = None,
    hash_bucket_count: int | None = None,
    hash_bucket_index: int | None = None,
    batch_size: int | None = None,
    aggregation_chunk_rows: int | None = None,
    resume: bool = False,
    pause_after_new_files: int | None = None,
    authorize_full_scope: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config, verified_config_sha256, verified_config_bytes = _read_config_snapshot(
        config_path
    )
    context = validate_parents(config)
    full_scope = _full_scope_from_arguments(
        max_files_per_table=max_files_per_table,
        hash_bucket_count=hash_bucket_count,
        hash_bucket_index=hash_bucket_index,
    )
    if full_scope and not authorize_full_scope:
        raise RelationAuditError(
            "full-scope execution requires explicit authorize_full_scope after review"
        )
    receipt = _resolve_content_receipt(config, content_freeze_receipt)
    content_context = _validate_content_freeze_parent(
        config, receipt, required=full_scope
    )
    if resume and _has_valid_terminal_completion_receipt(output_dir):
        raise RelationAuditError(
            "completed output is terminal and cannot be resumed"
        )
    source_contract = require_mapping(config.get("source"), "source")
    taxonomy_contract = require_mapping(config.get("taxonomy"), "taxonomy")
    resolved_source = resolve_root(
        source_root,
        str(source_contract.get("root_env")),
        field="source root",
    )
    resolved_taxonomy = resolve_root(
        taxonomy_root,
        str(taxonomy_contract.get("root_env")),
        field="taxonomy root",
    )
    require(
        not _paths_overlap(resolved_source, resolved_taxonomy),
        "source and taxonomy roots may not overlap",
    )
    topic_bits = require_int(
        config.get("identifier_codec", {}).get("topic_bits"), "topic bits"
    )
    _, allowed_topics, taxonomy_topic_set_sha = load_topic_ids(
        resolved_taxonomy,
        taxonomy_contract,
        topic_bits=topic_bits,
    )
    execution = require_mapping(config.get("execution"), "execution")
    effective_batch = (
        batch_size
        if batch_size is not None
        else require_int(execution.get("batch_size"), "batch size")
    )
    effective_chunk = (
        aggregation_chunk_rows
        if aggregation_chunk_rows is not None
        else require_int(execution.get("aggregation_chunk_rows"), "chunk rows")
    )
    plan = build_plan(
        config,
        context,
        verified_config_sha256=verified_config_sha256,
        verified_config_bytes=verified_config_bytes,
        source_root=resolved_source,
        max_files_per_table=max_files_per_table,
        hash_bucket_count=hash_bucket_count,
        hash_bucket_index=hash_bucket_index,
        batch_size=effective_batch,
        aggregation_chunk_rows=effective_chunk,
        taxonomy_topic_set_sha256=taxonomy_topic_set_sha,
    )
    require(
        bool(plan["full_scope"]) == full_scope,
        "argument-derived scope differs from the materialized plan",
    )
    _bind_current_source_to_content_baseline(plan, content_context)
    plan["current_content_baseline_parent"] = content_context["commitment"]
    plan["task_fingerprint_sha256"] = _plan_fingerprint(plan)
    output = prepare_output_dir(
        output_dir,
        source_root=resolved_source,
        taxonomy_root=resolved_taxonomy,
        resume=resume,
    )
    enforce_resource_preflight(plan, output_dir=output)
    with output_lock(output):
        _validate_output_tree(output)
        plan_path = output / "audit_plan.json"
        if plan_path.exists():
            require(resume, "existing audit plan requires --resume")
            require(
                read_json(plan_path, "existing audit plan") == plan,
                "resume audit plan mismatch",
            )
        else:
            atomic_write_json(plan_path, plan)
        checkpoints, event_rows, paused = execute_scan(
            output,
            source_root=resolved_source,
            config=config,
            allowed_topics=allowed_topics,
            plan=plan,
            resume=resume,
            pause_after_new_files=pause_after_new_files,
        )
        if paused:
            return {
                "schema_version": "openalex_g1_work_topic_relation_pause.v1",
                "status": "PAUSED",
                "completed_files": len(checkpoints),
                "total_files": len(plan["tasks"]),
                "event_rows": event_rows,
                "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
                "operator_paths_redacted": True,
                "full_component_established": False,
            }
        sort_manifest = prepare_sorted_events(
            output,
            plan=plan,
            event_rows=event_rows,
            resume=resume,
        )
        sorted_path = output / "work_topic_relation.events.sorted.bin"
        edge, dedup = aggregate_edge_groups(
            sorted_path,
            event_rows=event_rows,
            output_dir=output,
            plan=plan,
            topic_bits=topic_bits,
            chunk_rows=effective_chunk,
        )
        work = aggregate_work_groups(
            sorted_path,
            event_rows=event_rows,
            topic_bits=topic_bits,
            chunk_rows=effective_chunk,
        )
        checkpoints, verified_event_rows = revalidate_checkpoints_from_disk(
            output,
            source_root=resolved_source,
            plan=plan,
        )
        require(
            verified_event_rows == event_rows,
            "final checkpoint event count differs from aggregation input",
        )
        source_identity_evidence = _validate_source_tree_end(
            source_root=resolved_source,
            plan=plan,
        )
        _revalidate_content_freeze_parent(
            plan=plan, content_context=content_context
        )
        report = build_report(
            config,
            context,
            plan,
            checkpoints,
            sort_manifest=sort_manifest,
            edge=edge,
            work=work,
            dedup_artifact=dedup,
            source_identity_evidence=source_identity_evidence,
        )
        final_paths = (
            output / "work_topic_relation_audit.json",
            output / "work_topic_relation_audit.md",
            output / "work_topic_relation_audit.outputs.json",
        )
        try:
            atomic_write_json(final_paths[0], report)
            atomic_write_text(final_paths[1], render_markdown(report))
            atomic_write_json(
                output / "run_state.json",
                {
                    "schema_version": "openalex_g1_work_topic_relation_state.v1",
                    "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
                    "status": "COMPLETE_PENDING_RECEIPT",
                    "completion_receipt_relative_path": (
                        "work_topic_relation_audit.outputs.json"
                    ),
                    "complete_only_if_receipt_exists": True,
                    "completed_files": len(checkpoints),
                    "total_files": len(plan["tasks"]),
                    "event_rows": event_rows,
                    "result_fingerprint_sha256": report["provenance"][
                        "result_fingerprint_sha256"
                    ],
                },
            )
            _validate_output_tree(output)
            outputs = _output_manifest(output, report=report)
            # The outputs manifest is the completion receipt and is published last.
            atomic_write_json(final_paths[2], outputs)
            _validate_output_tree(output)
            validate_final_publication_from_disk(
                output,
                report=report,
                outputs=outputs,
                plan=plan,
                parent_hashes=context["hashes"],
                source_root=resolved_source,
                event_rows=event_rows,
                content_context=content_context,
            )
        except BaseException:
            _invalidate_failed_final_publication(
                output,
                plan=plan,
                completed_files=len(checkpoints),
                event_rows=event_rows,
            )
            raise
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--taxonomy-root", type=Path)
    parser.add_argument("--content-freeze-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files-per-table", type=int)
    parser.add_argument("--hash-bucket-count", type=int)
    parser.add_argument("--hash-bucket-index", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--aggregation-chunk-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after-new-files", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required execution acknowledgement",
    )
    parser.add_argument(
        "--allow-full-scope",
        action="store_true",
        help="separate full-scope authorization after review",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        raise RelationAuditError("execution requires --execute")
    report = run_audit(
        config_path=args.config,
        source_root=args.source_root,
        taxonomy_root=args.taxonomy_root,
        content_freeze_receipt=args.content_freeze_receipt,
        output_dir=args.output_dir,
        max_files_per_table=args.max_files_per_table,
        hash_bucket_count=args.hash_bucket_count,
        hash_bucket_index=args.hash_bucket_index,
        batch_size=args.batch_size,
        aggregation_chunk_rows=args.aggregation_chunk_rows,
        resume=args.resume,
        pause_after_new_files=args.pause_after_new_files,
        authorize_full_scope=args.allow_full_scope,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def redacted_error_message(exc: BaseException) -> str:
    return (
        f"{type(exc).__name__}: relation audit failed; "
        "operator paths and payload details are redacted"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RelationAuditError as exc:
        print(f"ERROR: {redacted_error_message(exc)}", file=sys.stderr)
        raise SystemExit(2) from exc
