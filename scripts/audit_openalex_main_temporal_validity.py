#!/usr/bin/env python3
"""Audit current-snapshot temporal validity of the OpenAlex main-work table.

Only ``id``, ``publication_date``, ``publication_year``, ``created_date``, and
``updated_date`` are projected.  Citation edges and any ``available_at``
convention are deliberately outside this component.  The runner is CPU/I/O
only, file-checkpointed, resumable, and fail-closed.  A max-files execution is
always a parser pilot and can never receive the full component status.
"""

from __future__ import annotations

import argparse
import fcntl
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
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml"
)

CONFIG_SCHEMA = "openalex_g1_main_temporal_validity_config.v1"
PLAN_SCHEMA = "openalex_g1_main_temporal_validity_plan.v1"
CHECKPOINT_SCHEMA = "openalex_g1_main_temporal_validity_checkpoint.v1"
REPORT_SCHEMA = "openalex_g1_main_temporal_validity_report.v1"
OUTPUTS_SCHEMA = "openalex_g1_main_temporal_validity_outputs.v1"
STATE_SCHEMA = "openalex_g1_main_temporal_validity_state.v1"
PAUSE_SCHEMA = "openalex_g1_main_temporal_validity_pause.v1"

PRODUCTION_CONFIG_SHA256 = (
    "d208b45cc87de577726a8310a899eb66882a519bae29df5f7dbbf7833562b172"
)
# Synthetic tests bind this name to their closed fixture config.  The CLI has
# no digest override, so production accepts only the repository freeze.
FROZEN_CONFIG_SHA256 = PRODUCTION_CONFIG_SHA256

FULL_STATUS = (
    "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_VALIDITY_COMPONENT_WITH_FROZEN_EXCLUSIONS"
)
PILOT_STATUS = "PILOT_MAIN_TEMPORAL_VALIDITY_ONLY"
BLOCKED_STATUS = "BLOCKED_MAIN_TEMPORAL_VALIDITY_COMPONENT"
PROJECTED_COLUMNS = (
    "id",
    "publication_date",
    "publication_year",
    "created_date",
    "updated_date",
)
PROJECTED_TYPES = {
    "id": "string",
    "publication_date": "string",
    "publication_year": "double",
    "created_date": "string",
    "updated_date": "string",
}
CLAIM_BOUNDARY = (
    "A full component pass establishes exact current-snapshot parsing, pairwise "
    "publication-date/year concordance, frozen null and chronology buckets, and "
    "created-date <= updated-date compliance under the frozen exact exception "
    "policy for the declared main-work projection. It does not establish edge "
    "observation time or citation available_at, historical content identity, "
    "author entity or candidate eligibility, G-1 or G0, empirical G1, field "
    "selection, NABM effects, or any scientific effect. A max-files run is "
    "PILOT_MAIN_TEMPORAL_VALIDITY_ONLY and can never receive the full component "
    "status."
)
SAMPLE_LIMIT = 20
ALLOWED_TOP_LEVEL = {
    "audit.lock",
    "audit_plan.json",
    "run_state.json",
    "checkpoints",
    "main_temporal_validity_audit.json",
    "main_temporal_validity_audit.md",
    "main_temporal_validity_audit.outputs.json",
}


class TemporalAuditError(RuntimeError):
    """Raised when an evidence, policy, or execution contract is violated."""


@dataclass(frozen=True)
class FileTask:
    file_index: int
    table_name: str
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
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TemporalAuditError(message)


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
    require(path.parts and all(part not in {"", ".", ".."} for part in path.parts), f"{field} is unsafe")
    return path


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


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _assert_no_symlink_components(path: Path, *, field: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            break
        try:
            info = current.lstat()
        except OSError as exc:
            raise TemporalAuditError(f"cannot inspect {field}") from exc
        require(not stat.S_ISLNK(info.st_mode), f"{field} path component is a symlink")


def _read_regular_bytes_snapshot(
    path: Path, *, field: str
) -> tuple[bytes, str, dict[str, int]]:
    path = _absolute(path)
    _assert_no_symlink_components(path, field=field)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise TemporalAuditError(f"cannot open {field}") from exc
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
        raise TemporalAuditError(f"cannot read {field}") from exc
    finally:
        os.close(descriptor)


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    path = _absolute(path)
    _assert_no_symlink_components(path, field="hash input")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise TemporalAuditError("cannot open hash input") from exc
    digest = hashlib.sha256()
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "hash input is not regular")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while block := handle.read(chunk_bytes):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _rich_regular_identity(path: Path, *, field: str) -> dict[str, int]:
    path = _absolute(path)
    _assert_no_symlink_components(path, field=field)
    try:
        info = path.lstat()
    except OSError as exc:
        raise TemporalAuditError(f"cannot stat {field}") from exc
    require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
    return _rich_identity_from_stat(info)


def secure_join_regular(root: Path, relative: Path, *, field: str) -> Path:
    root = _absolute(root)
    _assert_no_symlink_components(root, field=f"{field} root")
    require(root.is_dir() and not root.is_symlink(), f"{field} root is not a directory")
    path = root / relative
    _assert_no_symlink_components(path, field=field)
    require(path.is_file() and not path.is_symlink(), f"{field} is not a regular file")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TemporalAuditError(f"{field} escapes root") from exc
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    left = _absolute(left)
    right = _absolute(right)
    return left == right or left in right.parents or right in left.parents


def resolve_root(explicit: Path | None, env_name: str) -> Path:
    raw = explicit if explicit is not None else Path(os.environ.get(env_name, ""))
    require(str(raw) not in {"", "."}, f"set {env_name} or pass --source-root")
    root = _absolute(raw)
    _assert_no_symlink_components(root, field="source root")
    require(root.is_dir() and not root.is_symlink(), "source root is not a directory")
    return root


def _trusted_directory_fd(path: Path, *, field: str) -> int:
    path = _absolute(path)
    _assert_no_symlink_components(path, field=field)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise TemporalAuditError(f"cannot open {field}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise TemporalAuditError(f"{field} is not a directory")
    return descriptor


def _validate_regular_output_entry(directory_fd: int, name: str, *, field: str) -> None:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise TemporalAuditError(f"cannot inspect {field}") from exc
    require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path = _absolute(path)
    directory_fd = _trusted_directory_fd(path.parent, field="artifact parent")
    temporary = f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    try:
        _validate_regular_output_entry(directory_fd, path.name, field="existing artifact")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.chmod(path.name, 0o600, dir_fd=directory_fd, follow_symlinks=False)
        os.fsync(directory_fd)
    except OSError as exc:
        raise TemporalAuditError("cannot atomically publish output artifact") from exc
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
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def _unlink_output(path: Path) -> None:
    directory_fd = _trusted_directory_fd(path.parent, field="artifact parent")
    try:
        _validate_regular_output_entry(directory_fd, path.name, field="output artifact")
        try:
            os.unlink(path.name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            pass
    finally:
        os.close(directory_fd)


def read_json(path: Path, field: str) -> dict[str, Any]:
    payload, _, _ = _read_regular_bytes_snapshot(path, field=field)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemporalAuditError(f"cannot parse {field}") from exc
    value = require_mapping(value, field)
    require(
        payload == canonical_json_bytes(value) + b"\n",
        f"{field} is not exact canonical JSON bytes",
    )
    return value


def _parse_content_freeze_json(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = require_mapping(json.loads(payload), field)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemporalAuditError(f"cannot parse {field}") from exc
    require(
        payload == content_freeze_published_json_bytes(value),
        f"{field} is not exact content-freeze published JSON bytes",
    )
    return value


def _validate_trusted_output_chain(output: Path) -> dict[str, Any]:
    euid = os.geteuid()
    current = Path(output.anchor)
    checked = 0
    for part in output.parts[1:]:
        current /= part
        info = current.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), "output chain is not a directory")
        require(info.st_uid in {0, euid}, "output chain has an untrusted owner")
        shared_write = bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
        if current == output:
            require(info.st_uid == euid, "output directory is not operator-owned")
            require(stat.S_IMODE(info.st_mode) == 0o700, "output directory must be mode 0700")
        elif shared_write:
            require(bool(info.st_mode & stat.S_ISVTX), "writable output ancestor lacks sticky protection")
        checked += 1
    return {
        "status": "VERIFIED_PRIVATE_0700_OWNER_CHAIN",
        "components_checked": checked,
        "operator_paths_redacted": True,
    }


def _secure_subdir(output: Path, *parts: str) -> Path:
    current = output
    for part in parts:
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", part) is not None, "unsafe output directory name")
        candidate = current / part
        if candidate.exists() or candidate.is_symlink():
            info = candidate.lstat()
            require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), "output subdirectory is unsafe")
        else:
            descriptor = _trusted_directory_fd(current, field="output parent")
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.chmod(candidate, 0o700, follow_symlinks=False)
        current = candidate
    return current


def prepare_output_dir(output_dir: Path, *, source_root: Path, resume: bool) -> Path:
    output = _absolute(output_dir)
    require(not _paths_overlap(output, source_root), "output may not overlap source root")
    require(not _paths_overlap(output, REPO_ROOT), "output may not overlap repository")
    _assert_no_symlink_components(output, field="output directory")
    if output.exists() or output.is_symlink():
        require(output.is_dir() and not output.is_symlink(), "output is not a directory")
    else:
        missing: list[Path] = []
        current = output
        while not current.exists() and not current.is_symlink():
            missing.append(current)
            current = current.parent
        output.mkdir(parents=True, mode=0o700)
        for created in reversed(missing):
            os.chmod(created, 0o700, follow_symlinks=False)
    trust = _validate_trusted_output_chain(output)
    require(trust["status"] == "VERIFIED_PRIVATE_0700_OWNER_CHAIN", "output trust failure")
    entries = {path.name for path in output.iterdir()}
    require(not (entries - ALLOWED_TOP_LEVEL), "output contains undeclared entries")
    if resume:
        require("audit_plan.json" in entries, "resume output lacks audit plan")
        require(
            not entries.intersection(
                {
                    "main_temporal_validity_audit.json",
                    "main_temporal_validity_audit.md",
                    "main_temporal_validity_audit.outputs.json",
                }
            ),
            "completed or stale publication is terminal and may not be resumed",
        )
    else:
        require(not entries, "non-empty output requires --resume")
    _secure_subdir(output, "checkpoints", "main")
    return output


@contextmanager
def output_lock(output_dir: Path) -> Iterator[None]:
    descriptor = _trusted_directory_fd(output_dir, field="output directory")
    try:
        _validate_regular_output_entry(descriptor, "audit.lock", field="audit lock")
        lock_fd = os.open(
            "audit.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
    finally:
        os.close(descriptor)
    os.fchmod(lock_fd, 0o600)
    handle = os.fdopen(lock_fd, "r+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TemporalAuditError("another temporal audit holds the output lock") from exc
        yield
    finally:
        handle.close()


def _expected_gate_common() -> dict[str, Any]:
    return {
        "raw_source_id_validity": "BLOCKED",
        "w_zero_entity_validity": "UNRESOLVED",
        "parent_content_identity": "UNVERIFIED",
        "temporal_available_at": "PENDING_NOT_IN_THIS_COMPONENT",
        "citation_edge_chronology": "PENDING_NOT_IN_THIS_COMPONENT",
        "author_entity_validity": "PENDING",
        "candidate_eligibility": "PENDING",
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def _expected_gates(kind: str) -> dict[str, Any]:
    common = _expected_gate_common()
    if kind == "full":
        return {
            "main_temporal_validity_component": "PASS",
            "publication_date_year_validity": "PASS_WITH_FROZEN_EXCLUSIONS",
            "created_updated_chronology": "PASS_WITH_FROZEN_EXACT_EXCEPTION_POLICY",
            "current_content_baseline_component": "PASS",
            "historical_content_identity": "UNVERIFIED",
            **common,
        }
    if kind == "pilot":
        return {
            "main_temporal_validity_component": "PILOT_ONLY",
            "publication_date_year_validity": "NOT_ESTABLISHED_PILOT",
            "created_updated_chronology": "NOT_ESTABLISHED_PILOT",
            "current_content_baseline_component": "NOT_REQUIRED_UNPROMOTABLE_PILOT",
            "historical_content_identity": "UNVERIFIED",
            **common,
        }
    return {
        "main_temporal_validity_component": "BLOCKED",
        "publication_date_year_validity": "BLOCKED",
        "created_updated_chronology": "BLOCKED",
        "current_content_baseline_component": "PASS",
        "historical_content_identity": "UNVERIFIED",
        **common,
    }


def _validate_config_contract(config: Mapping[str, Any]) -> None:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    source = require_mapping(config.get("source"), "source")
    require(str(source.get("lineage_id", "")), "source lineage is empty")
    snapshot = str(source.get("snapshot_date"))
    try:
        date.fromisoformat(snapshot)
    except ValueError as exc:
        raise TemporalAuditError("source snapshot date is invalid") from exc
    table = require_mapping(config.get("table"), "table")
    require(table.get("key") == "main", "table key drift")
    require(str(table.get("name", "")), "table name is empty")
    require(require_int(table.get("expected_files"), "expected files") > 0, "expected files must be positive")
    require(require_int(table.get("expected_footer_rows"), "expected footer rows") > 0, "expected footer rows must be positive")
    require(require_int(table.get("expected_bytes"), "expected bytes") > 0, "expected bytes must be positive")
    require(tuple(table.get("projected_columns", [])) == PROJECTED_COLUMNS, "projected columns drift")
    require(table.get("projected_types") == PROJECTED_TYPES, "projected types drift")

    identifier = require_mapping(config.get("identifier_policy"), "identifier policy")
    require(identifier.get("work_pattern") == r"^https://openalex[.]org/W(0|[1-9][0-9]*)$", "work ID grammar drift")
    w_minus = require_mapping(identifier.get("w_minus_1"), "W-1 policy")
    require(w_minus.get("raw_value") == "https://openalex.org/W-1", "W-1 token drift")
    require(w_minus.get("expected_count") == 1, "W-1 count drift")
    require(w_minus.get("disposition") == "frozen_source_row_quarantine", "W-1 disposition drift")
    safe_relative_path(w_minus.get("relative_path"), "W-1 path")
    require(require_int(w_minus.get("row_group"), "W-1 row group") >= 0, "W-1 row group is negative")
    require(require_int(w_minus.get("file_row"), "W-1 file row") >= 0, "W-1 file row is negative")
    w_zero = require_mapping(identifier.get("w_zero"), "W0 policy")
    require(w_zero == {
        "raw_value": "https://openalex.org/W0",
        "expected_count": 1,
        "disposition": "retain_with_entity_validity_unresolved",
    }, "W0 policy drift")

    policy = require_mapping(config.get("temporal_policy"), "temporal policy")
    require(policy.get("snapshot_date") == snapshot, "source and temporal snapshot differ")
    require(policy.get("strict_date_format") == "YYYY-MM-DD", "date format drift")
    require(policy.get("calendar") == "proleptic_gregorian", "calendar drift")
    publication = require_mapping(policy.get("publication_date_year"), "publication policy")
    require(publication.get("require_pairwise_null_concordance") is True, "null concordance policy drift")
    require(publication.get("require_finite_integral_year") is True, "year policy drift")
    require(publication.get("require_date_derived_year_equality") is True, "date-year policy drift")
    require(publication.get("both_null_disposition") == "exclude_from_temporal_analysis", "both-null disposition drift")
    require(publication.get("pre_1900_disposition") == "exclude_from_temporal_analysis", "pre-1900 disposition drift")
    require(publication.get("valid_window_start_inclusive") == "1900-01-01", "valid-window start drift")
    require(publication.get("snapshot_year_on_or_before_snapshot_disposition") == "retain_for_later_policy", "snapshot-year retained disposition drift")
    require(publication.get("snapshot_year_after_snapshot_disposition") == "exclude_from_temporal_analysis", "snapshot-year future disposition drift")
    require(publication.get("post_snapshot_year_disposition") == "exclude_from_temporal_analysis", "post-snapshot disposition drift")
    require(publication.get("invalid_parse_or_mismatch_disposition") == "block_component_pass", "publication failure disposition drift")
    require(publication.get("aggregate_regression_scope") == "grammar_valid_w_minus_1_quarantined_w0_retained", "aggregate regression scope drift")
    regression = require_mapping(publication.get("required_aggregate_regression"), "aggregate regression")
    regression_keys = {
        "publication_year_null_rows",
        "publication_year_pre_1900_rows",
        "publication_year_snapshot_year_rows",
        "publication_year_post_snapshot_year_rows",
        "publication_year_min",
        "publication_year_max",
    }
    require(set(regression) == regression_keys, "aggregate regression key drift")
    for key, value in regression.items():
        require(type(value) is int, f"aggregate regression {key} must be an integer")
    require(all(regression[key] >= 0 for key in regression_keys - {"publication_year_min", "publication_year_max"}), "aggregate regression count is negative")
    require(1 <= regression["publication_year_min"] <= regression["publication_year_max"] <= 9999, "aggregate regression extrema are invalid")
    created = require_mapping(policy.get("created_updated"), "created-updated policy")
    require(created.get("relation") == "created_date_lte_updated_date", "chronology relation drift")
    require(created.get("null_policy") == "reject_any_null", "chronology null policy drift")
    require(created.get("parse_error_policy") == "block_component_pass", "chronology parse policy drift")
    require(created.get("reversal_policy") == "exact_row_allowlist_only", "exception policy drift")
    require(created.get("exception_policy_resolution") == "FROZEN_EMPTY_EXACT_ALLOWLIST_V1", "exception resolution drift")
    require(tuple(created.get("exception_key_fields", [])) == (
        "id", "created_date", "updated_date", "relative_path", "row_group", "file_row"
    ), "exception key ABI drift")
    require(created.get("exceptions") == [], "chronology exception allowlist is not frozen empty")
    require(created.get("require_every_exception_observed_exactly_once") is True, "exception accounting drift")
    require(created.get("require_created_not_after_snapshot") is True, "created snapshot policy drift")
    require(created.get("require_updated_not_after_snapshot") is True, "updated snapshot policy drift")
    require(created.get("future_metadata_policy") == "exact_row_allowlist_only", "future metadata policy drift")
    require(created.get("future_exception_policy_resolution") == "FROZEN_EMPTY_EXACT_ALLOWLIST_V1", "future exception resolution drift")
    require(tuple(created.get("future_exception_key_fields", [])) == (
        "field", "id", "raw_date", "relative_path", "row_group", "file_row"
    ), "future exception key ABI drift")
    require(created.get("future_metadata_exceptions") == [], "future exception allowlist is not frozen empty")
    require(created.get("require_every_future_exception_observed_exactly_once") is True, "future exception accounting drift")

    parents = require_mapping(config.get("parents"), "parents")
    grammar_parent = require_mapping(parents.get("grammar_valid_main_recount"), "grammar parent")
    grammar = require_mapping(grammar_parent.get("required_exact_counts"), "grammar exact counts")
    grammar_keys = {
        "body_rows_scanned",
        "null_rows",
        "quarantined_malformed_rows",
        "unexpected_malformed_rows",
        "grammar_valid_rows",
        "grammar_valid_distinct_ids",
        "grammar_valid_duplicate_id_groups",
        "grammar_valid_duplicate_row_excess",
        "grammar_valid_max_multiplicity",
        "grammar_valid_min_numeric_id",
        "grammar_valid_max_numeric_id",
        "zero_id_rows_retained",
    }
    require(set(grammar) == grammar_keys, "grammar exact-count key drift")
    for key, value in grammar.items():
        require(type(value) is int and value >= 0, f"grammar exact count {key} is invalid")
    require(grammar["body_rows_scanned"] == table["expected_footer_rows"], "grammar/table row count drift")
    require(grammar["body_rows_scanned"] == grammar["null_rows"] + grammar["quarantined_malformed_rows"] + grammar["unexpected_malformed_rows"] + grammar["grammar_valid_rows"], "grammar row partition drift")
    require(grammar["null_rows"] == 0, "grammar null-row freeze drift")
    require(grammar["quarantined_malformed_rows"] == 1, "grammar quarantine count drift")
    require(grammar["unexpected_malformed_rows"] == 0, "grammar malformed count drift")
    require(grammar["grammar_valid_distinct_ids"] == grammar["grammar_valid_rows"], "grammar distinct-row freeze drift")
    require(grammar["grammar_valid_duplicate_id_groups"] == 0, "grammar duplicate-group freeze drift")
    require(grammar["grammar_valid_duplicate_row_excess"] == 0, "grammar duplicate-row freeze drift")
    require(grammar["grammar_valid_max_multiplicity"] == 1, "grammar multiplicity freeze drift")
    require(grammar["zero_id_rows_retained"] == 1, "grammar W0 freeze drift")

    content_parent = require_mapping(parents.get("current_content_baseline"), "content-freeze parent")
    require(content_parent.get("expected_basename") == "content_freeze_audit.outputs.json", "content receipt basename drift")
    require(content_parent.get("schema_version") == "openalex_g1_content_freeze_outputs.v1", "content receipt schema drift")
    require(content_parent.get("required_status") == "PASS_CURRENT_CONTENT_BASELINE_COMPONENT", "content receipt status drift")
    require_sha256(content_parent.get("required_plan_sha256"), "content plan anchor")
    require_sha256(content_parent.get("required_task_fingerprint_sha256"), "content task anchor")
    require(require_int(content_parent.get("required_file_count"), "content file count") > 0, "content file count must be positive")
    require(require_int(content_parent.get("required_bytes"), "content bytes") > 0, "content bytes must be positive")
    require(set(content_parent.get("required_receipt_artifacts", [])) == {
        "audit_plan.json", "content_freeze_audit.json", "content_freeze_audit.md"
    } and len(content_parent.get("required_receipt_artifacts", [])) == 3, "content artifact ABI drift")
    require(content_parent.get("separately_bound_run_state") == "run_state.json", "content state path drift")
    require(content_parent.get("required_run_state") == "COMPLETE", "content run-state drift")
    require(content_parent.get("full_scope_required") is True, "content full-scope requirement drift")
    require(content_parent.get("pilot_may_omit_only_because_unpromotable") is True, "content pilot omission drift")

    execution = require_mapping(config.get("execution"), "execution")
    require(execution.get("workload_class") == "local_cpu_io", "workload class drift")
    require(execution.get("gpu_used") is False, "GPU use is forbidden for this audit")
    require(execution.get("deterministic_table_order") == ["main"], "table order drift")
    require(execution.get("deterministic_file_order") == "lexical_relative_path", "file order drift")
    require(execution.get("checkpoint_unit") == "input_file", "checkpoint unit drift")
    require(execution.get("resume_verification") == "rehash_current_projected_payload_before_reusing_metrics", "resume policy drift")
    require(execution.get("verify_source_identity_before_and_after_file") is True, "source stability drift")
    require(execution.get("verify_config_and_parents_at_finalization") is True, "input stability drift")
    require(execution.get("refuse_symlink_inputs") is True, "symlink policy drift")
    require(execution.get("require_private_output_mode") == "0700", "output mode drift")
    require(execution.get("output_files_mode") == "0600", "file mode drift")
    require(execution.get("require_output_outside_source_and_repository") is True, "output isolation drift")
    require(execution.get("max_files_is_parser_pilot_only") is True, "pilot scope drift")
    resources = require_mapping(execution.get("resource_preflight"), "resource preflight")
    require(resources.get("full_scope_only") is True, "resource scope drift")
    require(resources.get("pilot_promotion_allowed") is False, "pilot promotion drift")
    success = require_mapping(config.get("success_contract"), "success contract")
    require(success.get("full_success_status") == FULL_STATUS, "full status drift")
    require(success.get("pilot_status") == PILOT_STATUS, "pilot status drift")
    require(success.get("blocked_status") == BLOCKED_STATUS, "blocked status drift")
    require(success.get("required_full_gate_status") == _expected_gates("full"), "full gate contract drift")
    require(success.get("required_pilot_gate_status") == _expected_gates("pilot"), "pilot gate contract drift")
    require(success.get("required_blocked_gate_status") == _expected_gates("blocked"), "blocked gate contract drift")
    require(" ".join(str(config.get("claim_boundary", "")).split()) == CLAIM_BOUNDARY, "claim boundary drift")
    out = set(str(value) for value in require_list(config.get("out_of_scope"), "out of scope"))
    require({"citation_edge_available_at", "referenced_work_edges", "counts_by_year", "fixed_horizon_outcomes"} <= out, "temporal edge exclusion drift")


def _read_config_snapshot(path: Path) -> tuple[dict[str, Any], str, int, dict[str, int]]:
    payload, observed, identity = _read_regular_bytes_snapshot(path, field="config")
    require(observed == FROZEN_CONFIG_SHA256, "config SHA-256 differs from frozen contract")
    try:
        config = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise TemporalAuditError("cannot parse config") from exc
    config = require_mapping(config, "config")
    _validate_config_contract(config)
    return config, observed, len(payload), identity


def _require_gate_subset(observed: Mapping[str, Any], expected: Mapping[str, Any], *, field: str) -> None:
    for key, value in expected.items():
        require(observed.get(key) == value, f"{field} mismatch")


def _repo_parent_snapshot(spec: Mapping[str, Any], *, field: str) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = safe_relative_path(spec.get("relative_path"), f"{field} path")
    path = secure_join_regular(REPO_ROOT, relative, field=field)
    payload, observed, identity = _read_regular_bytes_snapshot(path, field=field)
    require(observed == require_sha256(spec.get("sha256"), f"{field} sha256"), f"{field} SHA-256 mismatch")
    try:
        value = require_mapping(json.loads(payload), field)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemporalAuditError(f"cannot parse {field}") from exc
    require(value.get("schema_version") == spec.get("schema_version"), f"{field} schema mismatch")
    return value, {
        "relative_path": relative.as_posix(),
        "sha256": observed,
        "bytes": len(payload),
        "rich_identity": identity,
    }


def validate_parents(config: Mapping[str, Any]) -> dict[str, Any]:
    parents = require_mapping(config.get("parents"), "parents")
    inventory_spec = require_mapping(parents.get("inventory"), "inventory parent")
    inventory, inventory_snapshot = _repo_parent_snapshot(inventory_spec, field="inventory parent")
    root = require_mapping(inventory.get("root"), "inventory root")
    source = require_mapping(config.get("source"), "source")
    require(root.get("id") == source.get("root_id"), "inventory root ID mismatch")
    require(root.get("snapshot_date") == source.get("snapshot_date"), "inventory snapshot mismatch")
    require(root.get("lineage_id") == source.get("lineage_id"), "inventory lineage mismatch")
    require(root.get("inventory_fingerprint_sha256") == inventory_spec.get("inventory_fingerprint_sha256"), "inventory fingerprint mismatch")
    require(root.get("stability_fingerprint_sha256") == inventory_spec.get("stability_fingerprint_sha256"), "inventory stability mismatch")
    require(inventory_spec.get("content_sha256_complete") is False, "historical content identity must remain unverified")

    grammar_spec = require_mapping(parents.get("grammar_valid_main_recount"), "grammar parent")
    grammar, grammar_snapshot = _repo_parent_snapshot(grammar_spec, field="grammar parent")
    require(grammar.get("status") == grammar_spec.get("required_status"), "grammar status mismatch")
    require(grammar.get("exact_counts") == grammar_spec.get("required_exact_counts"), "grammar exact-count mismatch")
    _require_gate_subset(
        require_mapping(grammar.get("gate_status"), "grammar gates"),
        require_mapping(grammar_spec.get("required_gate_states"), "required grammar gates"),
        field="grammar gate",
    )

    work_spec = require_mapping(parents.get("work_source_membership"), "work-source parent")
    work, work_snapshot = _repo_parent_snapshot(work_spec, field="work-source parent")
    require(work.get("status") == work_spec.get("required_status"), "work-source status mismatch")
    require(work.get("provenance", {}).get("result_fingerprint_sha256") == work_spec.get("result_fingerprint_sha256"), "work-source fingerprint mismatch")
    _require_gate_subset(
        require_mapping(work.get("gate_status"), "work-source gates"),
        require_mapping(work_spec.get("required_gate_states"), "required work-source gates"),
        field="work-source gate",
    )
    return {
        "inventory": inventory,
        "grammar": grammar,
        "work_source": work,
        "snapshots": {
            "inventory": inventory_snapshot,
            "grammar_valid_main_recount": grammar_snapshot,
            "work_source_membership": work_snapshot,
        },
    }


def _projected_schema(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fields = require_mapping(record.get("schema_fields"), "inventory schema")
    result: dict[str, dict[str, Any]] = {}
    for column in PROJECTED_COLUMNS:
        require(column in fields, f"inventory schema missing {column}")
        result[column] = require_mapping(fields[column], f"schema {column}")
        require(result[column].get("type") == PROJECTED_TYPES[column], f"projected type mismatch: {column}")
    return result


def _task_from_mapping(value: Mapping[str, Any]) -> FileTask:
    return FileTask(
        file_index=require_int(value["file_index"], "file index"),
        table_name=str(value["table_name"]),
        relative_path=str(value["relative_path"]),
        bytes=require_int(value["bytes"], "file bytes"),
        mtime_ns=require_int(value["mtime_ns"], "file mtime"),
        footer_rows=require_int(value["footer_rows"], "footer rows"),
        projected_columns=tuple(str(item) for item in value["projected_columns"]),
        projected_schema={str(key): dict(item) for key, item in value["projected_schema"].items()},
        projected_schema_sha256=str(value["projected_schema_sha256"]),
        source_rich_identity={str(key): require_int(item, f"source identity {key}") for key, item in value["source_rich_identity"].items()},
        task_sha256=str(value["task_sha256"]),
    )


def _source_tree(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = [
        {
            "relative_path": str(task["relative_path"]),
            "rich_identity": dict(task["source_rich_identity"]),
        }
        for task in tasks
    ]
    return {
        "file_count": len(records),
        "tree_sha256": sha256_bytes(canonical_json_bytes(records)),
        "records": records,
    }


def _resource_estimate(config: Mapping[str, Any], *, selected_files: int, batch_size: int) -> dict[str, Any]:
    resources = require_mapping(config.get("execution", {}).get("resource_preflight"), "resource preflight")
    row_bytes = require_int(resources.get("projected_bytes_per_row_upper_bound"), "projected row bytes")
    checkpoint_bytes = require_int(resources.get("checkpoint_bytes_per_file_upper_bound"), "checkpoint bytes")
    report_bytes = require_int(resources.get("report_bytes_upper_bound"), "report bytes")
    disk_multiplier = require_int(resources.get("disk_safety_multiplier_milli"), "disk multiplier")
    memory_multiplier = require_int(resources.get("memory_safety_multiplier_milli"), "memory multiplier")
    require(disk_multiplier >= 1000 and memory_multiplier >= 1000, "resource multiplier is unsafe")
    artifact_upper = selected_files * checkpoint_bytes + report_bytes
    batch_upper = batch_size * row_bytes
    return {
        "artifact_bytes_upper_bound": artifact_upper,
        "batch_memory_bytes_upper_bound": batch_upper,
        "required_free_disk_bytes": (artifact_upper * disk_multiplier + 999) // 1000 + require_int(resources.get("disk_reserve_bytes"), "disk reserve"),
        "required_available_memory_bytes": (batch_upper * memory_multiplier + 999) // 1000 + require_int(resources.get("memory_reserve_bytes"), "memory reserve"),
        "full_scope_only": True,
        "pilot_promotion_allowed": False,
    }


def build_plan(
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    verified_config_sha256: str,
    verified_config_bytes: int,
    verified_config_identity: Mapping[str, int],
    source_root: Path,
    max_files: int | None,
    batch_size: int,
) -> dict[str, Any]:
    require(verified_config_sha256 == FROZEN_CONFIG_SHA256, "plan config digest mismatch")
    require(verified_config_bytes > 0, "config snapshot is empty")
    require(max_files is None or max_files > 0, "max-files must be positive")
    require(batch_size > 0, "batch size must be positive")
    table_contract = require_mapping(config.get("table"), "table")
    table_name = str(table_contract.get("name"))
    tables = {
        str(item.get("name")): require_mapping(item, "inventory table")
        for item in require_list(context["inventory"].get("tables"), "inventory tables")
    }
    require(table_name in tables, "inventory lacks main table")
    manifest = tables[table_name]
    require(manifest.get("file_count") == table_contract.get("expected_files"), "main file-count mismatch")
    require(manifest.get("footer_rows") == table_contract.get("expected_footer_rows"), "main footer-row mismatch")
    require(manifest.get("bytes") == table_contract.get("expected_bytes"), "main byte-count mismatch")
    files = sorted(
        [require_mapping(item, "inventory file") for item in require_list(manifest.get("files"), "inventory files")],
        key=lambda item: str(item.get("relative_path")),
    )
    selected = files if max_files is None else files[:max_files]
    tasks: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        relative = safe_relative_path(record.get("relative_path"), "source file")
        path = secure_join_regular(source_root, relative, field="source Parquet")
        identity = _rich_regular_identity(path, field="source Parquet")
        require(identity["bytes"] == require_int(record.get("bytes"), "file bytes"), "source byte count differs from inventory")
        require(identity["mtime_ns"] == require_int(record.get("mtime_ns"), "file mtime"), "source mtime differs from inventory")
        projected = _projected_schema(record)
        core = {
            "file_index": index,
            "table_name": table_name,
            "relative_path": relative.as_posix(),
            "bytes": identity["bytes"],
            "mtime_ns": identity["mtime_ns"],
            "footer_rows": require_int(record.get("footer_rows"), "footer rows"),
            "projected_columns": list(PROJECTED_COLUMNS),
            "projected_schema": projected,
            "projected_schema_sha256": sha256_bytes(canonical_json_bytes(projected)),
            "source_rich_identity": identity,
        }
        core["task_sha256"] = sha256_bytes(canonical_json_bytes(core))
        tasks.append(core)
    tool_payload, tool_sha, tool_identity = _read_regular_bytes_snapshot(Path(__file__), field="audit tool")
    del tool_payload
    plan = {
        "schema_version": PLAN_SCHEMA,
        "label": config.get("label"),
        "full_scope": max_files is None,
        "pilot_mode": None if max_files is None else "max_files_parser_pilot",
        "batch_size": batch_size,
        "config_sha256": verified_config_sha256,
        "config_snapshot_bytes": verified_config_bytes,
        "config_rich_identity": dict(verified_config_identity),
        "tool_sha256": tool_sha,
        "tool_rich_identity": tool_identity,
        "parent_snapshots": context["snapshots"],
        "table": {
            "name": table_name,
            "parent_files": len(files),
            "selected_files": len(selected),
            "parent_footer_rows": int(manifest["footer_rows"]),
            "selected_footer_rows": sum(int(item["footer_rows"]) for item in selected),
            "parent_bytes": int(manifest["bytes"]),
            "selected_bytes": sum(int(item["bytes"]) for item in selected),
        },
        "tasks": tasks,
        "source_tree_start": _source_tree(tasks),
        "resource_estimate": _resource_estimate(config, selected_files=len(tasks), batch_size=batch_size),
        "operator_paths_redacted": True,
    }
    plan["task_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


def _available_disk_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _available_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError) as exc:
        raise TemporalAuditError("cannot determine available memory") from exc


def enforce_resource_preflight(plan: Mapping[str, Any], *, output_dir: Path) -> None:
    estimate = require_mapping(plan.get("resource_estimate"), "resource estimate")
    require(estimate.get("pilot_promotion_allowed") is False, "pilot resource contract drift")
    if not bool(plan.get("full_scope")):
        return
    require(
        _available_disk_bytes(output_dir) >= require_int(estimate.get("required_free_disk_bytes"), "required disk"),
        "BLOCKED_FULL_EXECUTION_RESOURCE_PREFLIGHT: insufficient free disk",
    )
    require(
        _available_memory_bytes() >= require_int(estimate.get("required_available_memory_bytes"), "required memory"),
        "BLOCKED_FULL_EXECUTION_RESOURCE_PREFLIGHT: insufficient available memory",
    )


def _source_file(task: FileTask, source_root: Path) -> Path:
    return secure_join_regular(
        source_root,
        safe_relative_path(task.relative_path, "task path"),
        field="source Parquet",
    )


def _actual_projected_schema(parquet: pq.ParquetFile) -> dict[str, dict[str, Any]]:
    schema = parquet.schema_arrow
    result: dict[str, dict[str, Any]] = {}
    for column in PROJECTED_COLUMNS:
        field = schema.field(column)
        result[column] = {"nullable": bool(field.nullable), "type": str(field.type)}
    return result


def _open_validate_parquet(
    task: FileTask, source_root: Path
) -> tuple[Path, Any, dict[str, int], pq.ParquetFile]:
    path = _source_file(task, source_root)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise TemporalAuditError("cannot open source Parquet") from exc
    try:
        handle = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        identity = _rich_identity_from_stat(os.fstat(handle.fileno()))
        require(identity == task.source_rich_identity, "source rich identity differs from frozen plan")
        parquet = pq.ParquetFile(handle)
        require(parquet.metadata.num_rows == task.footer_rows, "source footer row count differs")
        schema = _actual_projected_schema(parquet)
        require(schema == task.projected_schema, "source projected schema differs from inventory")
        require(sha256_bytes(canonical_json_bytes(schema)) == task.projected_schema_sha256, "source projected schema hash mismatch")
    except BaseException:
        handle.close()
        raise
    return path, handle, identity, parquet


def _bool_array(value: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return np.asarray(value.to_numpy(zero_copy_only=False), dtype=bool)


def _strict_date(value: pa.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    null = _bool_array(pc.is_null(value))
    text = pc.fill_null(value, "")
    shaped = _bool_array(pc.match_substring_regex(text, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"))
    parsed = pc.strptime(text, format="%Y-%m-%d", unit="s", error_is_null=True)
    parsed_nonnull = ~_bool_array(pc.is_null(parsed))
    formatted = pc.fill_null(pc.strftime(parsed, format="%Y-%m-%d"), "")
    roundtrip = _bool_array(pc.equal(formatted, text))
    seconds = np.asarray(pc.fill_null(pc.cast(parsed, pa.int64()), 0).to_numpy(zero_copy_only=False), dtype=np.int64)
    days = seconds // 86_400
    years = np.asarray(pc.fill_null(pc.year(parsed), 0).to_numpy(zero_copy_only=False), dtype=np.int64)
    valid = (
        (~null)
        & shaped
        & parsed_nonnull
        & roundtrip
        & (years >= 1)
        & (years <= 9999)
    )
    return null, valid, days, years


def _year_values(value: pa.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    null = _bool_array(pc.is_null(value))
    values = np.asarray(pc.fill_null(value, 0.0).to_numpy(zero_copy_only=False), dtype=np.float64)
    valid = (~null) & np.isfinite(values) & (values == np.trunc(values)) & (values >= 1) & (values <= 9999)
    integers = np.zeros(len(values), dtype=np.int64)
    integers[valid] = values[valid].astype(np.int64)
    return null, valid, integers


METRIC_KEYS = (
    "raw_rows",
    "quarantined_w_minus_1_rows",
    "retained_rows",
    "null_id_rows",
    "grammar_valid_id_rows",
    "unexpected_malformed_id_rows",
    "w_zero_rows",
    "publication_both_null_rows",
    "publication_exactly_one_null_rows",
    "publication_both_nonnull_rows",
    "publication_date_parse_error_rows",
    "publication_year_invalid_rows",
    "publication_concordance_checked_rows",
    "publication_date_year_match_rows",
    "publication_date_year_mismatch_rows",
    "publication_pre_1900_rows",
    "publication_1900_before_snapshot_year_rows",
    "publication_snapshot_year_on_or_before_snapshot_rows",
    "publication_snapshot_year_after_snapshot_rows",
    "publication_post_snapshot_year_rows",
    "publication_year_null_rows",
    "publication_year_pre_1900_rows",
    "publication_year_snapshot_year_rows",
    "publication_year_post_snapshot_year_rows",
    "created_updated_both_null_rows",
    "created_updated_exactly_one_null_rows",
    "created_updated_both_nonnull_rows",
    "created_date_parse_error_rows",
    "updated_date_parse_error_rows",
    "created_updated_parsed_pair_rows",
    "created_lte_updated_rows",
    "created_gt_updated_rows",
    "created_gt_updated_allowlisted_rows",
    "created_gt_updated_unexpected_rows",
    "created_after_snapshot_rows",
    "updated_after_snapshot_rows",
    "future_metadata_allowlisted_rows",
    "future_metadata_unexpected_rows",
)


def _empty_metrics() -> dict[str, int]:
    return {key: 0 for key in METRIC_KEYS}


def _increment(metrics: dict[str, int], key: str, mask: np.ndarray) -> None:
    metrics[key] += int(np.count_nonzero(mask))


def _row_locator(task: FileTask, row_group: int, file_row: int, reason: str) -> dict[str, Any]:
    return {
        "file_index": task.file_index,
        "row_group": row_group,
        "file_row": file_row,
        "reason": reason,
        "operator_paths_redacted": True,
    }


def _exception_token(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(value)))


def _allowed_exception_tokens(config: Mapping[str, Any], key: str) -> set[str]:
    created = require_mapping(config.get("temporal_policy", {}).get("created_updated"), "created-updated policy")
    return {
        _exception_token(require_mapping(item, f"{key} item"))
        for item in require_list(created.get(key), key)
    }


def scan_task(
    task: FileTask,
    *,
    source_root: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    path, handle, identity, parquet = _open_validate_parquet(task, source_root)
    metrics = _empty_metrics()
    samples: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    chronology_observations: dict[str, int] = {}
    future_observations: dict[str, int] = {}
    publication_year_min: int | None = None
    publication_year_max: int | None = None
    digest = hashlib.sha256()
    source = require_mapping(config.get("source"), "source")
    id_policy = require_mapping(config.get("identifier_policy"), "identifier policy")
    allowed_chronology = _allowed_exception_tokens(config, "exceptions")
    allowed_future = _allowed_exception_tokens(config, "future_metadata_exceptions")
    snapshot = date.fromisoformat(str(source.get("snapshot_date")))
    snapshot_day = (snapshot - date(1970, 1, 1)).days
    snapshot_year = snapshot.year
    start_day = (date(1900, 1, 1) - date(1970, 1, 1)).days
    rows = 0
    try:
        for row_group in range(parquet.metadata.num_row_groups):
            group_offset = 0
            for batch in parquet.iter_batches(
                batch_size=require_int(plan.get("batch_size"), "batch size"),
                row_groups=[row_group],
                columns=list(PROJECTED_COLUMNS),
                use_threads=False,
            ):
                serialized = batch.serialize().to_pybytes()
                digest.update(len(serialized).to_bytes(8, "little"))
                digest.update(serialized)
                n = batch.num_rows
                metrics["raw_rows"] += n
                ids = batch.column(0)
                id_null = _bool_array(pc.is_null(ids))
                id_text = pc.fill_null(ids, "")
                grammar = _bool_array(pc.match_substring_regex(id_text, str(id_policy.get("work_pattern"))))
                w_minus = _bool_array(pc.equal(id_text, str(id_policy.get("w_minus_1", {}).get("raw_value"))))
                w_zero = _bool_array(pc.equal(id_text, str(id_policy.get("w_zero", {}).get("raw_value"))))
                retained = ~w_minus
                _increment(metrics, "quarantined_w_minus_1_rows", w_minus)
                _increment(metrics, "retained_rows", retained)
                _increment(metrics, "null_id_rows", retained & id_null)
                _increment(metrics, "grammar_valid_id_rows", retained & grammar)
                _increment(metrics, "unexpected_malformed_id_rows", retained & ~grammar)
                _increment(metrics, "w_zero_rows", retained & w_zero)
                for index in np.flatnonzero(w_minus):
                    quarantine.append(
                        {
                            "relative_path": task.relative_path,
                            "row_group": row_group,
                            "file_row": rows + group_offset + int(index),
                        }
                    )

                pub_date = batch.column(1)
                pub_year = batch.column(2)
                pub_null, pub_valid, pub_days, pub_derived_year = _strict_date(pub_date)
                year_null, year_valid, year_int = _year_values(pub_year)
                both_null = retained & pub_null & year_null
                exactly_one = retained & (pub_null ^ year_null)
                both_nonnull = retained & ~pub_null & ~year_null
                _increment(metrics, "publication_both_null_rows", both_null)
                _increment(metrics, "publication_exactly_one_null_rows", exactly_one)
                _increment(metrics, "publication_both_nonnull_rows", both_nonnull)
                _increment(metrics, "publication_date_parse_error_rows", retained & ~pub_null & ~pub_valid)
                _increment(metrics, "publication_year_invalid_rows", retained & ~year_null & ~year_valid)
                checked = retained & pub_valid & year_valid
                matched = checked & (pub_derived_year == year_int)
                mismatch = checked & ~matched
                _increment(metrics, "publication_concordance_checked_rows", checked)
                _increment(metrics, "publication_date_year_match_rows", matched)
                _increment(metrics, "publication_date_year_mismatch_rows", mismatch)
                _increment(metrics, "publication_pre_1900_rows", retained & pub_valid & (pub_days < start_day))
                _increment(metrics, "publication_1900_before_snapshot_year_rows", retained & pub_valid & (pub_days >= start_day) & (pub_derived_year < snapshot_year))
                _increment(metrics, "publication_snapshot_year_on_or_before_snapshot_rows", retained & pub_valid & (pub_derived_year == snapshot_year) & (pub_days <= snapshot_day))
                _increment(metrics, "publication_snapshot_year_after_snapshot_rows", retained & pub_valid & (pub_derived_year == snapshot_year) & (pub_days > snapshot_day))
                _increment(metrics, "publication_post_snapshot_year_rows", retained & pub_valid & (pub_derived_year > snapshot_year))
                _increment(metrics, "publication_year_null_rows", retained & year_null)
                valid_year_rows = retained & year_valid
                _increment(metrics, "publication_year_pre_1900_rows", valid_year_rows & (year_int < 1900))
                _increment(metrics, "publication_year_snapshot_year_rows", valid_year_rows & (year_int == snapshot_year))
                _increment(metrics, "publication_year_post_snapshot_year_rows", valid_year_rows & (year_int > snapshot_year))
                if np.any(valid_year_rows):
                    current_min = int(np.min(year_int[valid_year_rows]))
                    current_max = int(np.max(year_int[valid_year_rows]))
                    publication_year_min = current_min if publication_year_min is None else min(publication_year_min, current_min)
                    publication_year_max = current_max if publication_year_max is None else max(publication_year_max, current_max)

                created = batch.column(3)
                updated = batch.column(4)
                created_null, created_valid, created_days, _ = _strict_date(created)
                updated_null, updated_valid, updated_days, _ = _strict_date(updated)
                temporal_both_null = retained & created_null & updated_null
                temporal_one_null = retained & (created_null ^ updated_null)
                temporal_nonnull = retained & ~created_null & ~updated_null
                parsed_pair = retained & created_valid & updated_valid
                ordered = parsed_pair & (created_days <= updated_days)
                reversed_rows = parsed_pair & (created_days > updated_days)
                _increment(metrics, "created_updated_both_null_rows", temporal_both_null)
                _increment(metrics, "created_updated_exactly_one_null_rows", temporal_one_null)
                _increment(metrics, "created_updated_both_nonnull_rows", temporal_nonnull)
                _increment(metrics, "created_date_parse_error_rows", retained & ~created_null & ~created_valid)
                _increment(metrics, "updated_date_parse_error_rows", retained & ~updated_null & ~updated_valid)
                _increment(metrics, "created_updated_parsed_pair_rows", parsed_pair)
                _increment(metrics, "created_lte_updated_rows", ordered)
                _increment(metrics, "created_gt_updated_rows", reversed_rows)
                created_future = retained & created_valid & (created_days > snapshot_day)
                updated_future = retained & updated_valid & (updated_days > snapshot_day)
                _increment(metrics, "created_after_snapshot_rows", created_future)
                _increment(metrics, "updated_after_snapshot_rows", updated_future)

                interesting: dict[str, np.ndarray] = {
                    "publication_null_mismatch": exactly_one,
                    "publication_date_parse_error": retained & ~pub_null & ~pub_valid,
                    "publication_year_invalid": retained & ~year_null & ~year_valid,
                    "publication_date_year_mismatch": mismatch,
                    "created_date_parse_error": retained & ~created_null & ~created_valid,
                    "updated_date_parse_error": retained & ~updated_null & ~updated_valid,
                }
                for reason, mask in interesting.items():
                    for index in np.flatnonzero(mask)[: max(0, SAMPLE_LIMIT - len(samples))]:
                        samples.append(_row_locator(task, row_group, rows + group_offset + int(index), reason))

                for index in np.flatnonzero(reversed_rows):
                    idx = int(index)
                    record = {
                        "id": ids[idx].as_py(),
                        "created_date": created[idx].as_py(),
                        "updated_date": updated[idx].as_py(),
                        "relative_path": task.relative_path,
                        "row_group": row_group,
                        "file_row": rows + group_offset + idx,
                    }
                    token = _exception_token(record)
                    if token in allowed_chronology:
                        metrics["created_gt_updated_allowlisted_rows"] += 1
                        chronology_observations[token] = chronology_observations.get(token, 0) + 1
                    else:
                        metrics["created_gt_updated_unexpected_rows"] += 1
                        if len(samples) < SAMPLE_LIMIT:
                            samples.append(_row_locator(task, row_group, rows + group_offset + idx, "created_gt_updated_unexpected"))

                for field, values, mask in (
                    ("created_date", created, created_future),
                    ("updated_date", updated, updated_future),
                ):
                    for index in np.flatnonzero(mask):
                        idx = int(index)
                        record = {
                            "field": field,
                            "id": ids[idx].as_py(),
                            "raw_date": values[idx].as_py(),
                            "relative_path": task.relative_path,
                            "row_group": row_group,
                            "file_row": rows + group_offset + idx,
                        }
                        token = _exception_token(record)
                        if token in allowed_future:
                            metrics["future_metadata_allowlisted_rows"] += 1
                            future_observations[token] = future_observations.get(token, 0) + 1
                        else:
                            metrics["future_metadata_unexpected_rows"] += 1
                            if len(samples) < SAMPLE_LIMIT:
                                samples.append(_row_locator(task, row_group, rows + group_offset + idx, f"{field}_after_snapshot_unexpected"))
                group_offset += n
            rows += parquet.metadata.row_group(row_group).num_rows
        require(_rich_identity_from_stat(os.fstat(handle.fileno())) == identity, "open source file changed during scan")
        require(_rich_regular_identity(path, field="source Parquet") == identity, "source path changed during scan")
    except (OSError, pa.ArrowException, ValueError, OverflowError) as exc:
        raise TemporalAuditError("cannot scan source temporal projection") from exc
    finally:
        handle.close()
    require(rows == task.footer_rows, "projected row count differs from footer")
    require(metrics["raw_rows"] == task.footer_rows, "scanned batch rows differ from footer")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "task_sha256": task.task_sha256,
        "file_index": task.file_index,
        "relative_path": task.relative_path,
        "source_identity": {
            **task.source_rich_identity,
            "footer_rows": task.footer_rows,
            "projected_schema_sha256": task.projected_schema_sha256,
        },
        "projected_payload_sha256": digest.hexdigest(),
        "metrics": metrics,
        "metrics_sha256": sha256_bytes(canonical_json_bytes(metrics)),
        "publication_year_extrema": {
            "minimum": publication_year_min,
            "maximum": publication_year_max,
        },
        "quarantine_observations": quarantine,
        "chronology_exception_observations": chronology_observations,
        "future_exception_observations": future_observations,
        "violation_samples": samples,
        "operator_paths_redacted": True,
    }
    payload["checkpoint_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def checkpoint_path(output_dir: Path, task: FileTask) -> Path:
    token = sha256_bytes(task.relative_path.encode("utf-8"))
    return output_dir / "checkpoints" / "main" / f"{token}.json"


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    task: FileTask,
    *,
    source_root: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(checkpoint)
    digest = value.pop("checkpoint_sha256", None)
    require(checkpoint.get("schema_version") == CHECKPOINT_SCHEMA, "checkpoint schema mismatch")
    require(digest == sha256_bytes(canonical_json_bytes(value)), "checkpoint payload hash mismatch")
    require(checkpoint.get("task_fingerprint_sha256") == plan.get("task_fingerprint_sha256"), "checkpoint plan mismatch")
    require(checkpoint.get("task_sha256") == task.task_sha256, "checkpoint task mismatch")
    require(checkpoint.get("relative_path") == task.relative_path, "checkpoint path mismatch")
    require(checkpoint.get("metrics_sha256") == sha256_bytes(canonical_json_bytes(checkpoint.get("metrics"))), "checkpoint metrics hash mismatch")
    current = scan_task(task, source_root=source_root, config=config, plan=plan)
    require(current == checkpoint, "current projected payload or metrics differ from checkpoint")
    return dict(checkpoint)


def _checkpoint_tree(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    commitments = [
        {
            "file_index": int(item["file_index"]),
            "checkpoint_sha256": str(item["checkpoint_sha256"]),
            "projected_payload_sha256": str(item["projected_payload_sha256"]),
            "metrics_sha256": str(item["metrics_sha256"]),
        }
        for item in checkpoints
    ]
    return {
        "checkpoint_count": len(commitments),
        "checkpoint_tree_sha256": sha256_bytes(canonical_json_bytes(commitments)),
        "resume_verification": "REHASH_CURRENT_PROJECTED_PAYLOAD_BEFORE_REUSING_METRICS",
        "commitments": commitments,
    }


def _state(plan: Mapping[str, Any], status: str, completed: int, *, error: BaseException | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": STATE_SCHEMA,
        "status": status,
        "completed_files": completed,
        "total_files": len(plan["tasks"]),
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "operator_paths_redacted": True,
    }
    if error is not None:
        value["error"] = {
            "type": type(error).__name__[:80],
            "message": "audit paused; exception detail omitted for path redaction",
            "detail_omitted": True,
        }
    return value


def execute_scan(
    output_dir: Path,
    *,
    source_root: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    resume: bool,
    pause_after_new_files: int | None,
) -> tuple[list[dict[str, Any]], bool]:
    require(pause_after_new_files is None or pause_after_new_files > 0, "pause count must be positive")
    tasks = [_task_from_mapping(item) for item in plan["tasks"]]
    checkpoints: list[dict[str, Any]] = []
    gap_seen = False
    for task in tasks:
        path = checkpoint_path(output_dir, task)
        if path.exists() or path.is_symlink():
            require(resume, "existing checkpoint requires --resume")
            require(not gap_seen, "checkpoint tree is not a contiguous prefix")
            checkpoints.append(
                validate_checkpoint(
                    read_json(path, "checkpoint"),
                    task,
                    source_root=source_root,
                    config=config,
                    plan=plan,
                )
            )
        else:
            gap_seen = True
    new_files = 0
    for task in tasks[len(checkpoints) :]:
        checkpoint = scan_task(task, source_root=source_root, config=config, plan=plan)
        atomic_write_json(checkpoint_path(output_dir, task), checkpoint)
        checkpoints.append(checkpoint)
        new_files += 1
        atomic_write_json(output_dir / "run_state.json", _state(plan, "RUNNING", len(checkpoints)))
        if pause_after_new_files is not None and new_files >= pause_after_new_files and len(checkpoints) < len(tasks):
            atomic_write_json(output_dir / "run_state.json", _state(plan, "PAUSED", len(checkpoints)))
            return checkpoints, True
    return checkpoints, False


def revalidate_checkpoints_from_disk(
    output_dir: Path,
    *,
    source_root: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tasks = [_task_from_mapping(item) for item in plan["tasks"]]
    expected = [checkpoint_path(output_dir, task) for task in tasks]
    observed = sorted((output_dir / "checkpoints" / "main").glob("*.json"))
    require({path.name for path in observed} == {path.name for path in expected}, "checkpoint file set differs from plan")
    return [
        validate_checkpoint(
            read_json(path, "final checkpoint"),
            task,
            source_root=source_root,
            config=config,
            plan=plan,
        )
        for task, path in zip(tasks, expected, strict=True)
    ]


def _validate_source_tree_end(source_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    current: list[dict[str, Any]] = []
    for raw in plan["tasks"]:
        task = _task_from_mapping(raw)
        identity = _rich_regular_identity(_source_file(task, source_root), field="source Parquet")
        require(identity == task.source_rich_identity, "final source rich-stat tree differs from plan")
        item = dict(raw)
        item["source_rich_identity"] = identity
        current.append(item)
    tree = _source_tree(current)
    require(tree == plan.get("source_tree_start"), "final source tree commitment differs")
    return {
        "status": "PASS_RICH_STAT_TREE_START_END_EQUAL",
        "file_count": tree["file_count"],
        "tree_sha256": tree["tree_sha256"],
        "projected_payload_rehashed_from_checkpoints": True,
    }


def _validate_content_freeze_parent(
    config: Mapping[str, Any], receipt_path: Path | None, *, required: bool
) -> dict[str, Any]:
    contract = require_mapping(config.get("parents", {}).get("current_content_baseline"), "content-freeze parent")
    if receipt_path is None:
        require(not required, "full scope requires current-content outputs receipt")
        return {
            "commitment": {
                "status": "OMITTED_UNPROMOTABLE_PILOT",
                "current_content_baseline_component": "NOT_REQUIRED_UNPROMOTABLE_PILOT",
                "historical_content_identity": "UNVERIFIED",
                "operator_paths_redacted": True,
            },
            "runtime": None,
        }
    receipt_path = _absolute(receipt_path)
    require(receipt_path.name == contract.get("expected_basename"), "content-freeze receipt basename mismatch")
    _validate_trusted_output_chain(receipt_path.parent)
    receipt_payload, receipt_sha, receipt_identity = _read_regular_bytes_snapshot(receipt_path, field="content-freeze receipt")
    receipt = _parse_content_freeze_json(
        receipt_payload,
        field="content-freeze receipt",
    )
    require(set(receipt) == {
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
    }, "content-freeze receipt key set differs")
    require(receipt.get("schema_version") == contract.get("schema_version"), "content-freeze receipt schema mismatch")
    require(receipt.get("status") == contract.get("required_status"), "content-freeze status mismatch")
    require(receipt.get("plan_sha256") == contract.get("required_plan_sha256"), "content-freeze plan anchor mismatch")
    require(receipt.get("task_fingerprint_sha256") == contract.get("required_task_fingerprint_sha256"), "content-freeze task anchor mismatch")
    checkpoint_tree = require_mapping(receipt.get("checkpoint_tree"), "content-freeze checkpoint tree")
    require(checkpoint_tree.get("files") == contract.get("required_file_count"), "content-freeze file count mismatch")
    require(receipt.get("run_state_bound_separately") is True, "content-freeze state binding mismatch")
    require(receipt.get("completion_contract") == "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST", "content-freeze completion contract mismatch")
    require(receipt.get("path_redacted") is True, "content-freeze receipt path redaction mismatch")

    artifact_items = [
        require_mapping(item, "content-freeze artifact")
        for item in require_list(receipt.get("artifacts"), "content-freeze artifacts")
    ]
    artifact_paths = [str(item.get("relative_path")) for item in artifact_items]
    require(len(artifact_paths) == len(set(artifact_paths)), "content-freeze receipt has duplicate artifacts")
    artifacts = dict(zip(artifact_paths, artifact_items, strict=True))
    required_artifacts = set(str(item) for item in require_list(contract.get("required_receipt_artifacts"), "required content-freeze artifacts"))
    require(required_artifacts == set(artifacts), "content-freeze receipt artifact set mismatch")
    runtime_files: dict[str, dict[str, Any]] = {}
    decoded: dict[str, Any] = {}
    for relative_text in sorted(required_artifacts):
        relative = safe_relative_path(relative_text, "content-freeze artifact path")
        artifact_path = secure_join_regular(receipt_path.parent, relative, field="content-freeze artifact")
        payload, observed, identity = _read_regular_bytes_snapshot(artifact_path, field="content-freeze artifact")
        artifact = artifacts[relative_text]
        require(set(artifact) == {"relative_path", "bytes", "sha256"}, "content-freeze artifact key set differs")
        require_sha256(artifact.get("sha256"), "content-freeze artifact SHA-256")
        require(require_int(artifact.get("bytes"), "content-freeze artifact bytes") >= 0, "content-freeze artifact bytes are negative")
        require(observed == artifact.get("sha256"), "content-freeze artifact hash mismatch")
        require(len(payload) == artifact.get("bytes"), "content-freeze artifact byte mismatch")
        runtime_files[relative_text] = {"path": artifact_path, "sha256": observed, "rich_identity": identity}
        if relative.suffix == ".json":
            decoded[relative_text] = _parse_content_freeze_json(
                payload,
                field="content-freeze artifact JSON",
            )
    plan = require_mapping(decoded.get("audit_plan.json"), "content-freeze plan")
    report = require_mapping(decoded.get("content_freeze_audit.json"), "content-freeze report")
    require(plan.get("schema_version") == "openalex_g1_content_freeze_plan.v1", "content-freeze plan schema mismatch")
    require(plan.get("full_scope") is True, "content-freeze plan is not full scope")
    require(plan.get("max_files") is None, "content-freeze full plan has max-files")
    require(plan.get("plan_sha256") == contract.get("required_plan_sha256"), "content-freeze plan digest mismatch")
    plan_payload = dict(plan)
    plan_digest = plan_payload.pop("plan_sha256", None)
    require(plan_digest == sha256_bytes(canonical_json_bytes(plan_payload)), "content-freeze plan self-digest mismatch")
    require(plan.get("task_fingerprint_sha256") == contract.get("required_task_fingerprint_sha256"), "content-freeze plan task mismatch")
    require(plan.get("declared_file_count") == contract.get("required_file_count"), "content-freeze declared files mismatch")
    require(plan.get("declared_bytes") == contract.get("required_bytes"), "content-freeze declared bytes mismatch")
    require(plan.get("selected_file_count") == contract.get("required_file_count"), "content-freeze selected files mismatch")
    require(plan.get("selected_bytes") == contract.get("required_bytes"), "content-freeze selected bytes mismatch")
    require(len(require_list(plan.get("tasks"), "content-freeze tasks")) == contract.get("required_file_count"), "content-freeze task count mismatch")
    require(report.get("schema_version") == "openalex_g1_content_freeze_report.v1", "content-freeze report schema mismatch")
    require(report.get("status") == contract.get("required_status"), "content-freeze report status mismatch")
    require(report.get("scope", {}).get("full_scope") is True, "content-freeze report is not full scope")
    require(report.get("pass_current_content_baseline_component") is True, "content-freeze report did not pass component")
    baseline = require_mapping(report.get("content_baseline"), "content baseline")
    require(baseline.get("files") == contract.get("required_file_count"), "content baseline files mismatch")
    require(baseline.get("bytes") == contract.get("required_bytes"), "content baseline bytes mismatch")
    require(baseline.get("historical_identity_established") is False, "historical identity overclaim")
    require(baseline.get("prior_component_run_identity_established") is False, "prior-run identity overclaim")
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
    require(report.get("gate_status") == required_content_gates, "content baseline gate mismatch")
    scope = require_mapping(report.get("scope"), "content-freeze scope")
    require(scope.get("full_scope") == plan.get("full_scope"), "content report/plan full-scope mismatch")
    require(scope.get("max_files") == plan.get("max_files"), "content report/plan max-files mismatch")
    require(scope.get("declared_files") == plan.get("declared_file_count") == contract.get("required_file_count"), "content report declared files mismatch")
    require(scope.get("declared_bytes") == plan.get("declared_bytes") == contract.get("required_bytes"), "content report declared bytes mismatch")
    require(scope.get("selected_files") == plan.get("selected_file_count") == contract.get("required_file_count"), "content report selected files mismatch")
    require(scope.get("selected_bytes") == plan.get("selected_bytes") == contract.get("required_bytes"), "content report selected bytes mismatch")
    require(scope.get("gpu_used") is False, "content report GPU boundary mismatch")
    require(report.get("parent_inventory") == plan.get("parent_inventory"), "content report/plan parent inventory mismatch")
    require(report.get("execution", {}).get("path_redacted") is True, "content report execution path redaction mismatch")
    require(report.get("provenance", {}).get("path_redacted") is True, "content report provenance path redaction mismatch")
    require(receipt.get("content_tree_sha256") == baseline.get("content_tree_sha256"), "content-freeze content tree mismatch")
    require(receipt.get("result_fingerprint_sha256") == report.get("provenance", {}).get("result_fingerprint_sha256"), "content-freeze result fingerprint mismatch")
    deterministic_core = {
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "full_scope": plan.get("full_scope"),
        "max_files": plan.get("max_files"),
        "parent_manifest_sha256": plan.get("parent_inventory", {}).get("manifest_sha256"),
        "files": baseline.get("files"),
        "bytes": baseline.get("bytes"),
        "content_tree_sha256": baseline.get("content_tree_sha256"),
        "gate_status": report.get("gate_status"),
        "claim_boundary": report.get("claim_boundary"),
        "same_size_restored_mtime_limitation": report.get("same_size_restored_mtime_limitation"),
        "checkpoint_trust_limitation": report.get("checkpoint_trust_limitation"),
        "publication_completion_contract": report.get("publication", {}).get("completion_contract"),
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
    report_checkpoints = require_mapping(report.get("checkpoint_evidence"), "content-freeze checkpoint evidence")
    require(
        checkpoint_tree
        == {
            "files": report_checkpoints.get("checkpoint_files"),
            "bytes": report_checkpoints.get("checkpoint_bytes"),
            "tree_sha256": report_checkpoints.get("checkpoint_tree_sha256"),
        },
        "content-freeze checkpoint tree mismatch",
    )

    state_relative = safe_relative_path(contract.get("separately_bound_run_state"), "content-freeze state path")
    state_path = secure_join_regular(receipt_path.parent, state_relative, field="content-freeze run state")
    state_payload, state_sha, state_identity = _read_regular_bytes_snapshot(state_path, field="content-freeze run state")
    state = _parse_content_freeze_json(
        state_payload,
        field="content-freeze run state",
    )
    require(state.get("schema_version") == "openalex_g1_content_freeze_state.v1", "content-freeze state schema mismatch")
    require(state.get("status") == contract.get("required_run_state"), "content-freeze state is not COMPLETE")
    require(state.get("plan_sha256") == contract.get("required_plan_sha256"), "content-freeze state plan mismatch")
    require(state.get("task_fingerprint_sha256") == contract.get("required_task_fingerprint_sha256"), "content-freeze state task mismatch")
    require(state.get("completed_files") == contract.get("required_file_count"), "content-freeze state files mismatch")
    require(state.get("completed_bytes") == contract.get("required_bytes"), "content-freeze state bytes mismatch")
    require(state.get("total_files") == contract.get("required_file_count"), "content-freeze state total files mismatch")
    require(state.get("total_bytes") == contract.get("required_bytes"), "content-freeze state total bytes mismatch")
    require(state.get("path_redacted") is True, "content-freeze state path redaction mismatch")
    require("error" not in state, "content-freeze COMPLETE state contains error")
    runtime_files[state_relative.as_posix()] = {"path": state_path, "sha256": state_sha, "rich_identity": state_identity}
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
    main_items = [
        item
        for item in require_list(plan.get("tasks"), "content-freeze tasks")
        if str(item.get("table_name")) == str(config.get("table", {}).get("name"))
    ]
    source_tasks = {
        str(item.get("relative_path")): require_mapping(
            item.get("source_stat"),
            "content-freeze task source stat",
        )
        for item in main_items
    }
    require(len(main_items) == len(source_tasks), "content-freeze main task paths are duplicated")
    require(len(source_tasks) == config.get("table", {}).get("expected_files"), "content-freeze main task count mismatch")
    return {
        "commitment": commitment,
        "runtime": {
            "receipt": {"path": receipt_path, "sha256": receipt_sha, "rich_identity": receipt_identity},
            "files": runtime_files,
            "main_source_tasks": source_tasks,
        },
    }


def _bind_current_source_to_content_baseline(
    temporal_plan: Mapping[str, Any], content_context: dict[str, Any]
) -> None:
    runtime = content_context.get("runtime")
    if runtime is None:
        content_context["commitment"]["main_source_rich_stat_bridge"] = "OMITTED_PILOT"
        return
    frozen = runtime["main_source_tasks"]
    common_fields = ("device", "inode", "mode", "bytes", "mtime_ns", "ctime_ns")
    current = {
        str(item["relative_path"]): {
            field: (
                stat.S_IMODE(item["source_rich_identity"][field])
                if field == "mode"
                else item["source_rich_identity"][field]
            )
            for field in common_fields
        }
        for item in temporal_plan["tasks"]
    }
    require(set(current) == set(frozen), "content-freeze main task set differs from temporal plan")
    for relative, identity in current.items():
        require(set(frozen[relative]) == set(common_fields), "content-freeze source-stat ABI mismatch")
        require(identity == frozen[relative], "current source rich identity differs from content-freeze plan")
    records = [
        {"relative_path": relative, "rich_identity": current[relative]}
        for relative in sorted(current)
    ]
    content_context["commitment"]["main_source_rich_stat_bridge"] = {
        "status": "PASS_EXACT_MAIN_TASK_SET_AND_RICH_STAT_EQUALITY",
        "files": len(records),
        "tree_sha256": sha256_bytes(canonical_json_bytes(records)),
        "mode_representation": "permission_bits_matching_content_freeze_source_stat",
    }


def _revalidate_snapshot(path: Path, expected: Mapping[str, Any], *, field: str) -> None:
    _, observed, identity = _read_regular_bytes_snapshot(path, field=field)
    require(observed == expected.get("sha256"), f"{field} content changed")
    require(identity == expected.get("rich_identity"), f"{field} identity changed")


def _revalidate_frozen_inputs(
    *,
    config_path: Path,
    plan: Mapping[str, Any],
    content_context: Mapping[str, Any],
) -> None:
    _revalidate_snapshot(
        config_path,
        {"sha256": plan["config_sha256"], "rich_identity": plan["config_rich_identity"]},
        field="final config",
    )
    _revalidate_snapshot(
        Path(__file__),
        {"sha256": plan["tool_sha256"], "rich_identity": plan["tool_rich_identity"]},
        field="final audit tool",
    )
    for name, snapshot in plan["parent_snapshots"].items():
        path = REPO_ROOT / safe_relative_path(snapshot["relative_path"], "parent path")
        _revalidate_snapshot(path, snapshot, field=f"final {name} parent")
    runtime = content_context.get("runtime")
    if runtime is not None:
        _revalidate_snapshot(runtime["receipt"]["path"], runtime["receipt"], field="final content-freeze receipt")
        for name, snapshot in runtime["files"].items():
            _revalidate_snapshot(snapshot["path"], snapshot, field=f"final content-freeze artifact {name}")


def _merge_checkpoints(checkpoints: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, Any]]:
    metrics = _empty_metrics()
    minimum: int | None = None
    maximum: int | None = None
    quarantine: list[dict[str, Any]] = []
    chronology: dict[str, int] = {}
    future: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for key in METRIC_KEYS:
            metrics[key] += require_int(checkpoint["metrics"].get(key), f"checkpoint metric {key}")
        extrema = checkpoint["publication_year_extrema"]
        if extrema["minimum"] is not None:
            minimum = int(extrema["minimum"]) if minimum is None else min(minimum, int(extrema["minimum"]))
        if extrema["maximum"] is not None:
            maximum = int(extrema["maximum"]) if maximum is None else max(maximum, int(extrema["maximum"]))
        quarantine.extend(checkpoint["quarantine_observations"])
        for token, count in checkpoint["chronology_exception_observations"].items():
            chronology[token] = chronology.get(token, 0) + int(count)
        for token, count in checkpoint["future_exception_observations"].items():
            future[token] = future.get(token, 0) + int(count)
        samples.extend(checkpoint["violation_samples"][: max(0, SAMPLE_LIMIT - len(samples))])
    return metrics, {
        "publication_year_minimum": minimum,
        "publication_year_maximum": maximum,
        "quarantine": quarantine,
        "chronology_exception_observations": chronology,
        "future_exception_observations": future,
        "violation_samples": samples,
    }


def _exception_accounting(config: Mapping[str, Any], observations: Mapping[str, int], key: str) -> dict[str, Any]:
    expected = _allowed_exception_tokens(config, key)
    unused = sorted(expected - set(observations))
    unexpected = sorted(set(observations) - expected)
    repeated = sorted(token for token, count in observations.items() if count != 1)
    return {
        "declared": len(expected),
        "observed": sum(int(value) for value in observations.values()),
        "unused_tokens": unused,
        "unexpected_tokens": unexpected,
        "non_unit_tokens": repeated,
        "exact": not unused and not unexpected and not repeated,
    }


def build_report(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    source_evidence: Mapping[str, Any],
    content_commitment: Mapping[str, Any],
) -> dict[str, Any]:
    metrics, details = _merge_checkpoints(checkpoints)
    full_scope = bool(plan["full_scope"])
    policy = require_mapping(config.get("temporal_policy"), "temporal policy")
    created = require_mapping(policy.get("created_updated"), "created-updated policy")
    chronology = _exception_accounting(config, details["chronology_exception_observations"], "exceptions")
    future = _exception_accounting(config, details["future_exception_observations"], "future_metadata_exceptions")
    identifier = require_mapping(config.get("identifier_policy"), "identifier policy")
    expected_q = require_mapping(identifier.get("w_minus_1"), "W-1 policy")
    quarantine_exact = (
        len(details["quarantine"]) == int(expected_q["expected_count"])
        and all(
            item == {
                "relative_path": expected_q["relative_path"],
                "row_group": expected_q["row_group"],
                "file_row": expected_q["file_row"],
            }
            for item in details["quarantine"]
        )
    )
    policy_resolved = (
        policy.get("resolution") == "FROZEN_MAIN_TEMPORAL_POLICY_V1"
        and created.get("exception_policy_resolution") == "FROZEN_EMPTY_EXACT_ALLOWLIST_V1"
        and created.get("future_exception_policy_resolution") == "FROZEN_EMPTY_EXACT_ALLOWLIST_V1"
    )
    grammar_expected = require_mapping(
        config.get("parents", {}).get("grammar_valid_main_recount", {}).get("required_exact_counts"),
        "grammar expected counts",
    )
    expected_w0 = int(identifier.get("w_zero", {}).get("expected_count", -1))
    publication_parseable = (
        metrics["retained_rows"]
        - metrics["publication_both_null_rows"]
        - metrics["publication_exactly_one_null_rows"]
        - metrics["publication_date_parse_error_rows"]
    )
    publication_bucket_total = sum(
        metrics[key]
        for key in (
            "publication_pre_1900_rows",
            "publication_1900_before_snapshot_year_rows",
            "publication_snapshot_year_on_or_before_snapshot_rows",
            "publication_snapshot_year_after_snapshot_rows",
            "publication_post_snapshot_year_rows",
        )
    )
    accounting = {
        "raw_rows_equal_selected_footer_rows": metrics["raw_rows"] == plan["table"]["selected_footer_rows"],
        "raw_equals_quarantine_plus_retained": metrics["raw_rows"] == metrics["quarantined_w_minus_1_rows"] + metrics["retained_rows"],
        "retained_equals_grammar_plus_unexpected": metrics["retained_rows"] == metrics["grammar_valid_id_rows"] + metrics["unexpected_malformed_id_rows"],
        "publication_null_partition_conserves_retained": metrics["retained_rows"] == metrics["publication_both_null_rows"] + metrics["publication_exactly_one_null_rows"] + metrics["publication_both_nonnull_rows"],
        "publication_concordance_partition": metrics["publication_concordance_checked_rows"] == metrics["publication_date_year_match_rows"] + metrics["publication_date_year_mismatch_rows"],
        "publication_buckets_conserve_parseable_dates": publication_parseable == publication_bucket_total,
        "created_updated_null_partition_conserves_retained": metrics["retained_rows"] == metrics["created_updated_both_null_rows"] + metrics["created_updated_exactly_one_null_rows"] + metrics["created_updated_both_nonnull_rows"],
        "created_updated_order_partition": metrics["created_updated_parsed_pair_rows"] == metrics["created_lte_updated_rows"] + metrics["created_gt_updated_rows"],
        "reversal_exception_partition": metrics["created_gt_updated_rows"] == metrics["created_gt_updated_allowlisted_rows"] + metrics["created_gt_updated_unexpected_rows"],
        "future_metadata_exception_partition": metrics["created_after_snapshot_rows"] + metrics["updated_after_snapshot_rows"] == metrics["future_metadata_allowlisted_rows"] + metrics["future_metadata_unexpected_rows"],
    }
    if full_scope:
        accounting.update(
            {
                "retained_equals_parent_grammar_valid_rows": metrics["retained_rows"] == grammar_expected.get("grammar_valid_rows"),
                "quarantine_equals_parent_and_frozen_count": metrics["quarantined_w_minus_1_rows"] == grammar_expected.get("quarantined_malformed_rows") == expected_q.get("expected_count"),
                "w0_equals_frozen_count": metrics["w_zero_rows"] == expected_w0,
                "null_ids_equal_parent_count": metrics["null_id_rows"] == grammar_expected.get("null_rows"),
                "unexpected_malformed_equal_parent_count": metrics["unexpected_malformed_id_rows"] == grammar_expected.get("unexpected_malformed_rows"),
            }
        )
    violations = {
        "unexpected_malformed_id_rows": metrics["unexpected_malformed_id_rows"],
        "publication_pairwise_null_mismatch_rows": metrics["publication_exactly_one_null_rows"],
        "publication_date_parse_error_rows": metrics["publication_date_parse_error_rows"],
        "publication_year_invalid_rows": metrics["publication_year_invalid_rows"],
        "publication_date_year_mismatch_rows": metrics["publication_date_year_mismatch_rows"],
        "created_updated_null_rows": metrics["created_updated_both_null_rows"] + metrics["created_updated_exactly_one_null_rows"],
        "created_date_parse_error_rows": metrics["created_date_parse_error_rows"],
        "updated_date_parse_error_rows": metrics["updated_date_parse_error_rows"],
        "created_gt_updated_unexpected_rows": metrics["created_gt_updated_unexpected_rows"],
        "future_metadata_unexpected_rows": metrics["future_metadata_unexpected_rows"],
        "chronology_exception_contract_failures": 0 if chronology["exact"] else 1,
        "future_exception_contract_failures": 0 if future["exact"] else 1,
        "quarantine_contract_failures": 0 if (quarantine_exact or not full_scope) else 1,
        "policy_resolution_failures": 0 if policy_resolved else 1,
        "accounting_contract_failures": sum(not value for value in accounting.values()),
    }
    regression_expected = require_mapping(policy.get("publication_date_year", {}).get("required_aggregate_regression"), "aggregate regression")
    regression_observed = {
        "publication_year_null_rows": metrics["publication_year_null_rows"],
        "publication_year_pre_1900_rows": metrics["publication_year_pre_1900_rows"],
        "publication_year_snapshot_year_rows": metrics["publication_year_snapshot_year_rows"],
        "publication_year_post_snapshot_year_rows": metrics["publication_year_post_snapshot_year_rows"],
        "publication_year_min": details["publication_year_minimum"],
        "publication_year_max": details["publication_year_maximum"],
    }
    regression_exact = regression_observed == regression_expected if full_scope else False
    violations["aggregate_regression_failures"] = 0 if (regression_exact or not full_scope) else 1
    total_violations = sum(violations.values())
    if not full_scope:
        status = PILOT_STATUS
        gates = _expected_gates("pilot")
    elif total_violations == 0:
        status = FULL_STATUS
        gates = _expected_gates("full")
    else:
        status = BLOCKED_STATUS
        gates = _expected_gates("blocked")
    deterministic = {
        "status": status,
        "gate_status": gates,
        "scope": {"full_scope": full_scope, "pilot_mode": plan["pilot_mode"], "selected_files": plan["table"]["selected_files"], "selected_footer_rows": plan["table"]["selected_footer_rows"]},
        "metrics": metrics,
        "publication_year_extrema": {"minimum": details["publication_year_minimum"], "maximum": details["publication_year_maximum"]},
        "aggregate_regression": {"expected": regression_expected, "observed": regression_observed, "exact": regression_exact},
        "violations": violations,
        "accounting_contract": accounting,
        "exception_accounting": {"chronology": chronology, "future_metadata": future},
        "checkpoint_evidence": _checkpoint_tree(checkpoints),
        "source_identity_evidence": dict(source_evidence),
        "content_baseline_parent": dict(content_commitment),
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    fingerprint = sha256_bytes(canonical_json_bytes(deterministic))
    return {
        "schema_version": REPORT_SCHEMA,
        "label": config.get("label"),
        **deterministic,
        "pass_main_temporal_validity_component": status == FULL_STATUS,
        "pass_g1": False,
        "publication_buckets": {
            key: metrics[key]
            for key in (
                "publication_both_null_rows",
                "publication_pre_1900_rows",
                "publication_1900_before_snapshot_year_rows",
                "publication_snapshot_year_on_or_before_snapshot_rows",
                "publication_snapshot_year_after_snapshot_rows",
                "publication_post_snapshot_year_rows",
            )
        },
        "created_updated_snapshot_buckets": {
            "created_after_snapshot_rows": metrics["created_after_snapshot_rows"],
            "updated_after_snapshot_rows": metrics["updated_after_snapshot_rows"],
        },
        "quarantine": {
            "w_minus_1_excluded_from_all_temporal_metrics": True,
            "observed_rows": len(details["quarantine"]),
            "exact_full_scope_match": quarantine_exact if full_scope else None,
        },
        "violation_samples": details["violation_samples"],
        "policy": {
            "resolution": policy.get("resolution"),
            "fully_resolved_for_pass": policy_resolved,
            "available_at_or_citation_edges_included": False,
        },
        "provenance": {
            "config_sha256": plan["config_sha256"],
            "config_snapshot_bytes": plan["config_snapshot_bytes"],
            "tool_sha256": plan["tool_sha256"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "result_fingerprint_sha256": fingerprint,
            "operator_paths_redacted": True,
        },
        "release_boundary": {
            "repository_surfaces": ["script", "frozen_config", "tests", "contract_docs"],
            "runtime_checkpoints_reports_receipts": "EXCLUDED_FROM_PACKAGE_AND_RELEASE",
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    scope = report["scope"]
    return "\n".join(
        [
            "# OpenAlex Main Temporal Validity Audit",
            "",
            f"- Status: `{report['status']}`",
            f"- Full scope: `{str(scope['full_scope']).lower()}`",
            f"- Selected files: `{scope['selected_files']}`",
            f"- Selected footer rows: `{scope['selected_footer_rows']}`",
            f"- Component pass: `{str(report['pass_main_temporal_validity_component']).lower()}`",
            f"- Current-content baseline: `{report['gate_status']['current_content_baseline_component']}`",
            f"- Historical content identity: `{report['gate_status']['historical_content_identity']}`",
            "- Citation available_at / edge chronology: `PENDING_NOT_IN_THIS_COMPONENT`",
            "- G-1 / G0 / empirical G1 / scientific effect: `PENDING / PENDING / NOT_RUN / NOT_ESTABLISHED`",
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )


def _output_manifest(output_dir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        output_dir / "audit_plan.json",
        output_dir / "run_state.json",
        output_dir / "main_temporal_validity_audit.json",
        output_dir / "main_temporal_validity_audit.md",
        *sorted((output_dir / "checkpoints" / "main").glob("*.json")),
    ]
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
        "task_fingerprint_sha256": report["provenance"]["task_fingerprint_sha256"],
        "artifacts": artifacts,
        "completion_contract": "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED",
        "operator_paths_redacted": True,
    }


def _validate_output_tree(output_dir: Path, *, final: bool) -> None:
    root_info = output_dir.lstat()
    require(stat.S_ISDIR(root_info.st_mode) and stat.S_IMODE(root_info.st_mode) == 0o700, "output root mode drift")
    entries = {path.name: path for path in output_dir.iterdir()}
    require(not (set(entries) - ALLOWED_TOP_LEVEL), "output contains undeclared entries")
    if final:
        require(set(entries) == ALLOWED_TOP_LEVEL, "final output entry set differs")
    for name, path in entries.items():
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "output entry is a symlink")
        if name == "checkpoints":
            require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700, "checkpoint directory mode drift")
        else:
            require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600, "output file mode drift")
    checkpoint_root = output_dir / "checkpoints"
    if checkpoint_root.exists():
        checkpoint_entries = {path.name: path for path in checkpoint_root.iterdir()}
        require(set(checkpoint_entries) == {"main"}, "checkpoint root entry set differs")
        main = checkpoint_root / "main"
        require(main.is_dir() and not main.is_symlink() and stat.S_IMODE(main.lstat().st_mode) == 0o700, "main checkpoint directory drift")
        for path in main.iterdir():
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), "checkpoint artifact is unsafe")
            require(stat.S_IMODE(info.st_mode) == 0o600, "checkpoint file mode drift")
            require(re.fullmatch(r"[0-9a-f]{64}[.]json", path.name) is not None, "checkpoint filename is undeclared")


def _report_fingerprint(report: Mapping[str, Any]) -> str:
    keys = (
        "status",
        "gate_status",
        "scope",
        "metrics",
        "publication_year_extrema",
        "aggregate_regression",
        "violations",
        "accounting_contract",
        "exception_accounting",
        "checkpoint_evidence",
        "source_identity_evidence",
        "content_baseline_parent",
        "task_fingerprint_sha256",
        "claim_boundary",
    )
    return sha256_bytes(canonical_json_bytes({key: report[key] for key in keys}))


def validate_final_publication(
    output_dir: Path,
    *,
    report: Mapping[str, Any],
    outputs: Mapping[str, Any],
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_root: Path,
    config_path: Path,
    content_context: Mapping[str, Any],
) -> None:
    _validate_output_tree(output_dir, final=True)
    disk_plan = read_json(output_dir / "audit_plan.json", "final audit plan")
    disk_report = read_json(output_dir / "main_temporal_validity_audit.json", "final temporal report")
    disk_outputs = read_json(output_dir / "main_temporal_validity_audit.outputs.json", "final temporal receipt")
    require(disk_plan == plan, "final plan differs after write")
    require(disk_report == report, "final report differs after write")
    require(disk_outputs == outputs, "final receipt differs after write")
    require(disk_report.get("schema_version") == REPORT_SCHEMA, "final report schema mismatch")
    require(disk_outputs.get("schema_version") == OUTPUTS_SCHEMA, "final receipt schema mismatch")
    require(_report_fingerprint(disk_report) == disk_report.get("provenance", {}).get("result_fingerprint_sha256"), "final report fingerprint mismatch")
    markdown_payload, _, _ = _read_regular_bytes_snapshot(output_dir / "main_temporal_validity_audit.md", field="final temporal Markdown")
    require(markdown_payload.decode("utf-8") == render_markdown(disk_report), "final Markdown differs from report")
    artifacts = require_mapping(disk_outputs.get("artifacts"), "receipt artifacts")
    for key, raw in artifacts.items():
        contract = require_mapping(raw, f"receipt artifact {key}")
        relative = safe_relative_path(contract.get("relative_path"), "receipt path")
        require(relative.as_posix() == key, "receipt artifact key/path mismatch")
        path = secure_join_regular(output_dir, relative, field="receipt artifact")
        require(_rich_regular_identity(path, field="receipt artifact")["bytes"] == contract.get("bytes"), "receipt artifact byte mismatch")
        require(sha256_file(path) == contract.get("sha256"), "receipt artifact hash mismatch")
    checkpoints = revalidate_checkpoints_from_disk(
        output_dir,
        source_root=source_root,
        config=config,
        plan=plan,
    )
    require(_checkpoint_tree(checkpoints) == disk_report.get("checkpoint_evidence"), "final checkpoint evidence mismatch")
    source_evidence = _validate_source_tree_end(source_root, plan)
    require(source_evidence == disk_report.get("source_identity_evidence"), "final source evidence mismatch")
    _revalidate_frozen_inputs(config_path=config_path, plan=plan, content_context=content_context)
    state = read_json(output_dir / "run_state.json", "final run state")
    require(state.get("status") == "COMPLETE_PENDING_RECEIPT", "final state lifecycle mismatch")
    require(state.get("completion_receipt_relative_path") == "main_temporal_validity_audit.outputs.json", "final receipt path mismatch")
    require(state.get("complete_only_if_receipt_exists") is True, "final receipt requirement mismatch")
    require(state.get("result_fingerprint_sha256") == report["provenance"]["result_fingerprint_sha256"], "final state result mismatch")
    _validate_output_tree(output_dir, final=True)


def _remove_final_publication(output_dir: Path) -> None:
    for name in (
        "main_temporal_validity_audit.outputs.json",
        "main_temporal_validity_audit.md",
        "main_temporal_validity_audit.json",
    ):
        descriptor = _trusted_directory_fd(output_dir, field="output directory")
        try:
            try:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            require(
                stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode),
                "cannot remove non-file final publication",
            )
            os.unlink(name, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _resolve_content_receipt(config: Mapping[str, Any], explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    contract = require_mapping(config.get("parents", {}).get("current_content_baseline"), "content-freeze parent")
    raw = os.environ.get(str(contract.get("receipt_path_env")), "")
    return None if not raw else Path(raw)


def run_audit(
    *,
    config_path: Path,
    source_root: Path | None,
    content_freeze_receipt: Path | None,
    output_dir: Path,
    max_files: int | None = None,
    batch_size: int | None = None,
    resume: bool = False,
    pause_after_new_files: int | None = None,
    authorize_full_scope: bool = False,
) -> dict[str, Any]:
    config_path = _absolute(config_path)
    config, config_sha, config_bytes, config_identity = _read_config_snapshot(config_path)
    context = validate_parents(config)
    full_scope = max_files is None
    if full_scope and not authorize_full_scope:
        raise TemporalAuditError("full-scope execution requires explicit --authorize-full")
    receipt = _resolve_content_receipt(config, content_freeze_receipt)
    content_context = _validate_content_freeze_parent(config, receipt, required=full_scope)
    source_contract = require_mapping(config.get("source"), "source")
    root = resolve_root(source_root, str(source_contract.get("root_env")))
    execution = require_mapping(config.get("execution"), "execution")
    effective_batch = batch_size if batch_size is not None else require_int(execution.get("batch_size"), "batch size")
    plan = build_plan(
        config,
        context,
        verified_config_sha256=config_sha,
        verified_config_bytes=config_bytes,
        verified_config_identity=config_identity,
        source_root=root,
        max_files=max_files,
        batch_size=effective_batch,
    )
    _bind_current_source_to_content_baseline(plan, content_context)
    plan["current_content_baseline_parent"] = content_context["commitment"]
    plan["task_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes({key: value for key, value in plan.items() if key != "task_fingerprint_sha256"}))
    output = prepare_output_dir(output_dir, source_root=root, resume=resume)
    enforce_resource_preflight(plan, output_dir=output)
    with output_lock(output):
        _validate_output_tree(output, final=False)
        plan_path = output / "audit_plan.json"
        if plan_path.exists():
            require(resume, "existing audit plan requires --resume")
            require(read_json(plan_path, "existing audit plan") == plan, "resume audit plan mismatch")
        else:
            atomic_write_json(plan_path, plan)
        finalization_started = False
        try:
            atomic_write_json(output / "run_state.json", _state(plan, "RUNNING", 0))
            checkpoints, paused = execute_scan(
                output,
                source_root=root,
                config=config,
                plan=plan,
                resume=resume,
                pause_after_new_files=pause_after_new_files,
            )
            if paused:
                return {
                    "schema_version": PAUSE_SCHEMA,
                    "status": "PAUSED",
                    "completed_files": len(checkpoints),
                    "total_files": len(plan["tasks"]),
                    "full_component_established": False,
                    "operator_paths_redacted": True,
                }
            checkpoints = revalidate_checkpoints_from_disk(
                output,
                source_root=root,
                config=config,
                plan=plan,
            )
            source_evidence = _validate_source_tree_end(root, plan)
            _revalidate_frozen_inputs(config_path=config_path, plan=plan, content_context=content_context)
            report = build_report(
                config,
                plan,
                checkpoints,
                source_evidence=source_evidence,
                content_commitment=content_context["commitment"],
            )
            report_path = output / "main_temporal_validity_audit.json"
            markdown_path = output / "main_temporal_validity_audit.md"
            receipt_path = output / "main_temporal_validity_audit.outputs.json"
            try:
                finalization_started = True
                atomic_write_json(report_path, report)
                atomic_write_text(markdown_path, render_markdown(report))
                state = _state(plan, "COMPLETE_PENDING_RECEIPT", len(checkpoints))
                state.update(
                    {
                        "completion_receipt_relative_path": receipt_path.name,
                        "complete_only_if_receipt_exists": True,
                        "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
                    }
                )
                atomic_write_json(output / "run_state.json", state)
                _validate_output_tree(output, final=False)
                outputs = _output_manifest(output, report)
                # Completion receipt is intentionally the final publication write.
                atomic_write_json(receipt_path, outputs)
                validate_final_publication(
                    output,
                    report=report,
                    outputs=outputs,
                    config=config,
                    plan=plan,
                    source_root=root,
                    config_path=config_path,
                    content_context=content_context,
                )
            except BaseException:
                _remove_final_publication(output)
                atomic_write_json(output / "run_state.json", _state(plan, "FINALIZATION_FAILED_NO_RECEIPT", len(checkpoints)))
                raise
            return report
        except BaseException as exc:
            if not (output / "main_temporal_validity_audit.outputs.json").exists():
                _remove_final_publication(output)
                if not finalization_started:
                    atomic_write_json(output / "run_state.json", _state(plan, "PAUSED", len(locals().get("checkpoints", [])), error=exc))
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--content-freeze-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after-new-files", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-full", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser


def redacted_error_message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: temporal audit failed; operator paths and payload details are redacted"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print("ERROR: execution requires --execute", file=sys.stderr)
        return 2
    try:
        report = run_audit(
            config_path=args.config,
            source_root=args.source_root,
            content_freeze_receipt=args.content_freeze_receipt,
            output_dir=args.output_dir,
            max_files=args.max_files,
            batch_size=args.batch_size,
            resume=args.resume,
            pause_after_new_files=args.pause_after_new_files,
            authorize_full_scope=args.authorize_full,
        )
    except KeyboardInterrupt:
        print("PAUSED: interrupted; resume with --resume", file=sys.stderr)
        return 130
    except (TemporalAuditError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {redacted_error_message(exc)}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "operator_paths_redacted": True}, sort_keys=True))
    if args.require_pass and report["status"] != FULL_STATUS:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
