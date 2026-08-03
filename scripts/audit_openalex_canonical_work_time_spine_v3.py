#!/usr/bin/env python3
"""Production-capable, temporal-v3-bound canonical work-time spine runner.

The prior v2 runner remains a frozen synthetic-contract implementation.  This
successor exact-loads that file and reuses only its pure row, classifier,
schema, path, and canonical-JSON helpers.  Corpus execution is implemented
here as a bounded streaming pipeline: source -> pending per-file staging ->
bounded-FD partition spools -> chunked external sort -> streaming Parquet.

No production run is implicit.  Full execution requires ``--execute``,
``--authorize-full``, and ``--acknowledge-exclusive-source-scan`` together.
A production-source pilot has a distinct schema, status, receipt basename,
and no partition product.  The closed-fixture API
uses the same streaming engine but can never emit a production status.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import heapq
import importlib.util
import json
import math
import os
import platform
import resource
import secrets
import shutil
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, Iterator, Mapping, NoReturn, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml"
)
PREDECESSOR_PATH = REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v2.py"
PREDECESSOR_SHA256 = (
    "6a2d1d87dd4a3e7d3c97056fabbe951a1528cfd62d8448acc74ae337f344b7b1"
)
# Frozen after the config's runner_contract is finalized.  The digest is over
# canonical JSON with only runner_contract.required_tool_sha256 replaced by
# sixty-four ASCII zeroes, which avoids a config/tool hash cycle.
FROZEN_SEMANTIC_CONFIG_SHA256 = "3e21393832cc7c2624177fd7f73f0c1ddb80ece21dfd2b4d84e173a7f32f741a"

DESIGN_CONFIG_SCHEMA = "openalex_g1_canonical_work_time_spine_design.v3"
FULL_SCOPE = "FULL_CURRENT_SNAPSHOT_PRODUCTION"
PILOT_SCOPE = "PRODUCTION_SOURCE_PREFIX_PILOT_UNPROMOTABLE"
CLOSED_FIXTURE_SCOPE = "CLOSED_FIXTURE_STREAMING_ENGINE_ONLY"
FRESH_INVOCATION = "FRESH"
NONTERMINAL_RESUME_INVOCATION = "NONTERMINAL_RESUME"
PRODUCTION_ROOT_ID = "openalex_works_20260330_repairreplay_20260410_190630"
PRODUCTION_NOT_RUN = "PRODUCTION_NOT_RUN"
FULL_STATUS = (
    "PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_TEMPORAL_"
    "V3_PARENT"
)
PILOT_STATUS = (
    "PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE"
)
CLOSED_FIXTURE_STATUS = (
    "PASS_TEST_FIXTURE_CANONICAL_WORK_TIME_SPINE_EXTERNAL_MEMORY_CONTRACT_ONLY"
)
FAILURE_STATUS = "FINALIZATION_FAILED_NO_VALID_RECEIPT"
COMPLETION_ABI = "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
TEMPORAL_V3_JSON_ABI = "CANONICAL_JSON_SORTED_KEYS_COMPACT_UTF8_PLUS_LF"
PRETTY_JSON_ABI = "JSON_SORTED_KEYS_INDENT_2_UTF8_PLUS_LF"
TEMPORAL_V3_CHECKPOINT_KEYS = {
    "checkpoint_sha256",
    "file_index",
    "metrics",
    "metrics_sha256",
    "operator_paths_redacted",
    "ordering_diagnostic_samples",
    "projected_payload_sha256",
    "publication_year_extrema",
    "quarantine_observations",
    "relative_path",
    "schema_version",
    "source_identity",
    "task_fingerprint_sha256",
    "task_sha256",
    "violation_samples",
}

FULL_SCHEMAS = {
    "plan": "openalex_g1_canonical_work_time_spine_production_plan.v1",
    "preflight_evidence": "openalex_g1_canonical_work_time_spine_production_preflight_evidence.v1",
    "scan_checkpoint": (
        "openalex_g1_canonical_work_time_spine_production_scan_checkpoint.v1"
    ),
    "partition_checkpoint": (
        "openalex_g1_canonical_work_time_spine_production_partition_checkpoint.v1"
    ),
    "staging_manifest": (
        "openalex_g1_canonical_work_time_spine_production_staging_manifest.v1"
    ),
    "spine_manifest": (
        "openalex_g1_canonical_work_time_spine_production_manifest.v1"
    ),
    "report": "openalex_g1_canonical_work_time_spine_production_report.v1",
    "outputs": "openalex_g1_canonical_work_time_spine_production_outputs.v1",
    "state": "openalex_g1_canonical_work_time_spine_production_state.v1",
}
PILOT_SCHEMAS = {
    "plan": "openalex_g1_canonical_work_time_spine_pilot_plan.v1",
    "preflight_evidence": "openalex_g1_canonical_work_time_spine_pilot_preflight_evidence.v1",
    "scan_checkpoint": (
        "openalex_g1_canonical_work_time_spine_pilot_scan_checkpoint.v1"
    ),
    "staging_manifest": (
        "openalex_g1_canonical_work_time_spine_pilot_staging_manifest.v1"
    ),
    "report": "openalex_g1_canonical_work_time_spine_pilot_report.v1",
    "outputs": "openalex_g1_canonical_work_time_spine_pilot_outputs.v1",
    "state": "openalex_g1_canonical_work_time_spine_pilot_state.v1",
}
CLOSED_FIXTURE_SCHEMAS = {
    "plan": "openalex_g1_canonical_work_time_spine_test_fixture_plan.v1",
    "preflight_evidence": "openalex_g1_canonical_work_time_spine_test_fixture_preflight_evidence.v1",
    "scan_checkpoint": (
        "openalex_g1_canonical_work_time_spine_test_fixture_scan_checkpoint.v1"
    ),
    "partition_checkpoint": (
        "openalex_g1_canonical_work_time_spine_test_fixture_partition_checkpoint.v1"
    ),
    "staging_manifest": (
        "openalex_g1_canonical_work_time_spine_test_fixture_staging_manifest.v1"
    ),
    "spine_manifest": (
        "openalex_g1_canonical_work_time_spine_test_fixture_manifest.v1"
    ),
    "report": "openalex_g1_canonical_work_time_spine_test_fixture_report.v1",
    "outputs": "openalex_g1_canonical_work_time_spine_test_fixture_outputs.v1",
    "state": "openalex_g1_canonical_work_time_spine_test_fixture_state.v1",
}

FULL_BASENAMES = {
    "report": "work_time_spine_audit.json",
    "markdown": "work_time_spine_audit.md",
    "receipt": "work_time_spine_audit.outputs.json",
}
PILOT_BASENAMES = {
    "report": "work_time_spine_pilot.json",
    "markdown": "work_time_spine_pilot.md",
    "receipt": "work_time_spine_pilot.outputs.json",
}
FIXTURE_BASENAMES = {
    "report": "work_time_spine_test_fixture.json",
    "markdown": "work_time_spine_test_fixture.md",
    "receipt": "work_time_spine_test_fixture.outputs.json",
}

PARTITION_COUNT = 4096
RECORD_BYTES = 16
MAX_OPEN_SPOOL_FILES = 4096
SPOOL_BUFFER_RECORDS_PER_FILE = 8192
SPOOL_DEMUX_CHUNK_ROWS = 1_048_576
DEFAULT_BATCH_SIZE = 262_144
DEFAULT_SORT_CHUNK_ROWS = 1_048_576
DEFAULT_PROJECTED_ARROW_BATCH_BYTES_BOUND = 256 << 20
DEFAULT_PROJECTED_IPC_BATCH_BYTES_BOUND = 256 << 20
DEFAULT_ENCODED_BATCH_BYTES_BOUND = DEFAULT_BATCH_SIZE * RECORD_BYTES
DEFAULT_PROJECTED_BATCH_BYTES_BOUND = (
    DEFAULT_PROJECTED_ARROW_BATCH_BYTES_BOUND
    + DEFAULT_PROJECTED_IPC_BATCH_BYTES_BOUND
    + DEFAULT_ENCODED_BATCH_BYTES_BOUND
)
DEFAULT_SORT_MEMORY_MULTIPLIER = 4
DEFAULT_RAM_RESERVE_BYTES = 8 << 30
DEFAULT_DISK_RESERVE_BYTES = 8 << 30
DEFAULT_CONTROL_BYTES_BOUND = 2 << 30
DEFAULT_PARQUET_BYTES_PER_ROW_BOUND = 32
DEFAULT_PARQUET_FIXED_BYTES_PER_PARTITION_BOUND = 131_072
DEFAULT_DISK_SAFETY_MULTIPLIER_MILLI = 1500
MAX_CLOSED_FIXTURE_TASKS = 8
MAX_CLOSED_FIXTURE_ROWS = 10_000
MAX_CLOSED_FIXTURE_BYTES = 64 << 20
RECORD_DTYPE = np.dtype(
    [
        ("work_id_numeric", "<u8"),
        ("publication_day", "<i4"),
        ("publication_year", "<u2"),
        ("temporal_disposition", "u1"),
        ("reserved", "u1"),
    ],
    align=False,
)
FRAMED_RECORD_DTYPE = np.dtype(
    [("encoded_length", "<u8"), ("record", f"V{RECORD_BYTES}")], align=False
)
require_record_dtype_size = RECORD_DTYPE.itemsize == RECORD_BYTES
if not require_record_dtype_size:  # pragma: no cover - import-time ABI guard
    raise RuntimeError("NumPy staging dtype is not the frozen 16-byte ABI")


def _raise_file_integrity_error(
    message: str, cause: BaseException | None = None
) -> NoReturn:
    error_type = globals().get("SpineAuditError", RuntimeError)
    if cause is None:
        raise error_type(message)
    raise error_type(message) from cause


def _integrity_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_uid),
        int(info.st_gid),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _assert_integrity_path_has_no_symlink(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            _raise_file_integrity_error(f"cannot inspect {field}", exc)
        if stat.S_ISLNK(info.st_mode):
            _raise_file_integrity_error(f"{field} contains a symlink")
    return absolute


def _open_integrity_descriptor(path: Path, *, field: str) -> tuple[int, tuple[int, ...]]:
    absolute = _assert_integrity_path_has_no_symlink(path, field=field)
    try:
        descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        _raise_file_integrity_error(f"cannot open {field}", exc)
    try:
        start = os.fstat(descriptor)
        if not stat.S_ISREG(start.st_mode):
            _raise_file_integrity_error(f"{field} is not a regular file")
        start_identity = _integrity_identity(start)
        if _integrity_identity(absolute.lstat()) != start_identity:
            _raise_file_integrity_error(f"{field} path identity changed before read")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, start_identity


def _validate_integrity_descriptor(
    descriptor: int,
    path: Path,
    start_identity: tuple[int, ...],
    *,
    field: str,
) -> None:
    try:
        end_identity = _integrity_identity(os.fstat(descriptor))
        path_identity = _integrity_identity(path.lstat())
    except OSError as exc:
        _raise_file_integrity_error(f"cannot restat {field}", exc)
    if end_identity != start_identity or path_identity != start_identity:
        _raise_file_integrity_error(f"{field} identity changed during read")


def _sha256_regular_file_snapshot(
    path: Path, *, field: str, chunk_bytes: int = 16 << 20
) -> tuple[str, tuple[int, ...]]:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    descriptor, start_identity = _open_integrity_descriptor(absolute, field=field)
    digest = hashlib.sha256()
    try:
        while block := os.read(descriptor, chunk_bytes):
            digest.update(block)
        _validate_integrity_descriptor(
            descriptor,
            absolute,
            start_identity,
            field=field,
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest(), start_identity


def _sha256_file(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    return _sha256_regular_file_snapshot(
        path, field="hash input", chunk_bytes=chunk_bytes
    )[0]


def _read_regular_file_snapshot(
    path: Path, *, field: str
) -> tuple[bytes, str, tuple[int, ...]]:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    descriptor, start_identity = _open_integrity_descriptor(absolute, field=field)
    blocks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        while block := os.read(descriptor, 1 << 20):
            blocks.append(block)
            digest.update(block)
        _validate_integrity_descriptor(
            descriptor,
            absolute,
            start_identity,
            field=field,
        )
    finally:
        os.close(descriptor)
    return b"".join(blocks), digest.hexdigest(), start_identity


def _read_regular_file_bytes(path: Path, *, field: str) -> bytes:
    return _read_regular_file_snapshot(path, field=field)[0]


def _load_predecessor() -> ModuleType:
    if _sha256_file(PREDECESSOR_PATH) != PREDECESSOR_SHA256:
        raise RuntimeError("canonical v2 predecessor SHA-256 mismatch")
    name = f"_canonical_spine_v2_{PREDECESSOR_SHA256[:12]}"
    spec = importlib.util.spec_from_file_location(name, PREDECESSOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical v2 predecessor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V2 = _load_predecessor()
SpineAuditError = _V2.SpineAuditError


class TerminalCompatibilityError(SpineAuditError):
    """A valid terminal surface is incompatible with the current invocation."""


Record = _V2.Record
ARROW_SCHEMA = _V2.ARROW_SCHEMA
SPINE_METRIC_KEYS = tuple(_V2.SPINE_METRIC_KEYS)
TEMPORAL_V3_METRIC_KEYS = tuple(_V2.TEMPORAL_V3_METRIC_KEYS)
DISPOSITION_NAMES = dict(_V2.DISPOSITION_NAMES)
WORK_PATTERN = _V2.WORK_PATTERN
W_MINUS_ONE = _V2.W_MINUS_ONE
RECORD_HASH_DOMAIN = _V2.RECORD_HASH_DOMAIN


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpineAuditError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return _V2.canonical_json_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path, field: str) -> dict[str, Any]:
    return _V2.read_json(path, field)


def _decode_json_payload(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        return _require_mapping(json.loads(payload.decode("utf-8")), field)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpineAuditError(f"cannot parse {field}") from exc


def _read_parent_json_snapshot(
    path: Path, *, field: str
) -> tuple[dict[str, Any], bytes, str, tuple[int, ...]]:
    payload, digest, identity = _read_regular_file_snapshot(path, field=field)
    value = _decode_json_payload(payload, field=field)
    require(
        payload == _V2.published_json_bytes(value),
        f"{field} bytes are not canonical",
    )
    return value, payload, digest, identity


def _read_temporal_v3_json_snapshot(
    path: Path, *, field: str
) -> tuple[dict[str, Any], bytes, str, tuple[int, ...]]:
    """Read the exact compact-newline JSON ABI published by temporal v3."""

    payload, digest, identity = _read_regular_file_snapshot(path, field=field)
    value = _decode_json_payload(payload, field=field)
    require(
        payload == _canonical_json_bytes(value) + b"\n",
        f"{field} bytes do not match the temporal v3 published JSON ABI",
    )
    return value, payload, digest, identity


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    return _V2.require_mapping(value, field)


def _require_list(value: Any, field: str) -> list[Any]:
    return _V2.require_list(value, field)


def _require_int(value: Any, field: str) -> int:
    return _V2.require_int(value, field)


def _require_sha256(value: Any, field: str) -> str:
    return _V2.require_sha256(value, field)


def _validate_temporal_checkpoint_source_identity(
    checkpoint: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    task_source_identity = _require_mapping(
        task.get("source_rich_identity"), "temporal task source identity"
    )
    checkpoint_source_identity = _require_mapping(
        checkpoint.get("source_identity"),
        "temporal checkpoint source identity",
    )
    require(
        set(checkpoint_source_identity)
        == set(task_source_identity)
        | {"footer_rows", "projected_schema_sha256"},
        "temporal checkpoint source identity key set mismatch",
    )
    require(
        {
            key: checkpoint_source_identity[key]
            for key in task_source_identity
        }
        == task_source_identity,
        "temporal checkpoint rich source identity mismatch",
    )
    require(
        checkpoint_source_identity.get("footer_rows") == task.get("footer_rows"),
        "temporal checkpoint source footer mismatch",
    )
    require(
        checkpoint_source_identity.get("projected_schema_sha256")
        == task.get("projected_schema_sha256"),
        "temporal checkpoint source schema mismatch",
    )


def _validate_temporal_checkpoint_shape(
    checkpoint: Mapping[str, Any], task: Mapping[str, Any]
) -> None:
    require(
        set(checkpoint) == TEMPORAL_V3_CHECKPOINT_KEYS,
        "temporal checkpoint key set mismatch",
    )
    require(
        checkpoint.get("file_index") == task.get("file_index"),
        "temporal checkpoint file index mismatch",
    )
    require(
        checkpoint.get("operator_paths_redacted") is True,
        "temporal checkpoint path redaction mismatch",
    )


def _validate_temporal_module_config_contract(
    module: ModuleType,
    *,
    observed_config_sha256: str,
    required_config_sha256: str,
) -> None:
    production_sha = _require_sha256(
        getattr(module, "PRODUCTION_CONFIG_SHA256", None),
        "temporal producer production config SHA",
    )
    frozen_sha = _require_sha256(
        getattr(module, "FROZEN_CONFIG_SHA256", None),
        "temporal producer frozen config SHA",
    )
    require(
        production_sha
        == frozen_sha
        == observed_config_sha256
        == required_config_sha256,
        "temporal producer config provenance mismatch",
    )


def _validate_temporal_module_repo_root(
    module: ModuleType, temporal_repo_root: Path
) -> None:
    try:
        module_repo_root = Path(os.fspath(module.REPO_ROOT)).resolve(strict=True)
    except (AttributeError, OSError, TypeError) as exc:
        raise SpineAuditError("cannot validate temporal producer repository root") from exc
    require(
        module_repo_root
        == REPO_ROOT.resolve(strict=True)
        == _absolute(temporal_repo_root),
        "temporal producer repository root mismatch",
    )


def _absolute(path: Path) -> Path:
    return _V2._absolute(path)


def _safe_relative(value: Any, field: str) -> Path:
    return _V2.safe_relative_path(value, field)


def _rich_identity(path: Path, *, field: str) -> dict[str, int]:
    return _V2._regular_identity(path, field=field)


def _identity_mapping(identity: tuple[int, ...]) -> dict[str, int]:
    fields = (
        "device",
        "inode",
        "mode",
        "uid",
        "gid",
        "links",
        "bytes",
        "mtime_ns",
        "ctime_ns",
    )
    require(len(identity) == len(fields), "regular-file identity field count drift")
    return dict(zip(fields, identity, strict=True))


def _frozen_file_snapshot(path: Path, *, field: str) -> dict[str, Any]:
    digest, identity = _sha256_regular_file_snapshot(path, field=field)
    return {
        "sha256": digest,
        "rich_identity": _identity_mapping(identity),
    }


def _atomic_json(path: Path, value: Any) -> None:
    _V2.atomic_write_json(path, value)


def _atomic_bytes(path: Path, value: bytes) -> None:
    _V2.atomic_write_bytes(path, value)


def _mkdir_private(path: Path) -> None:
    _V2._mkdir_private(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tree_digest(items: Sequence[Mapping[str, Any]]) -> str:
    return _V2._tree_digest(items)


def _schemas(scope: str) -> Mapping[str, str]:
    if scope == FULL_SCOPE:
        return FULL_SCHEMAS
    if scope == PILOT_SCOPE:
        return PILOT_SCHEMAS
    require(scope == CLOSED_FIXTURE_SCOPE, "unknown execution scope")
    return CLOSED_FIXTURE_SCHEMAS


def _basenames(scope: str) -> Mapping[str, str]:
    if scope == PILOT_SCOPE:
        return PILOT_BASENAMES
    if scope == CLOSED_FIXTURE_SCOPE:
        return FIXTURE_BASENAMES
    require(scope == FULL_SCOPE, "unknown execution scope")
    return FULL_BASENAMES


def _status(scope: str) -> str:
    if scope == FULL_SCOPE:
        return FULL_STATUS
    if scope == PILOT_SCOPE:
        return PILOT_STATUS
    require(scope == CLOSED_FIXTURE_SCOPE, "unknown execution scope")
    return CLOSED_FIXTURE_STATUS


def _semantic_config_sha256(config: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(config))
    runner = _require_mapping(normalized.get("runner_contract"), "runner contract")
    require(
        set(runner) >= {"required_tool_path", "required_tool_sha256"},
        "runner contract key set is incomplete",
    )
    _require_sha256(runner.get("required_tool_sha256"), "runner tool SHA")
    runner["required_tool_sha256"] = "0" * 64
    normalized["runner_contract"] = runner
    return _sha256_bytes(_canonical_json_bytes(normalized))


def _validate_production_runner_contract(
    config_path: Path, overlay: Mapping[str, Any]
) -> None:
    require(
        _absolute(config_path) == DEFAULT_CONFIG.resolve(strict=True),
        "production and pilot require the repository default config",
    )
    runner = _require_mapping(overlay.get("runner_contract"), "runner contract")
    require(
        runner.get("required_tool_path")
        == Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        "runner contract tool path mismatch",
    )
    expected_tool_sha = _require_sha256(
        runner.get("required_tool_sha256"), "runner contract tool SHA"
    )
    require(
        _sha256_file(Path(__file__).resolve()) == expected_tool_sha,
        "production runner SHA-256 mismatch",
    )
    require(
        _semantic_config_sha256(overlay) == FROZEN_SEMANTIC_CONFIG_SHA256,
        "production semantic config fingerprint mismatch",
    )


def _load_config(path: Path) -> tuple[dict[str, Any], str, dict[str, int]]:
    path = _absolute(path)
    require(
        path == DEFAULT_CONFIG.resolve(strict=True),
        "production and pilot require the repository default config",
    )
    overlay, digest, identity = _V2._load_yaml_snapshot(path)
    _validate_production_runner_contract(path, overlay)
    require(overlay.get("schema_version") == DESIGN_CONFIG_SCHEMA, "config schema drift")
    lifecycle = _require_mapping(overlay.get("lifecycle"), "lifecycle")
    require(
        lifecycle.get("production_status") == PRODUCTION_NOT_RUN,
        "production status must remain NOT_RUN before execution",
    )
    require(lifecycle.get("production_executable") is True, "production is disabled")
    require(
        lifecycle.get("production_result_exists") is False,
        "repository config cannot claim a production result",
    )
    foundation_contract = _require_mapping(
        overlay.get("foundation_contract"), "foundation contract"
    )
    require(
        foundation_contract.get("load_and_validate_foundation_before_overlay")
        is True,
        "foundation load policy drift",
    )
    foundation_relative = _safe_relative(
        foundation_contract.get("config_path"), "foundation config path"
    )
    foundation_path = REPO_ROOT / foundation_relative
    foundation, foundation_sha, foundation_identity = _V2._load_yaml_snapshot(
        foundation_path
    )
    require(
        foundation_sha == foundation_contract.get("config_sha256"),
        "foundation config SHA-256 mismatch",
    )
    require(
        foundation_contract.get("synthetic_runner_path")
        == PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(),
        "foundation predecessor path drift",
    )
    require(
        foundation_contract.get("synthetic_runner_sha256")
        == PREDECESSOR_SHA256,
        "foundation predecessor SHA drift",
    )
    _V2._validate_temporal_v3_parent_spec(foundation)

    inherited = tuple(foundation_contract.get("inherited_without_semantic_change", ()))
    required_inherited = {
        "component_decision",
        "required_parent_execution_order",
        "parent_cross_binding",
        "table",
        "spine_schema",
        "disposition_classifier",
        "identifier_accounting",
        "publication_accounting",
        "row_level_parent_artifact_limitation",
        "staging_record_abi",
        "partition_contract",
        "cross_stage_conservation",
        "consumer_contract",
        "release_privacy",
    }
    require(required_inherited <= set(inherited), "foundation inherited surface drift")
    effective = copy.deepcopy(foundation)
    for key in (
        "lifecycle",
        "source",
        "production_parent_anchors",
        "parent_validation_io_boundary",
        "production_output_abi",
        "execution_modes",
        "streaming_scan_contract",
        "bounded_partition_and_sort_contract",
        "runtime_binding",
        "resource_envelope",
        "resource_bound_semantics",
        "preflight_evidence_contract",
        "pre_full_operational_gates",
        "resume_and_corruption_contract",
        "output_security_contract",
        "completion_contract",
        "consumer_rebinding_boundary",
        "future_success_contract",
        "claim_boundary",
    ):
        effective[key] = copy.deepcopy(overlay[key])
    effective["schema_version"] = DESIGN_CONFIG_SCHEMA
    effective["label"] = overlay["label"]
    effective["foundation_contract"] = copy.deepcopy(foundation_contract)
    effective["runner_contract"] = copy.deepcopy(
        _require_mapping(overlay.get("runner_contract"), "runner contract")
    )
    effective["_contract_provenance"] = {
        "overlay_sha256": digest,
        "foundation_sha256": foundation_sha,
        "foundation_rich_identity": foundation_identity,
        "foundation_config_relative_path": foundation_relative.as_posix(),
    }

    anchors = _require_mapping(
        overlay.get("production_parent_anchors"), "production parent anchors"
    )
    temporal_anchor = _require_mapping(
        anchors.get("main_temporal_validity"), "temporal production anchors"
    )
    temporal_foundation = _require_mapping(
        foundation.get("parents", {}).get("main_temporal_validity"),
        "foundation temporal parent",
    )
    translated = {
        "required_receipt_basename": "required_receipt_basename",
        "required_receipt_sha256": "required_receipt_sha256",
        "required_schema_version": "schema_version",
        "required_plan_schema_version": "required_plan_schema_version",
        "required_checkpoint_schema_version": "required_checkpoint_schema_version",
        "required_report_schema_version": "required_report_schema_version",
        "required_state_schema_version": "required_state_schema_version",
        "required_status": "required_status",
        "required_state_status": "required_state_status",
        "required_completion_contract": "required_completion_contract",
        "required_config_path": "required_config_path",
        "required_config_sha256": "required_config_sha256",
        "required_tool_path": "required_tool_path",
        "required_tool_sha256": "required_tool_sha256",
        "required_report_sha256": "required_report_sha256",
        "required_task_fingerprint_sha256": (
            "required_exact_task_fingerprint_sha256"
        ),
        "required_result_fingerprint_sha256": (
            "required_exact_result_fingerprint_sha256"
        ),
        "required_checkpoint_tree_sha256": (
            "required_exact_checkpoint_tree_sha256"
        ),
        "required_source_tree_sha256": "required_exact_source_tree_sha256",
        "required_full_task_count": "required_full_task_count",
        "required_raw_rows": "required_raw_rows",
        "required_quarantined_rows": "required_quarantined_rows",
        "required_retained_rows": "required_retained_rows",
        "required_source_bytes": "required_source_bytes",
        "required_exact_total_receipt_artifact_count": (
            "exact_total_receipt_artifact_count"
        ),
    }
    for overlay_key, foundation_key in translated.items():
        require(
            temporal_anchor.get(overlay_key)
            == temporal_foundation.get(foundation_key),
            f"overlay/foundation temporal anchor mismatch: {overlay_key}",
        )
    schemas = _require_mapping(
        overlay.get("production_output_abi", {}).get("schemas"),
        "production output schemas",
    )
    require(schemas == FULL_SCHEMAS, "production schema family drift")
    modes = _require_mapping(overlay.get("execution_modes"), "execution modes")
    require(
        modes.get("production_pilot", {}).get("status") == PILOT_STATUS,
        "pilot status drift",
    )
    require(
        modes.get("closed_test_fixture", {}).get("status")
        == CLOSED_FIXTURE_STATUS,
        "fixture status drift",
    )
    runtime_binding = _require_mapping(
        overlay.get("runtime_binding"), "runtime binding"
    )
    require(
        runtime_binding.get("exact_fields") == list(_runtime_fingerprint()),
        "runtime fingerprint field contract drift",
    )
    require(
        runtime_binding.get("bind_to_plan_task_fingerprint_and_receipt") is True
        and runtime_binding.get("resume_requires_exact_equality") is True,
        "runtime binding policy drift",
    )
    completion = _require_mapping(
        overlay.get("completion_contract"), "completion contract"
    )
    for policy_key in (
        "production_and_pilot_pre_receipt_own_frozen_input_revalidation_required",
        "pre_receipt_default_config_hash_and_nine_field_identity_revalidated",
        "pre_receipt_current_tool_hash_and_nine_field_identity_revalidated",
        "pre_receipt_foundation_config_sha256_revalidated",
        "pre_receipt_pinned_predecessor_tool_sha256_revalidated",
        "closed_fixture_exempt_from_repository_frozen_input_revalidation",
        "pre_receipt_validation_failure_leaves_receipt_absent",
    ):
        require(
            completion.get(policy_key) is True,
            f"own frozen-input completion policy drift: {policy_key}",
        )
    for boundary_key in (
        "pre_receipt_file_snapshot_hash_is_loaded_code_attestation",
        "running_process_code_object_byte_identity_attested",
    ):
        require(
            completion.get(boundary_key) is False,
            f"own frozen-input claim boundary drift: {boundary_key}",
        )
    _validate_resource_config(effective)
    return effective, digest, identity


@dataclass(frozen=True)
class Task:
    file_index: int
    relative_path: str
    footer_rows: int
    source_rich_identity: dict[str, int]
    temporal_task_sha256: str
    temporal_checkpoint_sha256: str
    projected_schema: dict[str, Any]
    projected_schema_sha256: str
    temporal_metrics: dict[str, int] | None
    publication_year_extrema: dict[str, int | None] | None


@dataclass(frozen=True)
class ParentContext:
    tasks: tuple[Task, ...]
    parents: dict[str, Any]
    temporal_receipt_sha256: str
    temporal_output_root: Path
    parent_snapshot_paths: tuple[tuple[Path, str], ...]


@dataclass(frozen=True)
class RunArtifact:
    path: Path
    rows: int
    bytes: int
    sha256: str


def _artifact_contract(root: Path, relative_text: str) -> dict[str, Any]:
    relative = _safe_relative(relative_text, "parent artifact path")
    path = root / relative
    digest, identity = _sha256_regular_file_snapshot(
        path, field="parent artifact"
    )
    return {
        "relative_path": relative.as_posix(),
        "bytes": identity[6],
        "sha256": digest,
    }


def _load_temporal_module(config: Mapping[str, Any], temporal_repo_root: Path) -> ModuleType:
    parent = _require_mapping(
        config.get("parents", {}).get("main_temporal_validity"), "temporal parent"
    )
    relative = _safe_relative(parent.get("required_tool_path"), "temporal tool path")
    tool_path = REPO_ROOT / relative
    expected = _require_sha256(parent.get("required_tool_sha256"), "temporal tool SHA")
    require(_sha256_file(tool_path) == expected, "temporal tool SHA-256 mismatch")
    name = f"_canonical_v3_temporal_{expected[:12]}"
    spec = importlib.util.spec_from_file_location(name, tool_path)
    require(spec is not None and spec.loader is not None, "cannot load temporal tool")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _validate_temporal_module_repo_root(module, temporal_repo_root)
    return module


def _validate_parent_control_plane(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    content_receipt_path: Path,
    temporal_receipt_path: Path,
    temporal_config_path: Path,
    temporal_repo_root: Path,
) -> ParentContext:
    """Validate frozen parent control artifacts without a temporal data-page scan."""

    require(
        _absolute(temporal_repo_root) == REPO_ROOT.resolve(strict=True),
        "temporal repository root override is forbidden",
    )

    parent = _require_mapping(
        config.get("parents", {}).get("main_temporal_validity"), "temporal parent"
    )
    overlay_anchors = _require_mapping(
        config.get("production_parent_anchors"), "production parent anchors"
    )
    for policy_key in (
        "imported_temporal_module_globals_mutation_forbidden",
        "imported_temporal_module_repo_root_must_equal_current_repository_root",
        "temporal_module_production_and_frozen_config_constants_must_equal_anchored_config_sha256",
    ):
        require(
            overlay_anchors.get(policy_key) is True,
            f"temporal module provenance policy drift: {policy_key}",
        )
    temporal_anchor = _require_mapping(
        overlay_anchors.get("main_temporal_validity"), "temporal production anchors"
    )
    content_anchor = _require_mapping(
        overlay_anchors.get("current_content_baseline"), "content production anchors"
    )
    require(
        temporal_anchor.get("required_json_serialization_abi")
        == TEMPORAL_V3_JSON_ABI,
        "temporal parent JSON serialization ABI drift",
    )
    require(
        content_anchor.get("required_json_serialization_abi") == PRETTY_JSON_ABI,
        "content parent JSON serialization ABI drift",
    )
    control_snapshots: dict[
        Path, tuple[bytes, str, tuple[int, ...]]
    ] = {}

    def read_parent_json(
        path: Path, field: str, *, temporal_v3: bool
    ) -> dict[str, Any]:
        absolute = _absolute(path)
        reader = (
            _read_temporal_v3_json_snapshot
            if temporal_v3
            else _read_parent_json_snapshot
        )
        value, payload, digest, identity = reader(absolute, field=field)
        control_snapshots[absolute] = (payload, digest, identity)
        return value

    receipt_path = _absolute(temporal_receipt_path)
    require(
        receipt_path.name == parent.get("required_receipt_basename"),
        "temporal receipt basename mismatch",
    )
    receipt = read_parent_json(
        receipt_path, "temporal receipt", temporal_v3=True
    )
    temporal_receipt_sha = control_snapshots[receipt_path][1]
    require(
        temporal_receipt_sha == parent.get("required_receipt_sha256"),
        "temporal receipt SHA-256 mismatch",
    )
    require(
        temporal_receipt_sha == temporal_anchor.get("required_receipt_sha256"),
        "temporal overlay receipt SHA-256 mismatch",
    )
    output_root = receipt_path.parent
    require(
        output_root.is_dir()
        and not output_root.is_symlink()
        and stat.S_IMODE(output_root.stat().st_mode) == 0o700,
        "temporal output root is not private",
    )

    temporal_config_path = _absolute(temporal_config_path)
    _V2._assert_no_symlink_components(
        temporal_config_path, field="temporal config path"
    )
    temporal_config_payload, temporal_config_sha, temporal_config_identity = (
        _read_regular_file_snapshot(
        temporal_config_path, field="temporal config"
        )
    )
    control_snapshots[temporal_config_path] = (
        temporal_config_payload,
        temporal_config_sha,
        temporal_config_identity,
    )
    require(
        temporal_config_sha == parent.get("required_config_sha256"),
        "temporal config SHA-256 mismatch",
    )
    try:
        temporal_config = _require_mapping(
            yaml.load(temporal_config_payload, Loader=_V2.UniqueKeyLoader),
            "temporal config",
        )
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise SpineAuditError("cannot parse temporal config") from exc
    temporal_module = _load_temporal_module(config, temporal_repo_root)
    _validate_temporal_module_config_contract(
        temporal_module,
        observed_config_sha256=temporal_config_sha,
        required_config_sha256=_require_sha256(
            parent.get("required_config_sha256"), "temporal required config SHA"
        ),
    )

    require(
        set(receipt) == set(parent.get("exact_receipt_keyset", [])),
        "temporal receipt key set mismatch",
    )
    require(receipt.get("schema_version") == parent.get("schema_version"), "temporal receipt schema mismatch")
    require(receipt.get("status") == parent.get("required_status"), "temporal receipt status mismatch")
    require(receipt.get("completion_contract") == COMPLETION_ABI, "temporal completion ABI mismatch")
    require(receipt.get("operator_paths_redacted") is True, "temporal path redaction mismatch")

    plan_path = output_root / "audit_plan.json"
    report_path = output_root / "main_temporal_validity_audit.json"
    state_path = output_root / "run_state.json"
    plan = read_parent_json(plan_path, "temporal plan", temporal_v3=True)
    report = read_parent_json(report_path, "temporal report", temporal_v3=True)
    state = read_parent_json(state_path, "temporal state", temporal_v3=True)
    require(plan.get("schema_version") == parent.get("required_plan_schema_version"), "temporal plan schema mismatch")
    require(report.get("schema_version") == parent.get("required_report_schema_version"), "temporal report schema mismatch")
    require(state.get("schema_version") == parent.get("required_state_schema_version"), "temporal state schema mismatch")
    require(plan.get("full_scope") is True and plan.get("pilot_mode") is None, "temporal parent is not full scope")
    require(state.get("status") == parent.get("required_state_status"), "temporal state mismatch")
    require(report.get("status") == parent.get("required_status"), "temporal report status mismatch")
    require(control_snapshots[_absolute(report_path)][1] == parent.get("required_report_sha256"), "temporal report SHA mismatch")

    raw_tasks = [_require_mapping(item, "temporal task") for item in _require_list(plan.get("tasks"), "temporal tasks")]
    require(len(raw_tasks) == parent.get("required_full_task_count"), "temporal task count mismatch")
    require(
        [item.get("file_index") for item in raw_tasks] == list(range(len(raw_tasks))),
        "temporal parent task indices are not contiguous",
    )
    relative_paths = [str(item.get("relative_path")) for item in raw_tasks]
    require(len(set(relative_paths)) == len(relative_paths), "temporal task paths repeat")
    require(relative_paths == sorted(relative_paths), "temporal task paths are not lexical")
    artifacts = _require_mapping(receipt.get("artifacts"), "temporal receipt artifacts")
    expected_artifacts = set(str(item) for item in parent.get("exact_fixed_receipt_artifact_paths", []))
    expected_artifacts.update(
        f"checkpoints/main/{_sha256_bytes(str(task['relative_path']).encode('utf-8'))}.json"
        for task in raw_tasks
    )
    require(len(expected_artifacts) == parent.get("exact_total_receipt_artifact_count"), "temporal artifact count mismatch")
    require(set(artifacts) == expected_artifacts, "temporal artifact set mismatch")
    for relative in sorted(expected_artifacts):
        contract = _require_mapping(artifacts[relative], "temporal artifact")
        require(set(contract) == {"relative_path", "bytes", "sha256"}, "temporal artifact contract drift")
        artifact_path = _absolute(output_root / _safe_relative(relative, "temporal artifact path"))
        cached = control_snapshots.get(artifact_path)
        observed_contract = (
            {
                "relative_path": relative,
                "bytes": len(cached[0]),
                "sha256": cached[1],
            }
            if cached is not None
            else _artifact_contract(output_root, relative)
        )
        require(contract == observed_contract, "temporal artifact bytes/hash mismatch")

    plan_core = dict(plan)
    task_fp = plan_core.pop("task_fingerprint_sha256", None)
    require(task_fp == _sha256_bytes(_canonical_json_bytes(plan_core)), "temporal task fingerprint mismatch")
    require(task_fp == parent.get("required_exact_task_fingerprint_sha256"), "temporal exact task fingerprint mismatch")
    result_fp = temporal_module._report_fingerprint(report)
    require(result_fp == parent.get("required_exact_result_fingerprint_sha256"), "temporal exact result fingerprint mismatch")
    require(receipt.get("task_fingerprint_sha256") == task_fp, "temporal receipt task mismatch")
    require(receipt.get("result_fingerprint_sha256") == result_fp, "temporal receipt result mismatch")

    checkpoints: list[dict[str, Any]] = []
    tasks: list[Task] = []
    for raw in raw_tasks:
        raw_core = dict(raw)
        raw_task_sha = raw_core.pop("task_sha256", None)
        require(
            raw_task_sha == _sha256_bytes(_canonical_json_bytes(raw_core)),
            "temporal task self hash mismatch",
        )
        require(
            raw.get("projected_columns")
            == [
                "id",
                "publication_date",
                "publication_year",
                "created_date",
                "updated_date",
            ],
            "temporal task projected columns drift",
        )
        require(
            raw.get("projected_schema_sha256")
            == _sha256_bytes(_canonical_json_bytes(raw.get("projected_schema"))),
            "temporal task projected schema hash mismatch",
        )
        relative = str(raw["relative_path"])
        checkpoint_path = output_root / "checkpoints/main" / f"{_sha256_bytes(relative.encode('utf-8'))}.json"
        checkpoint = read_parent_json(
            checkpoint_path, "temporal checkpoint", temporal_v3=True
        )
        checkpoint_snapshot = control_snapshots[_absolute(checkpoint_path)]
        checkpoint_relative = checkpoint_path.relative_to(output_root).as_posix()
        require(
            artifacts[checkpoint_relative]
            == {
                "relative_path": checkpoint_relative,
                "bytes": len(checkpoint_snapshot[0]),
                "sha256": checkpoint_snapshot[1],
            },
            "temporal checkpoint snapshot differs from receipt",
        )
        core = dict(checkpoint)
        checkpoint_sha = core.pop("checkpoint_sha256", None)
        require(checkpoint_sha == _sha256_bytes(_canonical_json_bytes(core)), "temporal checkpoint self hash mismatch")
        _validate_temporal_checkpoint_shape(checkpoint, raw)
        require(checkpoint.get("schema_version") == parent.get("required_checkpoint_schema_version"), "temporal checkpoint schema mismatch")
        require(
            checkpoint.get("task_fingerprint_sha256") == task_fp,
            "temporal checkpoint task fingerprint mismatch",
        )
        require(
            checkpoint.get("task_sha256") == raw_task_sha,
            "temporal checkpoint task SHA mismatch",
        )
        require(
            checkpoint.get("relative_path") == relative,
            "temporal checkpoint path mismatch",
        )
        _validate_temporal_checkpoint_source_identity(checkpoint, raw)
        require(set(_require_mapping(checkpoint.get("metrics"), "temporal metrics")) == set(TEMPORAL_V3_METRIC_KEYS), "temporal metric keyset mismatch")
        require(checkpoint.get("metrics_sha256") == _sha256_bytes(_canonical_json_bytes(checkpoint.get("metrics"))), "temporal metrics hash mismatch")
        checkpoints.append(checkpoint)
        tasks.append(
            Task(
                file_index=_require_int(raw.get("file_index"), "temporal file index"),
                relative_path=relative,
                footer_rows=_require_int(raw.get("footer_rows"), "temporal footer rows"),
                source_rich_identity=dict(_require_mapping(raw.get("source_rich_identity"), "temporal source identity")),
                temporal_task_sha256=_require_sha256(raw_task_sha, "temporal task SHA"),
                temporal_checkpoint_sha256=_require_sha256(checkpoint_sha, "temporal checkpoint SHA"),
                projected_schema={
                    name: dict(
                        _require_mapping(
                            _require_mapping(raw.get("projected_schema"), "temporal projected schema").get(name),
                            f"temporal projected schema {name}",
                        )
                    )
                    for name in ("id", "publication_date", "publication_year")
                },
                projected_schema_sha256="",
                temporal_metrics=dict(_require_mapping(checkpoint.get("metrics"), "temporal metrics")),
                publication_year_extrema=dict(_require_mapping(checkpoint.get("publication_year_extrema"), "temporal extrema")),
            )
        )
        projected = tasks[-1].projected_schema
        tasks[-1] = Task(
            **{
                **tasks[-1].__dict__,
                "projected_schema_sha256": _sha256_bytes(
                    _canonical_json_bytes(projected)
                ),
            }
        )
    checkpoint_tree = temporal_module._checkpoint_tree(checkpoints)
    require(checkpoint_tree == report.get("checkpoint_evidence"), "temporal checkpoint tree mismatch")
    require(checkpoint_tree.get("checkpoint_tree_sha256") == parent.get("required_exact_checkpoint_tree_sha256"), "temporal checkpoint tree anchor mismatch")
    source_tree = temporal_module._source_tree(raw_tasks)
    require(source_tree == plan.get("source_tree_start"), "temporal source tree mismatch")
    require(source_tree.get("tree_sha256") == parent.get("required_exact_source_tree_sha256"), "temporal source tree anchor mismatch")

    adapted = _V2._temporal_parent_config_for_content(temporal_config, config)
    content_receipt_path = _absolute(content_receipt_path)
    require(
        content_receipt_path.name == content_anchor.get("required_receipt_basename"),
        "content receipt basename mismatch",
    )
    content_receipt = read_parent_json(
        content_receipt_path, "content receipt", temporal_v3=False
    )
    content_receipt_sha = control_snapshots[content_receipt_path][1]
    require(
        content_receipt_sha == content_anchor.get("required_receipt_sha256"),
        "content receipt SHA-256 mismatch",
    )
    expected_content_values = {
        "schema_version": content_anchor.get("required_schema_version"),
        "status": content_anchor.get("required_status"),
        "plan_sha256": content_anchor.get("required_plan_sha256"),
        "task_fingerprint_sha256": content_anchor.get(
            "required_task_fingerprint_sha256"
        ),
        "result_fingerprint_sha256": content_anchor.get(
            "required_result_fingerprint_sha256"
        ),
        "content_tree_sha256": content_anchor.get("required_content_tree_sha256"),
        "completion_contract": content_anchor.get("required_completion_contract"),
    }
    for key, expected in expected_content_values.items():
        require(
            content_receipt.get(key) == expected,
            f"content receipt anchor mismatch: {key}",
        )
    require(
        content_receipt.get("checkpoint_tree", {}).get("tree_sha256")
        == content_anchor.get("required_checkpoint_tree_sha256"),
        "content checkpoint-tree anchor mismatch",
    )
    content_config_path = REPO_ROOT / _safe_relative(
        content_anchor.get("required_config_path"), "content config path"
    )
    content_tool_path = REPO_ROOT / _safe_relative(
        content_anchor.get("required_tool_path"), "content tool path"
    )
    content_config_sha = _sha256_file(content_config_path)
    require(
        content_config_sha == content_anchor.get("required_config_sha256"),
        "content config SHA-256 mismatch",
    )
    content_tool_sha = _sha256_file(content_tool_path)
    require(
        content_tool_sha == content_anchor.get("required_tool_sha256"),
        "content tool SHA-256 mismatch",
    )
    content_context = temporal_module._validate_content_freeze_parent(
        adapted, content_receipt_path, required=True
    )
    require(
        _sha256_file(content_receipt_path) == content_receipt_sha,
        "content receipt changed during parent validation",
    )
    temporal_module._bind_current_source_to_content_baseline(plan, content_context)
    require(content_context["commitment"] == report.get("content_baseline_parent"), "content/temporal commitment mismatch")

    source_root = _absolute(source_root)
    for task in tasks:
        path = source_root / _safe_relative(task.relative_path, "source task")
        identity = _rich_identity(path, field="source pre-scan identity")
        require(
            identity == task.source_rich_identity,
            "live source identity differs from temporal parent",
        )

    parents = {
        "content": {
            "receipt_sha256": content_context["commitment"]["receipt_sha256"],
            "task_fingerprint_sha256": content_context["commitment"]["task_fingerprint_sha256"],
            "content_tree_sha256": content_context["commitment"]["content_tree_sha256"],
        },
        "temporal": {
            "receipt_sha256": temporal_receipt_sha,
            "task_fingerprint_sha256": task_fp,
            "result_fingerprint_sha256": result_fp,
            "checkpoint_tree_sha256": checkpoint_tree["checkpoint_tree_sha256"],
            "source_tree_sha256": source_tree["tree_sha256"],
            "config_sha256": temporal_config_sha,
            "tool_sha256": parent["required_tool_sha256"],
        },
    }
    snapshot_items: list[tuple[Path, str]] = [
        (receipt_path, temporal_receipt_sha),
        (temporal_config_path, temporal_config_sha),
        (REPO_ROOT / _safe_relative(parent["required_tool_path"], "temporal tool"), parent["required_tool_sha256"]),
        (content_receipt_path, content_receipt_sha),
        (content_config_path, content_config_sha),
        (content_tool_path, content_tool_sha),
    ]
    snapshot_items.extend(
        (output_root / relative, str(artifacts[relative]["sha256"]))
        for relative in sorted(expected_artifacts)
    )
    content_output_root = content_receipt_path.parent
    content_artifacts = _require_list(
        content_receipt.get("artifacts"), "content receipt artifacts"
    )
    require(
        {item.get("relative_path") for item in content_artifacts}
        == {
            "audit_plan.json",
            "content_freeze_audit.json",
            "content_freeze_audit.md",
        },
        "content receipt artifact set mismatch",
    )
    for raw_artifact in content_artifacts:
        artifact = _require_mapping(raw_artifact, "content receipt artifact")
        relative = str(artifact["relative_path"])
        artifact_path = content_output_root / _safe_relative(
            relative, "content artifact path"
        )
        payload, artifact_sha, _ = _read_regular_file_snapshot(
            artifact_path, field="content artifact"
        )
        if relative.endswith(".json"):
            parsed = _decode_json_payload(payload, field="content artifact JSON")
            require(
                payload == _V2.published_json_bytes(parsed),
                "content artifact JSON bytes are not canonical",
            )
        observed = {
            "relative_path": relative,
            "bytes": len(payload),
            "sha256": artifact_sha,
        }
        require(observed == artifact, "content artifact bytes/hash mismatch")
        snapshot_items.append((artifact_path, artifact_sha))
    content_state_path = content_output_root / "run_state.json"
    content_state, _, content_state_sha, _ = _read_parent_json_snapshot(
        content_state_path, field="content state"
    )
    require(
        content_state.get("status") == content_anchor.get("required_run_state_status"),
        "content state status mismatch",
    )
    snapshot_items.append((content_state_path, content_state_sha))
    snapshots = tuple(dict.fromkeys(snapshot_items))
    return ParentContext(
        tuple(tasks), parents, temporal_receipt_sha, output_root, snapshots
    )


def _revalidate_parent_control(context: ParentContext, source_root: Path) -> None:
    for path, expected in context.parent_snapshot_paths:
        require(_sha256_file(path) == expected, "parent control artifact changed")
    for task in context.tasks:
        path = _absolute(source_root) / _safe_relative(task.relative_path, "source task")
        require(_rich_identity(path, field="source final identity") == task.source_rich_identity, "source identity changed during spine execution")


def _available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    return pages * page_size


def _validate_resource_config(config: Mapping[str, Any]) -> None:
    source = _require_mapping(config.get("source"), "source")
    identifiers = _require_mapping(
        config.get("identifier_accounting"), "identifier accounting"
    )
    resources = _require_mapping(config.get("resource_envelope"), "resources")
    require(
        resources.get("declared_input_files") == source.get("expected_files"),
        "resource input-file anchor mismatch",
    )
    require(
        resources.get("declared_raw_rows") == source.get("expected_raw_rows"),
        "resource raw-row anchor mismatch",
    )
    require(
        resources.get("declared_retained_rows")
        == identifiers.get("retained_rows")
        == source.get("expected_retained_rows"),
        "resource retained-row anchor mismatch",
    )
    require(
        resources.get("declared_source_bytes") == source.get("expected_source_bytes"),
        "resource source-byte anchor mismatch",
    )
    require(
        resources.get("source_batch_rows") == DEFAULT_BATCH_SIZE,
        "resource source-batch anchor mismatch",
    )
    require(
        resources.get("sort_chunk_rows") == DEFAULT_SORT_CHUNK_ROWS,
        "resource sort-chunk anchor mismatch",
    )
    _resource_requirements(
        retained_rows=_require_int(
            identifiers.get("retained_rows"), "retained-row resource anchor"
        ),
        batch_size=DEFAULT_BATCH_SIZE,
        sort_chunk_rows=DEFAULT_SORT_CHUNK_ROWS,
        scope=FULL_SCOPE,
        config=config,
    )
    semantics = _require_mapping(
        config.get("resource_bound_semantics"), "resource bound semantics"
    )
    require(
        semantics.get("total_process_rss_upper_bound_claimed") is False
        and semantics.get("native_allocator_scratch_exactly_measured") is False
        and semantics.get("estimates_still_bound_to_plan_and_resume_runtime_fingerprint")
        is True,
        "resource-bound claim semantics drift",
    )


def _resource_requirements(
    *,
    retained_rows: int,
    batch_size: int,
    sort_chunk_rows: int,
    scope: str,
    config: Mapping[str, Any] | None,
) -> dict[str, int]:
    require(retained_rows >= 0, "retained rows are negative")
    require(batch_size > 0, "batch size must be positive")
    require(sort_chunk_rows > 0, "partition sort chunk must be positive")
    resources = (
        _require_mapping(config.get("resource_envelope"), "resource envelope")
        if config is not None
        else {}
    )

    def configured(name: str, default: int) -> int:
        value = resources.get(name, default)
        return _require_int(value, f"resource {name}")

    staging = retained_rows * RECORD_BYTES
    spool = staging
    sorted_runs = staging
    parquet = (
        retained_rows
        * configured(
            "parquet_bytes_per_row_upper_bound",
            DEFAULT_PARQUET_BYTES_PER_ROW_BOUND,
        )
        + PARTITION_COUNT
        * configured(
            "parquet_fixed_bytes_per_partition_upper_bound",
            DEFAULT_PARQUET_FIXED_BYTES_PER_PARTITION_BOUND,
        )
    )
    control = configured(
        "control_artifacts_bytes_upper_bound", DEFAULT_CONTROL_BYTES_BOUND
    )
    atomic_headroom = configured(
        "largest_next_atomic_artifact_bytes_upper_bound", max(parquet, staging)
    )
    reserve = configured("disk_reserve_bytes", DEFAULT_DISK_RESERVE_BYTES)
    multiplier = configured(
        "disk_safety_multiplier_milli", DEFAULT_DISK_SAFETY_MULTIPLIER_MILLI
    )
    require(multiplier >= 1000, "disk safety multiplier is below one")
    working = staging + spool + sorted_runs + parquet + control + atomic_headroom
    base_disk_required = math.ceil(multiplier * working / 1000) + reserve
    fresh_disk_required = base_disk_required
    resume_disk_required = base_disk_required
    preferred_disk = base_disk_required

    projected_arrow = configured(
        "projected_arrow_batch_bytes_upper_bound",
        DEFAULT_PROJECTED_ARROW_BATCH_BYTES_BOUND,
    )
    projected_ipc = configured(
        "projected_ipc_serialized_bytes_upper_bound",
        DEFAULT_PROJECTED_IPC_BATCH_BYTES_BOUND,
    )
    encoded_batch = configured(
        "encoded_record_batch_bytes_upper_bound",
        DEFAULT_ENCODED_BATCH_BYTES_BOUND,
    )
    projected = configured(
        "projected_batch_bytes_upper_bound",
        DEFAULT_PROJECTED_BATCH_BYTES_BOUND,
    )
    require(
        projected == projected_arrow + projected_ipc + encoded_batch,
        "scan retained-payload resource formula mismatch",
    )
    sort_memory = configured(
        "sort_working_bytes_preflight_estimate",
        sort_chunk_rows * RECORD_BYTES * DEFAULT_SORT_MEMORY_MULTIPLIER,
    )
    demux_memory = configured(
        "spool_demultiplex_working_bytes_preflight_estimate",
        SPOOL_DEMUX_CHUNK_ROWS * RECORD_BYTES * 8,
    )
    spool_memory = configured(
        "spool_buffer_bytes_upper_bound",
        PARTITION_COUNT * SPOOL_BUFFER_RECORDS_PER_FILE * RECORD_BYTES,
    )
    parquet_memory = configured(
        "parquet_writer_buffer_bytes_preflight_estimate", 16 << 20
    )
    merge_heap = configured("merge_heap_bytes_preflight_estimate", 64 << 10)
    ram_reserve = configured("ram_reserve_bytes", DEFAULT_RAM_RESERVE_BYTES)
    minimum_nofile = configured("minimum_nofile_soft_limit", 8192)
    calculated_ram = (
        max(projected, sort_memory, demux_memory)
        + spool_memory
        + parquet_memory
        + merge_heap
        + ram_reserve
    )
    if resources:
        require(
            calculated_ram
            == configured(
                "calculated_ram_preflight_sizing_bytes", calculated_ram
            ),
            "RAM preflight sizing formula mismatch",
        )
    required_ram = calculated_ram
    if scope == FULL_SCOPE:
        require(
            retained_rows == configured("declared_retained_rows", retained_rows),
            "full retained-row resource anchor mismatch",
        )
        require(
            configured("fixed_record_bytes", RECORD_BYTES) == RECORD_BYTES,
            "fixed record-byte resource anchor mismatch",
        )
        require(
            staging == configured("staging_payload_bytes_exact", staging),
            "full staging-byte anchor mismatch",
        )
        require(
            spool
            == configured("partition_spool_payload_bytes_upper_bound", spool),
            "full spool-byte formula mismatch",
        )
        require(
            sorted_runs
            == configured("external_run_payload_bytes_upper_bound", sorted_runs),
            "full external-run-byte formula mismatch",
        )
        require(
            parquet == configured("parquet_payload_bytes_upper_bound", parquet),
            "full Parquet-byte formula mismatch",
        )
        require(
            atomic_headroom
            == retained_rows
            * configured(
                "parquet_bytes_per_row_upper_bound",
                DEFAULT_PARQUET_BYTES_PER_ROW_BOUND,
            )
            + configured(
                "parquet_fixed_bytes_per_partition_upper_bound",
                DEFAULT_PARQUET_FIXED_BYTES_PER_PARTITION_BOUND,
            ),
            "full atomic-headroom formula mismatch",
        )
        require(
            working
            == configured("worst_case_remaining_bytes_before_multiplier", working),
            "full remaining-disk formula mismatch",
        )
        require(
            base_disk_required
            == configured(
                "base_required_free_disk_bytes_full", base_disk_required
            ),
            "full base disk formula mismatch",
        )
        fresh_disk_required = configured(
            "required_free_disk_bytes_full_fresh",
            base_disk_required + working,
        )
        require(
            fresh_disk_required == base_disk_required + working,
            "full fresh restartability disk formula mismatch",
        )
        resume_disk_required = configured(
            "required_free_disk_bytes_full_nonterminal_resume",
            base_disk_required,
        )
        require(
            resume_disk_required == base_disk_required,
            "full nonterminal-resume disk formula mismatch",
        )
        preferred_disk = configured(
            "preferred_free_disk_bytes_full", 160 << 30
        )
        require(
            preferred_disk == 160 << 30
            and preferred_disk >= fresh_disk_required,
            "full preferred disk anchor mismatch",
        )
        required_ram = max(
            calculated_ram,
            configured("minimum_available_ram_bytes_full", calculated_ram),
        )
    if scope == CLOSED_FIXTURE_SCOPE:
        base_disk_required = min(base_disk_required, 512 << 20)
        fresh_disk_required = base_disk_required
        resume_disk_required = base_disk_required
        preferred_disk = base_disk_required
        calculated_ram = min(calculated_ram, 512 << 20)
        required_ram = min(required_ram, 512 << 20)
    return {
        "staging_bytes_upper_bound": staging,
        "partition_spool_bytes_upper_bound": spool,
        "sorted_run_bytes_upper_bound": sorted_runs,
        "parquet_bytes_upper_bound": parquet,
        "parquet_bytes_per_row_upper_bound": configured(
            "parquet_bytes_per_row_upper_bound",
            DEFAULT_PARQUET_BYTES_PER_ROW_BOUND,
        ),
        "parquet_fixed_bytes_per_partition_upper_bound": configured(
            "parquet_fixed_bytes_per_partition_upper_bound",
            DEFAULT_PARQUET_FIXED_BYTES_PER_PARTITION_BOUND,
        ),
        "control_bytes_upper_bound": control,
        "atomic_publish_headroom_bytes": atomic_headroom,
        "disk_safety_multiplier_milli": multiplier,
        "disk_reserve_bytes": reserve,
        "base_required_free_disk_bytes_full": base_disk_required,
        "required_free_disk_bytes_full_fresh": fresh_disk_required,
        "required_free_disk_bytes_full_nonterminal_resume": resume_disk_required,
        "preferred_free_disk_bytes_full": preferred_disk,
        "projected_arrow_batch_bytes_upper_bound": projected_arrow,
        "projected_ipc_serialized_bytes_upper_bound": projected_ipc,
        "encoded_record_batch_bytes_upper_bound": encoded_batch,
        "projected_batch_bytes_upper_bound": projected,
        "sort_working_bytes_preflight_estimate": sort_memory,
        "spool_demultiplex_working_bytes_preflight_estimate": demux_memory,
        "spool_buffer_bytes_upper_bound": spool_memory,
        "parquet_writer_buffer_bytes_preflight_estimate": parquet_memory,
        "merge_heap_bytes_preflight_estimate": merge_heap,
        "ram_reserve_bytes": ram_reserve,
        "calculated_ram_preflight_sizing_bytes": calculated_ram,
        "required_available_ram_bytes": required_ram,
        "minimum_nofile_soft_limit": minimum_nofile,
    }


def _atomic_rename_probe(output_parent: Path) -> None:
    probe_root = output_parent
    while not probe_root.exists():
        probe_root = probe_root.parent
    probe_dir = probe_root / f".canonical-spine-preflight-{secrets.token_hex(8)}"
    probe_dir.mkdir(mode=0o700)
    source = probe_dir / "source.tmp"
    target = probe_dir / "target.bin"
    payload = secrets.token_bytes(64)
    try:
        descriptor = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(source, target)
        directory_fd = os.open(probe_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        info = target.lstat()
        require(stat.S_ISREG(info.st_mode), "atomic-rename probe target is not regular")
        require(info.st_nlink == 1, "atomic-rename probe target is hard-linked")
        require(target.read_bytes() == payload, "atomic-rename readback differs")
    finally:
        if source.exists() or source.is_symlink():
            source.unlink()
        if target.exists() or target.is_symlink():
            target.unlink()
        probe_dir.rmdir()


def _enforce_resource_preflight(
    output_parent: Path,
    requirements: Mapping[str, int],
    *,
    invocation_kind: str = FRESH_INVOCATION,
    minimum_nofile: int = 8192,
) -> dict[str, int]:
    probe = output_parent
    while not probe.exists():
        require(probe != probe.parent, "cannot locate output filesystem")
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    available_memory = _available_memory_bytes()
    nofile_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    selected_disk = _selected_required_available_disk_bytes(
        requirements, invocation_kind
    )
    require(
        usage.free >= selected_disk,
        "insufficient free disk for declared production envelope",
    )
    require(
        available_memory >= requirements["required_available_ram_bytes"],
        "insufficient available RAM for declared production envelope",
    )
    require(nofile_soft >= minimum_nofile, "open-file soft limit is too low")
    _atomic_rename_probe(output_parent)
    return {
        "observed_free_disk_bytes": usage.free,
        "observed_available_ram_bytes": available_memory,
        "observed_nofile_soft_limit": int(nofile_soft),
        "required_nofile_soft_limit": minimum_nofile,
        **dict(requirements),
    }


def _selected_required_available_disk_bytes(
    requirements: Mapping[str, int], invocation_kind: str
) -> int:
    if invocation_kind == FRESH_INVOCATION:
        key = "required_free_disk_bytes_full_fresh"
    else:
        require(
            invocation_kind == NONTERMINAL_RESUME_INVOCATION,
            "unknown preflight invocation kind",
        )
        key = "required_free_disk_bytes_full_nonterminal_resume"
    return _require_int(requirements.get(key), f"plan resource {key}")


def _build_preflight_evidence(
    plan: Mapping[str, Any],
    observed: Mapping[str, int],
    *,
    invocation_kind: str,
    source_root: Path,
    output: Path,
    distinct_device_required: bool,
) -> dict[str, Any]:
    requirements = _require_mapping(
        plan.get("resource_requirements"), "plan resource requirements"
    )
    source_device = _filesystem_device(source_root)
    output_device = _filesystem_device(output)
    if distinct_device_required:
        require(source_device != output_device, "source/output device gate differs")
    return {
        "schema_version": _schemas(str(plan["scope"]))["preflight_evidence"],
        "scope": plan["scope"],
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "invocation_kind": invocation_kind,
        "selected_required_available_disk_bytes": (
            _selected_required_available_disk_bytes(
                requirements, invocation_kind
            )
        ),
        "observed_free_disk_bytes": observed["observed_free_disk_bytes"],
        "required_available_ram_bytes": requirements[
            "required_available_ram_bytes"
        ],
        "observed_available_ram_bytes": observed[
            "observed_available_ram_bytes"
        ],
        "required_nofile_soft_limit": requirements["minimum_nofile_soft_limit"],
        "observed_nofile_soft_limit": observed["observed_nofile_soft_limit"],
        "atomic_rename_fsync_readback_probe": "PASS",
        "source_st_dev": source_device,
        "output_st_dev": output_device,
        "source_output_device_gate": (
            "PASS_DISTINCT_DEVICES"
            if distinct_device_required
            else "EXEMPT_CLOSED_TEST_FIXTURE"
        ),
        "latest_nonterminal_invocation_only": True,
        "operator_paths_redacted": True,
    }


def _validate_preflight_evidence(
    output: Path,
    plan: Mapping[str, Any],
    *,
    source_root: Path,
    validate_live_devices: bool = True,
) -> dict[str, Any]:
    evidence = _read_json(output / "preflight_evidence.json", "preflight evidence")
    scope = str(plan["scope"])
    expected_keys = {
        "schema_version",
        "scope",
        "task_fingerprint_sha256",
        "invocation_kind",
        "selected_required_available_disk_bytes",
        "observed_free_disk_bytes",
        "required_available_ram_bytes",
        "observed_available_ram_bytes",
        "required_nofile_soft_limit",
        "observed_nofile_soft_limit",
        "atomic_rename_fsync_readback_probe",
        "source_st_dev",
        "output_st_dev",
        "source_output_device_gate",
        "latest_nonterminal_invocation_only",
        "operator_paths_redacted",
    }
    require(set(evidence) == expected_keys, "preflight evidence key set differs")
    require(evidence.get("schema_version") == _schemas(scope)["preflight_evidence"], "preflight schema differs")
    require(evidence.get("scope") == scope, "preflight scope differs")
    require(evidence.get("task_fingerprint_sha256") == plan["task_fingerprint_sha256"], "preflight task binding differs")
    requirements = _require_mapping(plan.get("resource_requirements"), "plan resources")
    invocation_kind = evidence.get("invocation_kind")
    require(
        invocation_kind in (FRESH_INVOCATION, NONTERMINAL_RESUME_INVOCATION),
        "preflight invocation kind differs",
    )
    selected_disk = _selected_required_available_disk_bytes(
        requirements, str(invocation_kind)
    )
    require(
        evidence.get("selected_required_available_disk_bytes") == selected_disk,
        "preflight selected disk requirement differs",
    )
    for evidence_key, requirement_key in (
        ("required_available_ram_bytes", "required_available_ram_bytes"),
        ("required_nofile_soft_limit", "minimum_nofile_soft_limit"),
    ):
        require(evidence.get(evidence_key) == requirements[requirement_key], f"preflight requirement differs: {evidence_key}")
    require(_require_int(evidence.get("observed_free_disk_bytes"), "observed disk") >= selected_disk, "recorded disk preflight did not pass")
    require(_require_int(evidence.get("observed_available_ram_bytes"), "observed RAM") >= requirements["required_available_ram_bytes"], "recorded RAM preflight did not pass")
    require(_require_int(evidence.get("observed_nofile_soft_limit"), "observed nofile") >= requirements["minimum_nofile_soft_limit"], "recorded nofile preflight did not pass")
    require(evidence.get("atomic_rename_fsync_readback_probe") == "PASS", "atomic probe did not pass")
    if validate_live_devices:
        require(evidence.get("source_st_dev") == _filesystem_device(source_root), "source device changed")
        require(evidence.get("output_st_dev") == _filesystem_device(output), "output device changed")
    if scope in (FULL_SCOPE, PILOT_SCOPE):
        require(evidence.get("source_output_device_gate") == "PASS_DISTINCT_DEVICES" and evidence["source_st_dev"] != evidence["output_st_dev"], "production device gate did not pass")
    else:
        require(evidence.get("source_output_device_gate") == "EXEMPT_CLOSED_TEST_FIXTURE", "fixture device exemption differs")
    require(evidence.get("latest_nonterminal_invocation_only") is True and evidence.get("operator_paths_redacted") is True, "preflight evidence policy differs")
    return evidence


def _filesystem_device(path: Path) -> int:
    current = _absolute(path)
    while not current.exists():
        require(current != current.parent, "cannot locate filesystem ancestor")
        current = current.parent
    return int(current.stat().st_dev)


def _enforce_distinct_device(source_root: Path, output_dir: Path) -> None:
    require(
        _filesystem_device(source_root) != _filesystem_device(output_dir),
        "production source and output must be on distinct filesystems",
    )


def _validate_output_root_policy(output: Path) -> None:
    info = output.lstat()
    require(stat.S_ISDIR(info.st_mode), "output root is not a directory")
    require(not output.is_symlink(), "output root is a symlink")
    require(stat.S_IMODE(info.st_mode) == 0o700, "output root mode is not 0700")
    require(info.st_uid == os.geteuid(), "output root owner differs from executor")


def _paths_overlap(left: Path, right: Path) -> bool:
    return _V2._paths_overlap(left, right)


def _validate_output_location(
    output_dir: Path,
    *,
    source_root: Path,
    parent_roots: Sequence[Path] = (),
) -> Path:
    output = _absolute(output_dir)
    source = _absolute(source_root)
    require(not _paths_overlap(output, source), "output overlaps source")
    require(not _paths_overlap(output, REPO_ROOT), "output overlaps repository")
    for parent_root in parent_roots:
        require(
            not _paths_overlap(output, _absolute(parent_root)),
            "output overlaps parent output",
        )
    _V2._assert_no_symlink_components(output, field="output")
    if output.exists():
        _validate_output_root_policy(output)
    return output


@contextmanager
def _output_lock(output: Path, *, allow_create: bool = True) -> Iterator[None]:
    lock_path = output / "audit.lock"
    created = False
    try:
        if allow_create:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
                created = True
            except FileExistsError:
                descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
        else:
            descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    except OSError as exc:
        raise SpineAuditError("cannot open output lock") from exc
    try:
        if created:
            os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), "output lock is not regular")
        require(stat.S_IMODE(info.st_mode) == 0o600, "output lock mode drift")
        require(info.st_uid == os.geteuid(), "output lock owner differs")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpineAuditError("output directory is locked by another run") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _prepare_output(
    output_dir: Path,
    *,
    source_root: Path,
    parent_roots: Sequence[Path] = (),
) -> Path:
    output = _validate_output_location(
        output_dir, source_root=source_root, parent_roots=parent_roots
    )
    _mkdir_private(output)
    output.chmod(0o700)
    for relative in (
        "checkpoints",
        "checkpoints/scan",
        "checkpoints/scan/main",
        "checkpoints/partition",
        "staging",
        "staging/main",
        "spine",
        ".work_time_spine_tmp",
        ".work_time_spine_tmp/staging",
        ".work_time_spine_tmp/spool",
        ".work_time_spine_tmp/runs",
        ".work_time_spine_tmp/sorted",
    ):
        _mkdir_private(output / relative)
    _validate_output_root_policy(output)
    return output


def _runtime_fingerprint() -> dict[str, str]:
    cpp = getattr(pa, "cpp_build_info", None)
    return {
        "pyarrow_version": str(pa.__version__),
        "parquet_cpp_version": str(getattr(cpp, "version", "UNKNOWN")),
        "numpy_version": str(np.__version__),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
    }


def _record_digest_file(path: Path, rows: int) -> str:
    require(rows >= 0, "negative record count")
    digest = hashlib.sha256()
    digest.update(len(RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    digest.update(RECORD_HASH_DOMAIN)
    digest.update(rows.to_bytes(8, "little"))
    observed = 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        while payload := handle.read(RECORD_BYTES * 65_536):
            require(len(payload) % RECORD_BYTES == 0, "staging byte width mismatch")
            records = np.frombuffer(payload, dtype=RECORD_DTYPE)
            _validate_record_array(records)
            _update_framed_record_digest(digest, records)
            observed += len(records)
    require(observed == rows, "staging row count mismatch")
    return digest.hexdigest()


def _validate_record_array(records: np.ndarray[Any, Any]) -> None:
    require(records.dtype == RECORD_DTYPE, "record array dtype differs")
    if len(records) == 0:
        return
    dispositions = records["temporal_disposition"]
    require(bool(np.all(records["reserved"] == 0)), "staging reserved byte is nonzero")
    require(
        bool(np.all(dispositions <= max(DISPOSITION_NAMES))),
        "unknown temporal disposition",
    )
    nulls = dispositions == 0
    require(
        bool(np.all(records["publication_day"][nulls] == -(2**31)))
        and bool(np.all(records["publication_year"][nulls] == 0)),
        "both-null staging sentinel mismatch",
    )
    require(
        bool(np.all(records["publication_day"][~nulls] != -(2**31)))
        and bool(np.all(records["publication_year"][~nulls] != 0)),
        "nonnull staging record uses null sentinel",
    )


def _update_framed_record_digest(
    digest: Any, records: np.ndarray[Any, Any]
) -> None:
    framed = np.empty(len(records), dtype=FRAMED_RECORD_DTYPE)
    framed["encoded_length"] = RECORD_BYTES
    framed["record"] = records.view(f"V{RECORD_BYTES}").reshape(-1)
    digest.update(memoryview(framed).cast("B"))


def _empty_metrics() -> dict[str, int]:
    return {key: 0 for key in SPINE_METRIC_KEYS}


def _increment(metrics: dict[str, int], key: str, condition: bool) -> None:
    if condition:
        metrics[key] += 1


def _scan_task_to_pending(
    source_root: Path,
    output: Path,
    task: Task,
    *,
    batch_size: int,
    snapshot: date,
    checkpoint_schema: str,
    task_fingerprint_sha256: str,
    projected_arrow_batch_bytes_bound: int,
    projected_ipc_batch_bytes_bound: int,
    encoded_batch_bytes_bound: int,
    retained_payload_bytes_bound: int,
) -> dict[str, Any]:
    for value in (
        projected_arrow_batch_bytes_bound,
        projected_ipc_batch_bytes_bound,
        encoded_batch_bytes_bound,
        retained_payload_bytes_bound,
    ):
        require(value > 0, "scan payload byte bound is invalid")
    relative = _safe_relative(task.relative_path, "source task")
    source_path = _absolute(source_root) / relative
    _V2._assert_no_symlink_components(source_path, field="source Parquet")
    try:
        descriptor = os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise SpineAuditError("cannot open source Parquet") from exc
    handle = os.fdopen(descriptor, "rb")
    start_identity = _V2._rich_identity_from_stat(os.fstat(handle.fileno()))
    require(start_identity == task.source_rich_identity, "source identity differs before scan")
    try:
        parquet = pq.ParquetFile(handle)
        require(parquet.metadata.num_rows == task.footer_rows, "source footer row mismatch")
        projected_schema = {
            name: {
                "nullable": bool(parquet.schema_arrow.field(name).nullable),
                "type": str(parquet.schema_arrow.field(name).type),
            }
            for name in ("id", "publication_date", "publication_year")
        }
        require(projected_schema == task.projected_schema, "source projected schema differs")
        require(_sha256_bytes(_canonical_json_bytes(projected_schema)) == task.projected_schema_sha256, "source projected schema hash differs")
    except BaseException:
        handle.close()
        raise

    token = _sha256_bytes(task.relative_path.encode("utf-8"))
    pending_path = output / ".work_time_spine_tmp/staging" / f"{token}.bin"
    pending_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp_path = pending_path.with_name(f".{pending_path.name}.{secrets.token_hex(8)}.tmp")
    projected_digest = hashlib.sha256()
    staging_digest = hashlib.sha256()
    metrics = _empty_metrics()
    disposition_counts = {name: 0 for name in DISPOSITION_NAMES.values()}
    minimum_year: int | None = None
    maximum_year: int | None = None
    retained_rows = 0
    scanned_rows = 0
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as staging_handle:
            for row_group in range(parquet.metadata.num_row_groups):
                for batch in parquet.iter_batches(
                    batch_size=batch_size,
                    row_groups=[row_group],
                    columns=["id", "publication_date", "publication_year"],
                    use_threads=False,
                ):
                    require(
                        batch.nbytes <= projected_arrow_batch_bytes_bound,
                        "projected Arrow batch exceeds its individual bound",
                    )
                    serialized = batch.serialize()
                    require(
                        serialized.size <= projected_ipc_batch_bytes_bound,
                        "serialized projected batch exceeds its individual bound",
                    )
                    encoded_upper_bound = batch.num_rows * RECORD_BYTES
                    projected_digest.update(serialized.size.to_bytes(8, "little"))
                    projected_digest.update(memoryview(serialized))
                    require(
                        encoded_upper_bound <= encoded_batch_bytes_bound,
                        "encoded staging batch exceeds its individual bound",
                    )
                    require(
                        batch.nbytes + serialized.size + encoded_upper_bound
                        <= retained_payload_bytes_bound,
                        "scan retained payload buffers exceed the combined bound",
                    )
                    del serialized
                    encoded_batch = bytearray()
                    for index in range(batch.num_rows):
                        raw_id = batch.column(0)[index].as_py()
                        raw_date = batch.column(1)[index].as_py()
                        raw_year = batch.column(2)[index].as_py()
                        metrics["raw_rows"] += 1
                        if raw_id == W_MINUS_ONE:
                            metrics["quarantined_w_minus_1_rows"] += 1
                            continue
                        metrics["retained_rows"] += 1
                        retained_rows += 1
                        grammar = isinstance(raw_id, str) and WORK_PATTERN.fullmatch(raw_id) is not None
                        _increment(metrics, "null_id_rows", raw_id is None)
                        _increment(metrics, "grammar_valid_id_rows", grammar)
                        _increment(metrics, "unexpected_malformed_id_rows", not grammar)
                        _increment(metrics, "w_zero_rows", raw_id == "https://openalex.org/W0")

                        date_null = raw_date is None
                        year_null = raw_year is None
                        both_null = date_null and year_null
                        exactly_one = date_null ^ year_null
                        _increment(metrics, "publication_both_null_rows", both_null)
                        _increment(metrics, "publication_exactly_one_null_rows", exactly_one)
                        _increment(metrics, "publication_both_nonnull_rows", not date_null and not year_null)
                        _increment(metrics, "publication_year_null_rows", year_null)

                        date_valid = False
                        day = 0
                        derived_year = 0
                        if not date_null:
                            try:
                                date_valid, day, derived_year = _V2._strict_date(raw_date)
                            except SpineAuditError:
                                metrics["publication_date_parse_error_rows"] += 1
                        year_valid = False
                        year = 0
                        if not year_null:
                            try:
                                year_valid, year = _V2._strict_year(raw_year)
                            except SpineAuditError:
                                metrics["publication_year_invalid_rows"] += 1
                        checked = date_valid and year_valid
                        matched = checked and derived_year == year
                        _increment(metrics, "publication_concordance_checked_rows", checked)
                        _increment(metrics, "publication_date_year_match_rows", matched)
                        _increment(metrics, "publication_date_year_mismatch_rows", checked and not matched)
                        if date_valid:
                            boundary = (date(1900, 1, 1) - date(1970, 1, 1)).days
                            snapshot_day = (snapshot - date(1970, 1, 1)).days
                            _increment(metrics, "publication_pre_1900_rows", day < boundary)
                            _increment(metrics, "publication_1900_before_snapshot_year_rows", day >= boundary and derived_year < snapshot.year)
                            _increment(metrics, "publication_snapshot_year_on_or_before_snapshot_rows", derived_year == snapshot.year and day <= snapshot_day)
                            _increment(metrics, "publication_snapshot_year_after_snapshot_rows", derived_year == snapshot.year and day > snapshot_day)
                            _increment(metrics, "publication_post_snapshot_year_rows", derived_year > snapshot.year)
                        if year_valid:
                            _increment(metrics, "publication_year_pre_1900_rows", year < 1900)
                            _increment(metrics, "publication_year_snapshot_year_rows", year == snapshot.year)
                            _increment(metrics, "publication_year_post_snapshot_year_rows", year > snapshot.year)
                            minimum_year = year if minimum_year is None else min(minimum_year, year)
                            maximum_year = year if maximum_year is None else max(maximum_year, year)

                        require(grammar, "retained work ID is null or malformed")
                        suffix = WORK_PATTERN.fullmatch(str(raw_id)).group(1)  # type: ignore[union-attr]
                        work_id_numeric = int(suffix)
                        require(work_id_numeric <= 2**64 - 1, "work ID exceeds uint64")
                        disposition, normalized_day, normalized_year = _V2.classify_publication(
                            raw_date, raw_year, snapshot=snapshot
                        )
                        record = Record(work_id_numeric, normalized_day, normalized_year, disposition)
                        encoded = record.encode()
                        encoded_batch.extend(encoded)
                        disposition_counts[DISPOSITION_NAMES[disposition]] += 1
                    staging_handle.write(encoded_batch)
                    staging_digest.update(encoded_batch)
                    del encoded_batch
                    scanned_rows += batch.num_rows
            staging_handle.flush()
            os.fsync(staging_handle.fileno())
        require(_V2._rich_identity_from_stat(os.fstat(handle.fileno())) == start_identity, "open source changed during scan")
        require(_rich_identity(source_path, field="source after scan") == start_identity, "source path changed during scan")
    except BaseException as exc:
        if temp_path.exists():
            temp_path.unlink()
        if isinstance(exc, (KeyboardInterrupt, SystemExit, SpineAuditError)):
            raise
        raise SpineAuditError("cannot scan source projection") from exc
    finally:
        handle.close()
    require(scanned_rows == task.footer_rows, "scanned rows differ from footer")
    os.replace(temp_path, pending_path)
    _fsync_directory(pending_path.parent)
    pending_path.chmod(0o600)
    require(pending_path.stat().st_size == retained_rows * RECORD_BYTES, "staging byte count mismatch")

    extrema = {"minimum": minimum_year, "maximum": maximum_year}
    if task.temporal_metrics is not None:
        expected = {key: task.temporal_metrics[key] for key in SPINE_METRIC_KEYS}
        require(metrics == expected, "spine/temporal per-file metric mismatch")
        require(extrema == task.publication_year_extrema, "spine/temporal extrema mismatch")
    for key in (
        "null_id_rows",
        "unexpected_malformed_id_rows",
        "publication_exactly_one_null_rows",
        "publication_date_parse_error_rows",
        "publication_year_invalid_rows",
        "publication_date_year_mismatch_rows",
    ):
        require(metrics[key] == 0, f"rejected-row metric is nonzero: {key}")

    core = {
        "schema_version": checkpoint_schema,
        "task_fingerprint_sha256": task_fingerprint_sha256,
        "file_index": task.file_index,
        "relative_path": task.relative_path,
        "source_rich_identity": start_identity,
        "footer_rows": scanned_rows,
        "temporal_parent_task_sha256": task.temporal_task_sha256,
        "temporal_parent_checkpoint_sha256": task.temporal_checkpoint_sha256,
        "projected_schema_sha256": task.projected_schema_sha256,
        "projected_payload_sha256": projected_digest.hexdigest(),
        "disposition_metrics": metrics,
        "disposition_metrics_sha256": _sha256_bytes(_canonical_json_bytes(metrics)),
        "publication_year_extrema": extrema,
        "disposition_counts": disposition_counts,
        "row_level_disposition_payload_sha256": _record_digest_file(pending_path, retained_rows),
        "staging_shard_relative_path": f"staging/main/{token}.bin",
        "staging_shard_rows": retained_rows,
        "staging_shard_bytes": pending_path.stat().st_size,
        "staging_shard_sha256": staging_digest.hexdigest(),
        "operator_paths_redacted": True,
    }
    core["checkpoint_sha256"] = _sha256_bytes(_canonical_json_bytes(core))
    return core


class _BoundedSpoolWriter:
    def __init__(self, root: Path, maximum_open: int = MAX_OPEN_SPOOL_FILES) -> None:
        require(
            maximum_open >= PARTITION_COUNT,
            "spool open-file bound must cover every partition",
        )
        self.root = root
        self.maximum_open = maximum_open
        self.handles: dict[int, BinaryIO] = {}
        self.counts = [0] * PARTITION_COUNT
        self.digests = [hashlib.sha256() for _ in range(PARTITION_COUNT)]

    def _handle(self, partition_id: int) -> BinaryIO:
        handle = self.handles.get(partition_id)
        if handle is not None:
            return handle
        require(
            len(self.handles) < self.maximum_open,
            "partition spool descriptor bound exceeded",
        )
        path = self.root / f"part-{partition_id:04d}.bin"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        handle = os.fdopen(
            descriptor,
            "wb",
            buffering=RECORD_BYTES * SPOOL_BUFFER_RECORDS_PER_FILE,
        )
        self.handles[partition_id] = handle
        return handle

    def write_records(self, records: np.ndarray[Any, Any]) -> None:
        require(records.dtype == RECORD_DTYPE, "spool record dtype differs")
        if not len(records):
            return
        partition_ids = np.remainder(
            records["work_id_numeric"], PARTITION_COUNT
        ).astype(np.uint16, copy=False)
        order = np.argsort(partition_ids, kind="stable")
        ordered_partitions = partition_ids[order]
        ordered_records = np.ascontiguousarray(records[order])
        boundaries = (
            np.flatnonzero(ordered_partitions[1:] != ordered_partitions[:-1])
            + 1
        )
        starts = np.concatenate((np.array([0]), boundaries))
        stops = np.concatenate((boundaries, np.array([len(records)])))
        for start, stop in zip(starts, stops, strict=True):
            start_index = int(start)
            stop_index = int(stop)
            partition_id = int(ordered_partitions[start_index])
            payload = memoryview(
                ordered_records[start_index:stop_index]
            ).cast("B")
            self._handle(partition_id).write(payload)
            self.counts[partition_id] += stop_index - start_index
            self.digests[partition_id].update(payload)

    def close(self) -> None:
        errors: list[BaseException] = []
        for handle in self.handles.values():
            try:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            except BaseException as exc:  # pragma: no cover - defensive aggregation
                errors.append(exc)
        self.handles.clear()
        if errors:
            raise SpineAuditError("cannot durably close partition spool") from errors[0]

    def __enter__(self) -> _BoundedSpoolWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _iter_encoded_records(path: Path) -> Iterator[tuple[Record, bytes]]:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "record artifact is not regular")
    require(info.st_nlink == 1, "record artifact is hard-linked")
    require(info.st_size % RECORD_BYTES == 0, "record artifact width mismatch")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        while payload := handle.read(RECORD_BYTES * 65_536):
            require(len(payload) % RECORD_BYTES == 0, "record chunk width mismatch")
            for offset in range(0, len(payload), RECORD_BYTES):
                encoded = payload[offset : offset + RECORD_BYTES]
                yield Record.decode(encoded), encoded


def _build_partition_spools(
    output: Path,
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spool_root = output / ".work_time_spine_tmp/spool"
    if spool_root.exists():
        _V2._assert_no_symlink_components(spool_root, field="partition spool")
        shutil.rmtree(spool_root)
    _mkdir_private(spool_root)
    total_rows = 0
    with _BoundedSpoolWriter(spool_root) as writer:
        for checkpoint in checkpoints:
            token = _sha256_bytes(str(checkpoint["relative_path"]).encode("utf-8"))
            staging_path = output / "staging/main" / f"{token}.bin"
            expected_rows = _require_int(
                checkpoint.get("staging_shard_rows"), "spool staging rows"
            )
            info = staging_path.lstat()
            require(stat.S_ISREG(info.st_mode), "spool input is not regular")
            require(
                info.st_size
                == checkpoint.get("staging_shard_bytes")
                == expected_rows * RECORD_BYTES,
                "spool input byte count mismatch",
            )
            raw_digest = hashlib.sha256()
            record_digest = hashlib.sha256()
            record_digest.update(len(RECORD_HASH_DOMAIN).to_bytes(8, "little"))
            record_digest.update(RECORD_HASH_DOMAIN)
            record_digest.update(expected_rows.to_bytes(8, "little"))
            observed = 0
            descriptor = os.open(staging_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as staging_handle:
                while payload := staging_handle.read(
                    RECORD_BYTES * SPOOL_DEMUX_CHUNK_ROWS
                ):
                    require(
                        len(payload) % RECORD_BYTES == 0,
                        "spool input record is truncated",
                    )
                    raw_digest.update(payload)
                    records = np.frombuffer(payload, dtype=RECORD_DTYPE)
                    _validate_record_array(records)
                    _update_framed_record_digest(record_digest, records)
                    writer.write_records(records)
                    observed += len(records)
            require(
                observed == expected_rows,
                "staging/spool input row mismatch",
            )
            require(
                raw_digest.hexdigest() == checkpoint.get("staging_shard_sha256"),
                "spool input SHA-256 mismatch",
            )
            require(
                record_digest.hexdigest()
                == checkpoint.get("row_level_disposition_payload_sha256"),
                "spool input record digest mismatch",
            )
            total_rows += observed
        counts = tuple(writer.counts)
        spool_digests = tuple(item.hexdigest() for item in writer.digests)
    partitions: list[dict[str, Any]] = []
    for partition_id in range(PARTITION_COUNT):
        path = spool_root / f"part-{partition_id:04d}.bin"
        if not path.exists():
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            os.close(descriptor)
        path.chmod(0o600)
        info = path.lstat()
        require(info.st_nlink == 1, "partition spool is hard-linked")
        require(
            info.st_size == counts[partition_id] * RECORD_BYTES,
            "partition spool byte count mismatch",
        )
        partitions.append(
            {
                "partition_id": partition_id,
                "relative_path": f".work_time_spine_tmp/spool/part-{partition_id:04d}.bin",
                "rows": counts[partition_id],
                "bytes": info.st_size,
                "sha256": spool_digests[partition_id],
            }
        )
    require(sum(counts) == total_rows, "partition spool conservation mismatch")
    manifest = {
        "schema_version": "openalex_g1_canonical_work_time_spine_temporary_spool_manifest.v1",
        "partition_count": PARTITION_COUNT,
        "rows": total_rows,
        "bytes": total_rows * RECORD_BYTES,
        "maximum_open_files": MAX_OPEN_SPOOL_FILES,
        "partition_rows": list(counts),
        "partition_tree_sha256": _tree_digest(partitions),
        "partitions": partitions,
        "operator_paths_redacted": True,
    }
    _atomic_json(output / ".work_time_spine_tmp/spool_manifest.json", manifest)
    return manifest


def _write_sorted_runs(
    output: Path,
    partition_id: int,
    *,
    sort_chunk_rows: int,
    spool_contract: Mapping[str, Any],
) -> tuple[list[RunArtifact], int]:
    require(sort_chunk_rows > 0, "sort chunk rows must be positive")
    partition_run_root = output / ".work_time_spine_tmp/runs" / f"part-{partition_id:04d}"
    if partition_run_root.exists():
        _V2._assert_no_symlink_components(partition_run_root, field="sort-run root")
        shutil.rmtree(partition_run_root)
    _mkdir_private(partition_run_root)
    spool_path = output / ".work_time_spine_tmp/spool" / f"part-{partition_id:04d}.bin"
    require(
        spool_contract.get("partition_id") == partition_id
        and spool_contract.get("relative_path")
        == f".work_time_spine_tmp/spool/part-{partition_id:04d}.bin",
        "partition spool contract binding mismatch",
    )
    info = spool_path.lstat()
    require(stat.S_ISREG(info.st_mode), "partition spool is not regular")
    require(info.st_nlink == 1, "partition spool is hard-linked")
    require(info.st_size == spool_contract.get("bytes"), "partition spool bytes differ")
    runs: list[RunArtifact] = []
    total = 0
    spool_digest = hashlib.sha256()
    descriptor = os.open(spool_path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as spool_handle:
        while payload := spool_handle.read(sort_chunk_rows * RECORD_BYTES):
            require(len(payload) % RECORD_BYTES == 0, "partition spool is truncated")
            spool_digest.update(payload)
            records = np.frombuffer(payload, dtype=RECORD_DTYPE)
            _validate_record_array(records)
            require(
                bool(
                    np.all(
                        records["work_id_numeric"] % PARTITION_COUNT
                        == partition_id
                    )
                ),
                "partition spool modulo mismatch",
            )
            order = np.argsort(records["work_id_numeric"], kind="stable")
            sorted_payload = records[order].tobytes(order="C")
            run_path = partition_run_root / f"run-{len(runs):04d}.bin"
            run_descriptor = os.open(
                run_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(run_descriptor, "wb") as run_handle:
                run_handle.write(sorted_payload)
                run_handle.flush()
                os.fsync(run_handle.fileno())
            runs.append(
                RunArtifact(
                    path=run_path,
                    rows=len(records),
                    bytes=len(sorted_payload),
                    sha256=_sha256_bytes(sorted_payload),
                )
            )
            total += len(records)
    _fsync_directory(partition_run_root)
    require(total == spool_contract.get("rows"), "partition spool rows differ")
    require(
        total * RECORD_BYTES == spool_contract.get("bytes"),
        "partition spool row/byte conservation differs",
    )
    require(
        spool_digest.hexdigest() == spool_contract.get("sha256"),
        "partition spool SHA-256 differs",
    )
    require(len(runs) <= 512, "partition external-run count exceeds open-file bound")
    return runs, total


class _RunReader:
    def __init__(self, contract: RunArtifact) -> None:
        self.contract = contract
        self.handle = os.fdopen(
            os.open(contract.path, os.O_RDONLY | os.O_NOFOLLOW), "rb"
        )
        self.digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0
        self.verified = False

    def next(self) -> tuple[Record, bytes] | None:
        encoded = self.handle.read(RECORD_BYTES)
        if not encoded:
            self._verify_complete()
            return None
        require(len(encoded) == RECORD_BYTES, "sort run record is truncated")
        self.digest.update(encoded)
        self.rows += 1
        self.bytes += len(encoded)
        return Record.decode(encoded), encoded

    def _verify_complete(self) -> None:
        require(self.rows == self.contract.rows, "sort run row count differs")
        require(self.bytes == self.contract.bytes, "sort run byte count differs")
        require(self.digest.hexdigest() == self.contract.sha256, "sort run SHA-256 differs")
        self.verified = True

    def close(self) -> None:
        self.handle.close()


def _encoded_records_to_table(payload: bytes | bytearray) -> pa.Table:
    require(len(payload) % RECORD_BYTES == 0, "Parquet row-group payload is truncated")
    records = np.frombuffer(payload, dtype=RECORD_DTYPE)
    _validate_record_array(records)
    dispositions = records["temporal_disposition"]
    null_mask = dispositions == 0
    return pa.Table.from_arrays(
        [
            pa.array(records["work_id_numeric"], type=pa.uint64()),
            pa.array(records["publication_day"], mask=null_mask, type=pa.int32()),
            pa.array(records["publication_year"], mask=null_mask, type=pa.uint16()),
            pa.array(dispositions, type=pa.uint8()),
        ],
        schema=ARROW_SCHEMA,
    ).combine_chunks().replace_schema_metadata(None)


def _new_parquet_writer(target: BinaryIO) -> pq.ParquetWriter:
    return pq.ParquetWriter(
        target,
        ARROW_SCHEMA,
        version="2.6",
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        data_page_version="1.0",
        write_batch_size=262_144,
        use_byte_stream_split=False,
        write_page_index=False,
        write_page_checksum=False,
        store_schema=False,
        sorting_columns=[],
    )


def _stream_validate_parquet(
    path: Path,
    *,
    partition_id: int,
    expected_rows: int,
) -> dict[str, Any]:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "partition output is not regular")
    require(info.st_nlink == 1, "partition output is hard-linked")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    digest.update(len(RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    digest.update(RECORD_HASH_DOMAIN)
    digest.update(expected_rows.to_bytes(8, "little"))
    rows = 0
    previous: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    publication_year_minimum: int | None = None
    publication_year_maximum: int | None = None
    disposition_counts = {name: 0 for name in DISPOSITION_NAMES.values()}
    w_zero_disposition_counts = {
        name: 0 for name in DISPOSITION_NAMES.values()
    }
    nullability = {
        "publication_day_null_rows": 0,
        "publication_year_null_rows": 0,
    }
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            parquet = pq.ParquetFile(handle)
            _V2._validate_parquet_schema(parquet)
            require(parquet.metadata.num_rows == expected_rows, "partition footer row mismatch")
            for row_group in range(parquet.metadata.num_row_groups):
                metadata = parquet.metadata.row_group(row_group)
                for column_index in range(metadata.num_columns):
                    column = metadata.column(column_index)
                    require(column.compression == "UNCOMPRESSED", "partition compression drift")
                    require(column.is_stats_set is False, "partition statistics drift")
            for batch in parquet.iter_batches(batch_size=262_144, use_threads=False):
                ids = batch.column(0)
                days = batch.column(1)
                years = batch.column(2)
                dispositions = batch.column(3)
                id_values = np.asarray(
                    ids.to_numpy(zero_copy_only=False), dtype=np.uint64
                )
                disposition_values = np.asarray(
                    dispositions.to_numpy(zero_copy_only=False), dtype=np.uint8
                )
                day_nulls = np.asarray(
                    days.is_null().to_numpy(zero_copy_only=False), dtype=np.bool_
                )
                year_nulls = np.asarray(
                    years.is_null().to_numpy(zero_copy_only=False), dtype=np.bool_
                )
                require(
                    bool(np.array_equal(day_nulls, year_nulls)),
                    "decoded publication nullability differs",
                )
                require(
                    bool(np.array_equal(day_nulls, disposition_values == 0)),
                    "decoded disposition/nullability differs",
                )
                day_values = np.asarray(
                    pc.fill_null(days, -(2**31)).to_numpy(zero_copy_only=False),
                    dtype=np.int32,
                )
                year_values = np.asarray(
                    pc.fill_null(years, 0).to_numpy(zero_copy_only=False),
                    dtype=np.uint16,
                )
                require(
                    bool(np.all(id_values % PARTITION_COUNT == partition_id)),
                    "decoded partition modulo mismatch",
                )
                if len(id_values):
                    require(
                        previous is None or int(id_values[0]) > previous,
                        "decoded partition IDs are not strictly increasing",
                    )
                    require(
                        bool(np.all(id_values[1:] > id_values[:-1])),
                        "decoded partition IDs are not strictly increasing",
                    )
                    previous = int(id_values[-1])
                    minimum = int(id_values[0]) if minimum is None else minimum
                    maximum = int(id_values[-1])
                if bool(np.any(~day_nulls)):
                    non_null_years = year_values[~year_nulls]
                    batch_year_minimum = int(np.min(non_null_years))
                    batch_year_maximum = int(np.max(non_null_years))
                    publication_year_minimum = (
                        batch_year_minimum
                        if publication_year_minimum is None
                        else min(publication_year_minimum, batch_year_minimum)
                    )
                    publication_year_maximum = (
                        batch_year_maximum
                        if publication_year_maximum is None
                        else max(publication_year_maximum, batch_year_maximum)
                    )
                    date_values = pa.array(
                        day_values, mask=day_nulls, type=pa.date32()
                    )
                    derived_years = np.asarray(
                        pc.fill_null(pc.year(date_values), 0).to_numpy(
                            zero_copy_only=False
                        ),
                        dtype=np.int64,
                    )
                    require(
                        bool(
                            np.all(
                                derived_years[~day_nulls]
                                == year_values[~day_nulls]
                            )
                        ),
                        "decoded partition year/day mismatch",
                    )
                counts = np.bincount(
                    disposition_values, minlength=max(DISPOSITION_NAMES) + 1
                )
                require(
                    len(counts) == len(DISPOSITION_NAMES),
                    "decoded disposition is unknown",
                )
                for code, name in DISPOSITION_NAMES.items():
                    disposition_counts[name] += int(counts[code])
                w_zero_mask = id_values == 0
                if bool(np.any(w_zero_mask)):
                    w_zero_counts = np.bincount(
                        disposition_values[w_zero_mask],
                        minlength=max(DISPOSITION_NAMES) + 1,
                    )
                    require(
                        len(w_zero_counts) == len(DISPOSITION_NAMES),
                        "decoded W0 disposition is unknown",
                    )
                    for code, name in DISPOSITION_NAMES.items():
                        w_zero_disposition_counts[name] += int(
                            w_zero_counts[code]
                        )
                null_count = int(np.count_nonzero(day_nulls))
                nullability["publication_day_null_rows"] += null_count
                nullability["publication_year_null_rows"] += null_count
                encoded = np.empty(batch.num_rows, dtype=RECORD_DTYPE)
                encoded["work_id_numeric"] = id_values
                encoded["publication_day"] = day_values
                encoded["publication_year"] = year_values
                encoded["temporal_disposition"] = disposition_values
                encoded["reserved"] = 0
                _validate_record_array(encoded)
                _update_framed_record_digest(digest, encoded)
                rows += batch.num_rows
    finally:
        os.close(descriptor)
    require(rows == expected_rows, "decoded partition row mismatch")
    return {
        "rows": rows,
        "minimum_work_id_numeric": minimum,
        "maximum_work_id_numeric": maximum,
        "publication_year_extrema": {
            "minimum": publication_year_minimum,
            "maximum": publication_year_maximum,
        },
        "record_sha256": digest.hexdigest(),
        "disposition_counts": disposition_counts,
        "w_zero_disposition_counts": w_zero_disposition_counts,
        "nullability_counts": nullability,
    }


def _merge_runs_to_temporary_parquet(
    output: Path,
    partition_id: int,
    runs: Sequence[RunArtifact],
    expected_rows: int,
    *,
    task_fingerprint_sha256: str,
    checkpoint_schema: str,
    scan_tree_sha256: str,
    staging_tree_sha256: str,
    parquet_bytes_per_row_bound: int = DEFAULT_PARQUET_BYTES_PER_ROW_BOUND,
    parquet_fixed_bytes_bound: int = DEFAULT_PARQUET_FIXED_BYTES_PER_PARTITION_BOUND,
) -> dict[str, Any]:
    temporary_path = (
        output / ".work_time_spine_tmp/sorted" / f"part-{partition_id:04d}.parquet"
    )
    if temporary_path.exists() or temporary_path.is_symlink():
        temporary_path.unlink()
    descriptor = os.open(
        temporary_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    parquet_handle = os.fdopen(descriptor, "wb")
    writer = _new_parquet_writer(parquet_handle)
    digest = hashlib.sha256()
    digest.update(len(RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    digest.update(RECORD_HASH_DOMAIN)
    digest.update(expected_rows.to_bytes(8, "little"))
    previous: int | None = None
    rows = 0
    try:
        if len(runs) == 1:
            run_contract = runs[0]
            run_digest = hashlib.sha256()
            run_rows = 0
            run_bytes = 0
            run_descriptor = os.open(
                run_contract.path, os.O_RDONLY | os.O_NOFOLLOW
            )
            with os.fdopen(run_descriptor, "rb") as run_handle:
                while payload := run_handle.read(262_144 * RECORD_BYTES):
                    require(
                        len(payload) % RECORD_BYTES == 0,
                        "single sort run is truncated",
                    )
                    records = np.frombuffer(payload, dtype=RECORD_DTYPE)
                    _validate_record_array(records)
                    run_digest.update(payload)
                    run_rows += len(records)
                    run_bytes += len(payload)
                    identifiers = records["work_id_numeric"]
                    require(
                        bool(
                            np.all(identifiers % PARTITION_COUNT == partition_id)
                        ),
                        "merged record modulo mismatch",
                    )
                    if len(identifiers):
                        require(
                            previous is None or int(identifiers[0]) > previous,
                            "duplicate numeric work ID detected within sort run",
                        )
                        require(
                            bool(np.all(identifiers[1:] > identifiers[:-1])),
                            "duplicate numeric work ID detected within sort run",
                        )
                        previous = int(identifiers[-1])
                    _update_framed_record_digest(digest, records)
                    writer.write_table(
                        _encoded_records_to_table(payload),
                        row_group_size=262_144,
                    )
                    rows += len(records)
            require(run_rows == run_contract.rows, "single sort run rows differ")
            require(run_bytes == run_contract.bytes, "single sort run bytes differ")
            require(
                run_digest.hexdigest() == run_contract.sha256,
                "single sort run SHA-256 differs",
            )
        else:
            readers = [_RunReader(contract) for contract in runs]
            heap: list[tuple[int, int, Record, bytes]] = []
            row_group = bytearray()
            try:
                for index, reader in enumerate(readers):
                    item = reader.next()
                    if item is not None:
                        record, encoded = item
                        heapq.heappush(
                            heap, (record.work_id_numeric, index, record, encoded)
                        )
                while heap:
                    _, reader_index, record, encoded = heapq.heappop(heap)
                    require(
                        previous is None or record.work_id_numeric > previous,
                        "duplicate numeric work ID detected across external sort runs",
                    )
                    previous = record.work_id_numeric
                    require(
                        record.work_id_numeric % PARTITION_COUNT == partition_id,
                        "merged record modulo mismatch",
                    )
                    digest.update(RECORD_BYTES.to_bytes(8, "little"))
                    digest.update(encoded)
                    row_group.extend(encoded)
                    rows += 1
                    if len(row_group) == 262_144 * RECORD_BYTES:
                        writer.write_table(
                            _encoded_records_to_table(row_group),
                            row_group_size=262_144,
                        )
                        row_group.clear()
                    next_item = readers[reader_index].next()
                    if next_item is not None:
                        next_record, next_encoded = next_item
                        heapq.heappush(
                            heap,
                            (
                                next_record.work_id_numeric,
                                reader_index,
                                next_record,
                                next_encoded,
                            ),
                        )
                if row_group:
                    writer.write_table(
                        _encoded_records_to_table(row_group),
                        row_group_size=262_144,
                    )
                require(
                    all(reader.verified for reader in readers),
                    "sort run was not completely consumed and verified",
                )
            finally:
                for reader in readers:
                    reader.close()
        writer.close()
        parquet_handle.flush()
        os.fsync(parquet_handle.fileno())
        parquet_handle.close()
    except BaseException:
        try:
            writer.close()
        except BaseException:
            pass
        if not parquet_handle.closed:
            parquet_handle.close()
        if temporary_path.exists() or temporary_path.is_symlink():
            temporary_path.unlink()
        raise
    require(rows == expected_rows, "external merge row count mismatch")
    temporary_path.chmod(0o600)
    decoded = _stream_validate_parquet(
        temporary_path,
        partition_id=partition_id,
        expected_rows=expected_rows,
    )
    require(
        decoded["record_sha256"] == digest.hexdigest(),
        "pre/post Parquet record digest mismatch",
    )
    output_bytes = temporary_path.stat().st_size
    require(
        output_bytes
        <= expected_rows * parquet_bytes_per_row_bound + parquet_fixed_bytes_bound,
        "partition Parquet bytes exceed the declared envelope",
    )
    schema_contract = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in ARROW_SCHEMA
    ]
    core = {
        "schema_version": checkpoint_schema,
        "task_fingerprint_sha256": task_fingerprint_sha256,
        "runtime_fingerprint": _runtime_fingerprint(),
        "scan_checkpoint_tree_sha256": scan_tree_sha256,
        "staging_shard_tree_sha256": staging_tree_sha256,
        "partition_id": partition_id,
        "partition_formula": "work_id_numeric_mod_4096",
        "output_relative_path": f"spine/part-{partition_id:04d}.parquet",
        "output_rows": rows,
        "minimum_work_id_numeric": decoded["minimum_work_id_numeric"],
        "maximum_work_id_numeric": decoded["maximum_work_id_numeric"],
        "output_schema": schema_contract,
        "output_schema_sha256": _sha256_bytes(
            _canonical_json_bytes(schema_contract)
        ),
        "pre_parquet_sorted_record_sha256": digest.hexdigest(),
        "decoded_parquet_record_sha256": decoded["record_sha256"],
        "disposition_counts": decoded["disposition_counts"],
        "w_zero_disposition_counts": decoded["w_zero_disposition_counts"],
        "publication_year_extrema": decoded["publication_year_extrema"],
        "nullability_counts": decoded["nullability_counts"],
        "output_bytes": output_bytes,
        "output_sha256": _sha256_file(temporary_path),
        "strictly_increasing_ids": True,
        "operator_paths_redacted": True,
    }
    core["checkpoint_sha256"] = _sha256_bytes(_canonical_json_bytes(core))
    return core


def _task_plan_record(task: Task) -> dict[str, Any]:
    return {
        "file_index": task.file_index,
        "relative_path": task.relative_path,
        "footer_rows": task.footer_rows,
        "source_rich_identity": task.source_rich_identity,
        "temporal_parent_task_sha256": task.temporal_task_sha256,
        "temporal_parent_checkpoint_sha256": task.temporal_checkpoint_sha256,
        "projected_columns": ["id", "publication_date", "publication_year"],
        "projected_schema": task.projected_schema,
        "projected_schema_sha256": task.projected_schema_sha256,
    }


def _own_frozen_input_plan_contract(
    *,
    scope: str,
    config: Mapping[str, Any] | None,
    config_sha256: str,
    config_identity: Mapping[str, int],
) -> dict[str, Any]:
    if scope == CLOSED_FIXTURE_SCOPE:
        tool = _frozen_file_snapshot(Path(__file__), field="fixture runner snapshot")
        return {
            "own_frozen_input_revalidation_required": False,
            "frozen_file_snapshot_is_loaded_code_attestation": False,
            "config_relative_path": None,
            "config_sha256": config_sha256,
            "config_rich_identity": dict(config_identity),
            "foundation_config_relative_path": None,
            "foundation_config_sha256": "0" * 64,
            "foundation_config_rich_identity": {},
            "foundation_synthetic_runner_relative_path": None,
            "foundation_synthetic_runner_sha256": PREDECESSOR_SHA256,
            "foundation_synthetic_runner_rich_identity": {},
            "tool_relative_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "tool_sha256": tool["sha256"],
            "tool_rich_identity": tool["rich_identity"],
        }

    require(config is not None, "production own-input config is missing")
    config_snapshot = _frozen_file_snapshot(
        DEFAULT_CONFIG, field="production default config snapshot"
    )
    require(
        config_snapshot
        == {
            "sha256": config_sha256,
            "rich_identity": dict(config_identity),
        },
        "production default config changed before plan construction",
    )

    runner = _require_mapping(config.get("runner_contract"), "runner contract")
    tool_relative = _safe_relative(
        runner.get("required_tool_path"), "production runner path"
    )
    tool_path = REPO_ROOT / tool_relative
    require(
        tool_path.resolve(strict=True) == Path(__file__).resolve(strict=True),
        "production runner path differs from executing file",
    )
    tool_snapshot = _frozen_file_snapshot(
        tool_path, field="production runner snapshot"
    )
    require(
        tool_snapshot["sha256"]
        == _require_sha256(
            runner.get("required_tool_sha256"), "production runner pinned SHA"
        ),
        "production runner changed before plan construction",
    )

    foundation_contract = _require_mapping(
        config.get("foundation_contract"), "foundation contract"
    )
    provenance = _require_mapping(
        config.get("_contract_provenance"), "contract provenance"
    )
    foundation_relative = _safe_relative(
        foundation_contract.get("config_path"), "foundation config path"
    )
    require(
        foundation_relative.as_posix()
        == provenance.get("foundation_config_relative_path"),
        "foundation config path provenance mismatch",
    )
    foundation_snapshot = _frozen_file_snapshot(
        REPO_ROOT / foundation_relative,
        field="foundation config snapshot",
    )
    require(
        foundation_snapshot
        == {
            "sha256": provenance.get("foundation_sha256"),
            "rich_identity": dict(
                _require_mapping(
                    provenance.get("foundation_rich_identity"),
                    "foundation config identity",
                )
            ),
        },
        "foundation config changed before plan construction",
    )

    predecessor_relative = _safe_relative(
        foundation_contract.get("synthetic_runner_path"),
        "foundation predecessor path",
    )
    require(
        (REPO_ROOT / predecessor_relative).resolve(strict=True)
        == PREDECESSOR_PATH.resolve(strict=True),
        "foundation predecessor path mismatch",
    )
    predecessor_snapshot = _frozen_file_snapshot(
        PREDECESSOR_PATH, field="foundation predecessor snapshot"
    )
    require(
        predecessor_snapshot["sha256"] == PREDECESSOR_SHA256,
        "foundation predecessor changed before plan construction",
    )
    return {
        "own_frozen_input_revalidation_required": True,
        "frozen_file_snapshot_is_loaded_code_attestation": False,
        "config_relative_path": DEFAULT_CONFIG.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": config_snapshot["sha256"],
        "config_rich_identity": config_snapshot["rich_identity"],
        "foundation_config_relative_path": foundation_relative.as_posix(),
        "foundation_config_sha256": foundation_snapshot["sha256"],
        "foundation_config_rich_identity": foundation_snapshot["rich_identity"],
        "foundation_synthetic_runner_relative_path": predecessor_relative.as_posix(),
        "foundation_synthetic_runner_sha256": predecessor_snapshot["sha256"],
        "foundation_synthetic_runner_rich_identity": predecessor_snapshot[
            "rich_identity"
        ],
        "tool_relative_path": tool_relative.as_posix(),
        "tool_sha256": tool_snapshot["sha256"],
        "tool_rich_identity": tool_snapshot["rich_identity"],
    }


def _revalidate_own_frozen_inputs(plan: Mapping[str, Any]) -> None:
    scope = str(plan.get("scope"))
    if scope == CLOSED_FIXTURE_SCOPE:
        require(
            plan.get("own_frozen_input_revalidation_required") is False,
            "closed fixture own-input exemption drift",
        )
        return
    require(
        scope in {FULL_SCOPE, PILOT_SCOPE}
        and plan.get("own_frozen_input_revalidation_required") is True,
        "production own-input revalidation policy drift",
    )
    require(
        plan.get("frozen_file_snapshot_is_loaded_code_attestation") is False,
        "frozen-file snapshot overclaims loaded-code attestation",
    )
    contracts = (
        (
            "config",
            "config_relative_path",
            "config_sha256",
            "config_rich_identity",
        ),
        (
            "tool",
            "tool_relative_path",
            "tool_sha256",
            "tool_rich_identity",
        ),
        (
            "foundation config",
            "foundation_config_relative_path",
            "foundation_config_sha256",
            "foundation_config_rich_identity",
        ),
        (
            "foundation predecessor",
            "foundation_synthetic_runner_relative_path",
            "foundation_synthetic_runner_sha256",
            "foundation_synthetic_runner_rich_identity",
        ),
    )
    for field, path_key, sha_key, identity_key in contracts:
        relative = _safe_relative(plan.get(path_key), f"{field} frozen path")
        observed = _frozen_file_snapshot(
            REPO_ROOT / relative, field=f"pre-receipt {field} snapshot"
        )
        require(
            observed
            == {
                "sha256": plan.get(sha_key),
                "rich_identity": dict(
                    _require_mapping(plan.get(identity_key), f"{field} identity")
                ),
            },
            f"{field} changed after plan construction",
        )


def _build_plan(
    *,
    scope: str,
    config_sha256: str,
    config_identity: Mapping[str, int],
    config: Mapping[str, Any] | None,
    parents: Mapping[str, Any],
    tasks: Sequence[Task],
    batch_size: int,
    sort_chunk_rows: int,
    resources: Mapping[str, int],
) -> dict[str, Any]:
    schemas = _schemas(scope)
    task_records = [_task_plan_record(task) for task in tasks]
    own_inputs = _own_frozen_input_plan_contract(
        scope=scope,
        config=config,
        config_sha256=config_sha256,
        config_identity=config_identity,
    )
    plan: dict[str, Any] = {
        "schema_version": schemas["plan"],
        "scope": scope,
        "status_if_complete": _status(scope),
        "production_status_before_receipt": PRODUCTION_NOT_RUN,
        **own_inputs,
        "runtime_fingerprint": _runtime_fingerprint(),
        "parents": dict(parents),
        "tasks": task_records,
        "source_tree_sha256": _sha256_bytes(
            _canonical_json_bytes(
                [
                    {
                        "relative_path": item["relative_path"],
                        "source_rich_identity": item["source_rich_identity"],
                    }
                    for item in task_records
                ]
            )
        ),
        "batch_size": batch_size,
        "partition_sort_chunk_rows": sort_chunk_rows,
        "external_sort_algorithm": "numpy_argsort_work_id_numeric_stable",
        "external_sort_record_dtype": {
            "description": "<u8,<i4,<u2,u1,u1",
            "itemsize": RECORD_DTYPE.itemsize,
            "field_offsets": {
                name: int(RECORD_DTYPE.fields[name][1])
                for name in RECORD_DTYPE.names or ()
            },
        },
        "partition_count": PARTITION_COUNT if scope != PILOT_SCOPE else 0,
        "partition_formula": (
            "work_id_numeric_mod_4096" if scope != PILOT_SCOPE else None
        ),
        "expected_receipt_artifact_count": (
            6 + 2 * len(tasks)
            if scope == PILOT_SCOPE
            else 7 + 2 * len(tasks) + 2 * PARTITION_COUNT
        ),
        "resource_requirements": dict(resources),
        "operator_paths_redacted": True,
    }
    plan["task_fingerprint_sha256"] = _sha256_bytes(_canonical_json_bytes(plan))
    return plan


def _state(
    plan: Mapping[str, Any],
    status: str,
    *,
    completed_scan_files: int,
    completed_partitions: int,
    result_fingerprint_sha256: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": _schemas(str(plan["scope"]))["state"],
        "scope": plan["scope"],
        "production_status": PRODUCTION_NOT_RUN,
        "status": status,
        "completed_scan_files": completed_scan_files,
        "total_scan_files": len(plan["tasks"]),
        "completed_partitions": completed_partitions,
        "total_partitions": (
            PARTITION_COUNT if str(plan["scope"]) != PILOT_SCOPE else 0
        ),
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "operator_paths_redacted": True,
    }
    if result_fingerprint_sha256 is not None:
        value.update(
            {
                "result_fingerprint_sha256": result_fingerprint_sha256,
                "completion_receipt_relative_path": _basenames(str(plan["scope"]))[
                    "receipt"
                ],
                "complete_only_if_receipt_exists": True,
            }
        )
    return value


def _validate_checkpoint_self_hash(checkpoint: Mapping[str, Any]) -> None:
    core = dict(checkpoint)
    observed = core.pop("checkpoint_sha256", None)
    require(
        observed == _sha256_bytes(_canonical_json_bytes(core)),
        "checkpoint self hash mismatch",
    )


def _validate_reusable_scan_checkpoint(
    output: Path,
    plan: Mapping[str, Any],
    task: Task,
    checkpoint: Mapping[str, Any],
    *,
    source_root: Path,
    validate_source_identity: bool = True,
) -> dict[str, Any]:
    checkpoint = dict(checkpoint)
    _validate_checkpoint_self_hash(checkpoint)
    require(
        checkpoint.get("schema_version")
        == _schemas(str(plan["scope"]))["scan_checkpoint"],
        "reusable scan checkpoint schema mismatch",
    )
    expected = {
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "file_index": task.file_index,
        "relative_path": task.relative_path,
        "source_rich_identity": task.source_rich_identity,
        "footer_rows": task.footer_rows,
        "temporal_parent_task_sha256": task.temporal_task_sha256,
        "temporal_parent_checkpoint_sha256": task.temporal_checkpoint_sha256,
        "projected_schema_sha256": task.projected_schema_sha256,
        "operator_paths_redacted": True,
    }
    for key, value in expected.items():
        require(checkpoint.get(key) == value, f"reusable scan binding mismatch: {key}")
    metrics = _require_mapping(
        checkpoint.get("disposition_metrics"), "reusable scan metrics"
    )
    require(set(metrics) == set(SPINE_METRIC_KEYS), "reusable scan metric key drift")
    require(
        checkpoint.get("disposition_metrics_sha256")
        == _sha256_bytes(_canonical_json_bytes(metrics)),
        "reusable scan metric hash mismatch",
    )
    if task.temporal_metrics is not None:
        require(
            metrics == {key: task.temporal_metrics[key] for key in SPINE_METRIC_KEYS},
            "reusable scan metrics differ from temporal parent",
        )
        require(
            checkpoint.get("publication_year_extrema")
            == task.publication_year_extrema,
            "reusable scan publication extrema differ from temporal parent",
        )
    rows = _require_int(checkpoint.get("staging_shard_rows"), "staging rows")
    relative = _safe_relative(
        checkpoint.get("staging_shard_relative_path"), "staging shard path"
    )
    token = _sha256_bytes(task.relative_path.encode("utf-8"))
    require(
        relative.as_posix() == f"staging/main/{token}.bin",
        "reusable staging path mismatch",
    )
    staging_path = output / relative
    info = staging_path.lstat()
    require(stat.S_ISREG(info.st_mode), "reusable staging shard is not regular")
    require(info.st_nlink == 1, "reusable staging shard is hard-linked")
    require(
        info.st_size == checkpoint.get("staging_shard_bytes") == rows * RECORD_BYTES,
        "reusable staging byte count mismatch",
    )
    require(
        _sha256_file(staging_path) == checkpoint.get("staging_shard_sha256"),
        "reusable staging SHA-256 mismatch",
    )
    require(
        _record_digest_file(staging_path, rows)
        == checkpoint.get("row_level_disposition_payload_sha256"),
        "reusable staging record digest mismatch",
    )
    if validate_source_identity:
        source_path = _absolute(source_root) / _safe_relative(
            task.relative_path, "reusable source task"
        )
        require(
            _rich_identity(source_path, field="reusable source identity")
            == task.source_rich_identity,
            "reusable scan source identity changed",
        )
    return checkpoint


def _publish_scans(
    output: Path,
    plan: Mapping[str, Any],
    tasks: Sequence[Task],
    *,
    source_root: Path,
    snapshot: date,
    resume: bool,
    pause_after_new: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    if pause_after_new is not None:
        require(pause_after_new >= 0, "scan pause count is negative")
    checkpoints: list[dict[str, Any]] = []
    new_count = 0
    for index, task in enumerate(tasks):
        token = _sha256_bytes(task.relative_path.encode("utf-8"))
        checkpoint_path = output / "checkpoints/scan/main" / f"{token}.json"
        final_staging = output / "staging/main" / f"{token}.bin"
        checkpoint_exists = checkpoint_path.exists() or checkpoint_path.is_symlink()
        staging_exists = final_staging.exists() or final_staging.is_symlink()
        require(
            checkpoint_exists == staging_exists,
            "scan checkpoint/staging pair is incomplete",
        )
        if checkpoint_exists:
            require(resume, "existing scan artifacts require resume")
            checkpoint = _validate_reusable_scan_checkpoint(
                output,
                plan,
                task,
                _read_json(checkpoint_path, "reusable scan checkpoint"),
                source_root=source_root,
            )
            checkpoints.append(checkpoint)
            continue
        checkpoint = _scan_task_to_pending(
            source_root,
            output,
            task,
            batch_size=int(plan["batch_size"]),
            snapshot=snapshot,
            checkpoint_schema=_schemas(str(plan["scope"]))["scan_checkpoint"],
            task_fingerprint_sha256=str(plan["task_fingerprint_sha256"]),
            projected_arrow_batch_bytes_bound=int(
                plan["resource_requirements"][
                    "projected_arrow_batch_bytes_upper_bound"
                ]
            ),
            projected_ipc_batch_bytes_bound=int(
                plan["resource_requirements"][
                    "projected_ipc_serialized_bytes_upper_bound"
                ]
            ),
            encoded_batch_bytes_bound=int(
                plan["resource_requirements"][
                    "encoded_record_batch_bytes_upper_bound"
                ]
            ),
            retained_payload_bytes_bound=int(
                plan["resource_requirements"][
                    "projected_batch_bytes_upper_bound"
                ]
            ),
        )
        pending = output / ".work_time_spine_tmp/staging" / f"{token}.bin"
        os.replace(pending, final_staging)
        _fsync_directory(final_staging.parent)
        _atomic_json(checkpoint_path, checkpoint)
        checkpoints.append(checkpoint)
        new_count += 1
        _atomic_json(
            output / "run_state.json",
            _state(
                plan,
                "RUNNING_SCAN",
                completed_scan_files=index + 1,
                completed_partitions=0,
            ),
        )
        if pause_after_new is not None and new_count >= pause_after_new:
            return checkpoints, True
    return checkpoints, False


def _scan_tree(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    commitments = [
        {
            "file_index": item["file_index"],
            "relative_path": item["relative_path"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "projected_payload_sha256": item["projected_payload_sha256"],
            "row_level_disposition_payload_sha256": item[
                "row_level_disposition_payload_sha256"
            ],
        }
        for item in checkpoints
    ]
    return {
        "checkpoint_count": len(commitments),
        "checkpoint_tree_sha256": _tree_digest(commitments),
        "commitments": commitments,
    }


def _staging_manifest(
    plan: Mapping[str, Any], checkpoints: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    shards = [
        {
            "relative_path": item["staging_shard_relative_path"],
            "rows": item["staging_shard_rows"],
            "bytes": item["staging_shard_bytes"],
            "sha256": item["staging_shard_sha256"],
            "record_sha256": item["row_level_disposition_payload_sha256"],
        }
        for item in checkpoints
    ]
    return {
        "schema_version": _schemas(str(plan["scope"]))["staging_manifest"],
        "scope": plan["scope"],
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "scan_checkpoint_tree_sha256": _scan_tree(checkpoints)[
            "checkpoint_tree_sha256"
        ],
        "shard_count": len(shards),
        "rows": sum(int(item["rows"]) for item in shards),
        "bytes": sum(int(item["bytes"]) for item in shards),
        "staging_shard_tree_sha256": _tree_digest(shards),
        "shards": shards,
        "operator_paths_redacted": True,
    }


def _partition_tree(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    commitments = [
        {
            "partition_id": item["partition_id"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "output_sha256": item["output_sha256"],
            "pre_parquet_sorted_record_sha256": item[
                "pre_parquet_sorted_record_sha256"
            ],
            "decoded_parquet_record_sha256": item[
                "decoded_parquet_record_sha256"
            ],
        }
        for item in checkpoints
    ]
    return {
        "checkpoint_count": len(commitments),
        "partition_checkpoint_tree_sha256": _tree_digest(commitments),
        "commitments": commitments,
    }


def _validate_reusable_partition_checkpoint(
    output: Path,
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    partition_id: int,
    expected_rows: int,
    scan_tree_sha256: str,
    staging_tree_sha256: str,
) -> dict[str, Any]:
    checkpoint = dict(checkpoint)
    _validate_checkpoint_self_hash(checkpoint)
    expected = {
        "schema_version": _schemas(str(plan["scope"]))["partition_checkpoint"],
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "scan_checkpoint_tree_sha256": scan_tree_sha256,
        "staging_shard_tree_sha256": staging_tree_sha256,
        "partition_id": partition_id,
        "partition_formula": "work_id_numeric_mod_4096",
        "output_relative_path": f"spine/part-{partition_id:04d}.parquet",
        "output_rows": expected_rows,
        "strictly_increasing_ids": True,
        "operator_paths_redacted": True,
    }
    for key, value in expected.items():
        require(
            checkpoint.get(key) == value,
            f"reusable partition binding mismatch: {key}",
        )
    output_path = output / str(checkpoint["output_relative_path"])
    info = output_path.lstat()
    require(stat.S_ISREG(info.st_mode), "reusable partition is not regular")
    require(info.st_nlink == 1, "reusable partition is hard-linked")
    require(info.st_size == checkpoint.get("output_bytes"), "partition bytes differ")
    requirements = _require_mapping(
        plan.get("resource_requirements"), "plan resource requirements"
    )
    require(
        info.st_size
        <= expected_rows * requirements["parquet_bytes_per_row_upper_bound"]
        + requirements["parquet_fixed_bytes_per_partition_upper_bound"],
        "partition Parquet bytes exceed the declared envelope",
    )
    require(
        _sha256_file(output_path) == checkpoint.get("output_sha256"),
        "partition SHA-256 differs",
    )
    decoded = _stream_validate_parquet(
        output_path, partition_id=partition_id, expected_rows=expected_rows
    )
    exact_decoded = {
        "minimum_work_id_numeric": decoded["minimum_work_id_numeric"],
        "maximum_work_id_numeric": decoded["maximum_work_id_numeric"],
        "decoded_parquet_record_sha256": decoded["record_sha256"],
        "disposition_counts": decoded["disposition_counts"],
        "w_zero_disposition_counts": decoded["w_zero_disposition_counts"],
        "publication_year_extrema": decoded["publication_year_extrema"],
        "nullability_counts": decoded["nullability_counts"],
    }
    for key, value in exact_decoded.items():
        require(
            checkpoint.get(key) == value,
            f"reusable partition decoded evidence mismatch: {key}",
        )
    require(
        checkpoint.get("pre_parquet_sorted_record_sha256")
        == decoded["record_sha256"],
        "reusable partition record bridge differs",
    )
    schema_contract = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in ARROW_SCHEMA
    ]
    require(checkpoint.get("output_schema") == schema_contract, "partition schema differs")
    require(
        checkpoint.get("output_schema_sha256")
        == _sha256_bytes(_canonical_json_bytes(schema_contract)),
        "partition schema hash differs",
    )
    return checkpoint


def _publish_partitions(
    output: Path,
    plan: Mapping[str, Any],
    spool_manifest: Mapping[str, Any],
    *,
    scan_tree_sha256: str,
    staging_tree_sha256: str,
    resume: bool,
    pause_after_new: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    require(str(plan["scope"]) != PILOT_SCOPE, "pilot cannot publish partitions")
    if pause_after_new is not None:
        require(pause_after_new >= 0, "partition pause count is negative")
    counts = _require_list(spool_manifest.get("partition_rows"), "spool rows")
    spool_contracts = _require_list(
        spool_manifest.get("partitions"), "spool partition contracts"
    )
    require(len(counts) == PARTITION_COUNT, "spool partition count mismatch")
    require(
        len(spool_contracts) == PARTITION_COUNT,
        "spool partition contract count mismatch",
    )
    checkpoints: list[dict[str, Any]] = []
    new_count = 0
    for partition_id in range(PARTITION_COUNT):
        checkpoint_path = (
            output / "checkpoints/partition" / f"part-{partition_id:04d}.json"
        )
        final_path = output / "spine" / f"part-{partition_id:04d}.parquet"
        checkpoint_exists = checkpoint_path.exists() or checkpoint_path.is_symlink()
        output_exists = final_path.exists() or final_path.is_symlink()
        require(
            checkpoint_exists == output_exists,
            "partition checkpoint/output pair is incomplete",
        )
        expected_rows = _require_int(counts[partition_id], "spool partition rows")
        if checkpoint_exists:
            require(resume, "existing partition artifacts require resume")
            checkpoint = _validate_reusable_partition_checkpoint(
                output,
                plan,
                _read_json(checkpoint_path, "reusable partition checkpoint"),
                partition_id=partition_id,
                expected_rows=expected_rows,
                scan_tree_sha256=scan_tree_sha256,
                staging_tree_sha256=staging_tree_sha256,
            )
            checkpoints.append(checkpoint)
            continue
        if pause_after_new == 0:
            return checkpoints, True
        runs, observed_rows = _write_sorted_runs(
            output,
            partition_id,
            sort_chunk_rows=int(plan["partition_sort_chunk_rows"]),
            spool_contract=_require_mapping(
                spool_contracts[partition_id], "spool partition contract"
            ),
        )
        require(observed_rows == expected_rows, "sort-run row count mismatch")
        checkpoint = _merge_runs_to_temporary_parquet(
            output,
            partition_id,
            runs,
            expected_rows,
            task_fingerprint_sha256=str(plan["task_fingerprint_sha256"]),
            checkpoint_schema=_schemas(str(plan["scope"]))[
                "partition_checkpoint"
            ],
            scan_tree_sha256=scan_tree_sha256,
            staging_tree_sha256=staging_tree_sha256,
            parquet_bytes_per_row_bound=int(
                plan["resource_requirements"]["parquet_bytes_per_row_upper_bound"]
            ),
            parquet_fixed_bytes_bound=int(
                plan["resource_requirements"][
                    "parquet_fixed_bytes_per_partition_upper_bound"
                ]
            ),
        )
        temporary_path = (
            output
            / ".work_time_spine_tmp/sorted"
            / f"part-{partition_id:04d}.parquet"
        )
        os.replace(temporary_path, final_path)
        _fsync_directory(final_path.parent)
        _atomic_json(checkpoint_path, checkpoint)
        checkpoints.append(checkpoint)
        partition_run_root = (
            output / ".work_time_spine_tmp/runs" / f"part-{partition_id:04d}"
        )
        if partition_run_root.exists():
            shutil.rmtree(partition_run_root)
        new_count += 1
        _atomic_json(
            output / "run_state.json",
            _state(
                plan,
                "RUNNING_PARTITION_SORT",
                completed_scan_files=len(plan["tasks"]),
                completed_partitions=partition_id + 1,
            ),
        )
        if pause_after_new is not None and new_count >= pause_after_new:
            return checkpoints, True
    return checkpoints, False


def _spine_manifest(
    plan: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    scan_tree_sha256: str,
    staging_tree_sha256: str,
) -> dict[str, Any]:
    partitions = [
        {
            "partition_id": item["partition_id"],
            "relative_path": item["output_relative_path"],
            "rows": item["output_rows"],
            "bytes": item["output_bytes"],
            "sha256": item["output_sha256"],
            "record_sha256": item["decoded_parquet_record_sha256"],
        }
        for item in checkpoints
    ]
    total_bytes = sum(int(item["bytes"]) for item in partitions)
    requirements = _require_mapping(
        plan.get("resource_requirements"), "plan resource requirements"
    )
    require(
        total_bytes <= requirements["parquet_bytes_upper_bound"],
        "aggregate Parquet bytes exceed the declared envelope",
    )
    return {
        "schema_version": _schemas(str(plan["scope"]))["spine_manifest"],
        "scope": plan["scope"],
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "scan_checkpoint_tree_sha256": scan_tree_sha256,
        "staging_shard_tree_sha256": staging_tree_sha256,
        "partition_count": len(partitions),
        "rows": sum(int(item["rows"]) for item in partitions),
        "bytes": total_bytes,
        "spine_data_tree_sha256": _tree_digest(partitions),
        "partitions": partitions,
        "operator_paths_redacted": True,
    }


def _merge_metrics(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = _empty_metrics()
    for checkpoint in checkpoints:
        metrics = _require_mapping(
            checkpoint.get("disposition_metrics"), "scan disposition metrics"
        )
        for key in SPINE_METRIC_KEYS:
            totals[key] += _require_int(metrics.get(key), f"scan metric {key}")
    return totals


def _aggregate_publication_year_extrema(
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, int | None]:
    minimum: int | None = None
    maximum: int | None = None
    for checkpoint in checkpoints:
        extrema = _require_mapping(
            checkpoint.get("publication_year_extrema"),
            "scan publication-year extrema",
        )
        require(
            set(extrema) == {"minimum", "maximum"},
            "scan publication-year extrema key set differs",
        )
        observed_minimum = extrema.get("minimum")
        observed_maximum = extrema.get("maximum")
        require(
            (observed_minimum is None) == (observed_maximum is None),
            "scan publication-year extrema nullability differs",
        )
        if observed_minimum is None:
            continue
        low = _require_int(observed_minimum, "scan publication-year minimum")
        high = _require_int(observed_maximum, "scan publication-year maximum")
        require(low <= high, "scan publication-year extrema are reversed")
        minimum = low if minimum is None else min(minimum, low)
        maximum = high if maximum is None else max(maximum, high)
    return {"minimum": minimum, "maximum": maximum}


def _global_evidence(
    config: Mapping[str, Any] | None,
    scope: str,
    scan_checkpoints: Sequence[Mapping[str, Any]],
    partition_checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = _merge_metrics(scan_checkpoints)
    publication_year_extrema = _aggregate_publication_year_extrema(
        scan_checkpoints
    )
    disposition_counts = {name: 0 for name in DISPOSITION_NAMES.values()}
    scan_disposition_counts = {name: 0 for name in DISPOSITION_NAMES.values()}
    for checkpoint in scan_checkpoints:
        for name, value in _require_mapping(
            checkpoint.get("disposition_counts"), "scan dispositions"
        ).items():
            scan_disposition_counts[name] += _require_int(value, "scan disposition")
    require(
        sum(scan_disposition_counts.values()) == metrics["retained_rows"],
        "scan disposition accounting differs",
    )
    rows = 0
    minimum: int | None = None
    maximum: int | None = None
    partition_nullability = {
        "publication_day_null_rows": 0,
        "publication_year_null_rows": 0,
    }
    decoded_w_zero_disposition_counts = {
        name: 0 for name in DISPOSITION_NAMES.values()
    }
    if scope == PILOT_SCOPE:
        disposition_counts.update(scan_disposition_counts)
        rows = metrics["retained_rows"]
    for checkpoint in partition_checkpoints:
        count = _require_int(checkpoint.get("output_rows"), "partition rows")
        rows += count
        if count:
            observed_minimum = _require_int(
                checkpoint.get("minimum_work_id_numeric"), "partition minimum"
            )
            observed_maximum = _require_int(
                checkpoint.get("maximum_work_id_numeric"), "partition maximum"
            )
            minimum = (
                observed_minimum if minimum is None else min(minimum, observed_minimum)
            )
            maximum = (
                observed_maximum if maximum is None else max(maximum, observed_maximum)
            )
        for name, value in _require_mapping(
            checkpoint.get("disposition_counts"), "partition dispositions"
        ).items():
            disposition_counts[name] += _require_int(value, "partition disposition")
        for name, value in _require_mapping(
            checkpoint.get("nullability_counts"), "partition nullability"
        ).items():
            partition_nullability[name] += _require_int(
                value, "partition nullability"
            )
        for name, value in _require_mapping(
            checkpoint.get("w_zero_disposition_counts"),
            "partition W0 dispositions",
        ).items():
            require(
                name in decoded_w_zero_disposition_counts,
                "partition W0 disposition key differs",
            )
            decoded_w_zero_disposition_counts[name] += _require_int(
                value, "partition W0 disposition"
            )
    require(rows == metrics["retained_rows"], "scan/partition retained rows differ")
    require(sum(disposition_counts.values()) == rows, "disposition accounting differs")
    if scope != PILOT_SCOPE:
        decoded_publication_year_extrema = _aggregate_publication_year_extrema(
            partition_checkpoints
        )
        require(
            disposition_counts == scan_disposition_counts,
            "scan/partition disposition counts differ",
        )
        require(
            decoded_publication_year_extrema == publication_year_extrema,
            "scan/decoded publication-year extrema differ",
        )
        require(
            sum(decoded_w_zero_disposition_counts.values())
            == metrics["w_zero_rows"],
            "scan/decoded W0 row count differs",
        )
        for name, count in decoded_w_zero_disposition_counts.items():
            require(
                count <= disposition_counts[name],
                "decoded W0 disposition exceeds global disposition count",
            )
        both_null_rows = metrics["publication_both_null_rows"]
        require(
            partition_nullability["publication_day_null_rows"]
            == partition_nullability["publication_year_null_rows"]
            == both_null_rows
            == metrics["publication_year_null_rows"]
            == disposition_counts["BOTH_NULL_EXCLUDED"],
            "scan/partition publication nullability differs",
        )
    if scope == FULL_SCOPE:
        require(config is not None, "full evidence requires config")
        identifiers = _require_mapping(
            config.get("identifier_accounting"), "identifier accounting"
        )
        expected = {
            "raw_rows": identifiers["raw_rows"],
            "quarantined_w_minus_1_rows": identifiers[
                "frozen_w_minus_1_quarantine_rows"
            ],
            "retained_rows": identifiers["retained_rows"],
        }
        for key, value in expected.items():
            require(metrics[key] == value, f"full accounting mismatch: {key}")
        require(metrics["null_id_rows"] == identifiers["null_id_rows"] == 0, "full null-ID anchor mismatch")
        require(
            metrics["unexpected_malformed_id_rows"]
            == identifiers["unexpected_malformed_id_rows"]
            == 0,
            "full malformed-ID anchor mismatch",
        )
        require(
            metrics["grammar_valid_id_rows"] == metrics["retained_rows"],
            "full grammar-valid ID anchor mismatch",
        )
        require(
            metrics["w_zero_rows"] == identifiers["w_zero_rows_retained"] == 1,
            "full W0 anchor mismatch",
        )
        decoded_contract = _require_mapping(
            config.get("bounded_partition_and_sort_contract", {}).get(
                "decoded_partition_conservation"
            ),
            "decoded partition conservation",
        )
        require(
            decoded_contract.get(
                "partition_checkpoint_binds_publication_year_extrema"
            )
            is True
            and decoded_contract.get(
                "partition_checkpoint_binds_w_zero_disposition_counts"
            )
            is True
            and decoded_contract.get(
                "reusable_partition_checkpoint_revalidates_decoded_conservation"
            )
            is True,
            "decoded partition conservation contract drift",
        )
        require(
            decoded_contract.get("full_global_w_zero_rows") == 1
            and decoded_contract.get(
                "w_zero_expected_disposition_category_precommitted"
            )
            is False
            and decoded_contract.get("w_zero_entity_validity") == "UNRESOLVED",
            "decoded W0 contract drift",
        )
        require(
            sum(decoded_w_zero_disposition_counts.values()) == 1
            and sum(
                count == 1
                for count in decoded_w_zero_disposition_counts.values()
            )
            == 1,
            "full decoded W0 cross-tab mismatch",
        )
        require(
            rows == identifiers["retained_distinct_numeric_ids"],
            "full distinct-ID anchor mismatch",
        )
        require(minimum == identifiers["minimum_numeric_id"], "full minimum ID mismatch")
        require(maximum == identifiers["maximum_numeric_id"], "full maximum ID mismatch")
        publication = _require_mapping(
            config.get("publication_accounting"), "publication accounting"
        )
        required_year_minimum = _require_int(
            publication.get("required_publication_year_minimum"),
            "required publication-year minimum",
        )
        required_year_maximum = _require_int(
            publication.get("required_publication_year_maximum"),
            "required publication-year maximum",
        )
        require(
            publication_year_extrema
            == {"minimum": required_year_minimum, "maximum": required_year_maximum},
            "full publication-year extrema mismatch",
        )
        require(
            decoded_contract.get("full_global_publication_year_minimum")
            == required_year_minimum
            and decoded_contract.get("full_global_publication_year_maximum")
            == required_year_maximum
            and decoded_contract.get(
                "full_report_publication_accounting_includes_year_extrema"
            )
            is True
            and decoded_contract.get(
                "full_report_identifier_accounting_includes_w_zero_disposition_cross_tab"
            )
            is True,
            "decoded full-report bridge contract drift",
        )
        for name, value in _require_mapping(
            publication.get("fixed_disposition_anchors"), "fixed dispositions"
        ).items():
            require(disposition_counts[name] == value, f"full disposition mismatch: {name}")
        split = _require_mapping(
            publication.get("snapshot_year_split"), "snapshot-year split"
        )
        for name in (
            "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT",
            "VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED",
        ):
            require(disposition_counts[name] == split[name], f"full disposition mismatch: {name}")
    identifier_accounting: dict[str, Any] = {
        "raw_rows": metrics["raw_rows"],
        "quarantined_w_minus_1_rows": metrics["quarantined_w_minus_1_rows"],
        "retained_rows": rows,
        "distinct_numeric_ids": rows,
        "duplicate_numeric_id_groups": 0,
        "duplicate_numeric_row_excess": 0,
        "minimum_numeric_id": minimum,
        "maximum_numeric_id": maximum,
        "w_zero_rows": metrics["w_zero_rows"],
        "w_zero_disposition_counts": (
            None
            if scope == PILOT_SCOPE
            else decoded_w_zero_disposition_counts
        ),
        "w_zero_disposition_category": (
            None
            if scope == PILOT_SCOPE or metrics["w_zero_rows"] != 1
            else next(
                (
                    name
                    for name, count in decoded_w_zero_disposition_counts.items()
                    if count == 1
                ),
                None,
            )
        ),
        "w_zero_disposition_cross_tab_status": (
            "NOT_EVALUATED_IN_PREFIX_PILOT"
            if scope == PILOT_SCOPE
            else "DECODED_FINAL_PARQUET_VERIFIED"
        ),
        "w_zero_entity_validity": "UNRESOLVED",
        "global_uniqueness_verified": True,
    }
    if scope == PILOT_SCOPE:
        identifier_accounting.update(
            {
                "distinct_numeric_ids": None,
                "duplicate_numeric_id_groups": None,
                "duplicate_numeric_row_excess": None,
                "minimum_numeric_id": None,
                "maximum_numeric_id": None,
                "global_uniqueness_verified": False,
                "global_uniqueness_status": "NOT_EVALUATED_IN_PREFIX_PILOT",
            }
        )
    return {
        "metrics": metrics,
        "identifier_accounting": identifier_accounting,
        "publication_accounting": {
            "publication_year_minimum": publication_year_extrema["minimum"],
            "publication_year_maximum": publication_year_extrema["maximum"],
            "publication_year_extrema_status": (
                "FULL_FROZEN_ANCHOR_VERIFIED"
                if scope == FULL_SCOPE
                else (
                    "OBSERVED_PREFIX_ONLY_NONPROMOTABLE"
                    if scope == PILOT_SCOPE
                    else "OBSERVED_CLOSED_FIXTURE_ONLY"
                )
            ),
            "w_zero_disposition_counts": (
                None
                if scope == PILOT_SCOPE
                else decoded_w_zero_disposition_counts
            ),
            "disposition_counts": disposition_counts,
            "failure_metrics": {
                key: metrics[key]
                for key in (
                    "publication_exactly_one_null_rows",
                    "publication_date_parse_error_rows",
                    "publication_year_invalid_rows",
                    "publication_date_year_mismatch_rows",
                )
            },
        },
    }


def _claim_boundary(scope: str) -> str:
    if scope == FULL_SCOPE:
        return (
            "This pass creates a deterministic current-snapshot row-level canonical "
            "work-time spine bound to the exact content and temporal-v3 parents. It "
            "does not establish historical content identity, W0 entity validity, "
            "citation chronology or observation time, author or candidate validity, "
            "G-1, G0, empirical G1, an NABM effect, or any scientific effect."
        )
    if scope == PILOT_SCOPE:
        return (
            "This prefix pilot validates bounded source scanning and staging only. "
            "It is nonpromotable and establishes neither global ID uniqueness nor "
            "full-snapshot accounting, a production component pass, or a scientific "
            "effect."
        )
    return (
        "This closed fixture validates the external-memory engine contract only. "
        "It is Python-API-only, nonproduction, nonpromotable, and establishes no "
        "OpenAlex or scientific claim."
    )


def _report_fingerprint(report: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(report))
    provenance = _require_mapping(payload.get("provenance"), "report provenance")
    provenance.pop("result_fingerprint_sha256", None)
    payload["provenance"] = provenance
    return _sha256_bytes(_canonical_json_bytes(payload))


def _build_report(
    plan: Mapping[str, Any],
    scan_checkpoints: Sequence[Mapping[str, Any]],
    partition_checkpoints: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    scope = str(plan["scope"])
    scan_tree = _scan_tree(scan_checkpoints)
    staging = _staging_manifest(plan, scan_checkpoints)
    partition_tree = _partition_tree(partition_checkpoints)
    spine = (
        _spine_manifest(
            plan,
            partition_checkpoints,
            scan_tree_sha256=str(scan_tree["checkpoint_tree_sha256"]),
            staging_tree_sha256=str(staging["staging_shard_tree_sha256"]),
        )
        if scope != PILOT_SCOPE
        else None
    )
    production_pass = scope == FULL_SCOPE
    gates = {
        "canonical_work_time_spine_component": (
            "PASS" if production_pass else "NONPROMOTABLE_VALIDATION_ONLY"
        ),
        "current_content_baseline_component": (
            "PASS" if scope != CLOSED_FIXTURE_SCOPE else "FIXTURE_ONLY"
        ),
        "main_temporal_validity_component": (
            "PASS" if scope != CLOSED_FIXTURE_SCOPE else "FIXTURE_ONLY"
        ),
        "row_level_spine_commitment": (
            "CREATED_CURRENT_SNAPSHOT" if production_pass else "NONPRODUCTION_ONLY"
        ),
        "row_level_temporal_parent_artifact_identity": (
            "NOT_AVAILABLE_TEMPORAL_V3"
        ),
        "historical_content_identity": "UNVERIFIED",
        "raw_source_id_validity": "BLOCKED",
        "w_zero_entity_validity": "UNRESOLVED",
        "citation_edge_chronology": "PENDING_NOT_IN_THIS_COMPONENT",
        "openalex_edge_observation_time": (
            "PENDING_UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT"
        ),
        "fixed_horizon_outcome_maturation": "PENDING_SEPARATE_G0_POLICY",
        "author_entity_validity": "PENDING",
        "candidate_eligibility": "PENDING",
        "cutoff_aware_t_minus_1_researcher_graph": "PENDING",
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }
    report: dict[str, Any] = {
        "schema_version": _schemas(scope)["report"],
        "status": _status(scope),
        "scope": {
            "execution_scope": scope,
            "production_status": _status(scope) if production_pass else PRODUCTION_NOT_RUN,
            "source_files": len(plan["tasks"]),
            "source_bytes": sum(
                _require_int(
                    _require_mapping(
                        task.get("source_rich_identity"), "report source identity"
                    ).get("bytes"),
                    "report source bytes",
                )
                for task in plan["tasks"]
            ),
            "raw_rows": evidence["metrics"]["raw_rows"],
            "retained_rows": evidence["identifier_accounting"]["retained_rows"],
            "partition_count": len(partition_checkpoints),
            "receipt_artifact_count": plan["expected_receipt_artifact_count"],
        },
        "parents": dict(plan["parents"]),
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "scan_checkpoint_tree_sha256": scan_tree["checkpoint_tree_sha256"],
        "staging_shard_tree_sha256": staging["staging_shard_tree_sha256"],
        "partition_checkpoint_tree_sha256": (
            partition_tree["partition_checkpoint_tree_sha256"]
            if scope != PILOT_SCOPE
            else None
        ),
        "spine_data_tree_sha256": (
            spine["spine_data_tree_sha256"] if spine is not None else None
        ),
        "identifier_accounting": evidence["identifier_accounting"],
        "publication_accounting": evidence["publication_accounting"],
        "gate_status": gates,
        "claim_boundary": _claim_boundary(scope),
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "production_component_pass": production_pass,
        "pass_g1": False,
        "provenance": {
            "config_sha256": plan["config_sha256"],
            "tool_sha256": plan["tool_sha256"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "operator_paths_redacted": True,
        },
    }
    report["provenance"]["result_fingerprint_sha256"] = _report_fingerprint(
        report
    )
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    scope = _require_mapping(report.get("scope"), "report scope")
    return "\n".join(
        [
            "# OpenAlex Canonical Work-Time Spine Audit",
            "",
            f"- Status: `{report['status']}`",
            f"- Scope: `{scope['execution_scope']}`",
            f"- Source files / retained rows: `{scope['source_files']}` / `{scope['retained_rows']}`",
            f"- Published partitions: `{scope['partition_count']}`",
            f"- Production component pass: `{str(report['production_component_pass']).lower()}`",
            "- Empirical G1 / scientific effect: `NOT_RUN / NOT_ESTABLISHED`",
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )


def _receipt_artifact_paths(plan: Mapping[str, Any]) -> set[str]:
    scope = str(plan["scope"])
    names = _basenames(scope)
    paths = {
        "audit_plan.json",
        "preflight_evidence.json",
        "run_state.json",
        "staging_manifest.json",
        names["report"],
        names["markdown"],
    }
    for task in plan["tasks"]:
        token = _sha256_bytes(str(task["relative_path"]).encode("utf-8"))
        paths.add(f"checkpoints/scan/main/{token}.json")
        paths.add(f"staging/main/{token}.bin")
    if scope != PILOT_SCOPE:
        paths.add("work_time_spine_manifest.json")
        for partition_id in range(PARTITION_COUNT):
            paths.add(f"checkpoints/partition/part-{partition_id:04d}.json")
            paths.add(f"spine/part-{partition_id:04d}.parquet")
    require(
        len(paths) == plan["expected_receipt_artifact_count"],
        "receipt artifact count differs from plan",
    )
    return paths


def _published_regular_paths(output: Path) -> set[str]:
    return _published_tree_paths(output)[0]


def _published_tree_paths(output: Path) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    directories: set[str] = set()
    for path in output.rglob("*"):
        relative = path.relative_to(output).as_posix()
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "output tree contains a symlink")
        if stat.S_ISREG(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o600, "output file mode drift")
            paths.add(relative)
        elif stat.S_ISDIR(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o700, "output directory mode drift")
            directories.add(relative)
        else:
            raise SpineAuditError("output tree contains a non-file entry")
    return paths, directories


def _final_output_directories() -> set[str]:
    return {
        "checkpoints",
        "checkpoints/scan",
        "checkpoints/scan/main",
        "checkpoints/partition",
        "staging",
        "staging/main",
        "spine",
    }


def _output_receipt(
    output: Path, plan: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    paths = _receipt_artifact_paths(plan)
    artifacts = {
        relative: _artifact_contract(output, relative) for relative in sorted(paths)
    }
    scope = str(plan["scope"])
    return {
        "schema_version": _schemas(scope)["outputs"],
        "status": _status(scope),
        "scope": scope,
        "production_status": _status(scope) if scope == FULL_SCOPE else PRODUCTION_NOT_RUN,
        "production_component_pass": scope == FULL_SCOPE,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "result_fingerprint_sha256": report["provenance"][
            "result_fingerprint_sha256"
        ],
        "scan_checkpoint_tree_sha256": report["scan_checkpoint_tree_sha256"],
        "staging_shard_tree_sha256": report["staging_shard_tree_sha256"],
        "partition_checkpoint_tree_sha256": report[
            "partition_checkpoint_tree_sha256"
        ],
        "spine_data_tree_sha256": report["spine_data_tree_sha256"],
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "completion_contract": COMPLETION_ABI,
        "operator_paths_redacted": True,
    }


def _remove_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()
        _fsync_directory(path.parent)


def _invalidate_finalization(output: Path, plan: Mapping[str, Any]) -> None:
    names = _basenames(str(plan["scope"]))
    failures: list[BaseException] = []
    try:
        _remove_if_exists(output / names["receipt"])
    except BaseException as exc:  # pragma: no cover - injected cleanup failure
        failures.append(exc)
    try:
        _atomic_json(
            output / "run_state.json",
            _state(
                plan,
                FAILURE_STATUS,
                completed_scan_files=len(plan["tasks"]),
                completed_partitions=(
                    PARTITION_COUNT if str(plan["scope"]) != PILOT_SCOPE else 0
                ),
            ),
        )
    except BaseException as exc:  # pragma: no cover - injected cleanup failure
        failures.append(exc)
    for name in (names["report"], names["markdown"]):
        try:
            _remove_if_exists(output / name)
        except BaseException as exc:  # pragma: no cover
            failures.append(exc)
    if failures:
        raise SpineAuditError("finalization invalidation was incomplete") from failures[0]


def _validate_receipt_artifacts(
    output: Path,
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    receipt_published: bool,
) -> None:
    paths = _receipt_artifact_paths(plan)
    artifacts = _require_mapping(receipt.get("artifacts"), "receipt artifacts")
    require(set(artifacts) == paths, "receipt artifact set differs")
    require(receipt.get("artifact_count") == len(paths), "receipt count differs")
    for relative in sorted(paths):
        contract = _require_mapping(artifacts[relative], "receipt artifact")
        require(
            set(contract) == {"relative_path", "bytes", "sha256"},
            "receipt artifact contract drift",
        )
        require(
            contract == _artifact_contract(output, relative),
            "receipt artifact bytes/hash mismatch",
        )
    expected_files = paths | {"audit.lock"}
    if receipt_published:
        expected_files.add(_basenames(str(plan["scope"]))["receipt"])
    observed_files, observed_directories = _published_tree_paths(output)
    require(
        observed_files == expected_files,
        "published file set contains missing or undeclared entries",
    )
    require(
        observed_directories == _final_output_directories(),
        "published directory set contains missing or undeclared entries",
    )


def _validate_published_resource_envelope(
    output: Path, plan: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    requirements = _require_mapping(
        plan.get("resource_requirements"), "plan resource requirements"
    )
    artifacts = _require_mapping(receipt.get("artifacts"), "receipt artifacts")
    control_bytes = (
        len(_V2.published_json_bytes(receipt)) + (output / "audit.lock").stat().st_size
    )
    atomic_headroom = _require_int(
        requirements.get("atomic_publish_headroom_bytes"), "atomic headroom"
    )
    require(
        (output / "audit.lock").stat().st_size <= atomic_headroom,
        "audit lock exceeds atomic headroom",
    )
    require(
        len(_V2.published_json_bytes(receipt)) <= atomic_headroom,
        "receipt candidate exceeds atomic headroom",
    )
    for relative, raw_contract in artifacts.items():
        contract = _require_mapping(raw_contract, "resource artifact")
        size = _require_int(contract.get("bytes"), "resource artifact bytes")
        require(size <= atomic_headroom, f"atomic artifact exceeds headroom: {relative}")
        if not str(relative).endswith((".bin", ".parquet")):
            control_bytes += size
    require(
        control_bytes <= requirements["control_bytes_upper_bound"],
        "published control artifacts exceed the declared envelope",
    )


def _validate_final_publication(
    output: Path,
    *,
    plan: Mapping[str, Any],
    tasks: Sequence[Task],
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source_root: Path,
    config: Mapping[str, Any] | None,
    receipt_published: bool,
    validate_current_environment: bool = True,
) -> None:
    scope = str(plan["scope"])
    names = _basenames(scope)
    _validate_output_root_policy(output)
    require(not (output / ".work_time_spine_tmp").exists(), "temporary tree remains")
    disk_plan = _read_json(output / "audit_plan.json", "final plan")
    disk_report = _read_json(output / names["report"], "final report")
    state = _read_json(output / "run_state.json", "final state")
    _validate_preflight_evidence(
        output,
        plan,
        source_root=source_root,
        validate_live_devices=validate_current_environment,
    )
    require(disk_plan == plan, "final plan differs")
    require(disk_report == report, "final report differs")
    if receipt_published:
        observed_receipt, receipt_payload, _, _ = _read_parent_json_snapshot(
            output / names["receipt"], field="final receipt"
        )
        require(
            observed_receipt == receipt
            and receipt_payload == _V2.published_json_bytes(receipt),
            "final receipt differs",
        )
    else:
        require(
            not (output / names["receipt"]).exists(),
            "receipt exists before final validation",
        )
    _validate_receipt_artifacts(
        output, plan, receipt, receipt_published=receipt_published
    )
    _validate_published_resource_envelope(output, plan, receipt)
    require(report.get("schema_version") == _schemas(scope)["report"], "report schema differs")
    require(receipt.get("schema_version") == _schemas(scope)["outputs"], "receipt schema differs")
    require(report.get("status") == receipt.get("status") == _status(scope), "status differs")
    require(receipt.get("completion_contract") == COMPLETION_ABI, "completion ABI differs")
    production_pass = scope == FULL_SCOPE
    require(report.get("production_component_pass") is production_pass, "report production-pass drift")
    require(receipt.get("production_component_pass") is production_pass, "receipt production-pass drift")
    result_fp = _report_fingerprint(report)
    require(report.get("provenance", {}).get("result_fingerprint_sha256") == result_fp, "report fingerprint differs")
    require(receipt.get("result_fingerprint_sha256") == result_fp, "receipt fingerprint differs")
    require(state.get("status") == "COMPLETE_PENDING_RECEIPT", "final state differs")
    require(state.get("result_fingerprint_sha256") == result_fp, "state result differs")
    require(
        state
        == _state(
            plan,
            "COMPLETE_PENDING_RECEIPT",
            completed_scan_files=len(tasks),
            completed_partitions=(PARTITION_COUNT if scope != PILOT_SCOPE else 0),
            result_fingerprint_sha256=result_fp,
        ),
        "final state does not reconstruct",
    )
    require(
        (output / names["markdown"]).read_text(encoding="utf-8")
        == render_markdown(report),
        "final Markdown differs",
    )

    scans: list[dict[str, Any]] = []
    for task in tasks:
        token = _sha256_bytes(task.relative_path.encode("utf-8"))
        scans.append(
            _validate_reusable_scan_checkpoint(
                output,
                plan,
                task,
                _read_json(
                    output / "checkpoints/scan/main" / f"{token}.json",
                    "final scan checkpoint",
                ),
                source_root=source_root,
                validate_source_identity=validate_current_environment,
            )
        )
    staging = _staging_manifest(plan, scans)
    require(
        _read_json(output / "staging_manifest.json", "final staging manifest")
        == staging,
        "final staging manifest differs",
    )
    partitions: list[dict[str, Any]] = []
    if scope != PILOT_SCOPE:
        scan_tree_sha = _scan_tree(scans)["checkpoint_tree_sha256"]
        staging_tree_sha = staging["staging_shard_tree_sha256"]
        manifest = _read_json(output / "work_time_spine_manifest.json", "final spine manifest")
        partition_rows = {
            int(item["partition_id"]): int(item["rows"])
            for item in _require_list(manifest.get("partitions"), "manifest partitions")
        }
        require(len(partition_rows) == PARTITION_COUNT, "final partition count differs")
        for partition_id in range(PARTITION_COUNT):
            partitions.append(
                _validate_reusable_partition_checkpoint(
                    output,
                    plan,
                    _read_json(
                        output
                        / "checkpoints/partition"
                        / f"part-{partition_id:04d}.json",
                        "final partition checkpoint",
                    ),
                    partition_id=partition_id,
                    expected_rows=partition_rows[partition_id],
                    scan_tree_sha256=str(scan_tree_sha),
                    staging_tree_sha256=str(staging_tree_sha),
                )
            )
        expected_manifest = _spine_manifest(
            plan,
            partitions,
            scan_tree_sha256=str(scan_tree_sha),
            staging_tree_sha256=str(staging_tree_sha),
        )
        require(manifest == expected_manifest, "final spine manifest differs")
    evidence = _global_evidence(config, scope, scans, partitions)
    require(
        report == _build_report(plan, scans, partitions, evidence),
        "final report does not reconstruct",
    )
    require(receipt == _output_receipt(output, plan, report), "final receipt differs")


def _validate_published_receipt(
    output: Path, plan: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    receipt_path = output / _basenames(str(plan["scope"]))["receipt"]
    observed, payload, _, _ = _read_parent_json_snapshot(
        receipt_path, field="published receipt"
    )
    require(
        observed == receipt and payload == _V2.published_json_bytes(receipt),
        "published receipt readback differs",
    )
    require(
        receipt_path.stat().st_size
        <= plan["resource_requirements"]["atomic_publish_headroom_bytes"],
        "receipt exceeds atomic publication headroom",
    )
    observed_files, observed_directories = _published_tree_paths(output)
    require(
        observed_files
        == _receipt_artifact_paths(plan) | {"audit.lock", receipt_path.name},
        "post-receipt published file set differs",
    )
    require(
        observed_directories == _final_output_directories(),
        "post-receipt published directory set differs",
    )


def _remove_temporary_tree(output: Path) -> None:
    temporary = output / ".work_time_spine_tmp"
    if temporary.exists():
        _V2._assert_no_symlink_components(temporary, field="temporary tree")
        shutil.rmtree(temporary)
        _fsync_directory(output)


def _finalize(
    output: Path,
    *,
    plan: Mapping[str, Any],
    tasks: Sequence[Task],
    scan_checkpoints: Sequence[Mapping[str, Any]],
    partition_checkpoints: Sequence[Mapping[str, Any]],
    source_root: Path,
    config: Mapping[str, Any] | None,
    parent_context: ParentContext | None,
) -> dict[str, Any]:
    scope = str(plan["scope"])
    names = _basenames(scope)
    _validate_output_root_policy(output)
    evidence = _global_evidence(config, scope, scan_checkpoints, partition_checkpoints)
    report = _build_report(plan, scan_checkpoints, partition_checkpoints, evidence)
    result_fp = str(report["provenance"]["result_fingerprint_sha256"])
    finalization_started = False
    receipt_published = False
    try:
        finalization_started = True
        _atomic_json(output / names["report"], report)
        _atomic_bytes(output / names["markdown"], render_markdown(report).encode("utf-8"))
        _atomic_json(
            output / "run_state.json",
            _state(
                plan,
                "COMPLETE_PENDING_RECEIPT",
                completed_scan_files=len(scan_checkpoints),
                completed_partitions=len(partition_checkpoints),
                result_fingerprint_sha256=result_fp,
            ),
        )
        if parent_context is not None:
            _revalidate_parent_control(parent_context, source_root)
        receipt = _output_receipt(output, plan, report)
        _validate_final_publication(
            output,
            plan=plan,
            tasks=tasks,
            report=report,
            receipt=receipt,
            source_root=source_root,
            config=config,
            receipt_published=False,
        )
        if parent_context is not None:
            _revalidate_parent_control(parent_context, source_root)
        _revalidate_own_frozen_inputs(plan)
        # The authoritative receipt is the final mutation.  Everything it
        # commits to has already been reconstructed and rehashed above.
        _atomic_json(output / names["receipt"], receipt)
        receipt_published = True
        _validate_published_receipt(output, plan, receipt)
    except BaseException as exc:
        if receipt_published and isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if finalization_started:
            try:
                _invalidate_finalization(output, plan)
            except BaseException as cleanup_exc:
                raise SpineAuditError(
                    "finalization failed and invalidation was incomplete"
                ) from cleanup_exc
        if isinstance(exc, (KeyboardInterrupt, SystemExit, SpineAuditError)):
            raise
        raise SpineAuditError("spine finalization failed") from exc
    return report


def _authorize_execution(
    *,
    execute: bool,
    authorize_full: bool,
    acknowledge_exclusive_source_scan: bool,
    max_files: int | None,
) -> str:
    if max_files is None:
        require(
            execute and authorize_full and acknowledge_exclusive_source_scan,
            "full execution requires --execute, --authorize-full, and "
            "--acknowledge-exclusive-source-scan together",
        )
        return FULL_SCOPE
    require(execute, "production pilot requires --execute")
    require(max_files > 0, "--max-files must be positive")
    require(max_files == 1, "production pilot is frozen to exactly one source file")
    require(
        not authorize_full,
        "--authorize-full is forbidden for a production-source pilot",
    )
    return PILOT_SCOPE


def _validate_production_runtime_sizes(
    batch_size: int, partition_sort_chunk_rows: int
) -> None:
    require(
        batch_size == DEFAULT_BATCH_SIZE,
        "production and pilot require the frozen source batch size",
    )
    require(
        partition_sort_chunk_rows == DEFAULT_SORT_CHUNK_ROWS,
        "production and pilot require the frozen sort chunk size",
    )


def _fixture_tasks(
    source_root: Path, relative_paths: Sequence[str] | None = None
) -> tuple[Task, ...]:
    source_root = _absolute(source_root)
    _V2._assert_no_symlink_components(source_root, field="fixture source root")
    require(
        source_root.is_dir() and not source_root.is_symlink(),
        "fixture source root is invalid",
    )
    require(
        PRODUCTION_ROOT_ID not in source_root.resolve(strict=True).parts,
        "production source root is forbidden in the closed fixture API",
    )
    if relative_paths is None:
        relatives = sorted(
            path.relative_to(source_root).as_posix()
            for path in source_root.rglob("*.parquet")
        )
    else:
        relatives = sorted(
            {_safe_relative(value, "fixture relative path").as_posix() for value in relative_paths}
        )
    require(relatives, "closed fixture contains no Parquet tasks")
    require(len(relatives) <= MAX_CLOSED_FIXTURE_TASKS, "closed fixture has too many tasks")
    tasks: list[Task] = []
    total_rows = 0
    total_bytes = 0
    for index, relative_text in enumerate(relatives):
        relative = _safe_relative(relative_text, "fixture task")
        path = source_root / relative
        _V2._assert_no_symlink_components(path, field="fixture Parquet")
        identity = _rich_identity(path, field="fixture Parquet")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                parquet = pq.ParquetFile(handle)
                rows = int(parquet.metadata.num_rows)
                projected_schema = {
                    name: {
                        "nullable": bool(parquet.schema_arrow.field(name).nullable),
                        "type": str(parquet.schema_arrow.field(name).type),
                    }
                    for name in ("id", "publication_date", "publication_year")
                }
        finally:
            os.close(descriptor)
        require(
            _rich_identity(path, field="fixture post-footer identity") == identity,
            "fixture source changed while reading footer",
        )
        total_rows += rows
        total_bytes += identity["bytes"]
        task_core = {
            "file_index": index,
            "relative_path": relative.as_posix(),
            "footer_rows": rows,
            "source_rich_identity": identity,
            "projected_schema": projected_schema,
        }
        task_sha = _sha256_bytes(_canonical_json_bytes(task_core))
        tasks.append(
            Task(
                file_index=index,
                relative_path=relative.as_posix(),
                footer_rows=rows,
                source_rich_identity=identity,
                temporal_task_sha256=task_sha,
                temporal_checkpoint_sha256=task_sha,
                projected_schema=projected_schema,
                projected_schema_sha256=_sha256_bytes(
                    _canonical_json_bytes(projected_schema)
                ),
                temporal_metrics=None,
                publication_year_extrema=None,
            )
        )
    require(total_rows <= MAX_CLOSED_FIXTURE_ROWS, "closed fixture has too many rows")
    require(total_bytes <= MAX_CLOSED_FIXTURE_BYTES, "closed fixture is too large")
    return tuple(tasks)


def _recreate_temporary_tree(output: Path) -> None:
    temporary = output / ".work_time_spine_tmp"
    if temporary.exists() or temporary.is_symlink():
        _V2._assert_no_symlink_components(temporary, field="temporary tree")
        shutil.rmtree(temporary)
    for relative in ("staging", "spool", "runs", "sorted"):
        _mkdir_private(temporary / relative)


def _assert_fresh_output(output: Path) -> None:
    regular = _published_regular_paths(output)
    require(regular == {"audit.lock"}, "fresh output directory is not empty")
    allowed_directories = {
        "checkpoints",
        "checkpoints/scan",
        "checkpoints/scan/main",
        "checkpoints/partition",
        "staging",
        "staging/main",
        "spine",
        ".work_time_spine_tmp",
        ".work_time_spine_tmp/staging",
        ".work_time_spine_tmp/spool",
        ".work_time_spine_tmp/runs",
        ".work_time_spine_tmp/sorted",
    }
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_dir()
    }
    require(observed == allowed_directories, "fresh output directory layout differs")


def _decode_canonical_control_json(path: Path, *, field: str) -> dict[str, Any]:
    return _read_parent_json_snapshot(path, field=field)[0]


def _validated_completed_pair_prefix(
    pairs: Sequence[tuple[Path, Path]], *, field: str
) -> int:
    present: list[bool] = []
    for left, right in pairs:
        left_exists = left.exists() or left.is_symlink()
        right_exists = right.exists() or right.is_symlink()
        require(left_exists == right_exists, f"{field} pair is incomplete")
        if left_exists:
            for path in (left, right):
                try:
                    info = path.lstat()
                except OSError as exc:
                    raise SpineAuditError(f"cannot inspect {field} pair") from exc
                require(
                    stat.S_ISREG(info.st_mode),
                    f"{field} pair contains a non-regular entry",
                )
        present.append(left_exists)
    prefix = 0
    while prefix < len(present) and present[prefix]:
        prefix += 1
    require(not any(present[prefix:]), f"{field} pairs are not a prefix")
    return prefix


def _classify_preflight_invocation(
    output: Path, plan: Mapping[str, Any], *, resume: bool
) -> str:
    """Select the lower resume floor only from an existing valid plan/state pair."""

    if not resume:
        return FRESH_INVOCATION
    plan_path = output / "audit_plan.json"
    state_path = output / "run_state.json"
    if not (plan_path.exists() or plan_path.is_symlink()):
        return FRESH_INVOCATION
    require(
        state_path.exists() or state_path.is_symlink(),
        "resume plan exists without a nonterminal state",
    )
    observed_plan = _decode_canonical_control_json(
        plan_path, field="resume preflight plan"
    )
    require(observed_plan == plan, "resume preflight plan differs")
    state = _decode_canonical_control_json(
        state_path, field="resume preflight state"
    )
    status = state.get("status")
    require(
        status
        in {
            "RUNNING_SCAN",
            "RUNNING_PARTITION_SORT",
            "COMPLETE_PENDING_RECEIPT",
            FAILURE_STATUS,
        },
        "resume preflight state is not a recognized nonterminal state",
    )
    completed_scans = _require_int(
        state.get("completed_scan_files"), "resume completed scan files"
    )
    completed_partitions = _require_int(
        state.get("completed_partitions"), "resume completed partitions"
    )
    total_scans = len(plan["tasks"])
    total_partitions = (
        0 if str(plan["scope"]) == PILOT_SCOPE else PARTITION_COUNT
    )
    require(
        0 <= completed_scans <= total_scans
        and 0 <= completed_partitions <= total_partitions,
        "resume preflight state progress is out of bounds",
    )
    scan_pairs = []
    for raw_task in plan["tasks"]:
        task = _require_mapping(raw_task, "resume plan task")
        token = _sha256_bytes(str(task["relative_path"]).encode("utf-8"))
        scan_pairs.append(
            (
                output / "checkpoints/scan/main" / f"{token}.json",
                output / "staging/main" / f"{token}.bin",
            )
        )
    scan_prefix = _validated_completed_pair_prefix(
        scan_pairs, field="resume scan"
    )
    partition_prefix = 0
    if total_partitions:
        partition_prefix = _validated_completed_pair_prefix(
            [
                (
                    output
                    / "checkpoints/partition"
                    / f"part-{partition_id:04d}.json",
                    output / "spine" / f"part-{partition_id:04d}.parquet",
                )
                for partition_id in range(total_partitions)
            ],
            field="resume partition",
        )
    result_fingerprint: str | None = None
    if status == "RUNNING_SCAN":
        require(
            completed_partitions == 0,
            "scan-phase resume state has completed partitions",
        )
        require(
            scan_prefix in {completed_scans, min(completed_scans + 1, total_scans)},
            "resume scan prefix differs from durable state",
        )
        require(
            partition_prefix == 0,
            "scan-phase resume has partition artifacts",
        )
    elif status == "RUNNING_PARTITION_SORT":
        require(
            completed_scans == total_scans,
            "partition-phase resume state has incomplete scans",
        )
        require(scan_prefix == total_scans, "resume scan prefix is incomplete")
        require(
            partition_prefix
            in {
                completed_partitions,
                min(completed_partitions + 1, total_partitions),
            },
            "resume partition prefix differs from durable state",
        )
    elif status == "COMPLETE_PENDING_RECEIPT":
        require(
            completed_scans == total_scans
            and completed_partitions == total_partitions,
            "pending-receipt resume state is incomplete",
        )
        require(
            scan_prefix == total_scans
            and partition_prefix == total_partitions,
            "pending-receipt artifact prefix is incomplete",
        )
        result_fingerprint = _require_sha256(
            state.get("result_fingerprint_sha256"),
            "pending-receipt result fingerprint",
        )
    else:
        require(
            completed_scans == total_scans
            and completed_partitions == total_partitions,
            "failed-finalization resume state is incomplete",
        )
        require(
            scan_prefix == total_scans
            and partition_prefix == total_partitions,
            "failed-finalization artifact prefix is incomplete",
        )
    expected = _state(
        plan,
        str(status),
        completed_scan_files=completed_scans,
        completed_partitions=completed_partitions,
        result_fingerprint_sha256=result_fingerprint,
    )
    require(state == expected, "resume preflight state differs")
    return NONTERMINAL_RESUME_INVOCATION


def _write_or_validate_plan(
    output: Path, plan: Mapping[str, Any], *, resume: bool
) -> None:
    plan_path = output / "audit_plan.json"
    exists = plan_path.exists() or plan_path.is_symlink()
    if exists:
        require(resume, "existing audit plan requires --resume")
        observed_plan, observed_bytes, _, _ = _read_parent_json_snapshot(
            plan_path, field="existing audit plan"
        )
        require(
            observed_bytes == _V2.published_json_bytes(plan),
            "resume plan bytes differ from the canonical plan bytes",
        )
        require(observed_plan == plan, "resume plan differs")
    else:
        require(
            not any(
                path.exists() or path.is_symlink()
                for path in (
                    output / "run_state.json",
                    output / "staging_manifest.json",
                    output / "work_time_spine_manifest.json",
                )
            ),
            "orphaned output artifacts require a fresh empty output root",
        )
        _assert_fresh_output(output)
        _atomic_json(plan_path, plan)


def _validate_intrinsic_terminal_surface(
    output: Path,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    scope = str(plan.get("scope"))
    require(
        scope in (FULL_SCOPE, PILOT_SCOPE, CLOSED_FIXTURE_SCOPE),
        "terminal disk scope is unknown",
    )
    plan_core = dict(plan)
    task_fingerprint = plan_core.pop("task_fingerprint_sha256", None)
    require(
        task_fingerprint == _sha256_bytes(_canonical_json_bytes(plan_core)),
        "terminal disk plan fingerprint differs",
    )
    result_fingerprint = _report_fingerprint(report)
    require(
        report.get("provenance", {}).get("result_fingerprint_sha256")
        == result_fingerprint,
        "terminal disk report fingerprint differs",
    )
    require(
        receipt == _output_receipt(output, plan, report),
        "terminal disk receipt does not reconstruct",
    )
    state = _read_parent_json_snapshot(
        output / "run_state.json", field="terminal intrinsic state"
    )[0]
    require(
        state
        == _state(
            plan,
            "COMPLETE_PENDING_RECEIPT",
            completed_scan_files=len(plan["tasks"]),
            completed_partitions=(0 if scope == PILOT_SCOPE else PARTITION_COUNT),
            result_fingerprint_sha256=result_fingerprint,
        ),
        "terminal intrinsic state differs",
    )


def _terminal_resume(
    output: Path,
    *,
    expected_plan: Mapping[str, Any],
    tasks: Sequence[Task],
    source_root: Path,
    config: Mapping[str, Any] | None,
    parent_context: ParentContext | None,
) -> dict[str, Any]:
    names = _basenames(str(expected_plan["scope"]))
    disk_plan = _read_parent_json_snapshot(
        output / "audit_plan.json", field="terminal plan"
    )[0]
    report = _read_parent_json_snapshot(
        output / names["report"], field="terminal report"
    )[0]
    receipt = _read_parent_json_snapshot(
        output / names["receipt"], field="terminal receipt"
    )[0]
    # Terminal resume is strictly read-only.  Any intrinsic mismatch is
    # reported for explicit operator repair; existing evidence is preserved.
    _validate_receipt_artifacts(
        output, disk_plan, receipt, receipt_published=True
    )
    _validate_intrinsic_terminal_surface(output, disk_plan, report, receipt)
    if disk_plan != expected_plan:
        raise TerminalCompatibilityError(
            "valid terminal output is incompatible with current inputs; no files changed"
        )
    _validate_final_publication(
        output,
        plan=disk_plan,
        tasks=tasks,
        report=report,
        receipt=receipt,
        source_root=source_root,
        config=config,
        receipt_published=True,
        validate_current_environment=False,
    )
    try:
        _validate_preflight_evidence(
            output,
            disk_plan,
            source_root=source_root,
            validate_live_devices=True,
        )
        for task in tasks:
            source_path = _absolute(source_root) / _safe_relative(
                task.relative_path, "terminal current source task"
            )
            require(
                _rich_identity(
                    source_path, field="terminal current source identity"
                )
                == task.source_rich_identity,
                "terminal current source identity changed",
            )
        if parent_context is not None:
            _revalidate_parent_control(parent_context, source_root)
    except SpineAuditError as exc:
        raise TerminalCompatibilityError(
            "valid terminal output is incompatible with current source or parents; "
            "no files changed"
        ) from exc
    return report


def _execute_pipeline(
    *,
    scope: str,
    config: Mapping[str, Any] | None,
    config_sha256: str,
    config_identity: Mapping[str, int],
    tasks: Sequence[Task],
    parents: Mapping[str, Any],
    parent_context: ParentContext | None,
    source_root: Path,
    output_dir: Path,
    snapshot: date,
    batch_size: int,
    sort_chunk_rows: int,
    resume: bool,
    enforce_distinct_device: bool,
    pause_after_new_scan_files: int | None = None,
    pause_after_new_partitions: int | None = None,
) -> dict[str, Any]:
    retained_bound = sum(
        (
            int(task.temporal_metrics["retained_rows"])
            if task.temporal_metrics is not None
            else task.footer_rows
        )
        for task in tasks
    )
    requirements = _resource_requirements(
        retained_rows=retained_bound,
        batch_size=batch_size,
        sort_chunk_rows=sort_chunk_rows,
        scope=scope,
        config=config,
    )
    plan = _build_plan(
        scope=scope,
        config_sha256=config_sha256,
        config_identity=config_identity,
        config=config,
        parents=parents,
        tasks=tasks,
        batch_size=batch_size,
        sort_chunk_rows=sort_chunk_rows,
        resources=requirements,
    )
    parent_roots = (
        tuple({path.parent for path, _ in parent_context.parent_snapshot_paths})
        if parent_context is not None
        else ()
    )
    output = _validate_output_location(
        output_dir,
        source_root=source_root,
        parent_roots=parent_roots,
    )
    receipt_path = output / _basenames(scope)["receipt"]
    if receipt_path.exists() or receipt_path.is_symlink():
        require(resume, "completed output requires --resume")
        require(output.is_dir() and not output.is_symlink(), "output root is invalid")
        with _output_lock(output, allow_create=False):
            return _terminal_resume(
                output,
                expected_plan=plan,
                tasks=tasks,
                source_root=source_root,
                config=config,
                parent_context=parent_context,
            )

    if enforce_distinct_device:
        _enforce_distinct_device(source_root, output)
    invocation_kind = _classify_preflight_invocation(
        output,
        plan,
        resume=resume,
    )
    preflight_observations = _enforce_resource_preflight(
        output,
        requirements,
        invocation_kind=invocation_kind,
        minimum_nofile=int(requirements["minimum_nofile_soft_limit"]),
    )
    output = _prepare_output(
        output,
        source_root=source_root,
        parent_roots=parent_roots,
    )
    with _output_lock(output):
        if receipt_path.exists() or receipt_path.is_symlink():
            require(resume, "completed output requires --resume")
            return _terminal_resume(
                output,
                expected_plan=plan,
                tasks=tasks,
                source_root=source_root,
                config=config,
                parent_context=parent_context,
            )
        locked_invocation_kind = _classify_preflight_invocation(
            output,
            plan,
            resume=resume,
        )
        require(
            locked_invocation_kind == invocation_kind,
            "output state changed during live preflight",
        )
        _write_or_validate_plan(output, plan, resume=resume)
        _atomic_json(
            output / "preflight_evidence.json",
            _build_preflight_evidence(
                plan,
                preflight_observations,
                invocation_kind=invocation_kind,
                source_root=source_root,
                output=output,
                distinct_device_required=enforce_distinct_device,
            ),
        )
        _recreate_temporary_tree(output)
        if invocation_kind == FRESH_INVOCATION:
            _atomic_json(
                output / "run_state.json",
                _state(
                    plan,
                    "RUNNING_SCAN",
                    completed_scan_files=0,
                    completed_partitions=0,
                ),
            )
        scans, paused = _publish_scans(
            output,
            plan,
            tasks,
            source_root=source_root,
            snapshot=snapshot,
            resume=resume,
            pause_after_new=pause_after_new_scan_files,
        )
        if paused:
            return _read_json(output / "run_state.json", "paused state")
        staging_manifest = _staging_manifest(plan, scans)
        _validate_output_root_policy(output)
        staging_manifest_path = output / "staging_manifest.json"
        if staging_manifest_path.exists() or staging_manifest_path.is_symlink():
            require(resume, "existing staging manifest requires resume")
            require(
                _read_json(staging_manifest_path, "existing staging manifest")
                == staging_manifest,
                "resume staging manifest differs",
            )
        else:
            _atomic_json(staging_manifest_path, staging_manifest)

        partitions: list[dict[str, Any]] = []
        if scope != PILOT_SCOPE:
            current_state = _decode_canonical_control_json(
                output / "run_state.json", field="partition transition state"
            )
            if current_state.get("status") == "RUNNING_SCAN":
                _atomic_json(
                    output / "run_state.json",
                    _state(
                        plan,
                        "RUNNING_PARTITION_SORT",
                        completed_scan_files=len(tasks),
                        completed_partitions=0,
                    ),
                )
            _validate_output_root_policy(output)
            spool_manifest = _build_partition_spools(output, scans)
            partitions, paused = _publish_partitions(
                output,
                plan,
                spool_manifest,
                scan_tree_sha256=str(
                    _scan_tree(scans)["checkpoint_tree_sha256"]
                ),
                staging_tree_sha256=str(
                    staging_manifest["staging_shard_tree_sha256"]
                ),
                resume=resume,
                pause_after_new=pause_after_new_partitions,
            )
            if paused:
                return _read_json(output / "run_state.json", "paused state")
            spine_manifest = _spine_manifest(
                plan,
                partitions,
                scan_tree_sha256=str(
                    _scan_tree(scans)["checkpoint_tree_sha256"]
                ),
                staging_tree_sha256=str(
                    staging_manifest["staging_shard_tree_sha256"]
                ),
            )
            spine_manifest_path = output / "work_time_spine_manifest.json"
            if spine_manifest_path.exists() or spine_manifest_path.is_symlink():
                require(resume, "existing spine manifest requires resume")
                require(
                    _read_json(spine_manifest_path, "existing spine manifest")
                    == spine_manifest,
                    "resume spine manifest differs",
                )
            else:
                _atomic_json(spine_manifest_path, spine_manifest)
        _remove_temporary_tree(output)
        _validate_output_root_policy(output)
        return _finalize(
            output,
            plan=plan,
            tasks=tasks,
            scan_checkpoints=scans,
            partition_checkpoints=partitions,
            source_root=source_root,
            config=config,
            parent_context=parent_context,
        )


def run_closed_fixture(
    *,
    source_root: Path,
    output_dir: Path,
    relative_paths: Sequence[str] | None = None,
    snapshot: date = date(2026, 3, 30),
    batch_size: int = DEFAULT_BATCH_SIZE,
    partition_sort_chunk_rows: int = DEFAULT_SORT_CHUNK_ROWS,
    resume: bool = False,
    pause_after_new_scan_files: int | None = None,
    pause_after_new_partitions: int | None = None,
) -> dict[str, Any]:
    """Run the bounded engine on a small closed fixture; never emit production status."""

    require(
        0 < batch_size <= DEFAULT_BATCH_SIZE,
        "closed-fixture batch size exceeds the bounded engine cap",
    )
    require(
        0 < partition_sort_chunk_rows <= DEFAULT_SORT_CHUNK_ROWS,
        "closed-fixture sort chunk exceeds the bounded engine cap",
    )
    tasks = _fixture_tasks(source_root, relative_paths)
    return _execute_pipeline(
        scope=CLOSED_FIXTURE_SCOPE,
        config=None,
        config_sha256="0" * 64,
        config_identity={},
        tasks=tasks,
        parents={
            "content": {"scope": "CLOSED_FIXTURE_ONLY"},
            "temporal": {"scope": "CLOSED_FIXTURE_ONLY"},
        },
        parent_context=None,
        source_root=_absolute(source_root),
        output_dir=output_dir,
        snapshot=snapshot,
        batch_size=batch_size,
        sort_chunk_rows=partition_sort_chunk_rows,
        resume=resume,
        enforce_distinct_device=False,
        pause_after_new_scan_files=pause_after_new_scan_files,
        pause_after_new_partitions=pause_after_new_partitions,
    )


def run_production(
    *,
    config_path: Path = DEFAULT_CONFIG,
    source_root: Path,
    content_receipt_path: Path,
    temporal_receipt_path: Path,
    temporal_config_path: Path,
    temporal_repo_root: Path = REPO_ROOT,
    output_dir: Path,
    execute: bool,
    authorize_full: bool,
    acknowledge_exclusive_source_scan: bool,
    max_files: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    partition_sort_chunk_rows: int = DEFAULT_SORT_CHUNK_ROWS,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a full production audit or a distinct, nonpromotable prefix pilot."""

    scope = _authorize_execution(
        execute=execute,
        authorize_full=authorize_full,
        acknowledge_exclusive_source_scan=acknowledge_exclusive_source_scan,
        max_files=max_files,
    )
    _validate_production_runtime_sizes(batch_size, partition_sort_chunk_rows)
    require(
        _absolute(temporal_repo_root) == REPO_ROOT.resolve(strict=True),
        "temporal repository root override is forbidden",
    )
    config, config_sha, config_identity = _load_config(config_path)
    source_root = _absolute(source_root)
    _V2._assert_no_symlink_components(source_root, field="production source root")
    require(
        source_root.is_dir() and not source_root.is_symlink(),
        "production source root is invalid",
    )
    parents = _validate_parent_control_plane(
        config,
        source_root=source_root,
        content_receipt_path=content_receipt_path,
        temporal_receipt_path=temporal_receipt_path,
        temporal_config_path=temporal_config_path,
        temporal_repo_root=temporal_repo_root,
    )
    tasks = parents.tasks
    if scope == PILOT_SCOPE:
        require(max_files is not None, "pilot file bound is missing")
        require(max_files < len(tasks), "pilot must be a strict task prefix")
        tasks = tasks[:max_files]
    else:
        source = _require_mapping(config.get("source"), "source")
        require(len(tasks) == source.get("expected_files"), "full task count differs")
    return _execute_pipeline(
        scope=scope,
        config=config,
        config_sha256=config_sha,
        config_identity=config_identity,
        tasks=tasks,
        parents=parents.parents,
        parent_context=parents,
        source_root=source_root,
        output_dir=output_dir,
        snapshot=date.fromisoformat(str(config["source"]["snapshot_date"])),
        batch_size=batch_size,
        sort_chunk_rows=partition_sort_chunk_rows,
        resume=resume,
        enforce_distinct_device=True,
    )


def _env_or_path(value: Path | None, variable: str, field: str) -> Path:
    if value is not None:
        return value
    raw = os.environ.get(variable)
    require(bool(raw), f"{field} is required (argument or {variable})")
    return Path(str(raw))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--content-receipt", type=Path)
    parser.add_argument("--temporal-receipt", type=Path)
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=REPO_ROOT / "experiments/configs/openalex_g1_main_temporal_validity_v2.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-full", action="store_true")
    parser.add_argument("--acknowledge-exclusive-source-scan", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--partition-sort-chunk-rows", type=int, default=DEFAULT_SORT_CHUNK_ROWS
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _authorize_execution(
            execute=args.execute,
            authorize_full=args.authorize_full,
            acknowledge_exclusive_source_scan=(
                args.acknowledge_exclusive_source_scan
            ),
            max_files=args.max_files,
        )
        _validate_production_runtime_sizes(
            args.batch_size, args.partition_sort_chunk_rows
        )
        report = run_production(
            config_path=args.config,
            source_root=_env_or_path(
                args.source_root, "OPENALEX_DERIVED_ROOT", "source root"
            ),
            content_receipt_path=_env_or_path(
                args.content_receipt,
                "OPENALEX_CONTENT_FREEZE_RECEIPT",
                "content receipt",
            ),
            temporal_receipt_path=_env_or_path(
                args.temporal_receipt,
                "OPENALEX_TEMPORAL_V3_RECEIPT",
                "temporal receipt",
            ),
            temporal_config_path=args.temporal_config,
            temporal_repo_root=REPO_ROOT,
            output_dir=args.output_dir,
            execute=args.execute,
            authorize_full=args.authorize_full,
            acknowledge_exclusive_source_scan=(
                args.acknowledge_exclusive_source_scan
            ),
            max_files=args.max_files,
            batch_size=args.batch_size,
            partition_sort_chunk_rows=args.partition_sort_chunk_rows,
            resume=args.resume,
        )
    except SpineAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
