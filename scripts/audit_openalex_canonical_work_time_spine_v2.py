#!/usr/bin/env python3
"""Temporal-v3-parent-only synthetic contract for the canonical work-time spine.

The runner exercises the frozen 16-byte staging ABI, the exact 4,096-way
numeric partition surface, Parquet writer ABI, checkpoint/resume lifecycle,
and receipt-last publication semantics.  It deliberately refuses the
repository production config and the production OpenAlex root.  Production
authorization requires a separate review and code change.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import secrets
import stat
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml"
)

DESIGN_CONFIG_SCHEMA = "openalex_g1_canonical_work_time_spine_design.v2"
PLAN_SCHEMA = "openalex_g1_canonical_work_time_spine_synthetic_plan.v2"
SCAN_CHECKPOINT_SCHEMA = (
    "openalex_g1_canonical_work_time_spine_synthetic_scan_checkpoint.v2"
)
PARTITION_CHECKPOINT_SCHEMA = (
    "openalex_g1_canonical_work_time_spine_synthetic_partition_checkpoint.v2"
)
STAGING_MANIFEST_SCHEMA = (
    "openalex_g1_canonical_work_time_spine_synthetic_staging_manifest.v2"
)
SPINE_MANIFEST_SCHEMA = "openalex_g1_canonical_work_time_spine_synthetic_manifest.v2"
REPORT_SCHEMA = "openalex_g1_canonical_work_time_spine_synthetic_report.v2"
OUTPUTS_SCHEMA = "openalex_g1_canonical_work_time_spine_synthetic_outputs.v2"
STATE_SCHEMA = "openalex_g1_canonical_work_time_spine_synthetic_state.v2"

SYNTHETIC_STATUS = (
    "PASS_SYNTHETIC_CANONICAL_WORK_TIME_SPINE_TEMPORAL_V3_PARENT_CONTRACT_ONLY"
)
SYNTHETIC_SCOPE = "SYNTHETIC_FIXTURE_ONLY"
PRODUCTION_STATUS = "PRODUCTION_NOT_RUN"
FAILURE_STATUS = "FINALIZATION_FAILED_NO_VALID_RECEIPT"
COMPLETION_ABI = "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"

PRODUCTION_ROOT_ID = "openalex_works_20260330_repairreplay_20260410_190630"
MAX_SYNTHETIC_INPUT_TASKS = 8
MAX_SYNTHETIC_FOOTER_ROWS = 10_000
MAX_SYNTHETIC_SOURCE_BYTES = 64 * 1024 * 1024
MAX_SYNTHETIC_RETAINED_ROWS = 10_000
PARTITION_COUNT = 4096
RECORD_STRUCT = struct.Struct("<QiHBB")
RECORD_BYTES = 16
NULL_DAY_SENTINEL = -(2**31)
RECORD_HASH_DOMAIN = b"openalex-work-time-records-v1"
WORK_PATTERN = re.compile(r"^https://openalex[.]org/W(0|[1-9][0-9]*)$")
W_MINUS_ONE = "https://openalex.org/W-1"

DISPOSITION_NAMES = {
    0: "BOTH_NULL_EXCLUDED",
    1: "VALID_PRE_1900_EXCLUDED",
    2: "USABLE_1900_BEFORE_SNAPSHOT_YEAR",
    3: "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT",
    4: "VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED",
    5: "VALID_POST_SNAPSHOT_YEAR_EXCLUDED",
}
SPINE_METRIC_KEYS = (
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
)
TEMPORAL_V3_METRIC_KEYS = SPINE_METRIC_KEYS + (
    "created_updated_both_null_rows",
    "created_updated_exactly_one_null_rows",
    "created_updated_both_nonnull_rows",
    "created_date_parse_error_rows",
    "updated_date_parse_error_rows",
    "created_updated_parsed_pair_rows",
    "created_lte_updated_rows",
    "created_gt_updated_rows",
    "created_after_snapshot_rows",
    "updated_after_snapshot_rows",
    "future_metadata_field_values",
)
TEMPORAL_V3_CONFIG_PATH = (
    "experiments/configs/openalex_g1_main_temporal_validity_v2.yaml"
)
TEMPORAL_V3_TOOL_PATH = "scripts/audit_openalex_main_temporal_validity_v3.py"
TEMPORAL_V3_OUTPUTS_SCHEMA = "openalex_g1_main_temporal_validity_outputs.v3"
TEMPORAL_V3_PLAN_SCHEMA = "openalex_g1_main_temporal_validity_plan.v3"
TEMPORAL_V3_CHECKPOINT_SCHEMA = "openalex_g1_main_temporal_validity_checkpoint.v3"
TEMPORAL_V3_REPORT_SCHEMA = "openalex_g1_main_temporal_validity_report.v3"
TEMPORAL_V3_STATE_SCHEMA = "openalex_g1_main_temporal_validity_state.v3"
TEMPORAL_V3_STATUS = (
    "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_"
    "MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE"
)
TEMPORAL_V3_TOOL_SHA256 = (
    "76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d"
)
PRODUCTION_TEMPORAL_V3_ANCHORS = {
    "required_config_sha256": (
        "672940908090fedd0b562dd938ad4583f5e8238830fc1ff5b613be9618f033f7"
    ),
    "required_receipt_sha256": (
        "2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe"
    ),
    "required_report_sha256": (
        "de98be73fb0365672cec7ee770c40de090ae012d45a0422faebd34c0b9d19200"
    ),
    "required_exact_task_fingerprint_sha256": (
        "5d86a797481a14e95d08ecbe333bb207ff476bf7dc49886aec1db7b5b366001d"
    ),
    "required_exact_result_fingerprint_sha256": (
        "a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5"
    ),
    "required_exact_checkpoint_tree_sha256": (
        "725cf22ce96eca153fa1c9542bd09bd628a4ea1687a2fee3e5610a8cded95392"
    ),
    "required_exact_source_tree_sha256": (
        "da3fa570a5fb49a36668dffdee008c2fea4d4b6fd5a3f7f17873df78b603b2a0"
    ),
}
TEMPORAL_V3_REPORT_FINGERPRINT_KEYS = (
    "status",
    "pilot_local_contract_clean",
    "gate_status",
    "scope",
    "metrics",
    "publication_year_extrema",
    "aggregate_regression",
    "violations",
    "accounting_contract",
    "created_updated_relation_diagnostic",
    "checkpoint_evidence",
    "source_identity_evidence",
    "content_baseline_parent",
    "task_fingerprint_sha256",
    "claim_boundary",
)
FORBIDDEN_PRIOR_TEMPORAL_SHA256 = {
    "f6eec88337fb8482400ad3a25c7b7b7099b1613d575e0c29c6c26575e8612780",
    "d208b45cc87de577726a8310a899eb66882a519bae29df5f7dbbf7833562b172",
    "b4b39c016ba4c6867a3e08366f70f6125230c2a0a18c1e8c45d40826f373f563",
    "ec59f3c649baa4a6a5333daace92a5e46f6492fad62e4c1ed42a1702394425f4",
    "11a668b8f2b10c83e2d36b3f5d2cd16f1e5f86b41ef64f58c79f10cabebc30e8",
    "b86f2496bbf7258c4e981f9a7ad8b1618fdcdc34a7385de628b133e0082a89dc",
}
FINAL_REPORT_NAMES = (
    "work_time_spine_audit.outputs.json",
    "work_time_spine_audit.json",
    "work_time_spine_audit.md",
)
ALLOWED_TOP_LEVEL = {
    "audit.lock",
    "audit_plan.json",
    "run_state.json",
    "checkpoints",
    "staging",
    "spine",
    "staging_manifest.json",
    "work_time_spine_manifest.json",
    "work_time_spine_audit.json",
    "work_time_spine_audit.md",
    "work_time_spine_audit.outputs.json",
}

ARROW_SCHEMA = pa.schema(
    [
        pa.field("work_id_numeric", pa.uint64(), nullable=False),
        pa.field("publication_day", pa.int32(), nullable=True),
        pa.field("publication_year", pa.uint16(), nullable=True),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
    ],
    metadata=None,
)
_EMPTY_PARQUET_BYTES: bytes | None = None
_EMPTY_PARQUET_VALIDATED = False
DURABLE_FSYNC = True


class SpineAuditError(RuntimeError):
    """Raised when a synthetic evidence or lifecycle invariant fails."""


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that fails closed on duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class Record:
    work_id_numeric: int
    publication_day: int | None
    publication_year: int | None
    temporal_disposition: int

    def encode(self) -> bytes:
        if self.temporal_disposition == 0:
            require(
                self.publication_day is None and self.publication_year is None,
                "both-null disposition has non-null publication values",
            )
            day = NULL_DAY_SENTINEL
            year = 0
        else:
            require(
                self.publication_day is not None and self.publication_year is not None,
                "non-null disposition has null publication values",
            )
            day = int(self.publication_day)
            year = int(self.publication_year)
            require(day != NULL_DAY_SENTINEL and year != 0, "null sentinel leaked")
        try:
            return RECORD_STRUCT.pack(
                self.work_id_numeric,
                day,
                year,
                self.temporal_disposition,
                0,
            )
        except struct.error as exc:
            raise SpineAuditError("record does not fit the frozen 16-byte ABI") from exc

    @classmethod
    def decode(cls, value: bytes) -> Record:
        require(len(value) == RECORD_BYTES, "staging record length mismatch")
        work_id, day, year, disposition, padding = RECORD_STRUCT.unpack(value)
        require(padding == 0, "staging reserved padding is nonzero")
        require(disposition in DISPOSITION_NAMES, "unknown temporal disposition")
        if disposition == 0:
            require(
                day == NULL_DAY_SENTINEL and year == 0,
                "both-null staging sentinel mismatch",
            )
            return cls(work_id, None, None, disposition)
        require(
            day != NULL_DAY_SENTINEL and year != 0,
            "nonnull staging record uses null sentinel",
        )
        return cls(work_id, day, year, disposition)


@dataclass(frozen=True)
class ScanResult:
    file_index: int
    relative_path: str
    source_rich_identity: dict[str, int]
    footer_rows: int
    projected_schema_sha256: str
    projected_payload_sha256: str
    metrics: dict[str, int]
    publication_year_extrema: dict[str, int | None]
    records: tuple[Record, ...]
    disposition_counts: dict[str, int]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpineAuditError(message)


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


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def published_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    _assert_no_symlink_components(path, field="hash input")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise SpineAuditError("cannot open hash input") from exc
    digest = hashlib.sha256()
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "hash input is not regular")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while block := handle.read(chunk_bytes):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _paths_overlap(left: Path, right: Path) -> bool:
    left = _absolute(left)
    right = _absolute(right)
    return left == right or left in right.parents or right in left.parents


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
            raise SpineAuditError(f"cannot inspect {field}") from exc
        require(not stat.S_ISLNK(info.st_mode), f"{field} contains a symlink")


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


def _regular_identity(path: Path, *, field: str) -> dict[str, int]:
    _assert_no_symlink_components(path, field=field)
    try:
        info = path.lstat()
    except OSError as exc:
        raise SpineAuditError(f"cannot stat {field}") from exc
    require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
    return _rich_identity_from_stat(info)


def _parquet_footer_identity_and_rows(
    path: Path, *, field: str
) -> tuple[dict[str, int], int]:
    """Read only a live Parquet footer and identity, never a column payload."""

    _assert_no_symlink_components(path, field=field)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise SpineAuditError(f"cannot open {field}") from exc
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
        identity = _rich_identity_from_stat(info)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                parquet = pq.ParquetFile(handle)
                footer_rows = int(parquet.metadata.num_rows)
        except Exception as exc:
            raise SpineAuditError(f"cannot read {field} footer") from exc
        require(footer_rows >= 0, f"{field} footer row count is negative")
        return identity, footer_rows
    finally:
        os.close(descriptor)


def safe_relative_path(value: Any, field: str) -> Path:
    path = Path(str(value))
    require(not path.is_absolute(), f"{field} must be relative")
    require(path.parts and path != Path("."), f"{field} is empty")
    require(".." not in path.parts, f"{field} contains '..'")
    return path


def secure_regular(root: Path, relative: Path, *, field: str) -> Path:
    root = _absolute(root)
    _assert_no_symlink_components(root, field=f"{field} root")
    require(root.is_dir() and not root.is_symlink(), f"{field} root is invalid")
    path = root / relative
    _assert_no_symlink_components(path, field=field)
    require(path.is_file() and not path.is_symlink(), f"{field} is not regular")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SpineAuditError(f"{field} escapes root") from exc
    return path


def read_json(path: Path, field: str) -> dict[str, Any]:
    _assert_no_symlink_components(path, field=field)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            return require_mapping(json.load(handle), field)
    except (OSError, json.JSONDecodeError) as exc:
        raise SpineAuditError(f"cannot read {field}") from exc


def _directory_fd(path: Path, *, field: str) -> int:
    _assert_no_symlink_components(path, field=field)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise SpineAuditError(f"cannot open {field}") from exc
    require(stat.S_ISDIR(os.fstat(descriptor).st_mode), f"{field} is not a directory")
    return descriptor


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = _absolute(path)
    descriptor = _directory_fd(path.parent, field="artifact parent")
    temporary = f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    handle: int | None = None
    try:
        try:
            current = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
            require(stat.S_ISREG(current.st_mode), "existing artifact is not regular")
        except FileNotFoundError:
            pass
        handle = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
        with os.fdopen(handle, "wb") as output:
            handle = None
            output.write(payload)
            output.flush()
            if DURABLE_FSYNC:
                os.fsync(output.fileno())
        os.rename(temporary, path.name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
        os.chmod(path.name, 0o600, dir_fd=descriptor, follow_symlinks=False)
        if DURABLE_FSYNC:
            os.fsync(descriptor)
    except OSError as exc:
        raise SpineAuditError("cannot atomically publish artifact") from exc
    finally:
        if handle is not None:
            os.close(handle)
        try:
            os.unlink(temporary, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        finally:
            os.close(descriptor)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, published_json_bytes(value))


def _remove_if_exists(path: Path) -> None:
    descriptor = _directory_fd(path.parent, field="cleanup parent")
    try:
        try:
            info = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        require(
            stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode),
            "cleanup target is not a file",
        )
        os.unlink(path.name, dir_fd=descriptor)
        if DURABLE_FSYNC:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(path.is_dir() and not path.is_symlink(), "output directory is unsafe")
    path.chmod(0o700)


@contextmanager
def output_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / "audit.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    os.chmod(lock_path, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpineAuditError("another spine audit holds the output lock") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_yaml_snapshot(path: Path) -> tuple[dict[str, Any], str, dict[str, int]]:
    _assert_no_symlink_components(path, field="spine config")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise SpineAuditError("cannot open spine config") from exc
    try:
        start = _rich_identity_from_stat(os.fstat(descriptor))
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        end = _rich_identity_from_stat(os.fstat(descriptor))
        require(start == end, "spine config changed during snapshot")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        value = yaml.load(payload, Loader=UniqueKeyLoader)
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise SpineAuditError("cannot parse spine config") from exc
    return require_mapping(value, "spine config"), sha256_bytes(payload), start


def _runtime_fingerprint() -> dict[str, str]:
    cpp = getattr(pa, "cpp_build_info", None)
    return {
        "pyarrow_version": str(pa.__version__),
        "parquet_cpp_version": str(getattr(cpp, "version", "UNKNOWN")),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
    }


def _validate_temporal_v3_parent_spec(config: Mapping[str, Any]) -> None:
    spec = require_mapping(
        config.get("parents", {}).get("main_temporal_validity"),
        "temporal parent spec",
    )
    expected = {
        "parent_version_policy": "TEMPORAL_V3_PARENT_ONLY_NO_V1_OR_V2_FALLBACK",
        "schema_version": TEMPORAL_V3_OUTPUTS_SCHEMA,
        "required_plan_schema_version": TEMPORAL_V3_PLAN_SCHEMA,
        "required_checkpoint_schema_version": TEMPORAL_V3_CHECKPOINT_SCHEMA,
        "required_report_schema_version": TEMPORAL_V3_REPORT_SCHEMA,
        "required_state_schema_version": TEMPORAL_V3_STATE_SCHEMA,
        "required_status": TEMPORAL_V3_STATUS,
        "required_config_path": TEMPORAL_V3_CONFIG_PATH,
        "required_tool_path": TEMPORAL_V3_TOOL_PATH,
        "required_tool_sha256": TEMPORAL_V3_TOOL_SHA256,
        "inherited_policy": "FROZEN_MAIN_TEMPORAL_POLICY_V3",
    }
    for key, value in expected.items():
        require(spec.get(key) == value, f"temporal-v3-only parent drift: {key}")
    require(
        tuple(spec.get("report_result_fingerprint_payload_keyset", ()))
        == TEMPORAL_V3_REPORT_FINGERPRINT_KEYS,
        "temporal v3 report fingerprint ABI drift",
    )
    sha_fields = (
        "required_config_sha256",
        "required_tool_sha256",
        "required_receipt_sha256",
        "required_report_sha256",
        "required_exact_task_fingerprint_sha256",
        "required_exact_result_fingerprint_sha256",
        "required_exact_checkpoint_tree_sha256",
        "required_exact_source_tree_sha256",
    )
    for key in sha_fields:
        value = require_sha256(spec.get(key), f"temporal parent {key}")
        require(
            value not in FORBIDDEN_PRIOR_TEMPORAL_SHA256,
            f"temporal v1/v2 parent evidence is forbidden: {key}",
        )
    forbidden = require_mapping(
        spec.get("forbidden_prior_parent_sha256"), "forbidden temporal parents"
    )
    require(
        set(forbidden.values()) == FORBIDDEN_PRIOR_TEMPORAL_SHA256,
        "forbidden temporal v1/v2 evidence set drift",
    )
    require(
        spec.get("created_updated_order_profile_is_descriptive_nonblocking") is True,
        "created/updated relation was promoted to a validity gate",
    )
    exact_counts = {
        "required_raw_rows": 492293207,
        "required_quarantined_rows": 1,
        "required_retained_rows": 492293206,
        "required_metadata_parsed_pair_rows": 492293206,
        "required_created_lte_updated_rows": 492293203,
        "required_created_gt_updated_rows": 3,
        "required_future_metadata_field_values": 0,
        "required_created_updated_null_rows": 0,
        "required_created_date_parse_error_rows": 0,
        "required_updated_date_parse_error_rows": 0,
        "required_snapshot_year_on_or_before_rows": 11509999,
        "required_snapshot_year_after_snapshot_rows": 6736,
    }
    synthetic_source = str(config.get("source", {}).get("root_id", "")).startswith(
        "synthetic-"
    )
    if synthetic_source:
        synthetic = require_mapping(
            config.get("synthetic_implementation"), "synthetic implementation"
        )
        require(
            synthetic.get(
                "temporal_parent_anchor_override_allowed_only_for_closed_synthetic_fixture"
            )
            is True,
            "synthetic temporal parent anchor override policy drift",
        )
    else:
        for key, value in PRODUCTION_TEMPORAL_V3_ANCHORS.items():
            require(
                spec.get(key) == value,
                f"production temporal v3 SHA anchor drift: {key}",
            )
        for key, value in exact_counts.items():
            require(
                spec.get(key) == value, f"production temporal v3 anchor drift: {key}"
            )


def _validate_config(config: Mapping[str, Any]) -> None:
    require(config.get("schema_version") == DESIGN_CONFIG_SCHEMA, "config schema drift")
    require(
        "temporal_v3_parent_only" in str(config.get("label", "")),
        "temporal-v3-only config label drift",
    )
    _validate_temporal_v3_parent_spec(config)
    lifecycle = require_mapping(config.get("lifecycle"), "lifecycle")
    require(
        lifecycle.get("design_status")
        == "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN",
        "synthetic implementation lifecycle drift",
    )
    require(
        lifecycle.get("production_status") == PRODUCTION_STATUS,
        "production status drift",
    )
    require(
        lifecycle.get("production_executable") is False,
        "production execution is forbidden",
    )
    require(
        lifecycle.get("synthetic_executable") is True, "synthetic runner is not enabled"
    )
    synthetic = require_mapping(
        config.get("synthetic_implementation"), "synthetic implementation"
    )
    require(synthetic.get("scope") == SYNTHETIC_SCOPE, "synthetic scope drift")
    require(synthetic.get("status") == SYNTHETIC_STATUS, "synthetic status drift")
    require(
        synthetic.get("production_authorized") is False,
        "production authorization drift",
    )
    require(
        synthetic.get("partition_count_override_allowed") is False,
        "partition-count override is forbidden",
    )
    require(
        synthetic.get("cli_source_argument") == "--synthetic-source-root",
        "synthetic source CLI drift",
    )
    require(
        synthetic.get("generic_source_root_argument_rejected") is True,
        "generic source CLI boundary drift",
    )
    require(
        synthetic.get("resolved_source_path_must_not_contain_production_root_id")
        is True,
        "production source path boundary drift",
    )
    frozen_caps = {
        "hard_max_input_tasks": MAX_SYNTHETIC_INPUT_TASKS,
        "hard_max_footer_rows": MAX_SYNTHETIC_FOOTER_ROWS,
        "hard_max_source_bytes": MAX_SYNTHETIC_SOURCE_BYTES,
        "hard_max_retained_rows": MAX_SYNTHETIC_RETAINED_ROWS,
    }
    for key, expected in frozen_caps.items():
        require(synthetic.get(key) == expected, f"synthetic hard cap drift: {key}")
    require(
        synthetic.get(
            "hard_caps_validated_from_live_parquet_footers_before_any_column_payload_scan"
        )
        is True,
        "synthetic cap ordering drift",
    )
    require(
        synthetic.get(
            "temporal_parent_anchor_override_allowed_only_for_closed_synthetic_fixture"
        )
        is True,
        "synthetic temporal parent anchor override policy drift",
    )

    table = require_mapping(config.get("table"), "table")
    identifiers = require_mapping(
        config.get("identifier_accounting"), "identifier accounting"
    )
    declared = {
        "input tasks": require_int(table.get("expected_files"), "expected files"),
        "footer rows": require_int(table.get("expected_raw_rows"), "expected rows"),
        "source bytes": require_int(
            table.get("expected_source_bytes"), "expected source bytes"
        ),
        "retained rows": require_int(
            identifiers.get("retained_rows"), "expected retained rows"
        ),
    }
    caps = {
        "input tasks": MAX_SYNTHETIC_INPUT_TASKS,
        "footer rows": MAX_SYNTHETIC_FOOTER_ROWS,
        "source bytes": MAX_SYNTHETIC_SOURCE_BYTES,
        "retained rows": MAX_SYNTHETIC_RETAINED_ROWS,
    }
    for key, observed in declared.items():
        require(0 <= observed <= caps[key], f"synthetic {key} exceeds hard cap")

    partition = require_mapping(config.get("partition_contract"), "partition contract")
    require(
        partition.get("partition_count") == PARTITION_COUNT, "partition count drift"
    )
    require(
        partition.get("formula") == "work_id_numeric_mod_4096",
        "partition formula drift",
    )
    require(
        partition.get("exact_file_count") == PARTITION_COUNT,
        "partition file count drift",
    )
    writer = require_mapping(partition.get("writer_abi"), "writer ABI")
    expected_writer = {
        "parquet_format_version": "2.6",
        "compression": "NONE",
        "dictionary_encoding": False,
        "write_statistics": False,
        "row_group_rows": 262144,
        "write_batch_size": 262144,
        "data_page_version": "1.0",
        "write_page_index": False,
        "write_page_checksum": False,
        "use_byte_stream_split": False,
        "store_arrow_schema": False,
        "sorting_columns": [],
        "combine_chunks_before_write": True,
        "input_arrow_schema_metadata": None,
    }
    for key, value in expected_writer.items():
        require(writer.get(key) == value, f"writer ABI drift: {key}")
    require(
        writer.get("resume_requires_exact_runtime_fingerprint") is True,
        "runtime resume binding drift",
    )

    staging = require_mapping(config.get("staging_record_abi"), "staging ABI")
    require(staging.get("record_bytes") == RECORD_BYTES, "record byte width drift")
    require(
        staging.get("both_null_publication_day_sentinel") == NULL_DAY_SENTINEL,
        "day sentinel drift",
    )
    require(
        staging.get("both_null_publication_year_sentinel") == 0, "year sentinel drift"
    )
    require(staging.get("reserved_padding_required_value") == 0, "padding drift")
    require(
        staging.get("canonical_record_hash", {}).get("domain_utf8")
        == RECORD_HASH_DOMAIN.decode("utf-8"),
        "record digest domain drift",
    )

    classifier = require_mapping(config.get("disposition_classifier"), "classifier")
    require(
        classifier.get("snapshot_date")
        == config.get("source", {}).get("snapshot_date"),
        "snapshot date drift",
    )
    require(
        tuple(classifier.get("temporal_parent_per_file_metric_keyset", []))
        == TEMPORAL_V3_METRIC_KEYS,
        "temporal metric key ABI drift",
    )
    require(
        tuple(classifier.get("spine_recomputed_per_file_metric_keyset", []))
        == SPINE_METRIC_KEYS,
        "spine metric key ABI drift",
    )
    require(
        classifier.get("require_temporal_checkpoint_metric_keyset_exactly_v3") is True,
        "temporal checkpoint metric keyset closure drift",
    )
    require(
        classifier.get(
            "require_no_staging_or_partition_publication_after_any_rejected_row"
        )
        is True,
        "fail-before-staging drift",
    )

    completion = require_mapping(config.get("completion_contract"), "completion")
    require(
        completion.get("receipt_exact_artifact_count_full") == 9184,
        "production artifact anchor drift",
    )
    require(
        9184 == 6 + 2 * 493 + 2 * PARTITION_COUNT, "production artifact formula drift"
    )
    require(
        completion.get("receipt_completion_contract") == COMPLETION_ABI,
        "completion ABI drift",
    )
    require(
        completion.get("failure_state_status") == FAILURE_STATUS, "failure state drift"
    )
    require(completion.get("receipt_written_last") is True, "receipt-last drift")


def _validate_synthetic_parent_caps(
    config: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    source_root: Path | None,
) -> dict[str, int]:
    """Fail closed on metadata plus live footers before any column payload scan."""

    synthetic = require_mapping(
        config.get("synthetic_implementation"), "synthetic implementation"
    )
    task_count = len(tasks)
    footer_rows = sum(
        require_int(task.get("footer_rows"), "temporal task footer rows")
        for task in tasks
    )
    table = require_mapping(plan.get("table"), "temporal plan table")
    selected_bytes = require_int(table.get("selected_bytes"), "temporal selected bytes")
    task_source_bytes = sum(
        require_int(
            require_mapping(
                task.get("source_rich_identity"), "temporal source identity"
            ).get("bytes"),
            "temporal task source bytes",
        )
        for task in tasks
    )
    require(
        selected_bytes == task_source_bytes,
        "temporal selected/task source-byte metadata mismatch",
    )
    metrics = require_mapping(report.get("metrics"), "temporal report metrics")
    raw_rows = require_int(metrics.get("raw_rows"), "temporal raw rows")
    retained_rows = require_int(metrics.get("retained_rows"), "temporal retained rows")
    require(raw_rows == footer_rows, "temporal footer/raw-row metadata mismatch")
    require(
        0 <= retained_rows <= footer_rows,
        "temporal retained-row metadata exceeds raw rows",
    )

    observed = {
        "input_tasks": task_count,
        "footer_rows": footer_rows,
        "source_bytes": task_source_bytes,
        "retained_rows": retained_rows,
    }
    configured_caps = {
        "input_tasks": require_int(
            synthetic.get("hard_max_input_tasks"), "task hard cap"
        ),
        "footer_rows": require_int(
            synthetic.get("hard_max_footer_rows"), "footer-row hard cap"
        ),
        "source_bytes": require_int(
            synthetic.get("hard_max_source_bytes"), "source-byte hard cap"
        ),
        "retained_rows": require_int(
            synthetic.get("hard_max_retained_rows"), "retained-row hard cap"
        ),
    }
    for key, value in observed.items():
        require(0 <= value <= configured_caps[key], f"synthetic {key} exceeds hard cap")
    if source_root is not None:
        actual_source_bytes = 0
        actual_footer_rows = 0
        for task in tasks:
            relative = safe_relative_path(
                task.get("relative_path"), "capped temporal source task"
            )
            expected_identity = require_mapping(
                task.get("source_rich_identity"), "temporal source identity"
            )
            actual_identity, task_footer_rows = _parquet_footer_identity_and_rows(
                _absolute(source_root) / relative,
                field="capped synthetic source file",
            )
            require(
                actual_identity == expected_identity,
                "synthetic source identity differs before payload open",
            )
            actual_source_bytes += actual_identity["bytes"]
            actual_footer_rows += task_footer_rows
            require(
                actual_footer_rows <= configured_caps["footer_rows"],
                "synthetic live footer_rows exceeds hard cap",
            )
        require(
            actual_source_bytes == task_source_bytes,
            "actual/task synthetic source-byte mismatch",
        )
        require(
            actual_footer_rows == footer_rows,
            "live/declared synthetic footer-row mismatch",
        )
        observed["footer_rows"] = actual_footer_rows
    return observed


def _load_temporal_module(tool_path: Path, expected_sha256: str) -> ModuleType:
    require(sha256_file(tool_path) == expected_sha256, "temporal tool SHA-256 mismatch")
    name = f"_spine_temporal_parent_{expected_sha256[:12]}"
    spec = importlib.util.spec_from_file_location(name, tool_path)
    require(spec is not None and spec.loader is not None, "cannot load temporal tool")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _content_contract_adapter(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = require_mapping(
        config.get("parents", {}).get("current_content_baseline"),
        "content parent",
    )
    return {
        "expected_basename": parent.get("required_receipt_basename"),
        "schema_version": parent.get("schema_version"),
        "required_status": parent.get("required_status"),
        "required_plan_sha256": parent.get("required_plan_sha256"),
        "required_task_fingerprint_sha256": parent.get(
            "required_task_fingerprint_sha256"
        ),
        "required_file_count": parent.get("required_file_count"),
        "required_bytes": parent.get("required_source_bytes"),
        "required_receipt_artifacts": [
            "audit_plan.json",
            "content_freeze_audit.json",
            "content_freeze_audit.md",
        ],
        "separately_bound_run_state": "run_state.json",
        "required_run_state": parent.get("required_run_state"),
        "full_scope_required": True,
        "pilot_may_omit_only_because_unpromotable": True,
    }


def _temporal_parent_config_for_content(
    temporal_config: Mapping[str, Any],
    spine_config: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(temporal_config)
    parents = dict(require_mapping(value.get("parents"), "temporal parents"))
    parents["current_content_baseline"] = _content_contract_adapter(spine_config)
    value["parents"] = parents
    return value


def _artifact_contract(
    output_root: Path,
    relative_text: str,
) -> dict[str, Any]:
    relative = safe_relative_path(relative_text, "artifact path")
    path = secure_regular(output_root, relative, field="artifact")
    identity = _regular_identity(path, field="artifact")
    return {
        "relative_path": relative.as_posix(),
        "bytes": identity["bytes"],
        "sha256": sha256_file(path),
    }


def _validate_temporal_parent(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    temporal_receipt_path: Path,
    temporal_config_path: Path,
    temporal_repo_root: Path,
    content_receipt_path: Path,
    revalidate_source_payload: bool,
) -> dict[str, Any]:
    """Validate the exact temporal-v3-only parent ABI.

    The full path delegates final source/checkpoint recomputation to the frozen
    temporal implementation after independently checking its exact receipt
    surface.  The terminal path performs the same receipt, fingerprint, and
    parent-output checks without opening an OpenAlex source payload.
    """

    spec = require_mapping(
        config.get("parents", {}).get("main_temporal_validity"),
        "temporal parent spec",
    )
    temporal_receipt_path = _absolute(temporal_receipt_path)
    require(
        temporal_receipt_path.name == spec.get("required_receipt_basename"),
        "temporal receipt basename mismatch",
    )
    output_root = temporal_receipt_path.parent
    require(
        output_root.is_dir()
        and not output_root.is_symlink()
        and stat.S_IMODE(output_root.lstat().st_mode) == 0o700,
        "temporal output root is not private",
    )
    require(
        sha256_file(temporal_receipt_path)
        == require_sha256(spec.get("required_receipt_sha256"), "temporal receipt SHA"),
        "temporal v3 receipt SHA-256 mismatch",
    )

    temporal_config_path = _absolute(temporal_config_path)
    temporal_config_payload = temporal_config_path.read_bytes()
    temporal_config_sha = sha256_bytes(temporal_config_payload)
    require(
        temporal_config_sha == spec.get("required_config_sha256"),
        "temporal config SHA-256 mismatch",
    )
    try:
        temporal_config = require_mapping(
            yaml.load(temporal_config_payload, Loader=UniqueKeyLoader),
            "temporal config",
        )
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise SpineAuditError("cannot parse temporal config") from exc

    tool_path = secure_regular(
        REPO_ROOT,
        safe_relative_path(spec.get("required_tool_path"), "temporal tool path"),
        field="temporal tool",
    )
    tool_sha = require_sha256(spec.get("required_tool_sha256"), "temporal tool SHA")
    module = _load_temporal_module(tool_path, tool_sha)
    module.REPO_ROOT = _absolute(temporal_repo_root)
    module.FROZEN_CONFIG_SHA256 = temporal_config_sha

    receipt = read_json(temporal_receipt_path, "temporal receipt")
    require(
        list(receipt) == list(spec.get("exact_receipt_keyset", []))
        or set(receipt) == set(spec.get("exact_receipt_keyset", [])),
        "temporal receipt key set mismatch",
    )
    require(
        receipt.get("schema_version") == spec.get("schema_version"),
        "temporal receipt schema mismatch",
    )
    require(
        receipt.get("status") == spec.get("required_status"),
        "temporal receipt status mismatch",
    )
    require(
        receipt.get("completion_contract") == spec.get("required_completion_contract"),
        "temporal completion ABI mismatch",
    )
    require(
        receipt.get("operator_paths_redacted") is True,
        "temporal receipt path redaction mismatch",
    )

    artifacts = require_mapping(receipt.get("artifacts"), "temporal artifacts")
    plan = read_json(output_root / "audit_plan.json", "temporal plan")
    tasks = [
        require_mapping(item, "temporal task")
        for item in require_list(plan.get("tasks"), "temporal tasks")
    ]
    require(
        len(tasks) == spec.get("required_full_task_count"),
        "temporal task count mismatch",
    )
    expected_artifacts = set(
        str(item) for item in spec.get("exact_fixed_receipt_artifact_paths", [])
    )
    expected_artifacts.update(
        f"checkpoints/main/{sha256_bytes(str(task['relative_path']).encode('utf-8'))}.json"
        for task in tasks
    )
    require(
        len(expected_artifacts) == spec.get("exact_total_receipt_artifact_count"),
        "temporal expected artifact count mismatch",
    )
    require(set(artifacts) == expected_artifacts, "temporal artifact set mismatch")
    for relative_text in sorted(expected_artifacts):
        contract = require_mapping(
            artifacts[relative_text], "temporal artifact contract"
        )
        require(
            set(contract) == {"relative_path", "bytes", "sha256"},
            "temporal artifact contract key drift",
        )
        require(
            contract.get("relative_path") == relative_text,
            "temporal artifact key/path mismatch",
        )
        observed = _artifact_contract(output_root, relative_text)
        require(observed == contract, "temporal artifact bytes or hash mismatch")

    report = read_json(
        output_root / "main_temporal_validity_audit.json", "temporal report"
    )
    require(
        sha256_file(output_root / "main_temporal_validity_audit.json")
        == require_sha256(spec.get("required_report_sha256"), "temporal report SHA"),
        "temporal v3 report SHA-256 mismatch",
    )
    state = read_json(output_root / "run_state.json", "temporal state")
    synthetic_cap_evidence = _validate_synthetic_parent_caps(
        config,
        plan=plan,
        tasks=tasks,
        report=report,
        source_root=source_root if revalidate_source_payload else None,
    )
    require(
        plan.get("schema_version") == spec.get("required_plan_schema_version"),
        "temporal plan schema mismatch",
    )
    require(
        report.get("schema_version") == spec.get("required_report_schema_version"),
        "temporal report schema mismatch",
    )
    require(
        state.get("schema_version") == spec.get("required_state_schema_version"),
        "temporal state schema mismatch",
    )
    require(
        plan.get("full_scope") is True and plan.get("pilot_mode") is None,
        "temporal parent is not full scope",
    )
    require(
        state.get("status") == spec.get("required_state_status"),
        "temporal state completion mismatch",
    )
    require(
        state.get("completion_receipt_relative_path") == temporal_receipt_path.name,
        "temporal state receipt path mismatch",
    )
    require(
        state.get("complete_only_if_receipt_exists") is True,
        "temporal state receipt requirement mismatch",
    )
    require(
        report.get("status") == spec.get("required_status"),
        "temporal report status mismatch",
    )
    require(
        report.get("scope", {}).get("selected_files") == len(tasks),
        "temporal report file count mismatch",
    )
    require(
        report.get("scope", {}).get("selected_footer_rows")
        == spec.get("required_raw_rows"),
        "temporal report row count mismatch",
    )
    require(
        plan.get("table", {}).get("selected_bytes")
        == spec.get("required_source_bytes"),
        "temporal source byte count mismatch",
    )
    require(
        plan.get("config_sha256") == temporal_config_sha,
        "temporal plan config mismatch",
    )
    require(plan.get("tool_sha256") == tool_sha, "temporal plan tool mismatch")

    plan_payload = dict(plan)
    task_fingerprint = plan_payload.pop("task_fingerprint_sha256", None)
    require(
        task_fingerprint == sha256_bytes(canonical_json_bytes(plan_payload)),
        "temporal task fingerprint mismatch",
    )
    require(
        receipt.get("task_fingerprint_sha256") == task_fingerprint,
        "temporal receipt task mismatch",
    )
    require(
        task_fingerprint
        == require_sha256(
            spec.get("required_exact_task_fingerprint_sha256"),
            "temporal exact task fingerprint",
        ),
        "temporal v3 exact task fingerprint mismatch",
    )
    report_fingerprint = module._report_fingerprint(report)
    require(
        report.get("provenance", {}).get("result_fingerprint_sha256")
        == report_fingerprint,
        "temporal report fingerprint mismatch",
    )
    require(
        receipt.get("result_fingerprint_sha256") == report_fingerprint,
        "temporal receipt result mismatch",
    )
    require(
        state.get("result_fingerprint_sha256") == report_fingerprint,
        "temporal state result mismatch",
    )
    require(
        report_fingerprint
        == require_sha256(
            spec.get("required_exact_result_fingerprint_sha256"),
            "temporal exact result fingerprint",
        ),
        "temporal v3 exact result fingerprint mismatch",
    )

    checkpoints: list[dict[str, Any]] = []
    checkpoints_by_path: dict[str, dict[str, Any]] = {}
    for task in tasks:
        relative = str(task["relative_path"])
        checkpoint_path = (
            output_root
            / "checkpoints/main"
            / f"{sha256_bytes(relative.encode('utf-8'))}.json"
        )
        checkpoint = read_json(checkpoint_path, "temporal checkpoint")
        require(
            checkpoint.get("schema_version")
            == spec.get("required_checkpoint_schema_version"),
            "temporal checkpoint schema mismatch",
        )
        checkpoint_payload = dict(checkpoint)
        checkpoint_sha = checkpoint_payload.pop("checkpoint_sha256", None)
        require(
            checkpoint_sha == sha256_bytes(canonical_json_bytes(checkpoint_payload)),
            "temporal checkpoint self hash mismatch",
        )
        require(
            checkpoint.get("task_fingerprint_sha256") == task_fingerprint,
            "temporal checkpoint task fingerprint mismatch",
        )
        require(
            checkpoint.get("task_sha256") == task.get("task_sha256"),
            "temporal checkpoint task mismatch",
        )
        require(
            checkpoint.get("relative_path") == relative,
            "temporal checkpoint path mismatch",
        )
        require(
            checkpoint.get("metrics_sha256")
            == sha256_bytes(canonical_json_bytes(checkpoint.get("metrics"))),
            "temporal checkpoint metrics hash mismatch",
        )
        require(
            set(require_mapping(checkpoint.get("metrics"), "temporal metrics"))
            == set(TEMPORAL_V3_METRIC_KEYS),
            "temporal-v3 checkpoint metric keyset mismatch",
        )
        checkpoints.append(checkpoint)
        checkpoints_by_path[relative] = checkpoint
    checkpoint_tree = module._checkpoint_tree(checkpoints)
    require(
        checkpoint_tree == report.get("checkpoint_evidence"),
        "temporal checkpoint tree mismatch",
    )
    require(
        checkpoint_tree.get("checkpoint_tree_sha256")
        == require_sha256(
            spec.get("required_exact_checkpoint_tree_sha256"),
            "temporal exact checkpoint tree",
        ),
        "temporal v3 checkpoint tree anchor mismatch",
    )
    source_tree = module._source_tree(tasks)
    require(
        source_tree == plan.get("source_tree_start"),
        "temporal source tree mismatch",
    )
    require(
        source_tree.get("tree_sha256")
        == require_sha256(
            spec.get("required_exact_source_tree_sha256"),
            "temporal exact source tree",
        ),
        "temporal v3 source tree anchor mismatch",
    )

    metrics = require_mapping(report.get("metrics"), "temporal report metrics")
    require(
        set(metrics) == set(TEMPORAL_V3_METRIC_KEYS),
        "temporal-v3 report metric keyset mismatch",
    )
    expected_metrics = {
        "raw_rows": spec.get("required_raw_rows"),
        "quarantined_w_minus_1_rows": spec.get("required_quarantined_rows"),
        "retained_rows": spec.get("required_retained_rows"),
        "created_updated_parsed_pair_rows": spec.get(
            "required_metadata_parsed_pair_rows"
        ),
        "created_lte_updated_rows": spec.get("required_created_lte_updated_rows"),
        "created_gt_updated_rows": spec.get("required_created_gt_updated_rows"),
        "future_metadata_field_values": spec.get(
            "required_future_metadata_field_values"
        ),
        "created_date_parse_error_rows": spec.get(
            "required_created_date_parse_error_rows"
        ),
        "updated_date_parse_error_rows": spec.get(
            "required_updated_date_parse_error_rows"
        ),
        "publication_snapshot_year_on_or_before_snapshot_rows": spec.get(
            "required_snapshot_year_on_or_before_rows"
        ),
        "publication_snapshot_year_after_snapshot_rows": spec.get(
            "required_snapshot_year_after_snapshot_rows"
        ),
    }
    for key, value in expected_metrics.items():
        require(metrics.get(key) == value, f"temporal v3 exact metric mismatch: {key}")
    require(
        metrics["created_updated_both_null_rows"]
        + metrics["created_updated_exactly_one_null_rows"]
        == spec.get("required_created_updated_null_rows"),
        "temporal v3 created/updated null anchor mismatch",
    )
    relation = require_mapping(
        report.get("created_updated_relation_diagnostic"),
        "temporal created/updated relation diagnostic",
    )
    require(
        relation.get("validity_gate") is False
        and relation.get("interpretation")
        == "DESCRIPTIVE_ONLY_NO_VENDOR_ORDER_INVARIANT"
        and relation.get("partition_exact") is True,
        "temporal v3 relation diagnostic contract mismatch",
    )

    adapted_temporal_config = _temporal_parent_config_for_content(
        temporal_config,
        config,
    )
    content_context = module._validate_content_freeze_parent(
        adapted_temporal_config,
        _absolute(content_receipt_path),
        required=True,
    )
    module._bind_current_source_to_content_baseline(plan, content_context)
    require(
        content_context["commitment"] == report.get("content_baseline_parent"),
        "temporal embedded content commitment mismatch",
    )

    # The frozen temporal validator checks its exact final tree, all checkpoint
    # payloads, current five-column projections, source tree, and parent bytes.
    if revalidate_source_payload:
        module.validate_final_publication(
            output_root,
            report=report,
            outputs=receipt,
            config=adapted_temporal_config,
            plan=plan,
            source_root=_absolute(source_root),
            config_path=temporal_config_path,
            content_context=content_context,
        )
    else:
        module._validate_output_tree(output_root, final=True)
        module._revalidate_frozen_inputs(
            config_path=temporal_config_path,
            plan=plan,
            content_context=content_context,
        )

    content_runtime = require_mapping(content_context.get("runtime"), "content runtime")
    content_tasks = require_mapping(
        content_runtime.get("main_source_tasks"), "content main tasks"
    )
    temporal_tasks = {str(task["relative_path"]): task for task in tasks}
    require(len(temporal_tasks) == len(tasks), "temporal task paths are duplicated")
    require(
        set(content_tasks) == set(temporal_tasks), "content/temporal task sets differ"
    )
    six = ("device", "inode", "mode", "bytes", "mtime_ns", "ctime_ns")
    nine = (
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
    for relative, task in temporal_tasks.items():
        rich = require_mapping(
            task.get("source_rich_identity"), "temporal rich identity"
        )
        content_stat = require_mapping(content_tasks[relative], "content source stat")
        require(set(rich) == set(nine), "temporal nine-field identity ABI drift")
        require(set(content_stat) == set(six), "content six-field identity ABI drift")
        normalized = {
            key: (stat.S_IMODE(int(rich[key])) if key == "mode" else rich[key])
            for key in six
        }
        require(normalized == content_stat, "content/temporal source identity mismatch")

    return {
        "module": module,
        "output_root": output_root,
        "receipt": receipt,
        "receipt_sha256": sha256_file(temporal_receipt_path),
        "plan": plan,
        "report": report,
        "state": state,
        "checkpoints_by_path": checkpoints_by_path,
        "content_context": content_context,
        "synthetic_cap_evidence": synthetic_cap_evidence,
        "temporal_config_sha256": temporal_config_sha,
        "temporal_tool_sha256": tool_sha,
    }


def _projected_schema_from_temporal_task(
    task: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    schema = require_mapping(task.get("projected_schema"), "temporal projected schema")
    result = {
        name: require_mapping(schema.get(name), f"projected schema {name}")
        for name in ("id", "publication_date", "publication_year")
    }
    require(
        result
        == {
            "id": {"nullable": True, "type": "string"},
            "publication_date": {"nullable": True, "type": "string"},
            "publication_year": {"nullable": True, "type": "double"},
        },
        "three-column projected schema drift",
    )
    return result


def _build_plan(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    config_identity: Mapping[str, int],
    temporal: Mapping[str, Any],
    batch_size: int,
    partition_sort_chunk_rows: int,
) -> dict[str, Any]:
    require(batch_size > 0, "batch size must be positive")
    require(partition_sort_chunk_rows > 0, "partition sort chunk must be positive")
    tool_path = Path(__file__).resolve()
    tool_identity = _regular_identity(tool_path, field="spine tool")
    tool_sha = sha256_file(tool_path)
    temporal_plan = require_mapping(temporal.get("plan"), "temporal plan")
    checkpoints = require_mapping(
        temporal.get("checkpoints_by_path"),
        "temporal checkpoint map",
    )
    content_runtime = require_mapping(
        temporal.get("content_context", {}).get("runtime"),
        "content runtime",
    )
    content_tasks = require_mapping(
        content_runtime.get("main_source_tasks"), "content tasks"
    )
    tasks: list[dict[str, Any]] = []
    for raw in require_list(temporal_plan.get("tasks"), "temporal tasks"):
        task = require_mapping(raw, "temporal task")
        relative = str(task["relative_path"])
        checkpoint = checkpoints[relative]
        projected_schema = _projected_schema_from_temporal_task(task)
        core = {
            "file_index": require_int(task.get("file_index"), "temporal file index"),
            "relative_path": relative,
            "footer_rows": require_int(task.get("footer_rows"), "temporal footer rows"),
            "source_rich_identity": dict(task["source_rich_identity"]),
            "content_source_stat": dict(content_tasks[relative]),
            "temporal_task_sha256": require_sha256(
                task.get("task_sha256"), "temporal task SHA"
            ),
            "temporal_checkpoint_sha256": require_sha256(
                checkpoint.get("checkpoint_sha256"), "temporal checkpoint SHA"
            ),
            "projected_columns": ["id", "publication_date", "publication_year"],
            "projected_schema": projected_schema,
            "projected_schema_sha256": sha256_bytes(
                canonical_json_bytes(projected_schema)
            ),
        }
        core["spine_task_sha256"] = sha256_bytes(canonical_json_bytes(core))
        tasks.append(core)
    tasks.sort(key=lambda item: str(item["relative_path"]))
    require(
        [int(item["file_index"]) for item in tasks]
        == sorted(int(item["file_index"]) for item in tasks),
        "temporal lexical/file-index order drift",
    )

    content_commitment = require_mapping(
        temporal.get("content_context", {}).get("commitment"),
        "content commitment",
    )
    content_receipt_snapshot = require_mapping(
        content_runtime.get("receipt"),
        "content receipt snapshot",
    )
    content_receipt = read_json(
        Path(str(content_receipt_snapshot["path"])),
        "content receipt",
    )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "scope": SYNTHETIC_SCOPE,
        "production_status": PRODUCTION_STATUS,
        "config_sha256": config_sha256,
        "config_rich_identity": dict(config_identity),
        "tool_sha256": tool_sha,
        "tool_rich_identity": tool_identity,
        "runtime_fingerprint": _runtime_fingerprint(),
        "synthetic_cap_evidence": dict(
            require_mapping(
                temporal.get("synthetic_cap_evidence"), "synthetic cap evidence"
            )
        ),
        "source": dict(require_mapping(config.get("source"), "source")),
        "parents": {
            "content": {
                "receipt_sha256": content_commitment.get("receipt_sha256"),
                "result_fingerprint_sha256": content_receipt.get(
                    "result_fingerprint_sha256"
                ),
                "task_fingerprint_sha256": content_commitment.get(
                    "task_fingerprint_sha256"
                ),
                "content_tree_sha256": content_commitment.get("content_tree_sha256"),
                "files": content_commitment.get("files"),
                "bytes": content_commitment.get("bytes"),
            },
            "temporal": {
                "receipt_sha256": temporal.get("receipt_sha256"),
                "result_fingerprint_sha256": temporal["receipt"][
                    "result_fingerprint_sha256"
                ],
                "task_fingerprint_sha256": temporal["receipt"][
                    "task_fingerprint_sha256"
                ],
                "checkpoint_tree_sha256": temporal["report"]["checkpoint_evidence"][
                    "checkpoint_tree_sha256"
                ],
                "source_tree_sha256": temporal["plan"]["source_tree_start"][
                    "tree_sha256"
                ],
                "config_sha256": temporal.get("temporal_config_sha256"),
                "tool_sha256": temporal.get("temporal_tool_sha256"),
            },
        },
        "batch_size": batch_size,
        "partition_sort_chunk_rows": partition_sort_chunk_rows,
        "partition_count": PARTITION_COUNT,
        "partition_formula": "work_id_numeric_mod_4096",
        "record_abi": "OPENALEX_WORK_TIME_STAGING_RECORD_LE16_V1",
        "writer_abi": dict(config["partition_contract"]["writer_abi"]),
        "tasks": tasks,
        "source_tree_sha256": sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "relative_path": item["relative_path"],
                        "source_rich_identity": item["source_rich_identity"],
                    }
                    for item in tasks
                ]
            )
        ),
        "receipt_artifact_formula": "6 + 2*N + 2*4096",
        "synthetic_receipt_artifact_count": 6 + 2 * len(tasks) + 2 * PARTITION_COUNT,
        "production_receipt_artifact_count_n_493": 9184,
        "operator_paths_redacted": True,
    }
    plan["task_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


def record_digest(records: Sequence[Record]) -> str:
    digest = hashlib.sha256()
    digest.update(len(RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    digest.update(RECORD_HASH_DOMAIN)
    digest.update(len(records).to_bytes(8, "little"))
    for record in records:
        encoded = record.encode()
        digest.update(RECORD_BYTES.to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def decode_staging(payload: bytes) -> tuple[Record, ...]:
    require(len(payload) % RECORD_BYTES == 0, "staging shard byte width mismatch")
    return tuple(
        Record.decode(payload[offset : offset + RECORD_BYTES])
        for offset in range(0, len(payload), RECORD_BYTES)
    )


def _strict_date(value: Any) -> tuple[bool, int, int]:
    if value is None:
        return False, 0, 0
    require(isinstance(value, str), "publication date is not a string")
    require(
        re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is not None,
        "publication date does not match YYYY-MM-DD",
    )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SpineAuditError("publication date is not a valid calendar day") from exc
    require(parsed.isoformat() == value, "publication date is not canonical")
    return True, (parsed - date(1970, 1, 1)).days, parsed.year


def _strict_year(value: Any) -> tuple[bool, int]:
    if value is None:
        return False, 0
    require(type(value) in {int, float}, "publication year is not numeric")
    number = float(value)
    require(math.isfinite(number), "publication year is not finite")
    require(number == math.trunc(number), "publication year is not integral")
    integer = int(number)
    require(1 <= integer <= 9999, "publication year is outside 1..9999")
    return True, integer


def classify_publication(
    publication_date: Any,
    publication_year: Any,
    *,
    snapshot: date,
) -> tuple[int, int | None, int | None]:
    """Return the frozen disposition and normalized day/year."""

    date_null = publication_date is None
    year_null = publication_year is None
    if date_null and year_null:
        return 0, None, None
    require(not (date_null ^ year_null), "publication pair is exactly-one-null")
    date_valid, day, derived_year = _strict_date(publication_date)
    year_valid, year = _strict_year(publication_year)
    require(date_valid and year_valid, "publication pair is invalid")
    require(derived_year == year, "publication date/year mismatch")
    pre_1900 = (date(1900, 1, 1) - date(1970, 1, 1)).days
    snapshot_day = (snapshot - date(1970, 1, 1)).days
    if day < pre_1900:
        disposition = 1
    elif derived_year < snapshot.year:
        disposition = 2
    elif derived_year == snapshot.year and day <= snapshot_day:
        disposition = 3
    elif derived_year == snapshot.year:
        disposition = 4
    else:
        disposition = 5
    return disposition, day, year


def _empty_metrics() -> dict[str, int]:
    return {key: 0 for key in SPINE_METRIC_KEYS}


def _increment(metrics: dict[str, int], key: str, condition: bool) -> None:
    if condition:
        metrics[key] += 1


def _source_payload_batches(
    path: Path,
    *,
    task: Mapping[str, Any],
    batch_size: int,
) -> tuple[Any, Any, dict[str, int], pq.ParquetFile]:
    """Open a source payload; isolated so terminal tests can forbid this call."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise SpineAuditError("cannot open source Parquet") from exc
    try:
        handle = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        identity = _rich_identity_from_stat(os.fstat(handle.fileno()))
        require(
            identity == task.get("source_rich_identity"),
            "source rich identity differs from parent plan",
        )
        parquet = pq.ParquetFile(handle)
        require(
            parquet.metadata.num_rows == task.get("footer_rows"),
            "source footer rows differ",
        )
        actual_schema = {
            name: {
                "nullable": bool(parquet.schema_arrow.field(name).nullable),
                "type": str(parquet.schema_arrow.field(name).type),
            }
            for name in ("id", "publication_date", "publication_year")
        }
        require(
            actual_schema == task.get("projected_schema"),
            "source projected schema differs",
        )
        require(
            sha256_bytes(canonical_json_bytes(actual_schema))
            == task.get("projected_schema_sha256"),
            "source projected schema hash differs",
        )
    except BaseException:
        handle.close()
        raise
    return handle, identity, parquet, batch_size


def _scan_task(
    source_root: Path,
    task: Mapping[str, Any],
    temporal_checkpoint: Mapping[str, Any],
    *,
    batch_size: int,
    snapshot: date,
) -> ScanResult:
    relative = safe_relative_path(task.get("relative_path"), "source task path")
    path = secure_regular(source_root, relative, field="source Parquet")
    handle, identity, parquet, _ = _source_payload_batches(
        path,
        task=task,
        batch_size=batch_size,
    )
    metrics = _empty_metrics()
    records: list[Record] = []
    digest = hashlib.sha256()
    year_minimum: int | None = None
    year_maximum: int | None = None
    disposition_counts = {name: 0 for name in DISPOSITION_NAMES.values()}
    scanned_rows = 0
    try:
        for row_group in range(parquet.metadata.num_row_groups):
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group],
                columns=["id", "publication_date", "publication_year"],
                use_threads=False,
            ):
                serialized = batch.serialize().to_pybytes()
                digest.update(len(serialized).to_bytes(8, "little"))
                digest.update(serialized)
                for index in range(batch.num_rows):
                    raw_id = batch.column(0)[index].as_py()
                    raw_date = batch.column(1)[index].as_py()
                    raw_year = batch.column(2)[index].as_py()
                    metrics["raw_rows"] += 1
                    if raw_id == W_MINUS_ONE:
                        metrics["quarantined_w_minus_1_rows"] += 1
                        continue

                    metrics["retained_rows"] += 1
                    id_null = raw_id is None
                    grammar = (
                        isinstance(raw_id, str)
                        and WORK_PATTERN.fullmatch(raw_id) is not None
                    )
                    _increment(metrics, "null_id_rows", id_null)
                    _increment(metrics, "grammar_valid_id_rows", grammar)
                    _increment(metrics, "unexpected_malformed_id_rows", not grammar)
                    _increment(
                        metrics, "w_zero_rows", raw_id == "https://openalex.org/W0"
                    )

                    date_null = raw_date is None
                    year_null = raw_year is None
                    both_null = date_null and year_null
                    exactly_one = date_null ^ year_null
                    both_nonnull = not date_null and not year_null
                    _increment(metrics, "publication_both_null_rows", both_null)
                    _increment(
                        metrics, "publication_exactly_one_null_rows", exactly_one
                    )
                    _increment(metrics, "publication_both_nonnull_rows", both_nonnull)
                    _increment(metrics, "publication_year_null_rows", year_null)

                    date_valid = False
                    day = 0
                    derived_year = 0
                    if not date_null:
                        try:
                            date_valid, day, derived_year = _strict_date(raw_date)
                        except SpineAuditError:
                            metrics["publication_date_parse_error_rows"] += 1
                    year_valid = False
                    year = 0
                    if not year_null:
                        try:
                            year_valid, year = _strict_year(raw_year)
                        except SpineAuditError:
                            metrics["publication_year_invalid_rows"] += 1

                    checked = date_valid and year_valid
                    matched = checked and derived_year == year
                    _increment(metrics, "publication_concordance_checked_rows", checked)
                    _increment(metrics, "publication_date_year_match_rows", matched)
                    _increment(
                        metrics,
                        "publication_date_year_mismatch_rows",
                        checked and not matched,
                    )
                    if date_valid:
                        pre_1900_day = (date(1900, 1, 1) - date(1970, 1, 1)).days
                        snapshot_day = (snapshot - date(1970, 1, 1)).days
                        _increment(
                            metrics, "publication_pre_1900_rows", day < pre_1900_day
                        )
                        _increment(
                            metrics,
                            "publication_1900_before_snapshot_year_rows",
                            day >= pre_1900_day and derived_year < snapshot.year,
                        )
                        _increment(
                            metrics,
                            "publication_snapshot_year_on_or_before_snapshot_rows",
                            derived_year == snapshot.year and day <= snapshot_day,
                        )
                        _increment(
                            metrics,
                            "publication_snapshot_year_after_snapshot_rows",
                            derived_year == snapshot.year and day > snapshot_day,
                        )
                        _increment(
                            metrics,
                            "publication_post_snapshot_year_rows",
                            derived_year > snapshot.year,
                        )
                    if year_valid:
                        _increment(
                            metrics, "publication_year_pre_1900_rows", year < 1900
                        )
                        _increment(
                            metrics,
                            "publication_year_snapshot_year_rows",
                            year == snapshot.year,
                        )
                        _increment(
                            metrics,
                            "publication_year_post_snapshot_year_rows",
                            year > snapshot.year,
                        )
                        year_minimum = (
                            year if year_minimum is None else min(year_minimum, year)
                        )
                        year_maximum = (
                            year if year_maximum is None else max(year_maximum, year)
                        )

                    # Identifier and publication rejection occurs before any
                    # staging shard is published.  The full scan preflight in
                    # _execute_scan validates every task before publication.
                    require(grammar, "retained work ID is null or malformed")
                    suffix = WORK_PATTERN.fullmatch(str(raw_id)).group(1)  # type: ignore[union-attr]
                    work_id_numeric = int(suffix)
                    require(work_id_numeric <= 2**64 - 1, "work ID exceeds uint64")
                    disposition, normalized_day, normalized_year = classify_publication(
                        raw_date,
                        raw_year,
                        snapshot=snapshot,
                    )
                    record = Record(
                        work_id_numeric,
                        normalized_day,
                        normalized_year,
                        disposition,
                    )
                    record.encode()
                    records.append(record)
                    disposition_counts[DISPOSITION_NAMES[disposition]] += 1
                scanned_rows += batch.num_rows
        require(
            _rich_identity_from_stat(os.fstat(handle.fileno())) == identity,
            "open source changed during scan",
        )
        require(
            _regular_identity(path, field="source Parquet") == identity,
            "source path changed during scan",
        )
    except (OSError, pa.ArrowException, OverflowError, ValueError) as exc:
        raise SpineAuditError("cannot scan source projection") from exc
    finally:
        handle.close()
    require(scanned_rows == task.get("footer_rows"), "scanned rows differ from footer")

    temporal_metrics = require_mapping(
        temporal_checkpoint.get("metrics"), "temporal metrics"
    )
    require(
        set(temporal_metrics) == set(TEMPORAL_V3_METRIC_KEYS),
        "temporal-v3 checkpoint metric keyset mismatch",
    )
    expected_metrics = {
        key: require_int(temporal_metrics.get(key), f"temporal metric {key}")
        for key in SPINE_METRIC_KEYS
    }
    require(metrics == expected_metrics, "spine/temporal per-file metric mismatch")
    extrema = {
        "minimum": year_minimum,
        "maximum": year_maximum,
    }
    require(
        extrema == temporal_checkpoint.get("publication_year_extrema"),
        "spine/temporal year-extrema mismatch",
    )
    for key in (
        "null_id_rows",
        "unexpected_malformed_id_rows",
        "publication_exactly_one_null_rows",
        "publication_date_parse_error_rows",
        "publication_year_invalid_rows",
        "publication_date_year_mismatch_rows",
    ):
        require(metrics[key] == 0, f"rejected-row metric is nonzero: {key}")
    require(
        len(records) == metrics["retained_rows"], "retained staging row count mismatch"
    )
    return ScanResult(
        file_index=require_int(task.get("file_index"), "task file index"),
        relative_path=str(task["relative_path"]),
        source_rich_identity=identity,
        footer_rows=scanned_rows,
        projected_schema_sha256=str(task["projected_schema_sha256"]),
        projected_payload_sha256=digest.hexdigest(),
        metrics=metrics,
        publication_year_extrema=extrema,
        records=tuple(records),
        disposition_counts=disposition_counts,
    )


def _sum_metrics(results: Sequence[ScanResult]) -> dict[str, int]:
    totals = _empty_metrics()
    for result in results:
        for key in SPINE_METRIC_KEYS:
            totals[key] += result.metrics[key]
    return totals


def _tree_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(records)))


def _scan_checkpoint_core(
    result: ScanResult,
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    staging_relative = (
        f"staging/main/{sha256_bytes(result.relative_path.encode('utf-8'))}.bin"
    )
    payload = b"".join(record.encode() for record in result.records)
    return {
        "schema_version": SCAN_CHECKPOINT_SCHEMA,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "spine_task_sha256": task["spine_task_sha256"],
        "temporal_parent_task_sha256": task["temporal_task_sha256"],
        "temporal_parent_checkpoint_sha256": task["temporal_checkpoint_sha256"],
        "file_index": result.file_index,
        "relative_path": result.relative_path,
        "source_rich_identity": result.source_rich_identity,
        "footer_rows": result.footer_rows,
        "projected_schema_sha256": result.projected_schema_sha256,
        "projected_payload_sha256": result.projected_payload_sha256,
        "disposition_metrics": result.metrics,
        "disposition_metrics_sha256": sha256_bytes(
            canonical_json_bytes(result.metrics)
        ),
        "publication_year_extrema": result.publication_year_extrema,
        "row_level_disposition_payload_sha256": record_digest(result.records),
        "disposition_counts": result.disposition_counts,
        "staging_shard_relative_path": staging_relative,
        "staging_shard_rows": len(result.records),
        "staging_shard_bytes": len(payload),
        "staging_shard_sha256": sha256_bytes(payload),
        "operator_paths_redacted": True,
    }


def _with_self_hash(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = sha256_bytes(canonical_json_bytes(result))
    return result


def _validate_scan_checkpoint(
    output_dir: Path,
    checkpoint: Mapping[str, Any],
    expected_core: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(checkpoint)
    digest = value.pop("checkpoint_sha256", None)
    require(
        digest == sha256_bytes(canonical_json_bytes(value)),
        "scan checkpoint self hash mismatch",
    )
    require(value == expected_core, "scan checkpoint differs from current projection")
    staging_relative = safe_relative_path(
        value["staging_shard_relative_path"], "staging shard"
    )
    staging_path = secure_regular(output_dir, staging_relative, field="staging shard")
    payload = staging_path.read_bytes()
    require(
        len(payload) == value["staging_shard_bytes"],
        "staging shard byte count mismatch",
    )
    require(
        sha256_bytes(payload) == value["staging_shard_sha256"],
        "staging shard hash mismatch",
    )
    records = decode_staging(payload)
    require(
        len(records) == value["staging_shard_rows"], "staging shard row count mismatch"
    )
    require(
        record_digest(records) == value["row_level_disposition_payload_sha256"],
        "staging row digest mismatch",
    )
    return dict(checkpoint)


def _validate_global_scan(
    config: Mapping[str, Any],
    temporal: Mapping[str, Any],
    results: Sequence[ScanResult],
) -> dict[str, Any]:
    metrics = _sum_metrics(results)
    identifiers = require_mapping(
        config.get("identifier_accounting"), "identifier accounting"
    )
    require(
        metrics["raw_rows"] == identifiers.get("raw_rows"), "raw row anchor mismatch"
    )
    require(
        metrics["quarantined_w_minus_1_rows"]
        == identifiers.get("frozen_w_minus_1_quarantine_rows"),
        "W-1 row anchor mismatch",
    )
    require(
        metrics["retained_rows"] == identifiers.get("retained_rows"),
        "retained row anchor mismatch",
    )
    records = [record for result in results for record in result.records]
    numeric_ids = [record.work_id_numeric for record in records]
    unique_ids = set(numeric_ids)
    duplicate_excess = len(numeric_ids) - len(unique_ids)
    duplicate_groups = len(
        {item for item in numeric_ids if numeric_ids.count(item) > 1}
    )
    require(
        len(records) == identifiers.get("retained_rows"), "spine retained row mismatch"
    )
    require(
        len(unique_ids) == identifiers.get("retained_distinct_numeric_ids"),
        "distinct numeric ID mismatch",
    )
    require(
        duplicate_groups == identifiers.get("duplicate_numeric_id_groups"),
        "duplicate ID group mismatch",
    )
    require(
        duplicate_excess == identifiers.get("duplicate_numeric_row_excess"),
        "duplicate row excess mismatch",
    )
    require(
        min(numeric_ids) == identifiers.get("minimum_numeric_id"),
        "minimum numeric ID mismatch",
    )
    require(
        max(numeric_ids) == identifiers.get("maximum_numeric_id"),
        "maximum numeric ID mismatch",
    )
    require(
        numeric_ids.count(0) == identifiers.get("w_zero_rows_retained"),
        "W0 row mismatch",
    )

    temporal_metrics = require_mapping(
        temporal.get("report", {}).get("metrics"), "temporal global metrics"
    )
    require(
        set(temporal_metrics) == set(TEMPORAL_V3_METRIC_KEYS),
        "temporal-v3 global metric keyset mismatch",
    )
    require(
        metrics == {key: temporal_metrics.get(key) for key in SPINE_METRIC_KEYS},
        "spine/temporal global metric mismatch",
    )
    counts = {name: 0 for name in DISPOSITION_NAMES.values()}
    for record in records:
        counts[DISPOSITION_NAMES[record.temporal_disposition]] += 1
        if record.publication_day is not None:
            derived = (
                date(1970, 1, 1)
                .fromordinal(date(1970, 1, 1).toordinal() + record.publication_day)
                .year
            )
            require(
                derived == record.publication_year, "decoded record year/day mismatch"
            )
    publication = require_mapping(
        config.get("publication_accounting"), "publication accounting"
    )
    fixed = require_mapping(
        publication.get("fixed_disposition_anchors"), "fixed dispositions"
    )
    for name, expected in fixed.items():
        require(counts[name] == expected, f"fixed disposition mismatch: {name}")
    require(
        counts["USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT"]
        + counts["VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED"]
        == publication.get("snapshot_year_total_rows"),
        "snapshot-year split does not close",
    )
    require(
        sum(counts.values()) == len(records),
        "six-disposition accounting does not close",
    )
    nonnull_years = [
        record.publication_year
        for record in records
        if record.publication_year is not None
    ]
    require(
        min(nonnull_years) == publication.get("required_publication_year_minimum"),
        "publication-year minimum mismatch",
    )
    require(
        max(nonnull_years) == publication.get("required_publication_year_maximum"),
        "publication-year maximum mismatch",
    )
    return {
        "metrics": metrics,
        "records": records,
        "identifier_accounting": {
            "raw_rows": metrics["raw_rows"],
            "quarantined_w_minus_1_rows": metrics["quarantined_w_minus_1_rows"],
            "retained_rows": len(records),
            "distinct_numeric_ids": len(unique_ids),
            "duplicate_numeric_id_groups": duplicate_groups,
            "duplicate_numeric_row_excess": duplicate_excess,
            "minimum_numeric_id": min(numeric_ids),
            "maximum_numeric_id": max(numeric_ids),
            "w_zero_rows": numeric_ids.count(0),
            "w_zero_entity_validity": "UNRESOLVED",
        },
        "publication_accounting": {
            "disposition_counts": counts,
            "publication_year_minimum": min(nonnull_years),
            "publication_year_maximum": max(nonnull_years),
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


def _scan_checkpoint_path(output_dir: Path, relative_path: str) -> Path:
    token = sha256_bytes(relative_path.encode("utf-8"))
    return output_dir / "checkpoints/scan/main" / f"{token}.json"


def _scan_checkpoint_tree(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    checkpoints: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
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
        "schema_version": STAGING_MANIFEST_SCHEMA,
        "scope": SYNTHETIC_SCOPE,
        "production_status": PRODUCTION_STATUS,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "scan_checkpoint_tree_sha256": _scan_checkpoint_tree(checkpoints)[
            "checkpoint_tree_sha256"
        ],
        "shard_count": len(shards),
        "rows": sum(int(item["rows"]) for item in shards),
        "bytes": sum(int(item["bytes"]) for item in shards),
        "staging_shard_tree_sha256": _tree_digest(shards),
        "shards": shards,
        "operator_paths_redacted": True,
    }


def _state(
    plan: Mapping[str, Any],
    status: str,
    *,
    completed_scan_files: int,
    completed_partitions: int,
    result_fingerprint_sha256: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": STATE_SCHEMA,
        "scope": SYNTHETIC_SCOPE,
        "production_status": PRODUCTION_STATUS,
        "status": status,
        "completed_scan_files": completed_scan_files,
        "total_scan_files": len(plan["tasks"]),
        "completed_partitions": completed_partitions,
        "total_partitions": PARTITION_COUNT,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "operator_paths_redacted": True,
    }
    if result_fingerprint_sha256 is not None:
        value.update(
            {
                "result_fingerprint_sha256": result_fingerprint_sha256,
                "completion_receipt_relative_path": (
                    "work_time_spine_audit.outputs.json"
                ),
                "complete_only_if_receipt_exists": True,
            }
        )
    return value


def _execute_scan(
    output_dir: Path,
    source_root: Path,
    config: Mapping[str, Any],
    temporal: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    resume: bool,
    pause_after_new_scan_files: int | None,
) -> tuple[list[ScanResult], list[dict[str, Any]], bool, dict[str, Any]]:
    require(
        pause_after_new_scan_files is None or pause_after_new_scan_files > 0,
        "scan pause count must be positive",
    )
    snapshot = date.fromisoformat(str(config["source"]["snapshot_date"]))
    temporal_checkpoints = temporal["checkpoints_by_path"]

    # Global validation precedes the first published staging byte.  This is a
    # deliberate synthetic implementation of the contract's fail-before-
    # staging rule; production external-sort implementation remains blocked.
    results = [
        _scan_task(
            source_root,
            task,
            temporal_checkpoints[str(task["relative_path"])],
            batch_size=int(plan["batch_size"]),
            snapshot=snapshot,
        )
        for task in plan["tasks"]
    ]
    global_evidence = _validate_global_scan(config, temporal, results)

    checkpoints: list[dict[str, Any]] = []
    gap_seen = False
    for result, task in zip(results, plan["tasks"], strict=True):
        core = _scan_checkpoint_core(result, task, plan)
        checkpoint_path = _scan_checkpoint_path(output_dir, result.relative_path)
        staging_path = output_dir / safe_relative_path(
            core["staging_shard_relative_path"],
            "staging shard",
        )
        exists = checkpoint_path.exists() or checkpoint_path.is_symlink()
        shard_exists = staging_path.exists() or staging_path.is_symlink()
        require(exists == shard_exists, "orphan scan checkpoint or staging shard")
        if exists:
            require(resume, "existing scan checkpoint requires resume")
            require(not gap_seen, "scan checkpoint tree is not a contiguous prefix")
            checkpoints.append(
                _validate_scan_checkpoint(
                    output_dir,
                    read_json(checkpoint_path, "scan checkpoint"),
                    core,
                )
            )
        else:
            gap_seen = True

    new_files = 0
    for result, task in zip(
        results[len(checkpoints) :],
        plan["tasks"][len(checkpoints) :],
        strict=True,
    ):
        core = _scan_checkpoint_core(result, task, plan)
        staging_path = output_dir / safe_relative_path(
            core["staging_shard_relative_path"],
            "staging shard",
        )
        atomic_write_bytes(
            staging_path, b"".join(record.encode() for record in result.records)
        )
        checkpoint = _with_self_hash(core, "checkpoint_sha256")
        atomic_write_json(
            _scan_checkpoint_path(output_dir, result.relative_path),
            checkpoint,
        )
        checkpoints.append(checkpoint)
        new_files += 1
        atomic_write_json(
            output_dir / "run_state.json",
            _state(
                plan,
                "RUNNING_SCAN",
                completed_scan_files=len(checkpoints),
                completed_partitions=0,
            ),
        )
        if (
            pause_after_new_scan_files is not None
            and new_files >= pause_after_new_scan_files
            and len(checkpoints) < len(results)
        ):
            atomic_write_json(
                output_dir / "run_state.json",
                _state(
                    plan,
                    "PAUSED_SCAN",
                    completed_scan_files=len(checkpoints),
                    completed_partitions=0,
                ),
            )
            return results, checkpoints, True, global_evidence

    manifest = _staging_manifest(checkpoints, plan)
    manifest_path = output_dir / "staging_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        require(resume, "existing staging manifest requires resume")
        require(
            read_json(manifest_path, "staging manifest") == manifest,
            "staging manifest differs",
        )
    else:
        atomic_write_json(manifest_path, manifest)
    return results, checkpoints, False, global_evidence


def _records_to_table(records: Sequence[Record]) -> pa.Table:
    table = pa.Table.from_arrays(
        [
            pa.array([item.work_id_numeric for item in records], type=pa.uint64()),
            pa.array([item.publication_day for item in records], type=pa.int32()),
            pa.array([item.publication_year for item in records], type=pa.uint16()),
            pa.array(
                [item.temporal_disposition for item in records],
                type=pa.uint8(),
            ),
        ],
        schema=ARROW_SCHEMA,
    )
    return table.combine_chunks().replace_schema_metadata(None)


def _parquet_payload(records: Sequence[Record]) -> bytes:
    global _EMPTY_PARQUET_BYTES
    if not records and _EMPTY_PARQUET_BYTES is not None:
        return _EMPTY_PARQUET_BYTES
    sink = pa.BufferOutputStream()
    pq.write_table(
        _records_to_table(records),
        sink,
        version="2.6",
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        row_group_size=262144,
        data_page_version="1.0",
        write_batch_size=262144,
        use_byte_stream_split=False,
        write_page_index=False,
        write_page_checksum=False,
        store_schema=False,
    )
    payload = sink.getvalue().to_pybytes()
    if not records:
        _EMPTY_PARQUET_BYTES = payload
    return payload


def _validated_empty_parquet_payload() -> bytes:
    global _EMPTY_PARQUET_VALIDATED
    payload = _parquet_payload(())
    if not _EMPTY_PARQUET_VALIDATED:
        parquet = pq.ParquetFile(pa.BufferReader(payload))
        _validate_parquet_schema(parquet)
        require(parquet.metadata.num_rows == 0, "empty Parquet template has rows")
        _EMPTY_PARQUET_VALIDATED = True
    return payload


def _validate_parquet_schema(parquet: pq.ParquetFile) -> None:
    require(parquet.schema_arrow == ARROW_SCHEMA, "decoded Arrow schema differs")
    require(
        parquet.schema_arrow.metadata is None, "stored Arrow schema metadata leaked"
    )
    fields = [parquet.schema.column(index) for index in range(4)]
    observed = [
        {
            "name": field.name,
            "physical_type": field.physical_type,
            "logical_type": str(field.logical_type),
            "max_definition_level": field.max_definition_level,
        }
        for field in fields
    ]
    require(
        observed[0]["physical_type"] == "INT64", "work ID Parquet physical type drift"
    )
    require(
        "bitWidth=64" in observed[0]["logical_type"]
        and "isSigned=false" in observed[0]["logical_type"],
        "work ID Parquet logical type drift",
    )
    require(observed[0]["max_definition_level"] == 0, "work ID nullability drift")
    require(
        observed[1]["physical_type"] == "INT32", "publication day physical type drift"
    )
    require(observed[1]["logical_type"] == "None", "publication day logical type drift")
    require(
        observed[1]["max_definition_level"] == 1, "publication day nullability drift"
    )
    require(
        observed[2]["physical_type"] == "INT32", "publication year physical type drift"
    )
    require(
        "bitWidth=16" in observed[2]["logical_type"]
        and "isSigned=false" in observed[2]["logical_type"],
        "publication year logical type drift",
    )
    require(
        observed[2]["max_definition_level"] == 1, "publication year nullability drift"
    )
    require(observed[3]["physical_type"] == "INT32", "disposition physical type drift")
    require(
        "bitWidth=8" in observed[3]["logical_type"]
        and "isSigned=false" in observed[3]["logical_type"],
        "disposition logical type drift",
    )
    require(observed[3]["max_definition_level"] == 0, "disposition nullability drift")
    metadata = parquet.metadata.metadata or {}
    require(b"ARROW:schema" not in metadata, "Arrow schema metadata was stored")


def _decode_parquet(path: Path) -> tuple[Record, ...]:
    _assert_no_symlink_components(path, field="spine partition")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise SpineAuditError("cannot open spine partition") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            parquet = pq.ParquetFile(handle)
            _validate_parquet_schema(parquet)
            for row_group in range(parquet.metadata.num_row_groups):
                column = parquet.metadata.row_group(row_group).column(0)
                require(
                    column.compression == "UNCOMPRESSED", "partition compression drift"
                )
                require(
                    column.is_stats_set is False, "partition statistics were written"
                )
            table = parquet.read(use_threads=False)
    except (OSError, pa.ArrowException) as exc:
        raise SpineAuditError("cannot decode spine partition") from exc
    finally:
        os.close(descriptor)
    records: list[Record] = []
    ids = table.column("work_id_numeric").combine_chunks()
    days = table.column("publication_day").combine_chunks()
    years = table.column("publication_year").combine_chunks()
    dispositions = table.column("temporal_disposition").combine_chunks()
    for index in range(table.num_rows):
        disposition = int(dispositions[index].as_py())
        day = days[index].as_py()
        year = years[index].as_py()
        record = Record(
            int(ids[index].as_py()),
            None if day is None else int(day),
            None if year is None else int(year),
            disposition,
        )
        record.encode()
        records.append(record)
    return tuple(records)


def _partition_core(
    output_dir: Path,
    partition_id: int,
    records: Sequence[Record],
    *,
    plan: Mapping[str, Any],
    scan_tree_sha256: str,
    staging_tree_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    require(0 <= partition_id < PARTITION_COUNT, "partition ID out of range")
    ordered = tuple(sorted(records, key=lambda item: item.work_id_numeric))
    ids = [item.work_id_numeric for item in ordered]
    require(len(ids) == len(set(ids)), "duplicate ID within partition")
    require(
        all(left < right for left, right in zip(ids, ids[1:])),
        "partition IDs are not strictly increasing",
    )
    require(
        all(item % PARTITION_COUNT == partition_id for item in ids),
        "partition modulo assignment mismatch",
    )
    payload = _parquet_payload(ordered)
    relative = f"spine/part-{partition_id:04d}.parquet"
    path = output_dir / relative
    atomic_write_bytes(path, payload)
    if ordered:
        decoded = _decode_parquet(path)
    else:
        require(
            path.read_bytes() == _validated_empty_parquet_payload(),
            "published empty partition differs from canonical writer bytes",
        )
        decoded = ()
    require(
        decoded == ordered, "decoded partition rows differ from sorted staging rows"
    )
    pre_digest = record_digest(ordered)
    post_digest = record_digest(decoded)
    require(pre_digest == post_digest, "pre/post Parquet record digest mismatch")
    disposition_counts = {name: 0 for name in DISPOSITION_NAMES.values()}
    nullability = {
        "publication_day_null_rows": 0,
        "publication_year_null_rows": 0,
    }
    for item in decoded:
        disposition_counts[DISPOSITION_NAMES[item.temporal_disposition]] += 1
        nullability["publication_day_null_rows"] += item.publication_day is None
        nullability["publication_year_null_rows"] += item.publication_year is None
    schema_contract = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in ARROW_SCHEMA
    ]
    core = {
        "schema_version": PARTITION_CHECKPOINT_SCHEMA,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "scan_checkpoint_tree_sha256": scan_tree_sha256,
        "staging_shard_tree_sha256": staging_tree_sha256,
        "partition_id": partition_id,
        "partition_formula": "work_id_numeric_mod_4096",
        "output_relative_path": relative,
        "output_rows": len(decoded),
        "minimum_work_id_numeric": min(ids) if ids else None,
        "maximum_work_id_numeric": max(ids) if ids else None,
        "output_schema": schema_contract,
        "output_schema_sha256": sha256_bytes(canonical_json_bytes(schema_contract)),
        "pre_parquet_sorted_record_sha256": pre_digest,
        "decoded_parquet_record_sha256": post_digest,
        "disposition_counts": disposition_counts,
        "nullability_counts": nullability,
        "output_bytes": len(payload),
        "output_sha256": sha256_bytes(payload),
        "strictly_increasing_ids": True,
        "operator_paths_redacted": True,
    }
    return core, payload


def _expected_partition_core_without_publish(
    output_dir: Path,
    partition_id: int,
    records: Sequence[Record],
    *,
    plan: Mapping[str, Any],
    scan_tree_sha256: str,
    staging_tree_sha256: str,
) -> dict[str, Any]:
    ordered = tuple(sorted(records, key=lambda item: item.work_id_numeric))
    relative = f"spine/part-{partition_id:04d}.parquet"
    path = secure_regular(output_dir, Path(relative), field="spine partition")
    payload = path.read_bytes()
    if ordered:
        decoded = _decode_parquet(path)
    else:
        require(
            payload == _validated_empty_parquet_payload(),
            "existing empty partition differs from canonical writer bytes",
        )
        decoded = ()
    require(decoded == ordered, "existing partition decoded rows differ")
    ids = [item.work_id_numeric for item in decoded]
    require(
        all(item % PARTITION_COUNT == partition_id for item in ids),
        "existing partition modulo mismatch",
    )
    require(len(ids) == len(set(ids)), "existing partition duplicate IDs")
    require(
        all(left < right for left, right in zip(ids, ids[1:])),
        "existing partition sort mismatch",
    )
    disposition_counts = {name: 0 for name in DISPOSITION_NAMES.values()}
    for item in decoded:
        disposition_counts[DISPOSITION_NAMES[item.temporal_disposition]] += 1
    null_count = sum(item.publication_day is None for item in decoded)
    schema_contract = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in ARROW_SCHEMA
    ]
    return {
        "schema_version": PARTITION_CHECKPOINT_SCHEMA,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "scan_checkpoint_tree_sha256": scan_tree_sha256,
        "staging_shard_tree_sha256": staging_tree_sha256,
        "partition_id": partition_id,
        "partition_formula": "work_id_numeric_mod_4096",
        "output_relative_path": relative,
        "output_rows": len(decoded),
        "minimum_work_id_numeric": min(ids) if ids else None,
        "maximum_work_id_numeric": max(ids) if ids else None,
        "output_schema": schema_contract,
        "output_schema_sha256": sha256_bytes(canonical_json_bytes(schema_contract)),
        "pre_parquet_sorted_record_sha256": record_digest(ordered),
        "decoded_parquet_record_sha256": record_digest(decoded),
        "disposition_counts": disposition_counts,
        "nullability_counts": {
            "publication_day_null_rows": null_count,
            "publication_year_null_rows": null_count,
        },
        "output_bytes": len(payload),
        "output_sha256": sha256_bytes(payload),
        "strictly_increasing_ids": True,
        "operator_paths_redacted": True,
    }


def _partition_checkpoint_path(output_dir: Path, partition_id: int) -> Path:
    return output_dir / "checkpoints/partition" / f"part-{partition_id:04d}.json"


def _validate_partition_checkpoint(
    checkpoint: Mapping[str, Any],
    expected_core: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(checkpoint)
    digest = value.pop("checkpoint_sha256", None)
    require(
        digest == sha256_bytes(canonical_json_bytes(value)),
        "partition checkpoint self hash mismatch",
    )
    require(value == expected_core, "partition checkpoint differs from current output")
    return dict(checkpoint)


def _partition_checkpoint_tree(
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    commitments = [
        {
            "partition_id": item["partition_id"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "output_sha256": item["output_sha256"],
            "pre_parquet_sorted_record_sha256": item[
                "pre_parquet_sorted_record_sha256"
            ],
            "decoded_parquet_record_sha256": item["decoded_parquet_record_sha256"],
        }
        for item in checkpoints
    ]
    return {
        "checkpoint_count": len(commitments),
        "partition_checkpoint_tree_sha256": _tree_digest(commitments),
        "commitments": commitments,
    }


def _spine_manifest(
    checkpoints: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
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
    return {
        "schema_version": SPINE_MANIFEST_SCHEMA,
        "scope": SYNTHETIC_SCOPE,
        "production_status": PRODUCTION_STATUS,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "scan_checkpoint_tree_sha256": scan_tree_sha256,
        "staging_shard_tree_sha256": staging_tree_sha256,
        "partition_count": len(partitions),
        "rows": sum(int(item["rows"]) for item in partitions),
        "bytes": sum(int(item["bytes"]) for item in partitions),
        "spine_data_tree_sha256": _tree_digest(partitions),
        "partitions": partitions,
        "operator_paths_redacted": True,
    }


def _execute_partitions(
    output_dir: Path,
    results: Sequence[ScanResult],
    scan_checkpoints: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    resume: bool,
    pause_after_new_partitions: int | None,
) -> tuple[list[dict[str, Any]], bool]:
    require(
        pause_after_new_partitions is None or pause_after_new_partitions > 0,
        "partition pause count must be positive",
    )
    scan_tree = _scan_checkpoint_tree(scan_checkpoints)
    staging = _staging_manifest(scan_checkpoints, plan)
    by_partition: list[list[Record]] = [[] for _ in range(PARTITION_COUNT)]
    for result in results:
        for record in result.records:
            by_partition[record.work_id_numeric % PARTITION_COUNT].append(record)
    require(
        sum(len(items) for items in by_partition)
        == sum(len(result.records) for result in results),
        "partition assignment does not conserve rows",
    )

    checkpoints: list[dict[str, Any]] = []
    gap_seen = False
    for partition_id in range(PARTITION_COUNT):
        checkpoint_path = _partition_checkpoint_path(output_dir, partition_id)
        output_path = output_dir / f"spine/part-{partition_id:04d}.parquet"
        exists = checkpoint_path.exists() or checkpoint_path.is_symlink()
        output_exists = output_path.exists() or output_path.is_symlink()
        require(exists == output_exists, "orphan partition checkpoint or output")
        if exists:
            require(resume, "existing partition requires resume")
            require(
                not gap_seen, "partition checkpoint tree is not a contiguous prefix"
            )
            expected = _expected_partition_core_without_publish(
                output_dir,
                partition_id,
                by_partition[partition_id],
                plan=plan,
                scan_tree_sha256=scan_tree["checkpoint_tree_sha256"],
                staging_tree_sha256=staging["staging_shard_tree_sha256"],
            )
            checkpoints.append(
                _validate_partition_checkpoint(
                    read_json(checkpoint_path, "partition checkpoint"),
                    expected,
                )
            )
        else:
            gap_seen = True

    new_partitions = 0
    for partition_id in range(len(checkpoints), PARTITION_COUNT):
        core, _ = _partition_core(
            output_dir,
            partition_id,
            by_partition[partition_id],
            plan=plan,
            scan_tree_sha256=scan_tree["checkpoint_tree_sha256"],
            staging_tree_sha256=staging["staging_shard_tree_sha256"],
        )
        checkpoint = _with_self_hash(core, "checkpoint_sha256")
        atomic_write_json(
            _partition_checkpoint_path(output_dir, partition_id),
            checkpoint,
        )
        checkpoints.append(checkpoint)
        new_partitions += 1
        if partition_id % 256 == 255 or partition_id == PARTITION_COUNT - 1:
            atomic_write_json(
                output_dir / "run_state.json",
                _state(
                    plan,
                    "RUNNING_PARTITION",
                    completed_scan_files=len(scan_checkpoints),
                    completed_partitions=len(checkpoints),
                ),
            )
        if (
            pause_after_new_partitions is not None
            and new_partitions >= pause_after_new_partitions
            and len(checkpoints) < PARTITION_COUNT
        ):
            atomic_write_json(
                output_dir / "run_state.json",
                _state(
                    plan,
                    "PAUSED_PARTITION",
                    completed_scan_files=len(scan_checkpoints),
                    completed_partitions=len(checkpoints),
                ),
            )
            return checkpoints, True

    manifest = _spine_manifest(
        checkpoints,
        plan,
        scan_tree_sha256=scan_tree["checkpoint_tree_sha256"],
        staging_tree_sha256=staging["staging_shard_tree_sha256"],
    )
    manifest_path = output_dir / "work_time_spine_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        require(resume, "existing spine manifest requires resume")
        require(
            read_json(manifest_path, "spine manifest") == manifest,
            "spine manifest differs",
        )
    else:
        atomic_write_json(manifest_path, manifest)
    return checkpoints, False


def _claim_boundary() -> str:
    return (
        "This is a SYNTHETIC_FIXTURE_ONLY execution of the canonical work-time "
        "spine ABI. It validates the 16-byte row encoding, exact 4,096-way "
        "partition surface, checkpoint/resume rules, and receipt-last lifecycle "
        "against synthetic source bytes and an exact temporal-v3-only synthetic "
        "parent ABI. Temporal v1 and v2 receipts are rejected. Temporal v3 does "
        "not publish a row-level disposition artifact or digest, so this execution "
        "cannot establish direct row-level identity to a temporal-parent product. "
        "Production remains PRODUCTION_NOT_RUN. It does not establish the "
        "492,293,206-row production component, historical content identity, W0 "
        "entity validity, citation chronology, author or candidate validity, "
        "G-1, G0, empirical G1, an NABM effect, or any scientific effect."
    )


def _report(
    plan: Mapping[str, Any],
    temporal: Mapping[str, Any],
    scan_checkpoints: Sequence[Mapping[str, Any]],
    partition_checkpoints: Sequence[Mapping[str, Any]],
    global_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    scan_tree = _scan_checkpoint_tree(scan_checkpoints)
    staging = _staging_manifest(scan_checkpoints, plan)
    partition_tree = _partition_checkpoint_tree(partition_checkpoints)
    spine = _spine_manifest(
        partition_checkpoints,
        plan,
        scan_tree_sha256=scan_tree["checkpoint_tree_sha256"],
        staging_tree_sha256=staging["staging_shard_tree_sha256"],
    )
    parents = dict(plan["parents"])
    gates = {
        "canonical_work_time_spine_component": "SYNTHETIC_CONTRACT_ONLY",
        "production_canonical_work_time_spine_component": PRODUCTION_STATUS,
        "current_content_baseline_component": "SYNTHETIC_PARENT_VALIDATED",
        "main_temporal_validity_component": "SYNTHETIC_PARENT_VALIDATED",
        "row_level_spine_commitment": "SYNTHETIC_ONLY",
        "row_level_temporal_parent_artifact_identity": ("NOT_AVAILABLE_TEMPORAL_V3"),
        "historical_content_identity": "UNVERIFIED",
        "raw_source_id_validity": "BLOCKED",
        "w_zero_entity_validity": "UNRESOLVED",
        "citation_edge_chronology": "PENDING_NOT_IN_THIS_COMPONENT",
        "author_entity_validity": "PENDING",
        "candidate_eligibility": "PENDING",
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }
    deterministic = {
        "parents": parents,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "scan_checkpoint_tree_sha256": scan_tree["checkpoint_tree_sha256"],
        "staging_shard_tree_sha256": staging["staging_shard_tree_sha256"],
        "partition_checkpoint_tree_sha256": partition_tree[
            "partition_checkpoint_tree_sha256"
        ],
        "spine_data_tree_sha256": spine["spine_data_tree_sha256"],
        "identifier_accounting": global_evidence["identifier_accounting"],
        "publication_accounting": global_evidence["publication_accounting"],
        "gate_status": gates,
        "claim_boundary": _claim_boundary(),
    }
    result_fingerprint = sha256_bytes(canonical_json_bytes(deterministic))
    return {
        "schema_version": REPORT_SCHEMA,
        "status": SYNTHETIC_STATUS,
        "scope": {
            "execution_scope": SYNTHETIC_SCOPE,
            "production_status": PRODUCTION_STATUS,
            "synthetic_source_files": len(plan["tasks"]),
            "synthetic_raw_rows": global_evidence["metrics"]["raw_rows"],
            "synthetic_retained_rows": global_evidence["identifier_accounting"][
                "retained_rows"
            ],
            "partition_count": PARTITION_COUNT,
            "receipt_artifact_count": plan["synthetic_receipt_artifact_count"],
            "production_artifact_invariant_n_493": 9184,
            "production_row_invariant_met": False,
        },
        **deterministic,
        "runtime_fingerprint": plan["runtime_fingerprint"],
        "production_component_pass": False,
        "pass_g1": False,
        "provenance": {
            "config_sha256": plan["config_sha256"],
            "tool_sha256": plan["tool_sha256"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "result_fingerprint_sha256": result_fingerprint,
            "temporal_parent_result_fingerprint_sha256": temporal["receipt"][
                "result_fingerprint_sha256"
            ],
            "operator_paths_redacted": True,
        },
    }


def _report_fingerprint(report: Mapping[str, Any]) -> str:
    keys = (
        "parents",
        "task_fingerprint_sha256",
        "source_tree_sha256",
        "scan_checkpoint_tree_sha256",
        "staging_shard_tree_sha256",
        "partition_checkpoint_tree_sha256",
        "spine_data_tree_sha256",
        "identifier_accounting",
        "publication_accounting",
        "gate_status",
        "claim_boundary",
    )
    return sha256_bytes(canonical_json_bytes({key: report[key] for key in keys}))


def render_markdown(report: Mapping[str, Any]) -> str:
    scope = report["scope"]
    return "\n".join(
        [
            "# OpenAlex Canonical Work-Time Spine Synthetic Contract",
            "",
            f"- Status: `{report['status']}`",
            f"- Scope: `{scope['execution_scope']}`",
            f"- Production: `{scope['production_status']}`",
            f"- Synthetic files / retained rows: `{scope['synthetic_source_files']}` / `{scope['synthetic_retained_rows']}`",
            f"- Exact partitions: `{scope['partition_count']}`",
            f"- Synthetic receipt artifacts: `{scope['receipt_artifact_count']}`",
            "- Production component pass: `false`",
            "- G-1 / G0 / empirical G1 / scientific effect: `PENDING / PENDING / NOT_RUN / NOT_ESTABLISHED`",
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )


def _receipt_artifact_paths(plan: Mapping[str, Any]) -> set[str]:
    paths = {
        "audit_plan.json",
        "run_state.json",
        "staging_manifest.json",
        "work_time_spine_manifest.json",
        "work_time_spine_audit.json",
        "work_time_spine_audit.md",
    }
    for task in plan["tasks"]:
        token = sha256_bytes(str(task["relative_path"]).encode("utf-8"))
        paths.add(f"checkpoints/scan/main/{token}.json")
        paths.add(f"staging/main/{token}.bin")
    for partition_id in range(PARTITION_COUNT):
        paths.add(f"checkpoints/partition/part-{partition_id:04d}.json")
        paths.add(f"spine/part-{partition_id:04d}.parquet")
    require(
        len(paths) == 6 + 2 * len(plan["tasks"]) + 2 * PARTITION_COUNT,
        "receipt artifact formula mismatch",
    )
    require(
        len(paths) == plan["synthetic_receipt_artifact_count"],
        "plan synthetic artifact count mismatch",
    )
    return paths


def _published_regular_paths(output_dir: Path) -> set[str]:
    paths: set[str] = set()
    for path in output_dir.rglob("*"):
        relative = path.relative_to(output_dir).as_posix()
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "output tree contains a symlink")
        if stat.S_ISREG(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o600, "output file mode drift")
            paths.add(relative)
        elif stat.S_ISDIR(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o700, "output directory mode drift")
        else:
            raise SpineAuditError("output tree contains a non-file entry")
    return paths


def _output_receipt(
    output_dir: Path,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _receipt_artifact_paths(plan)
    artifacts = {
        relative: _artifact_contract(output_dir, relative) for relative in sorted(paths)
    }
    return {
        "schema_version": OUTPUTS_SCHEMA,
        "status": SYNTHETIC_STATUS,
        "scope": SYNTHETIC_SCOPE,
        "production_status": PRODUCTION_STATUS,
        "production_component_pass": False,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
        "scan_checkpoint_tree_sha256": report["scan_checkpoint_tree_sha256"],
        "staging_shard_tree_sha256": report["staging_shard_tree_sha256"],
        "partition_checkpoint_tree_sha256": report["partition_checkpoint_tree_sha256"],
        "spine_data_tree_sha256": report["spine_data_tree_sha256"],
        "artifacts": artifacts,
        "synthetic_artifact_count": len(artifacts),
        "production_artifact_invariant_n_493": 9184,
        "completion_contract": COMPLETION_ABI,
        "operator_paths_redacted": True,
    }


def _validate_output_layout(
    output_dir: Path,
    *,
    final: bool,
    plan: Mapping[str, Any] | None = None,
) -> None:
    require(
        output_dir.is_dir()
        and not output_dir.is_symlink()
        and stat.S_IMODE(output_dir.lstat().st_mode) == 0o700,
        "output root mode drift",
    )
    names = {path.name for path in output_dir.iterdir()}
    require(
        not (names - ALLOWED_TOP_LEVEL), "output contains undeclared top-level entries"
    )
    if final:
        require(names == ALLOWED_TOP_LEVEL, "final top-level entry set differs")
        require(plan is not None, "final layout requires plan")
        expected = _receipt_artifact_paths(plan) | {
            "audit.lock",
            "work_time_spine_audit.outputs.json",
        }
        require(
            _published_regular_paths(output_dir) == expected,
            "final published file set differs",
        )


def _validate_receipt_artifacts(
    output_dir: Path,
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    expected = _receipt_artifact_paths(plan)
    artifacts = require_mapping(receipt.get("artifacts"), "receipt artifacts")
    require(set(artifacts) == expected, "receipt artifact set differs")
    require(
        receipt.get("synthetic_artifact_count") == len(expected),
        "receipt artifact count differs",
    )
    for relative in sorted(expected):
        contract = require_mapping(artifacts[relative], "receipt artifact")
        require(
            set(contract) == {"relative_path", "bytes", "sha256"},
            "receipt artifact key drift",
        )
        require(
            contract.get("relative_path") == relative, "receipt artifact path mismatch"
        )
        require(
            _artifact_contract(output_dir, relative) == contract,
            "receipt artifact bytes/hash mismatch",
        )


def _revalidate_spine_artifacts_from_disk(
    output_dir: Path,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    """Recompute every local checkpoint/tree bridge without source payload I/O."""

    scan_checkpoints: list[dict[str, Any]] = []
    all_records: list[Record] = []
    for task in plan["tasks"]:
        relative = str(task["relative_path"])
        checkpoint = read_json(
            _scan_checkpoint_path(output_dir, relative),
            "published scan checkpoint",
        )
        payload = dict(checkpoint)
        checkpoint_sha = payload.pop("checkpoint_sha256", None)
        require(
            checkpoint_sha == sha256_bytes(canonical_json_bytes(payload)),
            "published scan checkpoint self hash mismatch",
        )
        require(
            checkpoint.get("schema_version") == SCAN_CHECKPOINT_SCHEMA,
            "published scan checkpoint schema mismatch",
        )
        require(
            checkpoint.get("task_fingerprint_sha256")
            == plan["task_fingerprint_sha256"],
            "published scan checkpoint plan mismatch",
        )
        require(
            checkpoint.get("spine_task_sha256") == task["spine_task_sha256"],
            "published scan task mismatch",
        )
        require(
            checkpoint.get("temporal_parent_task_sha256")
            == task["temporal_task_sha256"],
            "published temporal task binding mismatch",
        )
        require(
            checkpoint.get("temporal_parent_checkpoint_sha256")
            == task["temporal_checkpoint_sha256"],
            "published temporal checkpoint binding mismatch",
        )
        require(
            checkpoint.get("relative_path") == relative, "published scan path mismatch"
        )
        require(
            checkpoint.get("source_rich_identity") == task["source_rich_identity"],
            "published scan source identity mismatch",
        )
        require(
            checkpoint.get("footer_rows") == task["footer_rows"],
            "published scan footer mismatch",
        )
        require(
            checkpoint.get("projected_schema_sha256")
            == task["projected_schema_sha256"],
            "published scan schema binding mismatch",
        )
        require(
            checkpoint.get("disposition_metrics_sha256")
            == sha256_bytes(
                canonical_json_bytes(checkpoint.get("disposition_metrics"))
            ),
            "published disposition metrics hash mismatch",
        )
        staging_relative = safe_relative_path(
            checkpoint.get("staging_shard_relative_path"),
            "published staging shard",
        )
        expected_token = sha256_bytes(relative.encode("utf-8"))
        require(
            staging_relative.as_posix() == f"staging/main/{expected_token}.bin",
            "published staging shard path mismatch",
        )
        staging_path = secure_regular(
            output_dir,
            staging_relative,
            field="published staging shard",
        )
        staging_payload = staging_path.read_bytes()
        require(
            len(staging_payload) == checkpoint.get("staging_shard_bytes"),
            "published staging byte count mismatch",
        )
        require(
            sha256_bytes(staging_payload) == checkpoint.get("staging_shard_sha256"),
            "published staging hash mismatch",
        )
        records = decode_staging(staging_payload)
        require(
            len(records) == checkpoint.get("staging_shard_rows"),
            "published staging row count mismatch",
        )
        require(
            record_digest(records)
            == checkpoint.get("row_level_disposition_payload_sha256"),
            "published staging record digest mismatch",
        )
        observed_counts = {name: 0 for name in DISPOSITION_NAMES.values()}
        for record in records:
            observed_counts[DISPOSITION_NAMES[record.temporal_disposition]] += 1
        require(
            observed_counts == checkpoint.get("disposition_counts"),
            "published staging disposition count mismatch",
        )
        scan_checkpoints.append(checkpoint)
        all_records.extend(records)

    scan_tree = _scan_checkpoint_tree(scan_checkpoints)
    require(
        scan_tree["checkpoint_tree_sha256"]
        == report.get("scan_checkpoint_tree_sha256"),
        "published scan tree/report mismatch",
    )
    staging_manifest = _staging_manifest(scan_checkpoints, plan)
    require(
        read_json(output_dir / "staging_manifest.json", "published staging manifest")
        == staging_manifest,
        "published staging manifest mismatch",
    )
    require(
        staging_manifest["staging_shard_tree_sha256"]
        == report.get("staging_shard_tree_sha256"),
        "published staging tree/report mismatch",
    )

    by_partition: list[list[Record]] = [[] for _ in range(PARTITION_COUNT)]
    for record in all_records:
        by_partition[record.work_id_numeric % PARTITION_COUNT].append(record)
    partition_checkpoints: list[dict[str, Any]] = []
    for partition_id in range(PARTITION_COUNT):
        expected = _expected_partition_core_without_publish(
            output_dir,
            partition_id,
            by_partition[partition_id],
            plan=plan,
            scan_tree_sha256=scan_tree["checkpoint_tree_sha256"],
            staging_tree_sha256=staging_manifest["staging_shard_tree_sha256"],
        )
        checkpoint = read_json(
            _partition_checkpoint_path(output_dir, partition_id),
            "published partition checkpoint",
        )
        partition_checkpoints.append(
            _validate_partition_checkpoint(checkpoint, expected)
        )

    partition_tree = _partition_checkpoint_tree(partition_checkpoints)
    require(
        partition_tree["partition_checkpoint_tree_sha256"]
        == report.get("partition_checkpoint_tree_sha256"),
        "published partition tree/report mismatch",
    )
    spine_manifest = _spine_manifest(
        partition_checkpoints,
        plan,
        scan_tree_sha256=scan_tree["checkpoint_tree_sha256"],
        staging_tree_sha256=staging_manifest["staging_shard_tree_sha256"],
    )
    require(
        read_json(
            output_dir / "work_time_spine_manifest.json", "published spine manifest"
        )
        == spine_manifest,
        "published spine manifest mismatch",
    )
    require(
        spine_manifest["spine_data_tree_sha256"]
        == report.get("spine_data_tree_sha256"),
        "published spine data tree/report mismatch",
    )
    require(
        spine_manifest["rows"] == len(all_records),
        "published staging/partition row conservation mismatch",
    )
    require(
        len({record.work_id_numeric for record in all_records}) == len(all_records),
        "published staging contains duplicate IDs",
    )
    require(
        len(all_records)
        == report.get("identifier_accounting", {}).get("retained_rows"),
        "published retained-row/report mismatch",
    )


def _validate_final_publication(
    output_dir: Path,
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    _validate_output_layout(output_dir, final=True, plan=plan)
    disk_plan = read_json(output_dir / "audit_plan.json", "final plan")
    disk_report = read_json(output_dir / "work_time_spine_audit.json", "final report")
    disk_receipt = read_json(
        output_dir / "work_time_spine_audit.outputs.json",
        "final receipt",
    )
    state = read_json(output_dir / "run_state.json", "final state")
    require(disk_plan == plan, "final plan differs")
    require(disk_report == report, "final report differs")
    require(disk_receipt == receipt, "final receipt differs")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(receipt.get("schema_version") == OUTPUTS_SCHEMA, "receipt schema mismatch")
    require(
        report.get("status") == receipt.get("status") == SYNTHETIC_STATUS,
        "synthetic status mismatch",
    )
    require(
        report.get("scope", {}).get("execution_scope") == SYNTHETIC_SCOPE,
        "report synthetic scope mismatch",
    )
    require(receipt.get("scope") == SYNTHETIC_SCOPE, "receipt synthetic scope mismatch")
    require(
        receipt.get("production_status") == PRODUCTION_STATUS,
        "receipt production status mismatch",
    )
    require(
        receipt.get("production_component_pass") is False,
        "receipt production overclaim",
    )
    result_fp = _report_fingerprint(report)
    require(
        report.get("provenance", {}).get("result_fingerprint_sha256") == result_fp,
        "report result fingerprint mismatch",
    )
    require(
        receipt.get("result_fingerprint_sha256") == result_fp,
        "receipt result fingerprint mismatch",
    )
    require(
        state.get("status") == "COMPLETE_PENDING_RECEIPT", "final state status mismatch"
    )
    require(
        state.get("result_fingerprint_sha256") == result_fp,
        "final state result mismatch",
    )
    require(
        state.get("completion_receipt_relative_path")
        == "work_time_spine_audit.outputs.json",
        "final state receipt path mismatch",
    )
    require(
        state.get("complete_only_if_receipt_exists") is True,
        "final state receipt requirement mismatch",
    )
    require(
        (output_dir / "work_time_spine_audit.md").read_text(encoding="utf-8")
        == render_markdown(report),
        "final Markdown differs",
    )
    _validate_receipt_artifacts(output_dir, plan, receipt)
    _revalidate_spine_artifacts_from_disk(output_dir, plan, report)


def _validate_terminal_completion(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    config_identity: Mapping[str, int],
    temporal: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a completion without opening any source Parquet payload."""

    plan = read_json(output_dir / "audit_plan.json", "terminal plan")
    report = read_json(output_dir / "work_time_spine_audit.json", "terminal report")
    receipt = read_json(
        output_dir / "work_time_spine_audit.outputs.json",
        "terminal receipt",
    )
    require(plan.get("schema_version") == PLAN_SCHEMA, "terminal plan schema mismatch")
    payload = dict(plan)
    task_fp = payload.pop("task_fingerprint_sha256", None)
    require(
        task_fp == sha256_bytes(canonical_json_bytes(payload)),
        "terminal plan fingerprint mismatch",
    )
    require(plan.get("config_sha256") == config_sha256, "terminal config hash mismatch")
    require(
        plan.get("config_rich_identity") == config_identity,
        "terminal config identity mismatch",
    )
    require(
        plan.get("tool_sha256") == sha256_file(Path(__file__).resolve()),
        "terminal tool hash mismatch",
    )
    require(
        plan.get("tool_rich_identity")
        == _regular_identity(Path(__file__).resolve(), field="spine tool"),
        "terminal tool identity mismatch",
    )
    require(
        plan.get("runtime_fingerprint") == _runtime_fingerprint(),
        "terminal runtime fingerprint mismatch",
    )
    require(
        plan.get("partition_count") == PARTITION_COUNT,
        "terminal partition count mismatch",
    )
    expected_plan = _build_plan(
        config,
        config_sha256=config_sha256,
        config_identity=config_identity,
        temporal=temporal,
        batch_size=int(plan["batch_size"]),
        partition_sort_chunk_rows=int(plan["partition_sort_chunk_rows"]),
    )
    require(plan == expected_plan, "terminal plan differs from current frozen inputs")
    _validate_final_publication(
        output_dir,
        plan=plan,
        report=report,
        receipt=receipt,
    )
    return report


def _invalidation_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _state(
        plan,
        FAILURE_STATUS,
        completed_scan_files=len(plan["tasks"]),
        completed_partitions=PARTITION_COUNT,
        result_fingerprint_sha256=None,
    )


def _invalidate_finalization(output_dir: Path, plan: Mapping[str, Any]) -> None:
    """Invalidate in the frozen order and aggregate cleanup failures."""

    failures: list[BaseException] = []
    # 1. Authoritative receipt removal is attempted first.
    try:
        _remove_if_exists(output_dir / "work_time_spine_audit.outputs.json")
    except (
        BaseException
    ) as exc:  # pragma: no cover - exercised by injected cleanup tests
        failures.append(exc)
    # 2. Failure state contains no result fingerprint.
    try:
        atomic_write_json(output_dir / "run_state.json", _invalidation_state(plan))
    except (
        BaseException
    ) as exc:  # pragma: no cover - exercised by injected cleanup tests
        failures.append(exc)
    # 3. Descriptive pass surfaces are independently removed.
    for name in ("work_time_spine_audit.json", "work_time_spine_audit.md"):
        try:
            _remove_if_exists(output_dir / name)
        except BaseException as exc:  # pragma: no cover
            failures.append(exc)
    if failures:
        raise SpineAuditError(
            f"finalization invalidation had {len(failures)} cleanup failure(s)"
        ) from failures[0]


def _prepare_output(
    output_dir: Path,
    *,
    source_root: Path,
    temporal_output_root: Path,
) -> None:
    output_dir = _absolute(output_dir)
    for other, name in (
        (source_root, "source"),
        (REPO_ROOT, "repository"),
        (temporal_output_root, "temporal parent"),
    ):
        require(not _paths_overlap(output_dir, other), f"output overlaps {name}")
    _assert_no_symlink_components(output_dir, field="output")
    _mkdir_private(output_dir)
    for relative in (
        "checkpoints",
        "checkpoints/scan",
        "checkpoints/scan/main",
        "checkpoints/partition",
        "staging",
        "staging/main",
        "spine",
    ):
        _mkdir_private(output_dir / relative)
    _validate_output_layout(output_dir, final=False)


def run_audit(
    *,
    config_path: Path,
    synthetic_source_root: Path,
    content_receipt_path: Path,
    temporal_receipt_path: Path,
    temporal_config_path: Path,
    temporal_repo_root: Path,
    output_dir: Path,
    batch_size: int = 262144,
    partition_sort_chunk_rows: int = 1048576,
    resume: bool = False,
    pause_after_new_scan_files: int | None = None,
    pause_after_new_partitions: int | None = None,
    synthetic_fixture_only: bool = False,
) -> dict[str, Any]:
    """Execute the synthetic ABI.  Production roots/configs are refused."""

    require(synthetic_fixture_only, "explicit synthetic-fixture-only flag is required")
    config_path = _absolute(config_path)
    require(
        config_path != DEFAULT_CONFIG.resolve(),
        "repository production config is not executable",
    )
    config, config_sha, config_identity = _load_yaml_snapshot(config_path)
    _validate_config(config)
    source = require_mapping(config.get("source"), "source")
    require(
        source.get("root_id") != PRODUCTION_ROOT_ID,
        "production OpenAlex root is forbidden",
    )
    require(
        str(source.get("root_id", "")).startswith("synthetic-"),
        "source root is not an explicit synthetic fixture",
    )
    source_root = _absolute(synthetic_source_root)
    _assert_no_symlink_components(source_root, field="synthetic source root")
    require(
        source_root.is_dir() and not source_root.is_symlink(), "source root is invalid"
    )
    resolved_source_root = source_root.resolve(strict=True)
    require(
        PRODUCTION_ROOT_ID not in resolved_source_root.parts,
        "production OpenAlex source path is forbidden",
    )
    output_dir = _absolute(output_dir)
    _assert_no_symlink_components(output_dir, field="output")

    # Terminal path validates parent output bytes and the exact published spine
    # tree, but deliberately does not call the temporal source-payload validator.
    receipt_path = output_dir / "work_time_spine_audit.outputs.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        require(resume, "completed output requires resume")
        with output_lock(output_dir):
            disk_plan: dict[str, Any] | None = None
            try:
                disk_plan = read_json(output_dir / "audit_plan.json", "terminal plan")
                temporal = _validate_temporal_parent(
                    config,
                    source_root=source_root,
                    temporal_receipt_path=temporal_receipt_path,
                    temporal_config_path=temporal_config_path,
                    temporal_repo_root=temporal_repo_root,
                    content_receipt_path=content_receipt_path,
                    revalidate_source_payload=False,
                )
                return _validate_terminal_completion(
                    output_dir,
                    config=config,
                    config_sha256=config_sha,
                    config_identity=config_identity,
                    temporal=temporal,
                )
            except BaseException:
                if disk_plan is not None:
                    _invalidate_finalization(output_dir, disk_plan)
                else:
                    for name in FINAL_REPORT_NAMES:
                        _remove_if_exists(output_dir / name)
                raise

    temporal = _validate_temporal_parent(
        config,
        source_root=source_root,
        temporal_receipt_path=temporal_receipt_path,
        temporal_config_path=temporal_config_path,
        temporal_repo_root=temporal_repo_root,
        content_receipt_path=content_receipt_path,
        revalidate_source_payload=True,
    )
    _prepare_output(
        output_dir,
        source_root=source_root,
        temporal_output_root=temporal["output_root"],
    )
    with output_lock(output_dir):
        plan = _build_plan(
            config,
            config_sha256=config_sha,
            config_identity=config_identity,
            temporal=temporal,
            batch_size=batch_size,
            partition_sort_chunk_rows=partition_sort_chunk_rows,
        )
        plan_path = output_dir / "audit_plan.json"
        if plan_path.exists() or plan_path.is_symlink():
            require(resume, "existing plan requires resume")
            require(
                read_json(plan_path, "existing plan") == plan, "resume plan differs"
            )
        else:
            atomic_write_json(plan_path, plan)
        atomic_write_json(
            output_dir / "run_state.json",
            _state(
                plan,
                "RUNNING_SCAN",
                completed_scan_files=0,
                completed_partitions=0,
            ),
        )

        results, scan_checkpoints, paused, global_evidence = _execute_scan(
            output_dir,
            source_root,
            config,
            temporal,
            plan,
            resume=resume,
            pause_after_new_scan_files=pause_after_new_scan_files,
        )
        if paused:
            return read_json(output_dir / "run_state.json", "paused state")
        partition_checkpoints, paused = _execute_partitions(
            output_dir,
            results,
            scan_checkpoints,
            plan,
            resume=resume,
            pause_after_new_partitions=pause_after_new_partitions,
        )
        if paused:
            return read_json(output_dir / "run_state.json", "paused state")

        report = _report(
            plan,
            temporal,
            scan_checkpoints,
            partition_checkpoints,
            global_evidence,
        )
        result_fp = report["provenance"]["result_fingerprint_sha256"]
        finalization_started = False
        try:
            finalization_started = True
            atomic_write_json(output_dir / "work_time_spine_audit.json", report)
            atomic_write_bytes(
                output_dir / "work_time_spine_audit.md",
                render_markdown(report).encode("utf-8"),
            )
            atomic_write_json(
                output_dir / "run_state.json",
                _state(
                    plan,
                    "COMPLETE_PENDING_RECEIPT",
                    completed_scan_files=len(scan_checkpoints),
                    completed_partitions=len(partition_checkpoints),
                    result_fingerprint_sha256=result_fp,
                ),
            )

            # Revalidate parents and source payload at finalization before the
            # authoritative receipt is written.
            final_temporal = _validate_temporal_parent(
                config,
                source_root=source_root,
                temporal_receipt_path=temporal_receipt_path,
                temporal_config_path=temporal_config_path,
                temporal_repo_root=temporal_repo_root,
                content_receipt_path=content_receipt_path,
                revalidate_source_payload=True,
            )
            require(
                final_temporal["receipt_sha256"] == temporal["receipt_sha256"],
                "temporal parent changed before finalization",
            )
            receipt = _output_receipt(output_dir, plan, report)
            # The receipt is the last publication write.
            atomic_write_json(
                output_dir / "work_time_spine_audit.outputs.json",
                receipt,
            )
            _validate_final_publication(
                output_dir,
                plan=plan,
                report=report,
                receipt=receipt,
            )
        except BaseException as exc:
            if finalization_started:
                try:
                    _invalidate_finalization(output_dir, plan)
                except BaseException as cleanup_exc:
                    raise SpineAuditError(
                        "finalization failed and invalidation was incomplete"
                    ) from cleanup_exc
            if isinstance(exc, SpineAuditError):
                raise
            raise SpineAuditError("synthetic spine finalization failed") from exc
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--synthetic-source-root", type=Path, required=True)
    parser.add_argument("--content-receipt", type=Path, required=True)
    parser.add_argument("--temporal-receipt", type=Path, required=True)
    parser.add_argument("--temporal-config", type=Path, required=True)
    parser.add_argument("--temporal-repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--partition-sort-chunk-rows", type=int, default=1048576)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic-fixture-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_audit(
            config_path=args.config,
            synthetic_source_root=args.synthetic_source_root,
            content_receipt_path=args.content_receipt,
            temporal_receipt_path=args.temporal_receipt,
            temporal_config_path=args.temporal_config,
            temporal_repo_root=args.temporal_repo_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            partition_sort_chunk_rows=args.partition_sort_chunk_rows,
            resume=args.resume,
            synthetic_fixture_only=args.synthetic_fixture_only,
        )
    except SpineAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
