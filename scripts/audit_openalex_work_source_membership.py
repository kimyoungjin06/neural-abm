#!/usr/bin/env python3
"""Audit OpenAlex work-source membership, target coverage, and author-ID use.

This command performs a projected data-page scan of the frozen OpenAlex works
root.  It keeps three claims separate:

* every child ``id`` is a grammar-valid member of the frozen main-work view,
  apart from the exact predeclared W-1 quarantine;
* referenced/related work targets are measured as open-world snapshot coverage;
* authorship author-ID missingness is profiled at the work level without
  claiming author-entity validity.

Passing this component is not G-1 data readiness and is not empirical G1.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import re
import resource
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import audit_openalex_sentinel_incidence as sentinel  # noqa: E402
from scripts import recount_openalex_g1 as base  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_work_source_membership_v0.yaml"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "experiments/runs/20260716_openalex_g1_source_inventory/source_manifest.json"
)
CONFIG_SCHEMA_VERSION = "openalex_g1_work_source_membership_config.v1"
PLAN_SCHEMA_VERSION = "openalex_g1_work_source_membership_plan.v1"
CHECKPOINT_SCHEMA_VERSION = "openalex_g1_work_source_membership_checkpoint.v1"
REPORT_SCHEMA_VERSION = "openalex_g1_work_source_membership_report.v1"
OUTPUT_MANIFEST_SCHEMA_VERSION = "openalex_g1_work_source_membership_outputs.v1"

WORK_PREFIX = "https://openalex.org/W"
AUTHOR_PREFIX = "https://openalex.org/A"
FIELD_PREFIX = "https://openalex.org/fields/"
W_MINUS_1 = f"{WORK_PREFIX}-1"
W_ZERO = f"{WORK_PREFIX}0"
UINT64_MAX = int(np.iinfo(np.uint64).max)
YEAR_NULL = int(np.iinfo(np.int16).min)
FIELD_NULL = 0
FIELD_INVALID = 255
CANONICAL_MAIN_TABLE = "openalex_works_20260330"
CANONICAL_AUTHORSHIP = {
    "table": "openalex_works_20260330__authorships",
    "author_id_column": "authorships__author__id",
    "author_orcid_column": "authorships__author__orcid",
}
CANONICAL_TARGET_COLUMNS = {
    "openalex_works_20260330__referenced_works": "referenced_works",
    "openalex_works_20260330__related_works": "related_works",
}
CANONICAL_SUCCESS_STATUS = (
    "PASS_WORK_SOURCE_MEMBERSHIP_AND_PROFILE_AUDIT_WITH_FROZEN_QUARANTINE"
)
CANONICAL_PROFILE_OUTPUTS = {
    "one_way_margins_by_work_class_and_each_dimension",
    "joint_sparse_cube_work_class_by_year_by_primary_field_by_authors_count_band",
    "authors_count_exact_value_mismatch_profile",
    "w_zero_rows_reported_separately",
}
CANONICAL_TARGET_COVERAGE_METRICS = {
    "row_weighted_target_in_snapshot_fraction",
    "unique_target_in_snapshot_fraction",
    "target_not_in_snapshot_rows",
    "target_not_in_snapshot_unique_ids",
    "self_loop_rows",
}
CANONICAL_AUTHOR_WORK_CLASSES = {
    "no_authorship_rows",
    "no_author_id",
    "partial_author_id",
    "complete_author_id",
}
CANONICAL_AUTHOR_ROW_METRICS = {
    "retained_authorship_rows",
    "null_author_id_rows",
    "non_null_author_id_rows",
    "distinct_non_null_author_ids",
    "null_author_orcid_rows",
    "non_null_author_orcid_rows",
}
CANONICAL_AUTHORS_COUNT_CATEGORIES = {
    "main_authors_count_null",
    "exact_match",
    "mismatch",
}
MAX_IN_MEMORY_BLOCKING_LOCATORS_PER_FILE = 100_000

MAIN_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("year", "<i2"),
        ("field", "u1"),
        ("flags", "u1"),
        ("declared_authors", "<u4"),
    ]
)
AUTHOR_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("slots", "<u4"),
        ("null_ids", "<u4"),
        ("valid_ids", "<u4"),
        ("invalid_ids", "<u4"),
    ]
)
TARGET_DTYPE = np.dtype([("id", "<u8"), ("rows", "<u8")])


class MembershipAuditError(RuntimeError):
    """Raised when the audit cannot honor its frozen contract."""


@dataclass(frozen=True)
class FileTask:
    index: int
    table: str
    mode: str
    record: dict[str, Any]
    lineage: str


@dataclass(frozen=True)
class ParsedIDs:
    numeric: np.ndarray
    grammar_valid: np.ndarray
    null: np.ndarray
    w_minus_1: np.ndarray
    malformed_other: np.ndarray


@dataclass
class MasterIDsHandle:
    path: Path
    fd: int
    mapping: mmap.mmap
    array: np.ndarray
    start_identity: dict[str, int]
    sha256: str

    def close(self) -> None:
        self.array = np.empty(0, dtype=np.uint64)
        self.mapping.close()
        os.close(self.fd)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise MembershipAuditError(f"{field} must be a lowercase SHA-256")
    return text


def safe_relative_path(raw: str) -> Path:
    try:
        return base.safe_relative_path(raw)
    except base.RecountError as exc:
        raise MembershipAuditError(str(exc)) from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if partial.exists() or partial.is_symlink():
        raise MembershipAuditError(f"temporary file already exists: {partial.name}")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        partial.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if partial.exists() or partial.is_symlink():
        raise MembershipAuditError(f"temporary file already exists: {partial.name}")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        partial.unlink(missing_ok=True)


def atomic_save_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if path.exists() or path.is_symlink():
        raise MembershipAuditError(f"refusing to replace artifact: {path.name}")
    if partial.exists() or partial.is_symlink():
        raise MembershipAuditError(f"temporary artifact exists: {partial.name}")
    try:
        with partial.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        partial.unlink(missing_ok=True)
    record = {
        "relative_path": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "records": int(len(array)),
        "dtype": array.dtype.descr,
    }
    if (
        array.dtype.fields is None
        and np.issubdtype(array.dtype, np.integer)
        and len(array)
    ):
        record["minimum_numeric_id"] = int(array.min())
        record["maximum_numeric_id"] = int(array.max())
    return record


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MembershipAuditError("config must be a mapping")
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise MembershipAuditError("unsupported config schema_version")
    for key in (
        "source",
        "parent_inventory",
        "parent_grammar_valid_main_id_recount",
        "parent_sentinel_incidence",
        "master_work_id_set",
        "main_accounting",
        "child_source_membership",
        "sentinel_propagation",
        "work_target_coverage",
        "authorship_author_id_work_profile",
        "work_profile_strata",
        "execution",
        "success_contract",
    ):
        if not isinstance(value.get(key), dict):
            raise MembershipAuditError(f"config.{key} must be a mapping")

    source = value["source"]
    inventory_raw = value["parent_inventory"]
    grammar_raw = value["parent_grammar_valid_main_id_recount"]
    sentinel_raw = value["parent_sentinel_incidence"]
    master = value["master_work_id_set"]
    children = value["child_source_membership"]
    target_contracts = value["work_target_coverage"]["tables"]
    authorship = value["authorship_author_id_work_profile"]
    strata = value["work_profile_strata"]
    incidence = value["sentinel_propagation"]["source_incidence_by_table"]

    def expected_incidence(which: str) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for table, table_record in incidence.items():
            record = table_record[which]
            expected_rows = int(record["expected_rows"])
            paths = list(record["expected_files"])
            if expected_rows and len(paths) != 1:
                raise MembershipAuditError(
                    f"{table}.{which} must freeze one affected file"
                )
            output[str(table)] = {
                "files": [
                    {"relative_path": str(relative), "rows": expected_rows}
                    for relative in paths
                ]
            }
        return output

    value["parents"] = {
        "inventory": {
            "schema_version": "openalex_source_inventory.v1",
            "manifest_id": "openalex_g1_source_inventory_v0",
            "manifest_sha256": inventory_raw["manifest_sha256"],
            "source_root_env": source["root_env"],
            "source_root_id": source["root_id"],
            "expected_table_count": inventory_raw["expected_table_count"],
            "inventory_fingerprint_sha256": inventory_raw[
                "inventory_fingerprint_sha256"
            ],
            "stability_fingerprint_sha256": inventory_raw[
                "stability_fingerprint_sha256"
            ],
        },
        "grammar_recount": {
            "summary_path": grammar_raw["summary_relative_path"],
            "summary_sha256": grammar_raw["summary_sha256"],
            "expected_status": grammar_raw["required_status"],
            "sorted_ids_sha256": master["sha256"],
            "sorted_ids_bytes": master["exact_bytes"],
            "sorted_ids_rows": master["exact_count"],
            "min_numeric_id": master["minimum_numeric_id"],
            "max_numeric_id": master["maximum_numeric_id"],
        },
        "sentinel_incidence": {
            "summary_path": sentinel_raw["summary_relative_path"],
            "summary_sha256": sentinel_raw["summary_sha256"],
            "result_fingerprint_sha256": sentinel_raw["result_fingerprint_sha256"],
        },
    }
    value["scope"] = {
        "source_id_column": value["main_accounting"]["source_id_column"],
        "main_table": value["main_accounting"]["table"],
        "child_source_tables": list(children["tables"]),
        "work_target_columns": {
            str(table): str(record["target_column"])
            for table, record in target_contracts.items()
        },
        "authorship": {
            "table": authorship["table"],
            "author_id_column": authorship["author_id_column"],
            "author_orcid_column": authorship["author_orcid_column"],
        },
        "main_strata": {
            "publication_year_column": strata["dimensions"]["publication_year"][
                "column"
            ],
            "primary_field_column": strata["dimensions"]["primary_field"]["column"],
            "declared_authors_column": strata["dimensions"]["authors_count"]["column"],
        },
    }
    value["id_grammar"] = {
        "work_prefix": WORK_PREFIX,
        "author_prefix": AUTHOR_PREFIX,
        "field_prefix": FIELD_PREFIX,
    }
    value["quarantine"] = {
        "raw_value": value["sentinel_propagation"]["w_minus_1"]["raw_value"],
        "w0_value": value["sentinel_propagation"]["w_zero"]["raw_value"],
        "expected_w_minus_1": expected_incidence("w_minus_1"),
        "expected_w0": expected_incidence("w_zero"),
        "main_row_locators": value["sentinel_propagation"]["main_row_locators"],
    }
    value["pending"] = list(value["success_contract"]["pending_after_success"])
    value["execution"].setdefault("workers", 1)
    parents = value["parents"]
    for key in ("inventory", "grammar_recount", "sentinel_incidence"):
        if not isinstance(parents.get(key), dict):
            raise MembershipAuditError(f"config.parents.{key} must be a mapping")
    inventory = parents["inventory"]
    required_inventory = {
        "schema_version",
        "manifest_id",
        "manifest_sha256",
        "source_root_env",
        "source_root_id",
        "expected_table_count",
        "inventory_fingerprint_sha256",
        "stability_fingerprint_sha256",
    }
    missing = sorted(required_inventory - set(inventory))
    if missing:
        raise MembershipAuditError(f"inventory parent fields missing: {missing}")
    for parent, fields in (
        (
            inventory,
            (
                "manifest_sha256",
                "inventory_fingerprint_sha256",
                "stability_fingerprint_sha256",
            ),
        ),
        (parents["grammar_recount"], ("summary_sha256", "sorted_ids_sha256")),
        (
            parents["sentinel_incidence"],
            ("summary_sha256", "result_fingerprint_sha256"),
        ),
    ):
        for field in fields:
            validate_sha256(parent.get(field), field=field)
    scope = value["scope"]
    child_tables = scope.get("child_source_tables")
    target_columns = scope.get("work_target_columns")
    if not isinstance(child_tables, list) or len(child_tables) != 22:
        raise MembershipAuditError("scope must freeze exactly 22 child tables")
    if len(set(map(str, child_tables))) != 22:
        raise MembershipAuditError("child source tables must be unique")
    if not isinstance(target_columns, dict) or len(target_columns) != 2:
        raise MembershipAuditError("exactly two work-target columns are required")
    if str(scope.get("main_table")) != CANONICAL_MAIN_TABLE:
        raise MembershipAuditError("canonical main table differs")
    if str(scope.get("source_id_column")) != "id":
        raise MembershipAuditError("canonical source work-ID column differs")
    if {str(key): str(item) for key, item in target_columns.items()} != (
        CANONICAL_TARGET_COLUMNS
    ):
        raise MembershipAuditError("canonical work-target columns differ")
    if scope.get("authorship") != CANONICAL_AUTHORSHIP:
        raise MembershipAuditError("canonical authorship columns differ")
    if scope.get("main_strata") != {
        "publication_year_column": "publication_year",
        "primary_field_column": "primary_topic__field__id",
        "declared_authors_column": "authors_count",
    }:
        raise MembershipAuditError("canonical main-profile columns differ")
    if int(value["execution"].get("batch_size", 0)) < 1:
        raise MembershipAuditError("execution.batch_size must be positive")
    if int(value["execution"].get("workers", 0)) < 1:
        raise MembershipAuditError("execution.workers must be positive")
    if value["execution"].get("require_output_sha256") is not True:
        raise MembershipAuditError("final output SHA-256 must remain required")
    for required_flag in (
        "scan_all_source_id_data_pages",
        "verify_parent_before_resume",
        "verify_parent_stability_after_run",
        "refuse_symlink_inputs",
        "require_exact_scanned_file_and_row_counts",
        "require_per_file_disposition_counts",
        "require_complete_blocking_identifier_ledger",
    ):
        if value["execution"].get(required_flag) is not True:
            raise MembershipAuditError(f"execution.{required_flag} must remain true")
    resource_safety = value["execution"].get("resource_safety")
    if not isinstance(resource_safety, dict):
        raise MembershipAuditError("execution.resource_safety must be a mapping")
    if int(resource_safety.get("maximum_author_numeric_id_inclusive", -1)) < 1:
        raise MembershipAuditError("author numeric-ID safety cap must be positive")
    if int(resource_safety.get("maximum_estimated_dense_bytes", -1)) < 1:
        raise MembershipAuditError("dense-memory safety cap must be positive")
    if float(resource_safety.get("minimum_available_to_estimated_ratio", 0)) < 1:
        raise MembershipAuditError("available-memory safety ratio must be at least one")
    if set(str(item) for item in strata.get("required_outputs", [])) != (
        CANONICAL_PROFILE_OUTPUTS
    ):
        raise MembershipAuditError("required work-profile outputs differ")
    if (
        set(
            map(
                str,
                value["work_target_coverage"].get("required_coverage_metrics", []),
            )
        )
        != CANONICAL_TARGET_COVERAGE_METRICS
    ):
        raise MembershipAuditError("required target-coverage metrics differ")
    if set(map(str, authorship.get("required_work_classes", {}))) != (
        CANONICAL_AUTHOR_WORK_CLASSES
    ):
        raise MembershipAuditError("required author work classes differ")
    if set(map(str, authorship.get("required_row_metrics", []))) != (
        CANONICAL_AUTHOR_ROW_METRICS
    ):
        raise MembershipAuditError("required author row metrics differ")
    if set(map(str, authorship.get("authors_count_comparison_categories", []))) != (
        CANONICAL_AUTHORS_COUNT_CATEGORIES
    ):
        raise MembershipAuditError("authors-count comparison categories differ")
    grammar = value["id_grammar"]
    if grammar.get("work_prefix") != WORK_PREFIX:
        raise MembershipAuditError("unexpected work-ID prefix")
    if grammar.get("author_prefix") != AUTHOR_PREFIX:
        raise MembershipAuditError("unexpected author-ID prefix")
    if grammar.get("field_prefix") != FIELD_PREFIX:
        raise MembershipAuditError("unexpected field-ID prefix")
    if value["quarantine"].get("raw_value") != W_MINUS_1:
        raise MembershipAuditError("quarantine must freeze exact W-1")
    if value["quarantine"].get("w0_value") != W_ZERO:
        raise MembershipAuditError("W0 diagnostic must remain separate")
    if set(
        str(field)
        for field in value["sentinel_propagation"].get(
            "child_locator_ledger_required_fields", []
        )
    ) != {
        "table",
        "relative_path",
        "row_group",
        "file_row",
        "raw_source_id",
        "selected_values_sha256",
        "row_digest_sha256",
        "disposition",
    }:
        raise MembershipAuditError("sentinel locator-ledger fields differ")

    main = value["main_accounting"]
    child_records = children["tables"]
    if set(child_records) != set(child_tables):
        raise MembershipAuditError("child table names differ across config sections")
    child_files = sum(
        int(record["expected_files"]) for record in child_records.values()
    )
    child_rows = sum(
        int(record["expected_raw_rows"]) for record in child_records.values()
    )
    if child_files != int(children["expected_child_files"]):
        raise MembershipAuditError("child expected-file total is inconsistent")
    if child_rows != int(children["expected_raw_child_rows"]):
        raise MembershipAuditError("child expected-row total is inconsistent")
    if child_rows - int(children["expected_frozen_quarantine_child_rows"]) != int(
        children["expected_retained_child_rows_on_pass"]
    ):
        raise MembershipAuditError("child retained-row conservation is inconsistent")
    if int(main["expected_raw_rows"]) + child_rows != int(
        inventory_raw["expected_footer_rows"]
    ):
        raise MembershipAuditError("global source row total is inconsistent")
    if int(main["expected_files"]) + child_files != int(
        inventory_raw["expected_parquet_file_count"]
    ):
        raise MembershipAuditError("global source file total is inconsistent")
    global_identity = value["source_row_partition"]["global_identity_on_pass"]
    if int(global_identity["raw_rows"]) != int(
        inventory_raw["expected_footer_rows"]
    ) or int(global_identity["retained_rows"]) + int(
        global_identity["frozen_w_minus_1_quarantine_rows"]
    ) + int(global_identity["all_other_partition_rows"]) != int(
        global_identity["raw_rows"]
    ):
        raise MembershipAuditError("global row-conservation contract is inconsistent")
    if int(master["exact_bytes"]) != int(master["exact_count"]) * 8:
        raise MembershipAuditError("master-ID byte/count contract is inconsistent")
    if (
        int(master["exact_count"]) != int(master["exact_distinct_ids"])
        or int(master["duplicate_row_excess"]) != 0
        or int(master["minimum_numeric_id"]) != 0
        or int(master["maximum_numeric_id"]) < int(master["minimum_numeric_id"])
    ):
        raise MembershipAuditError("master-ID exactness contract is inconsistent")
    minus_total = sum(
        int(record["w_minus_1"]["expected_rows"]) for record in incidence.values()
    )
    zero_total = sum(
        int(record["w_zero"]["expected_rows"]) for record in incidence.values()
    )
    if minus_total != int(
        value["sentinel_propagation"]["w_minus_1"]["expected_total_source_rows"]
    ):
        raise MembershipAuditError("W-1 table incidence total is inconsistent")
    if zero_total != int(
        value["sentinel_propagation"]["w_zero"]["expected_total_source_rows"]
    ):
        raise MembershipAuditError("W0 table incidence total is inconsistent")
    if set(target_contracts) != set(value["scope"]["work_target_columns"]):
        raise MembershipAuditError("work-target table sets are inconsistent")
    for table, target_record in target_contracts.items():
        child_record = child_records[table]
        if int(target_record["expected_files"]) != int(
            child_record["expected_files"]
        ) or int(target_record["expected_raw_rows"]) != int(
            child_record["expected_raw_rows"]
        ):
            raise MembershipAuditError(f"target/child contract mismatch: {table}")
    required_states = value["success_contract"]["required_component_states"]
    if value["success_contract"].get("success_status") != CANONICAL_SUCCESS_STATUS:
        raise MembershipAuditError("canonical success status differs")
    frozen_states = {
        "raw_source_id_validity": "BLOCKED",
        "w_zero_entity_validity": "UNRESOLVED",
        "parent_content_identity": "UNVERIFIED",
        "g_minus_1_data_readiness": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
    }
    if any(
        required_states.get(key) != expected for key, expected in frozen_states.items()
    ):
        raise MembershipAuditError("success gate boundary was weakened")
    return value


def resolve_repo_file(raw: str, *, field: str) -> Path:
    relative = safe_relative_path(raw)
    path = REPO_ROOT / relative
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise MembershipAuditError(f"{field} is missing") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise MembershipAuditError(f"{field} must be a regular non-symlink file")
    return path


def resolve_root(config: dict[str, Any], explicit: Path | None) -> Path:
    parent = config["parents"]["inventory"]
    if explicit is None:
        env_name = str(parent["source_root_env"])
        raw = os.environ.get(env_name)
        if not raw:
            raise MembershipAuditError(f"set {env_name} or pass --root")
        root = Path(raw)
    else:
        root = explicit
    root = root.expanduser().resolve()
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise MembershipAuditError("source root is missing") from exc
    if not stat.S_ISDIR(mode) or root.is_symlink():
        raise MembershipAuditError("source root must be a non-symlink directory")
    if root.name != str(parent["source_root_id"]):
        raise MembershipAuditError("source root basename differs from frozen ID")
    return root


def sentinel_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_inventory": config["parents"]["inventory"],
        "targeted_incidence": {
            "source_id_column": str(config["scope"]["source_id_column"]),
            "work_target_columns": config["scope"]["work_target_columns"],
        },
    }


def validate_parent(
    *, root: Path, manifest_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], str, list[sentinel.FileSnapshot]]:
    try:
        manifest, manifest_sha256, snapshot = sentinel.validate_parent_and_live_root(
            root=root,
            manifest_path=manifest_path,
            config=sentinel_compatible_config(config),
        )
    except sentinel.SentinelAuditError as exc:
        raise MembershipAuditError(str(exc)) from exc
    expected_tables = {
        str(config["scope"]["main_table"]),
        *map(str, config["scope"]["child_source_tables"]),
    }
    observed_tables = {str(record.get("name")) for record in manifest["tables"]}
    if observed_tables != expected_tables:
        raise MembershipAuditError("manifest table set differs from canonical scope")
    return manifest, manifest_sha256, snapshot


def validate_evidence(config: dict[str, Any]) -> dict[str, Any]:
    parents = config["parents"]
    grammar = parents["grammar_recount"]
    sentinel_parent = parents["sentinel_incidence"]
    grammar_path = resolve_repo_file(
        str(grammar["summary_path"]), field="grammar recount summary"
    )
    sentinel_path = resolve_repo_file(
        str(sentinel_parent["summary_path"]), field="sentinel summary"
    )
    if sha256_file(grammar_path) != str(grammar["summary_sha256"]):
        raise MembershipAuditError("grammar recount summary SHA-256 mismatch")
    if sha256_file(sentinel_path) != str(sentinel_parent["summary_sha256"]):
        raise MembershipAuditError("sentinel summary SHA-256 mismatch")
    grammar_report = json.loads(grammar_path.read_text(encoding="utf-8"))
    sentinel_report = json.loads(sentinel_path.read_text(encoding="utf-8"))
    if grammar_report.get("status") != grammar.get("expected_status"):
        raise MembershipAuditError("grammar recount status mismatch")
    counts = grammar_report.get("exact_counts", {})
    expected = {
        "grammar_valid_rows": int(grammar["sorted_ids_rows"]),
        "grammar_valid_distinct_ids": int(grammar["sorted_ids_rows"]),
        "grammar_valid_min_numeric_id": int(grammar["min_numeric_id"]),
        "grammar_valid_max_numeric_id": int(grammar["max_numeric_id"]),
    }
    for field, expected_value in expected.items():
        if int(counts.get(field, -1)) != expected_value:
            raise MembershipAuditError(f"grammar recount field mismatch: {field}")
    sorted_commitment = grammar_report.get("local_artifact_commitments", {}).get(
        "grammar_valid_sorted_ids", {}
    )
    if (
        int(sorted_commitment.get("bytes", -1)) != int(grammar["sorted_ids_bytes"])
        or sorted_commitment.get("sha256") != grammar["sorted_ids_sha256"]
    ):
        raise MembershipAuditError("grammar summary sorted-master commitment mismatch")
    fingerprint = sentinel_report.get("provenance", {}).get("result_fingerprint_sha256")
    if fingerprint != sentinel_parent["result_fingerprint_sha256"]:
        raise MembershipAuditError("sentinel result fingerprint mismatch")

    frozen_files = (
        (
            config["parent_inventory"]["summary_relative_path"],
            config["parent_inventory"]["summary_sha256"],
            "inventory summary",
        ),
        (
            config["parent_inventory"]["config_relative_path"],
            config["parent_inventory"]["config_sha256"],
            "inventory config",
        ),
        (
            config["parent_grammar_valid_main_id_recount"]["config_relative_path"],
            config["parent_grammar_valid_main_id_recount"]["config_sha256"],
            "grammar recount config",
        ),
        (
            "scripts/recount_openalex_g1_canonical_quarantine.py",
            config["parent_grammar_valid_main_id_recount"]["tool_sha256"],
            "grammar recount tool",
        ),
        (
            config["parent_sentinel_incidence"]["config_relative_path"],
            config["parent_sentinel_incidence"]["config_sha256"],
            "sentinel config",
        ),
        (
            "scripts/audit_openalex_sentinel_incidence.py",
            config["parent_sentinel_incidence"]["tool_sha256"],
            "sentinel tool",
        ),
    )
    for raw_path, expected_sha256, label in frozen_files:
        frozen_path = resolve_repo_file(str(raw_path), field=label)
        if sha256_file(frozen_path) != str(expected_sha256):
            raise MembershipAuditError(f"{label} SHA-256 mismatch")
    grammar_provenance = grammar_report.get("provenance", {})
    if (
        grammar_provenance.get("raw_report_sha256")
        != config["parent_grammar_valid_main_id_recount"]["raw_report_sha256"]
    ):
        raise MembershipAuditError("grammar raw-report commitment mismatch")
    if sha256_file(REPO_ROOT / "scripts/recount_openalex_g1.py") != (
        grammar_provenance.get("base_dependency_sha256")
    ):
        raise MembershipAuditError("grammar base dependency SHA-256 mismatch")
    sentinel_provenance = sentinel_report.get("provenance", {})
    if (
        sentinel_provenance.get("raw_json_sha256")
        != config["parent_sentinel_incidence"]["raw_evidence_sha256"]
    ):
        raise MembershipAuditError("sentinel raw-evidence commitment mismatch")
    return {
        "grammar_summary_path": grammar_path,
        "grammar_summary_sha256": sha256_file(grammar_path),
        "grammar_report": grammar_report,
        "sentinel_summary_path": sentinel_path,
        "sentinel_summary_sha256": sha256_file(sentinel_path),
        "sentinel_report": sentinel_report,
    }


def sha256_fd(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    chunk_size = 16 * 1024 * 1024
    while offset < size:
        payload = os.pread(fd, min(chunk_size, size - offset), offset)
        if not payload:
            raise MembershipAuditError("file became short while hashing")
        digest.update(payload)
        offset += len(payload)
    return digest.hexdigest()


def validate_master_ids(path: Path, config: dict[str, Any]) -> MasterIDsHandle:
    parent = config["parents"]["grammar_recount"]
    if path.is_symlink():
        raise MembershipAuditError("sorted master IDs must be a regular file")
    expected_bytes = int(parent["sorted_ids_bytes"])
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise MembershipAuditError("cannot safely open sorted master IDs") from exc
    mapping: mmap.mmap | None = None
    ids: np.ndarray | None = None
    try:
        start_stat = os.fstat(fd)
        if not stat.S_ISREG(start_stat.st_mode):
            raise MembershipAuditError("sorted master IDs must be a regular file")
        if int(start_stat.st_size) != expected_bytes:
            raise MembershipAuditError("sorted master-ID byte count mismatch")
        observed_sha256 = sha256_fd(fd, expected_bytes)
        if observed_sha256 != str(parent["sorted_ids_sha256"]):
            raise MembershipAuditError("sorted master-ID SHA-256 mismatch")
        rows = int(parent["sorted_ids_rows"])
        if rows < 1:
            raise MembershipAuditError("sorted master IDs are empty")
        mapping = mmap.mmap(fd, expected_bytes, access=mmap.ACCESS_READ)
        ids = np.frombuffer(mapping, dtype="<u8", count=rows)
        if int(ids[0]) != int(parent["min_numeric_id"]):
            raise MembershipAuditError("sorted master-ID minimum mismatch")
        if int(ids[-1]) != int(parent["max_numeric_id"]):
            raise MembershipAuditError("sorted master-ID maximum mismatch")
        for start in range(1, rows, 10_000_000):
            end = min(rows, start + 10_000_000)
            left = ids[start - 1 : end - 1]
            right = ids[start:end]
            if np.any(right <= left):
                raise MembershipAuditError(
                    "sorted master IDs are not strictly increasing"
                )
        return MasterIDsHandle(
            path=path,
            fd=fd,
            mapping=mapping,
            array=ids,
            start_identity=file_identity(start_stat),
            sha256=observed_sha256,
        )
    except Exception:
        ids = None
        if "left" in locals():
            del left
        if "right" in locals():
            del right
        if mapping is not None:
            mapping.close()
        os.close(fd)
        raise


def validate_master_ids_stable(handle: MasterIDsHandle) -> None:
    current = file_identity(os.fstat(handle.fd))
    if current != handle.start_identity:
        raise MembershipAuditError("sorted master-ID file changed during audit")
    if sha256_fd(handle.fd, current["bytes"]) != handle.sha256:
        raise MembershipAuditError("sorted master-ID bytes changed during audit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        current_fd = os.open(handle.path, flags)
    except OSError as exc:
        raise MembershipAuditError(
            "sorted master-ID path changed during audit"
        ) from exc
    try:
        if file_identity(os.fstat(current_fd)) != handle.start_identity:
            raise MembershipAuditError("sorted master-ID path changed during audit")
    finally:
        os.close(current_fd)


def build_membership_index(ids: np.ndarray, max_id: int) -> np.ndarray:
    membership = np.zeros(max_id + 1, dtype=np.uint8)
    chunk = 10_000_000
    for start in range(0, len(ids), chunk):
        values = np.asarray(ids[start : start + chunk], dtype=np.uint64)
        membership[values] = 1
    set_count = 0
    for start in range(0, len(membership), 100_000_000):
        set_count += int(np.count_nonzero(membership[start : start + 100_000_000]))
    if set_count != len(ids):
        raise MembershipAuditError("dense membership index popcount mismatch")
    if not np.all(membership[np.asarray(ids[: min(len(ids), 10_000_000)])] == 1):
        raise MembershipAuditError("dense membership index verification failed")
    return membership


def _python_partition(values: pa.Array, *, prefix: str) -> ParsedIDs:
    grammar = re.compile(rf"{re.escape(prefix)}(0|[1-9][0-9]*)")
    raw_values = values.to_pylist()
    numeric = np.zeros(len(raw_values), dtype=np.uint64)
    valid = np.zeros(len(raw_values), dtype=bool)
    null = np.zeros(len(raw_values), dtype=bool)
    minus = np.zeros(len(raw_values), dtype=bool)
    malformed = np.zeros(len(raw_values), dtype=bool)
    for index, value in enumerate(raw_values):
        if value is None:
            null[index] = True
            continue
        if prefix == WORK_PREFIX and value == W_MINUS_1:
            minus[index] = True
            continue
        match = grammar.fullmatch(str(value))
        if match is None:
            malformed[index] = True
            continue
        parsed = int(match.group(1))
        if parsed > UINT64_MAX:
            malformed[index] = True
            continue
        numeric[index] = parsed
        valid[index] = True
    return ParsedIDs(numeric, valid, null, minus, malformed)


def partition_ids(values: pa.Array, *, prefix: str) -> ParsedIDs:
    suffix = pc.utf8_slice_codeunits(values, start=len(prefix))
    try:
        numeric_arrow = pc.cast(suffix, pa.uint64(), safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return _python_partition(values, prefix=prefix)
    prefix_ok = pc.fill_null(pc.starts_with(values, pattern=prefix), False)
    nonempty = pc.fill_null(pc.greater(pc.utf8_length(suffix), 0), False)
    leading_zero = pc.fill_null(
        pc.equal(pc.utf8_slice_codeunits(suffix, 0, 1), "0"), True
    )
    decimal_normal = pc.or_(
        pc.fill_null(pc.equal(suffix, "0"), False), pc.invert(leading_zero)
    )
    reconstructed = pc.cast(numeric_arrow, pa.string())
    exact = pc.fill_null(pc.equal(suffix, reconstructed), False)
    valid_arrow = pc.fill_null(
        pc.and_(
            pc.and_(prefix_ok, pc.and_(nonempty, decimal_normal)),
            pc.and_(pc.is_valid(numeric_arrow), exact),
        ),
        False,
    )
    valid = np.asarray(valid_arrow.to_numpy(zero_copy_only=False), dtype=bool)
    null = np.asarray(pc.is_null(values).to_numpy(zero_copy_only=False), dtype=bool)
    if prefix == WORK_PREFIX:
        minus_arrow = pc.fill_null(pc.equal(values, W_MINUS_1), False)
        minus = np.asarray(minus_arrow.to_numpy(zero_copy_only=False), dtype=bool)
    else:
        minus = np.zeros(len(values), dtype=bool)
    malformed = ~(valid | null | minus)
    numeric = np.asarray(
        pc.fill_null(numeric_arrow, 0).to_numpy(zero_copy_only=False),
        dtype=np.uint64,
    )
    return ParsedIDs(numeric, valid, null, minus, malformed)


def member_mask(numeric: np.ndarray, membership: np.ndarray) -> np.ndarray:
    result = np.zeros(len(numeric), dtype=bool)
    within = numeric < len(membership)
    if np.any(within):
        result[within] = membership[numeric[within]] != 0
    return result


def logical_stream_update(digest: Any, parsed: ParsedIDs) -> None:
    categories = np.zeros(len(parsed.numeric), dtype=np.uint8)
    categories[parsed.w_minus_1] = 1
    categories[parsed.null] = 2
    categories[parsed.malformed_other] = 3
    digest.update(categories.tobytes())
    digest.update(np.ascontiguousarray(parsed.numeric, dtype="<u8").tobytes())


def parse_field(values: pa.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parsed = partition_ids(values, prefix=FIELD_PREFIX)
    numeric = parsed.numeric
    valid = parsed.grammar_valid & (numeric > 0) & (numeric < FIELD_INVALID)
    null = parsed.null
    invalid = ~(valid | null)
    fields = np.zeros(len(values), dtype=np.uint8)
    fields[valid] = numeric[valid].astype(np.uint8)
    fields[invalid] = FIELD_INVALID
    return fields, null, invalid


def parse_year(values: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    null = np.asarray(pc.is_null(values).to_numpy(zero_copy_only=False), dtype=bool)
    raw = np.asarray(pc.fill_null(values, np.nan).to_numpy(zero_copy_only=False))
    valid = null | (
        np.isfinite(raw) & (raw == np.floor(raw)) & (raw >= -32767) & (raw <= 32767)
    )
    years = np.full(len(raw), YEAR_NULL, dtype=np.int16)
    usable = valid & ~null
    years[usable] = raw[usable].astype(np.int16)
    return years, ~valid


def parse_declared_authors(
    values: pa.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    null = np.asarray(pc.is_null(values).to_numpy(zero_copy_only=False), dtype=bool)
    raw = np.asarray(pc.fill_null(values, -1).to_numpy(zero_copy_only=False))
    invalid = ~null & ((raw < 0) | (raw > np.iinfo(np.uint32).max))
    result = np.zeros(len(raw), dtype=np.uint32)
    usable = ~(null | invalid)
    result[usable] = raw[usable].astype(np.uint32)
    return result, null, invalid


def file_identity(file_stat: os.stat_result) -> dict[str, int]:
    return {
        "dev": int(file_stat.st_dev),
        "inode": int(file_stat.st_ino),
        "bytes": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
        "ctime_ns": int(file_stat.st_ctime_ns),
    }


def checkpoint_digest(metric: dict[str, Any]) -> str:
    payload = {key: value for key, value in metric.items() if key != "metric_sha256"}
    return sha256_bytes(canonical_json_bytes(payload))


def lineage_from_path(relative_path: str) -> str:
    name = Path(relative_path).name
    prefix = name.split("-", 1)[0]
    if prefix not in {"part", "base", "delta", "repair"}:
        raise MembershipAuditError(f"unexpected source lineage prefix: {name}")
    return prefix


def table_mode(table: str, config: dict[str, Any]) -> str:
    scope = config["scope"]
    if table == str(scope["main_table"]):
        return "main"
    if table == str(scope["authorship"]["table"]):
        return "authorship"
    if table in scope["work_target_columns"]:
        return "target"
    return "source"


def build_tasks(
    manifest: dict[str, Any],
    config: dict[str, Any],
    *,
    max_files_per_table: int | None,
    selected_tables: set[str] | None,
) -> list[FileTask]:
    scope = config["scope"]
    expected = {str(scope["main_table"]), *map(str, scope["child_source_tables"])}
    observed = {str(table["name"]) for table in manifest["tables"]}
    if expected != observed:
        raise MembershipAuditError("configured source-table set differs from parent")
    if selected_tables is not None:
        unknown = sorted(selected_tables - expected)
        if unknown:
            raise MembershipAuditError(f"unknown selected tables: {unknown}")
    tasks: list[FileTask] = []
    index = 0
    for table in sorted(manifest["tables"], key=lambda row: str(row["name"])):
        name = str(table["name"])
        if selected_tables is not None and name not in selected_tables:
            continue
        records = sorted(table["files"], key=lambda row: str(row["relative_path"]))
        if max_files_per_table is not None:
            records = records[:max_files_per_table]
        for record in records:
            tasks.append(
                FileTask(
                    index=index,
                    table=name,
                    mode=table_mode(name, config),
                    record=record,
                    lineage=lineage_from_path(str(record["relative_path"])),
                )
            )
            index += 1
    return tasks


def file_record_sha256(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(record))


def plan_payload(
    *,
    config: dict[str, Any],
    config_path: Path,
    tool_path: Path,
    manifest_sha256: str,
    master_path: Path,
    tasks: list[FileTask],
    max_files_per_table: int | None,
    selected_tables: set[str] | None,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    full_scope = max_files_per_table is None and selected_tables is None
    task_rows = sum(int(task.record["footer_rows"]) for task in tasks)
    task_digest = hashlib.sha256()
    for task in tasks:
        task_digest.update(
            canonical_json_bytes(
                {
                    "table": task.table,
                    "mode": task.mode,
                    "record_sha256": file_record_sha256(task.record),
                }
            )
        )
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "label": str(config["label"]),
        "scope": "full" if full_scope else "engineering_pilot",
        "full_scope": full_scope,
        "selected_tables": sorted(selected_tables) if selected_tables else None,
        "max_files_per_table": max_files_per_table,
        "task_count": len(tasks),
        "task_footer_rows": task_rows,
        "task_fingerprint_sha256": task_digest.hexdigest(),
        "parent_manifest_sha256": manifest_sha256,
        "config_sha256": sha256_file(config_path),
        "tool_sha256": sha256_file(tool_path),
        "master_path_name": master_path.name,
        "master_sha256": str(config["parents"]["grammar_recount"]["sorted_ids_sha256"]),
        "batch_size": batch_size,
        "workers": workers,
    }
    payload["plan_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def checkpoint_path(output_dir: Path, task: FileTask) -> Path:
    return output_dir / "checkpoints" / task.table / f"{task.index:06d}.json"


def artifact_path(output_dir: Path, task: FileTask, kind: str) -> Path:
    return output_dir / "artifacts" / task.table / f"{task.index:06d}.{kind}.npy"


def relative_path_set_sha256(paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def expected_output_evidence(tasks: list[FileTask]) -> dict[str, Any]:
    checkpoint_paths = [
        f"checkpoints/{task.table}/{task.index:06d}.json" for task in tasks
    ]
    artifact_paths: list[str] = []
    for task in tasks:
        kinds = {
            "authorship": ("author", "author_ids"),
            "target": ("target",),
            "main": ("main",),
            "source": (),
        }[task.mode]
        artifact_paths.extend(
            f"artifacts/{task.table}/{task.index:06d}.{kind}.npy" for kind in kinds
        )
    return {
        "checkpoint_files": len(checkpoint_paths),
        "checkpoint_path_set_sha256": relative_path_set_sha256(checkpoint_paths),
        "data_artifact_files": len(artifact_paths),
        "data_artifact_path_set_sha256": relative_path_set_sha256(artifact_paths),
    }


def load_checkpoint(
    *, output_dir: Path, task: FileTask, plan: dict[str, Any]
) -> dict[str, Any] | None:
    path = checkpoint_path(output_dir, task)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise MembershipAuditError(f"checkpoint is not a regular file: {path.name}")
    metric = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        metric.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        metric.get("plan_sha256") == plan["plan_sha256"],
        metric.get("table") == task.table,
        metric.get("relative_path") == task.record["relative_path"],
        metric.get("file_record_sha256") == file_record_sha256(task.record),
        int(metric.get("footer_rows", -1)) == int(task.record["footer_rows"]),
        metric.get("metric_sha256") == checkpoint_digest(metric),
    )
    if not all(expected):
        raise MembershipAuditError(f"checkpoint validation failed: {path}")
    artifact = metric.get("artifact")
    if artifact:
        artifact_file = artifact_path(output_dir, task, str(artifact["kind"]))
        if artifact_file.is_symlink() or not artifact_file.is_file():
            raise MembershipAuditError(
                f"checkpoint artifact is missing: {artifact_file}"
            )
        if int(artifact_file.stat().st_size) != int(artifact["bytes"]):
            raise MembershipAuditError(
                f"checkpoint artifact size changed: {artifact_file}"
            )
        if sha256_file(artifact_file) != str(artifact["sha256"]):
            raise MembershipAuditError(
                f"checkpoint artifact SHA-256 changed: {artifact_file}"
            )
    for extra in metric.get("extra_artifacts", []):
        extra_file = artifact_path(output_dir, task, str(extra["kind"]))
        if extra_file.is_symlink() or not extra_file.is_file():
            raise MembershipAuditError(f"checkpoint artifact is missing: {extra_file}")
        if int(extra_file.stat().st_size) != int(extra["bytes"]):
            raise MembershipAuditError(
                f"checkpoint artifact size changed: {extra_file}"
            )
        if sha256_file(extra_file) != str(extra["sha256"]):
            raise MembershipAuditError(
                f"checkpoint artifact SHA-256 changed: {extra_file}"
            )
    return metric


def validate_checkpoint_source_identity(
    *, root: Path, task: FileTask, metric: dict[str, Any]
) -> None:
    """Refuse a stale checkpoint if its exact source inode identity changed."""

    relative = safe_relative_path(str(task.record["relative_path"]))
    source = root / relative
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(source, flags)
    try:
        current_stat = os.fstat(fd)
        if not stat.S_ISREG(current_stat.st_mode):
            raise MembershipAuditError(f"input is not a regular file: {relative}")
        current = file_identity(current_stat)
    finally:
        os.close(fd)
    if int(current["bytes"]) != int(task.record["bytes"]) or int(
        current["mtime_ns"]
    ) != int(task.record["mtime_ns"]):
        raise MembershipAuditError(f"input identity mismatch: {relative}")
    if current != metric.get("start_file_identity") or current != metric.get(
        "end_file_identity"
    ):
        raise MembershipAuditError(f"checkpoint source identity changed: {relative}")


def selected_columns(task: FileTask, config: dict[str, Any]) -> list[str]:
    scope = config["scope"]
    source = str(scope["source_id_column"])
    if task.mode == "authorship":
        return [
            source,
            str(scope["authorship"]["author_id_column"]),
            str(scope["authorship"]["author_orcid_column"]),
        ]
    if task.mode == "target":
        return [source, str(scope["work_target_columns"][task.table])]
    if task.mode == "main":
        strata = scope["main_strata"]
        return [
            source,
            str(strata["publication_year_column"]),
            str(strata["primary_field_column"]),
            str(strata["declared_authors_column"]),
        ]
    return [source]


def expected_column_types(task: FileTask, config: dict[str, Any]) -> dict[str, str]:
    columns = selected_columns(task, config)
    expected = {columns[0]: "string"}
    if task.mode == "authorship":
        expected[columns[1]] = "string"
        expected[columns[2]] = "string"
    elif task.mode == "target":
        expected[columns[1]] = "string"
    elif task.mode == "main":
        expected[columns[1]] = str(
            config["work_profile_strata"]["dimensions"]["publication_year"][
                "physical_type"
            ]
        )
        expected[columns[2]] = str(
            config["work_profile_strata"]["dimensions"]["primary_field"][
                "physical_type"
            ]
        )
        expected[columns[3]] = str(
            config["work_profile_strata"]["dimensions"]["authors_count"][
                "physical_type"
            ]
        )
    return expected


def _locator(
    *,
    task: FileTask,
    row_group: int,
    file_row: int,
    raw_value: str,
    selected_values: dict[str, Any],
) -> dict[str, Any]:
    disposition = (
        "frozen_w_minus_1_quarantine"
        if raw_value == W_MINUS_1
        else "retain_warn_zero_id_unresolved"
    )
    payload = {
        "table": task.table,
        "relative_path": str(task.record["relative_path"]),
        "row_group": row_group,
        "file_row": file_row,
        "raw_source_id": raw_value,
        "selected_values_sha256": sha256_bytes(canonical_json_bytes(selected_values)),
        "disposition": disposition,
    }
    return {
        **payload,
        "row_digest_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _blocking_locator(
    *,
    task: FileTask,
    row_group: int,
    file_row: int,
    raw_identifier: Any,
    column: str,
    disposition: str,
    selected_values: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "table": task.table,
        "relative_path": str(task.record["relative_path"]),
        "row_group": row_group,
        "file_row": file_row,
        "column": column,
        "raw_identifier": raw_identifier,
        "disposition": disposition,
        "selected_values_sha256": sha256_bytes(canonical_json_bytes(selected_values)),
    }
    return {
        **payload,
        "row_digest_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _aggregate_author_file(ids: np.ndarray, categories: np.ndarray) -> np.ndarray:
    if len(ids) == 0:
        return np.empty(0, dtype=AUTHOR_DTYPE)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    sorted_categories = categories[order]
    starts = np.flatnonzero(
        np.concatenate((np.array([True]), sorted_ids[1:] != sorted_ids[:-1]))
    )
    unique = sorted_ids[starts]
    ends = np.append(starts[1:], len(sorted_ids))
    output = np.empty(len(unique), dtype=AUTHOR_DTYPE)
    output["id"] = unique
    output["slots"] = (ends - starts).astype(np.uint32)
    output["null_ids"] = np.add.reduceat(
        (sorted_categories == 0).astype(np.uint32), starts
    )
    output["valid_ids"] = np.add.reduceat(
        (sorted_categories == 1).astype(np.uint32), starts
    )
    output["invalid_ids"] = np.add.reduceat(
        (sorted_categories == 2).astype(np.uint32), starts
    )
    return output


def _aggregate_target_file(ids: np.ndarray) -> np.ndarray:
    if len(ids) == 0:
        return np.empty(0, dtype=TARGET_DTYPE)
    unique, counts = np.unique(ids, return_counts=True)
    output = np.empty(len(unique), dtype=TARGET_DTYPE)
    output["id"] = unique
    output["rows"] = counts.astype(np.uint64)
    return output


def scan_file(
    *,
    root: Path,
    output_dir: Path,
    task: FileTask,
    config: dict[str, Any],
    plan: dict[str, Any],
    membership: np.ndarray,
    batch_size: int,
    replay_existing: bool = True,
) -> dict[str, Any]:
    relative = safe_relative_path(str(task.record["relative_path"]))
    ensure_task_output_directories(output_dir, task)
    existing = load_checkpoint(output_dir=output_dir, task=task, plan=plan)
    if existing is not None:
        validate_checkpoint_source_identity(root=root, task=task, metric=existing)
        if replay_existing:
            with tempfile.TemporaryDirectory(
                prefix=f".checkpoint-replay-{task.index:06d}-",
                dir=output_dir.parent,
            ) as temporary:
                replayed = scan_file(
                    root=root,
                    output_dir=Path(temporary),
                    task=task,
                    config=config,
                    plan=plan,
                    membership=membership,
                    batch_size=batch_size,
                    replay_existing=False,
                )
                nondeterministic = {"elapsed_seconds", "metric_sha256"}
                existing_payload = {
                    key: value
                    for key, value in existing.items()
                    if key not in nondeterministic
                }
                replayed_payload = {
                    key: value
                    for key, value in replayed.items()
                    if key not in nondeterministic
                }
                if canonical_json_bytes(existing_payload) != canonical_json_bytes(
                    replayed_payload
                ):
                    raise MembershipAuditError(
                        f"checkpoint semantic replay differs: {relative}"
                    )
        return existing
    orphan_kinds = {
        "authorship": ("author", "author_ids"),
        "target": ("target",),
        "main": ("main",),
        "source": (),
    }[task.mode]
    for kind in orphan_kinds:
        orphan = artifact_path(output_dir, task, kind)
        if orphan.is_symlink():
            raise MembershipAuditError(f"orphan artifact is a symlink: {orphan}")
        if orphan.exists():
            orphan.unlink()
    source = root / relative
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(source, flags)
    started = time.perf_counter()
    try:
        start_stat = os.fstat(fd)
        if not stat.S_ISREG(start_stat.st_mode):
            raise MembershipAuditError(f"input is not a regular file: {relative}")
        expected_identity = {
            "bytes": int(task.record["bytes"]),
            "mtime_ns": int(task.record["mtime_ns"]),
        }
        if (
            int(start_stat.st_size) != expected_identity["bytes"]
            or int(start_stat.st_mtime_ns) != expected_identity["mtime_ns"]
        ):
            raise MembershipAuditError(f"input identity mismatch: {relative}")
        handle = os.fdopen(fd, "rb", closefd=False)
        try:
            parquet = pq.ParquetFile(handle)
            columns = selected_columns(task, config)
            missing = sorted(set(columns) - set(parquet.schema_arrow.names))
            if missing:
                raise MembershipAuditError(f"columns missing in {relative}: {missing}")
            observed_physical_schema = sha256_bytes(
                parquet.schema_arrow.remove_metadata().serialize().to_pybytes()
            )
            if observed_physical_schema != task.record.get(
                "physical_schema_fingerprint_sha256"
            ):
                raise MembershipAuditError(
                    f"physical schema fingerprint changed: {relative}"
                )
            frozen_fields = task.record.get("schema_fields")
            if not isinstance(frozen_fields, dict):
                raise MembershipAuditError(
                    f"frozen schema fields are missing: {relative}"
                )
            for column, expected_type in expected_column_types(task, config).items():
                observed_type = str(parquet.schema_arrow.field(column).type)
                frozen_type = str(frozen_fields.get(column, {}).get("type"))
                if observed_type != expected_type or frozen_type != expected_type:
                    raise MembershipAuditError(
                        f"selected column type differs: {relative}:{column}"
                    )
            counters = {
                "raw_rows": 0,
                "source_member_rows": 0,
                "source_orphan_rows": 0,
                "source_null_rows": 0,
                "source_w_minus_1_rows": 0,
                "source_w_zero_rows": 0,
                "source_malformed_other_rows": 0,
                "target_in_snapshot_rows": 0,
                "target_external_rows": 0,
                "target_null_rows": 0,
                "target_w_minus_1_rows": 0,
                "target_w_zero_rows": 0,
                "target_malformed_rows": 0,
                "target_self_loop_rows": 0,
                "author_null_id_rows": 0,
                "author_valid_id_rows": 0,
                "author_invalid_id_rows": 0,
                "author_raw_null_id_rows": 0,
                "author_raw_valid_id_rows": 0,
                "author_raw_invalid_id_rows": 0,
                "author_w_minus_1_null_id_rows": 0,
                "author_w_zero_null_id_rows": 0,
                "author_orcid_null_rows": 0,
                "author_orcid_nonnull_rows": 0,
                "author_raw_orcid_null_rows": 0,
                "author_raw_orcid_nonnull_rows": 0,
                "main_year_invalid_rows": 0,
                "main_field_invalid_rows": 0,
                "main_declared_authors_null_rows": 0,
                "main_declared_authors_invalid_rows": 0,
            }
            source_digest = hashlib.sha256()
            target_digest = hashlib.sha256()
            w_minus_locators: list[dict[str, Any]] = []
            w_zero_locators: list[dict[str, Any]] = []
            blocking_identifier_locators: list[dict[str, Any]] = []

            def add_blocking_locator(locator: dict[str, Any]) -> None:
                if (
                    len(blocking_identifier_locators)
                    >= MAX_IN_MEMORY_BLOCKING_LOCATORS_PER_FILE
                ):
                    raise MembershipAuditError(
                        f"blocking identifier ledger safety cap exceeded: {relative}"
                    )
                blocking_identifier_locators.append(locator)

            author_ids: list[np.ndarray] = []
            author_categories: list[np.ndarray] = []
            valid_author_numeric_ids: list[np.ndarray] = []
            target_ids: list[np.ndarray] = []
            main_rows: list[np.ndarray] = []
            file_row = 0
            for row_group in range(parquet.num_row_groups):
                for batch in parquet.iter_batches(
                    batch_size=batch_size,
                    row_groups=[row_group],
                    columns=columns,
                    use_threads=True,
                ):
                    source_values = batch.column(0)
                    parsed = partition_ids(source_values, prefix=WORK_PREFIX)
                    logical_stream_update(source_digest, parsed)
                    valid_numbers = parsed.numeric[parsed.grammar_valid]
                    valid_members = member_mask(valid_numbers, membership)
                    counters["raw_rows"] += len(source_values)
                    counters["source_member_rows"] += int(
                        np.count_nonzero(valid_members)
                    )
                    counters["source_orphan_rows"] += int(
                        len(valid_members) - np.count_nonzero(valid_members)
                    )
                    counters["source_null_rows"] += int(np.count_nonzero(parsed.null))
                    counters["source_w_minus_1_rows"] += int(
                        np.count_nonzero(parsed.w_minus_1)
                    )
                    counters["source_w_zero_rows"] += int(
                        np.count_nonzero(parsed.grammar_valid & (parsed.numeric == 0))
                    )
                    counters["source_malformed_other_rows"] += int(
                        np.count_nonzero(parsed.malformed_other)
                    )
                    sentinel_indices = np.flatnonzero(
                        parsed.w_minus_1
                        | (parsed.grammar_valid & (parsed.numeric == 0))
                    )
                    for local in sentinel_indices.tolist():
                        selected = {
                            name: batch.column(index)[local].as_py()
                            for index, name in enumerate(columns)
                        }
                        raw_value = str(source_values[local].as_py())
                        locator = _locator(
                            task=task,
                            row_group=row_group,
                            file_row=file_row + local,
                            raw_value=raw_value,
                            selected_values=selected,
                        )
                        if raw_value == W_MINUS_1:
                            w_minus_locators.append(locator)
                        else:
                            w_zero_locators.append(locator)

                    grammar_positions = np.flatnonzero(parsed.grammar_valid)
                    orphan_positions = grammar_positions[~valid_members]
                    source_blocker_positions = np.sort(
                        np.concatenate(
                            (
                                np.flatnonzero(parsed.null),
                                np.flatnonzero(parsed.malformed_other),
                                orphan_positions,
                            )
                        )
                    )
                    for local in source_blocker_positions.tolist():
                        selected = {
                            name: batch.column(index)[local].as_py()
                            for index, name in enumerate(columns)
                        }
                        if parsed.null[local]:
                            disposition = "source_null_id"
                        elif parsed.malformed_other[local]:
                            disposition = "source_unexpected_malformed_id"
                        else:
                            disposition = "source_grammar_valid_orphan"
                        add_blocking_locator(
                            _blocking_locator(
                                task=task,
                                row_group=row_group,
                                file_row=file_row + local,
                                raw_identifier=source_values[local].as_py(),
                                column=columns[0],
                                disposition=disposition,
                                selected_values=selected,
                            )
                        )

                    member_positions = grammar_positions[valid_members]
                    member_ids = parsed.numeric[member_positions]
                    if task.mode == "authorship":
                        author_values = batch.column(1)
                        authors = partition_ids(author_values, prefix=AUTHOR_PREFIX)
                        categories = np.full(len(author_values), 2, dtype=np.uint8)
                        categories[authors.null] = 0
                        categories[authors.grammar_valid] = 1
                        counters["author_raw_null_id_rows"] += int(
                            np.count_nonzero(categories == 0)
                        )
                        counters["author_raw_valid_id_rows"] += int(
                            np.count_nonzero(categories == 1)
                        )
                        counters["author_raw_invalid_id_rows"] += int(
                            np.count_nonzero(categories == 2)
                        )
                        for local in np.flatnonzero(categories == 2).tolist():
                            selected = {
                                name: batch.column(index)[local].as_py()
                                for index, name in enumerate(columns)
                            }
                            add_blocking_locator(
                                _blocking_locator(
                                    task=task,
                                    row_group=row_group,
                                    file_row=file_row + local,
                                    raw_identifier=author_values[local].as_py(),
                                    column=columns[1],
                                    disposition="author_unexpected_malformed_id",
                                    selected_values=selected,
                                )
                            )
                        counters["author_w_minus_1_null_id_rows"] += int(
                            np.count_nonzero(parsed.w_minus_1 & authors.null)
                        )
                        counters["author_w_zero_null_id_rows"] += int(
                            np.count_nonzero(
                                parsed.grammar_valid
                                & (parsed.numeric == 0)
                                & authors.null
                            )
                        )
                        retained_categories = categories[member_positions]
                        author_ids.append(member_ids.copy())
                        author_categories.append(retained_categories.copy())
                        retained_author_numeric = authors.numeric[member_positions]
                        valid_author_numeric_ids.append(
                            retained_author_numeric[retained_categories == 1].copy()
                        )
                        counters["author_null_id_rows"] += int(
                            np.count_nonzero(retained_categories == 0)
                        )
                        counters["author_valid_id_rows"] += int(
                            np.count_nonzero(retained_categories == 1)
                        )
                        counters["author_invalid_id_rows"] += int(
                            np.count_nonzero(retained_categories == 2)
                        )
                        raw_orcid_null = np.asarray(
                            pc.is_null(batch.column(2)).to_numpy(zero_copy_only=False),
                            dtype=bool,
                        )
                        counters["author_raw_orcid_null_rows"] += int(
                            np.count_nonzero(raw_orcid_null)
                        )
                        counters["author_raw_orcid_nonnull_rows"] += int(
                            len(raw_orcid_null) - np.count_nonzero(raw_orcid_null)
                        )
                        orcid_null = raw_orcid_null[member_positions]
                        counters["author_orcid_null_rows"] += int(
                            np.count_nonzero(orcid_null)
                        )
                        counters["author_orcid_nonnull_rows"] += int(
                            len(orcid_null) - np.count_nonzero(orcid_null)
                        )
                    elif task.mode == "target":
                        targets = partition_ids(batch.column(1), prefix=WORK_PREFIX)
                        logical_stream_update(target_digest, targets)
                        target_zero = targets.grammar_valid & (targets.numeric == 0)
                        target_ordinary = targets.grammar_valid & ~target_zero
                        target_numbers = targets.numeric[target_ordinary]
                        target_members = member_mask(target_numbers, membership)
                        counters["target_in_snapshot_rows"] += int(
                            np.count_nonzero(target_members)
                        )
                        counters["target_external_rows"] += int(
                            len(target_members) - np.count_nonzero(target_members)
                        )
                        counters["target_null_rows"] += int(
                            np.count_nonzero(targets.null)
                        )
                        counters["target_w_minus_1_rows"] += int(
                            np.count_nonzero(targets.w_minus_1)
                        )
                        counters["target_w_zero_rows"] += int(
                            np.count_nonzero(target_zero)
                        )
                        counters["target_malformed_rows"] += int(
                            np.count_nonzero(targets.malformed_other)
                        )
                        target_blocker_positions = np.sort(
                            np.concatenate(
                                (
                                    np.flatnonzero(targets.null),
                                    np.flatnonzero(targets.w_minus_1),
                                    np.flatnonzero(target_zero),
                                    np.flatnonzero(targets.malformed_other),
                                )
                            )
                        )
                        for local in target_blocker_positions.tolist():
                            selected = {
                                name: batch.column(index)[local].as_py()
                                for index, name in enumerate(columns)
                            }
                            if targets.null[local]:
                                disposition = "target_null_id"
                            elif targets.w_minus_1[local]:
                                disposition = "target_w_minus_1"
                            elif target_zero[local]:
                                disposition = "target_w_zero"
                            else:
                                disposition = "target_unexpected_malformed_id"
                            add_blocking_locator(
                                _blocking_locator(
                                    task=task,
                                    row_group=row_group,
                                    file_row=file_row + local,
                                    raw_identifier=batch.column(1)[local].as_py(),
                                    column=columns[1],
                                    disposition=disposition,
                                    selected_values=selected,
                                )
                            )
                        aligned = parsed.grammar_valid & target_ordinary
                        counters["target_self_loop_rows"] += int(
                            np.count_nonzero(
                                parsed.numeric[aligned] == targets.numeric[aligned]
                            )
                        )
                        target_ids.append(target_numbers.copy())
                    elif task.mode == "main":
                        years, year_invalid = parse_year(batch.column(1))
                        fields, _, field_invalid = parse_field(batch.column(2))
                        declared, declared_null, declared_invalid = (
                            parse_declared_authors(batch.column(3))
                        )
                        counters["main_year_invalid_rows"] += int(
                            np.count_nonzero(year_invalid)
                        )
                        counters["main_field_invalid_rows"] += int(
                            np.count_nonzero(field_invalid)
                        )
                        for local in np.flatnonzero(field_invalid).tolist():
                            selected = {
                                name: batch.column(index)[local].as_py()
                                for index, name in enumerate(columns)
                            }
                            add_blocking_locator(
                                _blocking_locator(
                                    task=task,
                                    row_group=row_group,
                                    file_row=file_row + local,
                                    raw_identifier=batch.column(2)[local].as_py(),
                                    column=columns[2],
                                    disposition="main_primary_field_invalid_id",
                                    selected_values=selected,
                                )
                            )
                        counters["main_declared_authors_null_rows"] += int(
                            np.count_nonzero(declared_null)
                        )
                        counters["main_declared_authors_invalid_rows"] += int(
                            np.count_nonzero(declared_invalid)
                        )
                        retained = np.empty(len(member_positions), dtype=MAIN_DTYPE)
                        retained["id"] = member_ids
                        retained["year"] = years[member_positions]
                        retained["field"] = fields[member_positions]
                        retained["flags"] = (
                            declared_null[member_positions].astype(np.uint8)
                            | (year_invalid[member_positions].astype(np.uint8) << 1)
                            | (field_invalid[member_positions].astype(np.uint8) << 2)
                            | (declared_invalid[member_positions].astype(np.uint8) << 3)
                        )
                        retained["declared_authors"] = declared[member_positions]
                        main_rows.append(retained)
                    file_row += len(batch)
        finally:
            handle.close()
        end_stat = os.fstat(fd)
        if file_identity(start_stat) != file_identity(end_stat):
            raise MembershipAuditError(f"input changed while open: {relative}")
    finally:
        os.close(fd)

    footer_rows = int(task.record["footer_rows"])
    if counters["raw_rows"] != footer_rows:
        raise MembershipAuditError(f"data-page/footer row mismatch: {relative}")
    partition_total = (
        counters["source_member_rows"]
        + counters["source_orphan_rows"]
        + counters["source_null_rows"]
        + counters["source_w_minus_1_rows"]
        + counters["source_malformed_other_rows"]
    )
    if partition_total != footer_rows:
        raise MembershipAuditError(f"source row conservation failed: {relative}")

    if task.mode == "target":
        target_total = sum(
            counters[key]
            for key in (
                "target_in_snapshot_rows",
                "target_external_rows",
                "target_null_rows",
                "target_w_minus_1_rows",
                "target_w_zero_rows",
                "target_malformed_rows",
            )
        )
        if target_total != footer_rows:
            raise MembershipAuditError(f"target row conservation failed: {relative}")
    if task.mode == "authorship":
        author_total = sum(
            counters[key]
            for key in (
                "author_null_id_rows",
                "author_valid_id_rows",
                "author_invalid_id_rows",
            )
        )
        if author_total != counters["source_member_rows"]:
            raise MembershipAuditError(f"author row conservation failed: {relative}")
        if (
            counters["author_raw_null_id_rows"]
            + counters["author_raw_valid_id_rows"]
            + counters["author_raw_invalid_id_rows"]
            != footer_rows
        ):
            raise MembershipAuditError(
                f"raw author-ID row conservation failed: {relative}"
            )
        if (
            counters["author_orcid_null_rows"] + counters["author_orcid_nonnull_rows"]
            != counters["source_member_rows"]
        ):
            raise MembershipAuditError(f"author ORCID conservation failed: {relative}")
        if (
            counters["author_raw_orcid_null_rows"]
            + counters["author_raw_orcid_nonnull_rows"]
            != footer_rows
        ):
            raise MembershipAuditError(
                f"raw author ORCID conservation failed: {relative}"
            )

    artifact: dict[str, Any] | None = None
    extra_artifacts: list[dict[str, Any]] = []
    if task.mode == "authorship":
        ids = np.concatenate(author_ids) if author_ids else np.empty(0, dtype=np.uint64)
        categories = (
            np.concatenate(author_categories)
            if author_categories
            else np.empty(0, dtype=np.uint8)
        )
        summary = _aggregate_author_file(ids, categories)
        artifact_file = artifact_path(output_dir, task, "author")
        artifact = {"kind": "author", **atomic_save_array(artifact_file, summary)}
        all_valid_author_ids = (
            np.concatenate(valid_author_numeric_ids)
            if valid_author_numeric_ids
            else np.empty(0, dtype=np.uint64)
        )
        unique_valid_author_ids = np.unique(all_valid_author_ids)
        author_ids_file = artifact_path(output_dir, task, "author_ids")
        extra_artifacts.append(
            {
                "kind": "author_ids",
                **atomic_save_array(author_ids_file, unique_valid_author_ids),
            }
        )
    elif task.mode == "target":
        ids = np.concatenate(target_ids) if target_ids else np.empty(0, dtype=np.uint64)
        summary = _aggregate_target_file(ids)
        artifact_file = artifact_path(output_dir, task, "target")
        artifact = {"kind": "target", **atomic_save_array(artifact_file, summary)}
    elif task.mode == "main":
        summary = (
            np.concatenate(main_rows) if main_rows else np.empty(0, dtype=MAIN_DTYPE)
        )
        artifact_file = artifact_path(output_dir, task, "main")
        artifact = {"kind": "main", **atomic_save_array(artifact_file, summary)}

    metric: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "task_index": task.index,
        "table": task.table,
        "mode": task.mode,
        "lineage": task.lineage,
        "relative_path": str(task.record["relative_path"]),
        "file_record_sha256": file_record_sha256(task.record),
        "footer_rows": footer_rows,
        "columns": selected_columns(task, config),
        "start_file_identity": file_identity(start_stat),
        "end_file_identity": file_identity(end_stat),
        "source_logical_stream_sha256": source_digest.hexdigest(),
        "target_logical_stream_sha256": (
            target_digest.hexdigest() if task.mode == "target" else None
        ),
        "counts": counters,
        "w_minus_1_locators": w_minus_locators,
        "w_zero_locators": w_zero_locators,
        "blocking_identifier_locators": blocking_identifier_locators,
        "artifact": artifact,
        "extra_artifacts": extra_artifacts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    metric["metric_sha256"] = checkpoint_digest(metric)
    atomic_write_json(checkpoint_path(output_dir, task), metric)
    return metric


def execute_tasks(
    *,
    root: Path,
    output_dir: Path,
    tasks: list[FileTask],
    config: dict[str, Any],
    plan: dict[str, Any],
    membership: np.ndarray,
    batch_size: int,
    workers: int,
    progress: Callable[[int, int, dict[str, Any]], None] | None,
) -> list[dict[str, Any]]:
    def execute(task: FileTask) -> dict[str, Any]:
        try:
            return scan_file(
                root=root,
                output_dir=output_dir,
                task=task,
                config=config,
                plan=plan,
                membership=membership,
                batch_size=batch_size,
            )
        except MembershipAuditError:
            raise
        except Exception as exc:
            raise MembershipAuditError(
                f"scan failed for {task.record['relative_path']}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    results: list[dict[str, Any]] = []
    if workers == 1:
        iterator: Iterable[dict[str, Any]] = (execute(task) for task in tasks)
        for completed, metric in enumerate(iterator, start=1):
            results.append(metric)
            if progress is not None:
                progress(completed, len(tasks), metric)
        return results
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, metric in enumerate(executor.map(execute, tasks), start=1):
            results.append(metric)
            if progress is not None:
                progress(completed, len(tasks), metric)
    return results


def expected_sentinel_tables(
    config: dict[str, Any], key: str
) -> dict[str, dict[str, int]]:
    raw = config["quarantine"][key]
    result: dict[str, dict[str, int]] = {}
    for table, record in raw.items():
        files = record.get("files", [])
        expected_files = {
            str(item["relative_path"]): int(item["rows"]) for item in files
        }
        if expected_files:
            result[str(table)] = expected_files
    return result


def aggregate_table_metrics(
    manifest: dict[str, Any], metrics: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        grouped.setdefault(str(metric["table"]), []).append(metric)
    parent_tables = {str(row["name"]): row for row in manifest["tables"]}
    rows: list[dict[str, Any]] = []
    for table in sorted(grouped):
        items = sorted(grouped[table], key=lambda row: str(row["relative_path"]))
        counts = {
            key: sum(int(item["counts"][key]) for item in items)
            for key in items[0]["counts"]
        }
        rows.append(
            {
                "table": table,
                "mode": items[0]["mode"],
                "expected_parent_files": int(parent_tables[table]["file_count"]),
                "expected_parent_footer_rows": int(parent_tables[table]["footer_rows"]),
                "scanned_files": len(items),
                "scanned_footer_rows": sum(int(item["footer_rows"]) for item in items),
                "counts": counts,
                "source_row_conservation": counts["raw_rows"]
                == counts["source_member_rows"]
                + counts["source_orphan_rows"]
                + counts["source_null_rows"]
                + counts["source_w_minus_1_rows"]
                + counts["source_malformed_other_rows"],
                "w_minus_1_affected_files": {
                    str(item["relative_path"]): int(
                        item["counts"]["source_w_minus_1_rows"]
                    )
                    for item in items
                    if int(item["counts"]["source_w_minus_1_rows"])
                },
                "w_zero_affected_files": {
                    str(item["relative_path"]): int(
                        item["counts"]["source_w_zero_rows"]
                    )
                    for item in items
                    if int(item["counts"]["source_w_zero_rows"])
                },
                "w_minus_1_locator_ledger": [
                    locator
                    for item in items
                    for locator in item.get("w_minus_1_locators", [])
                ],
                "w_zero_locator_ledger": [
                    locator
                    for item in items
                    for locator in item.get("w_zero_locators", [])
                ],
                "blocking_identifier_ledger": [
                    locator
                    for item in items
                    for locator in item.get("blocking_identifier_locators", [])
                ],
                "source_logical_stream_fingerprint_sha256": sha256_bytes(
                    canonical_json_bytes(
                        [
                            {
                                "relative_path": item["relative_path"],
                                "sha256": item["source_logical_stream_sha256"],
                            }
                            for item in items
                        ]
                    )
                ),
            }
        )
    return rows


def validate_sentinel_contract(
    *, table_rows: list[dict[str, Any]], config: dict[str, Any], full_scope: bool
) -> dict[str, Any]:
    observed_minus = {
        row["table"]: row["w_minus_1_affected_files"]
        for row in table_rows
        if row["w_minus_1_affected_files"]
    }
    observed_zero = {
        row["table"]: row["w_zero_affected_files"]
        for row in table_rows
        if row["w_zero_affected_files"]
    }
    expected_minus = expected_sentinel_tables(config, "expected_w_minus_1")
    expected_zero = expected_sentinel_tables(config, "expected_w0")
    main_table = str(config["scope"]["main_table"])
    main_row = next((row for row in table_rows if row["table"] == main_table), None)

    def locator_ledger_exact(
        *, ledger_key: str, expected: dict[str, dict[str, int]], raw_value: str
    ) -> bool:
        disposition = (
            "frozen_w_minus_1_quarantine"
            if raw_value == W_MINUS_1
            else "retain_warn_zero_id_unresolved"
        )
        for row in table_rows:
            table = str(row["table"])
            expected_files = expected.get(table, {})
            ledger = row.get(ledger_key)
            if not isinstance(ledger, list):
                return False
            observed_files: dict[str, int] = {}
            coordinates: set[tuple[str, int, int]] = set()
            required_fields = set(
                map(
                    str,
                    config["sentinel_propagation"][
                        "child_locator_ledger_required_fields"
                    ],
                )
            )
            for locator in ledger:
                if not isinstance(locator, dict) or set(locator) != required_fields:
                    return False
                if (
                    type(locator.get("row_group")) is not int
                    or int(locator["row_group"]) < 0
                    or type(locator.get("file_row")) is not int
                    or int(locator["file_row"]) < 0
                ):
                    return False
                relative = str(locator.get("relative_path"))
                observed_files[relative] = observed_files.get(relative, 0) + 1
                coordinate = (
                    relative,
                    int(locator.get("row_group", -1)),
                    int(locator.get("file_row", -1)),
                )
                digest_payload = {
                    key: value
                    for key, value in locator.items()
                    if key != "row_digest_sha256"
                }
                if (
                    locator.get("table") != table
                    or locator.get("raw_source_id") != raw_value
                    or locator.get("disposition") != disposition
                    or re.fullmatch(
                        r"[0-9a-f]{64}", str(locator.get("row_digest_sha256"))
                    )
                    is None
                    or re.fullmatch(
                        r"[0-9a-f]{64}", str(locator.get("selected_values_sha256"))
                    )
                    is None
                    or locator.get("row_digest_sha256")
                    != sha256_bytes(canonical_json_bytes(digest_payload))
                    or coordinate in coordinates
                ):
                    return False
                coordinates.add(coordinate)
            if observed_files != expected_files:
                return False
        return set(expected).issubset({str(row["table"]) for row in table_rows})

    minus_ledger_exact = locator_ledger_exact(
        ledger_key="w_minus_1_locator_ledger",
        expected=expected_minus,
        raw_value=W_MINUS_1,
    )
    zero_ledger_exact = locator_ledger_exact(
        ledger_key="w_zero_locator_ledger",
        expected=expected_zero,
        raw_value=W_ZERO,
    )
    locator_checks: dict[str, bool | None] = {
        "w_minus_1": None,
        "w_zero": None,
    }
    if full_scope and main_row is not None:
        for key, ledger_key in (
            ("w_minus_1", "w_minus_1_locator_ledger"),
            ("w_zero", "w_zero_locator_ledger"),
        ):
            expected_locator = config["quarantine"]["main_row_locators"][key]
            ledger = main_row[ledger_key]
            locator_checks[key] = len(ledger) == 1 and all(
                ledger[0][field] == expected_locator[field]
                for field in ("relative_path", "row_group", "file_row")
            )
    return {
        "full_scope": full_scope,
        "w_minus_1_observed": observed_minus,
        "w_minus_1_expected": expected_minus,
        "w_minus_1_exact_match": observed_minus == expected_minus
        if full_scope
        else None,
        "w_minus_1_locator_ledger_exact_match": (
            minus_ledger_exact if full_scope else None
        ),
        "w0_observed": observed_zero,
        "w0_expected": expected_zero,
        "w0_exact_match": observed_zero == expected_zero if full_scope else None,
        "w0_locator_ledger_exact_match": zero_ledger_exact if full_scope else None,
        "main_locator_exact_match": locator_checks,
    }


BLOCKING_DISPOSITION_DOMAIN = {
    "source_null_id": "source",
    "source_unexpected_malformed_id": "source",
    "source_grammar_valid_orphan": "source",
    "target_null_id": "target",
    "target_w_minus_1": "target",
    "target_w_zero": "target",
    "target_unexpected_malformed_id": "target",
    "author_unexpected_malformed_id": "author",
    "main_primary_field_invalid_id": "main_field",
}
BLOCKING_LEDGER_DOMAINS = ("source", "target", "author", "main_field")
BLOCKING_LOCATOR_REQUIRED_FIELDS = {
    "table",
    "relative_path",
    "row_group",
    "file_row",
    "column",
    "raw_identifier",
    "disposition",
    "selected_values_sha256",
    "row_digest_sha256",
}


def expected_blocking_disposition_counts(
    *, mode: str, counts: dict[str, Any]
) -> dict[str, int]:
    expected = {
        "source_null_id": int(counts["source_null_rows"]),
        "source_unexpected_malformed_id": int(counts["source_malformed_other_rows"]),
        "source_grammar_valid_orphan": int(counts["source_orphan_rows"]),
    }
    if mode == "target":
        expected.update(
            {
                "target_null_id": int(counts["target_null_rows"]),
                "target_w_minus_1": int(counts["target_w_minus_1_rows"]),
                "target_w_zero": int(counts["target_w_zero_rows"]),
                "target_unexpected_malformed_id": int(
                    counts["target_malformed_rows"]
                ),
            }
        )
    if mode == "authorship":
        expected["author_unexpected_malformed_id"] = int(
            counts["author_raw_invalid_id_rows"]
        )
    if mode == "main":
        expected["main_primary_field_invalid_id"] = int(
            counts["main_field_invalid_rows"]
        )
    return expected


def validate_per_file_blocking_identifier_ledgers(
    *, tasks: list[FileTask], metrics: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], str]]]:
    domain_passes = {domain: True for domain in BLOCKING_LEDGER_DOMAINS}
    checked_files_by_domain = {domain: 0 for domain in BLOCKING_LEDGER_DOMAINS}
    validated_locators: list[tuple[dict[str, Any], str]] = []
    tasks_by_index = {task.index: task for task in tasks}
    metrics_by_index: dict[int, dict[str, Any]] = {}
    duplicate_metric_indices: set[int] = set()
    for metric in metrics:
        raw_index = metric.get("task_index")
        if type(raw_index) is not int:
            for domain in domain_passes:
                domain_passes[domain] = False
            continue
        index = raw_index
        if index in metrics_by_index:
            duplicate_metric_indices.add(index)
        metrics_by_index[index] = metric
    if (
        len(tasks_by_index) != len(tasks)
        or set(metrics_by_index) != set(tasks_by_index)
        or duplicate_metric_indices
    ):
        for domain in domain_passes:
            domain_passes[domain] = False

    for index, task in tasks_by_index.items():
        metric = metrics_by_index.get(index)
        mode = task.mode
        expected = (
            expected_blocking_disposition_counts(mode=mode, counts=metric["counts"])
            if metric is not None and isinstance(metric.get("counts"), dict)
            else {}
        )
        relevant_domains = {
            BLOCKING_DISPOSITION_DOMAIN[disposition] for disposition in expected
        }
        if (
            metric is None
            or metric.get("table") != task.table
            or metric.get("mode") != task.mode
            or metric.get("relative_path") != task.record["relative_path"]
        ):
            for domain in relevant_domains or set(BLOCKING_LEDGER_DOMAINS):
                domain_passes[domain] = False
            continue
        ledger = metric.get("blocking_identifier_locators")
        if not isinstance(ledger, list):
            for domain in relevant_domains:
                domain_passes[domain] = False
            continue
        observed = {disposition: 0 for disposition in expected}
        file_valid_domains = {domain: True for domain in relevant_domains}
        for locator in ledger:
            if (
                not isinstance(locator, dict)
                or set(locator) != BLOCKING_LOCATOR_REQUIRED_FIELDS
            ):
                for domain in relevant_domains:
                    file_valid_domains[domain] = False
                continue
            disposition = str(locator.get("disposition"))
            if disposition not in observed:
                for domain in relevant_domains:
                    file_valid_domains[domain] = False
                continue
            domain = BLOCKING_DISPOSITION_DOMAIN[disposition]
            if (
                locator.get("table") != task.table
                or locator.get("relative_path") != task.record["relative_path"]
                or type(locator.get("row_group")) is not int
                or int(locator["row_group"]) < 0
                or type(locator.get("file_row")) is not int
                or int(locator["file_row"]) < 0
            ):
                file_valid_domains[domain] = False
                observed[disposition] += 1
                continue
            observed[disposition] += 1
            validated_locators.append((locator, domain))
        for disposition, expected_count in expected.items():
            domain = BLOCKING_DISPOSITION_DOMAIN[disposition]
            if observed[disposition] != expected_count:
                file_valid_domains[domain] = False
        for domain, passed in file_valid_domains.items():
            domain_passes[domain] = domain_passes[domain] and passed
            if passed:
                checked_files_by_domain[domain] += 1
    contract = {
        "pass": all(domain_passes.values()),
        "domain_passes": domain_passes,
        "checked_files_by_domain": checked_files_by_domain,
        "validated_locator_rows": len(validated_locators),
    }
    return contract, validated_locators


def classify_blocking_identifier(
    *, raw_identifier: Any, domain: str, membership: np.ndarray
) -> str | None:
    try:
        values = pa.array([raw_identifier], type=pa.string())
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
        return None
    if domain == "source":
        parsed = partition_ids(values, prefix=WORK_PREFIX)
        if bool(parsed.null[0]):
            return "source_null_id"
        if bool(parsed.malformed_other[0]):
            return "source_unexpected_malformed_id"
        if bool(parsed.grammar_valid[0]) and not bool(
            member_mask(parsed.numeric, membership)[0]
        ):
            return "source_grammar_valid_orphan"
        return None
    if domain == "target":
        parsed = partition_ids(values, prefix=WORK_PREFIX)
        if bool(parsed.null[0]):
            return "target_null_id"
        if bool(parsed.w_minus_1[0]):
            return "target_w_minus_1"
        if bool(parsed.grammar_valid[0]) and int(parsed.numeric[0]) == 0:
            return "target_w_zero"
        if bool(parsed.malformed_other[0]):
            return "target_unexpected_malformed_id"
        return None
    if domain == "author":
        parsed = partition_ids(values, prefix=AUTHOR_PREFIX)
        return (
            "author_unexpected_malformed_id"
            if bool(parsed.malformed_other[0])
            else None
        )
    if domain == "main_field":
        _, _, invalid = parse_field(values)
        return "main_primary_field_invalid_id" if bool(invalid[0]) else None
    return None


def validate_blocking_locator_source_rows(
    *,
    root: Path,
    tasks: list[FileTask],
    metrics: list[dict[str, Any]],
    locators: list[tuple[dict[str, Any], str]],
    config: dict[str, Any],
    membership: np.ndarray,
) -> dict[str, Any]:
    domain_checks = {domain: True for domain in BLOCKING_LEDGER_DOMAINS}
    checked_by_domain = {domain: 0 for domain in BLOCKING_LEDGER_DOMAINS}
    tasks_by_path: dict[str, FileTask] = {}
    duplicate_task_paths: set[str] = set()
    for task in tasks:
        relative = str(task.record["relative_path"])
        if relative in tasks_by_path:
            duplicate_task_paths.add(relative)
        tasks_by_path[relative] = task
    metrics_by_index = {
        metric["task_index"]: metric
        for metric in metrics
        if type(metric.get("task_index")) is int
    }
    grouped: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for locator, domain in locators:
        grouped.setdefault(str(locator.get("relative_path")), []).append(
            (locator, domain)
        )

    for relative, items in grouped.items():
        affected_domains = {domain for _, domain in items}
        task = tasks_by_path.get(relative)
        if task is None or relative in duplicate_task_paths:
            for domain in affected_domains:
                domain_checks[domain] = False
            continue
        metric = metrics_by_index.get(task.index)
        if (
            metric is None
            or metric.get("relative_path") != relative
            or metric.get("table") != task.table
        ):
            for domain in affected_domains:
                domain_checks[domain] = False
            continue
        try:
            safe_relative = safe_relative_path(relative)
        except MembershipAuditError:
            for domain in affected_domains:
                domain_checks[domain] = False
            continue
        source = root / safe_relative
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            fd = os.open(source, flags)
        except OSError:
            for domain in affected_domains:
                domain_checks[domain] = False
            continue
        try:
            start_stat = os.fstat(fd)
            current_identity = file_identity(start_stat)
            if (
                not stat.S_ISREG(start_stat.st_mode)
                or current_identity != metric.get("start_file_identity")
                or current_identity != metric.get("end_file_identity")
                or int(start_stat.st_size) != int(task.record["bytes"])
                or int(start_stat.st_mtime_ns) != int(task.record["mtime_ns"])
            ):
                for domain in affected_domains:
                    domain_checks[domain] = False
                continue
            handle = os.fdopen(fd, "rb", closefd=False)
            try:
                parquet = pq.ParquetFile(handle)
                columns = selected_columns(task, config)
                if (
                    parquet.metadata.num_rows != int(task.record["footer_rows"])
                    or parquet.num_row_groups != int(task.record["row_groups"])
                    or set(columns) - set(parquet.schema_arrow.names)
                ):
                    for domain in affected_domains:
                        domain_checks[domain] = False
                    continue
                offsets: list[int] = []
                offset = 0
                for row_group in range(parquet.num_row_groups):
                    offsets.append(offset)
                    offset += int(parquet.metadata.row_group(row_group).num_rows)
                if offset != int(task.record["footer_rows"]):
                    for domain in affected_domains:
                        domain_checks[domain] = False
                    continue
                by_row_group: dict[int, list[tuple[dict[str, Any], str]]] = {}
                for locator, domain in items:
                    row_group = int(locator["row_group"])
                    if row_group < 0 or row_group >= parquet.num_row_groups:
                        domain_checks[domain] = False
                        continue
                    by_row_group.setdefault(row_group, []).append((locator, domain))
                for row_group, row_locators in by_row_group.items():
                    table = parquet.read_row_group(
                        row_group, columns=columns, use_threads=True
                    )
                    row_group_rows = len(table)
                    for locator, domain in row_locators:
                        file_row = int(locator["file_row"])
                        local_row = file_row - offsets[row_group]
                        if local_row < 0 or local_row >= row_group_rows:
                            domain_checks[domain] = False
                            continue
                        actual_values = {
                            name: table.column(index)[local_row].as_py()
                            for index, name in enumerate(columns)
                        }
                        column = str(locator["column"])
                        if column not in actual_values:
                            domain_checks[domain] = False
                            continue
                        actual_raw = actual_values[column]
                        actual_disposition = classify_blocking_identifier(
                            raw_identifier=actual_raw,
                            domain=domain,
                            membership=membership,
                        )
                        if (
                            locator.get("table") != task.table
                            or locator.get("raw_identifier") != actual_raw
                            or locator.get("disposition") != actual_disposition
                            or locator.get("selected_values_sha256")
                            != sha256_bytes(canonical_json_bytes(actual_values))
                        ):
                            domain_checks[domain] = False
                            continue
                        checked_by_domain[domain] += 1
            finally:
                handle.close()
            if file_identity(os.fstat(fd)) != current_identity:
                for domain in affected_domains:
                    domain_checks[domain] = False
        finally:
            os.close(fd)
    return {
        "pass": all(domain_checks.values()),
        "domain_passes": domain_checks,
        "checked_source_rows_by_domain": checked_by_domain,
    }


def validate_blocking_identifier_ledger(
    table_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    root: Path,
    tasks: list[FileTask],
    metrics: list[dict[str, Any]],
    membership: np.ndarray,
) -> dict[str, Any]:
    source_column = str(config["scope"]["source_id_column"])
    expected_global: dict[str, int] = {}
    observed_global: dict[str, int] = {}
    table_checks: dict[str, bool] = {}
    table_domain_checks: dict[str, dict[str, bool]] = {}
    for row in table_rows:
        table = str(row["table"])
        counts = row["counts"]
        expected = expected_blocking_disposition_counts(
            mode=str(row["mode"]), counts=counts
        )
        ledger = row.get("blocking_identifier_ledger")
        relevant_domains = {
            BLOCKING_DISPOSITION_DOMAIN[disposition] for disposition in expected
        }
        domain_valid = {domain: True for domain in relevant_domains}
        valid = isinstance(ledger, list)
        observed = {key: 0 for key in expected}
        coordinates: dict[tuple[str, int, int, str], str] = {}
        if not valid:
            for domain in domain_valid:
                domain_valid[domain] = False
        if valid:
            for locator in ledger:
                if (
                    not isinstance(locator, dict)
                    or set(locator) != BLOCKING_LOCATOR_REQUIRED_FIELDS
                ):
                    for domain in domain_valid:
                        domain_valid[domain] = False
                    break
                disposition = str(locator.get("disposition"))
                if disposition not in observed:
                    for domain in domain_valid:
                        domain_valid[domain] = False
                    break
                domain = BLOCKING_DISPOSITION_DOMAIN[disposition]
                row_group = locator.get("row_group")
                file_row = locator.get("file_row")
                column = str(locator.get("column"))
                if disposition.startswith("target_"):
                    expected_column = str(config["scope"]["work_target_columns"][table])
                elif disposition.startswith("author_"):
                    expected_column = str(
                        config["scope"]["authorship"]["author_id_column"]
                    )
                elif disposition.startswith("main_primary_field_"):
                    expected_column = str(
                        config["scope"]["main_strata"]["primary_field_column"]
                    )
                else:
                    expected_column = source_column
                coordinate = (
                    str(locator.get("relative_path")),
                    row_group if type(row_group) is int else -1,
                    file_row if type(file_row) is int else -1,
                    column,
                )
                digest_payload = {
                    key: value
                    for key, value in locator.items()
                    if key != "row_digest_sha256"
                }
                locator_valid = not (
                    locator.get("table") != table
                    or type(row_group) is not int
                    or row_group < 0
                    or type(file_row) is not int
                    or file_row < 0
                    or column != expected_column
                    or coordinate in coordinates
                    or re.fullmatch(
                        r"[0-9a-f]{64}", str(locator.get("selected_values_sha256"))
                    )
                    is None
                    or locator.get("row_digest_sha256")
                    != sha256_bytes(canonical_json_bytes(digest_payload))
                )
                if coordinate in coordinates:
                    domain_valid[domain] = False
                    domain_valid[coordinates[coordinate]] = False
                    locator_valid = False
                if not locator_valid:
                    domain_valid[domain] = False
                else:
                    coordinates[coordinate] = domain
                observed[disposition] += 1
        for disposition, expected_count in expected.items():
            domain = BLOCKING_DISPOSITION_DOMAIN[disposition]
            if observed[disposition] != expected_count:
                domain_valid[domain] = False
        valid = valid and all(domain_valid.values()) and observed == expected
        table_checks[table] = bool(valid)
        table_domain_checks[table] = domain_valid
        for disposition, count in expected.items():
            expected_global[disposition] = expected_global.get(disposition, 0) + count
        for disposition, count in observed.items():
            observed_global[disposition] = observed_global.get(disposition, 0) + count
    per_file_contract, per_file_locators = (
        validate_per_file_blocking_identifier_ledgers(tasks=tasks, metrics=metrics)
    )
    per_file_by_table_domain: dict[tuple[str, str], list[bytes]] = {}
    for locator, domain in per_file_locators:
        key = (str(locator["table"]), domain)
        per_file_by_table_domain.setdefault(key, []).append(
            canonical_json_bytes(locator)
        )
    for row in table_rows:
        table = str(row["table"])
        aggregate_by_domain: dict[str, list[bytes]] = {
            domain: [] for domain in table_domain_checks[table]
        }
        for locator in row.get("blocking_identifier_ledger", []):
            if not isinstance(locator, dict):
                continue
            domain = BLOCKING_DISPOSITION_DOMAIN.get(
                str(locator.get("disposition"))
            )
            if domain in aggregate_by_domain:
                aggregate_by_domain[domain].append(canonical_json_bytes(locator))
        for domain, aggregate_records in aggregate_by_domain.items():
            if sorted(aggregate_records) != sorted(
                per_file_by_table_domain.get((table, domain), [])
            ):
                table_domain_checks[table][domain] = False
    source_binding = validate_blocking_locator_source_rows(
        root=root,
        tasks=tasks,
        metrics=metrics,
        locators=per_file_locators,
        config=config,
        membership=membership,
    )
    domain_passes = {
        domain: all(
            checks.get(domain, True) for checks in table_domain_checks.values()
        )
        and bool(per_file_contract["domain_passes"][domain])
        and bool(source_binding["domain_passes"][domain])
        for domain in BLOCKING_LEDGER_DOMAINS
    }
    return {
        "pass": all(table_checks.values())
        and source_binding["pass"]
        and all(domain_passes.values()),
        "expected_by_disposition": expected_global,
        "observed_by_disposition": observed_global,
        "table_checks": table_checks,
        "table_domain_checks": table_domain_checks,
        "domain_passes": domain_passes,
        "per_file_contract": per_file_contract,
        "source_row_binding": source_binding,
    }


def load_artifact(
    output_dir: Path, task: FileTask, metric: dict[str, Any], kind: str
) -> np.ndarray:
    record = metric.get("artifact")
    if not record or record.get("kind") != kind:
        raise MembershipAuditError(
            f"missing {kind} artifact: {task.record['relative_path']}"
        )
    path = artifact_path(output_dir, task, kind)
    if sha256_file(path) != str(record["sha256"]):
        raise MembershipAuditError(f"{kind} artifact SHA-256 changed: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if len(array) != int(record["records"]):
        raise MembershipAuditError(f"{kind} artifact record count changed: {path}")
    return array


def load_extra_artifact(
    output_dir: Path,
    task: FileTask,
    metric: dict[str, Any],
    kind: str,
) -> np.ndarray:
    matches = [
        record
        for record in metric.get("extra_artifacts", [])
        if record.get("kind") == kind
    ]
    if len(matches) != 1:
        raise MembershipAuditError(
            f"missing {kind} artifact: {task.record['relative_path']}"
        )
    record = matches[0]
    path = artifact_path(output_dir, task, kind)
    if sha256_file(path) != str(record["sha256"]):
        raise MembershipAuditError(f"{kind} artifact SHA-256 changed: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if len(array) != int(record["records"]):
        raise MembershipAuditError(f"{kind} artifact record count changed: {path}")
    return array


def target_unique_coverage(
    *,
    output_dir: Path,
    tasks: list[FileTask],
    metrics_by_index: dict[int, dict[str, Any]],
    membership: np.ndarray,
    table: str,
) -> dict[str, Any]:
    seen = np.zeros(len(membership), dtype=np.uint8)
    external_above_max: set[int] = set()
    rle_rows = 0
    rle_records = 0
    for task in tasks:
        if task.table != table or task.mode != "target":
            continue
        array = load_artifact(output_dir, task, metrics_by_index[task.index], "target")
        ids = np.asarray(array["id"], dtype=np.uint64)
        rle_rows += int(np.asarray(array["rows"], dtype=np.uint64).sum(dtype=np.uint64))
        rle_records += len(ids)
        within = ids < len(seen)
        if np.any(within):
            seen[ids[within]] = 1
        if np.any(~within):
            external_above_max.update(map(int, ids[~within].tolist()))
    unique_within = 0
    unique_in_snapshot = 0
    chunk = 100_000_000
    for start in range(0, len(seen), chunk):
        end = min(len(seen), start + chunk)
        observed = seen[start:end]
        unique_within += int(np.count_nonzero(observed))
        unique_in_snapshot += int(np.count_nonzero(observed & membership[start:end]))
    unique_external = unique_within - unique_in_snapshot + len(external_above_max)
    return {
        "rle_input_rows": rle_rows,
        "per_file_rle_records": rle_records,
        "unique_valid_targets": unique_within + len(external_above_max),
        "unique_targets_in_snapshot": unique_in_snapshot,
        "unique_targets_external_to_snapshot": unique_external,
        "unique_target_coverage": (
            unique_in_snapshot / (unique_within + len(external_above_max))
            if unique_within + len(external_above_max)
            else None
        ),
        "external_ids_above_main_max": len(external_above_max),
    }


def validate_target_coverage_contract(
    coverage: dict[str, Any],
    table_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_tables = set(map(str, config["scope"]["work_target_columns"]))
    checks: dict[str, bool] = {}
    if set(coverage) != expected_tables:
        return {"pass": False, "table_checks": checks}
    rows_by_table = {str(row["table"]): row for row in table_rows}
    for table in sorted(expected_tables):
        record = coverage[table]
        row_counts = record.get("row_counts", {})
        expected_row_keys = {
            "target_in_snapshot_rows",
            "target_external_rows",
            "target_null_rows",
            "target_w_minus_1_rows",
            "target_w_zero_rows",
            "target_malformed_rows",
            "target_self_loop_rows",
        }
        valid = isinstance(row_counts, dict) and set(row_counts) == expected_row_keys
        try:
            counts = {key: int(row_counts[key]) for key in expected_row_keys}
            valid = valid and min(counts.values()) >= 0
            ordinary_rows = (
                counts["target_in_snapshot_rows"] + counts["target_external_rows"]
            )
            partition_rows = ordinary_rows + sum(
                counts[key]
                for key in (
                    "target_null_rows",
                    "target_w_minus_1_rows",
                    "target_w_zero_rows",
                    "target_malformed_rows",
                )
            )
            unique_in = int(record["unique_targets_in_snapshot"])
            unique_external = int(record["unique_targets_external_to_snapshot"])
            unique_valid = int(record["unique_valid_targets"])
            rle_rows = int(record["rle_input_rows"])
            rle_records = int(record["per_file_rle_records"])
            above_max = int(record["external_ids_above_main_max"])
            expected_row_coverage = (
                counts["target_in_snapshot_rows"] / ordinary_rows
                if ordinary_rows
                else None
            )
            expected_unique_coverage = (
                unique_in / unique_valid if unique_valid else None
            )
            duplicate = record["duplicate_source_target_rows"]
            valid = valid and all(
                (
                    record.get("column")
                    == config["scope"]["work_target_columns"][table],
                    partition_rows == int(rows_by_table[table]["counts"]["raw_rows"]),
                    rle_rows == ordinary_rows,
                    record.get("rle_rows_match_ordinary_valid_target_rows") is True,
                    unique_valid == unique_in + unique_external,
                    0 <= unique_in <= counts["target_in_snapshot_rows"],
                    0 <= unique_external <= counts["target_external_rows"],
                    0 <= rle_records <= ordinary_rows,
                    0 <= above_max <= unique_external,
                    counts["target_self_loop_rows"]
                    <= counts["target_in_snapshot_rows"],
                    record.get("row_target_coverage") == expected_row_coverage,
                    record.get("unique_target_coverage") == expected_unique_coverage,
                    duplicate
                    == {
                        "status": "NOT_MEASURED_EXTENSION_PENDING",
                        "deduplication_applied": False,
                    },
                )
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            valid = False
        checks[table] = bool(valid)
    return {"pass": all(checks.values()), "table_checks": checks}


def author_status_codes(total: np.ndarray, valid: np.ndarray) -> np.ndarray:
    status = np.full(len(total), 3, dtype=np.uint8)  # zero-authorship row
    present = total > 0
    status[present & (valid == total)] = 0  # complete
    status[present & (valid > 0) & (valid < total)] = 1  # partial
    status[present & (valid == 0)] = 2  # none usable
    return status


def _add_cross_counts(
    accumulator: np.ndarray, category: np.ndarray, status: np.ndarray
) -> None:
    code = category.astype(np.int64) * accumulator.shape[1] + status.astype(np.int64)
    counts = np.bincount(code, minlength=accumulator.size).reshape(accumulator.shape)
    accumulator += counts.astype(np.uint64)


def available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))


def validate_dense_resource_preflight(
    *,
    max_work_id: int,
    maximum_author_id: int,
    resource_safety: dict[str, Any],
) -> dict[str, Any]:
    if max_work_id < 0 or maximum_author_id < 0:
        raise MembershipAuditError("dense aggregation IDs must be nonnegative")
    author_cap = int(resource_safety["maximum_author_numeric_id_inclusive"])
    if maximum_author_id > author_cap:
        raise MembershipAuditError("author numeric-ID safety cap exceeded")
    estimated_dense_bytes = (max_work_id + 1) * 14 + maximum_author_id + 1
    dense_cap = int(resource_safety["maximum_estimated_dense_bytes"])
    if estimated_dense_bytes > dense_cap:
        raise MembershipAuditError("estimated dense-memory safety cap exceeded")
    available_bytes = available_memory_bytes()
    minimum_available = int(
        np.ceil(
            estimated_dense_bytes
            * float(resource_safety["minimum_available_to_estimated_ratio"])
        )
    )
    if available_bytes < minimum_available:
        raise MembershipAuditError("insufficient available memory for dense aggregation")
    return {
        "maximum_author_numeric_id": maximum_author_id,
        "maximum_author_numeric_id_inclusive": author_cap,
        "estimated_dense_bytes": estimated_dense_bytes,
        "maximum_estimated_dense_bytes": dense_cap,
        "minimum_available_bytes": minimum_available,
        "passed": True,
    }


def author_work_profile(
    *,
    output_dir: Path,
    tasks: list[FileTask],
    metrics_by_index: dict[int, dict[str, Any]],
    max_id: int,
    resource_safety: dict[str, Any],
) -> dict[str, Any]:
    maximum_author_id = 0
    for task in tasks:
        if task.mode != "authorship":
            continue
        metric = metrics_by_index[task.index]
        extras = [
            record
            for record in metric.get("extra_artifacts", [])
            if record.get("kind") == "author_ids"
        ]
        if len(extras) != 1:
            raise MembershipAuditError("author-ID artifact metadata is incomplete")
        maximum_author_id = max(
            maximum_author_id, int(extras[0].get("maximum_numeric_id", 0))
        )
    resource_preflight = validate_dense_resource_preflight(
        max_work_id=max_id,
        maximum_author_id=maximum_author_id,
        resource_safety=resource_safety,
    )
    total = np.zeros(max_id + 1, dtype=np.uint32)
    null_ids = np.zeros(max_id + 1, dtype=np.uint32)
    valid_ids = np.zeros(max_id + 1, dtype=np.uint32)
    lineage = np.zeros(max_id + 1, dtype=np.uint8)
    lineage_bits = {"part": 1, "base": 1, "delta": 2, "repair": 4}
    retained_author_rows = 0
    author_null_id_rows = 0
    author_valid_id_rows = 0
    author_invalid_id_rows = 0
    author_orcid_null_rows = 0
    author_orcid_nonnull_rows = 0
    for task in tasks:
        if task.mode != "authorship":
            continue
        metric = metrics_by_index[task.index]
        array = load_artifact(output_dir, task, metric, "author")
        if np.any(
            np.asarray(array["slots"], dtype=np.uint64)
            != np.asarray(array["null_ids"], dtype=np.uint64)
            + np.asarray(array["valid_ids"], dtype=np.uint64)
            + np.asarray(array["invalid_ids"], dtype=np.uint64)
        ):
            raise MembershipAuditError("author summary artifact is not conserved")
        ids = np.asarray(array["id"], dtype=np.uint64)
        old = total[ids]
        slots = np.asarray(array["slots"], dtype=np.uint32)
        if np.any(old > np.iinfo(np.uint32).max - slots):
            raise MembershipAuditError("authorship work-slot counter overflow")
        total[ids] = old + slots
        null_ids[ids] += np.asarray(array["null_ids"], dtype=np.uint32)
        valid_ids[ids] += np.asarray(array["valid_ids"], dtype=np.uint32)
        lineage[ids] |= np.uint8(lineage_bits[task.lineage])
        retained_author_rows += int(metric["counts"]["source_member_rows"])
        author_null_id_rows += int(metric["counts"]["author_null_id_rows"])
        author_valid_id_rows += int(metric["counts"]["author_valid_id_rows"])
        author_invalid_id_rows += int(metric["counts"]["author_invalid_id_rows"])
        author_orcid_null_rows += int(metric["counts"]["author_orcid_null_rows"])
        author_orcid_nonnull_rows += int(metric["counts"]["author_orcid_nonnull_rows"])

    author_seen = np.zeros(maximum_author_id + 1, dtype=np.uint8)
    for task in tasks:
        if task.mode != "authorship":
            continue
        author_ids_array = load_extra_artifact(
            output_dir,
            task,
            metrics_by_index[task.index],
            "author_ids",
        )
        author_seen[np.asarray(author_ids_array, dtype=np.uint64)] = 1
    distinct_valid_author_ids = 0
    for start in range(0, len(author_seen), 100_000_000):
        distinct_valid_author_ids += int(
            np.count_nonzero(author_seen[start : start + 100_000_000])
        )

    status_names = ["complete", "partial", "none_usable", "zero_authorship_row"]
    overall = np.zeros(4, dtype=np.uint64)
    raw_null_overall = np.zeros(4, dtype=np.uint64)
    year_counts = np.zeros((65536, 4), dtype=np.uint64)
    field_counts = np.zeros((256, 4), dtype=np.uint64)
    team_counts = np.zeros((7, 4), dtype=np.uint64)
    declared_band_counts = np.zeros((9, 4), dtype=np.uint64)
    lineage_counts = np.zeros((8, 4), dtype=np.uint64)
    team_labels = ["0", "1", "2", "3-5", "6-10", "11-50", "51+"]
    declared_band_labels = [
        "NULL_OR_INVALID",
        "0",
        "1",
        "2",
        "3-5",
        "6-10",
        "11-50",
        "51-100",
        "101+",
    ]
    joint_cube: dict[int, int] = {}
    declared_mismatch_pairs: dict[int, int] = {}
    main_rows = 0
    declared_match = 0
    declared_mismatch = 0
    declared_null = 0
    declared_invalid = 0
    zero_record: dict[str, Any] | None = None
    invalid_author_rows = 0

    for task in tasks:
        if task.mode != "main":
            continue
        array = load_artifact(output_dir, task, metrics_by_index[task.index], "main")
        ids = np.asarray(array["id"], dtype=np.uint64)
        observed_total = total[ids]
        observed_null = null_ids[ids]
        observed_valid = valid_ids[ids]
        observed_invalid = observed_total - observed_null - observed_valid
        invalid_author_rows += int(observed_invalid.sum(dtype=np.uint64))
        status = author_status_codes(observed_total, observed_valid)
        overall += np.bincount(status, minlength=4).astype(np.uint64)
        raw_status = author_status_codes(observed_total, observed_total - observed_null)
        raw_null_overall += np.bincount(raw_status, minlength=4).astype(np.uint64)
        years = np.asarray(array["year"], dtype=np.int16)
        year_index = years.astype(np.int32) - YEAR_NULL
        _add_cross_counts(year_counts, year_index, status)
        fields = np.asarray(array["field"], dtype=np.uint8)
        _add_cross_counts(field_counts, fields, status)
        team_bin = np.zeros(len(ids), dtype=np.uint8)
        team_bin[observed_total == 1] = 1
        team_bin[observed_total == 2] = 2
        team_bin[(observed_total >= 3) & (observed_total <= 5)] = 3
        team_bin[(observed_total >= 6) & (observed_total <= 10)] = 4
        team_bin[(observed_total >= 11) & (observed_total <= 50)] = 5
        team_bin[observed_total >= 51] = 6
        _add_cross_counts(team_counts, team_bin, status)
        _add_cross_counts(lineage_counts, lineage[ids], status)
        declared = np.asarray(array["declared_authors"], dtype=np.uint32)
        flags = np.asarray(array["flags"], dtype=np.uint8)
        declared_null_mask = (flags & 1) != 0
        declared_invalid_mask = (flags & 8) != 0
        comparable = ~(declared_null_mask | declared_invalid_mask)
        matches = comparable & (declared == observed_total)
        declared_match += int(np.count_nonzero(matches))
        declared_mismatch += int(
            np.count_nonzero(comparable & (declared != observed_total))
        )
        declared_null += int(np.count_nonzero(declared_null_mask))
        declared_invalid += int(np.count_nonzero(declared_invalid_mask))

        declared_band = np.zeros(len(declared), dtype=np.uint8)
        declared_band[comparable & (declared == 0)] = 1
        declared_band[comparable & (declared == 1)] = 2
        declared_band[comparable & (declared == 2)] = 3
        declared_band[comparable & (declared >= 3) & (declared <= 5)] = 4
        declared_band[comparable & (declared >= 6) & (declared <= 10)] = 5
        declared_band[comparable & (declared >= 11) & (declared <= 50)] = 6
        declared_band[comparable & (declared >= 51) & (declared <= 100)] = 7
        declared_band[comparable & (declared >= 101)] = 8
        _add_cross_counts(declared_band_counts, declared_band, status)
        mismatch_positions = np.flatnonzero(comparable & (declared != observed_total))
        if len(mismatch_positions):
            mismatch_codes = (
                declared[mismatch_positions].astype(np.uint64) << np.uint64(32)
            ) | observed_total[mismatch_positions].astype(np.uint64)
            unique_mismatches, mismatch_counts = np.unique(
                mismatch_codes, return_counts=True
            )
            for code, count in zip(
                unique_mismatches.tolist(), mismatch_counts.tolist(), strict=True
            ):
                declared_mismatch_pairs[int(code)] = declared_mismatch_pairs.get(
                    int(code), 0
                ) + int(count)
        cube_code = (
            (year_index.astype(np.uint64) * np.uint64(256) + fields.astype(np.uint64))
            * np.uint64(len(declared_band_labels))
            + declared_band.astype(np.uint64)
        ) * np.uint64(4) + status.astype(np.uint64)
        unique_codes, unique_counts = np.unique(cube_code, return_counts=True)
        for code, count in zip(
            unique_codes.tolist(), unique_counts.tolist(), strict=True
        ):
            joint_cube[int(code)] = joint_cube.get(int(code), 0) + int(count)
        zero_positions = np.flatnonzero(ids == 0)
        if len(zero_positions):
            position = int(zero_positions[0])
            zero_record = {
                "observed_authorship_slots": int(observed_total[position]),
                "valid_author_ids": int(observed_valid[position]),
                "null_author_ids": int(observed_null[position]),
                "invalid_author_ids": int(observed_invalid[position]),
                "status": status_names[int(status[position])],
            }
        main_rows += len(ids)

    def status_mapping(values: np.ndarray) -> dict[str, int]:
        return {name: int(values[index]) for index, name in enumerate(status_names)}

    years_output: list[dict[str, Any]] = []
    for index in np.flatnonzero(year_counts.sum(axis=1)):
        year = int(index + YEAR_NULL)
        years_output.append(
            {
                "year": None if year == YEAR_NULL else year,
                "statuses": status_mapping(year_counts[index]),
            }
        )
    fields_output: list[dict[str, Any]] = []
    for index in np.flatnonzero(field_counts.sum(axis=1)):
        label: int | str | None
        if index == FIELD_NULL:
            label = None
        elif index == FIELD_INVALID:
            label = "INVALID"
        else:
            label = int(index)
        fields_output.append(
            {"field_id": label, "statuses": status_mapping(field_counts[index])}
        )
    cube_output: list[dict[str, Any]] = []
    for encoded, count in sorted(joint_cube.items()):
        remainder, status_index = divmod(encoded, 4)
        remainder, band_index = divmod(remainder, len(declared_band_labels))
        year_index_value, field_index = divmod(remainder, 256)
        year_value = int(year_index_value + YEAR_NULL)
        if field_index == FIELD_NULL:
            field_value: int | str | None = None
        elif field_index == FIELD_INVALID:
            field_value = "INVALID"
        else:
            field_value = int(field_index)
        cube_output.append(
            {
                "publication_year": None if year_value == YEAR_NULL else year_value,
                "primary_field_id": field_value,
                "declared_authors_band": declared_band_labels[band_index],
                "author_id_status": status_names[status_index],
                "works": count,
            }
        )
    return {
        "resource_preflight": resource_preflight,
        "master_work_rows_profiled": main_rows,
        "grammar_usable_author_id_status": status_mapping(overall),
        "raw_null_only_status": status_mapping(raw_null_overall),
        "invalid_nonnull_author_id_rows": invalid_author_rows,
        "author_id_rows": {
            "null": author_null_id_rows,
            "grammar_valid_non_null": author_valid_id_rows,
            "invalid_non_null": author_invalid_id_rows,
        },
        "distinct_grammar_valid_nonnull_author_ids": distinct_valid_author_ids,
        "retained_authorship_rows": retained_author_rows,
        "author_orcid_rows": {
            "null": author_orcid_null_rows,
            "non_null": author_orcid_nonnull_rows,
        },
        "authorship_partition_conserved": int(overall.sum()) == main_rows,
        "declared_authors_count": {
            "null": declared_null,
            "invalid_nonnull": declared_invalid,
            "matches_observed_slots": declared_match,
            "mismatches_observed_slots": declared_mismatch,
        },
        "by_declared_authors_count_band": [
            {
                "declared_authors_band": label,
                "statuses": status_mapping(declared_band_counts[index]),
            }
            for index, label in enumerate(declared_band_labels)
            if int(declared_band_counts[index].sum())
        ],
        "declared_authors_count_mismatch_profile": [
            {
                "declared_authors_count": encoded >> 32,
                "observed_retained_authorship_slots": encoded & 0xFFFFFFFF,
                "works": count,
            }
            for encoded, count in sorted(declared_mismatch_pairs.items())
        ],
        "by_publication_year": years_output,
        "by_primary_field": fields_output,
        "joint_sparse_cube": cube_output,
        "by_observed_team_size": [
            {"team_size": label, "statuses": status_mapping(team_counts[index])}
            for index, label in enumerate(team_labels)
            if int(team_counts[index].sum())
        ],
        "by_authorship_source_lineage_mask": [
            {
                "lineage_mask": int(index),
                "statuses": status_mapping(lineage_counts[index]),
            }
            for index in range(len(lineage_counts))
            if int(lineage_counts[index].sum())
        ],
        "w0": zero_record,
        "candidate_eligibility_stratum": "NOT_EVALUABLE_G0_PENDING",
        "author_entity_validity_established": False,
    }


def validate_author_work_profile_contract(
    profile: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Bind every required author/work profile output to exact conservation gates."""

    status_names = ("complete", "partial", "none_usable", "zero_authorship_row")

    def status_total(mapping: Any) -> int:
        if not isinstance(mapping, dict) or set(mapping) != set(status_names):
            return -1
        try:
            values = [int(mapping[name]) for name in status_names]
            return sum(values) if all(value >= 0 for value in values) else -1
        except (KeyError, TypeError, ValueError):
            return -1

    def margin_total(records: Any) -> int:
        if not isinstance(records, list):
            return -1
        total_rows = 0
        for record in records:
            if not isinstance(record, dict):
                return -1
            value = status_total(record.get("statuses"))
            if value < 0:
                return -1
            total_rows += value
        return total_rows

    def status_vector(mapping: Any) -> tuple[int, ...] | None:
        if not isinstance(mapping, dict) or set(mapping) != set(status_names):
            return None
        try:
            values = tuple(int(mapping[name]) for name in status_names)
        except (KeyError, TypeError, ValueError):
            return None
        return values if all(value >= 0 for value in values) else None

    def margin_status_vector(records: Any) -> tuple[int, ...] | None:
        if not isinstance(records, list):
            return None
        accumulator = [0] * len(status_names)
        for record in records:
            vector = (
                status_vector(record.get("statuses"))
                if isinstance(record, dict)
                else None
            )
            if vector is None:
                return None
            for index, value in enumerate(vector):
                accumulator[index] += value
        return tuple(accumulator)

    def margin_projection(
        records: Any, label_key: str
    ) -> dict[str, tuple[int, ...]] | None:
        if not isinstance(records, list):
            return None
        output: dict[str, list[int]] = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {label_key, "statuses"}:
                return None
            vector = status_vector(record.get("statuses"))
            if vector is None:
                return None
            label = json.dumps(record[label_key], sort_keys=True)
            if label in output:
                return None
            bucket = output.setdefault(label, [0] * len(status_names))
            for index, value in enumerate(vector):
                bucket[index] += value
        return {key: tuple(values) for key, values in output.items()}

    expected_main_rows = int(config["parents"]["grammar_recount"]["sorted_ids_rows"])
    authorship_contract = config["authorship_author_id_work_profile"]
    expected_authorship_rows = int(
        authorship_contract["expected_retained_rows_on_membership_pass"]
    )
    observed_main_rows = int(profile.get("master_work_rows_profiled", -1))
    observed_authorship_rows = int(profile.get("retained_authorship_rows", -1))
    grammar_status_total = status_total(profile.get("grammar_usable_author_id_status"))
    raw_status_total = status_total(profile.get("raw_null_only_status"))
    overall_vector = status_vector(profile.get("grammar_usable_author_id_status"))
    declared_raw = profile.get("declared_authors_count", {})
    declared = declared_raw if isinstance(declared_raw, dict) else {}
    declared_total = (
        sum(
            int(declared.get(key, -expected_main_rows - 1))
            for key in (
                "null",
                "invalid_nonnull",
                "matches_observed_slots",
                "mismatches_observed_slots",
            )
        )
        if isinstance(declared_raw, dict)
        else -1
    )
    try:
        declared_nonnegative = (
            set(declared)
            == {
                "null",
                "invalid_nonnull",
                "matches_observed_slots",
                "mismatches_observed_slots",
            }
            and min(int(value) for value in declared.values()) >= 0
        )
    except (TypeError, ValueError):
        declared_nonnegative = False
    mismatch_profile = profile.get("declared_authors_count_mismatch_profile")
    mismatch_rows = (
        sum(
            int(record.get("works", -expected_main_rows - 1))
            for record in mismatch_profile
        )
        if isinstance(mismatch_profile, list)
        else -1
    )
    mismatch_profile_valid = isinstance(mismatch_profile, list)
    mismatch_pairs: set[tuple[int, int]] = set()
    if mismatch_profile_valid:
        for record in mismatch_profile:
            if not isinstance(record, dict) or set(record) != {
                "declared_authors_count",
                "observed_retained_authorship_slots",
                "works",
            }:
                mismatch_profile_valid = False
                break
            try:
                pair = (
                    int(record["declared_authors_count"]),
                    int(record["observed_retained_authorship_slots"]),
                )
                works = int(record["works"])
            except (KeyError, TypeError, ValueError):
                mismatch_profile_valid = False
                break
            if (
                min(pair) < 0
                or pair[0] == pair[1]
                or works <= 0
                or pair in mismatch_pairs
            ):
                mismatch_profile_valid = False
                break
            mismatch_pairs.add(pair)
    orcid = profile.get("author_orcid_rows", {})
    orcid_total = (
        int(orcid.get("null", -expected_authorship_rows - 1))
        + int(orcid.get("non_null", -expected_authorship_rows - 1))
        if isinstance(orcid, dict)
        else -1
    )
    try:
        orcid_nonnegative = (
            isinstance(orcid, dict)
            and set(orcid) == {"null", "non_null"}
            and min(int(orcid["null"]), int(orcid["non_null"])) >= 0
        )
    except (KeyError, TypeError, ValueError):
        orcid_nonnegative = False
    author_rows_raw = profile.get("author_id_rows", {})
    author_rows = author_rows_raw if isinstance(author_rows_raw, dict) else {}
    author_null_rows = int(author_rows.get("null", -1))
    author_valid_rows = int(author_rows.get("grammar_valid_non_null", -1))
    author_invalid_rows = int(author_rows.get("invalid_non_null", -1))
    author_id_row_total = author_null_rows + author_valid_rows + author_invalid_rows
    distinct_author_ids = int(
        profile.get("distinct_grammar_valid_nonnull_author_ids", -1)
    )
    zero = profile.get("w0")
    expected_zero_slots = int(authorship_contract["w_zero_expected_source_rows"])
    zero_exact = isinstance(zero, dict) and all(
        (
            int(zero.get("observed_authorship_slots", -1)) == expected_zero_slots,
            int(zero.get("null_author_ids", -1)) == expected_zero_slots,
            int(zero.get("valid_author_ids", -1)) == 0,
            int(zero.get("invalid_author_ids", -1)) == 0,
            zero.get("status") == "none_usable",
        )
    )
    year_margin = profile.get("by_publication_year")
    field_margin = profile.get("by_primary_field")
    declared_band_margin = profile.get("by_declared_authors_count_band")
    team_margin = profile.get("by_observed_team_size")
    lineage_margin = profile.get("by_authorship_source_lineage_mask")
    year_projection = margin_projection(year_margin, "year")
    field_projection = margin_projection(field_margin, "field_id")
    declared_band_projection = margin_projection(
        declared_band_margin, "declared_authors_band"
    )
    team_projection = margin_projection(team_margin, "team_size")
    lineage_projection = margin_projection(lineage_margin, "lineage_mask")
    cube = profile.get("joint_sparse_cube")
    cube_status = [0] * len(status_names)
    cube_year: dict[str, list[int]] = {}
    cube_field: dict[str, list[int]] = {}
    cube_declared: dict[str, list[int]] = {}
    cube_valid = isinstance(cube, list)
    cube_cells: set[tuple[str, str, str, str]] = set()
    if cube_valid:
        for record in cube:
            if not isinstance(record, dict) or set(record) != {
                "publication_year",
                "primary_field_id",
                "declared_authors_band",
                "author_id_status",
                "works",
            }:
                cube_valid = False
                break
            try:
                status_index = status_names.index(str(record["author_id_status"]))
                works = int(record["works"])
                if works <= 0:
                    raise ValueError
                labels = (
                    (cube_year, record["publication_year"]),
                    (cube_field, record["primary_field_id"]),
                    (cube_declared, record["declared_authors_band"]),
                )
            except (KeyError, TypeError, ValueError):
                cube_valid = False
                break
            cell = (
                json.dumps(record["publication_year"], sort_keys=True),
                json.dumps(record["primary_field_id"], sort_keys=True),
                json.dumps(record["declared_authors_band"], sort_keys=True),
                str(record["author_id_status"]),
            )
            if cell in cube_cells:
                cube_valid = False
                break
            cube_cells.add(cell)
            cube_status[status_index] += works
            for projection, raw_label in labels:
                label = json.dumps(raw_label, sort_keys=True)
                projection.setdefault(label, [0] * len(status_names))[status_index] += (
                    works
                )
    cube_status_vector = tuple(cube_status) if cube_valid else None
    cube_year_projection = (
        {key: tuple(values) for key, values in cube_year.items()}
        if cube_valid
        else None
    )
    cube_field_projection = (
        {key: tuple(values) for key, values in cube_field.items()}
        if cube_valid
        else None
    )
    cube_declared_projection = (
        {key: tuple(values) for key, values in cube_declared.items()}
        if cube_valid
        else None
    )
    totals = {
        "grammar_usable_author_id_status": grammar_status_total,
        "raw_null_only_status": raw_status_total,
        "publication_year_margin": margin_total(year_margin),
        "primary_field_margin": margin_total(field_margin),
        "declared_authors_band_margin": margin_total(declared_band_margin),
        "observed_team_size_margin": margin_total(team_margin),
        "authorship_lineage_margin": margin_total(lineage_margin),
        "joint_sparse_cube": (
            sum(
                int(record.get("works", -expected_main_rows - 1))
                for record in profile.get("joint_sparse_cube", [])
            )
            if isinstance(profile.get("joint_sparse_cube"), list)
            else -1
        ),
        "declared_authors_categories": declared_total,
        "declared_authors_mismatch_profile": mismatch_rows,
        "author_id_rows": author_id_row_total,
        "author_orcid_rows": orcid_total,
    }
    checks = {
        "master_work_rows_exact": observed_main_rows == expected_main_rows,
        "retained_authorship_rows_exact": (
            observed_authorship_rows == expected_authorship_rows
        ),
        "grammar_usable_partition_conserved": (
            profile.get("authorship_partition_conserved") is True
            and grammar_status_total == expected_main_rows
        ),
        "raw_null_only_partition_conserved": raw_status_total == expected_main_rows,
        "publication_year_status_projection_exact": (
            margin_status_vector(year_margin) == overall_vector
        ),
        "primary_field_status_projection_exact": (
            margin_status_vector(field_margin) == overall_vector
        ),
        "declared_authors_band_status_projection_exact": (
            margin_status_vector(declared_band_margin) == overall_vector
        ),
        "observed_team_size_status_projection_exact": (
            team_projection is not None
            and margin_status_vector(team_margin) == overall_vector
        ),
        "authorship_lineage_status_projection_exact": (
            lineage_projection is not None
            and margin_status_vector(lineage_margin) == overall_vector
        ),
        "joint_cube_status_projection_exact": cube_status_vector == overall_vector,
        "joint_cube_year_projection_exact": (
            year_projection is not None and cube_year_projection == year_projection
        ),
        "joint_cube_primary_field_projection_exact": (
            field_projection is not None and cube_field_projection == field_projection
        ),
        "joint_cube_declared_authors_projection_exact": (
            declared_band_projection is not None
            and cube_declared_projection == declared_band_projection
        ),
        "publication_year_margin_conserved": (
            totals["publication_year_margin"] == expected_main_rows
        ),
        "primary_field_margin_conserved": (
            totals["primary_field_margin"] == expected_main_rows
        ),
        "declared_authors_band_margin_conserved": (
            totals["declared_authors_band_margin"] == expected_main_rows
        ),
        "observed_team_size_margin_conserved": (
            totals["observed_team_size_margin"] == expected_main_rows
        ),
        "authorship_lineage_margin_conserved": (
            totals["authorship_lineage_margin"] == expected_main_rows
        ),
        "joint_sparse_cube_conserved": (
            totals["joint_sparse_cube"] == expected_main_rows
        ),
        "declared_authors_categories_conserved": (
            declared_nonnegative
            and totals["declared_authors_categories"] == expected_main_rows
        ),
        "declared_authors_mismatch_profile_conserved": (
            mismatch_profile_valid
            and totals["declared_authors_mismatch_profile"]
            == int(declared.get("mismatches_observed_slots", -1))
        ),
        "author_orcid_rows_conserved": (
            orcid_nonnegative
            and totals["author_orcid_rows"] == expected_authorship_rows
        ),
        "author_id_rows_conserved": (
            totals["author_id_rows"] == expected_authorship_rows
            and min(author_null_rows, author_valid_rows, author_invalid_rows) >= 0
        ),
        "distinct_author_ids_bounded": (
            0 <= distinct_author_ids <= author_valid_rows
            and (author_valid_rows == 0 or distinct_author_ids > 0)
        ),
        "invalid_nonnull_author_ids_absent": (
            int(profile.get("invalid_nonnull_author_id_rows", -1)) == 0
            and author_invalid_rows == 0
        ),
        "w_zero_profile_exact": zero_exact,
        "candidate_eligibility_not_inferred": (
            profile.get("candidate_eligibility_stratum") == "NOT_EVALUABLE_G0_PENDING"
        ),
        "author_entity_validity_not_inferred": (
            profile.get("author_entity_validity_established") is False
        ),
    }
    return {
        "pass": all(checks.values()),
        "expected": {
            "master_work_rows": expected_main_rows,
            "retained_authorship_rows": expected_authorship_rows,
            "w_zero_authorship_slots": expected_zero_slots,
        },
        "observed": {
            "master_work_rows": observed_main_rows,
            "retained_authorship_rows": observed_authorship_rows,
            **totals,
        },
        "checks": checks,
    }


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def aggregate_report(
    *,
    root: Path,
    config: dict[str, Any],
    config_path: Path,
    tool_path: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    plan: dict[str, Any],
    tasks: list[FileTask],
    metrics: list[dict[str, Any]],
    output_dir: Path,
    membership: np.ndarray,
    started_at: str,
    elapsed_seconds: float,
    parent_stable: bool,
) -> dict[str, Any]:
    full_scope = bool(plan["full_scope"])
    table_rows = aggregate_table_metrics(manifest, metrics)
    sentinel_contract = validate_sentinel_contract(
        table_rows=table_rows, config=config, full_scope=full_scope
    )
    blocking_identifier_ledger_contract = validate_blocking_identifier_ledger(
        table_rows,
        config,
        root=root,
        tasks=tasks,
        metrics=metrics,
        membership=membership,
    )
    metrics_by_index = {int(row["task_index"]): row for row in metrics}
    target_coverage: dict[str, Any] = {}
    for table in config["scope"]["work_target_columns"]:
        table_row = next(row for row in table_rows if row["table"] == table)
        row_counts = table_row["counts"]
        valid_rows = int(row_counts["target_in_snapshot_rows"]) + int(
            row_counts["target_external_rows"]
        )
        unique = target_unique_coverage(
            output_dir=output_dir,
            tasks=tasks,
            metrics_by_index=metrics_by_index,
            membership=membership,
            table=str(table),
        )
        target_coverage[str(table)] = {
            "column": config["scope"]["work_target_columns"][table],
            "row_counts": {
                key: int(row_counts[key])
                for key in (
                    "target_in_snapshot_rows",
                    "target_external_rows",
                    "target_null_rows",
                    "target_w_minus_1_rows",
                    "target_w_zero_rows",
                    "target_malformed_rows",
                    "target_self_loop_rows",
                )
            },
            "row_target_coverage": (
                int(row_counts["target_in_snapshot_rows"]) / valid_rows
                if valid_rows
                else None
            ),
            **unique,
            "rle_rows_match_ordinary_valid_target_rows": unique["rle_input_rows"]
            == valid_rows,
            "duplicate_source_target_rows": {
                "status": "NOT_MEASURED_EXTENSION_PENDING",
                "deduplication_applied": False,
            },
        }
    target_coverage_contract = validate_target_coverage_contract(
        target_coverage, table_rows, config
    )
    author_profile = author_work_profile(
        output_dir=output_dir,
        tasks=tasks,
        metrics_by_index=metrics_by_index,
        max_id=len(membership) - 1,
        resource_safety=config["execution"]["resource_safety"],
    )
    author_profile_contract = validate_author_work_profile_contract(
        author_profile, config
    )
    authorship_table = str(config["scope"]["authorship"]["table"])
    authorship_row = next(row for row in table_rows if row["table"] == authorship_table)
    authorship_counts = authorship_row["counts"]
    authorship_expected = config["authorship_author_id_work_profile"]
    author_source_contract = {
        "raw_rows_match": int(authorship_counts["raw_rows"])
        == int(authorship_expected["expected_raw_rows"]),
        "retained_rows_match": int(authorship_counts["source_member_rows"])
        == int(authorship_expected["expected_retained_rows_on_membership_pass"]),
        "raw_null_author_id_rows_match": int(
            authorship_counts["author_raw_null_id_rows"]
        )
        == int(authorship_expected["expected_raw_author_id_null_rows_from_footer"]),
        "raw_invalid_nonnull_author_id_rows": int(
            authorship_counts["author_raw_invalid_id_rows"]
        ),
        "w_minus_1_author_id_null_rows": int(
            authorship_counts["author_w_minus_1_null_id_rows"]
        ),
        "w_zero_author_id_null_rows": int(
            authorship_counts["author_w_zero_null_id_rows"]
        ),
    }
    author_source_contract["pass"] = all(
        (
            author_source_contract["raw_rows_match"],
            author_source_contract["retained_rows_match"],
            author_source_contract["raw_null_author_id_rows_match"],
            author_source_contract["raw_invalid_nonnull_author_id_rows"] == 0,
            author_source_contract["w_minus_1_author_id_null_rows"] == 1,
            author_source_contract["w_zero_author_id_null_rows"] == 1,
        )
    )
    source_blocks = sum(
        int(row["counts"][key])
        for row in table_rows
        for key in (
            "source_orphan_rows",
            "source_null_rows",
            "source_malformed_other_rows",
        )
    )
    target_blocks = sum(
        int(row["counts"][key])
        for row in table_rows
        if row["mode"] == "target"
        for key in (
            "target_null_rows",
            "target_w_minus_1_rows",
            "target_w_zero_rows",
            "target_malformed_rows",
        )
    )
    main_profile_blocks = sum(
        int(row["counts"][key])
        for row in table_rows
        if row["mode"] == "main"
        for key in (
            "main_year_invalid_rows",
            "main_field_invalid_rows",
            "main_declared_authors_invalid_rows",
        )
    )
    target_rle_conserved = all(
        record["rle_rows_match_ordinary_valid_target_rows"]
        for record in target_coverage.values()
    )
    complete_files = all(
        row["scanned_files"] == row["expected_parent_files"] for row in table_rows
    )
    complete_rows = all(
        row["scanned_footer_rows"] == row["expected_parent_footer_rows"]
        for row in table_rows
    )
    expected_table_names = {
        str(config["scope"]["main_table"]),
        *map(str, config["scope"]["child_source_tables"]),
    }
    table_set_exact = {str(row["table"]) for row in table_rows} == expected_table_names
    source_rows_conserved = all(
        row.get("source_row_conservation") is True for row in table_rows
    )
    global_expected = config["source_row_partition"]["global_identity_on_pass"]
    global_observed = {
        "files": len(metrics),
        "raw_rows": sum(int(row["counts"]["raw_rows"]) for row in table_rows),
        "retained_rows": sum(
            int(row["counts"]["source_member_rows"]) for row in table_rows
        ),
        "frozen_w_minus_1_quarantine_rows": sum(
            int(row["counts"]["source_w_minus_1_rows"]) for row in table_rows
        ),
        "all_other_partition_rows": source_blocks,
    }
    global_source_checks = {
        "files_exact": global_observed["files"]
        == int(config["parent_inventory"]["expected_parquet_file_count"]),
        "raw_rows_exact": global_observed["raw_rows"]
        == int(global_expected["raw_rows"]),
        "retained_rows_exact": global_observed["retained_rows"]
        == int(global_expected["retained_rows"]),
        "frozen_quarantine_rows_exact": global_observed[
            "frozen_w_minus_1_quarantine_rows"
        ]
        == int(global_expected["frozen_w_minus_1_quarantine_rows"]),
        "all_other_partition_rows_exact": global_observed["all_other_partition_rows"]
        == int(global_expected["all_other_partition_rows"]),
    }
    global_source_contract = {
        "pass": all(global_source_checks.values()),
        "expected": {
            "files": int(config["parent_inventory"]["expected_parquet_file_count"]),
            **{key: int(value) for key, value in global_expected.items()},
        },
        "observed": global_observed,
        "checks": global_source_checks,
    }
    sentinel_exact = (
        sentinel_contract["w_minus_1_exact_match"] is True
        and sentinel_contract["w0_exact_match"] is True
        and sentinel_contract["w_minus_1_locator_ledger_exact_match"] is True
        and sentinel_contract["w0_locator_ledger_exact_match"] is True
        and all(
            value is True
            for value in sentinel_contract["main_locator_exact_match"].values()
        )
    )
    main_accounting_pass = (
        full_scope
        and table_set_exact
        and complete_files
        and complete_rows
        and author_profile_contract["checks"]["master_work_rows_exact"] is True
        and sentinel_exact
    )
    child_source_membership_pass = (
        full_scope
        and table_set_exact
        and complete_files
        and complete_rows
        and source_blocks == 0
        and sentinel_exact
        and blocking_identifier_ledger_contract["domain_passes"]["source"] is True
    )
    source_row_conservation_pass = (
        full_scope
        and table_set_exact
        and complete_files
        and complete_rows
        and source_rows_conserved
        and global_source_contract["pass"] is True
        and blocking_identifier_ledger_contract["domain_passes"]["source"] is True
    )
    target_scan_pass = (
        full_scope
        and complete_files
        and complete_rows
        and target_blocks == 0
        and target_rle_conserved
        and target_coverage_contract["pass"] is True
        and blocking_identifier_ledger_contract["domain_passes"]["target"] is True
    )
    authorship_profile_pass = (
        full_scope
        and author_profile_contract["pass"] is True
        and author_source_contract["pass"] is True
        and blocking_identifier_ledger_contract["domain_passes"]["author"] is True
    )
    work_profile_strata_pass = (
        full_scope
        and main_profile_blocks == 0
        and author_profile_contract["pass"] is True
        and blocking_identifier_ledger_contract["domain_passes"]["main_field"]
        is True
    )
    component_states = {
        "main_accounting": (
            "PASS_PARENT_RECOUNT_REVALIDATED" if main_accounting_pass else "NOT_PASS"
        ),
        "child_source_membership": (
            "PASS" if child_source_membership_pass else "NOT_PASS"
        ),
        "source_row_conservation": (
            "PASS" if source_row_conservation_pass else "NOT_PASS"
        ),
        "declared_work_target_scan": (
            "COMPLETE_COVERAGE_MEASURED" if target_scan_pass else "NOT_COMPLETE"
        ),
        "authorship_author_id_work_profile": (
            "COMPLETE_PROFILE_MEASURED" if authorship_profile_pass else "NOT_COMPLETE"
        ),
        "work_profile_strata": (
            "COMPLETE_PROFILE_MEASURED" if work_profile_strata_pass else "NOT_COMPLETE"
        ),
        "raw_source_id_validity": "BLOCKED",
        "w_zero_entity_validity": "UNRESOLVED",
        "parent_content_identity": "UNVERIFIED",
        "g_minus_1_data_readiness": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
    }
    required_component_states = config["success_contract"]["required_component_states"]
    required_component_states_match = component_states == required_component_states
    component_pass = (
        full_scope
        and main_accounting_pass
        and child_source_membership_pass
        and source_row_conservation_pass
        and target_scan_pass
        and authorship_profile_pass
        and work_profile_strata_pass
        and required_component_states_match
        and parent_stable
    )
    status = (
        str(config["success_contract"]["success_status"])
        if component_pass
        else ("COMPLETE_ENGINEERING_PILOT" if not full_scope else "BLOCKED")
    )
    output_evidence_expectation = expected_output_evidence(tasks)
    deterministic = {
        "plan_sha256": plan["plan_sha256"],
        "table_rows": table_rows,
        "sentinel_contract": sentinel_contract,
        "target_coverage": target_coverage,
        "target_coverage_contract": target_coverage_contract,
        "author_profile": author_profile,
        "author_profile_contract": author_profile_contract,
        "author_source_contract": author_source_contract,
        "global_source_contract": global_source_contract,
        "blocking_identifier_ledger_contract": blocking_identifier_ledger_contract,
        "component_states": component_states,
        "required_component_states_match": required_component_states_match,
        "master_ids_stable_after_audit": True,
        "output_evidence_expectation": output_evidence_expectation,
        "component_pass": component_pass,
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": str(config["label"]),
        "generated_at_utc": utc_now(),
        "status": status,
        "gate_status": {
            **component_states,
            "required_component_states_match": required_component_states_match,
            "work_source_membership_component": "PASS"
            if component_pass
            else "NOT_PASS",
            "target_coverage": "MEASURED" if full_scope else "PILOT_ONLY",
            "target_space_closed_world": False,
            "analysis_view_materialized": False,
            "author_missingness_profile": "MEASURED" if full_scope else "PILOT_ONLY",
            "author_entity_validity_established": False,
            "parent_content_identity_verified": False,
            "g1_empirical_signature": "NOT_RUN",
        },
        "scope": {
            "full_scope": full_scope,
            "source_table_count": len(table_rows),
            "child_source_table_count": 22 if full_scope else None,
            "files_scanned": len(metrics),
            "data_page_rows_scanned": sum(
                int(row["counts"]["raw_rows"]) for row in table_rows
            ),
            "work_target_columns": config["scope"]["work_target_columns"],
        },
        "source": {
            "source_root_id": config["parents"]["inventory"]["source_root_id"],
            "parent_manifest_sha256": manifest_sha256,
            "parent_inventory_fingerprint_sha256": config["parents"]["inventory"][
                "inventory_fingerprint_sha256"
            ],
            "parent_stability_fingerprint_sha256": config["parents"]["inventory"][
                "stability_fingerprint_sha256"
            ],
            "parent_stable_after_audit": parent_stable,
            "master_ids_stable_after_audit": True,
        },
        "provenance": {
            "config": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": plan["config_sha256"],
            "tool": tool_path.relative_to(REPO_ROOT).as_posix(),
            "tool_sha256": plan["tool_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "grammar_recount_summary_sha256": config["parents"]["grammar_recount"][
                "summary_sha256"
            ],
            "sorted_master_ids_sha256": config["parents"]["grammar_recount"][
                "sorted_ids_sha256"
            ],
            "sentinel_result_fingerprint_sha256": config["parents"][
                "sentinel_incidence"
            ]["result_fingerprint_sha256"],
            "result_fingerprint_sha256": sha256_bytes(
                canonical_json_bytes(deterministic)
            ),
            "git": git_metadata(),
            "path_redacted": True,
        },
        "execution": {
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": elapsed_seconds,
            "batch_size": int(plan["batch_size"]),
            "workers": int(plan["workers"]),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "platform": platform.platform(),
            },
        },
        "output_evidence_expectation": output_evidence_expectation,
        "source_tables": table_rows,
        "global_source_contract": global_source_contract,
        "sentinel_contract": sentinel_contract,
        "blocking_identifier_ledger_contract": blocking_identifier_ledger_contract,
        "target_coverage": target_coverage,
        "target_coverage_contract": target_coverage_contract,
        "author_work_profile": author_profile,
        "author_profile_contract": author_profile_contract,
        "author_source_contract": author_source_contract,
        "warnings": [
            "WARN_ZERO_ID_UNRESOLVED",
            "WARN_PARENT_CONTENT_SHA256_PENDING",
            "WARN_ANALYSIS_VIEW_NOT_MATERIALIZED",
            "WARN_AUTHOR_ENTITY_VALIDITY_PENDING",
            "WARN_CANDIDATE_ELIGIBILITY_G0_PENDING",
        ],
        "pending": list(config.get("pending", [])),
        "claim_boundary": str(config["claim_boundary"]),
    }
    encoded = json.dumps(report, sort_keys=True)
    if str(output_dir) in encoded:
        raise MembershipAuditError("absolute output path leaked into report")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# OpenAlex Work-Source Membership and Author-Use Audit",
        "",
        f"- Status: `{report['status']}`",
        "- Raw source ID validity remains `BLOCKED` by the frozen W-1 row.",
        "- G-1 data readiness is `PENDING`; empirical G1 is `NOT_RUN`.",
        "- The audit does not materialize a filtered analysis view.",
        "",
        "## Source membership",
        "",
        "| Table | Raw rows | Retained members | W-1 | W0 | Orphans | Null | Other malformed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["source_tables"]:
        counts = row["counts"]
        lines.append(
            f"| `{row['table']}` | {counts['raw_rows']:,} | "
            f"{counts['source_member_rows']:,} | "
            f"{counts['source_w_minus_1_rows']:,} | "
            f"{counts['source_w_zero_rows']:,} | "
            f"{counts['source_orphan_rows']:,} | {counts['source_null_rows']:,} | "
            f"{counts['source_malformed_other_rows']:,} |"
        )
    lines.extend(["", "## Work-target coverage", ""])
    for table, coverage in report["target_coverage"].items():
        rows = coverage["row_counts"]
        lines.extend(
            [
                f"### `{table}.{coverage['column']}`",
                "",
                f"- In-snapshot target rows: {rows['target_in_snapshot_rows']:,}",
                f"- Snapshot-external valid target rows: {rows['target_external_rows']:,}",
                f"- Unique valid targets: {coverage['unique_valid_targets']:,}",
                f"- Unique targets in snapshot: {coverage['unique_targets_in_snapshot']:,}",
                f"- Null / malformed / W-1 targets: {rows['target_null_rows']:,} / "
                f"{rows['target_malformed_rows']:,} / {rows['target_w_minus_1_rows']:,}",
                "- Snapshot-external targets are coverage observations, not source orphans.",
                "",
            ]
        )
    profile = report["author_work_profile"]
    lines.extend(["## Work-level author-ID profile", ""])
    for name in ("complete", "partial", "none_usable", "zero_authorship_row"):
        count = profile["grammar_usable_author_id_status"][name]
        lines.append(f"- `{name}`: {count:,} works")
    lines.extend(
        [
            f"- Non-null malformed author-ID rows: {profile['invalid_nonnull_author_id_rows']:,}",
            f"- Declared/observed team-size mismatches: "
            f"{profile['declared_authors_count']['mismatches_observed_slots']:,}",
            "- Grammar-valid author IDs are not evidence of entity validity.",
            "",
            "## Provenance and remaining boundary",
            "",
            f"- Parent manifest SHA-256: `{report['source']['parent_manifest_sha256']}`",
            f"- Config SHA-256: `{report['provenance']['config_sha256']}`",
            f"- Tool SHA-256: `{report['provenance']['tool_sha256']}`",
            f"- Result fingerprint: `{report['provenance']['result_fingerprint_sha256']}`",
            "- Parent path/size/mtime matched at start and end; full content SHA-256 is pending.",
            "- Author split/merge validity, dates/censoring/leakage, a representative pilot, "
            "G-1, empirical G1, and every scientific effect remain pending.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_output_dir(output_dir: Path, plan: dict[str, Any]) -> None:
    if output_dir.exists() and output_dir.is_symlink():
        raise MembershipAuditError("output directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "audit_plan.json"
    if plan_path.exists():
        observed = json.loads(plan_path.read_text(encoding="utf-8"))
        if observed != plan:
            raise MembershipAuditError("existing output plan differs from this run")
    else:
        atomic_write_json(plan_path, plan)
    for directory_name in ("checkpoints", "artifacts"):
        directory = output_dir / directory_name
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise MembershipAuditError(f"invalid output directory: {directory_name}")
        directory.mkdir(exist_ok=True)


def ensure_task_output_directories(output_dir: Path, task: FileTask) -> None:
    if Path(task.table).name != task.table or task.table in {"", ".", ".."}:
        raise MembershipAuditError("invalid output table path component")
    directory_names = ["checkpoints"]
    if task.mode in {"authorship", "target", "main"}:
        directory_names.append("artifacts")
    for directory_name in directory_names:
        root = output_dir / directory_name
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise MembershipAuditError(f"invalid output directory: {directory_name}")
        root.mkdir(parents=True, exist_ok=True)
        table_directory = root / task.table
        if table_directory.exists() and (
            table_directory.is_symlink() or not table_directory.is_dir()
        ):
            raise MembershipAuditError(
                f"invalid output table directory: {directory_name}/{task.table}"
            )
        table_directory.mkdir(exist_ok=True)


def validate_output_location(root: Path, output_dir: Path) -> Path:
    output = Path(os.path.abspath(os.fspath(output_dir.expanduser())))
    source = root.resolve()
    current = output
    while True:
        if current.is_symlink():
            raise MembershipAuditError("output path must not contain symlinks")
        if current == current.parent:
            break
        current = current.parent
    resolved_output = output.resolve(strict=False)
    if resolved_output == source or resolved_output.is_relative_to(source):
        raise MembershipAuditError("output directory must be outside source root")
    return output


FINAL_REPORT_JSON = "work_source_membership_audit.json"
FINAL_REPORT_MARKDOWN = "work_source_membership_audit.md"
FINAL_OUTPUT_MANIFEST = "work_source_membership_audit.outputs.json"


def _final_artifact_record(path: Path) -> dict[str, Any]:
    file_stat = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
        raise MembershipAuditError(f"final output must be a regular file: {path.name}")
    return {
        "relative_path": path.name,
        "bytes": int(file_stat.st_size),
        "sha256": sha256_file(path),
    }


def output_tree_commitment(output_dir: Path, directory_name: str) -> dict[str, Any]:
    root = output_dir / directory_name
    if root.is_symlink() or not root.is_dir():
        raise MembershipAuditError(f"output tree is missing: {directory_name}")
    records: list[dict[str, Any]] = []

    def fail_walk(error: OSError) -> None:
        raise MembershipAuditError(
            f"cannot traverse output tree {directory_name}: {error}"
        ) from error

    for current, directories, filenames in os.walk(
        root, followlinks=False, onerror=fail_walk
    ):
        current_path = Path(current)
        if current_path != root and not directories and not filenames:
            raise MembershipAuditError(
                f"output tree contains an empty directory: {directory_name}"
            )
        for name in directories:
            directory = current_path / name
            if directory.is_symlink():
                raise MembershipAuditError(
                    f"output tree contains a symlink: {directory_name}"
                )
        for name in filenames:
            path = current_path / name
            file_stat = path.stat(follow_symlinks=False)
            if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
                raise MembershipAuditError(
                    f"output tree contains a non-regular file: {directory_name}"
                )
            records.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "bytes": int(file_stat.st_size),
                    "sha256": sha256_file(path),
                }
            )
    records.sort(key=lambda record: str(record["relative_path"]))
    digest = hashlib.sha256()
    path_digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json_bytes(record))
        digest.update(b"\n")
        path_digest.update(str(record["relative_path"]).encode("utf-8"))
        path_digest.update(b"\n")
    return {
        "relative_path": directory_name,
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "path_set_sha256": path_digest.hexdigest(),
        "tree_sha256": digest.hexdigest(),
    }


def validate_final_outputs(
    output_dir: Path, *, expected_result_fingerprint: str | None = None
) -> dict[str, Any] | None:
    """Validate the committed final JSON/Markdown and their SHA sidecar."""

    json_path = output_dir / FINAL_REPORT_JSON
    markdown_path = output_dir / FINAL_REPORT_MARKDOWN
    manifest_path = output_dir / FINAL_OUTPUT_MANIFEST
    paths = (json_path, markdown_path, manifest_path)
    present = [path.exists() or path.is_symlink() for path in paths]
    if not any(present):
        return None
    if not all(present):
        raise MembershipAuditError("final output set is incomplete")
    for path in paths:
        file_stat = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
            raise MembershipAuditError(
                f"final output must be a regular file: {path.name}"
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_manifest_fields = {
        "schema_version",
        "result_fingerprint_sha256",
        "artifacts",
        "audit_plan",
        "checkpoint_tree",
        "data_artifact_tree",
        "output_evidence_expectation",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_manifest_fields:
        raise MembershipAuditError("final output manifest fields differ")
    if manifest.get("schema_version") != OUTPUT_MANIFEST_SCHEMA_VERSION:
        raise MembershipAuditError("final output manifest schema differs")
    result_fingerprint = validate_sha256(
        manifest.get("result_fingerprint_sha256"),
        field="final output result_fingerprint_sha256",
    )
    if (
        expected_result_fingerprint is not None
        and result_fingerprint != expected_result_fingerprint
    ):
        raise MembershipAuditError("existing final report fingerprint differs")
    expected_names = {
        "report_json": FINAL_REPORT_JSON,
        "report_markdown": FINAL_REPORT_MARKDOWN,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_names):
        raise MembershipAuditError("final output artifact manifest differs")
    for kind, name in expected_names.items():
        record = artifacts[kind]
        if (
            not isinstance(record, dict)
            or set(record) != {"relative_path", "bytes", "sha256"}
            or record.get("relative_path") != name
            or type(record.get("bytes")) is not int
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is None
        ):
            raise MembershipAuditError("final output relative path differs")
        path = output_dir / name
        observed = _final_artifact_record(path)
        if (
            int(record.get("bytes", -1)) != observed["bytes"]
            or record.get("sha256") != observed["sha256"]
        ):
            raise MembershipAuditError(f"final output SHA-256 changed: {name}")
    plan_record = manifest.get("audit_plan")
    if (
        not isinstance(plan_record, dict)
        or set(plan_record) != {"relative_path", "bytes", "sha256"}
        or type(plan_record.get("bytes")) is not int
        or re.fullmatch(r"[0-9a-f]{64}", str(plan_record.get("sha256"))) is None
    ):
        raise MembershipAuditError("audit-plan output commitment is missing")
    observed_plan = _final_artifact_record(output_dir / "audit_plan.json")
    if plan_record != observed_plan:
        raise MembershipAuditError("audit-plan output SHA-256 changed")
    for field, directory in (
        ("checkpoint_tree", "checkpoints"),
        ("data_artifact_tree", "artifacts"),
    ):
        tree_record = manifest.get(field)
        if (
            not isinstance(tree_record, dict)
            or set(tree_record)
            != {
                "relative_path",
                "files",
                "bytes",
                "path_set_sha256",
                "tree_sha256",
            }
            or type(tree_record.get("files")) is not int
            or type(tree_record.get("bytes")) is not int
            or re.fullmatch(
                r"[0-9a-f]{64}", str(tree_record.get("path_set_sha256"))
            )
            is None
            or re.fullmatch(r"[0-9a-f]{64}", str(tree_record.get("tree_sha256")))
            is None
        ):
            raise MembershipAuditError(f"{directory} output tree record differs")
        observed_tree = output_tree_commitment(output_dir, directory)
        if tree_record != observed_tree:
            raise MembershipAuditError(f"{directory} output tree changed")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
    ):
        raise MembershipAuditError("final JSON report schema differs")
    if (
        report.get("provenance", {}).get("result_fingerprint_sha256")
        != result_fingerprint
    ):
        raise MembershipAuditError("final JSON fingerprint differs from manifest")
    if markdown_path.read_text(encoding="utf-8") != render_markdown(report):
        raise MembershipAuditError("final Markdown does not render from final JSON")
    expectation = report.get("output_evidence_expectation")
    if manifest.get("output_evidence_expectation") != expectation:
        raise MembershipAuditError("output evidence expectation differs")
    if not isinstance(expectation, dict):
        raise MembershipAuditError("output evidence expectation is missing")
    if (
        set(expectation)
        != {
            "checkpoint_files",
            "checkpoint_path_set_sha256",
            "data_artifact_files",
            "data_artifact_path_set_sha256",
        }
        or type(expectation.get("checkpoint_files")) is not int
        or type(expectation.get("data_artifact_files")) is not int
        or re.fullmatch(
            r"[0-9a-f]{64}", str(expectation.get("checkpoint_path_set_sha256"))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(expectation.get("data_artifact_path_set_sha256"))
        )
        is None
    ):
        raise MembershipAuditError("output evidence expectation fields differ")
    if (
        int(manifest["checkpoint_tree"]["files"])
        != int(expectation.get("checkpoint_files", -1))
        or manifest["checkpoint_tree"].get("path_set_sha256")
        != expectation.get("checkpoint_path_set_sha256")
        or int(manifest["data_artifact_tree"]["files"])
        != int(expectation.get("data_artifact_files", -1))
        or manifest["data_artifact_tree"].get("path_set_sha256")
        != expectation.get("data_artifact_path_set_sha256")
    ):
        raise MembershipAuditError("output evidence set is incomplete")
    allowed_top_level = {
        "audit.lock",
        "audit_plan.json",
        FINAL_REPORT_JSON,
        FINAL_REPORT_MARKDOWN,
        FINAL_OUTPUT_MANIFEST,
        "checkpoints",
        "artifacts",
    }
    top_level = {path.name: path for path in output_dir.iterdir()}
    if set(top_level) != allowed_top_level:
        raise MembershipAuditError("unexpected top-level output entry")
    for name in (
        "audit.lock",
        "audit_plan.json",
        FINAL_REPORT_JSON,
        FINAL_REPORT_MARKDOWN,
        FINAL_OUTPUT_MANIFEST,
    ):
        path = top_level[name]
        file_stat = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
            raise MembershipAuditError(f"invalid top-level output file: {name}")
    for name in ("checkpoints", "artifacts"):
        path = top_level[name]
        if path.is_symlink() or not path.is_dir():
            raise MembershipAuditError(f"invalid top-level output directory: {name}")
    return manifest


def commit_final_outputs(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Atomically create, or strictly revalidate, the final report set."""

    result_fingerprint = validate_sha256(
        report.get("provenance", {}).get("result_fingerprint_sha256"),
        field="report result_fingerprint_sha256",
    )
    existing = validate_final_outputs(
        output_dir, expected_result_fingerprint=result_fingerprint
    )
    if existing is not None:
        return existing
    json_path = output_dir / FINAL_REPORT_JSON
    markdown_path = output_dir / FINAL_REPORT_MARKDOWN
    manifest_path = output_dir / FINAL_OUTPUT_MANIFEST
    expectation = report.get("output_evidence_expectation")
    if not isinstance(expectation, dict):
        raise MembershipAuditError("output evidence expectation is missing")
    checkpoint_tree = output_tree_commitment(output_dir, "checkpoints")
    data_artifact_tree = output_tree_commitment(output_dir, "artifacts")
    if (
        checkpoint_tree["files"] != int(expectation.get("checkpoint_files", -1))
        or checkpoint_tree["path_set_sha256"]
        != expectation.get("checkpoint_path_set_sha256")
        or data_artifact_tree["files"]
        != int(expectation.get("data_artifact_files", -1))
        or data_artifact_tree["path_set_sha256"]
        != expectation.get("data_artifact_path_set_sha256")
    ):
        raise MembershipAuditError("output evidence set is incomplete")
    atomic_write_json(json_path, report)
    atomic_write_text(markdown_path, render_markdown(report))
    manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA_VERSION,
        "result_fingerprint_sha256": result_fingerprint,
        "artifacts": {
            "report_json": _final_artifact_record(json_path),
            "report_markdown": _final_artifact_record(markdown_path),
        },
        "audit_plan": _final_artifact_record(output_dir / "audit_plan.json"),
        "checkpoint_tree": checkpoint_tree,
        "data_artifact_tree": data_artifact_tree,
        "output_evidence_expectation": expectation,
    }
    atomic_write_json(manifest_path, manifest)
    validated = validate_final_outputs(
        output_dir, expected_result_fingerprint=result_fingerprint
    )
    if validated is None:  # pragma: no cover - guarded by the writes above
        raise MembershipAuditError("final output commit disappeared")
    return validated


def audit(
    *,
    root: Path,
    output_dir: Path,
    config_path: Path,
    manifest_path: Path,
    master_ids_path: Path,
    workers: int | None = None,
    batch_size: int | None = None,
    max_files_per_table: int | None = None,
    selected_tables: set[str] | None = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    started_clock = time.monotonic()
    config = load_config(config_path)
    evidence = validate_evidence(config)
    del evidence
    manifest, manifest_sha256, start_snapshot = validate_parent(
        root=root, manifest_path=manifest_path, config=config
    )
    effective_workers = int(workers or config["execution"]["workers"])
    effective_batch_size = int(batch_size or config["execution"]["batch_size"])
    if effective_workers < 1 or effective_batch_size < 1:
        raise MembershipAuditError("workers and batch size must be positive")
    tasks = build_tasks(
        manifest,
        config,
        max_files_per_table=max_files_per_table,
        selected_tables=selected_tables,
    )
    tool_path = Path(__file__).resolve()
    plan = plan_payload(
        config=config,
        config_path=config_path,
        tool_path=tool_path,
        manifest_sha256=manifest_sha256,
        master_path=master_ids_path,
        tasks=tasks,
        max_files_per_table=max_files_per_table,
        selected_tables=selected_tables,
        batch_size=effective_batch_size,
        workers=effective_workers,
    )
    prepare_output_dir(output_dir, plan)
    master = validate_master_ids(master_ids_path, config)
    try:
        validate_dense_resource_preflight(
            max_work_id=int(master.array[-1]),
            maximum_author_id=int(
                config["execution"]["resource_safety"][
                    "maximum_author_numeric_id_inclusive"
                ]
            ),
            resource_safety=config["execution"]["resource_safety"],
        )
        membership = build_membership_index(master.array, int(master.array[-1]))
        metrics = execute_tasks(
            root=root,
            output_dir=output_dir,
            tasks=tasks,
            config=config,
            plan=plan,
            membership=membership,
            batch_size=effective_batch_size,
            workers=effective_workers,
            progress=progress,
        )
        _, end_manifest_sha256, end_snapshot = validate_parent(
            root=root, manifest_path=manifest_path, config=config
        )
        parent_stable = (
            end_manifest_sha256 == manifest_sha256 and end_snapshot == start_snapshot
        )
        if not parent_stable:
            raise MembershipAuditError("source inventory changed during audit")
        report = aggregate_report(
            root=root,
            config=config,
            config_path=config_path,
            tool_path=tool_path,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            plan=plan,
            tasks=tasks,
            metrics=metrics,
            output_dir=output_dir,
            membership=membership,
            started_at=started_at,
            elapsed_seconds=time.monotonic() - started_clock,
            parent_stable=parent_stable,
        )
        _, final_manifest_sha256, final_snapshot = validate_parent(
            root=root, manifest_path=manifest_path, config=config
        )
        if final_manifest_sha256 != manifest_sha256 or final_snapshot != start_snapshot:
            raise MembershipAuditError("source inventory changed during aggregation")
        validate_master_ids_stable(master)
        if sha256_file(config_path) != plan["config_sha256"]:
            raise MembershipAuditError("config changed during audit")
        if sha256_file(tool_path) != plan["tool_sha256"]:
            raise MembershipAuditError("tool changed during audit")
        return report
    finally:
        master.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inventory-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--master-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-files-per-table", type=int)
    parser.add_argument("--table", action="append", dest="tables")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lock_handle: Any | None = None
    try:
        config_path = args.config.expanduser().resolve()
        manifest_path = args.inventory_manifest.expanduser().resolve()
        config = load_config(config_path)
        root = resolve_root(config, args.root)
        output_dir = validate_output_location(root, args.output_dir)
        if output_dir.exists() and output_dir.is_symlink():
            raise MembershipAuditError("output directory must not be a symlink")
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = output_dir / "audit.lock"
        lock_flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        lock_fd = os.open(lock_path, lock_flags, 0o600)
        lock_stat = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_stat.st_mode):
            os.close(lock_fd)
            raise MembershipAuditError("output lock must be a regular file")
        lock_handle = os.fdopen(lock_fd, "r+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MembershipAuditError("another audit holds the output lock") from exc

        last_print = 0.0

        def progress(completed: int, total: int, metric: dict[str, Any]) -> None:
            nonlocal last_print
            now = time.monotonic()
            if completed == total or completed % 100 == 0 or now - last_print >= 30:
                print(
                    f"projected scans: {completed:,}/{total:,}; "
                    f"latest={metric['relative_path']}",
                    file=sys.stderr,
                    flush=True,
                )
                last_print = now

        report = audit(
            root=root,
            output_dir=output_dir,
            config_path=config_path,
            manifest_path=manifest_path,
            master_ids_path=Path(
                os.path.abspath(os.fspath(args.master_ids.expanduser()))
            ),
            workers=args.workers,
            batch_size=args.batch_size,
            max_files_per_table=args.max_files_per_table,
            selected_tables=set(args.tables) if args.tables else None,
            progress=progress,
        )
        output_manifest = commit_final_outputs(output_dir, report)
        output_manifest_sha256 = sha256_file(output_dir / FINAL_OUTPUT_MANIFEST)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "g_minus_1_data_readiness": report["gate_status"][
                        "g_minus_1_data_readiness"
                    ],
                    "g1_empirical_signature": report["gate_status"][
                        "g1_empirical_signature"
                    ],
                    "result_fingerprint_sha256": report["provenance"][
                        "result_fingerprint_sha256"
                    ],
                    "output_dir": output_dir.name,
                    "output_manifest_sha256": output_manifest_sha256,
                    "output_artifact_count": len(output_manifest["artifacts"]),
                },
                sort_keys=True,
            )
        )
        if args.require_pass and report["status"].startswith("PASS_") is False:
            return 3
        return 0
    except (
        MembershipAuditError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
