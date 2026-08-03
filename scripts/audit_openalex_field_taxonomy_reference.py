#!/usr/bin/env python3
"""Audit current-snapshot OpenAlex work-to-taxonomy references.

This is a resumable CPU/I/O component audit over projected columns from the
main works table and the exploded work-topics table.  A full pass establishes
only current-snapshot taxonomy-reference integrity.  A bounded run is always
``PILOT_REFERENCE_ONLY`` and cannot establish the full component status.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import summarize_openalex_cross_field_feasibility as cross_field  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_field_taxonomy_reference_v0.yaml"
)
CONFIG_SCHEMA_VERSION = "openalex_g1_field_taxonomy_reference_config.v1"
PLAN_SCHEMA_VERSION = "openalex_g1_field_taxonomy_reference_plan.v1"
CHECKPOINT_SCHEMA_VERSION = "openalex_g1_field_taxonomy_reference_checkpoint.v1"
REPORT_SCHEMA_VERSION = "openalex_g1_field_taxonomy_reference_report.v1"
OUTPUTS_SCHEMA_VERSION = "openalex_g1_field_taxonomy_reference_outputs.v1"
FULL_SUCCESS_STATUS = "PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT"
PILOT_STATUS = "PILOT_REFERENCE_ONLY"
BLOCKED_STATUS = "BLOCKED_TAXONOMY_REFERENCE_AUDIT"
WORK_PREFIX = "https://openalex.org/W"
W_MINUS_1 = f"{WORK_PREFIX}-1"
W_ZERO = f"{WORK_PREFIX}0"
WORK_ID_PATTERN = re.compile(r"^https://openalex[.]org/W(0|[1-9][0-9]*)$")
ENTITY_ORDER = ("topic", "domain", "field", "subfield")
EXPECTED_PENDING = (
    "duplicate_work_topic_edge_audit",
    "primary_topic_present_in_work_topic_rows",
    "publication_date_and_year_validity",
    "cutoff_aware_available_at_validation",
    "author_entity_and_split_merge_validity",
    "candidate_researcher_eligibility",
    "deterministic_representative_pilot",
    "g_minus_1_data_readiness",
    "g0_measurement_freeze",
    "empirical_g1_signature",
)
PENDING_GATE_KEYS = {
    "duplicate_work_topic_edge_audit": "duplicate_work_topic_edges",
    "primary_topic_present_in_work_topic_rows": "primary_topic_child_inclusion",
    "publication_date_and_year_validity": "publication_date_validity",
    "cutoff_aware_available_at_validation": "temporal_available_at",
    "author_entity_and_split_merge_validity": "author_entity_validity",
    "candidate_researcher_eligibility": "candidate_eligibility",
    "deterministic_representative_pilot": None,
    "g_minus_1_data_readiness": "g_minus_1_data_readiness",
    "g0_measurement_freeze": "g0_measurement_freeze",
    "empirical_g1_signature": "empirical_g1_signature",
}
EXPECTED_CLAIM_BOUNDARY = (
    "A full component pass establishes current-snapshot, projected-column row "
    "conservation and validates non-null primary-topic and work-topic hierarchy "
    "references against the seven-file canonical taxonomy graph, with the frozen "
    "W-1 main-row quarantine and retained W0 warning unchanged. It does not prove "
    "historical inventory-era byte identity, deduplicate work-topic edges, prove "
    "that a primary topic occurs among each work's child topic rows, validate "
    "publication dates or cutoff-aware available_at, validate author entities or "
    "candidate eligibility, establish G-1 or G0, run empirical G1, or establish a "
    "scientific effect. A max-files run is PILOT_REFERENCE_ONLY and cannot be "
    "promoted to the full component status."
)


class TaxonomyReferenceAuditError(RuntimeError):
    """Raised when the frozen audit contract cannot be honored."""


@dataclass(frozen=True)
class FileTask:
    table_key: str
    table_name: str
    hierarchy_prefix: str
    relative_path: str
    bytes: int
    mtime_ns: int
    footer_rows: int
    projected_columns: tuple[str, ...]
    projected_schema: dict[str, dict[str, Any]]
    projected_schema_sha256: str


@dataclass(frozen=True)
class FrozenTaxonomy:
    public: dict[str, Any]
    ids: dict[str, pa.Array]
    names: dict[str, np.ndarray]
    rows: dict[str, list[dict[str, Any]]]
    id_to_index: dict[str, dict[str, int]]
    topic_parent_ids: dict[str, np.ndarray]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TaxonomyReferenceAuditError(message)


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaxonomyReferenceAuditError(f"cannot read {field}: {exc}") from exc
    return require_mapping(value, field)


def read_config(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "config is not a regular file")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TaxonomyReferenceAuditError(f"cannot read config: {exc}") from exc
    config = require_mapping(value, "config")
    require(config.get("schema_version") == CONFIG_SCHEMA_VERSION, "bad config schema")
    success = require_mapping(config.get("success_contract"), "success_contract")
    require(success.get("full_success_status") == FULL_SUCCESS_STATUS, "bad full status")
    require(success.get("pilot_status") == PILOT_STATUS, "bad pilot status")
    require(success.get("blocked_status") == BLOCKED_STATUS, "bad blocked status")
    gates = require_mapping(success.get("required_gate_status"), "required_gate_status")
    frozen_gates = {
        "current_snapshot_taxonomy_reference_component": "PASS",
        "raw_source_id_validity": "BLOCKED",
        "w_zero_entity_validity": "UNRESOLVED",
        "parent_content_identity": "UNVERIFIED",
        "duplicate_work_topic_edges": "PENDING",
        "primary_topic_child_inclusion": "PENDING",
        "publication_date_validity": "PENDING",
        "temporal_available_at": "PENDING",
        "author_entity_validity": "PENDING",
        "candidate_eligibility": "PENDING",
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
    }
    require(gates == frozen_gates, "required gate status contract drift")
    rules = require_mapping(config.get("reference_rules"), "reference_rules")
    frozen_rules = {
        "id_grammar": {
            "topic": r"^https://openalex[.]org/T([1-9][0-9]*)$",
            "domain": r"^https://openalex[.]org/domains/([1-9][0-9]*)$",
            "field": r"^https://openalex[.]org/fields/([1-9][0-9]*)$",
            "subfield": r"^https://openalex[.]org/subfields/([1-9][0-9]*)$",
        },
        "zero_sentinels": {
            "topic": "https://openalex.org/T0",
            "domain": "https://openalex.org/domains/0",
            "field": "https://openalex.org/fields/0",
            "subfield": "https://openalex.org/subfields/0",
        },
        "score_range": {"minimum_inclusive": 0.0, "maximum_inclusive": 1.0},
        "allow_all_null_hierarchy": True,
        "allow_partial_hierarchy": False,
        "require_zero_grammar_or_unknown_id_rows": True,
        "require_zero_parent_chain_or_name_mismatch_rows": True,
        "require_zero_nonfinite_or_out_of_range_score_rows": True,
    }
    require(rules == frozen_rules, "reference rules contract drift")
    require(
        tuple(success.get("pending_after_success", ())) == EXPECTED_PENDING,
        "pending contract drift",
    )
    require(config.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY, "claim boundary drift")
    execution = require_mapping(config.get("execution"), "execution")
    require(
        execution.get("resume_verification")
        == "rehash_current_projected_payload_before_reusing_metrics",
        "resume verification contract drift",
    )
    return config


def resolve_operator_path(
    override: Path | None,
    contract: Mapping[str, Any],
    *,
    field: str,
) -> Path:
    if override is not None:
        path = override
    else:
        env_name = str(contract.get("path_env") or contract.get("root_env") or "")
        require(env_name != "", f"{field} environment alias is missing")
        raw = os.environ.get(env_name)
        require(raw is not None and raw != "", f"{env_name} is not set")
        path = Path(raw)
    require(path.is_absolute(), f"{field} must resolve to an absolute path")
    return path


def _regular_identity(path: Path, *, field: str) -> dict[str, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TaxonomyReferenceAuditError(f"cannot stat {field}: {exc}") from exc
    require(not stat.S_ISLNK(info.st_mode), f"{field} may not be a symlink")
    require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def secure_join_regular(root: Path, relative: Path, *, field: str) -> Path:
    require(root.is_dir() and not root.is_symlink(), f"{field} root is not a directory")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise TaxonomyReferenceAuditError(f"cannot stat {field}: {exc}") from exc
        require(not stat.S_ISLNK(info.st_mode), f"{field} path component is a symlink")
    _regular_identity(current, field=field)
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise TaxonomyReferenceAuditError(f"{field} escapes its declared root") from exc
    return current


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    require(not partial.exists() and not partial.is_symlink(), "temporary file exists")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    require(not partial.exists() and not partial.is_symlink(), "temporary file exists")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _read_taxonomy_rows(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    plural_to_singular = {
        "domains": "domain",
        "fields": "field",
        "subfields": "subfield",
        "topics": "topic",
    }
    rows: dict[str, list[dict[str, Any]]] = {key: [] for key in ENTITY_ORDER}
    for index, raw_spec in enumerate(require_list(contract.get("files"), "taxonomy.files")):
        spec = require_mapping(raw_spec, f"taxonomy.files[{index}]")
        plural = str(spec.get("entity", ""))
        require(plural in plural_to_singular, "unsupported taxonomy entity")
        path = secure_join_regular(
            root,
            safe_relative_path(spec.get("relative_path"), "taxonomy file path"),
            field="taxonomy file",
        )
        require(
            sha256_file(path) == require_sha256(spec.get("sha256"), "taxonomy file sha256"),
            "taxonomy file SHA-256 mismatch",
        )
        observed = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                require(line.strip() != "", "blank taxonomy row")
                value = json.loads(line)
                rows[plural_to_singular[plural]].append(
                    require_mapping(value, "taxonomy row")
                )
                observed += 1
        require(observed == require_int(spec.get("expected_rows"), "taxonomy rows"), "taxonomy row count mismatch")
    return rows


def load_taxonomy(root: Path, contract: Mapping[str, Any]) -> FrozenTaxonomy:
    require(root.is_dir() and not root.is_symlink(), "taxonomy root is not a directory")
    try:
        public = cross_field.load_and_validate_taxonomy(root, contract)
    except cross_field.CrossFieldFeasibilityError as exc:
        raise TaxonomyReferenceAuditError(str(exc)) from exc
    rows = _read_taxonomy_rows(root, contract)
    ids: dict[str, pa.Array] = {}
    names: dict[str, np.ndarray] = {}
    id_to_index: dict[str, dict[str, int]] = {}
    for entity in ENTITY_ORDER:
        entity_ids = [str(row["id"]) for row in rows[entity]]
        ids[entity] = pa.array(entity_ids, type=pa.string())
        names[entity] = np.asarray(
            [str(row["display_name"]) for row in rows[entity]], dtype=object
        )
        id_to_index[entity] = {value: index for index, value in enumerate(entity_ids)}
        require(len(id_to_index[entity]) == len(entity_ids), "duplicate taxonomy ID")

    topic_parent_ids: dict[str, np.ndarray] = {}
    for parent in ("domain", "field", "subfield"):
        topic_parent_ids[parent] = np.asarray(
            [str(row[parent]["id"]) for row in rows["topic"]], dtype=object
        )
    return FrozenTaxonomy(
        public=public,
        ids=ids,
        names=names,
        rows=rows,
        id_to_index=id_to_index,
        topic_parent_ids=topic_parent_ids,
    )


def _repo_commitment(
    contract: Mapping[str, Any], *, path_field: str = "relative_path"
) -> tuple[Path, dict[str, Any], str]:
    relative = safe_relative_path(contract.get(path_field), path_field)
    path = REPO_ROOT / relative
    observed = sha256_file(path)
    require(observed == require_sha256(contract.get("sha256"), f"{path_field}.sha256"), f"{path_field} SHA-256 mismatch")
    return path, read_json(path, path_field), observed


def _expected_cross_field_matrix(
    profile: Mapping[str, Any], statuses: Iterable[str]
) -> dict[int | None, dict[str, int]]:
    status_set = set(statuses)
    matrix: dict[int | None, dict[str, int]] = {}
    for raw_domain in require_list(profile.get("domains"), "cross_field.domains"):
        domain = require_mapping(raw_domain, "cross_field.domain")
        for raw_field in require_list(domain.get("fields"), "cross_field.fields"):
            field = require_mapping(raw_field, "cross_field.field")
            field_id = require_int(field.get("field_id"), "cross_field.field_id")
            counts = require_mapping(field.get("authorship_status_counts"), "status counts")
            require(set(counts) == status_set, "cross-field author status set mismatch")
            require(field_id not in matrix, "duplicate cross-field bucket")
            matrix[field_id] = {key: require_int(counts[key], key) for key in status_set}
    compact_profile = require_mapping(profile.get("profile"), "cross_field.profile")
    null_profile = require_mapping(
        compact_profile.get("null_primary_field_profile"), "null primary field profile"
    )
    null_counts = require_mapping(null_profile.get("authorship_status_counts"), "null status counts")
    require(set(null_counts) == status_set, "null author status set mismatch")
    matrix[None] = {key: require_int(null_counts[key], key) for key in status_set}
    return matrix


def validate_parent_inputs(
    *,
    config_path: Path,
    inventory_manifest_path: Path,
    full_report_path: Path,
    taxonomy_root: Path,
) -> dict[str, Any]:
    config = read_config(config_path)
    inventory_contract = require_mapping(config.get("parent_inventory"), "parent_inventory")
    inventory_hash = sha256_file(inventory_manifest_path)
    require(
        inventory_hash == require_sha256(inventory_contract.get("manifest_sha256"), "inventory sha256"),
        "parent inventory manifest SHA-256 mismatch",
    )
    inventory = read_json(inventory_manifest_path, "parent inventory manifest")
    root = require_mapping(inventory.get("root"), "inventory.root")
    require(root.get("inventory_fingerprint_sha256") == inventory_contract.get("required_inventory_fingerprint_sha256"), "inventory fingerprint mismatch")
    require(root.get("stability_fingerprint_sha256") == inventory_contract.get("required_stability_fingerprint_sha256"), "inventory stability fingerprint mismatch")
    require(root.get("id") == config.get("source", {}).get("root_id"), "source root ID mismatch")

    parent = require_mapping(config.get("parent_work_source_audit"), "parent work source")
    compact_path, compact, compact_hash = _repo_commitment(
        require_mapping(parent.get("compact_summary"), "compact summary")
    )
    full_contract = require_mapping(parent.get("full_report"), "full report")
    require(full_report_path.name == full_contract.get("expected_basename"), "full report basename mismatch")
    full_hash = sha256_file(full_report_path)
    require(full_hash == require_sha256(full_contract.get("sha256"), "full report sha256"), "full report SHA-256 mismatch")
    full = read_json(full_report_path, "full work-source report")
    outputs_contract = require_mapping(parent.get("full_outputs_manifest"), "full outputs")
    outputs_path = full_report_path.parent / safe_relative_path(
        outputs_contract.get("relative_to_full_report"), "full outputs relative path"
    )
    outputs_hash = sha256_file(outputs_path)
    require(outputs_hash == require_sha256(outputs_contract.get("sha256"), "full outputs sha256"), "full outputs SHA-256 mismatch")
    required_status = str(parent.get("required_status", ""))
    fingerprint = require_sha256(parent.get("result_fingerprint_sha256"), "parent fingerprint")
    for label, report in (("compact", compact), ("full", full)):
        require(report.get("status") == required_status, f"{label} parent status mismatch")
        require(report.get("provenance", {}).get("result_fingerprint_sha256") == fingerprint, f"{label} parent fingerprint mismatch")
    required_gates = require_mapping(parent.get("required_gate_states"), "parent gates")
    full_gates = require_mapping(full.get("gate_status"), "full gate status")
    for key, expected in required_gates.items():
        require(full_gates.get(key) == expected, f"parent gate mismatch: {key}")

    profile_contract = require_mapping(config.get("compact_cross_field_profile"), "cross-field profile")
    profile_path, profile, profile_hash = _repo_commitment(profile_contract)
    require(profile.get("status") == profile_contract.get("required_status"), "cross-field status mismatch")
    require(profile.get("provenance", {}).get("result_fingerprint_sha256") == profile_contract.get("result_fingerprint_sha256"), "cross-field fingerprint mismatch")
    profile_provenance = require_mapping(profile.get("provenance"), "cross-field provenance")
    require(profile_provenance.get("compact_summary_sha256") == compact_hash, "cross-field compact-parent binding mismatch")
    require(profile_provenance.get("full_report_sha256") == full_hash, "cross-field full-parent binding mismatch")
    taxonomy_contract = require_mapping(config.get("taxonomy"), "taxonomy")
    require(profile_provenance.get("taxonomy_canonical_sha256") == taxonomy_contract.get("canonical_sha256"), "cross-field taxonomy binding mismatch")
    taxonomy = load_taxonomy(taxonomy_root, taxonomy_contract)

    statuses = [str(value) for value in require_list(profile_contract.get("required_author_id_statuses"), "author statuses")]
    compact_matrix = _expected_cross_field_matrix(profile, statuses)
    full_profile = require_mapping(full.get("author_work_profile"), "author_work_profile")
    full_matrix: dict[int | None, dict[str, int]] = {}
    for raw in require_list(full_profile.get("by_primary_field"), "by_primary_field"):
        row = require_mapping(raw, "by_primary_field row")
        field_id = row.get("field_id")
        require(field_id is None or type(field_id) is int, "bad parent field ID")
        counts = require_mapping(row.get("statuses"), "parent statuses")
        require(set(counts) == set(statuses), "parent author status set mismatch")
        require(field_id not in full_matrix, "duplicate parent field bucket")
        full_matrix[field_id] = {key: require_int(counts[key], key) for key in statuses}
    require(full_matrix == compact_matrix, "parent field/status margins differ from compact profile")
    require(len(compact_matrix) == require_int(profile_contract.get("required_primary_field_buckets_including_null"), "field buckets"), "cross-field bucket count mismatch")
    require(sum(sum(row.values()) for row in compact_matrix.values()) == require_int(profile_contract.get("required_master_work_rows"), "master works"), "cross-field master work count mismatch")

    return {
        "config": config,
        "inventory": inventory,
        "compact": compact,
        "full": full,
        "cross_field_profile": profile,
        "cross_field_matrix": compact_matrix,
        "taxonomy": taxonomy,
        "hashes": {
            "audit_tool_sha256": sha256_file(Path(__file__).resolve()),
            "config_sha256": sha256_file(config_path),
            "inventory_manifest_sha256": inventory_hash,
            "work_source_compact_sha256": compact_hash,
            "work_source_full_report_sha256": full_hash,
            "work_source_full_outputs_sha256": outputs_hash,
            "work_source_result_fingerprint_sha256": fingerprint,
            "cross_field_profile_sha256": profile_hash,
            "cross_field_result_fingerprint_sha256": str(profile_contract.get("result_fingerprint_sha256")),
            "taxonomy_canonical_sha256": str(taxonomy_contract.get("canonical_sha256")),
            "taxonomy_validator_tool_sha256": sha256_file(
                Path(cross_field.__file__).resolve()
            ),
        },
        "paths": {
            "compact": compact_path,
            "profile": profile_path,
        },
    }


def _projected_columns(table: Mapping[str, Any], tables: Mapping[str, Any]) -> tuple[str, ...]:
    source_column = str(table.get("source_id_column"))
    prefix = str(table.get("hierarchy_prefix"))
    suffixes = [str(value) for value in require_list(tables.get("projected_suffixes"), "projected suffixes")]
    return (source_column, *(f"{prefix}{suffix}" for suffix in suffixes))


def _projected_schema_from_record(
    record: Mapping[str, Any], columns: Iterable[str]
) -> dict[str, dict[str, Any]]:
    schema = require_mapping(record.get("schema_fields"), "manifest schema_fields")
    projected: dict[str, dict[str, Any]] = {}
    for column in columns:
        require(column in schema, f"projected column missing from manifest: {column}")
        projected[column] = dict(require_mapping(schema[column], f"schema {column}"))
    return projected


def build_plan(
    context: Mapping[str, Any],
    *,
    max_files_per_table: int | None,
    batch_size: int,
) -> dict[str, Any]:
    require(max_files_per_table is None or max_files_per_table > 0, "max files must be positive")
    require(batch_size > 0, "batch size must be positive")
    config = require_mapping(context.get("config"), "context.config")
    table_contracts = require_mapping(config.get("tables"), "tables")
    manifest_tables = {
        str(value["name"]): value
        for value in require_list(context["inventory"].get("tables"), "manifest tables")
    }
    tasks: list[dict[str, Any]] = []
    table_plans: dict[str, dict[str, Any]] = {}
    for table_key in ("main", "topics"):
        contract = require_mapping(table_contracts.get(table_key), f"tables.{table_key}")
        name = str(contract.get("name", ""))
        require(name in manifest_tables, f"manifest table missing: {name}")
        manifest_table = require_mapping(manifest_tables[name], f"manifest table {name}")
        require(manifest_table.get("file_count") == contract.get("expected_files"), f"{name} file count mismatch")
        require(manifest_table.get("footer_rows") == contract.get("expected_footer_rows"), f"{name} row count mismatch")
        require(manifest_table.get("bytes") == contract.get("expected_bytes"), f"{name} byte count mismatch")
        files = sorted(
            [require_mapping(value, "manifest file") for value in require_list(manifest_table.get("files"), "manifest files")],
            key=lambda value: str(value.get("relative_path")),
        )
        selected = files if max_files_per_table is None else files[:max_files_per_table]
        columns = _projected_columns(contract, table_contracts)
        table_rows = 0
        table_bytes = 0
        for record in selected:
            projected_schema = _projected_schema_from_record(record, columns)
            projected_types = require_mapping(table_contracts.get("projected_types"), "projected types")
            expected_types = {
                columns[0]: str(projected_types.get("source_id")),
                **{
                    f"{contract['hierarchy_prefix']}{suffix}": str(projected_types.get(suffix))
                    for suffix in table_contracts["projected_suffixes"]
                },
            }
            for column, expected_type in expected_types.items():
                require(projected_schema[column].get("type") == expected_type, f"projected type mismatch: {column}")
            schema_hash = sha256_bytes(canonical_json_bytes(projected_schema))
            task = {
                "table_key": table_key,
                "table_name": name,
                "hierarchy_prefix": str(contract.get("hierarchy_prefix")),
                "relative_path": str(record.get("relative_path")),
                "bytes": require_int(record.get("bytes"), "file bytes"),
                "mtime_ns": require_int(record.get("mtime_ns"), "file mtime"),
                "footer_rows": require_int(record.get("footer_rows"), "file rows"),
                "projected_columns": list(columns),
                "projected_schema": projected_schema,
                "projected_schema_sha256": schema_hash,
            }
            tasks.append(task)
            table_rows += task["footer_rows"]
            table_bytes += task["bytes"]
        table_plans[table_key] = {
            "name": name,
            "parent_files": len(files),
            "selected_files": len(selected),
            "parent_footer_rows": manifest_table["footer_rows"],
            "selected_footer_rows": table_rows,
            "parent_bytes": manifest_table["bytes"],
            "selected_bytes": table_bytes,
        }
    plan_core = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "label": config.get("label"),
        "full_scope": max_files_per_table is None,
        "max_files_per_table": max_files_per_table,
        "batch_size": batch_size,
        "parent_hashes": context["hashes"],
        "taxonomy_file_commitments": context["taxonomy"].public["file_commitments"],
        "tables": table_plans,
        "tasks": tasks,
    }
    plan_core["task_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(plan_core))
    return plan_core


def _task_from_mapping(value: Mapping[str, Any]) -> FileTask:
    return FileTask(
        table_key=str(value["table_key"]),
        table_name=str(value["table_name"]),
        hierarchy_prefix=str(value["hierarchy_prefix"]),
        relative_path=str(value["relative_path"]),
        bytes=require_int(value["bytes"], "task bytes"),
        mtime_ns=require_int(value["mtime_ns"], "task mtime"),
        footer_rows=require_int(value["footer_rows"], "task rows"),
        projected_columns=tuple(str(item) for item in value["projected_columns"]),
        projected_schema={str(key): dict(item) for key, item in value["projected_schema"].items()},
        projected_schema_sha256=str(value["projected_schema_sha256"]),
    )


def _bool_array(value: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return pc.fill_null(value, False).to_numpy(zero_copy_only=False).astype(bool, copy=False)


def _equal_mask(value: pa.Array, scalar: str) -> np.ndarray:
    return _bool_array(pc.equal(value, pa.scalar(scalar, type=pa.string())))


def _grammar_mask(value: pa.Array, pattern: str) -> np.ndarray:
    return _bool_array(pc.match_substring_regex(value, pattern=pattern))


def _index_in(value: pa.Array, choices: pa.Array) -> np.ndarray:
    return pc.fill_null(pc.index_in(value, value_set=choices), -1).to_numpy(
        zero_copy_only=False
    ).astype(np.int64, copy=False)


def _metric_keys() -> tuple[str, ...]:
    common = [
        "raw_rows",
        "source_w_minus_1_rows",
        "source_w_zero_rows",
        "source_null_rows",
        "source_unexpected_malformed_rows",
        "quarantined_rows",
        "retained_rows",
        "hierarchy_all_null_rows",
        "hierarchy_partial_rows",
        "hierarchy_complete_rows",
        "valid_reference_rows",
        "complete_invalid_rows",
        "name_mismatch_rows",
        "parent_chain_mismatch_rows",
        "nonfinite_score_rows",
        "out_of_range_score_rows",
        "taxonomy_zero_sentinel_rows",
        "main_field_null_rows",
        "main_field_grammar_invalid_rows",
        "main_field_unknown_rows",
    ]
    for entity in ENTITY_ORDER:
        common.extend(
            [
                f"{entity}_id_grammar_invalid_rows",
                f"{entity}_id_unknown_rows",
                f"{entity}_zero_sentinel_rows",
                f"{entity}_name_mismatch_rows",
            ]
        )
    return tuple(common)


def _actual_projected_schema(parquet: pq.ParquetFile, columns: Iterable[str]) -> dict[str, dict[str, Any]]:
    schema = parquet.schema_arrow
    result: dict[str, dict[str, Any]] = {}
    for column in columns:
        require(column in schema.names, f"actual Parquet missing projected column: {column}")
        field = schema.field(column)
        result[column] = {"nullable": bool(field.nullable), "type": str(field.type)}
    return result


def projected_payload_sha256(
    task: FileTask,
    *,
    source_root: Path,
    batch_size: int,
) -> str:
    """Re-read and hash the current projected logical payload for safe resume."""
    relative = safe_relative_path(task.relative_path, "task.relative_path")
    path = secure_join_regular(source_root, relative, field="source Parquet")
    start_identity = _regular_identity(path, field="source Parquet")
    require(
        start_identity == {"bytes": task.bytes, "mtime_ns": task.mtime_ns},
        "source file identity differs from manifest",
    )
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == task.footer_rows, "source footer row count differs")
    observed_schema = _actual_projected_schema(parquet, task.projected_columns)
    require(observed_schema == task.projected_schema, "projected schema differs from manifest")
    digest = hashlib.sha256()
    observed_rows = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=list(task.projected_columns),
        use_threads=False,
    ):
        observed_rows += batch.num_rows
        serialized = batch.serialize().to_pybytes()
        digest.update(len(serialized).to_bytes(8, "little"))
        digest.update(serialized)
    require(observed_rows == task.footer_rows, "projected payload row count differs")
    require(
        _regular_identity(path, field="source Parquet") == start_identity,
        "source file changed during projected-payload verification",
    )
    return digest.hexdigest()


def scan_file(
    task: FileTask,
    *,
    source_root: Path,
    taxonomy: FrozenTaxonomy,
    config: Mapping[str, Any],
    batch_size: int,
    task_fingerprint_sha256: str,
) -> dict[str, Any]:
    relative = safe_relative_path(task.relative_path, "task.relative_path")
    path = secure_join_regular(source_root, relative, field="source Parquet")
    start_identity = _regular_identity(path, field="source Parquet")
    require(start_identity == {"bytes": task.bytes, "mtime_ns": task.mtime_ns}, "source file identity differs from manifest")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == task.footer_rows, "source footer row count differs")
    observed_schema = _actual_projected_schema(parquet, task.projected_columns)
    require(observed_schema == task.projected_schema, "projected schema differs from manifest")
    require(sha256_bytes(canonical_json_bytes(observed_schema)) == task.projected_schema_sha256, "projected schema hash mismatch")

    tables = require_mapping(config.get("tables"), "tables")
    rules = require_mapping(config.get("reference_rules"), "reference rules")
    grammar = require_mapping(rules.get("id_grammar"), "id grammar")
    zero_sentinels = require_mapping(rules.get("zero_sentinels"), "zero sentinels")
    score_range = require_mapping(rules.get("score_range"), "score range")
    score_min = float(score_range.get("minimum_inclusive"))
    score_max = float(score_range.get("maximum_inclusive"))
    metrics = {key: 0 for key in _metric_keys()}
    main_field_counts: dict[str, int] = defaultdict(int)
    projected_digest = hashlib.sha256()
    scanned_rows = 0
    prefix = task.hierarchy_prefix
    source_column = task.projected_columns[0]
    hierarchy_columns = [f"{prefix}{suffix}" for suffix in tables["projected_suffixes"]]

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=list(task.projected_columns),
        use_threads=False,
    ):
        batch_rows = batch.num_rows
        scanned_rows += batch_rows
        serialized = batch.serialize().to_pybytes()
        projected_digest.update(len(serialized).to_bytes(8, "little"))
        projected_digest.update(serialized)
        columns = {name: batch.column(index) for index, name in enumerate(batch.schema.names)}
        source = columns[source_column]
        source_null = pc.is_null(source).to_numpy(zero_copy_only=False)
        source_w_minus_1 = _equal_mask(source, W_MINUS_1)
        source_w_zero = _equal_mask(source, W_ZERO)
        source_grammar = _grammar_mask(source, WORK_ID_PATTERN.pattern)
        source_unexpected = ~(source_null | source_w_minus_1 | source_grammar)
        quarantine = source_w_minus_1 if task.table_key == "main" else np.zeros(batch_rows, dtype=bool)
        retained = ~quarantine

        metrics["raw_rows"] += batch_rows
        metrics["source_w_minus_1_rows"] += int(np.count_nonzero(source_w_minus_1))
        metrics["source_w_zero_rows"] += int(np.count_nonzero(source_w_zero))
        metrics["source_null_rows"] += int(np.count_nonzero(source_null))
        metrics["source_unexpected_malformed_rows"] += int(np.count_nonzero(source_unexpected))
        metrics["quarantined_rows"] += int(np.count_nonzero(quarantine))
        metrics["retained_rows"] += int(np.count_nonzero(retained))

        null_masks = [pc.is_null(columns[name]).to_numpy(zero_copy_only=False) for name in hierarchy_columns]
        all_null = np.logical_and.reduce(null_masks) & retained
        any_null = np.logical_or.reduce(null_masks)
        partial = any_null & ~all_null & retained
        complete = ~any_null & retained
        metrics["hierarchy_all_null_rows"] += int(np.count_nonzero(all_null))
        metrics["hierarchy_partial_rows"] += int(np.count_nonzero(partial))
        metrics["hierarchy_complete_rows"] += int(np.count_nonzero(complete))

        entity_indexes: dict[str, np.ndarray] = {}
        entity_known: dict[str, np.ndarray] = {}
        entity_id_nonnull: dict[str, np.ndarray] = {}
        entity_observed_ids: dict[str, np.ndarray] = {}
        invalid_any = np.zeros(batch_rows, dtype=bool)
        zero_any = np.zeros(batch_rows, dtype=bool)
        name_any = np.zeros(batch_rows, dtype=bool)
        for entity in ENTITY_ORDER:
            suffix = "id" if entity == "topic" else f"{entity}__id"
            name_suffix = "display_name" if entity == "topic" else f"{entity}__display_name"
            id_column = columns[f"{prefix}{suffix}"]
            name_column = columns[f"{prefix}{name_suffix}"]
            index = _index_in(id_column, taxonomy.ids[entity])
            grammar_valid = _grammar_mask(id_column, str(grammar[entity]))
            id_nonnull = ~pc.is_null(id_column).to_numpy(zero_copy_only=False) & retained
            name_nonnull = (
                ~pc.is_null(name_column).to_numpy(zero_copy_only=False) & retained
            )
            zero = _equal_mask(id_column, str(zero_sentinels[entity])) & retained
            grammar_invalid = id_nonnull & ~grammar_valid
            unknown = id_nonnull & grammar_valid & (index < 0)
            known = id_nonnull & (index >= 0)
            safe_index = np.maximum(index, 0)
            observed_names = name_column.to_numpy(zero_copy_only=False)
            name_mismatch = (
                known
                & name_nonnull
                & (observed_names != taxonomy.names[entity][safe_index])
            )
            metrics[f"{entity}_id_grammar_invalid_rows"] += int(np.count_nonzero(grammar_invalid))
            metrics[f"{entity}_id_unknown_rows"] += int(np.count_nonzero(unknown))
            metrics[f"{entity}_zero_sentinel_rows"] += int(np.count_nonzero(zero))
            metrics[f"{entity}_name_mismatch_rows"] += int(np.count_nonzero(name_mismatch))
            invalid_any |= grammar_invalid | unknown
            zero_any |= zero
            name_any |= name_mismatch
            entity_indexes[entity] = index
            entity_known[entity] = known
            entity_id_nonnull[entity] = id_nonnull
            entity_observed_ids[entity] = id_column.to_numpy(zero_copy_only=False)

        topic_index = entity_indexes["topic"]
        safe_topic_index = np.maximum(topic_index, 0)
        chain_evaluable = entity_known["topic"].copy()
        for parent in ("domain", "field", "subfield"):
            chain_evaluable &= entity_id_nonnull[parent]
        chain_mismatch = chain_evaluable
        chain_mismatch &= (
            (entity_observed_ids["domain"] != taxonomy.topic_parent_ids["domain"][safe_topic_index])
            | (entity_observed_ids["field"] != taxonomy.topic_parent_ids["field"][safe_topic_index])
            | (entity_observed_ids["subfield"] != taxonomy.topic_parent_ids["subfield"][safe_topic_index])
        )
        score_column = columns[f"{prefix}score"]
        score = score_column.to_numpy(zero_copy_only=False)
        score_nonnull = (
            ~pc.is_null(score_column).to_numpy(zero_copy_only=False) & retained
        )
        nonfinite_score = score_nonnull & ~np.isfinite(score)
        out_of_range_score = (
            score_nonnull
            & np.isfinite(score)
            & ((score < score_min) | (score > score_max))
        )
        invalid_any |= name_any | chain_mismatch | nonfinite_score | out_of_range_score
        valid = complete & ~invalid_any
        metrics["name_mismatch_rows"] += int(np.count_nonzero(name_any))
        metrics["parent_chain_mismatch_rows"] += int(np.count_nonzero(chain_mismatch))
        metrics["nonfinite_score_rows"] += int(np.count_nonzero(nonfinite_score))
        metrics["out_of_range_score_rows"] += int(np.count_nonzero(out_of_range_score))
        metrics["taxonomy_zero_sentinel_rows"] += int(np.count_nonzero(zero_any))
        metrics["complete_invalid_rows"] += int(np.count_nonzero(complete & invalid_any))
        metrics["valid_reference_rows"] += int(np.count_nonzero(valid))

        if task.table_key == "main":
            field_column = columns[f"{prefix}field__id"]
            field_null = pc.is_null(field_column).to_numpy(zero_copy_only=False) & retained
            field_index = entity_indexes["field"]
            field_grammar = _grammar_mask(field_column, str(grammar["field"]))
            field_unknown = retained & field_grammar & (field_index < 0)
            field_invalid = retained & ~field_null & ~field_grammar
            metrics["main_field_null_rows"] += int(np.count_nonzero(field_null))
            metrics["main_field_grammar_invalid_rows"] += int(np.count_nonzero(field_invalid))
            metrics["main_field_unknown_rows"] += int(np.count_nonzero(field_unknown))
            known_retained = retained & (field_index >= 0)
            if np.any(known_retained):
                bincount = np.bincount(
                    field_index[known_retained], minlength=len(taxonomy.rows["field"])
                )
                for index in np.flatnonzero(bincount):
                    field_id = str(taxonomy.rows["field"][int(index)]["id"])
                    numeric = field_id.rsplit("/", maxsplit=1)[-1]
                    main_field_counts[numeric] += int(bincount[index])

    require(scanned_rows == task.footer_rows, "data-page rows do not equal footer rows")
    require(
        metrics["raw_rows"] == metrics["quarantined_rows"] + metrics["retained_rows"],
        "raw/quarantine/retained conservation failed",
    )
    require(
        metrics["retained_rows"]
        == metrics["hierarchy_all_null_rows"]
        + metrics["hierarchy_partial_rows"]
        + metrics["hierarchy_complete_rows"],
        "hierarchy null partition conservation failed",
    )
    require(
        metrics["hierarchy_complete_rows"]
        == metrics["valid_reference_rows"] + metrics["complete_invalid_rows"],
        "complete reference partition conservation failed",
    )
    end_identity = _regular_identity(path, field="source Parquet")
    require(end_identity == start_identity, "source file changed during scan")
    payload = {
        "metrics": metrics,
        "main_field_counts": dict(sorted(main_field_counts.items(), key=lambda item: int(item[0]))),
        "projected_payload_sha256": projected_digest.hexdigest(),
    }
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task_fingerprint_sha256": task_fingerprint_sha256,
        "table_key": task.table_key,
        "table_name": task.table_name,
        "relative_path": task.relative_path,
        "source_identity": {
            "bytes": task.bytes,
            "mtime_ns": task.mtime_ns,
            "footer_rows": task.footer_rows,
            "projected_schema": task.projected_schema,
            "projected_schema_sha256": task.projected_schema_sha256,
        },
        "payload": payload,
    }
    checkpoint["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return checkpoint


def checkpoint_path(output_dir: Path, task: FileTask) -> Path:
    token = sha256_bytes(task.relative_path.encode("utf-8"))[:24]
    return output_dir / "checkpoints" / task.table_key / f"{token}.json"


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    task: FileTask,
    *,
    source_root: Path,
    task_fingerprint_sha256: str,
    batch_size: int,
) -> dict[str, Any]:
    require(checkpoint.get("schema_version") == CHECKPOINT_SCHEMA_VERSION, "checkpoint schema mismatch")
    require(checkpoint.get("task_fingerprint_sha256") == task_fingerprint_sha256, "checkpoint task fingerprint mismatch")
    require(checkpoint.get("relative_path") == task.relative_path, "checkpoint path mismatch")
    require(checkpoint.get("table_key") == task.table_key, "checkpoint table mismatch")
    expected_identity = {
        "bytes": task.bytes,
        "mtime_ns": task.mtime_ns,
        "footer_rows": task.footer_rows,
        "projected_schema": task.projected_schema,
        "projected_schema_sha256": task.projected_schema_sha256,
    }
    require(checkpoint.get("source_identity") == expected_identity, "checkpoint source identity mismatch")
    payload = require_mapping(checkpoint.get("payload"), "checkpoint payload")
    require(checkpoint.get("payload_sha256") == sha256_bytes(canonical_json_bytes(payload)), "checkpoint payload hash mismatch")
    source_path = secure_join_regular(
        source_root,
        safe_relative_path(task.relative_path, "checkpoint path"),
        field="source Parquet",
    )
    current = _regular_identity(source_path, field="source Parquet")
    require(current == {"bytes": task.bytes, "mtime_ns": task.mtime_ns}, "source identity changed since checkpoint")
    require(
        projected_payload_sha256(
            task,
            source_root=source_root,
            batch_size=batch_size,
        )
        == payload.get("projected_payload_sha256"),
        "current projected payload differs from checkpoint",
    )
    return dict(checkpoint)


def _merge_metrics(checkpoints: Iterable[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    metrics = {key: 0 for key in _metric_keys()}
    fields: dict[str, int] = defaultdict(int)
    for checkpoint in checkpoints:
        payload = require_mapping(checkpoint.get("payload"), "checkpoint payload")
        observed = require_mapping(payload.get("metrics"), "checkpoint metrics")
        require(set(observed) == set(metrics), "checkpoint metric set mismatch")
        for key in metrics:
            metrics[key] += require_int(observed[key], key)
        for key, value in require_mapping(payload.get("main_field_counts"), "main field counts").items():
            require(re.fullmatch(r"[1-9][0-9]*", str(key)) is not None, "bad main field key")
            fields[str(key)] += require_int(value, "main field count")
    return metrics, dict(sorted(fields.items(), key=lambda item: int(item[0])))


def _table_result(
    table_key: str,
    checkpoints: list[dict[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    metrics, fields = _merge_metrics(checkpoints)
    table_plan = require_mapping(plan["tables"][table_key], f"plan table {table_key}")
    rows_expected = require_int(table_plan.get("selected_footer_rows"), "selected rows")
    return {
        "table_key": table_key,
        "table_name": table_plan["name"],
        "selected_files": table_plan["selected_files"],
        "parent_files": table_plan["parent_files"],
        "selected_bytes": table_plan["selected_bytes"],
        "parent_bytes": table_plan["parent_bytes"],
        "selected_footer_rows": rows_expected,
        "parent_footer_rows": table_plan["parent_footer_rows"],
        "metrics": metrics,
        "main_field_counts": fields,
        "row_conservation": (
            metrics["raw_rows"] == rows_expected
            and metrics["raw_rows"] == metrics["quarantined_rows"] + metrics["retained_rows"]
            and metrics["retained_rows"]
            == metrics["hierarchy_all_null_rows"]
            + metrics["hierarchy_partial_rows"]
            + metrics["hierarchy_complete_rows"]
            and metrics["hierarchy_complete_rows"]
            == metrics["valid_reference_rows"] + metrics["complete_invalid_rows"]
        ),
    }


def _cross_field_works(matrix: Mapping[int | None, Mapping[str, int]]) -> tuple[dict[str, int], int]:
    fields: dict[str, int] = {}
    null_rows = 0
    for field_id, statuses in matrix.items():
        works = sum(statuses.values())
        if field_id is None:
            null_rows = works
        else:
            fields[str(field_id)] = works
    return dict(sorted(fields.items(), key=lambda item: int(item[0]))), null_rows


def build_report(
    context: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoints: list[dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    by_table = {
        key: [value for value in checkpoints if value["table_key"] == key]
        for key in ("main", "topics")
    }
    table_results = {
        key: _table_result(key, by_table[key], plan) for key in ("main", "topics")
    }
    config = context["config"]
    table_contracts = require_mapping(config.get("tables"), "tables")
    rules = require_mapping(config.get("reference_rules"), "reference rules")
    full_scope = bool(plan.get("full_scope"))
    local_checks: dict[str, bool] = {}
    for key, result in table_results.items():
        metrics = result["metrics"]
        local_checks[f"{key}_row_conservation"] = bool(result["row_conservation"])
        local_checks[f"{key}_source_null_zero"] = metrics["source_null_rows"] == 0
        local_checks[f"{key}_source_malformed_zero"] = metrics["source_unexpected_malformed_rows"] == 0
        local_checks[f"{key}_partial_hierarchy_zero"] = metrics["hierarchy_partial_rows"] == 0
        local_checks[f"{key}_invalid_complete_reference_zero"] = metrics["complete_invalid_rows"] == 0
        local_checks[f"{key}_taxonomy_zero_sentinel_zero"] = metrics["taxonomy_zero_sentinel_rows"] == 0
    main_metrics = table_results["main"]["metrics"]
    topic_metrics = table_results["topics"]["metrics"]
    main_contract = require_mapping(table_contracts.get("main"), "main contract")
    topics_contract = require_mapping(table_contracts.get("topics"), "topics contract")
    if full_scope:
        local_checks["main_w_minus_1_exact"] = main_metrics["source_w_minus_1_rows"] == main_contract["expected_w_minus_1_source_rows"]
        local_checks["main_w_zero_exact"] = main_metrics["source_w_zero_rows"] == main_contract["expected_w_zero_source_rows"]
    else:
        local_checks["main_w_minus_1_not_above_frozen_total"] = main_metrics["source_w_minus_1_rows"] <= main_contract["expected_w_minus_1_source_rows"]
        local_checks["main_w_zero_not_above_frozen_total"] = main_metrics["source_w_zero_rows"] <= main_contract["expected_w_zero_source_rows"]
    local_checks["topics_w_minus_1_zero"] = topic_metrics["source_w_minus_1_rows"] == topics_contract["expected_w_minus_1_source_rows"]
    local_checks["topics_w_zero_zero"] = topic_metrics["source_w_zero_rows"] == topics_contract["expected_w_zero_source_rows"]

    expected_fields, expected_null = _cross_field_works(context["cross_field_matrix"])
    observed_fields = table_results["main"]["main_field_counts"]
    observed_null = main_metrics["main_field_null_rows"]
    field_comparison = {
        "status": "PASS" if full_scope and observed_fields == expected_fields and observed_null == expected_null else ("NOT_RUN_PILOT" if not full_scope else "FAIL"),
        "parent_field_status_margins_equal_compact_profile": True,
        "observed_main_field_works_equal_compact_profile": (observed_fields == expected_fields) if full_scope else None,
        "observed_main_null_field_works_equal_compact_profile": (observed_null == expected_null) if full_scope else None,
        "observed_field_counts": observed_fields if full_scope else None,
        "observed_null_field_rows": observed_null if full_scope else None,
        "expected_field_counts": expected_fields,
        "expected_null_field_rows": expected_null,
    }
    if full_scope:
        local_checks["main_field_margin_equals_compact_profile"] = field_comparison["status"] == "PASS"
    local_pass = all(local_checks.values())
    if not local_pass:
        status = BLOCKED_STATUS
    elif full_scope:
        status = FULL_SUCCESS_STATUS
    else:
        status = PILOT_STATUS

    success = require_mapping(config.get("success_contract"), "success contract")
    gate_status = dict(require_mapping(success.get("required_gate_status"), "required gates"))
    gate_status["current_snapshot_taxonomy_reference_component"] = (
        "PASS" if status == FULL_SUCCESS_STATUS else "NOT_ESTABLISHED_PILOT" if status == PILOT_STATUS else "BLOCKED"
    )
    selected_bytes = sum(value["selected_bytes"] for value in table_results.values())
    parent_bytes = sum(value["parent_bytes"] for value in table_results.values())
    projected_seconds = (
        elapsed_seconds * parent_bytes / selected_bytes if selected_bytes > 0 else None
    )
    checkpoint_commitments = sorted(
        [
            {
                "table_key": value["table_key"],
                "relative_path": value["relative_path"],
                "payload_sha256": value["payload_sha256"],
                "projected_payload_sha256": value["payload"]["projected_payload_sha256"],
            }
            for value in checkpoints
        ],
        key=lambda value: (value["table_key"], value["relative_path"]),
    )
    checkpoint_tree_sha = sha256_bytes(canonical_json_bytes(checkpoint_commitments))
    scope = {
        "full_scope": full_scope,
        "max_files_per_table": plan.get("max_files_per_table"),
        "selected_files": sum(value["selected_files"] for value in table_results.values()),
        "parent_files": sum(value["parent_files"] for value in table_results.values()),
        "selected_footer_rows": sum(value["selected_footer_rows"] for value in table_results.values()),
        "parent_footer_rows": sum(value["parent_footer_rows"] for value in table_results.values()),
        "selected_source_bytes": selected_bytes,
        "parent_source_bytes": parent_bytes,
        "scientific_effect_estimation_performed": False,
        "gpu_used": False,
    }
    accounting = {
        "tables": table_results,
        "local_checks": local_checks,
        "all_local_checks_pass": local_pass,
    }
    reference_summary = {
        "all_null_hierarchy_rows": sum(value["metrics"]["hierarchy_all_null_rows"] for value in table_results.values()),
        "partial_hierarchy_rows": sum(value["metrics"]["hierarchy_partial_rows"] for value in table_results.values()),
        "complete_hierarchy_rows": sum(value["metrics"]["hierarchy_complete_rows"] for value in table_results.values()),
        "valid_reference_rows": sum(value["metrics"]["valid_reference_rows"] for value in table_results.values()),
        "complete_invalid_rows": sum(value["metrics"]["complete_invalid_rows"] for value in table_results.values()),
        "all_null_hierarchy_allowed": rules.get("allow_all_null_hierarchy") is True,
    }
    parent_hashes = dict(context["hashes"])
    fingerprint_payload = {
        "status": status,
        "gate_status": gate_status,
        "scope": scope,
        "accounting": accounting,
        "reference_summary": reference_summary,
        "field_status_margin_comparison": field_comparison,
        "taxonomy": {
            "canonical_sha256": context["taxonomy"].public["canonical_sha256"],
            "entity_counts": context["taxonomy"].public["counts"],
            "internal_graph_checks": context["taxonomy"].public["internal_graph_checks"],
        },
        "checkpoint_tree_sha256": checkpoint_tree_sha,
        "pending": success.get("pending_after_success"),
        "claim_boundary": config.get("claim_boundary"),
        "parent_hashes": parent_hashes,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
    }
    result_fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_payload))
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": config.get("label"),
        "generated_at_utc": utc_now(),
        "status": status,
        "gate_status": gate_status,
        "scope": scope,
        "accounting": accounting,
        "reference_summary": reference_summary,
        "field_status_margin_comparison": field_comparison,
        "taxonomy": fingerprint_payload["taxonomy"],
        "execution": {
            "workload_class": "local_cpu_io",
            "elapsed_seconds": elapsed_seconds,
            "projected_full_runtime_seconds_linear_by_source_bytes": projected_seconds,
            "projection_is_operational_estimate_not_scientific_result": True,
            "resumed_checkpoint_count": None,
        },
        "checkpoint_evidence": {
            "checkpoint_count": len(checkpoints),
            "checkpoint_tree_sha256": checkpoint_tree_sha,
            "per_file_binds_projected_payload_hash": True,
            "resume_verification": "REHASH_CURRENT_PROJECTED_PAYLOAD",
            "resume_reuses_metrics_only_after_projected_payload_rehash": True,
        },
        "pending": success.get("pending_after_success"),
        "claim_boundary": config.get("claim_boundary"),
        "provenance": {
            **parent_hashes,
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "result_fingerprint_sha256": result_fingerprint,
            "tool_relative_path": "scripts/audit_openalex_field_taxonomy_reference.py",
            "tool_sha256": sha256_file(Path(__file__).resolve()),
            "path_redacted": True,
        },
    }
    serialized = json.dumps(report)
    require("/home/" not in serialized and "/hdd/" not in serialized and "/ssd/" not in serialized, "report leaked operator path")
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    scope = report["scope"]
    reference = report["reference_summary"]
    comparison = report["field_status_margin_comparison"]
    execution = report["execution"]
    checkpoint = report["checkpoint_evidence"]
    gates = report["gate_status"]

    def state(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    lines = [
        "# OpenAlex Current-Snapshot Taxonomy Reference Audit",
        "",
        "## Status and boundary",
        "",
        f"- Status: `{report['status']}`",
        f"- Full scope: `{str(scope['full_scope']).lower()}`",
        f"- Files: `{scope['selected_files']:,}` / `{scope['parent_files']:,}`",
        f"- Projected rows: `{scope['selected_footer_rows']:,}` / `{scope['parent_footer_rows']:,}`",
        f"- Workload: `{state(execution['workload_class'])}`; "
        f"GPU used: `{state(scope['gpu_used'])}`",
        "- Scientific effect estimation: "
        f"`{state(scope['scientific_effect_estimation_performed'])}`",
        f"- Resume verification: `{state(checkpoint['resume_verification'])}`; "
        "metrics reused only after projected-payload rehash: "
        f"`{state(checkpoint['resume_reuses_metrics_only_after_projected_payload_rehash'])}`",
        "",
        "A bounded run is pilot evidence only. Even a full component pass is not",
        "G-1, G0, empirical G1, or evidence of a scientific effect.",
        "",
        "## Reference accounting",
        "",
        "| Category | Exact rows |",
        "| --- | ---: |",
        f"| All-null hierarchy | {reference['all_null_hierarchy_rows']:,} |",
        f"| Partial hierarchy | {reference['partial_hierarchy_rows']:,} |",
        f"| Complete hierarchy | {reference['complete_hierarchy_rows']:,} |",
        f"| Valid complete reference | {reference['valid_reference_rows']:,} |",
        f"| Invalid complete reference | {reference['complete_invalid_rows']:,} |",
        "",
        "## Per-table accounting",
        "",
        "| Table | Files | Rows | W-1 | W0 | All null | Partial | Complete | Invalid complete | Conserved |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("main", "topics"):
        table = report["accounting"]["tables"][key]
        metrics = table["metrics"]
        lines.append(
            f"| {table['table_name']} | {table['selected_files']:,} | {metrics['raw_rows']:,} "
            f"| {metrics['source_w_minus_1_rows']:,} | {metrics['source_w_zero_rows']:,} "
            f"| {metrics['hierarchy_all_null_rows']:,} | {metrics['hierarchy_partial_rows']:,} "
            f"| {metrics['hierarchy_complete_rows']:,} | {metrics['complete_invalid_rows']:,} "
            f"| {str(table['row_conservation']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Cross-field margin check",
            "",
            f"- Status: `{comparison['status']}`",
            "- Parent work-source field/status margins equal the compact cross-field "
            f"profile: `{state(comparison['parent_field_status_margins_equal_compact_profile'])}`",
            "- Observed main field works equal the compact profile: "
            f"`{state(comparison['observed_main_field_works_equal_compact_profile'])}`",
            "- Observed main null-field works equal the compact profile: "
            f"`{state(comparison['observed_main_null_field_works_equal_compact_profile'])}`",
            "",
            "## Gate status",
            "",
            "| Gate | State |",
            "| --- | --- |",
        ]
    )
    for key in sorted(gates):
        lines.append(f"| {key} | {state(gates[key])} |")
    lines.extend(
        [
            "",
            "## Remaining boundaries",
            "",
            "| Pending item | Gate | Current state |",
            "| --- | --- | --- |",
        ]
    )
    for item in report["pending"]:
        require(item in PENDING_GATE_KEYS, f"unmapped pending item: {item}")
        gate_key = PENDING_GATE_KEYS[item]
        if gate_key is None:
            current_state = "PENDING"
            rendered_gate = "none"
        else:
            require(gate_key in gates, f"pending gate missing from report: {gate_key}")
            current_state = state(gates[gate_key])
            rendered_gate = gate_key
        lines.append(f"| {item} | {rendered_gate} | {current_state} |")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Task fingerprint: `{report['provenance']['task_fingerprint_sha256']}`",
            f"- Result fingerprint: `{report['provenance']['result_fingerprint_sha256']}`",
            f"- Checkpoint tree: `{report['checkpoint_evidence']['checkpoint_tree_sha256']}`",
            f"- Taxonomy canonical SHA-256: `{report['taxonomy']['canonical_sha256']}`",
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, report: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    report_path = output_dir / "taxonomy_reference_audit.json"
    markdown_path = output_dir / "taxonomy_reference_audit.md"
    manifest_path = output_dir / "taxonomy_reference_audit.outputs.json"
    atomic_write_json(report_path, report)
    atomic_write_text(markdown_path, render_markdown(report))
    artifacts = []
    for path in (output_dir / "audit_plan.json", report_path, markdown_path):
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
        "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
        "checkpoint_tree_sha256": report["checkpoint_evidence"]["checkpoint_tree_sha256"],
        "artifacts": artifacts,
    }
    atomic_write_json(manifest_path, outputs)
    return outputs


def run_audit(
    *,
    config_path: Path,
    inventory_manifest_path: Path,
    source_root: Path,
    taxonomy_root: Path,
    full_report_path: Path,
    output_dir: Path,
    max_files_per_table: int | None = None,
    resume: bool = False,
    batch_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(source_root.is_dir() and not source_root.is_symlink(), "source root is not a directory")
    if output_dir.exists():
        require(output_dir.is_dir() and not output_dir.is_symlink(), "output path is not a directory")
        require(resume or not any(output_dir.iterdir()), "nonempty output requires --resume")
    else:
        output_dir.mkdir(parents=True)
    context = validate_parent_inputs(
        config_path=config_path,
        inventory_manifest_path=inventory_manifest_path,
        full_report_path=full_report_path,
        taxonomy_root=taxonomy_root,
    )
    configured_batch = require_int(context["config"]["execution"]["batch_size"], "batch size")
    effective_batch = configured_batch if batch_size is None else batch_size
    plan = build_plan(
        context,
        max_files_per_table=max_files_per_table,
        batch_size=effective_batch,
    )
    plan_path = output_dir / "audit_plan.json"
    if resume and plan_path.exists():
        prior_plan = read_json(plan_path, "audit plan")
        require(prior_plan == plan, "resume plan differs from current frozen plan")
    else:
        require(not resume or not any(output_dir.iterdir()), "resume output lacks audit_plan.json")
        atomic_write_json(plan_path, plan)

    started = time.monotonic()
    checkpoints: list[dict[str, Any]] = []
    resumed = 0
    for raw_task in plan["tasks"]:
        task = _task_from_mapping(raw_task)
        path = checkpoint_path(output_dir, task)
        if resume and path.exists():
            checkpoint = validate_checkpoint(
                read_json(path, "checkpoint"),
                task,
                source_root=source_root,
                task_fingerprint_sha256=plan["task_fingerprint_sha256"],
                batch_size=effective_batch,
            )
            resumed += 1
        else:
            checkpoint = scan_file(
                task,
                source_root=source_root,
                taxonomy=context["taxonomy"],
                config=context["config"],
                batch_size=effective_batch,
                task_fingerprint_sha256=plan["task_fingerprint_sha256"],
            )
            atomic_write_json(path, checkpoint)
        checkpoints.append(checkpoint)
    elapsed = time.monotonic() - started
    report = build_report(context, plan, checkpoints, elapsed_seconds=elapsed)
    report["execution"]["resumed_checkpoint_count"] = resumed
    report["execution"]["newly_scanned_checkpoint_count"] = len(checkpoints) - resumed
    if resumed == len(checkpoints):
        report["execution"][
            "projected_full_runtime_seconds_linear_by_source_bytes"
        ] = None
        report["execution"]["runtime_projection_status"] = (
            "NOT_AVAILABLE_RESUME_ONLY"
        )
    else:
        report["execution"]["runtime_projection_status"] = "ESTIMATED_FROM_THIS_SCAN"
    outputs = write_outputs(output_dir, report, plan)
    return report, outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inventory-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--taxonomy-root", type=Path)
    parser.add_argument("--full-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files-per-table", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = read_config(args.config)
        inventory_contract = require_mapping(config.get("parent_inventory"), "parent inventory")
        manifest = args.inventory_manifest or (
            REPO_ROOT
            / safe_relative_path(
                inventory_contract.get("manifest_relative_path"),
                "inventory manifest relative path",
            )
        )
        source_root = resolve_operator_path(
            args.source_root,
            require_mapping(config.get("source"), "source"),
            field="source root",
        )
        taxonomy_root = resolve_operator_path(
            args.taxonomy_root,
            require_mapping(config.get("taxonomy"), "taxonomy"),
            field="taxonomy root",
        )
        full_report = resolve_operator_path(
            args.full_report,
            require_mapping(
                require_mapping(config.get("parent_work_source_audit"), "parent").get("full_report"),
                "full report",
            ),
            field="full report",
        )
        report, outputs = run_audit(
            config_path=args.config,
            inventory_manifest_path=manifest,
            source_root=source_root,
            taxonomy_root=taxonomy_root,
            full_report_path=full_report,
            output_dir=args.output_dir,
            max_files_per_table=args.max_files_per_table,
            resume=args.resume,
            batch_size=args.batch_size,
        )
    except (TaxonomyReferenceAuditError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "reference_summary": report["reference_summary"],
                "execution": report["execution"],
                "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
                "outputs_manifest_sha256": sha256_file(
                    args.output_dir / "taxonomy_reference_audit.outputs.json"
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
