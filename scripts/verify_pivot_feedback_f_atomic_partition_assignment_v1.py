#!/usr/bin/env python3
"""Independently verify the atomic final-panel partition assignment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_atomic_partition_assignment_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_atomic_partition_assignment.v1"
VERIFY_SCHEMA = "pivot_feedback_f_atomic_partition_assignment_verification.v1"
PASS_STATUS = "PASS_INDEPENDENT_OUTCOME_BLIND_ATOMIC_EPISODE_FAMILY_PARTITION_ASSIGNMENT_NO_F_PASS"

EXPECTED_FILES = {
    "ATOMIC_PARTITION_ASSIGNMENT.md",
    "atomic_episode_partition_assignment.parquet",
    "atomic_family_partition_assignment.parquet",
    "atomic_partition_assignment.json",
    "atomic_partition_assignment.outputs.json",
    "atomic_partition_summary.parquet",
}

EPISODE_FIELDS = [
    ("episode_id", "uint16", False),
    ("source_slot_id", "string", False),
    ("registered_partition_id", "string", False),
    ("partition_role", "string", False),
    ("field_id", "uint8", False),
    ("topic_id", "uint32", False),
    ("subfield_id", "uint16", False),
    ("decision_year", "uint16", False),
    ("response_cohort_start_year", "uint16", False),
    ("response_cohort_end_year", "uint16", False),
    ("response_crosses_entry_calendar_boundary", "bool", False),
    ("whole_episode_assignment_action", "string", False),
    ("sealed", "bool", False),
    ("operator_only", "bool", False),
    ("author_cold", "bool", False),
    ("final_panel_count_closure", "bool", False),
    ("leaveout_surge_present", "bool", False),
]

FAMILY_FIELDS = [
    ("family_ordinal", "uint16", False),
    ("family_key", "string", False),
    ("field_id", "uint8", False),
    ("subfield_id", "uint16", False),
    ("registered_partition_id", "string", False),
    ("episode_count", "uint16", False),
    ("minimum_decision_year", "uint16", False),
    ("maximum_decision_year", "uint16", False),
    ("any_response_crosses_entry_calendar_boundary", "bool", False),
    ("g1_g4_cross_partition", "bool", False),
    ("sealed", "bool", False),
    ("operator_only", "bool", False),
    ("author_cold", "bool", False),
]

PARTITION_FIELDS = [
    ("partition_order", "uint8", False),
    ("partition_id", "string", False),
    ("entry_start_year", "uint16", True),
    ("entry_end_year", "uint16", False),
    ("followup_through_year", "uint16", True),
    ("role", "string", False),
    ("sealed", "bool", False),
    ("operator_only", "bool", False),
    ("author_cold", "bool", False),
    ("assigned_episode_count", "uint16", False),
    ("assigned_family_count", "uint16", False),
    ("response_boundary_crossing_episode_count", "uint16", False),
    ("episode_assignment_status", "string", False),
    ("row_level_membership_status", "string", False),
    ("model_selection_fold_status", "string", False),
]


class AtomicPartitionVerificationError(RuntimeError):
    """Fail-closed atomic partition verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AtomicPartitionVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _safe_repo_path(relative: Any) -> Path:
    require(isinstance(relative, str) and relative, "empty repository path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe repository path")
    path = (ROOT / pure).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    require(path.is_file() and not path.is_symlink(), "invalid repository path")
    return path


def _schema_version(path: Path) -> str:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return metadata.get(b"schema_version", b"").decode()


def _assert_schema(path: Path, expected: Sequence[tuple[str, str, bool]], version: str) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    observed = [(field.name, str(field.type), field.nullable) for field in schema]
    require(observed == list(expected), f"schema fields: {path.name}")
    require((schema.metadata or {}).get(b"schema_version", b"").decode() == version, f"schema version: {path.name}")


def _verify_repository_sources(config: Mapping[str, Any]) -> None:
    authorization = _mapping(config["authorization_parent"], "authorization parent")
    authorization_path = _safe_repo_path(authorization["relative_path"])
    require(_sha256(authorization_path) == authorization["sha256"], "authorization SHA")
    authorization_doc = _yaml(authorization_path)
    require(authorization_doc.get("schema_version") == authorization["required_schema_version"], "authorization schema")
    require(
        authorization_doc["authorization_after_pass"][authorization["required_authorization"]]
        == authorization["required_value"],
        "atomic assignment authorization",
    )

    sources = _mapping(config["bound_design_sources"], "bound sources")
    summary_binding = _mapping(sources["aligned_final_panel_summary"], "summary binding")
    summary_path = _safe_repo_path(summary_binding["relative_path"])
    require(_sha256(summary_path) == summary_binding["sha256"], "summary SHA")
    summary = _json(summary_path)
    require(summary.get("schema_version") == summary_binding["required_schema_version"], "summary schema")
    require(summary.get("status") == summary_binding["required_status"], "summary status")
    require(summary["fail_closed_state"]["atomic_partition_assignment_complete"] is False, "predecessor atomic state")

    scientific_binding = _mapping(sources["scientific_sections"], "scientific binding")
    scientific_path = _safe_repo_path(scientific_binding["relative_path"])
    require(_sha256(scientific_path) == scientific_binding["sha256"], "scientific SHA")
    scientific = _yaml(scientific_path)
    require(scientific.get("schema_version") == scientific_binding["required_schema_version"], "scientific schema")
    section = scientific["sections"][scientific_binding["required_section"]]
    contract = config["atomic_partition_contract"]
    require(section["atomicity"]["all_response_cohorts_of_episode_stay_together"] is True, "scientific episode atomicity")
    require(section["atomicity"]["shock_family_cannot_cross_g1_and_g4"] is True, "scientific family atomicity")
    require(section["atomicity"]["crossing_episode_action"] == contract["crossing_episode_action"], "scientific crossing action")
    for key in ("fold_count", "grouping", "assignment", "balance_priority", "component_may_be_split"):
        require(section["model_selection_folds"][key] == contract["model_selection_folds"][key], f"scientific fold rule: {key}")


def _verify_output_tree(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(root.is_dir() and not root.is_symlink(), "output root")
    require((root.stat().st_mode & 0o777) == 0o700, "output root mode")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"non-file output: {path.name}")
        require((path.stat().st_mode & 0o777) == 0o600, f"output mode: {path.name}")
    receipt_path = root / "atomic_partition_assignment.outputs.json"
    receipt = _json(receipt_path)
    output_contract = config["output_contract"]
    require(receipt.get("schema_version") == output_contract["receipt_schema_version"], "receipt schema")
    require(receipt.get("status") == output_contract["passing_status"], "receipt status")
    require(receipt.get("receipt_written_last") is True, "receipt last flag")
    artifacts = {
        str(row["relative_path"]): dict(row) for row in receipt.get("artifacts", [])
    }
    require(set(artifacts) == EXPECTED_FILES - {receipt_path.name}, "receipt artifact set")
    require(receipt.get("artifact_count") == len(artifacts), "receipt artifact count")
    tree = []
    for relative in sorted(artifacts):
        path = root / relative
        row = artifacts[relative]
        require(path.stat().st_size == row["bytes"], f"artifact bytes: {relative}")
        require(_sha256(path) == row["sha256"], f"artifact SHA: {relative}")
        if path.suffix == ".parquet":
            require(pq.ParquetFile(path).metadata.num_rows == row["rows"], f"artifact rows: {relative}")
        tree.append((relative, row["bytes"], row["sha256"]))
    require(hashlib.sha256(_canonical_bytes(tree)).hexdigest() == receipt["artifact_tree_sha256"], "artifact tree SHA")
    require(receipt_path.stat().st_mtime_ns > max((root / relative).stat().st_mtime_ns for relative in artifacts), "receipt not written last")
    require(receipt["atomic_episode_family_partition_assignment_complete"] is True, "atomic assignment incomplete")
    require(receipt["operator_only_private_holdout_materialization_authorized_next"] is True, "materialization not authorized")
    for key in (
        "row_level_model_selection_fold_assignment_complete",
        "private_holdout_materialization_complete",
        "row_level_g4_author_cold_enforcement_complete",
        "model_score_read",
        "model_fitted",
        "g4_open",
        "pass_f",
        "pass_g0",
    ):
        require(receipt[key] is False, f"unsafe receipt state: {key}")
    for key in (
        "author_token_membership_rows_read",
        "holdout_paths_enumerated",
        "holdout_bytes_hashed",
        "confirmatory_entry_or_persistence_labels_read",
    ):
        require(receipt[key] == 0, f"unsafe receipt count: {key}")
    return receipt, artifacts


def _load_parent(root: Path, binding: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "parent basename")
    require(root.is_dir() and not root.is_symlink(), "parent root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "parent receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "parent receipt schema")
    require(receipt.get("status") == binding["required_status"], "parent receipt status")
    require(receipt.get("final_panel_conformance_complete") is True, "parent final-panel state")
    require(receipt.get("atomic_partition_assignment_authorized_next") is True, "parent authorization")
    for key in ("pass_f", "pass_g0", "g4_open", "model_fitted"):
        require(receipt.get(key) is False, f"unsafe parent state: {key}")
    verification_path = Path(str(binding["verification_path"])).resolve(strict=True)
    require(_sha256(verification_path) == binding["verification_sha256"], "parent verification SHA")
    require(_json(verification_path).get("status") == binding["verification_status"], "parent verification status")
    episode_path = root / str(binding["episode_basename"])
    slot_path = root / str(binding["slot_basename"])
    require(_sha256(episode_path) == binding["episode_sha256"], "parent episode SHA")
    require(_sha256(slot_path) == binding["slot_sha256"], "parent slot SHA")
    require(_schema_version(episode_path) == binding["episode_schema_version"], "parent episode schema")
    require(_schema_version(slot_path) == binding["slot_schema_version"], "parent slot schema")
    episodes = [dict(row) for row in pq.read_table(episode_path).to_pylist()]
    slots = [dict(row) for row in pq.read_table(slot_path).to_pylist()]
    require(len(episodes) == binding["episode_rows"] == 65, "parent episode rows")
    require(len(slots) == binding["slot_rows"] == 4, "parent slot rows")
    require(all(row["aligned_count_closure"] and row["label_independent_support_pass"] and row["leaveout_surge_present"] for row in episodes), "parent episode conformance")
    require(all(row["final_panel_slot_pass"] for row in slots), "parent slot conformance")
    return episodes, slots


def _inside(year: int, partition: Mapping[str, Any]) -> bool:
    start = partition["entry_start_year"]
    return (start is None or year >= int(start)) and year <= int(partition["entry_end_year"])


def _expected_rows(
    source_episodes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    contract = config["atomic_partition_contract"]
    partitions = [dict(row) for row in contract["registered_partitions"]]
    by_id = {str(row["partition_id"]): row for row in partitions}
    offsets = [int(value) for value in contract["response_cohort_offsets"]]
    require(offsets == [0, 1, 2], "response offsets")
    episode_rows = []
    for source in sorted(source_episodes, key=lambda row: int(row["episode_id"])):
        partition_id = str(source["temporal_partition_id"])
        require(partition_id in by_id, "source partition")
        partition = by_id[partition_id]
        decision_year = int(source["decision_year"])
        matches = [row for row in partitions if _inside(decision_year, row)]
        require(len(matches) == 1 and matches[0]["partition_id"] == partition_id, "event-year partition closure")
        response_years = [decision_year + offset for offset in offsets]
        episode_rows.append(
            {
                "episode_id": int(source["episode_id"]),
                "source_slot_id": str(source["slot_id"]),
                "registered_partition_id": partition_id,
                "partition_role": str(partition["role"]),
                "field_id": int(source["field_id"]),
                "topic_id": int(source["topic_id"]),
                "subfield_id": int(source["subfield_id"]),
                "decision_year": decision_year,
                "response_cohort_start_year": min(response_years),
                "response_cohort_end_year": max(response_years),
                "response_crosses_entry_calendar_boundary": not all(
                    _inside(year, partition) for year in response_years
                ),
                "whole_episode_assignment_action": contract["crossing_episode_action"],
                "sealed": bool(partition["sealed"]),
                "operator_only": bool(partition["operator_only"]),
                "author_cold": bool(partition["author_cold"]),
                "final_panel_count_closure": bool(source["aligned_count_closure"]),
                "leaveout_surge_present": bool(source["leaveout_surge_present"]),
            }
        )

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        grouped[(row["field_id"], row["subfield_id"])].append(row)
    order = {str(row["partition_id"]): int(row["partition_order"]) for row in partitions}
    keys = sorted(grouped, key=lambda key: (order[grouped[key][0]["registered_partition_id"]], key[0], key[1]))
    family_rows = []
    for ordinal, key in enumerate(keys, 1):
        members = grouped[key]
        partition_ids = {str(row["registered_partition_id"]) for row in members}
        require(len(partition_ids) == 1, "family crosses partition")
        partition_id = next(iter(partition_ids))
        g1_g4_cross = len(partition_ids & {"g1_empirical_replication", "g4_temporal_holdout"}) > 1
        require(not g1_g4_cross, "family crosses G1 and G4")
        partition = by_id[partition_id]
        years = [int(row["decision_year"]) for row in members]
        family_rows.append(
            {
                "family_ordinal": ordinal,
                "family_key": f"field-{key[0]}-subfield-{key[1]}",
                "field_id": key[0],
                "subfield_id": key[1],
                "registered_partition_id": partition_id,
                "episode_count": len(members),
                "minimum_decision_year": min(years),
                "maximum_decision_year": max(years),
                "any_response_crosses_entry_calendar_boundary": any(row["response_crosses_entry_calendar_boundary"] for row in members),
                "g1_g4_cross_partition": False,
                "sealed": bool(partition["sealed"]),
                "operator_only": bool(partition["operator_only"]),
                "author_cold": bool(partition["author_cold"]),
            }
        )

    partition_rows = []
    expected_counts = contract["expected_final_panel_partition_counts"]
    for partition in sorted(partitions, key=lambda row: int(row["partition_order"])):
        partition_id = str(partition["partition_id"])
        selected_episodes = [row for row in episode_rows if row["registered_partition_id"] == partition_id]
        selected_families = [row for row in family_rows if row["registered_partition_id"] == partition_id]
        require(len(selected_episodes) == int(expected_counts[partition_id]), f"expected partition count: {partition_id}")
        episode_status = "ASSIGNED_FROM_VERIFIED_FINAL_PANEL" if selected_episodes else "REGISTERED_EMPTY_IN_FROZEN_FINAL_SHOCK_PANEL"
        if partition_id == "g4_temporal_holdout":
            row_status = "PENDING_OPERATOR_ONLY_MATERIALIZATION_AND_PREFLIGHT"
        elif partition_id == "model_selection":
            row_status = contract["model_selection_folds"]["row_level_assignment_status"]
        elif partition_id == "g1_empirical_replication":
            row_status = "DEFERRED_UNTIL_POST_F_AUTHORIZED_ANALYTIC_ROW_MATERIALIZATION"
        else:
            row_status = "NOT_APPLICABLE_TO_FINAL_SHOCK_PANEL"
        fold_status = contract["model_selection_folds"]["row_level_assignment_status"] if partition_id == "model_selection" else "NOT_APPLICABLE"
        partition_rows.append(
            {
                "partition_order": int(partition["partition_order"]),
                "partition_id": partition_id,
                "entry_start_year": partition["entry_start_year"],
                "entry_end_year": int(partition["entry_end_year"]),
                "followup_through_year": partition["followup_through_year"],
                "role": str(partition["role"]),
                "sealed": bool(partition["sealed"]),
                "operator_only": bool(partition["operator_only"]),
                "author_cold": bool(partition["author_cold"]),
                "assigned_episode_count": len(selected_episodes),
                "assigned_family_count": len(selected_families),
                "response_boundary_crossing_episode_count": sum(bool(row["response_crosses_entry_calendar_boundary"]) for row in selected_episodes),
                "episode_assignment_status": episode_status,
                "row_level_membership_status": row_status,
                "model_selection_fold_status": fold_status,
            }
        )
    return episode_rows, family_rows, partition_rows


def verify(
    *,
    config_path: Path,
    final_panel_root: Path,
    output_root: Path,
    verification_output: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    _verify_repository_sources(config)
    output_root = output_root.resolve(strict=True)
    require(output_root.name == config["output_contract"]["root_basename"], "output basename")
    verification_output = verification_output.resolve()
    require(not verification_output.exists() and verification_output.parent.is_dir(), "verification output must be new")
    receipt, artifacts = _verify_output_tree(output_root, config)
    source_episodes, source_slots = _load_parent(
        final_panel_root,
        _mapping(config["verified_aligned_final_panel_parent"], "parent binding"),
    )
    expected_episodes, expected_families, expected_partitions = _expected_rows(source_episodes, config)

    episode_path = output_root / "atomic_episode_partition_assignment.parquet"
    family_path = output_root / "atomic_family_partition_assignment.parquet"
    partition_path = output_root / "atomic_partition_summary.parquet"
    _assert_schema(episode_path, EPISODE_FIELDS, config["output_contract"]["episode_schema_version"])
    _assert_schema(family_path, FAMILY_FIELDS, config["output_contract"]["family_schema_version"])
    _assert_schema(partition_path, PARTITION_FIELDS, config["output_contract"]["partition_schema_version"])
    episode_rows = pq.read_table(episode_path).to_pylist()
    family_rows = pq.read_table(family_path).to_pylist()
    partition_rows = pq.read_table(partition_path).to_pylist()
    forbidden = {"author_id", "author_numeric_id", "work_id", "work_numeric_id", "focal_author_id", "peer_author_id"}
    for path in (episode_path, family_path, partition_path):
        require(not (forbidden & set(pq.ParquetFile(path).schema_arrow.names)), f"raw ID in output: {path.name}")
    require(episode_rows == expected_episodes, "episode assignment differs from independent reconstruction")
    require(family_rows == expected_families, "family assignment differs from independent reconstruction")
    require(partition_rows == expected_partitions, "partition summary differs from independent reconstruction")

    report = _json(output_root / "atomic_partition_assignment.json")
    require(report.get("schema_version") == config["output_contract"]["report_schema_version"], "report schema")
    require(report.get("status") == receipt["status"], "report status")
    execution = _mapping(report["execution"], "execution")
    require(execution["config_relative_path"] == config_path.relative_to(ROOT).as_posix(), "executed config path")
    require(execution["config_sha256"] == _sha256(config_path), "executed config SHA")
    builder = _safe_repo_path(execution["builder_relative_path"])
    require(_sha256(builder) == execution["builder_sha256"], "executed builder SHA")
    accounting = report["accounting"]
    require(accounting["source_episode_rows"] == len(source_episodes) == 65, "report source episodes")
    require(accounting["source_slot_rows"] == len(source_slots) == 4, "report source slots")
    require(accounting["atomic_episode_rows"] == len(episode_rows) == 65, "report episode rows")
    require(accounting["atomic_family_rows"] == len(family_rows) == 29, "report family rows")
    require(accounting["partition_rows"] == len(partition_rows) == 4, "report partition rows")
    crossing_count = sum(bool(row["response_crosses_entry_calendar_boundary"]) for row in episode_rows)
    require(accounting["response_boundary_crossing_episode_rows"] == crossing_count, "report crossing episodes")
    require(report["partition_summary"] == partition_rows, "report partition table")
    require(report["authorization"]["operator_only_private_holdout_materialization"] == "AUTHORIZED_NEXT", "next authorization")
    scope = report["scope"]
    require(scope["atomic_episode_family_partition_assignment_complete"] is True, "scope atomic state")
    for key in (
        "row_level_model_selection_fold_assignment_complete",
        "private_holdout_materialization_complete",
        "row_level_g4_author_cold_enforcement_complete",
        "model_score_read",
        "model_fitted",
        "g4_open",
        "pass_f",
        "pass_g0",
    ):
        require(scope[key] is False, f"unsafe report scope: {key}")
    for key in (
        "author_token_membership_rows_read",
        "holdout_paths_enumerated",
        "holdout_bytes_hashed",
        "confirmatory_entry_or_persistence_labels_read",
    ):
        require(scope[key] == 0, f"unsafe report scope count: {key}")

    verification = {
        "schema_version": VERIFY_SCHEMA,
        "status": PASS_STATUS,
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(output_root / "atomic_partition_assignment.outputs.json"),
        "artifact_count": len(artifacts),
        "episode_rows_verified": len(episode_rows),
        "family_rows_verified": len(family_rows),
        "partition_rows_verified": len(partition_rows),
        "partition_episode_counts_verified": {
            row["partition_id"]: int(row["assigned_episode_count"])
            for row in partition_rows
        },
        "partition_family_counts_verified": {
            row["partition_id"]: int(row["assigned_family_count"])
            for row in partition_rows
        },
        "response_boundary_crossing_episode_rows_verified": crossing_count,
        "whole_episode_and_family_atomicity_verified": True,
        "g1_g4_family_disjointness_verified": True,
        "raw_id_columns_absent": True,
        "atomic_episode_family_partition_assignment_complete": True,
        "operator_only_private_holdout_materialization_authorized_next": True,
        "row_level_model_selection_fold_assignment_complete": False,
        "private_holdout_materialization_complete": False,
        "row_level_g4_author_cold_enforcement_complete": False,
        "author_token_membership_rows_read": 0,
        "holdout_paths_enumerated": 0,
        "holdout_bytes_hashed": 0,
        "confirmatory_entry_or_persistence_labels_read": 0,
        "model_score_read": False,
        "model_fitted": False,
        "g4_open": False,
        "pass_f": False,
        "pass_g0": False,
    }
    verification_output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(verification_output, 0o600)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--final-panel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        final_panel_root=args.final_panel_root,
        output_root=args.output_root,
        verification_output=args.verification_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
