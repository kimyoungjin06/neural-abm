#!/usr/bin/env python3
"""Build the outcome-blind atomic final-panel partition assignment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_atomic_partition_assignment_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_atomic_partition_assignment.v1"

EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("source_slot_id", pa.string(), nullable=False),
        pa.field("registered_partition_id", pa.string(), nullable=False),
        pa.field("partition_role", pa.string(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("response_cohort_start_year", pa.uint16(), nullable=False),
        pa.field("response_cohort_end_year", pa.uint16(), nullable=False),
        pa.field("response_crosses_entry_calendar_boundary", pa.bool_(), nullable=False),
        pa.field("whole_episode_assignment_action", pa.string(), nullable=False),
        pa.field("sealed", pa.bool_(), nullable=False),
        pa.field("operator_only", pa.bool_(), nullable=False),
        pa.field("author_cold", pa.bool_(), nullable=False),
        pa.field("final_panel_count_closure", pa.bool_(), nullable=False),
        pa.field("leaveout_surge_present", pa.bool_(), nullable=False),
    ]
)

FAMILY_SCHEMA = pa.schema(
    [
        pa.field("family_ordinal", pa.uint16(), nullable=False),
        pa.field("family_key", pa.string(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
        pa.field("registered_partition_id", pa.string(), nullable=False),
        pa.field("episode_count", pa.uint16(), nullable=False),
        pa.field("minimum_decision_year", pa.uint16(), nullable=False),
        pa.field("maximum_decision_year", pa.uint16(), nullable=False),
        pa.field("any_response_crosses_entry_calendar_boundary", pa.bool_(), nullable=False),
        pa.field("g1_g4_cross_partition", pa.bool_(), nullable=False),
        pa.field("sealed", pa.bool_(), nullable=False),
        pa.field("operator_only", pa.bool_(), nullable=False),
        pa.field("author_cold", pa.bool_(), nullable=False),
    ]
)

PARTITION_SCHEMA = pa.schema(
    [
        pa.field("partition_order", pa.uint8(), nullable=False),
        pa.field("partition_id", pa.string(), nullable=False),
        pa.field("entry_start_year", pa.uint16(), nullable=True),
        pa.field("entry_end_year", pa.uint16(), nullable=False),
        pa.field("followup_through_year", pa.uint16(), nullable=True),
        pa.field("role", pa.string(), nullable=False),
        pa.field("sealed", pa.bool_(), nullable=False),
        pa.field("operator_only", pa.bool_(), nullable=False),
        pa.field("author_cold", pa.bool_(), nullable=False),
        pa.field("assigned_episode_count", pa.uint16(), nullable=False),
        pa.field("assigned_family_count", pa.uint16(), nullable=False),
        pa.field("response_boundary_crossing_episode_count", pa.uint16(), nullable=False),
        pa.field("episode_assignment_status", pa.string(), nullable=False),
        pa.field("row_level_membership_status", pa.string(), nullable=False),
        pa.field("model_selection_fold_status", pa.string(), nullable=False),
    ]
)


class AtomicPartitionBuildError(RuntimeError):
    """Fail-closed atomic partition build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AtomicPartitionBuildError(message)


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


def _verify_repository_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    authorization = _mapping(config["authorization_parent"], "authorization parent")
    authorization_path = _safe_repo_path(authorization["relative_path"])
    require(_sha256(authorization_path) == authorization["sha256"], "authorization SHA")
    authorization_doc = _yaml(authorization_path)
    require(
        authorization_doc.get("schema_version") == authorization["required_schema_version"],
        "authorization schema",
    )
    require(
        authorization_doc["authorization_after_pass"][authorization["required_authorization"]]
        == authorization["required_value"],
        "atomic assignment not authorized",
    )

    sources = _mapping(config["bound_design_sources"], "bound design sources")
    summary_binding = _mapping(sources["aligned_final_panel_summary"], "final summary")
    summary_path = _safe_repo_path(summary_binding["relative_path"])
    require(_sha256(summary_path) == summary_binding["sha256"], "final summary SHA")
    summary = _json(summary_path)
    require(summary.get("schema_version") == summary_binding["required_schema_version"], "final summary schema")
    require(summary.get("status") == summary_binding["required_status"], "final summary status")
    require(summary["fail_closed_state"]["pass_f"] is False, "final summary passed F")

    scientific_binding = _mapping(sources["scientific_sections"], "scientific sections")
    scientific_path = _safe_repo_path(scientific_binding["relative_path"])
    require(_sha256(scientific_path) == scientific_binding["sha256"], "scientific sections SHA")
    scientific = _yaml(scientific_path)
    require(scientific.get("schema_version") == scientific_binding["required_schema_version"], "scientific sections schema")
    section = _mapping(
        scientific["sections"][scientific_binding["required_section"]],
        "partition section",
    )
    contract = _mapping(config["atomic_partition_contract"], "atomic contract")
    atomicity = _mapping(section["atomicity"], "scientific atomicity")
    require(atomicity["all_response_cohorts_of_episode_stay_together"] is True, "episode atomicity not frozen")
    require(atomicity["shock_family_cannot_cross_g1_and_g4"] is True, "family boundary not frozen")
    require(atomicity["crossing_episode_action"] == contract["crossing_episode_action"], "crossing action drift")
    folds = _mapping(section["model_selection_folds"], "scientific folds")
    for key in ("fold_count", "grouping", "assignment", "balance_priority", "component_may_be_split"):
        require(folds[key] == contract["model_selection_folds"][key], f"model-selection fold drift: {key}")
    return {
        "authorization_relative_path": authorization["relative_path"],
        "authorization_sha256": authorization["sha256"],
        "final_summary_relative_path": summary_binding["relative_path"],
        "final_summary_sha256": summary_binding["sha256"],
        "scientific_sections_relative_path": scientific_binding["relative_path"],
        "scientific_sections_sha256": scientific_binding["sha256"],
    }


def _load_verified_parent(root: Path, binding: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "final-panel root basename")
    require(root.is_dir() and not root.is_symlink(), "final-panel root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "final-panel receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "final-panel receipt schema")
    require(receipt.get("status") == binding["required_status"], "final-panel receipt status")
    require(receipt.get("atomic_partition_assignment_authorized_next") is True, "parent atomic authorization")
    require(receipt.get("final_panel_conformance_complete") is True, "parent conformance")
    for key in ("pass_f", "pass_g0", "g4_open", "model_fitted"):
        require(receipt.get(key) is False, f"unsafe final-panel parent: {key}")

    verification_path = Path(str(binding["verification_path"])).resolve(strict=True)
    require(_sha256(verification_path) == binding["verification_sha256"], "final-panel verification SHA")
    verification = _json(verification_path)
    require(verification.get("status") == binding["verification_status"], "final-panel verification status")
    require(verification.get("final_panel_conformance_complete") is True, "final-panel verification state")

    episode_path = root / str(binding["episode_basename"])
    slot_path = root / str(binding["slot_basename"])
    require(_sha256(episode_path) == binding["episode_sha256"], "final-panel episode SHA")
    require(_sha256(slot_path) == binding["slot_sha256"], "final-panel slot SHA")
    require(_schema_version(episode_path) == binding["episode_schema_version"], "final-panel episode schema")
    require(_schema_version(slot_path) == binding["slot_schema_version"], "final-panel slot schema")
    episodes = [dict(row) for row in pq.read_table(episode_path).to_pylist()]
    slots = [dict(row) for row in pq.read_table(slot_path).to_pylist()]
    require(len(episodes) == binding["episode_rows"] == 65, "final-panel episode rows")
    require(len(slots) == binding["slot_rows"] == 4, "final-panel slot rows")
    require(all(row["aligned_count_closure"] for row in episodes), "unaligned final-panel count")
    require(all(row["label_independent_support_pass"] for row in episodes), "final-panel support failure")
    require(all(row["leaveout_surge_present"] for row in episodes), "final-panel leaveout failure")
    require(all(row["final_panel_slot_pass"] for row in slots), "final-panel slot failure")
    return episodes, slots


def _year_in_partition(year: int, partition: Mapping[str, Any]) -> bool:
    start = partition["entry_start_year"]
    return (start is None or year >= int(start)) and year <= int(partition["entry_end_year"])


def _build_rows(
    episodes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    contract = _mapping(config["atomic_partition_contract"], "atomic contract")
    partitions = [dict(row) for row in contract["registered_partitions"]]
    by_id = {str(row["partition_id"]): row for row in partitions}
    offsets = [int(value) for value in contract["response_cohort_offsets"]]
    require(offsets == [0, 1, 2], "response offsets")

    episode_rows: list[dict[str, Any]] = []
    for source in sorted(episodes, key=lambda row: int(row["episode_id"])):
        partition_id = str(source["temporal_partition_id"])
        require(partition_id in by_id, "unknown source partition")
        partition = by_id[partition_id]
        decision_year = int(source["decision_year"])
        require(_year_in_partition(decision_year, partition), "event year outside registered partition")
        calendar_matches = [row for row in partitions if _year_in_partition(decision_year, row)]
        require(len(calendar_matches) == 1 and calendar_matches[0]["partition_id"] == partition_id, "non-atomic event calendar")
        response_years = [decision_year + offset for offset in offsets]
        crosses = not all(_year_in_partition(year, partition) for year in response_years)
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
                "response_crosses_entry_calendar_boundary": crosses,
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
    family_rows: list[dict[str, Any]] = []
    keys = sorted(
        grouped,
        key=lambda key: (
            order[grouped[key][0]["registered_partition_id"]],
            key[0],
            key[1],
        ),
    )
    for ordinal, key in enumerate(keys, 1):
        members = grouped[key]
        partition_ids = {row["registered_partition_id"] for row in members}
        require(len(partition_ids) == 1, "shock family crosses registered partitions")
        partition_id = next(iter(partition_ids))
        partition = by_id[partition_id]
        g1_g4_cross = len(partition_ids & {"g1_empirical_replication", "g4_temporal_holdout"}) > 1
        require(not g1_g4_cross, "shock family crosses G1 and G4")
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
                "any_response_crosses_entry_calendar_boundary": any(
                    row["response_crosses_entry_calendar_boundary"] for row in members
                ),
                "g1_g4_cross_partition": g1_g4_cross,
                "sealed": bool(partition["sealed"]),
                "operator_only": bool(partition["operator_only"]),
                "author_cold": bool(partition["author_cold"]),
            }
        )

    expected_counts = contract["expected_final_panel_partition_counts"]
    partition_rows: list[dict[str, Any]] = []
    for partition in sorted(partitions, key=lambda row: int(row["partition_order"])):
        partition_id = str(partition["partition_id"])
        selected_episodes = [row for row in episode_rows if row["registered_partition_id"] == partition_id]
        selected_families = [row for row in family_rows if row["registered_partition_id"] == partition_id]
        require(len(selected_episodes) == int(expected_counts[partition_id]), f"partition episode count: {partition_id}")
        if selected_episodes:
            episode_status = "ASSIGNED_FROM_VERIFIED_FINAL_PANEL"
        else:
            episode_status = "REGISTERED_EMPTY_IN_FROZEN_FINAL_SHOCK_PANEL"
        if partition_id == "g4_temporal_holdout":
            row_status = "PENDING_OPERATOR_ONLY_MATERIALIZATION_AND_PREFLIGHT"
        elif partition_id == "model_selection":
            row_status = contract["model_selection_folds"]["row_level_assignment_status"]
        elif partition_id == "g1_empirical_replication":
            row_status = "DEFERRED_UNTIL_POST_F_AUTHORIZED_ANALYTIC_ROW_MATERIALIZATION"
        else:
            row_status = "NOT_APPLICABLE_TO_FINAL_SHOCK_PANEL"
        fold_status = (
            contract["model_selection_folds"]["row_level_assignment_status"]
            if partition_id == "model_selection"
            else "NOT_APPLICABLE"
        )
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
                "response_boundary_crossing_episode_count": sum(
                    bool(row["response_crosses_entry_calendar_boundary"])
                    for row in selected_episodes
                ),
                "episode_assignment_status": episode_status,
                "row_level_membership_status": row_status,
                "model_selection_fold_status": fold_status,
            }
        )
    return episode_rows, family_rows, partition_rows


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema, version: str) -> None:
    versioned = schema.with_metadata({b"schema_version": version.encode("utf-8")})
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows], schema=versioned), path)
    os.chmod(path, 0o600)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        row["rows"] = pq.ParquetFile(path).metadata.num_rows
    return row


def build(*, config_path: Path, final_panel_root: Path, output_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    repository_bindings = _verify_repository_bindings(config)
    parent_binding = _mapping(config["verified_aligned_final_panel_parent"], "final-panel parent")
    source_episodes, source_slots = _load_verified_parent(final_panel_root, parent_binding)
    episode_rows, family_rows, partition_rows = _build_rows(source_episodes, config)

    output_root = output_root.resolve()
    output_contract = _mapping(config["output_contract"], "output contract")
    require(output_root.name == output_contract["root_basename"], "output basename")
    require(not output_root.exists() and output_root.parent.is_dir(), "output directory must be new")
    output_root.mkdir(mode=0o700)
    os.chmod(output_root, 0o700)

    episode_path = output_root / "atomic_episode_partition_assignment.parquet"
    family_path = output_root / "atomic_family_partition_assignment.parquet"
    partition_path = output_root / "atomic_partition_summary.parquet"
    _write_parquet(episode_path, episode_rows, EPISODE_SCHEMA, output_contract["episode_schema_version"])
    _write_parquet(family_path, family_rows, FAMILY_SCHEMA, output_contract["family_schema_version"])
    _write_parquet(partition_path, partition_rows, PARTITION_SCHEMA, output_contract["partition_schema_version"])

    crossing_count = sum(bool(row["response_crosses_entry_calendar_boundary"]) for row in episode_rows)
    report = {
        "schema_version": output_contract["report_schema_version"],
        "status": output_contract["passing_status"],
        "execution": {
            "config_relative_path": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": _sha256(config_path),
            "builder_relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "builder_sha256": _sha256(Path(__file__).resolve()),
        },
        "repository_bindings": repository_bindings,
        "input_bindings": {
            "final_panel_root_basename": final_panel_root.resolve().name,
            "final_panel_receipt_sha256": parent_binding["receipt_sha256"],
            "final_panel_episode_sha256": parent_binding["episode_sha256"],
            "final_panel_slot_sha256": parent_binding["slot_sha256"],
            "final_panel_verification_sha256": parent_binding["verification_sha256"],
        },
        "accounting": {
            "source_episode_rows": len(source_episodes),
            "source_slot_rows": len(source_slots),
            "atomic_episode_rows": len(episode_rows),
            "atomic_family_rows": len(family_rows),
            "partition_rows": len(partition_rows),
            "response_boundary_crossing_episode_rows": crossing_count,
            "g1_episode_rows": sum(row["registered_partition_id"] == "g1_empirical_replication" for row in episode_rows),
            "g1_family_rows": sum(row["registered_partition_id"] == "g1_empirical_replication" for row in family_rows),
            "model_selection_episode_rows": sum(row["registered_partition_id"] == "model_selection" for row in episode_rows),
            "g4_episode_rows": sum(row["registered_partition_id"] == "g4_temporal_holdout" for row in episode_rows),
            "g4_family_rows": sum(row["registered_partition_id"] == "g4_temporal_holdout" for row in family_rows),
        },
        "partition_summary": partition_rows,
        "model_selection_fold_recipe": config["atomic_partition_contract"]["model_selection_folds"],
        "g4_boundary": config["g4_boundary"],
        "authorization": config["authorization_after_pass"],
        "scope": {
            "atomic_episode_family_partition_assignment_complete": True,
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
        },
        "claim_boundary": config["claim_boundary"],
    }
    report_path = output_root / "atomic_partition_assignment.json"
    _write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")

    markdown = f"""# Pivot-feedback atomic partition assignment v1

Status: `{report['status']}`

The verified 65-episode final panel is assigned without splitting an episode,
response cohort, or field-subfield shock family. The manifest contains
{report['accounting']['g1_episode_rows']} G1 episodes in
{report['accounting']['g1_family_rows']} families and
{report['accounting']['g4_episode_rows']} operator-only G4 episodes in
{report['accounting']['g4_family_rows']} families.

{crossing_count} episodes have k=0,1,2 response years that extend beyond their
registered entry-calendar interval. They remain wholly assigned by registered
event year. Discovery and model-selection contain zero selected shock episodes;
no episode was fabricated to fill either cell. The five-fold model-selection
component recipe remains bound but row-level fold assignment is deferred.

This artifact reads no author-token membership, private holdout row, entry or
persistence label, realized citation effect, model score, or G4 outcome. It
authorizes operator-only private holdout materialization next, including the
required row-level author-cold enforcement. `pass_f=false`, `pass_g0=false`.
"""
    markdown_path = output_root / "ATOMIC_PARTITION_ASSIGNMENT.md"
    _write_text(markdown_path, markdown)

    artifact_paths = [markdown_path, report_path, episode_path, family_path, partition_path]
    artifacts = [_artifact(path, output_root) for path in sorted(artifact_paths)]
    tree = [(row["relative_path"], row["bytes"], row["sha256"]) for row in artifacts]
    receipt = {
        "schema_version": output_contract["receipt_schema_version"],
        "status": output_contract["passing_status"],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifact_tree_sha256": hashlib.sha256(_canonical_bytes(tree)).hexdigest(),
        "receipt_written_last": True,
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
    receipt_path = output_root / "atomic_partition_assignment.outputs.json"
    _write_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--final-panel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build(
        config_path=args.config,
        final_panel_root=args.final_panel_root,
        output_root=args.output_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
