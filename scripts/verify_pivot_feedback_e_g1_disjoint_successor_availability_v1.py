#!/usr/bin/env python3
"""Verify that no current frozen partition is an eligible disjoint G1 successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORITY = ROOT / "experiments/configs/pivot_feedback_e_g1_disjoint_successor_availability_v1.yaml"
ATOMIC_EPISODES = Path("/ssd/pivot-feedback-f-atomic-partition-assignment-v1-20260804/atomic_episode_partition_assignment.parquet")
ATOMIC_EPISODES_SHA = "169f0bd4bedaf4f34b3dec53214f1393587c6dcc26e0dac7eae60d589e91a17e"
ATOMIC_EPISODES_ROWS = 65
AUTHORITY_STATUS = "BLOCKED_NO_CURRENT_REGISTERED_DISJOINT_SUCCESSOR_COHORT_NEW_SOURCE_REQUIRED"
VERIFY_STATUS = "PASS_INDEPENDENT_NO_CURRENT_REGISTERED_DISJOINT_SUCCESSOR_COHORT_NEW_SOURCE_REQUIRED_NO_AUTHORITY_CHANGE"


class DisjointSuccessorAvailabilityVerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DisjointSuccessorAvailabilityVerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_repo_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    pure = PurePosixPath(str(binding["relative_path"]))
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe repository binding")
    path = ROOT.joinpath(*pure.parts)
    require(path.is_file() and not path.is_symlink(), "invalid repository binding")
    require(_sha256(path) == binding["sha256"], f"repository hash drift: {path.name}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix == ".yaml" else json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == binding["required_schema_version"], f"repository schema drift: {path.name}")
    require(payload.get("status") == binding["required_status"], f"repository status drift: {path.name}")
    require(binding.get("mutation_allowed") is False, f"mutable repository binding: {path.name}")
    return payload


def verify(authority_path: Path) -> dict[str, Any]:
    authority_path = authority_path.resolve(strict=True)
    authority = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    require(authority.get("schema_version") == "pivot_feedback_e_g1_disjoint_successor_availability.v1", "authority schema drift")
    require(authority.get("status") == AUTHORITY_STATUS, "authority status drift")
    require(authority.get("mode") == "POST_TERMINAL_METADATA_ONLY_ROUTING_CLOSURE_NOT_NEW_GATE_NO_AUTHORITY_CHANGE", "authority mode drift")

    current = _load_repo_binding(authority["active_state_parent"]["authority"])
    current_summary = _load_repo_binding(authority["active_state_parent"]["independent_verification"])
    terminal = _load_repo_binding(authority["active_state_parent"]["terminal_execution_verification"])
    atomic_contract = _load_repo_binding(authority["frozen_partition_sources"]["atomic_partition_contract"])
    atomic_summary = _load_repo_binding(authority["frozen_partition_sources"]["atomic_partition_verification"])
    field_contract = _load_repo_binding(authority["frozen_partition_sources"]["field_episode_contract"])
    _load_repo_binding(authority["frozen_partition_sources"]["field_episode_verification"])

    require(current["lifecycle"]["empirical_g1_status"] == "BLOCKED", "current G1 not terminal blocked")
    require(current["lifecycle"]["confirmatory_or_holdout_rows_read"] == 72_715_221, "current confirmatory read drift")
    require(current["authorization"]["same_cohort_g1_retry"] == "NOT_AUTHORIZED", "same cohort retry drift")
    require(current["authorization"]["disjoint_g1_successor"] == "REQUIRES_NEW_OUTCOME_BLIND_PREAUTHORIZATION_AND_FRESH_FREEZE", "disjoint successor boundary drift")
    require(current_summary["routing_replay"]["pass_e"] is False, "E unexpectedly passed")
    require(current_summary["routing_replay"]["H_state"] == "BYTE_SEALED_UNOPENED", "H seal state drift")
    require(terminal["registered_execution_disposition"]["terminal_status"] == "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION", "terminal result drift")
    require(terminal["registered_execution_disposition"]["failed_unit_count"] == 2, "support failure count drift")

    assignment = authority["frozen_partition_sources"]["atomic_episode_assignment"]
    require(assignment["path"] == str(ATOMIC_EPISODES), "atomic episode path drift")
    require(ATOMIC_EPISODES.is_file() and not ATOMIC_EPISODES.is_symlink(), "invalid atomic episode artifact")
    require(stat.S_IMODE(ATOMIC_EPISODES.stat().st_mode) == 0o600, "atomic episode mode drift")
    require(ATOMIC_EPISODES.stat().st_size == assignment["bytes"] == 6124, "atomic episode bytes drift")
    require(_sha256(ATOMIC_EPISODES) == assignment["sha256"] == ATOMIC_EPISODES_SHA, "atomic episode hash drift")
    parquet = pq.ParquetFile(ATOMIC_EPISODES)
    require(parquet.metadata.num_rows == assignment["rows"] == ATOMIC_EPISODES_ROWS, "atomic episode rows drift")
    require((parquet.schema_arrow.metadata or {}).get(b"schema_version", b"").decode() == assignment["schema_version"], "atomic episode schema drift")
    expected_columns = assignment["metadata_columns_read"]
    require(expected_columns == [
        "episode_id", "registered_partition_id", "partition_role", "field_id",
        "decision_year", "sealed", "operator_only", "author_cold",
    ], "metadata projection drift")
    rows = pq.read_table(ATOMIC_EPISODES, columns=expected_columns).to_pylist()
    require(len(rows) == ATOMIC_EPISODES_ROWS, "atomic episode scan closure")
    counts = {name: 0 for name in (
        "discovery_calibration", "g1_empirical_replication", "model_selection", "g4_temporal_holdout"
    )}
    field_sets: dict[str, set[int]] = {name: set() for name in counts}
    years: dict[str, set[int]] = {name: set() for name in counts}
    for row in rows:
        partition = str(row["registered_partition_id"])
        require(partition in counts, "unexpected partition")
        counts[partition] += 1
        field_sets[partition].add(int(row["field_id"]))
        years[partition].add(int(row["decision_year"]))
        if partition == "g4_temporal_holdout":
            require(row["sealed"] is True and row["operator_only"] is True and row["author_cold"] is True, "G4 boundary drift")
        else:
            require(row["sealed"] is False and row["operator_only"] is False, "open partition boundary drift")
    require(counts == {
        "discovery_calibration": 0,
        "g1_empirical_replication": 28,
        "model_selection": 0,
        "g4_temporal_holdout": 37,
    }, "partition count replay drift")
    require(field_sets["g1_empirical_replication"] == {11, 16, 32}, "G1 field replay drift")
    require(years["g1_empirical_replication"] == {2013, 2014, 2015, 2016}, "G1 year replay drift")
    require(field_sets["g4_temporal_holdout"] == {27}, "G4 field replay drift")
    require(years["g4_temporal_holdout"] == {2020, 2021, 2022}, "G4 year replay drift")

    partition_contract = atomic_contract["atomic_partition_contract"]
    registered = {row["partition_id"]: row for row in partition_contract["registered_partitions"]}
    require(registered["model_selection"]["role"] == "OPEN_ONLY_AFTER_E", "model selection role drift")
    require(partition_contract["selected_episode_reselection"] == "FORBIDDEN", "episode reselection drift")
    require(partition_contract["outcome_based_assignment"] == "FORBIDDEN", "outcome assignment drift")
    require(partition_contract["expected_final_panel_partition_counts"] == counts, "contract partition count drift")
    require(field_contract["episode_assignment"]["reserve_activation_after_outcome_read"] is False, "reserve activation drift")
    require(field_contract["episode_assignment"]["outcome_based_candidate_or_field_substitution"] is False, "outcome substitution drift")
    summary_counts = {
        row["partition_id"]: int(row["assigned_episode_count"])
        for row in atomic_summary["partition_assignment"]
    }
    require(summary_counts == counts, "independent atomic summary drift")

    replay = authority["partition_replay"]
    require(all(row["successor_available"] is False for row in replay.values()), "current successor marked available")
    require(replay["discovery_calibration"]["final_panel_episode_count"] == 0, "discovery count drift")
    require(replay["g1_empirical_replication"]["final_panel_episode_count"] == 28, "G1 count drift")
    require(replay["model_selection"]["final_panel_episode_count"] == 0, "model count drift")
    require(replay["g4_temporal_holdout"]["final_panel_episode_count"] == 37, "G4 count drift")
    disposition = authority["availability_disposition"]
    require(disposition == {
        "current_registered_disjoint_episode_count": 0,
        "current_registered_disjoint_family_count": 0,
        "current_registered_disjoint_cohort_available": False,
        "external_or_future_source_identity_bound": False,
        "successor_execution_authorized": False,
        "terminal_status": AUTHORITY_STATUS,
        "this_is_new_scientific_gate": False,
        "this_changes_current_state_v4": False,
        "current_g1_terminal_status_preserved": "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION",
    }, "availability disposition drift")
    requirements = authority["disjoint_successor_requirements"]
    require(requirements["source_identity"] == "NEW_IMMUTABLE_SNAPSHOT_OR_EXTERNAL_DATASET_NOT_CURRENT_REGISTERED_PARTITION", "new source requirement drift")
    require(requirements["source_receipt_before_row_access"] == "REQUIRED", "source receipt requirement drift")
    require(requirements["claim_and_estimand_freeze_before_outcome_labels"] == "REQUIRED", "fresh claim freeze drift")
    require(requirements["holdout_seal_state"] == "BYTE_SEALED_UNOPENED" and requirements["g4_open"] is False, "holdout boundary drift")
    validation = authority["validation_contract"]
    require(validation["repository_and_open_partition_metadata_only"] is True, "validation mode drift")
    require(validation["current_g1_label_files_may_be_read"] is False, "G1 label access drift")
    require(validation["model_selection_rows_may_be_read"] is False, "model row access drift")
    require(validation["holdout_or_private_g4_files_may_be_read"] is False, "G4 access drift")
    require(validation["seal_may_be_opened"] is False and validation["model_may_be_fitted"] is False, "unsafe validation")

    return {
        "schema_version": "pivot_feedback_e_g1_disjoint_successor_availability_summary.v1",
        "status": VERIFY_STATUS,
        "authority": {
            "relative_path": authority_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(authority_path),
            "current_state_v4_sha256": authority["active_state_parent"]["authority"]["sha256"],
            "terminal_execution_summary_sha256": authority["active_state_parent"]["terminal_execution_verification"]["sha256"],
        },
        "partition_metadata_replay": {
            "atomic_episode_assignment_sha256": ATOMIC_EPISODES_SHA,
            "atomic_episode_rows_read": len(rows),
            "projected_metadata_columns": expected_columns,
            "episode_counts": counts,
            "field_ids": {key: sorted(value) for key, value in field_sets.items()},
            "decision_years": {key: sorted(value) for key, value in years.items()},
            "current_registered_disjoint_episode_count": 0,
        },
        "route_replay": replay,
        "availability_disposition": disposition,
        "next_executable_prerequisite": {
            "status": "NEW_IMMUTABLE_MATURED_SOURCE_OR_EXTERNAL_DATASET_REQUIRED",
            "metadata_only_source_discovery_and_inventory": "AUTHORIZED_NEXT",
            "successor_row_access": "NOT_AUTHORIZED_UNTIL_FRESH_FREEZE_AND_INDEPENDENT_PREFLIGHT",
            "same_cohort_or_current_partition_reuse": "NOT_AUTHORIZED",
        },
        "access_accounting": {
            "repository_documents_read": 7,
            "open_partition_metadata_rows_read": len(rows),
            "current_g1_label_rows_read": 0,
            "model_selection_rows_read": 0,
            "holdout_or_private_g4_rows_read": 0,
            "seal_open_events": 0,
            "g4_open": False,
            "models_fitted": 0,
        },
        "authorization": authority["authorization_after_independent_pass"],
        "claim_boundary": authority["claim_boundary"],
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise DisjointSuccessorAvailabilityVerificationError(f"output exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify(args.authority)
    if args.output is not None:
        _write_create_only(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
