#!/usr/bin/env python3
"""Build the metadata-only E/G1 execution-input preflight contract.

This tool reads only repository governance documents and operator metadata
receipts.  It never opens an analytic artifact, materializes a label, runs G1,
fits a predictive model, or touches the private G4 seal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

CURRENT_STATE = ROOT / "experiments/configs/pivot_feedback_current_state_v2.yaml"
CURRENT_STATE_SUMMARY = (
    ROOT
    / "experiments/evidence/results/pivot_feedback_current_state_v2_20260805.summary.json"
)
SECTIONS = (
    ROOT / "experiments/configs/pivot_feedback_f_pre_g1_scientific_sections_v1.yaml"
)
SECTIONS_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_pre_g1_scientific_sections_v1_20260803.summary.json"
)
TOLERANCE = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml"
)
TOLERANCE_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_g0_g1_replication_tolerance_precommitment_20260803.summary.json"
)
FINAL_PANEL_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_aligned_final_panel_conformance_v1_20260804.summary.json"
)
PARTITION_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_atomic_partition_assignment_v1_20260804.summary.json"
)

WORK_ROOT = Path("/ssd/pivot-feedback-cutoff-primitives-v0-full-20260723")
WORK_RECEIPT = WORK_ROOT / "pivot_feedback_cutoff_panel.outputs.json"
HISTORY_ROOT = Path("/ssd/openalex-g1-author-history-v3-full-20260722")
HISTORY_RECEIPT = HISTORY_ROOT / "author_history_audit.outputs.json"
FINAL_PANEL_ROOT = Path(
    "/ssd/pivot-feedback-f-aligned-final-panel-conformance-v1-20260804"
)
FINAL_PANEL_RECEIPT = FINAL_PANEL_ROOT / "aligned_final_panel_conformance.outputs.json"
PARTITION_ROOT = Path("/ssd/pivot-feedback-f-atomic-partition-assignment-v1-20260804")
PARTITION_RECEIPT = PARTITION_ROOT / "atomic_partition_assignment.outputs.json"

DEFAULT_OUTPUT = (
    ROOT
    / "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v1.yaml"
)

EXPECTED_REPOSITORY_DOCUMENTS = {
    CURRENT_STATE: (
        "ca431afb63da4cf2e09d0563c0509e3685efac29dc8679df0f9cc7f1ae7851ab",
        "pivot_feedback_current_state.v2",
        "ACTIVE_E_PHASE_F_ATOMICALLY_CLOSED_G0_COMPATIBILITY_PASS_HOLDOUT_SEALED_UNOPENED",
    ),
    CURRENT_STATE_SUMMARY: (
        "25600b8bb2cc3cc3578081134b4908607f1440151302f0ea31a38290ed279b61",
        "pivot_feedback_current_state_v2_summary.v1",
        "PASS_ATOMIC_F_CLOSURE_G0_COMPATIBILITY_PASS_E_ENTRY_READY_FAIL_CLOSED",
    ),
    SECTIONS: (
        "cc58a651667b7cecb4cc61d96b6700aa665f1121c7c0b0a254368fb0299041dd",
        "pivot_feedback_f_pre_g1_scientific_sections.v1",
        "FROZEN_THIRTEEN_PRE_G1_SECTIONS_PENDING_INDEPENDENT_VALIDATION_NO_F_PASS",
    ),
    SECTIONS_SUMMARY: (
        "c0c7da0d7eb26a510b02a20fd4c85b371cc05aad0603d893dc9d7e45c1a8b2ee",
        "pivot_feedback_f_pre_g1_scientific_sections_summary.v1",
        "PASS_THIRTEEN_PRE_G1_SCIENTIFIC_SECTIONS_FROZEN_NO_F_PASS",
    ),
    TOLERANCE: (
        "1bcf91d197a9b927286d43f3505528f3f3f7fb022ffd4fe89d95cb24c041184d",
        "pivot_feedback_g0_g1_replication_tolerance_precommitment.v1",
        "FROZEN_SINGLE_G1_REPLICATION_TOLERANCE_PENDING_INDEPENDENT_VALIDATION_NO_G0_PROMOTION",
    ),
    TOLERANCE_SUMMARY: (
        "72b653528d9cf5c8f1dd823f304934b48ac21a0dd4a50011ac3d02ef9079edb3",
        "pivot_feedback_g0_g1_replication_tolerance_precommitment_summary.v1",
        "PASS_ONE_OUTCOME_BLIND_G1_REPLICATION_TOLERANCE_AND_CLOSED_FIXTURE",
    ),
    FINAL_PANEL_SUMMARY: (
        "e18f0e954e109baf953b161eb1321d8be4a584ee6f4fc383989b8aa3fb2bc35e",
        "pivot_feedback_f_aligned_final_panel_conformance_summary.v1",
        "PASS_OUTCOME_BLIND_ALIGNED_FINAL_PANEL_LEAVEOUT_AND_SUPPORT_CONFORMANCE_ATOMIC_PARTITION_AUTHORIZED_NO_F_PASS",
    ),
    PARTITION_SUMMARY: (
        "a1c57c541dd7f7a6042e524d6340da2bcd1246eb48021958bb60ddd72e313b0d",
        "pivot_feedback_f_atomic_partition_assignment_summary.v1",
        "PASS_OUTCOME_BLIND_ATOMIC_EPISODE_FAMILY_PARTITION_ASSIGNMENT_PRIVATE_HOLDOUT_MATERIALIZATION_AUTHORIZED_NO_F_PASS",
    ),
}

EXPECTED_OPERATOR_RECEIPTS = {
    WORK_RECEIPT: (
        "91e3f689e574d524eabbe7daa58f5e678da37f851ba9b7af0561dab57b70846e",
        "pivot_feedback_cutoff_primitives_outputs.v1",
        "PASS_NORMALIZED_CURRENT_SNAPSHOT_CUTOFF_PRIMITIVES_WITH_RETROSPECTIVE_TOPIC_AND_CITATION_PROXY_LIMITATIONS",
    ),
    HISTORY_RECEIPT: (
        "7b3613eaac6900f376a999a551f15e20dd184de2b4dc6758a60d93f3eedb51a8",
        "openalex_g1_work_side_author_history_outputs.v3",
        "PASS_CURRENT_SNAPSHOT_WORK_SIDE_AUTHOR_TOKEN_LINKED_RETROSPECTIVE_HISTORY_ACCOUNTING_AND_SENSITIVITY_COMPONENT_AUTHOR_ENTITY_AND_PAST_OBSERVABILITY_UNVERIFIED",
    ),
    FINAL_PANEL_RECEIPT: (
        "222888f538652593c1a26fddf6eb25dca37a2edecd71afe744fa63ac4e695505",
        "pivot_feedback_f_aligned_final_panel_conformance_outputs.v1",
        "PASS_OUTCOME_BLIND_ALIGNED_FINAL_PANEL_LEAVEOUT_AND_SUPPORT_CONFORMANCE_ATOMIC_PARTITION_AUTHORIZED_NO_F_PASS",
    ),
    PARTITION_RECEIPT: (
        "0ddd5ec51d924167b332a2f756b5310c25454e2ad44db5d86859276814c4f2b4",
        "pivot_feedback_f_atomic_partition_assignment_outputs.v1",
        "PASS_OUTCOME_BLIND_ATOMIC_EPISODE_FAMILY_PARTITION_ASSIGNMENT_PRIVATE_HOLDOUT_MATERIALIZATION_AUTHORIZED_NO_F_PASS",
    ),
}


class G1InputPreflightBuildError(RuntimeError):
    """Raised when an exact predecessor or metadata receipt drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _load_repository(path: Path) -> dict[str, Any]:
    expected_sha, expected_schema, expected_status = EXPECTED_REPOSITORY_DOCUMENTS[path]
    if not path.is_file():
        raise G1InputPreflightBuildError(f"missing repository document: {path}")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise G1InputPreflightBuildError(
            f"repository document hash drift: {path}: {actual_sha}"
        )
    if path.suffix == ".yaml":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema:
        raise G1InputPreflightBuildError(f"schema drift: {path}")
    if payload.get("status") != expected_status:
        raise G1InputPreflightBuildError(f"status drift: {path}")
    return payload


def _load_operator_receipt(path: Path) -> dict[str, Any]:
    expected_sha, expected_schema, expected_status = EXPECTED_OPERATOR_RECEIPTS[path]
    if not path.parent.is_dir() or not path.is_file():
        raise G1InputPreflightBuildError(f"missing operator metadata receipt: {path}")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise G1InputPreflightBuildError(
            f"operator receipt hash drift: {path}: {actual_sha}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema:
        raise G1InputPreflightBuildError(f"operator receipt schema drift: {path}")
    if payload.get("status") != expected_status:
        raise G1InputPreflightBuildError(f"operator receipt status drift: {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise G1InputPreflightBuildError(f"operator receipt has no artifact list: {path}")
    relative_paths = [entry.get("relative_path") for entry in artifacts]
    if len(relative_paths) != len(set(relative_paths)):
        raise G1InputPreflightBuildError(f"duplicate artifact paths in receipt: {path}")
    return payload


def _canonical_inventory(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    normalized = sorted(entries, key=lambda entry: entry["relative_path"])
    encoded = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return normalized, hashlib.sha256(encoded).hexdigest()


def _select_prefix(receipt: dict[str, Any], prefix: str) -> tuple[list[dict[str, Any]], str]:
    selected = [
        entry for entry in receipt["artifacts"] if entry["relative_path"].startswith(prefix)
    ]
    if not selected:
        raise G1InputPreflightBuildError(f"empty artifact prefix selection: {prefix}")
    return _canonical_inventory(selected)


def _select_paths(
    receipt: dict[str, Any], required_paths: list[str]
) -> tuple[list[dict[str, Any]], str]:
    by_path = {entry["relative_path"]: entry for entry in receipt["artifacts"]}
    missing = [path for path in required_paths if path not in by_path]
    if missing:
        raise G1InputPreflightBuildError(f"missing bound artifacts: {missing}")
    return _canonical_inventory([by_path[path] for path in required_paths])


def _repository_binding(path: Path) -> dict[str, Any]:
    expected_sha, expected_schema, expected_status = EXPECTED_REPOSITORY_DOCUMENTS[path]
    return {
        "relative_path": _relative(path),
        "sha256": expected_sha,
        "required_schema_version": expected_schema,
        "required_status": expected_status,
        "mutation_allowed": False,
    }


def _operator_binding(
    *,
    root: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
    selector: dict[str, Any],
    entries: list[dict[str, Any]],
    inventory_sha256: str,
    include_entries: bool,
) -> dict[str, Any]:
    expected_sha, expected_schema, expected_status = EXPECTED_OPERATOR_RECEIPTS[
        receipt_path
    ]
    binding: dict[str, Any] = {
        "root": str(root),
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": expected_sha,
        "receipt_schema_version": expected_schema,
        "required_status": expected_status,
        "receipt_artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "selector": selector,
        "selected_artifact_count": len(entries),
        "selected_artifact_bytes": sum(int(entry["bytes"]) for entry in entries),
        "selected_artifact_inventory_sha256": inventory_sha256,
        "inventory_canonicalization": "SORT_RELATIVE_PATH_JSON_SORT_KEYS_COMPACT_UTF8",
        "receipt_sha256_transitively_binds_exact_artifact_array": True,
        "analytic_artifact_files_opened_during_preflight": 0,
        "analytic_artifact_bytes_read_during_preflight": 0,
        "mutation_allowed": False,
    }
    if include_entries:
        binding["selected_artifacts"] = entries
    else:
        binding["first_relative_path"] = entries[0]["relative_path"]
        binding["last_relative_path"] = entries[-1]["relative_path"]
    return binding


def _require_exact_predecessor_semantics(
    current: dict[str, Any],
    sections: dict[str, Any],
    tolerance: dict[str, Any],
    final_panel: dict[str, Any],
    partitions: dict[str, Any],
) -> None:
    if current["active_routing"]["active_phase"] != "E":
        raise G1InputPreflightBuildError("active phase is not E")
    phase_e = next(
        phase for phase in current["active_routing"]["phases"] if phase["phase_id"] == "E"
    )
    if phase_e["state"] != "READY_NOT_RUN" or phase_e["complete"] is not False:
        raise G1InputPreflightBuildError("E is not READY_NOT_RUN")
    lifecycle = current["lifecycle"]
    required_lifecycle = {
        "pass_f": True,
        "pass_g0": True,
        "empirical_g1_status": "NOT_RUN",
        "confirmatory_or_holdout_rows_read": 0,
        "model_fitted": False,
        "g4_open": False,
        "holdout_seal_opened": False,
    }
    for key, value in required_lifecycle.items():
        if lifecycle.get(key) != value:
            raise G1InputPreflightBuildError(f"unsafe current lifecycle value: {key}")
    if sections["completion_accounting"]["frozen_section_count"] != 13:
        raise G1InputPreflightBuildError("thirteen scientific sections are not frozen")
    if sections["section_order"] != list(sections["sections"]):
        raise G1InputPreflightBuildError("scientific section order/key mismatch")
    choice = tolerance["precommitment_choices"]["g1_replication_tolerance"]
    if choice["interval_rule"]["lower_noninferiority_margin_cohort_years"] != -1.0:
        raise G1InputPreflightBuildError("G1 replication margin drift")
    if choice["interval_rule"]["interval_boundary_counts_as_pass"] is not False:
        raise G1InputPreflightBuildError("G1 replication boundary drift")

    g1_slots = [
        slot
        for slot in final_panel["slot_support"]
        if slot["slot_id"].startswith("g1_domain_slot_")
    ]
    if [slot["aligned_selected_field_id"] for slot in g1_slots] != [16, 32, 11]:
        raise G1InputPreflightBuildError("G1 field slate drift")
    if sum(slot["aligned_candidate_episode_count"] for slot in g1_slots) != 28:
        raise G1InputPreflightBuildError("G1 episode count drift")
    if sum(slot["aligned_subfield_family_count"] for slot in g1_slots) != 13:
        raise G1InputPreflightBuildError("G1 family count drift")
    g1_partition = next(
        row
        for row in partitions["partition_assignment"]
        if row["partition_id"] == "g1_empirical_replication"
    )
    if g1_partition != {
        "assigned_episode_count": 28,
        "assigned_family_count": 13,
        "entry_years": "2013_2016",
        "partition_id": "g1_empirical_replication",
        "sealed": False,
        "status": "ASSIGNED_FROM_VERIFIED_FINAL_PANEL",
    }:
        raise G1InputPreflightBuildError("G1 partition assignment drift")
    if partitions["atomicity_accounting"]["g1_g4_cross_partition_families"] != 0:
        raise G1InputPreflightBuildError("G1/G4 family disjointness drift")


def build_contract() -> dict[str, Any]:
    current = _load_repository(CURRENT_STATE)
    _load_repository(CURRENT_STATE_SUMMARY)
    sections = _load_repository(SECTIONS)
    _load_repository(SECTIONS_SUMMARY)
    tolerance = _load_repository(TOLERANCE)
    _load_repository(TOLERANCE_SUMMARY)
    final_panel = _load_repository(FINAL_PANEL_SUMMARY)
    partitions = _load_repository(PARTITION_SUMMARY)
    _require_exact_predecessor_semantics(
        current, sections, tolerance, final_panel, partitions
    )

    work_receipt = _load_operator_receipt(WORK_RECEIPT)
    history_receipt = _load_operator_receipt(HISTORY_RECEIPT)
    final_receipt = _load_operator_receipt(FINAL_PANEL_RECEIPT)
    partition_receipt = _load_operator_receipt(PARTITION_RECEIPT)

    work_entries, work_inventory = _select_prefix(work_receipt, "work_metadata/")
    history_entries, history_inventory = _select_prefix(history_receipt, "events/")
    final_entries, final_inventory = _select_paths(
        final_receipt,
        [
            "aligned_final_panel_episode_conformance.parquet",
            "aligned_final_panel_slot_support.parquet",
        ],
    )
    partition_entries, partition_inventory = _select_paths(
        partition_receipt,
        [
            "atomic_episode_partition_assignment.parquet",
            "atomic_family_partition_assignment.parquet",
            "atomic_partition_summary.parquet",
        ],
    )
    if (len(work_entries), sum(entry["bytes"] for entry in work_entries)) != (
        4096,
        4_120_921_809,
    ):
        raise G1InputPreflightBuildError("normalized work-metadata inventory drift")
    if (len(history_entries), sum(entry["bytes"] for entry in history_entries)) != (
        4096,
        5_006_392_019,
    ):
        raise G1InputPreflightBuildError("author-history inventory drift")

    rules = sections["sections"]
    h3 = rules["outcome_horizons_and_h3_timing"]
    controls = rules["negative_control_decision_gates"]
    support = rules["minimum_support_and_suppression"]
    sampling = rules["risk_set_sampling_and_weights"]
    estimation = rules["estimators_and_uncertainty"]
    multiplicity = rules["hierarchical_multiplicity"]

    return {
        "schema_version": "pivot_feedback_e_g1_execution_input_preflight.v1",
        "label": "pivot_feedback_e_g1_execution_input_preflight_v1",
        "mode": "E_PHASE_METADATA_ONLY_G1_EXECUTION_INPUT_PREFLIGHT_AND_CONTRACT_NO_ANALYTIC_READ",
        "status": "E_G1_EXECUTION_INPUT_PREFLIGHT_COMPLETE_PENDING_INDEPENDENT_VALIDATION_NO_G1_RUN",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "authorization_parent": {
            **_repository_binding(CURRENT_STATE),
            "required_active_phase": "E",
            "required_phase_state": "READY_NOT_RUN",
            "required_pass_f": True,
            "required_pass_g0_compatibility": True,
            "required_empirical_g1_status": "NOT_RUN",
            "required_holdout_seal_opened": False,
        },
        "bound_frozen_contracts": {
            "current_state_independent_closure": _repository_binding(
                CURRENT_STATE_SUMMARY
            ),
            "thirteen_scientific_sections": _repository_binding(SECTIONS),
            "thirteen_scientific_sections_verification": _repository_binding(
                SECTIONS_SUMMARY
            ),
            "g1_replication_tolerance": _repository_binding(TOLERANCE),
            "g1_replication_tolerance_verification": _repository_binding(
                TOLERANCE_SUMMARY
            ),
            "aligned_final_panel_conformance": _repository_binding(
                FINAL_PANEL_SUMMARY
            ),
            "atomic_partition_assignment": _repository_binding(PARTITION_SUMMARY),
        },
        "registered_execution_scope": {
            "phase_id": "E",
            "legacy_gate_id": "G1",
            "partition_id": "g1_empirical_replication",
            "entry_year_start": 2013,
            "entry_year_end": 2016,
            "selected_field_ids_in_registered_slot_order": [16, 32, 11],
            "selected_field_count": 3,
            "assigned_episode_count": 28,
            "assigned_family_count": 13,
            "analysis_unit": "CURRENT_SNAPSHOT_OPENALEX_AUTHOR_TOKEN_X_PRIMARY_TOPIC_X_DECISION_YEAR",
            "episode_key_columns": ["field_id", "topic_id", "decision_year"],
            "response_cohort_offsets": [0, 1, 2],
            "partition_predicate_must_be_applied_before_label_or_estimator_logic": True,
            "field_allowlist_must_be_applied_before_label_or_estimator_logic": True,
            "selected_field_replacement": "FORBIDDEN",
            "selected_episode_reselection": "FORBIDDEN",
            "outcome_based_scope_change": "FORBIDDEN",
            "g4_field_id": 27,
            "g4_field_allowed": False,
            "g4_entry_years_allowed": False,
            "g4_private_artifact_root_or_manifest_is_execution_input": False,
            "g4_seal_open_required_or_allowed": False,
        },
        "exact_non_holdout_inputs": {
            "normalized_work_metadata": _operator_binding(
                root=WORK_ROOT,
                receipt_path=WORK_RECEIPT,
                receipt=work_receipt,
                selector={
                    "kind": "RELATIVE_PATH_PREFIX",
                    "value": "work_metadata/",
                },
                entries=work_entries,
                inventory_sha256=work_inventory,
                include_entries=False,
            ),
            "author_history_events": _operator_binding(
                root=HISTORY_ROOT,
                receipt_path=HISTORY_RECEIPT,
                receipt=history_receipt,
                selector={"kind": "RELATIVE_PATH_PREFIX", "value": "events/"},
                entries=history_entries,
                inventory_sha256=history_inventory,
                include_entries=False,
            ),
            "aligned_final_panel_metadata": _operator_binding(
                root=FINAL_PANEL_ROOT,
                receipt_path=FINAL_PANEL_RECEIPT,
                receipt=final_receipt,
                selector={
                    "kind": "EXACT_RELATIVE_PATH_LIST",
                    "value": [entry["relative_path"] for entry in final_entries],
                },
                entries=final_entries,
                inventory_sha256=final_inventory,
                include_entries=True,
            ),
            "atomic_partition_metadata": _operator_binding(
                root=PARTITION_ROOT,
                receipt_path=PARTITION_RECEIPT,
                receipt=partition_receipt,
                selector={
                    "kind": "EXACT_RELATIVE_PATH_LIST",
                    "value": [entry["relative_path"] for entry in partition_entries],
                },
                entries=partition_entries,
                inventory_sha256=partition_inventory,
                include_entries=True,
            ),
            "private_g4_holdout": {
                "included": False,
                "root_read": False,
                "manifest_read": False,
                "artifact_path_enumerated": False,
                "artifact_byte_read": False,
                "seal_opened": False,
            },
        },
        "label_materialization_contract": {
            "source_of_truth": "BOUND_THIRTEEN_SCIENTIFIC_SECTIONS_EXACT_HASH",
            "materialization_occurs_in_this_preflight": False,
            "entry_outcome": h3["entry_outcome"],
            "entry_rule": rules["pivot_and_robustness_definitions"]["primary"],
            "persistence_outcome": h3["persistence_outcome"],
            "persistence_horizon_offsets": rules[
                "temporal_availability_and_maturation"
            ]["persistence_label"]["horizon_offsets"],
            "persistence_matured_rule": rules[
                "temporal_availability_and_maturation"
            ]["persistence_label"]["matured_rule"],
            "one_off_entry": h3["one_off_entry"],
            "failure_label_allowed": h3["failure_label_allowed"],
            "alternative_pivot": rules["pivot_and_robustness_definitions"][
                "substantively_different_alternative"
            ],
            "same_decision_year_information_as_feature": rules[
                "temporal_availability_and_maturation"
            ]["same_decision_year_information_as_feature"],
            "outcome_imputation": estimation["missingness"]["outcome_imputation"],
            "label_dependent_support": {
                "cohort_floor": support["cohort_floor"],
                "capacity_cell_floor": support["capacity_cell_floor"],
                "failure_action": support["support_or_collinearity_failure"],
                "may_change_scope_or_threshold_after_result": False,
            },
            "risk_set_sampling_and_weights": sampling,
        },
        "negative_control_contract": {
            "source_of_truth": "BOUND_THIRTEEN_SCIENTIFIC_SECTIONS_EXACT_HASH",
            "materialization_or_decision_occurs_in_this_preflight": False,
            "probability_difference_equivalence": controls[
                "probability_difference_equivalence"
            ],
            "true_minus_placebo": controls["true_minus_placebo"],
            "deterministic_constructions": controls["deterministic_constructions"],
            "required_control_count": 7,
            "any_required_control_failure": controls[
                "any_required_control_failure"
            ],
        },
        "estimand_and_decision_contract": {
            "source_of_truth": "BOUND_THIRTEEN_SCIENTIFIC_SECTIONS_AND_G1_TOLERANCE_EXACT_HASHES",
            "estimation_occurs_in_this_preflight": False,
            "primary_probability_model": estimation["primary_probability_model"],
            "estimands": estimation["estimands"],
            "h3_adjustment": estimation["h3_adjustment"],
            "uncertainty": estimation["uncertainty"],
            "h3_primary_and_replication": h3["h3"],
            "replication_noninferiority": h3["replication_noninferiority"],
            "hierarchical_multiplicity": multiplicity,
            "predictive_or_nabm_model_selection": "NOT_AUTHORIZED_IN_E",
        },
        "execution_output_contract": {
            "output_root_basename_pattern": "pivot-feedback-e-g1-execution-v1-YYYYMMDD",
            "output_directory_must_be_new": True,
            "output_directory_mode": "0700",
            "output_file_mode": "0600",
            "claim_bearing_execution_tool_sha256_required": True,
            "this_preflight_sha256_required": True,
            "receipt_written_last": True,
            "independent_verification_required": True,
            "raw_author_or_work_ids_allowed_only_in_operator_private_row_artifact": True,
            "repository_summary_must_not_contain_raw_author_or_work_ids": True,
            "artifacts": [
                {
                    "relative_path": "g1_risk_set_and_labels.parquet",
                    "schema_version": "pivot_feedback_e_g1_risk_set_and_labels.v1",
                    "confidentiality": "OPERATOR_PRIVATE_NON_HOLDOUT_ROW_LEVEL",
                    "exact_columns": [
                        "analysis_partition_id",
                        "primary_field_id",
                        "certified_episode_id",
                        "topic_id",
                        "decision_year",
                        "opportunity_id",
                        "current_snapshot_author_token",
                        "entry_outcome",
                        "persistence_outcome",
                        "one_off_entry",
                        "alternative_pivot_entry",
                        "peer_active",
                        "peer_visible_success",
                        "capacity_bridge",
                        "entry_mode",
                        "sampling_role",
                        "inclusion_probability",
                        "raw_sampling_weight",
                        "final_analytic_weight",
                        "exclusion_reason",
                    ],
                },
                {
                    "relative_path": "g1_episode_estimates.parquet",
                    "schema_version": "pivot_feedback_e_g1_episode_estimates.v1",
                    "confidentiality": "AGGREGATE_SUPPRESSION_REQUIRED",
                    "exact_columns": [
                        "hypothesis_id",
                        "primary_field_id",
                        "certified_episode_id",
                        "pivot_definition",
                        "point_estimate",
                        "interval_lower",
                        "interval_upper",
                        "effective_sample_size",
                        "support_status",
                        "estimator_status",
                    ],
                },
                {
                    "relative_path": "g1_field_replication_decisions.parquet",
                    "schema_version": "pivot_feedback_e_g1_field_replication_decisions.v1",
                    "confidentiality": "AGGREGATE_SUPPRESSION_REQUIRED",
                    "exact_columns": [
                        "primary_field_id",
                        "independent_episode_count",
                        "field_point_estimate_cohort_years",
                        "field_minus_other_fields_one_sided_95_lower_bound_cohort_years",
                        "leave_one_field_out_point_estimate_cohort_years",
                        "alternative_pivot_field_point_estimate_cohort_years",
                        "direction_pass",
                        "replication_noninferiority_pass",
                        "failure_reasons",
                    ],
                },
                {
                    "relative_path": "g1_negative_control_decisions.parquet",
                    "schema_version": "pivot_feedback_e_g1_negative_control_decisions.v1",
                    "confidentiality": "AGGREGATE_SUPPRESSION_REQUIRED",
                    "exact_columns": [
                        "control_id",
                        "control_family",
                        "point_estimate",
                        "interval_lower",
                        "interval_upper",
                        "registered_margin_lower",
                        "registered_margin_upper",
                        "decision",
                        "failure_reason",
                    ],
                },
                {
                    "relative_path": "g1_hypothesis_decisions.parquet",
                    "schema_version": "pivot_feedback_e_g1_hypothesis_decisions.v1",
                    "confidentiality": "AGGREGATE_SUPPRESSION_REQUIRED",
                    "exact_columns": [
                        "hypothesis_id",
                        "stage",
                        "point_estimate",
                        "interval_lower",
                        "interval_upper",
                        "raw_p_value",
                        "holm_adjusted_p_value",
                        "prerequisites_pass",
                        "registered_decision",
                        "failure_reason",
                    ],
                },
                {
                    "relative_path": "g1_execution_report.json",
                    "schema_version": "pivot_feedback_e_g1_execution_report.v1",
                    "confidentiality": "REPOSITORY_SAFE_AGGREGATE_AFTER_VERIFICATION",
                    "exact_top_level_fields": [
                        "schema_version",
                        "status",
                        "input_bindings",
                        "access_accounting",
                        "scope_accounting",
                        "support_and_weight_diagnostics",
                        "h3_primary_and_replication",
                        "negative_control_decisions",
                        "field_and_alternative_pivot_stability",
                        "hierarchical_hypothesis_decisions",
                        "claim_boundary",
                    ],
                },
                {
                    "relative_path": "g1_execution.outputs.json",
                    "schema_version": "pivot_feedback_e_g1_execution_outputs.v1",
                    "confidentiality": "RECEIPT_NO_RAW_IDS",
                    "exact_top_level_fields": [
                        "schema_version",
                        "status",
                        "artifact_count",
                        "artifacts",
                        "artifact_tree_sha256",
                        "input_preflight_sha256",
                        "execution_tool_sha256",
                        "confirmatory_rows_read",
                        "holdout_rows_read",
                        "model_selection_scores_read",
                        "g4_open",
                        "receipt_written_last",
                    ],
                },
            ],
            "allowed_terminal_statuses": [
                "PASS_E_G1_BOUNDED_OBSERVATIONAL_SIGNATURE",
                "FAIL_E_G1_REGISTERED_DECISION_NO_ROUTE_PROMOTION",
                "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION",
            ],
        },
        "preflight_access_accounting": {
            "repository_yaml_documents_read": 3,
            "repository_json_documents_read": 5,
            "operator_metadata_receipts_read": 4,
            "operator_roots_checked": 4,
            "analytic_parquet_or_row_level_files_opened": 0,
            "analytic_artifact_bytes_read": 0,
            "confirmatory_rows_read": 0,
            "holdout_rows_read": 0,
            "entry_or_persistence_labels_materialized": 0,
            "negative_control_values_materialized": 0,
            "estimators_run": 0,
            "model_selection_scores_read": 0,
            "predictive_or_nabm_models_fitted": 0,
            "seal_open_events": 0,
            "g4_open": False,
        },
        "lifecycle": {
            "input_preflight_contract_authored": True,
            "independent_preflight_verification_complete": False,
            "g1_execution_run": False,
            "empirical_g1_status": "NOT_RUN",
            "pass_e": False,
            "pass_g1": False,
            "model_selection_started": False,
            "model_fitted": False,
            "holdout_seal_opened": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": {
            "metadata_only_preflight": "COMPLETE_PENDING_INDEPENDENT_VALIDATION",
            "exact_g1_execution_successor": "AUTHORIZED_ONLY_AFTER_INDEPENDENT_PREFLIGHT_PASS",
            "confirmatory_analytic_access": "AUTHORIZED_ONLY_DURING_SEPARATE_EXECUTION_SUCCESSOR_FOR_BOUND_G1_SCOPE",
            "scope_or_threshold_change_after_confirmatory_access": "NOT_AUTHORIZED",
            "g4_artifact_access": "NOT_AUTHORIZED",
            "g4_seal_open": "NOT_AUTHORIZED",
            "model_selection_score_read": "NOT_AUTHORIZED",
            "predictive_or_nabm_model_fitting": "NOT_AUTHORIZED",
            "g2_or_g3_choice_resolution": "NOT_AUTHORIZED",
            "causal_claim": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "validation_contract": {
            "verifier_relative_path": "scripts/verify_pivot_feedback_e_g1_execution_input_preflight_v1.py",
            "summary_json_relative_path": "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.json",
            "summary_markdown_relative_path": "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.md",
            "repository_and_operator_metadata_receipts_only": True,
            "analytic_artifact_files_may_be_opened": False,
            "confirmatory_or_holdout_rows_may_be_read": False,
            "labels_may_be_materialized": False,
            "estimators_may_be_run": False,
            "seal_may_be_opened": False,
        },
        "claim_boundary": (
            "This metadata-only preflight binds the exact frozen 2013-2016, "
            "three-field, 28-episode, 13-family G1 scope; receipt-bound non-holdout "
            "inputs; label, sampling, negative-control, estimator, uncertainty, "
            "multiplicity, and output-schema contracts. It opens no analytic "
            "artifact, reads no confirmatory or holdout row, materializes no label "
            "or control, runs no estimator or G1 decision, reads no model-selection "
            "score, fits no predictive or NABM model, includes no private G4 "
            "artifact, opens no seal, and establishes no association, null, "
            "prediction, transport, causality, or scientific effect."
        ),
    }


def _write_yaml_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise G1InputPreflightBuildError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the metadata-only E/G1 execution input preflight contract."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = build_contract()
    _write_yaml_create_only(args.output, contract)
    print(
        json.dumps(
            {
                "schema_version": contract["schema_version"],
                "status": contract["status"],
                "partition_id": contract["registered_execution_scope"][
                    "partition_id"
                ],
                "field_count": contract["registered_execution_scope"][
                    "selected_field_count"
                ],
                "episode_count": contract["registered_execution_scope"][
                    "assigned_episode_count"
                ],
                "confirmatory_rows_read": contract["preflight_access_accounting"][
                    "confirmatory_rows_read"
                ],
                "holdout_rows_read": contract["preflight_access_accounting"][
                    "holdout_rows_read"
                ],
                "g1_status": contract["lifecycle"]["empirical_g1_status"],
                "g4_open": contract["lifecycle"]["g4_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
