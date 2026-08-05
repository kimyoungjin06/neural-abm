#!/usr/bin/env python3
"""Independently verify the metadata-only E/G1 input preflight contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORITY = (
    ROOT
    / "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v1.yaml"
)
BUILDER = ROOT / "scripts/build_pivot_feedback_e_g1_execution_input_preflight_v1.py"
VERIFIER = Path(__file__).resolve()

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

REPOSITORY_EXPECTED = {
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

OPERATOR_EXPECTED = {
    WORK_RECEIPT: (
        "91e3f689e574d524eabbe7daa58f5e678da37f851ba9b7af0561dab57b70846e",
        "pivot_feedback_cutoff_primitives_outputs.v1",
        "PASS_NORMALIZED_CURRENT_SNAPSHOT_CUTOFF_PRIMITIVES_WITH_RETROSPECTIVE_TOPIC_AND_CITATION_PROXY_LIMITATIONS",
        "work_metadata/",
        4096,
        4_120_921_809,
        "68d57f5425d56a5e008694fdcb6a36521ba977a3fbabb5935c3ad023abcaa5e9",
    ),
    HISTORY_RECEIPT: (
        "7b3613eaac6900f376a999a551f15e20dd184de2b4dc6758a60d93f3eedb51a8",
        "openalex_g1_work_side_author_history_outputs.v3",
        "PASS_CURRENT_SNAPSHOT_WORK_SIDE_AUTHOR_TOKEN_LINKED_RETROSPECTIVE_HISTORY_ACCOUNTING_AND_SENSITIVITY_COMPONENT_AUTHOR_ENTITY_AND_PAST_OBSERVABILITY_UNVERIFIED",
        "events/",
        4096,
        5_006_392_019,
        "59cfe5d5b235f26d4a2b8ad78f5679c15d341ac0778f92d3beb30929cb17aa4c",
    ),
}

SMALL_OPERATOR_EXPECTED = {
    FINAL_PANEL_RECEIPT: (
        "222888f538652593c1a26fddf6eb25dca37a2edecd71afe744fa63ac4e695505",
        "pivot_feedback_f_aligned_final_panel_conformance_outputs.v1",
        "PASS_OUTCOME_BLIND_ALIGNED_FINAL_PANEL_LEAVEOUT_AND_SUPPORT_CONFORMANCE_ATOMIC_PARTITION_AUTHORIZED_NO_F_PASS",
        [
            "aligned_final_panel_episode_conformance.parquet",
            "aligned_final_panel_slot_support.parquet",
        ],
    ),
    PARTITION_RECEIPT: (
        "0ddd5ec51d924167b332a2f756b5310c25454e2ad44db5d86859276814c4f2b4",
        "pivot_feedback_f_atomic_partition_assignment_outputs.v1",
        "PASS_OUTCOME_BLIND_ATOMIC_EPISODE_FAMILY_PARTITION_ASSIGNMENT_PRIVATE_HOLDOUT_MATERIALIZATION_AUTHORIZED_NO_F_PASS",
        [
            "atomic_episode_partition_assignment.parquet",
            "atomic_family_partition_assignment.parquet",
            "atomic_partition_summary.parquet",
        ],
    ),
}

OUTPUT_SCHEMAS = {
    "g1_risk_set_and_labels.parquet": (
        "pivot_feedback_e_g1_risk_set_and_labels.v1",
        [
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
    ),
    "g1_episode_estimates.parquet": (
        "pivot_feedback_e_g1_episode_estimates.v1",
        [
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
    ),
    "g1_field_replication_decisions.parquet": (
        "pivot_feedback_e_g1_field_replication_decisions.v1",
        [
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
    ),
    "g1_negative_control_decisions.parquet": (
        "pivot_feedback_e_g1_negative_control_decisions.v1",
        [
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
    ),
    "g1_hypothesis_decisions.parquet": (
        "pivot_feedback_e_g1_hypothesis_decisions.v1",
        [
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
    ),
}


class G1InputPreflightVerificationError(RuntimeError):
    """Raised when independent replay finds drift or an unsafe boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G1InputPreflightVerificationError(message)


def _load_repository(path: Path) -> dict[str, Any]:
    expected_sha, expected_schema, expected_status = REPOSITORY_EXPECTED[path]
    _require(path.is_file(), f"missing repository document: {path}")
    _require(_sha256(path) == expected_sha, f"repository hash drift: {path}")
    if path.suffix == ".yaml":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == expected_schema, f"schema drift: {path}")
    _require(payload.get("status") == expected_status, f"status drift: {path}")
    return payload


def _load_operator(path: Path, expected: tuple[Any, ...]) -> dict[str, Any]:
    expected_sha, expected_schema, expected_status = expected[:3]
    _require(path.parent.is_dir(), f"missing operator root: {path.parent}")
    _require(path.is_file(), f"missing operator receipt: {path}")
    _require(_sha256(path) == expected_sha, f"operator receipt hash drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == expected_schema, f"operator schema drift: {path}")
    _require(payload.get("status") == expected_status, f"operator status drift: {path}")
    artifacts = payload.get("artifacts")
    _require(isinstance(artifacts, list) and bool(artifacts), f"empty artifact list: {path}")
    names = [entry.get("relative_path") for entry in artifacts]
    _require(len(names) == len(set(names)), f"duplicate artifact paths: {path}")
    return payload


def _canonical(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    ordered = sorted(entries, key=lambda entry: entry["relative_path"])
    encoded = json.dumps(
        ordered,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ordered, hashlib.sha256(encoded).hexdigest()


def _verify_repository_binding(binding: dict[str, Any], path: Path) -> None:
    expected_sha, expected_schema, expected_status = REPOSITORY_EXPECTED[path]
    _require(binding["relative_path"] == _relative(path), f"binding path drift: {path}")
    _require(binding["sha256"] == expected_sha, f"binding hash drift: {path}")
    _require(
        binding["required_schema_version"] == expected_schema,
        f"binding schema drift: {path}",
    )
    _require(
        binding["required_status"] == expected_status,
        f"binding status drift: {path}",
    )
    _require(binding["mutation_allowed"] is False, f"mutable binding: {path}")


def _verify_operator_binding(
    binding: dict[str, Any],
    *,
    root: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
    entries: list[dict[str, Any]],
    inventory_sha256: str,
) -> None:
    expected = (
        OPERATOR_EXPECTED.get(receipt_path)
        or SMALL_OPERATOR_EXPECTED[receipt_path]
    )
    expected_sha, expected_schema, expected_status = expected[:3]
    _require(binding["root"] == str(root), f"operator root drift: {root}")
    _require(binding["root_basename"] == root.name, f"root basename drift: {root}")
    _require(binding["receipt_basename"] == receipt_path.name, "receipt basename drift")
    _require(binding["receipt_sha256"] == expected_sha, "operator receipt binding drift")
    _require(binding["receipt_schema_version"] == expected_schema, "operator schema binding drift")
    _require(binding["required_status"] == expected_status, "operator status binding drift")
    _require(
        binding["receipt_artifact_tree_sha256"] == receipt["artifact_tree_sha256"],
        "receipt tree binding drift",
    )
    _require(binding["selected_artifact_count"] == len(entries), "artifact count drift")
    _require(
        binding["selected_artifact_bytes"] == sum(int(entry["bytes"]) for entry in entries),
        "artifact byte count drift",
    )
    _require(
        binding["selected_artifact_inventory_sha256"] == inventory_sha256,
        "artifact inventory fingerprint drift",
    )
    _require(
        binding["inventory_canonicalization"]
        == "SORT_RELATIVE_PATH_JSON_SORT_KEYS_COMPACT_UTF8",
        "inventory canonicalization drift",
    )
    _require(
        binding["receipt_sha256_transitively_binds_exact_artifact_array"] is True,
        "receipt no longer binds exact inventory",
    )
    _require(binding["analytic_artifact_files_opened_during_preflight"] == 0, "analytic file opened")
    _require(binding["analytic_artifact_bytes_read_during_preflight"] == 0, "analytic bytes read")
    _require(binding["mutation_allowed"] is False, "operator input is mutable")


def _verify_output_contract(output: dict[str, Any]) -> None:
    _require(output["output_directory_must_be_new"] is True, "output may overwrite")
    _require(output["output_directory_mode"] == "0700", "unsafe output directory mode")
    _require(output["output_file_mode"] == "0600", "unsafe output file mode")
    _require(output["receipt_written_last"] is True, "receipt is not last")
    _require(output["independent_verification_required"] is True, "verification optional")
    artifacts = output["artifacts"]
    _require(len(artifacts) == 7, "output artifact count drift")
    by_path = {artifact["relative_path"]: artifact for artifact in artifacts}
    _require(len(by_path) == 7, "duplicate output artifact path")
    for path, (schema, columns) in OUTPUT_SCHEMAS.items():
        _require(path in by_path, f"missing output schema: {path}")
        _require(by_path[path]["schema_version"] == schema, f"schema drift: {path}")
        _require(by_path[path]["exact_columns"] == columns, f"column drift: {path}")
    _require(
        by_path["g1_execution_report.json"]["schema_version"]
        == "pivot_feedback_e_g1_execution_report.v1",
        "report schema drift",
    )
    _require(
        by_path["g1_execution.outputs.json"]["schema_version"]
        == "pivot_feedback_e_g1_execution_outputs.v1",
        "receipt schema drift",
    )
    _require(
        output["allowed_terminal_statuses"]
        == [
            "PASS_E_G1_BOUNDED_OBSERVATIONAL_SIGNATURE",
            "FAIL_E_G1_REGISTERED_DECISION_NO_ROUTE_PROMOTION",
            "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION",
        ],
        "terminal status set drift",
    )


def verify(authority_path: Path) -> dict[str, Any]:
    _require(authority_path.is_file(), f"missing authority: {authority_path}")
    authority = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    _require(
        authority.get("schema_version")
        == "pivot_feedback_e_g1_execution_input_preflight.v1",
        "preflight schema drift",
    )
    _require(
        authority.get("status")
        == "E_G1_EXECUTION_INPUT_PREFLIGHT_COMPLETE_PENDING_INDEPENDENT_VALIDATION_NO_G1_RUN",
        "preflight status drift",
    )

    current = _load_repository(CURRENT_STATE)
    current_summary = _load_repository(CURRENT_STATE_SUMMARY)
    sections = _load_repository(SECTIONS)
    sections_summary = _load_repository(SECTIONS_SUMMARY)
    tolerance = _load_repository(TOLERANCE)
    tolerance_summary = _load_repository(TOLERANCE_SUMMARY)
    final_panel = _load_repository(FINAL_PANEL_SUMMARY)
    partitions = _load_repository(PARTITION_SUMMARY)

    parent = authority["authorization_parent"]
    _verify_repository_binding(parent, CURRENT_STATE)
    _require(parent["required_active_phase"] == "E", "parent active phase binding drift")
    _require(parent["required_phase_state"] == "READY_NOT_RUN", "parent E state binding drift")
    _require(parent["required_pass_f"] is True, "parent F pass binding drift")
    _require(parent["required_pass_g0_compatibility"] is True, "parent G0 binding drift")
    _require(parent["required_empirical_g1_status"] == "NOT_RUN", "parent G1 status drift")
    _require(parent["required_holdout_seal_opened"] is False, "parent seal binding drift")

    phase_e = next(
        phase for phase in current["active_routing"]["phases"] if phase["phase_id"] == "E"
    )
    _require(current["active_routing"]["active_phase"] == "E", "current active phase drift")
    _require(phase_e["state"] == "READY_NOT_RUN" and phase_e["complete"] is False, "unsafe E state")
    _require(current["lifecycle"]["pass_f"] is True, "F is not passed")
    _require(current["lifecycle"]["empirical_g1_status"] == "NOT_RUN", "G1 already ran")
    _require(current["lifecycle"]["holdout_seal_opened"] is False, "holdout seal is open")
    _require(current_summary["fail_closed_replay"]["seal_opened"] is False, "closure seal replay unsafe")

    bindings = authority["bound_frozen_contracts"]
    binding_paths = {
        "current_state_independent_closure": CURRENT_STATE_SUMMARY,
        "thirteen_scientific_sections": SECTIONS,
        "thirteen_scientific_sections_verification": SECTIONS_SUMMARY,
        "g1_replication_tolerance": TOLERANCE,
        "g1_replication_tolerance_verification": TOLERANCE_SUMMARY,
        "aligned_final_panel_conformance": FINAL_PANEL_SUMMARY,
        "atomic_partition_assignment": PARTITION_SUMMARY,
    }
    _require(set(bindings) == set(binding_paths), "frozen contract binding set drift")
    for name, path in binding_paths.items():
        _verify_repository_binding(bindings[name], path)
    _require(sections_summary["section_inventory_replay"]["frozen_section_count"] == 13, "section replay drift")
    _require(tolerance_summary["policy_replay"]["lower_margin_cohort_years"] == -1.0, "tolerance replay drift")

    scope = authority["registered_execution_scope"]
    _require(scope["partition_id"] == "g1_empirical_replication", "partition scope drift")
    _require([scope["entry_year_start"], scope["entry_year_end"]] == [2013, 2016], "year scope drift")
    _require(scope["selected_field_ids_in_registered_slot_order"] == [16, 32, 11], "field scope drift")
    _require(scope["assigned_episode_count"] == 28, "episode count drift")
    _require(scope["assigned_family_count"] == 13, "family count drift")
    _require(scope["partition_predicate_must_be_applied_before_label_or_estimator_logic"] is True, "late partition filter")
    _require(scope["field_allowlist_must_be_applied_before_label_or_estimator_logic"] is True, "late field filter")
    _require(scope["selected_field_replacement"] == "FORBIDDEN", "field replacement allowed")
    _require(scope["selected_episode_reselection"] == "FORBIDDEN", "episode reselection allowed")
    _require(scope["g4_field_id"] == 27 and scope["g4_field_allowed"] is False, "G4 field included")
    _require(scope["g4_entry_years_allowed"] is False, "G4 years included")
    _require(scope["g4_private_artifact_root_or_manifest_is_execution_input"] is False, "G4 private input included")
    _require(scope["g4_seal_open_required_or_allowed"] is False, "G4 seal open allowed")

    g1_slots = [
        slot
        for slot in final_panel["slot_support"]
        if slot["slot_id"].startswith("g1_domain_slot_")
    ]
    _require([slot["aligned_selected_field_id"] for slot in g1_slots] == [16, 32, 11], "final-panel field replay drift")
    _require(sum(slot["aligned_candidate_episode_count"] for slot in g1_slots) == 28, "final-panel episode replay drift")
    _require(partitions["atomicity_accounting"]["g1_g4_cross_partition_families"] == 0, "G1/G4 family overlap")

    work_receipt = _load_operator(WORK_RECEIPT, OPERATOR_EXPECTED[WORK_RECEIPT])
    history_receipt = _load_operator(HISTORY_RECEIPT, OPERATOR_EXPECTED[HISTORY_RECEIPT])
    final_receipt = _load_operator(FINAL_PANEL_RECEIPT, SMALL_OPERATOR_EXPECTED[FINAL_PANEL_RECEIPT])
    partition_receipt = _load_operator(PARTITION_RECEIPT, SMALL_OPERATOR_EXPECTED[PARTITION_RECEIPT])

    inputs = authority["exact_non_holdout_inputs"]
    input_specs = [
        ("normalized_work_metadata", WORK_ROOT, WORK_RECEIPT, work_receipt),
        ("author_history_events", HISTORY_ROOT, HISTORY_RECEIPT, history_receipt),
    ]
    replay: dict[str, Any] = {}
    for name, root, receipt_path, receipt in input_specs:
        _, _, _, prefix, count, byte_count, expected_inventory = OPERATOR_EXPECTED[receipt_path]
        entries, inventory = _canonical(
            [entry for entry in receipt["artifacts"] if entry["relative_path"].startswith(prefix)]
        )
        _require(len(entries) == count, f"{name} count replay drift")
        _require(sum(int(entry["bytes"]) for entry in entries) == byte_count, f"{name} byte replay drift")
        _require(inventory == expected_inventory, f"{name} inventory replay drift")
        _verify_operator_binding(
            inputs[name],
            root=root,
            receipt_path=receipt_path,
            receipt=receipt,
            entries=entries,
            inventory_sha256=inventory,
        )
        _require(inputs[name]["selector"] == {"kind": "RELATIVE_PATH_PREFIX", "value": prefix}, f"{name} selector drift")
        _require(inputs[name]["first_relative_path"] == entries[0]["relative_path"], f"{name} first path drift")
        _require(inputs[name]["last_relative_path"] == entries[-1]["relative_path"], f"{name} last path drift")
        replay[name] = {
            "artifact_count": len(entries),
            "artifact_bytes": byte_count,
            "inventory_sha256": inventory,
        }

    small_specs = [
        ("aligned_final_panel_metadata", FINAL_PANEL_ROOT, FINAL_PANEL_RECEIPT, final_receipt),
        ("atomic_partition_metadata", PARTITION_ROOT, PARTITION_RECEIPT, partition_receipt),
    ]
    for name, root, receipt_path, receipt in small_specs:
        required_paths = SMALL_OPERATOR_EXPECTED[receipt_path][3]
        by_path = {entry["relative_path"]: entry for entry in receipt["artifacts"]}
        entries, inventory = _canonical([by_path[path] for path in required_paths])
        _verify_operator_binding(
            inputs[name],
            root=root,
            receipt_path=receipt_path,
            receipt=receipt,
            entries=entries,
            inventory_sha256=inventory,
        )
        _require(inputs[name]["selector"] == {"kind": "EXACT_RELATIVE_PATH_LIST", "value": [entry["relative_path"] for entry in entries]}, f"{name} selector drift")
        _require(inputs[name]["selected_artifacts"] == entries, f"{name} exact artifacts drift")
        replay[name] = {
            "artifact_count": len(entries),
            "artifact_bytes": sum(int(entry["bytes"]) for entry in entries),
            "inventory_sha256": inventory,
        }

    private = inputs["private_g4_holdout"]
    _require(set(private.values()) == {False}, "private G4 input or access present")

    rules = sections["sections"]
    labels = authority["label_materialization_contract"]
    _require(labels["materialization_occurs_in_this_preflight"] is False, "labels materialized")
    _require(labels["entry_rule"] == rules["pivot_and_robustness_definitions"]["primary"], "entry rule drift")
    _require(labels["persistence_outcome"] == rules["outcome_horizons_and_h3_timing"]["persistence_outcome"], "persistence rule drift")
    _require(labels["alternative_pivot"] == rules["pivot_and_robustness_definitions"]["substantively_different_alternative"], "alternative pivot drift")
    _require(labels["risk_set_sampling_and_weights"] == rules["risk_set_sampling_and_weights"], "sampling rule drift")
    _require(labels["outcome_imputation"] == "NONE", "outcome imputation enabled")

    controls = authority["negative_control_contract"]
    frozen_controls = rules["negative_control_decision_gates"]
    _require(controls["materialization_or_decision_occurs_in_this_preflight"] is False, "controls materialized")
    _require(controls["probability_difference_equivalence"] == frozen_controls["probability_difference_equivalence"], "equivalence controls drift")
    _require(controls["true_minus_placebo"] == frozen_controls["true_minus_placebo"], "placebo controls drift")
    _require(controls["deterministic_constructions"] == frozen_controls["deterministic_constructions"], "control construction drift")
    control_count = len(controls["probability_difference_equivalence"]["control_ids"]) + len(controls["true_minus_placebo"]["control_ids"])
    _require(control_count == controls["required_control_count"] == 7, "negative control count drift")

    estimand = authority["estimand_and_decision_contract"]
    frozen_estimation = rules["estimators_and_uncertainty"]
    _require(estimand["estimation_occurs_in_this_preflight"] is False, "estimator ran")
    _require(estimand["primary_probability_model"] == frozen_estimation["primary_probability_model"], "estimator model drift")
    _require(estimand["estimands"] == frozen_estimation["estimands"], "estimand drift")
    _require(estimand["uncertainty"] == frozen_estimation["uncertainty"], "uncertainty drift")
    _require(estimand["hierarchical_multiplicity"] == rules["hierarchical_multiplicity"], "multiplicity drift")
    frozen_tolerance = tolerance["precommitment_choices"]["g1_replication_tolerance"]
    _require(frozen_tolerance["interval_rule"]["lower_noninferiority_margin_cohort_years"] == -1.0, "tolerance value drift")
    _require(frozen_tolerance["interval_rule"]["interval_boundary_counts_as_pass"] is False, "tolerance boundary drift")

    _verify_output_contract(authority["execution_output_contract"])

    expected_access = {
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
    }
    _require(authority["preflight_access_accounting"] == expected_access, "access accounting drift")
    lifecycle = authority["lifecycle"]
    _require(lifecycle["input_preflight_contract_authored"] is True, "preflight not authored")
    _require(lifecycle["independent_preflight_verification_complete"] is False, "authority self-asserts verification")
    _require(lifecycle["g1_execution_run"] is False, "G1 execution already ran")
    _require(lifecycle["empirical_g1_status"] == "NOT_RUN", "empirical G1 status drift")
    _require(lifecycle["pass_e"] is False and lifecycle["pass_g1"] is False, "E/G1 passed in preflight")
    _require(lifecycle["holdout_seal_opened"] is False and lifecycle["g4_open"] is False, "holdout opened")
    authorization = authority["authorization"]
    _require(
        authorization["exact_g1_execution_successor"]
        == "AUTHORIZED_ONLY_AFTER_INDEPENDENT_PREFLIGHT_PASS",
        "unsafe execution authorization",
    )
    _require(authorization["g4_artifact_access"] == "NOT_AUTHORIZED", "G4 access authorized")
    _require(authorization["g4_seal_open"] == "NOT_AUTHORIZED", "G4 seal open authorized")
    _require(authorization["model_selection_score_read"] == "NOT_AUTHORIZED", "model scores authorized")

    return {
        "schema_version": "pivot_feedback_e_g1_execution_input_preflight_summary.v1",
        "status": "PASS_INDEPENDENT_E_G1_EXECUTION_INPUT_PREFLIGHT_EXACT_NON_HOLDOUT_SCOPE_NO_G1_RUN",
        "repository_artifacts": {
            "preflight_contract": {
                "relative_path": _relative(authority_path),
                "sha256": _sha256(authority_path),
            },
            "builder": {
                "relative_path": _relative(BUILDER),
                "sha256": _sha256(BUILDER),
            },
            "independent_verifier": {
                "relative_path": _relative(VERIFIER),
                "sha256": _sha256(VERIFIER),
            },
        },
        "active_authority_replay": {
            "current_state_sha256": REPOSITORY_EXPECTED[CURRENT_STATE][0],
            "active_phase": "E",
            "phase_state": "READY_NOT_RUN",
            "pass_f": True,
            "pass_g0_compatibility": True,
            "empirical_g1_status": "NOT_RUN",
            "holdout_seal_opened": False,
        },
        "registered_scope_replay": {
            "partition_id": "g1_empirical_replication",
            "entry_years": [2013, 2016],
            "field_ids_in_registered_slot_order": [16, 32, 11],
            "field_count": 3,
            "episode_count": 28,
            "family_count": 13,
            "g1_g4_cross_partition_families": 0,
            "g4_field_excluded": True,
            "g4_private_artifacts_included": False,
        },
        "exact_input_inventory_replay": replay,
        "scientific_contract_replay": {
            "frozen_section_count": 13,
            "label_materialization_rule_bound": True,
            "risk_sampling_and_weight_rule_bound": True,
            "negative_control_count": 7,
            "negative_control_rules_bound": True,
            "h3_primary_and_replication_rule_bound": True,
            "replication_margin_cohort_years": -1.0,
            "strict_replication_boundary": True,
            "estimator_and_uncertainty_rule_bound": True,
            "hierarchical_multiplicity_rule_bound": True,
            "output_artifact_schema_count": 7,
        },
        "independent_preflight": {
            "repository_bindings_exact": True,
            "operator_receipt_hashes_exact": True,
            "receipt_artifact_arrays_replayed_from_metadata": True,
            "non_holdout_inventory_fingerprints_exact": True,
            "scope_and_output_contract_exact": True,
            "complete": True,
            "this_is_new_scientific_gate": False,
        },
        "access_accounting": expected_access,
        "fail_closed_state": {
            "g1_execution_run": False,
            "empirical_g1_status": "NOT_RUN",
            "pass_e": False,
            "pass_g1": False,
            "confirmatory_rows_read": 0,
            "holdout_rows_read": 0,
            "labels_materialized": 0,
            "negative_controls_materialized": 0,
            "estimators_run": 0,
            "model_selection_scores_read": 0,
            "predictive_or_nabm_models_fitted": 0,
            "holdout_seal_opened": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": {
            "exact_g1_execution_successor": "AUTHORIZED_NEXT_FROM_INDEPENDENTLY_VERIFIED_PREFLIGHT",
            "confirmatory_analytic_access": "AUTHORIZED_ONLY_DURING_SEPARATE_EXECUTION_SUCCESSOR_FOR_BOUND_G1_SCOPE",
            "scope_or_threshold_change_after_confirmatory_access": "NOT_AUTHORIZED",
            "g4_artifact_access": "NOT_AUTHORIZED",
            "g4_seal_open": "NOT_AUTHORIZED",
            "model_selection_score_read": "NOT_AUTHORIZED",
            "predictive_or_nabm_model_fitting": "NOT_AUTHORIZED",
            "causal_claim": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "The independent metadata replay verifies the exact frozen G1 scope, "
            "receipt-bound non-holdout inputs, label and negative-control rules, "
            "estimand and decision rules, and seven output schemas. No analytic "
            "artifact or row was opened, no label, control, estimate, or G1 result "
            "was produced, no model-selection score was read, no predictive or "
            "NABM model was fitted, no private G4 artifact was included, the seal "
            "remains unopened, and no scientific effect is established."
        ),
    }


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise G1InputPreflightVerificationError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify the E/G1 execution input preflight."
    )
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify(args.authority)
    if args.output is not None:
        _write_json_create_only(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
