#!/usr/bin/env python3
"""Verify the active pivot-feedback milestone authority without scientific reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORITY = (
    ROOT / "experiments/configs/pivot_feedback_current_state_v1.yaml"
)

EXPECTED_BINDINGS = {
    "previous_authority": {
        "container": "authority_transition",
        "path": "experiments/configs/pivot_feedback_gate_state_v61.yaml",
        "schema": "pivot_feedback_gate_state.v61",
        "status": "VALID_G0_V4_TWELVE_RESOLVED_SEVEN_PENDING_G1_TOLERANCE_PRECOMMITMENT_AUTHORIZED",
    },
    "shadow_design": {
        "container": "authority_transition",
        "path": "experiments/configs/pivot_feedback_governance_compaction_v1.yaml",
        "schema": "pivot_feedback_governance_compaction.v1",
        "status": None,
    },
    "semantic_equivalence_receipt": {
        "container": "authority_transition",
        "path": "experiments/evidence/results/pivot_feedback_governance_compaction_v1_20260803.summary.json",
        "schema": "pivot_feedback_governance_compaction_summary.v1",
        "status": "PASS_SHADOW_SIX_PHASE_SEMANTIC_EQUIVALENCE_NO_AUTHORITY_CHANGE",
    },
    "g0_v4": {
        "container": "bound_freeze_inputs",
        "path": "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml",
        "schema": "pivot_feedback_g0_measurement_freeze.v5",
        "status": "PENDING_G0_V4_TWELVE_RESOLVED_SEVEN_CHOICES_AND_FIFTEEN_SECTIONS_NO_SEAL",
    },
    "g1_tolerance_contract": {
        "container": "bound_freeze_inputs",
        "path": "experiments/configs/pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml",
        "schema": "pivot_feedback_g0_g1_replication_tolerance_precommitment.v1",
        "status": "FROZEN_SINGLE_G1_REPLICATION_TOLERANCE_PENDING_INDEPENDENT_VALIDATION_NO_G0_PROMOTION",
    },
    "g1_tolerance_validation_receipt": {
        "container": "bound_freeze_inputs",
        "path": "experiments/evidence/results/pivot_feedback_g0_g1_replication_tolerance_precommitment_20260803.summary.json",
        "schema": "pivot_feedback_g0_g1_replication_tolerance_precommitment_summary.v1",
        "status": "PASS_ONE_OUTCOME_BLIND_G1_REPLICATION_TOLERANCE_AND_CLOSED_FIXTURE",
    },
}


class CurrentStateVerificationError(RuntimeError):
    """Raised when the milestone authority fails closed verification."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentStateVerificationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_document(path: Path) -> Mapping[str, Any]:
    label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
    require(path.is_file(), f"missing bound file: {label}")
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, label)


def _load_bound_documents(
    authority: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    documents: dict[str, Mapping[str, Any]] = {}
    for name, expected in EXPECTED_BINDINGS.items():
        container = _mapping(authority.get(expected["container"]), expected["container"])
        binding = _mapping(container.get(name), name)
        require(binding.get("relative_path") == expected["path"], f"{name} path")
        path = (ROOT / expected["path"]).resolve()
        require(path.is_relative_to(ROOT), f"{name} path escapes repository")
        require(binding.get("sha256") == _sha256(path), f"{name} hash")
        require(binding.get("mutation_allowed") is False, f"{name} immutable")
        document = _read_document(path)
        require(document.get("schema_version") == expected["schema"], f"{name} schema")
        required_schema = binding.get("required_schema_version")
        require(required_schema == expected["schema"], f"{name} required schema")
        if expected["status"] is not None:
            require(document.get("status") == expected["status"], f"{name} status")
            require(binding.get("required_status") == expected["status"], f"{name} required status")
        documents[name] = document
    return documents


def _validate_transition(
    authority: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    transition = _mapping(authority.get("authority_transition"), "authority transition")
    require(
        transition.get("activation_basis") == "EXPLICIT_USER_APPROVAL_2026_08_03",
        "activation basis",
    )
    require(
        transition.get("activation_scope")
        == "ADMINISTRATIVE_ROUTING_AND_F_PHASE_CLOSURE_ONLY",
        "activation scope",
    )
    for key in (
        "previous_authority_remains_immutable_provenance",
        "legacy_gate_files_remain_immutable_provenance",
        "active_routing_uses_this_authority",
    ):
        require(transition.get(key) is True, key)
    require(transition.get("delete_or_rewrite_legacy_files") is False, "legacy deletion")

    compaction_receipt = documents["semantic_equivalence_receipt"]
    replay = _mapping(compaction_receipt.get("semantic_equivalence_replay"), "semantic replay")
    require(replay.get("authority_relaxation_detected") is False, "semantic authority relaxation")
    require(replay.get("claim_ceiling_raise_detected") is False, "semantic claim raise")
    return {
        "previous_authority": documents["previous_authority"]["schema_version"],
        "semantic_equivalence_receipt": "PASS",
        "parallel_read_replay": "PASS",
        "active_authority": authority["schema_version"],
        "legacy_files_mutated_or_deleted": False,
    }


def _validate_routing(
    authority: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    routing = _mapping(authority.get("active_routing"), "active routing")
    phases = routing.get("phases")
    require(isinstance(phases, list) and len(phases) == 6, "six active phases")
    expected_order = ["D", "C", "F", "E", "M", "H"]
    require(routing.get("phase_order") == expected_order, "phase order")
    require(routing.get("active_phase") == "F", "active phase")
    require([phase.get("phase_id") for phase in phases] == expected_order, "phase IDs")

    shadow_phases = documents["shadow_design"].get("proposed_live_phases")
    require(isinstance(shadow_phases, list) and len(shadow_phases) == 6, "shadow phases")
    copied_keys = (
        "phase_id",
        "legacy_gate_ids",
        "requires",
        "may_read_discovery_outcomes",
        "may_read_confirmatory_or_holdout_rows",
        "may_fit_model",
        "transition_receipts_required",
        "claim_ceiling",
    )
    for active, shadow in zip(phases, shadow_phases, strict=True):
        for key in copied_keys:
            require(active.get(key) == shadow.get(key), f"phase {active.get('phase_id')} {key}")

    by_id = {phase["phase_id"]: phase for phase in phases}
    require(by_id["D"].get("complete") is True, "D completion")
    require(by_id["C"].get("complete") is True, "C completion")
    require(by_id["F"].get("complete") is False, "F must remain incomplete")
    require(all(by_id[item].get("complete") is False for item in ("E", "M", "H")), "future phases incomplete")
    optional = _mapping(routing.get("optional_causal_lane"), "optional causal lane")
    shadow_optional = _mapping(documents["shadow_design"]["optional_lanes"][0], "shadow causal lane")
    for key in ("legacy_gate_ids", "requires", "mandatory_for_bounded_noncausal_paper", "claim_ceiling"):
        require(optional.get(key) == shadow_optional.get(key), f"causal {key}")
    require(optional.get("state") == "NOT_RUN", "causal state")
    return {
        "phase_order": expected_order,
        "active_phase": "F",
        "bounded_complete_phases": ["D", "C"],
        "incomplete_phases": ["F", "E", "M", "H"],
        "mapping_matches_validated_shadow": True,
    }


def _stage_values(container: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = list(container.values())
    require(all(isinstance(value, Mapping) for value in values), "stage mappings")
    return values


def _validate_decisions_and_sections(
    authority: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision_registry = _mapping(authority.get("staged_decision_registry"), "decision registry")
    decisions = _stage_values(decision_registry)
    require([item.get("choice_count") for item in decisions] == [13, 3, 3], "decision counts")
    staged_choices = [choice for item in decisions for choice in item.get("choice_ids", [])]
    require(len(staged_choices) == len(set(staged_choices)) == 19, "decision uniqueness")

    shadow_decisions = documents["shadow_design"]["decision_staging_proposal"]["stages"]
    require(
        [item.get("choice_ids") for item in decisions]
        == [item.get("choices") for item in shadow_decisions],
        "decision staging equivalence",
    )
    g0_resolution = documents["g0_v4"]["resolution"]
    g0_choices = set(g0_resolution["resolved_measurements"])
    tolerance_choices = set(documents["g1_tolerance_contract"]["scope"]["choice_ids"])
    require(set(decisions[0]["choice_ids"]) == g0_choices | tolerance_choices, "pre-G1 binding")
    require(len(g0_choices) == 12 and tolerance_choices == {"g1_replication_tolerance"}, "pre-G1 sources")
    require(decisions[0].get("state") == "COMPLETE_OUTCOME_BLIND_HASH_BOUND", "pre-G1 decision state")
    require(decisions[1].get("state") == "DEFERRED_UNRESOLVED_BLOCKS_M_ONLY", "G2 decision state")
    require(decisions[2].get("state") == "DEFERRED_UNRESOLVED_BLOCKS_G3_ONLY", "G3 decision state")

    section_registry = _mapping(authority.get("staged_section_registry"), "section registry")
    sections = _stage_values(section_registry)
    require([item.get("section_count") for item in sections] == [13, 1, 1], "section counts")
    staged_sections = [section for item in sections for section in item.get("section_ids", [])]
    require(len(staged_sections) == len(set(staged_sections)) == 15, "section uniqueness")
    shadow_sections = documents["shadow_design"]["section_staging_proposal"]["stages"]
    require(
        [item.get("section_ids") for item in sections]
        == [item.get("section_ids") for item in shadow_sections],
        "section staging equivalence",
    )
    require(set(staged_sections) == set(g0_resolution["unresolved_required_sections"]), "section coverage")
    require(sections[0].get("state") == "PENDING_THIRTEEN_SECTIONS", "pre-G1 section state")
    require(sections[0].get("freeze_ready_section_count") == 0, "freeze-ready sections")
    return (
        {
            "pre_g1_choice_count": 13,
            "pre_g1_registry_complete": True,
            "later_stage_choice_count": 6,
            "later_choices_block_f": False,
            "all_legacy_choices_staged_exactly_once": True,
        },
        {
            "pre_g1_section_count": 13,
            "pre_g1_freeze_ready_section_count": 0,
            "later_stage_section_count": 2,
            "later_sections_block_f": False,
            "all_legacy_sections_staged_exactly_once": True,
        },
    )


def _validate_f_phase(authority: Mapping[str, Any]) -> dict[str, Any]:
    closure = _mapping(authority.get("f_phase_closure"), "F closure")
    require(closure.get("completed_receipts") == ["PRE_G1_DECISION_REGISTRY_COMPLETE"], "completed F receipt")
    expected_pending = [
        "THIRTEEN_PRE_G1_REQUIRED_SECTIONS_COMPLETE",
        "OUTCOME_BLIND_FINAL_PANEL_CONFORMANCE",
        "ATOMIC_PARTITION_AND_HOLDOUT_BYTE_SEAL",
    ]
    require(closure.get("pending_receipts") == expected_pending, "pending F receipts")
    require(closure.get("completion_is_one_atomic_milestone") is True, "atomic F milestone")
    require(
        closure.get("intermediate_receipts_may_record_provenance_but_are_not_new_gates") is True,
        "no new micro-gates",
    )
    expected_actions = {
        "COMPLETE_THIRTEEN_PRE_G1_SECTIONS_OUTCOME_BLIND",
        "BUILD_AND_VERIFY_FINAL_PANEL_CONFORMANCE",
        "ASSIGN_ATOMIC_PARTITIONS_WITHOUT_CONFIRMATORY_ANALYTIC_READ",
        "MATERIALIZE_PRIVATE_HOLDOUT_ARTIFACTS_OPERATOR_ONLY",
        "RUN_HOLDOUT_PREFLIGHT_AND_WRITE_BYTE_SEAL_LAST",
    }
    require(set(closure.get("authorized_actions", [])) == expected_actions, "F authorization")
    require(closure.get("operator_only_holdout_materialization_is_not_holdout_open") is True, "holdout boundary")
    require(closure.get("first_confirmatory_read_requires_f_pass") is True, "confirmatory hard stop")
    require(closure.get("pass_f_now") is False, "F pass")
    require(closure.get("pass_g0_now") is False, "G0 pass")
    return {
        "completed_receipt_count": 1,
        "pending_receipt_count": 3,
        "pending_receipts": expected_pending,
        "one_atomic_completion_milestone": True,
        "micro_gate_added": False,
    }


def _validate_fail_closed(authority: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(authority.get("lifecycle"), "lifecycle")
    required_true = (
        "authority_active",
        "parallel_read_replay_complete",
        "g1_tolerance_bound_into_pre_g1_registry",
    )
    for key in required_true:
        require(lifecycle.get(key) is True, key)
    expected_false = (
        "final_panel_conformance_run",
        "holdout_receipt_bound",
        "pass_f",
        "pass_g0",
        "model_fitted",
        "g4_open",
    )
    for key in expected_false:
        require(lifecycle.get(key) is False, key)
    require(lifecycle.get("pre_g1_pending_choice_count") == 0, "pending pre-G1 choices")
    require(lifecycle.get("pre_g1_pending_section_count") == 13, "pending pre-G1 sections")
    require(lifecycle.get("confirmatory_or_holdout_rows_read") == 0, "row reads")
    require(lifecycle.get("empirical_g1_status") == "NOT_RUN", "G1 status")
    require(lifecycle.get("scientific_effect") == "NOT_ESTABLISHED", "effect")

    boundaries = _mapping(authority.get("irreversible_boundaries"), "boundaries")
    require(boundaries.get("all_fail_closed") is True, "fail-closed boundaries")
    authorization = _mapping(authority.get("authorization"), "authorization")
    require(authorization.get("six_phase_active_routing") == "ACTIVE", "active routing")
    require(authorization.get("f_phase_closure") == "AUTHORIZED_EXACT_SCOPE", "F scope")
    expected_denials = {
        "e_g1_execution": "NOT_AUTHORIZED_UNTIL_F_PASS",
        "g2_or_g3_choice_resolution": "NOT_AUTHORIZED_UNTIL_REGISTERED_ENTRY_BOUNDARY",
        "confirmatory_or_holdout_analytic_access": "NOT_AUTHORIZED",
        "outcome_based_threshold_selection": "NOT_AUTHORIZED",
        "pass_f_without_all_receipts": "NOT_AUTHORIZED",
        "pass_g0_without_all_receipts": "NOT_AUTHORIZED",
        "model_fitting": "NOT_AUTHORIZED",
        "g4_open": "NOT_AUTHORIZED",
        "causal_claim": "NOT_AUTHORIZED",
        "scientific_effect": "NOT_ESTABLISHED",
    }
    for key, value in expected_denials.items():
        require(authorization.get(key) == value, f"denial {key}")
    return {
        "pass_f": False,
        "pass_g0": False,
        "confirmatory_or_holdout_rows_read": 0,
        "empirical_g1_status": "NOT_RUN",
        "model_fitted": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "denial_count": len(expected_denials),
    }


def verify(path: Path = DEFAULT_AUTHORITY) -> dict[str, Any]:
    authority = _read_document(path)
    require(authority.get("schema_version") == "pivot_feedback_current_state.v1", "authority schema")
    require(authority.get("mode") == "ACTIVE_SIX_PHASE_MILESTONE_AUTHORITY_FAIL_CLOSED", "authority mode")
    require(
        authority.get("status")
        == "ACTIVE_F_PHASE_PRE_G1_REGISTRY_BOUND_THIRTEEN_SECTIONS_PANEL_AND_SEAL_PENDING",
        "authority status",
    )
    documents = _load_bound_documents(authority)
    transition = _validate_transition(authority, documents)
    routing = _validate_routing(authority, documents)
    decisions, sections = _validate_decisions_and_sections(authority, documents)
    f_phase = _validate_f_phase(authority)
    fail_closed = _validate_fail_closed(authority)
    return {
        "schema_version": "pivot_feedback_current_state_summary.v1",
        "status": "PASS_ACTIVE_SIX_PHASE_AUTHORITY_F_IN_PROGRESS_FAIL_CLOSED",
        "authority_transition_replay": transition,
        "active_routing_replay": routing,
        "decision_registry_replay": decisions,
        "section_registry_replay": sections,
        "f_phase_replay": f_phase,
        "fail_closed_replay": fail_closed,
        "access_accounting": {
            "repository_yaml_documents_read": 5,
            "repository_json_documents_read": 2,
            "operator_or_row_level_files_read": 0,
            "discovery_rows_read": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "repository_artifacts": {
            "active_authority": {
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
        },
        "authorization": {
            "active_routing": "COMPLETE",
            "f_phase_closure": "AUTHORIZED_EXACT_SCOPE",
            "e_g1_execution": "NOT_AUTHORIZED_UNTIL_F_PASS",
            "confirmatory_or_holdout_analytic_access": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "The milestone authority activates the validated six-phase routing and binds "
            "the outcome-blind G1 tolerance into a complete thirteen-choice pre-G1 "
            "registry. F remains in progress with thirteen sections, final-panel "
            "conformance, atomic partitioning, and holdout byte sealing pending. No "
            "confirmatory or holdout row was read, F and G0 did not pass, G1 was not "
            "run, no model was fitted, G4 remains unopened, and no scientific effect "
            "is established."
        ),
    }


def main() -> int:
    try:
        result = verify()
    except (CurrentStateVerificationError, OSError, ValueError, yaml.YAMLError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
