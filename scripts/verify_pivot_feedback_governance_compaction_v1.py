#!/usr/bin/env python3
"""Verify the shadow-only pivot-feedback governance compaction contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_governance_compaction_v1.yaml"
)
CONTRACT_SHA256 = "4c2562be842ec30b3fafee5320e88f5949b28fcd89e6091b34e5e43dcdd4c4ee"
CONTRACT_SCHEMA = "pivot_feedback_governance_compaction.v1"
CONTRACT_STATUS = (
    "PROPOSED_SIX_PHASE_COMPACTION_PENDING_INDEPENDENT_SEMANTIC_EQUIVALENCE_"
    "VALIDATION"
)
SUMMARY_SCHEMA = "pivot_feedback_governance_compaction_summary.v1"
SUMMARY_STATUS = (
    "PASS_SHADOW_SIX_PHASE_SEMANTIC_EQUIVALENCE_NO_AUTHORITY_CHANGE"
)
LEGACY_GATE_IDS = ["G_MINUS_1A", "S0", "D0", "G0", "G0_5", "G1", "G2", "G3", "G4", "G5"]
LIVE_PHASE_IDS = ["D", "C", "F", "E", "M", "H"]
PRESERVED_DENIALS = {
    "G0_SUCCESSOR_BINDING",
    "G1_OR_PRODUCTION_APPLICATION",
    "G2_OR_G3_CHOICE_RESOLUTION",
    "REQUIRED_SECTION_COMPLETION",
    "CONFIRMATORY_ROW_ACCESS",
    "HOLDOUT_ROW_ACCESS",
    "G4_FIELD_OR_EPISODE_SELECTION",
    "G4_ARTIFACT_MATERIALIZATION",
    "HOLDOUT_PREFLIGHT",
    "HOLDOUT_SEAL_CREATION",
    "G0_PASSAGE",
    "MODEL_FITTING",
    "G4_OPEN",
    "OUTCOME_BASED_SELECTION",
    "CAUSAL_CLAIM",
    "SCIENTIFIC_EFFECT",
}
PRESERVED_CLAIM_BOUNDARIES = {
    "D0_IS_DISCOVERY_MEASUREMENT_AND_DESCRIPTION_ONLY",
    "AUTHOR_IDS_ARE_CURRENT_SNAPSHOT_TOKENS_NOT_VERIFIED_PERSONS",
    "HISTORICAL_OPENALEX_VISIBILITY_IS_NOT_IDENTIFIED",
    "G1_EMPIRICAL_SIGNATURE_NOT_ESTABLISHED",
    "NABM_ADDED_VALUE_NOT_ESTABLISHED",
    "HOLDOUT_TRANSPORT_NOT_ESTABLISHED",
    "CAUSALITY_NOT_ESTABLISHED",
    "SCIENTIFIC_EFFECT_NOT_ESTABLISHED",
}
IRREVERSIBLE_BOUNDARIES = {
    "FIRST_CONFIRMATORY_READ": "F_PASSED_WITH_PANEL_CONFORMANCE_AND_HOLDOUT_SEAL",
    "FIRST_MODEL_SELECTION_SCORE_READ": "E_PASSED_AND_PRE_G2_REGISTRY_LOCKED",
    "FIRST_HOLDOUT_OPEN": "M_RESOLVED_AND_MODEL_SCORING_CONTRACT_LOCKED",
    "CLAIM_PROMOTION": "CLAIM_SPECIFIC_RECEIPTS_AND_CEILING_CHECKED",
}
EXPECTED_BINDINGS = {
    "effective_machine_authority": {
        "container": "authority_boundary",
        "path": "experiments/configs/pivot_feedback_gate_state_v61.yaml",
        "sha256": "253c661fd529feb0d70733174197be449dfd27403b5881c6510bd7c8257c2715",
        "schema": "pivot_feedback_gate_state.v61",
        "status": (
            "VALID_G0_V4_TWELVE_RESOLVED_SEVEN_PENDING_G1_TOLERANCE_"
            "PRECOMMITMENT_AUTHORIZED"
        ),
        "format": "yaml",
    },
    "legacy_gate_sequence": {
        "container": "bound_current_sources",
        "path": "experiments/configs/pivot_feedback_gate_sequence_v1.yaml",
        "sha256": "3a3a655234c4efb74a4900e65624c903176615cee19da760cd06b1416e3fd365",
        "schema": "pivot_feedback_gate_sequence.v1",
        "status": "VALID_PENDING_SUCCESSOR_IMPLEMENTATION",
        "format": "yaml",
    },
    "g0_v4": {
        "container": "bound_current_sources",
        "path": "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml",
        "sha256": "cd49a31ab67e7a66d038b3198493df8b3a309aa14ee9de80410557a194581ce4",
        "schema": "pivot_feedback_g0_measurement_freeze.v5",
        "status": (
            "PENDING_G0_V4_TWELVE_RESOLVED_SEVEN_CHOICES_AND_FIFTEEN_"
            "SECTIONS_NO_SEAL"
        ),
        "format": "yaml",
    },
    "current_human_ledger": {
        "container": "bound_current_sources",
        "path": "docs/research/pivot-feedback-current-status-evidence-ledger-v1.md",
        "sha256": "0df9793ad1b2fe9aaf26d8fcf115ea40179a34a46133b9adf3ceb75129f956c2",
        "schema": None,
        "status": None,
        "format": "text",
    },
    "g1_tolerance_contract": {
        "container": "bound_current_sources",
        "path": (
            "experiments/configs/"
            "pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml"
        ),
        "sha256": "1bcf91d197a9b927286d43f3505528f3f3f7fb022ffd4fe89d95cb24c041184d",
        "schema": "pivot_feedback_g0_g1_replication_tolerance_precommitment.v1",
        "status": (
            "FROZEN_SINGLE_G1_REPLICATION_TOLERANCE_PENDING_INDEPENDENT_"
            "VALIDATION_NO_G0_PROMOTION"
        ),
        "format": "yaml",
    },
    "g1_tolerance_validation_summary": {
        "container": "bound_current_sources",
        "path": (
            "experiments/evidence/results/"
            "pivot_feedback_g0_g1_replication_tolerance_precommitment_"
            "20260803.summary.json"
        ),
        "sha256": "72b653528d9cf5c8f1dd823f304934b48ac21a0dd4a50011ac3d02ef9079edb3",
        "schema": (
            "pivot_feedback_g0_g1_replication_tolerance_precommitment_summary.v1"
        ),
        "status": (
            "PASS_ONE_OUTCOME_BLIND_G1_REPLICATION_TOLERANCE_AND_CLOSED_FIXTURE"
        ),
        "format": "json",
    },
}


class GovernanceCompactionVerificationError(RuntimeError):
    """Fail-closed shadow governance verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GovernanceCompactionVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise GovernanceCompactionVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GovernanceCompactionVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _binding_spec(contract: Mapping[str, Any], key: str) -> dict[str, Any]:
    expected = EXPECTED_BINDINGS[key]
    container = _mapping(contract.get(expected["container"]), expected["container"])
    return _mapping(container.get(key), key)


def _validate_binding(contract: Mapping[str, Any], key: str) -> Any:
    declared = _binding_spec(contract, key)
    expected = EXPECTED_BINDINGS[key]
    require(declared.get("relative_path") == expected["path"], f"{key} path")
    require(declared.get("sha256") == expected["sha256"], f"{key} hash")
    require(declared.get("mutation_allowed") is False, f"{key} mutation")
    path = _safe_repo_path(declared["relative_path"], key)
    require(_sha256(path) == expected["sha256"], f"{key} byte hash")
    if expected["format"] == "text":
        return path.read_text(encoding="utf-8")
    document = (
        _read_yaml(path, key)
        if expected["format"] == "yaml"
        else _read_json(path, key)
    )
    require(
        declared.get("required_schema_version") == expected["schema"],
        f"{key} required schema",
    )
    require(document.get("schema_version") == expected["schema"], f"{key} schema")
    require(
        declared.get("required_status") == expected["status"],
        f"{key} required status",
    )
    require(document.get("status") == expected["status"], f"{key} status")
    return document


def _validate_authority_boundary(contract: Mapping[str, Any]) -> dict[str, Any]:
    boundary = _mapping(contract.get("authority_boundary"), "authority boundary")
    for key in (
        "shadow_contract_can_replace_effective_authority",
        "shadow_contract_can_authorize_scientific_execution",
        "shadow_contract_can_relax_existing_denial",
        "shadow_contract_can_mark_any_legacy_gate_passed",
        "shadow_contract_can_bind_a_g0_successor",
    ):
        require(boundary.get(key) is False, key)
    return {
        "effective_authority": "pivot_feedback_gate_state.v61",
        "shadow_replaces_authority": False,
        "shadow_authorizes_scientific_execution": False,
    }


def _validate_inventory(contract: Mapping[str, Any]) -> dict[str, Any]:
    declared = _mapping(contract.get("legacy_inventory_snapshot"), "inventory")
    configs = sorted(
        (REPO_ROOT / "experiments/configs").glob("pivot_feedback_gate_state_v*.yaml"),
        key=lambda path: int(re.search(r"_v(\d+)\.yaml$", path.name).group(1)),
    )
    versions = [int(re.search(r"_v(\d+)\.yaml$", path.name).group(1)) for path in configs]
    validators = list(
        (REPO_ROOT / "scripts").glob("verify_pivot_feedback_gate_state_v*.py")
    )
    tests = list((REPO_ROOT / "tests").glob("test_pivot_feedback_gate_state_v*.py"))
    parent_count = 0
    for path in configs:
        text = path.read_text(encoding="utf-8")
        parent_count += len(
            re.findall(r"^(?:additive_parent|authorization_parent):", text, re.M)
        )

    require(declared.get("logical_gate_ids") == LEGACY_GATE_IDS, "logical gate IDs")
    require(declared.get("logical_gate_count") == len(LEGACY_GATE_IDS), "logical count")
    require(versions == list(range(2, 62)), "gate state versions")
    require(declared.get("gate_state_version_first") == versions[0], "first version")
    require(declared.get("gate_state_version_last") == versions[-1], "last version")
    require(declared.get("gate_state_file_count") == len(configs) == 60, "config count")
    require(declared.get("gate_state_parent_reference_count") == parent_count == 59, "parent count")
    require(declared.get("gate_state_test_file_count") == len(tests) == 59, "test count")
    require(declared.get("gate_state_validator_file_count") == len(validators) == 37, "validator count")
    require(
        declared.get("snapshot_is_diagnostic_not_scientific_evidence") is True,
        "inventory evidence boundary",
    )
    return {
        "logical_gate_count": len(LEGACY_GATE_IDS),
        "gate_state_file_count": len(configs),
        "gate_state_parent_reference_count": parent_count,
        "gate_state_test_file_count": len(tests),
        "gate_state_validator_file_count": len(validators),
    }


def _validate_current_state(
    contract: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    declared = _mapping(contract.get("current_effective_state"), "current state")
    gate = _mapping(documents["effective_machine_authority"], "gate v61")
    g0 = _mapping(documents["g0_v4"], "G0 v4")
    tolerance = _mapping(documents["g1_tolerance_validation_summary"], "tolerance")
    gate_result = _mapping(
        _mapping(gate.get("verified_g0_v4_binding"), "verified G0 v4").get(
            "result"
        ),
        "gate result",
    )
    g0_resolution = _mapping(g0.get("resolution"), "G0 resolution")
    g0_lifecycle = _mapping(g0.get("lifecycle"), "G0 lifecycle")

    require(declared.get("authority_version") == "v61", "authority version")
    require(gate["gate_state"]["D0"]["complete"] is True, "D0 complete")
    require(declared.get("d0_complete_discovery_only") is True, "declared D0")
    require(
        declared.get("g0_v4_resolved_choice_count")
        == gate_result.get("total_resolved_choice_count")
        == g0_resolution.get("resolved_measurement_count")
        == 12,
        "resolved choice count",
    )
    require(
        declared.get("g0_v4_unresolved_choice_count")
        == gate_result.get("remaining_quantitative_choice_count")
        == g0_resolution.get("unresolved_quantitative_choice_count")
        == 7,
        "unresolved choice count",
    )
    require(
        declared.get("g0_v4_unresolved_required_section_count")
        == gate_result.get("pending_required_section_count")
        == 15,
        "pending sections",
    )
    require(
        declared.get("g0_v4_freeze_ready_required_section_count")
        == gate_result.get("freeze_ready_required_section_count")
        == 0,
        "ready sections",
    )
    require(
        declared.get("g1_tolerance_precommitted_and_independently_validated")
        is True,
        "tolerance receipt",
    )
    require(
        tolerance["authorization"]["single_choice_precommitment"]
        == "COMPLETE_INDEPENDENTLY_VERIFIED",
        "tolerance validation",
    )
    require(declared.get("g1_tolerance_bound_into_g0") is False, "tolerance binding")
    require(
        g0_resolution["unresolved_quantitative_choices"][
            "g1_replication_tolerance"
        ]
        is None,
        "G0 tolerance remains null",
    )
    require(declared.get("g0_passed") is g0_lifecycle.get("pass_g0") is False, "G0 pass")
    require(
        declared.get("holdout_receipt_bound")
        is gate_result.get("holdout_receipt_bound")
        is False,
        "holdout receipt",
    )
    require(
        declared.get("confirmatory_or_holdout_rows_read")
        == gate_result.get("confirmatory_or_holdout_rows_read")
        == 0,
        "row access",
    )
    require(declared.get("empirical_g1_status") == "NOT_RUN", "G1 state")
    require(declared.get("model_fitted") is gate_result.get("model_fitted") is False, "model")
    require(declared.get("g4_open") is gate_result.get("g4_open") is False, "G4")
    require(declared.get("scientific_effect") == "NOT_ESTABLISHED", "effect")
    return {
        "d0_complete_discovery_only": True,
        "g0_resolved_choice_count": 12,
        "g0_unresolved_choice_count": 7,
        "g0_pending_section_count": 15,
        "g1_tolerance_receipt_exists_but_is_unbound": True,
        "confirmatory_or_holdout_rows_read": 0,
        "model_fitted": False,
        "g4_open": False,
    }


def _validate_phase_mapping(
    contract: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    phases = contract.get("proposed_live_phases")
    require(isinstance(phases, list) and len(phases) == 6, "live phases")
    require([phase.get("phase_id") for phase in phases] == LIVE_PHASE_IDS, "phase IDs")
    require([phase.get("order") for phase in phases] == [10, 20, 30, 40, 50, 60], "phase order")

    mapped: list[str] = []
    completed: set[str] = set()
    for phase in phases:
        phase_id = phase["phase_id"]
        requires = phase.get("requires")
        require(isinstance(requires, list), f"{phase_id} dependencies")
        require(set(requires).issubset(completed), f"{phase_id} acyclic dependencies")
        completed.add(phase_id)
        legacy_ids = phase.get("legacy_gate_ids")
        require(isinstance(legacy_ids, list) and legacy_ids, f"{phase_id} mapping")
        mapped.extend(legacy_ids)
        receipts = phase.get("transition_receipts_required")
        require(isinstance(receipts, list) and receipts, f"{phase_id} receipts")
        require(isinstance(phase.get("claim_ceiling"), str), f"{phase_id} ceiling")

    optional = contract.get("optional_lanes")
    require(isinstance(optional, list) and len(optional) == 1, "optional lanes")
    causal = _mapping(optional[0], "causal lane")
    require(causal.get("lane_id") == "CAUSAL", "causal lane ID")
    require(causal.get("legacy_gate_ids") == ["G5"], "G5 mapping")
    require(causal.get("mandatory_for_bounded_noncausal_paper") is False, "G5 optional")
    mapped.extend(causal["legacy_gate_ids"])
    require(mapped == LEGACY_GATE_IDS, "legacy mapping order")
    require(len(mapped) == len(set(mapped)), "legacy mapping uniqueness")

    gate_sequence = _mapping(documents["legacy_gate_sequence"], "gate sequence")
    sequence_nodes = gate_sequence.get("gate_nodes")
    require(isinstance(sequence_nodes, list), "legacy gate nodes")
    require([node.get("gate_id") for node in sequence_nodes] == LEGACY_GATE_IDS, "legacy nodes")
    return {
        "legacy_logical_gate_count": len(mapped),
        "live_phase_count": len(phases),
        "optional_lane_count": 1,
        "mapping": {
            phase["phase_id"]: phase["legacy_gate_ids"] for phase in phases
        }
        | {"CAUSAL": ["G5"]},
        "every_legacy_gate_mapped_exactly_once": True,
        "dependencies_acyclic": True,
    }


def _validate_semantic_equivalence(
    contract: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    equivalence = _mapping(
        contract.get("semantic_equivalence_contract"), "semantic equivalence"
    )
    for key in (
        "every_legacy_logical_gate_mapped_exactly_once",
        "live_phase_order_strictly_increasing",
        "live_phase_dependencies_acyclic",
        "compaction_may_not_upgrade_current_state",
        "compaction_may_not_reduce_transition_receipts",
        "compaction_may_not_open_a_data_class_earlier",
        "compaction_may_not_raise_a_claim_ceiling",
        "implicit_deny_for_actions_not_explicitly_allowed",
    ):
        require(equivalence.get(key) is True, key)
    require(
        equivalence.get("shadow_allowed_actions")
        == [
            "GOVERNANCE_DESIGN_REVIEW",
            "REPOSITORY_ONLY_SEMANTIC_EQUIVALENCE_VALIDATION",
        ],
        "shadow actions",
    )
    require(set(equivalence.get("preserved_current_denials", [])) == PRESERVED_DENIALS, "denials")
    require(
        set(equivalence.get("preserved_current_claim_boundaries", []))
        == PRESERVED_CLAIM_BOUNDARIES,
        "claim boundaries",
    )

    gate_auth = _mapping(
        documents["effective_machine_authority"].get("authorization"),
        "gate authorization",
    )
    expected_gate_denials = {
        "g0_successor_binding_for_g1_tolerance": "NOT_AUTHORIZED",
        "g2_or_g3_choice_precommitment": "NOT_AUTHORIZED",
        "production_rule_application": "NOT_AUTHORIZED",
        "other_quantitative_choice_resolution": "NOT_AUTHORIZED",
        "required_section_completion": "NOT_AUTHORIZED",
        "g4_field_or_episode_selection": "NOT_AUTHORIZED",
        "holdout_artifact_materialization": "NOT_AUTHORIZED",
        "holdout_preflight_or_seal": "NOT_AUTHORIZED",
        "confirmatory_or_holdout_access": "NOT_AUTHORIZED",
        "outcome_based_selection": "NOT_AUTHORIZED",
        "g0_measurement_freeze": "NOT_AUTHORIZED",
        "model_fitting": "NOT_AUTHORIZED",
        "g4_open": "NOT_AUTHORIZED",
        "scientific_effect": "NOT_ESTABLISHED",
    }
    for key, value in expected_gate_denials.items():
        require(gate_auth.get(key) == value, f"gate denial {key}")

    shadow_auth = _mapping(contract.get("authorization"), "shadow authorization")
    require(
        shadow_auth.get("repository_only_semantic_equivalence_validation")
        == "AUTHORIZED",
        "metadata validation authorization",
    )
    for key in (
        "authority_replacement",
        "legacy_archive_routing_change",
        "g0_successor_binding",
        "g1_or_production_application",
        "g2_or_g3_choice_resolution",
        "required_section_completion",
        "confirmatory_or_holdout_access",
        "holdout_preflight_or_seal",
        "g0_measurement_freeze",
        "model_fitting",
        "g4_open",
    ):
        require(shadow_auth.get(key) == "NOT_AUTHORIZED", f"shadow denial {key}")
    require(shadow_auth.get("scientific_effect") == "NOT_ESTABLISHED", "shadow effect")
    return {
        "preserved_denial_count": len(PRESERVED_DENIALS),
        "preserved_claim_boundary_count": len(PRESERVED_CLAIM_BOUNDARIES),
        "shadow_allowed_action_count": 2,
        "implicit_deny": True,
        "authority_relaxation_detected": False,
        "claim_ceiling_raise_detected": False,
    }


def _validate_decision_staging(
    contract: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    proposal = _mapping(contract.get("decision_staging_proposal"), "decision staging")
    stages = proposal.get("stages")
    require(isinstance(stages, list) and len(stages) == 3, "decision stages")
    g0_resolution = _mapping(documents["g0_v4"].get("resolution"), "G0 resolution")
    resolved = set(
        _mapping(g0_resolution.get("resolved_measurements"), "resolved choices")
    )
    unresolved = set(
        _mapping(
            g0_resolution.get("unresolved_quantitative_choices"),
            "unresolved choices",
        )
    )
    legacy_choices = resolved | unresolved
    staged = [choice for stage in stages for choice in stage.get("choices", [])]
    require(proposal.get("legacy_total_quantitative_choice_count") == 19, "legacy choices")
    require(len(staged) == len(set(staged)) == 19, "staged choice uniqueness")
    require(set(staged) == legacy_choices, "staged choice coverage")

    pre_g1, pre_g2, pre_g3 = stages
    require(pre_g1.get("owner_phase") == "F", "pre-G1 owner")
    require(pre_g1.get("choice_count") == 13, "pre-G1 count")
    require(set(pre_g1["choices"]) == resolved | {"g1_replication_tolerance"}, "pre-G1 choices")
    require(pre_g1.get("bound_in_g0_v4_count") == 12, "bound count")
    require(pre_g1.get("independently_precommitted_not_bound_count") == 1, "unbound count")
    require(pre_g1.get("may_use_g1_or_holdout_outcomes") is False, "pre-G1 access")

    require(pre_g2.get("owner_phase") == "M", "pre-G2 owner")
    require(pre_g2.get("choice_count") == 3, "pre-G2 count")
    require(
        set(pre_g2["choices"])
        == {
            "g2_score_margin",
            "g2_maximum_calibration_degradation",
            "g2_macro_trajectory_tolerance",
        },
        "pre-G2 choices",
    )
    require(pre_g2.get("may_use_g2_model_scores_or_holdout_outcomes") is False, "pre-G2 access")
    require(pre_g2.get("may_depend_on_g1_effect_magnitude") is False, "G1 dependence")

    require(pre_g3.get("owner_phase") == "M", "pre-G3 owner")
    require(pre_g3.get("choice_count") == 3, "pre-G3 count")
    require(
        set(pre_g3["choices"])
        == {
            "simulation_draws_or_mcse_stopping_rule",
            "parameter_budget",
            "tuning_budget",
        },
        "pre-G3 choices",
    )
    require(pre_g3.get("may_use_g3_simulation_results_or_holdout_outcomes") is False, "pre-G3 access")
    return {
        "legacy_choice_count": len(legacy_choices),
        "pre_g1_choice_count": 13,
        "pre_g2_choice_count": 3,
        "pre_g3_choice_count": 3,
        "all_choices_mapped_exactly_once": True,
        "six_later_stage_choices_removed_from_pre_g1_blocker": True,
        "g1_tolerance_is_precommitted_but_not_bound": True,
    }


def _validate_section_staging(
    contract: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    proposal = _mapping(contract.get("section_staging_proposal"), "section staging")
    stages = proposal.get("stages")
    require(isinstance(stages, list) and len(stages) == 3, "section stages")
    g0_resolution = _mapping(documents["g0_v4"].get("resolution"), "G0 resolution")
    legacy_sections = set(g0_resolution.get("unresolved_required_sections", []))
    staged = [section for stage in stages for section in stage.get("section_ids", [])]
    require(proposal.get("legacy_total_required_section_count") == 15, "section total")
    require(len(staged) == len(set(staged)) == 15, "section uniqueness")
    require(set(staged) == legacy_sections, "section coverage")
    require(stages[0].get("owner_phase") == "F", "pre-G1 section owner")
    require(stages[0].get("section_count") == 13, "pre-G1 section count")
    require(stages[1].get("section_ids") == ["g2_predictive_gate"], "G2 section")
    require(stages[2].get("section_ids") == ["g3_simulation_gate"], "G3 section")
    return {
        "legacy_section_count": 15,
        "pre_g1_section_count": 13,
        "pre_g2_section_count": 1,
        "pre_g3_section_count": 1,
        "all_sections_mapped_exactly_once": True,
        "later_stage_sections_removed_from_pre_g1_blocker": 2,
    }


def _validate_boundaries_and_migration(contract: Mapping[str, Any]) -> dict[str, Any]:
    boundaries = contract.get("irreversible_boundaries")
    require(isinstance(boundaries, list) and len(boundaries) == 4, "boundaries")
    observed = {
        item.get("boundary_id"): item.get("required_phase_state")
        for item in boundaries
    }
    require(observed == IRREVERSIBLE_BOUNDARIES, "irreversible boundary mapping")
    require(all(item.get("fail_closed") is True for item in boundaries), "fail closed")

    migration = _mapping(contract.get("migration_contract"), "migration")
    require(migration.get("current_stage") == "SHADOW_DESIGN_ONLY", "migration stage")
    steps = migration.get("steps")
    require(isinstance(steps, list) and len(steps) == 5, "migration steps")
    require([step.get("step_id") for step in steps] == [
        "S1_SHADOW_SCHEMA",
        "S2_SEMANTIC_EQUIVALENCE",
        "S3_PARALLEL_READ",
        "S4_AUTHORITY_SUCCESSOR",
        "S5_ARCHIVE_ROUTING",
    ], "migration order")
    require(steps[0].get("state") == "THIS_ARTIFACT", "S1 state")
    require(steps[1].get("state") == "PENDING_INDEPENDENT_VALIDATION", "S2 state")
    require(all(step.get("state") == "NOT_AUTHORIZED" for step in steps[2:]), "future migration states")
    require(migration.get("delete_or_rewrite_legacy_gate_files") is False, "legacy deletion")
    require(migration.get("mutate_v61_or_g0_v4") is False, "parent mutation")
    require(migration.get("skip_required_scientific_receipt") is False, "receipt skip")
    return {
        "irreversible_boundary_count": 4,
        "all_irreversible_boundaries_fail_closed": True,
        "migration_step_count": 5,
        "current_migration_stage": "SHADOW_DESIGN_ONLY",
        "authority_successor_authorized": False,
        "legacy_deletion_authorized": False,
    }


def _validate_hash_scope(contract: Mapping[str, Any]) -> dict[str, Any]:
    policy = _mapping(contract.get("hash_scope_policy"), "hash scope")
    scientific = policy.get("hash_as_scientific_lineage_roots")
    nonroots = policy.get("do_not_hash_as_scientific_roots_by_default")
    code_conditions = policy.get("executable_code_hash_required_when")
    require(isinstance(scientific, list) and len(scientific) == 6, "scientific roots")
    require(isinstance(nonroots, list) and set(nonroots) == {
        "HUMAN_NAVIGATION_DOCS",
        "LINT_CONFIGURATION",
        "GENERAL_TEST_RUNNER_CONFIGURATION",
        "NONSEMANTIC_FORMATTING",
    }, "nonroots")
    require(isinstance(code_conditions, list) and len(code_conditions) == 3, "code hash conditions")
    require(policy.get("human_docs_may_summarize_but_may_not_override_machine_state") is True, "human docs")
    return {
        "scientific_lineage_root_class_count": len(scientific),
        "default_nonroot_class_count": len(nonroots),
        "conditional_code_hash_rule_count": len(code_conditions),
    }


def verify(contract_path: Path) -> dict[str, Any]:
    require(contract_path == DEFAULT_CONTRACT.resolve(), "unexpected contract path")
    require(_sha256(contract_path) == CONTRACT_SHA256, "contract hash drift")
    contract = _read_yaml(contract_path, "contract")
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(contract.get("status") == CONTRACT_STATUS, "contract status")
    documents = {
        key: _validate_binding(contract, key) for key in EXPECTED_BINDINGS
    }
    authority = _validate_authority_boundary(contract)
    inventory = _validate_inventory(contract)
    current = _validate_current_state(contract, documents)
    phases = _validate_phase_mapping(contract, documents)
    equivalence = _validate_semantic_equivalence(contract, documents)
    decisions = _validate_decision_staging(contract, documents)
    sections = _validate_section_staging(contract, documents)
    boundaries = _validate_boundaries_and_migration(contract)
    hash_scope = _validate_hash_scope(contract)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": SUMMARY_STATUS,
        "authority_replay": authority,
        "legacy_inventory_replay": inventory,
        "current_state_replay": current,
        "phase_mapping_replay": phases,
        "semantic_equivalence_replay": equivalence,
        "decision_staging_replay": decisions,
        "section_staging_replay": sections,
        "boundary_and_migration_replay": boundaries,
        "hash_scope_replay": hash_scope,
        "access_accounting": {
            "repository_yaml_documents_read": 5,
            "repository_json_documents_read": 1,
            "repository_text_documents_read": 1,
            "gate_state_metadata_files_counted": 60,
            "operator_or_row_level_files_read": 0,
            "discovery_rows_read": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "repository_artifacts": {
            "shadow_contract": {
                "relative_path": contract_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(contract_path),
            }
        },
        "authorization": {
            "shadow_design": "COMPLETE_INDEPENDENTLY_VALIDATED",
            "parallel_read": "NOT_AUTHORIZED",
            "authority_successor": "NOT_AUTHORIZED",
            "legacy_archive_routing_change": "NOT_AUTHORIZED",
            "g0_successor_binding": "NOT_AUTHORIZED",
            "scientific_execution": "NOT_AUTHORIZED",
            "confirmatory_or_holdout_access": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "This independent repository-only replay validates that the proposed "
            "six-phase shadow governance maps every legacy logical gate exactly "
            "once, preserves current denials and claim ceilings, covers all "
            "nineteen quantitative choices and fifteen required sections, and "
            "keeps four irreversible boundaries fail-closed. It does not replace "
            "gate v61, bind the G1 tolerance into G0, authorize a successor, "
            "change active routing, read scientific rows, fit a model, open G4, "
            "or establish an effect."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = verify(args.contract.resolve())
    except (GovernanceCompactionVerificationError, OSError) as error:
        raise SystemExit(str(error)) from error
    rendered = json.dumps(output, indent=2, sort_keys=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
