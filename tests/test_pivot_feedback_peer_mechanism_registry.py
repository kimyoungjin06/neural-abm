from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    ROOT
    / "experiments"
    / "configs"
    / "pivot_feedback_peer_mechanism_registry_v0.yaml"
)
DESIGN_PATH = (
    ROOT
    / "docs"
    / "research"
    / "peer-exposure-identification-design.md"
)
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "case-studies"
    / "researcher-pivot"
    / "pivot-feedback-empirical-nabm-protocol.md"
)


def _registry() -> dict[str, Any]:
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    assert all(isinstance(row, dict) for row in rows)
    resolved = {row[key]: row for row in rows}
    assert len(resolved) == len(rows)
    return resolved


def test_peer_registry_is_prospective_closed_and_noncausal() -> None:
    registry = _registry()

    assert set(registry) == {
        "schema_version",
        "label",
        "mode",
        "status",
        "parent_contract",
        "gate_state",
        "interpretation_boundary",
        "exposure_constructs",
        "contrast_ladder",
        "competing_explanations",
        "capacity_bridge_conditional_contrast",
        "entry_mode_strata",
        "negative_controls",
        "claim_levels",
        "claim_downgrade_rules",
    }
    assert registry["schema_version"] == (
        "pivot_feedback_peer_mechanism_registry.v1"
    )
    assert registry["mode"] == "PROSPECTIVE_G0_COMPONENT"
    assert registry["status"] == "PENDING_G0_MEASUREMENT_FREEZE"
    assert registry["parent_contract"] == {
        "relative_path": (
            "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml"
        ),
        "required_schema_version": "pivot_feedback_cross_field_matrix.v1",
        "binding_status": "PATH_AND_SCHEMA_ONLY_PENDING_G0_HASH_FREEZE",
    }
    parent_path = ROOT / registry["parent_contract"]["relative_path"]
    assert parent_path.is_file()
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    assert parent["schema_version"] == registry["parent_contract"][
        "required_schema_version"
    ]
    assert registry["gate_state"] == {
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "external_causal_axis": "NOT_RUN",
        "execution_authorized": False,
    }

    boundary = registry["interpretation_boundary"]
    assert set(boundary) == {
        "observed_object",
        "direct_information_receipt_observed",
        "belief_update_observed",
        "causal_social_learning_identified",
        "strongest_openalex_only_interpretation",
        "forbidden_claims",
    }
    assert boundary["direct_information_receipt_observed"] is False
    assert boundary["belief_update_observed"] is False
    assert boundary["causal_social_learning_identified"] is False
    assert set(boundary["forbidden_claims"]) == {
        "RESEARCHERS_SAW_THE_PEER_PIVOT",
        "RESEARCHERS_CHANGED_BELIEFS_BECAUSE_OF_PEERS",
        "PEER_EXPOSURE_CAUSED_ENTRY",
        "PEER_EXPOSURE_CAUSED_PERSISTENCE",
    }


def test_peer_registry_is_linked_to_the_protocol_and_human_readable_design() -> None:
    design = " ".join(DESIGN_PATH.read_text(encoding="utf-8").split())
    protocol = " ".join(PROTOCOL_PATH.read_text(encoding="utf-8").split())

    for required in (
        "publicly available target-topic event",
        "actual receipt or attention [not observed in OpenAlex]",
        "belief or policy update [not observed in OpenAlex]",
        "`chaperoned_entry`",
        "`independent_entry`",
        "survives the independent-entry branch and every registered placebo",
        "Bibliometric silence is not an observed failure",
    ):
        assert required in design

    for required in (
        "`docs/research/peer-exposure-identification-design.md`",
        "`experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml`",
        "`PENDING_G0_MEASUREMENT_FREEZE`",
        "bibliographic outcome-availability lag",
        "prior-collaborator exposure-opportunity timescale",
        "Matched non-neighbor and shuffled-network controls plus pre-tie similarity diagnostics",
    ):
        assert required in protocol

    assert "matched-neighbor controls remain required" not in protocol
    assert "survives both entry pathways" not in design


def test_peer_registry_separates_activity_from_visible_success() -> None:
    registry = _registry()
    constructs = _by_id(registry["exposure_constructs"], "construct_id")
    contrasts = _by_id(registry["contrast_ladder"], "contrast_id")

    assert set(constructs) == {"peer_active", "peer_visible_success"}
    active = constructs["peer_active"]
    visible = constructs["peer_visible_success"]
    assert set(active) == {
        "construct_id",
        "label",
        "peer_set_rule",
        "event_rule",
        "availability_rule",
        "success_required",
        "excludes",
    }
    assert set(visible) == {
        "construct_id",
        "label",
        "subset_of",
        "peer_set_rule",
        "event_rule",
        "availability_rule",
        "success_rule_status",
        "excludes",
    }
    assert active["success_required"] is False
    assert visible["subset_of"] == "peer_active"
    assert visible["success_rule_status"] == "PENDING_G0_MEASUREMENT_FREEZE"
    assert active["peer_set_rule"] == visible["peer_set_rule"] == (
        "PRE_EXISTING_T_MINUS_1_COLLABORATORS_ONLY"
    )
    assert "POST_DECISION_CITATIONS" in active["excludes"]
    assert "POST_DECISION_CITATIONS" in visible["excludes"]
    assert "AT_OR_BEFORE_FOCAL_DECISION_CUTOFF" in active["availability_rule"]
    assert "AT_OR_BEFORE_FOCAL_DECISION_CUTOFF" in visible["availability_rule"]

    assert set(contrasts) == {
        "peer_active_increment",
        "peer_visible_success_increment",
    }
    assert all(
        set(contrast)
        == {
            "contrast_id",
            "focal_construct",
            "reference",
            "estimand",
            "required_comparators",
        }
        for contrast in contrasts.values()
    )
    assert contrasts["peer_active_increment"]["reference"] == (
        "NO_PRE_DECISION_TARGET_ACTIVE_PEER"
    )
    success_increment = contrasts["peer_visible_success_increment"]
    assert success_increment["reference"] == (
        "PEER_ACTIVE_WITHOUT_VISIBLE_SUCCESS_AT_DECISION_CUTOFF"
    )
    assert success_increment["estimand"] == (
        "INCREMENT_BEYOND_PEER_ACTIVITY_FIELD_WIDE_ATTENTION_AND_PRE_ENTRY_COVARIATES"
    )


def test_peer_registry_freezes_competing_explanations_and_capacity_contrast() -> None:
    registry = _registry()
    explanations = _by_id(
        registry["competing_explanations"],
        "explanation_id",
    )

    assert set(explanations) == {
        "field_wide_attention_or_common_opportunity",
        "selection_or_homophily",
        "recruitment_or_capacity_transfer",
        "network_local_information",
    }
    assert all(
        set(explanation)
        == {
            "explanation_id",
            "required_separation",
            (
                "interpretation_if_separated"
                if explanation_id == "network_local_information"
                else "interpretation_if_not_separated"
            ),
        }
        for explanation_id, explanation in explanations.items()
    )
    assert explanations["field_wide_attention_or_common_opportunity"][
        "required_separation"
    ] == [
        "CONDITION_ON_INCUMBENT_ONLY_FIELD_ATTENTION",
        "MATCHED_NON_NEIGHBOR_GATE",
        "MATCHED_NON_SURGE_TOPIC_GATE",
    ]
    assert explanations["selection_or_homophily"]["required_separation"] == [
        "PRE_TIE_TOPIC_SIMILARITY_AND_TRAJECTORY_DIAGNOSTICS",
        "FUTURE_TIE_GATE",
        "DEGREE_PRESERVING_SHUFFLED_NETWORK_GATE",
    ]
    assert explanations["recruitment_or_capacity_transfer"][
        "required_separation"
    ] == [
        "CHAPERONED_VERSUS_INDEPENDENT_ENTRY_CONTRAST",
        "CAPACITY_BRIDGE_CONDITIONAL_CONTRAST",
    ]
    assert explanations["network_local_information"]["required_separation"] == [
        "PUBLIC_SUCCESS_INCREMENT_BEYOND_PEER_ACTIVITY",
        "INDEPENDENT_ENTRY_INCREMENT",
        "ALL_SEVEN_NEGATIVE_CONTROL_GATES",
    ]

    capacity = registry["capacity_bridge_conditional_contrast"]
    assert set(capacity) == {
        "contrast_id",
        "bridge_rule",
        "focal_entry_work_allowed_in_bridge_construction",
        "conditioning_set",
        "primary_cells",
        "primary_outcome",
        "interaction_required",
        "causal_capacity_effect_identified",
        "failure_action",
    }
    assert capacity["focal_entry_work_allowed_in_bridge_construction"] is False
    assert capacity["conditioning_set"] == [
        "PEER_ACTIVE",
        "PEER_VISIBLE_SUCCESS",
        "INCUMBENT_ONLY_FIELD_ATTENTION",
        "PRE_ENTRY_RESEARCHER_AND_NETWORK_COVARIATES",
    ]
    assert capacity["primary_cells"] == [
        "NEITHER",
        "PUBLIC_SUCCESS_ONLY",
        "CAPACITY_BRIDGE_ONLY",
        "BOTH",
    ]
    assert capacity["primary_outcome"] == (
        "FIXED_HORIZON_TARGET_TOPIC_PERSISTENCE"
    )
    assert capacity["interaction_required"] is True
    assert capacity["causal_capacity_effect_identified"] is False
    assert capacity["failure_action"] == (
        "COLLAPSE_CHANNELS_AND_ABANDON_SEPARATE_PUBLIC_SUCCESS_VERSUS_CAPACITY_CLAIM"
    )


def test_peer_registry_separates_chaperoned_and_independent_entry() -> None:
    strata = _by_id(_registry()["entry_mode_strata"], "stratum_id")

    assert set(strata) == {
        "chaperoned_entry",
        "independent_entry",
        "unresolved_entry_mode",
    }
    assert all(
        set(stratum)
        == {"stratum_id", "rule", "interpretation", "social_learning_sufficient"}
        for stratum in strata.values()
    )
    assert strata["chaperoned_entry"]["rule"] == (
        "FIRST_QUALIFYING_TARGET_TOPIC_WORK_INCLUDES_ANY_EXPOSURE_CONTRIBUTING_PEER"
    )
    assert strata["independent_entry"]["rule"] == (
        "FIRST_QUALIFYING_TARGET_TOPIC_WORK_INCLUDES_NO_EXPOSURE_CONTRIBUTING_PEER"
    )
    assert strata["independent_entry"]["interpretation"] == (
        "RELATIONALLY_INDEPENDENT_OF_EXPOSING_PEERS_NOT_PROOF_OF_UNOBSERVED_INFORMATION_TRANSFER"
    )
    assert all(stratum["social_learning_sufficient"] is False for stratum in strata.values())


def test_peer_registry_freezes_required_placebos_and_their_claim_blocks() -> None:
    controls = _by_id(_registry()["negative_controls"], "control_id")

    assert set(controls) == {
        "lead_attention",
        "same_peer_unrelated_topic_success",
        "post_decision_future_citation_success",
        "future_tie",
        "degree_preserving_shuffled_network",
        "matched_non_neighbor",
        "matched_non_surge_topic",
    }
    assert len(controls) == 7
    base_control_keys = {
        "control_id",
        "exposure_time",
        "construction",
        "required_gate",
        "not_merely",
        "failure_interpretation",
        "blocks_claims",
    }
    assert set(controls["post_decision_future_citation_success"]) == {
        *base_control_keys,
        "feature_use",
    }
    assert all(
        set(control) == base_control_keys
        for control_id, control in controls.items()
        if control_id != "post_decision_future_citation_success"
    )
    unrelated = controls["same_peer_unrelated_topic_success"]
    assert unrelated["exposure_time"] == "AT_OR_BEFORE_FOCAL_DECISION_CUTOFF"
    assert unrelated["required_gate"] == "EQUIVALENCE_TO_FROZEN_NULL_REGION"
    assert "TOPIC_SPECIFIC_VISIBLE_SUCCESS_CHANNEL" in unrelated["blocks_claims"]

    future = controls["post_decision_future_citation_success"]
    assert future["exposure_time"] == "STRICTLY_AFTER_FOCAL_DECISION_CUTOFF"
    assert future["feature_use"] == (
        "PLACEBO_ONLY_NEVER_PRIMARY_OR_ROBUSTNESS_EXPOSURE"
    )
    assert future["required_gate"] == "EQUIVALENCE_TO_FROZEN_NULL_REGION"
    assert "TEMPORALLY_VALID_VISIBLE_SUCCESS_CHANNEL" in future["blocks_claims"]

    lead = controls["lead_attention"]
    assert lead["exposure_time"] == "T_PLUS_1_OR_T_PLUS_2_RELATIVE_TO_ENTRY_AT_T"
    assert lead["required_gate"] == "EQUIVALENCE_TO_FROZEN_NULL_REGION"
    assert "DELAYED_FEEDBACK_MECHANISM" in lead["blocks_claims"]

    non_surge = controls["matched_non_surge_topic"]
    assert non_surge["exposure_time"] == "SAME_FIELD_AND_CALENDAR_PERIOD"
    assert non_surge["required_gate"] == "EQUIVALENCE_TO_FROZEN_NULL_REGION"
    assert "ATTENTION_SURGE_SPECIFICITY" in non_surge["blocks_claims"]

    assert all(
        control["not_merely"] == "NONSIGNIFICANT_NULL_HYPOTHESIS_TEST"
        for control in controls.values()
    )


def test_peer_registry_claim_downgrades_cover_competing_explanations() -> None:
    registry = _registry()
    rules = _by_id(registry["claim_downgrade_rules"], "rule_id")
    levels = registry["claim_levels"]

    assert levels == [
        "DESCRIPTIVE_TIMING_ONLY",
        "PEER_ACTIVITY_ASSOCIATION_NONCAUSAL",
        "TOPIC_SPECIFIC_VISIBLE_SUCCESS_ASSOCIATION_NONCAUSAL",
        "NETWORK_LOCAL_INFORMATION_CHANNEL_COMPATIBLE_NONCAUSAL",
        "CAUSAL_SOCIAL_LEARNING_EXTERNAL_AXIS_ONLY",
    ]
    assert set(rules) == {
        "peer_active_increment_not_established",
        "visible_success_increment_not_established",
        "visible_success_confined_to_chaperoned_entry",
        "unrelated_topic_control_fails",
        "future_citation_control_fails",
        "network_locality_control_fails",
        "temporal_or_non_surge_control_fails",
        "capacity_bridge_not_separable",
        "independent_visible_success_survives_openalex_gates",
        "external_causal_axis_absent",
    }
    assert all(
        set(rule)
        == {"rule_id", "trigger", "claim_ceiling", "required_interpretation"}
        for rule in rules.values()
    )
    assert rules["visible_success_confined_to_chaperoned_entry"]["claim_ceiling"] == (
        "PEER_ACTIVITY_ASSOCIATION_NONCAUSAL"
    )
    assert rules["unrelated_topic_control_fails"]["required_interpretation"] == (
        "GENERIC_PEER_PRESTIGE_ACCESS_OR_SELECTION_NOT_TARGET_TOPIC_LEARNING"
    )
    assert rules["future_citation_control_fails"]["required_interpretation"] == (
        "OBSERVABLE_AT_DECISION_VISIBLE_SUCCESS_CHANNEL_NOT_ESTABLISHED"
    )
    assert rules["network_locality_control_fails"]["claim_ceiling"] == (
        "DESCRIPTIVE_TIMING_ONLY"
    )
    assert rules["temporal_or_non_surge_control_fails"]["claim_ceiling"] == (
        "DESCRIPTIVE_TIMING_ONLY"
    )
    assert rules["capacity_bridge_not_separable"]["required_interpretation"] == (
        "NO_SEPARATE_NETWORK_INFORMATION_VERSUS_CAPACITY_CHANNEL_CLAIM"
    )
    assert rules["independent_visible_success_survives_openalex_gates"][
        "claim_ceiling"
    ] == "NETWORK_LOCAL_INFORMATION_CHANNEL_COMPATIBLE_NONCAUSAL"
    assert rules["external_causal_axis_absent"]["required_interpretation"] == (
        "NO_CAUSAL_SOCIAL_LEARNING_OR_POLICY_EFFECT_CLAIM"
    )
    assert all(rule["claim_ceiling"] in levels for rule in rules.values())
