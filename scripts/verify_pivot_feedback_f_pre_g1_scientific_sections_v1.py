#!/usr/bin/env python3
"""Independently replay the thirteen outcome-blind pre-G1 sections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT / "experiments/configs/pivot_feedback_f_pre_g1_scientific_sections_v1.yaml"
)

EXPECTED_SOURCES = {
    "authorization_parent": (
        None,
        "experiments/configs/pivot_feedback_current_state_v1.yaml",
        "pivot_feedback_current_state.v1",
    ),
    "g0_v4": (
        "bound_design_sources",
        "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml",
        "pivot_feedback_g0_measurement_freeze.v5",
    ),
    "g1_tolerance": (
        "bound_design_sources",
        "experiments/configs/pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml",
        "pivot_feedback_g0_g1_replication_tolerance_precommitment.v1",
    ),
    "peer_registry": (
        "bound_design_sources",
        "experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml",
        "pivot_feedback_peer_mechanism_registry.v1",
    ),
    "cross_field_matrix": (
        "bound_design_sources",
        "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml",
        "pivot_feedback_cross_field_matrix.v1",
    ),
    "measurement_proxy_policy": (
        "bound_design_sources",
        "experiments/configs/pivot_feedback_g0_measurement_proxy_precommitment_v1.yaml",
        "pivot_feedback_g0_measurement_proxy_precommitment.v1",
    ),
    "risk_weight_support_policy": (
        "bound_design_sources",
        "experiments/configs/pivot_feedback_g0_risk_set_weight_support_precommitment_v1.yaml",
        "pivot_feedback_g0_risk_set_weight_support_precommitment.v1",
    ),
    "primary_pivot_definition": (
        "bound_design_sources",
        "experiments/configs/pivot_feedback_researcher_pivot_definition_v2.yaml",
        "pivot_feedback_researcher_pivot_definition.v2",
    ),
    "predecision_measurement": (
        "bound_design_sources",
        "experiments/configs/pivot_feedback_d0_certified_predecision_author_measurement_v1.yaml",
        "pivot_feedback_d0_certified_predecision_author_measurement_config.v1",
    ),
    "persistence_contract": (
        "bound_design_sources",
        "experiments/configs/pivot_feedback_d0_persistence_outcome_contract_v1.yaml",
        "pivot_feedback_d0_persistence_outcome_contract.v1",
    ),
    "empirical_protocol": (
        "bound_design_sources",
        "docs/case-studies/researcher-pivot/pivot-feedback-empirical-nabm-protocol.md",
        None,
    ),
}

EXPECTED_SECTIONS = [
    "population_and_author_eligibility",
    "pivot_and_robustness_definitions",
    "topic_attention_and_surge",
    "temporal_availability_and_maturation",
    "peer_exposure_and_source_assignment",
    "capacity_bridge_and_entry_mode",
    "risk_set_sampling_and_weights",
    "outcome_horizons_and_h3_timing",
    "negative_control_decision_gates",
    "minimum_support_and_suppression",
    "partition_and_atomic_fold_resolution",
    "estimators_and_uncertainty",
    "hierarchical_multiplicity",
]


class SectionFreezeVerificationError(RuntimeError):
    """Raised when a section freeze violates its bound scientific contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SectionFreezeVerificationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    require(path.is_file(), f"missing file {path.name}")
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _load_sources(contract: Mapping[str, Any]) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    for name, (container_name, relative_path, schema) in EXPECTED_SOURCES.items():
        container = contract if container_name is None else _mapping(
            contract.get(container_name), container_name
        )
        binding = _mapping(container.get(name), name)
        require(binding.get("relative_path") == relative_path, f"{name} path")
        path = (ROOT / relative_path).resolve()
        require(path.is_relative_to(ROOT), f"{name} path escape")
        require(binding.get("sha256") == _sha256(path), f"{name} hash")
        require(binding.get("mutation_allowed") is False, f"{name} immutable")
        if schema is None:
            documents[name] = path.read_text(encoding="utf-8")
        else:
            document = _load_yaml(path)
            require(document.get("schema_version") == schema, f"{name} schema")
            require(binding.get("required_schema_version") == schema, f"{name} required schema")
            documents[name] = document
    parent = documents["authorization_parent"]
    binding = contract["authorization_parent"]
    require(parent.get("status") == binding.get("required_status"), "parent status")
    return documents


def _validate_inventory(
    contract: Mapping[str, Any], documents: Mapping[str, Any]
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    require(contract.get("section_order") == EXPECTED_SECTIONS, "section order")
    sections = _mapping(contract.get("sections"), "sections")
    require(list(sections) == EXPECTED_SECTIONS, "section keys")
    require(
        all(_mapping(sections[item], item).get("freeze_state") == "FROZEN_OUTCOME_BLIND" for item in EXPECTED_SECTIONS),
        "section freeze states",
    )
    parent_sections = documents["authorization_parent"]["staged_section_registry"]
    require(
        parent_sections["before_e_confirmatory_access"]["section_ids"]
        == EXPECTED_SECTIONS,
        "parent pre-G1 sections",
    )
    require("g2_predictive_gate" not in sections, "G2 section entered F")
    require("g3_simulation_gate" not in sections, "G3 section entered F")
    return sections, {
        "required_section_count": 13,
        "frozen_section_count": 13,
        "later_stage_sections_included": 0,
        "outcome_blind": True,
    }


def _validate_population_and_pivot(
    sections: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    population = sections["population_and_author_eligibility"]
    opportunity = population["opportunity_rule"]
    predecision = documents["predecision_measurement"]["registered_author_membership"]
    require(
        opportunity["required_predecision_activity"]
        == "AT_LEAST_ONE_ELIGIBLE_FOCAL_FIELD_WORK_IN_T_MINUS_3_THROUGH_T_MINUS_1",
        "opportunity activity",
    )
    require(predecision["upstream_structural_predicate"]["value"] == 1, "D0 activity floor")
    require(opportunity["minimum_history_work_count_beyond_activity_rule"] == "NONE", "history threshold")
    require(population["author_token_is_natural_person"] is False, "author-token boundary")
    require(population["eligible_work_type_codes"] == [1, 2, 3, 4, 7, 16, 18, 21], "work types")

    pivot = sections["pivot_and_robustness_definitions"]
    source_pivot = documents["primary_pivot_definition"]["primary_event"]
    require(pivot["primary"]["event_state"] == source_pivot["pivot_state"], "primary pivot state")
    require(pivot["primary"]["minimum_history_threshold"] == "NONE", "primary history threshold")
    alternative = pivot["substantively_different_alternative"]
    require(alternative["taxonomy_level"] == "CURRENT_SNAPSHOT_SUBFIELD", "alternative level")
    require(alternative["trailing_window_offsets"] == [-3, -2, -1], "alternative window")
    require(alternative["older_target_subfield_history_allowed"] is True, "alternative substantiveness")
    require(alternative["may_replace_primary_after_outcome_read"] is False, "primary substitution")
    return {
        "author_unit": "CURRENT_SNAPSHOT_OPENALEX_AUTHOR_TOKEN",
        "minimum_history_threshold": "NONE",
        "primary_pivot": "OBSERVABLE_HISTORY_NEW_PRIMARY_TOPIC_ENTRY_V2",
        "alternative_pivot": "TRAILING_THREE_YEAR_BROAD_SUBFIELD_ENTRY_V1",
        "true_lifetime_pivot_claim": False,
    }


def _validate_measurement_rules(
    sections: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    policy = documents["measurement_proxy_policy"]["precommitment_choices"]
    attention = sections["topic_attention_and_surge"]
    bound_attention = policy["attention_surge_threshold"]
    require(attention["rule_id"] == bound_attention["rule_id"], "attention rule")
    require(attention["recent_window_offsets"] == bound_attention["recent_window_offsets"], "attention recent window")
    require(attention["reference_window_offsets"] == bound_attention["reference_window_offsets"], "attention reference window")
    require(attention["surge_threshold_inclusive"] == 3.0, "attention threshold")
    require(attention["threshold_relaxation_for_coverage"] is False, "attention relaxation")

    temporal = sections["temporal_availability_and_maturation"]
    require(temporal["same_decision_year_information_as_feature"] == "FORBIDDEN", "same-year leakage")
    require(temporal["citation_success_proxy"]["horizon_complete_calendar_years"] == 3, "citation horizon")
    require(temporal["persistence_label"]["horizon_offsets"] == [1, 2, 3], "persistence horizon")
    require(
        temporal["holdout_rule"]
        == "MATERIALIZATION_AND_HASHING_MAY_OCCUR_OPERATOR_ONLY_BUT_ANALYTIC_READ_REQUIRES_H_ENTRY",
        "holdout access boundary",
    )
    return {
        "attention_rule": bound_attention["rule_id"],
        "attention_modified_z_threshold": 3.0,
        "citation_maturation_years": 3,
        "persistence_offsets": [1, 2, 3],
        "same_year_feature_leakage_allowed": False,
    }


def _validate_peer_capacity_and_controls(
    sections: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    peer = sections["peer_exposure_and_source_assignment"]
    require(peer["future_ties_allowed_in_primary"] is False, "future ties")
    require(peer["peer_visible_success"]["subset_of_peer_active"] is True, "nested peer contrast")
    require(peer["source_assignment"]["canonical_source_is_not_exclusive_exposure_source"] is True, "source aggregation")
    require(
        peer["claim_label"]
        == "NETWORK_LOCAL_PUBLIC_AVAILABILITY_PROXY_NOT_RECEIPT_BELIEF_UPDATE_OR_CAUSAL_LEARNING",
        "peer claim boundary",
    )

    capacity = sections["capacity_bridge_and_entry_mode"]
    require(capacity["capacity_bridge"]["focal_entry_work_may_define_bridge"] is False, "capacity leakage")
    require(capacity["four_cells"] == ["NEITHER", "PUBLIC_SUCCESS_ONLY", "CAPACITY_BRIDGE_ONLY", "BOTH"], "capacity cells")
    require(capacity["interaction_required"] is True, "capacity interaction")

    controls = sections["negative_control_decision_gates"]
    equivalence_ids = controls["probability_difference_equivalence"]["control_ids"]
    contrast_ids = controls["true_minus_placebo"]["control_ids"]
    registered = [item["control_id"] for item in documents["peer_registry"]["negative_controls"]]
    require(set(equivalence_ids + contrast_ids) == set(registered), "seven control coverage")
    require(len(equivalence_ids) == 5 and len(contrast_ids) == 2, "control split")
    bound_margin = documents["measurement_proxy_policy"]["precommitment_choices"][
        "negative_control_equivalence_margins"
    ]
    require(controls["probability_difference_equivalence"]["practical_null"] == [-0.01, 0.01], "equivalence margin")
    require(bound_margin["symmetric_practical_null_lower"] == -0.01, "bound lower margin")
    require(bound_margin["symmetric_practical_null_upper"] == 0.01, "bound upper margin")
    require(controls["deterministic_constructions"]["shuffled_network"]["component_seed"] == 20260718, "shuffle seed")
    return {
        "peer_active_nested_baseline": True,
        "visible_success_is_increment": True,
        "capacity_cell_count": 4,
        "negative_control_count": 7,
        "equivalence_control_count": 5,
        "true_minus_placebo_control_count": 2,
        "causal_learning_claim_allowed": False,
    }


def _validate_sampling_support(
    sections: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    sampling = sections["risk_set_sampling_and_weights"]
    support = sections["minimum_support_and_suppression"]
    policy = documents["risk_weight_support_policy"]["precommitment_choices"]
    require(sampling["controls_per_case"] == policy["negative_sampling_ratio"]["ratio_controls_per_case"], "control ratio")
    require(sampling["truncation"] == "NEAREST_RANK_P01_P99_WINSORIZATION", "weight truncation")
    require(sampling["full_risk_set_sensitivity_required"] is True, "full risk-set sensitivity")
    cohort_policy = policy["minimum_cohort_size"]
    require(support["cohort_floor"]["minimum_total"] == cohort_policy["minimum_total_opportunity_rows"], "cohort total")
    require(support["cohort_floor"]["minimum_entry_events"] == cohort_policy["minimum_entry_events"], "cohort events")
    require(support["capacity_cell_floor"]["minimum_total"] == 50, "capacity total")
    require(support["capacity_cell_floor"]["minimum_entry_events"] == 11, "capacity events")
    require(support["collinearity"]["strict_upper_bound"] == 5.0, "VIF")
    require(support["reporting_suppression"]["suppress_inclusive_counts"] == [1, 10], "suppression")
    return {
        "controls_per_case": 10,
        "weight_truncation": "P01_P99",
        "cohort_floor": [500, 100, 100],
        "capacity_cell_floor": [50, 11, 11],
        "vif_strict_upper_bound": 5.0,
        "suppressed_counts": [1, 10],
    }


def _validate_h3_estimation_and_multiplicity(
    sections: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    timing = sections["outcome_horizons_and_h3_timing"]
    h3 = timing["h3"]
    tolerance = documents["g1_tolerance"]["precommitment_choices"]["g1_replication_tolerance"]
    require(h3["event_time_grid"] == tolerance["estimand"]["episode_value_grid_cohort_years"], "H3 grid")
    require(h3["primary_minimum_point_cohort_years"] == 1.0, "H3 point minimum")
    require(h3["minimum_independent_episodes_per_field"] == 3, "H3 episode floor")
    replication = timing["replication_noninferiority"]
    require(replication["lower_margin_cohort_years"] == -1.0, "replication margin")
    require(replication["boundary_counts_as_pass"] is False, "replication strictness")

    estimator = sections["estimators_and_uncertainty"]
    require(estimator["primary_probability_model"]["family"] == "BINOMIAL_LOGIT", "estimator family")
    require(estimator["uncertainty"]["procedure"] == "FIELD_STRATIFIED_EPISODE_CLUSTER_BOOTSTRAP", "uncertainty procedure")
    require(estimator["uncertainty"]["bootstrap_replicates"] == 9999, "bootstrap count")
    require(estimator["uncertainty"]["component_seed"] == 20260718, "bootstrap seed")
    require(estimator["missingness"]["outcome_imputation"] == "NONE", "outcome imputation")

    multiplicity = sections["hierarchical_multiplicity"]
    require(multiplicity["familywise_alpha"] == 0.05, "familywise alpha")
    require(multiplicity["stage_1"]["hypothesis"] == "H3_FIELD_BALANCED_TIMING_GAP", "stage 1")
    require(multiplicity["stage_1"]["must_pass_before_stage_2"] is True, "hierarchy")
    require(multiplicity["stage_2"]["hypotheses"] == ["H1_ATTENTION", "H2_VISIBLE_SUCCESS_INCREMENT", "H4_CAPACITY_BRIDGE"], "Holm family")
    require(multiplicity["stage_2"]["method"] == "HOLM_STEP_DOWN", "Holm method")
    return {
        "h3_event_time_grid": [-2, -1, 0, 1, 2],
        "h3_primary_point_minimum": 1.0,
        "replication_margin": -1.0,
        "bootstrap_replicates": 9999,
        "stage_1": "H3",
        "stage_2_holm_hypotheses": ["H1", "H2", "H4"],
        "familywise_alpha": 0.05,
    }


def _validate_partitions(
    sections: Mapping[str, Any], documents: Mapping[str, Any]
) -> dict[str, Any]:
    partition = sections["partition_and_atomic_fold_resolution"]
    observed = partition["temporal_partitions"]
    expected_rows = documents["cross_field_matrix"]["temporal_partitions"]
    expected = {row["partition_id"]: row for row in expected_rows}
    for name, row in observed.items():
        source = expected[name]
        for source_key, local_key in (
            ("entry_start_year", "entry_start_year"),
            ("entry_end_year", "entry_end_year"),
            ("outcome_followup_through_year", "followup_through_year"),
            ("sealed", "sealed"),
        ):
            if local_key in row:
                require(row[local_key] == source[source_key], f"partition {name} {local_key}")
    atomicity = partition["atomicity"]
    require(atomicity["all_response_cohorts_of_episode_stay_together"] is True, "episode atomicity")
    require(atomicity["g4_authors_removed_from_all_earlier_fit_or_selection_rows"] is True, "author cold")
    require(partition["model_selection_folds"]["component_may_be_split"] is False, "component splitting")
    require(partition["holdout_manifest"]["seal_receipt_written_last"] is True, "seal order")
    require(partition["holdout_manifest"]["analytic_open_before_h_entry"] is False, "holdout open")
    return {
        "discovery_end_year": 2012,
        "g1_years": [2013, 2016],
        "model_selection_years": [2017, 2019],
        "g4_entry_years": [2020, 2022],
        "g4_followup_through": 2025,
        "model_selection_fold_count": 5,
        "g4_author_cold": True,
        "seal_written_last": True,
    }


def _validate_fail_closed(contract: Mapping[str, Any]) -> dict[str, Any]:
    completion = _mapping(contract.get("completion_accounting"), "completion")
    require(completion.get("section_count") == 13, "section count")
    require(completion.get("frozen_section_count") == 13, "frozen count")
    require(completion.get("this_component_is_new_gate") is False, "micro-gate")
    require(completion.get("this_component_alone_may_pass_f") is False, "F pass")
    require(completion.get("this_component_alone_may_pass_g0") is False, "G0 pass")
    require(completion.get("final_panel_conformance_pending") is True, "panel pending")
    require(completion.get("atomic_partition_and_holdout_byte_seal_pending") is True, "seal pending")
    require(completion.get("confirmatory_or_holdout_rows_read") == 0, "row reads")
    require(completion.get("model_fitted") is False, "model")
    require(completion.get("g4_open") is False, "G4")

    authorization = _mapping(contract.get("authorization"), "authorization")
    expected = {
        "confirmatory_or_holdout_analytic_access": "NOT_AUTHORIZED",
        "f_passage": "NOT_AUTHORIZED_BY_THIS_COMPONENT_ALONE",
        "g0_passage": "NOT_AUTHORIZED_BY_THIS_COMPONENT_ALONE",
        "e_g1_execution": "NOT_AUTHORIZED",
        "model_fitting": "NOT_AUTHORIZED",
        "g4_open": "NOT_AUTHORIZED",
        "scientific_effect": "NOT_ESTABLISHED",
    }
    for key, value in expected.items():
        require(authorization.get(key) == value, f"authorization {key}")
    return {
        "new_gate_created": False,
        "pass_f": False,
        "pass_g0": False,
        "final_panel_conformance_pending": True,
        "holdout_byte_seal_pending": True,
        "confirmatory_or_holdout_rows_read": 0,
        "model_fitted": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def verify(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = _load_yaml(path)
    require(contract.get("schema_version") == "pivot_feedback_f_pre_g1_scientific_sections.v1", "schema")
    require(
        contract.get("status")
        == "FROZEN_THIRTEEN_PRE_G1_SECTIONS_PENDING_INDEPENDENT_VALIDATION_NO_F_PASS",
        "status",
    )
    documents = _load_sources(contract)
    sections, inventory = _validate_inventory(contract, documents)
    population = _validate_population_and_pivot(sections, documents)
    measurement = _validate_measurement_rules(sections, documents)
    peer = _validate_peer_capacity_and_controls(sections, documents)
    sampling = _validate_sampling_support(sections, documents)
    h3 = _validate_h3_estimation_and_multiplicity(sections, documents)
    partitions = _validate_partitions(sections, documents)
    fail_closed = _validate_fail_closed(contract)
    return {
        "schema_version": "pivot_feedback_f_pre_g1_scientific_sections_summary.v1",
        "status": "PASS_THIRTEEN_PRE_G1_SCIENTIFIC_SECTIONS_FROZEN_NO_F_PASS",
        "section_inventory_replay": inventory,
        "population_and_pivot_replay": population,
        "measurement_and_maturation_replay": measurement,
        "peer_capacity_and_control_replay": peer,
        "sampling_support_replay": sampling,
        "h3_estimation_multiplicity_replay": h3,
        "partition_replay": partitions,
        "fail_closed_replay": fail_closed,
        "access_accounting": {
            "repository_yaml_documents_read": 10,
            "repository_text_documents_read": 1,
            "operator_or_row_level_files_read": 0,
            "discovery_rows_read": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "repository_artifacts": {
            "section_contract": {
                "relative_path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
        },
        "authorization": {
            "section_freeze": "COMPLETE_INDEPENDENTLY_VERIFIED",
            "next_f_work": "FINAL_PANEL_CONFORMANCE_THEN_OPERATOR_ONLY_PARTITION_AND_SEAL",
            "confirmatory_or_holdout_analytic_access": "NOT_AUTHORIZED",
            "f_pass": "NOT_AUTHORIZED_BY_THIS_COMPONENT_ALONE",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "Thirteen outcome-blind pre-G1 scientific sections are frozen and "
            "independently replayed as one F-phase provenance component, not a new "
            "gate. Final-panel conformance and the atomic private holdout byte seal "
            "remain pending; F and G0 do not pass, no confirmatory or holdout row "
            "was read, G1 was not run, no model was fitted, G4 remains unopened, "
            "and no scientific effect is established."
        ),
    }


def main() -> int:
    try:
        result = verify()
    except (SectionFreezeVerificationError, OSError, ValueError, yaml.YAMLError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
