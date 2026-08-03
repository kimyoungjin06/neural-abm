from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v60.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v60_binds_six_choice_precommitment_evidence() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v60"
    assert gate["status"] == (
        "VALID_SIX_RISK_SET_WEIGHT_SUPPORT_CHOICES_FIXTURE_ONLY_"
        "G0_V4_BINDING_AUTHORIZED"
    )
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_risk_set_weight_support_precommitment"]
    for key in (
        "contract",
        "application_tool",
        "independent_validator",
        "fixture_input",
        "fixture_expected",
        "research_note",
        "summary_json",
        "summary_markdown",
    ):
        binding = verified[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
    summary = json.loads(
        (ROOT / verified["summary_json"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["schema_version"] == verified["summary_json"][
        "required_schema_version"
    ]
    assert summary["status"] == verified["summary_json"]["required_status"]
    assert summary["repository_artifacts"]["contract"]["sha256"] == (
        verified["contract"]["sha256"]
    )
    assert summary["repository_artifacts"]["application_tool"]["sha256"] == (
        verified["application_tool"]["sha256"]
    )
    assert summary["repository_artifacts"]["validator"]["sha256"] == (
        verified["independent_validator"]["sha256"]
    )


def test_gate_v60_records_policy_and_fixture_boundaries() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_risk_set_weight_support_precommitment"]["result"]
    decision = gate["precommitment_decision"]

    assert result["choice_count"] == 6
    assert result["controls_per_case"] == 10
    assert result["weight_percentiles"] == [0.01, 0.99]
    assert result["minimum_cohort"] == [500, 100, 100]
    assert result["suppressed_counts"] == [1, 10]
    assert result["capacity_cell_minimum"] == [50, 11, 11]
    assert result["vif_strict_upper_bound"] == 5.0
    assert result["fixture_capacity_support_fail_count"] == 1
    assert result["fixture_capacity_vif_fail_count"] == 1
    assert result["row_data_library_imports"] == []
    assert result["discovery_production_application_run"] is False
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert decision["values_are_outcome_blind_policy_not_empirical_truth"] is True
    assert decision["values_are_power_or_precision_guarantees"] is False
    assert decision["cms_policy_applies_to_openalex"] is False
    assert decision["remaining_quantitative_choice_count_after_binding"] == 7


def test_gate_v60_authorizes_only_exact_g0_v4_binding() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_g0_v4_binding_scope"]

    assert scope["exact_choice_ids"] == [
        "negative_sampling_ratio",
        "inverse_probability_weight_truncation",
        "minimum_cohort_size",
        "small_cell_suppression_threshold",
        "capacity_cell_support_threshold",
        "collinearity_threshold",
    ]
    for key in (
        "must_copy_g0_v3_except_versioned_parent_and_exact_six_choice_binding",
        "must_bind_g0_v3_byte_for_byte",
        "must_bind_risk_set_weight_support_contract_byte_for_byte",
        "must_bind_this_gate_byte_for_byte",
    ):
        assert scope[key] is True
    for key in (
        "may_resolve_other_quantitative_choices",
        "may_complete_required_sections",
        "may_apply_rules_to_discovery_production_data",
        "may_read_confirmatory_or_holdout_rows",
        "may_change_temporal_partitions",
        "may_select_g4_field_or_episode",
        "may_materialize_g4_artifacts",
        "may_create_holdout_preflight_or_seal",
        "may_pass_g0",
        "may_fit_model",
        "may_open_g4",
    ):
        assert scope[key] is False

    assert gate["gate_state"]["G0"]["complete"] is False
    assert gate["gate_state"]["G4"]["state"] == (
        "LOGICALLY_RESERVED_NOT_BYTE_SEALED_NOT_OPENED"
    )
    authorization = gate["authorization"]
    assert authorization["g0_v4_six_choice_binding_only_successor"] == (
        "AUTHORIZED_EXACT_SCOPE"
    )
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
