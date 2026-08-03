from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v58.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v58_binds_precommitment_evidence() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v58"
    assert gate["status"] == (
        "VALID_FOUR_MEASUREMENT_PROXY_CHOICES_FIXTURE_ONLY_G0_V3_BINDING_AUTHORIZED"
    )
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_measurement_proxy_precommitment"]
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


def test_gate_v58_records_four_policy_choices_and_fixture_edges() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_measurement_proxy_precommitment"]["result"]
    decision = gate["precommitment_decision"]

    assert result["choice_count"] == 4
    assert result["attention_modified_z_threshold"] == 3.0
    assert result["public_success_target_fraction"] == 0.10
    assert result["public_success_boundary_ties_inclusive"] is True
    assert result["citation_maturation_lag_years"] == 3
    assert result["negative_control_practical_null"] == [-0.01, 0.01]
    assert result["tost_one_sided_alpha"] == 0.05
    assert result["fixture_primary_surge_count"] == 1
    assert result["fixture_mad_zero_group_count"] == 1
    assert result["fixture_unresolved_small_cohort_count"] == 1
    assert result["fixture_inclusive_tie_inflation_count"] == 1
    assert result["row_data_library_imports"] == []
    assert result["discovery_production_application_run"] is False
    assert result["g0_v2_changed"] is False
    assert result["confirmatory_or_holdout_rows_read"] == 0

    assert decision["values_are_outcome_blind_policy_not_empirical_truth"] is True
    assert decision["citation_current_snapshot_bibliographic_proxy_only"] is True
    assert decision["historical_platform_observation_time_established"] is False
    assert decision["researcher_visibility_or_information_receipt_established"] is False
    assert decision["remaining_quantitative_choice_count_after_binding"] == 13


def test_gate_v58_authorizes_only_exact_g0_v3_binding() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_g0_v3_binding_scope"]

    assert scope["exact_choice_ids"] == [
        "attention_surge_threshold",
        "public_success_threshold",
        "citation_maturation_lag_years",
        "negative_control_equivalence_margins",
    ]
    for key in (
        "must_copy_g0_v2_except_versioned_parent_and_exact_four_choice_binding",
        "must_bind_g0_v2_byte_for_byte",
        "must_bind_measurement_proxy_contract_byte_for_byte",
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
    assert authorization["g0_v3_four_choice_binding_only_successor"] == (
        "AUTHORIZED_EXACT_SCOPE"
    )
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
