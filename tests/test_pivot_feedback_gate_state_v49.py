from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v49.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v49_binds_validated_metadata_only_inventory() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v49"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_g0_readiness_inventory"]
    for key in (
        "inventory",
        "validator",
        "research_note",
        "summary_json",
        "summary_markdown",
    ):
        binding = verified[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])

    result = verified["result"]
    assert result["repository_documents_read"] == 9
    assert result["operator_json_metadata_files_read"] == 8
    assert result["operator_parquet_files_opened"] == 0
    assert result["production_row_level_files_opened"] == 0
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert result["original_quantitative_choices"] == 19
    assert result["d0_resolved_quantitative_choices"] == 2
    assert result["unresolved_quantitative_choices"] == 17
    assert result["required_sections"] == 15
    assert result["freeze_ready_sections"] == 0
    assert result["g_minus_1_receipt_state"] == "PENDING_NULL_BINDING"
    assert result["holdout_access_seal_state"] == "PENDING_NULL_BINDING"
    assert result["g0_freeze_ready"] is False
    assert result["model_fitted"] is False
    assert result["g4_open"] is False


def test_gate_v49_authorizes_only_gap_preserving_successor_skeleton() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    decision = gate["g0_readiness_decision"]
    scope = gate["authorized_g0_successor_skeleton_scope"]

    assert decision["status"] == "NOT_READY_FOR_G0_FREEZE"
    assert decision["d0_core_calibration_status"] == (
        "COMPLETE_INDEPENDENTLY_VERIFIED"
    )
    assert decision["resolved_successor_values"] == {
        "author_minimum_history_works": "NONE",
        "primary_pivot_thresholds": (
            "FOCAL_GE_1_AND_PRE_ENTRY_EQ_0_WITH_FOUR_STATE_PRECEDENCE"
        ),
    }
    assert decision["resolved_values_selected_from_outcomes"] is False
    assert decision["unresolved_choice_count"] == 17
    assert decision["not_freeze_ready_section_count"] == 15
    assert decision["old_pending_overlay_may_be_mutated"] is False

    assert scope["mode"] == (
        "VERSIONED_METADATA_ONLY_G0_SUCCESSOR_SKELETON_TWO_RESOLVED_SEVENTEEN_PENDING"
    )
    assert scope["must_bind_old_overlay_byte_for_byte"] is True
    assert scope["may_copy_fixed_design_choices_from_old_overlay"] is True
    assert scope["may_serialize_author_minimum_history_works_as_none"] is True
    assert scope["may_serialize_threshold_free_primary_pivot_rules"] is True
    assert scope["may_mark_any_other_quantitative_choice_resolved"] is False
    assert scope["must_keep_unresolved_quantitative_choice_count"] == 17
    assert scope["must_keep_not_freeze_ready_section_count"] == 15
    assert scope["must_record_g_minus_1_receipt_as_pending"] is True
    assert scope["must_record_holdout_access_seal_as_pending"] is True
    for key in (
        "may_open_operator_parquet_or_row_level_data",
        "may_read_confirmatory_or_holdout_rows",
        "may_create_or_submit_api_requests",
        "may_choose_values_from_observed_outcomes",
        "may_change_confirmatory_periods",
        "may_rank_or_select_fields_episodes_or_thresholds_by_outcome",
        "may_establish_final_candidate_eligibility",
        "may_seal_or_pass_g0",
        "may_fit_or_score_a_model",
        "may_open_g4",
    ):
        assert scope[key] is False

    assert gate["gate_state"]["D0"]["complete"] is True
    assert gate["gate_state"]["G0"]["complete"] is False
    authorization = gate["authorization"]
    assert authorization["g0_readiness_and_gap_inventory"] == (
        "COMPLETE_INDEPENDENTLY_VALIDATED"
    )
    assert authorization["g0_versioned_freeze_successor_skeleton"].startswith(
        "AUTHORIZED_METADATA_ONLY"
    )
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
