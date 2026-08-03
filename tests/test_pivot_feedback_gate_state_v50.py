from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v50.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v50_binds_independently_validated_g0_v1_skeleton() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v50"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_g0_v1_successor_skeleton"]
    for key in (
        "skeleton",
        "validator",
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
    assert summary["status"] == verified["summary_json"]["required_status"]
    assert summary["repository_artifacts"]["skeleton"]["sha256"] == (
        verified["skeleton"]["sha256"]
    )
    assert summary["repository_artifacts"]["validator"]["sha256"] == (
        verified["validator"]["sha256"]
    )


def test_gate_v50_records_exact_validated_boundary() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_g0_v1_successor_skeleton"]["result"]
    decision = gate["g0_v1_successor_skeleton_decision"]

    assert result["repository_documents_read"] == 4
    assert result["operator_json_metadata_files_read"] == 0
    assert result["operator_parquet_files_opened"] == 0
    assert result["production_row_level_files_opened"] == 0
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert result["api_requests_created_or_submitted"] == 0
    assert result["immutable_predecessor_bound_byte_for_byte"] is True
    assert result["predecessor_copied_subtree_count"] == 6
    assert result["resolved_measurement_count"] == 2
    assert result["resolved_values_selected_from_observed_outcomes"] is False
    assert result["unresolved_quantitative_choice_count"] == 17
    assert result["unresolved_required_section_count"] == 15
    assert result["freeze_ready_required_section_count"] == 0
    assert result["pending_contract_sealable"] is False
    assert result["pass_g0"] is False
    assert result["model_fitted"] is False
    assert result["g4_open"] is False

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VALIDATED"
    assert decision["g0_freeze_ready"] is False
    assert decision["serialized_successor_values"] == {
        "author_minimum_history_works": "NONE",
        "primary_pivot_thresholds": (
            "FOCAL_GE_1_AND_PRE_ENTRY_EQ_0_WITH_FOUR_STATE_PRECEDENCE"
        ),
    }
    assert decision["serialized_values_selected_from_outcomes"] is False
    assert decision["g0_v1_skeleton_is_a_freeze_or_seal"] is False


def test_gate_v50_authorizes_only_metadata_closure_plan() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_g0_prerequisite_closure_plan_scope"]

    assert scope["mode"] == (
        "VERSIONED_METADATA_ONLY_G0_PREREQUISITE_CLOSURE_PLAN_NO_EXECUTION"
    )
    assert scope["must_separate_workstreams"] == [
        "G_MINUS_1_RECEIPT_CLOSURE",
        "HOLDOUT_ACCESS_SEAL_RECEIPT_CLOSURE",
        "SEVENTEEN_QUANTITATIVE_CHOICE_PRECOMMITMENTS",
        "FIFTEEN_REQUIRED_SECTION_COMPLETION_CONTRACTS",
    ]
    assert scope["must_preserve_g0_v1_skeleton_byte_for_byte"] is True
    assert scope["must_keep_all_current_blockers_explicit"] is True
    for key in (
        "may_open_operator_parquet_or_row_level_data",
        "may_read_confirmatory_or_holdout_rows",
        "may_create_or_submit_api_requests",
        "may_issue_g_minus_1_receipt",
        "may_issue_or_open_holdout_access_seal",
        "may_resolve_any_quantitative_choice",
        "may_complete_any_required_section",
        "may_choose_values_from_observed_outcomes",
        "may_establish_final_candidate_eligibility",
        "may_seal_or_pass_g0",
        "may_fit_or_score_a_model",
        "may_open_g4",
    ):
        assert scope[key] is False

    assert gate["gate_state"]["D0"]["complete"] is True
    assert gate["gate_state"]["G0"]["complete"] is False
    authorization = gate["authorization"]
    assert authorization["g0_versioned_freeze_successor_skeleton"] == (
        "COMPLETE_INDEPENDENTLY_VALIDATED"
    )
    assert authorization["g0_prerequisite_closure_plan"] == (
        "AUTHORIZED_METADATA_ONLY_NO_EXECUTION"
    )
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
