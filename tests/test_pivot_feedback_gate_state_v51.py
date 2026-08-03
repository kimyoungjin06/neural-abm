from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v51.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v51_binds_validated_prerequisite_closure_plan() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v51"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_g0_prerequisite_closure_plan"]
    for key in (
        "plan",
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
    assert summary["repository_artifacts"]["plan"]["sha256"] == (
        verified["plan"]["sha256"]
    )
    assert summary["repository_artifacts"]["validator"]["sha256"] == (
        verified["validator"]["sha256"]
    )


def test_gate_v51_records_contract_mismatch_not_evidence_absence() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_g0_prerequisite_closure_plan"]["result"]
    decision = gate["closure_plan_decision"]

    assert result["repository_documents_read"] == 9
    assert result["operator_json_metadata_files_read"] == 0
    assert result["operator_parquet_files_opened"] == 0
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert result["api_requests_created_or_submitted"] == 0
    assert result["workstream_count"] == 4
    assert result["workstream_dag_acyclic"] is True
    assert result["passing_g_minus_1a_receipt_exists"] is True
    assert result["g_minus_1_finding"] == (
        "CONTRACT_GENERATION_MISMATCH_NOT_EVIDENCE_ABSENCE"
    )
    assert result["legacy_g_minus_1_remains_pending"] is True
    assert result["unresolved_quantitative_choice_count"] == 17
    assert result["newly_resolved_value_count"] == 0
    assert result["required_section_count"] == 15
    assert result["freeze_ready_required_section_count"] == 0
    assert result["actual_g0_v2_binding_change_authorized"] is False
    assert result["pass_g0"] is False

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VALIDATED"
    assert decision["first_blocker_diagnosis"] == (
        "CONTRACT_GENERATION_MISMATCH_NOT_EVIDENCE_ABSENCE"
    )
    assert decision["legacy_g_minus_1_status"] == "G_MINUS_1_PENDING"
    assert decision["legacy_g_minus_1_may_be_promoted"] is False
    assert decision["g0_v1_may_be_mutated"] is False


def test_gate_v51_authorizes_only_binding_reconciliation_contract() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_g_minus_1a_binding_reconciliation_contract_scope"]

    assert scope["mode"] == (
        "VERSIONED_METADATA_ONLY_G_MINUS_1A_BINDING_RECONCILIATION_"
        "CONTRACT_NO_G0_V2_CHANGE"
    )
    for key in (
        "must_bind_g0_v1_skeleton_byte_for_byte",
        "must_bind_closure_plan_byte_for_byte",
        "must_bind_existing_g_minus_1a_receipt_config_and_verifier",
        "must_use_exact_g_minus_1a_schema_and_status_without_relabeling",
        "must_preserve_current_snapshot_proxy_and_author_token_caveats",
        "must_record_legacy_g_minus_1_as_not_passed",
        "must_keep_final_cutoff_panel_owned_by_g0_5",
        "may_define_future_versioned_g0_successor_binding",
        "independent_validation_required",
        "successor_gate_required_before_any_g0_v2_change",
    ):
        assert scope[key] is True
    for key in (
        "may_mutate_g0_v1",
        "may_write_or_modify_g0_v2",
        "may_promote_legacy_g_minus_1",
        "may_reissue_g_minus_1a_receipt",
        "may_write_holdout_seal_contract_or_receipt",
        "may_resolve_quantitative_choices",
        "may_complete_required_sections",
        "may_read_operator_parquet_or_row_level_data",
        "may_read_confirmatory_or_holdout_rows",
        "may_create_or_submit_api_requests",
        "may_seal_or_pass_g0",
        "may_fit_or_score_a_model",
        "may_open_g4",
    ):
        assert scope[key] is False

    assert gate["gate_state"]["G0"]["complete"] is False
    authorization = gate["authorization"]
    assert authorization["g0_prerequisite_closure_plan"] == (
        "COMPLETE_INDEPENDENTLY_VALIDATED"
    )
    assert authorization["g_minus_1a_binding_reconciliation_contract"] == (
        "AUTHORIZED_METADATA_ONLY_NO_G0_V2_CHANGE"
    )
    assert authorization["actual_g0_v2_binding_change"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
