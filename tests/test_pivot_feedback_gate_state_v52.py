from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v52.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v52_binds_validated_g_minus_1a_reconciliation() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v52"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_g_minus_1a_binding_reconciliation"]
    for key in (
        "contract",
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
    assert summary["repository_artifacts"]["contract"]["sha256"] == (
        verified["contract"]["sha256"]
    )
    assert summary["repository_artifacts"]["validator"]["sha256"] == (
        verified["validator"]["sha256"]
    )


def test_gate_v52_records_exact_no_relabel_boundary() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_g_minus_1a_binding_reconciliation"]["result"]
    decision = gate["binding_reconciliation_decision"]

    assert result["repository_documents_read"] == 9
    assert result["operator_parquet_files_opened"] == 0
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert result["api_requests_created_or_submitted"] == 0
    assert result["mismatch_class"] == (
        "CONTRACT_GENERATION_MISMATCH_NOT_EVIDENCE_ABSENCE"
    )
    assert result["schema_aliasing_allowed"] is False
    assert result["status_relabeling_allowed"] is False
    assert result["legacy_pass_inference_allowed"] is False
    assert result["future_field_name"] == "g_minus_1a_binding"
    assert result["receipt_unresolved_boundary_count"] == 14
    assert result["receipt_claim_boundary_preserved_exactly"] is True
    assert result["g0_v1_preserved_subtree_count"] == 8
    assert result["permitted_semantic_delta_count"] == 2
    assert result["unresolved_quantitative_choice_count"] == 17
    assert result["not_freeze_ready_section_count"] == 15
    assert result["future_successor_authored"] is False
    assert result["legacy_g_minus_1_passed"] is False
    assert result["pass_g0"] is False

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VALIDATED"
    assert decision["exact_g_minus_1a_receipt_binding_ready"] is True
    assert decision["receipt_semantics_preserved"] is True
    assert decision["legacy_g_minus_1_status"] == "G_MINUS_1_PENDING"
    assert decision["legacy_g_minus_1_passed"] is False
    assert decision["g0_freeze_ready"] is False


def test_gate_v52_authorizes_only_g0_v2_binding_successor() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_g0_v2_binding_only_successor_scope"]

    assert scope["mode"] == (
        "VERSIONED_G0_V2_BINDING_ONLY_SUCCESSOR_NO_OTHER_G0_CLOSURE"
    )
    for key in (
        "must_bind_g0_v1_byte_for_byte",
        "must_bind_reconciliation_contract_byte_for_byte",
        "may_add_version_metadata_and_parent_bindings",
        "may_retain_legacy_binding_as_pending_immutable_lineage",
        "may_add_exact_g_minus_1a_binding",
        "may_mark_g_minus_1a_bounded_prerequisite_satisfied",
        "must_keep_holdout_binding_pending_and_null",
        "must_keep_pass_g0_false",
        "must_keep_model_fitting_unauthorized",
        "must_keep_g4_unopened",
        "independent_validation_required",
        "successor_gate_required_before_next_closure_workstream",
    ):
        assert scope[key] is True
    assert scope["may_remove_only_readiness_blocker"] == (
        "G_MINUS_1_RECEIPT_ABSENT"
    )
    assert scope["must_keep_unresolved_quantitative_choice_count"] == 17
    assert scope["must_keep_not_freeze_ready_section_count"] == 15
    for key in (
        "may_change_any_other_measurement_or_design_subtree",
        "may_relabel_g_minus_1a_as_legacy_g_minus_1",
        "may_promote_legacy_g_minus_1",
        "may_reissue_or_modify_g_minus_1a_receipt",
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
    assert authorization["g_minus_1a_binding_reconciliation_contract"] == (
        "COMPLETE_INDEPENDENTLY_VALIDATED"
    )
    assert authorization["g0_v2_binding_only_successor"] == (
        "AUTHORIZED_NO_OTHER_G0_CLOSURE"
    )
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
