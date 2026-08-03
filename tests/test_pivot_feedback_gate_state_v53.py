from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v53.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v53_binds_validated_g0_v2_binding_successor() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v53"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_g0_v2_binding_only_successor"]
    for key in (
        "config",
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
    assert summary["repository_artifacts"]["config"]["sha256"] == (
        verified["config"]["sha256"]
    )
    assert summary["repository_artifacts"]["validator"]["sha256"] == (
        verified["validator"]["sha256"]
    )


def test_gate_v53_records_only_g_minus_1_absence_resolved() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_g0_v2_binding_only_successor"]["result"]
    decision = gate["g0_v2_binding_decision"]

    assert result["repository_documents_read"] == 7
    assert result["operator_parquet_files_opened"] == 0
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert result["api_requests_created_or_submitted"] == 0
    assert result["predecessor_preserved_subtree_count"] == 8
    assert result["predecessor_mutated"] is False
    assert result["legacy_g_minus_1_state"] == (
        "PENDING_NOT_PASSED_IMMUTABLE_LINEAGE"
    )
    assert result["legacy_g_minus_1_passed"] is False
    assert result["g_minus_1a_state"] == (
        "SATISFIED_G_MINUS_1A_BOUNDED_EVIDENCE_ONLY"
    )
    assert result["resolved_blocker"] == "G_MINUS_1_RECEIPT_ABSENT"
    assert result["remaining_blocker_count"] == 3
    assert result["holdout_binding_state"] == "PENDING_NULL_BINDING"
    assert result["unresolved_quantitative_choice_count"] == 17
    assert result["not_freeze_ready_section_count"] == 15
    assert result["pass_g0"] is False

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VALIDATED"
    assert decision["g_minus_1a_bounded_prerequisite_satisfied"] is True
    assert decision["legacy_g_minus_1_passed"] is False
    assert decision["g0_freeze_ready"] is False
    assert decision["next_workstream"] == "HOLDOUT_ACCESS_SEAL_RECEIPT_CLOSURE"


def test_gate_v53_authorizes_only_metadata_holdout_contract() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_holdout_seal_contract_scope"]

    assert scope["mode"] == (
        "VERSIONED_METADATA_ONLY_HOLDOUT_SEAL_CONTRACT_"
        "NO_SEAL_CREATION_OR_OPENING"
    )
    for key in (
        "must_bind_g0_v2_byte_for_byte",
        "must_bind_prerequisite_closure_plan_byte_for_byte",
        "must_separate_seal_creation_from_seal_opening",
        "must_require_atomic_manifest_and_hash_closure_before_any_confirmatory_read",
        "must_require_receipt_written_last_and_independently_verified",
        "must_fail_closed_on_missing_changed_or_previously_opened_artifact",
        "must_keep_receipt_nonauthorizing_for_model_fitting_or_g4_open",
        "may_define_future_manifest_and_receipt_schemas",
        "independent_validation_required",
        "successor_gate_required_before_seal_creation_or_receipt_issuance",
    ):
        assert scope[key] is True
    assert scope["must_freeze_partition"] == {
        "entry_years": [2020, 2021, 2022],
        "outcome_followup_through_year": 2025,
        "domain_and_author_cold": True,
    }
    for key in (
        "may_create_or_issue_holdout_seal",
        "may_open_holdout_seal",
        "may_enumerate_or_hash_holdout_artifacts",
        "may_read_operator_parquet_or_row_level_data",
        "may_read_confirmatory_or_holdout_rows",
        "may_create_or_submit_api_requests",
        "may_change_g0_v2",
        "may_resolve_quantitative_choices",
        "may_complete_required_sections",
        "may_seal_or_pass_g0",
        "may_fit_or_score_a_model",
        "may_open_g4",
    ):
        assert scope[key] is False

    assert gate["gate_state"]["G0"]["complete"] is False
    authorization = gate["authorization"]
    assert authorization["g0_v2_binding_only_successor"] == (
        "COMPLETE_INDEPENDENTLY_VALIDATED"
    )
    assert authorization["holdout_access_seal_contract"] == (
        "AUTHORIZED_METADATA_ONLY_NO_SEAL_ACTION"
    )
    assert authorization["holdout_seal_creation_or_receipt_issuance"] == (
        "NOT_AUTHORIZED"
    )
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
