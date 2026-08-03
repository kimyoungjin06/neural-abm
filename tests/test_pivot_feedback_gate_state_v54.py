from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v54.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v54_binds_validated_holdout_seal_contract() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v54"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_holdout_access_seal_contract"]
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


def test_gate_v54_records_contract_only_result_and_remaining_g0_hold() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_holdout_access_seal_contract"]["result"]
    decision = gate["holdout_seal_contract_decision"]

    assert result["repository_documents_read"] == 4
    assert result["operator_parquet_files_opened"] == 0
    assert result["holdout_artifact_paths_enumerated"] == 0
    assert result["holdout_artifact_bytes_hashed"] == 0
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert result["partition_entry_years"] == [2020, 2021, 2022]
    assert result["outcome_followup_through_year"] == 2025
    assert result["domain_cold"] is True
    assert result["author_cold"] is True
    assert result["seal_created"] is False
    assert result["receipt_written"] is False
    assert result["pass_g0"] is False

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VALIDATED"
    assert decision["partition_frozen"] is True
    assert decision["creation_and_opening_separate"] is True
    assert decision["seal_created"] is False
    assert decision["g0_freeze_ready"] is False
    assert decision["next_workstream"] == (
        "HOLDOUT_ACCESS_SEAL_CREATION_AND_RECEIPT_ISSUANCE"
    )


def test_gate_v54_authorizes_only_private_hash_seal_creation() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_seal_creation_scope"]

    assert scope["mode"] == (
        "PRIVATE_OPERATOR_EXPLICIT_ALLOWLIST_BYTE_HASH_SEAL_CREATION_"
        "NO_ROW_PARSING_OR_OPENING"
    )
    for key in (
        "must_bind_contract_byte_for_byte",
        "must_bind_g0_v2_byte_for_byte",
        "must_bind_prerequisite_closure_plan_byte_for_byte",
        "must_bind_this_gate_byte_for_byte",
        "explicit_operator_artifact_allowlist_required",
        "operator_preflight_required",
        "private_staging_directory_required",
        "may_enumerate_and_hash_only_explicitly_allowlisted_artifacts",
        "must_reject_missing_changed_nonregular_symlinked_duplicate_or_previously_opened_artifacts",
        "must_use_atomic_manifest_and_directory_finalization",
        "must_revalidate_final_hash_closure_before_receipt",
        "must_write_receipt_last_and_create_only",
        "independent_read_only_verification_required",
        "may_create_or_issue_holdout_seal",
        "successor_gate_required_before_seal_opening",
    ):
        assert scope[key] is True
    for key in (
        "may_parse_or_analyze_holdout_rows",
        "may_open_holdout_seal",
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
    assert authorization["holdout_access_seal_contract"] == (
        "COMPLETE_INDEPENDENTLY_VALIDATED"
    )
    assert authorization["holdout_seal_creation_or_receipt_issuance"] == (
        "AUTHORIZED_CONDITIONAL_EXPLICIT_ALLOWLIST_AND_PREFLIGHT"
    )
    assert authorization["holdout_seal_opening"] == "NOT_AUTHORIZED"
    assert authorization["confirmatory_or_holdout_row_read"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
