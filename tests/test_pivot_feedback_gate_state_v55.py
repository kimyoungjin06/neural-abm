from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v55.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v55_binds_closed_fixture_tooling_evidence() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v55"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_holdout_seal_tooling"]
    for key in (
        "creator",
        "independent_verifier",
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
    assert summary["tooling"]["creator_sha256"] == verified["creator"]["sha256"]
    assert summary["tooling"]["verifier_sha256"] == (
        verified["independent_verifier"]["sha256"]
    )


def test_gate_v55_keeps_production_explicitly_not_run() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    fixture = gate["verified_holdout_seal_tooling"]["closed_fixture_result"]
    production = gate["verified_holdout_seal_tooling"]["production_state"]
    decision = gate["tooling_decision"]

    assert fixture["test_count"] == 10
    assert fixture["artifact_count"] == 2
    assert fixture["artifact_bytes_total"] == 39
    assert fixture["row_data_libraries_imported"] == []
    assert fixture["operator_parquet_files_parsed"] == 0
    assert fixture["confirmatory_or_holdout_rows_read"] == 0
    assert fixture["fail_closed_check_count"] == 9

    assert production["operator_preflight_present"] is False
    assert production["production_artifact_paths_enumerated"] == 0
    assert production["production_artifact_bytes_hashed"] == 0
    assert production["production_seal_created"] is False
    assert production["production_receipt_written"] is False
    assert production["pass_g0"] is False
    assert production["model_fitted"] is False
    assert production["g4_open"] is False

    assert decision["status"] == "COMPLETE_CLOSED_FIXTURE_ONLY"
    assert decision["production_status"] == "NOT_RUN_MISSING_OPERATOR_PREFLIGHT"
    assert decision["g0_freeze_ready"] is False
    assert decision["next_workstream"] == (
        "HOLDOUT_ACCESS_SEAL_OPERATOR_PREFLIGHT_AND_EXECUTION"
    )


def test_gate_v55_authorizes_only_exact_operator_preflight() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_operator_preflight_scope"]

    assert scope["mode"] == (
        "OPERATOR_LOCAL_EXPLICIT_PATH_SIZE_SHA256_ALLOWLIST_NO_ROW_PARSING"
    )
    assert scope["required_schema_version"] == (
        "pivot_feedback_holdout_access_seal_preflight.v1"
    )
    assert scope["required_status"] == (
        "PASS_EXPLICIT_ALLOWLIST_HASHES_BOUND_ZERO_PRIOR_OPEN_EVENTS"
    )
    for key in (
        "must_bind_gate_v54_byte_for_byte",
        "must_bind_holdout_contract_byte_for_byte",
        "must_bind_g0_v2_byte_for_byte",
        "must_bind_prerequisite_closure_plan_byte_for_byte",
        "must_use_sorted_unique_canonical_relative_paths",
        "each_entry_requires_expected_byte_size_and_sha256",
        "must_attest_explicit_artifact_set_complete",
        "must_attest_paths_and_hashes_created_under_v54_authorization",
        "must_attest_private_access_controlled_source_root",
        "must_attest_zero_row_parsing_or_analysis",
        "must_attest_zero_prior_open_events",
        "must_attest_zero_confirmatory_or_holdout_rows_read",
        "production_execution_requires_preflight_file",
    ):
        assert scope[key] is True
    for key in (
        "may_implicitly_discover_artifacts",
        "may_use_wildcards_or_recursive_directory_inputs",
        "may_parse_or_analyze_holdout_rows",
        "may_open_holdout_seal",
        "may_fit_or_score_a_model",
        "may_open_g4",
    ):
        assert scope[key] is False

    assert gate["gate_state"]["G0"]["complete"] is False
    authorization = gate["authorization"]
    assert authorization["operator_preflight_authoring"] == (
        "AUTHORIZED_EXACT_V54_CONTRACT_ONLY"
    )
    assert authorization["production_seal_creation_or_receipt_issuance"] == (
        "CONDITIONAL_NOT_RUN_MISSING_OPERATOR_PREFLIGHT"
    )
    assert authorization["holdout_seal_opening"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
