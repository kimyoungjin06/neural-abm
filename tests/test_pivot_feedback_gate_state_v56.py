from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v56.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v56_binds_preflight_tooling_evidence() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v56"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_preflight_tooling"]
    for key in (
        "preflight_author",
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
    assert summary["tooling"]["preflight_author_sha256"] == (
        verified["preflight_author"]["sha256"]
    )


def test_gate_v56_records_closed_fixture_and_zero_production_access() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    fixture = gate["verified_preflight_tooling"]["closed_fixture_result"]
    production = gate["verified_preflight_tooling"]["production_state"]
    decision = gate["preflight_tooling_decision"]

    assert fixture["test_count"] == 14
    assert fixture["artifact_count"] == 2
    assert fixture["artifact_bytes_total"] == 27
    assert fixture["unlisted_source_files_discovered"] == 0
    assert fixture["row_data_libraries_imported"] == []
    assert fixture["operator_parquet_files_parsed"] == 0
    assert fixture["confirmatory_or_holdout_rows_read"] == 0
    assert fixture["fail_closed_check_count"] == 11

    assert production["source_root_supplied"] is False
    assert production["explicit_artifact_list_supplied"] is False
    assert production["operator_preflight_written"] is False
    assert production["production_artifact_paths_enumerated"] == 0
    assert production["production_artifact_bytes_hashed"] == 0
    assert production["production_seal_created"] is False
    assert production["production_receipt_written"] is False
    assert production["pass_g0"] is False
    assert production["g4_open"] is False

    assert decision["status"] == "COMPLETE_CLOSED_FIXTURE_ONLY"
    assert decision["production_status"] == (
        "NOT_RUN_MISSING_SOURCE_ROOT_AND_EXPLICIT_ARTIFACT_LIST"
    )
    assert decision["implicit_discovery_permitted"] is False
    assert decision["next_workstream"] == (
        "OPERATOR_SOURCE_ROOT_AND_EXPLICIT_ARTIFACT_LIST_BINDING"
    )


def test_gate_v56_requires_explicit_inputs_and_forbids_discovery() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    contract = gate["production_input_contract"]

    assert contract["required_user_or_operator_inputs"] == [
        "PRIVATE_SOURCE_ROOT_ABSOLUTE_PATH",
        "COMPLETE_EXPLICIT_ARTIFACT_RELATIVE_PATH_LIST",
    ]
    for key in (
        "source_root_must_exist_and_be_private_access_controlled",
        "artifact_paths_must_be_relative_to_source_root",
        "artifact_paths_must_be_explicit_files_only",
        "artifact_paths_must_be_sorted_unique_and_canonical_in_output",
        "artifact_set_completeness_attestation_required",
        "zero_prior_open_event_attestation_required",
        "zero_row_parse_and_holdout_row_read_attestation_required",
        "operator_control_output_parent_required",
        "private_seal_operator_root_required",
        "repository_receipt_parent_required",
        "production_preflight_command_authorized_after_inputs_bound",
        "production_seal_command_authorized_only_after_preflight_pass",
        "independent_receipt_verification_required_after_seal",
        "successor_gate_required_after_verified_production_receipt",
    ):
        assert contract[key] is True
    for key in (
        "may_search_or_scan_for_missing_inputs",
        "may_infer_artifact_membership_from_directory_contents",
        "may_use_globs_or_recursive_directory_inputs",
        "may_run_without_both_required_inputs",
    ):
        assert contract[key] is False

    assert gate["gate_state"]["G0"]["complete"] is False
    authorization = gate["authorization"]
    assert authorization["production_preflight"] == (
        "NOT_RUN_MISSING_SOURCE_ROOT_AND_EXPLICIT_ARTIFACT_LIST"
    )
    assert authorization["production_seal_creation_or_receipt_issuance"] == (
        "NOT_RUN_NO_PRODUCTION_PREFLIGHT"
    )
    assert authorization["holdout_seal_opening"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
