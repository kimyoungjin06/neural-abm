from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v57.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v57_binds_input_resolution_evidence() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v57"
    assert gate["status"] == (
        "VALID_PRODUCTION_INPUT_RESOLUTION_NO_MATERIALIZED_G4_ARTIFACT_SET_NO_SEAL"
    )
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_production_input_resolution"]
    for key in (
        "resolution_contract",
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
    assert summary["schema_version"] == verified["summary_json"][
        "required_schema_version"
    ]
    assert summary["status"] == verified["summary_json"]["required_status"]
    assert summary["repository_artifacts"]["resolution_contract"]["sha256"] == (
        verified["resolution_contract"]["sha256"]
    )
    assert summary["repository_artifacts"]["validator"]["sha256"] == (
        verified["validator"]["sha256"]
    )


def test_gate_v57_records_materialization_blocker_without_operator_scan() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_production_input_resolution"]["result"]
    decision = gate["input_resolution_decision"]

    assert result["logical_partition_reserved"] is True
    assert result["byte_seal_created"] is False
    assert result["g4_domain_selected"] is False
    assert result["g4_field_selected"] is False
    assert result["g4_artifact_set_materialized"] is False
    assert result["available_d0_entry_year_maximum"] == 2012
    assert result["requested_g4_entry_years"] == [2020, 2021, 2022]
    assert result["operator_directories_listed"] == 0
    assert result["operator_paths_searched_or_scanned"] == 0
    assert result["holdout_artifact_paths_enumerated"] == 0
    assert result["holdout_artifact_bytes_hashed"] == 0
    assert result["confirmatory_or_holdout_rows_read"] == 0

    assert decision["blocker_class"] == (
        "PREREQUISITE_ARTIFACT_NOT_MATERIALIZED_NOT_MISSING_USER_PATH"
    )
    assert decision["production_status"] == (
        "NOT_RUN_NO_MATERIALIZED_G4_ARTIFACT_SET"
    )
    assert decision["logical_partition_reservation_equals_byte_seal"] is False
    assert decision["existing_d0_artifact_substitution_valid"] is False
    assert decision["operator_path_request_is_sufficient_next_step"] is False
    assert decision["production_preflight_may_run_now"] is False
    assert decision["production_seal_may_run_now"] is False


def test_gate_v57_corrects_g4_state_and_keeps_execution_closed() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["gate_state"]["G0"]["complete"] is False
    assert gate["gate_state"]["G4"] == {
        "state": "LOGICALLY_RESERVED_NOT_BYTE_SEALED_NOT_OPENED",
        "complete": False,
    }
    authorization = gate["authorization"]
    assert authorization["production_input_metadata_resolution"] == (
        "COMPLETE_INDEPENDENTLY_VERIFIED"
    )
    assert authorization["production_preflight"] == (
        "NOT_AUTHORIZED_PREREQUISITE_ARTIFACT_NOT_MATERIALIZED"
    )
    assert authorization["production_seal_creation_or_receipt_issuance"] == (
        "NOT_AUTHORIZED_NO_VALID_PREFLIGHT"
    )
    assert authorization["holdout_seal_opening"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
