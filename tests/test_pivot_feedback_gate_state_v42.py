from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v42.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v42_binds_validated_threshold_free_definition() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v42"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_threshold_free_pivot_definition_v2"]
    for key in (
        "contract",
        "validator",
        "research_note",
        "summary_json",
        "summary_markdown",
    ):
        binding = verified[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])

    result = verified["result"]
    assert result["states"] == [
        "NO_FOCAL_ENTRY",
        "NEW_TOPIC_ENTRY",
        "RETURN",
        "CONTINUATION",
    ]
    assert result["primary_minimum_history_threshold"] == "NONE"
    assert result["bounded_exhaustive_valid_count_tuples"] == 45
    assert result["bounded_exhaustive_mismatches"] == 0
    assert result["sensitivity_state_changes"] == 0
    assert result["production_sources_read"] is False
    assert result["outcomes_read"] is False
    assert result["api_requests_created"] == 0


def test_gate_v42_authorizes_only_source_contract_design() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_entry_source_contract_scope"]

    assert scope["mode"] == "CONTRACT_AND_FIXTURES_ONLY_NO_PRODUCTION_ROW_READ"
    sources = scope["required_source_resolutions"]
    assert (
        sources["pre_entry_target_topic_work_count"]["current_state"]
        == "VERIFIED_COLUMN_AVAILABLE"
    )
    assert (
        sources["trailing_3y_target_topic_work_count"]["current_state"]
        == "NOT_YET_CONTRACTED"
    )
    assert (
        sources["focal_confirmed_eligible_target_topic_work_count"][
            "current_state"
        ]
        == "NOT_YET_CONTRACTED"
    )
    assert scope["contract_may_read_production_parquet_rows"] is False
    assert scope["contract_may_create_api_requests"] is False
    assert scope["contract_may_submit_paid_api_calls"] is False
    assert scope["contract_may_build_pivot_labels"] is False
    assert scope["contract_may_read_persistence_or_impact"] is False

    authorization = gate["authorization"]
    assert authorization["d0_entry_outcome_source_and_classifier_contract"].startswith(
        "AUTHORIZED_DESIGN"
    )
    assert authorization["d0_entry_outcome_source_build"] == "NOT_AUTHORIZED"
    assert authorization["d0_production_pivot_classification"] == "NOT_AUTHORIZED"
    assert authorization["d0_discovery_outcome_opening"] == "NOT_AUTHORIZED"
    assert authorization["d0_new_api_request_creation"] == "NOT_AUTHORIZED"
    assert authorization["d0_paid_api_submission"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
    assert gate["gate_state"]["D0"]["complete"] is False
