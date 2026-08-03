from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v45.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v45_binds_independently_verified_production_classification() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v45"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_production_pivot_classification"]
    for key in (
        "config",
        "producer",
        "verifier",
        "research_note",
        "summary_json",
        "summary_markdown",
    ):
        binding = verified[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])

    result = verified["result"]
    assert result["registered_author_episode_rows"] == 3_256_271
    assert result["source_columns_preserved"] == 8
    assert result["classification_columns"] == 10
    assert result["no_focal_entry_rows"] == 3_246_984
    assert result["new_topic_entry_rows"] == 4_301
    assert result["return_rows"] == 125
    assert result["continuation_rows"] == 4_861
    assert sum(
        result[key]
        for key in (
            "no_focal_entry_rows",
            "new_topic_entry_rows",
            "return_rows",
            "continuation_rows",
        )
    ) == result["registered_author_episode_rows"]
    assert result["source_column_mismatches"] == 0
    assert result["state_value_mismatches"] == 0
    assert result["primary_boolean_mismatches"] == 0
    assert result["source_rows_filtered"] == 0
    assert result["source_feature_value_mutations"] == 0
    assert result["post_entry_offsets_read"] == []
    assert result["persistence_or_impact_rows_read"] is False
    assert result["api_requests_created"] == 0


def test_gate_v45_authorizes_only_persistence_contract_and_fixtures() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    decision = gate["production_classification_decision"]
    scope = gate["authorized_persistence_outcome_contract_scope"]

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VERIFIED"
    assert decision["primary_pivot_state"] == "NEW_TOPIC_ENTRY"
    assert decision["primary_pivot_rows"] == 4_301
    assert decision["minimum_history_threshold"] == "NONE"
    assert decision["observable_history_claim_only"] is True
    assert decision["true_lifetime_first_claim_allowed"] is False

    assert scope["mode"] == (
        "DESIGN_AND_SYNTHETIC_FIXTURES_ONLY_NO_PRODUCTION_POST_ENTRY_ROW_READ"
    )
    assert scope["eligible_entry_year_end"] == 2012
    assert scope["confirmatory_entry_years_may_be_read"] is False
    assert scope["confirmatory_periods_may_be_changed"] is False
    assert scope["fields_may_be_selected_by_effect_size"] is False
    candidate = scope["provisional_outcome_candidate"]
    assert candidate["start_offset"] == 1
    assert candidate["end_offset"] == 3
    assert candidate["final_g0_freeze_status"] == "PENDING"
    assert scope["contract_may_inspect_repository_configs_and_receipt_metadata"] is True
    assert scope["contract_may_read_production_classification_rows"] is False
    assert scope["contract_may_read_raw_work_or_author_event_partitions"] is False
    assert scope["contract_may_read_production_post_entry_rows"] is False
    assert scope["contract_may_materialize_production_persistence_labels"] is False
    assert scope["contract_may_create_api_requests"] is False
    assert scope["contract_may_submit_paid_api_calls"] is False
    assert scope["contract_may_establish_final_candidate_eligibility"] is False
    assert scope["contract_may_freeze_g0"] is False
    assert scope["contract_may_fit_or_score_a_model"] is False

    authorization = gate["authorization"]
    assert authorization["d0_production_pivot_classification"] == (
        "COMPLETE_INDEPENDENTLY_VERIFIED"
    )
    assert authorization[
        "d0_persistence_outcome_source_and_definition_contract"
    ].startswith("AUTHORIZED_DESIGN")
    assert authorization["d0_production_post_entry_persistence_source_build"] == (
        "NOT_AUTHORIZED"
    )
    assert authorization["d0_discovery_outcome_opening"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
    assert gate["gate_state"]["D0"]["complete"] is False
