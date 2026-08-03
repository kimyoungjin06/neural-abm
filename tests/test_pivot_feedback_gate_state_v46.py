from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v46.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v46_binds_independently_validated_persistence_contract() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v46"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_persistence_outcome_contract"]
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
    assert result["repository_bindings_validated"] == 11
    assert result["operator_receipts_read"] == 3
    assert result["operator_parquet_files_opened"] == 0
    assert result["production_classification_rows_read"] == 0
    assert result["production_post_entry_rows_read"] == 0
    assert result["primary_pivot_state"] == "NEW_TOPIC_ENTRY"
    assert result["primary_pivot_rows"] == 4_301
    assert result["outcome_window_offsets"] == [1, 2, 3]
    assert result["exact_four_all_matured"] is True
    assert result["fixture_row_mismatches"] == 0
    assert result["fixture_accounting_mismatches"] == 0
    assert result["partial_positive_right_censored_fixture_passed"] is True
    assert result["new_api_request_required"] is False
    assert result["api_requests_created"] == 0


def test_gate_v46_authorizes_only_exact_production_source_build() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    decision = gate["persistence_contract_decision"]
    scope = gate["authorized_production_persistence_source_scope"]

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VALIDATED"
    assert decision["exact_denominator_state"] == "NEW_TOPIC_ENTRY"
    assert decision["exact_denominator_rows"] == 4_301
    assert decision["exact_outcome_window_offsets"] == [1, 2, 3]
    assert decision["true_researcher_topic_persistence_claim_allowed"] is False

    assert scope["mode"] == (
        "EXACT_4301_NEW_TOPIC_ENTRY_T_PLUS_1_THROUGH_T_PLUS_3_SOURCE_ONLY"
    )
    assert scope["exact_output_rows"] == 4_301
    assert scope["permitted_post_entry_offsets"] == [1, 2, 3]
    assert scope["source_builder_may_read_bound_production_classification_rows"] is True
    assert scope["source_builder_may_read_bound_work_and_author_event_partitions"] is True
    assert scope["source_builder_may_read_confirmatory_entry_year_rows"] is False
    assert scope["source_builder_may_change_confirmatory_periods"] is False
    assert scope["source_builder_may_substitute_episode_field_topic_or_entry_year"] is False
    assert scope["source_builder_may_filter_any_new_topic_entry_row"] is False
    assert scope["source_builder_may_apply_history_or_team_threshold"] is False
    assert scope[
        "source_builder_may_materialize_exact_persistence_counts_states_and_boolean"
    ] is True
    assert scope[
        "source_builder_may_materialize_impact_citation_peer_model_or_g4_columns"
    ] is False
    assert scope["source_builder_may_create_api_requests"] is False
    assert scope["source_builder_may_analyze_exposure_outcome_associations"] is False
    assert scope["source_builder_may_freeze_g0"] is False
    assert scope["source_builder_may_fit_or_score_a_model"] is False

    authorization = gate["authorization"]
    assert authorization[
        "d0_persistence_outcome_source_and_definition_contract"
    ] == "COMPLETE_INDEPENDENTLY_VALIDATED"
    assert authorization[
        "d0_production_post_entry_persistence_source_build"
    ].startswith("AUTHORIZED_EXACT_4301")
    assert authorization["d0_discovery_outcome_opening"].startswith(
        "AUTHORIZED_SOURCE_BUILD_ONLY"
    )
    assert authorization[
        "d0_discovery_outcome_analysis_or_interpretation"
    ] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
    assert gate["gate_state"]["D0"]["complete"] is False
