from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v43.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v43_binds_verified_entry_source_classifier_contract() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v43"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_entry_source_classifier_contract"]
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
    assert result["exact_certified_episode_rows"] == 4
    assert result["registered_author_episode_rows"] == 3_256_271
    assert result["repository_bindings_validated"] == 10
    assert result["operator_receipts_read_and_hash_validated"] == 5
    assert result["operator_parquet_files_opened"] == 0
    assert result["production_parquet_rows_read"] == 0
    assert result["fixture_row_mismatches"] == 0
    assert result["fixture_accounting_mismatches"] == 0
    assert result["all_four_states_covered"] is True
    assert result["individual_work_api_label_required"] is False
    assert result["api_requests_created"] == 0
    assert result["outcomes_read"] is False


def test_gate_v43_authorizes_only_three_count_production_source() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    decision = gate["source_contract_decision"]

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VERIFIED"
    assert (
        decision["content_confirmation_unit"]
        == "CERTIFIED_EPISODE_TOPIC_NOT_INDIVIDUAL_WORK"
    )
    assert decision["individual_work_api_or_human_label_required"] is False
    assert decision["exact_four_expected_temporal_disposition"] == 2
    assert decision["graph_team_cap"] == "NONE"
    assert decision["post_entry_offsets_allowed"] == []

    scope = gate["authorized_production_source_build_scope"]
    assert scope["mode"] == (
        "EXACT_FOUR_THREE_COUNT_FEATURE_SOURCE_ONLY_NO_PRODUCTION_STATE_LABELS"
    )
    assert scope["source_builder_may_read_bound_production_parquet_rows"] is True
    assert (
        scope[
            "source_builder_may_write_operator_local_feature_and_accounting_artifacts"
        ]
        is True
    )
    assert scope["source_builder_may_apply_production_classifier_states"] is False
    assert scope["source_builder_may_open_persistence_or_impact"] is False
    assert scope["source_builder_may_create_api_requests"] is False
    assert scope["source_builder_may_submit_paid_api_calls"] is False
    assert scope["source_builder_may_use_unbound_source_or_candidate_substitution"] is False
    assert scope["successor_gate_required_before_production_classifier_states"] is True

    authorization = gate["authorization"]
    assert authorization["d0_entry_outcome_source_and_classifier_contract"] == (
        "COMPLETE_INDEPENDENTLY_VALIDATED"
    )
    assert authorization["d0_entry_outcome_source_build"].startswith(
        "AUTHORIZED_EXACT_FOUR_FEATURE_SOURCE"
    )
    assert authorization["d0_production_pivot_classification"] == "NOT_AUTHORIZED"
    assert authorization["d0_discovery_outcome_opening"] == "NOT_AUTHORIZED"
    assert authorization["d0_new_api_request_creation"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
    assert gate["gate_state"]["D0"]["complete"] is False
