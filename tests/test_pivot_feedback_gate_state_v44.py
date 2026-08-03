from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v44.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v44_binds_independently_verified_production_source() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v44"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_production_entry_source"]
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
    assert result["work_source_rows_replayed"] == 492_293_206
    assert result["author_event_source_rows_replayed"] == 779_630_258
    assert result["pre_entry_value_mismatches"] == 0
    assert result["trailing_value_mismatches"] == 0
    assert result["focal_value_mismatches"] == 0
    assert result["invariant_violations"] == 0
    assert result["production_classifier_state_columns"] == 0
    assert result["pivot_label_columns"] == 0
    assert result["outcomes_read"] is False
    assert result["api_requests_created"] == 0


def test_gate_v44_authorizes_only_deterministic_four_state_application() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    decision = gate["production_source_decision"]
    scope = gate["authorized_production_classifier_scope"]

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VERIFIED"
    assert decision["registered_author_episode_rows"] == 3_256_271
    assert decision["minimum_history_threshold"] == "NONE"
    assert decision["graph_team_cap"] == "NONE"
    assert decision["post_entry_offsets_read"] == []

    assert scope["mode"] == (
        "DETERMINISTIC_FOUR_STATE_APPLICATION_TO_VERIFIED_THREE_COUNT_SOURCE_ONLY"
    )
    assert scope["exact_states"] == [
        "NO_FOCAL_ENTRY",
        "NEW_TOPIC_ENTRY",
        "RETURN",
        "CONTINUATION",
    ]
    assert scope["primary_pivot_state"] == "NEW_TOPIC_ENTRY"
    assert scope["classifier_may_read_verified_three_count_source"] is True
    assert scope["classifier_may_materialize_exact_four_state_column"] is True
    assert scope["classifier_may_change_or_filter_registered_population"] is False
    assert scope["classifier_may_read_raw_work_or_author_event_partitions"] is False
    assert scope["classifier_may_read_post_entry_persistence_or_impact"] is False
    assert scope["classifier_may_create_api_requests"] is False
    assert scope["classifier_may_fit_or_score_a_model"] is False

    authorization = gate["authorization"]
    assert authorization["d0_entry_outcome_source_build"] == (
        "COMPLETE_INDEPENDENTLY_VERIFIED"
    )
    assert authorization["d0_production_pivot_classification"].startswith(
        "AUTHORIZED_EXACT_FOUR"
    )
    assert authorization["d0_discovery_outcome_opening"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
    assert gate["gate_state"]["D0"]["complete"] is False
