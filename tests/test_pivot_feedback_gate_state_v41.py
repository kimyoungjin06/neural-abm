from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v41.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v41_binds_verified_join_and_decides_predecision_completeness() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v41"
    assert gate["additive_parent"]["sha256"] == _sha256(
        ROOT / gate["additive_parent"]["relative_path"]
    )
    verified = gate["verified_predecision_author_measurement"]
    for key in ("config", "producer", "verifier", "summary_json", "summary_markdown"):
        binding = verified[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
    result = verified["result"]
    assert result["author_episode_rows"] == 3_256_271
    assert result["measurement_columns"] == 74
    assert result["component_value_mismatches"] == [0, 0, 0, 0]
    assert result["component_bidirectional_key_differences"] == [0, 0, 0, 0]
    assert result["forbidden_measurement_columns"] == 0
    assert result["unclassified_columns"] == 0

    decision = gate["measurement_completeness_decision"]
    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VERIFIED"
    assert decision["primary_minimum_history_threshold"] == "NONE"
    assert decision["graph_team_cap_selected"] is False
    assert decision["peer_visible_success_role"].startswith("OPTIONAL_LATER")
    assert decision["predecision_surface_complete_does_not_mean_d0_discovery_complete"]


def test_gate_v41_supersedes_old_three_work_gate_without_opening_outcomes() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    legacy = gate["legacy_simple_pivot_definition_v1"]
    assert legacy["sha256"] == _sha256(ROOT / legacy["relative_path"])
    assert legacy["historical_role"] == "SUPERSEDED_FOR_CURRENT_PRODUCTION_REGISTRATION"
    assert legacy["superseded_choice"] == {
        "minimum_prior_observable_span_years": 3,
        "minimum_prior_eligible_work_count": 3,
    }

    successor = gate["authorized_threshold_free_pivot_definition_v2_scope"]
    assert (
        successor["required_current_principles"][
            "no_minimum_prior_year_or_work_threshold"
        ]
        is True
    )
    assert successor["definition_registration_may_read_focal_or_future_rows"] is False
    assert successor["production_classifier_build_authorized_after_definition_alone"] is False
    assert successor["paid_api_execution_authorized"] is False

    authorization = gate["authorization"]
    assert authorization["d0_threshold_free_researcher_pivot_definition_v2"].startswith(
        "AUTHORIZED_DEFINITION"
    )
    assert authorization["d0_production_pivot_classification"] == "NOT_AUTHORIZED"
    assert authorization["d0_discovery_outcome_opening"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
    assert gate["gate_state"]["D0"]["complete"] is False
