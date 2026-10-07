from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_observable_one_sided_v1_20261007.summary.json"


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_and_reproduced() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "c0a8b1a5e119e685688d82d6f0c72a0f42502bfc"
    assert result["execution"]["raw_row_count"] == 6 * 4 * 3 * 4 * 16
    assert result["execution"]["deterministic_rerun"]["all_four_artifacts_byte_identical"] is True
    assert result["validation"]["nesting_checks"] == "PASS_EXACT"


def test_the_sufficient_sign_is_confirmed_for_both_observables() -> None:
    disposition = _result()["registered_result_disposition"]
    assert disposition["status_mix"] == {"active_cells": 13, "discriminating": True, "inactive_cells": 11}
    for entry in disposition["one_sided_tests"].values():
        assert entry["disposition"] == "SUFFICIENT_SIGN_CONFIRMED"
        assert (entry["flagged_cells"], entry["false_active"], entry["detected_active"]) == (11, 0, 11)
    assert disposition["one_sided_either_observable"]["false_active"] == 0


def test_only_the_outcome_trend_passes_its_no_learning_control_and_is_carried() -> None:
    result = _result()
    disposition = result["registered_result_disposition"]
    assert disposition["no_learning_control_by_observable"] == {
        "cohort_persistence_drop": False,
        "outcome_selection_trend": True,
    }
    assert disposition["no_learning_control"]["cohort_persistence_drop"]["cells_reading_inactive"] == 16
    assert disposition["carried_to_the_empirical_stage_as_a_one_sided_indicator"] == ["outcome_selection_trend"]
    for cell in result["cells"]:
        assert cell["fixed_rule_outcome_selection_trend"] < 0.098417


def test_post_hoc_records_the_broken_weak_channel_reading() -> None:
    post = _result()["post_hoc_descriptive"]
    assert post["status"] == "POST_HOC_DESCRIPTIVE_NOT_REGISTERED_NO_NEW_EXECUTION"
    assert post["missed_cells_passing_pairs"]["hazard_q0p15/failure_lag_4"] == 2
    assert post["false_actives_across_n05_n08_n09"] == 0
