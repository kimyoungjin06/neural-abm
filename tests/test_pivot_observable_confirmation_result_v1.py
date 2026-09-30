from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_observable_confirmation_v1_20261001.summary.json"
)


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_and_reproduced() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "b270e5137e37a846e9e30f8d3acb4c1448edf886"
    assert result["execution"]["raw_row_count"] == 6 * 4 * 3 * 4 * 16
    assert result["execution"]["deterministic_rerun"]["all_four_artifacts_byte_identical"] is True
    assert result["validation"]["nesting_checks"] == "PASS_EXACT"


def test_both_observables_hold_in_most_cells_with_only_false_inactives() -> None:
    disposition = _result()["registered_result_disposition"]
    assert disposition["status_mix"] == {"active_cells": 10, "discriminating": True, "inactive_cells": 14}
    tests = disposition["observable_tests"]
    assert (tests["cohort_persistence_drop"]["correct"], tests["cohort_persistence_drop"]["false_active"]) == (23, 0)
    assert (tests["outcome_selection_trend"]["correct"], tests["outcome_selection_trend"]["false_active"]) == (22, 0)
    for entry in tests.values():
        assert entry["disposition"] == "HOLDS_IN_MOST_CELLS"
    assert disposition["joint_rule_both_observables"]["correct"] == 22
    assert disposition["carried_to_the_empirical_stage_as_a_confirmed_discriminant"] == []


def test_the_no_learning_control_reads_inactive_everywhere() -> None:
    result = _result()
    control = result["registered_result_disposition"]["no_learning_control"]
    assert control["disposition"] == "CONTROL_READS_INACTIVE_EVERYWHERE"
    for cell in result["cells"]:
        assert cell["fixed_rule_cohort_persistence_drop"] > 0.023175
        assert cell["fixed_rule_outcome_selection_trend"] < 0.098417


def test_post_hoc_misses_are_weak_channels_only() -> None:
    result = _result()
    strength = result["post_hoc_descriptive"]["channel_strength"]
    assert result["post_hoc_descriptive"]["status"] == "POST_HOC_DESCRIPTIVE_NOT_REGISTERED_NO_NEW_EXECUTION"
    missed = set()
    for entry in result["registered_result_disposition"]["observable_tests"].values():
        missed |= set(entry["missed_active_cells"])
    assert missed == set(strength["n08_weak_active_cells"])
    assert strength["false_actives_across_n05_and_n08"] == 0
    assert result["scientific_assessment"]["status"] == "NOT_CARRIED_AS_A_CONFIRMED_TWO_SIDED_DISCRIMINANT"
