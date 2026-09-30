from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_observable_discriminant_v1_20260930.summary.json"
)


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_and_reproduced() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "5b73c19c571a6f870fb7994ea576206e314380fe"
    execution = result["execution"]
    assert execution["raw_row_count"] == 6 * 4 * 3 * 4 * 16
    assert execution["deterministic_rerun"]["all_four_artifacts_byte_identical"] is True
    assert result["validation"]["nesting_checks"] == "PASS_EXACT"
    assert result["boundaries"]["empirical_rows_read"] == 0


def test_the_design_was_informative() -> None:
    disposition = _result()["registered_result_disposition"]
    assert disposition["status_mix"] == {"active_cells": 10, "discriminating": True, "inactive_cells": 14}
    assert disposition["design_check"]["disposition"] == "SURGE_CLOCK_PRODUCES_INTERIOR_PEAKS"
    assert disposition["design_check"]["mean_interior_peak_share"] > 0.75


def test_the_post_failure_share_mostly_holds_with_only_false_actives() -> None:
    pfes = _result()["registered_result_disposition"]["post_failure_entry_share"]
    assert pfes["disposition"] == "HOLDS_IN_MOST_CELLS"
    assert (pfes["correct"], pfes["cells"], pfes["false_active"], pfes["false_inactive"]) == (18, 24, 6, 0)
    assert pfes["separates_in_this_study"] == "DOES_NOT_SEPARATE"
    comparator = _result()["registered_result_disposition"]["feature_arm_relative_delay"]
    assert (comparator["correct"], comparator["false_inactive"]) == (19, 0)


def test_two_observables_separate_every_cell() -> None:
    result = _result()
    disposition = result["registered_result_disposition"]
    assert disposition["separating_observables"] == ["cohort_persistence_drop", "outcome_selection_trend"]
    screen = disposition["observable_screen"]
    assert screen["cohort_persistence_drop"]["direction"] == "ACTIVE_LOWER"
    assert screen["outcome_selection_trend"]["direction"] == "ACTIVE_HIGHER"
    for cell in result["cells"]:
        active = cell["status"] == "ENTRY_CHANNEL_ACTIVE"
        assert (cell["cohort_persistence_drop"] < 0.01) == active
        assert (cell["outcome_selection_trend"] > 0.1) == active
    for name in ("post_decline_entry_share", "post_first_cohort_entry_share", "post_peak_entry_share", "attribute_selection_trend"):
        assert screen[name]["disposition"] == "DOES_NOT_SEPARATE"


def test_post_hoc_notes_the_n01_counterexample_and_bounds_the_claim() -> None:
    result = _result()
    post = result["post_hoc_descriptive"]
    assert post["status"] == "POST_HOC_DESCRIPTIVE_NOT_REGISTERED_NO_NEW_EXECUTION"
    assert post["cohort_persistence_drop_in_n01"]["separates"] is False
    assert result["pre_freeze_n01_screen"]["separating"] == ["post_failure_entry_share"]
    assessment = result["scientific_assessment"]
    assert assessment["status"] == "CANDIDATES_FOR_A_THIRD_REGISTERED_TEST_NOT_CONFIRMED_DISCRIMINANTS"
    assert assessment["model_selected"] == "NO"
