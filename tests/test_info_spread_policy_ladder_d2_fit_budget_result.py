from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_d2_fit_budget_check_20261014.summary.json"
GROUPS = ("POS_flip/A/linear", "POS_flip/A/mlp", "POS_flip/B/linear", "POS_flip/B/mlp")


def _comparison() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))["comparison"]


def test_ensembling_is_significant_before_the_shift_with_pivot_settings() -> None:
    comparison = _comparison()
    for group in GROUPS:
        assert comparison[group]["pivot"]["contrasts"]["T0_1 - T0_4 (ensembling)"]["pre_shift"]["interval_95"][1] < 0


def test_the_closed_loop_at_matched_evidence_stays_indistinguishable() -> None:
    comparison = _comparison()
    label = "L0_full - T0_1 (closed loop at matched evidence)"
    for group in GROUPS:
        for cohort in ("pre_shift", "post_shift"):
            low, high = comparison[group]["pivot"]["contrasts"][label][cohort]["interval_95"]
            assert low <= 0 <= high


def test_forgetting_pays_after_the_flip_and_costs_before_it() -> None:
    comparison = _comparison()
    label = "L0_window - L0_full (forgetting)"
    for group in GROUPS:
        contrast = comparison[group]["pivot"]["contrasts"][label]
        assert contrast["post_shift"]["interval_95"][0] > 0
        assert contrast["pre_shift"]["interval_95"][1] < 0


def test_linear_learners_move_further_with_the_larger_budget() -> None:
    comparison = _comparison()
    for group in ("POS_flip/A/linear", "POS_flip/B/linear"):
        for arm in ("L0_full", "L0_window"):
            d1 = comparison[group]["d1"]["credibility_weight"][arm]["[16,20)"]
            pivot = comparison[group]["pivot"]["credibility_weight"][arm]["[16,20)"]
            assert pivot < d1
