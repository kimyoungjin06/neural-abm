from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_label_selection_20261014.summary.json"


def _groups() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))["groups"]


def test_selection_biases_what_pooled_learners_see() -> None:
    for group in ("A/linear", "A/mlp"):
        arms = _groups()[group]["arms"]
        assert arms["S_shares"]["data_weight_at_last_refit_before_shift"] > arms["U_matched"]["data_weight_at_last_refit_before_shift"]
        assert arms["U_matched"]["policy_weight_8_steps_before_shift"] < arms["S_shares"]["policy_weight_8_steps_before_shift"]
        # Volume on top of unselected rows changes little.
        assert abs(arms["F_all"]["policy_weight_8_steps_before_shift"] - arms["U_matched"]["policy_weight_8_steps_before_shift"]) < 0.05


def test_per_agent_mlp_overconfidence_is_not_selection() -> None:
    arms = _groups()["B/mlp"]["arms"]
    assert arms["U_matched"]["policy_weight_8_steps_before_shift"] > 10
    assert arms["U_matched"]["data_weight_at_last_refit_before_shift"] < 4


def test_removing_selection_lifts_post_shift_precision_everywhere() -> None:
    for group in _groups().values():
        for contrast in group["precision_contrasts"].values():
            assert contrast["post_shift"]["interval_95"][0] > 0
