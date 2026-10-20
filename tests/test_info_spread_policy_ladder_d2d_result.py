from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_d2d_controls_20261018.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_d2d_is_adequate_on_the_unverified_measure() -> None:
    result = _result()
    assert result["primary_measure"] == "unverified_share_precision"
    reading = result["registered_reading"]
    assert reading["neg_no_material_pair"] and reading["pos_sign_closed_loop_material_and_positive"]
    assert reading["null_no_material_positive_learner_gain"] and reading["adequate"]
    assert result["boundaries"]["study_cells_scored"] is False


def test_the_null_control_detects_the_old_measures_leak_and_not_the_new() -> None:
    result = _result()
    assert result["registered_reading"]["null_old_measure_material_positive_groups"] == ["NULL/B/partial_mlp"]
    entry = result["table"]["NULL/B/partial_mlp/L0-F1"]
    assert entry["share_precision"]["material"] and entry["share_precision"]["mean_cell_delta"] > 0.02
    assert not entry["unverified_share_precision"]["material"]


def test_the_closed_loop_effect_survives_the_change_of_measure() -> None:
    table = _result()["table"]
    for key, entry in table.items():
        if key.startswith("POS_sign/"):
            rule = entry["unverified_share_precision"]
            assert rule["material"] and rule["mean_cell_delta"] > 0.08
