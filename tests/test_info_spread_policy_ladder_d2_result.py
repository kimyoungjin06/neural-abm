from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_d2_controls_20261014.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_d2_scored_controls_only() -> None:
    result = _result()
    assert result["boundaries"]["study_cells_scored"] is False
    assert all(key.split("/")[0] in {"NEG_two_tables", "POS_flat", "POS_flip"} for key in result["table"])
    assert result["next"]["D3"] == "NOT_FROZEN_RULE_AND_CONTROLS_NEED_REVISION"


def test_false_positives_pass_and_power_fails() -> None:
    result = _result()
    for reading in result["registered_reading"].values():
        assert reading == {"neg_false_positive_ok": True, "pos_flip_power_ok": False}
    for regime in ("A", "B"):
        for family in ("linear", "mlp"):
            assert result["table"][f"NEG_two_tables/{regime}/{family}"]["material"] <= 1
            assert result["table"][f"POS_flip/{regime}/{family}"]["material"] < 16


def test_the_adaptive_arm_is_worse_in_both_shift_controls() -> None:
    table = _result()["table"]
    for key, row in table.items():
        if key.startswith("POS_"):
            assert row["mean_delta_share_precision"] < 0 and row["mean_delta_false_reach"] > 0
