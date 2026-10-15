from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_d2b_controls_20261015.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_d2b_is_not_adequate_and_scores_no_study_cell() -> None:
    result = _result()
    assert result["registered_reading"] == {
        "neg_no_material_pair": True,
        "pos_sign_closed_loop_material_and_positive": False,
        "adequate": False,
    }
    assert result["boundaries"]["study_cells_scored"] is False


def test_only_pooled_linear_fails_the_sign_control() -> None:
    table = _result()["table"]
    assert not table["POS_sign/A/linear/L0-T0_1"]["group"]["material"]
    for group in ("A/mlp", "B/partial_linear", "B/partial_mlp"):
        entry = table[f"POS_sign/{group}/L0-T0_1"]["group"]
        assert entry["material"] and entry["mean_cell_delta"] > 0.1


def test_pooled_linear_never_learns_the_reversed_sign() -> None:
    by_sign = _result()["pos_sign_precision_by_run_sign"]
    assert by_sign["A/linear/sign-1"]["L0"] < 0.3
    for group in ("A/mlp", "B/partial_linear", "B/partial_mlp"):
        assert by_sign[f"{group}/sign-1"]["L0"] > 0.5
