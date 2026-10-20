from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_d2c_controls_20261016.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_d2c_is_adequate_and_scores_no_study_cell() -> None:
    result = _result()
    assert result["registered_reading"] == {
        "neg_no_material_pair": True,
        "pos_sign_closed_loop_material_and_positive": True,
        "adequate": True,
    }
    assert result["boundaries"]["study_cells_scored"] is False


def test_no_neg_pair_is_material_and_every_sign_group_is() -> None:
    table = _result()["table"]
    assert not any(v["group"]["material"] for k, v in table.items() if k.startswith("NEG/"))
    for key, value in table.items():
        if key.startswith("POS_sign/"):
            assert value["group"]["material"] and value["group"]["mean_cell_delta"] > 0.08


def test_the_exact_linear_learner_learns_a_reversed_sign() -> None:
    by_sign = _result()["pos_sign_precision_by_run_sign"]
    assert by_sign["A/linear_exact/sign-1"]["L0"] > 0.55
