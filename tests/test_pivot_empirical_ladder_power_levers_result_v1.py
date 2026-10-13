from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_empirical_ladder_power_levers_v1_20261013.summary.json"
SCRIPT = ROOT / "scripts" / "explore_pivot_empirical_ladder_power_levers_v1.py"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_screen_is_exploratory_and_bound_to_its_script() -> None:
    result = _result()
    assert result["status"] == "EXPLORATORY_NOT_REGISTERED_NO_LEVER_ADOPTABLE"
    assert result["script"]["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert result["execution"]["seed"] == 20261105 and result["execution"]["tasks"] == 72
    assert result["boundaries"]["empirical_rows_read"] == result["boundaries"]["empirical_outcome_rows_read"] == 0
    assert result["boundaries"]["registered"] is False


def test_joint_tests_falsely_flag_anticipation() -> None:
    cells = _result()["cells"]
    for config in ("base", "C_no_g4_guard"):
        assert cells[f"{config}/W1_learning"]["A2_joint_ramp_confirmatory"]["wins"] == 8
        assert cells[f"{config}/W6_static_aligned_drift"]["A2_joint_ramp_confirmatory"]["wins"] >= 5


def test_no_lever_lifts_the_specific_rule() -> None:
    cells = _result()["cells"]
    for config in ("base", "B_field_history", "C_no_g4_guard"):
        assert cells[f"{config}/W1_learning"]["A0_three_way"]["wins"] <= 3
        for world in ("W3_static", "W6_static_aligned_drift"):
            assert cells[f"{config}/{world}"]["A0_three_way"]["wins"] == 0
    assert cells["B_field_history/W3_static"]["A1_joint_confirmatory"]["wins"] == 2


def test_nothing_is_proposed_from_the_screen() -> None:
    nxt = _result()["next"]
    assert nxt["v4_freeze"] == "NOT_PROPOSED_FROM_THESE_LEVERS"
    assert nxt["N11c"] == nxt["N11d"] == "NOT_PROPOSED"
