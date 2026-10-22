from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_flip_rate_v1_20261021.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_o26_reproduced_under_its_freeze() -> None:
    result = _result()
    assert result["precommit_commit"] == "86d3e80928359b58b03e65997a5a130792dfcf07"
    assert result["run_status"] == "STUDY_RESULT"
    assert result["reproduction"]["byte_identical"] == {"cells.jsonl": True, "summary.json": True}


def test_the_onset_is_a_quarter_in_every_family_and_the_control_holds() -> None:
    dispositions = _result()["dispositions"]
    assert dispositions["design_check"] == "POSITIVE_CONTROL_HOLDS_AT_0.5"
    assert set(dispositions["onset"].values()) == {"0.25"}
    for family_labels in dispositions["labels"].values():
        assert family_labels["0"]["best_table"] == "NEGATIVE"
        assert family_labels["0.1"]["closed_loop"] == "POSITIVE"
        assert family_labels["0.25"]["best_table"] == "POSITIVE"
