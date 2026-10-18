from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_study_v2_d4_20261019.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_d4_v2_reproduced_byte_for_byte_under_the_frozen_precommit() -> None:
    result = _result()
    assert result["precommit_commit"] == "dcc6e7495d378c7375544cdd882471d837aaf8eb"
    assert result["run_status"] == "STUDY_RESULT"
    assert result["primary_measure"] == "unverified_share_precision"
    assert result["reproduction"]["byte_identical"] == {"cells.jsonl": True, "summary.json": True}


def test_the_registered_dispositions() -> None:
    reading = _result()["registered_reading"]
    assert reading["design_check"] == "NULL_WORLD_PASSES"
    assert reading["closed_loop"] == "TABLE_SUFFICES_EVEN_IN_B"
    assert reading["closed_loop_material_positive_groups"] == {"A": 1, "B": 0}
    for regime in ("A", "B"):
        assert reading["best_table_counts"][regime]["BEST_TABLE_BEATS_ADAPTIVE"] == 6


def test_every_learner_beats_the_hand_rule_in_the_study_groups() -> None:
    table = _result()["table"]
    for key, entry in table.items():
        rule = entry["unverified_share_precision"]["hand_rule"]
        if key.endswith("/null"):
            assert not rule["material"]
        else:
            assert rule["material"] and rule["mean_cell_delta"] > 0.03
