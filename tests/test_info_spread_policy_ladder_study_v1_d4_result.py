from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_study_v1_d4_20261016.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_d4_reproduced_byte_for_byte_under_the_frozen_precommit() -> None:
    result = _result()
    assert result["precommit_commit"] == "daf5d80b5ac25086b247ae7c2595f9b7ca0a32d1"
    assert result["run_status"] == "STUDY_RESULT"
    assert result["reproduction"]["byte_identical"] == {"cells.jsonl": True, "summary.json": True}
    outputs = result["reproduction"]["outputs"]
    assert outputs["development"] == outputs["reproduction"]


def test_the_null_check_fails_and_blocks_the_study_groups() -> None:
    result = _result()
    reading = result["registered_reading"]
    assert reading["design_check"] == "DESIGN_DEFECT_IN_NULL_WORLD"
    assert reading["null_violations"] == ["B/partial_mlp/null"]
    assert reading["study_groups_read"] is False
    violation = result["table"]["B/partial_mlp/null"]["hand_rule"]
    assert violation["material"] and violation["mean_cell_delta"] > 0.02
    assert not result["table"]["A/mlp/null"]["hand_rule"]["material"]


def test_every_group_has_twenty_cells() -> None:
    table = _result()["table"]
    assert len(table) == 16
    assert {entry["cells"] for entry in table.values()} == {20}
