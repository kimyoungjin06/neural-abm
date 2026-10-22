from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_transfer_v1_20261021.summary.json"
MEASURE = "unverified_share_precision"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_o23_reproduced_under_its_freeze_with_parity() -> None:
    result = _result()
    assert result["precommit_commit"] == "fded050c6c0146c8dad7075488f17ba10019e873"
    assert result["run_status"] == "STUDY_RESULT"
    assert result["reproduction"]["byte_identical"] == {"cells.jsonl": True, "summary.json": True}
    assert result["parity_failures"] == []


def test_the_transferred_table_suffices_in_both_regimes() -> None:
    dispositions = _result()["dispositions"]
    assert dispositions["design_check"] == "NULL_WORLD_PASSES"
    assert dispositions["transfer_best_table"] == {"A": "TRANSFERRED_TABLE_SUFFICES", "B": "TRANSFERRED_TABLE_SUFFICES"}
    for regime in ("A", "B"):
        assert dispositions["counts"]["transfer_best_table"][regime]["NEGATIVE"] == 6


def test_same_world_donors_were_worth_nothing_material() -> None:
    table = _result()["table"]
    study = [v for k, v in table.items() if not k.endswith("/null")]
    assert not any(v[MEASURE]["transfer_cost"]["material"] for v in study)
    assert not any(v[MEASURE]["transfer_closed_loop"]["material"] for v in study)
