from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "n14_pivot_parity_20261019.summary.json"
RECORDED = "1f532d98663767cd2c9bdfa74022b620efbac4fd4827f1d6e85a2f9bb3b3d6ca"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_the_re_expressed_pivot_run_matches_its_record_byte_for_byte() -> None:
    result = _result()
    assert result["status"] == "N14_PARITY_HOLDS_BYTE_FOR_BYTE"
    assert result["target"]["recorded_runs_jsonl_sha256"] == RECORDED
    parity = result["parity"]
    assert parity["runs_jsonl_sha256"] == RECORDED and parity["whole_file_identical"] is True
    assert parity["rows"] == parity["identical_rows"] == 1152
    assert {arm: v["identical"] for arm, v in parity["by_arm"].items()} == {
        "fixed_rule": 288, "linear_batch": 288, "tree_batch": 288, "mlp_batch": 288}


def test_the_baseline_reproduced_and_nothing_else_changed() -> None:
    result = _result()
    assert result["baseline"]["rows"] == result["baseline"]["identical_rows"] == 72
    assert result["boundaries"]["generic_nabm_api_changed"] is False
    assert set(result["semantic_differences_found"]) == {"fixed_rule_numerics", "outcome_row_order_under_delays"}
