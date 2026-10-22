from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_utility_v1_20261022.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_o25_reproduced_under_its_freeze_with_parity() -> None:
    result = _result()
    assert result["precommit_commit"] == "ebd2c8d1ce12b33ec5f445e501e9fbec49b4b170"
    assert result["run_status"] == "STUDY_RESULT"
    assert result["reproduction"]["byte_identical"] == {"cells.jsonl": True, "summary.json": True}
    assert result["parity_failures"] == []


def test_the_framework_comparisons_hold_and_the_llm_prior_ones_change() -> None:
    dispositions = _result()["dispositions"]
    assert dispositions["overall"] == "SOME_CONCLUSIONS_CHANGE_UNDER_UTILITY"
    assert dispositions["disagreements"] == [
        "closed_loop_on_llm_prior/B", "closed_loop_on_unclipped_prior/B", "llm_prior_in_learner/A",
        "llm_prior_in_learner/B", "unclipped_prior_in_learner/B",
    ]
    rows = dispositions["by_comparison"]
    for name in ("best_table", "closed_loop", "hand_rule", "nabm_vs_archetypes"):
        assert rows[f"{name}/A"]["agrees"] and rows[f"{name}/B"]["agrees"]
    assert rows["llm_prior_in_learner/A"]["utility"] == "MOSTLY_NEGATIVE"
