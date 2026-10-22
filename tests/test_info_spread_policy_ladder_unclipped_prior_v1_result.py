from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_unclipped_prior_v1_20261021.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_o24_reproduced_under_its_freeze_with_parity() -> None:
    result = _result()
    assert result["precommit_commit"] == "eb4905787cc1bce2747536633e03944fb2db58f2"
    assert result["run_status"] == "STUDY_RESULT"
    assert result["reproduction"]["byte_identical"] == {"cells.jsonl": True, "summary.json": True}
    assert result["parity_failures"] == []


def test_the_unclipped_prior_carries_information_but_adds_little_to_the_learner() -> None:
    result = _result()
    assert result["unclipped_priors"]["A"]["p0u_distillation_r_squared"] > 0.75
    by = result["dispositions"]["by_comparison"]
    assert by["unclipped_prior_vs_hand"] == {"A": "MOSTLY_POSITIVE", "B": "MOSTLY_POSITIVE"}
    assert by["unclipped_prior_in_learner"] == {"A": "MIXED_OR_NONE", "B": "MIXED_OR_NONE"}
    counts = result["dispositions"]["counts"]["unclipped_prior_in_learner"]
    assert counts["A"]["NEGATIVE"] == counts["B"]["NEGATIVE"] == 0
    assert result["dispositions"]["design_check"] == "NULL_WORLD_PASSES"
