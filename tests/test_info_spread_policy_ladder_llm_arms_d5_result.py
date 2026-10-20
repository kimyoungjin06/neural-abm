from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_llm_arms_d5_20261019.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_d5_reproduced_under_the_freeze_with_parity_and_a_deterministic_elicitation() -> None:
    result = _result()
    assert result["precommit_commit"] == "dcc6e7495d378c7375544cdd882471d837aaf8eb"
    assert result["run_status"] == "STUDY_RESULT"
    assert result["reproduction"]["byte_identical"] == {"cells.jsonl": True, "summary.json": True}
    assert result["elicitation"]["byte_identical"] is True
    assert result["parity_failures"] == []


def test_the_registered_dispositions() -> None:
    reading = _result()["registered_reading"]
    assert reading["design_check"] == "NULL_WORLD_PASSES"
    assert reading["by_comparison"] == {
        "llm_prior_in_learner": {"A": "MIXED_OR_NONE", "B": "MIXED_OR_NONE"},
        "nabm_vs_archetypes": {"A": "MOSTLY_POSITIVE", "B": "MOSTLY_POSITIVE"},
        "closed_loop_on_llm_prior": {"A": "MOSTLY_POSITIVE", "B": "MIXED_OR_NONE"},
    }
    assert reading["counts"]["nabm_vs_archetypes"]["A"]["POSITIVE"] == 6
    assert reading["counts"]["nabm_vs_archetypes"]["B"]["POSITIVE"] == 6


def test_p0_in_regime_a_saturated_under_the_clip() -> None:
    result = _result()
    assert result["llm_arms"]["A"]["p0_distillation_r_squared"] < 0.05
    assert result["llm_arms"]["B"]["p0_distillation_r_squared"] > 0.4
    assert result["post_run_description_not_registered"]["p0_readout_saturation"]["share_of_p0_log_odds_below_the_clip"]["A"] > 0.95
