from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_d1_feasibility_20261013.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_d1_reports_feasibility_only() -> None:
    result = _result()
    assert result["status"] == "D1_FEASIBILITY_ONLY_NO_BEHAVIORAL_OUTCOME_REPORTED"
    assert result["boundaries"] == {
        "behavioral_outcomes_reported": False, "llm_calls": 0,
        "empirical_rows_read": 0, "registered": False,
    }
    for row in result["feasibility"].values():
        assert set(row) <= {
            "regime", "arm", "median_microseconds_per_agent_step", "median_decisions_per_run",
            "revealed_outcomes_per_agent_median", "archetype_queries_per_run_median",
        }


def test_nesting_and_parity_passed() -> None:
    result = _result()
    assert result["nesting_checks"]["failed"] == result["core_parity"]["failed"] == 0
    assert result["nesting_checks"]["passed"] == 14 and result["core_parity"]["passed"] == 10


def test_per_agent_learners_see_evidence_in_every_geometry() -> None:
    feasibility = _result()["feasibility"]
    for arm in ("B/L0_per_agent_linear", "B/L0_per_agent_mlp"):
        assert feasibility[arm]["revealed_outcomes_per_agent_median"]["min"] >= 10
