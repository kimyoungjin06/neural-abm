from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
RESULT = RESULTS / "info_spread_policy_ladder_prior_calibration_v1_20261023.summary.json"
SUMMARY = RESULTS / "info_spread_policy_ladder_prior_calibration_v1_20261023.summary.md"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _r(value: float) -> str:
    text = str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return text if text.startswith("-") else f"+{text}"


def test_o28_reproduced_under_its_freeze_with_parity_and_consistency() -> None:
    result = _result()
    assert result["precommit_commit"] == "555fd497b36576cb86a93007357b3b3a61e5e60f"
    assert result["run_status"] == "STUDY_RESULT"
    assert result["reproduction"]["byte_identical"] == {"cells.jsonl": True, "summary.json": True}
    assert result["parity_failures"] == [] and result["consistency_failures"] == []
    assert all(row["moves_to_its_calibration_source"] for row in result["calibration_check"].values())


def test_calibration_carries_d5s_prior_but_not_the_unclipped_one() -> None:
    dispositions = _result()["dispositions"]
    assert dispositions["overall"] == "O28_ATTRIBUTION_DIFFERS_ACROSS_COSTED_CASES"
    assert dispositions["costed_cases"] == ["P0/A", "P0/B", "P0u/B"]
    assert {case: row["attribution"] for case, row in dispositions["by_case"].items()} == {
        "P0/A": "CALIBRATION", "P0/B": "CALIBRATION", "P0u/A": "NEITHER_CONSISTENTLY", "P0u/B": "NEITHER_CONSISTENTLY",
    }
    p0u_b = dispositions["by_case"]["P0u/B"]["counts"]
    assert p0u_b["answers_given_hand_calibration"]["NEGATIVE"] == 6
    assert p0u_b["answers_given_llm_calibration"]["POSITIVE"] == 4


def test_the_summary_quotes_the_recorded_ranges() -> None:
    table = _result()["table"]
    text = " ".join(SUMMARY.read_text(encoding="utf-8").split())
    for prior, regime, name in (("P0", "A", "calibration_given_hand_answers"), ("P0u", "B", "answers_given_hand_calibration"),
                                ("P0u", "A", "answers_given_llm_calibration")):
        values = [v[f"{prior}/{name}"]["mean_cell_delta"] for k, v in table.items() if k.startswith(f"{regime}/")]
        span = f"{_r(min(values))} to {_r(max(values))}".replace("-", "−").lstrip("+")
        assert span in text, (prior, regime, name, span)
