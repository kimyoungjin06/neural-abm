from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
RESULT = RESULTS / "info_spread_policy_ladder_world_calibration_v1_20261024.summary.json"
SUMMARY = RESULTS / "info_spread_policy_ladder_world_calibration_v1_20261024.summary.md"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _r(value: float, places: str = "0.1") -> str:
    text = str(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))
    return text if text.startswith("-") else f"+{text}"


def test_o33_reproduced_under_its_freeze_with_parity_and_consistency() -> None:
    result = _result()
    assert result["precommit_commit"] == "a926c25faf2dd359ec731c0f2834839c8f6faf06"
    assert result["run_status"] == "STUDY_RESULT"
    assert result["reproduction"]["byte_identical"] == {"cells.jsonl": True, "summary.json": True}
    assert result["parity_failures"] == [] and result["consistency_failures"] == []


def test_each_prior_matched_its_source_in_the_world() -> None:
    result = _result()
    for regime, report in result["world_calibration"].items():
        for name, row in report.items():
            if name == "source_shares":
                continue
            assert abs(row["achieved"] / row["target"] - 1) < 0.0002, (regime, name)
            assert row["method"] == ("KEPT_COHORT_LOCATION" if (name, regime) == ("F1_pcalw", "A") else "BISECTED")
    check = result["world_match_check"]
    assert len(check) == 8 and all(row["matches"] for row in check.values())
    ratios = [row["ratio_to_calibration_source"] for row in check.values()]
    assert (min(ratios), max(ratios)) == (0.9842, 1.0011)


def test_c63_holds_only_in_regime_a_and_c64_does_not_hold() -> None:
    dispositions = _result()["dispositions"]
    assert dispositions["overall"] == "O33_NEITHER_HOLDS"
    assert {case: (row["attribution"], row["o28_attribution"]) for case, row in dispositions["by_case"].items()} == {
        "P0/A": ("CALIBRATION", "CALIBRATION"), "P0/B": ("BOTH", "CALIBRATION"),
        "P0u/A": ("NEITHER_CONSISTENTLY", "NEITHER_CONSISTENTLY"), "P0u/B": ("ANSWERS", "NEITHER_CONSISTENTLY"),
    }
    p0u_b = dispositions["by_case"]["P0u/B"]["counts"]
    assert p0u_b["answers_given_hand_calibration"]["NEGATIVE"] == 6
    assert p0u_b["answers_given_llm_calibration"]["NEGATIVE"] == 5


def test_the_summary_quotes_the_recorded_ranges() -> None:
    result = _result()
    table = result["table"]
    text = " ".join(SUMMARY.read_text(encoding="utf-8").split())
    for prior, regime, name in (("P0", "A", "calibration_given_hand_answers"), ("P0", "B", "answers_given_hand_calibration"),
                                ("P0u", "B", "answers_given_llm_calibration"), ("P0u", "B", "calibration_given_hand_answers")):
        values = [v[f"{prior}/{name}"]["mean_cell_delta"] for k, v in table.items() if k.startswith(f"{regime}/")]
        span = f"{_r(min(values))} to {_r(max(values))}".replace("-", "−")
        assert span in text, (prior, regime, name, span)
    # Regime B, at matched calibration: an LLM direction against F1's direction.
    pairs = (("L1_hcalw", "L0"), ("L1", "L0_pcalw"), ("L1u_hcalw", "L0"), ("L1u", "L0_ucalw"))
    rows = [v for k, v in result["learner_description"].items() if k.startswith("B/")]
    shares = [row[a]["mean_unverified_shares"] / row[b]["mean_unverified_shares"] - 1 for row in rows for a, b in pairs]
    precision = [row[a]["mean_unverified_share_precision"] - row[b]["mean_unverified_share_precision"]
                 for row in rows for a, b in pairs]
    assert f"between {-round(100 * min(shares))}% fewer and {_r(100 * max(shares)).lstrip('+')}% more" in text
    assert f"from {_r(min(precision), '0.001')} to {_r(max(precision), '0.001')}".replace("-", "−") in text
