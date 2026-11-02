from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
RESULT = RESULTS / "pivot_crossover_prediction_v1_20261102.summary.json"
SUMMARY = RESULTS / "pivot_crossover_prediction_v1_20261102.summary.md"
TABLE_SIZES = (1, 2, 4, 8, 16)
BINS = ("1 to 14", "15 to 49", "50 to 149", "150 or more")
WINNER = {"POSITIVE": "learner", "NEGATIVE": "table", "INDISTINGUISHABLE": "—", "NEUTRAL": "—"}


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _text() -> str:
    return " ".join(SUMMARY.read_text(encoding="utf-8").split())


def _q(value: float, places: str = "0.0001", sign: bool = False) -> str:
    text = str(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))
    if text.startswith("-"):
        return text.replace("-", "−")
    return f"+{text}" if sign else text


def test_reproduced_under_its_freeze_on_the_discovery_partition() -> None:
    result = _result()
    assert result["precommit_commit"] == "015eea5fa5720406491b1504433dbd6dea4d257f"
    assert result["execution"]["byte_identical"] == {
        "summary.json": True, "receipt.json": True, "private/episode_quarters.parquet": True}
    assert result["execution"]["host"] == "THE_WORKSTATION"
    assert result["data_access"] == {"partition": "DISCOVERY_ONLY", "labels_read": 6291,
                                     "confirmatory_outcome_rows_read": 0, "g4_outcome_rows_read": 0}
    assert result["checks"]["after_the_labels"] == {"labelled_precise_entrants": True}
    assert all(result["checks"]["before_any_label"].values())


def test_the_generator_check_passed_before_the_labels() -> None:
    check = _result()["generator_check"]
    assert check["passes"] is True and check["replicates"] == 40
    for row in check["taus"].values():
        assert row["agreement_share"] >= 0.8 and row["reversal_share"] <= 0.05


def test_the_learner_wins_where_determined_and_the_rule_agrees() -> None:
    result = _result()
    assert result["status"] == "E3S2_THE_LEARNER_WINS_WHERE_DETERMINED_AND_THE_RULE_AGREES"
    score = result["score"]
    assert score["signs"] == ["POSITIVE"] and score["reversals"] == 0
    assert score["agreements"] == score["determined"] == score["baseline_learner_always_wins_agreements"] == 7
    groups = result["groups"]
    assert all(groups[f"k={k}/bin=3"]["label"] == "POSITIVE" for k in TABLE_SIZES)
    assert all(groups[f"k={k}/bin=0"]["label"] == "INDISTINGUISHABLE" and groups[f"k={k}/bin=0"]["mean"] < 0 for k in TABLE_SIZES)
    assert {key: row["held"] for key, row in result["expectations"].items()} == {
        "X1": True, "X2": True, "X3": False, "X4": True, "X5": True}


def test_the_summary_quotes_the_record() -> None:
    result = _result()
    text = _text()
    for b, name in enumerate(BINS):
        for k in TABLE_SIZES:
            g = result["groups"][f"k={k}/bin={b}"]
            row = (f"| {name} | {k} | {g['forecasts']:,} ({g['episodes']}) | {_q(g['mean'], sign=True)} | "
                   f"[{_q(g['low'], sign=True)}, {_q(g['high'], sign=True)}] | {WINNER[g['label']]} | "
                   f"{_q(g['predicted'], sign=True)} | {WINNER[g['predicted_label']]} |")
            assert row in text, row
    for tau, row in result["generator_check"]["taus"].items():
        line = (f"| {tau.split('=')[1]} | {row['determined']} | {_q(row['agreement_share'], '0.01')} | {_q(row['match_share'], '0.01')} | "
                f"{row['reversals']} | {row['replicates_with_a_positive_group']} of 40 | Yes |")
        assert line in text, line
    described = result["described"]
    for k in TABLE_SIZES:
        row = described["tables"][f"k={k}"]
        assert (f"| {k} | {_q(row['H'], '0.00001')} | {_q(row['crossing_labels'], '0.1')} | "
                f"{_q(100 * row['share_of_forecasts_beyond_the_crossing'], '0.1')}% |") in text
    assert f"V = {_q(described['V'])}" in text
    assert f"standard deviation of true shares is {_q(described['between_episode_sd_of_shares'], '0.001')}" in text
    assert f"unweighted mean share of {_q(described['mean_episode_share'], '0.001')}" in text
    ratio = described["V"] / described["between_episode_variance_of_shares"]
    assert f"variance of shares, {_q(ratio, '0.1')} here" in text
    walls = result["execution"]["wall_seconds"]
    assert f"{round(walls['development'])} s for the development run and {round(walls['reproduction'])} s" in text
    assert f"Status: `{result['status']}`" in text
