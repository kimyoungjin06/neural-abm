from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
RESULT = RESULTS / "info_spread_crossover_prediction_v1_20261030.summary.json"
SUMMARY = RESULTS / "info_spread_crossover_prediction_v1_20261030.summary.md"
VISIBILITIES = (("0.2", "0.2"), ("0.5", "0.5"), ("1", "1.0"))
RATES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _text() -> str:
    return " ".join(SUMMARY.read_text(encoding="utf-8").split())


def _q(value: float, places: str, sign: bool = False) -> str:
    text = str(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))
    if text.startswith("-"):
        return text.replace("-", "−")
    return f"+{text}" if sign else text


def test_reproduced_under_its_freeze() -> None:
    result = _result()
    assert result["precommit_commit"] == "871ace34cda273bce086a232c70f77c9b5e8a254"
    assert result["execution"]["byte_identical"] == {"measurement.jsonl": True, "cells.jsonl": True, "summary.json": True}
    assert result["execution"]["host"] == "GPU_SERVER_CPU_ONLY"
    assert result["boundary"]["empirical_rows_read"] == 0


def test_the_rule_beats_both_baselines_but_misses_the_bound() -> None:
    result = _result()
    scoring = result["scoring"]
    assert result["status"] == scoring["status"] == "E3_THE_RULE_BEATS_THE_BASELINES_BUT_MISSES_BY_MORE_THAN_THE_BOUND"
    errors = scoring["mean_absolute_error"]
    assert 0.05 < errors["predicted"] < min(errors["baseline_constant"], errors["baseline_table_size_only"])
    assert scoring["within_bound"] == {"predicted": 6, "baseline_constant": 2, "baseline_table_size_only": 2}
    assert {key: row["held"] for key, row in result["expectations"].items()} == {
        "X1": True, "X2": True, "X3": True, "X4": False, "X5": True}
    series = scoring["series"]
    late = [series[f"pi={v}/k={k}"]["predicted"] > series[f"pi={v}/k={k}"]["observed"] for v in ("0.2", "0.5") for k in (2, 4, 8)]
    assert all(late)
    assert all(abs(series[f"pi=1/k={k}"]["predicted"] - series[f"pi=1/k={k}"]["observed"]) <= 0.01 for k in (2, 4, 8))


def test_the_summary_quotes_the_record() -> None:
    result = _result()
    text = _text()
    series = result["scoring"]["series"]
    for key, label in VISIBILITIES:
        for k in (2, 4, 8):
            s = series[f"pi={key}/k={k}"]
            row = (f"| {label} | {k} | {_q(s['observed'], '0.001')} | {_q(s['predicted'], '0.001')} | "
                   f"{_q(s['baseline_constant'], '0.001')} | {_q(s['baseline_table_size_only'], '0.001')} |")
            assert row in text, row
        components = result["predicted"][f"pi={key}/components"]
        assert f"| {label} | {_q(components['mean_Dd'], '0.0001')} | {_q(components['mean_V'], '0.0001')} |" in text
        for k in (1, 2, 4, 8):
            cells = []
            for p in RATES:
                group = result["observed"][f"pi={key}/k={k}/p={p:g}"]
                mark = {"POSITIVE": "+", "NEGATIVE": "−", "INDISTINGUISHABLE": "0"}[group["label"]]
                cells.append(f"{_q(group['mean_cell_delta'], '0.001', sign=True)} {mark}")
            assert f"| {label} | {k} | " + " | ".join(cells) + " |" in text, (key, k)
    errors, within = result["scoring"]["mean_absolute_error"], result["scoring"]["within_bound"]
    assert f"| **The rule** | {_q(errors['predicted'], '0.0001')} | {within['predicted']} of 9 |" in text
    assert f"| Constant (0.175) | {_q(errors['baseline_constant'], '0.0001')} | {within['baseline_constant']} of 9 |" in text
    assert f"| Table size only | {_q(errors['baseline_table_size_only'], '0.0001')} | {within['baseline_table_size_only']} of 9 |" in text
    misses = [abs(series[f"pi=0.2/k={k}"]["predicted"] - series[f"pi=0.2/k={k}"]["observed"]) for k in (2, 4, 8)]
    assert f"late by {_q(min(misses), '0.001')} to {_q(max(misses), '0.001')}" in text
    rest = [abs(series[f"pi={v}/k={k}"]["predicted"] - series[f"pi={v}/k={k}"]["observed"]) for v in ("0.5", "1") for k in (2, 4, 8)]
    assert f"its mean error would be {_q(sum(rest) / 6, '0.001')}" in text
    walls = result["execution"]["wall_seconds"]
    assert f"{round(walls['development']):,} s for the development run and {round(walls['reproduction']):,} s" in text
