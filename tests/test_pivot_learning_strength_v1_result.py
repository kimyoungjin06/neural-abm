from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
RESULT = RESULTS / "pivot_learning_strength_v1_20261030.summary.json"
SUMMARY = RESULTS / "pivot_learning_strength_v1_20261030.summary.md"
SCENARIOS = ("baseline", "interdisciplinary_seed_grants", "hot_field_hype")
RATE_NAMES = ("1.5", "0.75", "0.375", "0.1875", "0.09375", "0.046875", "0.0234375", "0.01171875")


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _text() -> str:
    return " ".join(SUMMARY.read_text(encoding="utf-8").split())


def _r(value: float | None, places: str) -> str:
    if value is None:
        return "undefined"
    text = str(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))
    return text.replace("-", "−") if text.startswith("-") else f"+{text}"


def _ci(row: dict, places: str = "0.0001") -> str:
    low, high = row["mean_ci95"]
    return f"{_r(row['mean'], places)} [{_r(low, places)}, {_r(high, places)}]"


def test_reproduced_under_its_freeze_with_every_check_passing() -> None:
    result = _result()
    assert result["precommit_commit"] == "42b682ef93e61c5d19c0e24631f7849dbde094c2"
    assert result["execution"]["byte_identical"] == {"o39_summary.json": True, "o39_receipt.json": True}
    checks = result["checks"]
    assert checks["study2_parity"] == {"tracked_runs_checked": 270, "tracked_runs_equal": 270, "tracked_comparisons_checked": 12,
                                       "tracked_comparisons_equal": 12, "frozen_delay_3_runs_equal": 90, "passes": True}
    assert checks["e4_estimator_parity"]["passes"] and checks["e4_estimator_parity"]["largest_difference"] == 0.0
    assert checks["full_sample_fits"] == {"fits": 162, "unconverged": []} and checks["bootstrap_unconverged_fits"] == 0
    assert result["data_access"]["empirical_rows_read"] == 0


def test_the_loop_falls_below_material_at_the_data_allowed_rate() -> None:
    result = _result()
    assert result["status"] == "O39_THE_LOOP_FALLS_BELOW_MATERIAL_AT_THE_DATA_ALLOWED_STRENGTH"
    for delay in ("delay_0", "delay_3"):
        row = result["decision"]["by_delay"][delay]
        assert row["rate"] == 1.5 / 64 and row["disposition"] == result["decision"]["disposition"]
        assert row["classes"] == {"baseline": "BELOW_MATERIAL_NOT_DETECTED", "interdisciplinary_seed_grants": "BELOW_MATERIAL_DETECTED"}
        assert all(r["mean_ci95"][1] < 0.05 for r in row["loop_rows"].values())
    assert all(row["held"] for row in result["decision"]["expectations"].values())
    secondary = result["calibration"]["secondary"]
    assert secondary["delay_0"]["rate"] == 1.5 / 128 and secondary["delay_3"]["rate"] is None
    bound = 0.789381
    for delay in ("delay_0", "delay_3"):
        allowed = [row for row in result["calibration"]["primary"][delay]["rows"] if row["rate"] <= 1.5 / 64]
        assert all(v is None or v <= bound for row in allowed for v in row["gaps"].values())


def test_the_summary_quotes_the_record() -> None:
    result = _result()
    text = _text()
    rows = {delay: {r["rate_index"]: r for r in result["calibration"]["primary"][delay]["rows"]} for delay in ("delay_0", "delay_3")}
    for k, name in enumerate(RATE_NAMES):
        gaps = [_r(rows[d][k]["gaps"][s], "0.01") for d in ("delay_0", "delay_3") for s in SCENARIOS]
        assert f"| {name} | " + " | ".join(gaps) + " |" in text, name
        deltas = [_r(result["comparisons"][f"imitative/{k}/{s}/delay_{d}/failed_pivot_rate"]["mean"], "0.001")
                  for d in (0, 3) for s in SCENARIOS[:2]]
        assert f"| {name} | " + " | ".join(deltas) + " |" in text, name
    loop = {d: result["decision"]["by_delay"][d]["loop_rows"] for d in ("delay_0", "delay_3")}
    assert f"| 3 (headline) | {_ci(loop['delay_3']['baseline'])} | Below material, not detected | " \
           f"{_ci(loop['delay_3']['interdisciplinary_seed_grants'])} | Below material, detected |" in text
    assert f"| 0 | {_ci(loop['delay_0']['baseline'])} | Below material, not detected | " \
           f"{_ci(loop['delay_0']['interdisciplinary_seed_grants'])} | Below material, detected |" in text
    comparisons = result["comparisons"]
    assert f"positive: {_ci(comparisons['imitative/0/hot_field_hype/delay_3/failed_pivot_rate'])}" in text
    assert f"−0.0200 [{_r(comparisons['cautionary/6/hot_field_hype/delay_0/failed_pivot_rate']['mean_ci95'][0], '0.0001')}, " in text
    assert f"{_ci(comparisons['cautionary/6/hot_field_hype/delay_3/failed_pivot_rate'])} at delay 3" in text
    post = result["post_hoc"]["estimator_at_the_allowed_rate"]
    for key in ("interdisciplinary_seed_grants/delay_0", "hot_field_hype/delay_0", "interdisciplinary_seed_grants/delay_3",
                "hot_field_hype/delay_3"):
        row = post[key]["learning_attributable_outcome_gap_visible"]
        assert f"{_r(row['mean'], '0.001')} [{_r(row['low'], '0.001')}, {_r(row['high'], '0.001')}]" in text, key
    material = result["post_hoc"]["smallest_rate_from_the_top_with_a_material_loop"]
    assert {key: row["multiple_of_the_data_allowed_rate"] for key, row in material.items()} == {
        "baseline/delay_0": 16.0, "baseline/delay_3": 32.0,
        "interdisciplinary_seed_grants/delay_0": 4.0, "interdisciplinary_seed_grants/delay_3": 4.0}
    walls = result["execution"]["wall_seconds"]
    assert f"{round(walls['development'])} s for the development run and {round(walls['reproduction'])} s for the reproduction run" in text
