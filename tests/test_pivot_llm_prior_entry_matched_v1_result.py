from __future__ import annotations

import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
RESULT = RESULTS / "pivot_llm_prior_entry_matched_v1_20261029.summary.json"
SUMMARY = RESULTS / "pivot_llm_prior_entry_matched_v1_20261029.summary.md"
V3 = RESULTS / "pivot_llm_prior_pilot_v3_20260929.summary.json"
WORLDS = ("linear_fit", "smooth_interaction_fit", "axis_interaction_fit")


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _text() -> str:
    return " ".join(SUMMARY.read_text(encoding="utf-8").split())


def _r(value: float, places: str = "0.001") -> str:
    text = str(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))
    return text.replace("-", "−") if text.startswith("-") else f"+{text}"


def test_reproduced_under_its_freeze_with_every_check_passing() -> None:
    result = _result()
    assert result["precommit_commit"] == "243dad757f69254f666ec92865ddccf162901f2b"
    assert all(result["reproduction"]["byte_identical"].values())
    assert set(result["reproduction"]["byte_identical"]) == {
        "calibration.json", "runs.jsonl", "scorecard.json", "scorecard.md", "receipt.json",
    }
    assert result["decision_status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert result["calibration"]["hand_slot_nesting"] == "PASS_EXACT"
    assert result["execution"]["raw_row_count"] == 1152
    assert result["execution"]["new_llm_calls"] == 0


def test_entry_was_matched_in_the_world() -> None:
    result = _result()
    records = result["calibration"]["records"]
    assert len(records) == 6
    for level in ("llm_entry", "hand_entry"):
        shifts = {row["intercept_shift"] for row in records if row["level"] == level}
        assert len(shifts) == 1, level
    assert all(abs(row["in_sample_ratio"] - 1) <= 0.01 for row in records)
    check = result["decisions"]["entry_match_check"]
    assert check["status"] == "PASS"
    registered = [row for row in check["records"] if row["registered_check"]]
    assert len(registered) == 6 and all(0.95 <= row["fixed_rule_entry_ratio"] <= 1.05 for row in registered)
    gap = result["post_hoc_descriptive"]["residual_entry_gap"]
    assert max(row["implied_precision_advantage_to_the_matched_hand"] for row in gap.values()) < 0.005


def test_c22_precision_is_selectivity_and_the_expectations_are_recorded() -> None:
    result = _result()
    decisions = result["decisions"]
    assert result["status"] == "C22_PRECISION_IS_SELECTIVITY"
    fixed = decisions["ranking_at_llm_selectivity_fixed_rule"]["cells"]
    assert {world: fixed[f"{world}/fixed_rule"]["outcome"] for world in WORLDS} == {
        "linear_fit": "FIRST_BETTER",
        "smooth_interaction_fit": "NO_CLEAR_DIFFERENCE",
        "axis_interaction_fit": "NO_CLEAR_DIFFERENCE",
    }
    assert decisions["ranking_at_llm_selectivity_learners"]["counts"] == {
        "FIRST_BETTER": 1, "SECOND_BETTER": 1, "NO_CLEAR_DIFFERENCE": 7,
    }
    assert decisions["ranking_at_hand_selectivity"]["counts"] == {
        "FIRST_BETTER": 0, "SECOND_BETTER": 7, "NO_CLEAR_DIFFERENCE": 5,
    }
    matched = decisions["c22_against_entry_matched_hand"]
    assert matched["precision_better_cells"] == [] and matched["dominating_cells"] == []
    assert matched["precision_worse_cells"] == ["linear_fit/mlp_batch"]
    replication = decisions["c22_replication_against_hand"]
    assert len(replication["precision_better_cells"]) == 9
    assert replication["dominating_cells"] == ["linear_fit/mlp_batch", "linear_fit/tree_batch"]
    assert len(replication["mean_only_dominating_cells"]) == 4
    held = {key: row["held"] for key, row in result["expectations"].items()}
    assert held == {"E1": True, "E2": True, "E3": False}


def test_the_v3_interval_recount_is_reproducible_from_the_v3_record() -> None:
    cells = json.loads(V3.read_text(encoding="utf-8"))["registered_decisions"]["hybrid_20"]["cells"]
    passing = 0
    for cell in cells.values():
        gap = cell["learning_noninferior_versus_hand"]["hand_minus_condition_probe_brier"]
        values = gap["values"]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        low = mean - student_t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))
        passing += cell["precision_versus_hand"]["pass"] and low >= -0.005
    assert f"{passing} of 9" == _result()["c22_across_seed_sets"]["v3_interval_rule_recomputed_post_hoc"]


def test_the_summary_quotes_the_record() -> None:
    result = _result()
    text = _text()
    decisions = result["decisions"]
    for world, label in zip(WORLDS, ("linear", "smooth interaction", "axis interaction"), strict=True):
        cell = decisions["ranking_at_llm_selectivity_fixed_rule"]["cells"][f"{world}/fixed_rule"]
        bounds = cell["interval_95"]
        quoted = f"| {label} | `{cell['outcome']}` | {_r(bounds['mean'])} [{_r(bounds['low'])}, {_r(bounds['high'])}] |"
        assert quoted in text, quoted
    worse = decisions["c22_against_entry_matched_hand"]["cells"]["linear_fit/mlp_batch"]["precision"]["interval_95"]
    assert f"{_r(worse['mean'])} [{_r(worse['low'])}, {_r(worse['high'])}]" in text
    others = [
        cell["precision"]["interval_95"]["mean"]
        for key, cell in decisions["c22_against_entry_matched_hand"]["cells"].items()
        if key != "linear_fit/mlp_batch"
    ]
    assert f"range from {_r(min(others))} to {_r(max(others))}" in text
    gains = [cell["precision"]["interval_95"]["mean"] for cell in decisions["c22_replication_against_hand"]["cells"].values()]
    assert f"from {_r(min(gains))} to {_r(max(gains))}" in text
    records = {row["level"]: row for row in result["calibration"]["records"]}
    assert f"{_r(records['llm_entry']['intercept_shift'])} |" in text
    assert f"{_r(records['hand_entry']['intercept_shift'])} |" in text
    for row in decisions["entry_match_check"]["records"]:
        if row["registered_check"]:
            assert f"| {_r(row['fixed_rule_entry_ratio'])[1:]} |" in text
    ratio = next(r for r in decisions["entry_match_check"]["records"] if r["matched"] == "hand_at_llm_entry")
    assert f"{_r((1 - ratio['fixed_rule_entry_ratio']) * 100, '0.1')[1:]}% fewer agents" in text
    walls = result["execution"]["wall_seconds"]
    assert f"{round(walls['development'])} s for the development run" in text
    assert f"{round(walls['reproduction'])} s for the reproduction run" in text
    seeds = result["c22_across_seed_sets"]
    assert f"| v3 (C22) | {seeds['v3_original_mean_only_rule']} | {seeds['v3_interval_rule_recomputed_post_hoc']}" in text
    assert f"| this study | {seeds['this_study_mean_only_rule']} | {seeds['this_study_interval_rule']} |" in text
