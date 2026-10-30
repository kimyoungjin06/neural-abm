from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prototype_pivot_learning_strength_v1.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
o39 = importlib.import_module(SCRIPT.stem)
CONTRACT = o39.load_contract()
E4, E4_CONTRACT, STUDY2 = o39.load_modules(CONTRACT)
TRACKED = json.loads(o39._binding(CONTRACT, o39.TRACKED_ROLE).read_text(encoding="utf-8"))
RECORD = json.loads(o39._binding(CONTRACT, o39.E4_RECORD_ROLE).read_text(encoding="utf-8"))
SCENARIOS = {s.name: s for s in STUDY2.study1.study_scenarios()}
RATES = CONTRACT["simulation"]["learning_rates"]
GRANTS = "interdisciplinary_seed_grants"


def _narrow(scenarios: list[str], delays: list[int], rates: list[float]) -> dict:
    contract = copy.deepcopy(CONTRACT)
    contract["simulation"].update(scenarios=scenarios, outcome_delays=delays, learning_rates=rates)
    return contract


# -- contract ----------------------------------------------------------------------------


def test_the_contract_binds_e4_study2_and_the_ledger() -> None:
    roles = {b["role"] for b in CONTRACT["source_bindings"]}
    assert {o39.STUDY2_ROLE, o39.TRACKED_ROLE, o39.E4_SCRIPT_ROLE, o39.E4_CONTRACT_ROLE, o39.E4_RECORD_ROLE} <= roles
    assert CONTRACT["simulation"]["scenarios"] == E4_CONTRACT["simulation"]["scenarios"]
    assert CONTRACT["calibration"]["primary"]["value"] == RECORD["data"]["estimates"]["state/outcome_gap_visible"]["high"]
    assert CONTRACT["calibration"]["secondary"]["value"] == RECORD["data"]["estimates"]["state/outcome_gap_visible"]["mean"]
    assert CONTRACT["boundary"]["empirical_rows_read"] == 0 and CONTRACT["boundary"]["calibrates_study2"] is False
    assert CONTRACT["execution"]["identity_artifacts"] == o39.IDENTITY


def test_the_grid_halves_the_tracked_rate_seven_times() -> None:
    assert RATES[0] == STUDY2.LearningStudyConfig().learning_rate == 1.5
    assert all(a / b == 2.0 for a, b in zip(RATES, RATES[1:]))
    assert len(RATES) == 8 and RATES[-1] < 0.0118


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("simulation", "learning_rates"), [1.5, 0.5], "halved"),
        (("simulation", "outcome_delays"), [0, 2], "delays"),
        (("simulation", "estimated_arms"), ["imitative"], "frozen and imitative"),
        (("simulation", "scenarios"), ["baseline", "hot_field_hype", "interdisciplinary_seed_grants"], "E4's order"),
        (("calibration", "primary", "value"), 1.0, "primary data bound"),
        (("calibration", "quantity"), "state/visibility_did", "visible outcome gap"),
        (("decision", "material_failed_pivot_delta"), 0.1, "material"),
        (("decision", "primary_delay"), 0, "headline"),
        (("boundary", "empirical_rows_read"), 1, "boundary"),
        (("execution", "thread_variables"), ["OMP_NUM_THREADS"], "thread"),
    ],
)
def test_contract_mutations_are_refused(path: tuple[str, ...], value: object, message: str) -> None:
    contract = copy.deepcopy(CONTRACT)
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(o39.StrengthError, match=message):
        o39.validate_contract(contract)


def test_a_drifted_binding_is_refused() -> None:
    contract = copy.deepcopy(CONTRACT)
    contract["source_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(o39.StrengthError, match="drifted"):
        o39.validate_contract(contract)


# -- Study 2 over the grid ---------------------------------------------------------------


def test_the_tasks_cover_every_arm_rate_scenario_and_delay_once() -> None:
    tasks = o39.simulation_tasks(CONTRACT)
    assert len(tasks) == len(set(tasks)) == 3 * 2 * (1 + 2 * len(RATES))
    assert [t for t in tasks if t[0] == "frozen"] == [("frozen", None, s, d) for s in CONTRACT["simulation"]["scenarios"] for d in (0, 3)]


def test_the_tracked_rate_reproduces_study2_and_a_zero_rate_is_the_frozen_arm() -> None:
    for arm in ("imitative", "cautionary"):
        runs, _logs = o39.run_task(E4, STUDY2, SCENARIOS, (arm, 0, GRANTS, 0), rates=RATES, replicates=1)
        recorded = TRACKED["scenarios"][GRANTS][arm]["replicate_runs"][0]
        assert all(runs[0][key] == recorded[key] for key in E4.PARITY_KEYS), arm
    _runs, frozen = o39.run_task(E4, STUDY2, SCENARIOS, ("frozen", None, GRANTS, 0), rates=RATES, replicates=1)
    _runs, zero = o39.run_task(E4, STUDY2, SCENARIOS, ("imitative", 0, GRANTS, 0), rates=[0.0], replicates=1)
    assert zero == frozen
    _runs, small = o39.run_task(E4, STUDY2, SCENARIOS, ("imitative", 7, GRANTS, 0), rates=RATES, replicates=1)
    _runs, tracked = o39.run_task(E4, STUDY2, SCENARIOS, ("imitative", 0, GRANTS, 0), rates=RATES, replicates=1)
    assert small != tracked
    assert STUDY2.LearningStudyConfig().learning_rate == 1.5
    assert STUDY2.PivotEnvironment.__name__ == "PivotEnvironment" and STUDY2.PivotPolicyAgent.__name__ == "PivotPolicyAgent"


def test_comparisons_pair_replicates_with_study2s_distribution() -> None:
    contract = _narrow(["baseline"], [3], [0.75])
    frozen = [{"failed_pivot_rate": 0.1, "productive_pivot_rate": 0.2}, {"failed_pivot_rate": 0.0, "productive_pivot_rate": 0.4}]
    learning = [{"failed_pivot_rate": 0.3, "productive_pivot_rate": 0.2}, {"failed_pivot_rate": 0.2, "productive_pivot_rate": 0.1}]
    results = {("frozen", None, "baseline", 3): {"runs": frozen}, ("imitative", 0, "baseline", 3): {"runs": learning},
               ("cautionary", 0, "baseline", 3): {"runs": frozen}}
    rows = o39.comparisons(contract, STUDY2, results)
    row = rows["imitative/0/baseline/delay_3/failed_pivot_rate"]
    assert row["paired_deltas"] == [0.2, 0.2] and row["rate"] == 0.75
    assert {k: row[k] for k in ("mean", "mean_ci95")} == {k: STUDY2._distribution([0.2, 0.2])[k] for k in ("mean", "mean_ci95")}
    assert rows["cautionary/0/baseline/delay_3/productive_pivot_rate"]["paired_deltas"] == [0.0, 0.0]
    assert len(rows) == 2 * 2


# -- the estimator -----------------------------------------------------------------------


def test_the_reused_estimator_gives_e4s_recorded_points_at_the_tracked_rate() -> None:
    contract = _narrow([GRANTS], [3], [1.5])
    results = {}
    for task in (("frozen", None, GRANTS, 3), ("imitative", 0, GRANTS, 3)):
        runs, logs = o39.run_task(E4, STUDY2, SCENARIOS, task, rates=[1.5], replicates=30)
        results[task] = {"runs": runs, "logs": logs}
    problems, points, full_fits = o39.estimation_problems(contract, E4, E4_CONTRACT, STUDY2, results)
    assert full_fits == {"fits": 6, "unconverged": []}
    for problem in problems.values():
        problem["multiplicities"] = problem["multiplicities"][:2]
    draws = o39.run_bootstrap(o39.DEFAULT_CONFIG, E4, problems, E4._settings(E4_CONTRACT["estimator"]), workers=1)
    report = o39.estimation_report(E4, problems, points, draws)
    recorded = RECORD["simulation"][f"simulation/{GRANTS}/delay_3"]
    for arm, key in (("frozen", "frozen/none"), ("imitative", "imitative/0")):
        for label, row in recorded["by_arm"][arm].items():
            assert report[f"{key}/{GRANTS}/delay_3"]["rows"][label]["mean"] == row["mean"], (arm, label)
    for label, row in recorded["learning_attributable"]["imitative"].items():
        assert report[f"imitative/0/{GRANTS}/delay_3"]["learning_attributable"][label]["mean"] == row["mean"], label
    assert report[f"imitative/0/{GRANTS}/delay_3"]["state_events"] == recorded["state_events"]["imitative"]


def test_spawned_bootstrap_workers_equal_the_in_process_bootstrap() -> None:
    rng = np.random.default_rng(5)
    rows = 400
    group = rng.integers(0, 8, rows)
    cluster = group // 2
    x = rng.integers(0, 2, rows).astype(np.float64)
    events = (rng.random(rows) < 1 / (1 + np.exp(-(-1.0 + 0.8 * x)))).astype(np.float64)
    cells = E4.collapse(group, cluster, x[:, None], np.ones(rows), events, ["exposed"])
    settings = E4._settings(E4_CONTRACT["estimator"])
    fit = E4.fit_logit(cells, **settings)
    problem = {"models": [{"arm": "imitative", "model": "exposure", "cells": cells, "beta": fit["beta"], "reference": "P_F",
                           "quantities": ["beta:exposed"], "converged": True}],
               "multiplicities": E4.multiplicities(4, 6, 1, 1, 0, 3)}
    serial = o39.run_bootstrap(o39.DEFAULT_CONFIG, E4, {"p": problem}, settings, workers=1, chunk=4)
    spawned = o39.run_bootstrap(o39.DEFAULT_CONFIG, E4, {"p": problem}, settings, workers=2, chunk=4)
    assert np.array_equal(serial["p"]["draws"], spawned["p"]["draws"]) and serial["p"]["draws"].shape == (6, 1)
    rebuilt = E4.Cells(**o39._transport({"p": problem})["p"]["models"][0]["cells"])
    assert np.array_equal(rebuilt.design, cells.design) and rebuilt.names == cells.names


def test_e4_parity_flags_a_value_beyond_the_tolerance_and_a_definedness_change() -> None:
    contract = _narrow(["baseline"], [0], [1.5])
    row = {"mean": 1.0, "low": 0.5, "high": 1.5}
    record = {"simulation": {"simulation/baseline/delay_0": {"by_arm": {"frozen": {"q": row}, "imitative": {"q": row}},
                                                            "learning_attributable": {"imitative": {"q": row}}}}}
    report = {"frozen/none/baseline/delay_0": {"rows": {"q": dict(row)}},
              "imitative/0/baseline/delay_0": {"rows": {"q": dict(row)}, "learning_attributable": {"q": dict(row)}}}
    assert o39.e4_parity(contract, record, report)["passes"]
    report["imitative/0/baseline/delay_0"]["rows"]["q"]["low"] = 0.5 + 2e-5
    assert o39.e4_parity(contract, record, report)["mismatched"] == ["baseline/0/q/low"]
    report["imitative/0/baseline/delay_0"]["rows"]["q"] = {"mean": None, "low": None, "high": None}
    assert len(o39.e4_parity(contract, record, report)["mismatched"]) == 3


# -- calibration and decision ------------------------------------------------------------


def _report(gaps: dict[int, dict[str, float | None]], delay: int = 3) -> dict:
    return {o39.problem_key("imitative", k, s, delay): {"rows": {"state/outcome_gap_visible": {"mean": value}}}
            for k, by_scenario in gaps.items() for s, value in by_scenario.items()}


def test_calibration_takes_the_largest_rate_below_which_every_rate_is_allowed() -> None:
    contract = _narrow(["baseline", GRANTS], [3], [1.5, 0.75, 0.375, 0.1875])
    gaps = {0: {"baseline": 9.0, GRANTS: 8.0}, 1: {"baseline": None, GRANTS: 0.7}, 2: {"baseline": None, GRANTS: 0.9},
            3: {"baseline": None, GRANTS: 0.2}}
    out = o39.calibrate(contract, _report(gaps), 0.789381)["delay_3"]
    assert out["rate_index"] == 3 and out["rate"] == 0.1875
    assert [row["allowed"] for row in out["rows"]] == [True, False, True, False]
    gaps[2][GRANTS] = 0.5
    assert o39.calibrate(contract, _report(gaps), 0.789381)["delay_3"]["rate"] == 0.75
    gaps[1] = {"baseline": None, GRANTS: None}
    assert o39.calibrate(contract, _report(gaps), 0.789381)["delay_3"]["rate"] == 0.375
    gaps[3][GRANTS] = 0.8
    out = o39.calibrate(contract, _report(gaps), 0.789381)["delay_3"]
    assert out["rate_index"] is None and out["rate"] is None


@pytest.mark.parametrize(
    ("interval", "label"),
    [((0.05, 0.2), "MATERIAL"), ((0.01, 0.049), "BELOW_MATERIAL_DETECTED"), ((0.0, 0.03), "BELOW_MATERIAL_NOT_DETECTED"),
     ((0.04, 0.06), "UNRESOLVED"), ((-0.01, 0.05), "UNRESOLVED")],
)
def test_the_material_classes(interval: tuple[float, float], label: str) -> None:
    assert o39.classify({"mean_ci95": list(interval)}, 0.05) == label


@pytest.mark.parametrize(
    ("baseline", "grants", "status"),
    [("MATERIAL", "MATERIAL", "THE_LOOP_HOLDS_AT_THE_DATA_ALLOWED_STRENGTH"),
     ("BELOW_MATERIAL_NOT_DETECTED", "BELOW_MATERIAL_DETECTED", "THE_LOOP_FALLS_BELOW_MATERIAL_AT_THE_DATA_ALLOWED_STRENGTH"),
     ("BELOW_MATERIAL_NOT_DETECTED", "MATERIAL", "THE_LOOP_HOLDS_ONLY_UNDER_GRANTS"),
     ("MATERIAL", "BELOW_MATERIAL_DETECTED", "THE_LOOP_HOLDS_ONLY_AT_BASELINE"),
     ("UNRESOLVED", "MATERIAL", "UNRESOLVED_AT_THE_DATA_ALLOWED_STRENGTH"),
     (None, None, "NOT_CALIBRATED_THE_SMALLEST_RATE_EXCEEDS_THE_DATA_BOUND")],
)
def test_the_registered_dispositions(baseline: str | None, grants: str | None, status: str) -> None:
    classes = None if baseline is None else {"baseline": baseline, GRANTS: grants}
    assert o39.disposition(classes) == status


def test_decide_reads_the_delay_3_rate_and_records_the_expectations() -> None:
    contract = _narrow(["baseline", GRANTS, "hot_field_hype"], [0, 3], [1.5, 0.75, 0.375, 0.1875])
    gaps = {k: {"baseline": None, GRANTS: g, "hot_field_hype": g} for k, g in enumerate([9.0, 4.0, 1.0, 0.3])}
    report = {**_report(gaps, 3), **_report(gaps, 0)}
    calibration = {"primary": o39.calibrate(contract, report, 0.789381)}
    rows = {}
    for k in range(4):
        for delay in (0, 3):
            for scenario, interval in (("baseline", [-0.01, 0.02]), (GRANTS, [0.06, 0.1])):
                rows[f"imitative/{k}/{scenario}/delay_{delay}/failed_pivot_rate"] = {"mean": sum(interval) / 2, "mean_ci95": interval}
    decision = o39.decide(contract, calibration, rows)
    assert decision["status"] == "O39_THE_LOOP_HOLDS_ONLY_UNDER_GRANTS"
    assert decision["by_delay"]["delay_3"]["rate"] == 0.1875 and decision["by_delay"]["delay_0"]["rate"] == 0.1875
    assert {key: row["held"] for key, row in decision["expectations"].items()} == {"X1": True, "X2": True, "X3": True, "X4": True}
    assert decision["expectations"]["X1"]["spearman"][f"{GRANTS}/delay_3"] == 1.0


def test_run_refuses_without_single_threaded_libraries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setattr(sys, "argv", ["x", "run", "--precommit-sha", "0" * 40, "--output-dir", str(tmp_path / "out")])
    with pytest.raises(o39.StrengthError, match="OMP_NUM_THREADS"):
        o39.main()
