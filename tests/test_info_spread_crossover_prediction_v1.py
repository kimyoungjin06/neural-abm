from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_crossover_prediction_v1 as e3  # noqa: E402
import run_info_spread_policy_ladder_flip_rate_v1 as flip  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as v2  # noqa: E402


CONTRACT = e3.load_contract()
V2 = e3.load_v2_contract()
v2.apply_learner_settings(V2)


def _components(steps: int = 6, **values: float) -> dict[str, list[float]]:
    base = {"A_plus": 0.0, "A_minus": 0.0, "C_plus": 0.0, "C_minus": 0.0, "Dd": 0.0, "V_plus": 0.0, "V_minus": 0.0}
    base.update(values)
    return {key: [value] * steps for key, value in base.items()}


# -- contract ----------------------------------------------------------------------------


def test_the_contract_binds_its_sources_and_keeps_the_worlds_apart() -> None:
    roles = {b["role"] for b in CONTRACT["source_bindings"]}
    assert {"FLIP_RATE_STREAM", "O26_RESULT_THE_RULE_WAS_WRITTEN_WITH_IN_VIEW", "GROUP_RULE"} <= roles
    assert CONTRACT["seeds"]["donor_replicates"][:4] == V2["seeds"]["donor_replicates"]
    used = {6000, 7000, 9000, 11000, 15000}
    assert not {CONTRACT["seeds"]["world_seed_offset"], CONTRACT["measurement"]["world_seed_offset"]} & used
    assert CONTRACT["prediction"]["fitted_parameters"] == "NONE"
    assert CONTRACT["boundary"]["empirical_rows_read"] == 0


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("design", "flip_rates"), [0.0, 0.25], "rise from 0 to 0.5"),
        (("design", "table_sizes"), [1, 4], "table sizes"),
        (("design", "visibilities"), [0.5], "visibilities"),
        (("design", "family"), "mlp", "Regime A"),
        (("measurement", "world_seed_offset"), 18000, "not the comparison worlds"),
        (("scoring", "accuracy_bound"), 0.1, "accuracy bound"),
        (("scoring", "scored_table_sizes"), [4], "scored"),
        (("execution", "thread_variables"), ["OMP_NUM_THREADS"], "thread"),
        (("boundary", "empirical_rows_read"), 1, "no empirical row"),
    ],
)
def test_contract_mutations_are_refused(path: tuple[str, str], value: object, message: str) -> None:
    contract = copy.deepcopy(CONTRACT)
    contract[path[0]][path[1]] = value
    with pytest.raises(e3.CrossoverError, match=message):
        e3.validate_contract(contract)


# -- worlds and streams ------------------------------------------------------------------


def test_the_world_at_visibility_0_5_is_the_flip_rate_studys() -> None:
    flip_contract = flip.load_contract()
    mine = e3.build_world(V2, "smoke", 3, 0.5, flip_contract["seeds"]["world_seed_offset"])
    theirs = flip.build_world(flip_contract, V2, "smoke", "A", 3)
    assert mine.config == theirs.config and mine.intercept == theirs.intercept
    assert np.array_equal(mine.thresholds, theirs.thresholds) and (mine.adjacency != theirs.adjacency).nnz == 0


def test_revealed_items_nest_across_visibilities_and_veracity_is_shared() -> None:
    streams = {v: flip.build_stream(e3.build_world(V2, "smoke", 1, v, 18000), 4, 0.25) for v in (0.2, 0.5, 1.0)}
    assert np.all(streams[0.2].revealed <= streams[0.5].revealed) and np.all(streams[0.5].revealed <= streams[1.0].revealed)
    assert streams[1.0].revealed.all()
    for v in (0.2, 0.5):
        assert np.array_equal(streams[v].veracity, streams[1.0].veracity)
        assert np.array_equal(streams[v].features, streams[1.0].features)
        assert streams[v].credibility_sign == streams[1.0].credibility_sign


def test_the_measurement_signs_are_fixed() -> None:
    world = e3.build_world(V2, "smoke", 0, 0.5, 19000)
    assert {flip.build_stream(world, r, 0.0).credibility_sign for r in range(30)} == {1}
    assert {flip.build_stream(world, r, 1.0).credibility_sign for r in range(30)} == {-1}


# -- the plug-in rule --------------------------------------------------------------------


def test_with_no_heterogeneity_a_table_of_k_donors_cuts_the_variance_k_fold() -> None:
    c = _components(A_plus=0.02, V_plus=0.004, V_minus=0.006, Dd=0.03, A_minus=0.05, C_minus=-0.01)
    for k in (1, 2, 4, 8):
        assert e3.plug_in_gap(c, k, 0.0) == pytest.approx(-0.004 * (1 - 1 / k))


def test_the_symmetric_case_reduces_to_the_simple_threshold() -> None:
    dd, v = 0.04, 0.003
    c = _components(A_minus=dd, C_minus=-dd, Dd=dd, V_plus=v, V_minus=v)
    for k in (1, 2, 4, 8):
        for p in (0.05, 0.1, 0.25, 0.5):
            assert e3.plug_in_gap(c, k, p) == pytest.approx(p * (1 - p) * dd * (1 + 1 / k) - v * (1 - 1 / k))
            positive = e3.simple_index(c, p) > (k - 1) / (k + 1)
            assert (e3.plug_in_gap(c, k, p) > 0) == positive


def test_sustained_crossings() -> None:
    rates = [0.0, 0.1, 0.2, 0.3]
    assert e3._sustained_crossing(rates, [-0.2, -0.1, 0.1, 0.3]) == pytest.approx(0.15)
    assert e3._sustained_crossing(rates, [-0.2, 0.1, -0.1, 0.1]) == pytest.approx(0.25)
    assert e3._sustained_crossing(rates, [-0.2, -0.1, -0.05, 0.0]) is None
    assert e3._sustained_crossing(rates, [0.1, 0.2, 0.3, 0.4]) == 0.0


def test_the_baselines() -> None:
    assert e3.baseline_crossing("constant", 8, CONTRACT) == 0.175
    assert e3.baseline_crossing("table_size_only", 4, CONTRACT) == pytest.approx(0.175)
    assert e3.baseline_crossing("table_size_only", 2, CONTRACT) < 0.175 < e3.baseline_crossing("table_size_only", 8, CONTRACT)


def _table(crossings: dict[tuple[float, int], float | None]) -> dict:
    table = {}
    for v in CONTRACT["design"]["visibilities"]:
        for k in CONTRACT["design"]["table_sizes"]:
            crossing = crossings.get((v, k))
            for p in CONTRACT["design"]["flip_rates"]:
                delta = -0.02 if crossing is None else float(p) - crossing
                if k == 1 and float(p) == 0.0:
                    delta = 0.0
                table[e3.group_key(v, k, p)] = {"exact_mean_cell_delta": delta, "mean_cell_delta": round(delta, 4),
                                                "label": "INDISTINGUISHABLE" if abs(delta) < 0.01 else
                                                ("POSITIVE" if delta > 0 else "NEGATIVE")}
    return table


def _predicted(crossings: dict[tuple[float, int], float | None]) -> dict:
    return {f"pi={v:g}/k={k}": {"crossing": crossings.get((v, k))}
            for v in CONTRACT["design"]["visibilities"] for k in CONTRACT["design"]["table_sizes"]}


def test_the_registered_dispositions() -> None:
    truth = {(v, k): base + shift for v, base in ((0.2, 0.2), (0.5, 0.1), (1.0, 0.03)) for k, shift in ((1, 0.0), (2, 0.02),
                                                                                                         (4, 0.06), (8, 0.1))}
    table = _table(truth)
    exact = e3.score(CONTRACT, _predicted(truth), table)
    assert exact["status"] == "E3_THE_RULE_PREDICTS_THE_CROSSOVER" and exact["within_bound"]["predicted"] == 9
    assert exact["scored_series"] == sorted(f"pi={v:g}/k={k}" for v in (0.2, 0.5, 1.0) for k in (2, 4, 8))
    far = e3.score(CONTRACT, _predicted({key: value + 0.06 for key, value in truth.items()}), table)
    assert far["disposition"] == "THE_RULE_BEATS_THE_BASELINES_BUT_MISSES_BY_MORE_THAN_THE_BOUND"
    flat = e3.score(CONTRACT, _predicted({key: 0.4 for key in truth}), table)
    assert flat["disposition"] == "THE_RULE_DOES_NOT_BEAT_THE_BASELINES"
    expected = e3.expectations(CONTRACT, table, exact)
    assert {key: row["held"] for key, row in expected.items()} == {"X1": True, "X2": True, "X3": True, "X4": True, "X5": True}
    missing = e3.score(CONTRACT, _predicted({**truth, (0.2, 8): None}), _table({**truth, (0.2, 8): None}))
    assert missing["series"]["pi=0.2/k=8"]["observed"] is None and missing["within_bound"]["predicted"] == 9


# -- runs --------------------------------------------------------------------------------


def test_a_learners_trajectory_starts_at_the_prior() -> None:
    world = e3.build_world(V2, "smoke", 0, 0.5, 19000)
    rows, f_plus, f_minus = e3.probe(world, 50)
    assert rows.shape == (50, 6) and np.all((f_plus > 0) & (f_plus < 1))
    assert np.corrcoef(rows[:, 0], f_plus)[0, 1] > 0.5 and np.corrcoef(rows[:, 0], f_minus)[0, 1] < -0.5
    result = spread.simulate(world, flip.build_stream(world, 0, 0.0), e3.learner(V2, world, 0, 0), 0)
    path = e3.trajectory(result.snapshots, rows, world.config.steps)
    prior = spread.hand_prior(world.config).probability(rows)
    first_refit = result.snapshots.steps[1]
    assert np.array_equal(path[0], prior) and np.array_equal(path[first_refit - 1], prior)
    assert not np.array_equal(path[first_refit], prior)


def test_at_visibility_0_5_the_first_four_donors_reproduce_the_flip_rate_arms() -> None:
    flip_contract = flip.load_contract()
    contract = copy.deepcopy(CONTRACT)
    contract["seeds"]["world_seed_offset"] = flip_contract["seeds"]["world_seed_offset"]
    mine = e3.run_cell(contract, V2, "smoke", 0.5, 0.25, 0)
    theirs = flip.run_cell(flip_contract, V2, "smoke", "A", "linear_exact", 0.25, 0, "cpu")
    assert mine["donor_signs"][:4] == theirs["donor_signs"]
    for row, reference in zip(mine["rows"], theirs["rows"], strict=True):
        for arm in ("L0", "T0_1", "T0_4"):
            assert row["arms"][arm] == reference["arms"][arm], arm
        assert set(row["arms"]) == {"L0", "T0_1", "T0_2", "T0_4", "T0_8"}


def test_a_smoke_measurement_is_deterministic() -> None:
    first = e3.measure_cell(CONTRACT, V2, "smoke", 0.5, 0)
    second = e3.measure_cell(CONTRACT, V2, "smoke", 0.5, 0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    components = first["components"]
    assert set(components) == {"A_plus", "A_minus", "C_plus", "C_minus", "Dd", "V_plus", "V_minus"}
    assert all(len(values) == V2["profiles"]["smoke"]["steps"] for values in components.values())
    assert min(components["V_plus"]) >= 0 and components["Dd"][0] == 0.0
