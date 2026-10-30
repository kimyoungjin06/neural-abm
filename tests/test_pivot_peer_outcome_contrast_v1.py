from __future__ import annotations

import copy
import datetime as dt
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prototype_pivot_peer_outcome_contrast_v1.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
e4 = importlib.import_module(SCRIPT.stem)
CONTRACT = e4.load_contract()
N11B = e4.load_n11b(CONTRACT)
N11B_CONTRACT = N11B.load_contract(e4._binding(CONTRACT, e4.N11B_CONTRACT_ROLE))
S1 = N11B.load_s1(N11B_CONTRACT)
N10 = S1.load_n10(N11B.s1_contract(N11B_CONTRACT, S1))
GUARD_COLUMN = N11B.s1_contract(N11B_CONTRACT, S1)["build"]["g4_guard"]["column"]
ESTIMATOR = CONTRACT["estimator"]
SETTINGS = e4._settings(ESTIMATOR)

D = dt.date
J1, J2, I1, I2, I3, I4, G, K, A10 = 11, 12, 21, 22, 23, 24, 16, 17, 30
# work_id: (date, topic, subfield, field, authors). Topic 100 is discovery episode 0's topic (t = 2005).
WORKS = {
    1: (D(2004, 3, 1), 300, 1605, 16, [J1, I1]),
    2: (D(2004, 4, 1), 300, 1605, 16, [J2, I1]),
    3: (D(2004, 5, 1), 300, 1605, 16, [J1, I2]),
    4: (D(2004, 6, 1), 300, 1605, 16, [G, I3]),       # G is guarded
    5: (D(2004, 7, 1), 500, 3201, 32, [K, I3]),       # K is never field-active in 16 before entering
    6: (D(2003, 2, 2), 300, 1605, 16, [I4]),
    7: (D(2005, 2, 15), 100, 1605, 16, [J1]),         # J1 enters, offset 0
    8: (D(2005, 8, 1), 100, 1605, 16, [J1]),          # J1's follow-up, offset 2: a success
    9: (D(2005, 5, 10), 100, 1605, 16, [J2]),         # J2 enters, offset 1, no follow-up: a failure
    10: (D(2006, 8, 1), 100, 1605, 16, [I1]),         # I1 enters alone, offset 6
    11: (D(2005, 11, 11), 100, 1605, 16, [I2, J1]),   # I2 enters with J1, offset 3: recruited
    12: (D(2005, 10, 10), 100, 1605, 16, [G]),
    13: (D(2006, 3, 3), 100, 1605, 16, [K]),
    14: (D(2006, 6, 6), 100, 1605, 16, [K]),          # K's follow-up: never read
    15: (D(2005, 5, 5), 400, 1606, 16, [A10]),
    16: (D(2007, 8, 1), 200, 1606, 16, [A10]),        # A10 enters confirmatory episode 1
}
CONFIRMATORY_FOLLOW_UP = {17: (D(2008, 1, 15), 200, 1606, 16, [A10])}
EPISODES = [
    {"episode_id": 0, "field": 16, "topic": 100, "subfield": 1605, "t": 2005, "partition": "discovery", "screen": "RETAINED", "meets_floor": True},
    {"episode_id": 1, "field": 16, "topic": 200, "subfield": 1606, "t": 2006, "partition": "confirmatory", "screen": "RETAINED", "meets_floor": True},
]
ENTRANTS = [
    (0, J1, D(2005, 2, 15), 0, True),
    (0, J2, D(2005, 5, 10), 1, True),
    (0, I2, D(2005, 11, 11), 3, True),
    (0, I1, D(2006, 8, 1), 6, True),
    (1, A10, D(2007, 8, 1), 6, True),
]


def _write_inputs(tmp_path: Path, works: dict) -> tuple[dict, Path]:
    table = pa.table({
        "work_id": pa.array(list(works), pa.uint64()),
        "year": pa.array([v[0].year for v in works.values()], pa.int16()),
        "date": pa.array([v[0] for v in works.values()], pa.date32()),
        "topic": pa.array([v[1] for v in works.values()], pa.uint32()),
        "subfield": pa.array([v[2] for v in works.values()], pa.uint32()),
        "field": pa.array([v[3] for v in works.values()], pa.uint8()),
    })
    pairs = sorted((w, a) for w, v in works.items() for a in v[4])
    authorships = pa.table({"work_id": pa.array([p[0] for p in pairs], pa.uint64()), "author_id": pa.array([p[1] for p in pairs], pa.uint64())})
    entrants = pa.table({
        "episode_id": pa.array([e[0] for e in ENTRANTS], pa.int64()), "author_id": pa.array([e[1] for e in ENTRANTS], pa.uint64()),
        "entry_date": pa.array([e[2] for e in ENTRANTS], pa.date32()), "offset": pa.array([e[3] for e in ENTRANTS], pa.int64()),
        "precise": pa.array([e[4] for e in ENTRANTS], pa.bool_()),
    })
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, value in (("works", table), ("authorships", authorships), ("episodes", pa.Table.from_pylist(EPISODES)), ("entrants", entrants)):
        paths[name] = tmp_path / f"{name}.parquet"
        pq.write_table(value, paths[name])
    guard = tmp_path / "guard.parquet"
    pq.write_table(pa.table({GUARD_COLUMN: pa.array([G], pa.uint64())}), guard)
    return paths, guard


def _build(tmp_path: Path, works: dict = WORKS, n11b_works: dict | None = None) -> dict:
    """N11b on the fixture with everyone sampled, then this study's build on the same extract."""

    contract = copy.deepcopy(N11B_CONTRACT)
    contract["sampling"]["floor_authors"] = 1000
    n11b_paths, n11b_guard = _write_inputs(tmp_path / "n11b", n11b_works or works)
    built = N11B.build(n11b_paths, n11b_guard, GUARD_COLUMN, contract, N10, threads=2)
    assert built["status"] == "N11B_PASS", built["invariants"]
    private = {"panel": tmp_path / "panel.parquet", "episodes_sampling": tmp_path / "episodes_sampling.parquet"}
    pq.write_table(built["panel"], private["panel"])
    pq.write_table(built["episodes_sampling"], private["episodes_sampling"])
    paths, guard = _write_inputs(tmp_path / "e4", works)
    return e4.build_empirical(paths, guard, GUARD_COLUMN, private, N11B, ESTIMATOR, threads=2)


def _cells(built: dict) -> list[dict]:
    return built["cells"].to_pylist()


# -- contract and authorization ----------------------------------------------------------


def test_the_contract_binds_study2_n11b_and_the_ledger() -> None:
    roles = {b["role"] for b in CONTRACT["source_bindings"]}
    assert {e4.STUDY2_ROLE, e4.STUDY1_ROLE, e4.TRACKED_ROLE, e4.N11B_BUILDER_ROLE, e4.N11B_CONTRACT_ROLE, e4.N11B_RESULT_ROLE} <= roles
    assert CONTRACT["data_access"]["partition"] == "DISCOVERY_ONLY"
    assert CONTRACT["authorization"]["confirmatory_outcomes"] == "NOT_AUTHORIZED"
    assert CONTRACT["boundary"]["identifies_learning"] is False
    assert sorted(CONTRACT["estimator"]["contrasts"]) == sorted(e4.CONTRASTS)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("data_access", "partition"), "ALL", "discovery partition only"),
        (("authorization", "confirmatory_outcomes"), "AUTHORIZED", "confirmatory"),
        (("simulation", "outcome_delays"), [0, 2], "delays"),
        (("boundary", "identifies_learning"), True, "does not identify"),
        (("inputs", "n11b_receipt_sha256"), "0" * 64, "N11b receipt"),
        (("precondition", "n11a_v3_status"), "RECOVERY_PASSES", "N11a v3"),
    ],
)
def test_contract_mutations_are_refused(path: tuple[str, str], value: object, message: str) -> None:
    contract = copy.deepcopy(CONTRACT)
    contract[path[0]][path[1]] = value
    with pytest.raises(e4.ContrastError, match=message):
        e4.validate_contract(contract)


def test_the_authorization_keeps_confirmatory_and_g4_outcomes_sealed(tmp_path: Path) -> None:
    good = {
        "schema_version": e4.AUTHORIZATION_SCHEMA, "status": "AUTHORIZED_DISCOVERY_PEER_OUTCOME_CONTRAST",
        "contract_sha256": e4._sha256(e4.DEFAULT_CONFIG), "script_sha256": e4._sha256(SCRIPT),
        "human_authorization": {"user_instruction": "x"}, "outcomes": "DISCOVERY_PARTITION_ONLY",
        "confirmatory_outcomes": "NOT_AUTHORIZED", "g4_outcomes": "NOT_AUTHORIZED",
    }
    path = tmp_path / "good.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert e4.load_authorization(path, e4.DEFAULT_CONFIG, SCRIPT)["status"] == good["status"]
    for change, message in ((("confirmatory_outcomes", "AUTHORIZED"), "confirmatory"), (("outcomes", "ALL"), "discovery only"),
                            (("script_sha256", "0" * 64), "another script"), (("human_authorization", {}), "quote the user")):
        path = tmp_path / f"{change[0]}.json"
        path.write_text(json.dumps({**good, change[0]: change[1]}), encoding="utf-8")
        with pytest.raises(e4.ContrastError, match=message):
            e4.load_authorization(path, e4.DEFAULT_CONFIG, SCRIPT)


def test_the_committed_authorization_names_this_contract_and_script() -> None:
    authorization = e4.load_authorization(e4.DEFAULT_AUTHORIZATION, e4.DEFAULT_CONFIG, SCRIPT)
    assert authorization["outcomes"] == "DISCOVERY_PARTITION_ONLY"


# -- Study 2 through the slots -----------------------------------------------------------


def test_delay_0_reproduces_the_tracked_runs_and_the_frozen_arm_ignores_the_delay() -> None:
    study2 = e4.load_study2(CONTRACT)
    tracked = json.loads(e4._binding(CONTRACT, e4.TRACKED_ROLE).read_text(encoding="utf-8"))
    config = study2.LearningStudyConfig()
    scenarios = {s.name: s for s in study2.study1.study_scenarios()}
    for name in CONTRACT["simulation"]["scenarios"]:
        for mode in CONTRACT["simulation"]["arms"]:
            run, log = e4.run_logged_arm(study2, scenario=scenarios[name], config=config, replicate=1, mode=mode, delay=0)
            recorded = tracked["scenarios"][name][mode]["replicate_runs"][1]
            assert all(run[key] == recorded[key] for key in e4.PARITY_KEYS), (name, mode)
            assert sum(step >= 0 for step in log["pivot_step"]) == round(run["pivot_rate"] * config.agent_count)
        _run, delayed = e4.run_logged_arm(study2, scenario=scenarios[name], config=config, replicate=1, mode="frozen", delay=3)
        assert delayed == e4.run_logged_arm(study2, scenario=scenarios[name], config=config, replicate=1, mode="frozen", delay=0)[1]
    assert study2.PivotEnvironment.__name__ == "PivotEnvironment" and study2.PivotPolicyAgent.__name__ == "PivotPolicyAgent"


def test_the_delayed_environment_hands_agents_the_events_of_step_s_minus_1_minus_delay() -> None:
    study2 = e4.load_study2(CONTRACT)
    environment = e4.delayed_environment_class(study2)(neighbors=[[1], [0]], config=study2.LearningStudyConfig(), replicate=0, delay=2)
    seen = []
    for step in range(1, 7):
        environment.advance()
        seen.append(list(environment.last_events))
        environment.new_events.append(step)
    assert seen == [[], [], [], [1], [2], [3]]


def test_simulation_rows_follow_visibility_and_the_delay() -> None:
    log = {"stage": [0, 1, 2], "pivot_step": [3, 4, -1], "productive": [True, False, False], "neighbors": [[1], [0, 2], [0, 1]]}
    rows = e4.simulation_rows([log], delay=1, burn_in=2, steps=6)
    agent2 = [(t, e, n, lag, s, f, ps, pf) for t, st, e, n, lag, s, f, ps, pf in zip(
        rows["time"], rows["stage"], rows["event"], rows["n_visible"], rows["lag"], rows["S"], rows["F"], rows["P_S"], rows["P_F"],
        strict=True) if st == 2]
    # Agent 0 pivots productively at step 3 (visible from 4, outcome from 5); agent 1 fails at step 4 (visible 5, outcome 6).
    assert agent2 == [(0, 0, 0, -1, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0, 1, 0), (2, 0, 2, 1, 1, 0, 0, 1), (3, 0, 2, 2, 1, 1, 0, 0)]
    assert list(rows["entry"]["time"]) == [0, 1] and list(rows["entry"]["label"]) == [1, 0]
    assert [e for e, st in zip(rows["event"], rows["stage"], strict=True) if st == 1] == [0, 1]


# -- the estimator -----------------------------------------------------------------------


def _dense_fit(cells: e4.Cells, ridge: float) -> np.ndarray:
    events, trials = np.bincount(cells.group, weights=cells.events), np.bincount(cells.group, weights=cells.trials)
    keep = (events[cells.group] > 0) & (events[cells.group] < trials[cells.group])
    groups, g = np.unique(cells.group[keep], return_inverse=True)
    x = np.column_stack([np.eye(groups.size)[g], cells.design[keep]])
    n, y = cells.trials[keep], cells.events[keep]
    theta = np.zeros(x.shape[1])
    penalty = np.r_[np.zeros(groups.size), np.full(cells.design.shape[1], ridge)]
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(x @ theta)))
        grad = x.T @ (y - n * mu) - penalty * theta
        hess = (x * (n * mu * (1 - mu))[:, None]).T @ x + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        theta += step
        if np.abs(step).max() < 1e-12:
            break
    return theta[groups.size:]


def test_the_block_newton_fit_equals_a_dense_fit_and_recovers_the_truth() -> None:
    rng = np.random.default_rng(7)
    groups, per = 40, 400
    group = np.repeat(np.arange(groups), per)
    design = rng.integers(0, 2, size=(group.size, 3)).astype(float)
    truth = np.array([0.8, -0.5, 0.0])
    eta = rng.normal(-3, 0.5, groups)[group] + design @ truth
    events = (rng.random(group.size) < 1 / (1 + np.exp(-eta))).astype(float)
    group[: per] = groups  # one more group with no events at all
    events[: per] = 0
    group[per: 2 * per] = groups + 1  # and one with only events
    events[per: 2 * per] = 1
    cells = e4.collapse(group, group % 5, design, np.ones(group.size), events, ["a", "b", "c"])
    fit = e4.fit_logit(cells, **SETTINGS)
    assert fit["converged"] and fit["groups"] == groups - 2
    np.testing.assert_allclose(fit["beta"], _dense_fit(cells, SETTINGS["ridge"]), atol=1e-8)
    assert np.all(np.abs(fit["beta"] - truth) < 0.2)
    weights = np.where(cells.cluster == 0, 2.0, 1.0)
    doubled = e4.Cells(cells.group, cells.cluster, cells.trials * weights, cells.events * weights, cells.design, cells.names)
    np.testing.assert_allclose(e4.fit_logit(cells, weights=weights, **SETTINGS)["beta"], e4.fit_logit(doubled, **SETTINGS)["beta"], atol=1e-9)


def test_contrasts_drop_the_reference_and_need_every_state() -> None:
    beta = np.array([0.5, 0.2, 0.3, 9.0])
    full = e4.contrast_values(beta, ["S", "F", "P_S", "n1"], "P_F")
    assert full["visibility_did"] == pytest.approx(0.5 - 0.2 - 0.3) and full["failure_avoidance"] == pytest.approx(0.2)
    assert full["success_following"] == pytest.approx(0.2) and full["outcome_gap_before_visible"] == pytest.approx(0.3)
    immediate = e4.contrast_values(np.array([0.5]), ["S"], "F")
    assert immediate["outcome_gap_visible"] == pytest.approx(0.5) and immediate["visibility_did"] is None
    model = {"cells": e4.Cells(np.zeros(1, int), np.zeros(1, int), np.ones(1), np.ones(1), np.zeros((1, 1)), ["S"],
                               {"S": 12.0, "F": 9.0, "P_S": 0.0, "P_F": 0.0}), "reference": "F"}
    assert not e4._quantity_defined(model, "outcome_gap_visible")
    model["cells"].state_events["F"] = 10.0
    assert e4._quantity_defined(model, "outcome_gap_visible") and not e4._quantity_defined(model, "visibility_did")


def test_bootstrap_draws_are_deterministic_and_resample_clusters() -> None:
    m = e4.multiplicities(24, 100, 20261030, 2)
    assert m.shape == (100, 24) and np.all(m.sum(axis=1) == 24)
    assert np.array_equal(m, e4.multiplicities(24, 100, 20261030, 2)) and not np.array_equal(m, e4.multiplicities(24, 100, 20261030, 3))
    rng = np.random.default_rng(3)
    group = np.repeat(np.arange(24 * 4), 200)
    design = rng.integers(0, 2, size=(group.size, 1)).astype(float)
    events = (rng.random(group.size) < 0.05 + 0.05 * design[:, 0]).astype(float)
    cells = e4.collapse(group, group // 4, design, np.ones(group.size), events, ["exposed"])
    fit = e4.fit_logit(cells, **SETTINGS)
    problem = {"models": [{"arm": "data", "model": "exposure", "cells": cells, "beta": fit["beta"], "reference": "P_F",
                           "quantities": ["beta:exposed"]}], "multiplicities": m[:20]}
    first = e4.bootstrap({"p": problem}, SETTINGS, workers=1, chunk=7)
    second = e4.bootstrap({"p": problem}, SETTINGS, workers=1, chunk=20)
    assert np.array_equal(first["p"]["draws"], second["p"]["draws"]) and first["p"]["unconverged_fits"] == 0
    assert first["p"]["draws"].shape == (20, 1) and np.std(first["p"]["draws"]) > 0


# -- decisions ---------------------------------------------------------------------------


def _row(mean: float | None, low: float | None, high: float | None) -> dict:
    return {"mean": mean, "low": low, "high": high, "finite_draws": 10}


def _reports(frozen_did: tuple, did: tuple, following: tuple, avoidance: tuple) -> tuple[dict, dict]:
    simulation = {}
    for scenario in CONTRACT["simulation"]["scenarios"]:
        simulation[f"simulation/{scenario}/delay_0"] = {"learning_attributable": {"imitative": {"state/outcome_gap_visible": _row(0.5, 0.2, 0.8)}}}
        simulation[f"simulation/{scenario}/delay_3"] = {"by_arm": {"frozen": {"state/visibility_did": _row(*frozen_did)}}}
    data = {"estimates": {"exposure/exposed": _row(1.0, 0.8, 1.2), "state/outcome_gap_before_visible": _row(0.2, 0.1, 0.3),
                          "state/visibility_did": _row(*did), "state/success_following": _row(*following),
                          "state/failure_avoidance": _row(*avoidance)}}
    return simulation, data


@pytest.mark.parametrize(
    ("frozen_did", "did", "following", "avoidance", "status"),
    [
        ((0.0, -0.2, 0.2), (0.4, 0.1, 0.7), (0.3, 0.1, 0.5), (-0.3, -0.5, -0.1), "E4_CONSISTENT_WITH_IMITATIVE"),
        ((0.0, -0.2, 0.2), (0.4, 0.1, 0.7), (0.0, -0.2, 0.2), (-0.3, -0.5, -0.1), "E4_CONSISTENT_WITH_CAUTIONARY"),
        ((0.0, -0.2, 0.2), (0.4, 0.1, 0.7), (0.3, 0.1, 0.5), (0.0, -0.2, 0.2), "E4_SUCCESS_FOLLOWING_ONLY_NO_STUDY2_ARM"),
        ((0.0, -0.2, 0.2), (0.4, 0.1, 0.7), (0.1, -0.1, 0.3), (0.0, -0.2, 0.2), "E4_OUTCOME_DEPENDENCE_EMERGES_WITHOUT_A_DIRECTION"),
        ((0.0, -0.2, 0.2), (-0.4, -0.7, -0.1), (0.0, -0.2, 0.2), (0.0, -0.2, 0.2), "E4_OUTCOME_ASSOCIATION_WEAKENS_ONCE_VISIBLE"),
        ((0.0, -0.2, 0.2), (0.0, -0.3, 0.3), (0.0, -0.2, 0.2), (0.0, -0.2, 0.2), "E4_NO_VISIBILITY_EFFECT_DETECTED"),
        ((0.3, 0.1, 0.5), (0.4, 0.1, 0.7), (0.3, 0.1, 0.5), (-0.3, -0.5, -0.1), "E4_NOT_INTERPRETABLE_THE_ESTIMATOR_FLAGS_STUDY2_FROZEN_ARM"),
        ((0.0, -0.2, 0.2), (None, None, None), (None, None, None), (None, None, None), "E4_UNDEFINED_TOO_FEW_EVENTS"),
    ],
)
def test_the_registered_dispositions(frozen_did: tuple, did: tuple, following: tuple, avoidance: tuple, status: str) -> None:
    simulation, data = _reports(frozen_did, did, following, avoidance)
    decision = e4.decide(CONTRACT, simulation, data)
    assert decision["status"] == status
    assert decision["expectations"]["ES2"]["held"] and decision["expectations"]["EE1"]["held"] and decision["expectations"]["EE2"]["held"]


def test_es3_needs_two_defined_scenarios_and_counts_only_defined_ones() -> None:
    simulation, data = _reports((0.0, -0.2, 0.2), (0.4, 0.1, 0.7), (0.3, 0.1, 0.5), (-0.3, -0.5, -0.1))
    simulation["simulation/baseline/delay_3"]["by_arm"]["frozen"]["state/visibility_did"] = _row(None, None, None)
    assert e4.decide(CONTRACT, simulation, data)["expectations"]["ES3"]["held"]
    simulation["simulation/hot_field_hype/delay_3"]["by_arm"]["frozen"]["state/visibility_did"] = _row(None, None, None)
    assert e4.decide(CONTRACT, simulation, data)["status"] == "E4_NOT_INTERPRETABLE_THE_ESTIMATOR_FLAGS_STUDY2_FROZEN_ARM"


# -- the empirical build -----------------------------------------------------------------


def test_the_build_reproduces_n11b_and_reads_the_registered_states(tmp_path: Path) -> None:
    built = _build(tmp_path)
    assert not any(built["parity"].values()), built["parity"]
    counts = built["counts"]
    assert counts["discovery_entrants"] == 4 and counts["labelled_precise_entrants"] == 1
    assert counts["recruited_precise_entrants"] == 1 and counts["exposed_authors"] == 5
    # Year 0: J1-I1, J1-I2, J2-I1, I2-J1, I1-J1, I1-J2, G-I3, K-I3; year 1: J1-I1, J2-I1, G-I3, K-I3; year 2: G-I3, K-I3.
    assert counts["dyads"] == 14 and counts["dyads_with_an_s1_coauthor"] == 8
    exposed = [c for c in _cells(built) if c["n_bin"] >= 0]
    by_quarter: dict[tuple, tuple] = {}
    for c in exposed:
        key = (c["quarter"], c["s"], c["f"], c["p_s"], c["p_f"], c["u"])
        previous = by_quarter.get(key, (0, 0, 0, 0))
        by_quarter[key] = tuple(a + b for a, b in zip(previous, (c["rows"], c["events"], c["rows_all"], c["events_all"]), strict=True))
    # I1: J1 pending success from quarter 1, J2 pending failure from 2, J1's success visible at 5, J2's failure at 6, entry at 6.
    assert by_quarter[(1, 0, 0, 1, 0, 0)] == (2, 0, 2, 0)            # I1 and the recruited I2
    assert by_quarter[(3, 0, 0, 1, 0, 0)] == (0, 0, 1, 1)            # I2's recruited entry is censored, counted only as recruited
    assert by_quarter[(2, 0, 0, 1, 1, 0)] == (1, 0, 1, 0)
    assert by_quarter[(5, 1, 0, 0, 1, 0)] == (1, 0, 1, 0)
    assert by_quarter[(6, 1, 1, 0, 0, 0)] == (1, 1, 1, 1)
    # I3's coauthor entrants are a guarded author and a never-eligible one: state U only.
    assert {(c["quarter"], c["n_bin"], c["lag_bin"]) for c in exposed if c["u"]} == {(4, 0, 0), *((q, 1, min(q - 4, 6) if q - 4 < 5 else 5 if q - 4 < 8 else 6) for q in range(5, 12))}
    # Unexposed: I4 at risk in years 0 and 1, and A10 (the confirmatory entrant) in years 1 and 2 of episode 0.
    assert sum(c["rows"] for c in _cells(built) if c["weight_class"] == "pi") == 16
    assert built["entry_cells"] == [(0, 0, 1, 1), (0, 1, 1, 0), (0, 3, 1, 0), (0, 6, 1, 0)]


def test_no_confirmatory_or_non_s1_follow_up_reaches_any_output(tmp_path: Path) -> None:
    base = _build(tmp_path / "base")
    without_k = {k: v for k, v in WORKS.items() if k != 14}
    more = {**WORKS, **CONFIRMATORY_FOLLOW_UP, 18: (D(2007, 1, 15), 100, 1605, 16, [J1])}   # J1's work outside its label window
    for name, works in (("without_k", without_k), ("more", more)):
        other = _build(tmp_path / name, works)
        assert other["cells"].equals(base["cells"]) and other["entry_cells"] == base["entry_cells"], name
        assert {k: v for k, v in other["counts"].items() if k != "coauthor_entrants"} == {k: v for k, v in base["counts"].items() if k != "coauthor_entrants"}


def test_a_discovery_follow_up_inside_the_window_changes_the_state(tmp_path: Path) -> None:
    works = {**WORKS, 19: (D(2005, 9, 1), 100, 1605, 16, [J2])}
    built = _build(tmp_path, works)
    assert not any(built["parity"].values())
    exposed = {(c["quarter"], c["s"], c["f"], c["p_s"], c["p_f"]) for c in _cells(built) if c["n_bin"] >= 0 and not c["u"]}
    assert (6, 1, 0, 0, 0) in exposed and (2, 0, 0, 1, 0) in exposed


def test_a_build_that_differs_from_n11b_stops_before_any_label(tmp_path: Path) -> None:
    n11b_works = {**WORKS, 20: (D(2004, 9, 9), 300, 1605, 16, [I1, I4])}
    built = _build(tmp_path, WORKS, n11b_works=n11b_works)
    assert built.get("stopped") and any(built["parity"].values())
    assert "cells" not in built


def test_empirical_cells_weight_the_subcohort_and_keep_states_to_exposed_rows(tmp_path: Path) -> None:
    built = _build(tmp_path)
    built = {**built, "probability": {0: 0.25}}
    cells = e4.empirical_cells(built, ESTIMATOR)
    assert cells["state"].names[:3] == ["S", "F", "P_S"] and cells["state"].state_events["reference"] == "P_F"
    pi_rows = sum(c["rows"] for c in _cells(built) if c["weight_class"] == "pi")
    assert cells["exposure"].trials.sum() == pytest.approx(sum(c["rows"] for c in _cells(built)) + 3 * pi_rows)
    assert cells["exposure_recruited_as_events"].events.sum() == cells["exposure"].events.sum() + 1
    assert cells["state"].trials.sum() == sum(c["rows"] for c in _cells(built) if c["n_bin"] >= 0)


def test_a_nearly_separated_group_still_converges_to_the_dense_fit() -> None:
    # The cells of a group that once sent the fit to a fixed effect of 1e291: a 2-of-2 exposed cell beside unexposed failures.
    rng = np.random.default_rng(11)
    group = np.repeat(np.arange(30), 50)
    design = rng.integers(0, 2, size=(group.size, 2)).astype(float)
    events = (rng.random(group.size) < 0.1 + 0.3 * design[:, 0]).astype(float)
    extra_group = np.array([30, 30, 30, 30])
    extra_design = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
    cells = e4.collapse(np.r_[group, extra_group], np.r_[group, extra_group] % 6, np.vstack([design, extra_design]),
                        np.r_[np.ones(group.size), [2.0, 48.0, 2.0, 26.0]], np.r_[events, [0.0, 10.0, 2.0, 8.0]], ["exposed", "stage"])
    fit = e4.fit_logit(cells, **SETTINGS)
    assert fit["converged"]
    np.testing.assert_allclose(fit["beta"], _dense_fit(cells, SETTINGS["ridge"]), atol=1e-8)


def test_structure_only_stops_after_the_parity_and_reads_no_label(tmp_path: Path) -> None:
    contract = copy.deepcopy(N11B_CONTRACT)
    contract["sampling"]["floor_authors"] = 1000
    n11b_paths, n11b_guard = _write_inputs(tmp_path / "n11b", WORKS)
    built = N11B.build(n11b_paths, n11b_guard, GUARD_COLUMN, contract, N10, threads=2)
    private = {"panel": tmp_path / "panel.parquet", "episodes_sampling": tmp_path / "episodes_sampling.parquet"}
    pq.write_table(built["panel"], private["panel"])
    pq.write_table(built["episodes_sampling"], private["episodes_sampling"])
    paths, guard = _write_inputs(tmp_path / "e4", WORKS)
    out = e4.build_empirical(paths, guard, GUARD_COLUMN, private, N11B, ESTIMATOR, threads=2, structure_only=True)
    assert out["stopped"] is False and not any(out["parity"].values())
    # Rows: J1 1, J2 2, I1 7, I2 4, I3 12. Visible: I1 quarters 1-6, I2 1-3 (before recruitment censoring), I3 4-11.
    assert out["structure"] == {"dyads": 14, "exposed_authors": 5, "exposed_rows": 26, "exposed_rows_with_a_visible_entry": 17}
    assert "cells" not in out and "discovery_entrants_without_a_label" not in out["parity"]
