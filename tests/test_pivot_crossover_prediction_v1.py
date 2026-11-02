from __future__ import annotations

import copy
import datetime as dt
import importlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
crossover = importlib.import_module("run_pivot_crossover_prediction_v1")
e4 = importlib.import_module("prototype_pivot_peer_outcome_contrast_v1")
CONTRACT_PATH = crossover.DEFAULT_CONTRACT
CONTRACT = crossover.load_contract()
N11B = crossover.load_n11b(CONTRACT)
N11B_CONTRACT = N11B.load_contract(crossover._binding(CONTRACT, crossover.N11B_CONTRACT_ROLE))
S1 = N11B.load_s1(N11B_CONTRACT)
N10 = S1.load_n10(N11B.s1_contract(N11B_CONTRACT, S1))
GUARD_COLUMN = N11B.s1_contract(N11B_CONTRACT, S1)["build"]["g4_guard"]["column"]
E4_CONTRACT = e4.load_contract()
MINIMUM = crossover.minimum_effect({"labelled_precise_entrants": 920, "discovery_precise_entrants": 6291})


def _mutated(path: tuple[str, ...], value: object) -> dict:
    contract = copy.deepcopy(CONTRACT)
    node = contract
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return contract


# -- contract and authorization ----------------------------------------------------------


def test_the_contract_binds_e4_stage_1_and_the_ledger() -> None:
    roles = {b["role"] for b in CONTRACT["source_bindings"]}
    assert {crossover.N11B_BUILDER_ROLE, crossover.E4_RECORD_ROLE, crossover.STAGE_1_RECORD_ROLE, "CURRENT_LEDGER_CARRYING_O42"} <= roles
    assert CONTRACT["precondition"]["e4_parity_counts"] == {
        "discovery_episodes": 24, "discovery_precise_entrants": 6291, "labelled_precise_entrants": 920}
    assert CONTRACT["groups"]["minimum_effect"]["value"] == pytest.approx(MINIMUM, abs=5e-8)
    assert CONTRACT["data_access"]["partition"] == "DISCOVERY_ONLY"
    assert "THE_N11B_PANEL" in CONTRACT["data_access"]["not_read"]
    assert set(CONTRACT["precommit"]["precommit_paths"]) >= {
        "experiments/configs/pivot_crossover_prediction_authorization_v1.json", "tests/duckdb_overlay_tests.txt",
        "scripts/inspect_release_artifacts.py", "tests/test_release_artifact_inspection.py"}


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("policies", "table_sizes"), [1, 2, 4, 8], "table sizes"),
        (("groups", "evidence_bins", "edges"), [1, 10, 50, 150], "evidence bins"),
        (("groups", "minimum_effect", "value"), 0.001, "minimum effect"),
        (("groups", "bootstrap", "draws"), 1000, "2,000 draws"),
        (("scoring", "agreement_floor"), 0.7, "scoring floors"),
        (("generator_check", "taus"), [0.0, 0.5, 1.0], "generator check design"),
        (("generator_check", "passes_at_each_tau", "reversal_share_of_determined_at_most"), 0.1, "reversal ceiling"),
        (("generator_check", "failure"), "READ_THE_LABELS_ANYWAY", "stop before the labels"),
        (("authorization", "confirmatory_outcomes"), "AUTHORIZED", "must stay unauthorized"),
        (("data_access", "partition"), "ALL", "discovery partition only"),
        (("precondition", "e4_parity_counts", "labelled_precise_entrants"), 921, "E4 record differs"),
        (("execution", "identity_artifacts"), ["summary.json"], "identity artifacts"),
    ],
)
def test_contract_mutations_are_refused(path: tuple[str, ...], value: object, message: str) -> None:
    with pytest.raises(crossover.CrossoverError, match=message):
        crossover.validate_contract(_mutated(path, value))


def test_the_authorization_keeps_confirmatory_and_g4_outcomes_sealed(tmp_path: Path) -> None:
    script = Path(crossover.__file__).resolve()
    good = {
        "schema_version": crossover.AUTHORIZATION_SCHEMA, "status": "AUTHORIZED_DISCOVERY_CROSSOVER_PREDICTION",
        "contract_sha256": crossover._sha256(CONTRACT_PATH), "script_sha256": crossover._sha256(script),
        "human_authorization": {"user_instruction": "x"}, "outcomes": "DISCOVERY_PARTITION_ONLY",
        "confirmatory_outcomes": "NOT_AUTHORIZED", "g4_outcomes": "NOT_AUTHORIZED",
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    crossover.load_authorization(path, CONTRACT_PATH, script)
    for key, value in (("confirmatory_outcomes", "AUTHORIZED"), ("g4_outcomes", "AUTHORIZED"), ("outcomes", "ALL"),
                       ("contract_sha256", "0" * 64), ("human_authorization", {})):
        path.write_text(json.dumps({**good, key: value}), encoding="utf-8")
        with pytest.raises(crossover.CrossoverError):
            crossover.load_authorization(path, CONTRACT_PATH, script)


def test_the_committed_authorization_names_this_contract_and_script() -> None:
    authorization = crossover.load_authorization(crossover.DEFAULT_AUTHORIZATION, CONTRACT_PATH, Path(crossover.__file__).resolve())
    assert authorization["human_authorization"]["user_instruction"] == "PR 만들어서 병합하고 2단계 진행하자"
    assert authorization["scope"]["runs"] == ["development", "reproduction"]


# -- the policies and the rule -----------------------------------------------------------


def test_the_minimum_effect_is_one_percent_of_the_base_rate_brier() -> None:
    p0 = 920 / 6291
    assert MINIMUM == pytest.approx(0.01 * p0 * (1 - p0))


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_the_table_score_is_the_exact_expectation_over_k_subsets(k: int) -> None:
    shares = np.array([0.05, 0.1, 0.12, 0.2, 0.31, 0.4])
    for y in (0.0, 1.0):
        brute = np.mean([(np.mean(shares[list(c)]) - y) ** 2 for c in itertools.combinations(range(shares.size), k)])
        mean, spread, m = crossover._donor_moments(shares, np.arange(shares.size))
        assert (mean - y) ** 2 + crossover._table_variance(spread, m, k) == pytest.approx(brute)


def test_the_learner_sees_only_labels_five_quarters_old() -> None:
    entrants = np.arange(1, 13)
    follow = np.zeros(12, dtype=np.int64)
    assert crossover.learner_forecast(entrants, follow, 4) == (0, None)
    n, share = crossover.learner_forecast(entrants, follow, 7)
    assert n == 1 + 2 + 3
    for later in range(3, 12):
        changed = follow.copy()
        changed[later] = entrants[later]
        assert crossover.learner_forecast(entrants, changed, 7) == (n, share)
    changed = follow.copy()
    changed[2] = 3
    assert crossover.learner_forecast(entrants, changed, 7) == (6, 0.5)


def _shape(seed: int = 0, episodes: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 60, size=episodes)
    return rng.poisson(sizes[:, None] / 3.0, size=(episodes, 12)) + 1


def test_the_rule_reads_no_label_of_its_target() -> None:
    entrants = _shape()
    follow = crossover.synthetic_follow_ups(entrants, 0.15, 0.5, (1, 2, 3))
    base = crossover.comparison(entrants, follow)
    changed = follow.copy()
    changed[3] = entrants[3] - changed[3]
    other = crossover.comparison(entrants, changed)
    assert np.array_equal(base["predicted"][3], other["predicted"][3])
    assert not np.array_equal(base["predicted"][4], other["predicted"][4])
    assert not np.array_equal(base["observed"][3], other["observed"][3])


def test_the_rule_predicts_the_observed_gap_without_bias_for_exchangeable_entrants() -> None:
    entrants = _shape(7)
    diffs = []
    for r in range(300):
        follow = crossover.synthetic_follow_ups(entrants, 0.15, 0.5, (9, r))
        sums = crossover.comparison(entrants, follow)
        diffs.append((sums["observed"].sum(axis=(0, 2)) - sums["predicted"].sum(axis=(0, 2))) / sums["forecasts"].sum())
    diffs = np.array(diffs)
    se = diffs.std(axis=0) / np.sqrt(len(diffs))
    assert np.all(np.abs(diffs.mean(axis=0)) < 4 * se + 1e-5)


def test_the_simple_reading_crossing_moves_later_with_the_table_size() -> None:
    entrants = _shape(3)
    follow = crossover.synthetic_follow_ups(entrants, 0.15, 0.75, (4, 4))
    totals = entrants.sum(axis=1)
    rule = crossover.rule_components(totals, follow.sum(axis=1) / totals, None)
    assert np.all(np.diff(rule["H"]) < 0)
    assert rule["V"] > 0


def test_a_world_needs_donors_for_the_largest_table() -> None:
    entrants = _shape(episodes=17)
    with pytest.raises(crossover.CrossoverError, match="largest table"):
        crossover.comparison(entrants, np.zeros_like(entrants))


# -- groups, scoring, decision -----------------------------------------------------------


def test_groups_are_deterministic_and_label_by_the_interval_and_the_minimum() -> None:
    entrants = _shape(5, episodes=24)
    follow = crossover.synthetic_follow_ups(entrants, 0.15, 0.75, (5, 5))
    first = crossover.analyse(entrants, follow, MINIMUM, (1, 1), 400)
    again = crossover.analyse(entrants, follow, MINIMUM, (1, 1), 400)
    assert first["groups"] == again["groups"]
    assert len(first["groups"]) == 20
    for group in first["groups"].values():
        if group["forecasts"] == 0:
            continue
        if group["label"] == "POSITIVE":
            assert group["low"] > 0 and group["mean"] >= MINIMUM
        elif group["label"] == "NEGATIVE":
            assert group["high"] < 0 and group["mean"] <= -MINIMUM
        expected = "POSITIVE" if group["predicted"] >= MINIMUM else "NEGATIVE" if group["predicted"] <= -MINIMUM else "NEUTRAL"
        assert group["predicted_label"] == expected


def _group(label: str, predicted: float) -> dict:
    return {"label": label, "predicted": predicted, "predicted_label": crossover._predicted_label(predicted, MINIMUM), "covered": True}


PASSING = {"passes": True, "replicates": 40, "taus": {"tau=0.5": {"replicates_with_a_positive_group": 40}}}


@pytest.mark.parametrize(
    ("groups", "status"),
    [
        ([("NEGATIVE", -0.01), ("POSITIVE", 0.004), ("INDISTINGUISHABLE", 0.0)], "THE_RULE_ORDERS_THE_TABLE_AND_THE_LEARNER"),
        ([("NEGATIVE", -0.01), ("NEGATIVE", -0.0001)], "THE_TABLE_WINS_WHERE_DETERMINED_AND_THE_RULE_AGREES"),
        ([("POSITIVE", 0.01), ("POSITIVE", 0.0001)], "THE_LEARNER_WINS_WHERE_DETERMINED_AND_THE_RULE_AGREES"),
        ([("NEGATIVE", -0.01), ("POSITIVE", -0.004)], "THE_RULE_REVERSES_A_DETERMINED_COMPARISON"),
        ([("NEGATIVE", -0.01), ("POSITIVE", -0.0001), ("POSITIVE", 0.003)], "THE_RULE_MISSES_DETERMINED_COMPARISONS"),
        ([("NEGATIVE", -0.01), ("INDISTINGUISHABLE", 0.02)], "UNDETERMINED_TOO_FEW_DETERMINED_GROUPS"),
    ],
)
def test_the_registered_dispositions(groups: list, status: str) -> None:
    table = {f"g{i}": _group(label, value) for i, (label, value) in enumerate(groups)}
    assert crossover.decide(CONTRACT, PASSING, crossover.score(table)) == "E3S2_" + status


def test_a_failed_generator_check_stops_before_any_label() -> None:
    assert crossover.decide(CONTRACT, {**PASSING, "passes": False}, None) == "E3S2_STOPPED_THE_RULE_FAILS_ON_KNOWN_TRUTH"
    rows = crossover.expectations({**PASSING, "passes": False}, None, None)
    assert rows["X1"]["held"] is False and rows["X3"]["held"] is None


def test_the_generator_check_is_deterministic_on_a_small_shape() -> None:
    contract = _mutated(("generator_check", "replicates"), 2)
    entrants = _shape(11, episodes=24)
    first = crossover.generator_check(entrants, contract, 920 / 6291, MINIMUM)
    assert first == crossover.generator_check(entrants, contract, 920 / 6291, MINIMUM)
    assert set(first["taus"]) == {"tau=0", "tau=0.25", "tau=0.5", "tau=1"}
    assert first["replicates"] == 2


def test_run_reads_no_label_when_the_generator_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    entrants = _shape(2, episodes=24)
    entrants[:, :] = 0
    entrants[:, 0] = 6291 // 24
    entrants[: 6291 % 24, 1] = 1

    def fake_build(inputs, n11b, *, threads=8, structure_only=False):
        calls.append(structure_only)
        return {"episode_ids": list(range(24)), "entrants": entrants, "follow_ups": None}

    monkeypatch.setattr(crossover, "n11b_inputs", lambda contract, n11b: {"preflight": {"status": "PASS"}, "inputs": {}, "passes": True})
    monkeypatch.setattr(crossover, "build", fake_build)
    monkeypatch.setattr(crossover, "generator_check", lambda *args: {**PASSING, "passes": False})
    summary, built, _ = crossover.run(CONTRACT, threads=1)
    assert calls == [True]
    assert summary["status"] == "E3S2_STOPPED_THE_RULE_FAILS_ON_KNOWN_TRUTH"
    assert summary["data_access"]["labels_read"] == 0 and built["follow_ups"] is None


# -- the build on a fixture, against E4's label rule -------------------------------------


D = dt.date
J1, J2, I1, I2, I3, I4, I5, X1, G, K, A10 = 11, 12, 21, 22, 23, 24, 25, 26, 16, 17, 30
# work_id: (date, topic, subfield, field, authors). Topic 100 is discovery episode 0's topic (t = 2005).
WORKS = {
    1: (D(2004, 3, 1), 300, 1605, 16, [J1, I1]),
    2: (D(2004, 4, 1), 300, 1605, 16, [J2, I1]),
    3: (D(2004, 5, 1), 300, 1605, 16, [J1, I2]),
    4: (D(2004, 6, 1), 300, 1605, 16, [G, I3]),
    5: (D(2004, 7, 1), 500, 3201, 32, [K, I3]),
    6: (D(2003, 2, 2), 300, 1605, 16, [I4, I5]),
    18: (D(2004, 9, 9), 300, 1605, 16, [X1]),
    7: (D(2005, 2, 15), 100, 1605, 16, [J1]),         # J1 enters, offset 0
    8: (D(2005, 8, 1), 100, 1605, 16, [J1]),          # J1's follow-up at offset 2: a success
    9: (D(2005, 5, 10), 100, 1605, 16, [J2]),         # J2 enters, offset 1; follow-up at offset 6 is outside its window
    19: (D(2006, 8, 1), 100, 1605, 16, [J2]),
    10: (D(2006, 8, 1), 100, 1605, 16, [I1]),         # I1 enters, offset 6
    11: (D(2005, 11, 11), 100, 1605, 16, [I2, J1]),   # I2 enters with J1, offset 3
    20: (D(2005, 11, 1), 100, 1605, 16, [I3]),        # I3 enters, offset 3
    21: (D(2006, 1, 1), 100, 1605, 16, [I3]),         # a year-only follow-up: 2006 lies wholly in I3's window 4..7
    22: (D(2005, 4, 4), 100, 1605, 16, [I4]),         # I4 enters, offset 1
    23: (D(2006, 5, 1), 100, 1605, 16, [I4]),         # offset 5: inside I4's window 2..5
    24: (D(2005, 4, 4), 100, 1605, 16, [I5]),         # I5 enters, offset 1
    25: (D(2006, 1, 1), 100, 1605, 16, [I5]),         # a year-only follow-up: 2006 spans 4..7, not wholly in 2..5
    26: (D(2005, 1, 1), 100, 1605, 16, [X1]),         # X1 enters on a year-only date: imprecise
    12: (D(2005, 10, 10), 100, 1605, 16, [G]),
    13: (D(2006, 3, 3), 100, 1605, 16, [K]),
    15: (D(2005, 5, 5), 400, 1606, 16, [A10]),
    16: (D(2007, 8, 1), 200, 1606, 16, [A10]),        # A10 enters confirmatory episode 1
}
SEALED_FOLLOW_UPS = {
    17: (D(2008, 1, 15), 200, 1606, 16, [A10]),       # a confirmatory follow-up
    27: (D(2006, 3, 3), 100, 1605, 16, [X1]),         # an imprecise entrant's follow-up
}
EPISODES = [
    {"episode_id": 0, "field": 16, "topic": 100, "subfield": 1605, "t": 2005, "partition": "discovery", "screen": "RETAINED", "meets_floor": True},
    {"episode_id": 1, "field": 16, "topic": 200, "subfield": 1606, "t": 2006, "partition": "confirmatory", "screen": "RETAINED", "meets_floor": True},
]
ENTRANTS = [
    (0, J1, D(2005, 2, 15), 0, True),
    (0, J2, D(2005, 5, 10), 1, True),
    (0, I4, D(2005, 4, 4), 1, True),
    (0, I5, D(2005, 4, 4), 1, True),
    (0, I2, D(2005, 11, 11), 3, True),
    (0, I3, D(2005, 11, 1), 3, True),
    (0, I1, D(2006, 8, 1), 6, True),
    (0, X1, D(2005, 1, 1), 0, False),
    (1, A10, D(2007, 8, 1), 6, True),
]


def _write_inputs(directory: Path, works: dict) -> tuple[dict, Path]:
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
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, value in (("works", table), ("authorships", authorships), ("episodes", pa.Table.from_pylist(EPISODES)), ("entrants", entrants)):
        paths[name] = directory / f"{name}.parquet"
        pq.write_table(value, paths[name])
    guard = directory / "guard.parquet"
    pq.write_table(pa.table({GUARD_COLUMN: pa.array([G], pa.uint64())}), guard)
    return paths, guard


def _e4_entry_cells(tmp_path: Path) -> dict[tuple[int, int], tuple[int, int]]:
    """E4's build on the fixture: precise entrants and follow-ups by episode and entry quarter."""

    contract = copy.deepcopy(N11B_CONTRACT)
    contract["sampling"]["floor_authors"] = 1000
    paths, guard = _write_inputs(tmp_path / "n11b", WORKS)
    built = N11B.build(paths, guard, GUARD_COLUMN, contract, N10, threads=2)
    assert built["status"] == "N11B_PASS", built["invariants"]
    private = {"panel": tmp_path / "panel.parquet", "episodes_sampling": tmp_path / "episodes_sampling.parquet"}
    pq.write_table(built["panel"], private["panel"])
    pq.write_table(built["episodes_sampling"], private["episodes_sampling"])
    paths, guard = _write_inputs(tmp_path / "e4", WORKS)
    result = e4.build_empirical(paths, guard, GUARD_COLUMN, private, N11B, E4_CONTRACT["estimator"], threads=2)
    assert not any(result["parity"].values()), result["parity"]
    return {(int(e), int(q)): (int(n), int(y)) for e, q, n, y in result["entry_cells"]}


def test_the_build_reproduces_e4s_labels_on_a_fixture(tmp_path: Path) -> None:
    paths, _ = _write_inputs(tmp_path / "ours", WORKS)
    built = crossover.build(paths, N11B, threads=2)
    assert built["episode_ids"] == [0]
    cells = _e4_entry_cells(tmp_path)
    ours = {(0, q): (int(built["entrants"][0, q]), int(built["follow_ups"][0, q])) for q in range(12) if built["entrants"][0, q]}
    assert ours == cells
    # J1, I3 (year-only, wholly inside), and I4 follow up; J2 (outside), I2, I5 (year spans), and I1 do not.
    assert built["follow_ups"][0].tolist() == [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert built["entrants"][0].tolist() == [1, 3, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0]


def test_no_confirmatory_or_imprecise_follow_up_reaches_any_output(tmp_path: Path) -> None:
    base_paths, _ = _write_inputs(tmp_path / "base", WORKS)
    sealed_paths, _ = _write_inputs(tmp_path / "sealed", {**WORKS, **SEALED_FOLLOW_UPS})
    base = crossover.build(base_paths, N11B, threads=2)
    sealed = crossover.build(sealed_paths, N11B, threads=2)
    assert np.array_equal(base["entrants"], sealed["entrants"]) and np.array_equal(base["follow_ups"], sealed["follow_ups"])
    assert crossover.quarter_table(base).equals(crossover.quarter_table(sealed))


def test_structure_only_reads_no_work(tmp_path: Path) -> None:
    paths, _ = _write_inputs(tmp_path / "ours", WORKS)
    paths = {**paths, "works": tmp_path / "missing.parquet", "authorships": tmp_path / "missing.parquet"}
    built = crossover.build(paths, N11B, threads=2, structure_only=True)
    assert built["follow_ups"] is None
    assert int(built["entrants"].sum()) == 7


def test_the_contract_yaml_round_trips() -> None:
    assert yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8")) == CONTRACT
