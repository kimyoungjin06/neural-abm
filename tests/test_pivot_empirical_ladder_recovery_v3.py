from __future__ import annotations

import copy
import importlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prototype_pivot_empirical_ladder_recovery_v3.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
n11a3 = importlib.import_module(SCRIPT.stem)
N11B_RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_empirical_ladder_risk_sets_v1_20261012.summary.json"


def _contract() -> dict:
    return n11a3.load_contract()


def _world(contract: dict, world_id: str) -> dict:
    return next(w for w in contract["worlds"] if w["id"] == world_id)


# -- contract ----------------------------------------------------------------------------


def test_v3_keeps_the_v2_ladder_rule_and_mechanisms_with_fresh_seeds() -> None:
    contract = _contract()
    assert n11a3.MODELS[-1] == "B7_anticipatory" and "B7_anticipatory" in n11a3.NON_LEARNING
    roles = {w["id"]: w["role"] for w in contract["worlds"]}
    assert [k for k, r in roles.items() if r == "MUST_RECOVER"] == ["W1_learning", "W2_learning_confounded"]
    assert sum(r == "MUST_NOT_FALSE_RECOVER" for r in roles.values()) == 5
    assert contract["seeds"]["base_seed"] not in contract["seeds"]["opened"]
    assert len(n11a3.v1.verify_source_bindings(contract)) == 7


def test_the_episode_shape_is_the_recorded_n11b_shape() -> None:
    contract = _contract()
    n11b = json.loads(N11B_RESULT.read_text(encoding="utf-8"))
    shapes, discovery = n11a3.episode_shapes(contract, "development")
    by_partition = n11b["post_hoc_structural_diagnostics"]["by_partition"]
    assert discovery == by_partition["discovery"]["episodes"] == 24
    assert len(shapes) - discovery == by_partition["confirmatory"]["episodes"] == 39
    assert sum(s["at_risk"] for s in shapes) == n11b["risk_set"]["at_risk_authors_union_total"]
    assert sum(s["entrants"] for s in shapes) == n11b["sampling"]["entrants_total"]
    confirmatory = [s["entrants"] for s in shapes[discovery:]]
    assert float(np.median(confirmatory)) == by_partition["confirmatory"]["entrants"]["median"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["mechanisms"]["follow_up"].update(success_rate_per_quarter=0.5), "may not change the v2 follow_up"),
        (lambda c: c["ladder"]["learners"]["mlp"].update(hidden=32), "may not change the v2 ladder learners"),
        (lambda c: c["decision"].update(must_recover_at_least=6), "may not change the v2 decision"),
        (lambda c: _world(c, "W8_static_recruitment").update(kappa=0.2), "must not learn"),
        (lambda c: _world(c, "W1_learning").update(confounds=["telepathy"]), "unknown confound"),
        (lambda c: c["seeds"].update(base_seed=20261102), "fresh seeds"),
        (lambda c: c["episode_shape"]["confirmatory"][0].__setitem__(2, 0), "between zero"),
    ],
)
def test_contract_mutations_are_refused(mutate, message: str) -> None:
    contract = copy.deepcopy(_contract())
    mutate(contract)
    with pytest.raises(ValueError, match=message):
        n11a3.validate_contract(contract)


# -- features and the world ----------------------------------------------------------------


def test_feature_marginals_match_the_n11b_pooled_quantiles() -> None:
    spec = _contract()["features"]
    counts = n11a3._counts(np.random.default_rng(3), 200_000, spec)
    productivity, career, proximity, network = counts.T
    assert np.median(productivity) == 1 and np.quantile(productivity, 0.75) == 2
    assert np.median(career) == 3 and np.quantile(career, 0.25) == 2 and np.quantile(career, 0.75) in (8, 9)
    assert np.median(proximity) == 0 and 0.30 < np.quantile(proximity, 0.75) < 0.42
    assert 0.068 < (network > 0).mean() < 0.078
    z = n11a3.draw_features(np.random.default_rng(4), 200_000, spec, 20261104)
    assert np.allclose(z.mean(axis=0), 0, atol=0.02) and np.allclose(z.std(axis=0), 1, atol=0.02)


def _shape(at_risk: int = 20_000, entrants: int = 300) -> dict:
    return {"field": 11, "at_risk": at_risk, "entrants": entrants}


@pytest.mark.parametrize("world_id", ["W3_static", "W1s_strong_learning", "W6_static_aligned_drift"])
def test_selectivity_terms_keep_the_entry_level(world_id: str) -> None:
    contract = _contract()
    rare = _shape(at_risk=300_000, entrants=300)
    counts = [n11a3.simulate_episode(_world(contract, world_id), rare, contract, np.random.SeedSequence(s))["entrants"]
              for s in range(4)]
    assert 0.85 * 300 < np.mean(counts) < 1.15 * 300


def test_recruitment_adds_flagged_entrants_and_dates_are_thinned() -> None:
    contract = _contract()
    episode = n11a3.simulate_episode(_world(contract, "W8_static_recruitment"), _shape(), contract, np.random.SeedSequence(1))
    assert 0 < episode["recruited_entrants"] < episode["entrants"]
    assert episode["recruited"].sum() == episode["recruited_entrants"]
    assert 0.45 < episode["precise_entrants"] / episode["entrants"] < 0.70
    static = n11a3.simulate_episode(_world(contract, "W3_static"), _shape(), contract, np.random.SeedSequence(1))
    assert static["recruited_entrants"] == 0


def test_the_case_cohort_keeps_every_entrant_and_weights_rebuild_the_risk_set() -> None:
    contract = _contract()
    episode = n11a3.simulate_episode(_world(contract, "W3_static"), _shape(), contract, np.random.SeedSequence(2))
    assert episode["entrant"].sum() == episode["entrants"]
    assert episode["probability"] == pytest.approx(max(2000, 20 * episode["entrants"]) / 20_000)
    rows = n11a3.ladder_rows(episode, contract["ladder"], 10, np.random.SeedSequence(3))
    first = rows["year"] == 0
    quarter0 = np.zeros(rows["y"].size, dtype=bool)
    quarter0[: int((n11a3.last_rows(episode, True)[0] >= 0).sum())] = True
    assert rows["w"][quarter0].sum() == pytest.approx(20_000, rel=0.05)
    assert set(np.unique(rows["w"])) <= {1.0, 1.0 / episode["probability"]}
    assert first.any()


# -- the analysis --------------------------------------------------------------------------


def _follow(columns: list[int], width: int = n11a3.FOLLOW_QUARTERS) -> np.ndarray:
    row = np.zeros(width, dtype=bool)
    row[columns] = True
    return row


def test_analyst_labels_use_only_what_dated_records_show() -> None:
    entry = np.array([2, 2, 3, 5, 5])
    precise_entry = np.array([True, True, True, False, False])
    seen = np.vstack([_follow([1]), _follow([1]), _follow([0]), _follow([2]), _follow([1])])
    precise_follow = np.vstack([_follow([1]), _follow([]), _follow([]), _follow([]), _follow([])])
    label, visible = n11a3.analyst_labels(entry, precise_entry, seen, precise_follow)
    # quarter-dated follow-up in the window; year-only one not wholly inside it; year-only one filling the
    # window after a fourth-quarter entry; an imprecise entrant's follow-up in the next year (quarter 8) and
    # in its own year (quarter 7).
    assert label.tolist() == [1, 0, 1, 1, 0]
    assert visible.tolist() == [7, 7, 8, 12, 12]


def test_last_rows_events_and_censoring() -> None:
    episode = {
        "entry": np.array([-1, 6, 6, 6, 2]),
        "entrant": np.array([False, True, True, True, True]),
        "precise": np.array([False, True, True, False, False]),
        "recruited": np.array([False, False, True, False, False]),
    }
    last, event = n11a3.last_rows(episode, censor_recruited=True)
    assert last.tolist() == [11, 6, 5, 3, -1]
    assert event.tolist() == [False, True, False, False, False]
    last, event = n11a3.last_rows(episode, censor_recruited=False)
    assert last.tolist() == [11, 6, 6, 3, -1]
    assert event.tolist() == [False, True, True, False, False]


def test_learned_terms_are_centred_by_the_weighted_risk_set_mean() -> None:
    contract = _contract()
    episode = n11a3.simulate_episode(_world(contract, "W1s_strong_learning"), _shape(), contract, np.random.SeedSequence(5))
    rows = n11a3.ladder_rows(episode, contract["ladder"], 10, np.random.SeedSequence(6))
    last, _ = n11a3.last_rows(episode, True)
    start = 0
    for quarter in range(n11a3.QUARTERS):
        size = int((last >= quarter).sum())
        part = slice(start, start + size)
        for column in ("B4_linear", "B7_anticipatory"):
            assert abs(np.dot(rows[column][part], rows["w"][part])) < 1e-6 * rows["w"][part].sum()
        start += size
    assert start == rows["y"].size
    assert rows["B7_anticipatory"][: int((last >= 0).sum())].any()


def test_newton_fit_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40_000, 3))
    y = (rng.random(40_000) < 1 / (1 + np.exp(-(-4 + x @ [0.5, -0.3, 0.2])))).astype(float)
    w = np.where(y == 1, 1.0, 1.0 + 9 * (rng.random(40_000) < 0.5))
    beta = n11a3.fit_logit(x, y, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = LogisticRegression(C=np.inf, max_iter=5000, tol=1e-12).fit(x, y, sample_weight=w)
    assert np.allclose(beta, np.r_[reference.intercept_, reference.coef_[0]], atol=1e-4)


def test_smoke_task_is_deterministic_and_scores_recruitment_both_ways() -> None:
    contract = _contract()
    index = [w["id"] for w in contract["worlds"]].index("W8_static_recruitment")
    first = n11a3.run_task(contract, "smoke", index, 0)
    assert first == n11a3.run_task(contract, "smoke", index, 0)
    assert first["without_recruitment_censoring"] is not None
    assert set(first["comparisons"]) == {"best_non_learning", "b2", "anticipatory"}
    assert first["recruited_share"] > 0 and first["events"] > 0


def _row(world: str, role: str, win: bool) -> dict:
    comparison = {"won": win, "mean": 1.0 if win else -1.0}
    return {
        "world": world, "role": role, "nabm_win": win, "best_nabm": "B4_linear", "best_non_learning": "B2",
        "comparisons": {name: comparison for name in ("best_non_learning", "b2", "anticipatory")},
        "learned_coefficient": 1.0, "anticipatory_coefficient": 0.0, "entrants_per_episode_median": 100.0,
        "precise_share": 0.575, "recruited_share": 0.0, "without_recruitment_censoring": None,
        "confirmatory_evidence_pool_median": {f"quarter_{q}": 1.0 for q in n11a3.POOL_QUARTERS},
    }


def _rows(contract: dict, recover: int, false_wins: int) -> list[dict]:
    rows = []
    for world in contract["worlds"]:
        for replicate in range(8):
            if world["role"] == "MUST_RECOVER":
                win = replicate < recover
            elif world["role"] == "MUST_NOT_FALSE_RECOVER":
                win = world["id"] == "W8_static_recruitment" and replicate < false_wins
            else:
                win = True
            rows.append(_row(world["id"], world["role"], win))
    return rows


@pytest.mark.parametrize(
    ("recover", "false_wins", "disposition"),
    [
        (8, 1, "RECOVERY_PASSES_AT_THE_EMPIRICAL_SHAPE"),
        (6, 0, "UNDERPOWERED_AT_THE_EMPIRICAL_SHAPE"),
        (8, 2, "FALSE_RECOVERY_AT_THE_EMPIRICAL_SHAPE"),
        (3, 2, "FALSE_RECOVERY_AT_THE_EMPIRICAL_SHAPE"),
    ],
)
def test_dispositions(recover: int, false_wins: int, disposition: str) -> None:
    contract = _contract()
    decision = n11a3.decide(_rows(contract, recover, false_wins), contract)
    assert decision["disposition"] == disposition
    assert decision["action"] == contract["decision"]["actions"][disposition]


def test_run_requires_a_full_committed_sha() -> None:
    with pytest.raises(RuntimeError, match="full 40-character"):
        n11a3.v1.verify_precommit("abc1234", _contract())
