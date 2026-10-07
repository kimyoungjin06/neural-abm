from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prototype_pivot_empirical_ladder_recovery_v1.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
n11a = importlib.import_module(SCRIPT.stem)


def _contract() -> dict:
    return n11a.load_contract()


def _world(world_id: str) -> dict:
    return next(w for w in _contract()["worlds"] if w["id"] == world_id)


def test_the_contract_registers_roles_and_binds_its_sources() -> None:
    contract = _contract()
    assert len(n11a.verify_source_bindings(contract)) == 3
    roles = {w["id"]: w["role"] for w in contract["worlds"]}
    assert [w for w, r in roles.items() if r == "MUST_RECOVER"] == ["W1_learning", "W2_learning_confounded"]
    assert [w for w, r in roles.items() if r == "MUST_NOT_FALSE_RECOVER"] == ["W3_static", "W4_static_confounded", "W5_static_independent_drift"]
    assert contract["boundaries"]["empirical_rows_read"] == 0


def test_a_no_learning_world_may_not_learn() -> None:
    contract = copy.deepcopy(_contract())
    next(w for w in contract["worlds"] if w["id"] == "W4_static_confounded")["kappa"] = 0.5
    with pytest.raises(ValueError, match="must not learn"):
        n11a.validate_contract(contract)


def test_the_sign_test_uses_the_n10_v2_threshold() -> None:
    contract = copy.deepcopy(_contract())
    contract["reported_sign_test"]["threshold"] = 0.2
    with pytest.raises(ValueError, match="N10 v2"):
        n11a.validate_contract(contract)


def _episode(world_id: str, seed: int = 1, n: int = 300) -> dict:
    contract = _contract()
    params = {**n11a.profile_params(contract, "smoke"), "at_risk_authors": n}
    return n11a.simulate_episode(_world(world_id), params, contract["mechanisms"], np.random.SeedSequence(seed))


def test_simulation_is_deterministic_and_entry_removes_from_risk() -> None:
    a, b = _episode("W1_learning"), _episode("W1_learning")
    assert np.array_equal(a["entry"], b["entry"]) and np.array_equal(a["observed"], b["observed"])
    entrants = a["entry"] >= 0
    assert 0.05 < entrants.mean() < 0.8
    assert not a["observed"][~entrants].any()


def test_the_growth_confound_raises_later_follow_up() -> None:
    late, early = [], []
    for seed in range(30):
        e = _episode("W4_static_confounded", seed, 600)
        entered = np.flatnonzero(e["entry"] >= 0)
        follow = e["observed"][entered].any(axis=1)
        late.extend(follow[e["entry"][entered] >= 8])
        early.extend(follow[e["entry"][entered] <= 3])
    assert np.mean(late) > np.mean(early)


def test_learned_term_needs_evidence_and_is_centred() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4))
    y = (x[:, 0] > 0).astype(int)
    assert n11a.learned_term("linear", {"C": 1.0}, x[:5], y[:5], x, 0, 10) is None
    assert n11a.learned_term("linear", {"C": 1.0}, x, np.zeros(40, dtype=int), x, 0, 10) is None
    term = n11a.learned_term("linear", {"C": 1.0}, x, y, x, 0, 10)
    assert abs(term.mean()) < 1e-9 and np.corrcoef(term, x[:, 0])[0, 1] > 0.9


def test_ladder_rows_cover_the_risk_set_and_every_entry() -> None:
    contract = _contract()
    episode = _episode("W3_static")
    rows = n11a.ladder_rows(episode, contract["ladder"], 10, np.random.SeedSequence(3))
    entry = episode["entry"]
    expected = sum(int(((entry == -1) | (entry >= q)).sum()) for q in range(n11a.QUARTERS))
    assert rows["y"].size == expected
    assert int(rows["y"].sum()) == int((entry >= 0).sum())
    assert not rows["B4_linear"][rows["has_ev"] == 0].any()
    shapes = {model: n11a.design(model, rows).shape[1] for model in n11a.MODELS}
    assert shapes == {"B1": 1, "B2": 5, "B2t": 13, "B3": 7, "B4_linear": 6, "B5_tree": 6, "B6_mlp": 6}


def test_a_zero_learned_term_reproduces_b2() -> None:
    contract = _contract()
    rows = n11a.ladder_rows(_episode("W3_static"), contract["ladder"], 10, np.random.SeedSequence(3))
    rows["B4_linear"] = np.zeros_like(rows["B4_linear"])
    b2, b4 = n11a.fit("B2", rows), n11a.fit("B4_linear", rows)
    episode = np.zeros(rows["y"].size, dtype=int)
    assert n11a.episode_loglik(b2, "B2", rows, episode, 1)[0] == pytest.approx(n11a.episode_loglik(b4, "B4_linear", rows, episode, 1)[0], abs=1e-6)


def test_decide_applies_the_registered_counts() -> None:
    contract = _contract()
    rows = []
    for world in contract["worlds"]:
        wins = 8 if world["role"] == "MUST_RECOVER" else 1
        for r in range(8):
            rows.append({"world": world["id"], "nabm_win": r < wins, "sign_test": {"positive": False}, "best_nabm": "B4_linear",
                         "best_non_learning": "B2", "mean_episode_ll_difference": 0.0})
    assert n11a.decide(rows, contract)["disposition"] == "IDENTIFICATION_PASSES"
    rows[0]["nabm_win"] = False
    rows[1]["nabm_win"] = False
    assert n11a.decide(rows, contract)["disposition"] == "IDENTIFICATION_FAILS"


def test_run_task_smoke_has_the_registered_fields() -> None:
    result = n11a.run_task(_contract(), "smoke", 0, 0)
    assert result["world"] == "W1_learning"
    assert result["best_nabm"] in n11a.NABM and result["best_non_learning"] in n11a.NON_LEARNING
    assert set(result["confirmatory_ll"]) == set(n11a.MODELS)
    again = n11a.run_task(_contract(), "smoke", 0, 0)
    assert again == result
