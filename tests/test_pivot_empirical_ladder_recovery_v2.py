from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prototype_pivot_empirical_ladder_recovery_v2.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
n11a2 = importlib.import_module(SCRIPT.stem)


def _contract() -> dict:
    return n11a2.load_contract()


def test_v2_amends_only_the_ladder_the_rule_the_roles_and_the_seeds() -> None:
    contract = _contract()
    assert contract["amends"]["result_informed_successor"] is True
    assert n11a2.MODELS[-1] == "B7_anticipatory" and "B7_anticipatory" in n11a2.NON_LEARNING
    roles = {w["id"]: w["role"] for w in contract["worlds"]}
    assert roles["W6_static_aligned_drift"] == "MUST_NOT_FALSE_RECOVER"
    assert roles["W7_learning_with_anticipation"] == "REPORTED_BOTH_MECHANISMS"
    assert contract["seeds"]["base_seed"] not in contract["seeds"]["opened"]
    assert len(n11a2.v1.verify_source_bindings(contract)) == 6


@pytest.mark.parametrize("block", ["episode_structure", "mechanisms", "reported_sign_test", "profiles"])
def test_v2_may_not_change_a_v1_block(block: str) -> None:
    contract = copy.deepcopy(_contract())
    contract[block] = {**contract[block], "changed": True}
    with pytest.raises(ValueError, match="may not change"):
        n11a2.validate_contract(contract)


def test_reused_seeds_and_a_learning_w6_are_refused() -> None:
    contract = copy.deepcopy(_contract())
    contract["seeds"]["base_seed"] = 20261101
    with pytest.raises(ValueError, match="fresh seeds"):
        n11a2.validate_contract(contract)
    contract = copy.deepcopy(_contract())
    next(w for w in contract["worlds"] if w["id"] == "W6_static_aligned_drift")["kappa"] = 0.3
    with pytest.raises(ValueError, match="must not learn"):
        n11a2.validate_contract(contract)


def _episode(world_id: str, seed: int = 4) -> dict:
    contract = _contract()
    world = next(w for w in contract["worlds"] if w["id"] == world_id)
    params = {**n11a2.v1.profile_params(contract, "smoke"), "at_risk_authors": 600}
    return n11a2.v1.simulate_episode(world, params, contract["mechanisms"], np.random.SeedSequence(seed))


def test_the_anticipatory_term_acts_from_the_first_quarter() -> None:
    episode = _episode("W6_static_aligned_drift")
    term = n11a2.anticipatory_rows(episode, 10)
    entry = episode["entry"]
    first = int(((entry == -1) | (entry >= 0)).sum())
    assert term.size == sum(int(((entry == -1) | (entry >= q)).sum()) for q in range(n11a2.v1.QUARTERS))
    assert np.abs(term[:first]).sum() > 0
    assert abs(term[:first].mean()) < 1e-9
    rows = n11a2.v1.ladder_rows(episode, _contract()["ladder"], 10, np.random.SeedSequence(5))
    assert not rows["B4_linear"][:first].any()


def test_b7_design_and_fit() -> None:
    episode = _episode("W3_static")
    rows = n11a2.v1.ladder_rows(episode, _contract()["ladder"], 10, np.random.SeedSequence(5))
    rows["B7_anticipatory"] = n11a2.anticipatory_rows(episode, 10)
    rows["episode"] = np.zeros(rows["y"].size, dtype=int)
    assert n11a2.design("B7_anticipatory", rows).shape[1] == 6
    clf = n11a2.fit("B7_anticipatory", rows)
    assert n11a2.episode_loglik(clf, "B7_anticipatory", rows, 1)[0] < 0


def test_run_task_smoke_compares_against_three_rivals_and_is_deterministic() -> None:
    contract = _contract()
    result = n11a2.run_task(contract, "smoke", 0, 0)
    assert set(result["comparisons"]) == {"best_non_learning", "b2", "anticipatory"}
    assert result["comparisons"]["anticipatory"]["rival"] == "B7_anticipatory"
    assert result["nabm_win"] == all(c["won"] for c in result["comparisons"].values())
    assert n11a2.run_task(contract, "smoke", 0, 0) == result
