from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import info_spread_unverified_measure_v1 as unverified  # noqa: E402
import info_spread_utility_measure_v1 as utility  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_study_v1 as v1  # noqa: E402


def _world(regime: str) -> spread.World:
    return spread.build_world(spread.WorldConfig(agents=60, steps=20, items_per_step=4, regime=regime,
                                                 geometry="linear", epsilon=0.02, world_seed=12))


@pytest.mark.parametrize("regime", ["A", "B"])
def test_it_reproduces_the_unverified_measure_and_adds_utility(regime: str) -> None:
    contract = v1.load_contract()
    v1.apply_learner_settings(contract)
    world = _world(regime)
    stream = spread.build_stream(world, 1)
    family = "linear_exact" if regime == "A" else "partial_linear_exact"
    for make in (lambda: spread.prior_policy("F1", spread.hand_prior(world.config)),
                 lambda: v1._learner(contract, world, family, 0, 0, 1, "cpu"),
                 lambda: unverified.threshold_probability_policy(world)):
        result, ours = utility.measured_simulate(world, stream, make(), 1)
        plain, theirs = unverified.measured_simulate(world, stream, make(), 1)
        assert result.digest == plain.digest
        for key in ("share_precision", "unverified_share_precision", "false_reach", "decisions", "by_state"):
            assert ours[key] == theirs[key], key
        if regime == "A":
            assert ours["unverified_utility_per_agent"] == utility.utility_from_counts(theirs["by_state"],
                                                                                       world.config.agents)


def test_false_shares_cost_their_sharers_cost_in_regime_b() -> None:
    world = _world("B")
    stream = spread.build_stream(world, 3)
    # A policy that shares everything makes many false shares by agents of every cost.
    always = spread.Policy("share_all", probability=lambda ids, x, _t: np.ones(len(ids)))
    _result, measured = utility.measured_simulate(world, stream, always, 3)
    unit = utility.utility_from_counts(measured["by_state"], world.config.agents)
    assert len(set(world.costs)) == 3
    assert measured["unverified_utility_per_agent"] != unit
