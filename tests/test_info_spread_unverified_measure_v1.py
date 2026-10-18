from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_info_spread_null_leak_v1 as leak  # noqa: E402
import info_spread_unverified_measure_v1 as measure  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_study_v1 as d4  # noqa: E402


def _world(regime: str, geometry: str = "linear", epsilon: float = 0.02) -> spread.World:
    return spread.build_world(spread.WorldConfig(
        agents=60, steps=20, items_per_step=4, regime=regime, geometry=geometry, epsilon=epsilon, world_seed=9,
    ))


@pytest.mark.parametrize("regime", ["A", "B"])
def test_f0_in_probability_form_makes_the_threshold_rules_decisions(regime: str) -> None:
    world = _world(regime)
    for replicate in range(3):
        stream = spread.build_stream(world, replicate)
        plain = spread.simulate(world, stream, spread.threshold_policy(), replicate)
        result, measured = measure.measured_simulate(
            world, stream, measure.threshold_probability_policy(world), replicate)
        assert result.digest == plain.digest
        assert measured["share_precision"] == plain.behavior["share_precision"]


@pytest.mark.parametrize("regime", ["A", "B"])
def test_measuring_changes_no_decision_of_any_arm(regime: str) -> None:
    contract = d4.load_contract()
    d4.apply_learner_settings(contract)
    world = _world(regime)
    stream = spread.build_stream(world, 2)
    family = "linear_exact" if regime == "A" else "partial_linear_exact"
    for make in (
        lambda: spread.prior_policy("F1", spread.hand_prior(world.config)),
        lambda: d4._learner(contract, world, family, 0, 0, 2, "cpu"),
    ):
        plain = spread.simulate(world, stream, make(), 2)
        result, measured = measure.measured_simulate(world, stream, make(), 2)
        assert result.digest == plain.digest
        states = measured["by_state"]
        assert sum(s["decisions"] for s in states.values()) == plain.decisions
        assert sum(s["shares"] for s in states.values()) == int(plain.decision_shares.sum())
        unverified = states["unrevealed"]["shares"] + states["pending"]["shares"]
        true = states["unrevealed"]["true"] + states["pending"]["true"]
        assert measured["unverified_share_precision"] == true / unverified


def test_the_split_matches_the_diagnosis_without_an_exploration_floor() -> None:
    contract = d4.load_contract()
    d4.apply_learner_settings(contract)
    world = _world("B", geometry="null", epsilon=0.0)
    stream = spread.build_stream(world, 1)
    log: list[np.ndarray] = []
    spread.simulate(world, stream, leak.logged(d4._learner(contract, world, "partial_mlp", 3, 0, 1, "cpu"),
                                                world, stream, log), 1)
    expected = leak.split_precision(world, stream, log)
    _result, measured = measure.measured_simulate(
        world, stream, d4._learner(contract, world, "partial_mlp", 3, 0, 1, "cpu"), 1)
    assert measured["by_state"] == expected


def test_a_policy_without_probabilities_is_refused() -> None:
    world = _world("A")
    with pytest.raises(ValueError):
        measure.measured_simulate(world, spread.build_stream(world, 0), spread.threshold_policy(), 0)
