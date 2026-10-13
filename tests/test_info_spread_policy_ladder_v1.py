"""Stage D1 nesting checks for the information-spread policy ladder.

Each check is exact: two arms that should coincide must make the same
decisions in the same order, compared by digest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402


def _small(regime: str = "A", geometry: str = "linear", epsilon: float = 0.0) -> spread.WorldConfig:
    return spread.WorldConfig(
        agents=80, steps=16, items_per_step=4, regime=regime, geometry=geometry,
        world_seed=7, epsilon=epsilon, refit_every=3,
    )


def _run(config: spread.WorldConfig, policy: spread.Policy, replicate: int = 0) -> spread.RunResult:
    world = spread.build_world(config)
    return spread.simulate(world, spread.build_stream(world, replicate), policy, replicate)


def _threshold_reference(world: spread.World, stream: spread.Stream) -> np.ndarray:
    """An independent loop: synchronous threshold dynamics, absorbing shares."""

    neighbors = [world.adjacency.indices[world.adjacency.indptr[i]:world.adjacency.indptr[i + 1]]
                 for i in range(world.config.agents)]
    reach = np.zeros(len(stream.veracity), dtype=np.int64)
    for item in range(len(stream.veracity)):
        shared = set(int(a) for a in stream.seeds[item])
        frontier = set(shared)
        for _step in range(int(stream.born[item]), world.config.steps):
            if not frontier:
                break
            exposed = {n for a in frontier for n in neighbors[a]} - shared
            adopters = {
                i for i in exposed
                if sum(1 for n in neighbors[i] if n in shared) / max(len(neighbors[i]), 1)
                >= world.thresholds[i]
            }
            shared |= adopters
            frontier = adopters
        reach[item] = len(shared)
    return reach


def test_threshold_arm_reproduces_independent_threshold_dynamics() -> None:
    config = _small()
    world = spread.build_world(config)
    stream = spread.build_stream(world, 0)
    result = spread.simulate(world, stream, spread.threshold_policy(), 0)
    assert np.array_equal(result.reach, _threshold_reference(world, stream))


@pytest.mark.parametrize("raised_threshold", [False, True])
def test_granovetter_knife_edge(raised_threshold: bool) -> None:
    agents = 20
    config = spread.WorldConfig(agents=agents, steps=agents + 2, items_per_step=1, seeds_per_item=1)
    adjacency = sp.csr_matrix(nx.to_scipy_sparse_array(nx.complete_graph(agents)), dtype=np.float64)
    thresholds = np.arange(agents) / (agents - 1)
    if raised_threshold:
        thresholds[1] = 2 / (agents - 1)
    world = spread.World(config, adjacency, np.full(agents, agents - 1.0), thresholds, np.ones(agents), 0.0)
    count = config.steps
    stream = spread.Stream(
        features=np.full((count, 4), 0.5), veracity=np.ones(count, dtype=bool),
        revealed=np.zeros(count, dtype=bool), seeds=np.zeros((count, 1), dtype=np.int64),
        born=np.arange(count),
    )
    reach = spread.simulate(world, stream, spread.threshold_policy(), 0).reach
    assert reach[0] == (1 if raised_threshold else agents)


@pytest.mark.parametrize("regime", ["A", "B"])
def test_a_learner_with_learning_off_is_the_hand_rule(regime: str) -> None:
    config = _small(regime)
    prior = spread.hand_prior(config)
    if regime == "A":
        learner = spread.pooled_learning_policy("linear", prior, seed_components=(1,), learning=False)
    else:
        learner = spread.per_agent_learning_policy(
            "linear", prior, config.agents, seed_components=(1,), learning=False
        )
    assert _run(config, learner).digest == _run(config, spread.prior_policy("F1", prior)).digest


def test_a_constant_table_is_the_hand_rule() -> None:
    config = _small()
    prior = spread.hand_prior(config)
    donor = core.PolicySnapshots()
    donor.record(0, ("prior", prior))
    table = spread.table_policy("T0_const", [donor], final_only=False)
    assert _run(config, table).digest == _run(config, spread.prior_policy("F1", prior)).digest


@pytest.mark.parametrize(("regime", "family"), [("A", "linear"), ("A", "tree"), ("B", "linear")])
def test_replaying_a_run_own_trajectory_reproduces_it(regime: str, family: str) -> None:
    config = _small(regime, epsilon=0.05)
    prior = spread.hand_prior(config)
    if regime == "A":
        learner = spread.pooled_learning_policy(family, prior, seed_components=(2,))
    else:
        learner = spread.per_agent_learning_policy(family, prior, config.agents, seed_components=(2,))
    adaptive = _run(config, learner)
    assert len(adaptive.snapshots.steps) > 1  # it did refit
    replay = spread.table_policy("T0_self", [adaptive.snapshots], final_only=False)
    assert _run(config, replay).digest == adaptive.digest


def test_archetypes_of_one_with_the_hand_rule_as_answer_are_the_hand_rule() -> None:
    config = _small()
    prior = spread.hand_prior(config)
    log: list[int] = []

    def per_item(x: np.ndarray) -> np.ndarray:
        return np.unique(x[:, :4], axis=0, return_inverse=True)[1].ravel()

    stub = spread.archetype_stub_policy(
        "X0_identity", np.arange(config.agents), per_item,
        lambda _a, _c, _ids, x: prior.probability(x), log,
    )
    assert _run(config, stub).digest == _run(config, spread.prior_policy("F1", prior)).digest
    assert sum(log) > 0


def test_the_exploration_floor_at_zero_changes_nothing_and_above_zero_only_adds() -> None:
    prior = spread.hand_prior(_small())
    base = _run(_small(epsilon=0.0), spread.prior_policy("F1", prior))
    again = _run(_small(epsilon=0.0), spread.prior_policy("F1", prior))
    explored = _run(_small(epsilon=0.1), spread.prior_policy("F1", prior))
    assert base.digest == again.digest
    assert explored.digest != base.digest


def test_arms_share_the_stream_and_runs_are_deterministic() -> None:
    config = _small("B", "axis")
    world = spread.build_world(config)
    assert np.array_equal(spread.build_stream(world, 3).veracity, spread.build_stream(world, 3).veracity)
    assert not np.array_equal(spread.build_stream(world, 3).features, spread.build_stream(world, 4).features)
    prior = spread.hand_prior(config)
    first = _run(config, spread.per_agent_learning_policy("mlp", prior, config.agents, seed_components=(5,)))
    second = _run(config, spread.per_agent_learning_policy("mlp", prior, config.agents, seed_components=(5,)))
    assert first.digest == second.digest


def test_the_intercept_gives_the_base_rate() -> None:
    for geometry in spread.GEOMETRIES:
        world = spread.build_world(_small(geometry=geometry))
        rng = np.random.default_rng(0)
        shape = spread._veracity_logit_without_intercept(geometry, rng.random((50_000, 4)))
        assert abs(core.sigmoid(shape + world.intercept).mean() - 0.4) < 0.01


def test_the_feasibility_report_carries_no_behavioral_outcome() -> None:
    config = _small()
    result = _run(config, spread.prior_policy("F1", spread.hand_prior(config)))
    row = spread._feasibility_row(config, result, 0)
    assert "behavior" not in row and "share_precision" not in str(row)
