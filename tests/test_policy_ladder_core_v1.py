from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import prototype_pivot_matched_marginal_discriminator_v3 as pivot_v3  # noqa: E402
import prototype_pivot_nonlinear_policy_abm_v1 as pivot_abm1  # noqa: E402


def _pivot_prior() -> core.LinearPrior:
    weight, bias = pivot_v3.v2.study2._initial_policy()
    return core.LinearPrior(weight.detach().numpy().copy(), float(bias.detach()))


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    width = len(pivot_v3.v2.study2.FEATURES)
    features = rng.random((160, width))
    labels = (rng.random(160) < 0.35).astype(np.float64)
    return features, labels, rng.random((48, width))


@pytest.mark.parametrize("family", ["linear", "mlp"])
def test_residual_learners_reproduce_pivot_bytes(family: str) -> None:
    features, labels, evaluation = _data()
    settings = {"learning_rate": 0.05, "epochs": 40, "weight_decay": 1e-3, "hidden_width": 8}
    model = {
        "id": "matched_linear" if family == "linear" else "residual_mlp",
        "learning_rate": settings["learning_rate"],
        "epochs": settings["epochs"],
        "hidden_width": settings["hidden_width"],
    }
    expected = pivot_v3.fit_candidate(
        model, {"weight_decay": settings["weight_decay"]}, features, labels, evaluation, 11
    )
    observed = core.fit_residual(family, _pivot_prior(), features, labels, settings, 11)
    assert np.array_equal(observed.predict(evaluation), expected)


def test_boosted_trees_reproduce_pivot_bytes() -> None:
    features, labels, evaluation = _data(1)
    settings = {
        "rounds": 6, "learning_rate": 0.3, "leaf_l2": 1.0, "max_depth": 3,
        "min_samples_leaf": 5,
    }
    expected = pivot_abm1.fit_prior_boosted_trees(features, labels, evaluation, settings, 5)
    observed = core.fit_prior_boosted_trees(_pivot_prior(), features, labels, settings, 5)
    assert np.array_equal(observed.predict(evaluation), expected)


@pytest.mark.parametrize("family", ["linear", "mlp", "tree"])
def test_a_learner_without_rows_is_the_prior(family: str) -> None:
    prior = _pivot_prior()
    _features, _labels, evaluation = _data()
    predictor = core.fit_residual(family, prior, evaluation[:0], np.zeros(0), {}, 0)
    assert np.array_equal(predictor.predict(evaluation), prior.probability(evaluation))


@pytest.mark.parametrize("family", ["linear", "mlp"])
def test_batched_agents_learn_only_from_their_own_rows(family: str) -> None:
    prior = _pivot_prior()
    features, labels, evaluation = _data(2)
    owners = np.arange(len(labels)) % 3  # agents 0-2 have rows; agent 3 has none
    settings = {"learning_rate": 0.05, "epochs": 30, "weight_decay": 1e-3, "hidden_width": 4}
    state = core.fit_batched_residual(
        family, prior, owners, features, labels, agents=4, settings=settings, seed=3
    )
    predictor = core.BatchedPredictor(prior, state)
    idle = predictor.predict_for(np.full(len(evaluation), 3), evaluation)
    assert np.array_equal(idle, prior.probability(evaluation))

    # Agent 1's fit does not change when agent 0's rows change.
    changed = labels.copy()
    changed[owners == 0] = 1.0 - changed[owners == 0]
    other = core.fit_batched_residual(
        family, prior, owners, features, changed, agents=4, settings=settings, seed=3
    )
    agent1 = np.full(len(evaluation), 1)
    assert np.array_equal(
        core.BatchedPredictor(prior, other).predict_for(agent1, evaluation),
        predictor.predict_for(agent1, evaluation),
    )


def test_keyed_streams_and_exploration_floor() -> None:
    first = core.keyed_uniform((1, 2, 3), 5)
    assert np.array_equal(first, core.keyed_uniform((1, 2, 3), 5))
    assert not np.array_equal(first, core.keyed_uniform((1, 2, 4), 5))
    assert not core.ExplorationFloor(0.0).explores(np.zeros(4)).any()
    assert core.ExplorationFloor(0.5).explores(np.array([0.1, 0.9])).tolist() == [True, False]


def test_outcome_record_and_snapshots() -> None:
    record = core.OutcomeRecord(width=2)
    record.add(np.ones((2, 2)), np.array([1.0, 0.0]), np.array([0, 1]), visible_from=3)
    record.add(np.zeros((1, 2)), np.array([1.0]), np.array([1]), visible_from=5)
    assert record.visible(2)[1].size == 0
    assert record.visible(3)[1].tolist() == [1.0, 0.0]
    assert record.visible_counts_by_agent(5, 3).tolist() == [1, 2, 0]

    snapshots = core.PolicySnapshots()
    snapshots.record(0, "prior")
    snapshots.record(4, "refit")
    assert [snapshots.active(t) for t in (0, 3, 4, 9)] == ["prior", "prior", "refit", "refit"]
    with pytest.raises(ValueError):
        snapshots.record(4, "again")


def test_the_outcome_record_window_keeps_recent_rows_only() -> None:
    record = core.OutcomeRecord(width=1)
    for step in range(5):
        record.add(np.full((1, 1), step), np.array([1.0]), np.array([0]), visible_from=step)
    assert record.visible(4)[0].ravel().tolist() == [0, 1, 2, 3, 4]
    assert record.visible(4, since=2)[0].ravel().tolist() == [2, 3, 4]
