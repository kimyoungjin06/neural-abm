"""The package's policy-ladder primitives against the frozen repository core.

Decision 0017 extracts the semantics that two consumers share. These tests
check that the package versions reproduce scripts/policy_ladder_core_v1.py
byte for byte, change no global state, and follow the outcome-record ordering
rule.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_abm import policy_ladder as ladder
from neural_abm import policy_learners as learners


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402


def _priors(width: int = 6, seed: int = 0) -> tuple[ladder.LinearPrior, core.LinearPrior]:
    rng = np.random.default_rng(seed)
    weight, bias = rng.normal(size=width), float(rng.normal())
    return ladder.LinearPrior(weight.copy(), bias), core.LinearPrior(weight.copy(), bias)


def _data(width: int = 6, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.random((180, width))
    labels = (rng.random(180) < 0.4).astype(np.float64)
    return features, labels, rng.random((50, width))


@pytest.fixture(autouse=True)
def _single_thread() -> None:
    torch.set_num_threads(1)


def test_keyed_randomness_and_the_floor_match_the_core() -> None:
    for components in ((7, 3, 1), (20261023, 0, 83, 5, 11)):
        assert ladder.keyed_seed(*components) == core.keyed_seed(*components)
        assert np.array_equal(ladder.keyed_uniform(components, 40), core.keyed_uniform(components, 40))
    draws = ladder.keyed_uniform((1, 2), 1000)
    for epsilon in (0.0, 0.02, 0.5):
        assert np.array_equal(ladder.ExplorationFloor(epsilon).explores(draws),
                              core.ExplorationFloor(epsilon).explores(draws))


def test_the_prior_matches_the_core() -> None:
    package, frozen = _priors()
    features, _labels, _evaluation = _data()
    assert np.array_equal(package.logit(features), frozen.logit(features))
    assert np.array_equal(package.probability(features), frozen.probability(features))
    assert np.array_equal(ladder.PriorPredictor(package).predict(features), frozen.probability(features))


@pytest.mark.parametrize(
    ("family", "settings"),
    [("linear", {"learning_rate": 0.02, "epochs": 120, "weight_decay": 1e-4}),
     ("mlp", {"learning_rate": 0.02, "epochs": 120, "weight_decay": 1e-4, "hidden_width": 16}),
     ("mlp", {"learning_rate": 0.05, "epochs": 40, "weight_decay": 1e-3, "hidden_width": 8})],
)
def test_residual_learners_match_the_core_byte_for_byte(family: str, settings: dict) -> None:
    package, frozen = _priors()
    features, labels, evaluation = _data()
    expected = core.fit_residual(family, frozen, features, labels, settings, 13).predict(evaluation)
    observed = learners.fit_residual_policy(
        family, package, features, labels, learning_rate=settings["learning_rate"],
        epochs=settings["epochs"], weight_decay=settings["weight_decay"],
        hidden_width=settings.get("hidden_width"), seed=13,
    ).predict(evaluation)
    assert np.array_equal(observed, expected)


def test_the_mlp_starts_from_the_cores_weights_and_predicts_the_prior() -> None:
    package, frozen = _priors()
    features, _labels, _evaluation = _data()
    ours, theirs = learners.ResidualMLP(package, 16, 5), core.ResidualMLP(frozen, 16, 5)
    for name, value in theirs.state_dict().items():
        assert torch.equal(ours.state_dict()[name], value), name
    with torch.no_grad():
        logits = ours.to(torch.float64)(torch.as_tensor(features, dtype=torch.float64)).numpy()
    assert np.allclose(logits, package.logit(features), rtol=0, atol=1e-12)


def test_fitting_changes_no_global_state() -> None:
    package, _frozen = _priors()
    features, labels, _evaluation = _data()
    torch.manual_seed(1234)
    rng_state = torch.random.get_rng_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    for family in ("linear", "mlp"):
        learners.fit_residual_policy(family, package, features, labels, learning_rate=0.02, epochs=5,
                                     weight_decay=1e-4, hidden_width=4, seed=3)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_a_learner_without_rows_is_the_prior() -> None:
    package, _frozen = _priors()
    _features, _labels, evaluation = _data()
    empty_x, empty_y = np.zeros((0, 6)), np.zeros(0)
    for family in ("linear", "mlp"):
        predictor = learners.fit_residual_policy(family, package, empty_x, empty_y, learning_rate=0.1,
                                                 epochs=3, weight_decay=0.0, hidden_width=4, seed=1)
        assert np.array_equal(predictor.predict(evaluation), package.probability(evaluation))
    trees = ladder.fit_prior_boosted_trees(package, empty_x, empty_y, {"learning_rate": 0.1}, 0)
    assert np.array_equal(trees.predict(evaluation), package.probability(evaluation))
    with pytest.raises(ValueError):
        learners.fit_residual_policy("tree", package, empty_x, empty_y, learning_rate=0.1, epochs=1,
                                     weight_decay=0.0)


def test_boosted_trees_match_the_core_byte_for_byte() -> None:
    package, frozen = _priors()
    features, labels, evaluation = _data()
    settings = {"rounds": 40, "learning_rate": 0.1, "leaf_l2": 1.0, "max_depth": 3, "min_samples_leaf": 5}
    expected = core.fit_prior_boosted_trees(frozen, features, labels, settings, 9).predict(evaluation)
    observed = ladder.fit_prior_boosted_trees(package, features, labels, settings, 9).predict(evaluation)
    assert np.array_equal(observed, expected)


def _fill(record: object, steps: list[int], kinds: list[int]) -> None:
    rng = np.random.default_rng(4)
    for index, (step, kind) in enumerate(zip(steps, kinds, strict=True)):
        record.add(rng.random((3, 4)), (rng.random(3) < 0.5).astype(float), np.arange(3) + index,
                   visible_from=step, kinds=np.full(3, kind))


def test_the_record_matches_the_core_when_rows_arrive_in_release_order() -> None:
    steps, kinds = [0, 0, 1, 3, 3, 4, 7], [1, 0, 1, 1, 0, 1, 1]
    ours, theirs = ladder.OutcomeRecord(width=4), core.OutcomeRecord(width=4)
    _fill(ours, steps, kinds)
    _fill(theirs, steps, kinds)
    for step in range(9):
        for since in (None, 2):
            for kind in ((1,), None, (0,)):
                for a, b in zip(ours.visible(step, since, kinds=kind), theirs.visible(step, since, kinds=kind),
                                strict=True):
                    assert np.array_equal(a, b)
    assert np.array_equal(ours.visible_counts_by_agent(8, 12), theirs.visible_counts_by_agent(8, 12))


def test_the_record_returns_rows_in_release_order() -> None:
    record = ladder.OutcomeRecord(width=1)
    # Added at the event, with later visibility: a failure seen at step 5 is
    # added before a success seen at step 2.
    record.add(np.array([[1.0]]), np.array([0.0]), np.array([0]), visible_from=5)
    record.add(np.array([[2.0]]), np.array([1.0]), np.array([1]), visible_from=2)
    record.add(np.array([[3.0]]), np.array([1.0]), np.array([2]), visible_from=2)
    features, labels, agents = record.visible(6)
    assert features.ravel().tolist() == [2.0, 3.0, 1.0]
    assert agents.tolist() == [1, 2, 0]
    assert record.visible(4)[2].tolist() == [1, 2]
    with pytest.raises(ValueError):
        record.add(np.zeros((1, 2)), np.zeros(1), np.zeros(1), visible_from=0)


def test_the_information_spread_ladder_runs_unchanged_on_the_package_record(monkeypatch: pytest.MonkeyPatch) -> None:
    import prototype_info_spread_policy_ladder_v1 as spread  # noqa: PLC0415
    import run_info_spread_policy_ladder_study_v1 as d4  # noqa: PLC0415

    contract = d4.load_contract()
    d4.apply_learner_settings(contract)
    world = spread.build_world(spread.WorldConfig(agents=60, steps=20, items_per_step=4, regime="B",
                                                  geometry="smooth", epsilon=0.02, world_seed=8))
    stream = spread.build_stream(world, 1)

    def digests() -> list[str]:
        return [
            spread.simulate(world, stream, d4._learner(contract, world, family, 1, 0, 1, "cpu"), 1).digest
            for family in ("partial_linear_exact", "partial_mlp")
        ]

    frozen = digests()
    monkeypatch.setattr(core, "OutcomeRecord", ladder.OutcomeRecord)
    assert digests() == frozen


def test_the_torch_free_module_imports_neither_torch_nor_scikit_learn() -> None:
    code = (
        "import sys; import neural_abm.policy_ladder, neural_abm.api_lite; "
        "print(sorted(m for m in ('torch', 'sklearn', 'scipy') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
