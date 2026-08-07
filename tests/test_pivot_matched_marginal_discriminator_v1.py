from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "experiments" / "configs" / "pivot_matched_marginal_discriminator_v1.yaml"
)
SCRIPT = ROOT / "scripts" / "prototype_pivot_matched_marginal_discriminator_v1.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("pivot_matched_marginal_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_freezes_disjoint_cohorts_and_full_baseline_ladder() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert config["cohorts"]["training"]["base_seed"] != config["cohorts"][
        "evaluation"
    ]["base_seed"]
    assert config["matched_marginal_worlds"]["success_fraction"] == 0.5
    assert [row["id"] for row in config["model_ladder"]] == [
        "fixed_rule",
        "scalar_delta",
        "explicit_linear_logit",
        "histogram_tree",
        "residual_mlp",
    ]
    assert config["model_ladder"][2]["backpropagation"] is False
    assert config["boundaries"]["empirical_rows_read"] == 0
    assert config["boundaries"]["full_agent_dynamics_run"] is False
    assert config["boundaries"]["empirical_model_selected"] == "NO"


def test_source_bindings_and_plan_validate_without_model_execution() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    plan = module._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert plan["model_ids"] == list(module.MODEL_IDS)
    assert plan["world_ids"] == list(module.WORLD_IDS)
    assert len(plan["source_bindings"]) == 4
    assert plan["boundaries"]["neural_necessity"] == "NOT_ESTABLISHED"


def test_matched_worlds_share_features_and_exact_success_fraction() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    cohort = module.make_cohort(
        contract,
        role="training",
        replicate=0,
        agent_count=64,
    )
    labels = {
        world: module.world_labels(cohort, world) for world in module.WORLD_IDS
    }
    assert all(float(values.mean()) == 0.5 for values in labels.values())
    assert np.array_equal(
        labels["linear_support_aligned"],
        1.0 - labels["linear_support_reversed"],
    )
    assert np.array_equal(
        labels["interaction_aligned"],
        1.0 - labels["interaction_reversed"],
    )


def test_training_and_evaluation_cohorts_are_deterministic_and_disjoint() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    training = module.make_cohort(
        contract,
        role="training",
        replicate=1,
        agent_count=64,
    )
    training_again = module.make_cohort(
        contract,
        role="training",
        replicate=1,
        agent_count=64,
    )
    evaluation = module.make_cohort(
        contract,
        role="evaluation",
        replicate=1,
        agent_count=64,
    )
    assert np.array_equal(training.features, training_again.features)
    assert not np.array_equal(training.features, evaluation.features)


def test_scalar_delta_is_exactly_orientation_invariant() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    train = module.make_cohort(
        contract,
        role="training",
        replicate=0,
        agent_count=64,
    )
    evaluation = module.make_cohort(
        contract,
        role="evaluation",
        replicate=0,
        agent_count=64,
    )
    for left, right in module.PAIR_WORLDS.values():
        left_prediction = module.fit_predict(
            contract,
            model_id="scalar_delta",
            world_id=left,
            replicate=0,
            train_features=train.features,
            train_labels=module.world_labels(train, left),
            eval_features=evaluation.features,
        )
        right_prediction = module.fit_predict(
            contract,
            model_id="scalar_delta",
            world_id=right,
            replicate=0,
            train_features=train.features,
            train_labels=module.world_labels(train, right),
            eval_features=evaluation.features,
        )
        assert np.array_equal(left_prediction, right_prediction)


def test_residual_mlp_starts_at_exact_frozen_logistic_output() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    cohort = module.make_cohort(
        contract,
        role="training",
        replicate=0,
        agent_count=64,
    )
    policy = module.ResidualMLP(hidden_width=16, seed=123).to(dtype=torch.float64)
    features = torch.as_tensor(cohort.features, dtype=torch.float64)
    with torch.no_grad():
        actual = torch.sigmoid(policy(features)).numpy()
    expected = module._sigmoid(module._base_logit(cohort.features))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-16)


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        module.delayed.verify_precommit("short", contract)
