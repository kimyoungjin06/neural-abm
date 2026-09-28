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
    ROOT / "experiments" / "configs" / "pivot_matched_marginal_discriminator_v3.yaml"
)
SCRIPT = ROOT / "scripts" / "prototype_pivot_matched_marginal_discriminator_v3.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("pivot_matched_marginal_v3", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cohort(module: object, contract: dict, role: str, replicate: int, agents: int):
    return module.v2.make_cohort(
        contract, role=role, replicate=replicate, agent_count=agents
    )


def test_contract_uses_fresh_seeds_and_a_matched_optimizer() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert config["purpose"]["result_informed_successor"] is True
    seeds = {
        config["cohorts"]["training"]["base_seed"],
        config["cohorts"]["evaluation"]["base_seed"],
    }
    assert len(seeds) == 2
    assert not seeds & {20260808, 20260809}
    models = {row["id"]: row for row in config["model_ladder"]}
    linear, mlp = models["matched_linear"], models["residual_mlp"]
    for key in ("optimizer", "epochs", "learning_rate", "candidates"):
        assert linear[key] == mlp[key]
    assert len(models["histogram_tree"]["candidates"]) == 12
    assert config["hyperparameter_selection"]["data"] == "TRAINING_COHORT_ONLY"
    assert config["boundaries"]["empirical_rows_read"] == 0


def test_source_bindings_and_plan_validate_without_model_execution() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    plan = module._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert plan["world_ids"] == list(module.WORLD_IDS)
    assert len(plan["source_bindings"]) == 6


def test_reused_v2_seed_namespace_is_rejected() -> None:
    module = _module()
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    contract["cohorts"]["training"]["base_seed"] = 20260808
    with pytest.raises(ValueError, match="opened v2 seed"):
        module.validate_contract(contract)


def test_worlds_share_features_and_exact_success_fraction() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    cohort = _cohort(module, contract, "training", 0, 256)
    labels = {world: module.world_labels(cohort, world) for world in module.WORLD_IDS}
    assert all(float(values.mean()) == 0.5 for values in labels.values())
    for left, right in module.PAIR_WORLDS.values():
        assert np.array_equal(labels[left], 1.0 - labels[right])
    assert len(np.unique(module._axis_score(cohort.features))) == 256


def test_axis_score_matches_the_registered_corner_formula() -> None:
    module = _module()
    first = np.array([0.3, 0.3, -0.3, 0.1, -0.2])
    second = np.array([0.2, -0.2, -0.4, 0.4, 0.05])
    assert np.allclose(module._corner(first, second), [0.2, -0.2, 0.3, 0.1, -0.05])
    features = np.full((1, len(module.v2.study2.FEATURES)), 0.5)
    index = {name: position for position, name in enumerate(module.v2.study2.FEATURES)}
    features[0, index["resource_security"]] = 0.9
    features[0, index["network_support"]] = 0.7
    features[0, index["skill_distance"]] = 0.8
    features[0, index["openness"]] = 0.1
    # max(corner(0.4, 0.2), -corner(0.3, -0.4)) = max(0.2, 0.3)
    assert module._axis_score(features)[0] == pytest.approx(0.3)


def test_inner_folds_are_balanced_and_orientation_invariant() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    cohort = _cohort(module, contract, "training", 3, 256)
    for left, right in module.PAIR_WORLDS.values():
        labels = module.world_labels(cohort, left)
        folds = module.inner_folds(labels, cohort.agent_ids, 4)
        reversed_folds = module.inner_folds(
            module.world_labels(cohort, right), cohort.agent_ids, 4
        )
        assert np.array_equal(folds, reversed_folds)
        for fold in range(4):
            assert np.sum((folds == fold) & (labels == 1.0)) == 32
            assert np.sum((folds == fold) & (labels == 0.0)) == 32


@pytest.mark.parametrize("profile", ["smoke", "development"])
def test_scalar_delta_is_exactly_orientation_invariant_in_every_replicate(
    profile: str,
) -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    settings = module.v2._profile(contract, profile)
    for replicate in range(settings["replicate_count"]):
        train = _cohort(
            module, contract, "training", replicate, settings["training_agent_count"]
        )
        evaluation = _cohort(
            module,
            contract,
            "evaluation",
            replicate,
            settings["evaluation_agent_count"],
        )
        for left, right in module.PAIR_WORLDS.values():
            predictions = [
                module.fit_predict(
                    contract,
                    model_id="scalar_delta",
                    world_id=world,
                    replicate=replicate,
                    train=train,
                    train_labels=module.world_labels(train, world),
                    eval_features=evaluation.features,
                )[0]
                for world in (left, right)
            ]
            assert np.array_equal(predictions[0], predictions[1])


def test_matched_linear_starts_at_the_frozen_logistic_output() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    cohort = _cohort(module, contract, "training", 0, 64)
    policy = module.ResidualLinear().to(dtype=torch.float64)
    with torch.no_grad():
        actual = torch.sigmoid(
            policy(torch.as_tensor(cohort.features, dtype=torch.float64))
        ).numpy()
    expected = module.v2._sigmoid(module.v2._base_logit(cohort.features))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-16)
    assert sum(parameter.numel() for parameter in policy.parameters()) == 10


def test_mlp_training_loop_is_exactly_the_v2_loop() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    train = _cohort(module, contract, "training", 0, 64)
    evaluation = _cohort(module, contract, "evaluation", 0, 64)
    labels = module.world_labels(train, "smooth_interaction_aligned")
    model = module._model_contract(contract, "residual_mlp")
    candidate = model["candidates"][1]
    actual = module.fit_candidate(
        model, candidate, train.features, labels, evaluation.features, 99
    )
    expected = module.v2._fit_residual_mlp(
        train.features,
        labels,
        evaluation.features,
        {**model, "weight_decay": candidate["weight_decay"]},
        99,
    )
    assert np.array_equal(actual, expected)


def test_selection_takes_the_first_candidate_on_ties() -> None:
    module = _module()
    assert module.select_candidate([0.2, 0.1, 0.1, 0.3]) == 1
    assert module.select_candidate([0.1, 0.1]) == 0


def _synthetic_rows(module: object, briers: dict) -> list[dict]:
    rows = []
    for replicate in range(16):
        for world in module.WORLD_IDS:
            for model in module.MODEL_IDS:
                wobble = 0.001 * ((replicate % 4) - 1.5)
                rows.append(
                    {
                        "replicate": replicate,
                        "world_id": world,
                        "model_id": model,
                        "brier_score": briers[(world, model)] + wobble,
                    }
                )
    return rows


def _briers(module: object, **overrides: float) -> dict:
    base = {"fixed_rule": 0.26, "scalar_delta": 0.25, "matched_linear": 0.10}
    briers = {}
    for world in module.WORLD_IDS:
        interaction = world in module.INTERACTION_WORLDS
        briers[(world, "fixed_rule")] = base["fixed_rule"]
        briers[(world, "scalar_delta")] = base["scalar_delta"]
        briers[(world, "matched_linear")] = 0.24 if interaction else base["matched_linear"]
        briers[(world, "histogram_tree")] = 0.12 if interaction else 0.09
        briers[(world, "residual_mlp")] = 0.10 if interaction else 0.10
    for key, value in overrides.items():
        world, model = key.split("__")
        briers[(world, model)] = value
    return briers


def test_decision_routes_a_tree_that_wins_axis_worlds_to_paired_arms() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    briers = _briers(
        module,
        axis_interaction_aligned__histogram_tree=0.08,
        axis_interaction_reversed__histogram_tree=0.08,
    )
    decision = module.decide(_synthetic_rows(module, briers), contract)
    assert decision["feature_conditional_need"] == "SUPPORTED_ON_FROZEN_GENERATOR"
    assert decision["nonlinear_representation_need"] == "SUPPORTED_BOTH_GEOMETRIES"
    assert decision["nonlinear_family_dominance"] == "GENERATOR_DEPENDENT_OR_TIED"
    assert decision["neural_specific_advantage"] == "NOT_SUPPORTED_ON_FROZEN_GENERATOR"
    assert decision["abm_route"] == "PAIRED_CLASSICAL_AND_NEURAL_NONLINEAR_ARMS"


def test_neural_route_requires_interaction_specific_dominance() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    # MLP wins every interaction world by 0.02 and ties the tree on linear worlds.
    specific = _briers(
        module,
        linear_support_aligned__histogram_tree=0.10,
        linear_support_reversed__histogram_tree=0.10,
    )
    decision = module.decide(_synthetic_rows(module, specific), contract)
    assert decision["neural_interaction_specificity"] == "PASS"
    assert decision["abm_route"] == "NEURAL_NONLINEAR_PRIMARY_WITH_CLASSICAL_COMPARATOR"
    # The same interaction edge also present on linear worlds is not specific.
    generic = _briers(
        module,
        linear_support_aligned__histogram_tree=0.12,
        linear_support_reversed__histogram_tree=0.12,
    )
    decision = module.decide(_synthetic_rows(module, generic), contract)
    assert decision["nonlinear_family_dominance"] == "MLP_DOMINATES_ALL_FOUR_WORLDS"
    assert decision["neural_interaction_specificity"] == "FAIL"
    assert decision["abm_route"] == "PAIRED_CLASSICAL_AND_NEURAL_NONLINEAR_ARMS"


def test_linear_and_tree_routes() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    no_interaction = {
        f"{world}__{model}": 0.10
        for world in module.INTERACTION_WORLDS
        for model in ("matched_linear", "residual_mlp")
    }
    decision = module.decide(
        _synthetic_rows(module, _briers(module, **no_interaction)), contract
    )
    assert decision["nonlinear_representation_need"] == "NOT_SUPPORTED"
    assert decision["abm_route"] == "LINEAR_CONDITIONAL_POLICY"
    tree_wins = {f"{world}__histogram_tree": 0.08 for world in module.INTERACTION_WORLDS}
    decision = module.decide(
        _synthetic_rows(module, _briers(module, **tree_wins)), contract
    )
    assert decision["nonlinear_family_dominance"] == "TREE_DOMINATES_ALL_FOUR_WORLDS"
    assert decision["abm_route"] == (
        "CLASSICAL_NONLINEAR_POLICY_NEURAL_RETIRED_FROM_POLICY_LAYER"
    )


def test_smoke_pipeline_runs_without_evaluating_decisions() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    rows, summary = module.run_discriminator(contract, "smoke")
    assert len(rows) == 2 * len(module.WORLD_IDS) * len(module.MODEL_IDS)
    assert summary["scalar_orientation_invariance"] == "PASS_EXACT"
    assert summary["dispositions"]["abm_route"] == "NOT_EVALUATED"
    tuned = [row for row in rows if row["model_id"] in module.TUNED_MODEL_IDS]
    assert all(len(row["validation_briers"]) >= 3 for row in tuned)


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    module = _module()
    contract = module.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        module.delayed.verify_precommit("short", contract)
