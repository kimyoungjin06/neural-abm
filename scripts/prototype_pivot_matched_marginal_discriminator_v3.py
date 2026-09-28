#!/usr/bin/env python3
"""Run the precommitted Pivot matched-marginal policy discriminator v3."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prototype_pivot_matched_marginal_discriminator_v2 as v2  # noqa: E402


delayed = v2.delayed
DEFAULT_CONFIG = (
    ROOT / "experiments" / "configs" / "pivot_matched_marginal_discriminator_v3.yaml"
)
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
MODEL_IDS = (
    "fixed_rule",
    "scalar_delta",
    "matched_linear",
    "histogram_tree",
    "residual_mlp",
)
TUNED_MODEL_IDS = ("matched_linear", "histogram_tree", "residual_mlp")
PAIR_WORLDS = {
    "linear_orientation": ("linear_support_aligned", "linear_support_reversed"),
    "smooth_interaction_orientation": (
        "smooth_interaction_aligned",
        "smooth_interaction_reversed",
    ),
    "axis_interaction_orientation": (
        "axis_interaction_aligned",
        "axis_interaction_reversed",
    ),
}
WORLD_IDS = tuple(world for pair in PAIR_WORLDS.values() for world in pair)
LINEAR_WORLDS = PAIR_WORLDS["linear_orientation"]
SMOOTH_WORLDS = PAIR_WORLDS["smooth_interaction_orientation"]
AXIS_WORLDS = PAIR_WORLDS["axis_interaction_orientation"]
INTERACTION_WORLDS = SMOOTH_WORLDS + AXIS_WORLDS
LINEAR_REFERENCE = {
    world: LINEAR_WORLDS[0] if world.endswith("_aligned") else LINEAR_WORLDS[1]
    for world in INTERACTION_WORLDS
}
WORLD_COMPONENT_IDS = {
    "linear_support_aligned": 11,
    "linear_support_reversed": 12,
    "smooth_interaction_aligned": 21,
    "smooth_interaction_reversed": 22,
    "axis_interaction_aligned": 31,
    "axis_interaction_reversed": 32,
}
MODEL_COMPONENT_IDS = {"matched_linear": 1, "histogram_tree": 2, "residual_mlp": 3}
METRICS = v2.METRICS
IMPROVEMENT_RULE = v2.IMPROVEMENT_RULE
DID_RULE = "INTERACTION_WORLD_GAP_MINUS_LINEAR_WORLD_GAP_SAME_ORIENTATION_AND_REPLICATE"
ROUTES = (
    ("FEATURE_CONDITIONAL_NEED_NOT_SUPPORTED", "SCALAR_ADAPTATION_POLICY"),
    ("NONLINEAR_REPRESENTATION_NEED_NOT_SUPPORTED", "LINEAR_CONDITIONAL_POLICY"),
    (
        "NEURAL_SPECIFIC_ADVANTAGE_SUPPORTED",
        "NEURAL_NONLINEAR_PRIMARY_WITH_CLASSICAL_COMPARATOR",
    ),
    (
        "TREE_DOMINATES_ALL_FOUR_WORLDS",
        "CLASSICAL_NONLINEAR_POLICY_NEURAL_RETIRED_FROM_POLICY_LAYER",
    ),
    ("OTHERWISE", "PAIRED_CLASSICAL_AND_NEURAL_NONLINEAR_ARMS"),
)


class ResidualLinear(torch.nn.Module):
    """The v2 residual MLP with its tanh hidden branch removed."""

    def __init__(self) -> None:
        super().__init__()
        base_weight, base_bias = v2.study2._initial_policy()
        self.register_buffer("base_weight", base_weight.detach().clone())
        self.register_buffer("base_bias", base_bias.detach().clone())
        self.linear_delta = torch.nn.Linear(len(v2.study2.FEATURES), 1)
        torch.nn.init.zeros_(self.linear_delta.weight)
        torch.nn.init.zeros_(self.linear_delta.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base = features @ self.base_weight + self.base_bias
        return base + self.linear_delta(features).reshape(-1)


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("matched-marginal contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_matched_marginal_discriminator.v3":
        raise ValueError("unexpected matched-marginal schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("matched-marginal contract is not a precommit design")
    cohorts = contract.get("cohorts", {})
    if cohorts.get("scenario") != "hot_field_hype":
        raise ValueError("v3 must remain bound to hot_field_hype features")
    seeds = [
        cohorts.get("training", {}).get("base_seed"),
        cohorts.get("evaluation", {}).get("base_seed"),
    ]
    if seeds[0] == seeds[1]:
        raise ValueError("training and evaluation seed namespaces must be disjoint")
    if set(seeds) & set(cohorts.get("forbidden_base_seeds", [])):
        raise ValueError("v3 must not reuse opened v2 seed namespaces")
    worlds = contract.get("matched_marginal_worlds", {})
    if worlds.get("success_fraction") != 0.5:
        raise ValueError("v3 requires an exact one-half success fraction")
    configured_worlds = [
        world["id"] for pair in worlds.get("pairs", []) for world in pair["worlds"]
    ]
    if configured_worlds != list(WORLD_IDS):
        raise ValueError("matched-marginal worlds changed")
    models = [row["id"] for row in contract.get("model_ladder", [])]
    if models != list(MODEL_IDS):
        raise ValueError("matched-marginal model ladder changed")
    scalar = _model_contract(contract, "scalar_delta")
    if scalar.get("evidence_reduction") != "TRAINING_SUCCESS_COUNT_ONLY":
        raise ValueError("scalar baseline must consume only the success count")
    linear = _model_contract(contract, "matched_linear")
    mlp = _model_contract(contract, "residual_mlp")
    for key in ("optimizer", "epochs", "learning_rate", "candidates", "initialization"):
        if linear.get(key) != mlp.get(key):
            raise ValueError(f"matched linear must share the MLP {key}")
    for model_id in TUNED_MODEL_IDS:
        if not _model_contract(contract, model_id).get("candidates"):
            raise ValueError(f"{model_id} requires registered candidates")
    selection = contract.get("hyperparameter_selection", {})
    if selection.get("data") != "TRAINING_COHORT_ONLY":
        raise ValueError("selection must use the training cohort only")
    if selection.get("evaluation_cohort_used_for_selection") is not False:
        raise ValueError("the evaluation cohort must not inform selection")
    decisions = contract.get("decision_contract", {})
    if decisions.get("improvement_rule") != IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("difference_in_differences_rule") != DID_RULE:
        raise ValueError("difference-in-differences rule changed")
    if decisions.get("scalar_invariance_check", {}).get("required") is not True:
        raise ValueError("scalar orientation invariance must remain required")
    routes = [
        (row["condition"], row["route"])
        for row in decisions.get("abm_routing", {}).get("evaluated_in_order", [])
    ]
    if routes != list(ROUTES):
        raise ValueError("ABM routing changed")
    boundaries = contract.get("boundaries", {})
    if boundaries.get("empirical_rows_read") != 0:
        raise ValueError("empirical rows are outside this discriminator")
    if boundaries.get("generic_nabm_api_changed") is not False:
        raise ValueError("generic NABM changes are outside this discriminator")
    if boundaries.get("full_agent_dynamics_run") is not False:
        raise ValueError("full agent dynamics are outside this discriminator")


def _model_contract(contract: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    return next(row for row in contract["model_ladder"] if row["id"] == model_id)


def _corner(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.sign(first) * np.sign(second) * np.minimum(np.abs(first), np.abs(second))


def _axis_score(features: np.ndarray) -> np.ndarray:
    index = {name: position for position, name in enumerate(v2.study2.FEATURES)}
    resource = features[:, index["resource_security"]] - 0.5
    network = features[:, index["network_support"]] - 0.5
    skill = features[:, index["skill_distance"]] - 0.5
    openness = features[:, index["openness"]] - 0.5
    return np.maximum(_corner(resource, network), -_corner(skill, openness))


def world_labels(cohort: v2.Cohort, world_id: str) -> np.ndarray:
    if world_id.endswith("_reversed"):
        aligned_world = world_id.removesuffix("_reversed") + "_aligned"
        return 1.0 - world_labels(cohort, aligned_world)
    if world_id.startswith("linear_"):
        score = v2._linear_score(cohort.features)
    elif world_id.startswith("smooth_interaction_"):
        score = v2._interaction_score(cohort.features)
    elif world_id.startswith("axis_interaction_"):
        score = _axis_score(cohort.features)
    else:
        raise ValueError(f"unknown world: {world_id}")
    if cohort.features.shape[0] % 2:
        raise ValueError("exact matched marginal requires an even cohort size")
    # np.lexsort uses the final key as primary: descending score, then agent id.
    order = np.lexsort((cohort.agent_ids, -score))
    labels = np.zeros(cohort.features.shape[0], dtype=np.float64)
    labels[order[: cohort.features.shape[0] // 2]] = 1.0
    return labels


def inner_folds(labels: np.ndarray, agent_ids: np.ndarray, fold_count: int) -> np.ndarray:
    folds = np.empty(labels.shape[0], dtype=np.int64)
    for value in (0.0, 1.0):
        members = np.flatnonzero(labels == value)
        members = members[np.argsort(agent_ids[members], kind="stable")]
        folds[members] = np.arange(members.size) % fold_count
    return folds


def _seed(
    contract: Mapping[str, Any],
    *,
    replicate: int,
    world_id: str,
    model_id: str,
    fold_component: int,
) -> int:
    return int(
        np.random.SeedSequence(
            (
                int(contract["cohorts"]["training"]["base_seed"]),
                replicate,
                WORLD_COMPONENT_IDS[world_id],
                MODEL_COMPONENT_IDS[model_id],
                fold_component,
            )
        ).generate_state(1)[0]
    )


def _train_residual_policy(
    policy: torch.nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
) -> np.ndarray:
    # Same loop as v2._fit_residual_mlp, shared by the MLP and matched linear.
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    inputs = torch.as_tensor(train_features, dtype=torch.float64)
    targets = torch.as_tensor(train_labels, dtype=torch.float64)
    policy = policy.to(dtype=torch.float64)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            policy(inputs), targets
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        evaluation = torch.as_tensor(eval_features, dtype=torch.float64)
        return torch.sigmoid(policy(evaluation)).detach().numpy()


def fit_candidate(
    model: Mapping[str, Any],
    candidate: Mapping[str, Any],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    seed: int,
) -> np.ndarray:
    model_id = model["id"]
    if model_id == "histogram_tree":
        settings = {**candidate, "l2_regularization": model["l2_regularization"]}
        return v2._fit_histogram_tree(
            train_features, train_labels, eval_features, settings, seed
        )
    if model_id == "residual_mlp":
        policy: torch.nn.Module = v2.ResidualMLP(int(model["hidden_width"]), seed)
    elif model_id == "matched_linear":
        policy = ResidualLinear()
    else:
        raise ValueError(f"not a tuned model: {model_id}")
    return _train_residual_policy(
        policy,
        train_features,
        train_labels,
        eval_features,
        learning_rate=float(model["learning_rate"]),
        epochs=int(model["epochs"]),
        weight_decay=float(candidate["weight_decay"]),
    )


def select_candidate(validation_briers: Sequence[float]) -> int:
    # np.argmin returns the first minimum, the registered tie-break.
    return int(np.argmin(np.asarray(validation_briers, dtype=np.float64)))


def fit_predict(
    contract: Mapping[str, Any],
    *,
    model_id: str,
    world_id: str,
    replicate: int,
    train: v2.Cohort,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = _model_contract(contract, model_id)
    if model_id == "fixed_rule":
        return v2._sigmoid(v2._base_logit(eval_features)), {}
    if model_id == "scalar_delta":
        return (
            v2._fit_scalar_delta(train.features, train_labels, eval_features, model),
            {},
        )
    candidates = model["candidates"]
    fold_count = int(contract["hyperparameter_selection"]["inner_folds"])
    folds = inner_folds(train_labels, train.agent_ids, fold_count)
    fold_briers = np.zeros((len(candidates), fold_count), dtype=np.float64)
    for fold in range(fold_count):
        fit_rows = folds != fold
        seed = _seed(
            contract,
            replicate=replicate,
            world_id=world_id,
            model_id=model_id,
            fold_component=fold + 1,
        )
        for position, candidate in enumerate(candidates):
            prediction = fit_candidate(
                model,
                candidate,
                train.features[fit_rows],
                train_labels[fit_rows],
                train.features[~fit_rows],
                seed,
            )
            fold_briers[position, fold] = float(
                np.mean((prediction - train_labels[~fit_rows]) ** 2)
            )
    validation_briers = fold_briers.mean(axis=1)
    selected = select_candidate(validation_briers)
    seed = _seed(
        contract,
        replicate=replicate,
        world_id=world_id,
        model_id=model_id,
        fold_component=0,
    )
    prediction = fit_candidate(
        model, candidates[selected], train.features, train_labels, eval_features, seed
    )
    return prediction, {
        "selected_candidate": selected,
        "validation_briers": [round(float(value), 10) for value in validation_briers],
    }


def run_replicate(
    contract: Mapping[str, Any], profile: str, replicate: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    settings = v2._profile(contract, profile)
    train = v2.make_cohort(
        contract,
        role="training",
        replicate=replicate,
        agent_count=settings["training_agent_count"],
    )
    evaluation = v2.make_cohort(
        contract,
        role="evaluation",
        replicate=replicate,
        agent_count=settings["evaluation_agent_count"],
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for world_id in WORLD_IDS:
        train_labels = world_labels(train, world_id)
        eval_labels = world_labels(evaluation, world_id)
        if train_labels.mean() != 0.5 or eval_labels.mean() != 0.5:
            raise RuntimeError("matched marginal construction failed")
        for model_id in MODEL_IDS:
            prediction, selection = fit_predict(
                contract,
                model_id=model_id,
                world_id=world_id,
                replicate=replicate,
                train=train,
                train_labels=train_labels,
                eval_features=evaluation.features,
            )
            predictions[(world_id, model_id)] = prediction
            rows.append(
                {
                    "replicate": replicate,
                    "world_id": world_id,
                    "model_id": model_id,
                    "training_feature_sha256": v2._array_sha256(train.features),
                    "evaluation_feature_sha256": v2._array_sha256(evaluation.features),
                    "training_label_sha256": v2._array_sha256(train_labels),
                    "evaluation_label_sha256": v2._array_sha256(eval_labels),
                    "prediction_sha256": v2._array_sha256(prediction),
                    "training_success_fraction": float(train_labels.mean()),
                    "evaluation_success_fraction": float(eval_labels.mean()),
                    **selection,
                    **v2.predictive_metrics(eval_labels, prediction),
                }
            )
    for pair_id, (left, right) in PAIR_WORLDS.items():
        if not np.array_equal(
            predictions[(left, "scalar_delta")],
            predictions[(right, "scalar_delta")],
        ):
            raise RuntimeError(f"scalar orientation invariance failed for {pair_id}")
    return rows, predictions


def _gaps(
    indexed: Mapping[tuple[int, str, str], Mapping[str, Any]],
    replicates: Sequence[int],
    *,
    world_id: str,
    model_id: str,
    reference_model_id: str,
) -> np.ndarray:
    return np.asarray(
        [
            float(indexed[(replicate, world_id, reference_model_id)]["brier_score"])
            - float(indexed[(replicate, world_id, model_id)]["brier_score"])
            for replicate in replicates
        ],
        dtype=np.float64,
    )


def _judge(values: np.ndarray, minimum_mean: float, minimum_positive: int) -> dict[str, Any]:
    distribution = v2._distribution(values)
    positive = int(np.sum(values > 0.0))
    return {
        "brier_improvement": distribution,
        "positive_replicates": positive,
        "pass": distribution["mean"] >= minimum_mean and positive >= minimum_positive,
    }


def _comparison(
    indexed: Mapping[tuple[int, str, str], Mapping[str, Any]],
    replicates: Sequence[int],
    rule: Mapping[str, Any],
    *,
    model_id: str,
    reference_model_id: str,
    difference_in_differences: bool,
) -> dict[str, Any]:
    minimum_mean = float(rule["brier_improvement_at_least"])
    minimum_positive = int(rule["improved_replicates_at_least"])
    worlds: dict[str, Any] = {}
    for world_id in rule["worlds"]:
        values = _gaps(
            indexed,
            replicates,
            world_id=world_id,
            model_id=model_id,
            reference_model_id=reference_model_id,
        )
        if difference_in_differences:
            values = values - _gaps(
                indexed,
                replicates,
                world_id=LINEAR_REFERENCE[world_id],
                model_id=model_id,
                reference_model_id=reference_model_id,
            )
        worlds[world_id] = _judge(values, minimum_mean, minimum_positive)
    return {
        "model_id": model_id,
        "reference_model_id": reference_model_id,
        "statistic": "DIFFERENCE_IN_DIFFERENCES" if difference_in_differences else "RAW_GAP",
        "minimum_mean_brier_improvement": minimum_mean,
        "minimum_positive_replicates": minimum_positive,
        "worlds": worlds,
        "pass": all(report["pass"] for report in worlds.values()),
    }


def route(dispositions: Mapping[str, Any]) -> str:
    if dispositions["feature_conditional_need"] != "SUPPORTED_ON_FROZEN_GENERATOR":
        return ROUTES[0][1]
    if dispositions["nonlinear_representation_need"] == "NOT_SUPPORTED":
        return ROUTES[1][1]
    if dispositions["neural_specific_advantage"] == "SUPPORTED_ON_FROZEN_GENERATOR_ONLY":
        return ROUTES[2][1]
    if dispositions["nonlinear_family_dominance"] == "TREE_DOMINATES_ALL_FOUR_WORLDS":
        return ROUTES[3][1]
    return ROUTES[4][1]


def decide(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    indexed = {
        (int(row["replicate"]), str(row["world_id"]), str(row["model_id"])): row
        for row in rows
    }
    replicates = sorted({int(row["replicate"]) for row in rows})
    decision = contract["decision_contract"]
    feature = _comparison(
        indexed,
        replicates,
        decision["feature_conditional_need"],
        model_id="matched_linear",
        reference_model_id="scalar_delta",
        difference_in_differences=False,
    )
    nonlinear = _comparison(
        indexed,
        replicates,
        decision["nonlinear_representation_need"],
        model_id="residual_mlp",
        reference_model_id="matched_linear",
        difference_in_differences=True,
    )
    smooth_pass = all(nonlinear["worlds"][world]["pass"] for world in SMOOTH_WORLDS)
    axis_pass = all(nonlinear["worlds"][world]["pass"] for world in AXIS_WORLDS)
    nonlinear_disposition = {
        (True, True): "SUPPORTED_BOTH_GEOMETRIES",
        (True, False): "SUPPORTED_SMOOTH_ONLY",
        (False, True): "SUPPORTED_AXIS_ONLY",
        (False, False): "NOT_SUPPORTED",
    }[(smooth_pass, axis_pass)]
    dominance_rule = decision["nonlinear_family_dominance"]
    mlp_over_tree = _comparison(
        indexed,
        replicates,
        dominance_rule,
        model_id="residual_mlp",
        reference_model_id="histogram_tree",
        difference_in_differences=False,
    )
    tree_over_mlp = _comparison(
        indexed,
        replicates,
        dominance_rule,
        model_id="histogram_tree",
        reference_model_id="residual_mlp",
        difference_in_differences=False,
    )
    if mlp_over_tree["pass"]:
        dominance = "MLP_DOMINATES_ALL_FOUR_WORLDS"
    elif tree_over_mlp["pass"]:
        dominance = "TREE_DOMINATES_ALL_FOUR_WORLDS"
    else:
        dominance = "GENERATOR_DEPENDENT_OR_TIED"
    specificity = _comparison(
        indexed,
        replicates,
        decision["neural_interaction_specificity"],
        model_id="residual_mlp",
        reference_model_id="histogram_tree",
        difference_in_differences=True,
    )
    neural_specific = dominance == "MLP_DOMINATES_ALL_FOUR_WORLDS" and specificity["pass"]
    dispositions: dict[str, Any] = {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "feature_conditional_need": (
            "SUPPORTED_ON_FROZEN_GENERATOR"
            if feature["pass"]
            else "NOT_SUPPORTED_ON_FROZEN_GENERATOR"
        ),
        "nonlinear_representation_need": nonlinear_disposition,
        "nonlinear_family_dominance": dominance,
        "neural_interaction_specificity": "PASS" if specificity["pass"] else "FAIL",
        "neural_specific_advantage": (
            "SUPPORTED_ON_FROZEN_GENERATOR_ONLY"
            if neural_specific
            else "NOT_SUPPORTED_ON_FROZEN_GENERATOR"
        ),
    }
    dispositions["abm_route"] = route(dispositions)
    dispositions["reports"] = {
        "feature_conditional_need": feature,
        "nonlinear_representation_need": nonlinear,
        "mlp_over_tree": mlp_over_tree,
        "tree_over_mlp": tree_over_mlp,
        "neural_interaction_specificity": specificity,
    }
    return dispositions


def summarize(
    rows: Sequence[Mapping[str, Any]],
    prediction_sets: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    contract: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["world_id"]), str(row["model_id"]))].append(row)
    cells = []
    for world_id in WORLD_IDS:
        for model_id in MODEL_IDS:
            cell_rows = grouped[(world_id, model_id)]
            cell: dict[str, Any] = {
                "world_id": world_id,
                "model_id": model_id,
                **{
                    metric: v2._distribution([float(row[metric]) for row in cell_rows])
                    for metric in METRICS
                },
            }
            if model_id in TUNED_MODEL_IDS:
                counts = Counter(int(row["selected_candidate"]) for row in cell_rows)
                cell["selected_candidate_counts"] = {
                    str(position): counts[position] for position in sorted(counts)
                }
            cells.append(cell)
    orientation_distances = []
    for pair_id, (left_world, right_world) in PAIR_WORLDS.items():
        for model_id in MODEL_IDS:
            distances = []
            for _replicate, predictions in sorted(prediction_sets.items()):
                left = predictions[(left_world, model_id)]
                right = predictions[(right_world, model_id)]
                distances.append(float(np.mean(np.abs(left - right))))
                if model_id == "scalar_delta" and not np.array_equal(left, right):
                    raise RuntimeError("scalar invariance failed during summary")
            orientation_distances.append(
                {
                    "pair_id": pair_id,
                    "model_id": model_id,
                    "mean_absolute_prediction_distance": v2._distribution(distances),
                }
            )
    settings = v2._profile(contract, profile)
    minimum_positive = int(
        contract["decision_contract"]["feature_conditional_need"][
            "improved_replicates_at_least"
        ]
    )
    if settings["replicate_count"] < minimum_positive:
        dispositions: dict[str, Any] = {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
        for key in (
            "feature_conditional_need",
            "nonlinear_representation_need",
            "nonlinear_family_dominance",
            "neural_interaction_specificity",
            "neural_specific_advantage",
            "abm_route",
        ):
            dispositions[key] = "NOT_EVALUATED"
    else:
        dispositions = decide(rows, contract)
    return {
        "schema_version": "pivot_matched_marginal_scorecard.v3",
        "status": "SYNTHETIC_POLICY_LAYER_DISCRIMINATOR",
        "profile": profile,
        "cells": cells,
        "orientation_prediction_distances": orientation_distances,
        "scalar_orientation_invariance": "PASS_EXACT",
        "dispositions": dispositions,
        "boundaries": contract["boundaries"],
    }


def run_discriminator(
    contract: Mapping[str, Any], profile: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = v2._profile(contract, profile)
    rows: list[dict[str, Any]] = []
    prediction_sets: dict[int, Mapping[tuple[str, str], np.ndarray]] = {}
    for replicate in range(settings["replicate_count"]):
        replicate_rows, predictions = run_replicate(contract, profile, replicate)
        rows.extend(replicate_rows)
        prediction_sets[replicate] = predictions
    return rows, summarize(rows, prediction_sets, contract, profile)


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot matched-marginal policy scorecard v3",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| World | Model | Brier | Log loss | AUC | Separation |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        lines.append(
            "| {world_id} | {model_id} | {brier:.6f} | {loss:.6f} | "
            "{auc:.6f} | {separation:.6f} |".format(
                world_id=cell["world_id"],
                model_id=cell["model_id"],
                brier=cell["brier_score"]["mean"],
                loss=cell["log_loss"]["mean"],
                auc=cell["roc_auc"]["mean"],
                separation=cell["conditional_separation"]["mean"],
            )
        )
    dispositions = summary["dispositions"]
    lines.extend(
        [
            "",
            "## Frozen dispositions",
            "",
            f"- Feature-conditional need: `{dispositions['feature_conditional_need']}`",
            f"- Nonlinear representation need: `{dispositions['nonlinear_representation_need']}`",
            f"- Nonlinear family dominance: `{dispositions['nonlinear_family_dominance']}`",
            f"- Neural interaction specificity: `{dispositions['neural_interaction_specificity']}`",
            f"- Neural-specific advantage: `{dispositions['neural_specific_advantage']}`",
            f"- ABM route: `{dispositions['abm_route']}`",
            "",
            "This is a synthetic policy-layer result. It does not select an empirical "
            "model, establish a scientific effect, or authorize package promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "replicate_metrics.jsonl"
    scorecard_path = output_dir / "policy_scorecard.json"
    markdown_path = output_dir / "policy_scorecard.md"
    receipt_path = output_dir / "receipt.json"
    delayed._write_exclusive(
        rows_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    delayed._write_exclusive(
        scorecard_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    delayed._write_exclusive(markdown_path, _markdown(summary))
    receipt = {
        "schema_version": "pivot_matched_marginal_receipt.v3",
        "status": "COMPLETE",
        "row_count": len(rows),
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": v2._sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {
            path.name: {"sha256": v2._sha256(path), "bytes": path.stat().st_size}
            for path in (rows_path, scorecard_path, markdown_path)
        },
        "boundaries": summary["boundaries"],
    }
    delayed._write_exclusive(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_path": str(config_path.resolve()),
        "contract_sha256": v2._sha256(config_path),
        "model_ids": list(MODEL_IDS),
        "world_ids": list(WORLD_IDS),
        "profiles": contract["profiles"],
        "source_bindings": delayed.verify_source_bindings(contract),
        "decision_contract": contract["decision_contract"],
        "boundaries": contract["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile", choices=("smoke", "development"), default="development"
    )
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        print(json.dumps(_plan_payload(contract, config_path), indent=2, sort_keys=True))
        return
    if args.precommit_sha is None:
        parser.error("run requires --precommit-sha")
    if args.output_dir is None:
        parser.error("run requires --output-dir")
    if FULL_SHA_PATTERN.fullmatch(args.precommit_sha) is None:
        parser.error("run requires a full 40-character lowercase Git SHA")
    precommit = delayed.verify_precommit(args.precommit_sha, contract)
    rows, summary = run_discriminator(contract, args.profile)
    summary = {
        **summary,
        "precommit_sha": args.precommit_sha,
        "contract_sha256": v2._sha256(config_path),
    }
    receipt = write_outputs(
        output_dir=args.output_dir.resolve(),
        rows=rows,
        summary=summary,
        precommit=precommit,
        contract_path=config_path,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "row_count": receipt["row_count"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
