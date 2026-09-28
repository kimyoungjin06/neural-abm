#!/usr/bin/env python3
"""Run the precommitted Pivot matched-marginal policy discriminator v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SCRIPTS = ROOT / "scripts"
for search_path in (EXAMPLES, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import research_pivot_learning_study as study2  # noqa: E402
import research_pivot_study as study1  # noqa: E402
import prototype_pivot_delayed_feedback_learning_v1 as delayed  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "experiments" / "configs" / "pivot_matched_marginal_discriminator_v2.yaml"
)
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
MODEL_IDS = (
    "fixed_rule",
    "scalar_delta",
    "explicit_linear_logit",
    "histogram_tree",
    "residual_mlp",
)
WORLD_IDS = (
    "linear_support_aligned",
    "linear_support_reversed",
    "interaction_aligned",
    "interaction_reversed",
)
PAIR_WORLDS = {
    "linear_orientation": (
        "linear_support_aligned",
        "linear_support_reversed",
    ),
    "interaction_orientation": (
        "interaction_aligned",
        "interaction_reversed",
    ),
}
WORLD_COMPONENT_IDS = {
    "linear_support_aligned": 11,
    "linear_support_reversed": 12,
    "interaction_aligned": 21,
    "interaction_reversed": 22,
}
METRICS = ("brier_score", "log_loss", "roc_auc", "conditional_separation")
IMPROVEMENT_RULE = (
    "MEAN_PAIRED_BRIER_IMPROVEMENT_AT_LEAST_THRESHOLD_AND_STRICTLY_POSITIVE_"
    "IN_AT_LEAST_REQUIRED_REPLICATES"
)


@dataclass(frozen=True)
class Cohort:
    features: np.ndarray
    agent_ids: np.ndarray


class ResidualMLP(torch.nn.Module):
    """Frozen Pivot logit plus zero-started linear and nonlinear residuals."""

    def __init__(self, hidden_width: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        base_weight, base_bias = study2._initial_policy()
        self.register_buffer("base_weight", base_weight.detach().clone())
        self.register_buffer("base_bias", base_bias.detach().clone())
        self.linear_delta = torch.nn.Linear(len(study2.FEATURES), 1)
        self.hidden = torch.nn.Linear(len(study2.FEATURES), hidden_width)
        self.output = torch.nn.Linear(hidden_width, 1)
        torch.nn.init.zeros_(self.linear_delta.weight)
        torch.nn.init.zeros_(self.linear_delta.bias)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base = features @ self.base_weight + self.base_bias
        linear = self.linear_delta(features).reshape(-1)
        nonlinear = self.output(torch.tanh(self.hidden(features))).reshape(-1)
        return base + linear + nonlinear


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("matched-marginal contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_matched_marginal_discriminator.v2":
        raise ValueError("unexpected matched-marginal schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("matched-marginal contract is not a precommit design")
    cohorts = contract.get("cohorts", {})
    if cohorts.get("scenario") != "hot_field_hype":
        raise ValueError("v2 must remain bound to hot_field_hype features")
    if cohorts.get("training", {}).get("base_seed") == cohorts.get(
        "evaluation", {}
    ).get("base_seed"):
        raise ValueError("training and evaluation seed namespaces must be disjoint")
    worlds = contract.get("matched_marginal_worlds", {})
    if worlds.get("success_fraction") != 0.5:
        raise ValueError("v2 requires an exact one-half success fraction")
    configured_worlds = [
        world["id"]
        for pair in worlds.get("pairs", [])
        for world in pair.get("worlds", [])
    ]
    if configured_worlds != list(WORLD_IDS):
        raise ValueError("matched-marginal worlds changed")
    models = [row["id"] for row in contract.get("model_ladder", [])]
    if models != list(MODEL_IDS):
        raise ValueError("matched-marginal model ladder changed")
    scalar = _model_contract(contract, "scalar_delta")
    if scalar.get("adaptive_state") != "ONE_GLOBAL_LOGIT_OFFSET":
        raise ValueError("scalar baseline must remain one-state")
    if scalar.get("evidence_reduction") != "TRAINING_SUCCESS_COUNT_ONLY":
        raise ValueError("scalar baseline must consume only the success count")
    linear = _model_contract(contract, "explicit_linear_logit")
    if linear.get("backpropagation") is not False:
        raise ValueError("explicit linear baseline must remain non-backpropagated")
    decisions = contract.get("decision_contract", {})
    if decisions.get("scalar_invariance_check", {}).get("required") is not True:
        raise ValueError("scalar orientation invariance must remain required")
    if decisions.get("improvement_rule") != IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    boundaries = contract.get("boundaries", {})
    if boundaries.get("empirical_rows_read") != 0:
        raise ValueError("empirical rows are outside this discriminator")
    if boundaries.get("generic_nabm_api_changed") is not False:
        raise ValueError("generic NABM changes are outside this discriminator")
    if boundaries.get("full_agent_dynamics_run") is not False:
        raise ValueError("full agent dynamics are outside this discriminator")


def _model_contract(contract: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    return next(row for row in contract["model_ladder"] if row["id"] == model_id)


def _profile(contract: Mapping[str, Any], profile: str) -> dict[str, int]:
    try:
        values = contract["profiles"][profile]
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile}") from exc
    return {
        "replicate_count": int(values["replicate_count"]),
        "training_agent_count": int(values["training_agent_count"]),
        "evaluation_agent_count": int(values["evaluation_agent_count"]),
    }


def _scenario(contract: Mapping[str, Any]) -> Any:
    name = contract["cohorts"]["scenario"]
    return next(row for row in study1.study_scenarios() if row.name == name)


def make_cohort(
    contract: Mapping[str, Any],
    *,
    role: str,
    replicate: int,
    agent_count: int,
) -> Cohort:
    if role not in {"training", "evaluation"}:
        raise ValueError(f"unknown cohort role: {role}")
    base_seed = int(contract["cohorts"][role]["base_seed"])
    config = study1.StudyConfig(
        agent_count=agent_count,
        steps=1,
        replicates=1,
        base_seed=base_seed,
    )
    scenario = _scenario(contract)
    context = SimpleNamespace(
        scenario=scenario,
        replicate=replicate,
        rng=study1.component_rng(base_seed, replicate, "population"),
    )
    researchers = study1.make_researchers(context, config)
    features = np.stack([study2._feature_vector(row) for row in researchers])
    agent_ids = np.asarray([row.agent_id for row in researchers], dtype=np.int64)
    return Cohort(features=features, agent_ids=agent_ids)


def _linear_score(features: np.ndarray) -> np.ndarray:
    index = {name: position for position, name in enumerate(study2.FEATURES)}
    return (
        0.20 * features[:, index["field_opportunity"]]
        + 0.45 * features[:, index["resource_security"]]
        + 0.35 * features[:, index["network_support"]]
        - 0.45 * features[:, index["skill_distance"]]
        - 0.25 * features[:, index["reputation_risk"]]
        + 0.15 * features[:, index["openness"]]
    )


def _interaction_score(features: np.ndarray) -> np.ndarray:
    index = {name: position for position, name in enumerate(study2.FEATURES)}
    resource = features[:, index["resource_security"]] - 0.5
    network = features[:, index["network_support"]] - 0.5
    skill = features[:, index["skill_distance"]] - 0.5
    openness = features[:, index["openness"]] - 0.5
    return resource * network - skill * openness


def world_labels(cohort: Cohort, world_id: str) -> np.ndarray:
    if world_id.endswith("_reversed"):
        aligned_world = world_id.removesuffix("_reversed") + "_aligned"
        return 1.0 - world_labels(cohort, aligned_world)
    if world_id.startswith("linear_"):
        score = _linear_score(cohort.features)
    elif world_id.startswith("interaction_"):
        score = _interaction_score(cohort.features)
    else:
        raise ValueError(f"unknown world: {world_id}")
    if cohort.features.shape[0] % 2:
        raise ValueError("exact matched marginal requires an even cohort size")
    # np.lexsort uses the final key as primary: descending score, then agent id.
    order = np.lexsort((cohort.agent_ids, -score))
    labels = np.zeros(cohort.features.shape[0], dtype=np.float64)
    labels[order[: cohort.features.shape[0] // 2]] = 1.0
    return labels


def _base_logit(features: np.ndarray) -> np.ndarray:
    weight, bias = study2._initial_policy()
    return features @ weight.detach().numpy() + float(bias.detach())


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_scalar_delta(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    model: Mapping[str, Any],
) -> np.ndarray:
    # Reduce labels to their count first. Averaging the elementwise residual
    # rounds differently for label vectors with equal counts (the v1 stop).
    success_fraction = float(np.count_nonzero(train_labels)) / train_labels.shape[0]
    offset = 0.0
    train_base = _base_logit(train_features)
    for _epoch in range(int(model["epochs"])):
        residual = success_fraction - float(np.mean(_sigmoid(train_base + offset)))
        offset += float(model["eta"]) * residual - float(model["prior_anchor"]) * offset
    return _sigmoid(_base_logit(eval_features) + offset)


def _fit_explicit_linear(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    model: Mapping[str, Any],
) -> np.ndarray:
    delta_weight = np.zeros(train_features.shape[1], dtype=np.float64)
    delta_bias = 0.0
    train_base = _base_logit(train_features)
    learning_rate = float(model["learning_rate"])
    anchor = float(model["prior_anchor"])
    for _epoch in range(int(model["epochs"])):
        prediction = _sigmoid(train_base + train_features @ delta_weight + delta_bias)
        residual = prediction - train_labels
        gradient_weight = train_features.T @ residual / train_features.shape[0]
        gradient_weight += anchor * delta_weight
        gradient_bias = float(np.mean(residual)) + anchor * delta_bias
        delta_weight -= learning_rate * gradient_weight
        delta_bias -= learning_rate * gradient_bias
    return _sigmoid(_base_logit(eval_features) + eval_features @ delta_weight + delta_bias)


def _fit_histogram_tree(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    model: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    estimator = HistGradientBoostingClassifier(
        learning_rate=float(model["learning_rate"]),
        max_iter=int(model["max_iter"]),
        max_depth=int(model["max_depth"]),
        min_samples_leaf=int(model["min_samples_leaf"]),
        l2_regularization=float(model["l2_regularization"]),
        random_state=seed,
    )
    estimator.fit(train_features, train_labels.astype(np.int64))
    return estimator.predict_proba(eval_features)[:, 1].astype(np.float64)


def _fit_residual_mlp(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    model: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    policy = ResidualMLP(int(model["hidden_width"]), seed)
    inputs = torch.as_tensor(train_features, dtype=torch.float64)
    targets = torch.as_tensor(train_labels, dtype=torch.float64)
    policy = policy.to(dtype=torch.float64)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=float(model["learning_rate"]),
        weight_decay=float(model["weight_decay"]),
    )
    for _epoch in range(int(model["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            policy(inputs), targets
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        evaluation = torch.as_tensor(eval_features, dtype=torch.float64)
        return torch.sigmoid(policy(evaluation)).detach().numpy()


def fit_predict(
    contract: Mapping[str, Any],
    *,
    model_id: str,
    world_id: str,
    replicate: int,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
) -> np.ndarray:
    model = _model_contract(contract, model_id)
    if model_id == "fixed_rule":
        return _sigmoid(_base_logit(eval_features))
    if model_id == "scalar_delta":
        return _fit_scalar_delta(train_features, train_labels, eval_features, model)
    if model_id == "explicit_linear_logit":
        return _fit_explicit_linear(train_features, train_labels, eval_features, model)
    seed = int(
        np.random.SeedSequence(
            (
                int(contract["cohorts"]["training"]["base_seed"]),
                replicate,
                WORLD_COMPONENT_IDS[world_id],
            )
        ).generate_state(1)[0]
    )
    if model_id == "histogram_tree":
        return _fit_histogram_tree(
            train_features, train_labels, eval_features, model, seed
        )
    if model_id == "residual_mlp":
        return _fit_residual_mlp(
            train_features, train_labels, eval_features, model, seed
        )
    raise ValueError(f"unknown model: {model_id}")


def predictive_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    clipped = np.clip(prediction, 1e-12, 1.0 - 1e-12)
    return {
        "brier_score": float(np.mean((prediction - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        ),
        "roc_auc": float(roc_auc_score(labels, prediction)),
        "conditional_separation": float(
            prediction[labels == 1.0].mean() - prediction[labels == 0.0].mean()
        ),
    }


def run_replicate(
    contract: Mapping[str, Any], profile: str, replicate: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    settings = _profile(contract, profile)
    train = make_cohort(
        contract,
        role="training",
        replicate=replicate,
        agent_count=settings["training_agent_count"],
    )
    evaluation = make_cohort(
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
            prediction = fit_predict(
                contract,
                model_id=model_id,
                world_id=world_id,
                replicate=replicate,
                train_features=train.features,
                train_labels=train_labels,
                eval_features=evaluation.features,
            )
            predictions[(world_id, model_id)] = prediction
            rows.append(
                {
                    "replicate": replicate,
                    "world_id": world_id,
                    "model_id": model_id,
                    "training_feature_sha256": _array_sha256(train.features),
                    "evaluation_feature_sha256": _array_sha256(evaluation.features),
                    "training_label_sha256": _array_sha256(train_labels),
                    "evaluation_label_sha256": _array_sha256(eval_labels),
                    "prediction_sha256": _array_sha256(prediction),
                    "training_success_fraction": float(train_labels.mean()),
                    "evaluation_success_fraction": float(eval_labels.mean()),
                    **predictive_metrics(eval_labels, prediction),
                }
            )
    for pair_id, worlds in PAIR_WORLDS.items():
        left, right = worlds
        if not np.array_equal(
            predictions[(left, "scalar_delta")],
            predictions[(right, "scalar_delta")],
        ):
            raise RuntimeError(f"scalar orientation invariance failed for {pair_id}")
    return rows, predictions


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 8),
        "std": round(float(array.std(ddof=1)), 8) if array.size > 1 else 0.0,
        "min": round(float(array.min()), 8),
        "max": round(float(array.max()), 8),
        "values": [round(float(value), 8) for value in array],
    }


def _decision_for_worlds(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    reference_model_id: str,
    worlds: Sequence[str],
    minimum_mean: float,
    minimum_positive: int,
) -> dict[str, Any]:
    indexed = {
        (int(row["replicate"]), str(row["world_id"]), str(row["model_id"])): row
        for row in rows
    }
    world_reports: dict[str, Any] = {}
    for world_id in worlds:
        improvements = [
            float(indexed[(replicate, world_id, reference_model_id)]["brier_score"])
            - float(indexed[(replicate, world_id, model_id)]["brier_score"])
            for replicate in sorted(
                {
                    int(row["replicate"])
                    for row in rows
                    if row["world_id"] == world_id
                }
            )
        ]
        distribution = _distribution(improvements)
        positive = sum(value > 0.0 for value in improvements)
        world_reports[world_id] = {
            "brier_improvement": distribution,
            "positive_replicates": positive,
            "pass": distribution["mean"] >= minimum_mean
            and positive >= minimum_positive,
        }
    return {
        "model_id": model_id,
        "reference_model_id": reference_model_id,
        "minimum_mean_brier_improvement": minimum_mean,
        "minimum_positive_replicates": minimum_positive,
        "worlds": world_reports,
        "pass": all(report["pass"] for report in world_reports.values()),
    }


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
            cells.append(
                {
                    "world_id": world_id,
                    "model_id": model_id,
                    **{
                        metric: _distribution(
                            [float(row[metric]) for row in cell_rows]
                        )
                        for metric in METRICS
                    },
                }
            )
    orientation_distances = []
    scalar_invariant = True
    for pair_id, worlds in PAIR_WORLDS.items():
        for model_id in MODEL_IDS:
            distances = []
            for replicate, predictions in sorted(prediction_sets.items()):
                left = predictions[(worlds[0], model_id)]
                right = predictions[(worlds[1], model_id)]
                distances.append(float(np.mean(np.abs(left - right))))
                if model_id == "scalar_delta" and not np.array_equal(left, right):
                    scalar_invariant = False
            orientation_distances.append(
                {
                    "pair_id": pair_id,
                    "model_id": model_id,
                    "mean_absolute_prediction_distance": _distribution(distances),
                }
            )
    if not scalar_invariant:
        raise RuntimeError("scalar invariance failed during summary")

    decision = contract["decision_contract"]
    settings = _profile(contract, profile)
    if settings["replicate_count"] < int(
        decision["feature_conditional_need"]["improved_replicates_at_least"]
    ):
        dispositions: dict[str, Any] = {
            "status": "NOT_EVALUATED_SMOKE_PROFILE",
            "feature_conditional_need": "NOT_EVALUATED",
            "nonlinear_representation_need": "NOT_EVALUATED",
            "neural_specific_advantage": "NOT_EVALUATED",
        }
    else:
        feature_rule = decision["feature_conditional_need"]
        feature_report = _decision_for_worlds(
            rows,
            model_id="explicit_linear_logit",
            reference_model_id="scalar_delta",
            worlds=feature_rule["worlds"],
            minimum_mean=float(feature_rule["brier_improvement_at_least"]),
            minimum_positive=int(feature_rule["improved_replicates_at_least"]),
        )
        nonlinear_rule = decision["nonlinear_representation_need"]
        nonlinear_report = _decision_for_worlds(
            rows,
            model_id="residual_mlp",
            reference_model_id="explicit_linear_logit",
            worlds=nonlinear_rule["worlds"],
            minimum_mean=float(nonlinear_rule["brier_improvement_at_least"]),
            minimum_positive=int(nonlinear_rule["improved_replicates_at_least"]),
        )
        neural_rule = decision["neural_specific_advantage"]
        neural_report = _decision_for_worlds(
            rows,
            model_id="residual_mlp",
            reference_model_id="histogram_tree",
            worlds=neural_rule["worlds"],
            minimum_mean=float(neural_rule["brier_improvement_at_least"]),
            minimum_positive=int(neural_rule["improved_replicates_at_least"]),
        )
        indexed = {
            (int(row["replicate"]), str(row["world_id"]), str(row["model_id"])): row
            for row in rows
        }
        linear_regressions = []
        for world_id in feature_rule["worlds"]:
            for replicate in range(settings["replicate_count"]):
                linear_regressions.append(
                    float(indexed[(replicate, world_id, "residual_mlp")]["brier_score"])
                    - float(
                        indexed[(replicate, world_id, "explicit_linear_logit")][
                            "brier_score"
                        ]
                    )
                )
        maximum_regression = max(linear_regressions)
        neural_report["maximum_linear_world_brier_regression"] = round(
            maximum_regression, 8
        )
        neural_report["linear_world_guard_pass"] = maximum_regression <= float(
            neural_rule["maximum_linear_world_brier_regression"]
        )
        neural_report["pass"] = bool(
            neural_report["pass"] and neural_report["linear_world_guard_pass"]
        )
        dispositions = {
            "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
            "feature_conditional_need": (
                "SUPPORTED_ON_FROZEN_GENERATOR"
                if feature_report["pass"]
                else "NOT_SUPPORTED_ON_FROZEN_GENERATOR"
            ),
            "nonlinear_representation_need": (
                "SUPPORTED_ON_FROZEN_GENERATOR"
                if nonlinear_report["pass"]
                else "NOT_SUPPORTED_ON_FROZEN_GENERATOR"
            ),
            "neural_specific_advantage": (
                "SUPPORTED_ON_FROZEN_GENERATOR_ONLY"
                if neural_report["pass"]
                else "NOT_SUPPORTED_ON_FROZEN_GENERATOR"
            ),
            "feature_report": feature_report,
            "nonlinear_report": nonlinear_report,
            "neural_report": neural_report,
        }
    return {
        "schema_version": "pivot_matched_marginal_scorecard.v2",
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
    settings = _profile(contract, profile)
    rows: list[dict[str, Any]] = []
    prediction_sets: dict[int, Mapping[tuple[str, str], np.ndarray]] = {}
    for replicate in range(settings["replicate_count"]):
        replicate_rows, predictions = run_replicate(contract, profile, replicate)
        rows.extend(replicate_rows)
        prediction_sets[replicate] = predictions
    return rows, summarize(rows, prediction_sets, contract, profile)


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot matched-marginal policy scorecard v2",
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
    lines.extend(
        [
            "",
            "## Frozen dispositions",
            "",
            f"- Feature-conditional need: `{summary['dispositions']['feature_conditional_need']}`",
            f"- Nonlinear representation need: `{summary['dispositions']['nonlinear_representation_need']}`",
            f"- Neural-specific advantage: `{summary['dispositions']['neural_specific_advantage']}`",
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
        "schema_version": "pivot_matched_marginal_receipt.v2",
        "status": "COMPLETE",
        "row_count": len(rows),
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": _sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (rows_path, scorecard_path, markdown_path)
        },
        "boundaries": summary["boundaries"],
    }
    delayed._write_exclusive(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def _plan_payload(
    contract: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_path": str(config_path.resolve()),
        "contract_sha256": _sha256(config_path),
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
        "contract_sha256": _sha256(config_path),
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
