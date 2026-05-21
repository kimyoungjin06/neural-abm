"""Offline learned basin phase-critic training and diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F

from neural_abm.basin_transition_samples import (
    BASIN_PHASE_FIELD_NAMES,
    DEFAULT_FUTURE_BASIN_HORIZON,
    ensure_future_basin_motion_labels,
)
from neural_abm.state_continuation import build_basin_phase_representation


BasinCriticStatus = Literal["pass", "fail", "inconclusive"]
BasinPhaseCriticLabelMode = Literal[
    "ceiling_horizon",
    "prototype_direction",
    "future_outcome_direction",
]
BasinReplayWeightTargetMode = Literal[
    "magnitude",
    "future_basin_motion",
    "intervention_pressure",
]

DEFAULT_BASIN_PHASE_CRITIC_FEATURES: tuple[str, ...] = (
    "candidate_action",
    "policy_action_probability",
    "phase_payoff_alignment",
    "phase_action_rate",
    "phase_policy_rate",
    "phase_consensus",
    "phase_payoff_stability",
)

CANDIDATE_CONTEXT_BASIN_PHASE_CRITIC_FEATURES: tuple[str, ...] = (
    "candidate_action",
    "policy_action_probability",
    "candidate_action_delta",
    "candidate_policy_delta",
    "phase_payoff_alignment",
    "candidate_phase_action_rate",
    "candidate_phase_policy_rate",
    "candidate_phase_consensus",
    "phase_payoff_stability",
)

DEFAULT_BASIN_REPLAY_WEIGHT_FEATURES: tuple[str, ...] = (
    "prototype_action1_advantage",
    "learned_action1_advantage",
    "learned_action_margin",
    "learned_uncertainty",
    "learned_abstain",
    "prototype_learned_agreement",
    "prototype_learned_disagreement",
    "score_action0",
    "score_action1",
    "score_std_action0",
    "score_std_action1",
    "action_observed",
    "policy_action_probability",
    "phase_payoff_alignment",
    "phase_action_rate",
    "phase_policy_rate",
    "phase_consensus",
    "phase_payoff_stability",
)

BASIN_PHASE_CRITIC_SUMMARY_FIELDS: tuple[str, ...] = (
    "label",
    "case",
    "toy",
    "status",
    "train_seeds",
    "eval_seeds",
    "label_mode",
    "label_horizon",
    "feature_count",
    "feature_columns",
    "train_n",
    "train_positive_n",
    "train_negative_n",
    "eval_n",
    "eval_positive_n",
    "eval_negative_n",
    "eval_auc",
    "eval_average_precision",
    "eval_accuracy",
    "eval_brier",
    "eval_log_loss",
    "eval_pairwise_rank_accuracy",
    "prototype_observed_auc",
    "prototype_score_correlation",
    "ensemble_size",
    "abstention_margin_threshold",
    "uncertainty_threshold",
    "eval_candidate_margin_mean",
    "eval_candidate_margin_abs_mean",
    "eval_candidate_uncertainty_mean",
    "eval_candidate_uncertainty_max",
    "eval_action1_advantage_positive_rate",
    "eval_abstention_rate",
    "eval_non_abstain_rate",
    "final_loss",
    "reasons",
    "model_path",
    "predictions_path",
    "summary_json_path",
)


@dataclass(frozen=True)
class BasinPhaseCriticTrainingConfig:
    train_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    label_mode: BasinPhaseCriticLabelMode = "ceiling_horizon"
    label_horizon: int = 5
    ceiling_tolerance: float = 1e-6
    feature_columns: tuple[str, ...] = DEFAULT_BASIN_PHASE_CRITIC_FEATURES
    max_epochs: int = 300
    learning_rate: float = 0.05
    pairwise_weight: float = 0.25
    max_pairwise_pairs: int = 8192
    ensemble_size: int = 5
    abstention_margin_threshold: float = 0.005
    uncertainty_threshold: float = 0.05
    random_seed: int = 0


@dataclass(frozen=True)
class LearnedBasinPhaseCritic:
    feature_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float

    def predict_logits(self, frame: pd.DataFrame) -> np.ndarray:
        features = feature_matrix(frame, self.feature_columns)
        scaled = (features - self.feature_mean) / self.feature_scale
        return scaled @ self.weights + self.bias

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        logits = self.predict_logits(frame)
        return 1.0 / (1.0 + np.exp(-logits))

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            feature_columns=np.asarray(self.feature_columns, dtype=object),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            weights=self.weights,
            bias=np.asarray([self.bias], dtype=np.float64),
        )


@dataclass(frozen=True)
class LearnedBasinPhaseCriticBundle:
    main: LearnedBasinPhaseCritic
    ensemble: tuple[LearnedBasinPhaseCritic, ...]
    path: Path | None = None


@dataclass(frozen=True)
class LearnedBasinRuntimeDiagnostics:
    score_action0: np.ndarray
    score_action1: np.ndarray
    score_std_action0: np.ndarray
    score_std_action1: np.ndarray
    score_observed: np.ndarray
    action1_advantage: np.ndarray
    action_margin: np.ndarray
    uncertainty: np.ndarray
    abstain: np.ndarray
    prototype_action1_advantage_correlation: float
    model_path: str
    ensemble_size: int
    abstention_margin_threshold: float
    uncertainty_threshold: float
    replay_weight_features: pd.DataFrame | None = None


@dataclass(frozen=True)
class LearnedBasinCreditSignal:
    action1_advantage: np.ndarray
    learned_credit_used_mask: np.ndarray
    replay_mask: np.ndarray
    replay_weight: np.ndarray
    source: str
    replay_selection: str
    replay_min_selected_rate: float


@dataclass(frozen=True)
class LearnedBasinReplayWeightScorer:
    feature_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float
    output_floor: float = 0.0
    output_ceiling: float = 1.0

    def predict_weight(self, frame: pd.DataFrame) -> np.ndarray:
        features = feature_matrix(frame, self.feature_columns)
        scaled = (features - self.feature_mean) / self.feature_scale
        logits = scaled @ self.weights + self.bias
        residual = 1.0 / (1.0 + np.exp(-logits))
        floor = float(self.output_floor)
        ceiling = float(self.output_ceiling)
        return np.clip(floor + (ceiling - floor) * residual, 0.0, 1.0)

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            kind=np.asarray(["basin_replay_weight_scorer_v1"], dtype=object),
            feature_columns=np.asarray(self.feature_columns, dtype=object),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            weights=self.weights,
            bias=np.asarray([self.bias], dtype=np.float64),
            output_floor=np.asarray([self.output_floor], dtype=np.float64),
            output_ceiling=np.asarray([self.output_ceiling], dtype=np.float64),
        )


@dataclass(frozen=True)
class BasinReplayWeightScorerTrainingConfig:
    train_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    feature_columns: tuple[str, ...] = DEFAULT_BASIN_REPLAY_WEIGHT_FEATURES
    target_mode: BasinReplayWeightTargetMode = "magnitude"
    target_column: str = "training_effective_advantage"
    target_quantile: float = 0.90
    future_horizon: int = DEFAULT_FUTURE_BASIN_HORIZON
    output_floor: float = 0.50
    output_ceiling: float = 1.0
    max_epochs: int = 300
    learning_rate: float = 0.05
    random_seed: int = 0


@dataclass(frozen=True)
class BasinPhaseCriticQualityResult:
    label: str
    case: str
    toy: str
    status: BasinCriticStatus
    metrics: dict[str, object]
    reasons: tuple[str, ...]
    critic: LearnedBasinPhaseCritic | None = None
    ensemble: tuple[LearnedBasinPhaseCritic, ...] = ()

    def summary_row(self) -> dict[str, object]:
        return {
            "label": self.label,
            "case": self.case,
            "toy": self.toy,
            "status": self.status,
            **self.metrics,
            "reasons": "; ".join(self.reasons),
        }


@dataclass(frozen=True)
class BasinPhaseCriticCase:
    name: str
    toy: str
    ceiling_tolerance: float
    label_mode: BasinPhaseCriticLabelMode = "ceiling_horizon"
    label_horizon: int = 5
    group: str = "nabm"


@dataclass(frozen=True)
class BasinPhaseCriticManifest:
    label: str
    runs_csv: Path
    output_dir: Path
    train_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    max_epochs: int
    learning_rate: float
    pairwise_weight: float
    ensemble_size: int
    abstention_margin_threshold: float
    uncertainty_threshold: float
    feature_columns: tuple[str, ...]
    cases: tuple[BasinPhaseCriticCase, ...]


@dataclass(frozen=True)
class BasinPhaseCriticWorkflowResult:
    summary_csv_path: Path
    summary_json_path: Path
    markdown_path: Path
    rows: list[dict[str, object]]
    results: list[BasinPhaseCriticQualityResult]


def load_basin_phase_critic_manifest(path: Path) -> BasinPhaseCriticManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected mapping YAML at {path}")
    required = {"label", "runs_csv", "train_seeds", "eval_seeds", "cases"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(
            "Basin phase critic manifest missing field(s): " + ", ".join(missing)
        )
    cases_raw = raw["cases"]
    if not isinstance(cases_raw, Sequence) or isinstance(cases_raw, str | bytes):
        raise ValueError("Basin phase critic manifest cases must be a list")
    cases = tuple(_critic_case_from_mapping(case) for case in cases_raw)
    if not cases:
        raise ValueError("Basin phase critic manifest must contain at least one case")
    return BasinPhaseCriticManifest(
        label=str(raw["label"]),
        runs_csv=Path(str(raw["runs_csv"])),
        output_dir=Path(str(raw.get("output_dir", "experiments/results/basin_critic"))),
        train_seeds=tuple(int(seed) for seed in _sequence(raw, "train_seeds")),
        eval_seeds=tuple(int(seed) for seed in _sequence(raw, "eval_seeds")),
        max_epochs=int(raw.get("max_epochs", 300)),
        learning_rate=float(raw.get("learning_rate", 0.05)),
        pairwise_weight=float(raw.get("pairwise_weight", 0.25)),
        ensemble_size=int(raw.get("ensemble_size", 5)),
        abstention_margin_threshold=float(
            raw.get("abstention_margin_threshold", 0.005)
        ),
        uncertainty_threshold=float(raw.get("uncertainty_threshold", 0.05)),
        feature_columns=tuple(
            str(column)
            for column in raw.get(
                "feature_columns",
                DEFAULT_BASIN_PHASE_CRITIC_FEATURES,
            )
        ),
        cases=cases,
    )


def run_basin_phase_critic_workflow(
    manifest_path: Path,
    *,
    output_dir: Path | None = None,
) -> BasinPhaseCriticWorkflowResult:
    manifest = load_basin_phase_critic_manifest(manifest_path)
    resolved_output_dir = manifest.output_dir if output_dir is None else output_dir
    rows: list[dict[str, object]] = []
    results: list[BasinPhaseCriticQualityResult] = []
    for case in manifest.cases:
        artifact_paths = artifact_paths_from_runs_csv(
            manifest.runs_csv,
            toy=case.toy,
            group=case.group,
        )
        samples = load_basin_transition_sample_artifacts(artifact_paths)
        config = BasinPhaseCriticTrainingConfig(
            train_seeds=manifest.train_seeds,
            eval_seeds=manifest.eval_seeds,
            label_mode=case.label_mode,
            label_horizon=case.label_horizon,
            ceiling_tolerance=case.ceiling_tolerance,
            max_epochs=manifest.max_epochs,
            learning_rate=manifest.learning_rate,
            pairwise_weight=manifest.pairwise_weight,
            ensemble_size=manifest.ensemble_size,
            abstention_margin_threshold=manifest.abstention_margin_threshold,
            uncertainty_threshold=manifest.uncertainty_threshold,
            feature_columns=manifest.feature_columns,
        )
        result = train_evaluate_basin_phase_critic(
            samples,
            label=manifest.label,
            case=case.name,
            toy=case.toy,
            config=config,
        )
        case_output_dir = resolved_output_dir / manifest.label
        row = write_basin_phase_critic_case_artifacts(
            case_output_dir,
            result,
            examples=basin_phase_critic_examples(samples, config=config),
        )
        rows.append(row)
        results.append(result)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = resolved_output_dir / f"{manifest.label}_summary.csv"
    summary_json_path = resolved_output_dir / f"{manifest.label}_summary.json"
    markdown_path = resolved_output_dir / f"{manifest.label}_summary.md"
    write_summary_csv(summary_csv_path, rows)
    summary_json_path.write_text(
        json.dumps({"label": manifest.label, "cases": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_basin_phase_critic_markdown(rows), encoding="utf-8")
    return BasinPhaseCriticWorkflowResult(
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        markdown_path=markdown_path,
        rows=rows,
        results=results,
    )


def artifact_paths_from_runs_csv(
    runs_csv: Path,
    *,
    toy: str,
    group: str,
) -> list[Path]:
    runs = pd.read_csv(runs_csv)
    required = {"toy", "group", "run_dir"}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"Run CSV missing field(s): {', '.join(missing)}")
    paths: list[Path] = []
    for run_dir in runs.loc[
        (runs["toy"].astype(str) == toy) & (runs["group"].astype(str) == group),
        "run_dir",
    ]:
        artifact = Path(str(run_dir)) / "basin_transition_samples.parquet"
        if artifact.exists():
            paths.append(artifact)
    if not paths:
        raise ValueError(f"No basin transition sample artifacts found for {toy}/{group}")
    return paths


def load_basin_transition_sample_artifacts(paths: Sequence[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("At least one basin transition sample artifact is required")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["source_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def basin_phase_critic_examples(
    samples: pd.DataFrame,
    *,
    config: BasinPhaseCriticTrainingConfig,
) -> pd.DataFrame:
    required = {
        "seed",
        "epoch",
        "agent_id",
        "action_observed",
        "policy_action_probability",
        "mean_payoff",
        "target_payoff",
        "time_to_ceiling",
        "score_observed",
        *BASIN_PHASE_FIELD_NAMES,
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(
            "Basin transition samples missing field(s): " + ", ".join(missing)
        )
    examples = (
        ensure_future_basin_motion_labels(
            samples,
            future_horizon=int(config.label_horizon),
            ceiling_tolerance=float(config.ceiling_tolerance),
        )
        if config.label_mode in {"prototype_direction", "future_outcome_direction"}
        else samples.copy()
    )
    examples["candidate_action"] = pd.to_numeric(
        examples["action_observed"],
        errors="coerce",
    )
    examples = candidate_phase_context_examples(examples)
    mean_payoff = pd.to_numeric(examples["mean_payoff"], errors="coerce")
    target_payoff = pd.to_numeric(examples["target_payoff"], errors="coerce")
    epoch = pd.to_numeric(examples["epoch"], errors="coerce")
    time_to_ceiling = pd.to_numeric(examples["time_to_ceiling"], errors="coerce")
    current_reached = mean_payoff >= (
        target_payoff - float(config.ceiling_tolerance)
    )
    epochs_to_ceiling = time_to_ceiling - epoch
    future_reached = (epochs_to_ceiling >= 0.0) & (
        epochs_to_ceiling <= float(config.label_horizon)
    )
    examples["current_ceiling_reached"] = current_reached.fillna(False)
    examples["epochs_to_ceiling"] = epochs_to_ceiling
    examples["target_reached_within_horizon"] = (
        examples["current_ceiling_reached"] | future_reached.fillna(False)
    )
    if config.label_mode == "prototype_direction":
        examples = prototype_direction_candidate_examples(examples)
    elif config.label_mode == "future_outcome_direction":
        examples = future_outcome_direction_candidate_examples(
            examples,
            ceiling_tolerance=float(config.ceiling_tolerance),
        )
    elif config.label_mode != "ceiling_horizon":
        raise ValueError(f"Unsupported basin phase critic label_mode: {config.label_mode}")
    feature_values = feature_matrix(examples, config.feature_columns)
    finite_mask = np.isfinite(feature_values).all(axis=1)
    return examples.loc[finite_mask].reset_index(drop=True)


def prototype_direction_candidate_examples(examples: pd.DataFrame) -> pd.DataFrame:
    """Expand rows into action candidates with prototype direction labels."""

    if "basin_action1_advantage" not in examples.columns:
        raise ValueError(
            "Basin transition samples missing field(s): basin_action1_advantage"
        )
    direction = pd.to_numeric(
        examples["basin_action1_advantage"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    eligible = np.isfinite(direction) & (np.abs(direction) > 1e-12)
    if "future_basin_score_delta" in examples.columns:
        future_delta = pd.to_numeric(
            examples["future_basin_score_delta"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        future_motion = np.isfinite(future_delta) & (future_delta > 0.0)
        eligible = eligible & (
            examples["target_reached_within_horizon"].astype(bool).to_numpy()
            | examples["current_ceiling_reached"].astype(bool).to_numpy()
            | future_motion
        )
    base = examples.loc[eligible].copy()
    if base.empty:
        raise ValueError("No finite prototype direction rows for basin phase critic")
    direction = pd.to_numeric(
        base["basin_action1_advantage"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    candidates: list[pd.DataFrame] = []
    for candidate_action in (0.0, 1.0):
        candidate = base.copy()
        candidate["candidate_action"] = candidate_action
        candidate["prototype_direction_margin"] = np.abs(direction)
        candidate["prototype_direction_action1_preferred"] = direction > 0.0
        candidate["target_reached_within_horizon"] = (
            direction > 0.0 if candidate_action >= 0.5 else direction < 0.0
        )
        candidates.append(candidate)
    expanded = pd.concat(candidates, ignore_index=True)
    return candidate_phase_context_examples(expanded)


def future_outcome_direction_candidate_examples(
    examples: pd.DataFrame,
    *,
    ceiling_tolerance: float,
) -> pd.DataFrame:
    """Expand rows into action candidates with observed future-outcome labels."""

    required = {
        "action_observed",
        "future_basin_score_delta",
        "current_ceiling_reached",
        "target_reached_within_horizon",
    }
    missing = sorted(required - set(examples.columns))
    if missing:
        raise ValueError(
            "Basin transition samples missing field(s): " + ", ".join(missing)
        )
    observed_action = pd.to_numeric(
        examples["action_observed"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if "action_counterfactual" in examples.columns:
        counterfactual_action = pd.to_numeric(
            examples["action_counterfactual"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
    else:
        counterfactual_action = 1.0 - observed_action
    future_delta = pd.to_numeric(
        examples["future_basin_score_delta"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    current_or_future_ceiling = (
        examples["current_ceiling_reached"].astype(bool).to_numpy()
        | examples["target_reached_within_horizon"].astype(bool).to_numpy()
    )
    margin_floor = max(float(ceiling_tolerance), 1e-12)
    positive_observed = (
        (np.isfinite(future_delta) & (future_delta > margin_floor))
        | current_or_future_ceiling
    )
    positive_counterfactual = np.isfinite(future_delta) & (
        future_delta < -margin_floor
    )
    eligible = (
        np.isfinite(observed_action)
        & np.isfinite(counterfactual_action)
        & (observed_action != counterfactual_action)
        & (positive_observed | positive_counterfactual)
    )
    base = examples.loc[eligible].copy()
    if base.empty:
        raise ValueError("No finite future-outcome direction rows for basin phase critic")

    observed_action = observed_action[eligible]
    counterfactual_action = counterfactual_action[eligible]
    future_delta = future_delta[eligible]
    positive_observed = positive_observed[eligible]
    positive_action = np.where(
        positive_observed,
        observed_action,
        counterfactual_action,
    )
    margin = np.maximum(np.abs(future_delta), margin_floor)
    candidates: list[pd.DataFrame] = []
    for candidate_action in (0.0, 1.0):
        candidate = base.copy()
        candidate["candidate_action"] = candidate_action
        candidate["future_direction_margin"] = margin
        candidate["future_direction_observed_preferred"] = positive_observed
        candidate["target_reached_within_horizon"] = (
            np.abs(positive_action - candidate_action) < 0.5
        )
        candidates.append(candidate)
    expanded = pd.concat(candidates, ignore_index=True)
    return candidate_phase_context_examples(expanded)


def train_evaluate_basin_phase_critic(
    samples: pd.DataFrame,
    *,
    label: str,
    case: str,
    toy: str,
    config: BasinPhaseCriticTrainingConfig,
) -> BasinPhaseCriticQualityResult:
    examples = basin_phase_critic_examples(samples, config=config)
    train_mask = examples["seed"].astype(int).isin(config.train_seeds)
    eval_mask = examples["seed"].astype(int).isin(config.eval_seeds)
    train_examples = examples.loc[train_mask].reset_index(drop=True)
    eval_examples = examples.loc[eval_mask].reset_index(drop=True)
    base_metrics = {
        "train_seeds": _format_sequence(config.train_seeds),
        "eval_seeds": _format_sequence(config.eval_seeds),
        "label_mode": config.label_mode,
        "label_horizon": config.label_horizon,
        "feature_count": len(config.feature_columns),
        "feature_columns": ", ".join(config.feature_columns),
        "ensemble_size": config.ensemble_size,
        "abstention_margin_threshold": config.abstention_margin_threshold,
        "uncertainty_threshold": config.uncertainty_threshold,
        "train_n": len(train_examples),
        "eval_n": len(eval_examples),
    }
    train_labels = label_vector(train_examples)
    eval_labels = label_vector(eval_examples)
    class_metrics = {
        "train_positive_n": int(np.sum(train_labels == 1)),
        "train_negative_n": int(np.sum(train_labels == 0)),
        "eval_positive_n": int(np.sum(eval_labels == 1)),
        "eval_negative_n": int(np.sum(eval_labels == 0)),
    }
    empty_metrics = _empty_quality_metrics()
    insufficient = _insufficient_class_reasons(train_labels, eval_labels)
    if insufficient:
        return BasinPhaseCriticQualityResult(
            label=label,
            case=case,
            toy=toy,
            status="inconclusive",
            metrics={**base_metrics, **class_metrics, **empty_metrics},
            reasons=tuple(insufficient),
            critic=None,
        )

    critic, final_loss = fit_linear_pairwise_critic(
        train_examples,
        labels=train_labels,
        config=config,
    )
    ensemble = fit_basin_phase_critic_ensemble(
        train_examples,
        labels=train_labels,
        config=config,
    )
    eval_scores = critic.predict_proba(eval_examples)
    prototype_scores = pd.to_numeric(
        eval_examples["score_observed"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    quality_metrics = quality_metrics_for_scores(
        labels=eval_labels,
        scores=eval_scores,
        prototype_scores=prototype_scores,
    )
    candidate_metrics = candidate_action_summary_metrics(
        candidate_action_diagnostics(
            ensemble or (critic,),
            eval_examples,
            config=config,
        )
    )
    status, reasons = _status_for_quality(quality_metrics)
    return BasinPhaseCriticQualityResult(
        label=label,
        case=case,
        toy=toy,
        status=status,
        metrics={
            **base_metrics,
            **class_metrics,
            **quality_metrics,
            **candidate_metrics,
            "final_loss": final_loss,
        },
        reasons=tuple(reasons),
        critic=critic,
        ensemble=ensemble,
    )


def fit_basin_phase_critic_ensemble(
    train_examples: pd.DataFrame,
    *,
    labels: np.ndarray,
    config: BasinPhaseCriticTrainingConfig,
) -> tuple[LearnedBasinPhaseCritic, ...]:
    """Train bootstrap heads for uncertainty diagnostics."""

    ensemble_size = max(int(config.ensemble_size), 0)
    if ensemble_size <= 0:
        return ()
    rng = np.random.default_rng(config.random_seed)
    critics: list[LearnedBasinPhaseCritic] = []
    for head_index in range(ensemble_size):
        sample_indices = rng.integers(0, len(train_examples), size=len(train_examples))
        critic, _ = fit_linear_pairwise_critic(
            train_examples.iloc[sample_indices].reset_index(drop=True),
            labels=labels[sample_indices],
            config=config,
            random_seed=config.random_seed + head_index + 1,
        )
        critics.append(critic)
    return tuple(critics)


def fit_linear_pairwise_critic(
    train_examples: pd.DataFrame,
    *,
    labels: np.ndarray,
    config: BasinPhaseCriticTrainingConfig,
    random_seed: int | None = None,
) -> tuple[LearnedBasinPhaseCritic, float]:
    features = feature_matrix(train_examples, config.feature_columns)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    scaled = (features - mean) / scale

    resolved_random_seed = config.random_seed if random_seed is None else random_seed
    torch.manual_seed(resolved_random_seed)
    model = nn.Linear(scaled.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x_tensor = torch.as_tensor(scaled, dtype=torch.float32)
    y_tensor = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32)
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    pos_weight = torch.as_tensor(
        [max(negatives / max(positives, 1.0), 1e-6)],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(resolved_random_seed)
    final_loss = float("nan")
    for _ in range(config.max_epochs):
        optimizer.zero_grad()
        logits = model(x_tensor).squeeze(dim=1)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            y_tensor,
            pos_weight=pos_weight,
        )
        ranking = sampled_pairwise_ranking_loss(
            logits,
            y_tensor,
            max_pairs=config.max_pairwise_pairs,
            generator=generator,
        )
        loss = bce + float(config.pairwise_weight) * ranking
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    weights = model.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    bias = float(model.bias.detach().cpu().numpy().reshape(-1)[0])
    return (
        LearnedBasinPhaseCritic(
            feature_columns=config.feature_columns,
            feature_mean=mean.astype(np.float64),
            feature_scale=scale.astype(np.float64),
            weights=weights,
            bias=bias,
        ),
        final_loss,
    )


def sampled_pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_pairs: int,
    generator: torch.Generator,
) -> torch.Tensor:
    positive = torch.nonzero(labels > 0.5, as_tuple=False).reshape(-1)
    negative = torch.nonzero(labels <= 0.5, as_tuple=False).reshape(-1)
    if positive.numel() == 0 or negative.numel() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    pair_count = min(int(max_pairs), int(positive.numel() * negative.numel()))
    positive_sample = positive[
        torch.randint(positive.numel(), (pair_count,), generator=generator)
    ]
    negative_sample = negative[
        torch.randint(negative.numel(), (pair_count,), generator=generator)
    ]
    return F.softplus(-(logits[positive_sample] - logits[negative_sample])).mean()


def quality_metrics_for_scores(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    prototype_scores: np.ndarray,
) -> dict[str, float]:
    binary_predictions = scores >= 0.5
    metrics = {
        "eval_auc": _auc_or_nan(labels, scores),
        "eval_average_precision": float(average_precision_score(labels, scores)),
        "eval_accuracy": float(accuracy_score(labels, binary_predictions)),
        "eval_brier": float(brier_score_loss(labels, scores)),
        "eval_log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "eval_pairwise_rank_accuracy": pairwise_rank_accuracy(labels, scores),
        "prototype_observed_auc": _auc_or_nan(labels, prototype_scores),
        "prototype_score_correlation": pearson_or_nan(scores, prototype_scores),
    }
    return metrics


def candidate_action_diagnostics(
    critics: Sequence[LearnedBasinPhaseCritic],
    examples: pd.DataFrame,
    *,
    config: BasinPhaseCriticTrainingConfig,
) -> dict[str, np.ndarray]:
    """Score candidate actions 0/1 and return uncertainty diagnostics."""

    if not critics:
        raise ValueError("At least one critic is required for candidate scoring")
    action0 = examples.copy()
    action1 = examples.copy()
    action0["candidate_action"] = 0.0
    action1["candidate_action"] = 1.0
    action0 = candidate_phase_context_examples(action0)
    action1 = candidate_phase_context_examples(action1)
    scores0 = np.vstack([critic.predict_proba(action0) for critic in critics])
    scores1 = np.vstack([critic.predict_proba(action1) for critic in critics])
    mean0 = np.mean(scores0, axis=0)
    mean1 = np.mean(scores1, axis=0)
    std0 = np.std(scores0, axis=0)
    std1 = np.std(scores1, axis=0)
    margin = mean1 - mean0
    uncertainty = np.maximum(std0, std1)
    abstain = (
        np.abs(margin) < float(config.abstention_margin_threshold)
    ) | (uncertainty > float(config.uncertainty_threshold))
    observed_actions = pd.to_numeric(
        examples["action_observed"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    observed_score = np.where(observed_actions >= 0.5, mean1, mean0)
    return {
        "candidate_score_action0": mean0,
        "candidate_score_action1": mean1,
        "candidate_score_std_action0": std0,
        "candidate_score_std_action1": std1,
        "learned_basin_score": observed_score,
        "learned_basin_action1_advantage": margin,
        "learned_basin_action_margin": np.abs(margin),
        "learned_basin_uncertainty": uncertainty,
        "learned_basin_abstain": abstain.astype(bool),
    }


def candidate_action_summary_metrics(
    diagnostics: Mapping[str, np.ndarray],
) -> dict[str, float]:
    margin = np.asarray(diagnostics["learned_basin_action1_advantage"], dtype=np.float64)
    uncertainty = np.asarray(diagnostics["learned_basin_uncertainty"], dtype=np.float64)
    abstain = np.asarray(diagnostics["learned_basin_abstain"], dtype=bool)
    if margin.size == 0:
        return {
            "eval_candidate_margin_mean": float("nan"),
            "eval_candidate_margin_abs_mean": float("nan"),
            "eval_candidate_uncertainty_mean": float("nan"),
            "eval_candidate_uncertainty_max": float("nan"),
            "eval_action1_advantage_positive_rate": float("nan"),
            "eval_abstention_rate": float("nan"),
            "eval_non_abstain_rate": float("nan"),
        }
    return {
        "eval_candidate_margin_mean": float(np.mean(margin)),
        "eval_candidate_margin_abs_mean": float(np.mean(np.abs(margin))),
        "eval_candidate_uncertainty_mean": float(np.mean(uncertainty)),
        "eval_candidate_uncertainty_max": float(np.max(uncertainty)),
        "eval_action1_advantage_positive_rate": float(np.mean(margin > 0.0)),
        "eval_abstention_rate": float(np.mean(abstain)),
        "eval_non_abstain_rate": float(np.mean(~abstain)),
    }


def basin_replay_weight_feature_frame(
    *,
    prototype_action1_advantage: np.ndarray,
    candidate_diagnostics: Mapping[str, np.ndarray],
    examples: pd.DataFrame,
) -> pd.DataFrame:
    """Build replay-weight scorer features from critic and phase diagnostics."""

    prototype = np.asarray(prototype_action1_advantage, dtype=np.float64)
    learned = np.asarray(
        candidate_diagnostics["learned_basin_action1_advantage"],
        dtype=np.float64,
    )
    abstain = np.asarray(candidate_diagnostics["learned_basin_abstain"], dtype=bool)
    if prototype.shape != learned.shape:
        raise ValueError(
            "prototype advantage shape "
            f"{prototype.shape} must match learned advantage shape {learned.shape}"
        )
    finite_pair = np.isfinite(prototype) & np.isfinite(learned)
    frame = pd.DataFrame(
        {
            "prototype_action1_advantage": prototype,
            "learned_action1_advantage": learned,
            "learned_action_margin": np.asarray(
                candidate_diagnostics["learned_basin_action_margin"],
                dtype=np.float64,
            ),
            "learned_uncertainty": np.asarray(
                candidate_diagnostics["learned_basin_uncertainty"],
                dtype=np.float64,
            ),
            "learned_abstain": abstain.astype(np.float64),
            "prototype_learned_agreement": (
                finite_pair & (prototype * learned > 0.0)
            ).astype(np.float64),
            "prototype_learned_disagreement": (
                finite_pair & (prototype * learned < 0.0)
            ).astype(np.float64),
            "score_action0": np.asarray(
                candidate_diagnostics["candidate_score_action0"],
                dtype=np.float64,
            ),
            "score_action1": np.asarray(
                candidate_diagnostics["candidate_score_action1"],
                dtype=np.float64,
            ),
            "score_std_action0": np.asarray(
                candidate_diagnostics["candidate_score_std_action0"],
                dtype=np.float64,
            ),
            "score_std_action1": np.asarray(
                candidate_diagnostics["candidate_score_std_action1"],
                dtype=np.float64,
            ),
        }
    )
    for column in (
        "action_observed",
        "policy_action_probability",
        "phase_payoff_alignment",
        "phase_action_rate",
        "phase_policy_rate",
        "phase_consensus",
        "phase_payoff_stability",
    ):
        if column in examples.columns:
            frame[column] = pd.to_numeric(examples[column], errors="coerce")
    return frame


def train_basin_replay_weight_scorer(
    samples: pd.DataFrame,
    *,
    critic_bundle: LearnedBasinPhaseCriticBundle,
    config: BasinReplayWeightScorerTrainingConfig,
) -> tuple[LearnedBasinReplayWeightScorer, dict[str, object]]:
    """Train a small frozen scorer for continuous basin replay weights."""

    samples_with_targets = ensure_future_basin_motion_labels(
        samples,
        future_horizon=int(config.future_horizon),
    )
    critic_config = BasinPhaseCriticTrainingConfig(
        train_seeds=config.train_seeds,
        eval_seeds=config.eval_seeds,
        feature_columns=critic_bundle.main.feature_columns,
    )
    examples = basin_phase_critic_examples(samples_with_targets, config=critic_config)
    candidate_diagnostics = candidate_action_diagnostics(
        critic_bundle.ensemble or (critic_bundle.main,),
        examples,
        config=critic_config,
    )
    features = basin_replay_weight_feature_frame(
        prototype_action1_advantage=pd.to_numeric(
            examples["basin_action1_advantage"],
            errors="coerce",
        ).to_numpy(dtype=np.float64),
        candidate_diagnostics=candidate_diagnostics,
        examples=examples,
    )
    train_mask = examples["seed"].astype(int).isin(config.train_seeds).to_numpy()
    eval_mask = examples["seed"].astype(int).isin(config.eval_seeds).to_numpy()
    (
        target_residual,
        target_column,
        target_scale,
        target_extra_metrics,
    ) = _replay_weight_target_residual_for_examples(
        examples,
        config=config,
        train_mask=train_mask,
    )
    finite_features = np.isfinite(feature_matrix(features, config.feature_columns)).all(
        axis=1
    )
    finite_target = np.isfinite(target_residual)
    train = train_mask & finite_features & finite_target
    eval_ = eval_mask & finite_features & finite_target
    if not np.any(train):
        raise ValueError("No finite train rows for basin replay weight scorer")
    if not np.any(eval_):
        raise ValueError("No finite eval rows for basin replay weight scorer")

    scorer, final_loss = fit_replay_weight_scorer(
        features.loc[train].reset_index(drop=True),
        target_residual=target_residual[train],
        config=config,
    )
    train_pred = scorer.predict_weight(features.loc[train])
    eval_pred = scorer.predict_weight(features.loc[eval_])
    train_target = _floor_target(target_residual[train], config=config)
    eval_target = _floor_target(target_residual[eval_], config=config)
    metrics = {
        "train_seeds": _format_sequence(config.train_seeds),
        "eval_seeds": _format_sequence(config.eval_seeds),
        "feature_count": len(config.feature_columns),
        "feature_columns": ", ".join(config.feature_columns),
        "target_mode": config.target_mode,
        "target_column": target_column,
        "target_quantile": config.target_quantile,
        "future_horizon": config.future_horizon,
        "target_scale": target_scale,
        **target_extra_metrics,
        "output_floor": config.output_floor,
        "output_ceiling": config.output_ceiling,
        "train_n": int(np.sum(train)),
        "eval_n": int(np.sum(eval_)),
        "train_target_mean": float(np.mean(train_target)),
        "eval_target_mean": float(np.mean(eval_target)),
        "train_weight_mean": float(np.mean(train_pred)),
        "eval_weight_mean": float(np.mean(eval_pred)),
        "eval_mse": float(np.mean(np.square(eval_pred - eval_target))),
        "eval_target_correlation": pearson_or_nan(eval_pred, eval_target),
        "final_loss": final_loss,
    }
    return scorer, metrics


def fit_replay_weight_scorer(
    features: pd.DataFrame,
    *,
    target_residual: np.ndarray,
    config: BasinReplayWeightScorerTrainingConfig,
) -> tuple[LearnedBasinReplayWeightScorer, float]:
    values = feature_matrix(features, config.feature_columns)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    scaled = (values - mean) / scale
    torch.manual_seed(int(config.random_seed))
    model = nn.Linear(scaled.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    x_tensor = torch.as_tensor(scaled, dtype=torch.float32)
    y_tensor = torch.as_tensor(
        np.clip(target_residual, 0.0, 1.0).astype(np.float32),
        dtype=torch.float32,
    )
    final_loss = float("nan")
    for _ in range(int(config.max_epochs)):
        optimizer.zero_grad()
        residual = torch.sigmoid(model(x_tensor).squeeze(dim=1))
        loss = F.mse_loss(residual, y_tensor)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    weights = model.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    bias = float(model.bias.detach().cpu().numpy().reshape(-1)[0])
    return (
        LearnedBasinReplayWeightScorer(
            feature_columns=tuple(config.feature_columns),
            feature_mean=mean.astype(np.float64),
            feature_scale=scale.astype(np.float64),
            weights=weights,
            bias=bias,
            output_floor=float(config.output_floor),
            output_ceiling=float(config.output_ceiling),
        ),
        final_loss,
    )


def load_learned_basin_replay_weight_scorer(
    path: Path,
) -> LearnedBasinReplayWeightScorer:
    with np.load(path, allow_pickle=True) as payload:
        return LearnedBasinReplayWeightScorer(
            feature_columns=tuple(str(value) for value in payload["feature_columns"]),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(np.asarray(payload["bias"], dtype=np.float64).reshape(-1)[0]),
            output_floor=float(
                np.asarray(
                    payload["output_floor"] if "output_floor" in payload else [0.0],
                    dtype=np.float64,
                )
                .reshape(-1)[0]
            ),
            output_ceiling=float(
                np.asarray(
                    (
                        payload["output_ceiling"]
                        if "output_ceiling" in payload
                        else [1.0]
                    ),
                    dtype=np.float64,
                )
                .reshape(-1)[0]
            ),
        )


def candidate_phase_context_examples(frame: pd.DataFrame) -> pd.DataFrame:
    """Add candidate-conditioned phase features for action-0/action-1 scoring."""

    examples = frame.copy()
    candidate = pd.to_numeric(examples["candidate_action"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    observed = pd.to_numeric(examples["action_observed"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    policy_probability = pd.to_numeric(
        examples["policy_action_probability"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    action_rate = _numeric_column(
        examples,
        "action_rate",
        fallback="phase_action_rate",
    )
    policy_rate = _numeric_column(
        examples,
        "policy_rate",
        fallback="phase_policy_rate",
    )
    agent_count = _candidate_agent_counts(examples)

    candidate_action_delta = candidate - observed
    candidate_policy_delta = candidate - policy_probability
    candidate_action_rate = np.clip(
        action_rate + candidate_action_delta / agent_count,
        0.0,
        1.0,
    )
    candidate_policy_rate = np.clip(
        policy_rate + candidate_policy_delta / agent_count,
        0.0,
        1.0,
    )
    examples["candidate_action_delta"] = candidate_action_delta
    examples["candidate_policy_delta"] = candidate_policy_delta
    examples["candidate_phase_action_rate"] = candidate_action_rate
    examples["candidate_phase_policy_rate"] = candidate_policy_rate
    examples["candidate_phase_consensus"] = np.clip(
        2.0 * np.abs(candidate_action_rate - 0.5),
        0.0,
        1.0,
    )
    return examples


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    fallback: str | None = None,
) -> np.ndarray:
    if column in frame.columns:
        source = column
    elif fallback is not None and fallback in frame.columns:
        source = fallback
    else:
        raise ValueError(f"Column missing: {column}")
    return pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=np.float64)


def _candidate_agent_counts(frame: pd.DataFrame) -> np.ndarray:
    if "agent_count" in frame.columns:
        counts = pd.to_numeric(frame["agent_count"], errors="coerce").to_numpy(
            dtype=np.float64
        )
    else:
        group_keys = [
            key
            for key in ("source_path", "run_id", "seed", "epoch")
            if key in frame.columns
        ]
        if group_keys:
            counts = (
                frame.groupby(group_keys, dropna=False)["agent_id"]
                .transform("count")
                .to_numpy(dtype=np.float64)
            )
        else:
            counts = np.full(len(frame), len(frame), dtype=np.float64)
    return np.maximum(counts, 1.0)


def load_learned_basin_phase_critic_bundle(
    path: Path,
) -> LearnedBasinPhaseCriticBundle:
    """Load a learned basin phase critic bundle written by the training CLI."""

    with np.load(path, allow_pickle=True) as payload:
        feature_columns = tuple(str(value) for value in payload["feature_columns"])
        main = LearnedBasinPhaseCritic(
            feature_columns=feature_columns,
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(np.asarray(payload["bias"], dtype=np.float64).reshape(-1)[0]),
        )
        if "ensemble_weights" not in payload:
            return LearnedBasinPhaseCriticBundle(main=main, ensemble=(main,), path=path)
        means = np.asarray(payload["ensemble_feature_mean"], dtype=np.float64)
        scales = np.asarray(payload["ensemble_feature_scale"], dtype=np.float64)
        weights = np.asarray(payload["ensemble_weights"], dtype=np.float64)
        biases = np.asarray(payload["ensemble_bias"], dtype=np.float64).reshape(-1)
    ensemble = tuple(
        LearnedBasinPhaseCritic(
            feature_columns=feature_columns,
            feature_mean=means[index],
            feature_scale=scales[index],
            weights=weights[index],
            bias=float(biases[index]),
        )
        for index in range(weights.shape[0])
    )
    return LearnedBasinPhaseCriticBundle(main=main, ensemble=ensemble, path=path)


def learned_basin_runtime_diagnostics(
    bundle: LearnedBasinPhaseCriticBundle,
    *,
    actions: np.ndarray,
    payoffs: np.ndarray,
    action_probabilities: np.ndarray,
    target_payoff: float,
    abstention_margin_threshold: float,
    uncertainty_threshold: float,
    prototype_action1_advantage: np.ndarray | None = None,
) -> LearnedBasinRuntimeDiagnostics:
    """Score runtime phase rows with a learned critic without changing policy."""

    examples = runtime_basin_phase_examples(
        actions=actions,
        payoffs=payoffs,
        action_probabilities=action_probabilities,
        target_payoff=target_payoff,
    )
    diagnostics = candidate_action_diagnostics(
        bundle.ensemble or (bundle.main,),
        examples,
        config=BasinPhaseCriticTrainingConfig(
            train_seeds=(),
            eval_seeds=(),
            abstention_margin_threshold=float(abstention_margin_threshold),
            uncertainty_threshold=float(uncertainty_threshold),
        ),
    )
    correlation = (
        float("nan")
        if prototype_action1_advantage is None
        else pearson_or_nan(
            np.asarray(prototype_action1_advantage, dtype=np.float64),
            diagnostics["learned_basin_action1_advantage"],
        )
    )
    replay_features = (
        None
        if prototype_action1_advantage is None
        else basin_replay_weight_feature_frame(
            prototype_action1_advantage=np.asarray(
                prototype_action1_advantage,
                dtype=np.float64,
            ),
            candidate_diagnostics=diagnostics,
            examples=examples,
        )
    )
    return LearnedBasinRuntimeDiagnostics(
        score_action0=diagnostics["candidate_score_action0"],
        score_action1=diagnostics["candidate_score_action1"],
        score_std_action0=diagnostics["candidate_score_std_action0"],
        score_std_action1=diagnostics["candidate_score_std_action1"],
        score_observed=diagnostics["learned_basin_score"],
        action1_advantage=diagnostics["learned_basin_action1_advantage"],
        action_margin=diagnostics["learned_basin_action_margin"],
        uncertainty=diagnostics["learned_basin_uncertainty"],
        abstain=diagnostics["learned_basin_abstain"].astype(bool),
        prototype_action1_advantage_correlation=correlation,
        model_path="" if bundle.path is None else str(bundle.path),
        ensemble_size=len(bundle.ensemble),
        abstention_margin_threshold=float(abstention_margin_threshold),
        uncertainty_threshold=float(uncertainty_threshold),
        replay_weight_features=replay_features,
    )


def learned_basin_credit_signal(
    diagnostics: LearnedBasinRuntimeDiagnostics | None,
    *,
    prototype_action1_advantage: np.ndarray,
    fallback: str = "prototype",
    replay_selection: str = "all",
    replay_mode: str = "hard",
    replay_min_selected_rate: float = 0.0,
    replay_floor_source: str = "prototype_abs",
    replay_soft_min_weight: float = 0.0,
    replay_soft_disagreement_weight: float = 0.25,
    replay_weight_scorer: LearnedBasinReplayWeightScorer | None = None,
    eligible_mask: np.ndarray | None = None,
) -> LearnedBasinCreditSignal:
    """Return the gated learned credit signal used for replay training."""

    prototype = np.asarray(prototype_action1_advantage, dtype=np.float64)
    eligible = _learned_replay_eligible_mask(eligible_mask, prototype.shape)
    if diagnostics is None:
        return LearnedBasinCreditSignal(
            action1_advantage=prototype,
            learned_credit_used_mask=np.zeros(prototype.shape, dtype=bool),
            replay_mask=np.ones(prototype.shape, dtype=bool),
            replay_weight=np.ones(prototype.shape, dtype=np.float64),
            source="prototype",
            replay_selection="all",
            replay_min_selected_rate=0.0,
        )
    learned = np.asarray(diagnostics.action1_advantage, dtype=np.float64)
    abstain = np.asarray(diagnostics.abstain, dtype=bool)
    if not (prototype.shape == learned.shape == abstain.shape):
        raise ValueError(
            "learned basin credit shapes must match prototype shape "
            f"{prototype.shape}; got learned {learned.shape}, abstain {abstain.shape}"
        )
    used = ~abstain
    agreement = np.isfinite(prototype) & np.isfinite(learned) & (prototype * learned > 0.0)
    disagreement = (
        np.isfinite(prototype) & np.isfinite(learned) & (prototype * learned < 0.0)
    )
    if replay_selection == "all":
        replay_mask = np.ones(prototype.shape, dtype=bool)
    elif replay_selection == "confident":
        replay_mask = used.copy()
    elif replay_selection == "confident_agreement":
        replay_mask = used & agreement
    elif replay_selection == "confident_disagreement":
        replay_mask = used & disagreement
    else:
        raise ValueError(f"Unsupported learned basin replay selection: {replay_selection}")
    if replay_mode == "hard":
        replay_mask = _apply_learned_replay_floor(
            replay_mask=replay_mask,
            eligible_mask=eligible,
            prototype=prototype,
            learned=learned,
            used_mask=used,
            min_selected_rate=float(replay_min_selected_rate),
            floor_source=replay_floor_source,
        )
        replay_weight = replay_mask.astype(np.float64)
    elif replay_mode == "soft_attention":
        replay_weight = _learned_soft_replay_weights(
            prototype=prototype,
            learned=learned,
            action_margin=np.asarray(diagnostics.action_margin, dtype=np.float64),
            uncertainty=np.asarray(diagnostics.uncertainty, dtype=np.float64),
            abstain=abstain,
            eligible_mask=eligible,
            replay_selection=replay_selection,
            min_weight=float(replay_soft_min_weight),
            disagreement_weight=float(replay_soft_disagreement_weight),
            margin_threshold=float(diagnostics.abstention_margin_threshold),
            uncertainty_threshold=float(diagnostics.uncertainty_threshold),
        )
        replay_mask = replay_weight > 0.0
    elif replay_mode == "learned_weight":
        if replay_weight_scorer is None:
            raise ValueError("learned basin replay mode requires a weight scorer")
        if diagnostics.replay_weight_features is None:
            raise ValueError(
                "learned basin replay weight scorer requires runtime feature rows"
            )
        replay_weight = replay_weight_scorer.predict_weight(
            diagnostics.replay_weight_features
        )
        if replay_weight.shape != prototype.shape:
            raise ValueError(
                "learned basin replay weight shape "
                f"{replay_weight.shape} must match prototype shape {prototype.shape}"
            )
        replay_weight = np.where(eligible, np.clip(replay_weight, 0.0, 1.0), 0.0)
        replay_mask = replay_weight > 0.0
    else:
        raise ValueError(f"Unsupported learned basin replay mode: {replay_mode}")
    if fallback == "prototype":
        fallback_advantage = prototype
    elif fallback == "zero":
        fallback_advantage = np.zeros(prototype.shape, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported learned basin credit fallback: {fallback}")
    return LearnedBasinCreditSignal(
        action1_advantage=np.where(used, learned, fallback_advantage),
        learned_credit_used_mask=used,
        replay_mask=replay_mask,
        replay_weight=replay_weight,
        source=f"learned_gated_{fallback}",
        replay_selection=replay_selection,
        replay_min_selected_rate=float(replay_min_selected_rate),
    )


def _learned_replay_eligible_mask(
    eligible_mask: np.ndarray | None,
    shape: tuple[int, ...],
) -> np.ndarray:
    if eligible_mask is None:
        return np.ones(shape, dtype=bool)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if eligible.shape != shape:
        raise ValueError(
            "learned basin replay eligible mask shape "
            f"{eligible.shape} must match credit shape {shape}"
        )
    return eligible


def _apply_learned_replay_floor(
    *,
    replay_mask: np.ndarray,
    eligible_mask: np.ndarray,
    prototype: np.ndarray,
    learned: np.ndarray,
    used_mask: np.ndarray,
    min_selected_rate: float,
    floor_source: str,
) -> np.ndarray:
    if min_selected_rate <= 0.0:
        return replay_mask
    if min_selected_rate > 1.0:
        raise ValueError(
            "learned basin replay min selected rate must be between 0 and 1"
        )
    eligible_count = int(np.sum(eligible_mask))
    if eligible_count == 0:
        return replay_mask
    selected = replay_mask.copy()
    target_count = int(np.ceil(float(min_selected_rate) * float(eligible_count)))
    current_count = int(np.sum(selected & eligible_mask))
    if current_count >= target_count:
        return selected
    candidate_mask = eligible_mask & ~selected
    if floor_source == "prototype_abs":
        scores = np.abs(prototype)
    elif floor_source == "learned_abs":
        scores = np.where(used_mask, np.abs(learned), -np.inf)
    else:
        raise ValueError(f"Unsupported learned basin replay floor source: {floor_source}")
    candidates = np.flatnonzero(candidate_mask & np.isfinite(scores))
    if candidates.size == 0:
        return selected
    needed = min(target_count - current_count, int(candidates.size))
    order = np.argsort(scores[candidates], kind="stable")
    selected[candidates[order[-needed:]]] = True
    return selected


def _learned_soft_replay_weights(
    *,
    prototype: np.ndarray,
    learned: np.ndarray,
    action_margin: np.ndarray,
    uncertainty: np.ndarray,
    abstain: np.ndarray,
    eligible_mask: np.ndarray,
    replay_selection: str,
    min_weight: float,
    disagreement_weight: float,
    margin_threshold: float,
    uncertainty_threshold: float,
) -> np.ndarray:
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("learned basin replay soft min weight must be between 0 and 1")
    if not 0.0 <= disagreement_weight <= 1.0:
        raise ValueError(
            "learned basin replay soft disagreement weight must be between 0 and 1"
        )
    finite = np.isfinite(prototype) & np.isfinite(learned)
    used = ~abstain
    agreement = finite & used & (prototype * learned > 0.0)
    disagreement = finite & used & (prototype * learned < 0.0)
    relation = np.zeros(prototype.shape, dtype=np.float64)
    if replay_selection == "all":
        relation[finite & used] = 1.0
        relation[eligible_mask & ~used] = disagreement_weight
    elif replay_selection == "confident":
        relation[finite & used] = 1.0
    elif replay_selection == "confident_agreement":
        relation[agreement] = 1.0
        relation[disagreement] = disagreement_weight
    elif replay_selection == "confident_disagreement":
        relation[disagreement] = 1.0
        relation[agreement] = disagreement_weight
    else:
        raise ValueError(f"Unsupported learned basin replay selection: {replay_selection}")
    margin_scale = max(float(margin_threshold), 1e-8)
    margin_confidence = np.clip(np.abs(action_margin) / margin_scale, 0.0, 1.0)
    if uncertainty_threshold <= 0.0:
        uncertainty_confidence = np.ones(prototype.shape, dtype=np.float64)
    else:
        uncertainty_confidence = np.clip(
            1.0 - (uncertainty / float(uncertainty_threshold)),
            0.0,
            1.0,
        )
    confidence = np.where(
        np.isfinite(margin_confidence) & np.isfinite(uncertainty_confidence),
        margin_confidence * uncertainty_confidence,
        0.0,
    )
    raw_weight = min_weight + (1.0 - min_weight) * relation * confidence
    return np.where(eligible_mask, np.clip(raw_weight, 0.0, 1.0), 0.0)


def runtime_basin_phase_examples(
    *,
    actions: np.ndarray,
    payoffs: np.ndarray,
    action_probabilities: np.ndarray,
    target_payoff: float,
) -> pd.DataFrame:
    """Build per-agent runtime examples compatible with learned critic features."""

    action_values = np.asarray(actions, dtype=np.int64)
    payoff_values = np.asarray(payoffs, dtype=np.float64)
    probability_values = np.asarray(action_probabilities, dtype=np.float64)
    if not (
        action_values.ndim == payoff_values.ndim == probability_values.ndim == 1
    ):
        raise ValueError("runtime basin critic inputs must be 1D")
    if not (len(action_values) == len(payoff_values) == len(probability_values)):
        raise ValueError("runtime basin critic inputs must have matching lengths")
    embedding = build_basin_phase_representation(
        actions=action_values,
        payoffs=payoff_values,
        target_payoff=float(target_payoff),
        action_probabilities=probability_values,
    )[0]
    phase_fields = dict(zip(BASIN_PHASE_FIELD_NAMES, map(float, embedding), strict=True))
    rows = [
        {
            "agent_id": agent_id,
            "agent_count": len(action_values),
            "action_observed": int(action_values[agent_id]),
            "candidate_action": int(action_values[agent_id]),
            "policy_action_probability": float(probability_values[agent_id]),
            "action_rate": float(np.mean(action_values)),
            "policy_rate": float(np.mean(probability_values)),
            **phase_fields,
        }
        for agent_id in range(len(action_values))
    ]
    return candidate_phase_context_examples(pd.DataFrame(rows))


_LEARNED_BASIN_RUNTIME_EMPTY_FIELDS: dict[str, object] = {
    "domain_basin_learned_model_path": "",
    "domain_basin_learned_ensemble_size": "",
    "domain_basin_learned_margin_threshold": "",
    "domain_basin_learned_uncertainty_threshold": "",
    "domain_basin_learned_score_action0_mean": "",
    "domain_basin_learned_score_action1_mean": "",
    "domain_basin_learned_action1_advantage_mean": "",
    "domain_basin_learned_action1_advantage_positive_rate": "",
    "domain_basin_learned_uncertainty_mean": "",
    "domain_basin_learned_abstention_rate": "",
    "domain_basin_learned_prototype_advantage_correlation": "",
}

_LEARNED_BASIN_RUNTIME_MICRO_EMPTY_FIELDS: dict[str, object] = {
    "domain_basin_learned_score_action0": "",
    "domain_basin_learned_score_action1": "",
    "domain_basin_learned_action1_advantage": "",
    "domain_basin_learned_uncertainty": "",
    "domain_basin_learned_abstain": "",
}


def learned_basin_runtime_aggregate_fields(
    diagnostics: LearnedBasinRuntimeDiagnostics | None,
) -> dict[str, object]:
    """Return aggregate read-only learned basin critic diagnostics."""

    if diagnostics is None or diagnostics.action1_advantage.size == 0:
        return dict(_LEARNED_BASIN_RUNTIME_EMPTY_FIELDS)
    return {
        "domain_basin_learned_model_path": diagnostics.model_path,
        "domain_basin_learned_ensemble_size": diagnostics.ensemble_size,
        "domain_basin_learned_margin_threshold": (
            diagnostics.abstention_margin_threshold
        ),
        "domain_basin_learned_uncertainty_threshold": diagnostics.uncertainty_threshold,
        "domain_basin_learned_score_action0_mean": float(
            np.mean(diagnostics.score_action0)
        ),
        "domain_basin_learned_score_action1_mean": float(
            np.mean(diagnostics.score_action1)
        ),
        "domain_basin_learned_action1_advantage_mean": float(
            np.mean(diagnostics.action1_advantage)
        ),
        "domain_basin_learned_action1_advantage_positive_rate": float(
            np.mean(diagnostics.action1_advantage > 0.0)
        ),
        "domain_basin_learned_uncertainty_mean": float(
            np.mean(diagnostics.uncertainty)
        ),
        "domain_basin_learned_abstention_rate": float(np.mean(diagnostics.abstain)),
        "domain_basin_learned_prototype_advantage_correlation": _finite_or_empty(
            diagnostics.prototype_action1_advantage_correlation
        ),
    }


def learned_basin_runtime_micro_fields(
    diagnostics: LearnedBasinRuntimeDiagnostics | None,
    agent_id: int,
) -> dict[str, object]:
    """Return per-agent read-only learned basin critic diagnostics."""

    if diagnostics is None or agent_id >= diagnostics.action1_advantage.size:
        return dict(_LEARNED_BASIN_RUNTIME_MICRO_EMPTY_FIELDS)
    return {
        "domain_basin_learned_score_action0": float(
            diagnostics.score_action0[agent_id]
        ),
        "domain_basin_learned_score_action1": float(
            diagnostics.score_action1[agent_id]
        ),
        "domain_basin_learned_action1_advantage": float(
            diagnostics.action1_advantage[agent_id]
        ),
        "domain_basin_learned_uncertainty": float(diagnostics.uncertainty[agent_id]),
        "domain_basin_learned_abstain": bool(diagnostics.abstain[agent_id]),
    }


def _finite_or_empty(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else ""


def write_basin_phase_critic_case_artifacts(
    output_dir: Path,
    result: BasinPhaseCriticQualityResult,
    *,
    examples: pd.DataFrame,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{result.label}_{result.case}"
    model_path = output_dir / f"{prefix}_model.npz"
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    summary_json_path = output_dir / f"{prefix}_summary.json"
    row = result.summary_row()
    if result.critic is not None:
        save_critic_bundle_npz(model_path, result.critic, result.ensemble)
        predictions = prediction_frame(
            result.critic,
            examples,
            ensemble=result.ensemble,
            config=BasinPhaseCriticTrainingConfig(
                train_seeds=(),
                eval_seeds=(),
                label_mode=str(
                    result.metrics.get("label_mode", "ceiling_horizon")
                ),
                ensemble_size=int(result.metrics.get("ensemble_size", 0)),
                abstention_margin_threshold=float(
                    result.metrics.get("abstention_margin_threshold", 0.005)
                ),
                uncertainty_threshold=float(
                    result.metrics.get("uncertainty_threshold", 0.05)
                ),
            ),
        )
        predictions.to_csv(predictions_path, index=False)
        row["model_path"] = str(model_path)
        row["predictions_path"] = str(predictions_path)
    else:
        row["model_path"] = ""
        row["predictions_path"] = ""
    row["summary_json_path"] = str(summary_json_path)
    summary_json_path.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return row


def save_critic_bundle_npz(
    path: Path,
    critic: LearnedBasinPhaseCritic,
    ensemble: Sequence[LearnedBasinPhaseCritic],
) -> None:
    """Persist the main critic and bootstrap heads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    heads = tuple(ensemble)
    np.savez(
        path,
        feature_columns=np.asarray(critic.feature_columns, dtype=object),
        feature_mean=critic.feature_mean,
        feature_scale=critic.feature_scale,
        weights=critic.weights,
        bias=np.asarray([critic.bias], dtype=np.float64),
        ensemble_feature_mean=np.vstack(
            [head.feature_mean for head in heads]
            if heads
            else [critic.feature_mean]
        ),
        ensemble_feature_scale=np.vstack(
            [head.feature_scale for head in heads]
            if heads
            else [critic.feature_scale]
        ),
        ensemble_weights=np.vstack(
            [head.weights for head in heads] if heads else [critic.weights]
        ),
        ensemble_bias=np.asarray(
            [head.bias for head in heads] if heads else [critic.bias],
            dtype=np.float64,
        ),
    )


def prediction_frame(
    critic: LearnedBasinPhaseCritic,
    examples: pd.DataFrame,
    *,
    ensemble: Sequence[LearnedBasinPhaseCritic] = (),
    config: BasinPhaseCriticTrainingConfig | None = None,
) -> pd.DataFrame:
    resolved_config = (
        BasinPhaseCriticTrainingConfig(train_seeds=(), eval_seeds=())
        if config is None
        else config
    )
    candidate_diagnostics = candidate_action_diagnostics(
        tuple(ensemble) or (critic,),
        examples,
        config=resolved_config,
    )
    columns = [
        "toy",
        "run_id",
        "seed",
        "epoch",
        "agent_id",
        "target_reached_within_horizon",
        "current_ceiling_reached",
        "epochs_to_ceiling",
        "score_observed",
    ]
    available = [column for column in columns if column in examples.columns]
    frame = examples.loc[:, available].copy()
    for column, values in candidate_diagnostics.items():
        frame[column] = values
    return frame


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(BASIN_PHASE_CRITIC_SUMMARY_FIELDS),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def render_basin_phase_critic_markdown(
    rows: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# Basin Phase Critic Quality",
        "",
        "| Case | Toy | Status | Train N | Eval N | Eval AUC | Pairwise Rank | Prototype AUC | Abstain | Uncertainty | Reasons |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {toy} | {status} | {train_n} | {eval_n} | "
            "{auc} | {rank} | {prototype_auc} | {abstain} | {uncertainty} | "
            "{reasons} |".format(
                case=row["case"],
                toy=row["toy"],
                status=row["status"],
                train_n=row["train_n"],
                eval_n=row["eval_n"],
                auc=_format_metric(row.get("eval_auc")),
                rank=_format_metric(row.get("eval_pairwise_rank_accuracy")),
                prototype_auc=_format_metric(row.get("prototype_observed_auc")),
                abstain=_format_metric(row.get("eval_abstention_rate")),
                uncertainty=_format_metric(row.get("eval_candidate_uncertainty_mean")),
                reasons=row.get("reasons", ""),
            )
        )
    return "\n".join(lines) + "\n"


def feature_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("Feature column(s) missing: " + ", ".join(missing))
    return frame.loc[:, list(columns)].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)


def label_vector(frame: pd.DataFrame) -> np.ndarray:
    if "target_reached_within_horizon" not in frame.columns:
        return np.empty(0, dtype=np.int64)
    return frame["target_reached_within_horizon"].astype(bool).to_numpy(
        dtype=np.int64
    )


def pairwise_rank_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if positive_scores.size == 0 or negative_scores.size == 0:
        return float("nan")
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    wins = np.sum(comparisons > 0.0)
    ties = np.sum(comparisons == 0.0)
    total = comparisons.size
    return float((wins + 0.5 * ties) / total)


def pearson_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if np.sum(mask) < 2:
        return float("nan")
    left_values = left[mask]
    right_values = right[mask]
    if np.std(left_values) <= 1e-12 or np.std(right_values) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _auc_or_nan(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _empty_quality_metrics() -> dict[str, object]:
    return {
        "eval_auc": "",
        "eval_average_precision": "",
        "eval_accuracy": "",
        "eval_brier": "",
        "eval_log_loss": "",
        "eval_pairwise_rank_accuracy": "",
        "prototype_observed_auc": "",
        "prototype_score_correlation": "",
        "eval_candidate_margin_mean": "",
        "eval_candidate_margin_abs_mean": "",
        "eval_candidate_uncertainty_mean": "",
        "eval_candidate_uncertainty_max": "",
        "eval_action1_advantage_positive_rate": "",
        "eval_abstention_rate": "",
        "eval_non_abstain_rate": "",
        "final_loss": "",
    }


def _insufficient_class_reasons(
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
) -> list[str]:
    reasons: list[str] = []
    if train_labels.size == 0:
        reasons.append("empty train split")
    elif len(set(train_labels.tolist())) < 2:
        reasons.append("train split has fewer than two label classes")
    if eval_labels.size == 0:
        reasons.append("empty eval split")
    elif len(set(eval_labels.tolist())) < 2:
        reasons.append("eval split has fewer than two label classes")
    return reasons


def _status_for_quality(metrics: Mapping[str, object]) -> tuple[BasinCriticStatus, list[str]]:
    auc = _metric_float(metrics.get("eval_auc"))
    rank = _metric_float(metrics.get("eval_pairwise_rank_accuracy"))
    loss = _metric_float(metrics.get("eval_log_loss"))
    reasons: list[str] = []
    if auc is None or rank is None or loss is None:
        return "inconclusive", ["missing finite quality metric"]
    if auc <= 0.5:
        reasons.append(f"eval AUC {auc:.6g} <= chance")
    if rank <= 0.5:
        reasons.append(f"pairwise rank accuracy {rank:.6g} <= chance")
    if not math.isfinite(loss):
        reasons.append("eval log loss is not finite")
    return ("fail", reasons) if reasons else ("pass", [])


def _metric_float(value: object) -> float | None:
    if value in {"", None} or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _replay_weight_target_source(
    examples: pd.DataFrame,
    *,
    config: BasinReplayWeightScorerTrainingConfig,
) -> tuple[np.ndarray, str, bool]:
    if config.target_mode == "magnitude":
        target_column = config.target_column
        absolute = True
    elif config.target_mode == "future_basin_motion":
        target_column = "future_basin_score_delta"
        absolute = False
    else:
        raise ValueError(
            "Unsupported basin replay weight target_mode: "
            f"{config.target_mode!r}"
        )
    if target_column not in examples.columns:
        raise ValueError(
            "Basin replay weight scorer samples missing target field: "
            f"{target_column}"
        )
    return (
        pd.to_numeric(examples[target_column], errors="coerce").to_numpy(
            dtype=np.float64
        ),
        target_column,
        absolute,
    )


def _replay_weight_target_residual_for_examples(
    examples: pd.DataFrame,
    *,
    config: BasinReplayWeightScorerTrainingConfig,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, str, float, dict[str, object]]:
    if config.target_mode == "intervention_pressure":
        if config.target_column not in examples.columns:
            raise ValueError(
                "Basin replay weight scorer samples missing target field: "
                f"{config.target_column}"
            )
        if "future_basin_score_delta" not in examples.columns:
            raise ValueError(
                "Basin replay weight scorer samples missing target field: "
                "future_basin_score_delta"
            )
        magnitude_residual, magnitude_scale = _replay_weight_target_residual(
            pd.to_numeric(
                examples[config.target_column],
                errors="coerce",
            ).to_numpy(dtype=np.float64),
            train_mask=train_mask,
            quantile=float(config.target_quantile),
            absolute=True,
        )
        future_residual, future_scale = _replay_weight_target_residual(
            pd.to_numeric(
                examples["future_basin_score_delta"],
                errors="coerce",
            ).to_numpy(dtype=np.float64),
            train_mask=train_mask,
            quantile=float(config.target_quantile),
            absolute=False,
        )
        return (
            np.maximum(magnitude_residual, future_residual),
            f"{config.target_column}+future_basin_score_delta",
            float(max(magnitude_scale, future_scale)),
            {
                "target_scale_magnitude": magnitude_scale,
                "target_scale_future_basin_motion": future_scale,
            },
        )

    target_source, target_column, target_absolute = _replay_weight_target_source(
        examples,
        config=config,
    )
    target_residual, target_scale = _replay_weight_target_residual(
        target_source,
        train_mask=train_mask,
        quantile=float(config.target_quantile),
        absolute=target_absolute,
    )
    return target_residual, target_column, target_scale, {}


def _replay_weight_target_residual(
    target_source: np.ndarray,
    *,
    train_mask: np.ndarray,
    quantile: float,
    absolute: bool,
) -> tuple[np.ndarray, float]:
    values = np.asarray(target_source, dtype=np.float64)
    target = np.abs(values) if absolute else np.maximum(values, 0.0)
    train_values = target[train_mask & np.isfinite(target)]
    positive = train_values[train_values > 0.0]
    if positive.size == 0:
        raise ValueError("Replay weight scorer target has no positive train values")
    scale = float(np.quantile(positive, np.clip(float(quantile), 0.0, 1.0)))
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(positive))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Replay weight scorer target scale is not finite")
    return np.clip(target / scale, 0.0, 1.0), scale


def _floor_target(
    residual: np.ndarray,
    *,
    config: BasinReplayWeightScorerTrainingConfig,
) -> np.ndarray:
    floor = float(config.output_floor)
    ceiling = float(config.output_ceiling)
    return np.clip(floor + (ceiling - floor) * residual, 0.0, 1.0)


def _format_metric(value: object) -> str:
    number = _metric_float(value)
    return "" if number is None else f"{number:.6g}"


def _format_sequence(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _sequence(raw: Mapping[str, object], key: str) -> Sequence[object]:
    value = raw[key]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"Basin phase critic manifest field {key} must be a list")
    return value


def _critic_case_from_mapping(raw: object) -> BasinPhaseCriticCase:
    if not isinstance(raw, Mapping):
        raise ValueError("Basin phase critic case must be a mapping")
    required = {"name", "toy", "ceiling_tolerance"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(
            "Basin phase critic case missing field(s): " + ", ".join(missing)
        )
    return BasinPhaseCriticCase(
        name=str(raw["name"]),
        toy=str(raw["toy"]),
        ceiling_tolerance=float(raw["ceiling_tolerance"]),
        label_mode=str(raw.get("label_mode", "ceiling_horizon")),
        label_horizon=int(raw.get("label_horizon", 5)),
        group=str(raw.get("group", "nabm")),
    )
