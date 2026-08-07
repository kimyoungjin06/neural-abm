#!/usr/bin/env python3
"""Run the precommitted fixed, delta, and neural Pivot model ladder v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXAMPLES = ROOT / "examples"
DEFAULT_CONFIG = (
    ROOT
    / "experiments"
    / "configs"
    / "pivot_adaptive_baseline_ladder_v1.yaml"
)
for import_path in (ROOT, SCRIPTS, EXAMPLES):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import prototype_pivot_delayed_feedback_learning_v1 as delayed  # noqa: E402
import prototype_pivot_persistence_exit_v1 as persistence  # noqa: E402
import research_pivot_learning_study as study2  # noqa: E402
import research_pivot_study as study1  # noqa: E402


FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
MODEL_IDS = (
    "fixed_rule",
    "delta_imitative",
    "delta_cautionary",
    "neural_imitative",
    "neural_cautionary",
)
FINAL_METRICS = persistence.FINAL_METRICS
DISTANCE_TRAJECTORIES = (
    "cumulative_entry_rate_trajectory",
    "active_productive_rate_trajectory",
    "active_failed_rate_trajectory",
    "exited_failed_rate_trajectory",
)


@dataclass
class DeltaPolicyAgent(study2.PivotPolicyAgent):
    """One-state, explicit, non-gradient adaptive policy."""

    delta_eta: float = 0.5
    delta_prior_anchor: float = 0.25
    initial_belief: float = field(init=False)
    adaptive_belief: float = field(init=False)
    evidence_batch_count: int = field(init=False, default=0)
    evidence_event_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.features = study2._feature_vector(self.base)
        self.prior_weight = self.weight.detach().clone()
        self.prior_bias = self.bias.detach().clone()
        with study2.torch.no_grad():
            initial = float(study2.torch.sigmoid(self.policy_logit(self.features)))
        self.initial_belief = initial
        self.adaptive_belief = initial
        self.last_policy_output = initial

    def policy_output(self, features: np.ndarray) -> float:
        del features
        return float(self.adaptive_belief)

    def attention_weight(self) -> float:
        return float(self.adaptive_belief)

    def act_or_predict(self, observation: Any) -> Any:
        del observation
        return study2.torch.tensor(
            [self.adaptive_belief],
            dtype=study2.torch.float64,
        )

    def _learn_from_events(
        self,
        events: Sequence[study2.PivotEvent],
    ) -> None:
        neighbor_ids = set(self.environment.neighbors[self.agent_id])
        samples = [
            event
            for event in events
            if event.agent_id != self.agent_id and event.agent_id in neighbor_ids
        ]
        if self.learning_mode == "cautionary":
            samples = [event for event in samples if not event.productive]
        if not samples:
            return
        target = float(np.mean([1.0 if event.productive else 0.0 for event in samples]))
        belief = float(self.adaptive_belief)
        updated = (
            belief
            + self.delta_eta * (target - belief)
            + self.delta_prior_anchor * (self.initial_belief - belief)
        )
        self.adaptive_belief = float(np.clip(updated, 0.0, 1.0))
        self.evidence_batch_count += 1
        self.evidence_event_count += len(samples)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("adaptive baseline contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_adaptive_baseline_ladder.v1":
        raise ValueError("unexpected adaptive baseline schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("ladder is not in the precommit design state")
    if [row["id"] for row in contract.get("model_ladder", [])] != list(MODEL_IDS):
        raise ValueError("model ladder changed")
    delta_rule = contract.get("delta_rule", {})
    if delta_rule.get("eta") != 0.5 or delta_rule.get("prior_anchor") != 0.25:
        raise ValueError("delta update parameters changed")
    if delta_rule.get("feature_specific_weights") is not False:
        raise ValueError("delta baseline must remain one-state")
    if delta_rule.get("backpropagation") is not False:
        raise ValueError("delta baseline must remain non-gradient")
    regimes = contract.get("visibility_regimes", [])
    if [row["id"] for row in regimes] != [
        "immediate",
        "failure_lag_2",
        "failure_lag_4",
        "selective_failure_lag_4",
    ]:
        raise ValueError("visibility regimes changed")
    for row in regimes:
        delayed.VisibilityRegime.from_mapping(row)
    targets = contract.get("common_target_vector", {})
    if tuple(targets.get("final", ())) != FINAL_METRICS:
        raise ValueError("final target vector changed")
    if targets.get("decision_thresholds") != "NONE_DESCRIPTIVE_ONLY":
        raise ValueError("ladder must remain threshold-free")
    if targets.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this ladder")
    boundaries = contract.get("boundaries", {})
    if boundaries.get("generic_nabm_api_changed") is not False:
        raise ValueError("generic NABM API change is outside v1")
    if boundaries.get("empirical_rows_read") != 0:
        raise ValueError("empirical rows are outside v1")


def _study_config(contract: Mapping[str, Any], profile: str) -> Any:
    try:
        profile_values = dict(contract["profiles"][profile])
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile}") from exc
    return study2.LearningStudyConfig(
        **profile_values,
        **dict(contract["inherited_learning_parameters"]),
        arms=("frozen", "imitative", "cautionary"),
    )


def _scenario(contract: Mapping[str, Any]) -> Any:
    name = contract["common_environment"]["scenario"]
    return next(row for row in study1.study_scenarios() if row.name == name)


def _regimes(contract: Mapping[str, Any]) -> list[delayed.VisibilityRegime]:
    return [
        delayed.VisibilityRegime.from_mapping(row)
        for row in contract["visibility_regimes"]
    ]


def _model_spec(model_id: str) -> tuple[str, str]:
    if model_id == "fixed_rule":
        return "fixed", "frozen"
    family, mode = model_id.split("_", maxsplit=1)
    return family, mode


def _run_delta_arm(
    *,
    scenario: Any,
    config: study2.LearningStudyConfig,
    visibility: delayed.VisibilityRegime,
    transition: Mapping[str, Any],
    delta_rule: Mapping[str, Any],
    replicate: int,
    learning_mode: str,
) -> dict[str, Any]:
    rng = study1.component_rng(config.base_seed, replicate, "population")
    study_config = study1.StudyConfig(
        agent_count=config.agent_count,
        steps=config.steps,
        replicates=config.replicates,
        base_seed=config.base_seed,
        neighbor_count=config.neighbor_count,
        within_stage_probability=config.within_stage_probability,
        local_alpha=config.local_alpha,
        local_noise_scale=config.local_noise_scale,
        social_alpha=config.social_alpha,
        peer_similarity_threshold=config.peer_similarity_threshold,
        pivot_threshold=config.pivot_threshold,
        productive_threshold=config.productive_threshold,
    )
    context = SimpleNamespace(scenario=scenario, replicate=replicate, rng=rng)
    researchers = study1.make_researchers(context, study_config)
    neighbors = study1.build_stage_assortative_network(
        researchers,
        context,
        study_config,
    )
    environment = delayed.DelayedPivotEnvironment(
        neighbors=neighbors,
        config=config,
        replicate=replicate,
        visibility=visibility,
    )
    agents = [
        DeltaPolicyAgent(
            agent_id=researcher.agent_id,
            base=researcher,
            environment=environment,
            learning_rate=float(delta_rule["eta"]),
            learning_mode=learning_mode,
            prior_anchor=0.0,
            delta_eta=float(delta_rule["eta"]),
            delta_prior_anchor=float(delta_rule["prior_anchor"]),
        )
        for researcher in researchers
    ]
    score = transition["failed_continuation_score"]
    adapter = persistence.PersistenceCommitAdapter(
        agents=agents,
        environment=environment,
        minimum_active_steps=int(transition["failed_minimum_active_steps"]),
        fit_weight=float(score["productive_fit_weight"]),
        policy_weight=float(score["current_policy_output_weight"]),
        exit_below=float(score["exit_below"]),
    )
    unit = study2.NABMUnit(
        agents=agents,
        step=study2.NABMStep(
            social_block=study2.SocialBlock(alpha=config.social_alpha),
            channel=study2.SocialChannel(
                name="pivot_readiness",
                kind=study2.BOUNDED_SCALAR_CHANNEL,
                commit_mode="domain_pivot_persistence_exit",
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            commit_adapter=adapter,
        ),
        peer_selector=study2._make_peer_selector(environment, config),
        social_value_builder=study2.scalar_message_values("pivot_readiness"),
    )

    count = len(agents)
    cumulative_entry: list[float] = []
    active_rate: list[float] = []
    active_productive: list[float] = []
    active_failed: list[float] = []
    exited_failed: list[float] = []
    new_failed_exits: list[int] = []
    public_success: list[int] = []
    public_failure: list[int] = []
    mean_policy_state: list[float] = []
    for _step in range(config.steps):
        environment.advance()
        unit.run(collect_logs=False)
        environment.ingest_private_events()
        entered = sum(
            state != persistence.NOT_YET_ENTERED for state in adapter.states
        )
        productive = sum(
            state == persistence.ACTIVE_PRODUCTIVE for state in adapter.states
        )
        failed = sum(state == persistence.ACTIVE_FAILED for state in adapter.states)
        exited = sum(
            state == persistence.EXITED_OR_REDIRECTED for state in adapter.states
        )
        cumulative_entry.append(entered / count)
        active_productive.append(productive / count)
        active_failed.append(failed / count)
        active_rate.append((productive + failed) / count)
        exited_failed.append(exited / count)
        new_failed_exits.append(len(adapter.exits_this_step))
        public_success.append(sum(event.productive for event in environment.last_events))
        public_failure.append(
            sum(not event.productive for event in environment.last_events)
        )
        mean_policy_state.append(
            float(np.mean([agent.adaptive_belief for agent in agents]))
        )

    final_active = sum(
        state in {persistence.ACTIVE_PRODUCTIVE, persistence.ACTIVE_FAILED}
        for state in adapter.states
    )
    final_productive = sum(
        state == persistence.ACTIVE_PRODUCTIVE for state in adapter.states
    )
    final_failed = sum(
        state == persistence.ACTIVE_FAILED for state in adapter.states
    )
    final_exited = sum(
        state == persistence.EXITED_OR_REDIRECTED for state in adapter.states
    )
    return {
        "cumulative_entry_rate": (final_active + final_exited) / count,
        "active_rate": final_active / count,
        "active_productive_rate": final_productive / count,
        "active_failed_rate": final_failed / count,
        "exited_failed_rate": final_exited / count,
        "productive_share_of_active": (
            final_productive / final_active if final_active else 0.0
        ),
        "cumulative_entry_rate_trajectory": cumulative_entry,
        "active_rate_trajectory": active_rate,
        "active_productive_rate_trajectory": active_productive,
        "active_failed_rate_trajectory": active_failed,
        "exited_failed_rate_trajectory": exited_failed,
        "new_failed_exits_per_step": new_failed_exits,
        "public_success_per_step": public_success,
        "public_failure_per_step": public_failure,
        "mean_policy_state": mean_policy_state,
        "evidence_batch_count": sum(
            agent.evidence_batch_count for agent in agents
        ),
        "evidence_event_count": sum(
            agent.evidence_event_count for agent in agents
        ),
        "adaptive_state_count_per_agent": 1,
        "gradient_based": False,
    }


def run_model(
    *,
    model_id: str,
    scenario: Any,
    config: study2.LearningStudyConfig,
    visibility: delayed.VisibilityRegime,
    contract: Mapping[str, Any],
    replicate: int,
) -> dict[str, Any]:
    family, learning_mode = _model_spec(model_id)
    if family == "delta":
        return _run_delta_arm(
            scenario=scenario,
            config=config,
            visibility=visibility,
            transition=contract["common_environment_transition"],
            delta_rule=contract["delta_rule"],
            replicate=replicate,
            learning_mode=learning_mode,
        )
    run = persistence.run_persistence_arm(
        scenario=scenario,
        config=config,
        visibility=visibility,
        transition=contract["common_environment_transition"],
        replicate=replicate,
        learning_mode=learning_mode,
    )
    return {
        **run,
        "adaptive_state_count_per_agent": 0 if family == "fixed" else 10,
        "gradient_based": family == "neural",
    }


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


def _paired_record(
    *,
    comparison: str,
    model_id: str,
    regime: str,
    reference_model_id: str,
    reference_regime: str,
    metric: str,
    values: Sequence[float],
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "model_id": model_id,
        "regime": regime,
        "reference_model_id": reference_model_id,
        "reference_regime": reference_regime,
        "metric": metric,
        "sign_counts": {
            "negative": sum(value < 0.0 for value in values),
            "zero": sum(value == 0.0 for value in values),
            "positive": sum(value > 0.0 for value in values),
        },
        "paired_delta": _distribution(values),
    }


def _model_distances(
    *,
    grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    regimes: Sequence[delayed.VisibilityRegime],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for regime in regimes:
        for mode in ("imitative", "cautionary"):
            delta_rows = {
                int(row["replicate"]): row
                for row in grouped[(f"delta_{mode}", regime.regime_id)]
            }
            neural_rows = {
                int(row["replicate"]): row
                for row in grouped[(f"neural_{mode}", regime.regime_id)]
            }
            final_mae: list[float] = []
            trajectory_rmse: list[float] = []
            for replicate in sorted(delta_rows):
                delta_row = delta_rows[replicate]
                neural_row = neural_rows[replicate]
                final_mae.append(
                    float(
                        np.mean(
                            [
                                abs(
                                    float(delta_row[metric])
                                    - float(neural_row[metric])
                                )
                                for metric in FINAL_METRICS
                            ]
                        )
                    )
                )
                squared: list[float] = []
                for metric in DISTANCE_TRAJECTORIES:
                    delta_values = np.asarray(delta_row[metric], dtype=np.float64)
                    neural_values = np.asarray(neural_row[metric], dtype=np.float64)
                    squared.extend(((delta_values - neural_values) ** 2).tolist())
                trajectory_rmse.append(float(np.sqrt(np.mean(squared))))
            reports.append(
                {
                    "regime": regime.regime_id,
                    "evidence_mode": mode,
                    "delta_model": f"delta_{mode}",
                    "neural_model": f"neural_{mode}",
                    "final_vector_mae": _distribution(final_mae),
                    "trajectory_rmse": _distribution(trajectory_rmse),
                    "mean_final_vector_delta": {
                        metric: round(
                            float(
                                np.mean(
                                    [
                                        float(delta_rows[index][metric])
                                        - float(neural_rows[index][metric])
                                        for index in sorted(delta_rows)
                                    ]
                                )
                            ),
                            8,
                        )
                        for metric in FINAL_METRICS
                    },
                }
            )
    return reports


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["regime"])].append(row)
    regimes = _regimes(contract)
    cells: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for regime in regimes:
            cell_rows = grouped[(model_id, regime.regime_id)]
            cells.append(
                {
                    "model_id": model_id,
                    "regime": regime.regime_id,
                    "replicates": len(cell_rows),
                    "metrics": {
                        metric: _distribution(
                            [float(row[metric]) for row in cell_rows]
                        )
                        for metric in FINAL_METRICS
                    },
                    "mean_trajectories": {
                        metric: [
                            round(float(value), 8)
                            for value in np.asarray(
                                [row[metric] for row in cell_rows],
                                dtype=np.float64,
                            ).mean(axis=0)
                        ]
                        for metric in DISTANCE_TRAJECTORIES
                    },
                }
            )

    comparisons: list[dict[str, Any]] = []
    for regime in regimes:
        fixed = {
            int(row["replicate"]): row
            for row in grouped[("fixed_rule", regime.regime_id)]
        }
        for model_id in MODEL_IDS[1:]:
            target = {
                int(row["replicate"]): row
                for row in grouped[(model_id, regime.regime_id)]
            }
            for metric in FINAL_METRICS:
                values = [
                    float(target[index][metric]) - float(fixed[index][metric])
                    for index in sorted(target)
                ]
                comparisons.append(
                    _paired_record(
                        comparison="adaptive_minus_fixed",
                        model_id=model_id,
                        regime=regime.regime_id,
                        reference_model_id="fixed_rule",
                        reference_regime=regime.regime_id,
                        metric=metric,
                        values=values,
                    )
                )
    for regime in regimes:
        for mode in ("imitative", "cautionary"):
            delta_rows = {
                int(row["replicate"]): row
                for row in grouped[(f"delta_{mode}", regime.regime_id)]
            }
            neural_rows = {
                int(row["replicate"]): row
                for row in grouped[(f"neural_{mode}", regime.regime_id)]
            }
            for metric in FINAL_METRICS:
                values = [
                    float(delta_rows[index][metric])
                    - float(neural_rows[index][metric])
                    for index in sorted(delta_rows)
                ]
                comparisons.append(
                    _paired_record(
                        comparison="delta_minus_neural",
                        model_id=f"delta_{mode}",
                        regime=regime.regime_id,
                        reference_model_id=f"neural_{mode}",
                        reference_regime=regime.regime_id,
                        metric=metric,
                        values=values,
                    )
                )
    for model_id in MODEL_IDS:
        immediate = {
            int(row["replicate"]): row
            for row in grouped[(model_id, "immediate")]
        }
        for regime in regimes:
            if regime.regime_id == "immediate":
                continue
            target = {
                int(row["replicate"]): row
                for row in grouped[(model_id, regime.regime_id)]
            }
            for metric in FINAL_METRICS:
                values = [
                    float(target[index][metric])
                    - float(immediate[index][metric])
                    for index in sorted(target)
                ]
                comparisons.append(
                    _paired_record(
                        comparison="delay_minus_immediate",
                        model_id=model_id,
                        regime=regime.regime_id,
                        reference_model_id=model_id,
                        reference_regime="immediate",
                        metric=metric,
                        values=values,
                    )
                )
    return {
        "schema_version": "pivot_adaptive_baseline_scorecard.v1",
        "status": "DESCRIPTIVE_SYNTHETIC_MODEL_COMPARISON",
        "profile": profile,
        "scenario": contract["common_environment"]["scenario"],
        "cells": cells,
        "paired_comparisons": comparisons,
        "delta_neural_distances": _model_distances(
            grouped=grouped,
            regimes=regimes,
        ),
        "registered_result_space": contract["registered_result_space"],
        "boundaries": contract["boundaries"],
    }


def run_model_ladder(
    contract: Mapping[str, Any],
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = _study_config(contract, profile)
    scenario = _scenario(contract)
    rows: list[dict[str, Any]] = []
    for replicate in range(config.replicates):
        for regime in _regimes(contract):
            for model_id in MODEL_IDS:
                run = run_model(
                    model_id=model_id,
                    scenario=scenario,
                    config=config,
                    visibility=regime,
                    contract=contract,
                    replicate=replicate,
                )
                rows.append(
                    {
                        "replicate": replicate,
                        "regime": regime.regime_id,
                        "model_id": model_id,
                        **run,
                    }
                )
    return rows, _summarize(rows, contract, profile)


def _write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    path.chmod(0o600)


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot adaptive baseline model scorecard v1",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "Status: `DESCRIPTIVE_SYNTHETIC_MODEL_COMPARISON`",
        "",
        "| Visibility | Model | Entry | Active | Active productive | "
        "Active failed | Exited failed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        metrics = cell["metrics"]
        lines.append(
            "| {regime} | {model} | {entry:.4f} | {active:.4f} | "
            "{productive:.4f} | {failed:.4f} | {exited:.4f} |".format(
                regime=cell["regime"],
                model=cell["model_id"],
                entry=metrics["cumulative_entry_rate"]["mean"],
                active=metrics["active_rate"]["mean"],
                productive=metrics["active_productive_rate"]["mean"],
                failed=metrics["active_failed_rate"]["mean"],
                exited=metrics["exited_failed_rate"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "Distances are descriptive. No empirical fit, predictive "
            "validation, model selection, neural necessity, or scientific "
            "effect is established.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    precommit: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    raw_path = output_dir / "replicates.jsonl"
    scorecard_path = output_dir / "model_scorecard.json"
    markdown_path = output_dir / "model_scorecard.md"
    receipt_path = output_dir / "receipt.json"
    _write_exclusive(
        raw_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    _write_exclusive(
        scorecard_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _write_exclusive(markdown_path, _markdown(summary))
    artifacts = [raw_path, scorecard_path, markdown_path]
    receipt = {
        "schema_version": "pivot_adaptive_baseline_receipt.v1",
        "status": "COMPLETE",
        "precommit": precommit,
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": _sha256(contract_path),
        },
        "row_count": len(rows),
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
        "boundaries": summary["boundaries"],
    }
    _write_exclusive(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def _contract_with_transition(contract: Mapping[str, Any]) -> dict[str, Any]:
    parent = persistence.load_contract(
        ROOT / "experiments" / "configs" / "pivot_persistence_exit_prototype_v1.yaml"
    )
    return {**contract, "common_environment_transition": parent["transition_contract"]}


def _plan_payload(
    contract: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_path": str(config_path.resolve()),
        "contract_sha256": _sha256(config_path),
        "model_ids": list(MODEL_IDS),
        "regimes": [regime.regime_id for regime in _regimes(contract)],
        "profiles": contract["profiles"],
        "source_bindings": delayed.verify_source_bindings(contract),
        "boundaries": contract["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("smoke", "development"),
        default="development",
    )
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = _contract_with_transition(load_contract(config_path))
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
    rows, summary = run_model_ladder(contract, args.profile)
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
                "output_dir": str(args.output_dir.resolve()),
                "row_count": receipt["row_count"],
                "receipt_sha256": _sha256(
                    args.output_dir.resolve() / "receipt.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
