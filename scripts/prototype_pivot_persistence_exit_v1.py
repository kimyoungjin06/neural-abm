#!/usr/bin/env python3
"""Run the precommitted Pivot persistence-and-exit prototype v1."""

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
    / "pivot_persistence_exit_prototype_v1.yaml"
)
for import_path in (ROOT, SCRIPTS, EXAMPLES):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import prototype_pivot_delayed_feedback_learning_v1 as delayed  # noqa: E402
import research_pivot_learning_study as study2  # noqa: E402
import research_pivot_study as study1  # noqa: E402


FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
NOT_YET_ENTERED = "NOT_YET_ENTERED"
ACTIVE_PRODUCTIVE = "ACTIVE_PRODUCTIVE"
ACTIVE_FAILED = "ACTIVE_FAILED"
EXITED_OR_REDIRECTED = "EXITED_OR_REDIRECTED"
STATE_MODELS = ("absorbing_entry", "persistence_exit_v1")
FINAL_METRICS = (
    "cumulative_entry_rate",
    "active_rate",
    "active_productive_rate",
    "active_failed_rate",
    "exited_failed_rate",
    "productive_share_of_active",
)
TRAJECTORY_FIELDS = (
    "cumulative_entry_rate_trajectory",
    "active_rate_trajectory",
    "active_productive_rate_trajectory",
    "active_failed_rate_trajectory",
    "exited_failed_rate_trajectory",
    "new_failed_exits_per_step",
    "public_success_per_step",
    "public_failure_per_step",
    "mean_attention_weight",
)


@dataclass
class PersistenceCommitAdapter:
    """Pivot-owned entry plus a nonabsorbing failed-persistence endpoint."""

    agents: Sequence[study2.PivotPolicyAgent]
    environment: delayed.DelayedPivotEnvironment
    minimum_active_steps: int
    fit_weight: float
    policy_weight: float
    exit_below: float
    states: list[str] = field(init=False)
    entry_steps: list[int | None] = field(init=False)
    exit_steps: list[int | None] = field(init=False)
    exits_this_step: list[int] = field(default_factory=list)
    continuation_scores_at_exit: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.states = [NOT_YET_ENTERED for _agent in self.agents]
        self.entry_steps = [None for _agent in self.agents]
        self.exit_steps = [None for _agent in self.agents]

    def _evaluate_failed_exit(
        self,
        index: int,
        agent: study2.PivotPolicyAgent,
    ) -> None:
        if self.states[index] != ACTIVE_FAILED:
            return
        entry_step = self.entry_steps[index]
        if entry_step is None:
            raise RuntimeError("active failed agent is missing entry_step")
        if self.environment.step - entry_step < self.minimum_active_steps:
            return
        continuation = (
            self.fit_weight * float(agent.base.productive_fit)
            + self.policy_weight * float(agent.last_policy_output)
        )
        if continuation >= self.exit_below:
            return
        self.states[index] = EXITED_OR_REDIRECTED
        self.exit_steps[index] = self.environment.step
        self.exits_this_step.append(agent.agent_id)
        self.continuation_scores_at_exit.append(float(continuation))

    def commit(self, mix_result: Any) -> Any:
        config = self.environment.config
        values = np.asarray(mix_result.mixed_values, dtype=np.float64)
        committed: list[int] = []
        losses = [0.0 for _agent in self.agents]
        transitions_open = self.environment.step > config.burn_in_steps
        self.exits_this_step = []
        for index, agent in enumerate(self.agents):
            if agent.base.pivoted:
                self._evaluate_failed_exit(index, agent)
                continue
            before = float(agent.base.pivot_readiness)
            after = float(np.clip(values[index], 0.0, 1.0))
            agent.base.pivot_readiness = after
            losses[index] = abs(after - before)
            committed.append(agent.agent_id)
            if transitions_open and after >= config.pivot_threshold:
                agent.base.pivoted = True
                agent.base.productive_pivot = (
                    agent.base.productive_fit >= config.productive_threshold
                )
                agent.pivot_step = self.environment.step
                self.entry_steps[index] = self.environment.step
                self.states[index] = (
                    ACTIVE_PRODUCTIVE
                    if agent.base.productive_pivot
                    else ACTIVE_FAILED
                )
                self.environment.new_events.append(
                    study2.PivotEvent(
                        agent_id=agent.agent_id,
                        features=agent.features.copy(),
                        productive=agent.base.productive_pivot,
                        step=self.environment.step,
                    )
                )
        return study2.CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
            losses=losses,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("persistence prototype contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_persistence_exit_prototype.v1":
        raise ValueError("unexpected persistence prototype schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("prototype is not in the precommit design state")
    if [row["id"] for row in contract.get("state_models", [])] != list(
        STATE_MODELS
    ):
        raise ValueError("state model ladder changed")
    if contract.get("learning_arms") != ["frozen", "imitative", "cautionary"]:
        raise ValueError("learning arms changed")
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
    transition = contract.get("transition_contract", {})
    score = transition.get("failed_continuation_score", {})
    if score != {
        "productive_fit_weight": 0.5,
        "current_policy_output_weight": 0.5,
        "exit_below": 0.4,
    }:
        raise ValueError("failed continuation score changed")
    if transition.get("failed_minimum_active_steps") != 2:
        raise ValueError("minimum active duration changed")
    if transition.get("reentry_after_exit") is not False:
        raise ValueError("reentry is outside v1")
    if contract.get("reporting", {}).get("decision_thresholds") != (
        "NONE_DESCRIPTIVE_ONLY"
    ):
        raise ValueError("v1 must remain descriptive")
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
        arms=tuple(contract["learning_arms"]),
    )


def _scenario(contract: Mapping[str, Any]) -> Any:
    name = contract["scenario"]["name"]
    return next(row for row in study1.study_scenarios() if row.name == name)


def _regimes(contract: Mapping[str, Any]) -> list[delayed.VisibilityRegime]:
    return [
        delayed.VisibilityRegime.from_mapping(row)
        for row in contract["visibility_regimes"]
    ]


def run_persistence_arm(
    *,
    scenario: Any,
    config: study2.LearningStudyConfig,
    visibility: delayed.VisibilityRegime,
    transition: Mapping[str, Any],
    replicate: int,
    learning_mode: str,
) -> dict[str, Any]:
    """Run the nonabsorbing endpoint while preserving predecessor entry."""

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
        study2.PivotPolicyAgent(
            agent_id=researcher.agent_id,
            base=researcher,
            environment=environment,
            learning_rate=(
                0.0 if learning_mode == "frozen" else config.learning_rate
            ),
            learning_mode=learning_mode,
            prior_anchor=config.prior_anchor,
        )
        for researcher in researchers
    ]
    score = transition["failed_continuation_score"]
    adapter = PersistenceCommitAdapter(
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
    mean_attention_weight: list[float] = []
    for _step in range(config.steps):
        environment.advance()
        unit.run(collect_logs=False)
        environment.ingest_private_events()
        entered = sum(state != NOT_YET_ENTERED for state in adapter.states)
        productive = sum(state == ACTIVE_PRODUCTIVE for state in adapter.states)
        failed = sum(state == ACTIVE_FAILED for state in adapter.states)
        exited = sum(state == EXITED_OR_REDIRECTED for state in adapter.states)
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
        mean_attention_weight.append(
            float(np.mean([agent.attention_weight() for agent in agents]))
        )

    final_active = sum(
        state in {ACTIVE_PRODUCTIVE, ACTIVE_FAILED} for state in adapter.states
    )
    final_productive = sum(
        state == ACTIVE_PRODUCTIVE for state in adapter.states
    )
    final_failed = sum(state == ACTIVE_FAILED for state in adapter.states)
    final_exited = sum(
        state == EXITED_OR_REDIRECTED for state in adapter.states
    )
    final_entered = final_active + final_exited
    return {
        "cumulative_entry_rate": final_entered / count,
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
        "mean_attention_weight": mean_attention_weight,
        "continuation_scores_at_exit": [
            round(value, 10) for value in adapter.continuation_scores_at_exit
        ],
    }


def _normalize_absorbing(
    run: Mapping[str, Any],
    steps: int,
    agent_count: int,
) -> dict[str, Any]:
    cumulative_entry = [
        (
            round(productive * agent_count) + round(failed * agent_count)
        )
        / agent_count
        for productive, failed in zip(
            run["cumulative_productive_rate"],
            run["cumulative_failed_rate"],
            strict=True,
        )
    ]
    return {
        "cumulative_entry_rate": float(run["pivot_rate"]),
        "active_rate": float(run["pivot_rate"]),
        "active_productive_rate": float(run["productive_pivot_rate"]),
        "active_failed_rate": float(run["failed_pivot_rate"]),
        "exited_failed_rate": 0.0,
        "productive_share_of_active": float(run["productive_share_of_pivots"]),
        "cumulative_entry_rate_trajectory": cumulative_entry,
        "active_rate_trajectory": cumulative_entry,
        "active_productive_rate_trajectory": list(
            run["cumulative_productive_rate"]
        ),
        "active_failed_rate_trajectory": list(run["cumulative_failed_rate"]),
        "exited_failed_rate_trajectory": [0.0 for _step in range(steps)],
        "new_failed_exits_per_step": [0 for _step in range(steps)],
        "public_success_per_step": list(run["public_success_per_step"]),
        "public_failure_per_step": list(run["public_failure_per_step"]),
        "mean_attention_weight": list(run["mean_attention_weight"]),
        "continuation_scores_at_exit": [],
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
    state_model: str,
    regime: str,
    arm: str,
    reference_state_model: str,
    reference_regime: str,
    reference_arm: str,
    metric: str,
    values: Sequence[float],
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "state_model": state_model,
        "regime": regime,
        "arm": arm,
        "reference_state_model": reference_state_model,
        "reference_regime": reference_regime,
        "reference_arm": reference_arm,
        "metric": metric,
        "sign_counts": {
            "negative": sum(value < 0.0 for value in values),
            "zero": sum(value == 0.0 for value in values),
            "positive": sum(value > 0.0 for value in values),
        },
        "paired_delta": _distribution(values),
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(row["state_model"], row["regime"], row["arm"])].append(row)

    cells: list[dict[str, Any]] = []
    for state_model in STATE_MODELS:
        for regime in _regimes(contract):
            for arm in contract["learning_arms"]:
                cell_rows = grouped[(state_model, regime.regime_id, arm)]
                cells.append(
                    {
                        "state_model": state_model,
                        "regime": regime.regime_id,
                        "arm": arm,
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
                            for metric in TRAJECTORY_FIELDS
                        },
                    }
                )

    comparisons: list[dict[str, Any]] = []
    for regime in _regimes(contract):
        for arm in contract["learning_arms"]:
            target = {
                int(row["replicate"]): row
                for row in grouped[("persistence_exit_v1", regime.regime_id, arm)]
            }
            reference = {
                int(row["replicate"]): row
                for row in grouped[("absorbing_entry", regime.regime_id, arm)]
            }
            for metric in FINAL_METRICS:
                values = [
                    float(target[index][metric])
                    - float(reference[index][metric])
                    for index in sorted(target)
                ]
                comparisons.append(
                    _paired_record(
                        comparison="persistence_minus_absorbing",
                        state_model="persistence_exit_v1",
                        regime=regime.regime_id,
                        arm=arm,
                        reference_state_model="absorbing_entry",
                        reference_regime=regime.regime_id,
                        reference_arm=arm,
                        metric=metric,
                        values=values,
                    )
                )
    for state_model in STATE_MODELS:
        for regime in _regimes(contract):
            if regime.regime_id == "immediate":
                continue
            for arm in contract["learning_arms"]:
                target = {
                    int(row["replicate"]): row
                    for row in grouped[(state_model, regime.regime_id, arm)]
                }
                reference = {
                    int(row["replicate"]): row
                    for row in grouped[(state_model, "immediate", arm)]
                }
                for metric in FINAL_METRICS:
                    values = [
                        float(target[index][metric])
                        - float(reference[index][metric])
                        for index in sorted(target)
                    ]
                    comparisons.append(
                        _paired_record(
                            comparison="delay_minus_immediate",
                            state_model=state_model,
                            regime=regime.regime_id,
                            arm=arm,
                            reference_state_model=state_model,
                            reference_regime="immediate",
                            reference_arm=arm,
                            metric=metric,
                            values=values,
                        )
                    )
    for state_model in STATE_MODELS:
        for regime in _regimes(contract):
            for arm in ("imitative", "cautionary"):
                target = {
                    int(row["replicate"]): row
                    for row in grouped[(state_model, regime.regime_id, arm)]
                }
                reference = {
                    int(row["replicate"]): row
                    for row in grouped[(state_model, regime.regime_id, "frozen")]
                }
                for metric in FINAL_METRICS:
                    values = [
                        float(target[index][metric])
                        - float(reference[index][metric])
                        for index in sorted(target)
                    ]
                    comparisons.append(
                        _paired_record(
                            comparison="learning_minus_frozen",
                            state_model=state_model,
                            regime=regime.regime_id,
                            arm=arm,
                            reference_state_model=state_model,
                            reference_regime=regime.regime_id,
                            reference_arm="frozen",
                            metric=metric,
                            values=values,
                        )
                    )

    return {
        "schema_version": "pivot_persistence_exit_map.v1",
        "status": "DESCRIPTIVE_SYNTHETIC_RESULT",
        "profile": profile,
        "scenario": contract["scenario"]["name"],
        "entry_invariance": {
            "status": "PASS",
            "paired_rows_checked": len(rows) // 2,
            "maximum_final_absolute_difference": 0.0,
            "maximum_trajectory_absolute_difference": 0.0,
        },
        "cells": cells,
        "paired_comparisons": comparisons,
        "registered_result_space": contract["registered_result_space"],
        "boundaries": {
            **contract["boundaries"],
            "model_selected": "NO",
            "scientific_effect": "NOT_ESTABLISHED",
        },
    }


def run_persistence_map(
    contract: Mapping[str, Any],
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = _study_config(contract, profile)
    scenario = _scenario(contract)
    rows: list[dict[str, Any]] = []
    transition = contract["transition_contract"]
    for replicate in range(config.replicates):
        for visibility in _regimes(contract):
            for arm in contract["learning_arms"]:
                predecessor = delayed.run_delayed_arm(
                    scenario=scenario,
                    config=config,
                    visibility=visibility,
                    replicate=replicate,
                    learning_mode=arm,
                )
                absorbing = _normalize_absorbing(
                    predecessor,
                    config.steps,
                    config.agent_count,
                )
                persistence = run_persistence_arm(
                    scenario=scenario,
                    config=config,
                    visibility=visibility,
                    transition=transition,
                    replicate=replicate,
                    learning_mode=arm,
                )
                if persistence["cumulative_entry_rate"] != absorbing[
                    "cumulative_entry_rate"
                ]:
                    raise RuntimeError("final cumulative entry invariance failed")
                if persistence["cumulative_entry_rate_trajectory"] != absorbing[
                    "cumulative_entry_rate_trajectory"
                ]:
                    raise RuntimeError("entry trajectory invariance failed")
                common = {
                    "replicate": replicate,
                    "regime": visibility.regime_id,
                    "arm": arm,
                }
                rows.append(
                    {"state_model": "absorbing_entry", **common, **absorbing}
                )
                rows.append(
                    {
                        "state_model": "persistence_exit_v1",
                        **common,
                        **persistence,
                    }
                )
    return rows, _summarize(rows, contract, profile)


def _write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    path.chmod(0o600)


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot persistence and exit map v1",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        "Status: `DESCRIPTIVE_SYNTHETIC_RESULT`",
        "",
        "| Visibility | Learning | Entry | Active | Active productive | "
        "Active failed | Exited failed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        if cell["state_model"] != "persistence_exit_v1":
            continue
        metrics = cell["metrics"]
        lines.append(
            "| {regime} | {arm} | {entry:.4f} | {active:.4f} | "
            "{productive:.4f} | {failed:.4f} | {exited:.4f} |".format(
                regime=cell["regime"],
                arm=cell["arm"],
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
            "Entry is exactly nested with the absorbing predecessor. Values "
            "describe a synthetic domain model; no empirical effect or model "
            "selection is established.",
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
    map_path = output_dir / "persistence_map.json"
    markdown_path = output_dir / "persistence_map.md"
    receipt_path = output_dir / "receipt.json"
    _write_exclusive(
        raw_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    _write_exclusive(
        map_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _write_exclusive(markdown_path, _markdown(summary))
    artifacts = [raw_path, map_path, markdown_path]
    receipt = {
        "schema_version": "pivot_persistence_exit_receipt.v1",
        "status": "COMPLETE",
        "precommit": precommit,
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": _sha256(contract_path),
        },
        "row_count": len(rows),
        "entry_invariance": summary["entry_invariance"],
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


def _plan_payload(
    contract: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_path": str(config_path.resolve()),
        "contract_sha256": _sha256(config_path),
        "state_models": list(STATE_MODELS),
        "regimes": [regime.regime_id for regime in _regimes(contract)],
        "learning_arms": contract["learning_arms"],
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
    rows, summary = run_persistence_map(contract, args.profile)
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
                "entry_invariance": receipt["entry_invariance"],
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
