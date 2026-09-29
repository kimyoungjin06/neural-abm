#!/usr/bin/env python3
"""Run the precommitted Pivot relative-delay and cost study v1."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXAMPLES = ROOT / "examples"
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_relative_delay_cost_v1.yaml"
for import_path in (ROOT, SCRIPTS, EXAMPLES):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import prototype_pivot_nonlinear_policy_abm_v2 as abm2  # noqa: E402


abm1 = abm2.abm1
delayed = abm2.delayed
persistence = abm2.persistence
study1 = abm2.study1
study2 = abm2.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = abm1.FIT_WORLDS
INTERACTION_FIT_WORLDS = abm1.INTERACTION_FIT_WORLDS
VISIBILITY_IDS = ("immediate", "failure_lag_1", "failure_lag_2", "failure_lag_4")
# v2 components are kept so q = 1 reproduces v2 seeds exactly.
VISIBILITY_COMPONENT = {**abm2.VISIBILITY_COMPONENT, "failure_lag_1": 4}
Q_VALUES = (1.0, 0.5, 0.25)
MAIN_ARMS = (
    ("fixed_rule", "none"),
    ("linear_batch", "field_public"),
    ("tree_batch", "field_public"),
    ("mlp_batch", "field_public"),
)
BENCHMARK_ARMS = abm2.ARM_KEYS
FEATURE_ARMS = ("linear_batch", "tree_batch", "mlp_batch")
NONLINEAR_ARMS = ("tree_batch", "mlp_batch")
PREDECESSOR_MODES = abm2.PREDECESSOR_MODES
HAZARD_COMPONENT = 71
FIELD_FOLD_COMPONENT = abm2.FIELD_FOLD_COMPONENT
OPENED_SEEDS = {20261003, 20261004, 20261005, 20261010, 20261011, 20261012, 20261013}
IMPROVEMENT_RULE = abm1.IMPROVEMENT_RULE
DID_RULE = (
    "INTERACTION_FIT_WORLD_GAP_MINUS_LINEAR_FIT_WORLD_GAP_SAME_HAZARD_VISIBILITY_"
    "AND_REPLICATE"
)
SATURATION_ENTRY = 0.98
PERSISTENCE_DROP = 0.05


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("relative-delay contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_relative_delay_cost.v1":
        raise ValueError("unexpected relative-delay schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("relative-delay contract is not a precommit design")
    if tuple(contract["entry_hazard"]["q_values"]) != Q_VALUES:
        raise ValueError("entry hazard values changed")
    if [row["id"] for row in contract["visibility_regimes"]] != list(VISIBILITY_IDS):
        raise ValueError("visibility regimes changed")
    for row in contract["visibility_regimes"]:
        delayed.VisibilityRegime.from_mapping(row)
    if [row["id"] for row in contract["fit_worlds"]["worlds"]] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    if [tuple(row) for row in contract["main_run_arms"]] != list(MAIN_ARMS):
        raise ValueError("main-run arms changed")
    if [tuple(row) for row in contract["cost_ledger"]["arms"]] != list(BENCHMARK_ARMS):
        raise ValueError("benchmark arms changed")
    field_regime = contract["evidence_regimes"][1]
    if field_regime.get("includes_own_public_outcome") is not False:
        raise ValueError("the field record must exclude the agent's own outcome")
    v2_learners = abm2.load_contract()["batch_learners"]
    for arm_id in abm2.BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v2_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from v2")
    seeds = {
        int(contract["profiles"]["development"]["base_seed"]),
        int(contract["probe_cohort"]["base_seed"]),
    }
    if len(seeds) != 2 or seeds & OPENED_SEEDS:
        raise ValueError("the study needs fresh seed namespaces")
    decisions = contract["decision_contract"]
    if decisions.get("improvement_rule") != IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("difference_in_differences_rule") != DID_RULE:
        raise ValueError("difference-in-differences rule changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside this study")
    boundaries = contract.get("boundaries", {})
    if boundaries.get("generic_nabm_api_changed") is not False:
        raise ValueError("generic NABM API change is outside this study")
    if boundaries.get("empirical_rows_read") != 0:
        raise ValueError("empirical rows are outside this study")


def study_config(contract: Mapping[str, Any], profile: str) -> Any:
    return abm1.study_config(contract, profile)


# -- entry hazard -----------------------------------------------------------


@dataclass
class HazardCommitAdapter(persistence.PersistenceCommitAdapter):
    """The persistence adapter with an entry hazard; q = 1 is the predecessor."""

    q: float = 1.0
    base_seed: int = 0
    replicate: int = 0

    def admits(self, agent_id: int) -> bool:
        if self.q >= 1.0:
            return True
        seed = np.random.SeedSequence(
            (self.base_seed, self.replicate, HAZARD_COMPONENT, self.environment.step, agent_id)
        )
        return float(np.random.default_rng(seed).random()) < self.q

    def commit(self, mix_result: Any) -> Any:
        # Copied from the predecessor; only the admits() test is added.
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
            if (
                transitions_open
                and after >= config.pivot_threshold
                and self.admits(agent.agent_id)
            ):
                agent.base.pivoted = True
                agent.base.productive_pivot = (
                    agent.base.productive_fit >= config.productive_threshold
                )
                agent.pivot_step = self.environment.step
                self.entry_steps[index] = self.environment.step
                self.states[index] = (
                    persistence.ACTIVE_PRODUCTIVE
                    if agent.base.productive_pivot
                    else persistence.ACTIVE_FAILED
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


# -- simulation-side timing observables -----------------------------------------


def timing_observables(
    entry_steps: Sequence[int | None],
    exit_steps: Sequence[int | None],
    *,
    steps: int,
    first_open_step: int,
    horizon: int,
) -> dict[str, Any]:
    incidence = [
        sum(1 for entry in entry_steps if entry == step) for step in range(1, steps + 1)
    ]
    total = sum(incidence)
    peaks = (
        [step for step, count in enumerate(incidence, start=1) if count == max(incidence)]
        if total
        else []
    )
    cohorts = []
    for step, count in enumerate(incidence, start=1):
        if count == 0 or step + horizon > steps:
            continue
        persisting = sum(
            1
            for entry, exit_step in zip(entry_steps, exit_steps, strict=True)
            if entry == step and (exit_step is None or exit_step > step + horizon)
        )
        cohorts.append([step, count, persisting])
    onset = None
    if cohorts:
        first_share = cohorts[0][2] / cohorts[0][1]
        onset = next(
            (
                step
                for step, count, persisting in cohorts[1:]
                if persisting / count <= first_share - PERSISTENCE_DROP
            ),
            None,
        )
    width = None
    if total:
        cumulative = np.cumsum(incidence)
        ninety = int(np.argmax(cumulative >= 0.9 * total - 1e-12)) + 1
        width = ninety - first_open_step + 1
    return {
        "entry_incidence": incidence,
        "entry_peak_steps": peaks,
        "cohort_persistence": cohorts,
        "decline_onset_step": onset,
        "model_h3_gap": (min(peaks) - onset) if peaks and onset is not None else None,
        "entry_wave_width": width,
    }


# -- ABM run loop -------------------------------------------------------------


def _learner_seed(
    config: Any,
    *,
    replicate: int,
    world_id: str,
    visibility_id: str,
    evidence_regime: str,
    arm_id: str,
    agent_component: int,
) -> tuple[int, ...]:
    return (
        int(config.base_seed),
        replicate,
        abm2.WORLD_COMPONENT[world_id],
        VISIBILITY_COMPONENT[visibility_id],
        abm2.EVIDENCE_COMPONENT[evidence_regime],
        abm2.ARM_COMPONENT[arm_id],
        agent_component,
    )


def run_arm(
    contract: Mapping[str, Any],
    config: Any,
    *,
    replicate: int,
    visibility_id: str,
    world_id: str,
    arm_id: str,
    evidence_regime: str,
    probe: tuple[np.ndarray, np.ndarray],
    q: float,
) -> dict[str, Any]:
    """The v2 run loop with an entry hazard and timing observables."""

    probe_features, probe_labels = probe
    scenario = abm1._scenario(contract)
    visibility = abm1._visibility(contract, visibility_id)
    transition = contract["common_environment"]["transition"]
    rng = study1.component_rng(config.base_seed, replicate, "population")
    population_config = study1.StudyConfig(
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
    researchers = study1.make_researchers(context, population_config)
    neighbors = study1.build_stage_assortative_network(
        researchers, context, population_config
    )
    researchers = abm1.apply_fit_world(researchers, world_id)
    environment = delayed.DelayedPivotEnvironment(
        neighbors=neighbors,
        config=config,
        replicate=replicate,
        visibility=visibility,
    )
    all_features = np.stack([study2._feature_vector(row) for row in researchers])
    field_record: abm2.CrossFitField | None = None
    agents: list[study2.PivotPolicyAgent]
    if arm_id in PREDECESSOR_MODES:
        mode = PREDECESSOR_MODES[arm_id]
        agents = [
            study2.PivotPolicyAgent(
                agent_id=row.agent_id,
                base=row,
                environment=environment,
                learning_rate=0.0 if mode == "frozen" else config.learning_rate,
                learning_mode=mode,
                prior_anchor=config.prior_anchor,
            )
            for row in researchers
        ]
    else:
        settings = contract["batch_learners"][arm_id]
        seed = {
            "replicate": replicate,
            "world_id": world_id,
            "visibility_id": visibility_id,
            "evidence_regime": evidence_regime,
            "arm_id": arm_id,
        }
        if evidence_regime == "field_public":
            folds = int(contract["evidence_regimes"][1]["cross_fit_folds"])
            field_record = abm2.CrossFitField(
                learners=[
                    abm2.ProbedLearner(
                        arm_id=arm_id,
                        settings=settings,
                        seed_components=_learner_seed(
                            config, agent_component=FIELD_FOLD_COMPONENT + fold, **seed
                        ),
                        eval_features=all_features,
                        probe_features=probe_features,
                        excluded_agent_ids=frozenset(
                            row.agent_id
                            for row in researchers
                            if row.agent_id % folds == fold
                        ),
                    )
                    for fold in range(folds)
                ]
            )
        agents = [
            abm2.ProbedPolicyAgent(
                agent_id=row.agent_id,
                base=row,
                environment=environment,
                learning_rate=0.0,
                learning_mode="imitative",
                prior_anchor=0.0,
                learner=(
                    None
                    if field_record is not None
                    else abm2.ProbedLearner(
                        arm_id=arm_id,
                        settings=settings,
                        seed_components=_learner_seed(
                            config, agent_component=row.agent_id + 1, **seed
                        ),
                        eval_features=all_features[row.agent_id : row.agent_id + 1],
                        probe_features=probe_features,
                    )
                ),
                field_record=field_record,
            )
            for row in researchers
        ]
    score = transition["failed_continuation_score"]
    adapter = HazardCommitAdapter(
        agents=agents,
        environment=environment,
        minimum_active_steps=int(transition["failed_minimum_active_steps"]),
        fit_weight=float(score["productive_fit_weight"]),
        policy_weight=float(score["current_policy_output_weight"]),
        exit_below=float(score["exit_below"]),
        q=float(q),
        base_seed=int(config.base_seed),
        replicate=replicate,
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
    trajectories: dict[str, list[Any]] = defaultdict(list)
    pre_entry_decisions = 0
    for _step in range(config.steps):
        pre_entry_decisions += sum(
            state == persistence.NOT_YET_ENTERED for state in adapter.states
        )
        environment.advance()
        if field_record is not None:
            field_record.update(environment.last_events)
        unit.run(collect_logs=False)
        environment.ingest_private_events()
        states = adapter.states
        entered = sum(state != persistence.NOT_YET_ENTERED for state in states)
        productive = sum(state == persistence.ACTIVE_PRODUCTIVE for state in states)
        failed = sum(state == persistence.ACTIVE_FAILED for state in states)
        exited = sum(state == persistence.EXITED_OR_REDIRECTED for state in states)
        trajectories["cumulative_entry_rate_trajectory"].append(entered / count)
        trajectories["active_productive_rate_trajectory"].append(productive / count)
        trajectories["active_failed_rate_trajectory"].append(failed / count)
        trajectories["active_rate_trajectory"].append((productive + failed) / count)
        trajectories["exited_failed_rate_trajectory"].append(exited / count)
        trajectories["new_failed_exits_per_step"].append(len(adapter.exits_this_step))
        trajectories["public_success_per_step"].append(
            sum(event.productive for event in environment.last_events)
        )
        trajectories["public_failure_per_step"].append(
            sum(not event.productive for event in environment.last_events)
        )
        trajectories["mean_attention_weight"].append(
            float(np.mean([agent.attention_weight() for agent in agents]))
        )
        trajectories["mean_policy_output"].append(
            float(np.mean([agent.last_policy_output for agent in agents]))
        )

    states = adapter.states
    final_active = sum(
        state in {persistence.ACTIVE_PRODUCTIVE, persistence.ACTIVE_FAILED}
        for state in states
    )
    final_productive = sum(state == persistence.ACTIVE_PRODUCTIVE for state in states)
    final_failed = sum(state == persistence.ACTIVE_FAILED for state in states)
    final_exited = sum(state == persistence.EXITED_OR_REDIRECTED for state in states)
    final_entered = final_active + final_exited
    outputs = np.asarray([agent.last_policy_output for agent in agents])
    labels = np.asarray(
        [
            float(agent.base.productive_fit >= config.productive_threshold)
            for agent in agents
        ]
    )
    evidence_events: float | None
    if field_record is not None:
        field_record.check_isolation()
        probe_predictions = [learner.probe_output for learner in field_record.learners]
        evidence_events = float(
            np.mean([len(learner.labels) for learner in field_record.learners])
        )
    elif arm_id == "fixed_rule":
        probe_predictions = [abm1._prior_output(probe_features)]
        evidence_events = None
    elif arm_id == "online_logistic":
        with torch.no_grad():
            inputs = torch.as_tensor(probe_features, dtype=torch.float64)
            probe_predictions = [
                torch.sigmoid(inputs @ agent.weight + agent.bias).numpy()
                for agent in agents
            ]
        evidence_events = None
    else:
        probe_predictions = [agent.learner.probe_output for agent in agents]  # type: ignore[attr-defined]
        evidence_events = float(
            np.mean([len(agent.learner.labels) for agent in agents])  # type: ignore[attr-defined]
        )
    horizon = int(contract["h3_adapter"]["cohort_persistence_horizon_steps"])
    return {
        "cumulative_entry_rate": final_entered / count,
        "active_rate": final_active / count,
        "active_productive_rate": final_productive / count,
        "active_failed_rate": final_failed / count,
        "exited_failed_rate": final_exited / count,
        "productive_share_of_active": (
            final_productive / final_active if final_active else 0.0
        ),
        "productive_share_of_entrants": (
            final_productive / final_entered if final_entered else 0.0
        ),
        "failed_entry_rate": (final_failed + final_exited) / count,
        "probe_brier": float(
            np.mean(
                [abm2._brier(prediction, probe_labels) for prediction in probe_predictions]
            )
        ),
        "probe_auc": abm2._auc(np.mean(probe_predictions, axis=0), probe_labels),
        "in_situ_policy_brier": abm2._brier(outputs, labels),
        "evidence_events_per_learner": evidence_events,
        **dict(trajectories),
        "continuation_scores_at_exit": [
            round(value, 10) for value in adapter.continuation_scores_at_exit
        ],
        **timing_observables(
            adapter.entry_steps,
            adapter.exit_steps,
            steps=config.steps,
            first_open_step=config.burn_in_steps + 1,
            horizon=horizon,
        ),
        "policy_queries": count * config.steps,
        "pre_entry_decisions": int(pre_entry_decisions),
    }


def check_nesting(
    contract: Mapping[str, Any],
    config: Any,
    *,
    replicate: int,
    visibility_id: str,
    runs: Mapping[str, Mapping[str, Any]],
    probe: tuple[np.ndarray, np.ndarray],
) -> None:
    predecessor = persistence.run_persistence_arm(
        scenario=abm1._scenario(contract),
        config=config,
        visibility=abm1._visibility(contract, visibility_id),
        transition=contract["common_environment"]["transition"],
        replicate=replicate,
        learning_mode="frozen",
    )
    for key, value in predecessor.items():
        if runs["fixed_rule"][key] != value:
            raise RuntimeError(f"fixed_rule does not nest the persistence arm at {key}")
    if visibility_id != "immediate":
        return
    v2_run = abm2.run_arm(
        contract,
        config,
        replicate=replicate,
        visibility_id=visibility_id,
        world_id="linear_fit",
        arm_id="mlp_batch",
        evidence_regime="field_public",
        probe=probe,
    )
    for key, value in v2_run.items():
        if runs["mlp_batch"][key] != value:
            raise RuntimeError(f"mlp_batch does not nest the v2 arm at {key}")


def run_task(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    visibility_id: str,
    world_id: str,
    q: float,
) -> list[dict[str, Any]]:
    config = study_config(contract, profile)
    probe = abm2.probe_cohort(contract, replicate, world_id)
    rows = []
    runs: dict[str, dict[str, Any]] = {}
    for arm_id, evidence_regime in MAIN_ARMS:
        run = run_arm(
            contract,
            config,
            replicate=replicate,
            visibility_id=visibility_id,
            world_id=world_id,
            arm_id=arm_id,
            evidence_regime=evidence_regime,
            probe=probe,
            q=q,
        )
        runs[arm_id] = run
        rows.append(
            {
                "replicate": replicate,
                "q": q,
                "visibility_id": visibility_id,
                "world_id": world_id,
                "arm_id": arm_id,
                "evidence_regime": evidence_regime,
                **run,
            }
        )
    if q == 1.0 and world_id == "linear_fit":
        check_nesting(
            contract,
            config,
            replicate=replicate,
            visibility_id=visibility_id,
            runs=runs,
            probe=probe,
        )
    return rows


_THREAD_LIMIT: Any = None


def _worker_init() -> None:
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _run_task_args(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    return run_task(*args)


def run_study(
    contract: Mapping[str, Any], profile: str, workers: int = 1
) -> list[dict[str, Any]]:
    config = study_config(contract, profile)
    tasks = [
        (contract, profile, replicate, visibility_id, world_id, float(q))
        for replicate in range(config.replicates)
        for q in contract["entry_hazard"]["q_values"]
        for visibility_id in VISIBILITY_IDS
        for world_id in FIT_WORLDS
    ]
    if workers <= 1:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            results = [_run_task_args(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_worker_init
        ) as pool:
            results = list(pool.map(_run_task_args, tasks))
    return [row for rows in results for row in rows]


# -- decisions ------------------------------------------------------------------


def relative_delay_disposition(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify cells with keys q, delay, relative_delay, and status."""

    active_delays: dict[float, set[int]] = defaultdict(set)
    for cell in cells:
        active_delays.setdefault(cell["q"], set())
        if cell["status"] == "ENTRY_CHANNEL_ACTIVE":
            active_delays[cell["q"]].add(int(cell["delay"]))
    identical = len({frozenset(value) for value in active_delays.values()}) == 1
    active = [c["relative_delay"] for c in cells if c["status"] == "ENTRY_CHANNEL_ACTIVE"]
    inactive = [c["relative_delay"] for c in cells if c["status"] != "ENTRY_CHANNEL_ACTIVE"]
    both = bool(active) and bool(inactive)
    separable = both and max(active) < min(inactive)
    if not both or (identical and separable):
        disposition = "NOT_DISCRIMINATED"
    elif not identical and separable:
        disposition = "RELATIVE_DELAY_SUPPORTED"
    elif identical:
        disposition = "ABSOLUTE_DELAY_SUPPORTED"
    else:
        disposition = "NEITHER_SIMPLE_RULE"
    return {
        "disposition": disposition,
        "identical_active_delays_across_q": identical,
        "separable_by_relative_delay": separable,
        "active_delays_by_q": {str(q): sorted(value) for q, value in active_delays.items()},
        "threshold_interval": [max(active), min(inactive)] if separable else None,
    }


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {
        (
            int(row["replicate"]),
            float(row["q"]),
            str(row["visibility_id"]),
            str(row["world_id"]),
            str(row["arm_id"]),
        ): row
        for row in rows
    }


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = _index(rows)
    replicates = sorted({int(row["replicate"]) for row in rows})
    decision = contract["decision_contract"]
    required = int(decision["improved_replicates_at_least"])
    minimum = float(decision["entry_channel"]["improvement_at_least"])
    delays = {
        row["id"]: int(row["failure_delay_steps"]) for row in contract["visibility_regimes"]
    }

    def values(q: float, visibility: str, world: str, arm: str, metric: str) -> np.ndarray:
        return np.asarray(
            [float(indexed[(r, q, visibility, world, arm)][metric]) for r in replicates]
        )

    cells = []
    for q in contract["entry_hazard"]["q_values"]:
        q = float(q)
        widths = [
            indexed[(r, q, "immediate", world, "fixed_rule")]["entry_wave_width"]
            for r in replicates
            for world in FIT_WORLDS
        ]
        width = float(np.mean([w for w in widths if w is not None]))
        for visibility in VISIBILITY_IDS:
            reports = {}
            share = "productive_share_of_entrants"
            for arm in NONLINEAR_ARMS:
                linear_gap = values(q, visibility, "linear_fit", arm, share) - values(
                    q, visibility, "linear_fit", "linear_batch", share
                )
                for world in INTERACTION_FIT_WORLDS:
                    world_gap = values(q, visibility, world, arm, share) - values(
                        q, visibility, world, "linear_batch", share
                    )
                    reports[f"{arm}/{world}"] = abm1._judge(
                        world_gap - linear_gap, minimum, required
                    )
            saturated = all(
                int(
                    np.sum(
                        values(q, visibility, world, arm, "cumulative_entry_rate")
                        >= SATURATION_ENTRY
                    )
                )
                >= required
                for world in FIT_WORLDS
                for arm in FEATURE_ARMS
            )
            passed = sorted(key for key, report in reports.items() if report["pass"])
            if passed:
                status = "ENTRY_CHANNEL_ACTIVE"
            elif saturated:
                status = "ENTRY_SATURATED"
            else:
                status = "ENTRY_CHANNEL_ABSENT"
            cells.append(
                {
                    "q": q,
                    "visibility_id": visibility,
                    "delay": delays[visibility],
                    "fixed_rule_entry_wave_width": round(width, 6),
                    "relative_delay": round((delays[visibility] + 1) / width, 6),
                    "status": status,
                    "saturated": saturated,
                    "passing_family_geometries": passed,
                    "entry_channel_reports": reports,
                }
            )
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "cells": cells,
        "relative_delay_test": relative_delay_disposition(cells),
    }


def _timing_summary(rows: Sequence[Mapping[str, Any]], first_open_step: int) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["q"])].append(row)
    summary = {}
    for q, q_rows in sorted(grouped.items()):
        peaks = [row["entry_peak_steps"] for row in q_rows if row["entry_peak_steps"]]
        gaps = [row["model_h3_gap"] for row in q_rows if row["model_h3_gap"] is not None]
        summary[q] = {
            "runs": len(q_rows),
            "share_of_runs_peaking_at_first_open_step": round(
                float(np.mean([min(p) == first_open_step for p in peaks])), 6
            ),
            "share_of_runs_with_decline_onset": round(
                float(np.mean([row["decline_onset_step"] is not None for row in q_rows])), 6
            ),
            "median_model_h3_gap": float(np.median(gaps)) if gaps else None,
        }
    return summary


def summarize(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    config = study_config(contract, profile)
    metrics = (
        "cumulative_entry_rate",
        "productive_share_of_entrants",
        "active_failed_rate",
        "probe_brier",
        "entry_wave_width",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["q"], row["visibility_id"], row["world_id"], row["arm_id"])].append(row)
    cells = []
    for (q, visibility, world, arm), cell_rows in sorted(grouped.items()):
        means = {}
        for metric in metrics:
            values = [row[metric] for row in cell_rows if row[metric] is not None]
            means[metric] = round(float(np.mean(values)), 6) if values else None
        cells.append(
            {
                "q": q,
                "visibility_id": visibility,
                "world_id": world,
                "arm_id": arm,
                "replicates": len(cell_rows),
                "means": means,
            }
        )
    required = int(contract["decision_contract"]["improved_replicates_at_least"])
    if config.replicates < required:
        dispositions: dict[str, Any] = {"status": "NOT_EVALUATED_SMOKE_PROFILE"}
    else:
        dispositions = decide(rows, contract)
    return {
        "schema_version": "pivot_relative_delay_cost_scorecard.v1",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "cells": cells,
        "timing_observables": _timing_summary(rows, config.burn_in_steps + 1),
        "llm_reference_counts": {
            "runs": len(rows),
            "policy_queries": int(sum(row["policy_queries"] for row in rows)),
            "pre_entry_decisions": int(sum(row["pre_entry_decisions"] for row in rows)),
        },
        "dispositions": dispositions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot relative-delay and cost scorecard v1",
        "",
        f"Profile: `{summary['profile']}`",
        "",
    ]
    dispositions = summary["dispositions"]
    if "cells" in dispositions:
        lines.extend(
            [
                "| q | Visibility | Relative delay | Status | Passing pairs |",
                "|---:|---|---:|---|---|",
            ]
        )
        for cell in dispositions["cells"]:
            lines.append(
                f"| {cell['q']} | {cell['visibility_id']} | {cell['relative_delay']:.3f} | "
                f"{cell['status']} | {', '.join(cell['passing_family_geometries']) or '-'} |"
            )
        lines.extend(
            [
                "",
                f"Relative-delay test: `{dispositions['relative_delay_test']['disposition']}`",
            ]
        )
    else:
        lines.append(f"Dispositions: `{dispositions['status']}`")
    lines.extend(
        [
            "",
            "This is a synthetic generative ABM comparison. It selects no model and "
            "establishes no empirical or scientific effect.",
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
    paths = {
        "runs.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        "scorecard.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "scorecard.md": _markdown(summary),
    }
    for name, content in paths.items():
        delayed._write_exclusive(output_dir / name, content)
    receipt = {
        "schema_version": "pivot_relative_delay_cost_receipt.v1",
        "status": "COMPLETE",
        "row_count": len(rows),
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": abm1._sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {
            name: {
                "sha256": abm1._sha256(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in paths
        },
        "boundaries": summary["boundaries"],
    }
    delayed._write_exclusive(
        output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


# -- cost ledger ------------------------------------------------------------------


def benchmark(
    contract: Mapping[str, Any], *, profile: str = "development", repeats: int | None = None
) -> dict[str, Any]:
    """Time each policy family on one fixed task. Not byte-reproducible."""

    ledger = contract["cost_ledger"]
    repeats = int(ledger["repeats"]) if repeats is None else repeats
    config = study_config(contract, profile)
    world = "axis_interaction_fit"
    probe = abm2.probe_cohort(contract, 0, world)
    agent_steps = config.agent_count * config.steps
    load_before = os.getloadavg()
    records = []
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for arm_id, evidence_regime in BENCHMARK_ARMS:
            seconds = []
            for _repeat in range(repeats):
                start = time.process_time()
                row = run_arm(
                    contract,
                    config,
                    replicate=0,
                    visibility_id="immediate",
                    world_id=world,
                    arm_id=arm_id,
                    evidence_regime=evidence_regime,
                    probe=probe,
                    q=1.0,
                )
                seconds.append(time.process_time() - start)
            median = statistics.median(seconds)
            records.append(
                {
                    "arm_id": arm_id,
                    "evidence_regime": evidence_regime,
                    "cpu_seconds_per_run": seconds,
                    "median_cpu_seconds_per_run": median,
                    "median_cpu_ms_per_agent_step": 1000.0 * median / agent_steps,
                    "policy_queries_per_run": row["policy_queries"],
                    "pre_entry_decisions_per_run": row["pre_entry_decisions"],
                }
            )
    fixed = next(record for record in records if record["arm_id"] == "fixed_rule")
    for record in records:
        record["cost_relative_to_fixed_rule"] = (
            record["median_cpu_seconds_per_run"] / fixed["median_cpu_seconds_per_run"]
        )
    return {
        "schema_version": "pivot_relative_delay_cost_ledger.v1",
        "status": "TIMING_NOT_BYTE_REPRODUCIBLE",
        "profile": profile,
        "task": {
            "replicate": 0,
            "visibility_id": "immediate",
            "world_id": world,
            "q": 1.0,
            "agent_count": config.agent_count,
            "steps": config.steps,
            "agent_steps_per_run": agent_steps,
        },
        "clock": "PROCESS_CPU_TIME_SINGLE_THREAD",
        "repeats": repeats,
        "load_average_before": list(load_before),
        "load_average_after": list(os.getloadavg()),
        "records": records,
        "llm_reference": ledger["llm_reference"],
    }


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_path": str(config_path.resolve()),
        "contract_sha256": abm1._sha256(config_path),
        "q_values": list(contract["entry_hazard"]["q_values"]),
        "visibility_regimes": list(VISIBILITY_IDS),
        "main_run_arms": [list(key) for key in MAIN_ARMS],
        "profiles": contract["profiles"],
        "source_bindings": delayed.verify_source_bindings(contract),
        "decision_contract": contract["decision_contract"],
        "boundaries": contract["boundaries"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "benchmark"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile", choices=("smoke", "development"), default="development"
    )
    parser.add_argument("--precommit-sha")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config_path = args.config.resolve()
    contract = load_contract(config_path)
    if args.command == "plan":
        print(json.dumps(_plan_payload(contract, config_path), indent=2, sort_keys=True))
        return
    if args.precommit_sha is None:
        parser.error(f"{args.command} requires --precommit-sha")
    if FULL_SHA_PATTERN.fullmatch(args.precommit_sha) is None:
        parser.error(f"{args.command} requires a full 40-character lowercase Git SHA")
    precommit = delayed.verify_precommit(args.precommit_sha, contract)
    if args.command == "benchmark":
        if args.output_file is None:
            parser.error("benchmark requires --output-file")
        ledger = {**benchmark(contract, profile=args.profile), "precommit": precommit}
        delayed._write_exclusive(
            args.output_file.resolve(), json.dumps(ledger, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps({"status": ledger["status"], "output_file": str(args.output_file)}))
        return
    if args.output_dir is None:
        parser.error("run requires --output-dir")
    rows = run_study(contract, args.profile, workers=args.workers)
    summary = {
        **summarize(rows, contract, args.profile),
        "precommit_sha": args.precommit_sha,
        "contract_sha256": abm1._sha256(config_path),
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
