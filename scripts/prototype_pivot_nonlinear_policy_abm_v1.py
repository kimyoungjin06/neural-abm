#!/usr/bin/env python3
"""Run the precommitted Pivot nonlinear policy ABM v1."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXAMPLES = ROOT / "examples"
DEFAULT_CONFIG = (
    ROOT / "experiments" / "configs" / "pivot_nonlinear_policy_abm_v1.yaml"
)
for import_path in (ROOT, SCRIPTS, EXAMPLES):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import prototype_pivot_delayed_feedback_learning_v1 as delayed  # noqa: E402
import prototype_pivot_matched_marginal_discriminator_v2 as v2  # noqa: E402
import prototype_pivot_matched_marginal_discriminator_v3 as v3  # noqa: E402
import prototype_pivot_persistence_exit_v1 as persistence  # noqa: E402
import research_pivot_learning_study as study2  # noqa: E402
import research_pivot_study as study1  # noqa: E402


FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = ("linear_fit", "smooth_interaction_fit", "axis_interaction_fit")
INTERACTION_FIT_WORLDS = FIT_WORLDS[1:]
VISIBILITY_IDS = ("immediate", "failure_lag_4")
EVIDENCE_REGIMES = ("neighbor_local", "field_public")
BATCH_ARMS = ("scalar_batch", "linear_batch", "tree_batch", "mlp_batch")
NONLINEAR_ARMS = ("tree_batch", "mlp_batch")
ARM_KEYS = (
    ("fixed_rule", "none"),
    ("online_logistic", "neighbor_local"),
    *((arm, evidence) for evidence in EVIDENCE_REGIMES for arm in BATCH_ARMS),
)
WORLD_COMPONENT = {world: position + 1 for position, world in enumerate(FIT_WORLDS)}
VISIBILITY_COMPONENT = {
    regime: position + 1 for position, regime in enumerate(VISIBILITY_IDS)
}
EVIDENCE_COMPONENT = {
    regime: position + 1 for position, regime in enumerate(EVIDENCE_REGIMES)
}
ARM_COMPONENT = {arm: position + 1 for position, arm in enumerate(BATCH_ARMS)}
PREDECESSOR_MODES = {"fixed_rule": "frozen", "online_logistic": "imitative"}
FINAL_METRICS = (
    *persistence.FINAL_METRICS,
    "productive_share_of_entrants",
    "failed_entry_rate",
    "policy_brier",
    "policy_auc",
    "evidence_events_per_learner",
)
IMPROVEMENT_RULE = (
    "MEAN_PAIRED_IMPROVEMENT_AT_LEAST_THRESHOLD_AND_STRICTLY_POSITIVE_"
    "IN_AT_LEAST_REQUIRED_REPLICATES"
)
DID_RULE = (
    "INTERACTION_FIT_WORLD_GAP_MINUS_LINEAR_FIT_WORLD_GAP_SAME_VISIBILITY_"
    "EVIDENCE_AND_REPLICATE"
)
TRANSFER_WORLDS = {
    "linear_fit": "linear_support_aligned",
    "smooth_interaction_fit": "smooth_interaction_aligned",
    "axis_interaction_fit": "axis_interaction_aligned",
}


@dataclass
class WorldResearcher(study1.Researcher):
    """A researcher whose structural fit comes from a rank-matched world."""

    assigned_fit: float = 0.0

    @property
    def productive_fit(self) -> float:  # type: ignore[override]
        return float(self.assigned_fit)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("nonlinear policy ABM contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_nonlinear_policy_abm.v1":
        raise ValueError("unexpected nonlinear policy ABM schema")
    if contract.get("status") != "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION":
        raise ValueError("ABM contract is not a precommit design")
    worlds = contract.get("fit_worlds", {}).get("worlds", [])
    if [row["id"] for row in worlds] != list(FIT_WORLDS):
        raise ValueError("fit worlds changed")
    if [row["id"] for row in contract.get("visibility_regimes", [])] != list(
        VISIBILITY_IDS
    ):
        raise ValueError("visibility regimes changed")
    for row in contract["visibility_regimes"]:
        delayed.VisibilityRegime.from_mapping(row)
    if [row["id"] for row in contract.get("evidence_regimes", [])] != list(
        EVIDENCE_REGIMES
    ):
        raise ValueError("evidence regimes changed")
    arms = [
        (row["id"], regime)
        for row in contract.get("policy_arms", [])
        for regime in row["evidence_regimes"]
    ]
    if sorted(arms) != sorted(ARM_KEYS):
        raise ValueError("policy arms changed")
    if contract.get("common_environment", {}).get("learning_mode") != "imitative":
        raise ValueError("v1 is limited to imitative learning")
    decisions = contract.get("decision_contract", {})
    if decisions.get("improvement_rule") != IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("difference_in_differences_rule") != DID_RULE:
        raise ValueError("difference-in-differences rule changed")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside v1")
    boundaries = contract.get("boundaries", {})
    if boundaries.get("generic_nabm_api_changed") is not False:
        raise ValueError("generic NABM API change is outside v1")
    if boundaries.get("empirical_rows_read") != 0:
        raise ValueError("empirical rows are outside v1")


def study_config(contract: Mapping[str, Any], profile: str) -> Any:
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


def _visibility(contract: Mapping[str, Any], regime_id: str) -> Any:
    row = next(row for row in contract["visibility_regimes"] if row["id"] == regime_id)
    return delayed.VisibilityRegime.from_mapping(row)


# -- fit worlds -----------------------------------------------------------


def world_score(features: np.ndarray, world_id: str) -> np.ndarray:
    if world_id == "smooth_interaction_fit":
        return v2._interaction_score(features)
    if world_id == "axis_interaction_fit":
        return v3._axis_score(features)
    raise ValueError(f"not an interaction fit world: {world_id}")


def apply_fit_world(researchers: Sequence[Any], world_id: str) -> list[Any]:
    if world_id == "linear_fit":
        return list(researchers)
    features = np.stack([study2._feature_vector(row) for row in researchers])
    agent_ids = np.asarray([row.agent_id for row in researchers], dtype=np.int64)
    order = np.lexsort((agent_ids, -world_score(features, world_id)))
    linear_fits = [float(row.productive_fit) for row in researchers]
    assigned = np.empty(len(researchers), dtype=np.float64)
    assigned[order] = sorted(linear_fits, reverse=True)
    if sorted(assigned.tolist()) != sorted(linear_fits):
        raise RuntimeError("rank matching changed the fit value multiset")
    return [
        WorldResearcher(**dataclasses.asdict(row), assigned_fit=float(value))
        for row, value in zip(researchers, assigned, strict=True)
    ]


# -- batch learners -------------------------------------------------------


def _prior_output(features: np.ndarray) -> np.ndarray:
    return v2._sigmoid(v2._base_logit(features))


def fit_prior_boosted_trees(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    settings: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    """Newton-boosted regression trees starting from the frozen prior logit."""

    margin = v2._base_logit(train_features)
    eval_margin = v2._base_logit(eval_features)
    learning_rate = float(settings["learning_rate"])
    leaf_l2 = float(settings["leaf_l2"])
    for _round in range(int(settings["rounds"])):
        probability = v2._sigmoid(margin)
        gradient = train_labels - probability
        hessian = np.maximum(probability * (1.0 - probability), 1e-6)
        tree = DecisionTreeRegressor(
            max_depth=int(settings["max_depth"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            random_state=seed,
        )
        tree.fit(train_features, gradient / hessian, sample_weight=hessian)
        train_leaves = tree.apply(train_features)
        node_count = tree.tree_.node_count
        values = np.bincount(train_leaves, weights=gradient, minlength=node_count) / (
            np.bincount(train_leaves, weights=hessian, minlength=node_count) + leaf_l2
        )
        margin = margin + learning_rate * values[train_leaves]
        eval_margin = eval_margin + learning_rate * values[tree.apply(eval_features)]
    return v2._sigmoid(eval_margin)


def fit_batch(
    arm_id: str,
    settings: Mapping[str, Any],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    seed: int,
) -> np.ndarray:
    if arm_id == "scalar_batch":
        return v2._fit_scalar_delta(train_features, train_labels, eval_features, settings)
    if arm_id == "tree_batch":
        return fit_prior_boosted_trees(
            train_features, train_labels, eval_features, settings, seed
        )
    if arm_id == "linear_batch":
        model = {"id": "matched_linear", **settings}
    elif arm_id == "mlp_batch":
        model = {"id": "residual_mlp", **settings}
    else:
        raise ValueError(f"unknown batch arm: {arm_id}")
    candidate = {"weight_decay": float(settings["weight_decay"])}
    return v3.fit_candidate(
        model, candidate, train_features, train_labels, eval_features, seed
    )


@dataclass
class BatchLearner:
    """Refit a registered family from the frozen prior on a growing buffer."""

    arm_id: str
    settings: Mapping[str, Any]
    seed_components: tuple[int, ...]
    eval_features: np.ndarray
    features: list[np.ndarray] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    refits: int = 0
    output: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.output = _prior_output(self.eval_features)

    def update(self, events: Sequence[study2.PivotEvent]) -> None:
        if not events:
            return
        for event in events:
            self.features.append(event.features)
            self.labels.append(1.0 if event.productive else 0.0)
        seed = int(
            np.random.SeedSequence((*self.seed_components, self.refits)).generate_state(
                1
            )[0]
        )
        self.refits += 1
        self.output = fit_batch(
            self.arm_id,
            self.settings,
            np.stack(self.features),
            np.asarray(self.labels, dtype=np.float64),
            self.eval_features,
            seed,
        )


@dataclass
class BatchPolicyAgent(study2.PivotPolicyAgent):
    """Pivot agent whose readiness target comes from a batch learner."""

    evidence_regime: str = "neighbor_local"
    learner: BatchLearner | None = None
    field_learner: BatchLearner | None = None

    def policy_output(self, features: np.ndarray) -> float:
        del features
        if self.field_learner is not None:
            return float(self.field_learner.output[self.agent_id])
        if self.learner is None:
            raise RuntimeError("batch agent has no learner")
        return float(self.learner.output[0])

    def act_or_predict(self, observation: Any) -> torch.Tensor:
        del observation
        return torch.tensor([self.policy_output(self.features)], dtype=torch.float64)

    def _learn_from_events(self, events: Sequence[study2.PivotEvent]) -> None:
        # The field learner is updated once per step by the run loop.
        if self.field_learner is not None:
            return
        neighbor_ids = set(self.environment.neighbors[self.agent_id])
        samples = [
            event
            for event in events
            if event.agent_id != self.agent_id and event.agent_id in neighbor_ids
        ]
        if samples and self.learner is not None:
            self.learner.update(samples)


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
        WORLD_COMPONENT[world_id],
        VISIBILITY_COMPONENT[visibility_id],
        EVIDENCE_COMPONENT[evidence_regime],
        ARM_COMPONENT[arm_id],
        agent_component,
    )


# -- ABM run loop ---------------------------------------------------------


def run_arm(
    contract: Mapping[str, Any],
    config: Any,
    *,
    replicate: int,
    visibility_id: str,
    world_id: str,
    arm_id: str,
    evidence_regime: str,
) -> dict[str, Any]:
    """Run one arm through the persistence predecessor's exact lifecycle."""

    scenario = _scenario(contract)
    visibility = _visibility(contract, visibility_id)
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
    researchers = apply_fit_world(researchers, world_id)
    environment = delayed.DelayedPivotEnvironment(
        neighbors=neighbors,
        config=config,
        replicate=replicate,
        visibility=visibility,
    )
    all_features = np.stack([study2._feature_vector(row) for row in researchers])
    field_learner: BatchLearner | None = None
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
            field_learner = BatchLearner(
                arm_id=arm_id,
                settings=settings,
                seed_components=_learner_seed(config, agent_component=0, **seed),
                eval_features=all_features,
            )
        agents = [
            BatchPolicyAgent(
                agent_id=row.agent_id,
                base=row,
                environment=environment,
                learning_rate=0.0,
                learning_mode="imitative",
                prior_anchor=0.0,
                evidence_regime=evidence_regime,
                learner=(
                    None
                    if field_learner is not None
                    else BatchLearner(
                        arm_id=arm_id,
                        settings=settings,
                        seed_components=_learner_seed(
                            config, agent_component=row.agent_id + 1, **seed
                        ),
                        eval_features=all_features[row.agent_id : row.agent_id + 1],
                    )
                ),
                field_learner=field_learner,
            )
            for row in researchers
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
    trajectories: dict[str, list[Any]] = defaultdict(list)
    for _step in range(config.steps):
        environment.advance()
        if field_learner is not None:
            field_learner.update(environment.last_events)
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
    if field_learner is not None:
        evidence_events: float | None = float(len(field_learner.labels))
    elif arm_id in PREDECESSOR_MODES:
        evidence_events = None
    else:
        evidence_events = float(
            np.mean([len(agent.learner.labels) for agent in agents])  # type: ignore[union-attr]
        )
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
        "policy_brier": float(np.mean((outputs - labels) ** 2)),
        "policy_auc": (
            float(roc_auc_score(labels, outputs)) if 0.0 < labels.mean() < 1.0 else None
        ),
        "evidence_events_per_learner": evidence_events,
        **dict(trajectories),
        "continuation_scores_at_exit": [
            round(value, 10) for value in adapter.continuation_scores_at_exit
        ],
    }


def check_nesting(
    contract: Mapping[str, Any],
    config: Any,
    *,
    replicate: int,
    visibility_id: str,
    runs: Mapping[str, Mapping[str, Any]],
) -> None:
    for arm_id, mode in PREDECESSOR_MODES.items():
        predecessor = persistence.run_persistence_arm(
            scenario=_scenario(contract),
            config=config,
            visibility=_visibility(contract, visibility_id),
            transition=contract["common_environment"]["transition"],
            replicate=replicate,
            learning_mode=mode,
        )
        for key, value in predecessor.items():
            if runs[arm_id][key] != value:
                raise RuntimeError(
                    f"{arm_id} does not nest the persistence predecessor at {key}"
                )


def run_task(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    visibility_id: str,
    world_id: str,
) -> list[dict[str, Any]]:
    config = study_config(contract, profile)
    rows = []
    runs: dict[str, dict[str, Any]] = {}
    for arm_id, evidence_regime in ARM_KEYS:
        run = run_arm(
            contract,
            config,
            replicate=replicate,
            visibility_id=visibility_id,
            world_id=world_id,
            arm_id=arm_id,
            evidence_regime=evidence_regime,
        )
        if evidence_regime != "field_public":
            runs[arm_id] = run
        rows.append(
            {
                "replicate": replicate,
                "visibility_id": visibility_id,
                "world_id": world_id,
                "arm_id": arm_id,
                "evidence_regime": evidence_regime,
                **run,
            }
        )
    if world_id == "linear_fit":
        check_nesting(
            contract,
            config,
            replicate=replicate,
            visibility_id=visibility_id,
            runs=runs,
        )
    return rows


_THREAD_LIMIT: Any = None


def _worker_init() -> None:
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def _run_task_args(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    return run_task(*args)


def run_abm(
    contract: Mapping[str, Any], profile: str, workers: int = 1
) -> list[dict[str, Any]]:
    config = study_config(contract, profile)
    tasks = [
        (contract, profile, replicate, visibility_id, world_id)
        for replicate in range(config.replicates)
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


# -- family transfer check ------------------------------------------------


def family_transfer_check(contract: Mapping[str, Any]) -> dict[str, Any]:
    check = contract["family_transfer_check"]
    cohort_contract = {
        "cohorts": {
            "scenario": contract["common_environment"]["scenario"],
            "training": {"base_seed": int(check["training_base_seed"])},
            "evaluation": {"base_seed": int(check["evaluation_base_seed"])},
        }
    }
    briers: dict[tuple[int, str, str], float] = {}
    for replicate in range(int(check["replicates"])):
        train = v2.make_cohort(
            cohort_contract,
            role="training",
            replicate=replicate,
            agent_count=int(check["agent_count"]),
        )
        evaluation = v2.make_cohort(
            cohort_contract,
            role="evaluation",
            replicate=replicate,
            agent_count=int(check["agent_count"]),
        )
        for world_id, v3_world in TRANSFER_WORLDS.items():
            train_labels = v3.world_labels(train, v3_world)
            eval_labels = v3.world_labels(evaluation, v3_world)
            for arm_id in BATCH_ARMS:
                seed = int(
                    np.random.SeedSequence(
                        (
                            int(check["training_base_seed"]),
                            replicate,
                            WORLD_COMPONENT[world_id],
                            ARM_COMPONENT[arm_id],
                        )
                    ).generate_state(1)[0]
                )
                prediction = fit_batch(
                    arm_id,
                    contract["batch_learners"][arm_id],
                    train.features,
                    train_labels,
                    evaluation.features,
                    seed,
                )
                briers[(replicate, world_id, arm_id)] = float(
                    np.mean((prediction - eval_labels) ** 2)
                )
    replicates = range(int(check["replicates"]))

    def wins(world: str, better: str, worse: str) -> int:
        return sum(
            briers[(replicate, world, better)] < briers[(replicate, world, worse)]
            for replicate in replicates
        )

    required = int(check["pass_when_all_hold_in_at_least_replicates"])
    conditions = {
        "tree_beats_linear_smooth": wins("smooth_interaction_fit", "tree_batch", "linear_batch"),
        "tree_beats_linear_axis": wins("axis_interaction_fit", "tree_batch", "linear_batch"),
        "mlp_beats_linear_smooth": wins("smooth_interaction_fit", "mlp_batch", "linear_batch"),
        "mlp_beats_linear_axis": wins("axis_interaction_fit", "mlp_batch", "linear_batch"),
        "mlp_beats_tree_smooth": wins("smooth_interaction_fit", "mlp_batch", "tree_batch"),
        "tree_beats_mlp_axis": wins("axis_interaction_fit", "tree_batch", "mlp_batch"),
    }
    return {
        "status": (
            "FAMILY_TRANSFERRED"
            if all(value >= required for value in conditions.values())
            else "FAMILY_NOT_TRANSFERRED"
        ),
        "winning_replicates": conditions,
        "required_replicates": required,
        "mean_brier": {
            f"{world}/{arm}": round(
                float(np.mean([briers[(replicate, world, arm)] for replicate in replicates])),
                6,
            )
            for world in TRANSFER_WORLDS
            for arm in BATCH_ARMS
        },
    }


# -- decisions ------------------------------------------------------------


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {
        (
            int(row["replicate"]),
            str(row["visibility_id"]),
            str(row["world_id"]),
            str(row["arm_id"]),
            str(row["evidence_regime"]),
        ): row
        for row in rows
    }


def _gap(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    *,
    metric: str,
    higher_is_better: bool,
    visibility_id: str,
    world_id: str,
    evidence_regime: str,
    arm_id: str,
    reference_arm_id: str,
) -> np.ndarray:
    sign = 1.0 if higher_is_better else -1.0
    return np.asarray(
        [
            sign
            * (
                float(indexed[(r, visibility_id, world_id, arm_id, evidence_regime)][metric])
                - float(
                    indexed[(r, visibility_id, world_id, reference_arm_id, evidence_regime)][
                        metric
                    ]
                )
            )
            for r in replicates
        ],
        dtype=np.float64,
    )


def _judge(values: np.ndarray, minimum: float, positive_required: int) -> dict[str, Any]:
    positive = int(np.sum(values > 0.0))
    distribution = v2._distribution(values)
    return {
        "improvement": distribution,
        "positive_replicates": positive,
        "pass": distribution["mean"] >= minimum and positive >= positive_required,
    }


def _did_report(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    rule: Mapping[str, Any],
    *,
    metric: str,
    higher_is_better: bool,
    evidence_regime: str,
) -> dict[str, Any]:
    minimum = float(rule["improvement_at_least"])
    positive_required = int(rule["improved_replicates_at_least"])
    report: dict[str, Any] = {}
    for arm_id in rule["families"]:
        for world_id in rule["worlds"]:
            cells = {}
            for visibility_id in VISIBILITY_IDS:
                common = {
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "visibility_id": visibility_id,
                    "evidence_regime": evidence_regime,
                    "arm_id": arm_id,
                    "reference_arm_id": "linear_batch",
                }
                values = _gap(indexed, replicates, world_id=world_id, **common) - _gap(
                    indexed, replicates, world_id="linear_fit", **common
                )
                cells[visibility_id] = _judge(values, minimum, positive_required)
            report[f"{arm_id}/{world_id}"] = {
                "visibility": cells,
                "pass": all(cell["pass"] for cell in cells.values()),
            }
    return report


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = _index(rows)
    replicates = sorted({int(row["replicate"]) for row in rows})
    decision = contract["decision_contract"]
    learning = _did_report(
        indexed,
        replicates,
        decision["policy_learning"],
        metric="policy_brier",
        higher_is_better=False,
        evidence_regime="field_public",
    )
    behavior = _did_report(
        indexed,
        replicates,
        decision["behavioral_consequence"],
        metric="productive_share_of_entrants",
        higher_is_better=True,
        evidence_regime="field_public",
    )
    local_learning = _did_report(
        indexed,
        replicates,
        decision["policy_learning"],
        metric="policy_brier",
        higher_is_better=False,
        evidence_regime="neighbor_local",
    )
    crossover_rule = decision["abm_geometry_crossover"]
    minimum = float(crossover_rule["improvement_at_least"])
    positive_required = int(crossover_rule["improved_replicates_at_least"])
    crossover: dict[str, Any] = {}
    for world_id in INTERACTION_FIT_WORLDS:
        for arm_id, reference in (("mlp_batch", "tree_batch"), ("tree_batch", "mlp_batch")):
            cells = {
                visibility_id: _judge(
                    _gap(
                        indexed,
                        replicates,
                        metric="policy_brier",
                        higher_is_better=False,
                        visibility_id=visibility_id,
                        world_id=world_id,
                        evidence_regime="field_public",
                        arm_id=arm_id,
                        reference_arm_id=reference,
                    ),
                    minimum,
                    positive_required,
                )
                for visibility_id in VISIBILITY_IDS
            }
            crossover[f"{arm_id}_over_{reference}/{world_id}"] = {
                "visibility": cells,
                "pass": all(cell["pass"] for cell in cells.values()),
            }
    mlp_smooth = crossover["mlp_batch_over_tree_batch/smooth_interaction_fit"]["pass"]
    mlp_axis = crossover["mlp_batch_over_tree_batch/axis_interaction_fit"]["pass"]
    tree_smooth = crossover["tree_batch_over_mlp_batch/smooth_interaction_fit"]["pass"]
    tree_axis = crossover["tree_batch_over_mlp_batch/axis_interaction_fit"]["pass"]
    if mlp_smooth and tree_axis:
        crossover_disposition = "CROSSOVER_TRANSFERS_TO_ABM"
    elif mlp_smooth and mlp_axis:
        crossover_disposition = "MLP_DOMINATES_IN_ABM"
    elif tree_smooth and tree_axis:
        crossover_disposition = "TREE_DOMINATES_IN_ABM"
    else:
        crossover_disposition = "TIED_OR_MIXED"
    learned = sorted(key for key, value in learning.items() if value["pass"])
    learned_locally = sorted(key for key, value in local_learning.items() if value["pass"])
    changed = sorted(key for key in learned if behavior[key]["pass"])
    if not learned:
        evidence_scale = "NO_NONLINEAR_LEARNING_UNDER_FIELD_EVIDENCE"
        result = "NO_NONLINEAR_LEARNING_IN_ABM"
    else:
        evidence_scale = (
            "NONLINEAR_LEARNING_SURVIVES_LOCAL_EVIDENCE"
            if learned_locally
            else "EVIDENCE_SCALE_GATES_NONLINEAR_LEARNING"
        )
        result = (
            "NONLINEAR_LEARNING_AND_BEHAVIOR_BOTH_CHANGE"
            if changed
            else "NONLINEAR_LEARNING_CHANGES_BUT_BEHAVIOR_DOES_NOT"
        )
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "result": result,
        "abm_geometry_crossover": crossover_disposition,
        "evidence_scale": evidence_scale,
        "learned_family_geometries_field": learned,
        "learned_family_geometries_local": learned_locally,
        "behavior_changing_family_geometries_field": changed,
        "reports": {
            "policy_learning_field": learning,
            "behavioral_consequence_field": behavior,
            "policy_learning_local": local_learning,
            "abm_geometry_crossover": crossover,
        },
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    transfer: Mapping[str, Any],
    contract: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["visibility_id"], row["world_id"], row["arm_id"], row["evidence_regime"])
        ].append(row)
    cells = []
    for visibility_id in VISIBILITY_IDS:
        for world_id in FIT_WORLDS:
            for arm_id, evidence_regime in ARM_KEYS:
                cell_rows = grouped[(visibility_id, world_id, arm_id, evidence_regime)]
                means = {}
                for metric in FINAL_METRICS:
                    values = [row[metric] for row in cell_rows if row[metric] is not None]
                    means[metric] = round(float(np.mean(values)), 6) if values else None
                cells.append(
                    {
                        "visibility_id": visibility_id,
                        "world_id": world_id,
                        "arm_id": arm_id,
                        "evidence_regime": evidence_regime,
                        "replicates": len(cell_rows),
                        "means": means,
                    }
                )
    config = study_config(contract, profile)
    minimum_positive = int(
        contract["decision_contract"]["policy_learning"]["improved_replicates_at_least"]
    )
    if config.replicates < minimum_positive:
        dispositions: dict[str, Any] = {
            "status": "NOT_EVALUATED_SMOKE_PROFILE",
            "result": "NOT_EVALUATED",
            "abm_geometry_crossover": "NOT_EVALUATED",
            "evidence_scale": "NOT_EVALUATED",
        }
    else:
        dispositions = decide(rows, contract)
    return {
        "schema_version": "pivot_nonlinear_policy_abm_scorecard.v1",
        "status": "SYNTHETIC_GENERATIVE_ABM_COMPARISON",
        "profile": profile,
        "nesting_checks": "PASS_EXACT",
        "family_transfer": transfer,
        "cells": cells,
        "dispositions": dispositions,
        "boundaries": contract["boundaries"],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot nonlinear policy ABM scorecard v1",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        f"Family transfer: `{summary['family_transfer']['status']}`",
        "",
        "| Visibility | Fit world | Arm | Evidence | Entry | Productive share of entrants | "
        "Active failed | Policy Brier |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        means = cell["means"]
        lines.append(
            "| {visibility} | {world} | {arm} | {evidence} | {entry:.4f} | "
            "{share:.4f} | {failed:.4f} | {brier:.4f} |".format(
                visibility=cell["visibility_id"],
                world=cell["world_id"],
                arm=cell["arm_id"],
                evidence=cell["evidence_regime"],
                entry=means["cumulative_entry_rate"],
                share=means["productive_share_of_entrants"],
                failed=means["active_failed_rate"],
                brier=means["policy_brier"],
            )
        )
    dispositions = summary["dispositions"]
    lines.extend(
        [
            "",
            "## Frozen dispositions",
            "",
            f"- Result: `{dispositions['result']}`",
            f"- ABM geometry crossover: `{dispositions['abm_geometry_crossover']}`",
            f"- Evidence scale: `{dispositions['evidence_scale']}`",
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
        "family_transfer.json": json.dumps(
            summary["family_transfer"], indent=2, sort_keys=True
        )
        + "\n",
        "runs.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        "abm_scorecard.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "abm_scorecard.md": _markdown(summary),
    }
    for name, content in paths.items():
        delayed._write_exclusive(output_dir / name, content)
    receipt = {
        "schema_version": "pivot_nonlinear_policy_abm_receipt.v1",
        "status": "COMPLETE",
        "row_count": len(rows),
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": _sha256(contract_path),
        "precommit": dict(precommit),
        "outputs": {
            name: {
                "sha256": _sha256(output_dir / name),
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


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_path": str(config_path.resolve()),
        "contract_sha256": _sha256(config_path),
        "fit_worlds": list(FIT_WORLDS),
        "visibility_regimes": list(VISIBILITY_IDS),
        "arm_keys": [list(key) for key in ARM_KEYS],
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
    parser.add_argument("--workers", type=int, default=32)
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
    transfer = family_transfer_check(contract)
    rows = run_abm(contract, args.profile, workers=args.workers)
    summary = {
        **summarize(rows, transfer, contract, args.profile),
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
