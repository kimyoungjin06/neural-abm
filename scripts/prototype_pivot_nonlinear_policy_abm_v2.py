#!/usr/bin/env python3
"""Run the precommitted Pivot nonlinear policy ABM v2."""

from __future__ import annotations

import argparse
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
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXAMPLES = ROOT / "examples"
DEFAULT_CONFIG = (
    ROOT / "experiments" / "configs" / "pivot_nonlinear_policy_abm_v2.yaml"
)
for import_path in (ROOT, SCRIPTS, EXAMPLES):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import prototype_pivot_nonlinear_policy_abm_v1 as abm1  # noqa: E402


delayed = abm1.delayed
persistence = abm1.persistence
study1 = abm1.study1
study2 = abm1.study2
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FIT_WORLDS = abm1.FIT_WORLDS
INTERACTION_FIT_WORLDS = abm1.INTERACTION_FIT_WORLDS
VISIBILITY_IDS = ("immediate", "failure_lag_2", "failure_lag_4")
EVIDENCE_REGIMES = abm1.EVIDENCE_REGIMES
BATCH_ARMS = abm1.BATCH_ARMS
NONLINEAR_ARMS = abm1.NONLINEAR_ARMS
FEATURE_ARMS = ("linear_batch", "tree_batch", "mlp_batch")
ARM_KEYS = abm1.ARM_KEYS
PREDECESSOR_MODES = abm1.PREDECESSOR_MODES
WORLD_COMPONENT = abm1.WORLD_COMPONENT
VISIBILITY_COMPONENT = {
    regime: position + 1 for position, regime in enumerate(VISIBILITY_IDS)
}
EVIDENCE_COMPONENT = abm1.EVIDENCE_COMPONENT
ARM_COMPONENT = abm1.ARM_COMPONENT
FINAL_METRICS = (
    *persistence.FINAL_METRICS,
    "productive_share_of_entrants",
    "failed_entry_rate",
    "probe_brier",
    "probe_auc",
    "in_situ_policy_brier",
    "evidence_events_per_learner",
)
IMPROVEMENT_RULE = abm1.IMPROVEMENT_RULE
DID_RULE = abm1.DID_RULE
OPENED_SEEDS = {20261003, 20261004, 20261005}
# Local learners use agent_id + 1; field fold learners sit above any agent id.
FIELD_FOLD_COMPONENT = 1_000_000


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("nonlinear policy ABM contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "pivot_nonlinear_policy_abm.v2":
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
    field_regime = contract["evidence_regimes"][1]
    if field_regime.get("includes_own_public_outcome") is not False:
        raise ValueError("the field record must exclude the agent's own outcome")
    if int(field_regime.get("cross_fit_folds", 0)) < 2:
        raise ValueError("field cross-fitting needs at least two folds")
    arms = [
        (row["id"], regime)
        for row in contract.get("policy_arms", [])
        for regime in row["evidence_regimes"]
    ]
    if sorted(arms) != sorted(ARM_KEYS):
        raise ValueError("policy arms changed")
    v1_learners = abm1.load_contract()["batch_learners"]
    for arm_id in BATCH_ARMS:
        for key, value in contract["batch_learners"][arm_id].items():
            if v1_learners[arm_id].get(key) != value:
                raise ValueError(f"{arm_id} setting {key} differs from v1")
    seeds = {
        int(contract["profiles"]["development"]["base_seed"]),
        int(contract["probe_cohort"]["base_seed"]),
        int(contract["family_transfer_check"]["training_base_seed"]),
        int(contract["family_transfer_check"]["evaluation_base_seed"]),
    }
    if len(seeds) != 4 or seeds & OPENED_SEEDS:
        raise ValueError("v2 needs four fresh seed namespaces")
    decisions = contract.get("decision_contract", {})
    if decisions.get("improvement_rule") != IMPROVEMENT_RULE:
        raise ValueError("paired improvement rule changed")
    if decisions.get("difference_in_differences_rule") != DID_RULE:
        raise ValueError("difference-in-differences rule changed")
    if decisions.get("evaluated_per_visibility_regime") is not True:
        raise ValueError("v2 decisions must be evaluated per visibility regime")
    if decisions.get("nesting_checks", {}).get("required") is not True:
        raise ValueError("nesting checks must remain required")
    if decisions.get("model_selection") != "FORBIDDEN":
        raise ValueError("model selection is outside v2")
    boundaries = contract.get("boundaries", {})
    if boundaries.get("generic_nabm_api_changed") is not False:
        raise ValueError("generic NABM API change is outside v2")
    if boundaries.get("empirical_rows_read") != 0:
        raise ValueError("empirical rows are outside v2")


def study_config(contract: Mapping[str, Any], profile: str) -> Any:
    return abm1.study_config(contract, profile)


# -- probe cohort ---------------------------------------------------------


def probe_cohort(
    contract: Mapping[str, Any], replicate: int, world_id: str
) -> tuple[np.ndarray, np.ndarray]:
    probe = contract["probe_cohort"]
    base_seed = int(probe["base_seed"])
    context = SimpleNamespace(
        scenario=abm1._scenario(contract),
        replicate=replicate,
        rng=study1.component_rng(base_seed, replicate, "population"),
    )
    population_config = study1.StudyConfig(
        agent_count=int(probe["agent_count"]), base_seed=base_seed
    )
    researchers = abm1.apply_fit_world(
        study1.make_researchers(context, population_config), world_id
    )
    threshold = float(contract["inherited_learning_parameters"]["productive_threshold"])
    features = np.stack([study2._feature_vector(row) for row in researchers])
    labels = np.asarray(
        [float(row.productive_fit >= threshold) for row in researchers]
    )
    return features, labels


def _brier(prediction: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((prediction - labels) ** 2))


def _auc(prediction: np.ndarray, labels: np.ndarray) -> float | None:
    if not 0.0 < float(labels.mean()) < 1.0:
        return None
    return float(roc_auc_score(labels, prediction))


# -- learners -------------------------------------------------------------


@dataclass
class ProbedLearner:
    """A v1 batch learner that also predicts the out-of-sample probe cohort."""

    arm_id: str
    settings: Mapping[str, Any]
    seed_components: tuple[int, ...]
    eval_features: np.ndarray
    probe_features: np.ndarray
    excluded_agent_ids: frozenset[int] = frozenset()
    features: list[np.ndarray] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    trained_agent_ids: list[int] = field(default_factory=list)
    refits: int = 0
    output: np.ndarray = field(init=False)
    probe_output: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.output = abm1._prior_output(self.eval_features)
        self.probe_output = abm1._prior_output(self.probe_features)

    def update(self, events: Sequence[study2.PivotEvent]) -> None:
        events = [
            event for event in events if event.agent_id not in self.excluded_agent_ids
        ]
        if not events:
            return
        for event in events:
            self.features.append(event.features)
            self.labels.append(1.0 if event.productive else 0.0)
            self.trained_agent_ids.append(int(event.agent_id))
        seed = int(
            np.random.SeedSequence((*self.seed_components, self.refits)).generate_state(
                1
            )[0]
        )
        self.refits += 1
        prediction = abm1.fit_batch(
            self.arm_id,
            self.settings,
            np.stack(self.features),
            np.asarray(self.labels, dtype=np.float64),
            np.vstack([self.eval_features, self.probe_features]),
            seed,
        )
        count = self.eval_features.shape[0]
        self.output = prediction[:count]
        self.probe_output = prediction[count:]


@dataclass
class CrossFitField:
    """Fold learners; each agent acts on a model that never saw its fold."""

    learners: list[ProbedLearner]

    @property
    def folds(self) -> int:
        return len(self.learners)

    def update(self, events: Sequence[study2.PivotEvent]) -> None:
        for learner in self.learners:
            learner.update(events)

    def output_for(self, agent_id: int) -> float:
        return float(self.learners[agent_id % self.folds].output[agent_id])

    def check_isolation(self) -> None:
        for fold, learner in enumerate(self.learners):
            if any(agent_id % self.folds == fold for agent_id in learner.trained_agent_ids):
                raise RuntimeError("a field fold learner saw its own fold")


@dataclass
class ProbedPolicyAgent(study2.PivotPolicyAgent):
    """Pivot agent acting on a local learner or its fold's field learner."""

    learner: ProbedLearner | None = None
    field_record: CrossFitField | None = None

    def policy_output(self, features: np.ndarray) -> float:
        del features
        if self.field_record is not None:
            return self.field_record.output_for(self.agent_id)
        if self.learner is None:
            raise RuntimeError("batch agent has no learner")
        return float(self.learner.output[0])

    def act_or_predict(self, observation: Any) -> torch.Tensor:
        del observation
        return torch.tensor([self.policy_output(self.features)], dtype=torch.float64)

    def _learn_from_events(self, events: Sequence[study2.PivotEvent]) -> None:
        # The field record is updated once per step by the run loop.
        if self.field_record is not None:
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
    probe: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Run one arm through the persistence predecessor's exact lifecycle."""

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
    field_record: CrossFitField | None = None
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
            field_record = CrossFitField(
                learners=[
                    ProbedLearner(
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
            ProbedPolicyAgent(
                agent_id=row.agent_id,
                base=row,
                environment=environment,
                learning_rate=0.0,
                learning_mode="imitative",
                prior_anchor=0.0,
                learner=(
                    None
                    if field_record is not None
                    else ProbedLearner(
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
    probe_predictions: list[np.ndarray]
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
            np.mean([_brier(prediction, probe_labels) for prediction in probe_predictions])
        ),
        "probe_auc": _auc(np.mean(probe_predictions, axis=0), probe_labels),
        "in_situ_policy_brier": _brier(outputs, labels),
        "evidence_events_per_learner": evidence_events,
        **dict(trajectories),
        "continuation_scores_at_exit": [
            round(value, 10) for value in adapter.continuation_scores_at_exit
        ],
    }


def run_task(
    contract: Mapping[str, Any],
    profile: str,
    replicate: int,
    visibility_id: str,
    world_id: str,
) -> list[dict[str, Any]]:
    config = study_config(contract, profile)
    probe = probe_cohort(contract, replicate, world_id)
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
            probe=probe,
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
        abm1.check_nesting(
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


# -- decisions ------------------------------------------------------------


def _values(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    *,
    metric: str,
    visibility_id: str,
    world_id: str,
    arm_key: tuple[str, str],
) -> np.ndarray:
    return np.asarray(
        [
            float(indexed[(replicate, visibility_id, world_id, *arm_key)][metric])
            for replicate in replicates
        ],
        dtype=np.float64,
    )


def _improvement(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    *,
    metric: str,
    higher_is_better: bool,
    visibility_id: str,
    world_id: str,
    arm_key: tuple[str, str],
    reference_key: tuple[str, str],
    difference_in_differences: bool,
) -> np.ndarray:
    def gap(world: str) -> np.ndarray:
        common = {"metric": metric, "visibility_id": visibility_id, "world_id": world}
        difference = _values(indexed, replicates, arm_key=arm_key, **common) - _values(
            indexed, replicates, arm_key=reference_key, **common
        )
        return difference if higher_is_better else -difference

    values = gap(world_id)
    if difference_in_differences:
        values = values - gap("linear_fit")
    return values


def _saturated(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    visibility_id: str,
    minimum: int,
) -> bool:
    return all(
        sum(
            float(
                indexed[(replicate, visibility_id, world, arm, "field_public")][
                    "cumulative_entry_rate"
                ]
            )
            == 1.0
            for replicate in replicates
        )
        >= minimum
        for world in FIT_WORLDS
        for arm in FEATURE_ARMS
    )


def decide_visibility(
    indexed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    replicates: Sequence[int],
    contract: Mapping[str, Any],
    visibility_id: str,
) -> dict[str, Any]:
    decision = contract["decision_contract"]
    required = int(decision["improved_replicates_at_least"])

    def judge(rule: str, **kwargs: Any) -> dict[str, Any]:
        minimum = float(decision[rule]["improvement_at_least"])
        values = _improvement(indexed, replicates, visibility_id=visibility_id, **kwargs)
        return abm1._judge(values, minimum, required)

    field = "field_public"
    learning = {
        f"{arm}/{world}": judge(
            "field_nonlinear_learning",
            metric="probe_brier",
            higher_is_better=False,
            world_id=world,
            arm_key=(arm, field),
            reference_key=("linear_batch", field),
            difference_in_differences=True,
        )
        for arm in NONLINEAR_ARMS
        for world in INTERACTION_FIT_WORLDS
    }
    crossover = {
        f"{arm}_over_{reference}/{world}": judge(
            "field_geometry_crossover",
            metric="probe_brier",
            higher_is_better=False,
            world_id=world,
            arm_key=(arm, field),
            reference_key=(reference, field),
            difference_in_differences=False,
        )
        for world in INTERACTION_FIT_WORLDS
        for arm, reference in (("mlp_batch", "tree_batch"), ("tree_batch", "mlp_batch"))
    }

    def passed(key: str) -> bool:
        return bool(crossover[key]["pass"])

    if passed("mlp_batch_over_tree_batch/smooth_interaction_fit") and passed(
        "tree_batch_over_mlp_batch/axis_interaction_fit"
    ):
        crossover_disposition = "CROSSOVER_TRANSFERS_TO_ABM"
    elif passed("mlp_batch_over_tree_batch/smooth_interaction_fit") and passed(
        "mlp_batch_over_tree_batch/axis_interaction_fit"
    ):
        crossover_disposition = "MLP_DOMINATES_IN_ABM"
    elif passed("tree_batch_over_mlp_batch/smooth_interaction_fit") and passed(
        "tree_batch_over_mlp_batch/axis_interaction_fit"
    ):
        crossover_disposition = "TREE_DOMINATES_IN_ABM"
    else:
        crossover_disposition = "TIED_OR_MIXED"
    useful = {
        regime: {
            f"{arm}/{world}": judge(
                "useful_learning",
                metric="probe_brier",
                higher_is_better=False,
                world_id=world,
                arm_key=(arm, regime),
                reference_key=("fixed_rule", "none"),
                difference_in_differences=False,
            )
            for arm in FEATURE_ARMS
            for world in INTERACTION_FIT_WORLDS
        }
        for regime in EVIDENCE_REGIMES
    }
    useful_field = sorted(k for k, v in useful["field_public"].items() if v["pass"])
    useful_local = sorted(k for k, v in useful["neighbor_local"].items() if v["pass"])
    if not useful_field:
        evidence_scale = "NO_USEFUL_LEARNING_UNDER_FIELD_EVIDENCE"
    elif useful_local:
        evidence_scale = "USEFUL_LEARNING_SURVIVES_LOCAL_EVIDENCE"
    else:
        evidence_scale = "EVIDENCE_SCALE_GATES_USEFUL_LEARNING"
    saturated = _saturated(indexed, replicates, visibility_id, required)
    entry = {
        f"{arm}/{world}": judge(
            "entry_behavior",
            metric="productive_share_of_entrants",
            higher_is_better=True,
            world_id=world,
            arm_key=(arm, field),
            reference_key=("linear_batch", field),
            difference_in_differences=True,
        )
        for arm in NONLINEAR_ARMS
        for world in INTERACTION_FIT_WORLDS
    }
    entry_passed = sorted(k for k, v in entry.items() if v["pass"])
    if saturated:
        entry_disposition = "ENTRY_SATURATED"
    elif entry_passed:
        entry_disposition = "ENTRY_CHANNEL_ACTIVE"
    else:
        entry_disposition = "ENTRY_CHANNEL_ABSENT"
    exit_channel = {
        f"{arm}/{world}": judge(
            "exit_channel",
            metric="active_failed_rate",
            higher_is_better=False,
            world_id=world,
            arm_key=(arm, field),
            reference_key=("linear_batch", field),
            difference_in_differences=True,
        )
        for arm in NONLINEAR_ARMS
        for world in INTERACTION_FIT_WORLDS
    }
    exit_passed = sorted(k for k, v in exit_channel.items() if v["pass"])
    learned = sorted(k for k, v in learning.items() if v["pass"])
    return {
        "field_nonlinear_learning": (
            "OUT_OF_SAMPLE_NONLINEAR_LEARNING" if learned else "NO_NONLINEAR_LEARNING"
        ),
        "learned_family_geometries": learned,
        "geometry_crossover": crossover_disposition,
        "evidence_scale": evidence_scale,
        "useful_learners_field": useful_field,
        "useful_learners_local": useful_local,
        "entry": entry_disposition,
        "entry_passing_family_geometries": entry_passed,
        "exit": "EXIT_CHANNEL_ACTIVE" if exit_passed else "EXIT_CHANNEL_ABSENT",
        "exit_passing_family_geometries": exit_passed,
        "reports": {
            "field_nonlinear_learning": learning,
            "field_geometry_crossover": crossover,
            "useful_learning": useful,
            "entry_behavior": entry,
            "exit_channel": exit_channel,
        },
    }


def decide(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    indexed = abm1._index(rows)
    replicates = sorted({int(row["replicate"]) for row in rows})
    return {
        "status": "FROZEN_SYNTHETIC_DECISION_COMPLETE",
        "by_visibility": {
            visibility_id: decide_visibility(indexed, replicates, contract, visibility_id)
            for visibility_id in VISIBILITY_IDS
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
    required = int(contract["decision_contract"]["improved_replicates_at_least"])
    if config.replicates < required:
        dispositions: dict[str, Any] = {
            "status": "NOT_EVALUATED_SMOKE_PROFILE",
            "by_visibility": {
                visibility_id: "NOT_EVALUATED" for visibility_id in VISIBILITY_IDS
            },
        }
    else:
        dispositions = decide(rows, contract)
    return {
        "schema_version": "pivot_nonlinear_policy_abm_scorecard.v2",
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
        "# Pivot nonlinear policy ABM scorecard v2",
        "",
        f"Profile: `{summary['profile']}`",
        "",
        f"Family transfer: `{summary['family_transfer']['status']}`",
        "",
        "| Visibility | Fit world | Arm | Evidence | Entry | Productive share of entrants | "
        "Active failed | Probe Brier |",
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
                brier=means["probe_brier"],
            )
        )
    lines.extend(["", "## Frozen dispositions", ""])
    by_visibility = summary["dispositions"]["by_visibility"]
    for visibility_id in VISIBILITY_IDS:
        entry = by_visibility[visibility_id]
        if not isinstance(entry, Mapping):
            lines.append(f"- {visibility_id}: `{entry}`")
            continue
        lines.append(
            "- {visibility}: learning `{learning}`, crossover `{crossover}`, "
            "evidence scale `{scale}`, entry `{entry}`, exit `{exit}`".format(
                visibility=visibility_id,
                learning=entry["field_nonlinear_learning"],
                crossover=entry["geometry_crossover"],
                scale=entry["evidence_scale"],
                entry=entry["entry"],
                exit=entry["exit"],
            )
        )
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
        "schema_version": "pivot_nonlinear_policy_abm_receipt.v2",
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


def _plan_payload(contract: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "status": "PRECOMMIT_PLAN_VALID",
        "contract_path": str(config_path.resolve()),
        "contract_sha256": abm1._sha256(config_path),
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
    parser.add_argument("--workers", type=int, default=16)
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
    transfer = abm1.family_transfer_check(contract)
    rows = run_abm(contract, args.profile, workers=args.workers)
    summary = {
        **summarize(rows, transfer, contract, args.profile),
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
