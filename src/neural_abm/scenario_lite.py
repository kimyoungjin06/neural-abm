"""Torch-free scenario comparison helpers for bounded-scalar ABM questions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neural_abm.workflow_lite import (
    BoundedScalarWorkflowSpec,
    run_bounded_scalar_workflow,
)

ScenarioAgentFactory = Callable[["ScenarioDefinition"], Sequence[Any]]
ScenarioNeighborBuilder = Callable[
    [Sequence[Any], "ScenarioDefinition"], list[list[int]]
]
ScenarioLocalUpdate = Callable[[Any, "ScenarioDefinition"], float]
ScenarioDomainTransition = Callable[
    [Any, float, "ScenarioDefinition"], Mapping[str, Any]
]
ScenarioMicroFields = Callable[[Any, "ScenarioDefinition"], Mapping[str, Any]]
ScenarioAggregateFields = Callable[
    [Sequence[Any], "ScenarioDefinition"], Mapping[str, Any]
]

ReplicatedAgentFactory = Callable[["ScenarioReplicateContext"], Sequence[Any]]
ReplicatedNeighborBuilder = Callable[
    [Sequence[Any], "ScenarioReplicateContext"],
    list[list[int]],
]
ReplicatedLocalUpdate = Callable[[Any, "ScenarioReplicateContext"], float]
ReplicatedDomainTransition = Callable[
    [Any, float, "ScenarioReplicateContext"],
    Mapping[str, Any],
]
ReplicatedMicroFields = Callable[[Any, "ScenarioReplicateContext"], Mapping[str, Any]]
ReplicatedAggregateFields = Callable[
    [Sequence[Any], "ScenarioReplicateContext"],
    Mapping[str, Any],
]

SUCCESS_DIRECTIONS: tuple[str, ...] = ("increase", "decrease")


@dataclass(frozen=True)
class ScenarioDefinition:
    """A named baseline or counterfactual scenario."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must be non-empty")
        if not self.description:
            raise ValueError("scenario description must be non-empty")


@dataclass(frozen=True)
class BoundedScalarScenarioSpec:
    """Public-facing settings for a bounded-scalar scenario comparison.

    The caller supplies the research question, outcome meaning, and comparison
    threshold. The scenario layer only orchestrates execution and reports
    whether the simulated delta mechanically satisfies that supplied threshold.
    """

    research_question: str
    state_field: str
    channel_name: str
    transition_label: str
    commit_mode: str
    outcome_field: str
    baseline_name: str = "baseline"
    success_direction: str = "increase"
    success_min_delta: float = 0.0
    steps: int = 1
    peer_rule: str = "output_similarity"
    peer_similarity_threshold: float = 0.70
    social_alpha: float = 0.45
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    round_digits: int | None = 4

    def __post_init__(self) -> None:
        if not self.research_question:
            raise ValueError("research_question must be non-empty")
        if not self.outcome_field:
            raise ValueError("outcome_field must be non-empty")
        if not self.baseline_name:
            raise ValueError("baseline_name must be non-empty")
        if self.success_direction not in SUCCESS_DIRECTIONS:
            allowed = ", ".join(SUCCESS_DIRECTIONS)
            raise ValueError(f"success_direction must be one of {allowed}")
        if self.success_min_delta < 0.0:
            raise ValueError("success_min_delta must be >= 0")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        BoundedScalarWorkflowSpec(
            domain_question=self.research_question,
            state_field=self.state_field,
            channel_name=self.channel_name,
            transition_label=self.transition_label,
            commit_mode=self.commit_mode,
            peer_rule=self.peer_rule,
            peer_similarity_threshold=self.peer_similarity_threshold,
            social_alpha=self.social_alpha,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            round_digits=self.round_digits,
        )


@dataclass(frozen=True)
class ScenarioComparison:
    """Outcome comparison between the baseline and one counterfactual."""

    baseline: str
    scenario: str
    outcome_field: str
    baseline_value: float
    scenario_value: float
    delta: float
    success_criterion: str
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "scenario": self.scenario,
            "outcome_field": self.outcome_field,
            "baseline_value": self.baseline_value,
            "scenario_value": self.scenario_value,
            "delta": self.delta,
            "success_criterion": self.success_criterion,
            "success": self.success,
        }


@dataclass(frozen=True)
class BoundedScalarScenarioResult:
    """Result envelope for bounded-scalar scenario comparisons."""

    spec: BoundedScalarScenarioSpec
    scenarios: dict[str, dict[str, Any]]
    comparisons: list[ScenarioComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "surface": "neural_abm.scenario_lite",
            "base_surface": "neural_abm.api_lite",
            "default_profile": "torch-free",
            "research_question": self.spec.research_question,
            "state_field": self.spec.state_field,
            "outcome_field": self.spec.outcome_field,
            "baseline": self.spec.baseline_name,
            "steps": self.spec.steps,
            "scenario_count": len(self.scenarios),
            "scenarios": self.scenarios,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }


def run_bounded_scalar_scenarios(
    *,
    scenarios: Sequence[ScenarioDefinition],
    spec: BoundedScalarScenarioSpec,
    make_agents: ScenarioAgentFactory,
    build_neighbors: ScenarioNeighborBuilder,
    local_update: ScenarioLocalUpdate,
    domain_transition: ScenarioDomainTransition,
    micro_fields: ScenarioMicroFields | None = None,
    aggregate_fields: ScenarioAggregateFields | None = None,
) -> BoundedScalarScenarioResult:
    """Run baseline and counterfactual bounded-scalar scenarios."""

    scenario_results: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        if scenario.name in scenario_results:
            raise ValueError(f"duplicate scenario name: {scenario.name}")
        scenario_results[scenario.name] = _run_single_scenario(
            scenario=scenario,
            spec=spec,
            make_agents=make_agents,
            build_neighbors=build_neighbors,
            local_update=local_update,
            domain_transition=domain_transition,
            micro_fields=micro_fields,
            aggregate_fields=aggregate_fields,
        )

    if spec.baseline_name not in scenario_results:
        raise ValueError(f"baseline scenario not found: {spec.baseline_name}")

    comparisons = _compare_to_baseline(
        scenario_results=scenario_results,
        spec=spec,
    )
    return BoundedScalarScenarioResult(
        spec=spec,
        scenarios=scenario_results,
        comparisons=comparisons,
    )


def _run_single_scenario(
    *,
    scenario: ScenarioDefinition,
    spec: BoundedScalarScenarioSpec,
    make_agents: ScenarioAgentFactory,
    build_neighbors: ScenarioNeighborBuilder,
    local_update: ScenarioLocalUpdate,
    domain_transition: ScenarioDomainTransition,
    micro_fields: ScenarioMicroFields | None,
    aggregate_fields: ScenarioAggregateFields | None,
) -> dict[str, Any]:
    agents = list(make_agents(scenario))
    history: list[dict[str, Any]] = []

    for step in range(spec.steps):
        workflow_spec = BoundedScalarWorkflowSpec(
            domain_question=f"{spec.research_question}:{scenario.name}:step_{step + 1}",
            state_field=spec.state_field,
            channel_name=spec.channel_name,
            transition_label=spec.transition_label,
            commit_mode=spec.commit_mode,
            peer_rule=spec.peer_rule,
            peer_similarity_threshold=spec.peer_similarity_threshold,
            social_alpha=spec.social_alpha,
            lower_bound=spec.lower_bound,
            upper_bound=spec.upper_bound,
            round_digits=spec.round_digits,
        )
        neighbors = build_neighbors(agents, scenario)
        workflow_result = run_bounded_scalar_workflow(
            agents=agents,
            neighbors=neighbors,
            spec=workflow_spec,
            local_update=lambda agent: local_update(agent, scenario),
            domain_transition=lambda agent, value: domain_transition(
                agent,
                value,
                scenario,
            ),
            micro_fields=(
                None
                if micro_fields is None
                else lambda agent: micro_fields(agent, scenario)
            ),
            aggregate_fields=(
                None
                if aggregate_fields is None
                else lambda scenario_agents: aggregate_fields(
                    scenario_agents,
                    scenario,
                )
            ),
        ).to_dict()
        workflow_result["step"] = step + 1
        history.append(workflow_result)

    final = history[-1]
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "parameters": dict(scenario.parameters),
        "final": final,
        "history": history,
    }


def _compare_to_baseline(
    *,
    scenario_results: Mapping[str, dict[str, Any]],
    spec: BoundedScalarScenarioSpec,
) -> list[ScenarioComparison]:
    baseline = scenario_results[spec.baseline_name]
    baseline_value = _outcome_value(baseline, spec.outcome_field)
    comparisons: list[ScenarioComparison] = []
    for name, result in scenario_results.items():
        if name == spec.baseline_name:
            continue
        scenario_value = _outcome_value(result, spec.outcome_field)
        delta = scenario_value - baseline_value
        comparisons.append(
            ScenarioComparison(
                baseline=spec.baseline_name,
                scenario=name,
                outcome_field=spec.outcome_field,
                baseline_value=baseline_value,
                scenario_value=scenario_value,
                delta=delta,
                success_criterion=_success_criterion(spec),
                success=_is_success(delta, spec),
            )
        )
    return comparisons


def _outcome_value(result: Mapping[str, Any], outcome_field: str) -> float:
    final = result["final"]
    aggregate = final["aggregate_audit"]
    if outcome_field not in aggregate:
        raise ValueError(f"outcome field missing from aggregate audit: {outcome_field}")
    value = aggregate[outcome_field]
    if isinstance(value, bool):
        raise ValueError(f"outcome field must be numeric, got bool: {outcome_field}")
    return float(value)


def _success_criterion(spec: BoundedScalarScenarioSpec) -> str:
    comparator = ">" if spec.success_direction == "increase" else "<"
    signed_threshold = (
        spec.success_min_delta
        if spec.success_direction == "increase"
        else -spec.success_min_delta
    )
    return f"delta {comparator} {signed_threshold:g}"


def _is_success(delta: float, spec: BoundedScalarScenarioSpec) -> bool:
    if spec.success_direction == "increase":
        return delta > spec.success_min_delta
    return delta < -spec.success_min_delta


@dataclass(frozen=True)
class ReplicationSpec:
    """Settings for seed-based replicated scenario runs.

    Replicate ``r`` uses the same seed in every scenario, so scenario
    comparisons are paired through common random numbers.
    """

    replicates: int
    base_seed: int = 0

    def __post_init__(self) -> None:
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        if self.base_seed < 0:
            raise ValueError("base_seed must be >= 0")


@dataclass(frozen=True)
class ScenarioReplicateContext:
    """One scenario replicate: the scenario, replicate index, and seeded rng."""

    scenario: ScenarioDefinition
    replicate: int
    rng: np.random.Generator


@dataclass(frozen=True)
class ReplicatedScenarioComparison:
    """Paired outcome comparison between the baseline and one counterfactual.

    Deltas are paired per replicate through common random numbers. ``success``
    applies the criterion to the mean delta; ``success_fraction`` reports the
    fraction of paired replicates that satisfy the criterion individually.
    """

    baseline: str
    scenario: str
    outcome_field: str
    replicates: int
    baseline_mean: float
    scenario_mean: float
    mean_delta: float
    delta_std: float
    delta_ci95: tuple[float, float]
    success_criterion: str
    success: bool
    success_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "scenario": self.scenario,
            "outcome_field": self.outcome_field,
            "replicates": self.replicates,
            "baseline_mean": self.baseline_mean,
            "scenario_mean": self.scenario_mean,
            "mean_delta": self.mean_delta,
            "delta_std": self.delta_std,
            "delta_ci95": list(self.delta_ci95),
            "success_criterion": self.success_criterion,
            "success": self.success,
            "success_fraction": self.success_fraction,
        }


@dataclass(frozen=True)
class ReplicatedScenarioResult:
    """Result envelope for replicated bounded-scalar scenario comparisons."""

    spec: BoundedScalarScenarioSpec
    replication: ReplicationSpec
    scenarios: dict[str, dict[str, Any]]
    comparisons: list[ReplicatedScenarioComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "surface": "neural_abm.scenario_lite",
            "base_surface": "neural_abm.api_lite",
            "default_profile": "torch-free",
            "research_question": self.spec.research_question,
            "state_field": self.spec.state_field,
            "outcome_field": self.spec.outcome_field,
            "baseline": self.spec.baseline_name,
            "steps": self.spec.steps,
            "replicates": self.replication.replicates,
            "base_seed": self.replication.base_seed,
            "scenario_count": len(self.scenarios),
            "scenarios": self.scenarios,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }


def run_replicated_bounded_scalar_scenarios(
    *,
    scenarios: Sequence[ScenarioDefinition],
    spec: BoundedScalarScenarioSpec,
    replication: ReplicationSpec,
    make_agents: ReplicatedAgentFactory,
    build_neighbors: ReplicatedNeighborBuilder,
    local_update: ReplicatedLocalUpdate,
    domain_transition: ReplicatedDomainTransition,
    micro_fields: ReplicatedMicroFields | None = None,
    aggregate_fields: ReplicatedAggregateFields | None = None,
) -> ReplicatedScenarioResult:
    """Run seed-paired replicates of baseline and counterfactual scenarios.

    Every callback receives a :class:`ScenarioReplicateContext` so agent
    populations, networks, and updates can draw from the replicate rng. The
    result reports per-scenario outcome distributions and paired baseline
    comparisons instead of a single deterministic delta.
    """

    scenario_results: dict[str, dict[str, Any]] = {}
    outcome_samples: dict[str, list[float]] = {}
    for scenario in scenarios:
        if scenario.name in scenario_results:
            raise ValueError(f"duplicate scenario name: {scenario.name}")
        summary, outcomes = _run_replicated_scenario(
            scenario=scenario,
            spec=spec,
            replication=replication,
            make_agents=make_agents,
            build_neighbors=build_neighbors,
            local_update=local_update,
            domain_transition=domain_transition,
            micro_fields=micro_fields,
            aggregate_fields=aggregate_fields,
        )
        scenario_results[scenario.name] = summary
        outcome_samples[scenario.name] = outcomes

    if spec.baseline_name not in scenario_results:
        raise ValueError(f"baseline scenario not found: {spec.baseline_name}")

    comparisons = _compare_replicated_to_baseline(
        outcome_samples=outcome_samples,
        spec=spec,
        replication=replication,
    )
    return ReplicatedScenarioResult(
        spec=spec,
        replication=replication,
        scenarios=scenario_results,
        comparisons=comparisons,
    )


def _replicate_rng(replication: ReplicationSpec, replicate: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence((replication.base_seed, replicate))
    )


def _run_replicated_scenario(
    *,
    scenario: ScenarioDefinition,
    spec: BoundedScalarScenarioSpec,
    replication: ReplicationSpec,
    make_agents: ReplicatedAgentFactory,
    build_neighbors: ReplicatedNeighborBuilder,
    local_update: ReplicatedLocalUpdate,
    domain_transition: ReplicatedDomainTransition,
    micro_fields: ReplicatedMicroFields | None,
    aggregate_fields: ReplicatedAggregateFields | None,
) -> tuple[dict[str, Any], list[float]]:
    outcomes: list[float] = []
    aggregate_samples: dict[str, list[float]] = {}
    trajectories: list[list[float]] = []
    first_replicate_final: dict[str, Any] | None = None
    mean_state_key = f"mean_{spec.state_field}"

    for replicate in range(replication.replicates):
        context = ScenarioReplicateContext(
            scenario=scenario,
            replicate=replicate,
            rng=_replicate_rng(replication, replicate),
        )
        replicate_result = _run_single_scenario(
            scenario=scenario,
            spec=spec,
            make_agents=lambda _scenario: make_agents(context),
            build_neighbors=lambda agents, _scenario: build_neighbors(agents, context),
            local_update=lambda agent, _scenario: local_update(agent, context),
            domain_transition=lambda agent, value, _scenario: domain_transition(
                agent,
                value,
                context,
            ),
            micro_fields=(
                None
                if micro_fields is None
                else lambda agent, _scenario: micro_fields(agent, context)
            ),
            aggregate_fields=(
                None
                if aggregate_fields is None
                else lambda agents, _scenario: aggregate_fields(agents, context)
            ),
        )
        outcomes.append(_outcome_value(replicate_result, spec.outcome_field))
        final_aggregate = replicate_result["final"]["aggregate_audit"]
        for key, value in final_aggregate.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            aggregate_samples.setdefault(key, []).append(float(value))
        trajectories.append(
            [
                float(step_result["aggregate_audit"][mean_state_key])
                for step_result in replicate_result["history"]
            ]
        )
        if first_replicate_final is None:
            first_replicate_final = replicate_result["final"]

    trajectory_array = np.asarray(trajectories, dtype=np.float64)
    summary = {
        "scenario": scenario.name,
        "description": scenario.description,
        "parameters": dict(scenario.parameters),
        "replicates": replication.replicates,
        "outcome_field": spec.outcome_field,
        "outcome_values": [
            _round_float(value, spec.round_digits) for value in outcomes
        ],
        "outcome_summary": _distribution_summary(outcomes, spec.round_digits),
        "aggregate_summaries": {
            key: _distribution_summary(values, spec.round_digits)
            for key, values in aggregate_samples.items()
        },
        "mean_state_trajectory": [
            _round_float(value, spec.round_digits)
            for value in trajectory_array.mean(axis=0)
        ],
        "std_state_trajectory": [
            _round_float(value, spec.round_digits)
            for value in trajectory_array.std(axis=0)
        ],
        "first_replicate_final": first_replicate_final,
    }
    return summary, outcomes


def _compare_replicated_to_baseline(
    *,
    outcome_samples: Mapping[str, Sequence[float]],
    spec: BoundedScalarScenarioSpec,
    replication: ReplicationSpec,
) -> list[ReplicatedScenarioComparison]:
    baseline_values = np.asarray(outcome_samples[spec.baseline_name], dtype=np.float64)
    comparisons: list[ReplicatedScenarioComparison] = []
    for name, values in outcome_samples.items():
        if name == spec.baseline_name:
            continue
        scenario_values = np.asarray(values, dtype=np.float64)
        deltas = scenario_values - baseline_values
        mean_delta = float(deltas.mean())
        delta_std = float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0
        ci_low, ci_high = (float(value) for value in np.percentile(deltas, [2.5, 97.5]))
        successes = [_is_success(float(delta), spec) for delta in deltas]
        digits = spec.round_digits
        comparisons.append(
            ReplicatedScenarioComparison(
                baseline=spec.baseline_name,
                scenario=name,
                outcome_field=spec.outcome_field,
                replicates=replication.replicates,
                baseline_mean=_round_float(float(baseline_values.mean()), digits),
                scenario_mean=_round_float(float(scenario_values.mean()), digits),
                mean_delta=_round_float(mean_delta, digits),
                delta_std=_round_float(delta_std, digits),
                delta_ci95=(
                    _round_float(ci_low, digits),
                    _round_float(ci_high, digits),
                ),
                success_criterion=f"mean {_success_criterion(spec)}",
                success=_is_success(mean_delta, spec),
                success_fraction=_round_float(
                    float(np.mean(successes)) if successes else 0.0,
                    digits,
                ),
            )
        )
    return comparisons


def _round_float(value: float, digits: int | None) -> float:
    numeric = float(value)
    return numeric if digits is None else round(numeric, digits)


def _distribution_summary(
    values: Sequence[float],
    digits: int | None,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    ci_low, ci_high = (float(value) for value in np.percentile(array, [2.5, 97.5]))
    return {
        "mean": _round_float(float(array.mean()), digits),
        "std": _round_float(std, digits),
        "min": _round_float(float(array.min()), digits),
        "max": _round_float(float(array.max()), digits),
        "ci95_low": _round_float(ci_low, digits),
        "ci95_high": _round_float(ci_high, digits),
    }


__all__ = [
    "BoundedScalarScenarioResult",
    "BoundedScalarScenarioSpec",
    "ReplicatedAgentFactory",
    "ReplicatedAggregateFields",
    "ReplicatedDomainTransition",
    "ReplicatedLocalUpdate",
    "ReplicatedMicroFields",
    "ReplicatedNeighborBuilder",
    "ReplicatedScenarioComparison",
    "ReplicatedScenarioResult",
    "ReplicationSpec",
    "ScenarioAggregateFields",
    "ScenarioAgentFactory",
    "ScenarioComparison",
    "ScenarioDefinition",
    "ScenarioDomainTransition",
    "ScenarioLocalUpdate",
    "ScenarioMicroFields",
    "ScenarioNeighborBuilder",
    "ScenarioReplicateContext",
    "SUCCESS_DIRECTIONS",
    "run_bounded_scalar_scenarios",
    "run_replicated_bounded_scalar_scenarios",
]
