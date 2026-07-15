"""Torch-free scenario comparison helpers for bounded-scalar ABM questions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
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
NORMAL_95_Z = 1.959963984540054


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
    ``round_digits`` applies to serialized audit rows only; scenario effects and
    distribution summaries retain the unrounded values used for analysis.
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
        if (
            isinstance(self.success_min_delta, bool)
            or not isinstance(self.success_min_delta, Real)
            or not np.isfinite(self.success_min_delta)
        ):
            raise ValueError("success_min_delta must be a finite real number")
        if self.success_min_delta < 0.0:
            raise ValueError("success_min_delta must be >= 0")
        if isinstance(self.steps, bool) or not isinstance(self.steps, Integral):
            raise ValueError("steps must be an integer")
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
            "analysis_precision": "unrounded",
            "audit_round_digits": self.spec.round_digits,
            "scenario_count": len(self.scenarios),
            "scenarios": {
                name: _serialize_single_scenario(result, self.spec.round_digits)
                for name, result in self.scenarios.items()
            },
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

    scenario_list = _validated_scenarios(scenarios, spec.baseline_name)
    executed_results: dict[str, dict[str, Any]] = {}
    for scenario in _baseline_first(scenario_list, spec.baseline_name):
        result = _run_single_scenario(
            scenario=scenario,
            spec=spec,
            make_agents=make_agents,
            build_neighbors=build_neighbors,
            local_update=local_update,
            domain_transition=domain_transition,
            micro_fields=micro_fields,
            aggregate_fields=aggregate_fields,
        )
        _outcome_value(result, spec.outcome_field)
        executed_results[scenario.name] = result

    scenario_results = {
        scenario.name: executed_results[scenario.name] for scenario in scenario_list
    }

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
        ).to_dict(round_values=False)
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
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"outcome field must be numeric: {outcome_field}") from error
    if not np.isfinite(numeric):
        raise ValueError(f"outcome field must be finite: {outcome_field}")
    return numeric


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

    Replicate ``r`` uses the same initial seed in every scenario, so outcomes
    are paired by replicate. Strict common-random-number alignment additionally
    requires caller callbacks to use scenario-independent component streams or
    otherwise consume identical draws.
    """

    replicates: int
    base_seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.replicates, bool) or not isinstance(
            self.replicates, Integral
        ):
            raise ValueError("replicates must be an integer")
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, Integral):
            raise ValueError("base_seed must be an integer")
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

    Deltas are paired by replicate. ``delta_ci95`` is a normal-approximation
    confidence interval for the paired mean effect. The separate empirical
    percentile interval describes the central spread of replicate deltas and
    is not a confidence interval for their mean. With one replicate the mean
    interval is unavailable rather than represented as a zero-width interval.
    ``success`` applies the criterion to the mean delta; ``success_fraction``
    reports the fraction of paired replicates that satisfy the criterion
    individually.
    """

    baseline: str
    scenario: str
    outcome_field: str
    replicates: int
    baseline_mean: float
    scenario_mean: float
    mean_delta: float
    delta_std: float
    delta_ci95: tuple[float, float] | None
    success_criterion: str
    success: bool
    success_fraction: float
    delta_empirical_percentile_interval95: tuple[float, float] | None = None
    paired_deltas: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        mean_interval = None if self.delta_ci95 is None else list(self.delta_ci95)
        payload = {
            "baseline": self.baseline,
            "scenario": self.scenario,
            "outcome_field": self.outcome_field,
            "replicates": self.replicates,
            "baseline_mean": self.baseline_mean,
            "scenario_mean": self.scenario_mean,
            "mean_delta": self.mean_delta,
            "delta_std": self.delta_std,
            "delta_ci95": mean_interval,
            "mean_effect_ci95": mean_interval,
            "delta_ci95_method": (
                "normal_approximation_for_paired_mean"
                if self.delta_ci95 is not None
                else "unavailable_requires_at_least_2_replicates"
            ),
            "success_criterion": self.success_criterion,
            "success": self.success,
            "success_fraction": self.success_fraction,
        }
        if self.delta_empirical_percentile_interval95 is not None:
            payload["delta_empirical_percentile_interval95"] = list(
                self.delta_empirical_percentile_interval95
            )
        if self.paired_deltas is not None:
            payload["paired_deltas"] = list(self.paired_deltas)
        return payload


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
            "analysis_precision": "unrounded",
            "audit_round_digits": self.spec.round_digits,
            "scenario_count": len(self.scenarios),
            "scenarios": {
                name: _serialize_replicated_scenario(result, self.spec.round_digits)
                for name, result in self.scenarios.items()
            },
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

    scenario_list = _validated_scenarios(scenarios, spec.baseline_name)
    executed_results: dict[str, dict[str, Any]] = {}
    executed_outcomes: dict[str, list[float]] = {}
    for scenario in _baseline_first(scenario_list, spec.baseline_name):
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
        executed_results[scenario.name] = summary
        executed_outcomes[scenario.name] = outcomes

    scenario_results = {
        scenario.name: executed_results[scenario.name] for scenario in scenario_list
    }
    outcome_samples = {
        scenario.name: executed_outcomes[scenario.name] for scenario in scenario_list
    }

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
        "outcome_values": list(outcomes),
        "outcome_summary": _distribution_summary(outcomes),
        "aggregate_summaries": {
            key: _distribution_summary(values)
            for key, values in aggregate_samples.items()
        },
        "mean_state_trajectory": [
            float(value) for value in trajectory_array.mean(axis=0)
        ],
        "std_state_trajectory": [
            float(value) for value in trajectory_array.std(axis=0)
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
        percentile_low, percentile_high = (
            float(value) for value in np.percentile(deltas, [2.5, 97.5])
        )
        mean_ci95 = _normal_mean_ci95(deltas)
        successes = [_is_success(float(delta), spec) for delta in deltas]
        comparisons.append(
            ReplicatedScenarioComparison(
                baseline=spec.baseline_name,
                scenario=name,
                outcome_field=spec.outcome_field,
                replicates=replication.replicates,
                baseline_mean=float(baseline_values.mean()),
                scenario_mean=float(scenario_values.mean()),
                mean_delta=mean_delta,
                delta_std=delta_std,
                delta_ci95=mean_ci95,
                success_criterion=f"mean {_success_criterion(spec)}",
                success=_is_success(mean_delta, spec),
                success_fraction=(float(np.mean(successes)) if successes else 0.0),
                delta_empirical_percentile_interval95=(
                    percentile_low,
                    percentile_high,
                ),
                paired_deltas=tuple(float(delta) for delta in deltas),
            )
        )
    return comparisons


def _distribution_summary(
    values: Sequence[float],
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    percentile_low, percentile_high = (
        float(value) for value in np.percentile(array, [2.5, 97.5])
    )
    mean_ci95 = _normal_mean_ci95(array)
    ci_low = None if mean_ci95 is None else mean_ci95[0]
    ci_high = None if mean_ci95 is None else mean_ci95[1]
    return {
        "mean": float(array.mean()),
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "mean_ci95": None if mean_ci95 is None else [ci_low, ci_high],
        "ci95_method": (
            "normal_approximation_for_mean"
            if mean_ci95 is not None
            else "unavailable_requires_at_least_2_replicates"
        ),
        "empirical_percentile_interval95": [percentile_low, percentile_high],
    }


def _normal_mean_ci95(
    values: Sequence[float] | np.ndarray,
) -> tuple[float, float] | None:
    """Return an analytic 95% CI for a replicated mean, or ``None`` for n < 2.

    The interval uses the sample standard error and the asymptotic normal
    critical value. For paired effects, callers pass the paired deltas rather
    than the two marginal outcome samples.
    """

    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.size <= 1:
        return None
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    half_width = NORMAL_95_Z * standard_error
    return mean - half_width, mean + half_width


def _validated_scenarios(
    scenarios: Sequence[ScenarioDefinition],
    baseline_name: str,
) -> tuple[ScenarioDefinition, ...]:
    scenario_list = tuple(scenarios)
    seen: set[str] = set()
    for scenario in scenario_list:
        if scenario.name in seen:
            raise ValueError(f"duplicate scenario name: {scenario.name}")
        seen.add(scenario.name)
    if baseline_name not in seen:
        raise ValueError(f"baseline scenario not found: {baseline_name}")
    return scenario_list


def _baseline_first(
    scenarios: Sequence[ScenarioDefinition],
    baseline_name: str,
) -> tuple[ScenarioDefinition, ...]:
    """Order execution for early baseline validation without changing output order."""

    baseline = next(
        scenario for scenario in scenarios if scenario.name == baseline_name
    )
    return (baseline, *(scenario for scenario in scenarios if scenario is not baseline))


def _round_presentation_value(value: Any, digits: int | None) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if digits is None else round(numeric, digits)
    if isinstance(value, Mapping):
        return {
            key: _round_presentation_value(item, digits) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_round_presentation_value(item, digits) for item in value]
    if isinstance(value, tuple):
        return tuple(_round_presentation_value(item, digits) for item in value)
    return value


def _serialize_workflow_payload(
    payload: Mapping[str, Any],
    digits: int | None,
) -> dict[str, Any]:
    serialized = dict(payload)
    serialized["aggregate_audit"] = _round_presentation_value(
        payload["aggregate_audit"], digits
    )
    serialized["micro_audit"] = _round_presentation_value(
        payload["micro_audit"], digits
    )
    return serialized


def _serialize_single_scenario(
    result: Mapping[str, Any],
    digits: int | None,
) -> dict[str, Any]:
    history = [
        _serialize_workflow_payload(step_result, digits)
        for step_result in result["history"]
    ]
    return {
        "scenario": result["scenario"],
        "description": result["description"],
        "parameters": dict(result["parameters"]),
        "final": history[-1],
        "history": history,
    }


def _serialize_replicated_scenario(
    result: Mapping[str, Any],
    digits: int | None,
) -> dict[str, Any]:
    serialized = dict(result)
    serialized["parameters"] = dict(result["parameters"])
    first_final = result.get("first_replicate_final")
    if first_final is not None:
        serialized["first_replicate_final"] = _serialize_workflow_payload(
            first_final,
            digits,
        )
    return serialized


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
