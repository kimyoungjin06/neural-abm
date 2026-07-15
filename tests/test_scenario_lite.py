"""Unit tests for torch-free scenario comparison and replication helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pytest

from neural_abm.scenario_lite import (
    BoundedScalarScenarioSpec,
    ReplicationSpec,
    ScenarioDefinition,
    ScenarioReplicateContext,
    run_bounded_scalar_scenarios,
    run_replicated_bounded_scalar_scenarios,
)


@dataclass
class _Agent:
    agent_id: int
    x: float
    hit: bool = False


def _spec(**overrides: Any) -> BoundedScalarScenarioSpec:
    settings: dict[str, Any] = {
        "research_question": "does_the_boost_scenario_raise_the_hit_rate",
        "state_field": "x",
        "channel_name": "x",
        "transition_label": "hit = x >= 0.5",
        "commit_mode": "domain_hit_threshold",
        "outcome_field": "hit_rate",
        "success_min_delta": 0.05,
        "steps": 2,
    }
    settings.update(overrides)
    return BoundedScalarScenarioSpec(**settings)


def _scenarios() -> tuple[ScenarioDefinition, ...]:
    return (
        ScenarioDefinition(name="baseline", description="No boost."),
        ScenarioDefinition(
            name="boost",
            description="Initial values shifted up.",
            parameters={"lift": 0.2},
        ),
    )


def _make_agents(scenario: ScenarioDefinition) -> list[_Agent]:
    lift = float(scenario.parameters.get("lift", 0.0))
    return [_Agent(agent_id=index, x=0.3 + lift) for index in range(4)]


def _build_neighbors(
    agents: list[_Agent],
    scenario: ScenarioDefinition,
) -> list[list[int]]:
    del scenario
    count = len(agents)
    return [[(index + 1) % count] for index in range(count)]


def _local_update(agent: _Agent, scenario: ScenarioDefinition) -> float:
    del scenario
    return 0.0


def _domain_transition(
    agent: _Agent,
    value: float,
    scenario: ScenarioDefinition,
) -> dict[str, Any]:
    del scenario
    agent.x = value
    agent.hit = agent.x >= 0.5
    return {"hit": agent.hit}


def _aggregate_fields(
    agents: list[_Agent],
    scenario: ScenarioDefinition,
) -> dict[str, Any]:
    del scenario
    return {"hit_rate": sum(agent.hit for agent in agents) / len(agents)}


def _run_single(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "scenarios": _scenarios(),
        "spec": _spec(),
        "make_agents": _make_agents,
        "build_neighbors": _build_neighbors,
        "local_update": _local_update,
        "domain_transition": _domain_transition,
        "aggregate_fields": _aggregate_fields,
    }
    settings.update(overrides)
    return run_bounded_scalar_scenarios(**settings).to_dict()


def test_scenario_definition_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="scenario name"):
        ScenarioDefinition(name="", description="d")
    with pytest.raises(ValueError, match="scenario description"):
        ScenarioDefinition(name="n", description="")


def test_scenario_spec_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="success_direction"):
        _spec(success_direction="sideways")
    with pytest.raises(ValueError, match="success_min_delta"):
        _spec(success_min_delta=-0.1)
    for invalid_threshold in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite real"):
            _spec(success_min_delta=invalid_threshold)
    with pytest.raises(ValueError, match="steps"):
        _spec(steps=0)
    for invalid_steps in (1.5, True):
        with pytest.raises(ValueError, match="steps must be an integer"):
            _spec(steps=invalid_steps)
    with pytest.raises(ValueError, match="social_alpha"):
        _spec(social_alpha=1.5)


def test_single_run_rejects_duplicate_scenario_names() -> None:
    duplicated = (
        ScenarioDefinition(name="baseline", description="a"),
        ScenarioDefinition(name="baseline", description="b"),
    )
    with pytest.raises(ValueError, match="duplicate scenario name"):
        _run_single(scenarios=duplicated)


def test_single_run_rejects_missing_baseline() -> None:
    callback_calls: list[str] = []

    def recording_factory(scenario: ScenarioDefinition) -> list[_Agent]:
        callback_calls.append(scenario.name)
        return _make_agents(scenario)

    with pytest.raises(ValueError, match="baseline scenario not found"):
        _run_single(
            spec=_spec(baseline_name="missing"),
            make_agents=recording_factory,
        )
    assert callback_calls == []


def test_single_run_rejects_missing_outcome_field() -> None:
    with pytest.raises(ValueError, match="outcome field missing"):
        _run_single(spec=_spec(outcome_field="unknown_field"))


def test_single_run_rejects_bool_outcome_field() -> None:
    def bool_aggregate(
        agents: list[_Agent],
        scenario: ScenarioDefinition,
    ) -> dict[str, Any]:
        del scenario
        return {"hit_rate": any(agent.hit for agent in agents)}

    with pytest.raises(ValueError, match="must be numeric"):
        _run_single(aggregate_fields=bool_aggregate)


@pytest.mark.parametrize("outcome", [float("nan"), float("inf"), float("-inf")])
def test_single_run_rejects_non_finite_outcome_field(outcome: float) -> None:
    def non_finite_aggregate(
        agents: list[_Agent],
        scenario: ScenarioDefinition,
    ) -> dict[str, Any]:
        del agents, scenario
        return {"hit_rate": outcome}

    with pytest.raises(ValueError, match="outcome field must be finite"):
        _run_single(aggregate_fields=non_finite_aggregate)


def test_invalid_baseline_fails_before_counterfactual_callbacks() -> None:
    scenarios = tuple(reversed(_scenarios()))
    callback_calls: list[str] = []

    def recording_factory(scenario: ScenarioDefinition) -> list[_Agent]:
        callback_calls.append(scenario.name)
        return _make_agents(scenario)

    def invalid_baseline_aggregate(
        agents: list[_Agent],
        scenario: ScenarioDefinition,
    ) -> dict[str, Any]:
        del agents
        return {"hit_rate": float("nan") if scenario.name == "baseline" else 1.0}

    with pytest.raises(ValueError, match="outcome field must be finite"):
        _run_single(
            scenarios=scenarios,
            make_agents=recording_factory,
            aggregate_fields=invalid_baseline_aggregate,
        )
    assert callback_calls == ["baseline"]


def test_single_run_reports_increase_comparison() -> None:
    payload = _run_single()
    assert payload["scenario_count"] == 2
    (comparison,) = payload["comparisons"]
    assert comparison["scenario"] == "boost"
    assert comparison["baseline_value"] == 0.0
    assert comparison["scenario_value"] == 1.0
    assert comparison["delta"] == 1.0
    assert comparison["success_criterion"] == "delta > 0.05"
    assert comparison["success"] is True


def test_single_run_reports_decrease_comparison() -> None:
    payload = _run_single(spec=_spec(success_direction="decrease"))
    (comparison,) = payload["comparisons"]
    assert comparison["success_criterion"] == "delta < -0.05"
    assert comparison["success"] is False


def test_round_digits_does_not_change_delta_or_success_calculation() -> None:
    scenarios = (
        ScenarioDefinition(
            name="baseline",
            description="Lower outcome.",
            parameters={"outcome": 0.049},
        ),
        ScenarioDefinition(
            name="boost",
            description="Slightly higher outcome.",
            parameters={"outcome": 0.051},
        ),
    )
    result = run_bounded_scalar_scenarios(
        scenarios=scenarios,
        spec=_spec(
            steps=1,
            round_digits=2,
            success_min_delta=0.0015,
        ),
        make_agents=_make_agents,
        build_neighbors=_build_neighbors,
        local_update=_local_update,
        domain_transition=_domain_transition,
        aggregate_fields=lambda agents, scenario: {
            "hit_rate": scenario.parameters["outcome"]
        },
    )

    (comparison,) = result.comparisons
    assert comparison.baseline_value == 0.049
    assert comparison.scenario_value == 0.051
    assert comparison.delta == pytest.approx(0.002)
    assert comparison.success is True

    payload = result.to_dict()
    (serialized_comparison,) = payload["comparisons"]
    assert serialized_comparison["delta"] == pytest.approx(0.002)
    assert serialized_comparison["success"] is True
    assert (
        payload["scenarios"]["baseline"]["final"]["aggregate_audit"]["hit_rate"] == 0.05
    )


def test_replication_spec_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="replicates"):
        ReplicationSpec(replicates=0)
    with pytest.raises(ValueError, match="base_seed"):
        ReplicationSpec(replicates=2, base_seed=-1)
    for invalid_replicates in (1.5, True):
        with pytest.raises(ValueError, match="replicates must be an integer"):
            ReplicationSpec(replicates=invalid_replicates)
    for invalid_seed in (1.5, True):
        with pytest.raises(ValueError, match="base_seed must be an integer"):
            ReplicationSpec(replicates=2, base_seed=invalid_seed)


def _replicated_make_agents(context: ScenarioReplicateContext) -> list[_Agent]:
    lift = float(context.scenario.parameters.get("lift", 0.0))
    return [
        _Agent(agent_id=index, x=float(context.rng.uniform(0.2, 0.4)) + lift)
        for index in range(4)
    ]


def _run_replicated(
    *,
    scenarios: tuple[ScenarioDefinition, ...] | None = None,
    replication: ReplicationSpec | None = None,
) -> dict[str, Any]:
    return run_replicated_bounded_scalar_scenarios(
        scenarios=_scenarios() if scenarios is None else scenarios,
        spec=_spec(),
        replication=(
            ReplicationSpec(replicates=8, base_seed=11)
            if replication is None
            else replication
        ),
        make_agents=_replicated_make_agents,
        build_neighbors=lambda agents, context: _build_neighbors(
            agents,
            context.scenario,
        ),
        local_update=lambda agent, context: 0.0,
        domain_transition=lambda agent, value, context: _domain_transition(
            agent,
            value,
            context.scenario,
        ),
        aggregate_fields=lambda agents, context: _aggregate_fields(
            agents,
            context.scenario,
        ),
    ).to_dict()


def test_replicated_run_is_deterministic_for_same_base_seed() -> None:
    assert _run_replicated() == _run_replicated()


def test_replicated_run_changes_with_base_seed() -> None:
    first = _run_replicated(replication=ReplicationSpec(replicates=8, base_seed=11))
    second = _run_replicated(replication=ReplicationSpec(replicates=8, base_seed=12))
    assert (
        first["scenarios"]["baseline"]["mean_state_trajectory"]
        != second["scenarios"]["baseline"]["mean_state_trajectory"]
    )


def test_replicated_invalid_baseline_fails_before_counterfactual_callbacks() -> None:
    callback_calls: list[str] = []

    def recording_factory(context: ScenarioReplicateContext) -> list[_Agent]:
        callback_calls.append(context.scenario.name)
        return [_Agent(agent_id=0, x=0.3)]

    with pytest.raises(ValueError, match="outcome field must be finite"):
        run_replicated_bounded_scalar_scenarios(
            scenarios=tuple(reversed(_scenarios())),
            spec=_spec(steps=1),
            replication=ReplicationSpec(replicates=2, base_seed=11),
            make_agents=recording_factory,
            build_neighbors=lambda agents, context: [[]],
            local_update=lambda agent, context: 0.0,
            domain_transition=lambda agent, value, context: {},
            aggregate_fields=lambda agents, context: {
                "hit_rate": (
                    float("nan") if context.scenario.name == "baseline" else 1.0
                )
            },
        )
    assert callback_calls == ["baseline"]


def test_replicated_run_pairs_scenarios_through_common_random_numbers() -> None:
    scenarios = (
        ScenarioDefinition(name="baseline", description="No boost."),
        ScenarioDefinition(name="clone", description="Same parameters as baseline."),
    )
    payload = _run_replicated(scenarios=scenarios)
    assert (
        payload["scenarios"]["clone"]["outcome_values"]
        == payload["scenarios"]["baseline"]["outcome_values"]
    )
    (comparison,) = payload["comparisons"]
    assert comparison["mean_delta"] == 0.0
    assert comparison["delta_std"] == 0.0
    assert comparison["delta_ci95"] == [0.0, 0.0]
    assert comparison["success"] is False
    assert comparison["success_fraction"] == 0.0


def test_replicated_run_reports_distribution_summaries() -> None:
    payload = _run_replicated()
    baseline = payload["scenarios"]["baseline"]
    assert payload["replicates"] == 8
    assert payload["base_seed"] == 11
    assert len(baseline["outcome_values"]) == 8
    summary = baseline["outcome_summary"]
    assert {
        "mean",
        "std",
        "min",
        "max",
        "ci95_low",
        "ci95_high",
        "mean_ci95",
        "ci95_method",
        "empirical_percentile_interval95",
    } == set(summary)
    assert summary["min"] <= summary["mean"] <= summary["max"]
    assert summary["mean_ci95"] == [summary["ci95_low"], summary["ci95_high"]]
    assert summary["ci95_method"] == "normal_approximation_for_mean"
    assert len(baseline["mean_state_trajectory"]) == 2
    assert "hit_rate" in baseline["aggregate_summaries"]
    (comparison,) = payload["comparisons"]
    assert comparison["success_criterion"] == "mean delta > 0.05"
    assert comparison["replicates"] == 8
    assert comparison["mean_effect_ci95"] == comparison["delta_ci95"]
    assert comparison["delta_ci95_method"] == ("normal_approximation_for_paired_mean")
    assert len(comparison["delta_empirical_percentile_interval95"]) == 2
    assert len(comparison["paired_deltas"]) == comparison["replicates"]


def test_single_replicate_marks_mean_intervals_unavailable() -> None:
    payload = _run_replicated(replication=ReplicationSpec(replicates=1, base_seed=11))
    baseline_summary = payload["scenarios"]["baseline"]["outcome_summary"]
    (comparison,) = payload["comparisons"]

    assert baseline_summary["mean_ci95"] is None
    assert baseline_summary["ci95_low"] is None
    assert baseline_summary["ci95_high"] is None
    assert baseline_summary["ci95_method"] == (
        "unavailable_requires_at_least_2_replicates"
    )
    assert comparison["delta_ci95"] is None
    assert comparison["mean_effect_ci95"] is None
    assert comparison["delta_ci95_method"] == (
        "unavailable_requires_at_least_2_replicates"
    )


def test_replicated_round_digits_rounds_audits_not_analysis() -> None:
    scenarios = (
        ScenarioDefinition(
            name="baseline",
            description="Lower outcome.",
            parameters={"outcome": 0.049},
        ),
        ScenarioDefinition(
            name="boost",
            description="Higher outcome.",
            parameters={"outcome": 0.051},
        ),
    )
    payload = run_replicated_bounded_scalar_scenarios(
        scenarios=scenarios,
        spec=_spec(steps=1, round_digits=2, success_min_delta=0.0015),
        replication=ReplicationSpec(replicates=2, base_seed=3),
        make_agents=lambda context: [_Agent(agent_id=0, x=0.12345)],
        build_neighbors=lambda agents, context: [[]],
        local_update=lambda agent, context: 0.0,
        domain_transition=lambda agent, value, context: {},
        aggregate_fields=lambda agents, context: {
            "hit_rate": context.scenario.parameters["outcome"]
        },
    ).to_dict()

    assert payload["analysis_precision"] == "unrounded"
    assert payload["audit_round_digits"] == 2
    assert payload["scenarios"]["baseline"]["outcome_values"] == [0.049, 0.049]
    assert (
        payload["scenarios"]["baseline"]["first_replicate_final"]["aggregate_audit"][
            "hit_rate"
        ]
        == 0.05
    )
    (comparison,) = payload["comparisons"]
    assert comparison["mean_delta"] == pytest.approx(0.002)
    assert comparison["success"] is True


def test_paired_mean_effect_ci_narrows_as_replication_grows() -> None:
    scenarios = (
        ScenarioDefinition(name="baseline", description="Zero outcome."),
        ScenarioDefinition(name="alternating", description="Alternating outcome."),
    )
    spec = _spec(steps=1, round_digits=None, success_min_delta=0.0)

    def run(replicates: int) -> dict[str, Any]:
        result = run_replicated_bounded_scalar_scenarios(
            scenarios=scenarios,
            spec=spec,
            replication=ReplicationSpec(replicates=replicates, base_seed=7),
            make_agents=lambda context: [_Agent(agent_id=0, x=0.3)],
            build_neighbors=lambda agents, context: [[]],
            local_update=lambda agent, context: 0.0,
            domain_transition=lambda agent, value, context: {},
            aggregate_fields=lambda agents, context: {
                "hit_rate": (
                    0.0
                    if context.scenario.name == "baseline"
                    else float(2 * (context.replicate % 2))
                )
            },
        )
        return result.to_dict()["comparisons"][0]

    small = run(10)
    large = run(1000)
    small_width = small["delta_ci95"][1] - small["delta_ci95"][0]
    large_width = large["delta_ci95"][1] - large["delta_ci95"][0]

    assert small["mean_delta"] == 1.0
    assert large["mean_delta"] == 1.0
    assert small["delta_empirical_percentile_interval95"] == [0.0, 2.0]
    assert large["delta_empirical_percentile_interval95"] == [0.0, 2.0]
    assert large_width < small_width / math.sqrt(50)
