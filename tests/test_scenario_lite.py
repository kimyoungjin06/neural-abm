"""Unit tests for torch-free scenario comparison and replication helpers."""

from __future__ import annotations

from dataclasses import dataclass
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
    with pytest.raises(ValueError, match="steps"):
        _spec(steps=0)
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
    with pytest.raises(ValueError, match="baseline scenario not found"):
        _run_single(spec=_spec(baseline_name="missing"))


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


def test_replication_spec_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="replicates"):
        ReplicationSpec(replicates=0)
    with pytest.raises(ValueError, match="base_seed"):
        ReplicationSpec(replicates=2, base_seed=-1)


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
    assert {"mean", "std", "min", "max", "ci95_low", "ci95_high"} == set(summary)
    assert summary["min"] <= summary["mean"] <= summary["max"]
    assert len(baseline["mean_state_trajectory"]) == 2
    assert "hit_rate" in baseline["aggregate_summaries"]
    (comparison,) = payload["comparisons"]
    assert comparison["success_criterion"] == "mean delta > 0.05"
    assert comparison["replicates"] == 8
