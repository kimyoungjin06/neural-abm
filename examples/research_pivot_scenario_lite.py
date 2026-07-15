"""Torch-free scenario comparison for a science-of-science PIVOT question.

The example asks whether researchers should pivot into a new field under
different scientific environments. It is not a prediction model. It shows how a
general researcher can express a baseline, counterfactual scenarios, local
decision pressure, peer influence, and outcome comparison through api_lite.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from neural_abm.api_lite import (
    BoundedScalarScenarioSpec,
    ScenarioDefinition,
    run_bounded_scalar_scenarios,
)


@dataclass
class ResearcherPivotAgent:
    agent_id: int
    career_stage: str
    skill_distance: float
    resource_security: float
    network_support: float
    field_opportunity: float
    reputation_risk: float
    openness: float
    pivot_readiness: float
    funding_signal: float = 0.0
    attention_signal: float = 0.0
    peer_success_signal: float = 0.0
    pivoted: bool = False
    productive_pivot: bool = False

    @property
    def pivot_pressure(self) -> float:
        pressure = (
            0.25 * self.field_opportunity
            + 0.25 * self.resource_security
            + 0.25 * self.network_support
            + 0.20 * self.funding_signal
            + 0.20 * self.attention_signal
            + 0.15 * self.peer_success_signal
            + 0.10 * self.openness
            - 0.25 * self.skill_distance
            - 0.20 * self.reputation_risk
            + 0.10
        )
        return float(np.clip(pressure, 0.0, 1.0))

    @property
    def productive_fit(self) -> float:
        fit = (
            0.35 * self.field_opportunity
            + 0.30 * self.network_support
            + 0.25 * self.resource_security
            + 0.10 * self.openness
            - 0.25 * self.skill_distance
            - 0.15 * self.reputation_risk
        )
        return float(np.clip(fit, 0.0, 1.0))


@dataclass(frozen=True)
class PivotConfig:
    local_alpha: float = 0.45
    social_alpha: float = 0.40
    peer_similarity_threshold: float = 0.68
    pivot_threshold: float = 0.53
    productive_threshold: float = 0.43
    steps: int = 3


BASE_RESEARCHER_ROWS: tuple[
    tuple[int, str, float, float, float, float, float, float, float],
    ...,
] = (
    (0, "early", 0.40, 0.45, 0.35, 0.55, 0.45, 0.75, 0.30),
    (1, "early", 0.55, 0.30, 0.25, 0.60, 0.50, 0.80, 0.22),
    (2, "mid", 0.30, 0.55, 0.50, 0.50, 0.35, 0.65, 0.42),
    (3, "mid", 0.65, 0.50, 0.35, 0.70, 0.55, 0.70, 0.28),
    (4, "senior", 0.25, 0.80, 0.70, 0.45, 0.20, 0.45, 0.48),
    (5, "senior", 0.50, 0.75, 0.60, 0.52, 0.30, 0.50, 0.40),
    (6, "peripheral", 0.45, 0.25, 0.20, 0.58, 0.55, 0.82, 0.20),
    (7, "interdisciplinary", 0.20, 0.50, 0.65, 0.62, 0.25, 0.85, 0.50),
)

PIVOT_NEIGHBORS: tuple[tuple[int, ...], ...] = (
    (1, 2, 7),
    (0, 6),
    (0, 3, 5),
    (2, 4),
    (3, 5, 7),
    (2, 4, 7),
    (0, 1),
    (0, 4, 5),
)


def scenarios() -> tuple[ScenarioDefinition, ...]:
    return (
        ScenarioDefinition(
            name="baseline",
            description="No explicit interdisciplinary pivot support.",
        ),
        ScenarioDefinition(
            name="interdisciplinary_seed_grants",
            description=(
                "Small grants, bridge ties, and reduced reputation risk support "
                "researchers considering a pivot."
            ),
            parameters={
                "field_signal": 0.04,
                "resource_signal": 0.12,
                "bridge_signal": 0.15,
                "funding_signal": 0.22,
                "peer_success_signal": 0.08,
                "reputation_penalty": -0.05,
            },
        ),
        ScenarioDefinition(
            name="hot_field_hype",
            description=(
                "A visible hot-field signal raises attention without providing "
                "bridge ties or resource security."
            ),
            parameters={
                "field_signal": 0.20,
                "resource_signal": -0.20,
                "bridge_signal": -0.20,
                "attention_signal": 0.70,
                "peer_success_signal": 0.35,
                "reputation_penalty": 0.15,
            },
        ),
    )


def _scenario_value(scenario: ScenarioDefinition, key: str) -> float:
    return float(scenario.parameters.get(key, 0.0))


def make_agents(scenario: ScenarioDefinition) -> list[ResearcherPivotAgent]:
    agents: list[ResearcherPivotAgent] = []
    for row in BASE_RESEARCHER_ROWS:
        (
            agent_id,
            career_stage,
            skill_distance,
            resource_security,
            network_support,
            field_opportunity,
            reputation_risk,
            openness,
            pivot_readiness,
        ) = row
        agents.append(
            ResearcherPivotAgent(
                agent_id=agent_id,
                career_stage=career_stage,
                skill_distance=skill_distance,
                resource_security=float(
                    np.clip(
                        resource_security
                        + _scenario_value(scenario, "resource_signal"),
                        0.0,
                        1.0,
                    )
                ),
                network_support=float(
                    np.clip(
                        network_support + _scenario_value(scenario, "bridge_signal"),
                        0.0,
                        1.0,
                    )
                ),
                field_opportunity=float(
                    np.clip(
                        field_opportunity + _scenario_value(scenario, "field_signal"),
                        0.0,
                        1.0,
                    )
                ),
                reputation_risk=float(
                    np.clip(
                        reputation_risk
                        + _scenario_value(scenario, "reputation_penalty"),
                        0.0,
                        1.0,
                    )
                ),
                openness=openness,
                pivot_readiness=pivot_readiness,
                funding_signal=_scenario_value(scenario, "funding_signal"),
                attention_signal=_scenario_value(scenario, "attention_signal"),
                peer_success_signal=_scenario_value(scenario, "peer_success_signal"),
            )
        )
    return agents


def build_neighbors(
    agents: list[ResearcherPivotAgent],
    scenario: ScenarioDefinition,
) -> list[list[int]]:
    del agents, scenario
    return [list(neighbors) for neighbors in PIVOT_NEIGHBORS]


def run_pivot_scenario_audit() -> dict[str, Any]:
    config = PivotConfig()
    spec = BoundedScalarScenarioSpec(
        research_question=(
            "should_researchers_pivot_under_different_scientific_environments"
        ),
        state_field="pivot_readiness",
        channel_name="pivot_readiness",
        transition_label="pivoted = pivot_readiness >= pivot_threshold",
        commit_mode="domain_pivot_threshold",
        outcome_field="productive_pivot_count",
        success_direction="increase",
        success_min_delta=2.0,
        steps=config.steps,
        peer_similarity_threshold=config.peer_similarity_threshold,
        social_alpha=config.social_alpha,
    )

    def local_update(
        agent: ResearcherPivotAgent, scenario: ScenarioDefinition
    ) -> float:
        del scenario
        before = float(agent.pivot_readiness)
        agent.pivot_readiness = float(
            np.clip(
                (1.0 - config.local_alpha) * before
                + config.local_alpha * agent.pivot_pressure,
                0.0,
                1.0,
            )
        )
        return abs(agent.pivot_readiness - before)

    def domain_transition(
        agent: ResearcherPivotAgent,
        mixed_readiness: float,
        scenario: ScenarioDefinition,
    ) -> dict[str, Any]:
        del scenario
        agent.pivot_readiness = float(np.clip(mixed_readiness, 0.0, 1.0))
        agent.pivoted = agent.pivot_readiness >= config.pivot_threshold
        agent.productive_pivot = (
            agent.pivoted and agent.productive_fit >= config.productive_threshold
        )
        return {
            "pivoted": agent.pivoted,
            "productive_pivot": agent.productive_pivot,
        }

    def micro_fields(
        agent: ResearcherPivotAgent,
        scenario: ScenarioDefinition,
    ) -> dict[str, Any]:
        return {
            "career_stage": agent.career_stage,
            "scenario": scenario.name,
            "skill_distance": agent.skill_distance,
            "resource_security": agent.resource_security,
            "network_support": agent.network_support,
            "field_opportunity": agent.field_opportunity,
            "reputation_risk": agent.reputation_risk,
            "openness": agent.openness,
            "funding_signal": agent.funding_signal,
            "attention_signal": agent.attention_signal,
            "peer_success_signal": agent.peer_success_signal,
            "pivot_pressure": agent.pivot_pressure,
            "productive_fit": agent.productive_fit,
        }

    def aggregate_fields(
        agents: list[ResearcherPivotAgent],
        scenario: ScenarioDefinition,
    ) -> dict[str, Any]:
        del scenario
        pivot_ids = [agent.agent_id for agent in agents if agent.pivoted]
        productive_ids = [agent.agent_id for agent in agents if agent.productive_pivot]
        return {
            "pivot_count": len(pivot_ids),
            "productive_pivot_count": len(productive_ids),
            "failed_pivot_count": len(pivot_ids) - len(productive_ids),
            "pivot_ids": pivot_ids,
            "productive_pivot_ids": productive_ids,
            "pivot_threshold": config.pivot_threshold,
            "productive_threshold": config.productive_threshold,
        }

    result = run_bounded_scalar_scenarios(
        scenarios=scenarios(),
        spec=spec,
        make_agents=make_agents,
        build_neighbors=build_neighbors,
        local_update=local_update,
        domain_transition=domain_transition,
        micro_fields=micro_fields,
        aggregate_fields=aggregate_fields,
    ).to_dict()
    result["torch_loaded"] = "torch" in sys.modules
    return result


def run_pivot_scenario_demo() -> dict[str, Any]:
    audit = run_pivot_scenario_audit()
    scenario_summaries = {
        name: {
            "description": scenario["description"],
            "parameters": scenario["parameters"],
            "final_aggregate": scenario["final"]["aggregate_audit"],
            "micro_audit_sample": scenario["final"]["micro_audit"][:2],
            "history_steps": len(scenario["history"]),
        }
        for name, scenario in audit["scenarios"].items()
    }
    return {
        "status": audit["status"],
        "surface": audit["surface"],
        "base_surface": audit["base_surface"],
        "default_profile": audit["default_profile"],
        "research_question": audit["research_question"],
        "state_field": audit["state_field"],
        "outcome_field": audit["outcome_field"],
        "baseline": audit["baseline"],
        "steps": audit["steps"],
        "scenario_count": audit["scenario_count"],
        "scenario_summaries": scenario_summaries,
        "comparisons": audit["comparisons"],
        "torch_loaded": audit["torch_loaded"],
    }


def main() -> None:
    print(json.dumps(run_pivot_scenario_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
