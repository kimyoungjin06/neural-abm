"""Classical opinion-dynamics models as switch settings of the NABM workflow.

Each demonstration below runs the same bounded-scalar lifecycle used by the
researcher-pivot studies (local adaptation -> typed peer exchange ->
domain-owned commit -> audit rows). Two are exact special cases and two are
explicitly labeled near variants:

- ``degroot``: peer-similarity threshold 0, no local update -> DeGroot (1974)
  iterated averaging: variance decays monotonically to consensus at the
  preserved initial mean.
- ``friedkin_johnsen``: same, plus a pre-mix local anchor toward each agent's
  initial opinion -> FJ-like anchored averaging: convergence to persistent
  disagreement, but not the canonical Friedkin-Johnsen (1990) recurrence.
- ``hegselmann_krause``: all-to-all, self-excluding neighbors with an
  output-similarity threshold -> an HK-style bounded-confidence variant:
  cluster count falls as the confidence bound grows, but the confidence set
  differs from canonical self-inclusive Hegselmann-Krause (2002).
- ``granovetter``: absorbing threshold transition over the adoption share
  among the other agents -> a Granovetter (1978) threshold-cascade special
  case, including the knife-edge result where shifting a single agent's
  threshold collapses a full cascade to one adopter.

All demonstrations are deterministic (linspace initial conditions, no random
draws), so the printed results are exactly reproducible and their claim-bearing
outcomes are pinned by ``tests/test_classical_reductions.py``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neural_abm.api_lite import (
    BoundedScalarScenarioSpec,
    ScenarioDefinition,
    run_bounded_scalar_scenarios,
)


@dataclass
class OpinionAgent:
    agent_id: int
    x: float
    anchor: float = field(default=0.0)


@dataclass
class ThresholdAgent:
    agent_id: int
    x: float
    threshold: float
    adopted: bool = False


def _ring_neighbors(count: int, reach: int = 2) -> list[list[int]]:
    return [
        sorted(
            {(index + offset) % count for offset in range(-reach, reach + 1)} - {index}
        )
        for index in range(count)
    ]


def _all_to_all_neighbors(count: int) -> list[list[int]]:
    return [
        [other for other in range(count) if other != index] for index in range(count)
    ]


def _opinion_spec(
    *,
    question: str,
    steps: int,
    social_alpha: float,
    peer_similarity_threshold: float,
) -> BoundedScalarScenarioSpec:
    return BoundedScalarScenarioSpec(
        research_question=question,
        state_field="x",
        channel_name="opinion",
        transition_label="opinion = mixed value",
        commit_mode="opinion_commit",
        outcome_field="state_variance",
        steps=steps,
        social_alpha=social_alpha,
        peer_similarity_threshold=peer_similarity_threshold,
        round_digits=6,
    )


def _opinion_aggregate(
    agents: list[OpinionAgent], _: ScenarioDefinition
) -> dict[str, Any]:
    values = np.asarray([agent.x for agent in agents], dtype=np.float64)
    return {
        "state_variance": float(values.var()),
        "state_mean": float(values.mean()),
        "state_range": float(values.max() - values.min()),
        "cluster_count": _cluster_count(values),
    }


def _cluster_count(values: np.ndarray, gap: float = 0.01) -> int:
    ordered = np.sort(values)
    return int(1 + np.sum(np.diff(ordered) > gap))


def run_degroot(*, agent_count: int = 24, steps: int = 300) -> dict[str, Any]:
    """DeGroot reduction: ring averaging, no confidence bound, no local rule."""

    spec = _opinion_spec(
        question="does_repeated_ring_averaging_reach_consensus",
        steps=steps,
        social_alpha=0.5,
        peer_similarity_threshold=0.0,
    )
    result = run_bounded_scalar_scenarios(
        scenarios=(ScenarioDefinition(name="baseline", description="DeGroot ring."),),
        spec=spec,
        make_agents=lambda scenario: [
            OpinionAgent(agent_id=index, x=float(value))
            for index, value in enumerate(np.linspace(0.0, 1.0, agent_count))
        ],
        build_neighbors=lambda agents, scenario: _ring_neighbors(len(agents)),
        local_update=lambda agent, scenario: 0.0,
        domain_transition=_commit_opinion,
        aggregate_fields=_opinion_aggregate,
    ).to_dict()
    history = result["scenarios"]["baseline"]["history"]
    variances = [step["aggregate_audit"]["state_variance"] for step in history]
    final = history[-1]["aggregate_audit"]
    return {
        "model": "degroot_1974",
        "switches": {
            "peer_similarity_threshold": 0.0,
            "local_update": "off",
            "social_alpha": 0.5,
            "network": "ring",
        },
        "initial_mean": 0.5,
        "final_mean": final["state_mean"],
        "final_range": final["state_range"],
        "final_variance": final["state_variance"],
        "variance_monotone_decreasing": bool(
            all(
                later <= earlier + 1e-12
                for earlier, later in zip(variances, variances[1:])
            )
        ),
        "consensus_reached": bool(final["state_range"] < 1e-3),
    }


def _commit_opinion(
    agent: OpinionAgent,
    value: float,
    _: ScenarioDefinition,
) -> dict[str, Any]:
    agent.x = float(np.clip(value, 0.0, 1.0))
    return {}


def run_friedkin_johnsen(
    *,
    agent_count: int = 24,
    steps: int = 300,
    anchor_weight: float = 0.3,
) -> dict[str, Any]:
    """Run the FJ-like pre-mix anchored-averaging variant."""

    def make_agents(_: ScenarioDefinition) -> list[OpinionAgent]:
        values = np.linspace(0.0, 1.0, agent_count)
        return [
            OpinionAgent(agent_id=index, x=float(value), anchor=float(value))
            for index, value in enumerate(values)
        ]

    def local_update(agent: OpinionAgent, _: ScenarioDefinition) -> float:
        before = agent.x
        agent.x = (1.0 - anchor_weight) * agent.x + anchor_weight * agent.anchor
        return abs(agent.x - before)

    spec = _opinion_spec(
        question="does_anchoring_preserve_disagreement_under_averaging",
        steps=steps,
        social_alpha=0.5,
        peer_similarity_threshold=0.0,
    )
    result = run_bounded_scalar_scenarios(
        scenarios=(
            ScenarioDefinition(
                name="baseline", description="FJ-like pre-mix anchored ring."
            ),
        ),
        spec=spec,
        make_agents=make_agents,
        build_neighbors=lambda agents, scenario: _ring_neighbors(len(agents)),
        local_update=local_update,
        domain_transition=_commit_opinion,
        aggregate_fields=_opinion_aggregate,
    ).to_dict()
    history = result["scenarios"]["baseline"]["history"]
    final = history[-1]["aggregate_audit"]
    previous = history[-2]["aggregate_audit"]
    # At the FJ steady state, anchor and social pulls balance, so per-step
    # shifts stay positive; convergence shows up as a stationary state
    # distribution, not as vanishing shifts.
    variance_delta = abs(final["state_variance"] - previous["state_variance"])
    return {
        "model": "friedkin_johnsen_like_pre_mix_anchor",
        "switches": {
            "peer_similarity_threshold": 0.0,
            "local_update": f"anchor_to_initial({anchor_weight})",
            "social_alpha": 0.5,
            "network": "ring",
        },
        "final_variance": final["state_variance"],
        "final_range": final["state_range"],
        "final_variance_delta": variance_delta,
        "converged": bool(variance_delta < 1e-8),
        "disagreement_persists": bool(final["state_range"] > 0.05),
    }


def run_hegselmann_krause(
    *,
    agent_count: int = 60,
    steps: int = 40,
    epsilons: tuple[float, ...] = (0.05, 0.15, 0.35),
) -> list[dict[str, Any]]:
    """Run the self-excluding HK bounded-confidence variant."""

    rows: list[dict[str, Any]] = []
    for epsilon in epsilons:
        spec = _opinion_spec(
            question="how_does_the_confidence_bound_set_the_cluster_count",
            steps=steps,
            social_alpha=1.0,
            peer_similarity_threshold=1.0 - epsilon,
        )
        result = run_bounded_scalar_scenarios(
            scenarios=(
                ScenarioDefinition(
                    name="baseline",
                    description=f"Self-excluding HK variant, epsilon={epsilon}.",
                ),
            ),
            spec=spec,
            make_agents=lambda scenario: [
                OpinionAgent(agent_id=index, x=float(value))
                for index, value in enumerate(np.linspace(0.0, 1.0, agent_count))
            ],
            build_neighbors=lambda agents, scenario: _all_to_all_neighbors(len(agents)),
            local_update=lambda agent, scenario: 0.0,
            domain_transition=_commit_opinion,
            aggregate_fields=_opinion_aggregate,
        ).to_dict()
        final = result["scenarios"]["baseline"]["history"][-1]["aggregate_audit"]
        rows.append(
            {
                "model": "hegselmann_krause_self_excluding_variant",
                "epsilon": epsilon,
                "switches": {
                    "peer_similarity_threshold": round(1.0 - epsilon, 6),
                    "local_update": "off",
                    "social_alpha": 1.0,
                    "network": "all_to_all",
                },
                "cluster_count": final["cluster_count"],
                "final_variance": final["state_variance"],
            }
        )
    return rows


def run_granovetter(*, agent_count: int = 100) -> dict[str, Any]:
    """Granovetter reduction: absorbing threshold cascade and its knife edge."""

    def make_agents(scenario: ScenarioDefinition) -> list[ThresholdAgent]:
        shifted = int(scenario.parameters.get("shifted_agent", -1))
        agents = []
        for index in range(agent_count):
            threshold = index / agent_count
            if index == shifted:
                threshold = (index + 1) / agent_count
            agents.append(ThresholdAgent(agent_id=index, x=0.0, threshold=threshold))
        return agents

    def domain_transition(
        agent: ThresholdAgent,
        adoption_share: float,
        _: ScenarioDefinition,
    ) -> dict[str, Any]:
        if not agent.adopted and adoption_share >= agent.threshold:
            agent.adopted = True
        agent.x = 1.0 if agent.adopted else 0.0
        return {"adopted": agent.adopted}

    def aggregate_fields(
        agents: list[ThresholdAgent],
        _: ScenarioDefinition,
    ) -> dict[str, Any]:
        return {"adoption_count": sum(agent.adopted for agent in agents)}

    spec = BoundedScalarScenarioSpec(
        research_question="does_one_hesitant_agent_break_the_full_cascade",
        state_field="x",
        channel_name="adoption",
        transition_label="adopt when adoption share >= own threshold",
        commit_mode="absorbing_threshold_adoption",
        outcome_field="adoption_count",
        success_direction="decrease",
        success_min_delta=50.0,
        steps=agent_count + 5,
        social_alpha=1.0,
        peer_similarity_threshold=0.0,
        round_digits=6,
    )
    result = run_bounded_scalar_scenarios(
        scenarios=(
            ScenarioDefinition(
                name="baseline",
                description="Uniform threshold ladder 0/N .. (N-1)/N.",
            ),
            ScenarioDefinition(
                name="one_agent_more_hesitant",
                description="Identical ladder, one threshold moved 1/N -> 2/N.",
                parameters={"shifted_agent": 1},
            ),
        ),
        spec=spec,
        make_agents=make_agents,
        build_neighbors=lambda agents, scenario: _all_to_all_neighbors(len(agents)),
        local_update=lambda agent, scenario: 0.0,
        domain_transition=domain_transition,
        aggregate_fields=aggregate_fields,
    ).to_dict()
    (comparison,) = result["comparisons"]
    return {
        "model": "granovetter_1978",
        "switches": {
            "transition": "absorbing_threshold",
            "local_update": "off",
            "social_alpha": 1.0,
            "network": "all_to_all_self_excluding",
            "peer_similarity_threshold": 0.0,
        },
        "baseline_adopters": result["scenarios"]["baseline"]["history"][-1][
            "aggregate_audit"
        ]["adoption_count"],
        "perturbed_adopters": result["scenarios"]["one_agent_more_hesitant"]["history"][
            -1
        ]["aggregate_audit"]["adoption_count"],
        "comparison": comparison,
        "knife_edge_reproduced": bool(comparison["success"]),
    }


def run_all_reductions() -> dict[str, Any]:
    return {
        "status": "ok",
        "surface": "neural_abm.scenario_lite",
        "base_surface": "neural_abm.api_lite",
        "default_profile": "torch-free",
        "torch_loaded": "torch" in sys.modules,
        "degroot": run_degroot(),
        "friedkin_johnsen": run_friedkin_johnsen(),
        "hegselmann_krause": run_hegselmann_krause(),
        "granovetter": run_granovetter(),
    }


def main() -> None:
    print(json.dumps(run_all_reductions(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
