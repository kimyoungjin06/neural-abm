"""Epidemic compliance behavior demo using the reusable NABM social unit."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from neural_abm import (
    SCALAR_PROBABILITY_CHANNEL,
    CommitReport,
    NABMAgent,
    NABMStep,
    ObservationSpec,
    SocialBlock,
    SocialChannel,
    SocialMessageSpec,
    SocialMixResult,
)


@dataclass(frozen=True)
class EpidemicConfig:
    initial_infection_rate: float = 0.12
    base_transmission: float = 0.34
    recovery_probability: float = 0.09
    local_learning_rate: float = 0.30
    social_alpha: float = 0.35
    peer_similarity_threshold: float = 0.0


@dataclass
class ComplianceAgent:
    agent_id: int
    infected: bool
    compliance_probability: float
    perceived_cost: float
    recent_exposure: float
    local_learning_rate: float

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="epidemic_compliance_context",
            tensor_shape=(4,),
            dtype=torch.float32,
            description="local risk, neighbor infection, perceived cost, exposure",
        )

    def social_message_spec(self) -> SocialMessageSpec:
        return SocialMessageSpec(
            required_keys=(
                "agent_id",
                "compliance_probability",
                "latent_summary",
                "confidence",
                "param_norm",
            ),
            tensor_keys=("latent_summary",),
            probability_keys=("compliance_probability",),
        )

    def observe(self, x: Mapping[str, float] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().clone().to(dtype=torch.float32)
        return torch.tensor(
            [
                float(x["local_infection_risk"]),
                float(x["neighbor_infection_rate"]),
                float(x["perceived_cost"]),
                float(x["recent_exposure"]),
            ],
            dtype=torch.float32,
        )

    def act_or_predict(self, observation: torch.Tensor) -> torch.Tensor:
        observed = self.observe(observation)
        risk = observed[..., 0]
        neighbor_rate = observed[..., 1]
        cost = observed[..., 2]
        exposure = observed[..., 3]
        target = torch.clamp(
            0.12 + 0.55 * risk + 0.30 * neighbor_rate + 0.25 * exposure - 0.36 * cost,
            0.0,
            1.0,
        )
        current = torch.full_like(target, float(self.compliance_probability))
        return torch.clamp(0.55 * current + 0.45 * target, 0.0, 1.0).reshape(-1)

    def local_update(self, observation: torch.Tensor | None = None) -> float:
        if observation is None:
            observation = torch.tensor(
                [0.0, 0.0, self.perceived_cost, self.recent_exposure],
                dtype=torch.float32,
            )
        target = float(self.act_or_predict(observation)[0])
        before = float(self.compliance_probability)
        self.compliance_probability = float(
            np.clip(
                (1.0 - self.local_learning_rate) * before
                + self.local_learning_rate * target,
                0.0,
                1.0,
            )
        )
        return abs(self.compliance_probability - before)

    def social_message(self, observation: torch.Tensor | None = None) -> dict[str, Any]:
        if observation is None:
            latent_summary = torch.tensor(
                [
                    float(self.infected),
                    float(self.compliance_probability),
                    float(self.perceived_cost),
                    float(self.recent_exposure),
                ],
                dtype=torch.float32,
            )
        else:
            latent_summary = self.observe(observation)
        confidence = abs(float(self.compliance_probability) - 0.5) * 2.0
        return {
            "agent_id": self.agent_id,
            "compliance_probability": float(
                np.clip(self.compliance_probability, 0.0, 1.0)
            ),
            "latent_summary": latent_summary.detach().clone(),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "param_norm": float(
                abs(self.compliance_probability)
                + abs(self.perceived_cost)
                + abs(self.recent_exposure)
            ),
        }

    def log_state(self, observation: torch.Tensor | None = None) -> dict[str, Any]:
        message = self.social_message(observation)
        return {
            "agent_id": self.agent_id,
            "infected": self.infected,
            "compliance_probability": message["compliance_probability"],
            "perceived_cost": self.perceived_cost,
            "recent_exposure": self.recent_exposure,
            "confidence": message["confidence"],
            "param_norm": message["param_norm"],
            "latent_norm": float(torch.linalg.vector_norm(message["latent_summary"])),
        }


@dataclass
class ScalarAttributeCommitAdapter:
    agents: Sequence[ComplianceAgent]
    attribute: str
    skip_empty_peers: bool = True

    def commit(self, mix_result: SocialMixResult) -> CommitReport:
        committed: list[int] = []
        for agent_id, agent in enumerate(self.agents):
            if self.skip_empty_peers and not mix_result.peer_ids[agent_id]:
                continue
            setattr(
                agent,
                self.attribute,
                float(np.clip(mix_result.mixed_values[agent_id], 0.0, 1.0)),
            )
            committed.append(int(getattr(agent, "agent_id", agent_id)))
        return CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
        )


def make_agent(agent_id: int = 0) -> ComplianceAgent:
    return ComplianceAgent(
        agent_id=agent_id,
        infected=agent_id == 0,
        compliance_probability=0.20 + 0.08 * (agent_id % 5),
        perceived_cost=0.20 + 0.05 * (agent_id % 4),
        recent_exposure=0.0,
        local_learning_rate=0.30,
    )


def _ring_neighbors(agent_count: int, radius: int = 2) -> list[list[int]]:
    neighbors: list[list[int]] = []
    for agent_id in range(agent_count):
        peers = {
            (agent_id + offset) % agent_count
            for offset in range(-radius, radius + 1)
            if offset != 0 and agent_count > 1
        }
        peers.discard(agent_id)
        neighbors.append(sorted(peers))
    return neighbors


def _initialize_agents(
    agent_count: int,
    config: EpidemicConfig,
    rng: np.random.Generator,
) -> list[ComplianceAgent]:
    infected_count = max(1, int(round(agent_count * config.initial_infection_rate)))
    infected_ids = set(rng.choice(agent_count, size=infected_count, replace=False))
    agents: list[ComplianceAgent] = []
    for agent_id in range(agent_count):
        agents.append(
            ComplianceAgent(
                agent_id=agent_id,
                infected=agent_id in infected_ids,
                compliance_probability=float(rng.uniform(0.10, 0.80)),
                perceived_cost=float(rng.uniform(0.10, 0.55)),
                recent_exposure=0.0,
                local_learning_rate=config.local_learning_rate,
            )
        )
    return agents


def _observation_for_agent(
    agent: ComplianceAgent,
    agents: Sequence[ComplianceAgent],
    neighbors: Sequence[Sequence[int]],
    config: EpidemicConfig,
) -> dict[str, float]:
    peer_ids = neighbors[agent.agent_id]
    infected_neighbors = sum(1 for peer_id in peer_ids if agents[peer_id].infected)
    neighbor_rate = infected_neighbors / len(peer_ids) if peer_ids else 0.0
    effective_contact = 1.0 - agent.compliance_probability
    risk = 1.0 - (1.0 - config.base_transmission * effective_contact) ** max(
        infected_neighbors,
        0,
    )
    return {
        "local_infection_risk": float(np.clip(risk, 0.0, 1.0)),
        "neighbor_infection_rate": float(neighbor_rate),
        "perceived_cost": float(agent.perceived_cost),
        "recent_exposure": float(agent.recent_exposure),
    }


def _advance_infections(
    agents: Sequence[ComplianceAgent],
    neighbors: Sequence[Sequence[int]],
    config: EpidemicConfig,
    rng: np.random.Generator,
) -> None:
    next_infected: list[bool] = []
    exposures: list[float] = []
    for agent in agents:
        peer_ids = neighbors[agent.agent_id]
        infected_neighbors = sum(1 for peer_id in peer_ids if agents[peer_id].infected)
        neighbor_rate = infected_neighbors / len(peer_ids) if peer_ids else 0.0
        exposures.append(neighbor_rate)
        if agent.infected:
            next_infected.append(rng.random() >= config.recovery_probability)
            continue
        peer_compliance = (
            float(np.mean([agents[peer_id].compliance_probability for peer_id in peer_ids]))
            if peer_ids
            else 0.0
        )
        effective_contact = (1.0 - agent.compliance_probability) * (
            1.0 - 0.45 * peer_compliance
        )
        infection_probability = 1.0 - (
            1.0 - config.base_transmission * effective_contact
        ) ** infected_neighbors
        next_infected.append(rng.random() < infection_probability)

    for agent, infected, exposure in zip(agents, next_infected, exposures, strict=True):
        agent.infected = bool(infected)
        agent.recent_exposure = float(
            np.clip(0.70 * agent.recent_exposure + 0.30 * exposure, 0.0, 1.0)
        )


def _metrics(agents: Sequence[ComplianceAgent]) -> dict[str, float]:
    infection_rate = float(np.mean([agent.infected for agent in agents]))
    compliance_rate = float(np.mean([agent.compliance_probability for agent in agents]))
    return {
        "infection_rate": infection_rate,
        "compliance_rate": compliance_rate,
        "contact_reduction": compliance_rate,
        "mean_recent_exposure": float(
            np.mean([agent.recent_exposure for agent in agents])
        ),
    }


def run_demo(seed: int = 11, steps: int = 16, agent_count: int = 40) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    config = EpidemicConfig()
    agents = _initialize_agents(agent_count=agent_count, config=config, rng=rng)
    assert all(isinstance(agent, NABMAgent) for agent in agents)
    neighbors = _ring_neighbors(agent_count)

    channel = SocialChannel(
        name="compliance_probability",
        kind=SCALAR_PROBABILITY_CHANNEL,
        commit_mode="scalar_attribute_commit",
    )
    social_step = NABMStep(
        social_block=SocialBlock(alpha=config.social_alpha),
        channel=channel,
        commit_adapter=ScalarAttributeCommitAdapter(
            agents=agents,
            attribute="compliance_probability",
        ),
    )

    history: list[dict[str, Any]] = []
    social_update_norms: list[float] = []
    for step_id in range(steps):
        observations = [
            agent.observe(_observation_for_agent(agent, agents, neighbors, config))
            for agent in agents
        ]
        for agent, observation in zip(agents, observations, strict=True):
            agent.observation_spec().validate(observation)
            agent.social_message_spec().validate(agent.social_message(observation))
            agent.local_update(observation)

        _advance_infections(agents, neighbors, config, rng)
        values = np.asarray([agent.compliance_probability for agent in agents], dtype=float)
        peer_selection = SocialBlock(alpha=0.0).select_scalar_output_peers(
            neighbors=[list(peer_ids) for peer_ids in neighbors],
            values=values,
            peer_rule="output_similarity",
            threshold=config.peer_similarity_threshold,
        )
        result = social_step.run(values=values, peer_ids=peer_selection.peer_ids)
        diagnostics = result.diagnostics.aggregate_row()
        metrics = _metrics(agents)
        social_update_norms.append(result.diagnostics.mean_update_norm)
        history.append(
            {
                "step": step_id,
                "metrics": metrics,
                "mean_peer_count": result.diagnostics.mean_peer_count,
                **diagnostics,
            }
        )

    final_metrics = history[-1]["metrics"] if history else _metrics(agents)
    return {
        "example": "epidemic_compliance",
        "seed": seed,
        "steps": steps,
        "agent_count": agent_count,
        "social_channel": channel.name,
        "commit_mode": channel.commit_mode,
        "mean_social_update_norm": float(np.mean(social_update_norms))
        if social_update_norms
        else 0.0,
        "max_social_update_norm": float(max(social_update_norms, default=0.0)),
        "social_update_norms": social_update_norms,
        "metrics": final_metrics,
        "history": history,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
