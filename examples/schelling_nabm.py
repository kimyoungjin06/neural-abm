"""Schelling-style segregation demo using the reusable NABM social unit."""

from __future__ import annotations

import json
import math
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

Coord = tuple[int, int]


@dataclass(frozen=True)
class SchellingConfig:
    occupancy_rate: float = 0.82
    satisfaction_threshold: float = 0.55
    local_learning_rate: float = 0.35
    social_alpha: float = 0.40
    peer_similarity_threshold: float = 0.0


@dataclass
class SchellingAgent:
    agent_id: int
    group: int
    location: Coord
    move_probability: float
    satisfaction_threshold: float
    local_learning_rate: float

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="schelling_local_context",
            tensor_shape=(3,),
            dtype=torch.float32,
            description="same-type neighbor ratio, vacancy pressure, local density",
        )

    def social_message_spec(self) -> SocialMessageSpec:
        return SocialMessageSpec(
            required_keys=(
                "agent_id",
                "move_probability",
                "latent_summary",
                "confidence",
                "param_norm",
            ),
            tensor_keys=("latent_summary",),
            probability_keys=("move_probability",),
        )

    def observe(self, x: Mapping[str, float] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().clone().to(dtype=torch.float32)
        return torch.tensor(
            [
                float(x["same_type_neighbor_ratio"]),
                float(x["vacancy_pressure"]),
                float(x["local_density"]),
            ],
            dtype=torch.float32,
        )

    def act_or_predict(self, observation: torch.Tensor) -> torch.Tensor:
        observed = self.observe(observation)
        same_ratio = observed[..., 0]
        vacancy_pressure = observed[..., 1]
        density = observed[..., 2]
        dissatisfaction = torch.clamp(self.satisfaction_threshold - same_ratio, 0.0, 1.0)
        domain_target = torch.clamp(
            0.08 + 0.82 * dissatisfaction + 0.20 * vacancy_pressure - 0.10 * density,
            0.0,
            1.0,
        )
        current = torch.full_like(domain_target, float(self.move_probability))
        probability = torch.clamp(0.60 * current + 0.40 * domain_target, 0.0, 1.0)
        return probability.reshape(-1)

    def local_update(self, observation: torch.Tensor | None = None) -> float:
        if observation is None:
            observation = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        observed = self.observe(observation)
        same_ratio = float(observed[0])
        vacancy_pressure = float(observed[1])
        density = float(observed[2])
        dissatisfaction = max(0.0, self.satisfaction_threshold - same_ratio)
        target = float(
            np.clip(
                0.05 + 0.90 * dissatisfaction + 0.20 * vacancy_pressure - 0.08 * density,
                0.0,
                1.0,
            )
        )
        before = float(self.move_probability)
        self.move_probability = float(
            np.clip(
                (1.0 - self.local_learning_rate) * before
                + self.local_learning_rate * target,
                0.0,
                1.0,
            )
        )
        return abs(self.move_probability - before)

    def social_message(self, observation: torch.Tensor | None = None) -> dict[str, Any]:
        if observation is None:
            latent_summary = torch.tensor(
                [
                    float(self.group),
                    float(self.move_probability),
                    float(self.satisfaction_threshold),
                ],
                dtype=torch.float32,
            )
            confidence = abs(float(self.move_probability) - 0.5) * 2.0
        else:
            observed = self.observe(observation)
            latent_summary = observed.detach().clone()
            confidence = abs(float(self.act_or_predict(observed)[0]) - 0.5) * 2.0
        return {
            "agent_id": self.agent_id,
            "move_probability": float(np.clip(self.move_probability, 0.0, 1.0)),
            "latent_summary": latent_summary,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "param_norm": float(
                abs(self.move_probability) + abs(self.satisfaction_threshold)
            ),
        }

    def log_state(self, observation: torch.Tensor | None = None) -> dict[str, Any]:
        message = self.social_message(observation)
        return {
            "agent_id": self.agent_id,
            "group": self.group,
            "row": self.location[0],
            "col": self.location[1],
            "move_probability": message["move_probability"],
            "confidence": message["confidence"],
            "param_norm": message["param_norm"],
            "latent_norm": float(torch.linalg.vector_norm(message["latent_summary"])),
        }


@dataclass
class ScalarAttributeCommitAdapter:
    agents: Sequence[SchellingAgent]
    attribute: str
    skip_empty_peers: bool = True

    def commit(self, mix_result: SocialMixResult) -> CommitReport:
        committed: list[int] = []
        for agent_id, agent in enumerate(self.agents):
            if self.skip_empty_peers and not mix_result.peer_ids[agent_id]:
                continue
            value = float(np.clip(mix_result.mixed_values[agent_id], 0.0, 1.0))
            setattr(agent, self.attribute, value)
            committed.append(int(getattr(agent, "agent_id", agent_id)))
        return CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
        )


def make_agent(agent_id: int = 0) -> SchellingAgent:
    return SchellingAgent(
        agent_id=agent_id,
        group=agent_id % 2,
        location=(0, agent_id),
        move_probability=0.25 + 0.10 * (agent_id % 3),
        satisfaction_threshold=0.55,
        local_learning_rate=0.35,
    )


def _grid_size(agent_count: int, occupancy_rate: float) -> int:
    size = max(3, math.ceil(math.sqrt(agent_count / occupancy_rate)))
    while size * size <= agent_count:
        size += 1
    return size


def _initialize_agents(
    *,
    agent_count: int,
    grid_size: int,
    config: SchellingConfig,
    rng: np.random.Generator,
) -> list[SchellingAgent]:
    cells = [(row, col) for row in range(grid_size) for col in range(grid_size)]
    chosen = rng.choice(len(cells), size=agent_count, replace=False)
    agents: list[SchellingAgent] = []
    for agent_id, cell_id in enumerate(chosen):
        group = int(rng.integers(0, 2))
        agents.append(
            SchellingAgent(
                agent_id=agent_id,
                group=group,
                location=cells[int(cell_id)],
                move_probability=float(rng.uniform(0.05, 0.75)),
                satisfaction_threshold=config.satisfaction_threshold,
                local_learning_rate=config.local_learning_rate,
            )
        )
    return agents


def _build_grid(agents: Sequence[SchellingAgent]) -> dict[Coord, int]:
    return {agent.location: agent.agent_id for agent in agents}


def _neighbor_cells(location: Coord, grid_size: int) -> list[Coord]:
    row, col = location
    cells: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            candidate = (row + dr, col + dc)
            if 0 <= candidate[0] < grid_size and 0 <= candidate[1] < grid_size:
                cells.append(candidate)
    return cells


def _features_for_agent(
    agent: SchellingAgent,
    agents: Sequence[SchellingAgent],
    grid: dict[Coord, int],
    grid_size: int,
) -> dict[str, float]:
    neighbor_cells = _neighbor_cells(agent.location, grid_size)
    occupied = [grid[cell] for cell in neighbor_cells if cell in grid]
    same_type = sum(1 for peer_id in occupied if agents[peer_id].group == agent.group)
    same_ratio = float(same_type / len(occupied)) if occupied else 1.0
    density = float(len(occupied) / len(neighbor_cells)) if neighbor_cells else 0.0
    vacancy_pressure = 1.0 - density
    return {
        "same_type_neighbor_ratio": same_ratio,
        "vacancy_pressure": vacancy_pressure,
        "local_density": density,
    }


def _neighbor_ids(
    agents: Sequence[SchellingAgent],
    grid: dict[Coord, int],
    grid_size: int,
) -> list[list[int]]:
    neighbors: list[list[int]] = []
    for agent in agents:
        peers = [
            int(grid[cell])
            for cell in _neighbor_cells(agent.location, grid_size)
            if cell in grid
        ]
        neighbors.append(peers)
    return neighbors


def _vacant_cells(grid: dict[Coord, int], grid_size: int) -> list[Coord]:
    return [
        (row, col)
        for row in range(grid_size)
        for col in range(grid_size)
        if (row, col) not in grid
    ]


def _relocate_agents(
    agents: Sequence[SchellingAgent],
    observations: Sequence[torch.Tensor],
    grid_size: int,
    rng: np.random.Generator,
) -> int:
    grid = _build_grid(agents)
    vacant = _vacant_cells(grid, grid_size)
    moved = 0
    for agent, observation in zip(agents, observations, strict=True):
        if not vacant:
            break
        move_probability = float(agent.act_or_predict(observation)[0])
        same_ratio = float(observation[0])
        unhappy = same_ratio < agent.satisfaction_threshold
        if unhappy and rng.random() < move_probability:
            old_location = agent.location
            target_id = int(rng.integers(0, len(vacant)))
            new_location = vacant.pop(target_id)
            del grid[old_location]
            agent.location = new_location
            grid[new_location] = agent.agent_id
            vacant.append(old_location)
            moved += 1
    return moved


def _metrics(
    agents: Sequence[SchellingAgent],
    grid: dict[Coord, int],
    grid_size: int,
    move_rate: float,
) -> dict[str, float]:
    features = [
        _features_for_agent(agent, agents, grid, grid_size) for agent in agents
    ]
    same_ratios = [item["same_type_neighbor_ratio"] for item in features]
    satisfied = [
        ratio >= agent.satisfaction_threshold
        for ratio, agent in zip(same_ratios, agents, strict=True)
    ]
    return {
        "mean_satisfaction": float(np.mean(satisfied)) if satisfied else 0.0,
        "segregation_index": float(np.mean(same_ratios)) if same_ratios else 0.0,
        "move_rate": float(move_rate),
        "mean_move_probability": float(
            np.mean([agent.move_probability for agent in agents])
        ),
    }


def run_demo(seed: int = 7, steps: int = 12, agent_count: int = 36) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    config = SchellingConfig()
    grid_size = _grid_size(agent_count, config.occupancy_rate)
    agents = _initialize_agents(
        agent_count=agent_count,
        grid_size=grid_size,
        config=config,
        rng=rng,
    )
    assert all(isinstance(agent, NABMAgent) for agent in agents)

    channel = SocialChannel(
        name="move_probability",
        kind=SCALAR_PROBABILITY_CHANNEL,
        commit_mode="scalar_attribute_commit",
    )
    social_step = NABMStep(
        social_block=SocialBlock(alpha=config.social_alpha),
        channel=channel,
        commit_adapter=ScalarAttributeCommitAdapter(
            agents=agents,
            attribute="move_probability",
        ),
    )

    history: list[dict[str, Any]] = []
    social_update_norms: list[float] = []
    for step_id in range(steps):
        grid = _build_grid(agents)
        observations = [
            agent.observe(_features_for_agent(agent, agents, grid, grid_size))
            for agent in agents
        ]
        for agent, observation in zip(agents, observations, strict=True):
            agent.observation_spec().validate(observation)
            agent.social_message_spec().validate(agent.social_message(observation))
            agent.local_update(observation)

        moved_count = _relocate_agents(agents, observations, grid_size, rng)
        grid = _build_grid(agents)
        neighbors = _neighbor_ids(agents, grid, grid_size)
        values = np.asarray([agent.move_probability for agent in agents], dtype=float)
        peer_selection = SocialBlock(alpha=0.0).select_scalar_output_peers(
            neighbors=neighbors,
            values=values,
            peer_rule="output_similarity",
            threshold=config.peer_similarity_threshold,
        )
        result = social_step.run(values=values, peer_ids=peer_selection.peer_ids)
        diagnostics = result.diagnostics.aggregate_row()
        metrics = _metrics(
            agents=agents,
            grid=grid,
            grid_size=grid_size,
            move_rate=moved_count / agent_count if agent_count else 0.0,
        )
        social_update_norms.append(result.diagnostics.mean_update_norm)
        history.append(
            {
                "step": step_id,
                "metrics": metrics,
                "mean_peer_count": result.diagnostics.mean_peer_count,
                **diagnostics,
            }
        )

    final_metrics = history[-1]["metrics"] if history else {}
    return {
        "example": "schelling",
        "seed": seed,
        "steps": steps,
        "agent_count": agent_count,
        "grid_size": grid_size,
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
