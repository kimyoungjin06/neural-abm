"""Market pricing and seller strategy demo using the reusable NABM social unit."""

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

PRICE_SCALE = 5.0


@dataclass(frozen=True)
class MarketConfig:
    base_demand_per_seller: float = 0.78
    initial_inventory: float = 10.0
    restock_rate: float = 1.75
    price_sensitivity: float = 1.45
    local_learning_rate: float = 0.28
    social_alpha: float = 0.35
    peer_similarity_threshold: float = 0.0


@dataclass
class MarketSellerAgent:
    agent_id: int
    inventory: float
    cash: float
    unit_cost: float
    price: float
    pricing_aggressiveness: float
    last_sales: float
    last_profit: float
    local_learning_rate: float

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="market_seller_context",
            tensor_shape=(4,),
            dtype=torch.float32,
            description="inventory, wealth, market price, demand signal",
        )

    def social_message_spec(self) -> SocialMessageSpec:
        return SocialMessageSpec(
            required_keys=(
                "agent_id",
                "pricing_aggressiveness",
                "latent_summary",
                "confidence",
                "param_norm",
            ),
            tensor_keys=("latent_summary",),
            probability_keys=("pricing_aggressiveness",),
        )

    def observe(self, x: Mapping[str, float] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().clone().to(dtype=torch.float32)
        return torch.tensor(
            [
                float(x["inventory_ratio"]),
                float(x["wealth_ratio"]),
                float(x["last_market_price"]) / PRICE_SCALE,
                float(x["demand_signal"]),
            ],
            dtype=torch.float32,
        )

    def act_or_predict(self, observation: torch.Tensor) -> torch.Tensor:
        observed = self.observe(observation)
        inventory_ratio = observed[..., 0]
        last_market_price = torch.clamp(observed[..., 2] * PRICE_SCALE, min=0.25)
        demand_signal = observed[..., 3]
        markup = 0.08 + 0.75 * float(self.pricing_aggressiveness)
        inventory_discount = torch.clamp(1.15 - 0.35 * inventory_ratio, 0.72, 1.18)
        demand_multiplier = torch.clamp(0.86 + 0.28 * demand_signal, 0.75, 1.20)
        price = last_market_price * (1.0 + markup) * inventory_discount * demand_multiplier
        floor = torch.full_like(price, self.unit_cost * 1.02)
        ceiling = torch.full_like(price, self.unit_cost * 8.0)
        return torch.minimum(torch.maximum(price, floor), ceiling).reshape(-1)

    def local_update(self, observation: torch.Tensor | None = None) -> float:
        if observation is None:
            observation = torch.tensor(
                [self.inventory / 10.0, self.cash / 100.0, self.price / PRICE_SCALE, 0.5],
                dtype=torch.float32,
            )
        observed = self.observe(observation)
        inventory_ratio = float(observed[0])
        demand_signal = float(observed[3])
        margin_signal = float(np.clip((self.price - self.unit_cost) / PRICE_SCALE, 0.0, 1.0))
        target = float(
            np.clip(
                0.18
                + 0.58 * demand_signal
                - 0.30 * inventory_ratio
                + 0.16 * margin_signal,
                0.0,
                1.0,
            )
        )
        before = float(self.pricing_aggressiveness)
        self.pricing_aggressiveness = float(
            np.clip(
                (1.0 - self.local_learning_rate) * before
                + self.local_learning_rate * target,
                0.0,
                1.0,
            )
        )
        return abs(self.pricing_aggressiveness - before)

    def social_message(self, observation: torch.Tensor | None = None) -> dict[str, Any]:
        if observation is None:
            latent_summary = torch.tensor(
                [
                    float(self.inventory / 10.0),
                    float(self.cash / 100.0),
                    float(self.price / PRICE_SCALE),
                    float(self.pricing_aggressiveness),
                ],
                dtype=torch.float32,
            )
        else:
            latent_summary = self.observe(observation)
        confidence = abs(float(self.pricing_aggressiveness) - 0.5) * 2.0
        return {
            "agent_id": self.agent_id,
            "pricing_aggressiveness": float(
                np.clip(self.pricing_aggressiveness, 0.0, 1.0)
            ),
            "latent_summary": latent_summary.detach().clone(),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "param_norm": float(
                abs(self.pricing_aggressiveness)
                + abs(self.price)
                + abs(self.inventory)
                + abs(self.cash / 100.0)
            ),
        }

    def log_state(self, observation: torch.Tensor | None = None) -> dict[str, Any]:
        message = self.social_message(observation)
        return {
            "agent_id": self.agent_id,
            "inventory": self.inventory,
            "cash": self.cash,
            "price": self.price,
            "pricing_aggressiveness": message["pricing_aggressiveness"],
            "last_sales": self.last_sales,
            "last_profit": self.last_profit,
            "confidence": message["confidence"],
            "param_norm": message["param_norm"],
            "latent_norm": float(torch.linalg.vector_norm(message["latent_summary"])),
        }


@dataclass
class ScalarAttributeCommitAdapter:
    agents: Sequence[MarketSellerAgent]
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


def make_agent(agent_id: int = 0) -> MarketSellerAgent:
    return MarketSellerAgent(
        agent_id=agent_id,
        inventory=10.0,
        cash=50.0,
        unit_cost=1.0 + 0.03 * (agent_id % 3),
        price=2.0 + 0.10 * (agent_id % 4),
        pricing_aggressiveness=0.20 + 0.08 * (agent_id % 5),
        last_sales=0.0,
        last_profit=0.0,
        local_learning_rate=0.28,
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
    config: MarketConfig,
    rng: np.random.Generator,
) -> list[MarketSellerAgent]:
    agents: list[MarketSellerAgent] = []
    for agent_id in range(agent_count):
        unit_cost = float(rng.uniform(0.85, 1.25))
        agents.append(
            MarketSellerAgent(
                agent_id=agent_id,
                inventory=float(rng.uniform(0.75, 1.25) * config.initial_inventory),
                cash=float(rng.uniform(35.0, 65.0)),
                unit_cost=unit_cost,
                price=float(rng.uniform(1.45, 2.60)),
                pricing_aggressiveness=float(rng.uniform(0.08, 0.82)),
                last_sales=0.0,
                last_profit=0.0,
                local_learning_rate=config.local_learning_rate,
            )
        )
    return agents


def _observation_for_agent(
    agent: MarketSellerAgent,
    last_market_price: float,
    demand_signal: float,
    config: MarketConfig,
) -> dict[str, float]:
    return {
        "inventory_ratio": float(
            np.clip(agent.inventory / (config.initial_inventory * 1.5), 0.0, 1.5)
        ),
        "wealth_ratio": float(np.clip(agent.cash / 100.0, 0.0, 2.0)),
        "last_market_price": float(last_market_price),
        "demand_signal": float(np.clip(demand_signal, 0.0, 1.0)),
    }


def _clear_market(
    agents: Sequence[MarketSellerAgent],
    observations: Sequence[torch.Tensor],
    demand_signal: float,
    config: MarketConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    prices = np.asarray(
        [
            float(agent.act_or_predict(observation)[0])
            for agent, observation in zip(agents, observations, strict=True)
        ],
        dtype=float,
    )
    for agent, price in zip(agents, prices, strict=True):
        agent.price = float(price)

    total_demand = max(
        1,
        int(round(len(agents) * config.base_demand_per_seller * (0.70 + demand_signal))),
    )
    relative_prices = prices / max(float(np.mean(prices)), 1e-6)
    weights = np.exp(-config.price_sensitivity * relative_prices)
    weights = weights / float(np.sum(weights))
    desired_quantities = total_demand * weights

    sold_quantities: list[float] = []
    for agent, desired in zip(agents, desired_quantities, strict=True):
        stochastic_round = math_floor_with_fraction(desired, rng)
        sold = min(float(stochastic_round), float(agent.inventory))
        revenue = sold * agent.price
        profit = sold * (agent.price - agent.unit_cost)
        agent.inventory = max(0.0, agent.inventory - sold)
        agent.cash += revenue
        agent.last_sales = sold
        agent.last_profit = profit
        restock = config.restock_rate * float(rng.uniform(0.75, 1.25))
        agent.inventory += restock
        sold_quantities.append(sold)

    total_sold = float(np.sum(sold_quantities))
    trade_rate = float(np.clip(total_sold / total_demand, 0.0, 1.0))
    if total_sold > 0.0:
        market_price = float(np.average(prices, weights=sold_quantities))
    else:
        market_price = float(np.mean(prices))
    return market_price, trade_rate


def math_floor_with_fraction(value: float, rng: np.random.Generator) -> int:
    base = int(np.floor(value))
    return base + int(rng.random() < value - base)


def _metrics(agents: Sequence[MarketSellerAgent], trade_rate: float) -> dict[str, float]:
    inventories = np.asarray([agent.inventory for agent in agents], dtype=float)
    prices = np.asarray([agent.price for agent in agents], dtype=float)
    profits = np.asarray([agent.last_profit for agent in agents], dtype=float)
    mean_inventory = float(np.mean(inventories)) if len(inventories) else 0.0
    return {
        "mean_price": float(np.mean(prices)) if len(prices) else 0.0,
        "trade_rate": float(np.clip(trade_rate, 0.0, 1.0)),
        "inventory_dispersion": float(
            np.std(inventories) / (mean_inventory + 1e-8)
        )
        if len(inventories)
        else 0.0,
        "profit_mean": float(np.mean(profits)) if len(profits) else 0.0,
        "mean_pricing_aggressiveness": float(
            np.mean([agent.pricing_aggressiveness for agent in agents])
        )
        if agents
        else 0.0,
    }


def run_demo(seed: int = 19, steps: int = 14, agent_count: int = 32) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    config = MarketConfig()
    agents = _initialize_agents(agent_count=agent_count, config=config, rng=rng)
    assert all(isinstance(agent, NABMAgent) for agent in agents)
    neighbors = _ring_neighbors(agent_count)
    last_market_price = float(np.mean([agent.price for agent in agents]))
    demand_signal = 0.55
    trade_rate = 0.0

    channel = SocialChannel(
        name="pricing_aggressiveness",
        kind=SCALAR_PROBABILITY_CHANNEL,
        commit_mode="scalar_attribute_commit",
    )
    social_step = NABMStep(
        social_block=SocialBlock(alpha=config.social_alpha),
        channel=channel,
        commit_adapter=ScalarAttributeCommitAdapter(
            agents=agents,
            attribute="pricing_aggressiveness",
        ),
    )

    history: list[dict[str, Any]] = []
    social_update_norms: list[float] = []
    for step_id in range(steps):
        observations = [
            agent.observe(
                _observation_for_agent(agent, last_market_price, demand_signal, config)
            )
            for agent in agents
        ]
        for agent, observation in zip(agents, observations, strict=True):
            agent.observation_spec().validate(observation)
            agent.social_message_spec().validate(agent.social_message(observation))
            agent.local_update(observation)

        last_market_price, trade_rate = _clear_market(
            agents=agents,
            observations=observations,
            demand_signal=demand_signal,
            config=config,
            rng=rng,
        )
        demand_signal = float(np.clip(0.65 * demand_signal + 0.35 * trade_rate, 0.0, 1.0))
        values = np.asarray(
            [agent.pricing_aggressiveness for agent in agents],
            dtype=float,
        )
        peer_selection = SocialBlock(alpha=0.0).select_scalar_output_peers(
            neighbors=[list(peer_ids) for peer_ids in neighbors],
            values=values,
            peer_rule="output_similarity",
            threshold=config.peer_similarity_threshold,
        )
        result = social_step.run(values=values, peer_ids=peer_selection.peer_ids)
        diagnostics = result.diagnostics.aggregate_row()
        metrics = _metrics(agents, trade_rate)
        social_update_norms.append(result.diagnostics.mean_update_norm)
        history.append(
            {
                "step": step_id,
                "metrics": metrics,
                "mean_peer_count": result.diagnostics.mean_peer_count,
                **diagnostics,
            }
        )

    final_metrics = history[-1]["metrics"] if history else _metrics(agents, trade_rate)
    return {
        "example": "market_pricing",
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
