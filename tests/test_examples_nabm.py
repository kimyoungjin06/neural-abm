from __future__ import annotations

import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest
import torch

from neural_abm import NABMAgent

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"


def load_example_module(name: str) -> ModuleType:
    module_name = f"examples.{name}"
    spec = spec_from_file_location(module_name, EXAMPLES_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load example module: {name}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


schelling_nabm = load_example_module("schelling_nabm")
epidemic_compliance_nabm = load_example_module("epidemic_compliance_nabm")
market_pricing_nabm = load_example_module("market_pricing_nabm")


@pytest.mark.parametrize(
    ("module", "agent_count", "expected_channel", "expected_metric_keys"),
    [
        (
            schelling_nabm,
            28,
            "move_probability",
            {"mean_satisfaction", "segregation_index", "move_rate"},
        ),
        (
            epidemic_compliance_nabm,
            30,
            "compliance_probability",
            {"infection_rate", "compliance_rate", "contact_reduction"},
        ),
        (
            market_pricing_nabm,
            26,
            "pricing_aggressiveness",
            {"mean_price", "trade_rate", "inventory_dispersion", "profit_mean"},
        ),
    ],
)
def test_examples_run_nabm_demo_summaries(
    module: ModuleType,
    agent_count: int,
    expected_channel: str,
    expected_metric_keys: set[str],
) -> None:
    summary = module.run_demo(seed=1, steps=10, agent_count=agent_count)

    assert summary["agent_count"] == agent_count
    assert summary["steps"] == 10
    assert summary["social_channel"] == expected_channel
    assert summary["commit_mode"] == "scalar_attribute_commit"
    assert len(summary["history"]) == 10
    assert len(summary["social_update_norms"]) == 10
    assert summary["mean_social_update_norm"] >= 0.0
    assert summary["max_social_update_norm"] > 0.0
    assert any(norm > 0.0 for norm in summary["social_update_norms"])
    assert expected_metric_keys <= set(summary["metrics"])

    for row in summary["history"]:
        assert row["social_channel"] == expected_channel
        assert row["commit_mode"] == "scalar_attribute_commit"
        assert row["mean_social_update_norm"] >= 0.0
        assert row["max_social_update_norm"] >= row["mean_social_update_norm"]
        assert row["active_social_agent_count"] > 0


def test_schelling_demo_metric_ranges() -> None:
    metrics = schelling_nabm.run_demo(seed=1, steps=10, agent_count=28)["metrics"]

    assert 0.0 <= metrics["mean_satisfaction"] <= 1.0
    assert 0.0 <= metrics["segregation_index"] <= 1.0
    assert 0.0 <= metrics["move_rate"] <= 1.0
    assert 0.0 <= metrics["mean_move_probability"] <= 1.0


def test_epidemic_demo_metric_ranges() -> None:
    metrics = epidemic_compliance_nabm.run_demo(
        seed=1,
        steps=10,
        agent_count=30,
    )["metrics"]

    assert 0.0 <= metrics["infection_rate"] <= 1.0
    assert 0.0 <= metrics["compliance_rate"] <= 1.0
    assert 0.0 <= metrics["contact_reduction"] <= 1.0
    assert 0.0 <= metrics["mean_recent_exposure"] <= 1.0


def test_market_demo_metric_ranges() -> None:
    metrics = market_pricing_nabm.run_demo(seed=1, steps=10, agent_count=26)["metrics"]

    assert 0.0 < metrics["mean_price"] < 50.0
    assert 0.0 <= metrics["trade_rate"] <= 1.0
    assert 0.0 <= metrics["inventory_dispersion"] < 5.0
    assert math.isfinite(metrics["profit_mean"])
    assert 0.0 <= metrics["mean_pricing_aggressiveness"] <= 1.0


@pytest.mark.parametrize(
    ("agent", "observation"),
    [
        (
            schelling_nabm.make_agent(agent_id=2),
            torch.tensor([0.4, 0.25, 0.75], dtype=torch.float32),
        ),
        (
            epidemic_compliance_nabm.make_agent(agent_id=2),
            torch.tensor([0.2, 0.25, 0.30, 0.10], dtype=torch.float32),
        ),
        (
            market_pricing_nabm.make_agent(agent_id=2),
            torch.tensor([0.7, 0.5, 0.4, 0.6], dtype=torch.float32),
        ),
    ],
)
def test_example_agents_satisfy_nabm_contract(
    agent: NABMAgent,
    observation: torch.Tensor,
) -> None:
    assert isinstance(agent, NABMAgent)

    observed = agent.observe(observation)
    agent.observation_spec().validate(observed)
    prediction = agent.act_or_predict(observed)
    message = agent.social_message(observed)
    agent.social_message_spec().validate(message)
    local_loss = agent.local_update(observed)
    state = agent.log_state(observed)

    assert torch.is_tensor(prediction)
    assert prediction.numel() >= 1
    assert all(torch.isfinite(prediction))
    assert local_loss >= 0.0
    assert message["agent_id"] == agent.agent_id
    assert state["agent_id"] == agent.agent_id
    assert 0.0 <= state["confidence"] <= 1.0
    assert state["param_norm"] >= 0.0
    assert state["latent_norm"] >= 0.0
