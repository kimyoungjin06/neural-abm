from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from binary_config_helpers import toy1_config
from neural_abm.config import load_toy1_config
from neural_abm.toy_classification import run_toy1


def write_tiny_config(tmp_path: Path, mixer: str, peer_rule: str) -> Path:
    config = toy1_config(
        {
            "run": {
                "name": f"tiny_{mixer}",
                "seed": 7,
                "output_dir": str(tmp_path / "runs"),
            },
            "simulation": {
                "epochs": 1,
                "sync_mode": "synchronous",
                "device": "cpu",
            },
            "data": {
                "boundary": "sine",
                "label_noise": 0.01,
                "train_pool_size": 500,
                "probe_size": 32,
                "test_size": 64,
            },
            "agents": {
                "count": 5,
                "init_mode": "same_init",
                "model": {
                    "input_dim": 2,
                    "hidden_dim": 8,
                    "output_dim": 2,
                    "activation": "relu",
                },
                "optimizer": {
                    "name": "adam",
                    "learning_rate": 0.01,
                },
                "training": {
                    "local_batch_size": 8,
                    "local_steps_per_epoch": 1,
                },
                "shards": {
                    "policy": "five_group_bias",
                    "groups": {
                        "left_region": {"count": 1, "samples_per_agent": 40},
                        "right_region": {"count": 1, "samples_per_agent": 40},
                        "boundary_region": {"count": 1, "samples_per_agent": 40},
                        "noisy_labels": {
                            "count": 1,
                            "samples_per_agent": 40,
                            "label_noise": 0.20,
                        },
                        "small_balanced": {"count": 1, "samples_per_agent": 20},
                    },
                },
            },
            "graph": {
                "type": "watts_strogatz",
                "k": 2,
                "rewire_probability": 0.0,
            },
            "social": {
                "mixer": mixer,
                "peer_rule": peer_rule,
                "alpha": 0.25,
                "threshold": 0.0,
                "communication_budget": {
                    "probe_predictions": 32,
                    "latent_dim": 8,
                    "scalar_summary": 8,
                },
            },
            "logging": {
                "micro_state": True,
                "interval": 1,
                "aggregate_metrics": True,
            },
        }
    )
    path = tmp_path / f"{mixer}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mixer", "peer_rule"),
    [
        ("none", "none"),
        ("output_average", "output_similarity"),
        ("latent_average", "latent_similarity"),
        ("parameter_average", "state_similarity"),
        ("parameter_aligned_average", "state_similarity"),
        ("parameter_aligned_average", "aligned_state_similarity"),
    ],
)
def test_toy1_runner_smoke(tmp_path: Path, mixer: str, peer_rule: str) -> None:
    config_path = write_tiny_config(tmp_path, mixer=mixer, peer_rule=peer_rule)
    config = load_toy1_config(config_path)
    result = run_toy1(config=config, config_path=config_path)

    assert result.run_dir.exists()
    assert (result.run_dir / "micro_state.csv").exists()
    assert (result.run_dir / "aggregate_metrics.csv").exists()
    assert (result.run_dir / "summary.json").exists()
    assert result.toy == "toy1"
    assert 0.0 <= result.domain_metrics["domain_final_mean_global_accuracy"] <= 1.0
    assert 0.0 <= result.domain_metrics["domain_final_mean_consensus"] <= 1.0


def test_toy1_output_average_uses_unit_social_diagnostics(tmp_path: Path) -> None:
    config_path = write_tiny_config(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
    )
    result = run_toy1(config=load_toy1_config(config_path), config_path=config_path)

    with (result.run_dir / "aggregate_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        aggregate_row = next(csv.DictReader(handle))
    with (result.run_dir / "micro_state.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        micro_row = next(csv.DictReader(handle))

    assert aggregate_row["social_channel"] == "probe_output_distribution"
    assert aggregate_row["commit_mode"] == "distillation_step"
    assert float(aggregate_row["active_social_agent_count"]) > 0.0
    assert micro_row["social_channel"] == "probe_output_distribution"
    assert micro_row["commit_mode"] == "distillation_step"
