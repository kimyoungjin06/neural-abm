#!/usr/bin/env python
"""Benchmark Toy2/4/5 neural binary runners on CPU and accelerator devices."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable

import torch
import yaml

from neural_abm.accelerator import resolve_neural_update_backend, resolve_torch_device
from neural_abm.config import (
    Toy2Config,
    Toy4Config,
    Toy5Config,
    load_toy2_config,
    load_toy4_config,
    load_toy5_config,
)
from neural_abm.spatial_binary import BinaryToyResult
from neural_abm.toy_contagion import run_toy5
from neural_abm.toy_pd import run_toy2
from neural_abm.toy_public_goods import run_toy4


ToyLoader = Callable[[Path], Toy2Config | Toy4Config | Toy5Config]
ToyRunner = Callable[..., BinaryToyResult]
PARITY_TOLERANCES = {
    "final_action_rate": 1e-9,
    "final_mean_policy_action_probability": 1e-5,
    "final_mean_reputation": 1e-9,
}
LOOP_COMPARISON_FIELDS = [
    "loop_speedup",
    "final_action_rate_diff_vs_loop",
    "final_mean_policy_action_probability_diff_vs_loop",
    "final_mean_reputation_diff_vs_loop",
    "parity_passed_vs_loop",
]


@dataclass(frozen=True)
class CaseMeasurement:
    seconds: float
    result: BinaryToyResult
    timing_rows: list[dict[str, object]]
    stage_seconds: dict[str, float]

TIMING_STAGES = [
    "make_run_dir",
    "write_metadata",
    "initial_state",
    "writer_setup",
    "initial_step_result",
    "initial_aggregate_row",
    "write_initial_aggregate",
    "sample_revision_mask",
    "hooked_step_total",
    "build_step_context",
    "local_step",
    "neural_context_peers",
    "build_observations",
    "policy_readout",
    "decision_selection",
    "local_training",
    "local_trainable_parameters",
    "local_loss_update",
    "local_loss_forward",
    "local_optimizer_update",
    "local_autograd_grad",
    "local_adam_update",
    "cache_refresh",
    "post_local_readout",
    "select_peers",
    "social_step",
    "social_distillation",
    "social_trainable_parameters",
    "social_loss_update",
    "social_mix",
    "social_loss_forward",
    "social_optimizer_update",
    "social_autograd_grad",
    "social_adam_update",
    "sample_actions",
    "commit_actions",
    "post_step_state_policy",
    "update_payoff_ema",
    "update_reputation_ema",
    "mobility_swaps",
    "finalize_hook_step",
    "aggregate_row",
    "write_aggregate",
    "write_micro",
]

CSV_FIELDS = [
    "toy",
    "scenario",
    "peer_rule",
    "training_backend",
    "resolved_training_backend",
    "device",
    "agent_count",
    "epochs",
    "seed",
    "warmup_runs",
    "repeats",
    "seconds",
    "seconds_std",
    "seconds_min",
    "seconds_max",
    "epochs_per_second",
    "epochs_per_second_std",
    "agent_steps_per_second",
    "agent_steps_per_second_std",
    "final_action_rate",
    "final_action_rate_std",
    "final_mean_policy_action_probability",
    "final_mean_policy_action_probability_std",
    "final_mean_reputation",
    "final_mean_reputation_std",
    *LOOP_COMPARISON_FIELDS,
    *[f"timing_{stage}_seconds" for stage in TIMING_STAGES],
    *[f"timing_{stage}_seconds_std" for stage in TIMING_STAGES],
]

STAGE_CSV_FIELDS = [
    "toy",
    "scenario",
    "peer_rule",
    "training_backend",
    "resolved_training_backend",
    "device",
    "agent_count",
    "epochs",
    "seed",
    "repeat_index",
    "warmup_runs",
    "repeats",
    "run_id",
    "epoch",
    "policy_rule",
    "coordination_mixer",
    "stage",
    "seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "auto"],
        help="Torch devices to benchmark, e.g. cpu cuda cuda:0 auto.",
    )
    parser.add_argument(
        "--toys",
        nargs="+",
        choices=["toy2", "toy4", "toy5"],
        default=["toy2", "toy4", "toy5"],
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=[64],
        help=(
            "Agent counts to benchmark. Toy2/Toy4 require counts that can be "
            "factored into grid_width > 1 and grid_height > 1."
        ),
    )
    parser.add_argument(
        "--mixers",
        nargs="+",
        choices=["none", "output_average"],
        default=["output_average"],
        help="Coordination mixers to benchmark.",
    )
    parser.add_argument(
        "--peer-rules",
        nargs="+",
        choices=["none", "output_similarity"],
        default=["none"],
        help="Coordination peer rules to benchmark.",
    )
    parser.add_argument(
        "--training-backends",
        nargs="+",
        choices=["loop", "batched", "tensor_batched", "auto"],
        default=["loop"],
        help=(
            "Neural update backend to benchmark."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Number of untimed warmup runs per benchmark case.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of measured repeats per benchmark case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/toy_gpu_core_benchmark.csv"),
    )
    parser.add_argument(
        "--stage-output",
        type=Path,
        default=None,
        help=(
            "Optional detailed stage timing CSV path. Defaults to "
            "<output stem>_stage_timings.csv."
        ),
    )
    parser.add_argument(
        "--no-stage-timing",
        action="store_true",
        help=(
            "Disable per-stage timing collection for measured repeats. This keeps "
            "the end-to-end wall-clock measurement free of timing hook overhead."
        ),
    )
    parser.add_argument(
        "--require-backend-parity",
        action="store_true",
        help=(
            "Exit nonzero when a non-loop backend has a loop baseline in the same "
            "benchmark group and exceeds parity tolerances."
        ),
    )
    parser.add_argument(
        "--run-output-dir",
        type=Path,
        default=Path("experiments/runs/toy_gpu_core_benchmark"),
    )
    args = parser.parse_args()
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_slug(device: torch.device) -> str:
    return str(device).replace(":", "_")


def grid_dimensions(agent_count: int) -> tuple[int, int]:
    """Return near-square grid dimensions for Toy2/Toy4 exact agent counts."""

    if agent_count < 4:
        raise ValueError("Toy2/Toy4 agent counts must be at least 4")
    for height in range(int(agent_count**0.5), 1, -1):
        if agent_count % height == 0:
            return agent_count // height, height
    raise ValueError(
        "Toy2/Toy4 agent counts must be composite so grid_width and grid_height "
        f"are both > 1; got {agent_count}"
    )


def coordination_config(
    mixer: str,
    *,
    peer_rule: str = "none",
    include_budget: bool = False,
) -> dict[str, Any]:
    if mixer not in {"none", "output_average"}:
        raise ValueError(f"Unsupported benchmark mixer: {mixer}")
    if peer_rule not in {"none", "output_similarity"}:
        raise ValueError(f"Unsupported benchmark peer rule: {peer_rule}")
    if mixer == "none" and peer_rule != "none":
        raise ValueError("Benchmark peer_rule must be 'none' when mixer is 'none'")
    config: dict[str, Any] = {
        "mixer": mixer,
        "peer_rule": peer_rule,
        "alpha": 0.25 if mixer == "output_average" else 0.0,
        "threshold": 0.0,
    }
    if include_budget:
        config["communication_budget"] = {
            "probe_predictions": 1,
            "latent_dim": 16,
            "scalar_summary": 8,
        }
    return config


def toy2_config(
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    output_dir: Path,
    agent_count: int,
    mixer: str,
    training_backend: str,
    peer_rule: str = "none",
) -> dict[str, Any]:
    grid_width, grid_height = grid_dimensions(agent_count)
    scenario = scenario_name(mixer, peer_rule)
    return {
        "run": {
            "name": (
                f"toy2_gpu_core_{scenario}_{training_backend}_"
                f"{agent_count}_{device_slug(device)}"
            ),
            "seed": seed,
            "output_dir": str(output_dir),
        },
        "simulation": {
            "epochs": epochs,
            "sync_mode": "synchronous",
            "device": str(device),
        },
        "model": {
            "policy": {
                "rule": "neural_policy",
                "learning_enabled": True,
                "revision_rate": 1.0,
                "selection_strength": 1.0,
                "temperature": 1.0,
                "neural_update_backend": training_backend,
                "decision": {
                    "mode": "sampled",
                    "action_temperature": 1.0,
                    "exploration_epsilon": 0.0,
                    "calibration": {"mode": "none", "strength": 4.0},
                },
                "domain": {
                    "local_update_rule": "counterfactual_advantage",
                    "neural_peer_mode": "spatial",
                    "interaction_mode": "spatial",
                    "payoff_transform": "linear",
                },
            },
            "agents": {
                "init_mode": "independent_init",
                "model": {
                    "input_dim": 6,
                    "hidden_dim": 16,
                    "output_dim": 2,
                    "activation": "relu",
                },
                "optimizer": {"name": "adam", "learning_rate": 0.01},
                "policy_prior_action_probability": None,
            },
            "coordination": coordination_config(
                mixer,
                peer_rule=peer_rule,
                include_budget=True,
            ),
            "state": binary_state_config(),
        },
        "domain": {
            "toy": "toy2",
            "environment": {
                "grid_width": grid_width,
                "grid_height": grid_height,
                "neighborhood": "von_neumann",
                "periodic": True,
                "initial_action_probability": 0.5,
                "reward_ema_decay": 0.9,
                "entropy_beta": 0.01,
                "payoff_R": 3.0,
                "payoff_S": 0.0,
                "payoff_T": 5.0,
                "payoff_P": 1.0,
            },
            "game": {
                "family": "prisoner_dilemma",
                "payoff": {"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0},
            },
        },
        "logging": logging_config(),
    }


def toy4_config(
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    output_dir: Path,
    agent_count: int,
    mixer: str,
    training_backend: str,
    peer_rule: str = "none",
) -> dict[str, Any]:
    grid_width, grid_height = grid_dimensions(agent_count)
    scenario = scenario_name(mixer, peer_rule)
    return {
        "run": {
            "name": (
                f"toy4_gpu_core_{scenario}_{training_backend}_"
                f"{agent_count}_{device_slug(device)}"
            ),
            "seed": seed,
            "output_dir": str(output_dir),
        },
        "simulation": {
            "epochs": epochs,
            "sync_mode": "synchronous",
            "device": str(device),
        },
        "model": {
            "policy": {
                "rule": "neural_policy",
                "learning_enabled": True,
                "revision_rate": 1.0,
                "selection_strength": 1.0,
                "temperature": 1.0,
                "neural_update_backend": training_backend,
                "decision": {
                    "mode": "sampled",
                    "action_temperature": 1.0,
                    "exploration_epsilon": 0.0,
                },
                "domain": {},
            },
            "agents": {
                "init_mode": "independent_init",
                "model": {
                    "input_dim": 6,
                    "hidden_dim": 16,
                    "output_dim": 2,
                    "activation": "relu",
                },
                "optimizer": {"name": "adam", "learning_rate": 0.01},
            },
            "coordination": coordination_config(mixer, peer_rule=peer_rule),
            "state": binary_state_config(),
        },
        "domain": {
            "toy": "toy4",
            "environment": {
                "grid_width": grid_width,
                "grid_height": grid_height,
                "initial_action_probability": 0.5,
                "reward_ema_decay": 0.9,
                "entropy_beta": 0.01,
                "resource_enabled": False,
                "resource_initial": 100.0,
                "resource_carrying_capacity": 100.0,
                "resource_recovery_rate": 0.05,
                "resource_extraction_per_defector": 1.0,
                "resource_collapse_threshold": 0.0,
            },
            "game": {
                "multiplier": 1.6,
                "contribution_cost": 1.0,
                "group_mode": "local_neighborhood",
            },
            "graph": {
                "type": "grid",
                "neighborhood": "von_neumann",
                "periodic": True,
            },
        },
        "logging": logging_config(),
    }


def toy5_config(
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    output_dir: Path,
    agent_count: int,
    mixer: str,
    training_backend: str,
    peer_rule: str = "none",
) -> dict[str, Any]:
    scenario = scenario_name(mixer, peer_rule)
    return {
        "run": {
            "name": (
                f"toy5_gpu_core_{scenario}_{training_backend}_"
                f"{agent_count}_{device_slug(device)}"
            ),
            "seed": seed,
            "output_dir": str(output_dir),
        },
        "simulation": {
            "epochs": epochs,
            "sync_mode": "synchronous",
            "device": str(device),
        },
        "model": {
            "policy": {
                "rule": "neural_policy",
                "learning_enabled": True,
                "revision_rate": 1.0,
                "temperature": 1.0,
                "neural_update_backend": training_backend,
                "decision": {
                    "mode": "sampled",
                    "action_temperature": 1.0,
                    "exploration_epsilon": 0.0,
                },
                "domain": {
                    "repeated_exposure_decay": 0.0,
                    "adoption_is_absorbing": True,
                },
            },
            "agents": {
                "count": agent_count,
                "init_mode": "independent_init",
                "model": {
                    "input_dim": 6,
                    "hidden_dim": 16,
                    "output_dim": 2,
                    "activation": "relu",
                },
                "optimizer": {"name": "adam", "learning_rate": 0.01},
            },
            "coordination": coordination_config(mixer, peer_rule=peer_rule),
            "state": binary_state_config(),
        },
        "domain": {
            "toy": "toy5",
            "environment": {
                "initial_action_fraction": 0.05,
                "seed_selection": "random",
                "threshold_mode": "homogeneous",
                "homogeneous_threshold": 0.25,
                "heterogeneous_threshold_low": 0.15,
                "heterogeneous_threshold_high": 0.55,
                "simple_contagion_probability": 0.08,
            },
            "graph": {
                "type": "watts_strogatz",
                "k": 6,
                "rewire_probability": 0.1,
            },
        },
        "logging": logging_config(),
    }


def binary_state_config() -> dict[str, Any]:
    return {
        "reputation": {
            "enabled": True,
            "decay": 0.9,
            "peer_rule": "spatial",
            "temperature": 1.0,
            "noise": 0.0,
            "observation_mode": "none",
        },
        "mobility": {
            "enabled": False,
            "rate": 0.0,
            "candidate_pool_size": 8,
            "selection_rule": "local_quality",
            "move_cost": 0.0,
        },
    }


def logging_config() -> dict[str, Any]:
    return {
        "micro_state": False,
        "interval": 1,
        "aggregate_metrics": True,
        "probe_predictions": False,
        "probe_prediction_interval": 1,
    }


TOY_SPECS: dict[str, tuple[Callable[..., dict[str, Any]], ToyLoader, ToyRunner]] = {
    "toy2": (toy2_config, load_toy2_config, run_toy2),
    "toy4": (toy4_config, load_toy4_config, run_toy4),
    "toy5": (toy5_config, load_toy5_config, run_toy5),
}


def config_path_for(
    *,
    run_output_dir: Path,
    toy: str,
    device: torch.device,
    seed: int,
    agent_count: int,
    mixer: str,
    peer_rule: str = "none",
    training_backend: str,
    run_label: str,
) -> Path:
    return (
        run_output_dir
        / "_configs"
        / (
            f"{toy}_{scenario_name(mixer, peer_rule)}_{agent_count}_"
            f"{training_backend}_{device_slug(device)}_seed{seed}_{run_label}.yaml"
        )
    )


def scenario_name(mixer: str, peer_rule: str = "none") -> str:
    if peer_rule == "none":
        return f"neural_{mixer}"
    return f"neural_{mixer}_{peer_rule}"


def write_config(path: Path, raw: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )


def configured_agent_count(config: Toy2Config | Toy4Config | Toy5Config) -> int:
    if isinstance(config, Toy2Config):
        return config.environment.grid_width * config.environment.grid_height
    if isinstance(config, Toy4Config):
        return config.agent_count
    return config.agents.count


def run_once(
    *,
    toy: str,
    device: torch.device,
    epochs: int,
    seed: int,
    agent_count: int,
    mixer: str,
    training_backend: str,
    peer_rule: str = "none",
    run_output_dir: Path,
    run_label: str,
    collect_timing: bool,
) -> CaseMeasurement:
    raw_builder, loader, runner = TOY_SPECS[toy]
    raw = raw_builder(
        seed=seed,
        epochs=epochs,
        device=device,
        output_dir=run_output_dir,
        agent_count=agent_count,
        mixer=mixer,
        training_backend=training_backend,
        peer_rule=peer_rule,
    )
    raw["run"]["name"] = f"{raw['run']['name']}_{run_label}"
    config_path = config_path_for(
        run_output_dir=run_output_dir,
        toy=toy,
        device=device,
        seed=seed,
        agent_count=agent_count,
        mixer=mixer,
        peer_rule=peer_rule,
        training_backend=training_backend,
        run_label=run_label,
    )
    write_config(config_path, raw)
    config = loader(config_path)

    case_timing_rows: list[dict[str, object]] | None = [] if collect_timing else None
    synchronize(device)
    start = time.perf_counter()
    runner_kwargs: dict[str, object] = {"timing_rows": case_timing_rows}
    if toy in {"toy2", "toy4", "toy5"}:
        runner_kwargs["neural_update_backend"] = training_backend
    result = runner(config, config_path, **runner_kwargs)
    synchronize(device)
    seconds = time.perf_counter() - start

    return CaseMeasurement(
        seconds=seconds,
        result=result,
        timing_rows=[] if case_timing_rows is None else case_timing_rows,
        stage_seconds=timing_summary([] if case_timing_rows is None else case_timing_rows),
    )


def run_case(
    *,
    toy: str,
    device: torch.device,
    epochs: int,
    seed: int,
    agent_count: int,
    mixer: str,
    training_backend: str,
    peer_rule: str = "none",
    warmup_runs: int,
    repeats: int,
    run_output_dir: Path,
    collect_stage_timing: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    for warmup_index in range(warmup_runs):
        run_once(
            toy=toy,
            device=device,
            epochs=epochs,
            seed=seed,
            agent_count=agent_count,
            mixer=mixer,
            training_backend=training_backend,
            peer_rule=peer_rule,
            run_output_dir=run_output_dir,
            run_label=f"warmup{warmup_index:02d}",
            collect_timing=False,
        )

    measurements: list[CaseMeasurement] = []
    all_timing_rows: list[dict[str, object]] = []
    for repeat_index in range(repeats):
        measurement = run_once(
            toy=toy,
            device=device,
            epochs=epochs,
            seed=seed,
            agent_count=agent_count,
            mixer=mixer,
            training_backend=training_backend,
            peer_rule=peer_rule,
            run_output_dir=run_output_dir,
            run_label=f"repeat{repeat_index:02d}",
            collect_timing=collect_stage_timing,
        )
        measurements.append(measurement)
        for row in measurement.timing_rows:
            row["repeat_index"] = repeat_index
            row["warmup_runs"] = warmup_runs
            row["repeats"] = repeats
        all_timing_rows.extend(measurement.timing_rows)

    count = agent_count
    scenario = scenario_name(mixer, peer_rule)
    resolved_training_backend = (
        resolve_neural_update_backend(
            training_backend,
            device=device,
            agent_count=count,
        )
        if toy in {"toy2", "toy4", "toy5"}
        else "loop"
    )
    for row in all_timing_rows:
        row["scenario"] = scenario
        row["peer_rule"] = peer_rule
        row["training_backend"] = training_backend
        row["resolved_training_backend"] = resolved_training_backend
        row["epochs"] = epochs
        row["toy"] = toy
        row["device"] = str(device)
        row["agent_count"] = count
        row["seed"] = seed
    seconds_values = [measurement.seconds for measurement in measurements]
    epochs_per_second_values = [
        epochs / seconds if seconds else 0.0 for seconds in seconds_values
    ]
    agent_steps_per_second_values = [
        count * epochs / seconds if seconds else 0.0 for seconds in seconds_values
    ]
    return {
        "toy": toy,
        "scenario": scenario,
        "peer_rule": peer_rule,
        "training_backend": training_backend,
        "resolved_training_backend": resolved_training_backend,
        "device": str(device),
        "agent_count": count,
        "epochs": epochs,
        "seed": seed,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "seconds": metric_mean(seconds_values),
        "seconds_std": metric_std(seconds_values),
        "seconds_min": min(seconds_values),
        "seconds_max": max(seconds_values),
        "epochs_per_second": metric_mean(epochs_per_second_values),
        "epochs_per_second_std": metric_std(epochs_per_second_values),
        "agent_steps_per_second": metric_mean(agent_steps_per_second_values),
        "agent_steps_per_second_std": metric_std(agent_steps_per_second_values),
        "final_action_rate": metric_mean(
            [measurement.result.final_action_rate for measurement in measurements]
        ),
        "final_action_rate_std": metric_std(
            [measurement.result.final_action_rate for measurement in measurements]
        ),
        "final_mean_policy_action_probability": metric_mean(
            [
                measurement.result.final_mean_policy_action_probability
                for measurement in measurements
            ]
        ),
        "final_mean_policy_action_probability_std": metric_std(
            [
                measurement.result.final_mean_policy_action_probability
                for measurement in measurements
            ]
        ),
        "final_mean_reputation": metric_mean(
            [measurement.result.final_mean_reputation for measurement in measurements]
        ),
        "final_mean_reputation_std": metric_std(
            [measurement.result.final_mean_reputation for measurement in measurements]
        ),
        **stage_summary(measurements),
    }, all_timing_rows


def annotate_loop_comparisons(rows: list[dict[str, object]]) -> None:
    """Add loop-relative speedup and parity columns in-place."""

    baselines: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        _clear_loop_comparison_fields(row)
        if row.get("training_backend") == "loop":
            baselines[_loop_comparison_key(row)] = row

    for row in rows:
        baseline = baselines.get(_loop_comparison_key(row))
        if baseline is None:
            continue
        row_seconds = _row_float(row, "seconds")
        loop_seconds = _row_float(baseline, "seconds")
        if row_seconds is not None and row_seconds > 0.0 and loop_seconds is not None:
            row["loop_speedup"] = loop_seconds / row_seconds
        parity_passed = True
        for metric, tolerance in PARITY_TOLERANCES.items():
            diff_key = f"{metric}_diff_vs_loop"
            metric_value = _row_float(row, metric)
            baseline_value = _row_float(baseline, metric)
            if metric_value is None or baseline_value is None:
                row[diff_key] = ""
                parity_passed = False
                continue
            diff = abs(metric_value - baseline_value)
            row[diff_key] = diff
            if diff > tolerance:
                parity_passed = False
        row["parity_passed_vs_loop"] = parity_passed


def backend_parity_failures(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("training_backend") != "loop"
        and row.get("parity_passed_vs_loop") is False
    ]


def _clear_loop_comparison_fields(row: dict[str, object]) -> None:
    for field in LOOP_COMPARISON_FIELDS:
        row[field] = ""


def _loop_comparison_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("toy"),
        row.get("scenario"),
        row.get("device"),
        row.get("agent_count"),
        row.get("epochs"),
        row.get("seed"),
        row.get("repeats"),
    )


def _row_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value in {None, ""}:
        return None
    return float(value)


def timing_summary(timing_rows: list[dict[str, object]]) -> dict[str, float]:
    totals = {f"timing_{stage}_seconds": 0.0 for stage in TIMING_STAGES}
    for row in timing_rows:
        stage = str(row["stage"])
        key = f"timing_{stage}_seconds"
        if key in totals:
            totals[key] += float(row["seconds"])
    return totals


def stage_summary(measurements: list[CaseMeasurement]) -> dict[str, float | str]:
    summary: dict[str, float | str] = {}
    for stage in TIMING_STAGES:
        key = f"timing_{stage}_seconds"
        values = [measurement.stage_seconds[key] for measurement in measurements]
        summary[key] = metric_mean(values)
        summary[f"{key}_std"] = metric_std(values)
    return summary


def metric_mean(values: list[float | int | None]) -> float | str:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return ""
    return fmean(numeric_values)


def metric_std(values: list[float | int | None]) -> float | str:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return ""
    if len(numeric_values) == 1:
        return 0.0
    return stdev(numeric_values)


def resolve_devices(device_names: list[str]) -> list[torch.device]:
    devices: list[torch.device] = []
    for name in device_names:
        try:
            device = resolve_torch_device(name)
        except RuntimeError as exc:
            print(f"Skipping {name}: {exc}", file=sys.stderr)
            continue
        if device not in devices:
            devices.append(device)
    if not devices:
        raise RuntimeError("No benchmark devices are available")
    return devices


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def stage_output_path(output: Path, stage_output: Path | None) -> Path:
    if stage_output is not None:
        return stage_output
    return output.with_name(f"{output.stem}_stage_timings{output.suffix}")


def write_stage_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in STAGE_CSV_FIELDS}
            for row in rows
        )


def main() -> None:
    args = parse_args()
    devices = resolve_devices(args.devices)
    rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    for toy in args.toys:
        for agent_count in args.agent_counts:
            for mixer in args.mixers:
                for peer_rule in args.peer_rules:
                    if mixer == "none" and peer_rule != "none":
                        continue
                    for training_backend in args.training_backends:
                        for device in devices:
                            row, case_stage_rows = run_case(
                                toy=toy,
                                device=device,
                                epochs=args.epochs,
                                seed=args.seed,
                                agent_count=agent_count,
                                mixer=mixer,
                                peer_rule=peer_rule,
                                training_backend=training_backend,
                                warmup_runs=args.warmup_runs,
                                repeats=args.repeats,
                                run_output_dir=args.run_output_dir,
                                collect_stage_timing=not args.no_stage_timing,
                            )
                            rows.append(row)
                            stage_rows.extend(case_stage_rows)
    annotate_loop_comparisons(rows)
    write_rows(args.output, rows)
    stage_path = stage_output_path(args.output, args.stage_output)
    write_stage_rows(stage_path, stage_rows)
    print(args.output)
    print(stage_path)
    if args.require_backend_parity:
        failures = backend_parity_failures(rows)
        if failures:
            for row in failures:
                print(
                    "Backend parity failed for "
                    f"{row['toy']} {row['scenario']} {row['device']} "
                    f"{row['training_backend']}",
                    file=sys.stderr,
                )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
