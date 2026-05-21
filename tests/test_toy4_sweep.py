from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from neural_abm.config import load_toy4_config


ROOT = Path(__file__).resolve().parents[1]


def import_script(name: str, relative_path: str):
    script_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_toy4_sweep = import_script("run_toy4_sweep", "scripts/run_toy4_sweep.py")
plot_toy4_sweep = import_script("plot_toy4_sweep", "scripts/plot_toy4_sweep.py")


def load_base_config(tmp_path: Path) -> dict:
    base_path = ROOT / "experiments/configs/toy4_public_goods_baseline.yaml"
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    raw["run"]["output_dir"] = str(tmp_path / "runs")
    return raw


def test_toy4_sweep_parse_args_preserves_legacy_common_defaults(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_toy4_sweep.py"])

    args = run_toy4_sweep.parse_args()

    assert args.base_config == Path("experiments/configs/toy4_public_goods_baseline.yaml")
    assert args.label == "toy4_reputation_sweep_seeds01_03"
    assert args.seeds == [1, 2, 3]
    assert args.epochs == 50
    assert args.mixers == ["none", "output_average"]
    assert args.peer_rules is None
    assert args.alpha == 0.25
    assert args.alphas is None
    assert args.thresholds == [0.0]
    assert args.config_dir == Path("experiments/configs/generated")
    assert args.results_dir == Path("experiments/results")


def test_toy4_sweep_writes_reputation_mobility_config(tmp_path: Path) -> None:
    args = SimpleNamespace(
        label="toy4_sweep_unit",
        epochs=2,
        alpha=0.25,
        revision_rate=1.0,
        resource_enabled=False,
        reputation_decay=0.8,
        reputation_temperature=1.5,
        reputation_noise=0.1,
        reputation_observation_mode="self_neighbor_mean",
        mobility_rate=0.5,
        mobility_move_cost=0.05,
        config_dir=tmp_path / "configs",
    )

    path = run_toy4_sweep.write_case_config(
        base=load_base_config(tmp_path),
        args=args,
        update_rule="neural_policy",
        mixer="output_average",
        seed=7,
        mobility_enabled=True,
    )
    config = load_toy4_config(path)

    assert config.policy.rule == "neural_policy"
    assert config.coordination.mixer == "output_average"
    assert config.state.reputation.enabled is True
    assert config.state.reputation.decay == 0.8
    assert config.state.reputation.observation_mode == "self_neighbor_mean"
    assert config.agents.model.input_dim == 8
    assert config.state.mobility.enabled is True
    assert config.state.mobility.rate == 0.5


def test_toy4_sweep_writes_output_similarity_threshold_config(
    tmp_path: Path,
) -> None:
    path = run_toy4_sweep.write_toy4_case_config(
        base=load_base_config(tmp_path),
        label="toy4_sweep_unit",
        update_rule="neural_policy",
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.5,
        seed=2,
        epochs=1,
        config_dir=tmp_path / "configs",
        mobility_enabled=False,
        revision_rate=1.0,
        resource_enabled=False,
        reputation_decay=0.9,
        reputation_temperature=1.0,
        reputation_noise=0.0,
        reputation_observation_mode="none",
        mobility_rate=0.25,
        mobility_move_cost=0.0,
        coordination_threshold=0.3,
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_toy4_config(path)

    assert raw["model"]["coordination"]["peer_rule"] == "output_similarity"
    assert raw["model"]["coordination"]["alpha"] == 0.5
    assert raw["model"]["coordination"]["threshold"] == 0.3
    assert config.coordination.peer_rule == "output_similarity"
    assert config.coordination.alpha == 0.5
    assert config.coordination.threshold == 0.3


def test_toy4_sweep_summary_includes_reputation_mobility_metrics(
    tmp_path: Path,
) -> None:
    row = run_toy4_sweep.result_row(
        label="toy4_sweep_unit",
        update_rule="neural_policy",
        mixer="output_average",
        peer_rule="output_similarity",
        seed=1,
        epochs=2,
        alpha=0.25,
        policy_revision_rate=1.0,
        resource_enabled=False,
        reputation_decay=0.8,
        reputation_temperature=1.5,
        reputation_noise=0.1,
        reputation_observation_mode="self_neighbor_mean",
        mobility_enabled=True,
        mobility_rate=0.5,
        mobility_move_cost=0.05,
        run_dir=tmp_path / "run",
        final_action_rate=0.4,
        final_mean_payoff=0.2,
        domain_payoff_gini=0.1,
        domain_resource_level=100.0,
        domain_collapse_time="",
        final_mean_reputation=0.6,
        final_reputation_dispersion=0.05,
        final_mobility_rate=0.25,
        final_mean_mobility_gain=0.02,
    )
    summary_path = tmp_path / "summary.csv"
    run_toy4_sweep.write_summary_csv(summary_path, [row])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)

    grouped = run_toy4_sweep.build_grouped_summary([row])

    assert "final_mean_reputation" in reader.fieldnames
    assert "final_mobility_rate" in reader.fieldnames
    assert "coordination_threshold" in reader.fieldnames
    assert written["reputation_observation_mode"] == "self_neighbor_mean"
    assert float(written["final_mobility_rate"]) == 0.25
    assert "reputation_mean" in grouped.columns
    assert "mobility_gain_mean" in grouped.columns


def test_toy4_sweep_plot_accepts_new_metrics(tmp_path: Path) -> None:
    rows = []
    for update_rule, mixer in plot_toy4_sweep.CONDITION_ORDER:
        rows.append(
            {
                "policy_rule": update_rule,
                "coordination_mixer": mixer,
                "mobility_enabled": False,
                "final_action_rate": 0.4,
                "final_mean_reputation": 0.5,
                "final_mobility_rate": 0.0,
                "final_mean_mobility_gain": 0.0,
            }
        )
    path = plot_toy4_sweep.plot_summary(pd.DataFrame(rows), tmp_path)

    assert path.exists()
    assert path.name == "toy4_reputation_mobility_sweep.png"
