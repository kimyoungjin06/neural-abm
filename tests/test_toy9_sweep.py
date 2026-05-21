from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from neural_abm.config import load_toy9_config
from neural_abm.toy_heterogeneous import run_toy9
from test_toy9_runner import tiny_config_dict


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_toy9_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_toy9_sweep", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy9_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy9_sweep
SPEC.loader.exec_module(run_toy9_sweep)


def load_base_config(tmp_path: Path) -> dict:
    return tiny_config_dict(tmp_path, mixer="output_average")


def test_toy9_sweep_writes_group_gate_config(tmp_path: Path) -> None:
    path = run_toy9_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy9_sweep_unit",
        threshold_group_fraction=0.75,
        coordination_gate_mode="all_enabled",
        environment_threshold=0.6,
        benefit=1.4,
        action_cost=0.25,
        payoff_learning_rate=0.3,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.5,
        seed=3,
        epochs=1,
        config_dir=tmp_path / "configs",
        coordination_threshold=0.8,
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_toy9_config(path)
    groups = raw["model"]["agents"]["groups"]

    assert groups[0]["fraction"] == pytest.approx(0.75)
    assert groups[1]["fraction"] == pytest.approx(0.25)
    assert groups[0]["coordination_enabled"] is True
    assert groups[1]["coordination_enabled"] is True
    assert groups[0]["threshold"] == pytest.approx(0.6)
    assert groups[1]["learning_rate"] == pytest.approx(0.3)
    assert config.environment.benefit == pytest.approx(1.4)
    assert config.environment.action_cost == pytest.approx(0.25)
    assert config.coordination.alpha == pytest.approx(0.5)
    assert config.coordination.threshold == pytest.approx(0.8)


def test_toy9_sweep_summary_includes_required_metrics(tmp_path: Path) -> None:
    row = run_toy9_sweep.result_row(
        label="toy9_sweep_unit",
        threshold_group_fraction=0.5,
        coordination_gate_mode="gated",
        environment_threshold=0.45,
        benefit=1.2,
        action_cost=0.35,
        payoff_learning_rate=0.18,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        coordination_threshold=0.0,
        seed=1,
        epochs=1,
        run_dir=tmp_path / "run",
        domain_final_action_rate=0.4,
        domain_final_mean_action_probability=0.45,
        domain_final_mean_payoff=0.1,
        domain_final_payoff_variance=0.02,
        domain_final_group_action_rate_gap=0.2,
        domain_final_coordination_enabled_action_rate=0.5,
        domain_final_coordination_disabled_action_rate=0.3,
        final_fragmentation_components=2,
    )
    summary_path = tmp_path / "summary.csv"
    run_toy9_sweep.write_summary_csv(summary_path, [row])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)
    grouped = run_toy9_sweep.build_grouped_summary([row])

    assert "domain_final_group_action_rate_gap" in reader.fieldnames
    assert "coordination_gate_mode" in reader.fieldnames
    assert float(written["domain_final_action_rate"]) == pytest.approx(0.4)
    assert written["coordination_gate_mode"] == "gated"
    assert "final_action_rate_mean" in grouped.columns
    assert "final_group_action_rate_gap_mean" in grouped.columns
    assert "final_fragmentation_components_mean" in grouped.columns


def test_toy9_sweep_generated_config_runs(tmp_path: Path) -> None:
    path = run_toy9_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy9_sweep_smoke",
        threshold_group_fraction=0.5,
        coordination_gate_mode="gated",
        environment_threshold=0.45,
        benefit=1.2,
        action_cost=0.35,
        payoff_learning_rate=0.18,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        seed=1,
        epochs=1,
        config_dir=tmp_path / "configs",
    )

    result = run_toy9(load_toy9_config(path), path)

    assert result.run_dir.exists()
    assert 0.0 <= result.domain_metrics["domain_final_action_rate"] <= 1.0
