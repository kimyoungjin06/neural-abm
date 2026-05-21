from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from neural_abm.config import load_toy7_config
from neural_abm.toy_resource import run_toy7
from test_toy7_runner import tiny_config_dict


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_toy7_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_toy7_sweep", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy7_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy7_sweep
SPEC.loader.exec_module(run_toy7_sweep)


def load_base_config(tmp_path: Path) -> dict:
    return tiny_config_dict(tmp_path, mixer="output_average")


def test_toy7_sweep_writes_resource_parameter_config(tmp_path: Path) -> None:
    path = run_toy7_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy7_sweep_unit",
        recovery_rate=0.1,
        extraction_cost=0.5,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.5,
        seed=3,
        epochs=1,
        config_dir=tmp_path / "configs",
        coordination_threshold=0.8,
        initial_intensity_mean=0.45,
        initial_intensity_std=0.2,
        exploration_std=0.08,
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_toy7_config(path)

    assert raw["domain"]["environment"]["resource_recovery_rate"] == pytest.approx(0.1)
    assert raw["domain"]["environment"]["extraction_cost"] == pytest.approx(0.5)
    assert config.coordination.mixer == "output_average"
    assert config.coordination.peer_rule == "output_similarity"
    assert config.coordination.alpha == pytest.approx(0.5)
    assert config.coordination.threshold == pytest.approx(0.8)
    assert config.environment.initial_intensity_mean == pytest.approx(0.45)
    assert config.environment.initial_intensity_std == pytest.approx(0.2)
    assert config.policy.exploration_std == pytest.approx(0.08)
    assert config.run.seed == 3
    assert config.simulation.epochs == 1


def test_toy7_sweep_summary_includes_required_metrics(tmp_path: Path) -> None:
    row = run_toy7_sweep.result_row(
        label="toy7_sweep_unit",
        recovery_rate=0.05,
        extraction_cost=0.35,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        coordination_threshold=0.8,
        initial_intensity_mean=0.45,
        initial_intensity_std=0.2,
        exploration_std=0.08,
        seed=1,
        epochs=1,
        run_dir=tmp_path / "run",
        domain_final_resource_fraction=0.7,
        domain_final_resource_level=70.0,
        domain_final_mean_intensity=0.4,
        domain_final_intensity_variance=0.01,
        domain_final_mean_payoff=0.2,
        final_fragmentation_components=2,
    )
    summary_path = tmp_path / "summary.csv"
    run_toy7_sweep.write_summary_csv(summary_path, [row])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)

    grouped = run_toy7_sweep.build_grouped_summary([row])

    assert "domain_final_resource_fraction" in reader.fieldnames
    assert "domain_final_mean_intensity" in reader.fieldnames
    assert "coordination_threshold" in reader.fieldnames
    assert "initial_intensity_std" in reader.fieldnames
    assert "exploration_std" in reader.fieldnames
    assert float(written["domain_final_resource_fraction"]) == pytest.approx(0.7)
    assert float(written["domain_final_mean_intensity"]) == pytest.approx(0.4)
    assert float(written["coordination_threshold"]) == pytest.approx(0.8)
    assert float(written["initial_intensity_std"]) == pytest.approx(0.2)
    assert float(written["exploration_std"]) == pytest.approx(0.08)
    assert "final_resource_fraction_mean" in grouped.columns
    assert "final_mean_intensity_mean" in grouped.columns
    assert "final_mean_payoff_mean" in grouped.columns
    assert "coordination_threshold" in grouped.columns
    assert "initial_intensity_std" in grouped.columns


def test_toy7_sweep_generated_config_runs(tmp_path: Path) -> None:
    path = run_toy7_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy7_sweep_smoke",
        recovery_rate=0.05,
        extraction_cost=0.35,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        seed=1,
        epochs=1,
        config_dir=tmp_path / "configs",
    )

    result = run_toy7(load_toy7_config(path), path)

    assert result.run_dir.exists()
    assert 0.0 <= result.domain_metrics["domain_final_resource_fraction"] <= 1.0
