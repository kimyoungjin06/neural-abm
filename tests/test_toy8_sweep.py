from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from neural_abm.config import load_toy8_config
from neural_abm.toy_async import run_toy8
from test_toy8_runner import tiny_config_dict


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_toy8_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_toy8_sweep", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy8_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy8_sweep
SPEC.loader.exec_module(run_toy8_sweep)


def load_base_config(tmp_path: Path) -> dict:
    return tiny_config_dict(tmp_path, mixer="output_average")


def test_toy8_sweep_writes_event_parameter_config(tmp_path: Path) -> None:
    path = run_toy8_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy8_sweep_unit",
        initial_active_fraction=0.2,
        initial_failed_fraction=0.1,
        base_activation_rate=0.04,
        peer_activation_rate=0.5,
        failure_rate=0.07,
        overload_failure_rate=0.2,
        recovery_rate=0.03,
        max_time=25.0,
        graph_k=4,
        graph_rewire_probability=0.2,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.5,
        seed=3,
        epochs=3,
        config_dir=tmp_path / "configs",
        coordination_threshold=0.8,
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_toy8_config(path)

    assert raw["domain"]["environment"]["initial_active_fraction"] == pytest.approx(0.2)
    assert raw["domain"]["environment"]["initial_failed_fraction"] == pytest.approx(0.1)
    assert config.environment.base_activation_rate == pytest.approx(0.04)
    assert config.environment.peer_activation_rate == pytest.approx(0.5)
    assert config.environment.failure_rate == pytest.approx(0.07)
    assert config.environment.overload_failure_rate == pytest.approx(0.2)
    assert config.environment.recovery_rate == pytest.approx(0.03)
    assert config.environment.max_time == pytest.approx(25.0)
    assert config.graph.k == 4
    assert config.graph.rewire_probability == pytest.approx(0.2)
    assert config.coordination.alpha == pytest.approx(0.5)
    assert config.coordination.threshold == pytest.approx(0.8)


def test_toy8_sweep_summary_includes_required_metrics(tmp_path: Path) -> None:
    row = run_toy8_sweep.result_row(
        label="toy8_sweep_unit",
        initial_active_fraction=0.1,
        initial_failed_fraction=0.0,
        base_activation_rate=0.02,
        peer_activation_rate=0.3,
        failure_rate=0.03,
        overload_failure_rate=0.08,
        recovery_rate=0.01,
        max_time=50.0,
        graph_k=4,
        graph_rewire_probability=0.1,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        coordination_threshold=0.8,
        seed=1,
        epochs=3,
        run_dir=tmp_path / "run",
        domain_final_time=2.0,
        domain_final_inactive_fraction=0.5,
        domain_final_active_fraction=0.3,
        domain_final_failed_fraction=0.2,
        domain_total_events=3,
        domain_activation_events=2,
        domain_failure_events=1,
        domain_recovery_events=0,
        domain_absorbed=False,
        final_fragmentation_components=2,
    )
    summary_path = tmp_path / "summary.csv"
    run_toy8_sweep.write_summary_csv(summary_path, [row])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)
    grouped = run_toy8_sweep.build_grouped_summary([row])

    assert "domain_final_active_fraction" in reader.fieldnames
    assert "domain_total_events" in reader.fieldnames
    assert "absorbed_rate" in grouped.columns
    assert float(written["domain_final_active_fraction"]) == pytest.approx(0.3)
    assert int(written["domain_total_events"]) == 3
    assert grouped["total_events_mean"].iloc[0] == pytest.approx(3.0)


def test_toy8_sweep_generated_config_runs(tmp_path: Path) -> None:
    path = run_toy8_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy8_sweep_smoke",
        initial_active_fraction=0.25,
        initial_failed_fraction=0.125,
        base_activation_rate=0.05,
        peer_activation_rate=0.4,
        failure_rate=0.03,
        overload_failure_rate=0.1,
        recovery_rate=0.02,
        max_time=30.0,
        graph_k=2,
        graph_rewire_probability=0.0,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        seed=1,
        epochs=2,
        config_dir=tmp_path / "configs",
    )

    result = run_toy8(load_toy8_config(path), path)

    assert result.run_dir.exists()
    assert 0.0 <= result.domain_metrics["domain_final_active_fraction"] <= 1.0
    assert result.domain_metrics["domain_total_events"] <= 2
