from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from neural_abm.config import load_toy10_config
from neural_abm.toy_market import run_toy10
from test_toy10_runner import tiny_config_dict


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_toy10_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_toy10_sweep", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy10_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy10_sweep
SPEC.loader.exec_module(run_toy10_sweep)


def load_base_config(tmp_path: Path) -> dict:
    return tiny_config_dict(tmp_path, mixer="output_average")


def test_toy10_sweep_writes_market_network_config(tmp_path: Path) -> None:
    path = run_toy10_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy10_sweep_unit",
        recovery_rate=0.1,
        extraction_cost=0.45,
        dynamic_rewire_rate=0.2,
        initial_price_expectation_mean=0.65,
        initial_conservation_norm_mean=0.5,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.5,
        seed=3,
        epochs=1,
        config_dir=tmp_path / "configs",
        social_harvest_gain=1.5,
        social_disagreement_penalty=0.7,
        conservation_harvest_weight=0.6,
        coordination_threshold=0.8,
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_toy10_config(path)

    assert raw["domain"]["environment"]["resource_recovery_rate"] == pytest.approx(0.1)
    assert raw["domain"]["environment"]["extraction_cost"] == pytest.approx(0.45)
    assert raw["domain"]["network"]["dynamic_rewire_rate"] == pytest.approx(0.2)
    assert config.environment.initial_price_expectation_mean == pytest.approx(0.65)
    assert config.environment.initial_conservation_norm_mean == pytest.approx(0.5)
    assert config.policy.social_harvest_gain == pytest.approx(1.5)
    assert config.policy.social_disagreement_penalty == pytest.approx(0.7)
    assert config.policy.conservation_harvest_weight == pytest.approx(0.6)
    assert config.coordination.alpha == pytest.approx(0.5)
    assert config.coordination.threshold == pytest.approx(0.8)


def test_toy10_sweep_summary_includes_required_metrics(tmp_path: Path) -> None:
    row = run_toy10_sweep.result_row(
        label="toy10_sweep_unit",
        recovery_rate=0.05,
        extraction_cost=0.3,
        dynamic_rewire_rate=0.05,
        initial_price_expectation_mean=0.5,
        initial_conservation_norm_mean=0.35,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        coordination_threshold=0.0,
        seed=1,
        epochs=1,
        run_dir=tmp_path / "run",
        domain_final_resource_fraction=0.7,
        domain_final_resource_level=70.0,
        domain_final_market_price=0.6,
        domain_final_market_imbalance=0.1,
        domain_final_mean_harvest_intensity=0.4,
        domain_final_mean_price_expectation=0.55,
        domain_final_mean_conservation_norm=0.35,
        domain_final_mean_payoff=0.2,
        domain_cumulative_rewired_edge_count=4,
        final_fragmentation_components=2,
        social_harvest_gain=1.5,
        social_disagreement_penalty=0.7,
        conservation_harvest_weight=0.6,
    )
    summary_path = tmp_path / "summary.csv"
    run_toy10_sweep.write_summary_csv(summary_path, [row])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)
    grouped = run_toy10_sweep.build_grouped_summary([row])

    assert "domain_final_resource_fraction" in reader.fieldnames
    assert "social_harvest_gain" in reader.fieldnames
    assert "social_disagreement_penalty" in reader.fieldnames
    assert "domain_cumulative_rewired_edge_count" in reader.fieldnames
    assert float(written["social_harvest_gain"]) == pytest.approx(1.5)
    assert float(written["social_disagreement_penalty"]) == pytest.approx(0.7)
    assert float(written["conservation_harvest_weight"]) == pytest.approx(0.6)
    assert float(written["domain_final_market_price"]) == pytest.approx(0.6)
    assert float(written["domain_cumulative_rewired_edge_count"]) == pytest.approx(4)
    assert "final_resource_fraction_mean" in grouped.columns
    assert "final_market_price_mean" in grouped.columns
    assert "cumulative_rewired_edge_count_mean" in grouped.columns


def test_toy10_sweep_generated_config_runs(tmp_path: Path) -> None:
    path = run_toy10_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy10_sweep_smoke",
        recovery_rate=0.05,
        extraction_cost=0.3,
        dynamic_rewire_rate=0.05,
        initial_price_expectation_mean=0.5,
        initial_conservation_norm_mean=0.35,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        seed=1,
        epochs=1,
        config_dir=tmp_path / "configs",
    )

    result = run_toy10(load_toy10_config(path), path)

    assert result.run_dir.exists()
    assert 0.0 <= result.domain_metrics["domain_final_resource_fraction"] <= 1.0
