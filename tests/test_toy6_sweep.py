from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from neural_abm.config import load_toy6_config
from neural_abm.toy_categorical import run_toy6
from test_toy6_runner import tiny_config_dict


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_toy6_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_toy6_sweep", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy6_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy6_sweep
SPEC.loader.exec_module(run_toy6_sweep)


def load_base_config(tmp_path: Path) -> dict:
    return tiny_config_dict(tmp_path, mixer="output_average")


def test_toy6_sweep_writes_strategy_distribution_config(tmp_path: Path) -> None:
    path = run_toy6_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy6_sweep_unit",
        strategy_count=4,
        initial_distribution_label="biased",
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.5,
        seed=3,
        epochs=1,
        config_dir=tmp_path / "configs",
        coordination_threshold=0.8,
        payoff_profile="win_bonus",
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_toy6_config(path)

    assert raw["domain"]["game"]["strategy_count"] == 4
    assert raw["domain"]["environment"]["initial_strategy_probabilities"] == (
        pytest.approx([0.6, 0.4 / 3.0, 0.4 / 3.0, 0.4 / 3.0])
    )
    assert config.coordination.mixer == "output_average"
    assert config.coordination.peer_rule == "output_similarity"
    assert config.coordination.alpha == pytest.approx(0.5)
    assert config.coordination.threshold == pytest.approx(0.8)
    assert config.game.win_payoff == pytest.approx(1.5)
    assert config.game.loss_payoff == pytest.approx(-1.0)
    assert config.run.seed == 3
    assert config.simulation.epochs == 1


def test_toy6_sweep_summary_includes_required_metrics(tmp_path: Path) -> None:
    row = run_toy6_sweep.result_row(
        label="toy6_sweep_unit",
        strategy_count=3,
        initial_distribution_label="balanced",
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        coordination_threshold=0.8,
        payoff_profile="loss_heavy",
        win_payoff=1.0,
        loss_payoff=-1.5,
        draw_payoff=0.0,
        seed=1,
        epochs=1,
        run_dir=tmp_path / "run",
        domain_final_mean_payoff=0.2,
        domain_final_strategy_entropy=0.9,
        domain_final_dominant_strategy=1,
        domain_final_dominant_strategy_fraction=0.5,
        final_fragmentation_components=2,
    )
    summary_path = tmp_path / "summary.csv"
    run_toy6_sweep.write_summary_csv(summary_path, [row])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)

    grouped = run_toy6_sweep.build_grouped_summary([row])

    assert "domain_final_strategy_entropy" in reader.fieldnames
    assert "domain_final_dominant_strategy_fraction" in reader.fieldnames
    assert "coordination_threshold" in reader.fieldnames
    assert "payoff_profile" in reader.fieldnames
    assert float(written["domain_final_strategy_entropy"]) == pytest.approx(0.9)
    assert int(written["domain_final_dominant_strategy"]) == 1
    assert written["payoff_profile"] == "loss_heavy"
    assert float(written["coordination_threshold"]) == pytest.approx(0.8)
    assert "final_strategy_entropy_mean" in grouped.columns
    assert "final_dominant_strategy_fraction_mean" in grouped.columns
    assert "final_fragmentation_components_mean" in grouped.columns
    assert "coordination_threshold" in grouped.columns
    assert "payoff_profile" in grouped.columns


def test_toy6_sweep_generated_config_runs(tmp_path: Path) -> None:
    path = run_toy6_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy6_sweep_smoke",
        strategy_count=3,
        initial_distribution_label="balanced",
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        seed=1,
        epochs=1,
        config_dir=tmp_path / "configs",
    )

    result = run_toy6(load_toy6_config(path), path)

    assert result.run_dir.exists()
    assert 0.0 <= result.domain_metrics["domain_final_strategy_entropy"] <= 1.0
