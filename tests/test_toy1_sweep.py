from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from neural_abm.config import load_toy1_config
from neural_abm.toy_classification import run_toy1
from test_toy1_runner import write_tiny_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_toy1_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_toy1_sweep", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy1_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy1_sweep
SPEC.loader.exec_module(run_toy1_sweep)


def load_base_config(tmp_path: Path) -> dict:
    path = write_tiny_config(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_toy1_sweep_parse_args_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_toy1_sweep.py"])

    args = run_toy1_sweep.parse_args()

    assert args.base_config == Path("experiments/configs/toy1_neural_hk_baseline.yaml")
    assert args.label is None
    assert args.seeds == [1]
    assert args.alphas == [0.0, 0.1, 0.25, 0.5]
    assert args.thresholds == [0.6, 0.8, 0.95]
    assert args.epochs is None
    assert args.cases is None
    assert args.config_dir == Path("experiments/configs/generated")
    assert args.results_dir == Path("experiments/results")


def test_toy1_sweep_writes_case_config_with_legacy_signature(
    tmp_path: Path,
) -> None:
    case = run_toy1_sweep.SweepCase(
        mixer="parameter_average",
        peer_rule="state_similarity",
        init_mode="independent_init",
    )
    path = run_toy1_sweep.write_case_config(
        load_base_config(tmp_path),
        case,
        3,
        0.5,
        0.8,
        "toy1_sweep_unit",
        tmp_path / "configs",
        epochs=1,
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_toy1_config(path)

    assert raw["run"]["name"] == (
        "toy1_sweep_unit_parameter_average_state_similarity_independent_init_"
        "a0p5_t0p8"
    )
    assert raw["simulation"]["epochs"] == 1
    assert config.run.seed == 3
    assert config.agents.init_mode == "independent_init"
    assert config.coordination.mixer == "parameter_average"
    assert config.coordination.peer_rule == "state_similarity"
    assert config.coordination.alpha == pytest.approx(0.5)
    assert config.coordination.threshold == pytest.approx(0.8)


def test_toy1_sweep_summary_includes_required_metrics(tmp_path: Path) -> None:
    row = run_toy1_sweep.result_row(
        label="toy1_sweep_unit",
        case="output_average_output_similarity_same_init",
        seed=1,
        mixer="output_average",
        peer_rule="output_similarity",
        model_init_mode="same_init",
        alpha=0.25,
        coordination_threshold=0.8,
        run_dir=tmp_path / "run",
        domain_final_mean_global_accuracy=0.7,
        domain_final_mean_consensus=0.9,
        final_fragmentation_components=2,
    )
    summary_path = tmp_path / "summary.csv"
    run_toy1_sweep.write_summary_csv(summary_path, [row])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)

    grouped = run_toy1_sweep.build_grouped_summary([row])

    assert "domain_final_mean_global_accuracy" in reader.fieldnames
    assert "threshold" in reader.fieldnames
    assert written["case"] == "output_average_output_similarity_same_init"
    assert float(written["threshold"]) == pytest.approx(0.8)
    assert float(written["domain_final_mean_consensus"]) == pytest.approx(0.9)
    assert "accuracy_mean" in grouped.columns
    assert "consensus_mean" in grouped.columns
    assert "fragmentation_mean" in grouped.columns


def test_toy1_sweep_generated_config_runs(tmp_path: Path) -> None:
    path = run_toy1_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy1_sweep_smoke",
        case="output_average_output_similarity_same_init",
        mixer="output_average",
        peer_rule="output_similarity",
        init_mode="same_init",
        alpha=0.25,
        coordination_threshold=0.0,
        seed=1,
        epochs=1,
        config_dir=tmp_path / "configs",
    )

    result = run_toy1(load_toy1_config(path), path)

    assert result.run_dir.exists()
    assert 0.0 <= result.domain_metrics["domain_final_mean_global_accuracy"] <= 1.0


def test_toy1_sweep_main_smoke_writes_summary_grouped_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_path = write_tiny_config(
        tmp_path,
        mixer="output_average",
        peer_rule="output_similarity",
    )
    config_dir = tmp_path / "configs"
    results_dir = tmp_path / "results"
    label = "toy1_main_smoke"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_toy1_sweep.py",
            "--base-config",
            str(base_path),
            "--label",
            label,
            "--seeds",
            "1",
            "--epochs",
            "1",
            "--cases",
            "output_average_output_similarity_same_init",
            "--alphas",
            "0.0",
            "--thresholds",
            "0.6",
            "--config-dir",
            str(config_dir),
            "--results-dir",
            str(results_dir),
        ],
    )

    run_toy1_sweep.main()

    assert (results_dir / f"{label}_summary.csv").exists()
    assert (results_dir / f"{label}_grouped_summary.csv").exists()
    assert (results_dir / f"{label}_grouped_summary.md").exists()
    assert len(list((config_dir / label).glob("*.yaml"))) == 1
