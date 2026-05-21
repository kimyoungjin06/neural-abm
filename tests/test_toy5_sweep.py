from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from neural_abm.config import load_toy5_config
from neural_abm.toy_contagion import run_toy5
from test_toy5_runner import tiny_config_dict


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_toy5_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_toy5_sweep", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy5_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy5_sweep
SPEC.loader.exec_module(run_toy5_sweep)


def load_base_config(tmp_path: Path) -> dict:
    return tiny_config_dict(tmp_path, update_rule="neural_policy", mixer="output_average")


def test_toy5_sweep_writes_threshold_adoption_config(tmp_path: Path) -> None:
    path = run_toy5_sweep.write_case_config(
        base=load_base_config(tmp_path),
        label="toy5_sweep_unit",
        update_rule="neural_policy",
        initial_action_fraction=0.1,
        threshold_mode="heterogeneous",
        homogeneous_threshold=0.4,
        heterogeneous_threshold_low=0.2,
        heterogeneous_threshold_high=0.7,
        simple_contagion_probability=0.15,
        repeated_exposure_decay=0.3,
        adoption_is_absorbing=False,
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.5,
        seed=3,
        epochs=1,
        config_dir=tmp_path / "configs",
        coordination_threshold=0.8,
        revision_rate=0.9,
        neural_update_backend="batched",
        reputation_decay=0.8,
        reputation_temperature=1.2,
        reputation_noise=0.1,
        reputation_observation_mode="self_neighbor_mean",
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_toy5_config(path)

    assert raw["domain"]["environment"]["initial_action_fraction"] == pytest.approx(
        0.1
    )
    assert raw["domain"]["environment"]["threshold_mode"] == "heterogeneous"
    assert raw["domain"]["environment"]["heterogeneous_threshold_high"] == (
        pytest.approx(0.7)
    )
    assert config.policy.rule == "neural_policy"
    assert config.policy.domain.repeated_exposure_decay == pytest.approx(0.3)
    assert config.policy.domain.adoption_is_absorbing is False
    assert config.policy.neural_update_backend == "batched"
    assert config.coordination.mixer == "output_average"
    assert config.coordination.peer_rule == "output_similarity"
    assert config.coordination.alpha == pytest.approx(0.5)
    assert config.coordination.threshold == pytest.approx(0.8)
    assert config.state.reputation.decay == pytest.approx(0.8)
    assert config.state.reputation.observation_mode == "self_neighbor_mean"
    assert config.agents.model.input_dim == 8
    assert config.run.seed == 3
    assert config.simulation.epochs == 1


def test_toy5_sweep_summary_includes_required_metrics(tmp_path: Path) -> None:
    row = run_toy5_sweep.result_row(
        label="toy5_sweep_unit",
        update_rule="complex_threshold",
        mixer="output_average",
        peer_rule="output_similarity",
        alpha=0.25,
        coordination_threshold=0.8,
        seed=1,
        epochs=1,
        initial_action_fraction=0.05,
        threshold_mode="homogeneous",
        homogeneous_threshold=0.25,
        heterogeneous_threshold_low=0.15,
        heterogeneous_threshold_high=0.55,
        simple_contagion_probability=0.08,
        policy_revision_rate=1.0,
        repeated_exposure_decay=0.0,
        adoption_is_absorbing=True,
        neural_update_backend="loop",
        reputation_decay=0.9,
        reputation_temperature=1.0,
        reputation_noise=0.0,
        reputation_observation_mode="none",
        run_dir=tmp_path / "run",
        final_action_rate=0.6,
        final_mean_payoff=0.2,
        final_mean_policy_action_probability=0.55,
        final_mean_reputation=0.5,
        final_reputation_dispersion=0.1,
        domain_cascade_size=6,
        domain_time_to_50_action=1,
        domain_failed_cascade=False,
        domain_mean_neighbor_action_rate=0.4,
        domain_mean_repeated_exposure_count=2.0,
        domain_low_threshold_action_rate="",
        domain_high_threshold_action_rate="",
        final_fragmentation_components=1,
    )
    summary_path = tmp_path / "summary.csv"
    run_toy5_sweep.write_summary_csv(summary_path, [row])

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = next(reader)

    grouped = run_toy5_sweep.build_grouped_summary([row])

    assert "domain_cascade_size" in reader.fieldnames
    assert "domain_failed_cascade" in reader.fieldnames
    assert "final_mean_policy_action_probability" in reader.fieldnames
    assert int(written["domain_cascade_size"]) == 6
    assert written["adoption_is_absorbing"] == "True"
    assert float(written["final_action_rate"]) == pytest.approx(0.6)
    assert "final_action_rate_mean" in grouped.columns
    assert "cascade_size_mean" in grouped.columns
    assert "final_fragmentation_components_mean" in grouped.columns


def test_toy5_sweep_generated_config_runs(tmp_path: Path) -> None:
    path = run_toy5_sweep.write_case_config(
        base=tiny_config_dict(tmp_path, update_rule="complex_threshold"),
        label="toy5_sweep_smoke",
        update_rule="complex_threshold",
        initial_action_fraction=1.0 / 6.0,
        threshold_mode="homogeneous",
        homogeneous_threshold=0.5,
        heterogeneous_threshold_low=0.25,
        heterogeneous_threshold_high=0.75,
        simple_contagion_probability=1.0,
        repeated_exposure_decay=0.0,
        adoption_is_absorbing=True,
        mixer="none",
        peer_rule="none",
        alpha=0.0,
        seed=1,
        epochs=1,
        config_dir=tmp_path / "configs",
    )

    result = run_toy5(load_toy5_config(path), path)

    assert result.run_dir.exists()
    assert result.domain_metrics["domain_cascade_size"] >= 1
