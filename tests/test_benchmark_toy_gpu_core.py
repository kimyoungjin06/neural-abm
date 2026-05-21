from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark_toy_gpu_core.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_toy_gpu_core", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
benchmark_toy_gpu_core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_toy_gpu_core
SPEC.loader.exec_module(benchmark_toy_gpu_core)


def test_benchmark_repeat_csv_fields_are_declared() -> None:
    assert "warmup_runs" in benchmark_toy_gpu_core.CSV_FIELDS
    assert "repeats" in benchmark_toy_gpu_core.CSV_FIELDS
    assert "resolved_training_backend" in benchmark_toy_gpu_core.CSV_FIELDS
    assert "seconds_std" in benchmark_toy_gpu_core.CSV_FIELDS
    assert "agent_steps_per_second_std" in benchmark_toy_gpu_core.CSV_FIELDS
    assert (
        "timing_local_training_seconds_std"
        in benchmark_toy_gpu_core.CSV_FIELDS
    )
    assert (
        "timing_local_trainable_parameters_seconds"
        in benchmark_toy_gpu_core.CSV_FIELDS
    )
    assert (
        "timing_social_loss_update_seconds_std"
        in benchmark_toy_gpu_core.CSV_FIELDS
    )
    assert (
        "timing_local_optimizer_update_seconds"
        in benchmark_toy_gpu_core.CSV_FIELDS
    )
    assert (
        "timing_local_autograd_grad_seconds"
        in benchmark_toy_gpu_core.CSV_FIELDS
    )
    assert (
        "timing_social_adam_update_seconds_std"
        in benchmark_toy_gpu_core.CSV_FIELDS
    )
    assert "timing_social_mix_seconds" in benchmark_toy_gpu_core.CSV_FIELDS
    assert "repeat_index" in benchmark_toy_gpu_core.STAGE_CSV_FIELDS
    assert (
        "resolved_training_backend"
        in benchmark_toy_gpu_core.STAGE_CSV_FIELDS
    )
    assert "loop_speedup" in benchmark_toy_gpu_core.CSV_FIELDS
    assert "final_action_rate_diff_vs_loop" in benchmark_toy_gpu_core.CSV_FIELDS
    assert (
        "final_mean_policy_action_probability_diff_vs_loop"
        in benchmark_toy_gpu_core.CSV_FIELDS
    )
    assert "final_mean_reputation_diff_vs_loop" in benchmark_toy_gpu_core.CSV_FIELDS
    assert "parity_passed_vs_loop" in benchmark_toy_gpu_core.CSV_FIELDS


def test_benchmark_toy_configs_write_neural_update_backend() -> None:
    toy2 = benchmark_toy_gpu_core.toy2_config(
        seed=1,
        epochs=1,
        device=benchmark_toy_gpu_core.torch.device("cpu"),
        output_dir=Path("/tmp/out"),
        agent_count=64,
        mixer="none",
        training_backend="batched",
    )
    toy4 = benchmark_toy_gpu_core.toy4_config(
        seed=1,
        epochs=1,
        device=benchmark_toy_gpu_core.torch.device("cpu"),
        output_dir=Path("/tmp/out"),
        agent_count=64,
        mixer="none",
        training_backend="auto",
    )
    toy5 = benchmark_toy_gpu_core.toy5_config(
        seed=1,
        epochs=1,
        device=benchmark_toy_gpu_core.torch.device("cpu"),
        output_dir=Path("/tmp/out"),
        agent_count=64,
        mixer="none",
        training_backend="tensor_batched",
    )

    assert toy2["model"]["policy"]["neural_update_backend"] == "batched"
    assert toy4["model"]["policy"]["neural_update_backend"] == "auto"
    assert toy5["model"]["policy"]["neural_update_backend"] == "tensor_batched"


def test_benchmark_metric_helpers_report_sample_std() -> None:
    assert benchmark_toy_gpu_core.metric_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert benchmark_toy_gpu_core.metric_std([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert benchmark_toy_gpu_core.metric_std([2.0]) == pytest.approx(0.0)
    assert benchmark_toy_gpu_core.metric_mean([None]) == ""
    assert benchmark_toy_gpu_core.metric_std([None]) == ""


def test_benchmark_stage_summary_reports_repeat_mean_and_std() -> None:
    first = benchmark_toy_gpu_core.CaseMeasurement(
        seconds=1.0,
        result=SimpleNamespace(),
        timing_rows=[],
        stage_seconds={
            key: 0.0
            for key in (
                f"timing_{stage}_seconds"
                for stage in benchmark_toy_gpu_core.TIMING_STAGES
            )
        },
    )
    second = benchmark_toy_gpu_core.CaseMeasurement(
        seconds=2.0,
        result=SimpleNamespace(),
        timing_rows=[],
        stage_seconds={
            key: 0.0
            for key in (
                f"timing_{stage}_seconds"
                for stage in benchmark_toy_gpu_core.TIMING_STAGES
            )
        },
    )
    first.stage_seconds["timing_local_training_seconds"] = 1.0
    second.stage_seconds["timing_local_training_seconds"] = 3.0

    summary = benchmark_toy_gpu_core.stage_summary([first, second])

    assert summary["timing_local_training_seconds"] == pytest.approx(2.0)
    assert summary["timing_local_training_seconds_std"] == pytest.approx(2.0**0.5)


def test_benchmark_run_case_can_disable_stage_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collect_timing_values: list[bool] = []

    def fake_run_once(**kwargs: object) -> benchmark_toy_gpu_core.CaseMeasurement:
        collect_timing = bool(kwargs["collect_timing"])
        collect_timing_values.append(collect_timing)
        timing_rows = (
            [{"stage": "local_training", "seconds": 1.0}]
            if collect_timing
            else []
        )
        return benchmark_toy_gpu_core.CaseMeasurement(
            seconds=1.0,
            result=SimpleNamespace(
                final_action_rate=0.5,
                final_mean_policy_action_probability=0.5,
                final_mean_reputation=0.0,
            ),
            timing_rows=timing_rows,
            stage_seconds=benchmark_toy_gpu_core.timing_summary(timing_rows),
        )

    monkeypatch.setattr(benchmark_toy_gpu_core, "run_once", fake_run_once)

    row, stage_rows = benchmark_toy_gpu_core.run_case(
        toy="toy4",
        device=benchmark_toy_gpu_core.torch.device("cpu"),
        epochs=3,
        seed=7,
        agent_count=64,
        mixer="output_average",
        training_backend="batched",
        warmup_runs=1,
        repeats=2,
        run_output_dir=Path("/tmp/out"),
        collect_stage_timing=False,
    )

    assert collect_timing_values == [False, False, False]
    assert stage_rows == []
    assert row["seconds"] == pytest.approx(1.0)
    assert row["timing_local_training_seconds"] == pytest.approx(0.0)


def test_benchmark_run_case_resolves_toy2_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_once(**kwargs: object) -> benchmark_toy_gpu_core.CaseMeasurement:
        del kwargs
        return benchmark_toy_gpu_core.CaseMeasurement(
            seconds=1.0,
            result=SimpleNamespace(
                final_action_rate=0.5,
                final_mean_policy_action_probability=0.5,
                final_mean_reputation=0.0,
            ),
            timing_rows=[],
            stage_seconds=benchmark_toy_gpu_core.timing_summary([]),
        )

    monkeypatch.setattr(benchmark_toy_gpu_core, "run_once", fake_run_once)

    row, _ = benchmark_toy_gpu_core.run_case(
        toy="toy2",
        device=benchmark_toy_gpu_core.torch.device("cpu"),
        epochs=3,
        seed=7,
        agent_count=64,
        mixer="output_average",
        training_backend="batched",
        warmup_runs=0,
        repeats=1,
        run_output_dir=Path("/tmp/out"),
    )

    assert row["resolved_training_backend"] == "batched"


def test_benchmark_loop_comparison_annotation() -> None:
    rows = [
        _benchmark_row(
            training_backend="loop",
            seconds=4.0,
            final_action_rate=0.25,
            final_mean_policy_action_probability=0.5,
            final_mean_reputation=0.75,
        ),
        _benchmark_row(
            training_backend="batched",
            seconds=2.0,
            final_action_rate=0.25,
            final_mean_policy_action_probability=0.500001,
            final_mean_reputation=0.75,
        ),
    ]

    benchmark_toy_gpu_core.annotate_loop_comparisons(rows)

    assert rows[0]["loop_speedup"] == pytest.approx(1.0)
    assert rows[0]["parity_passed_vs_loop"] is True
    assert rows[1]["loop_speedup"] == pytest.approx(2.0)
    assert rows[1]["final_action_rate_diff_vs_loop"] == pytest.approx(0.0)
    assert rows[1]["final_mean_policy_action_probability_diff_vs_loop"] == (
        pytest.approx(1e-6)
    )
    assert rows[1]["final_mean_reputation_diff_vs_loop"] == pytest.approx(0.0)
    assert rows[1]["parity_passed_vs_loop"] is True
    assert benchmark_toy_gpu_core.backend_parity_failures(rows) == []


def test_benchmark_loop_comparison_missing_baseline_stays_empty() -> None:
    rows = [
        _benchmark_row(
            training_backend="batched",
            seconds=2.0,
            final_action_rate=0.25,
            final_mean_policy_action_probability=0.5,
            final_mean_reputation=0.75,
        )
    ]

    benchmark_toy_gpu_core.annotate_loop_comparisons(rows)

    assert rows[0]["loop_speedup"] == ""
    assert rows[0]["final_action_rate_diff_vs_loop"] == ""
    assert rows[0]["parity_passed_vs_loop"] == ""


def test_benchmark_backend_parity_failures_reports_non_loop_drift() -> None:
    rows = [
        _benchmark_row(
            training_backend="loop",
            seconds=4.0,
            final_action_rate=0.25,
            final_mean_policy_action_probability=0.5,
            final_mean_reputation=0.75,
        ),
        _benchmark_row(
            training_backend="tensor_batched",
            seconds=2.0,
            final_action_rate=0.30,
            final_mean_policy_action_probability=0.5,
            final_mean_reputation=0.75,
        ),
    ]

    benchmark_toy_gpu_core.annotate_loop_comparisons(rows)

    assert rows[1]["parity_passed_vs_loop"] is False
    assert benchmark_toy_gpu_core.backend_parity_failures(rows) == [rows[1]]


def _benchmark_row(
    *,
    training_backend: str,
    seconds: float,
    final_action_rate: float,
    final_mean_policy_action_probability: float,
    final_mean_reputation: float,
) -> dict[str, object]:
    return {
        "toy": "toy2",
        "scenario": "neural_output_average",
        "training_backend": training_backend,
        "resolved_training_backend": training_backend,
        "device": "cpu",
        "agent_count": 64,
        "epochs": 3,
        "seed": 7,
        "warmup_runs": 0,
        "repeats": 1,
        "seconds": seconds,
        "final_action_rate": final_action_rate,
        "final_mean_policy_action_probability": (
            final_mean_policy_action_probability
        ),
        "final_mean_reputation": final_mean_reputation,
    }
