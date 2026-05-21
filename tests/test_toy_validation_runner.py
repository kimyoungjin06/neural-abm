from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from neural_abm.capabilities import supports_coordination


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_toy_validation.py"
SPEC = importlib.util.spec_from_file_location("run_toy_validation", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_toy_validation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_toy_validation
SPEC.loader.exec_module(run_toy_validation)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_fake_runner(
    *,
    summary: dict[str, Any] | None,
    aggregate_rows: list[dict[str, Any]] | None = None,
    micro_rows: list[dict[str, Any]] | None = None,
):
    def fake_runner(config: Any, config_path: Path) -> SimpleNamespace:
        run_dir = (
            Path(config.run.output_dir)
            / f"fake_{config.run.name}_seed{config.run.seed:02d}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        if summary is not None:
            payload = {"run_dir": str(run_dir), **summary}
            (run_dir / "summary.json").write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )

        rows = aggregate_rows if aggregate_rows is not None else [{"value": 1.0}]
        fieldnames = sorted({key for row in rows for key in row}) or ["value"]
        with (run_dir / "aggregate_metrics.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        micro = micro_rows if micro_rows is not None else [{"agent_id": 0}]
        micro_fields = sorted({key for row in micro for key in row}) or ["agent_id"]
        with (run_dir / "micro_state.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=micro_fields)
            writer.writeheader()
            writer.writerows(micro)

        return SimpleNamespace(run_dir=run_dir)

    return fake_runner


@pytest.mark.parametrize("spec", run_toy_validation.scenario_specs())
def test_validation_scenario_configs_load(tmp_path: Path, spec: Any) -> None:
    config_path = run_toy_validation.write_scenario_config(
        spec,
        label="unit_validation",
        seed=2,
        epochs=3,
        config_root=tmp_path / "configs",
        runs_dir=tmp_path / "runs",
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = run_toy_validation.TOY_HANDLERS[spec.toy].loader(config_path)

    assert config.run.seed == 2
    assert config.simulation.epochs == 3
    assert config.run.output_dir == tmp_path / "runs"

    coordination = raw.get("model", {}).get(
        "coordination",
        raw.get("coordination", raw.get("social", {})),
    )
    if coordination["mixer"] == "none":
        assert coordination["peer_rule"] == "none"
        assert coordination["alpha"] == pytest.approx(0.0)
    if coordination["mixer"] == "output_average":
        assert coordination["peer_rule"] == "output_similarity"
        assert coordination["alpha"] == pytest.approx(0.25)
    assert supports_coordination(
        spec.toy,
        str(coordination["mixer"]),
        str(coordination["peer_rule"]),
    )
    if spec.toy in {
        "toy1",
        "toy2",
        "toy3",
        "toy4",
        "toy5",
        "toy6",
        "toy7",
        "toy8",
        "toy9",
        "toy10",
    }:
        assert set(raw) == {"run", "simulation", "model", "domain", "logging"}
        assert "coordination" in raw["model"]
        if spec.toy != "toy1":
            assert "policy" in raw["model"]
        if spec.toy in {"toy2", "toy4", "toy5"}:
            assert "state" in raw["model"]
        assert raw["domain"]["toy"] == spec.toy
        assert not {
            "dynamics",
            "social",
            "agents",
            "graph",
            "environment",
            "rewiring",
            "reputation",
            "mobility",
        } & set(raw)
        if spec.toy != "toy1":
            assert hasattr(config, "policy")
        assert hasattr(config, "coordination")
        if spec.toy in {"toy2", "toy4", "toy5"}:
            assert hasattr(config, "state")


def test_validation_fake_success_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_toy_validation.TOY_HANDLERS["toy1"],
        "runner",
        make_fake_runner(
            summary={
                "domain_final_mean_global_accuracy": 0.75,
                "domain_final_mean_consensus": 0.8,
                "final_fragmentation_components": 1,
            },
            aggregate_rows=[
                {
                    "domain_mean_global_accuracy": 0.75,
                    "domain_mean_consensus": 0.8,
                },
            ],
        ),
    )

    result = run_toy_validation.run_validation(
        label="fake_success",
        seeds=[1, 2],
        epochs=1,
        config_dir=tmp_path / "configs",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        scenario_names=["toy1_no_social"],
    )

    assert result.runs_path.exists()
    assert result.metrics_path.exists()
    assert result.report_path.exists()
    assert len(result.records) == 2
    assert all(record.status == "pass" for record in result.records)
    assert {row["metric"] for row in read_csv(result.metrics_path)} >= {
        "domain_final_mean_global_accuracy",
        "domain_final_mean_consensus",
    }
    assert read_csv(result.runs_path)[0]["status"] == "pass"


@pytest.mark.parametrize(
    ("summary", "aggregate_rows", "expected"),
    [
        (
            None,
            [{"domain_mean_global_accuracy": 0.75, "domain_mean_consensus": 0.8}],
            "missing summary.json",
        ),
        (
            {
                "domain_final_mean_global_accuracy": 0.75,
                "domain_final_mean_consensus": 0.8,
                "final_fragmentation_components": 1,
            },
            [],
            "aggregate_metrics.csv has no data rows",
        ),
        (
            {
                "domain_final_mean_global_accuracy": 0.75,
                "domain_final_mean_consensus": 0.8,
                "final_fragmentation_components": 1,
            },
            [{"domain_mean_global_accuracy": 1.5, "domain_mean_consensus": 0.8}],
            "outside [0, 1]",
        ),
    ],
)
def test_validation_marks_failed_run_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary: dict[str, Any] | None,
    aggregate_rows: list[dict[str, Any]],
    expected: str,
) -> None:
    monkeypatch.setattr(
        run_toy_validation.TOY_HANDLERS["toy1"],
        "runner",
        make_fake_runner(summary=summary, aggregate_rows=aggregate_rows),
    )

    result = run_toy_validation.run_validation(
        label="fake_failure",
        seeds=[1],
        epochs=1,
        config_dir=tmp_path / "configs",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        scenario_names=["toy1_no_social"],
    )

    assert result.failed is True
    assert expected in result.records[0].to_row()["failed_checks"]
    assert read_csv(result.runs_path)[0]["status"] == "fail"


def test_validation_stop_on_failure_writes_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_toy_validation.TOY_HANDLERS["toy1"],
        "runner",
        make_fake_runner(summary=None),
    )

    result = run_toy_validation.run_validation(
        label="fake_stop",
        seeds=[1, 2],
        epochs=1,
        config_dir=tmp_path / "configs",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        scenario_names=["toy1_no_social", "toy1_output_average"],
        stop_on_failure=True,
    )

    assert result.failed is True
    assert len(result.records) == 1
    assert result.runs_path.exists()
    assert len(read_csv(result.runs_path)) == 1


def test_validation_tiny_actual_toy3_toy5_subset(tmp_path: Path) -> None:
    result = run_toy_validation.run_validation(
        label="tiny_actual",
        seeds=[1],
        epochs=1,
        config_dir=tmp_path / "configs",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        scenario_names=["toy3_hk_no_rewire", "toy5_neural_output_average"],
    )

    assert (tmp_path / "configs" / "tiny_actual").exists()
    assert result.runs_path.exists()
    assert result.metrics_path.exists()
    assert result.report_path.exists()
    rows = read_csv(result.runs_path)
    assert {row["scenario"] for row in rows} == {
        "toy3_hk_no_rewire",
        "toy5_neural_output_average",
    }


def test_validation_cli_presets_resolve_defaults() -> None:
    args = SimpleNamespace(
        preset="quick",
        label=None,
        seeds=None,
        epochs=None,
        scenarios=None,
    )

    label, seeds, epochs, scenarios = run_toy_validation.resolve_cli_selection(args)

    assert label == "toy_validation_quick"
    assert seeds == [1]
    assert epochs == 3
    assert scenarios == [
        "toy1_no_social",
        "toy2_harsh_pd_neural_none",
        "toy3_hk_no_rewire",
        "toy4_static_imitation_none",
        "toy5_low_threshold_cascade",
        "toy6_categorical_output_average",
        "toy7_resource_output_average",
        "toy8_async_output_average",
        "toy9_heterogeneous_output_average",
        "toy10_market_output_average",
    ]


def test_validation_cli_overrides_preset_defaults() -> None:
    args = SimpleNamespace(
        preset="quick",
        label="custom",
        seeds=[2, 4],
        epochs=7,
        scenarios=["toy3_hk_rewire"],
    )

    label, seeds, epochs, scenarios = run_toy_validation.resolve_cli_selection(args)

    assert label == "custom"
    assert seeds == [2, 4]
    assert epochs == 7
    assert scenarios == ["toy3_hk_rewire"]
