from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib


matplotlib.use("Agg")

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


run_toy_validation = import_script(
    "run_toy_validation_for_plot_tests",
    "scripts/run_toy_validation.py",
)
plot_toy_validation = import_script(
    "plot_toy_validation_for_tests",
    "scripts/plot_toy_validation.py",
)


def test_plot_validation_accepts_quick_preset_subset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = run_toy_validation.run_validation(
        label="quick_plot_unit",
        seeds=[1],
        epochs=1,
        config_dir=tmp_path / "configs",
        results_dir=tmp_path / "results",
        runs_dir=tmp_path / "runs",
        scenario_names=run_toy_validation.VALIDATION_PRESETS["quick"].scenarios,
    )
    output_path = tmp_path / "figures" / "quick_validation.png"

    monkeypatch.setattr(
        plot_toy_validation,
        "parse_args",
        lambda: SimpleNamespace(
            metrics=result.metrics_path,
            runs=result.runs_path,
            output=output_path,
        ),
    )

    plot_toy_validation.main()

    assert result.report_path.exists()
    assert output_path.exists()
    assert output_path.stat().st_size > 0
