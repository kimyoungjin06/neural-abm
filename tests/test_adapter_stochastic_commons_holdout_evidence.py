from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/evidence/adapter_only_stochastic_commons_quick.yaml"
RUNNER_PATH = ROOT / "scripts/run_adapter_stochastic_commons_holdout_evidence.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_adapter_stochastic_commons_holdout_evidence",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_stochastic_commons_manifest_contract() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["label"] == "adapter_only_stochastic_commons_quick"
    assert "endogenous state transitions" in manifest["description"]
    assert "src/neural_abm" in manifest["description"]
    assert {variant["group"] for variant in manifest["variants"]} == {
        "baseline",
        "negative_control",
        "main",
    }
    assert {case["name"] for case in manifest["cases"]} == {
        "steady_regen_commons",
        "localized_resource_shock",
        "heterogeneous_need_commons",
    }
    assert (
        manifest["success_criteria"]["localized_resource_shock"]["main"][
            "max_collapse_epochs"
        ]
        == 0
    )
    assert (
        manifest["success_criteria"]["heterogeneous_need_commons"]["main"][
            "min_recovery_hits"
        ]
        == 3
    )


def test_adapter_stochastic_commons_runner_writes_artifacts(tmp_path: Path) -> None:
    runner = _load_runner()
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["results_dir"] = str(tmp_path / "results")
    manifest["summary_dir"] = str(tmp_path / "summaries")
    manifest["findings_path"] = str(tmp_path / "findings.md")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    outputs = runner.run_adapter_stochastic_commons_holdout_evidence(manifest_path)

    for path in outputs.values():
        assert path.exists()
    rows = outputs["runs"].read_text(encoding="utf-8")
    assert "adapter_local_resource_main" in rows
    assert "collapse_epochs" in rows
    assert "unit_policy_lifecycle_used" in rows
    assert "source_changes_required" in rows
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    shock = next(
        case for case in summary["cases"] if case["case"] == "localized_resource_shock"
    )
    main = shock["variants"]["adapter_local_resource_main"]
    baseline = shock["variants"]["greedy_harvest_baseline"]
    negative = shock["variants"]["global_pressure_negative_control"]
    assert main["max_collapse_epochs"] == 0
    assert main["min_resource_mean"] >= 0.43
    assert main["recovery_hits"] == 3
    assert baseline["min_collapse_epochs"] >= 1
    assert negative["min_collapse_epochs"] >= 1
    findings = outputs["findings"].read_text(encoding="utf-8")
    assert "endogenous state transitions" in findings
    assert "not a full general-purpose ABM framework proof" in findings
