from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/evidence/adapter_only_congestion_holdout_quick.yaml"
RUNNER_PATH = ROOT / "scripts/run_adapter_congestion_holdout_evidence.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_adapter_congestion_holdout_evidence",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_congestion_holdout_manifest_contract() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["label"] == "adapter_only_congestion_holdout_quick"
    assert "balanced allocation" in manifest["description"]
    assert "threshold cascade" in manifest["description"]
    assert {variant["group"] for variant in manifest["variants"]} == {
        "baseline",
        "negative_control",
        "main",
    }
    assert {case["name"] for case in manifest["cases"]} == {
        "symmetric_capacity",
        "asymmetric_capacity",
        "noisy_preference_capacity",
    }
    assert (
        manifest["success_criteria"]["asymmetric_capacity"]["main"][
            "max_final_capacity_error_abs"
        ]
        == 0
    )
    assert (
        manifest["success_criteria"]["symmetric_capacity"]["negative_control"][
            "min_overcrowding_count"
        ]
        == 10
    )


def test_adapter_congestion_holdout_evidence_runner_writes_artifacts(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["results_dir"] = str(tmp_path / "results")
    manifest["summary_dir"] = str(tmp_path / "summaries")
    manifest["findings_path"] = str(tmp_path / "findings.md")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    outputs = runner.run_adapter_congestion_holdout_evidence(manifest_path)

    for path in outputs.values():
        assert path.exists()
    rows = outputs["runs"].read_text(encoding="utf-8")
    assert "adapter_capacity_policy_main" in rows
    assert "final_capacity_error_abs" in rows
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    asymmetric = next(
        case for case in summary["cases"] if case["case"] == "asymmetric_capacity"
    )
    assert asymmetric["variants"]["adapter_capacity_policy_main"][
        "max_capacity_error_abs"
    ] == 0
    assert asymmetric["variants"]["imitation_baseline"]["max_capacity_error_abs"] >= 14
    assert (
        asymmetric["variants"]["global_pressure_negative_control"][
            "max_overcrowding_count"
        ]
        >= 14
    )
    findings = outputs["findings"].read_text(encoding="utf-8")
    assert "threshold-cascade isomorphic" in findings
