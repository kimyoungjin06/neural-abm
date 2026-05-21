from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

from neural_abm.diagnostics.profile_index import build_evidence_profile_index


def write_manifest(path: Path, raw: dict) -> None:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def minimal_manifest(label: str) -> dict:
    return {
        "label": label,
        "seeds": [1],
        "epochs": 2,
        "success_criteria": {
            "main_group": "nabm",
            "require_without_teacher_bootstrap_replay": True,
            "cases": {
                "toy1_case": {
                    "final_ceiling_min_hits": 1,
                    "mean_time_to_ceiling_lt": 2,
                }
            },
        },
        "cases": [
            {
                "toy": "toy1",
                "name": "toy1_case",
                "base_config": "experiments/configs/toy1_classification_baseline.yaml",
                "primary_metric": "score",
                "direction": "maximize",
                "baseline_group": "baseline",
                "nabm_group": "nabm",
                "ceiling_metric": "score",
                "ceiling_value": 1.0,
                "variants": [
                    {
                        "name": "toy1_main",
                        "group": "nabm",
                        "updates": {},
                    }
                ],
            }
        ],
    }


def minimal_rows(label: str) -> list[dict[str, object]]:
    return [
        {
            "label": label,
            "case": "toy1_case",
            "toy": "toy1",
            "variant": "toy1_main",
            "group": "nabm",
            "baseline_group": "baseline",
            "nabm_group": "nabm",
            "seed": 1,
            "primary_metric": "score",
            "metric_value": 1.0,
            "direction": "maximize",
            "ceiling_metric": "score",
            "ceiling_value": 1.0,
            "ceiling_tolerance": 0.0,
            "ceiling_metric_value": 1.0,
            "ceiling_gap": 0.0,
            "final_within_ceiling": True,
            "ever_reached_ceiling": True,
            "time_to_ceiling": 1,
            "config_path": "",
            "run_dir": "",
            "nabm_status": "full",
            "neural_role": "",
            "social_channels": "",
            "reference_policies": "",
        }
    ]


def test_build_evidence_profile_index_profiles_existing_and_skips_missing(
    tmp_path: Path,
) -> None:
    manifests_dir = tmp_path / "manifests"
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "index"
    manifests_dir.mkdir()
    results_dir.mkdir()
    existing_manifest = manifests_dir / "existing.yaml"
    missing_manifest = manifests_dir / "missing.yaml"
    write_manifest(existing_manifest, minimal_manifest("existing"))
    write_manifest(missing_manifest, minimal_manifest("missing"))
    write_rows(results_dir / "existing_runs.csv", minimal_rows("existing"))

    output = build_evidence_profile_index(
        manifest_paths=[existing_manifest, missing_manifest],
        results_dir=results_dir,
        gate_output_dir=tmp_path / "gate",
        output_dir=output_dir,
        profile_output_dir=results_dir,
    )

    assert output.csv_path == output_dir / "evidence_profile_index.csv"
    assert output.markdown_path == output_dir / "evidence_profile_index.md"
    assert output.json_path == output_dir / "evidence_profile_index.json"
    assert output.csv_path.exists()
    assert output.markdown_path.exists()
    assert output.json_path.exists()
    assert len(output.rows) == 2
    pass_row = next(row for row in output.rows if row.label == "existing")
    skipped_row = next(row for row in output.rows if row.label == "missing")
    assert pass_row.status == "pass"
    assert pass_row.case == "toy1_case"
    assert pass_row.final_ceiling_hits == 1
    assert pass_row.profile_markdown_path.endswith("existing_profile.md")
    assert skipped_row.status == "skipped"
    assert skipped_row.skipped_reason == "missing_runs"

    index_json = json.loads(output.json_path.read_text(encoding="utf-8"))
    assert {row["label"] for row in index_json} == {"existing", "missing"}
    markdown = output.markdown_path.read_text(encoding="utf-8")
    assert "Status counts" in markdown
    assert "missing_runs" in markdown


def test_profile_evidence_artifacts_script_writes_index(
    tmp_path: Path,
) -> None:
    manifests_dir = tmp_path / "manifests"
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "index"
    manifests_dir.mkdir()
    results_dir.mkdir()
    manifest_path = manifests_dir / "existing.yaml"
    write_manifest(manifest_path, minimal_manifest("existing"))
    write_rows(results_dir / "existing_runs.csv", minimal_rows("existing"))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/profile_evidence_artifacts.py",
            "--manifest",
            str(manifest_path),
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Evidence profile index rows: 1" in completed.stdout
    assert (output_dir / "evidence_profile_index.csv").exists()
    assert (output_dir / "evidence_profile_index.md").exists()
    assert (output_dir / "evidence_profile_index.json").exists()
