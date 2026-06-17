from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSPECT_SCRIPT = ROOT / "scripts" / "inspect_release_artifacts.py"


def test_release_artifact_inspection_builds_and_reports_package_boundary() -> None:
    completed = subprocess.run(
        [sys.executable, str(INSPECT_SCRIPT), "--build", "--ephemeral", "--json"],
        check=True,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])

    assert payload["status"] == "pass"
    assert payload["blocking_issues"] == []
    assert payload["pyproject"]["name"] == "neural-abm"
    assert payload["pyproject"]["version"] == "0.1.0a5"
    assert payload["pyproject"]["requires_python"] == ">=3.11"
    assert payload["pyproject"]["default_dependencies"] == ["numpy", "pyyaml"]
    assert payload["pyproject"]["has_authors"] is True
    assert payload["pyproject"]["has_classifiers"] is True
    assert payload["pyproject"]["has_keywords"] is True
    assert payload["pyproject"]["has_license"] is True
    assert payload["pyproject"]["has_urls"] is True
    assert payload["wheel"]["metadata_name"] == "neural-abm"
    assert payload["wheel"]["metadata_version"] == "0.1.0a5"
    assert payload["wheel"]["default_requires"] == ["numpy", "pyyaml"]
    assert payload["wheel"]["required_modules_present"] is True
    assert payload["sdist"]["required_files_present"] is True
    assert payload["sdist"]["forbidden_files_present"] is False
    assert payload["sdist"]["forbidden_files"] == []
    assert not any(
        "project.license is not set" in item for item in payload["warnings"]
    )
    assert not any("project.urls is not set" in item for item in payload["warnings"])
