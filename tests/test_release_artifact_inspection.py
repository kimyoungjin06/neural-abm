from __future__ import annotations

import json
import runpy
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
    assert payload["pyproject"]["version"] == "0.1.0a6.dev0"
    assert payload["pyproject"]["requires_python"] == ">=3.11"
    assert payload["pyproject"]["default_dependencies"] == ["numpy", "pyyaml"]
    assert payload["pyproject"]["has_authors"] is True
    assert payload["pyproject"]["has_classifiers"] is True
    assert payload["pyproject"]["has_keywords"] is True
    assert payload["pyproject"]["has_license"] is True
    assert payload["pyproject"]["has_urls"] is True
    assert payload["wheel"]["metadata_name"] == "neural-abm"
    assert payload["wheel"]["metadata_version"] == "0.1.0a6.dev0"
    assert payload["wheel"]["runtime_version"] == "0.1.0a6.dev0"
    assert payload["wheel"]["default_requires"] == ["numpy", "pyyaml"]
    assert "neural_abm/scenario_lite.py" in payload["wheel"]["required_modules"]
    assert "neural_abm/workflow_lite.py" in payload["wheel"]["required_modules"]
    assert payload["wheel"]["required_modules_present"] is True
    assert (
        "examples/research_pivot_scenario_lite.py" in payload["sdist"]["required_files"]
    )
    claim_bearing_files = {
        "docs/case-studies/researcher-pivot/README.md",
        "docs/case-studies/researcher-pivot/data/learning_study_results.json",
        "docs/case-studies/researcher-pivot/data/study_results.json",
        "docs/classical-reductions.md",
        "docs/decisions/0015-researcher-scenario-lite-contract.md",
        "docs/figures/nabm_unit_recurrent_block.png",
        "docs/figures/nabm_unit_recurrent_block.svg",
        "examples/classical_reductions.py",
        "examples/research_pivot_learning_study.py",
        "examples/research_pivot_study.py",
        "scripts/inspect_release_artifacts.py",
        "scripts/plot_nabm_unit_schematic.py",
        "scripts/plot_research_pivot_study.py",
    }
    assert claim_bearing_files <= set(payload["sdist"]["required_files"])
    assert payload["sdist"]["required_files_present"] is True
    assert payload["sdist"]["internal_links_checked"] > 0
    assert payload["sdist"]["internal_links_ok"] is True
    assert payload["sdist"]["broken_internal_links"] == []
    assert payload["sdist"]["forbidden_files_present"] is False
    assert payload["sdist"]["forbidden_files"] == []
    assert not any("project.license is not set" in item for item in payload["warnings"])
    assert not any("project.urls is not set" in item for item in payload["warnings"])


def test_internal_markdown_link_inspection_catches_missing_targets() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    inspect_links = inspector["_inspect_internal_markdown_links"]
    names = {"README.md", "docs/guide.md"}
    markdown_sources = {
        "README.md": "\n".join(
            (
                "[valid](docs/guide.md#present-anchor)",
                "[missing file](docs/missing.md)",
                "[missing anchor](docs/guide.md#absent-anchor)",
                "[external](https://example.com/not-in-the-sdist)",
            )
        ),
        "docs/guide.md": "# Present Anchor\n",
    }

    issues, checked_count = inspect_links(names, markdown_sources)

    assert checked_count == 3
    assert len(issues) == 2
    assert "targets missing file 'docs/missing.md'" in issues[0]
    assert "targets missing anchor 'absent-anchor'" in issues[1]


def test_cross_artifact_metadata_check_rejects_stale_dist() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    cross_check = inspector["_cross_artifact_metadata_issues"]
    pyproject = {
        "name": "neural-abm",
        "version": "0.1.0a6.dev0",
        "requires_python": ">=3.11",
    }
    stale = {
        "metadata_name": "other-name",
        "metadata_version": "0.1.0a7.dev0",
        "requires_python": ">=3.12",
    }

    issues = cross_check(pyproject, stale, stale)

    assert len(issues) == 6
    assert any("wheel version differs from pyproject" in issue for issue in issues)
    assert any(
        "sdist requires_python differs from pyproject" in issue for issue in issues
    )
