"""Inspect built pre-release artifacts for package-boundary checks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_DEPENDENCIES = {"numpy", "pyyaml"}
REQUIRED_PYTHON = ">=3.11"
REQUIRED_EXTRAS = {"torch", "research", "full"}
REQUIRED_WHEEL_MODULES = {
    "neural_abm/__init__.py",
    "neural_abm/api.py",
    "neural_abm/api_lite.py",
    "neural_abm/capabilities.py",
    "neural_abm/scenario_lite.py",
    "neural_abm/social_core.py",
    "neural_abm/unit_core.py",
    "neural_abm/workflow_lite.py",
}
REQUIRED_SDIST_FILES = {
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "docs/early-git-user-handoff.md",
    "docs/git-distribution-flow.md",
    "docs/package-release-boundary.md",
    "docs/pre-release-artifact-flow.md",
    "examples/first_run.py",
    "examples/research_pivot_scenario_lite.py",
    "examples/toy_catalog.py",
    "scripts/smoke_package_profiles.py",
}
FORBIDDEN_SDIST_PATHS = {
    "docs/release-candidate-dry-run.md",
    "scripts/reproduce_early_git.py",
}
FORBIDDEN_SDIST_PREFIXES = (
    "archive/",
    "experiments/",
    "paper/",
    "ref/",
    "docs/nabm-unit-v1",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect wheel/sdist metadata for a pre-release artifact.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build artifacts with uv build into --dist-dir first.",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=ROOT / "dist",
        help="Directory containing built wheel and sdist artifacts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print compact JSON only.",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="With --build, build into a temporary directory instead of dist/.",
    )
    args = parser.parse_args()

    if args.build and args.ephemeral:
        with tempfile.TemporaryDirectory(prefix="neural-abm-release-inspect.") as temp:
            dist_dir = Path(temp)
            _build_dist(dist_dir)
            report = inspect_release_artifacts(dist_dir)
    elif args.build:
        _build_dist(args.dist_dir)
        report = inspect_release_artifacts(args.dist_dir)
    else:
        report = inspect_release_artifacts(args.dist_dir)

    text = json.dumps(report, indent=None if args.json else 2, sort_keys=True)
    print(text)
    if report["blocking_issues"]:
        raise SystemExit(1)


def inspect_release_artifacts(dist_dir: Path) -> dict[str, Any]:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    wheel = _single_artifact(dist_dir, "*.whl")
    sdist = _single_artifact(dist_dir, "*.tar.gz")
    wheel_report = _inspect_wheel(wheel)
    sdist_report = _inspect_sdist(sdist)
    pyproject_report = _inspect_pyproject(pyproject)
    blocking_issues = [
        *pyproject_report["blocking_issues"],
        *wheel_report["blocking_issues"],
        *sdist_report["blocking_issues"],
    ]
    warnings = [
        *pyproject_report["warnings"],
        *wheel_report["warnings"],
        *sdist_report["warnings"],
    ]
    return {
        "status": "pass" if not blocking_issues else "fail",
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "pyproject": pyproject_report["summary"],
        "wheel": wheel_report["summary"],
        "sdist": sdist_report["summary"],
    }


def _build_dist(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("neural_abm-*.whl", "neural_abm-*.tar.gz"):
        for artifact in output_dir.glob(pattern):
            artifact.unlink()
    subprocess.run(
        ["uv", "build", "--out-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
    )


def _single_artifact(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {pattern!r} artifact in {dist_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _inspect_pyproject(pyproject: dict[str, Any]) -> dict[str, Any]:
    project = pyproject["project"]
    optional_dependencies = project["optional-dependencies"]
    default_dependency_names = {
        _dependency_name(requirement)
        for requirement in project["dependencies"]
    }
    missing_extras = sorted(REQUIRED_EXTRAS - set(optional_dependencies))
    blocking_issues: list[str] = []
    warnings: list[str] = []

    if default_dependency_names != DEFAULT_DEPENDENCIES:
        blocking_issues.append(
            "default dependencies must stay limited to numpy and pyyaml"
        )
    if missing_extras:
        blocking_issues.append(f"missing required extras: {', '.join(missing_extras)}")
    if project.get("requires-python") != REQUIRED_PYTHON:
        blocking_issues.append(f"requires-python must stay at {REQUIRED_PYTHON}")
    if "license" not in project:
        warnings.append("project.license is not set; choose a license before release")
    if "urls" not in project:
        warnings.append("project.urls is not set; add project links before release")
    version = project.get("version", "")
    if not _is_pre_release_version(version):
        warnings.append(
            "version is not marked as an alpha, beta, rc, or dev release; "
            "reserve final versions for intentional public releases"
        )

    return {
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "summary": {
            "name": project["name"],
            "version": project["version"],
            "requires_python": project["requires-python"],
            "default_dependencies": sorted(default_dependency_names),
            "extras": sorted(optional_dependencies),
            "has_authors": bool(project.get("authors")),
            "has_classifiers": bool(project.get("classifiers")),
            "has_keywords": bool(project.get("keywords")),
            "has_license": "license" in project,
            "has_urls": "urls" in project,
        },
    }


def _inspect_wheel(wheel: Path) -> dict[str, Any]:
    blocking_issues: list[str] = []
    warnings: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(
            archive.read(metadata_name).decode("utf-8", errors="replace")
        )

    missing_modules = sorted(REQUIRED_WHEEL_MODULES - names)
    if missing_modules:
        blocking_issues.append(
            f"wheel missing required modules: {', '.join(missing_modules)}"
        )
    requires_dist = metadata.get_all("Requires-Dist") or []
    default_requires = sorted(
        {
            _dependency_name(requirement)
            for requirement in requires_dist
            if "extra ==" not in requirement
        }
    )
    extras = sorted(set(metadata.get_all("Provides-Extra") or []))
    if set(default_requires) != DEFAULT_DEPENDENCIES:
        blocking_issues.append(
            "wheel default Requires-Dist must stay limited to numpy and pyyaml"
        )
    if REQUIRED_EXTRAS - set(extras):
        blocking_issues.append(
            "wheel metadata missing required extras: "
            f"{', '.join(sorted(REQUIRED_EXTRAS - set(extras)))}"
        )
    if metadata.get("Description-Content-Type") != "text/markdown":
        warnings.append("wheel README metadata is not marked as text/markdown")

    return {
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "summary": {
            "path": str(wheel),
            "metadata_name": metadata.get("Name"),
            "metadata_version": metadata.get("Version"),
            "requires_python": metadata.get("Requires-Python"),
            "default_requires": default_requires,
            "extras": extras,
            "required_modules": sorted(REQUIRED_WHEEL_MODULES),
            "required_modules_present": not missing_modules,
            "description_content_type": metadata.get("Description-Content-Type"),
        },
    }


def _inspect_sdist(sdist: Path) -> dict[str, Any]:
    blocking_issues: list[str] = []
    with tarfile.open(sdist, "r:gz") as archive:
        names = {
            _strip_sdist_root(member.name)
            for member in archive.getmembers()
            if member.isfile()
        }
    missing_files = sorted(REQUIRED_SDIST_FILES - names)
    if missing_files:
        blocking_issues.append(
            f"sdist missing required files: {', '.join(missing_files)}"
        )
    forbidden_files = sorted(
        name
        for name in names
        if name in FORBIDDEN_SDIST_PATHS
        or any(name.startswith(prefix) for prefix in FORBIDDEN_SDIST_PREFIXES)
    )
    if forbidden_files:
        blocking_issues.append(
            "sdist includes internal-history files: "
            f"{', '.join(forbidden_files[:10])}"
        )
    return {
        "blocking_issues": blocking_issues,
        "warnings": [],
        "summary": {
            "path": str(sdist),
            "required_files": sorted(REQUIRED_SDIST_FILES),
            "required_files_present": not missing_files,
            "missing_required_files": missing_files,
            "forbidden_files_present": bool(forbidden_files),
            "forbidden_files": forbidden_files,
        },
    }


def _dependency_name(requirement: str) -> str:
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise ValueError(f"could not parse dependency name from {requirement!r}")
    return match.group(1).lower()


def _is_pre_release_version(version: str) -> bool:
    return bool(re.search(r"(a|b|rc|dev)\d*$", version))


def _strip_sdist_root(path: str) -> str:
    parts = Path(path).parts
    return "/".join(parts[1:]) if len(parts) > 1 else path


if __name__ == "__main__":
    main()
