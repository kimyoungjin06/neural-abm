"""Reproduce the early Git user paths for a tagged alpha release.

This is a maintainer support tool, not a package-facing example. It exercises
the fresh-clone path and the direct Git tag install path in temporary
directories, then verifies the default profile stays torch-free.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_REPO_URL = "https://github.com/kimyoungjin06/neural-abm.git"

VALIDATION_CODE = r"""
import importlib.metadata
import importlib.util
import json
import subprocess
import sys

import neural_abm
from neural_abm.api_lite import toy_catalog

try:
    git_head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
except Exception:
    git_head = None

print(json.dumps({
    "git_head": git_head,
    "version": neural_abm.__version__,
    "metadata_version": importlib.metadata.version("neural-abm"),
    "toy_count": len(toy_catalog()),
    "torch_installed": importlib.util.find_spec("torch") is not None,
    "torch_loaded": "torch" in sys.modules,
}, sort_keys=True))
"""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce clone-first and direct Git install alpha paths.",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Git ref to test. Defaults to v<project.version> from pyproject.toml.",
    )
    parser.add_argument(
        "--expected-version",
        default=None,
        help="Expected neural_abm.__version__ and package metadata version.",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Repository URL to clone and install from.",
    )
    parser.add_argument(
        "--python",
        default="3.11",
        help="Python version for the isolated direct Git install smoke.",
    )
    parser.add_argument(
        "--skip-fresh-clone",
        action="store_true",
        help="Skip the fresh clone reproduction path.",
    )
    parser.add_argument(
        "--skip-git-install",
        action="store_true",
        help="Skip the direct Git install reproduction path.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary reproduction directory and report its path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print compact JSON instead of pretty JSON.",
    )
    args = parser.parse_args(argv)

    expected_version = args.expected_version or _project_version()
    ref = args.ref or f"v{expected_version}"

    temp_root = Path(tempfile.mkdtemp(prefix="neural-abm-early-git."))
    try:
        report: dict[str, Any] = {
            "status": "pass",
            "ref": ref,
            "expected_version": expected_version,
            "repo_url": args.repo_url,
            "python": args.python,
            "temp_root": str(temp_root),
            "fresh_clone": None,
            "git_install": None,
        }
        if not args.skip_fresh_clone:
            report["fresh_clone"] = _run_fresh_clone(
                temp_root=temp_root,
                repo_url=args.repo_url,
                ref=ref,
                expected_version=expected_version,
            )
        if not args.skip_git_install:
            report["git_install"] = _run_git_install(
                temp_root=temp_root,
                repo_url=args.repo_url,
                ref=ref,
                python=args.python,
                expected_version=expected_version,
            )
        print(json.dumps(report, indent=None if args.json else 2, sort_keys=True))
    except Exception as exc:
        failure = {
            "status": "fail",
            "ref": ref,
            "expected_version": expected_version,
            "repo_url": args.repo_url,
            "temp_root": str(temp_root),
            "error": str(exc),
        }
        print(json.dumps(failure, indent=None if args.json else 2, sort_keys=True))
        raise SystemExit(1) from exc
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


def _project_version() -> str:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return str(pyproject["project"]["version"])


def _run_fresh_clone(
    *,
    temp_root: Path,
    repo_url: str,
    ref: str,
    expected_version: str,
) -> dict[str, Any]:
    clone_dir = temp_root / "fresh-clone"
    _run(["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(clone_dir)])
    first_run = _run_json(
        ["uv", "run", "--no-dev", "python", "examples/first_run.py"],
        cwd=clone_dir,
    )
    toy_catalog = _run_json(
        ["uv", "run", "--no-dev", "python", "examples/toy_catalog.py"],
        cwd=clone_dir,
    )
    validation = _run_json(
        ["uv", "run", "--no-dev", "python", "-c", VALIDATION_CODE],
        cwd=clone_dir,
    )

    if first_run.get("status") != "ok":
        raise RuntimeError(f"first_run status was {first_run.get('status')!r}")
    if first_run.get("toy_count") != 10:
        raise RuntimeError(f"first_run toy_count was {first_run.get('toy_count')!r}")
    if first_run.get("torch_loaded") is not False:
        raise RuntimeError("first_run loaded torch in the default profile")
    if toy_catalog.get("toy_count") != 10:
        raise RuntimeError(
            f"toy_catalog toy_count was {toy_catalog.get('toy_count')!r}"
        )
    if toy_catalog.get("torch_loaded") is not False:
        raise RuntimeError("toy_catalog loaded torch in the default profile")
    _assert_default_validation(validation, expected_version=expected_version)
    return {
        "status": "pass",
        "clone_dir": str(clone_dir),
        "first_run": {
            "status": first_run["status"],
            "toy_count": first_run["toy_count"],
            "torch_loaded": first_run["torch_loaded"],
        },
        "toy_catalog": {
            "toy_count": toy_catalog["toy_count"],
            "torch_loaded": toy_catalog["torch_loaded"],
        },
        "validation": validation,
    }


def _run_git_install(
    *,
    temp_root: Path,
    repo_url: str,
    ref: str,
    python: str,
    expected_version: str,
) -> dict[str, Any]:
    cache_dir = temp_root / "uv-cache"
    requirement = f"neural-abm @ git+{repo_url}@{ref}"
    validation = _run_json(
        [
            "uv",
            "run",
            "--quiet",
            "--isolated",
            "--no-project",
            "--python",
            python,
            "--with",
            requirement,
            "python",
            "-c",
            VALIDATION_CODE,
        ],
        cwd=temp_root,
        env={**os.environ, "UV_CACHE_DIR": str(cache_dir)},
    )
    _assert_default_validation(validation, expected_version=expected_version)
    return {
        "status": "pass",
        "requirement": requirement,
        "validation": validation,
    }


def _assert_default_validation(
    payload: dict[str, Any],
    *,
    expected_version: str,
) -> None:
    checks = {
        "version": expected_version,
        "metadata_version": expected_version,
        "toy_count": 10,
        "torch_installed": False,
        "torch_loaded": False,
    }
    for key, expected in checks.items():
        actual = payload.get(key)
        if actual != expected:
            raise RuntimeError(f"{key} was {actual!r}, expected {expected!r}")


def _run_json(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = _run(command, cwd=cwd, env=env)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"{command[0]!r} did not print JSON")
        return json.loads(lines[-1])


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = {
            "command": list(command),
            "cwd": str(cwd) if cwd is not None else None,
            "returncode": exc.returncode,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
        raise RuntimeError(json.dumps(details, indent=2, sort_keys=True)) from exc


if __name__ == "__main__":
    main()
