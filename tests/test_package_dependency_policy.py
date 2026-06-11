from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION_0014 = ROOT / "docs" / "decisions" / "0014-package-dependency-policy.md"
RELEASE_BOUNDARY = ROOT / "docs" / "package-release-boundary.md"
PYPROJECT = ROOT / "pyproject.toml"
RELEASE_INSPECT_SCRIPT = ROOT / "scripts" / "inspect_release_artifacts.py"
PROFILE_SMOKE_SCRIPT = ROOT / "scripts" / "smoke_package_profiles.py"


def _dependency_name(requirement: str) -> str:
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    assert match is not None
    return match.group(1).lower()


def _plain(text: str) -> str:
    return " ".join(text.split())


def test_package_dependency_policy_classifies_all_direct_dependencies() -> None:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dependencies = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["dependencies"]
    }
    optional_dependencies = {
        extra: {_dependency_name(requirement) for requirement in requirements}
        for extra, requirements in pyproject["project"]["optional-dependencies"].items()
    }
    dev_dependencies = {
        _dependency_name(requirement)
        for requirement in pyproject["dependency-groups"]["dev"]
    }
    decision = _plain(DECISION_0014.read_text(encoding="utf-8"))

    assert pyproject["project"]["authors"] == [
        {"name": "kimyoungjin06", "email": "kimyoungjin06@gmail.com"}
    ]
    assert pyproject["project"]["license"] == "Apache-2.0"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    assert pyproject["project"]["urls"] == {
        "Homepage": "https://github.com/kimyoungjin06/neural-abm",
        "Issues": "https://github.com/kimyoungjin06/neural-abm/issues",
        "Repository": "https://github.com/kimyoungjin06/neural-abm",
    }
    assert "agent-based modeling" in pyproject["project"]["keywords"]
    assert "Development Status :: 3 - Alpha" in pyproject["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.14" in pyproject["project"][
        "classifiers"
    ]
    assert dependencies == {"numpy", "pyyaml"}
    assert optional_dependencies["torch"] == {"torch"}
    assert optional_dependencies["config"] == {"pydantic"}
    assert optional_dependencies["plot"] == {"matplotlib"}
    assert optional_dependencies["cli"] == {"tqdm"}

    for heavy_dependency in (
        "matplotlib",
        "networkx",
        "pandas",
        "pyarrow",
        "pydantic",
        "scikit-learn",
        "scipy",
        "torch",
        "tqdm",
    ):
        assert heavy_dependency in optional_dependencies["full"]
        assert heavy_dependency in dev_dependencies
        assert heavy_dependency not in dependencies

    for required in (
        "The default install is the `api_lite` floor",
        "`numpy`",
        "`pyyaml`",
        "`config`",
        "`torch`",
        "`research`",
        "`plot`",
        "`cli`",
        "`full`",
        "The dev dependency group also includes the full research stack",
    ):
        assert required in decision


def test_package_dependency_policy_defers_torch_optionalization() -> None:
    decision = _plain(DECISION_0014.read_text(encoding="utf-8"))
    release_boundary = _plain(RELEASE_BOUNDARY.read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    module_readme = (ROOT / "src" / "neural_abm" / "README.md").read_text(
        encoding="utf-8"
    )

    for required in (
        "not yet a lightweight no-torch API boundary",
        "`unit.py` and `social.py`",
        "import `torch` at module load time",
        "The first import-time split is `neural_abm.api_lite`",
        "not a replacement for `neural_abm.api`",
        "torch-free social primitives",
        "`api_lite.SocialChannel` accepts only scalar/bounded",
        "lightweight lifecycle reports/local-step primitives",
        "built-wheel profile smokes prove",
        "default profile imports the package root",
        "`research` extra imports representative research dependencies",
        "must be lazy",
        "Amendment: Torch-Free Lifecycle Reports",
        "`neural_abm.unit` remains a compatibility module",
        "The full stable v0 facade should not be marketed as a no-torch API",
    ):
        assert required in decision

    for required in (
        "Package Release Boundary",
        "The default profile is the lightweight `api_lite` floor",
        "must not import or require `torch`",
        "Use `neural_abm.api_lite` for no-torch metadata",
        "from neural_abm.api_lite import toy_catalog, toys_by_taxonomy",
        "Use `neural_abm.api` for the stable torch-backed v0 lifecycle",
        "Run artifacts include these fields in `metadata.json` and `summary.json`",
        "uv run python scripts/smoke_package_profiles.py",
        "uv run python examples/toy_catalog.py",
        "default wheel metadata does not require",
        "Do not market `neural_abm.api` as no-torch",
        "Do not rename stable model IDs",
    ):
        assert required in release_boundary

    for required in (
        "Public API and Package Status",
        "from neural_abm.api import NABMUnit, SocialBlock, SocialChannel",
        "neural_abm.api_lite",
        "Package release boundary",
        "toy_catalog",
        "default package profile is now a lightweight torch-free install",
        "scalar/bounded scalar",
        "uv run python examples/toy_catalog.py",
        "uv run python scripts/smoke_package_profiles.py",
        "scripts/inspect_release_artifacts.py",
        "uv run python scripts/inspect_release_artifacts.py --build",
        "Decision 0014",
    ):
        assert required in readme

    for required in (
        "Package Dependency Boundary",
        "default package profile is the lightweight no-torch `api_lite` boundary",
        "src/neural_abm/api_lite.py",
        "src/neural_abm/social_core.py",
        "src/neural_abm/unit_core.py",
        "`api_lite.SocialChannel` accepts only scalar/bounded",
        "`unit` and `social`",
        "load `torch` at import time",
        "Decision 0014",
        "`torch`, `research`,",
        "`plot`, `cli`, or `full`",
    ):
        assert required in module_readme


def test_package_profile_smoke_script_covers_release_profiles() -> None:
    script = PROFILE_SMOKE_SCRIPT.read_text(encoding="utf-8")

    for required in (
        'DEFAULT_PROFILES = ("default", "torch", "research", "full")',
        '"default"',
        '"torch"',
        '"research"',
        '"full"',
        "torch_installed",
        "default_requires",
        "torch_required_by_default",
        "neural_abm.api_lite",
        "api_lite.SocialChannel",
        "LITE_SOCIAL_CHANNEL_KINDS",
        "api_lite.mix_scalar_probabilities",
        "api_lite.CommitReport",
        "api_lite.NABMLocalStep",
        "local_losses",
        "rejected_tensor_channel",
        "social_mix_values",
        "api_lite.toy_catalog",
        "toy_catalog_count",
        "taxonomy_binary_probability",
        "toy10_display_name",
        "from neural_abm.api import NABMUnit, SocialBlock, SocialChannel",
        "from neural_abm.config import Toy6Config",
        "from neural_abm.toy_market import run_toy10",
        "args.wheel.resolve()",
        "uv",
        "--isolated",
    ):
        assert required in script


def test_release_artifact_inspection_script_covers_dry_run_boundaries() -> None:
    script = RELEASE_INSPECT_SCRIPT.read_text(encoding="utf-8")

    for required in (
        "REQUIRED_WHEEL_MODULES",
        "REQUIRED_SDIST_FILES",
        '"LICENSE"',
        "docs/git-distribution-flow.md",
        "DEFAULT_DEPENDENCIES",
        "project.license is not set",
        "project.urls is not set",
        "requires-python is >=3.14",
        "FORBIDDEN_SDIST_PREFIXES",
        "docs/pre-release-artifact-flow.md",
        "version is not marked as an alpha",
        "Description-Content-Type",
        "Provides-Extra",
        "Requires-Dist",
        "uv",
        "build",
    ):
        assert required in script
