from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _compact(text: str) -> str:
    return " ".join(text.split())


def test_readme_quick_start_prefers_clone_first_flow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick_start = readme.index("## Quick Start")
    project_map = readme.index("## Project Map")
    scope = readme.index("## Scope")

    clone_command = (
        "git clone --depth 1 --branch v0.1.0a5 "
        "https://github.com/kimyoungjin06/neural-abm.git neural-abm"
    )
    tag_install_command = (
        'uv pip install "neural-abm @ '
        "git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a5\""
    )
    candidate_heading = "## Main / Next-Alpha Candidate"
    scenario_command = "uv run --no-dev python examples/research_pivot_scenario_lite.py"
    tag_path = readme[quick_start : readme.index(candidate_heading)]
    candidate_path = readme[
        readme.index(candidate_heading) : readme.index("## What You Just Ran")
    ]

    assert quick_start < project_map
    assert quick_start < scope
    assert "PyPI" not in readme
    assert "TestPyPI" not in readme
    assert "The current alpha is clone-first" in readme[:quick_start]
    assert "/home/kimyoungjin06" not in readme
    assert "Desktop/Workspace" not in readme
    assert readme.index(clone_command, quick_start) < readme.index(
        tag_install_command,
        quick_start,
    )
    assert "uv run --no-dev python examples/first_run.py" in readme
    assert "uv run --no-dev python examples/toy_catalog.py" in readme
    assert readme.index("examples/first_run.py", quick_start) < readme.index(
        "examples/toy_catalog.py",
        quick_start,
    )
    assert "The first-run output should report" in readme
    assert "Query the lightweight API from the clone" in readme
    assert "Use the default branch only when you intentionally want" in readme
    assert "reproduce against `v0.1.0a5` first" in readme
    assert scenario_command not in tag_path
    assert "is not present in `v0.1.0a5`" in candidate_path
    assert "--branch main" in candidate_path
    assert scenario_command in candidate_path
    assert "## Troubleshooting" in readme
    assert "git rev-parse --short HEAD" in readme
    assert "Open an issue with the failed command" in readme


def test_git_distribution_flow_documents_fresh_clone_smoke() -> None:
    doc = (ROOT / "docs" / "git-distribution-flow.md").read_text(encoding="utf-8")
    tag_path = doc[doc.index("## Current Mode") : doc.index("Use the default branch")]
    candidate_path = doc[
        doc.index("`examples/research_pivot_scenario_lite.py` is not part") : doc.index(
            "The repository can also be used locally"
        )
    ]
    scenario_command = "uv run --no-dev python examples/research_pivot_scenario_lite.py"

    for required in (
        "primary user path is a fresh clone",
        "git clone --depth 1 --branch v0.1.0a5 https://github.com/kimyoungjin06/neural-abm.git neural-abm",
        "uv run --no-dev python examples/first_run.py",
        "uv run --no-dev python examples/toy_catalog.py",
        'assert "torch" not in sys.modules',
        "v0.1.0a5",
    ):
        assert required in doc

    assert scenario_command not in tag_path
    assert "`main` / next-alpha candidate" in candidate_path
    assert "--branch main" in candidate_path
    assert scenario_command in candidate_path


def test_ci_keeps_clone_first_default_profile_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    for required in (
        "Smoke clone-first default environment",
        "UV_PROJECT_ENVIRONMENT=.venv-clone-smoke",
        "uv run --no-dev python examples/first_run.py",
        "uv run --no-dev python examples/toy_catalog.py",
        "uv run --no-dev python examples/research_pivot_scenario_lite.py",
        'assert payload["requires_python"] == ">=3.11"',
        'assert payload["torch_loaded"] is False',
        'assert payload["toy_count"] == 10',
    ):
        assert required in workflow


def test_release_readiness_records_verified_alpha_and_clone_first_next_gate() -> None:
    doc = (ROOT / "docs" / "release-readiness.md").read_text(encoding="utf-8")
    compact_doc = _compact(doc)

    for required in (
        "validated clone-first alpha path",
        "current verified alpha is `v0.1.0a5`",
        "git clone --depth 1 --branch v0.1.0a5 https://github.com/kimyoungjin06/neural-abm.git neural-abm",
        "uv run --no-dev python examples/first_run.py",
        "Smoke clone-first default environment",
        "A direct Git tag install reports matching",
        "The `v0.1.0a3` Git alpha gate is complete",
        "The `v0.1.0a4` Git alpha gate is complete",
        "CI is green on the release tag",
        "`pyproject.toml` and `neural_abm.__version__` both report `0.1.0a3`",
        "The current operational gate is `v0.1.0a5`",
        "The verified tag clone command as the first README and handoff path",
        "Maintainer reproduction tooling for fresh clone and direct Git tag install",
        "Fresh remote clone and direct Git tag install smoke for the `v0.1.0a5` tag",
    ):
        assert required in compact_doc


def test_early_git_user_handoff_documents_surface_boundaries() -> None:
    handoff = (
        ROOT / "docs" / "early-git-user-handoff.md"
    ).read_text(encoding="utf-8")
    compact_handoff = _compact(handoff)
    tag_path = handoff[
        handoff.index("## Current Use Path") : handoff.index(
            "## Main / Next-Alpha Candidate"
        )
    ]
    candidate_path = handoff[
        handoff.index("## Main / Next-Alpha Candidate") : handoff.index(
            "## Stable Surfaces"
        )
    ]
    scenario_command = "uv run --no-dev python examples/research_pivot_scenario_lite.py"

    for required in (
        "Early Git User Handoff",
        "uv run --no-dev python examples/first_run.py",
        "uv run --no-dev python examples/toy_catalog.py",
        "git clone --depth 1 --branch v0.1.0a5 https://github.com/kimyoungjin06/neural-abm.git neural-abm",
        "`neural_abm.api_lite`: torch-free",
        "Intentionally Torch-Backed Surfaces",
        "`neural_abm.api`: stable v0 lifecycle facade",
        "Experimental or Internal Surfaces",
        "What To Report",
        "Minimal Diagnostic Bundle",
        "git rev-parse --short HEAD",
        "direct Git tag install failure",
        "torch_installed=false",
        "v0.1.0a5",
    ):
        assert required in compact_handoff

    assert scenario_command not in tag_path
    assert "is not present in `v0.1.0a5`" in candidate_path
    assert "--branch main" in candidate_path
    assert scenario_command in candidate_path


def test_early_git_issue_template_collects_reproducible_failure_context() -> None:
    template = (
        ROOT / ".github" / "ISSUE_TEMPLATE" / "early-git-user-report.yml"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml"
    ).read_text(encoding="utf-8")

    for required in (
        "Early Git user report",
        "clone-first",
        "git clone --depth 1 --branch v0.1.0a5 https://github.com/kimyoungjin06/neural-abm.git neural-abm",
        "docs/early-git-user-handoff.md",
        "Direct Git tag install",
        "uv run --no-dev python examples/first_run.py",
        "uv run --no-dev python examples/toy_catalog.py",
        "git rev-parse --short HEAD",
        "Did you expect torch-backed behavior?",
        "Default clone-first usage should not install or load torch",
        "v0.1.0a5",
    ):
        assert required in template

    assert "docs/early-git-user-handoff.md" in config


def test_early_git_feedback_loop_design_records_triage_contract() -> None:
    design = (
        ROOT / "docs" / "early-git-feedback-loop-design.md"
    ).read_text(encoding="utf-8")
    compact_design = _compact(design)
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for required in (
        "Early Git Feedback Loop Design",
        "Triage Taxonomy",
        "`docs`",
        "`clone-smoke`",
        "`git-install`",
        "`dependency-profile`",
        "`api-boundary`",
        "`environment`",
        "`unsupported-surface`",
        "tmpdir=$(mktemp -d)",
        "uv run --isolated --no-project --python 3.11",
        "scripts/reproduce_early_git.py",
        "Preferred maintainer helper",
        "Regression Rules",
        "Release Decision",
        "Do not make package upload part of this loop",
        ".github/labels.yml",
        "early-git-maintainer-triage-checklist.md",
        "Implemented Support Steps",
        "Keep `scripts/reproduce_early_git.py` maintainer-only",
    ):
        assert required in compact_design

    assert "early-git-feedback-loop-design.md" in docs_index


def test_early_git_labels_and_maintainer_checklist_match_taxonomy() -> None:
    labels = (ROOT / ".github" / "labels.yml").read_text(encoding="utf-8")
    checklist = (
        ROOT / "docs" / "early-git-maintainer-triage-checklist.md"
    ).read_text(encoding="utf-8")
    compact_checklist = _compact(checklist)
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for label in (
        "early-git",
        "docs",
        "clone-smoke",
        "git-install",
        "dependency-profile",
        "api-boundary",
        "environment",
        "unsupported-surface",
    ):
        assert f"name: {label}" in labels
        assert f"`{label}`" in compact_checklist

    for required in (
        "Early Git Maintainer Triage Checklist",
        "Add exactly one primary taxonomy label",
        "Fresh clone",
        "Direct Git tag install",
        "Preferred helper",
        "uv run python scripts/reproduce_early_git.py --ref v0.1.0a5",
        "Fix Decision",
        "Regression Requirement",
        "Use a new alpha tag only when the fix changes",
        "Close as unsupported",
        "tests/test_clone_first_distribution.py",
    ):
        assert required in compact_checklist

    assert "early-git-maintainer-triage-checklist.md" in docs_index


def test_early_git_reproduction_script_records_maintainer_contract() -> None:
    script = (ROOT / "scripts" / "reproduce_early_git.py").read_text(
        encoding="utf-8"
    )
    scripts_index = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    for required in (
        "Reproduce the early Git user paths",
        "DEFAULT_REPO_URL",
        "--ref",
        "--expected-version",
        "--repo-url",
        "--python",
        "--skip-fresh-clone",
        "--skip-git-install",
        "--keep-temp",
        '"git"',
        '"clone"',
        "--depth",
        "--branch",
        "uv",
        "--no-dev",
        "examples/first_run.py",
        "examples/toy_catalog.py",
        "--isolated",
        "--no-project",
        "neural-abm @ git+",
        "metadata_version",
        "toy_count",
        "torch_installed",
        "torch_loaded",
    ):
        assert required in script

    assert "reproduce_early_git.py" in scripts_index
    assert "maintainer-only triage helper" in scripts_index


def test_v010a5_release_note_records_clone_first_gate() -> None:
    note = (
        ROOT / "docs" / "releases" / "v0.1.0a5.md"
    ).read_text(encoding="utf-8")
    compact_note = _compact(note)
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for required in (
        "`v0.1.0a5` is a clone-first Git alpha release",
        "git clone --depth 1 --branch v0.1.0a5",
        "uv run --no-dev python examples/toy_catalog.py",
        "Smoke clone-first default environment",
        "Kept package upload automation and planning outside the active release path",
        "scripts/reproduce_early_git.py",
        "Package metadata and `neural_abm.__version__` both report `0.1.0a5`",
        "## Troubleshooting",
        "git rev-parse --short HEAD",
        "Open an early Git user report",
    ):
        assert required in compact_note

    assert "releases/v0.1.0a5.md" in docs_index
    assert "releases/v0.1.0a4.md" in docs_index
    assert "releases/v0.1.0a3.md" in docs_index
