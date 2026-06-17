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

    clone_command = "git clone https://github.com/kimyoungjin06/neural-abm.git"
    tag_install_command = (
        'uv pip install "neural-abm @ '
        "git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a4\""
    )

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
    assert "## Troubleshooting" in readme
    assert "git rev-parse --short HEAD" in readme
    assert "Open an issue with the failed command" in readme


def test_git_distribution_flow_documents_fresh_clone_smoke() -> None:
    doc = (ROOT / "docs" / "git-distribution-flow.md").read_text(encoding="utf-8")

    for required in (
        "primary user path is a fresh clone",
        "git clone https://github.com/kimyoungjin06/neural-abm.git",
        "uv run --no-dev python examples/first_run.py",
        "uv run --no-dev python examples/toy_catalog.py",
        'assert "torch" not in sys.modules',
        "v0.1.0a4",
    ):
        assert required in doc


def test_ci_keeps_clone_first_default_profile_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    for required in (
        "Smoke clone-first default environment",
        "UV_PROJECT_ENVIRONMENT=.venv-clone-smoke",
        "uv run --no-dev python examples/first_run.py",
        "uv run --no-dev python examples/toy_catalog.py",
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
        "git clone https://github.com/kimyoungjin06/neural-abm.git",
        "uv run --no-dev python examples/first_run.py",
        "Smoke clone-first default environment",
        "A direct Git tag install reports matching",
        "The `v0.1.0a3` Git alpha gate is complete",
        "CI is green on the release tag",
        "`pyproject.toml` and `neural_abm.__version__` both report `0.1.0a3`",
        "The current operational gate is `v0.1.0a4`",
        "GitHub Actions on Node 24 compatible action versions",
        "Early Git user handoff guidance",
        "Fresh remote clone and direct Git tag install smoke for the `v0.1.0a4` tag",
    ):
        assert required in compact_doc


def test_early_git_user_handoff_documents_surface_boundaries() -> None:
    handoff = (
        ROOT / "docs" / "early-git-user-handoff.md"
    ).read_text(encoding="utf-8")
    compact_handoff = _compact(handoff)

    for required in (
        "Early Git User Handoff",
        "uv run --no-dev python examples/first_run.py",
        "uv run --no-dev python examples/toy_catalog.py",
        "`neural_abm.api_lite`: torch-free",
        "Intentionally Torch-Backed Surfaces",
        "`neural_abm.api`: stable v0 lifecycle facade",
        "Experimental or Internal Surfaces",
        "What To Report",
        "Minimal Diagnostic Bundle",
        "git rev-parse --short HEAD",
        "direct Git tag install failure",
        "torch_installed=false",
        "v0.1.0a4",
    ):
        assert required in compact_handoff


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
        "Direct Git tag install",
        "uv run --no-dev python examples/first_run.py",
        "uv run --no-dev python examples/toy_catalog.py",
        "git rev-parse --short HEAD",
        "Did you expect torch-backed behavior?",
        "Default clone-first usage should not install or load torch",
        "v0.1.0a4",
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
        "Regression Rules",
        "Release Decision",
        "Do not make package upload part of this loop",
        "Next Implementation Steps",
    ):
        assert required in compact_design

    assert "early-git-feedback-loop-design.md" in docs_index


def test_v010a4_release_note_records_clone_first_gate() -> None:
    note = (
        ROOT / "docs" / "releases" / "v0.1.0a4.md"
    ).read_text(encoding="utf-8")
    compact_note = _compact(note)
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for required in (
        "`v0.1.0a4` is a clone-first Git alpha release",
        "git clone --depth 1 --branch v0.1.0a4",
        "uv run --no-dev python examples/toy_catalog.py",
        "Smoke clone-first default environment",
        "Removed package upload automation and planning",
        "Package metadata and `neural_abm.__version__` both report `0.1.0a4`",
    ):
        assert required in compact_note

    assert "releases/v0.1.0a4.md" in docs_index
    assert "releases/v0.1.0a3.md" in docs_index
