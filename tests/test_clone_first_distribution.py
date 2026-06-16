from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_quick_start_prefers_clone_first_flow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick_start = readme.index("## Quick Start")

    clone_command = "git clone https://github.com/kimyoungjin06/neural-abm.git"
    tag_install_command = (
        'uv pip install "neural-abm @ '
        "git+https://github.com/kimyoungjin06/neural-abm.git@v0.1.0a2\""
    )

    assert readme.index(clone_command, quick_start) < readme.index(
        tag_install_command,
        quick_start,
    )
    assert "uv run --no-dev python examples/toy_catalog.py" in readme
    assert "Query the lightweight API from the clone" in readme


def test_git_distribution_flow_documents_fresh_clone_smoke() -> None:
    doc = (ROOT / "docs" / "git-distribution-flow.md").read_text(encoding="utf-8")

    for required in (
        "primary pre-PyPI user path is a fresh clone",
        "git clone https://github.com/kimyoungjin06/neural-abm.git",
        "uv run --no-dev python examples/toy_catalog.py",
        'assert "torch" not in sys.modules',
    ):
        assert required in doc


def test_ci_keeps_clone_first_default_profile_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    for required in (
        "Smoke clone-first default environment",
        "UV_PROJECT_ENVIRONMENT=.venv-clone-smoke",
        "uv run --no-dev python examples/toy_catalog.py",
        'assert payload["requires_python"] == ">=3.11"',
        'assert payload["torch_loaded"] is False',
        'assert payload["toy_count"] == 10',
    ):
        assert required in workflow
