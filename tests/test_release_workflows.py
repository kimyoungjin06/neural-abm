from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_publish_workflow_uses_trusted_publishing_boundaries() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )

    for required in (
        "workflow_dispatch:",
        "target:",
        "testpypi",
        "pypi",
        "confirm_pypi",
        "permissions:",
        "contents: read",
        "id-token: write",
        "environment:",
        "name: testpypi",
        "name: pypi",
        "pypa/gh-action-pypi-publish@release/v1",
        "repository-url: https://test.pypi.org/legacy/",
        "refs/tags/v${EXPECTED_VERSION}",
        "publish neural-abm ${EXPECTED_VERSION} to pypi",
        "scripts/inspect_release_artifacts.py --build",
        "scripts/smoke_package_profiles.py",
        "--default-index https://test.pypi.org/simple/",
        "--index https://pypi.org/simple/",
        "--with \"neural-abm==${VERSION}\"",
        "assert payload[\"requires_python\"] == \">=3.11\"",
        "assert payload[\"torch_installed\"] is False",
        "assert payload[\"torch_loaded\"] is False",
    ):
        assert required in workflow
