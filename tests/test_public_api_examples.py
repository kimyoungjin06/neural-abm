from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MINIMAL_API_EXAMPLE = ROOT / "examples" / "minimal_api_nabm.py"
FIRST_RUN_EXAMPLE = ROOT / "examples" / "first_run.py"
TOY_CATALOG_EXAMPLE = ROOT / "examples" / "toy_catalog.py"


def _neural_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    neural_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "neural_abm" or alias.name.startswith("neural_abm."):
                    neural_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "neural_abm" or module.startswith("neural_abm."):
                neural_imports.append(module)
    return neural_imports


def test_minimal_api_example_imports_only_stable_facade() -> None:
    neural_imports = _neural_imports(MINIMAL_API_EXAMPLE)
    assert neural_imports == ["neural_abm.api"]


def test_first_run_example_imports_only_api_lite() -> None:
    neural_imports = _neural_imports(FIRST_RUN_EXAMPLE)
    assert neural_imports == ["neural_abm.api_lite"]


def test_toy_catalog_example_imports_only_api_lite() -> None:
    neural_imports = _neural_imports(TOY_CATALOG_EXAMPLE)
    assert neural_imports == ["neural_abm.api_lite"]


def test_minimal_api_example_runs_as_script() -> None:
    completed = subprocess.run(
        [sys.executable, str(MINIMAL_API_EXAMPLE)],
        check=True,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["agent_count"] == 8
    assert payload["steps"] == 5
    assert payload["social_channel"] == "belief_probability"
    assert payload["commit_mode"] == "belief_probability_commit"
    assert len(payload["history"]) == 5
    assert 0.0 <= payload["mean_belief_probability"] <= 1.0
    assert payload["belief_dispersion"] >= 0.0


def test_first_run_example_runs_as_script_without_torch() -> None:
    completed = subprocess.run(
        [sys.executable, str(FIRST_RUN_EXAMPLE)],
        check=True,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "ok"
    assert payload["surface"] == "neural_abm.api_lite"
    assert payload["default_profile"] == "torch-free"
    assert payload["toy_count"] == 10
    assert payload["torch_loaded"] is False
    assert payload["next_example"] == "examples/toy_catalog.py"
    assert payload["binary_probability_toy_count"] == 4
    assert payload["parity_coverage_toy_count"] == 5
    assert payload["recommended_first_toys"] == [
        {"toy": "toy2", "display_name": "Spatial Prisoner's Dilemma"},
        {"toy": "toy4", "display_name": "Public Goods Commons"},
        {"toy": "toy5", "display_name": "Contagion Adoption"},
    ]


def test_toy_catalog_example_runs_as_script_without_torch() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOY_CATALOG_EXAMPLE)],
        check=True,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["toy_count"] == 10
    assert payload["torch_loaded"] is False
    assert payload["binary_probability_toys"] == ["toy2", "toy4", "toy5", "toy9"]
    assert payload["parity_coverage_toys"] == [
        "toy6",
        "toy7",
        "toy8",
        "toy9",
        "toy10",
    ]
    assert payload["market_ecology"]["display_name"] == "Market Ecology Network"
    assert payload["catalog"][0]["toy"] == "toy1"
    assert payload["catalog"][0]["display_name"] == "Neural HK Classification"


def test_minimal_api_example_run_demo_summary() -> None:
    namespace: dict[str, object] = {}
    exec(MINIMAL_API_EXAMPLE.read_text(encoding="utf-8"), namespace)
    run_demo = namespace["run_demo"]

    summary = run_demo(seed=3, steps=4, agent_count=6)

    assert summary["agent_count"] == 6
    assert summary["steps"] == 4
    assert summary["social_channel"] == "belief_probability"
    assert summary["commit_mode"] == "belief_probability_commit"
    assert len(summary["history"]) == 4
    for row in summary["history"]:
        assert row["social_channel"] == "belief_probability"
        assert row["commit_mode"] == "belief_probability_commit"
        assert row["mean_social_update_norm"] >= 0.0
