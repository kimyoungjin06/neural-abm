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
RESEARCH_PIVOT_SCENARIO_LITE_EXAMPLE = (
    ROOT / "examples" / "research_pivot_scenario_lite.py"
)
RESEARCH_PIVOT_STUDY_EXAMPLE = ROOT / "examples" / "research_pivot_study.py"
RESEARCH_PIVOT_LEARNING_STUDY_EXAMPLE = (
    ROOT / "examples" / "research_pivot_learning_study.py"
)


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


def test_research_pivot_scenario_lite_example_imports_only_api_lite() -> None:
    neural_imports = _neural_imports(RESEARCH_PIVOT_SCENARIO_LITE_EXAMPLE)
    assert neural_imports == ["neural_abm.api_lite"]


def test_research_pivot_study_example_imports_only_api_lite() -> None:
    neural_imports = _neural_imports(RESEARCH_PIVOT_STUDY_EXAMPLE)
    assert neural_imports == ["neural_abm.api_lite"]


def test_research_pivot_learning_study_example_imports_only_api() -> None:
    neural_imports = _neural_imports(RESEARCH_PIVOT_LEARNING_STUDY_EXAMPLE)
    assert neural_imports == ["neural_abm.api"]


def test_research_pivot_learning_study_quick_run() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RESEARCH_PIVOT_LEARNING_STUDY_EXAMPLE),
            "--quick",
        ],
        check=True,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "ok"
    assert payload["surface"] == "neural_abm.api"
    assert payload["default_profile"] == "torch-backed"
    assert payload["torch_loaded"] is True
    assert payload["config"]["replicates"] == 4
    assert set(payload["scenarios"]) == {
        "baseline",
        "interdisciplinary_seed_grants",
        "hot_field_hype",
    }
    for arms in payload["scenarios"].values():
        assert set(arms) == {"frozen", "imitative", "cautionary"}
        for summary in arms.values():
            assert len(summary["cumulative_failed_rate_mean"]) == 6
            assert len(summary["mean_attention_weight"]) == 6
    arms_in_comparisons = {row["arm"] for row in payload["comparisons"]}
    assert arms_in_comparisons == {"imitative - frozen", "cautionary - frozen"}


def test_research_pivot_study_quick_run_without_torch() -> None:
    completed = subprocess.run(
        [sys.executable, str(RESEARCH_PIVOT_STUDY_EXAMPLE), "--quick"],
        check=True,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "ok"
    assert payload["surface"] == "neural_abm.scenario_lite"
    assert payload["default_profile"] == "torch-free"
    assert payload["torch_loaded"] is False
    assert payload["outcome_field"] == "productive_pivot_rate"
    assert payload["config"]["replicates"] == 10
    assert {row["scenario"] for row in payload["comparisons"]} == {
        "interdisciplinary_seed_grants",
        "hot_field_hype",
        "hype_with_support",
    }
    for row in payload["comparisons"]:
        assert row["replicates"] == 10
        assert len(row["delta_ci95"]) == 2
    baseline = payload["scenarios"]["baseline"]
    assert len(baseline["outcome_values"]) == 10
    assert "productive_share_of_pivots" in baseline["aggregate_summaries"]


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


def test_research_pivot_scenario_lite_example_runs_as_script_without_torch() -> None:
    completed = subprocess.run(
        [sys.executable, str(RESEARCH_PIVOT_SCENARIO_LITE_EXAMPLE)],
        check=True,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "ok"
    assert payload["surface"] == "neural_abm.scenario_lite"
    assert payload["base_surface"] == "neural_abm.api_lite"
    assert payload["default_profile"] == "torch-free"
    assert payload["torch_loaded"] is False
    assert payload["research_question"] == (
        "should_researchers_pivot_under_different_scientific_environments"
    )
    assert payload["state_field"] == "pivot_readiness"
    assert payload["outcome_field"] == "productive_pivot_count"
    assert payload["baseline"] == "baseline"
    assert payload["steps"] == 3
    assert payload["scenario_count"] == 3

    comparisons = {row["scenario"]: row for row in payload["comparisons"]}
    assert comparisons["interdisciplinary_seed_grants"] == {
        "baseline": "baseline",
        "scenario": "interdisciplinary_seed_grants",
        "outcome_field": "productive_pivot_count",
        "baseline_value": 0.0,
        "scenario_value": 3.0,
        "delta": 3.0,
        "success_criterion": "delta > 2",
        "success": True,
    }
    assert comparisons["hot_field_hype"] == {
        "baseline": "baseline",
        "scenario": "hot_field_hype",
        "outcome_field": "productive_pivot_count",
        "baseline_value": 0.0,
        "scenario_value": 2.0,
        "delta": 2.0,
        "success_criterion": "delta > 2",
        "success": False,
    }

    summaries = payload["scenario_summaries"]
    baseline = summaries["baseline"]["final_aggregate"]
    supported = summaries["interdisciplinary_seed_grants"]["final_aggregate"]
    hype = summaries["hot_field_hype"]["final_aggregate"]
    assert baseline["agent_count"] == 8
    assert baseline["productive_pivot_count"] == 0
    assert supported["pivot_count"] == 3
    assert supported["productive_pivot_count"] == 3
    assert hype["pivot_count"] == 3
    assert hype["productive_pivot_count"] == 2
    assert hype["failed_pivot_count"] == 1
    assert summaries["baseline"]["history_steps"] == 3
    assert len(summaries["interdisciplinary_seed_grants"]["micro_audit_sample"]) == 2
    assert {
        "agent_id",
        "career_stage",
        "scenario",
        "skill_distance",
        "resource_security",
        "network_support",
        "field_opportunity",
        "reputation_risk",
        "funding_signal",
        "attention_signal",
        "peer_success_signal",
        "pivot_pressure",
        "productive_fit",
        "pivot_readiness",
        "pivoted",
        "productive_pivot",
        "peer_count",
        "local_shift",
        "social_shift",
    } <= set(summaries["interdisciplinary_seed_grants"]["micro_audit_sample"][0])


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
