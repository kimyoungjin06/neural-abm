from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_linear_learner_diagnosis_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_linear_learner_diagnosis_v1 as study  # noqa: E402


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_contract_plan_binds_sources_and_lists_the_refits() -> None:
    contract = study.load_contract(CONFIG)
    plan = study._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 16
    assert plan["refits"] == list(study.FITS)
    assert plan["source_arms"] == ["linear_batch", "mlp_batch"]


def test_reused_seeds_llm_calls_and_another_prior_are_rejected() -> None:
    contract = _raw()
    contract["probe_cohort"]["base_seed"] = 20261203
    with pytest.raises(ValueError, match="fresh seed"):
        study.validate_contract(contract)
    contract = _raw()
    contract["boundaries"]["new_llm_calls"] = 1
    with pytest.raises(ValueError, match="no LLM calls"):
        study.validate_contract(contract)
    contract = _raw()
    contract["prior"] = "LLM_BASE_RATE"
    with pytest.raises(ValueError, match="hand rule"):
        study.validate_contract(contract)


def test_the_learner_slot_records_and_is_restored_even_on_error() -> None:
    learner_class = study.recording_learner()
    with pytest.raises(ZeroDivisionError), study.learner_slot(learner_class):
        assert study.abm2.ProbedLearner is learner_class
        _ = 1 / 0
    assert study.abm2.ProbedLearner is study.ORIGINAL_LEARNER
    learner = learner_class(
        arm_id="linear_batch",
        settings={},
        seed_components=(1,),
        eval_features=np.zeros((1, 9)),
        probe_features=np.zeros((2, 9)),
    )
    assert learner_class.created == [learner]
    assert study.RecordingLearner.created == []


def test_the_converged_fit_minimizes_the_registered_objective() -> None:
    rng = np.random.default_rng(3)
    features = rng.random((120, 9))
    labels = (features @ np.linspace(-1.0, 1.0, 9) > 0.0).astype(np.float64)
    contract = study.load_contract(CONFIG)
    fits = {row["id"]: row for row in contract["refits"]}
    converged, diagnostics = study.refit(contract, fits["linear_converged"], features, labels, features, 0)
    registered, _ = study.refit(contract, fits["registered_linear"], features, labels, features, 0)
    assert diagnostics["max_abs_gradient"] <= 1e-6
    assert diagnostics["residual_norm"] > 0.0

    def loss(prediction: np.ndarray) -> float:
        clipped = np.clip(prediction, 1e-15, 1 - 1e-15)
        return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))

    assert loss(converged) < loss(registered)
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("source_arm", ["linear_batch", "mlp_batch"])
def test_a_smoke_task_reproduces_the_run_and_refits_every_fold(source_arm: str) -> None:
    contract = study.load_contract(CONFIG)
    row = study.run_task((contract, "smoke", 0, "linear_fit", source_arm))
    assert row["source_arm"] == source_arm and len(row["folds"]) == 4
    for fold in row["folds"]:
        assert set(fold["fits"]) == set(study.FITS)
        assert fold["evidence_events"] > 0
        assert fold["fits"]["linear_converged"]["max_abs_gradient"] <= 1e-6
    assert row == study.run_task((contract, "smoke", 0, "linear_fit", source_arm))


def _row(replicate: int, world: str, source: str, briers: dict[str, float], wobble: float) -> dict:
    fits = {
        fit_id: {"brier": briers.get(fit_id, 0.05) + wobble, "log_loss": 0.2, "auc": 0.99, "sharpness": 0.4}
        for fit_id in study.FITS
    }
    for fit_id in ("linear_converged", "linear_converged_weak_penalty"):
        fits[fit_id].update({"max_abs_gradient": 1e-9, "residual_norm": 10.0})
    folds = [
        {"fold": f, "evidence_events": 140, "evidence_productive_share": 0.4, "refits": 4, "fits": fits}
        for f in range(4)
    ]
    return {
        "replicate": replicate,
        "world_id": world,
        "source_arm": source,
        "probe_base_rate": 0.3,
        "run_probe_brier": 0.05,
        "run_probe_auc": 0.99,
        "run_cumulative_entry_rate": 0.8,
        "folds": folds,
    }


def _rows(briers: dict[tuple[str, str], dict[str, float]]) -> list[dict]:
    return [
        _row(r, world, source, briers.get((world, source), {}), 0.002 * ((r % 4) - 1.5) * (source == "linear_batch"))
        for r in range(16)
        for world in study.FIT_WORLDS
        for source in study.SOURCE_ARMS
    ]


def test_decisions_separate_the_learner_from_the_evidence_and_the_budget() -> None:
    contract = study.load_contract(CONFIG)
    learner_only = {"registered_linear": 0.06, "registered_mlp": 0.013, "linear_converged": 0.014}
    briers = {("linear_fit", source): learner_only for source in study.SOURCE_ARMS}
    # Smooth world: the evidence matters, and converging closes part of the gap only.
    briers[("smooth_interaction_fit", "linear_batch")] = {
        "registered_linear": 0.18, "registered_mlp": 0.10, "linear_converged": 0.16,
    }
    briers[("smooth_interaction_fit", "mlp_batch")] = {
        "registered_linear": 0.15, "registered_mlp": 0.08, "linear_converged": 0.14,
    }
    # Axis world: only the linear arm's evidence hurts, and converging changes nothing.
    briers[("axis_interaction_fit", "linear_batch")] = {
        "registered_linear": 0.08, "registered_mlp": 0.05, "linear_converged": 0.08,
    }
    decisions = study.decide(_rows(briers), contract)
    assert decisions["o35"] == "O35_LEARNER_OPTIMIZATION_BUDGET"
    linear = decisions["by_world"]["linear_fit"]
    assert linear["learner_gap"]["on_linear_batch_evidence"]["material"] is True
    assert linear["evidence_gap"]["registered_linear"]["material"] is False
    smooth = decisions["by_world"]["smooth_interaction_fit"]
    assert smooth["source_of_the_gap"] == "BOTH"
    assert smooth["optimization"] == "PARTLY_OPTIMIZATION"
    axis = decisions["by_world"]["axis_interaction_fit"]
    assert axis["source_of_the_gap"] == "EVIDENCE" and axis["optimization"] == "NOT_OPTIMIZATION"
    flat = study.decide(_rows({}), contract)["by_world"]["linear_fit"]
    assert flat["source_of_the_gap"] == "NEITHER" and flat["optimization"] == "NO_GAP_TO_EXPLAIN"


def test_development_summary_and_outputs_write(tmp_path: Path) -> None:
    contract = study.load_contract(CONFIG)
    rows = _rows({})
    summary = study.summarize(rows, contract, "development")
    assert summary["decisions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert len(summary["cells"]) == 3 * 2
    smoke = study.summarize(rows[:12], copy.deepcopy(contract), "smoke")
    assert smoke["decisions"]["status"] == "NOT_EVALUATED_SMOKE_PROFILE"
    receipt = study.write_outputs(
        output_dir=tmp_path / "out",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert set(receipt["outputs"]) == {"runs.jsonl", "scorecard.json", "scorecard.md"}
    assert "O35:" in (tmp_path / "out" / "scorecard.md").read_text()


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = study.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        study.delayed.verify_precommit("short", contract)
