from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
RESULT = RESULTS / "pivot_linear_learner_diagnosis_v1_20261029.summary.json"
SUMMARY = RESULTS / "pivot_linear_learner_diagnosis_v1_20261029.summary.md"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _text() -> str:
    return " ".join(SUMMARY.read_text(encoding="utf-8").split())


def _r(value: float, places: str = "0.001", sign: bool = True) -> str:
    text = str(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))
    if text.startswith("-"):
        return text.replace("-", "−")
    return f"+{text}" if sign else text


def _cell(result: dict, world: str, source: str) -> dict:
    return next(c for c in result["cells"] if c["world_id"] == world and c["source_arm"] == source)


def test_reproduced_under_its_freeze_with_every_check_passing() -> None:
    result = _result()
    assert result["precommit_commit"] == "8c7ffd8b4e26625988e7cee4eae6f69614ae1164"
    assert all(result["reproduction"]["byte_identical"].values()) and len(result["reproduction"]["byte_identical"]) == 4
    assert result["checks"] == "PASS_REFIT_PARITY_PROBE_BRIER_PARITY_AND_CONVERGENCE"
    assert result["decision_status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert result["execution"]["tasks"] == 96


def test_the_gap_is_the_learner_and_mostly_its_optimization_budget() -> None:
    result = _result()
    assert result["status"] == "O35_LEARNER_PARTLY_OPTIMIZATION"
    linear = result["decisions"]["by_world"]["linear_fit"]
    assert linear["source_of_the_gap"] == "LEARNER"
    assert all(row["material"] and row["mean"] > 0 for row in linear["learner_gap"].values())
    assert not any(row["material"] for row in linear["evidence_gap"].values())
    assert linear["optimization"] == "PARTLY_OPTIMIZATION"
    assert linear["converged_linear_minus_mlp"]["high"] > 0.005
    assert linear["registered_minus_converged_linear"]["material"] is True
    share = result["post_hoc_descriptive"]["share_of_the_gap_closed_by_convergence_on_the_linear_arms_evidence"]
    assert 0.70 < share < 0.72
    assert result["post_hoc_descriptive"]["weak_penalty_converged_minus_registered_mlp"]["high"] < 0.0
    for world in ("smooth_interaction_fit", "axis_interaction_fit"):
        row = result["decisions"]["by_world"][world]
        assert (row["source_of_the_gap"], row["optimization"]) == ("BOTH", "NOT_OPTIMIZATION")
        assert row["mlp_minus_linear_auc_on_linear_evidence"]["low"] > 0.15
    held = {key: row["held"] for key, row in result["expectations"].items()}
    assert held == {"E1": True, "E2": False, "E3": True}


def test_the_summary_quotes_the_record() -> None:
    result = _result()
    text = _text()
    linear = result["decisions"]["by_world"]["linear_fit"]
    fits = _cell(result, "linear_fit", "linear_batch")["fits"]
    for fit_id, label in (
        ("registered_linear", "registered linear"),
        ("linear_epochs_x10", "linear, 10x epochs"),
        ("linear_learning_rate_x10", "linear, 10x learning rate"),
        ("linear_converged", "linear, converged at the registered penalty"),
        ("linear_converged_weak_penalty", "linear, converged at a 100x weaker penalty (descriptive)"),
        ("registered_mlp", "registered MLP"),
    ):
        row = fits[fit_id]
        quoted = f"| {label} | {_r(row['brier'], '0.0001', False)} | {_r(row['auc'], '0.0001', False)} |"
        assert quoted in text, quoted
    for source, label in (("linear_batch", "linear arm's"), ("mlp_batch", "MLP arm's")):
        gap = linear["learner_gap"][f"on_{source}_evidence"]
        assert f"on the {label} evidence: {_r(gap['mean'])} [{_r(gap['low'])}, {_r(gap['high'])}]" in text
    closure = linear["registered_minus_converged_linear"]
    assert f"lowers the Brier by {_r(closure['mean'])} [{_r(closure['low'])}, {_r(closure['high'])}]" in text
    converged = linear["converged_linear_minus_mlp"]
    assert f"still {_r(converged['mean'], sign=False)} [{_r(converged['low'], sign=False)}, {_r(converged['high'], sign=False)}] worse" in text
    auc = linear["mlp_minus_linear_auc_on_linear_evidence"]
    assert f"by only {_r(auc['mean'], '0.0001', False)} [{_r(auc['low'], '0.0001', False)}, {_r(auc['high'], '0.0001', False)}]" in text
    share = result["post_hoc_descriptive"]["share_of_the_gap_closed_by_convergence_on_the_linear_arms_evidence"]
    assert f"closes {round(share * 100)}% of the gap" in text
    norms = result["post_hoc_descriptive"]["converged_residual_norm_linear_world"]
    assert f"norm is {round(norms['registered_penalty'])}" in text and f"it is {round(norms['weak_penalty'])}" in text
    walls = result["execution"]["wall_seconds"]
    assert f"{round(walls['development'])} s for the development run" in text
    assert f"{round(walls['reproduction'])} s for the reproduction run" in text
    for world, label in (("smooth_interaction_fit", "smooth"), ("axis_interaction_fit", "axis")):
        row = result["decisions"]["by_world"][world]
        own, other = (row["learner_gap"][f"on_{s}_evidence"]["mean"] for s in ("linear_batch", "mlp_batch"))
        moved = -row["registered_minus_converged_linear"]["mean"]
        auc_gap = row["mlp_minus_linear_auc_on_linear_evidence"]["mean"]
        assert f"| {label} | {_r(own)} / {_r(other)} | {_r(moved)} | {_r(auc_gap)} |" in text, label
