from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_converged_capacity_v1.yaml"
RESULTS = ROOT / "experiments" / "evidence" / "results"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_converged_capacity_v1 as study  # noqa: E402


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _recorded_view() -> dict:
    abm = json.loads((RESULTS / "pivot_nonlinear_policy_abm_v2_20260928.summary.json").read_text())
    policy = json.loads((RESULTS / "pivot_matched_marginal_discriminator_v3_20260928.summary.json").read_text())
    return {
        "policy_layer": policy["registered_result_disposition"],
        "abm": abm["registered_result_disposition"],
        "post_hoc": {
            "raw_entry_share_mlp_minus_tree_immediate": abm["post_hoc_descriptive"][
                "raw_entry_share_mlp_minus_tree_immediate"
            ]
        },
    }


def test_contract_plan_binds_sources_and_counts_the_tasks() -> None:
    contract = study.load_contract(CONFIG)
    plan = study._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 17
    assert plan["regimes"] == ["registered", "converged", "tuned"]
    # Development: 144 ABM cells times 7, 4 and 6 arm keys, plus 16 policy-layer replicates per regime.
    assert plan["tasks"] == {"development": 144 * 17 + 3 * 16, "smoke": 18 * 17 + 3 * 2}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c["boundaries"].update(new_llm_calls=1), "no LLM calls"),
        (lambda c: c["tuning"]["residual_menu"].pop(2), "nest the registered setting"),
        (lambda c: c["tuning"]["residual_menu"].pop(8), "nest every converged v3 candidate"),
        (lambda c: c["tuning"]["abm_tree_menu"]["candidates"].pop(8), "nest the registered tree"),
        (lambda c: c["tuning"]["abm_tree_menu"]["fixed"].update(leaf_l2=0.1), "leaf penalty"),
        (lambda c: c["tuning"]["abm_selection"].update(minimum_events_per_label=3), "every inner fold"),
        (lambda c: c["training_regimes"][1].update(abm_weight_decay=0.001), "registered ABM weight decay"),
        (lambda c: c["studies"]["abm"].update(recorded_rows_sha256="0" * 64), "row hash"),
        (lambda c: c["training_regimes"].pop(), "regimes changed"),
    ],
)
def test_contract_edits_that_break_the_design_are_rejected(edit, message) -> None:  # noqa: ANN001
    contract = _raw()
    edit(contract)
    with pytest.raises(ValueError, match=message):
        study.validate_contract(contract)


def test_both_slots_are_restored_even_on_error() -> None:
    contract = study.load_contract(CONFIG)
    learner_class = study.training_learner("converged", contract)
    with pytest.raises(ZeroDivisionError), study.learner_slot(learner_class):
        assert study.abm2.ProbedLearner is learner_class
        _ = 1 / 0
    with pytest.raises(ZeroDivisionError), study.candidate_slot(contract, study.new_log()):
        assert study.v3.fit_candidate is not study.ORIGINAL_FIT_CANDIDATE
        _ = 1 / 0
    study._check_slots()


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_converge_reaches_the_tolerance_below_the_adam_objective(kind: str) -> None:
    rng = np.random.default_rng(5)
    features = rng.random((120, 9))
    labels = (features @ np.linspace(-1.0, 1.0, 9) > 0.0).astype(np.float64)
    contract = study.load_contract(CONFIG)
    decay = 1e-4

    def objective(policy: torch.nn.Module) -> float:
        with torch.no_grad():
            inputs = torch.as_tensor(features, dtype=torch.float64)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                policy(inputs), torch.as_tensor(labels)
            )
            return float(loss + 0.5 * decay * sum((p**2).sum() for p in policy.parameters()))

    policy = study.residual_policy(kind, 16, 7)
    prediction, info = study.converge(policy, features, labels, features, weight_decay=decay, spec=contract["convergence"])
    assert info["converged"] and info["max_abs_gradient"] <= 1e-6
    adam = study.residual_policy(kind, 16, 7).to(dtype=torch.float64)
    optimizer = torch.optim.Adam(adam.parameters(), lr=0.02, weight_decay=decay)
    for _ in range(300):
        optimizer.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            adam(torch.as_tensor(features)), torch.as_tensor(labels)
        ).backward()
        optimizer.step()
    assert objective(policy) < objective(adam)
    again, _ = study.converge(
        study.residual_policy(kind, 16, 7), features, labels, features, weight_decay=decay, spec=contract["convergence"]
    )
    assert np.array_equal(prediction, again)


def _buffer(count: int, positives: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(count)
    features = rng.random((count, 9))
    labels = np.zeros(count)
    labels[:positives] = 1.0
    return features, labels, np.arange(count) * 3


@pytest.mark.parametrize("arm_id", ["linear_batch", "tree_batch", "mlp_batch"])
def test_tuning_falls_back_below_the_minimum_and_nests_the_registered_fit(arm_id: str) -> None:
    contract = study.load_contract(CONFIG)
    settings = study.abm2.load_contract()["batch_learners"][arm_id]
    features, labels, agent_ids = _buffer(40, 3)
    log = study.new_log()
    fallback = study.tuned_fit(contract, arm_id, settings, features, labels, agent_ids, features, 11, log)
    registered = study.abm1.fit_batch(arm_id, settings, features, labels, features, 11)
    assert np.array_equal(fallback, registered) and log["fallbacks"] == 1 and log["inner_fits"] == 0
    # With only the registered candidate on the menu, selection returns the registered fit exactly.
    narrowed = copy.deepcopy(contract)
    if arm_id == "tree_batch":
        keys = ("rounds", "learning_rate", "max_depth", "min_samples_leaf")
        narrowed["tuning"]["abm_tree_menu"]["candidates"] = [{key: settings[key] for key in keys}]
    else:
        narrowed["tuning"]["residual_menu"] = [{"training": study.ADAM, "weight_decay": settings["weight_decay"]}]
    features, labels, agent_ids = _buffer(40, 12)
    log = study.new_log()
    tuned = study.tuned_fit(narrowed, arm_id, settings, features, labels, agent_ids, features, 11, log)
    registered = study.abm1.fit_batch(arm_id, settings, features, labels, features, 11)
    assert np.array_equal(tuned, registered)
    assert log["inner_fits"] == 4 and log["acting_fits"] == 1 and dict(log["selected"]) == {0: 1}


def test_smoke_abm_tasks_match_the_unmodified_runner_and_retrain() -> None:
    contract = study.load_contract(CONFIG)
    cell = (0, "immediate", "smooth_interaction_fit")
    registered = study.abm_task((contract, "smoke", "registered", *cell, "mlp_batch", "field_public"))
    assert registered["smoke_parity"] is True
    converged = study.abm_task((contract, "smoke", "converged", *cell, "mlp_batch", "field_public"))
    assert converged["training"]["lbfgs_fits"] == converged["training"]["acting_fits"] > 0
    assert converged["probe_brier"] != registered["probe_brier"]
    neighbor = study.abm_task((contract, "smoke", "tuned", *cell, "linear_batch", "neighbor_local"))
    assert neighbor["training"]["fallbacks"] == neighbor["training"]["acting_fits"] > 0
    assert neighbor["training"]["inner_fits"] == 0
    study._check_slots()


def test_a_smoke_policy_task_matches_v3_through_the_candidate_slot() -> None:
    contract = study.load_contract(CONFIG)
    result = study.policy_task((contract, "smoke", "registered", 0))
    assert result["smoke_parity"] is True
    assert {row["model_id"] for row in result["rows"]} == set(study.v3.MODEL_IDS)
    assert study.policy_candidates(contract, "converged") == [
        {"training": study.LBFGS, "weight_decay": value} for value in (0.0001, 0.001, 0.01)
    ]


def test_every_capacity_claim_holds_on_the_recorded_dispositions() -> None:
    contract = study.load_contract(CONFIG)
    claims = study.evaluate_claims(contract, _recorded_view())
    assert claims["outcome"] == "CLAIMS_HOLD"
    assert len(claims["claims"]) == 9
    assert sum(len(row["checks"]) for row in claims["claims"].values()) == 21


@pytest.mark.parametrize(
    ("pointer", "value", "claim"),
    [
        ("abm/failure_lag_2/geometry_crossover", "MLP_DOMINATES_IN_ABM", "C09_family_choice_depends_on_boundary_geometry"),
        ("abm/immediate/exit_passing_family_geometries", ["tree_batch/smooth_interaction_fit"], "C13_tree_reduces_failed_stock_on_axis_boundaries"),
        ("post_hoc/raw_entry_share_mlp_minus_tree_immediate/axis_interaction_fit/mean", 0.001, "C14_raw_entry_precision_follows_the_crossover"),
        ("policy_layer/neural_specific_advantage", "SUPPORTED_ON_FROZEN_GENERATOR_ONLY", "R01_neural_policy_is_necessary_stays_rejected"),
    ],
)
def test_a_changed_disposition_changes_exactly_its_claim(pointer: str, value, claim) -> None:  # noqa: ANN001
    contract = study.load_contract(CONFIG)
    view = _recorded_view()
    *parents, last = pointer.split("/")
    target = view
    for part in parents:
        target = target[part]
    target[last] = value
    assert study.evaluate_claims(contract, view)["changed"] == [claim]


def test_regime_rows_and_the_study_status() -> None:
    rows = [
        {"regime": regime, "arm_id": arm, "evidence_regime": evidence}
        for regime in study.REGIMES
        for arm, evidence in study.REGIME_ARM_KEYS[regime]
    ]
    converged = study.regime_rows(rows, "converged")
    assert sorted((r["regime"], r["arm_id"]) for r in converged if r["evidence_regime"] != "neighbor_local") == [
        ("converged", "linear_batch"),
        ("converged", "mlp_batch"),
        ("registered", "fixed_rule"),
        ("registered", "tree_batch"),
    ]
    tuned = study.regime_rows(rows, "tuned")
    assert {r["regime"] for r in tuned if r["arm_id"] == "fixed_rule"} == {"registered"}
    assert {r["regime"] for r in tuned if r["arm_id"] != "fixed_rule"} == {"tuned"}
    assert len(tuned) == len(converged) == len(study.ARM_KEYS)
    assert study.STUDY_STATUS[(False, True)] == "O36_HOLD_WHEN_TUNED_ONLY"
