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
CONFIG = ROOT / "experiments" / "configs" / "pivot_converged_capacity_v2.yaml"
RESULTS = ROOT / "experiments" / "evidence" / "results"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_converged_capacity_v2 as study  # noqa: E402


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
    assert len(plan["source_bindings"]) == 19
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
        (lambda c: c["convergence"]["non_converged"].update(acting_ceiling_share=0.01), "at most 0.1%"),
        (lambda c: c["convergence"]["non_converged"].update(inner_fit="STOP"), "unavailable"),
        (lambda c: c["convergence"]["non_converged"].update(acting_fit="STOP"), "final iterate"),
        (lambda c: c["supersedes"].update(status="RECORDED"), "how v1 ended"),
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
        assert study.v3.select_candidate is not study.ORIGINAL_SELECT_CANDIDATE
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


def _missing(weak: float):  # noqa: ANN202
    """A converge that misses the tolerance whenever the weight decay is the given one."""

    original = study.converge

    def converge(policy, *args, weight_decay, spec):  # noqa: ANN001, ANN202
        prediction, info = original(policy, *args, weight_decay=weight_decay, spec=spec)
        if weight_decay == weak:
            info = {**info, "converged": False}
        return prediction, info

    return converge


def test_an_unconverged_inner_fit_makes_its_candidate_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = study.load_contract(CONFIG)
    settings = study.abm2.load_contract()["batch_learners"]["linear_batch"]
    narrowed = copy.deepcopy(contract)
    narrowed["tuning"]["residual_menu"] = [
        {"training": study.LBFGS, "weight_decay": 1e-6},
        {"training": study.ADAM, "weight_decay": settings["weight_decay"]},
    ]
    features, _labels, agent_ids = _buffer(60, 25)
    # A linear label, so the converged linear fit validates better than 300 Adam steps.
    labels = (features @ np.linspace(-1.0, 1.0, 9) > 0.0).astype(np.float64)
    clean = study.new_log()
    chosen = study.tuned_fit(narrowed, "linear_batch", settings, features, labels, agent_ids, features, 3, clean)
    monkeypatch.setattr(study, "converge", _missing(1e-6))
    log = study.new_log()
    fallback = study.tuned_fit(narrowed, "linear_batch", settings, features, labels, agent_ids, features, 3, log)
    # The converged candidate wins when available; when its first inner fit misses, it is skipped.
    assert dict(clean["selected"]) == {0: 1}
    assert dict(log["selected"]) == {1: 1} and log["unavailable_candidates"] == 1
    assert log["lbfgs_unconverged_inner"] == 1 and log["inner_fits"] == 1 + 4
    assert np.array_equal(fallback, study.abm1.fit_batch("linear_batch", settings, features, labels, features, 3))
    assert not np.array_equal(chosen, fallback)


def test_an_unconverged_acting_fit_keeps_its_final_iterate(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = study.load_contract(CONFIG)
    settings = study.abm2.load_contract()["batch_learners"]["mlp_batch"]
    features, labels, agent_ids = _buffer(30, 10)
    monkeypatch.setattr(study, "converge", _missing(settings["weight_decay"]))
    log = study.new_log()
    prediction = study.train_abm(
        contract, "converged", "mlp_batch", settings, features, labels, agent_ids, features, 5, log
    )
    assert np.all(np.isfinite(prediction))
    assert (log["lbfgs_acting_fits"], log["lbfgs_unconverged_acting"]) == (1, 1)
    check = study.convergence_check(contract, {"abm": {"converged": log}})
    assert check["within_ceiling"] is False
    log["lbfgs_acting_fits"] = 2000
    assert study.convergence_check(contract, {"abm": {"converged": log}})["within_ceiling"] is True


def test_the_policy_layer_skips_unconverged_inner_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = study.load_contract(CONFIG)
    log = study.new_log()
    monkeypatch.setattr(study, "converge", _missing(1e-6))
    with study.candidate_slot(contract, log):
        study.v3.fit_candidate({"id": "matched_linear"}, {"training": study.LBFGS, "weight_decay": 1e-6},
                               *_buffer(20, 8)[:2], np.zeros((3, 9)), 0)
        assert study.v3.select_candidate([np.nan, 0.2, 0.1]) == 2
        assert study.v3.select_candidate([0.05, np.nan, 0.1]) == 0
        acting = study.v3.fit_candidate({"id": "matched_linear"}, {"training": study.LBFGS, "weight_decay": 1e-6},
                                        *_buffer(20, 8)[:2], np.zeros((3, 9)), 0)
    assert np.all(np.isfinite(acting))
    assert log["unavailable_candidates"] == 2
    assert (log["lbfgs_unconverged_inner"], log["lbfgs_unconverged_acting"]) == (1, 1)
    study._check_slots()


def test_the_longest_tasks_run_first() -> None:
    contract = study.load_contract(CONFIG)
    tasks = study.plan_tasks(contract, "smoke", ("converged", "tuned"))
    first = [args[2:] for kind, args in tasks[:18]]
    assert all(row[0] == "tuned" and row[4:] == ("mlp_batch", "field_public") for row in first)
    assert [row[2] for row in first[:6]] == ["immediate"] * 6
