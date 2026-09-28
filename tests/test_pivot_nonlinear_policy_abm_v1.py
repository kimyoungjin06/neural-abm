from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_nonlinear_policy_abm_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_nonlinear_policy_abm_v1 as abm  # noqa: E402


def _population(contract: dict, replicate: int = 0) -> list:
    config = abm.study_config(contract, "development")
    context = SimpleNamespace(
        scenario=abm._scenario(contract),
        replicate=replicate,
        rng=abm.study1.component_rng(config.base_seed, replicate, "population"),
    )
    population_config = abm.study1.StudyConfig(
        agent_count=config.agent_count, base_seed=config.base_seed
    )
    return abm.study1.make_researchers(context, population_config)


def test_contract_plan_binds_sources_and_the_full_arm_grid() -> None:
    contract = abm.load_contract(CONFIG)
    plan = abm._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 11
    assert len(plan["arm_keys"]) == 10
    assert contract["boundaries"]["empirical_rows_read"] == 0
    assert contract["decision_contract"]["model_selection"] == "FORBIDDEN"


def test_changed_arm_grid_is_rejected() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    contract["policy_arms"][2]["evidence_regimes"] = ["field_public"]
    with pytest.raises(ValueError, match="policy arms changed"):
        abm.validate_contract(contract)


def test_linear_world_is_the_untouched_predecessor_population() -> None:
    contract = abm.load_contract(CONFIG)
    researchers = _population(contract)
    assert abm.apply_fit_world(researchers, "linear_fit") == researchers


@pytest.mark.parametrize("world", ["smooth_interaction_fit", "axis_interaction_fit"])
def test_rank_matching_keeps_fit_values_and_changes_who_is_productive(world: str) -> None:
    contract = abm.load_contract(CONFIG)
    researchers = _population(contract)
    matched = abm.apply_fit_world(researchers, world)
    linear = [row.productive_fit for row in researchers]
    assigned = [row.productive_fit for row in matched]
    assert sorted(linear) == sorted(assigned)
    threshold = contract["inherited_learning_parameters"]["productive_threshold"]
    linear_set = {row.agent_id for row in researchers if row.productive_fit >= threshold}
    world_set = {row.agent_id for row in matched if row.productive_fit >= threshold}
    assert len(linear_set) == len(world_set)
    assert linear_set != world_set
    features = np.stack([abm.study2._feature_vector(row) for row in researchers])
    score = abm.world_score(features, world)
    ranked = np.argsort(-score, kind="stable")
    assert matched[ranked[0]].productive_fit == max(linear)
    assert all(
        getattr(matched[0], name) == getattr(researchers[0], name)
        for name in ("resource_security", "network_support", "career_stage")
    )


def test_prior_boosted_trees_start_at_the_prior_and_handle_one_class() -> None:
    rng = np.random.default_rng(0)
    features = rng.random((40, 9))
    prior = abm._prior_output(features)
    settings = {
        "rounds": 0,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "leaf_l2": 1.0,
    }
    untrained = abm.fit_prior_boosted_trees(
        features, np.ones(40), features, settings, 1
    )
    np.testing.assert_array_equal(untrained, prior)
    one_class = abm.fit_prior_boosted_trees(
        features[:3], np.zeros(3), features, {**settings, "rounds": 20}, 1
    )
    assert np.all(one_class < prior)


def test_prior_boosted_trees_learn_an_axis_aligned_rule() -> None:
    rng = np.random.default_rng(1)
    features = rng.random((200, 9))
    labels = ((features[:, 1] > 0.5) == (features[:, 2] > 0.5)).astype(float)
    contract = abm.load_contract(CONFIG)
    prediction = abm.fit_prior_boosted_trees(
        features, labels, features, contract["batch_learners"]["tree_batch"], 3
    )
    prior = abm._prior_output(features)
    assert np.mean((prediction - labels) ** 2) < 0.5 * np.mean((prior - labels) ** 2)


@pytest.mark.parametrize("arm", abm.BATCH_ARMS)
def test_batch_learner_starts_at_prior_and_refits_deterministically(arm: str) -> None:
    contract = abm.load_contract(CONFIG)
    rng = np.random.default_rng(2)
    eval_features = rng.random((5, 9))
    events = [
        abm.study2.PivotEvent(
            agent_id=index, features=rng.random(9), productive=bool(index % 2), step=3
        )
        for index in range(12)
    ]
    learners = [
        abm.BatchLearner(
            arm_id=arm,
            settings=contract["batch_learners"][arm],
            seed_components=(1, 2, 3),
            eval_features=eval_features,
        )
        for _copy in range(2)
    ]
    np.testing.assert_array_equal(learners[0].output, abm._prior_output(eval_features))
    for learner in learners:
        learner.update(events[:6])
        learner.update([])
        learner.update(events[6:])
    assert learners[0].refits == 2
    assert len(learners[0].labels) == 12
    np.testing.assert_array_equal(learners[0].output, learners[1].output)


def test_smoke_task_nests_the_persistence_predecessor() -> None:
    contract = abm.load_contract(CONFIG)
    rows = abm.run_task(contract, "smoke", 0, "failure_lag_4", "linear_fit")
    assert [(row["arm_id"], row["evidence_regime"]) for row in rows] == list(
        abm.ARM_KEYS
    )
    field_rows = [row for row in rows if row["evidence_regime"] == "field_public"]
    public = [
        sum(row["public_success_per_step"]) + sum(row["public_failure_per_step"])
        for row in field_rows
    ]
    assert [row["evidence_events_per_learner"] for row in field_rows] == public


def test_nesting_check_detects_a_mismatch() -> None:
    contract = abm.load_contract(CONFIG)
    config = abm.study_config(contract, "smoke")
    runs = {
        arm: abm.run_arm(
            contract,
            config,
            replicate=0,
            visibility_id="immediate",
            world_id="linear_fit",
            arm_id=arm,
            evidence_regime=evidence,
        )
        for arm, evidence in abm.ARM_KEYS[:2]
    }
    runs["online_logistic"] = {**runs["online_logistic"], "active_rate": -1.0}
    with pytest.raises(RuntimeError, match="does not nest"):
        abm.check_nesting(
            contract, config, replicate=0, visibility_id="immediate", runs=runs
        )


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = abm.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    reduced["profiles"]["smoke"]["steps"] = 6
    serial = abm.run_abm(reduced, "smoke", workers=1)
    parallel = abm.run_abm(reduced, "smoke", workers=3)
    assert len(serial) == 2 * 3 * len(abm.ARM_KEYS)
    assert serial == parallel


def _synthetic_rows(briers: dict, shares: dict) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for visibility in abm.VISIBILITY_IDS:
            for world in abm.FIT_WORLDS:
                for arm, evidence in abm.ARM_KEYS:
                    key = (world, arm, evidence)
                    rows.append(
                        {
                            "replicate": replicate,
                            "visibility_id": visibility,
                            "world_id": world,
                            "arm_id": arm,
                            "evidence_regime": evidence,
                            "policy_brier": briers.get(key, 0.20) + wobble,
                            "productive_share_of_entrants": shares.get(key, 0.30)
                            + wobble,
                        }
                    )
    return rows


def test_decision_separates_learning_from_behavior_and_evidence_scale() -> None:
    contract = abm.load_contract(CONFIG)
    briers = {
        ("smooth_interaction_fit", "mlp_batch", "field_public"): 0.12,
        ("smooth_interaction_fit", "tree_batch", "field_public"): 0.16,
        ("axis_interaction_fit", "tree_batch", "field_public"): 0.12,
        ("axis_interaction_fit", "mlp_batch", "field_public"): 0.16,
    }
    decision = abm.decide(_synthetic_rows(briers, {}), contract)
    assert decision["result"] == "NONLINEAR_LEARNING_CHANGES_BUT_BEHAVIOR_DOES_NOT"
    assert decision["abm_geometry_crossover"] == "CROSSOVER_TRANSFERS_TO_ABM"
    assert decision["evidence_scale"] == "EVIDENCE_SCALE_GATES_NONLINEAR_LEARNING"
    assert "mlp_batch/smooth_interaction_fit" in decision[
        "learned_family_geometries_field"
    ]
    shares = {("smooth_interaction_fit", "mlp_batch", "field_public"): 0.35}
    decision = abm.decide(_synthetic_rows(briers, shares), contract)
    assert decision["result"] == "NONLINEAR_LEARNING_AND_BEHAVIOR_BOTH_CHANGE"
    assert decision["behavior_changing_family_geometries_field"] == [
        "mlp_batch/smooth_interaction_fit"
    ]


def test_decision_reports_no_learning_and_local_survival() -> None:
    contract = abm.load_contract(CONFIG)
    decision = abm.decide(_synthetic_rows({}, {}), contract)
    assert decision["result"] == "NO_NONLINEAR_LEARNING_IN_ABM"
    assert decision["abm_geometry_crossover"] == "TIED_OR_MIXED"
    briers = {
        (world, "tree_batch", evidence): 0.12
        for world in abm.INTERACTION_FIT_WORLDS
        for evidence in abm.EVIDENCE_REGIMES
    }
    decision = abm.decide(_synthetic_rows(briers, {}), contract)
    assert decision["evidence_scale"] == "NONLINEAR_LEARNING_SURVIVES_LOCAL_EVIDENCE"
    assert decision["abm_geometry_crossover"] == "TREE_DOMINATES_IN_ABM"


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = abm.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        abm.delayed.verify_precommit("short", contract)


def test_family_transfer_check_reports_every_condition() -> None:
    contract = abm.load_contract(CONFIG)
    transfer = abm.family_transfer_check(contract)
    assert transfer["status"] in {"FAMILY_TRANSFERRED", "FAMILY_NOT_TRANSFERRED"}
    assert len(transfer["winning_replicates"]) == 6
    assert len(transfer["mean_brier"]) == 3 * len(abm.BATCH_ARMS)


def test_development_summary_and_outputs_write_from_full_shaped_rows(
    tmp_path: Path,
) -> None:
    contract = abm.load_contract(CONFIG)
    template = abm.run_task(contract, "smoke", 0, "immediate", "linear_fit")
    rows = [
        {**row, "replicate": replicate, "visibility_id": visibility, "world_id": world}
        for replicate in range(16)
        for visibility in abm.VISIBILITY_IDS
        for world in abm.FIT_WORLDS
        for row in template
    ]
    transfer = {"status": "FAMILY_TRANSFERRED", "winning_replicates": {}, "mean_brier": {}}
    summary = abm.summarize(rows, transfer, contract, "development")
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert len(summary["cells"]) == 2 * 3 * len(abm.ARM_KEYS)
    receipt = abm.write_outputs(
        output_dir=tmp_path / "out",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert receipt["row_count"] == 960
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == sorted(
        [*receipt["outputs"], "receipt.json"]
    )
    with pytest.raises(FileExistsError):
        abm.write_outputs(
            output_dir=tmp_path / "out",
            rows=rows,
            summary=summary,
            precommit={},
            contract_path=CONFIG,
        )
