from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_nonlinear_policy_abm_v2.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_nonlinear_policy_abm_v2 as abm  # noqa: E402


def _raw_contract() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_contract_plan_binds_sources_and_the_full_grid() -> None:
    contract = abm.load_contract(CONFIG)
    plan = abm._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 11
    assert len(plan["arm_keys"]) == 10
    assert plan["visibility_regimes"] == ["immediate", "failure_lag_2", "failure_lag_4"]
    assert contract["purpose"]["result_informed_successor"] is True
    assert len(contract["purpose"]["corrects_v1_defects"]) == 4


def test_field_record_must_exclude_own_outcome() -> None:
    contract = _raw_contract()
    contract["evidence_regimes"][1]["includes_own_public_outcome"] = True
    with pytest.raises(ValueError, match="own outcome"):
        abm.validate_contract(contract)


def test_opened_v1_seed_namespaces_are_rejected() -> None:
    contract = _raw_contract()
    contract["probe_cohort"]["base_seed"] = 20261003
    with pytest.raises(ValueError, match="fresh seed"):
        abm.validate_contract(contract)


def test_learner_settings_must_equal_v1() -> None:
    contract = _raw_contract()
    contract["batch_learners"]["tree_batch"]["rounds"] = 100
    with pytest.raises(ValueError, match="differs from v1"):
        abm.validate_contract(contract)


@pytest.mark.parametrize("world", abm.FIT_WORLDS)
def test_probe_cohort_is_deterministic_and_uses_the_world_rule(world: str) -> None:
    contract = abm.load_contract(CONFIG)
    features, labels = abm.probe_cohort(contract, 2, world)
    again, labels_again = abm.probe_cohort(contract, 2, world)
    np.testing.assert_array_equal(features, again)
    np.testing.assert_array_equal(labels, labels_again)
    linear_features, linear_labels = abm.probe_cohort(contract, 2, "linear_fit")
    np.testing.assert_array_equal(features, linear_features)
    assert labels.sum() == linear_labels.sum()
    assert features.shape == (256, 9)
    other, _labels = abm.probe_cohort(contract, 3, world)
    assert not np.array_equal(features, other)


def test_probe_cohort_is_not_the_simulated_population() -> None:
    contract = abm.load_contract(CONFIG)
    config = abm.study_config(contract, "development")
    context = abm.SimpleNamespace(
        scenario=abm.abm1._scenario(contract),
        replicate=0,
        rng=abm.study1.component_rng(config.base_seed, 0, "population"),
    )
    population = abm.study1.make_researchers(
        context,
        abm.study1.StudyConfig(agent_count=config.agent_count, base_seed=config.base_seed),
    )
    simulated = np.stack([abm.study2._feature_vector(row) for row in population])
    probe, _labels = abm.probe_cohort(contract, 0, "linear_fit")
    assert not np.any(np.all(np.isclose(simulated[:, None, :], probe[None, :, :]), axis=2))


def _events(rng: np.random.Generator, agent_ids: range) -> list:
    return [
        abm.study2.PivotEvent(
            agent_id=agent_id,
            features=rng.random(9),
            productive=bool(agent_id % 3 == 0),
            step=3,
        )
        for agent_id in agent_ids
    ]


@pytest.mark.parametrize("arm", abm.BATCH_ARMS)
def test_probed_learner_starts_at_prior_and_excludes_agents(arm: str) -> None:
    contract = abm.load_contract(CONFIG)
    rng = np.random.default_rng(4)
    eval_features, probe_features = rng.random((3, 9)), rng.random((7, 9))
    learner = abm.ProbedLearner(
        arm_id=arm,
        settings=contract["batch_learners"][arm],
        seed_components=(1, 2),
        eval_features=eval_features,
        probe_features=probe_features,
        excluded_agent_ids=frozenset({0, 1, 2}),
    )
    np.testing.assert_array_equal(learner.output, abm.abm1._prior_output(eval_features))
    np.testing.assert_array_equal(
        learner.probe_output, abm.abm1._prior_output(probe_features)
    )
    learner.update(_events(rng, range(3)))
    assert learner.refits == 0
    learner.update(_events(rng, range(12)))
    assert learner.trained_agent_ids == list(range(3, 12))
    assert learner.output.shape == (3,)
    assert learner.probe_output.shape == (7,)


def test_cross_fit_field_routes_agents_to_their_fold_and_checks_isolation() -> None:
    contract = abm.load_contract(CONFIG)
    rng = np.random.default_rng(5)
    features = rng.random((8, 9))
    learners = [
        abm.ProbedLearner(
            arm_id="linear_batch",
            settings=contract["batch_learners"]["linear_batch"],
            seed_components=(fold,),
            eval_features=features,
            probe_features=features[:2],
            excluded_agent_ids=frozenset(i for i in range(8) if i % 4 == fold),
        )
        for fold in range(4)
    ]
    record = abm.CrossFitField(learners=learners)
    record.update(_events(rng, range(8)))
    record.check_isolation()
    for agent_id in range(8):
        assert record.output_for(agent_id) == float(learners[agent_id % 4].output[agent_id])
    learners[1].trained_agent_ids.append(5)
    with pytest.raises(RuntimeError, match="saw its own fold"):
        record.check_isolation()


def test_smoke_task_nests_predecessor_and_scores_every_arm_on_probes() -> None:
    contract = abm.load_contract(CONFIG)
    rows = abm.run_task(contract, "smoke", 0, "failure_lag_2", "linear_fit")
    assert [(row["arm_id"], row["evidence_regime"]) for row in rows] == list(
        abm.ARM_KEYS
    )
    assert all(0.0 <= row["probe_brier"] <= 1.0 for row in rows)
    field_rows = [row for row in rows if row["evidence_regime"] == "field_public"]
    public = [
        sum(row["public_success_per_step"]) + sum(row["public_failure_per_step"])
        for row in field_rows
    ]
    # Each of four fold learners sees the public events outside its fold.
    for row, events in zip(field_rows, public, strict=True):
        assert row["evidence_events_per_learner"] == pytest.approx(0.75 * events, abs=3)


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = abm.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    reduced["profiles"]["smoke"]["steps"] = 6
    serial = abm.run_abm(reduced, "smoke", workers=1)
    parallel = abm.run_abm(reduced, "smoke", workers=3)
    assert len(serial) == 3 * 3 * len(abm.ARM_KEYS)
    assert serial == parallel


def _rows(overrides: dict) -> list[dict]:
    base = {
        "probe_brier": 0.20,
        "productive_share_of_entrants": 0.30,
        "active_failed_rate": 0.05,
        "cumulative_entry_rate": 0.80,
    }
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for visibility in abm.VISIBILITY_IDS:
            for world in abm.FIT_WORLDS:
                for arm, evidence in abm.ARM_KEYS:
                    row = {
                        "replicate": replicate,
                        "visibility_id": visibility,
                        "world_id": world,
                        "arm_id": arm,
                        "evidence_regime": evidence,
                    }
                    for metric, value in base.items():
                        value = overrides.get((visibility, world, arm, evidence, metric), value)
                        row[metric] = value if metric == "cumulative_entry_rate" else value + wobble
                    rows.append(row)
    return rows


def _set(overrides: dict, visibility: str, metric: str, value: float, *, worlds, arms, evidence="field_public") -> None:
    for world in worlds:
        for arm in arms:
            overrides[(visibility, world, arm, evidence, metric)] = value


def test_decisions_separate_learning_entry_saturation_and_exit() -> None:
    contract = abm.load_contract(CONFIG)
    overrides: dict = {}
    # Field MLP learns smooth and tree learns axis, out of sample.
    _set(overrides, "immediate", "probe_brier", 0.12, worlds=["smooth_interaction_fit"], arms=["mlp_batch"])
    _set(overrides, "immediate", "probe_brier", 0.16, worlds=["smooth_interaction_fit"], arms=["tree_batch"])
    _set(overrides, "immediate", "probe_brier", 0.12, worlds=["axis_interaction_fit"], arms=["tree_batch"])
    _set(overrides, "immediate", "probe_brier", 0.16, worlds=["axis_interaction_fit"], arms=["mlp_batch"])
    _set(overrides, "immediate", "productive_share_of_entrants", 0.35, worlds=abm.INTERACTION_FIT_WORLDS, arms=["tree_batch"])
    # Under the longest delay every feature arm enters fully; exit still differs.
    _set(overrides, "failure_lag_4", "cumulative_entry_rate", 1.0, worlds=abm.FIT_WORLDS, arms=abm.FEATURE_ARMS)
    _set(overrides, "failure_lag_4", "active_failed_rate", 0.02, worlds=abm.INTERACTION_FIT_WORLDS, arms=["mlp_batch"])
    decision = abm.decide(_rows(overrides), contract)["by_visibility"]
    immediate = decision["immediate"]
    assert immediate["field_nonlinear_learning"] == "OUT_OF_SAMPLE_NONLINEAR_LEARNING"
    assert immediate["geometry_crossover"] == "CROSSOVER_TRANSFERS_TO_ABM"
    assert immediate["evidence_scale"] == "EVIDENCE_SCALE_GATES_USEFUL_LEARNING"
    assert immediate["entry"] == "ENTRY_CHANNEL_ACTIVE"
    assert immediate["entry_passing_family_geometries"] == [
        "tree_batch/axis_interaction_fit",
        "tree_batch/smooth_interaction_fit",
    ]
    assert immediate["exit"] == "EXIT_CHANNEL_ABSENT"
    lag4 = decision["failure_lag_4"]
    assert lag4["entry"] == "ENTRY_SATURATED"
    assert lag4["exit"] == "EXIT_CHANNEL_ACTIVE"
    assert lag4["field_nonlinear_learning"] == "NO_NONLINEAR_LEARNING"
    assert lag4["evidence_scale"] == "NO_USEFUL_LEARNING_UNDER_FIELD_EVIDENCE"
    assert decision["failure_lag_2"]["entry"] == "ENTRY_CHANNEL_ABSENT"


def test_useful_learning_uses_the_fixed_prior_as_reference() -> None:
    contract = abm.load_contract(CONFIG)
    overrides: dict = {}
    # Local learners beat the prior; a linear reference that degrades is irrelevant.
    _set(overrides, "failure_lag_2", "probe_brier", 0.15, worlds=abm.INTERACTION_FIT_WORLDS, arms=["tree_batch"], evidence="neighbor_local")
    _set(overrides, "failure_lag_2", "probe_brier", 0.15, worlds=abm.INTERACTION_FIT_WORLDS, arms=["tree_batch"])
    _set(overrides, "failure_lag_2", "probe_brier", 0.40, worlds=abm.INTERACTION_FIT_WORLDS, arms=["linear_batch"], evidence="neighbor_local")
    decision = abm.decide(_rows(overrides), contract)["by_visibility"]["failure_lag_2"]
    assert decision["evidence_scale"] == "USEFUL_LEARNING_SURVIVES_LOCAL_EVIDENCE"
    assert decision["useful_learners_local"] == [
        "tree_batch/axis_interaction_fit",
        "tree_batch/smooth_interaction_fit",
    ]
    assert decision["geometry_crossover"] == "TREE_DOMINATES_IN_ABM"


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
    assert set(summary["dispositions"]["by_visibility"]) == set(abm.VISIBILITY_IDS)
    assert len(summary["cells"]) == 3 * 3 * len(abm.ARM_KEYS)
    receipt = abm.write_outputs(
        output_dir=tmp_path / "out",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert receipt["row_count"] == 1440
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == sorted(
        [*receipt["outputs"], "receipt.json"]
    )


def test_family_transfer_check_runs_on_fresh_seeds() -> None:
    contract = abm.load_contract(CONFIG)
    transfer = abm.abm1.family_transfer_check(contract)
    assert transfer["status"] in {"FAMILY_TRANSFERRED", "FAMILY_NOT_TRANSFERRED"}
    assert len(transfer["winning_replicates"]) == 6


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = abm.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        abm.delayed.verify_precommit("short", contract)
