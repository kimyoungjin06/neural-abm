from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_nonlinear_policy_abm_v2_20260928.summary.json"
)
VISIBILITY = ("immediate", "failure_lag_2", "failure_lag_4")
INTERACTION = ("smooth_interaction_fit", "axis_interaction_fit")


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_nested_isolated_and_reproduced() -> None:
    result = _result()
    assert result["status"] == "SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED"
    assert result["precommit"]["commit_sha"] == (
        "c56d74112af6b962e3b704b724a8cf94ffeb0e34"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 1440
    assert execution["deterministic_rerun"]["all_five_artifacts_byte_identical"] is True
    assert execution["first_run"]["receipt_sha256"] == (
        execution["deterministic_rerun"]["receipt_sha256"]
    )
    validation = result["validation"]
    assert validation["nesting_and_fold_isolation_checks"] == "PASS_EXACT"
    assert validation["family_transfer"] == "FAMILY_TRANSFERRED"
    assert result["boundaries"]["model_selected"] == "NO"


def test_learning_crossover_and_evidence_scale_hold_in_every_regime() -> None:
    registered = _result()["registered_result_disposition"]
    for visibility in VISIBILITY:
        entry = registered[visibility]
        assert entry["field_nonlinear_learning"] == "OUT_OF_SAMPLE_NONLINEAR_LEARNING"
        assert len(entry["learned_family_geometries"]) == 4
        assert entry["geometry_crossover"] == "CROSSOVER_TRANSFERS_TO_ABM"
        assert entry["evidence_scale"] == "EVIDENCE_SCALE_GATES_USEFUL_LEARNING"
        assert len(entry["useful_learners_field"]) == 6
        assert entry["useful_learners_local"] == []


def test_entry_is_active_only_immediately_and_saturates_under_delay() -> None:
    result = _result()
    registered = result["registered_result_disposition"]
    assert registered["immediate"]["entry"] == "ENTRY_CHANNEL_ACTIVE"
    assert registered["immediate"]["entry_passing_family_geometries"] == [
        "tree_batch/axis_interaction_fit",
        "tree_batch/smooth_interaction_fit",
    ]
    full_entry = result["post_hoc_descriptive"]["field_full_entry_runs"]
    for visibility in ("failure_lag_2", "failure_lag_4"):
        assert registered[visibility]["entry"] == "ENTRY_SATURATED"
        for arm in ("linear_batch", "tree_batch", "mlp_batch"):
            assert full_entry[visibility][arm] == 48
    for arm in ("linear_batch", "tree_batch", "mlp_batch"):
        assert full_entry["immediate"][arm] == 0


def test_every_local_feature_learner_is_worse_than_the_prior() -> None:
    decisions = _result()["registered_decisions"]
    for visibility in VISIBILITY:
        local = decisions[visibility]["useful_learning"]["neighbor_local"]
        field = decisions[visibility]["useful_learning"]["field_public"]
        for key, report in local.items():
            assert report["positive_replicates"] == 0
            assert report["mean"] < -0.05
            assert field[key]["positive_replicates"] == 16


def test_raw_behavior_follows_the_crossover_and_mlp_gains_are_not_specific() -> None:
    result = _result()
    post = result["post_hoc_descriptive"]
    raw = post["raw_entry_share_mlp_minus_tree_immediate"]
    assert raw["smooth_interaction_fit"]["mean"] > 0.015
    assert raw["smooth_interaction_fit"]["positive_replicates"] >= 14
    assert raw["axis_interaction_fit"]["mean"] < -0.015
    assert raw["axis_interaction_fit"]["positive_replicates"] <= 2
    linear_world = post["mlp_minus_linear_raw_in_linear_world"]
    for visibility in VISIBILITY:
        assert linear_world[visibility]["probe_brier"]["positive_replicates"] == 16
        assert linear_world[visibility]["active_failed_rate"]["positive_replicates"] == 16
    assessment = result["scientific_assessment"]
    assert assessment["entry_interaction_specific_pass"] == "TREE_ONLY"
    assert assessment["v1_memorization_artifact"].startswith("REMOVED")
    assert assessment["neural_necessity"] == "NOT_ESTABLISHED"
