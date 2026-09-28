from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_nonlinear_policy_abm_v1_20260928.summary.json"
)
INTERACTION = ("smooth_interaction_fit", "axis_interaction_fit")
FEATURE_LEARNERS = ("linear_batch", "tree_batch", "mlp_batch")


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_nested_reproduced_and_nonselective() -> None:
    result = _result()
    assert result["status"] == "SYNTHETIC_GENERATIVE_ABM_COMPARISON_REPRODUCED"
    assert result["precommit"]["commit_sha"] == (
        "4ffc5dd7f372b4b212d30f1e6ee92d92aa0570ff"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 960
    assert execution["first_run"]["receipt_sha256"] == (
        execution["deterministic_rerun"]["receipt_sha256"]
    )
    assert execution["deterministic_rerun"]["all_five_artifacts_byte_identical"] is True
    assert result["validation"]["nesting_checks"] == "PASS_EXACT"
    assert result["validation"]["family_transfer"] == "FAMILY_TRANSFERRED"
    assert result["boundaries"]["model_selected"] == "NO"


def test_registered_dispositions_are_recorded_as_frozen() -> None:
    disposition = _result()["registered_result_disposition"]
    assert disposition["result"] == "NONLINEAR_LEARNING_CHANGES_BUT_BEHAVIOR_DOES_NOT"
    assert disposition["abm_geometry_crossover"] == "TREE_DOMINATES_IN_ABM"
    assert disposition["evidence_scale"] == "NONLINEAR_LEARNING_SURVIVES_LOCAL_EVIDENCE"
    assert disposition["behavior_changing_family_geometries_field"] == []


def test_entry_changes_under_immediate_visibility_and_saturates_under_delay() -> None:
    result = _result()
    post = result["post_hoc_descriptive"]
    assert len(post["behavior_passes_under_immediate_visibility"]) == 4
    behavior = result["registered_decisions"]["behavioral_consequence_field"]
    for report in behavior.values():
        assert report["visibility"]["failure_lag_4"]["mean"] == 0.0
    saturated = post["field_saturated_entry_runs"]["failure_lag_4"]
    for arm in FEATURE_LEARNERS:
        assert saturated[arm] == 48
        assert post["field_in_sample_share"]["failure_lag_4"][arm] == 1.0


def test_local_feature_learners_are_worse_than_the_prior_in_interaction_worlds() -> None:
    local = _result()["post_hoc_descriptive"]["local_policy_brier_minus_fixed_prior"]
    for visibility in ("immediate", "failure_lag_4"):
        for world in INTERACTION:
            for arm in FEATURE_LEARNERS:
                assert local[visibility][world][arm]["positive_replicates"] == 16
                assert local[visibility][world][arm]["mean"] > 0.03


def test_assessment_corrects_the_misleading_dispositions() -> None:
    result = _result()
    assessment = result["scientific_assessment"]
    assert assessment["evidence_scale_post_hoc"] == (
        "LOCAL_EVIDENCE_PREVENTS_USEFUL_FEATURE_LEARNING"
    )
    assert assessment["registered_crossover_disposition"].startswith(
        "NOT_A_GENERALIZATION_TEST"
    )
    assert assessment["neural_necessity"] == "NOT_ESTABLISHED"
    framework = result["framework_assessment"]
    assert len(framework["design_defects_found"]) == 4
    assert framework["no_post_result_tuning_of_v1"] is True
    transfer = result["family_transfer"]["winning_replicates"]
    assert transfer["mlp_beats_tree_smooth"] == 4
    assert transfer["tree_beats_mlp_axis"] == 4
