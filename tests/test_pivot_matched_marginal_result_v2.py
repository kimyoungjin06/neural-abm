from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_matched_marginal_discriminator_v2_20260928.summary.json"
)


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_reproduced_and_nonselective() -> None:
    result = _result()
    assert result["status"] == "SYNTHETIC_POLICY_LAYER_DISCRIMINATOR_REPRODUCED"
    assert result["precommit"]["commit_sha"] == (
        "4c5920c30e69e493c2eceb4bb90dc6a6b0bd45a7"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 320
    assert execution["first_run"]["receipt_sha256"] == (
        execution["deterministic_rerun"]["receipt_sha256"]
    )
    assert execution["deterministic_rerun"][
        "all_four_artifacts_byte_identical"
    ] is True
    validation = result["validation"]
    assert validation["scalar_orientation_invariance"] == "PASS_EXACT"
    assert validation["fixed_and_scalar_orientation_distance_zero"] is True
    assert validation["exact_half_success_fraction_all_rows"] is True
    assert result["boundaries"]["model_selected"] == "NO"
    assert result["boundaries"]["empirical_rows_read"] == 0


def test_all_three_registered_comparisons_pass() -> None:
    result = _result()
    assert result["registered_result_disposition"] == {
        "feature_conditional_need": "SUPPORTED_ON_FROZEN_GENERATOR",
        "nonlinear_representation_need": "SUPPORTED_ON_FROZEN_GENERATOR",
        "neural_specific_advantage": "SUPPORTED_ON_FROZEN_GENERATOR_ONLY",
    }
    decisions = result["registered_decisions"]
    for report in decisions.values():
        assert report["pass"] is True
        for world in report["worlds"].values():
            assert world["mean"] >= report["minimum_mean_brier_improvement"]
            assert world["positive_replicates"] >= report["minimum_positive_replicates"]
    assert decisions["neural_specific_advantage"]["linear_world_guard_pass"] is True


def test_nonlinear_brier_margin_is_mostly_matched_on_linear_worlds() -> None:
    post = _result()["post_hoc_descriptive"]
    assert post["status"] == "POST_HOC_DESCRIPTIVE_NOT_REGISTERED_NO_NEW_EXECUTION"
    for orientation in ("aligned", "reversed"):
        gaps = post["mlp_over_linear_brier"][orientation]
        assert gaps["linear_world"]["positive_replicates"] == 16
        assert gaps["linear_world"]["mean"] > 0.1
        assert gaps["interaction_minus_linear"]["mean"] < 0.02
        auc = post["tree_over_linear_auc"][orientation]
        assert auc["interaction_world"]["positive_replicates"] == 16
        assert auc["interaction_world"]["mean"] - auc["linear_world"]["mean"] > 0.25


def test_neural_advantage_over_tree_is_not_interaction_specific() -> None:
    result = _result()
    post = result["post_hoc_descriptive"]
    for orientation in ("aligned", "reversed"):
        gaps = post["mlp_over_tree_brier"][orientation]
        assert gaps["linear_world"]["mean"] > gaps["interaction_world"]["mean"]
        assert gaps["interaction_minus_linear"]["positive_replicates"] <= 2
    assessment = result["scientific_assessment"]
    assert assessment["neural_necessity"] == "NOT_ESTABLISHED"
    assert assessment["model_selected"] == "NO"
    framework = result["framework_assessment"]
    assert framework["unresolved_distinction"] == "NEURAL_VERSUS_CLASSICAL_NONLINEAR"
    assert framework["full_abm_integration_on_neural_grounds"] == "NOT_YET_WARRANTED"
    assert framework["no_post_result_tuning_of_v2"] is True
