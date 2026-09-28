from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_matched_marginal_discriminator_v3_20260928.summary.json"
)
SMOOTH = ("smooth_interaction_aligned", "smooth_interaction_reversed")
AXIS = ("axis_interaction_aligned", "axis_interaction_reversed")


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_reproduced_and_nonselective() -> None:
    result = _result()
    assert result["status"] == "SYNTHETIC_POLICY_LAYER_DISCRIMINATOR_REPRODUCED"
    assert result["precommit"]["commit_sha"] == (
        "14f37afb2e740df28f0a1c505cea63823fc1b9c1"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 480
    assert execution["first_run"]["receipt_sha256"] == (
        execution["deterministic_rerun"]["receipt_sha256"]
    )
    assert execution["deterministic_rerun"]["all_four_artifacts_byte_identical"] is True
    validation = result["validation"]
    assert validation["scalar_orientation_invariance"] == "PASS_EXACT"
    assert validation["fixed_and_scalar_orientation_distance_zero"] is True
    assert validation["exact_half_success_fraction_all_rows"] is True
    assert result["boundaries"]["model_selected"] == "NO"


def test_registered_route_is_paired_nonlinear_arms() -> None:
    disposition = _result()["registered_result_disposition"]
    assert disposition == {
        "feature_conditional_need": "SUPPORTED_ON_FROZEN_GENERATOR",
        "nonlinear_representation_need": "SUPPORTED_BOTH_GEOMETRIES",
        "nonlinear_family_dominance": "GENERATOR_DEPENDENT_OR_TIED",
        "neural_interaction_specificity": "PASS",
        "neural_specific_advantage": "NOT_SUPPORTED_ON_FROZEN_GENERATOR",
        "abm_route": "PAIRED_CLASSICAL_AND_NEURAL_NONLINEAR_ARMS",
    }


def test_families_cross_over_by_boundary_geometry() -> None:
    decisions = _result()["registered_decisions"]
    for world in SMOOTH:
        assert decisions["mlp_over_tree"]["worlds"][world]["positive_replicates"] == 16
        assert decisions["tree_over_mlp"]["worlds"][world]["pass"] is False
    for world in AXIS:
        assert decisions["tree_over_mlp"]["worlds"][world]["positive_replicates"] == 16
        assert decisions["mlp_over_tree"]["worlds"][world]["pass"] is False
    for world in SMOOTH + AXIS:
        report = decisions["nonlinear_representation_need"]["worlds"][world]
        assert report["positive_replicates"] == 16
        assert report["mean"] > 0.08


def test_log_loss_does_not_preserve_the_tree_axis_edge() -> None:
    result = _result()
    advantage = result["post_hoc_descriptive"]["mlp_minus_tree_advantage"]
    for world in AXIS:
        assert advantage[world]["brier"]["mean"] < -0.04
        assert -0.02 < advantage[world]["log_loss"]["mean"] < 0.0
        assert advantage[world]["log_loss"]["positive_replicates"] >= 8
    for world in SMOOTH:
        assert advantage[world]["log_loss"]["positive_replicates"] == 16
    framework = result["framework_assessment"]
    assert framework["carry_forward_policy_families"] == [
        "CLASSICAL_NONLINEAR",
        "NEURAL_NONLINEAR",
    ]
    assert framework["abm_integration_requires_separate_precommit"] is True
    assert result["scientific_assessment"]["neural_necessity"] == "NOT_ESTABLISHED"
