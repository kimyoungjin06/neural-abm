from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_adaptive_baseline_ladder_v1_20260807.summary.json"
)


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_reproduced_and_nonselective() -> None:
    result = _result()
    assert result["status"] == (
        "DESCRIPTIVE_SYNTHETIC_MODEL_COMPARISON_REPRODUCED"
    )
    assert result["precommit"]["commit_sha"] == (
        "bae8d48b0f0d97d433b9cdcb2b7708f290fb64a0"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 320
    assert execution["first_run"]["receipt_sha256"] == (
        execution["deterministic_rerun"]["receipt_sha256"]
    )
    assert execution["deterministic_rerun"][
        "all_four_artifacts_byte_identical"
    ] is True
    assert result["boundaries"]["model_selected"] == "NO"
    assert result["boundaries"]["predictive_validation"] == "NOT_RUN"


def test_fixed_rule_is_invariant_but_adaptive_models_are_not() -> None:
    result = _result()
    assert result["validation"][
        "fixed_rule_identical_across_visibility_regimes"
    ] is True
    contrast = result["registered_contrasts"][
        "failure_lag_4_minus_immediate_imitative"
    ]
    for family in ("delta", "neural"):
        assert contrast[family]["cumulative_entry_rate"] > 0.0
        assert contrast[family]["exited_failed_rate"] > 0.0
        assert contrast[family]["active_failed_rate"] < 0.0
        assert contrast[family]["entry_positive_replicates"] == 16
        assert contrast[family]["active_failed_negative_replicates"] == 16


def test_delta_reproduces_selective_missingness_residue() -> None:
    result = _result()
    contrast = result["registered_contrasts"][
        "selective_minus_full_failure_lag_4_imitative"
    ]
    for family in ("delta", "neural"):
        assert contrast[family]["active_failed_rate"] > 0.08
        assert contrast[family]["exited_failed_rate"] < -0.07
        assert contrast[family]["productive_share_of_active"] < -0.20


def test_cautionary_surface_does_not_require_neural_weights() -> None:
    result = _result()
    distances = {
        (row["regime"], row["evidence_mode"]): row
        for row in result["delta_neural_distance_means"]
    }
    for regime in ("failure_lag_2", "failure_lag_4"):
        row = distances[(regime, "cautionary")]
        assert row["final_vector_mae"] == 0.0
        assert row["trajectory_rmse"] == 0.0
    assert result["registered_result_disposition"][
        "CAUTIONARY_TRADEOFF_DOES_NOT_REQUIRE_NEURAL_WEIGHTS"
    ] == "OBSERVED"


def test_neural_imitation_adds_magnitude_not_unique_phase_direction() -> None:
    result = _result()
    contrast = result["registered_contrasts"][
        "failure_lag_4_minus_immediate_imitative"
    ]
    assert contrast["neural"]["cumulative_entry_rate"] > contrast["delta"][
        "cumulative_entry_rate"
    ]
    assert contrast["neural"]["exited_failed_rate"] > contrast["delta"][
        "exited_failed_rate"
    ]
    distances = {
        (row["regime"], row["evidence_mode"]): row
        for row in result["delta_neural_distance_means"]
    }
    assert distances[("immediate", "imitative")]["final_vector_mae"] > (
        distances[("failure_lag_4", "imitative")]["final_vector_mae"]
    )


def test_current_surface_does_not_support_neural_necessity() -> None:
    result = _result()
    assessment = result["scientific_assessment"]
    assert assessment["neural_necessity_on_current_synthetic_surface"] == (
        "NOT_SUPPORTED"
    )
    assert assessment["simpler_adaptive_rule_reproduces_core_surface"] is True
    assert assessment["neural_policy_is_behaviorally_identical_to_delta_everywhere"] is False
    assert assessment["model_selected"] == "NO"
    framework = result["framework_assessment"]
    assert framework["no_more_in_sample_synthetic_tuning"] is True
    assert framework["next_case_target"] == (
        "PRECOMMITTED_DISJOINT_SCENARIO_OR_EMPIRICAL_PREDICTIVE_GENERALIZATION_SCORECARD"
    )
    assert result["boundaries"]["empirical_rows_read"] == 0
    assert result["boundaries"]["neural_necessity"] == "NOT_ESTABLISHED"
