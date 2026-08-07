from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_persistence_exit_prototype_v1_20260807.summary.json"
)


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_bound_reproduced_and_entry_nested() -> None:
    result = _result()
    assert result["status"] == "DESCRIPTIVE_SYNTHETIC_RESULT_REPRODUCED"
    assert result["precommit"]["commit_sha"] == (
        "4347d5390742ad1a3fc9b62a62de025d8b7962b2"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 384
    assert execution["first_run"]["receipt_sha256"] == (
        execution["deterministic_rerun"]["receipt_sha256"]
    )
    assert execution["deterministic_rerun"][
        "all_four_artifacts_byte_identical"
    ] is True
    invariance = result["validation"]["entry_invariance"]
    assert invariance == {
        "status": "PASS",
        "paired_rows_checked": 192,
        "maximum_final_absolute_difference": 0.0,
        "maximum_trajectory_absolute_difference": 0.0,
    }


def test_frozen_persistence_is_identical_across_visibility() -> None:
    result = _result()
    frozen = [
        row for row in result["persistence_cell_means"] if row["arm"] == "frozen"
    ]
    fields = (
        "cumulative_entry_rate",
        "active_rate",
        "active_productive_rate",
        "active_failed_rate",
        "exited_failed_rate",
        "productive_share_of_active",
    )
    assert len(frozen) == 4
    assert len({tuple(row[field] for field in fields) for row in frozen}) == 1


def test_full_delay_creates_overshoot_then_purge() -> None:
    result = _result()
    contrast = result["key_paired_contrasts"][
        "failure_lag_4_minus_immediate_imitative"
    ]
    assert contrast["cumulative_entry_rate"] > 0.0
    assert contrast["exited_failed_rate"] > 0.0
    assert contrast["active_failed_rate"] < 0.0
    assert contrast["productive_share_of_active"] > 0.0
    assert contrast["active_failed_negative_replicates"] == 16
    mechanism = result["mechanism_readout"]["failure_lag_4_imitative"]
    assert mechanism["active_failed_rate_step_7"] > 0.4
    assert mechanism["final_active_failed_rate"] < 0.01


def test_selective_missingness_prevents_full_correction() -> None:
    result = _result()
    contrast = result["key_paired_contrasts"][
        "selective_failure_lag_4_minus_full_failure_lag_4_imitative"
    ]
    assert contrast["active_failed_rate"] > 0.09
    assert contrast["exited_failed_rate"] < -0.09
    assert contrast["final_attention_weight"] > 0.07
    full = result["mechanism_readout"]["failure_lag_4_imitative"]
    selective = result["mechanism_readout"][
        "selective_failure_lag_4_imitative"
    ]
    assert selective["mean_failure_releases_at_step_8"] < (
        full["mean_failure_releases_at_step_8"]
    )
    assert selective["final_active_failed_rate"] > full[
        "final_active_failed_rate"
    ]


def test_final_stock_equivalence_hides_different_paths() -> None:
    result = _result()
    contrast = result["key_paired_contrasts"][
        "selective_failure_lag_4_minus_immediate_imitative"
    ]
    assert abs(contrast["active_rate"]) < 0.001
    assert contrast["cumulative_entry_rate"] > 0.16
    assert contrast["exited_failed_rate"] > 0.16
    assert result["case_study_assessment"]["single_final_stock_is_insufficient"]


def test_next_target_is_a_fair_nonneural_baseline_not_another_state() -> None:
    result = _result()
    assessment = result["framework_assessment"]
    assert assessment["domain_owned_state_adapter_added_information"] is True
    assert assessment["generic_extraction_decision"] == (
        "DEFER_NO_SECOND_MODEL_FAMILY_YET"
    )
    assert assessment["next_case_target"] == (
        "FAIR_EXPLICIT_NONNEURAL_ADAPTIVE_BASELINE_ON_THE_SAME_PERSISTENCE_TARGETS"
    )
    boundaries = result["boundaries"]
    assert boundaries["empirical_rows_read"] == 0
    assert boundaries["empirical_fit"] == "NOT_RUN"
    assert boundaries["generic_nabm_api_changed"] is False
    assert boundaries["model_selected"] == "NO"
    assert boundaries["neural_necessity"] == "NOT_ESTABLISHED"
    assert boundaries["scientific_effect"] == "NOT_ESTABLISHED"
