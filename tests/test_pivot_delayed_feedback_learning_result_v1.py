from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_delayed_feedback_learning_prototype_v1_20260807.summary.json"
)


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_bound_to_the_precommit_and_exact_rerun() -> None:
    result = _result()
    assert result["status"] == "DESCRIPTIVE_SYNTHETIC_RESULT_REPRODUCED"
    assert result["precommit"]["commit_sha"] == (
        "3fc30cd5094e7fdcd0d59515bbf995d6ff6edede"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 240
    assert execution["first_run"]["receipt_sha256"] == (
        execution["deterministic_rerun"]["receipt_sha256"]
    )
    assert execution["deterministic_rerun"][
        "all_four_artifacts_byte_identical"
    ] is True
    assert all(len(value) == 64 for value in execution["artifact_sha256"].values())


def test_frozen_negative_control_is_identical_across_visibility_regimes() -> None:
    result = _result()
    frozen = [row for row in result["cell_means"] if row["arm"] == "frozen"]
    assert len(frozen) == 5
    fields = (
        "pivot_rate",
        "productive_pivot_rate",
        "failed_pivot_rate",
        "productive_share_of_pivots",
    )
    assert len({tuple(row[field] for field in fields) for row in frozen}) == 1
    assert result["validation"][
        "frozen_arm_identical_across_visibility_regimes"
    ] is True


def test_delayed_failure_amplifies_imitative_entry_and_failure() -> None:
    result = _result()
    contrasts = result["registered_contrasts"][
        "imitative_delay_minus_immediate"
    ]
    by_regime = {row["regime"]: row for row in contrasts}
    assert by_regime["failure_lag_2"]["pivot_rate"] > (
        by_regime["symmetric_lag_2"]["pivot_rate"]
    )
    assert by_regime["failure_lag_4"]["pivot_rate"] > (
        by_regime["failure_lag_2"]["pivot_rate"]
    )
    assert by_regime["failure_lag_4"]["failed_pivot_rate"] > (
        by_regime["failure_lag_2"]["failed_pivot_rate"]
    )
    assert by_regime["failure_lag_2"]["pivot_rate_positive_replicates"] == 16
    assert by_regime["failure_lag_4"]["pivot_rate_positive_replicates"] == 16


def test_cautionary_correction_weakens_and_costs_productive_pivots() -> None:
    result = _result()
    contrasts = result["registered_contrasts"][
        "cautionary_learning_minus_frozen"
    ]
    by_regime = {row["regime"]: row for row in contrasts}
    assert all(row["failed_pivot_rate"] < 0.0 for row in contrasts)
    assert all(row["productive_pivot_rate"] < 0.0 for row in contrasts)
    assert abs(by_regime["immediate"]["failed_pivot_rate"]) > abs(
        by_regime["failure_lag_2"]["failed_pivot_rate"]
    )
    assert abs(by_regime["failure_lag_2"]["failed_pivot_rate"]) > abs(
        by_regime["failure_lag_4"]["failed_pivot_rate"]
    )
    assert abs(by_regime["failure_lag_4"]["failed_pivot_rate"]) > abs(
        by_regime["selective_failure_lag_4"]["failed_pivot_rate"]
    )


def test_framework_decision_and_claim_boundaries_remain_narrow() -> None:
    result = _result()
    assessment = result["framework_assessment"]
    assert assessment["domain_owned_queue_worked_without_generic_api_change"] is True
    assert assessment["generic_extraction_decision"] == (
        "DEFER_NO_SECOND_MODEL_FAMILY_YET"
    )
    assert assessment["next_case_model_target"] == (
        "NONABSORBING_PERSISTENCE_EXIT_OR_REDIRECTION_STATE_MACHINE"
    )
    boundaries = result["boundaries"]
    assert boundaries["empirical_rows_read"] == 0
    assert boundaries["empirical_fit"] == "NOT_RUN"
    assert boundaries["generic_nabm_api_changed"] is False
    assert boundaries["model_selected"] == "NO"
    assert boundaries["neural_necessity"] == "NOT_ESTABLISHED"
    assert boundaries["scientific_effect"] == "NOT_ESTABLISHED"
