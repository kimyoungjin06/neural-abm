from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v48.yaml"
OPERATOR_ROOT = Path(
    "/ssd/pivot-feedback-d0-persistence-calibration-readout-v1-20260731"
)
OPERATOR_VERIFICATION = Path(
    "/ssd/"
    "pivot-feedback-d0-persistence-calibration-readout-v1-20260731"
    ".verification-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v48_binds_independently_verified_calibration_readout() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v48"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_descriptive_persistence_calibration"]
    for key in (
        "config",
        "producer",
        "verifier",
        "research_note",
        "summary_json",
        "summary_markdown",
    ):
        binding = verified[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])

    operator = verified["operator"]
    receipt_path = OPERATOR_ROOT / operator["receipt_basename"]
    report_path = OPERATOR_ROOT / operator["report_basename"]
    markdown_path = OPERATOR_ROOT / operator["markdown_basename"]
    assert operator["receipt_sha256"] == _sha256(receipt_path)
    assert operator["report_sha256"] == _sha256(report_path)
    assert operator["markdown_sha256"] == _sha256(markdown_path)
    assert operator["verification_sha256"] == _sha256(OPERATOR_VERIFICATION)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verification = json.loads(OPERATOR_VERIFICATION.read_text(encoding="utf-8"))
    assert receipt["status"] == operator["receipt_status"]
    assert verification["status"] == operator["verification_status"]

    result = verified["result"]
    assert result["source_rows_read"] == 4_301
    assert result["matured_persistence_rows"] == 1_487
    assert result["matured_no_persistence_rows"] == 2_814
    assert result["right_censored_rows"] == 0
    assert result["first_observed_t_plus_1_rows"] == 837
    assert result["first_observed_t_plus_2_rows"] == 401
    assert result["first_observed_t_plus_3_rows"] == 249
    assert result["all_three_followup_year_presence_rows"] == 190
    assert result["overall_value_mismatches"] == 0
    assert result["episode_value_mismatches"] == 0
    assert result["distribution_value_mismatches"] == 0
    assert result["invariant_value_mismatches"] == 0
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert result["joined_external_rows"] == 0
    assert result["author_level_rows_written"] == 0
    assert result["api_requests_created_or_submitted"] == 0
    assert result["adjusted_comparison_or_inferential_effect_computed"] is False


def test_gate_v48_completes_core_d0_but_not_g0() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    decision = gate["d0_core_calibration_decision"]
    scope = gate["authorized_g0_readiness_inventory_scope"]

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VERIFIED"
    assert decision["exact_registered_author_episode_rows_classified"] == 3_256_271
    assert decision["exact_new_topic_entry_rows"] == 4_301
    assert decision["exact_persistence_rows_calibrated"] == 4_301
    assert decision["descriptive_calibration_complete"] is True
    assert (
        decision["independent_all_row_and_all_derived_value_replay_complete"]
        is True
    )
    assert decision["optional_peer_visible_success_component_deferred"] is True
    assert decision["confirmatory_outcomes_inspected"] is False
    assert decision["g0_measurement_freeze_completed"] is False

    assert scope["mode"] == (
        "REPOSITORY_CONFIG_SUMMARY_AND_RECEIPT_METADATA_ONLY_NO_PRODUCTION_PARQUET_READ"
    )
    overlay = scope["required_pending_overlay"]
    assert overlay["sha256"] == _sha256(ROOT / overlay["relative_path"])
    assert scope["inventory_may_read_repository_configs_summaries_and_research_notes"] is True
    assert scope["inventory_may_open_operator_parquet_or_row_level_data"] is False
    assert scope["inventory_may_read_confirmatory_or_holdout_rows"] is False
    assert scope["inventory_may_create_or_submit_api_requests"] is False
    assert scope["inventory_may_choose_quantitative_values_from_observed_outcomes"] is False
    assert scope["inventory_may_rank_or_select_fields_episodes_or_thresholds_by_outcome"] is False
    assert scope["inventory_may_seal_or_pass_g0"] is False
    assert scope["inventory_may_fit_or_score_a_model"] is False
    assert scope["inventory_may_open_g4"] is False

    assert gate["gate_state"]["D0"]["complete"] is True
    assert gate["gate_state"]["G0"]["complete"] is False
    authorization = gate["authorization"]
    assert authorization["d0_core_entry_and_persistence_calibration"] == (
        "COMPLETE_INDEPENDENTLY_VERIFIED"
    )
    assert authorization["g0_readiness_and_gap_inventory"].startswith(
        "AUTHORIZED_METADATA"
    )
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
