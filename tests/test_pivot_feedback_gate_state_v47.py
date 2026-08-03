from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v47.yaml"
OPERATOR_ROOT = Path(
    "/ssd/pivot-feedback-d0-persistence-source-v1-full-20260731"
)
OPERATOR_VERIFICATION = Path(
    "/ssd/"
    "pivot-feedback-d0-persistence-source-v1-full-20260731"
    ".verification-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v47_binds_independently_verified_production_source() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v47"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_production_persistence_source"]
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
    artifact_path = OPERATOR_ROOT / operator["artifact_basename"]
    report_path = OPERATOR_ROOT / operator["report_basename"]
    markdown_path = OPERATOR_ROOT / operator["markdown_basename"]
    assert operator["receipt_sha256"] == _sha256(receipt_path)
    assert operator["artifact_sha256"] == _sha256(artifact_path)
    assert operator["report_sha256"] == _sha256(report_path)
    assert operator["markdown_sha256"] == _sha256(markdown_path)
    assert operator["verification_sha256"] == _sha256(OPERATOR_VERIFICATION)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verification = json.loads(OPERATOR_VERIFICATION.read_text(encoding="utf-8"))
    assert receipt["status"] == operator["receipt_status"]
    assert verification["status"] == operator["verification_status"]
    assert verification["output_receipt_sha256"] == operator["receipt_sha256"]
    assert verification["output_artifact_sha256"] == operator["artifact_sha256"]

    result = verified["result"]
    assert result["exact_denominator_state"] == "NEW_TOPIC_ENTRY"
    assert result["exact_denominator_rows"] == 4_301
    assert result["matured_persistence_rows"] == 1_487
    assert result["matured_no_persistence_rows"] == 2_814
    assert result["right_censored_rows"] == 0
    assert result["qualifying_author_work_pairs"] == 3_689
    assert result["work_partitions_rehashed_by_producer"] == 4_096
    assert result["author_event_partitions_rehashed_by_producer"] == 4_096
    assert result["work_partitions_rehashed_by_verifier"] == 4_096
    assert result["author_event_partitions_rehashed_by_verifier"] == 4_096
    assert result["denominator_rows_filtered"] == 0
    assert result["denominator_rows_mutated"] == 0
    assert result["output_value_mismatch_rows"] == 0
    assert result["source_metadata_mismatch_rows"] == 0
    assert result["confirmatory_entry_year_rows_read"] == 0
    assert result["api_requests_created"] == 0
    assert result["association_or_effect_computed"] is False


def test_gate_v47_authorizes_only_existing_artifact_descriptive_readout() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    decision = gate["production_persistence_source_decision"]
    scope = gate["authorized_descriptive_calibration_readout_scope"]

    assert decision["status"] == "COMPLETE_INDEPENDENTLY_VERIFIED"
    assert decision["exact_denominator_state"] == "NEW_TOPIC_ENTRY"
    assert decision["exact_denominator_rows"] == 4_301
    assert decision["exact_outcome_window_offsets"] == [1, 2, 3]
    assert decision["persistence_positive_rows"] == 1_487
    assert decision["persistence_negative_rows"] == 2_814
    assert decision["adjusted_episode_comparison_established"] is False
    assert decision["exposure_association_established"] is False

    assert scope["mode"] == (
        "EXISTING_VERIFIED_4301_ROW_ARTIFACT_ONLY_DESCRIPTIVE_MEASUREMENT_CALIBRATION"
    )
    assert scope["required_input"]["rows"] == 4_301
    assert scope["required_input"]["columns"] == 13
    assert scope["readout_may_rehash_and_read_existing_verified_artifact"] is True
    assert scope["readout_may_recompute_every_derived_value"] is True
    assert scope["readout_may_join_raw_work_or_author_event_partitions"] is False
    assert (
        scope["readout_may_join_exposure_peer_citation_impact_or_affiliation_data"]
        is False
    )
    assert scope["readout_may_read_confirmatory_or_holdout_rows"] is False
    assert scope["readout_may_create_or_submit_api_requests"] is False
    assert scope["readout_may_compute_adjusted_comparisons_or_inferential_effects"] is False
    assert (
        scope["readout_may_rank_or_select_fields_episodes_or_thresholds_by_outcome"]
        is False
    )
    assert scope["readout_may_freeze_g0"] is False
    assert scope["readout_may_fit_or_score_a_model"] is False
    assert scope["readout_may_open_g4"] is False

    authorization = gate["authorization"]
    assert authorization[
        "d0_production_post_entry_persistence_source_build"
    ] == "COMPLETE_INDEPENDENTLY_VERIFIED"
    assert authorization[
        "d0_descriptive_persistence_calibration_readout"
    ].startswith("AUTHORIZED_EXISTING_VERIFIED_4301")
    assert authorization["d0_exposure_or_outcome_association_analysis"] == (
        "NOT_AUTHORIZED"
    )
    assert authorization["d0_outcome_based_selection"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
    assert gate["gate_state"]["D0"]["complete"] is False
