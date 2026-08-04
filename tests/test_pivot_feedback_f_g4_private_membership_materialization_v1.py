from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_g4_private_membership_materialization_v1.yaml"
BUILDER = SCRIPTS / "build_pivot_feedback_f_g4_private_membership_materialization_v1.py"
VERIFIER = SCRIPTS / "verify_pivot_feedback_f_g4_private_membership_materialization_v1.py"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_f_g4_private_membership_materialization_v1_20260804.summary.json"
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v9.md"
PRODUCTION = Path("/ssd/pivot-feedback-f-g4-private-membership-v1-20260804")
VERIFICATION = Path("/ssd/pivot-feedback-f-g4-private-membership-v1-20260804.verification.json")
RECEIPT = PRODUCTION / "g4_private_membership_materialization.outputs.json"
REPORT = PRODUCTION / "g4_private_membership_materialization.json"

PASS_STATUS = (
    "PASS_OPERATOR_ONLY_G4_PREDECISION_MEMBERSHIP_AND_AUTHOR_COLD_GUARD_"
    "REMAINING_PRIVATE_PACKAGE_PENDING_NO_F_PASS"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_PRIVATE_G4_PREDECISION_MEMBERSHIP_AUTHOR_COLD_GUARD_"
    "NO_OUTCOME_READ_NO_F_PASS"
)
RISK_PREFIX = "private/g4_risk_measurement/"
FORBIDDEN_COLUMNS = {
    "work_id",
    "work_numeric_id",
    "entry_label",
    "persistence_label",
    "realized_citation_count",
    "model_score",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_contract_freezes_private_membership_author_cold_and_gate_boundary() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    membership = contract["private_membership_contract"]
    guard = contract["author_cold_guard_contract"]
    completion = contract["completion_accounting"]
    authorization = contract["authorization_after_pass"]

    assert contract["schema_version"] == (
        "pivot_feedback_f_g4_private_membership_materialization.v1"
    )
    assert membership["partition_id"] == "g4_temporal_holdout"
    assert membership["field_id"] == 27
    assert membership["entry_years"] == [2020, 2021, 2022]
    assert membership["episode_count"] == 37
    assert membership["family_count"] == 16
    assert membership["expected_risk_opportunity_rows"] == 217_839_267
    assert membership["raw_author_tokens_private"] is True
    assert membership["raw_work_ids_emitted"] is False
    assert membership["entry_or_persistence_labels_emitted"] is False
    assert membership["postdecision_target_events_emitted"] is False
    assert membership["realized_citation_effects_emitted"] is False
    assert membership["model_scores_emitted"] is False

    assert guard["guard_unit"] == "DISTINCT_G4_RISK_OPPORTUNITY_AUTHOR_TOKEN"
    assert guard["exact_projection_of_private_risk_rows"] is True
    assert guard["future_earlier_row_builds_must_anti_join_guard"] is True
    assert guard["earlier_row_filter_application_status"] == (
        "PENDING_FUTURE_ROW_MATERIALIZATION"
    )
    assert guard["natural_person_identity_claim"] == (
        "FORBIDDEN_CURRENT_SNAPSHOT_AUTHOR_TOKEN_ONLY"
    )

    assert completion["complete_private_g4_artifact_set"] is False
    assert completion["holdout_preflight_complete"] is False
    assert completion["holdout_byte_seal_complete"] is False
    assert completion["confirmatory_or_holdout_outcome_rows_read"] == 0
    assert completion["pass_f"] is False
    assert completion["pass_g0"] is False
    assert authorization[
        "remaining_private_predecision_feature_and_postdecision_source_materialization"
    ] == "AUTHORIZED_NEXT"
    assert authorization["explicit_file_list_preflight"] == (
        "NOT_AUTHORIZED_INCOMPLETE_PRIVATE_G4_ARTIFACT_SET"
    )
    assert authorization["g4_open"] == "NOT_AUTHORIZED"


def test_production_report_records_real_rows_without_outcome_materialization() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == PASS_STATUS
    assert report["accounting"] == {
        "author_cold_guard_rows": 8_174_698,
        "entry_or_persistence_labels_emitted": False,
        "focal_excluded_attention_work_incidences": 0,
        "model_scores_emitted": False,
        "modern_event_rows": 185_332_443,
        "network_leaveout_surge_rows": 217_832_376,
        "postdecision_target_events_emitted": False,
        "raw_work_ids_emitted": False,
        "realized_citation_effects_emitted": False,
        "risk_file_count": 37,
        "risk_opportunity_rows": 217_839_267,
        "target_event_rows": 2_239_321,
    }
    assert report["author_cold_guard"] == {
        "earlier_row_filter_application_status": "PENDING_FUTURE_ROW_MATERIALIZATION",
        "exact_projection_of_private_risk_rows": True,
        "exclude_from_discovery": True,
        "exclude_from_g1": True,
        "exclude_from_model_selection": True,
        "guard_rows": 8_174_698,
    }
    assert len(report["episode_manifest"]) == 37
    assert all(row["risk_count_closure"] for row in report["episode_manifest"])
    assert all(row["surge_count_closure"] for row in report["episode_manifest"])
    assert report["resource_preflight"]["passed_before_source_payload_open"] is True
    assert report["authorization"]["explicit_file_list_preflight"] == (
        "NOT_AUTHORIZED_INCOMPLETE_PRIVATE_G4_ARTIFACT_SET"
    )
    assert report["completion"] == {
        "complete_private_g4_artifact_set": False,
        "confirmatory_or_holdout_outcome_rows_read": 0,
        "earlier_partition_guard_application_complete": False,
        "entry_or_persistence_labels_materialized": False,
        "g4_author_cold_guard_complete": True,
        "g4_open": False,
        "g4_private_predecision_membership_complete": True,
        "holdout_byte_seal_complete": False,
        "holdout_preflight_complete": False,
        "model_fitted": False,
        "pass_f": False,
        "pass_g0": False,
        "postdecision_target_source_materialized": False,
    }


def test_receipt_closes_private_tree_and_parquet_metadata_without_content_scan() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    artifacts = receipt["artifacts"]
    artifact_paths = {row["relative_path"] for row in artifacts}
    actual_paths = {
        path.relative_to(PRODUCTION).as_posix()
        for path in PRODUCTION.rglob("*")
        if path.is_file() and path != RECEIPT
    }
    risk_rows = [
        row for row in artifacts if row["relative_path"].startswith(RISK_PREFIX)
    ]

    assert receipt["status"] == PASS_STATUS
    assert receipt["artifact_count"] == len(artifacts) == 41
    assert artifact_paths == actual_paths
    assert sum(row["bytes"] for row in artifacts) == 573_119_853
    assert len(risk_rows) == 37
    assert sum(row["bytes"] for row in risk_rows) == 548_720_762
    assert sum(row["rows"] for row in risk_rows) == 217_839_267
    assert _mode(PRODUCTION) == 0o700
    assert all(_mode(path) == 0o700 for path in PRODUCTION.rglob("*") if path.is_dir())
    assert all(_mode(path) == 0o600 for path in PRODUCTION.rglob("*") if path.is_file())
    assert all(
        (PRODUCTION / row["relative_path"]).stat().st_size == row["bytes"]
        for row in artifacts
    )
    assert all(
        pq.ParquetFile(PRODUCTION / row["relative_path"]).metadata.num_rows
        == row["rows"]
        for row in artifacts
        if row["relative_path"].endswith(".parquet")
    )

    for row in artifacts:
        if not row["relative_path"].endswith(".parquet"):
            continue
        schema = pq.ParquetFile(PRODUCTION / row["relative_path"]).schema_arrow
        assert schema.metadata is not None
        assert schema.metadata.get(b"schema_version")
        assert FORBIDDEN_COLUMNS.isdisjoint(schema.names)


def test_canonical_independent_verification_binds_full_row_replay() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))

    assert verification == {
        "artifact_count": 41,
        "author_cold_guard_exact_projection_verified": True,
        "author_cold_guard_rows_verified": 8_174_698,
        "complete_private_g4_artifact_set": False,
        "confirmatory_or_holdout_outcome_rows_read": 0,
        "earlier_partition_guard_application_complete": False,
        "forbidden_outcome_and_work_columns_absent": True,
        "g4_author_cold_guard_complete": True,
        "g4_open": False,
        "g4_private_predecision_membership_complete": True,
        "guard_projection_extra": 0,
        "guard_projection_missing": 0,
        "holdout_byte_seal_complete": False,
        "holdout_preflight_complete": False,
        "model_fitted": False,
        "model_score_read": False,
        "network_leaveout_surge_rows_verified": 217_832_376,
        "output_receipt_sha256": _sha(RECEIPT),
        "output_root_basename": PRODUCTION.name,
        "parent_count_and_row_algebra_verified": True,
        "pass_f": False,
        "pass_g0": False,
        "risk_episode_files_verified": 37,
        "risk_key_duplicates": 0,
        "risk_opportunity_rows_verified": 217_839_267,
        "schema_version": (
            "pivot_feedback_f_g4_private_membership_materialization_verification.v1"
        ),
        "status": VERIFY_STATUS,
    }
    assert summary["independent_verification"]["verification_sha256"] == _sha(
        VERIFICATION
    )
    assert summary["independent_verification"]["status"] == VERIFY_STATUS


def test_verifier_is_builder_independent_and_checks_guard_both_directions() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert "build_pivot_feedback_f_g4_private_membership_materialization_v1" not in source
    assert source.count("EXCEPT SELECT") == 2
    assert "EXCEPT SELECT author_numeric_id FROM {guard_sql}" in source
    assert "EXCEPT SELECT DISTINCT author_numeric_id FROM {risk_sql}" in source
    assert '"work_numeric_id"' in source
    assert '"entry_label"' in source
    assert '"persistence_label"' in source
    assert '"realized_citation_count"' in source
    assert '"model_score"' in source
    assert "private risk algebra or parent mismatch" in source
    assert "duplicate private risk key" in source


def test_repository_summary_and_ledger_bind_fail_closed_result() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")
    production = summary["production_artifact"]
    bindings = summary["repository_bindings"]

    assert summary["status"] == PASS_STATUS
    assert production["output_receipt_sha256"] == _sha(RECEIPT)
    assert production["report_sha256"] == _sha(REPORT)
    assert bindings["config_sha256"] == _sha(CONFIG)
    assert bindings["builder_sha256"] == _sha(BUILDER)
    assert bindings["verifier_sha256"] == _sha(VERIFIER)
    assert summary["materialization_accounting"]["non_surge_risk_rows"] == 6_891
    assert summary["author_cold_state"]["exact_projection_complete"] is True
    assert summary["author_cold_state"]["earlier_partition_guard_application_complete"] is False
    assert summary["fail_closed_state"]["complete_private_g4_artifact_set"] is False
    assert summary["fail_closed_state"]["pass_f"] is False
    assert summary["fail_closed_state"]["pass_g0"] is False
    assert "217,839,267 author-token by" in ledger
    assert "8,174,698 current-snapshot author" in ledger
    assert "Explicit-list preflight is not authorized" in ledger
    assert "`pass_f=false` and `pass_g0=false`" in ledger
