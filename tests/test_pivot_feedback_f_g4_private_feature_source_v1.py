from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_g4_private_feature_source_v1.yaml"
BUILDER = SCRIPTS / "build_pivot_feedback_f_g4_private_feature_source_v1.py"
VERIFIER = SCRIPTS / "verify_pivot_feedback_f_g4_private_feature_source_v1.py"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_f_g4_private_feature_source_v1_20260805.summary.json"
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v10.md"
PRODUCTION = Path("/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804")
VERIFICATION = Path("/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804.verification.json")
RECEIPT = PRODUCTION / "g4_private_feature_source_materialization.outputs.json"
REPORT = PRODUCTION / "g4_private_feature_source_materialization.json"

PASS_STATUS = (
    "PASS_OPERATOR_ONLY_COMPLETE_G4_FEATURE_AND_POSTDECISION_SOURCE_"
    "NO_LABEL_NO_ANALYTIC_OPEN_NO_F_PASS"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_COMPLETE_PRIVATE_G4_FEATURE_AND_POSTDECISION_SOURCE_"
    "NO_LABEL_NO_ANALYTIC_OPEN_NO_F_PASS"
)
PREFIXES = {
    "feature": "private/g4_predecision_feature/",
    "relation": "private/g4_contributing_peer_source/",
    "post": "private/g4_postdecision_target_event_source/",
    "byline": "private/g4_postdecision_target_work_byline_source/",
}
EXPECTED_PREFIX_ACCOUNTING = {
    "feature": (37, 1_834_343_869, 217_839_267),
    "relation": (37, 1_069_917_294, 267_491_648),
    "post": (37, 11_261_741, 1_607_759),
    "byline": (37, 17_982_075, 3_392_785),
}
FORBIDDEN_COLUMNS = {
    "entry_label",
    "persistence_label",
    "outcome_state",
    "descriptive_rate",
    "model_score",
    "predicted_probability",
    "treatment_effect",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_contract_freezes_complete_private_source_without_opening_analytics() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    feature = contract["feature_contract"]
    success = contract["public_success_contract"]
    post = contract["postdecision_source_contract"]
    completion = contract["completion_accounting"]
    authorization = contract["authorization_after_pass"]

    assert contract["schema_version"] == "pivot_feedback_f_g4_private_feature_source.v1"
    assert feature["partition_id"] == "g4_temporal_holdout"
    assert feature["field_id"] == 27
    assert feature["decision_years"] == [2020, 2021, 2022]
    assert feature["episode_count"] == 37
    assert feature["feature_rows"] == 217_839_267
    assert feature["contributing_peer_relation_rows"] == 267_491_648
    assert feature["raw_author_and_work_tokens_private"] is True

    assert success["rule_id"] == (
        "FIELD_YEAR_TOP_DECILE_AGE3_CITATION_PROXY_INCLUSIVE_TIES_V1"
    )
    assert success["expected_reference_work_rows"] == 30_198_415
    assert success["boundary_tie_rule"] == "INCLUDE_ALL_WORKS_AT_OR_ABOVE_CUTOFF"
    assert success["historical_platform_observation_time_established"] is False
    assert success["researcher_visibility_established"] is False

    assert post["permitted_offsets"] == [0, 1, 2, 3]
    assert post["entry_label_materialized"] is False
    assert post["persistence_label_materialized"] is False
    assert post["outcome_state_materialized"] is False
    assert post["descriptive_rate_computed"] is False
    assert post["model_score_computed"] is False

    assert completion["this_component_is_new_gate"] is False
    assert completion["this_component_alone_may_pass_f"] is False
    assert completion["complete_private_g4_artifact_set"] is False
    assert completion["holdout_preflight_complete"] is False
    assert completion["holdout_byte_seal_complete"] is False
    assert completion["confirmatory_or_holdout_analytic_rows_read"] == 0
    assert completion["pass_f"] is False
    assert completion["pass_g0"] is False
    assert authorization[
        "private_source_root_and_complete_explicit_artifact_list_binding"
    ] == "AUTHORIZED_NEXT"
    assert authorization["explicit_file_list_preflight"] == (
        "NOT_AUTHORIZED_BEFORE_SEPARATE_ROOT_AND_LIST_BINDING"
    )
    assert authorization["g4_open"] == "NOT_AUTHORIZED"


def test_production_report_records_sources_without_labels_or_analytic_read() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == PASS_STATUS
    assert report["accounting"] == {
        "analytic_outcome_rows_read": 0,
        "bad_feature_rows": 0,
        "contributing_peer_rows": 267_491_648,
        "feature_rows": 217_839_267,
        "guard_event_rows": 249_045_471,
        "labels_materialized": False,
        "peer_count_mismatch_rows": 0,
        "postdecision_offsets_read": [0, 1, 2, 3],
        "postdecision_target_event_rows": 1_607_759,
        "postdecision_target_work_byline_rows": 3_392_785,
        "prior_peer_year_rows": 1_589_253_481,
        "public_success_positive_work_rows": 3_493_298,
        "public_success_reference_work_rows": 30_198_415,
        "public_success_work_rows": 30_198_415,
        "recent_event_rows": 185_332_443,
        "risk_author_year_rows": 17_606_233,
        "risk_rows": 217_839_267,
        "target_event_rows": 4_261_278,
    }
    assert len(report["episode_manifest"]) == 37
    assert all(row["membership_count_closure"] for row in report["episode_manifest"])
    assert all(row["peer_count_closure"] for row in report["episode_manifest"])
    assert all(row["labels_materialized"] is False for row in report["episode_manifest"])
    assert report["resource_preflight"]["passed_before_source_payload_open"] is True
    assert report["authorization"]["explicit_file_list_preflight"] == (
        "NOT_AUTHORIZED_BEFORE_SEPARATE_ROOT_AND_LIST_BINDING"
    )
    assert report["completion"] == {
        "complete_private_g4_artifact_set": True,
        "confirmatory_or_holdout_analytic_rows_read": 0,
        "earlier_partition_guard_application_complete": False,
        "entry_or_persistence_labels_materialized": False,
        "g4_author_cold_guard_complete": True,
        "g4_open": False,
        "g4_postdecision_target_source_complete": True,
        "g4_private_predecision_features_complete": True,
        "g4_private_predecision_membership_complete": True,
        "holdout_byte_seal_complete": False,
        "holdout_preflight_complete": False,
        "model_fitted": False,
        "model_score_read": False,
        "pass_f": False,
        "pass_g0": False,
    }


def test_receipt_closes_private_tree_and_parquet_metadata() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    artifacts = receipt["artifacts"]
    artifact_paths = {row["relative_path"] for row in artifacts}
    actual_paths = {
        path.relative_to(PRODUCTION).as_posix()
        for path in PRODUCTION.rglob("*")
        if path.is_file() and path != RECEIPT
    }

    assert receipt["status"] == PASS_STATUS
    assert receipt["artifact_count"] == len(artifacts) == 153
    assert receipt["artifact_tree_sha256"] == (
        "62d6978c776c32464c4c862df12a52a580a6066dda45d4fb0dfe75910a562084"
    )
    assert artifact_paths == actual_paths
    assert sum(row["bytes"] for row in artifacts) == 3_067_046_872
    assert _mode(PRODUCTION) == 0o700
    assert all(_mode(path) == 0o700 for path in PRODUCTION.rglob("*") if path.is_dir())
    assert all(_mode(path) == 0o600 for path in PRODUCTION.rglob("*") if path.is_file())

    for label, prefix in PREFIXES.items():
        selected = [row for row in artifacts if row["relative_path"].startswith(prefix)]
        files, bytes_total, rows_total = EXPECTED_PREFIX_ACCOUNTING[label]
        assert len(selected) == files
        assert sum(row["bytes"] for row in selected) == bytes_total
        assert sum(row["rows"] for row in selected) == rows_total

    parquet_artifacts = [
        row for row in artifacts if row["relative_path"].endswith(".parquet")
    ]
    assert len(parquet_artifacts) == 151
    assert sum(row["rows"] for row in parquet_artifacts) == 528_704_609
    for row in parquet_artifacts:
        path = PRODUCTION / row["relative_path"]
        parquet = pq.ParquetFile(path)
        assert path.stat().st_size == row["bytes"]
        assert parquet.metadata.num_rows == row["rows"]
        assert parquet.schema_arrow.metadata is not None
        assert parquet.schema_arrow.metadata.get(b"schema_version")
        assert FORBIDDEN_COLUMNS.isdisjoint(parquet.schema_arrow.names)


def test_canonical_independent_verification_closes_row_algebra_and_source_replay() -> None:
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))

    assert verification["schema_version"] == (
        "pivot_feedback_f_g4_private_feature_source_verification.v1"
    )
    assert verification["status"] == VERIFY_STATUS
    assert verification["output_receipt_sha256"] == _sha(RECEIPT)
    assert verification["artifact_count"] == 153
    assert verification["feature_episode_files_verified"] == 37
    assert verification["relation_episode_files_verified"] == 37
    assert verification["postdecision_episode_files_verified"] == 37
    assert verification["byline_episode_files_verified"] == 37
    assert verification["feature_rows_verified"] == 217_839_267
    assert verification["contributing_peer_rows_verified"] == 267_491_648
    assert verification["public_success_work_rows_verified"] == 30_198_415
    assert verification["public_success_positive_work_rows_verified"] == 3_493_298
    assert verification["postdecision_target_event_rows_verified"] == 1_607_759
    assert verification["postdecision_target_work_byline_rows_verified"] == 3_392_785
    assert verification["output_row_algebra"] == {
        "byline_key_duplicates": 0,
        "feature_algebra_failures": 0,
        "feature_key_duplicates": 0,
        "missing_focal_byline_rows": 0,
        "orphan_byline_rows": 0,
        "parent_risk_mismatch_rows": 0,
        "post_algebra_failures": 0,
        "post_key_duplicates": 0,
        "relation_algebra_failures": 0,
        "relation_count_mismatch_rows": 0,
        "relation_key_duplicates": 0,
        "success_cohort_failures": 0,
        "success_cutoff_failures": 0,
        "success_key_duplicates": 0,
    }
    assert verification["source_replay"] == {
        "byline_mismatch_rows": 0,
        "feature_mismatch_rows": 0,
        "guard_event_rows_replayed": 249_045_471,
        "postdecision_mismatch_rows": 0,
        "prior_peer_year_rows_replayed": 1_589_253_481,
        "public_success_mismatch_rows": 0,
        "public_success_reference_work_rows_replayed": 30_198_415,
        "recent_event_rows_replayed": 185_332_443,
        "relation_mismatch_rows": 0,
        "relation_rows_replayed": 267_491_648,
        "risk_author_year_rows_replayed": 17_606_233,
        "target_event_rows_replayed": 4_261_278,
    }
    assert verification["source_replay_all_materialized_columns_verified"] is True
    assert verification["inclusive_tie_public_success_cutoff_verified"] is True
    assert verification["postdecision_focal_byline_closure_verified"] is True
    assert verification["forbidden_label_rate_score_columns_absent"] is True
    assert verification["complete_private_g4_artifact_set"] is True
    assert verification["holdout_preflight_complete"] is False
    assert verification["holdout_byte_seal_complete"] is False
    assert verification["confirmatory_or_holdout_analytic_rows_read"] == 0
    assert verification["g4_open"] is False
    assert verification["pass_f"] is False
    assert verification["pass_g0"] is False


def test_verifier_is_producer_independent_and_replays_each_source_family() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert "build_pivot_feedback_f_g4_private_feature_source_v1" not in source
    assert "source replay feature mismatch" in source
    assert "source replay relation mismatch" in source
    assert "source replay public-success mismatch" in source
    assert "source replay postdecision mismatch" in source
    assert "source replay byline mismatch" in source
    assert "feature versus parent risk mismatch" in source
    assert "relation-to-feature count mismatch" in source
    assert "success target-rank cutoff" in source
    assert "missing focal byline" in source
    assert '"entry_label"' in source
    assert '"persistence_label"' in source
    assert '"descriptive_rate"' in source
    assert '"model_score"' in source


def test_repository_summary_and_ledger_bind_fail_closed_result() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")
    production = summary["production_artifact"]
    bindings = summary["repository_bindings"]
    independent = summary["independent_verification"]

    assert summary["status"] == PASS_STATUS
    assert production["output_receipt_sha256"] == _sha(RECEIPT)
    assert production["report_sha256"] == _sha(REPORT)
    assert production["artifact_count"] == 153
    assert production["artifact_bytes_total"] == 3_067_046_872
    assert bindings["config_sha256"] == _sha(CONFIG)
    assert bindings["builder_sha256"] == _sha(BUILDER)
    assert bindings["verifier_sha256"] == _sha(VERIFIER)
    assert independent["verification_sha256"] == _sha(VERIFICATION)
    assert independent["output_row_algebra_mismatch_total"] == 0
    assert independent["source_replay_mismatch_total"] == 0
    assert summary["fail_closed_state"]["this_component_is_new_gate"] is False
    assert summary["fail_closed_state"]["complete_private_g4_artifact_set"] is True
    assert summary["fail_closed_state"]["holdout_preflight_complete"] is False
    assert summary["fail_closed_state"]["pass_f"] is False
    assert summary["fail_closed_state"]["pass_g0"] is False
    assert "217,839,267 predecision author-episode" in ledger
    assert "267,491,648" in ledger
    assert "All 14 recorded output mismatch counts are zero" in ledger
    assert "complete explicit 153-artifact list" in ledger
    assert "Explicit-list preflight remains unauthorized" in ledger
    assert "`pass_f=false`, `pass_g0=false`, and" in ledger
