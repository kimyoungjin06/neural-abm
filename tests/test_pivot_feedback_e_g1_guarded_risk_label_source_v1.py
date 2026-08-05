from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_e_g1_guarded_risk_label_source_v1.yaml"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_e_g1_guarded_risk_label_source_v1_20260805.summary.json"
BUILDER_PATH = ROOT / "scripts/build_pivot_feedback_e_g1_guarded_risk_label_source_v1.py"
VERIFIER_PATH = ROOT / "scripts/verify_pivot_feedback_e_g1_guarded_risk_label_source_v1.py"
PRODUCTION = Path("/ssd/pivot-feedback-e-g1-guarded-risk-label-source-v1-20260805")
RECEIPT = PRODUCTION / "g1_guarded_risk_label_source.outputs.json"
REPORT = PRODUCTION / "g1_guarded_risk_label_source.json"
MANIFEST = PRODUCTION / "g1_guarded_cohort_manifest.parquet"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_PATH, "pivot_feedback_e_g1_guarded_source_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_freezes_guard_first_scope_labels_and_support() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["registered_scope"] == {
        "partition_id": "g1_empirical_replication",
        "field_ids_in_registered_slot_order": [16, 32, 11],
        "shock_year_start": 2013,
        "shock_year_end": 2016,
        "episode_count": 28,
        "family_count": 13,
        "response_cohort_offsets": [0, 1, 2],
        "response_cohort_count": 84,
        "minimum_source_year": 2010,
        "maximum_followup_year": 2021,
    }
    assert config["label_contract"]["guard_anti_join_before_label_logic"] is True
    assert config["label_contract"]["outcome_imputation"] == "NONE"
    assert config["support_contract"]["minimum_total"] == 500
    assert config["support_contract"]["minimum_entry_events"] == 100
    assert config["support_contract"]["minimum_non_entry_opportunities"] == 100
    assert config["support_contract"]["pooling_to_rescue_failure"] is False
    assert config["authorization_after_pass"]["scope_or_threshold_change"] == "NOT_AUTHORIZED"
    assert config["authorization_after_pass"]["g4_open"] == "NOT_AUTHORIZED"


def test_builder_applies_guard_before_every_label_table() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")
    guarded = source.index("CREATE TABLE guarded_risk AS")

    assert source.index("CREATE TABLE pre_guard_risk AS") < guarded
    assert guarded < source.index("CREATE TABLE label_count AS")
    assert guarded < source.index("CREATE TABLE subfield_count AS")
    assert guarded < source.index("CREATE TABLE guarded_label AS")
    assert "ANTI JOIN guard" in source[guarded : source.index("CREATE TABLE label_count AS")]
    assert "SELECT author_numeric_id FROM read_parquet" in source


def test_production_tree_receipt_and_access_boundary_are_closed() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert receipt["artifact_count"] == 86
    assert receipt["confirmatory_rows_read"] == 72_715_221
    assert receipt["holdout_outcome_rows_read"] == 0
    assert receipt["other_private_g4_files_read"] == 0
    assert receipt["seal_open_events"] == 0
    assert receipt["g4_open"] is False
    assert receipt["estimators_run"] == 0
    assert receipt["receipt_written_last"] is True
    assert report["guard"]["guard_overlap_rows"] == 0
    assert report["guard"]["anti_join_before_label_logic"] is True
    assert report["execution"]["config_sha256"] == _sha(CONFIG)
    assert report["execution"]["builder_sha256"] == _sha(BUILDER_PATH)
    assert pq.ParquetFile(MANIFEST).metadata.num_rows == 84
    assert len(list((PRODUCTION / "private/g1_guarded_risk_labels").glob("*.parquet"))) == 84
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in PRODUCTION.rglob("*") if path.is_dir())
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in PRODUCTION.rglob("*") if path.is_file())


def test_independent_summary_records_row_replay_and_terminal_support_failure() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["status"] == (
        "PASS_INDEPENDENT_G1_GUARD_FIRST_RISK_AND_LABEL_SOURCE_ROW_REPLAY_ESTIMATION_NOT_RUN"
    )
    assert summary["repository_artifacts"]["config"]["sha256"] == _sha(CONFIG)
    assert summary["repository_artifacts"]["builder"]["sha256"] == _sha(BUILDER_PATH)
    assert summary["repository_artifacts"]["independent_verifier"]["sha256"] == _sha(VERIFIER_PATH)
    assert summary["production_artifact"]["receipt_sha256"] == _sha(RECEIPT)
    assert summary["production_artifact"]["report_sha256"] == _sha(REPORT)
    assert summary["production_artifact"]["manifest_sha256"] == _sha(MANIFEST)
    assert summary["guard_replay"]["materialized_guard_overlap_rows"] == 0
    assert summary["label_replay"]["label_logic_mismatches"] == 0
    assert summary["label_replay"]["author_order_or_duplicate_rows"] == 0
    assert summary["independent_row_replay"] == {
        "alternative_pivot_entry_rows": 10_086,
        "entry_rows": 17_411,
        "guard_excluded_rows": 22_677_450,
        "guarded_risk_rows": 72_715_221,
        "non_entry_rows": 72_697_810,
        "one_off_entry_rows": 14_227,
        "persistence_rows": 3_184,
        "pre_guard_risk_rows": 95_392_671,
    }
    disposition = summary["registered_execution_disposition"]
    assert disposition["terminal_status"] == (
        "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION"
    )
    assert disposition["failed_unit_action"] == (
        "EXCLUDE_UNIT_AND_FAIL_REGISTERED_COVERAGE_REQUIREMENT"
    )
    assert [row["episode_id"] for row in disposition["failed_units"]] == [11, 25]
    assert [row["entries"] for row in disposition["failed_units"]] == [58, 28]
    assert disposition["registered_sampling_weighting_and_estimation"] == (
        "NOT_RUN_BLOCKED_BY_SUPPORT"
    )


def test_independent_verifier_is_producer_independent() -> None:
    tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert not any("build_pivot_feedback_e_g1_guarded" in name for name in imported)
    assert VERIFIER.CONFIG_SHA256 == _sha(CONFIG)
    assert VERIFIER.BUILDER_SHA256 == _sha(BUILDER_PATH)
