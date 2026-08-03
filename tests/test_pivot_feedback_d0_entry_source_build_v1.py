from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_entry_source_build_v1.yaml"
)
BUILDER = ROOT / "scripts/build_pivot_feedback_d0_entry_source_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_d0_entry_source_v1.py"
PRODUCTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-entry-source-v1-full-20260731"
)
PRODUCTION_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-entry-source-v1-full-20260731.verification-v1.json"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD_MODULE = _load(BUILDER, "pivot_entry_source_builder_test_module")
VERIFY_MODULE = _load(VERIFIER, "pivot_entry_source_verifier_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_binds_v43_and_forbids_state_or_outcome_materialization() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gate = config["repository_bindings"]["gate_state"]

    assert config["status_before_execution"] == "AUTHORIZED_BY_GATE_STATE_V43"
    assert gate["sha256"] == _sha256(ROOT / gate["relative_path"])
    assert config["feature_contract"]["post_entry_offsets_allowed"] == []
    assert config["feature_contract"]["team_cap"] == "NONE"
    assert config["feature_contract"]["minimum_history_threshold"] == "NONE"
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def test_output_schema_contains_counts_but_no_classifier_state() -> None:
    assert tuple(BUILD_MODULE.OUTPUT_SCHEMA.names) == BUILD_MODULE.FEATURE_COLUMNS
    assert BUILD_MODULE.OUTPUT_SCHEMA.equals(
        VERIFY_MODULE.OUTPUT_SCHEMA,
        check_metadata=True,
    )
    assert not any(
        "state" in name or "label" in name
        for name in BUILD_MODULE.OUTPUT_SCHEMA.names
    )


def test_current_production_output_and_independent_verification_are_bound() -> None:
    receipt_path = PRODUCTION_ROOT / "pivot_entry_source_build.outputs.json"
    feature_path = PRODUCTION_ROOT / "pivot_entry_source_features.parquet"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verification = json.loads(PRODUCTION_VERIFICATION.read_text(encoding="utf-8"))

    assert receipt["status"] == BUILD_MODULE.STATUS
    assert receipt["feature_rows"] == 3_256_271
    assert receipt["feature_columns"] == 8
    assert receipt["production_classifier_state_materialized"] is False
    assert receipt["pivot_label_materialized"] is False
    assert receipt["outcomes_read"] is False
    assert receipt["api_requests_created"] == 0
    assert _sha256(receipt_path) == (
        "e56329675a51b1c1a2be1003df2e60fa3fdad1a93fce42a2102adc957284e060"
    )
    assert _sha256(feature_path) == (
        "812843969ca8b147659173090bcb6f6ea7434e9c13ebf37b4e852418b06f5384"
    )
    parquet = pq.ParquetFile(feature_path)
    assert parquet.metadata.num_rows == 3_256_271
    assert parquet.schema_arrow.equals(
        VERIFY_MODULE.OUTPUT_SCHEMA,
        check_metadata=True,
    )

    assert verification["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert verification["output_receipt_sha256"] == _sha256(receipt_path)
    assert verification["feature_replay"]["pre_entry_value_mismatches"] == 0
    assert verification["feature_replay"]["trailing_value_mismatches"] == 0
    assert verification["feature_replay"]["focal_value_mismatches"] == 0
    assert (
        verification["feature_replay"]["production_classifier_state_columns"]
        == 0
    )
    assert verification["scope"]["outcomes_read"] is False
    assert verification["scope"]["api_requests_created"] == 0


def test_independent_work_partition_replay_applies_precedence(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "work_id_numeric": 4096,
            "publication_day": 1,
            "publication_year": 2003,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 1,
            "primary_topic_id": 14430,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 8192,
            "publication_day": 1,
            "publication_year": 2003,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": None,
            "declared_authors_count": 1,
            "primary_topic_id": 14430,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 12288,
            "publication_day": 1,
            "publication_year": 2003,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 1,
            "primary_topic_id": None,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 16384,
            "publication_day": 1,
            "publication_year": 2003,
            "temporal_disposition": 2,
            "work_type_code": 6,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 1,
            "primary_topic_id": 14430,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
    ]
    table = pa.Table.from_pylist(rows, schema=VERIFY_MODULE.WORK_SCHEMA)
    path = tmp_path / "part-0000.parquet"
    pq.write_table(table, path)
    episodes = [
        {
            "certified_episode_index": 1,
            "field_id": 17,
            "topic_id": 14430,
            "entry_year": 2003,
        },
        {
            "certified_episode_index": 2,
            "field_id": 17,
            "topic_id": 10579,
            "entry_year": 2012,
        },
        {
            "certified_episode_index": 3,
            "field_id": 27,
            "topic_id": 10151,
            "entry_year": 1993,
        },
        {
            "certified_episode_index": 4,
            "field_id": 27,
            "topic_id": 12421,
            "entry_year": 1953,
        },
    ]

    result = VERIFY_MODULE._scan_work_partition((0, path), episodes)
    episode = result["accounting"][0]

    assert episode["candidate_window_rows"] == 4
    assert episode["missing_primary_topic_or_field_rows"] == 1
    assert episode["excluded_work_type_or_paratext_rows"] == 1
    assert episode["eligible_prelink_rows"] == 2
    assert episode["retained_null_retraction_rows"] == 1
    assert sum(len(values) for values in result["eligible"]) == 2


def test_independent_event_partition_replay_checks_metadata(
    tmp_path: Path,
) -> None:
    eligible = np.empty(
        1,
        dtype=[
            ("work_id", "<u8"),
            ("episode", "u1"),
            ("field", "u1"),
            ("entry_year", "<u2"),
            ("publication_year", "<u2"),
            ("disposition", "u1"),
            ("declared", "<u4"),
            ("is_focal", "?"),
        ],
    )
    eligible[0] = (4096, 1, 17, 2003, 2003, 2, 1, True)
    table = pa.Table.from_pylist(
        [
            {
                "author_numeric_id": 4096,
                "work_numeric_id": 4096,
                "publication_year": 2003,
                "temporal_disposition": 2,
                "declared_authors_count": 1,
                "primary_field_id": 17,
            },
            {
                "author_numeric_id": 8192,
                "work_numeric_id": 4096,
                "publication_year": 2002,
                "temporal_disposition": 2,
                "declared_authors_count": 1,
                "primary_field_id": 17,
            },
        ],
        schema=VERIFY_MODULE.EVENT_SCHEMA,
    )
    path = tmp_path / "part-0000.parquet"
    pq.write_table(table, path)

    result = VERIFY_MODULE._scan_event_partition((0, path), eligible)

    assert result["joined_rows"] == 2
    assert result["publication_year_mismatch_rows"] == 1
    assert len(result["valid"]) == 1
