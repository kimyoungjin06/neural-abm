from __future__ import annotations

import ast
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
    ROOT
    / "experiments/configs/pivot_feedback_d0_persistence_source_build_v1.yaml"
)
BUILDER = ROOT / "scripts/build_pivot_feedback_d0_persistence_source_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_d0_persistence_source_v1.py"
PRODUCTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-persistence-source-v1-full-20260731"
)
PRODUCTION_VERIFICATION = Path(
    "/ssd/"
    "pivot-feedback-d0-persistence-source-v1-full-20260731"
    ".verification-v1.json"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD_MODULE = _load(BUILDER, "persistence_source_builder_test_module")
VERIFY_MODULE = _load(VERIFIER, "persistence_source_verifier_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_binds_v46_and_exact_4301_source_scope() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gate = config["repository_bindings"]["gate_state"]
    source = config["source_contract"]

    assert config["status_before_execution"] == "AUTHORIZED_BY_GATE_STATE_V46"
    assert gate["sha256"] == _sha256(ROOT / gate["relative_path"])
    assert source["primary_pivot_state"] == "NEW_TOPIC_ENTRY"
    assert source["source_classification_rows"] == 3_256_271
    assert source["exact_output_rows"] == 4_301
    assert source["permitted_post_entry_offsets"] == [1, 2, 3]
    assert source["team_cap"] == "NONE"
    assert source["source_denominator_filtering"] == "FORBIDDEN"
    assert source["source_denominator_mutation"] == "FORBIDDEN"
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def test_builder_and_verifier_schemas_match_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert BUILD_MODULE.OUTPUT_COLUMNS == VERIFY_MODULE.OUTPUT_COLUMNS
    assert BUILD_MODULE.OUTPUT_SCHEMA.equals(
        VERIFY_MODULE.OUTPUT_SCHEMA,
        check_metadata=True,
    )
    assert tuple(config["output_contract"]["required_columns"]) == (
        BUILD_MODULE.OUTPUT_COLUMNS
    )
    assert BUILD_MODULE.OUTPUT_SCHEMA.metadata == {
        b"schema_version": b"pivot_feedback_d0_persistence_outcome_source.v1"
    }
    assert BUILD_MODULE.OUTPUT_SCHEMA.field(
        "three_year_target_topic_persistent"
    ).nullable


def test_independent_verifier_does_not_import_duckdb_or_builder() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "duckdb" not in imported
    assert not any("build_pivot_feedback" in name for name in imported)


def test_independent_work_scan_applies_exact_outcome_window_and_abi(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "work_id_numeric": 1,
            "publication_day": 1,
            "publication_year": 2013,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 1,
            "primary_topic_id": 10579,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 2,
            "publication_day": 1,
            "publication_year": 2014,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": None,
            "declared_authors_count": 1,
            "primary_topic_id": 10579,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 3,
            "publication_day": 1,
            "publication_year": 2015,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": True,
            "declared_authors_count": 1,
            "primary_topic_id": 10579,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 4,
            "publication_day": 1,
            "publication_year": 2016,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 1,
            "primary_topic_id": 10579,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 5,
            "publication_day": None,
            "publication_year": None,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 1,
            "primary_topic_id": 10579,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
    ]
    path = tmp_path / "part-0000.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=VERIFY_MODULE.WORK_SCHEMA), path)
    episodes = [[2, 17, 10579, 2012, 2015]]

    result = VERIFY_MODULE._scan_work_partition((0, path), episodes)

    assert result["source_rows"] == 5
    assert result["eligible"].size == 2
    assert result["eligible"]["offset"].tolist() == [1, 2]
    assert result["accounting"] == {
        "candidate_window_rows": 3,
        "excluded_wrong_topic_or_field_rows": 0,
        "excluded_temporal_disposition_rows": 0,
        "excluded_paratext_or_work_type_rows": 0,
        "excluded_retracted_rows": 1,
        "eligible_prelink_work_rows": 2,
        "retained_null_retraction_rows": 1,
        "excluded_missing_publication_year_target_rows": 1,
    }


def test_independent_event_scan_checks_metadata_and_keeps_raw_duplicates(
    tmp_path: Path,
) -> None:
    eligible = np.empty(1, dtype=VERIFY_MODULE.ELIGIBLE_DTYPE)
    eligible[0] = (10, 2, 17, 2013, 2, 1, 1)
    rows = [
        {
            "author_numeric_id": 101,
            "work_numeric_id": 10,
            "publication_year": 2013,
            "temporal_disposition": 2,
            "declared_authors_count": 1,
            "primary_field_id": 17,
        },
        {
            "author_numeric_id": 101,
            "work_numeric_id": 10,
            "publication_year": 2013,
            "temporal_disposition": 2,
            "declared_authors_count": 1,
            "primary_field_id": 17,
        },
        {
            "author_numeric_id": 102,
            "work_numeric_id": 10,
            "publication_year": 2014,
            "temporal_disposition": 2,
            "declared_authors_count": 1,
            "primary_field_id": 17,
        },
    ]
    path = tmp_path / "part-0000.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=VERIFY_MODULE.EVENT_SCHEMA), path)

    result = VERIFY_MODULE._scan_event_partition((0, path), eligible)

    assert result["joined_rows"] == 3
    assert result["mismatches"]["publication_year_mismatch_rows"] == 1
    assert result["pairs"].size == 2
    assert np.unique(result["pairs"]).size == 1


def test_independent_count_replay_preserves_zero_rows() -> None:
    full = np.empty(2, dtype=VERIFY_MODULE.FULL_KEY_DTYPE)
    full[0] = (1, 17, 14430, 2003, 101)
    full[1] = (1, 17, 14430, 2003, 102)
    primary = np.empty(2, dtype=VERIFY_MODULE.PRIMARY_KEY_DTYPE)
    primary[0] = (1, 101)
    primary[1] = (1, 102)
    pairs = np.empty(2, dtype=VERIFY_MODULE.PAIR_DTYPE)
    pairs[0] = (1, 101, 10, 1)
    pairs[1] = (1, 101, 11, 3)

    counts, accounting = VERIFY_MODULE._independent_counts(
        full,
        primary,
        pairs,
    )

    assert counts.tolist() == [[1, 0, 1], [0, 0, 0]]
    assert accounting == {
        "registered_primary_distinct_author_work_pairs": 2,
        "unregistered_or_nonprimary_distinct_author_work_pairs": 0,
    }


def test_current_production_source_and_independent_replay_are_bound() -> None:
    receipt_path = PRODUCTION_ROOT / "persistence_outcome_source.outputs.json"
    artifact_path = PRODUCTION_ROOT / "persistence_outcome_source.parquet"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verification = json.loads(PRODUCTION_VERIFICATION.read_text(encoding="utf-8"))

    assert receipt["status"] == BUILD_MODULE.STATUS
    assert receipt["persistence_source_rows"] == 4_301
    assert receipt["persistence_source_columns"] == 13
    assert receipt["matured_persistence_rows"] == 1_487
    assert receipt["matured_no_persistence_rows"] == 2_814
    assert receipt["right_censored_rows"] == 0
    assert receipt["qualifying_author_work_pairs"] == 3_689
    assert receipt["denominator_rows_filtered"] == 0
    assert receipt["denominator_rows_mutated"] == 0
    assert receipt["confirmatory_entry_year_rows_read"] == 0
    assert receipt["association_or_effect_computed"] is False
    assert receipt["api_requests_created"] == 0
    assert _sha256(receipt_path) == (
        "3aa45f93e4cbf40c0e9543482ec3c6a908b6026b7f9d23b719c615c68180295a"
    )
    assert _sha256(artifact_path) == (
        "0d347a296dbe08960e8a0fd2d5aaf9ff0afa7c9f55b08b761258e9874ae7f6c5"
    )
    parquet = pq.ParquetFile(artifact_path)
    assert parquet.metadata.num_rows == 4_301
    assert parquet.schema_arrow.equals(
        VERIFY_MODULE.OUTPUT_SCHEMA,
        check_metadata=True,
    )

    assert verification["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert verification["output_receipt_sha256"] == _sha256(receipt_path)
    assert verification["output_artifact_sha256"] == _sha256(artifact_path)
    assert verification["source_replay"]["work_partitions_rehashed"] == 4_096
    assert (
        verification["source_replay"]["author_event_partitions_rehashed"]
        == 4_096
    )
    replay = verification["outcome_replay"]
    assert replay["rows"] == 4_301
    assert replay["denominator_rows_filtered"] == 0
    assert replay["denominator_rows_mutated"] == 0
    assert all(value == 0 for value in replay["mismatches"].values())
    assert replay["totals"] == {
        "matured_no_persistence_rows": 2_814,
        "matured_persistence_rows": 1_487,
        "pair_rows_t_plus_1": 1_315,
        "pair_rows_t_plus_2": 1_281,
        "pair_rows_t_plus_3": 1_093,
        "qualifying_author_work_pairs": 3_689,
        "right_censored_rows": 0,
        "rows": 4_301,
    }
    assert verification["verifier"]["imports_producer_module"] is False
    assert verification["verifier"]["uses_producer_engine"] is False
    assert verification["scope"]["association_or_effect_computed"] is False
    assert verification["scope"]["api_requests_created"] == 0
