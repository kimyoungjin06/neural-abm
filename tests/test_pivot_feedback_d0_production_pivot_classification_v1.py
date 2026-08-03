from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_production_pivot_classification_v1.yaml"
)
BUILDER = (
    ROOT
    / "scripts/build_pivot_feedback_d0_production_pivot_classification_v1.py"
)
VERIFIER = (
    ROOT
    / "scripts/verify_pivot_feedback_d0_production_pivot_classification_v1.py"
)
PRODUCTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-production-pivot-classification-v1-20260731"
)
PRODUCTION_VERIFICATION = Path(
    "/ssd/"
    "pivot-feedback-d0-production-pivot-classification-v1-20260731"
    ".verification-v1.json"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD_MODULE = _load(BUILDER, "production_pivot_classifier_builder_test_module")
VERIFY_MODULE = _load(VERIFIER, "production_pivot_classifier_verifier_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_binds_v44_and_only_authorizes_four_state_application() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gate = config["repository_bindings"]["gate_state"]

    assert config["status_before_execution"] == "AUTHORIZED_BY_GATE_STATE_V44"
    assert gate["sha256"] == _sha256(ROOT / gate["relative_path"])
    contract = config["classification_contract"]
    assert contract["exact_states"] == list(BUILD_MODULE.STATES)
    assert contract["minimum_history_threshold"] == "NONE"
    assert contract["graph_team_cap"] == "NONE"
    assert contract["source_population_filtering"] == "FORBIDDEN"
    assert contract["source_feature_mutation"] == "FORBIDDEN"
    assert contract["post_entry_input_allowed"] is False
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def test_builder_and_verifier_output_schemas_match_without_future_outcomes() -> None:
    assert tuple(BUILD_MODULE.OUTPUT_ARROW_SCHEMA.names) == (
        BUILD_MODULE.OUTPUT_COLUMNS
    )
    assert BUILD_MODULE.OUTPUT_ARROW_SCHEMA.equals(
        VERIFY_MODULE.OUTPUT_ARROW_SCHEMA,
        check_metadata=True,
    )
    assert BUILD_MODULE.OUTPUT_COLUMNS[-2:] == (
        "observable_history_pivot_state",
        "observable_history_new_topic_entry",
    )
    assert not any(
        fragment in name
        for name in BUILD_MODULE.OUTPUT_COLUMNS
        for fragment in ("persistence", "impact", "citation", "model", "g4")
    )


@pytest.mark.parametrize(
    ("pre_entry", "trailing", "focal", "expected"),
    [
        (0, 0, 0, "NO_FOCAL_ENTRY"),
        (5, 2, 0, "NO_FOCAL_ENTRY"),
        (0, 0, 1, "NEW_TOPIC_ENTRY"),
        (5, 0, 1, "RETURN"),
        (5, 2, 1, "CONTINUATION"),
    ],
)
def test_independent_four_state_tables_match(
    pre_entry: int,
    trailing: int,
    focal: int,
    expected: str,
) -> None:
    assert BUILD_MODULE.classify_counts(pre_entry, trailing, focal) == expected
    assert VERIFY_MODULE.classify_counts(pre_entry, trailing, focal) == expected


def test_inconsistent_counts_fail_closed_in_both_engines() -> None:
    with pytest.raises(
        BUILD_MODULE.ProductionPivotClassificationError,
        match="trailing exceeds",
    ):
        BUILD_MODULE.classify_counts(1, 2, 1)
    with pytest.raises(
        VERIFY_MODULE.ProductionPivotClassificationVerificationError,
        match="trailing exceeds",
    ):
        VERIFY_MODULE.classify_counts(1, 2, 1)


def test_current_production_classification_and_independent_replay_are_bound() -> None:
    receipt_path = (
        PRODUCTION_ROOT / "production_pivot_classification.outputs.json"
    )
    artifact_path = PRODUCTION_ROOT / "production_pivot_classification.parquet"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verification = json.loads(PRODUCTION_VERIFICATION.read_text(encoding="utf-8"))

    assert receipt["status"] == BUILD_MODULE.STATUS
    assert receipt["classification_rows"] == 3_256_271
    assert receipt["classification_columns"] == 10
    assert receipt["new_topic_entry_rows"] == 4_301
    assert receipt["four_state_partition_complete"] is True
    assert receipt["source_rows_filtered"] == 0
    assert receipt["source_feature_mutations"] == 0
    assert receipt["outcomes_read"] is False
    assert receipt["api_requests_created"] == 0
    assert _sha256(receipt_path) == (
        "b9b064842ee2b71db8974ff9bef74f7c4bb9aa06f7cf588aec34eacaa3a812de"
    )
    assert _sha256(artifact_path) == (
        "32930717841455afabf79c093356ecfced94047be7a429afa4194e3606ab5f66"
    )
    parquet = pq.ParquetFile(artifact_path)
    assert parquet.metadata.num_rows == 3_256_271
    assert parquet.schema_arrow.equals(
        VERIFY_MODULE.OUTPUT_ARROW_SCHEMA,
        check_metadata=True,
    )

    assert verification["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert verification["output_receipt_sha256"] == _sha256(receipt_path)
    replay = verification["state_replay"]
    assert replay["source_column_mismatches"] == []
    assert replay["source_rows_filtered"] == 0
    assert replay["source_feature_value_mutations"] == 0
    assert replay["state_value_mismatches"] == 0
    assert replay["primary_boolean_mismatches"] == 0
    assert replay["totals"] == {
        "author_rows": 3_256_271,
        "continuation_rows": 4_861,
        "new_topic_entry_rows": 4_301,
        "no_focal_entry_rows": 3_246_984,
        "return_rows": 125,
    }
    assert verification["verifier"]["imports_producer_module"] is False
    assert verification["verifier"]["uses_producer_engine"] is False
    assert verification["scope"]["persistence_or_impact_rows_read"] is False
    assert verification["scope"]["api_requests_created"] == 0
