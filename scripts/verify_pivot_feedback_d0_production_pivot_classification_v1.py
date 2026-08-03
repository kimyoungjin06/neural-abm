#!/usr/bin/env python3
"""Independently verify every production observable-history pivot state."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_production_pivot_classification_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_d0_production_pivot_classification_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_production_pivot_classification_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_production_pivot_classification_outputs.v1"
ARTIFACT_SCHEMA = "pivot_feedback_d0_production_pivot_classification.v1"
VERIFY_SCHEMA = (
    "pivot_feedback_d0_production_pivot_classification_verification.v1"
)
SOURCE_SCHEMA = "pivot_feedback_d0_entry_source_features.v1"
SOURCE_VERIFY_SCHEMA = "pivot_feedback_d0_entry_source_verification.v1"
PRODUCER_STATUS = (
    "PASS_EXACT_FOUR_THRESHOLD_FREE_OBSERVABLE_HISTORY_FOUR_STATE_CLASSIFICATION"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_FOUR_ALL_ROW_OBSERVABLE_HISTORY_STATE_REPLAY"
)
SOURCE_STATUS = (
    "PASS_EXACT_FOUR_THREE_COUNT_PRODUCTION_FEATURE_SOURCE_NO_STATE_LABELS"
)
SOURCE_VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_FOUR_THREE_COUNT_SOURCE_REPLAY_NO_STATE_LABELS"
)
ROWS = 3_256_271
STATES = (
    "NO_FOCAL_ENTRY",
    "NEW_TOPIC_ENTRY",
    "RETURN",
    "CONTINUATION",
)
SOURCE_COLUMNS = (
    "certified_episode_index",
    "field_id",
    "topic_id",
    "entry_year",
    "author_numeric_id",
    "pre_entry_target_topic_work_count",
    "trailing_3y_target_topic_work_count",
    "focal_confirmed_eligible_target_topic_work_count",
)
OUTPUT_COLUMNS = (
    *SOURCE_COLUMNS,
    "observable_history_pivot_state",
    "observable_history_new_topic_entry",
)
SOURCE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field(
            "pre_entry_target_topic_work_count",
            pa.uint32(),
            nullable=False,
        ),
        pa.field(
            "trailing_3y_target_topic_work_count",
            pa.uint32(),
            nullable=False,
        ),
        pa.field(
            "focal_confirmed_eligible_target_topic_work_count",
            pa.uint32(),
            nullable=False,
        ),
    ],
    metadata={b"schema_version": SOURCE_SCHEMA.encode()},
)
OUTPUT_ARROW_SCHEMA = pa.schema(
    [
        *SOURCE_ARROW_SCHEMA,
        pa.field("observable_history_pivot_state", pa.string(), nullable=False),
        pa.field(
            "observable_history_new_topic_entry",
            pa.bool_(),
            nullable=False,
        ),
    ],
    metadata={b"schema_version": ARTIFACT_SCHEMA.encode()},
)
EXPECTED_FILES = {
    "PRODUCTION_PIVOT_CLASSIFICATION.md",
    "production_pivot_classification.json",
    "production_pivot_classification.outputs.json",
    "production_pivot_classification.parquet",
}


class ProductionPivotClassificationVerificationError(RuntimeError):
    """Fail-closed independent production-classification verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionPivotClassificationVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionPivotClassificationVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ProductionPivotClassificationVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _nested(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing bound field {dotted}",
        )
        current = current[part]
    return current


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _artifact_items(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = receipt.get("artifacts")
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for name, item_value in raw.items():
            item = _mapping(item_value, f"artifact {name}")
            relative = item.get("relative_path") or item.get("basename") or name
            require(isinstance(relative, str) and relative, "artifact path")
            result[relative] = item
    else:
        for item_value in _list(raw, "receipt artifacts"):
            item = _mapping(item_value, "receipt artifact")
            relative = item.get("relative_path")
            require(isinstance(relative, str) and relative, "artifact path")
            result[relative] = item
    return result


def _validate_repository(config: Mapping[str, Any]) -> dict[str, Any]:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V44",
        "v44 authorization",
    )
    result: dict[str, Any] = {}
    for label, raw in _mapping(
        config.get("repository_bindings"),
        "repository bindings",
    ).items():
        binding = _mapping(raw, f"repository binding {label}")
        path = _safe_repo_path(binding.get("relative_path"), label)
        require(_sha256(path) == binding.get("sha256"), f"{label} SHA-256")
        document = _read_yaml(path, label)
        require(
            document.get("schema_version") == binding.get("required_schema_version"),
            f"{label} schema",
        )
        if "required_status" in binding:
            require(
                document.get("status") == binding.get("required_status"),
                f"{label} status",
            )
        for dotted, expected in _mapping(
            binding.get("required_fields", {}),
            f"{label} fields",
        ).items():
            require(
                _nested(document, str(dotted)) == expected,
                f"{label} field {dotted}",
            )
        result[label] = {
            "relative_path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": binding["sha256"],
            "schema_version": document["schema_version"],
        }
    return result


def _validate_source(
    *,
    config: Mapping[str, Any],
    operator_base: Path,
    source_root: Path,
) -> dict[str, Any]:
    spec = _mapping(config.get("verified_operator_input"), "operator input")
    source_root = source_root.resolve(strict=True)
    require(
        source_root.parent == operator_base
        and source_root.name == spec.get("root_basename"),
        "source root",
    )
    require(stat.S_IMODE(source_root.stat().st_mode) == 0o700, "source root mode")
    receipt_path = source_root / str(spec.get("receipt_basename"))
    feature_path = source_root / str(spec.get("feature_basename"))
    verification_path = operator_base / str(spec.get("verification_basename"))
    require(_sha256(receipt_path) == spec.get("receipt_sha256"), "source receipt SHA")
    require(_sha256(feature_path) == spec.get("feature_sha256"), "source feature SHA")
    require(
        _sha256(verification_path) == spec.get("verification_sha256"),
        "source verification SHA",
    )
    receipt = _read_json(receipt_path, "source receipt")
    verification = _read_json(verification_path, "source verification")
    require(receipt.get("schema_version") == spec.get("receipt_schema_version"), "source receipt schema")
    require(receipt.get("status") == SOURCE_STATUS, "source status")
    require(receipt.get("feature_rows") == ROWS, "source rows")
    require(receipt.get("feature_columns") == 8, "source columns")
    require(receipt.get("production_classifier_state_materialized") is False, "source state boundary")
    require(receipt.get("pivot_label_materialized") is False, "source label boundary")
    require(verification.get("schema_version") == SOURCE_VERIFY_SCHEMA, "source verification schema")
    require(verification.get("status") == SOURCE_VERIFY_STATUS, "source verification status")
    require(
        verification.get("feature_replay", {}).get("pre_entry_value_mismatches")
        == 0,
        "source pre-entry replay",
    )
    require(
        verification.get("feature_replay", {}).get("trailing_value_mismatches")
        == 0,
        "source trailing replay",
    )
    require(
        verification.get("feature_replay", {}).get("focal_value_mismatches")
        == 0,
        "source focal replay",
    )
    parquet = pq.ParquetFile(feature_path)
    require(parquet.metadata.num_rows == ROWS, "source Parquet rows")
    require(
        parquet.schema_arrow.equals(SOURCE_ARROW_SCHEMA, check_metadata=True),
        "source schema",
    )
    return {
        "feature_path": feature_path,
        "receipt_sha256": spec["receipt_sha256"],
        "feature_sha256": spec["feature_sha256"],
        "verification_sha256": spec["verification_sha256"],
    }


def _validate_output(
    *,
    root: Path,
    config_path: Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "output root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "output root mode")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"unsafe output {path.name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"output mode {path.name}")
    receipt_path = root / "production_pivot_classification.outputs.json"
    report_path = root / "production_pivot_classification.json"
    artifact_path = root / "production_pivot_classification.parquet"
    receipt = _read_json(receipt_path, "classification receipt")
    report = _read_json(report_path, "classification report")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(receipt.get("status") == PRODUCER_STATUS, "receipt status")
    require(receipt.get("config_sha256") == _sha256(config_path), "receipt config")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema")
    require(report.get("status") == PRODUCER_STATUS, "report status")
    require(report.get("config_sha256") == _sha256(config_path), "report config")
    observed: list[dict[str, Any]] = []
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
        path = root / str(item.get("relative_path"))
        actual: dict[str, Any] = {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".parquet":
            actual["rows"] = pq.ParquetFile(path).metadata.num_rows
        require(actual == item, f"artifact {path.name}")
        observed.append(actual)
    observed.sort(key=lambda item: str(item["relative_path"]))
    require(receipt.get("artifact_count") == len(observed) == 3, "artifact count")
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(
            _canonical_bytes(
                [
                    (item["relative_path"], item["bytes"], item["sha256"])
                    for item in observed
                ]
            )
        ).hexdigest(),
        "artifact tree",
    )
    require(receipt.get("classification_rows") == ROWS, "classification rows")
    require(receipt.get("classification_columns") == 10, "classification columns")
    require(receipt.get("four_state_partition_complete") is True, "state closure")
    for key in (
        "post_entry_rows_read",
        "outcomes_read",
        "independent_verification_complete",
        "g0_measurement_freeze_authorized",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag {key}")
    for key in (
        "source_rows_filtered",
        "source_feature_mutations",
        "raw_work_partition_rows_read",
        "raw_author_event_partition_rows_read",
        "api_requests_created",
    ):
        require(receipt.get(key) == 0, f"unsafe receipt count {key}")
    require(receipt.get("receipt_written_last") is True, "receipt order")
    parquet = pq.ParquetFile(artifact_path)
    require(parquet.metadata.num_rows == ROWS, "artifact rows")
    require(
        parquet.schema_arrow.equals(OUTPUT_ARROW_SCHEMA, check_metadata=True),
        "artifact schema",
    )
    return {
        "receipt": receipt,
        "report": report,
        "receipt_path": receipt_path,
        "artifact_path": artifact_path,
        "artifacts": observed,
    }


def classify_counts(pre_entry: int, trailing: int, focal: int) -> str:
    for name, value in (
        ("pre-entry", pre_entry),
        ("trailing", trailing),
        ("focal", focal),
    ):
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{name} count",
        )
    require(trailing <= pre_entry, "trailing exceeds pre-entry")
    if focal == 0:
        return "NO_FOCAL_ENTRY"
    if pre_entry == 0:
        return "NEW_TOPIC_ENTRY"
    if trailing == 0:
        return "RETURN"
    return "CONTINUATION"


def _array_equal(left: pa.ChunkedArray, right: pa.ChunkedArray) -> bool:
    return left.combine_chunks().equals(right.combine_chunks())


def _replay_states(
    *,
    source_path: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    source = pq.read_table(source_path)
    output = pq.read_table(artifact_path)
    require(source.num_rows == output.num_rows == ROWS, "replay rows")
    source_mismatches: list[str] = []
    for name in SOURCE_COLUMNS:
        if not _array_equal(source[name], output[name]):
            source_mismatches.append(name)
    require(not source_mismatches, f"source column mismatch: {source_mismatches}")
    pre_entry = (
        source["pre_entry_target_topic_work_count"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    trailing = (
        source["trailing_3y_target_topic_work_count"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    focal = (
        source["focal_confirmed_eligible_target_topic_work_count"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    require(int(np.count_nonzero(trailing > pre_entry)) == 0, "source count invariant")
    expected = np.full(ROWS, "CONTINUATION", dtype="<U15")
    no_focal = focal == 0
    new_topic = (focal > 0) & (pre_entry == 0)
    return_state = (focal > 0) & (pre_entry > 0) & (trailing == 0)
    expected[no_focal] = "NO_FOCAL_ENTRY"
    expected[new_topic] = "NEW_TOPIC_ENTRY"
    expected[return_state] = "RETURN"
    observed_state = output["observable_history_pivot_state"].combine_chunks()
    state_mismatches = int(
        np.count_nonzero(
            np.asarray(observed_state.to_pylist(), dtype="<U15") != expected
        )
    )
    observed_boolean = (
        output["observable_history_new_topic_entry"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
    )
    boolean_mismatches = int(np.count_nonzero(observed_boolean != new_topic))
    require(state_mismatches == boolean_mismatches == 0, "state replay mismatch")
    episode = (
        source["certified_episode_index"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    field = source["field_id"].combine_chunks().to_numpy(zero_copy_only=False)
    topic = source["topic_id"].combine_chunks().to_numpy(zero_copy_only=False)
    entry_year = source["entry_year"].combine_chunks().to_numpy(zero_copy_only=False)
    by_episode: list[dict[str, int]] = []
    for episode_index in (1, 2, 3, 4):
        mask = episode == episode_index
        positions = np.flatnonzero(mask)
        counts = {
            state: int(np.count_nonzero(mask & (expected == state)))
            for state in STATES
        }
        by_episode.append(
            {
                "certified_episode_index": episode_index,
                "field_id": int(field[positions[0]]),
                "topic_id": int(topic[positions[0]]),
                "entry_year": int(entry_year[positions[0]]),
                "author_rows": len(positions),
                "no_focal_entry_rows": counts["NO_FOCAL_ENTRY"],
                "new_topic_entry_rows": counts["NEW_TOPIC_ENTRY"],
                "return_rows": counts["RETURN"],
                "continuation_rows": counts["CONTINUATION"],
            }
        )
    totals = {
        "author_rows": ROWS,
        "no_focal_entry_rows": int(np.count_nonzero(expected == "NO_FOCAL_ENTRY")),
        "new_topic_entry_rows": int(np.count_nonzero(expected == "NEW_TOPIC_ENTRY")),
        "return_rows": int(np.count_nonzero(expected == "RETURN")),
        "continuation_rows": int(np.count_nonzero(expected == "CONTINUATION")),
    }
    require(
        totals["author_rows"]
        == sum(totals[f"{state.lower()}_rows"] for state in STATES),
        "state partition closure",
    )
    return {
        "rows": ROWS,
        "source_columns_compared": len(SOURCE_COLUMNS),
        "source_column_mismatches": source_mismatches,
        "state_value_mismatches": state_mismatches,
        "primary_boolean_mismatches": boolean_mismatches,
        "by_episode": by_episode,
        "totals": totals,
        "source_rows_filtered": 0,
        "source_feature_value_mutations": 0,
        "raw_work_partition_rows_read": 0,
        "raw_author_event_partition_rows_read": 0,
        "post_entry_offsets_read": [],
    }


def verify(
    *,
    config_path: Path,
    operator_base: Path,
    source_root: Path,
    producer_root: Path,
    verification_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    config_path = config_path.resolve(strict=True)
    operator_base = operator_base.resolve(strict=True)
    producer_root = producer_root.resolve(strict=True)
    verification_path = verification_path.resolve()
    require(
        verification_path.parent == operator_base
        and not verification_path.exists(),
        "verification output",
    )
    config = _read_yaml(config_path, "config")
    repository = _validate_repository(config)
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden inputs and outputs",
    )
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary")
    source = _validate_source(
        config=config,
        operator_base=operator_base,
        source_root=source_root,
    )
    output = _validate_output(root=producer_root, config_path=config_path)
    builder_path = _safe_repo_path(
        output["report"]["execution"]["builder_relative_path"],
        "reported builder",
    )
    require(
        output["report"]["execution"]["builder_sha256"] == _sha256(builder_path),
        "reported builder SHA",
    )
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(name=node.module or "")]
        )
    }
    require(
        "build_pivot_feedback_d0_production_pivot_classification_v1"
        not in imported_modules,
        "verifier imports producer",
    )
    require("duckdb" not in imported_modules, "verifier uses producer engine")
    replay = _replay_states(
        source_path=source["feature_path"],
        artifact_path=output["artifact_path"],
    )
    reported = output["report"]["classification_accounting"]
    for key in (
        "by_episode",
        "totals",
        "source_rows_filtered",
        "source_feature_value_mutations",
    ):
        require(reported.get(key) == replay.get(key), f"producer report {key}")
    require(reported.get("state_rule_mismatches") == 0, "producer state mismatches")
    require(
        reported.get("primary_boolean_mismatches") == 0,
        "producer boolean mismatches",
    )
    verification = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "verifier": {
            "strategy": "PYARROW_NUMPY_EXACT_SOURCE_AND_ALL_ROW_STATE_REPLAY",
            "imports_producer_module": False,
            "uses_producer_engine": False,
            "producer_engine": "DUCKDB",
            "verification_engine": "PYARROW_NUMPY",
            "elapsed_seconds": round(time.monotonic() - started, 6),
        },
        "repository_bindings": repository,
        "source_binding": {
            "root_basename": source_root.name,
            "receipt_sha256": source["receipt_sha256"],
            "feature_sha256": source["feature_sha256"],
            "verification_sha256": source["verification_sha256"],
        },
        "output_root_basename": producer_root.name,
        "output_receipt_sha256": _sha256(output["receipt_path"]),
        "output_artifacts": output["artifacts"],
        "scope": {
            "exact_certified_episode_rows": 4,
            "registered_author_episode_rows": ROWS,
            "threshold_free": True,
            "minimum_history_threshold": "NONE",
            "observable_history_claim_only": True,
            "true_lifetime_first_claim_allowed": False,
            "raw_work_partition_rows_read": 0,
            "raw_author_event_partition_rows_read": 0,
            "post_entry_offsets_read": [],
            "persistence_or_impact_rows_read": False,
            "api_requests_created": 0,
        },
        "state_replay": replay,
        "authorization": {
            "production_pivot_classification_independently_verified": True,
            "successor_gate_may_decide_next_d0_scope": True,
            "discovery_outcome_opening_authorized_by_verifier_alone": False,
            "final_candidate_eligibility_authorized": False,
            "g0_measurement_freeze_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        },
        "claim_boundary": (
            "This independent PyArrow and NumPy pass exactly preserves all "
            "eight verified source columns and recomputes every one of the "
            "3,256,271 observable-history states and primary booleans with "
            "zero mismatch. NEW_TOPIC_ENTRY is zero prior observed target-"
            "topic work plus positive focal-year work under the current "
            "snapshot ABI, not true lifetime-first entry. The verifier reads "
            "no raw work or author-event partition, post-entry outcome, API "
            "result, model, or G4 result and establishes no association, "
            "causal mechanism, or scientific effect."
        ),
    }
    _write_json_new(verification_path, verification)
    os.chmod(verification_path, 0o600)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--verification-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        operator_base=args.operator_base,
        source_root=args.source_root,
        producer_root=args.producer_root,
        verification_path=args.verification_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
