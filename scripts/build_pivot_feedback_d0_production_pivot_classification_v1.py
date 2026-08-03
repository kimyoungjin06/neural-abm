#!/usr/bin/env python3
"""Apply the frozen four-state classifier to the verified three-count source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
from typing import Any, Mapping, Sequence

import duckdb
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
SOURCE_SCHEMA = "pivot_feedback_d0_entry_source_features.v1"
SOURCE_VERIFY_SCHEMA = "pivot_feedback_d0_entry_source_verification.v1"
SOURCE_STATUS = (
    "PASS_EXACT_FOUR_THREE_COUNT_PRODUCTION_FEATURE_SOURCE_NO_STATE_LABELS"
)
SOURCE_VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_FOUR_THREE_COUNT_SOURCE_REPLAY_NO_STATE_LABELS"
)
STATUS = (
    "PASS_EXACT_FOUR_THRESHOLD_FREE_OBSERVABLE_HISTORY_FOUR_STATE_CLASSIFICATION"
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


class ProductionPivotClassificationError(RuntimeError):
    """Fail-closed production pivot-classification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionPivotClassificationError(message)


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
        raise ProductionPivotClassificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ProductionPivotClassificationError(
            f"invalid {label}: {error}"
        ) from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


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


def _validate_input(
    *,
    config: Mapping[str, Any],
    operator_base: Path,
    source_root: Path,
) -> dict[str, Any]:
    spec = _mapping(config.get("verified_operator_input"), "operator input")
    source_root = source_root.resolve(strict=True)
    require(source_root.parent == operator_base, "source root parent")
    require(source_root.name == spec.get("root_basename"), "source root basename")
    require(
        source_root.is_dir() and not source_root.is_symlink(),
        "source root",
    )
    require(stat.S_IMODE(source_root.stat().st_mode) == 0o700, "source root mode")
    receipt_path = source_root / str(spec.get("receipt_basename"))
    require(_sha256(receipt_path) == spec.get("receipt_sha256"), "source receipt SHA")
    receipt = _read_json(receipt_path, "source receipt")
    require(
        receipt.get("schema_version") == spec.get("receipt_schema_version"),
        "source receipt schema",
    )
    require(receipt.get("status") == spec.get("receipt_status"), "source status")
    require(
        receipt.get("artifact_tree_sha256") == spec.get("artifact_tree_sha256"),
        "source artifact tree",
    )
    require(receipt.get("feature_rows") == spec.get("feature_rows") == ROWS, "source rows")
    require(receipt.get("feature_columns") == spec.get("feature_columns") == 8, "source columns")
    require(receipt.get("production_classifier_state_materialized") is False, "source state boundary")
    require(receipt.get("pivot_label_materialized") is False, "source label boundary")
    require(receipt.get("outcomes_read") is False, "source outcome boundary")
    require(receipt.get("api_requests_created") == 0, "source API boundary")
    feature_path = source_root / str(spec.get("feature_basename"))
    require(_sha256(feature_path) == spec.get("feature_sha256"), "source feature SHA")
    artifact = _artifact_items(receipt).get(feature_path.name)
    require(artifact is not None, "source feature absent from receipt")
    require(artifact.get("sha256") == spec.get("feature_sha256"), "source receipt feature SHA")
    parquet = pq.ParquetFile(feature_path)
    require(parquet.metadata.num_rows == ROWS, "source Parquet rows")
    require(
        parquet.schema_arrow.equals(SOURCE_ARROW_SCHEMA, check_metadata=True),
        "source Parquet schema",
    )
    require(
        tuple(spec.get("exact_input_columns", ())) == SOURCE_COLUMNS,
        "source input columns",
    )
    verification_path = operator_base / str(spec.get("verification_basename"))
    require(
        _sha256(verification_path) == spec.get("verification_sha256"),
        "source verification SHA",
    )
    verification = _read_json(verification_path, "source verification")
    require(
        verification.get("schema_version") == spec.get("verification_schema_version"),
        "source verification schema",
    )
    require(
        verification.get("status") == spec.get("verification_status"),
        "source verification status",
    )
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
    return {
        "feature_path": feature_path,
        "receipt_path": receipt_path,
        "binding": {
            "root_basename": source_root.name,
            "receipt_sha256": spec["receipt_sha256"],
            "artifact_tree_sha256": spec["artifact_tree_sha256"],
            "feature_sha256": spec["feature_sha256"],
            "verification_sha256": spec["verification_sha256"],
            "feature_rows": ROWS,
            "feature_columns": 8,
        },
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


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    row = connection.execute(query).fetchone()
    require(row is not None and len(row) == 1, "scalar query")
    return int(row[0])


def _write_classification(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_path: Path,
    output_path: Path,
) -> int:
    source = _sql_path(source_path)
    query = f"""
        SELECT
          certified_episode_index,
          field_id,
          topic_id,
          entry_year,
          author_numeric_id,
          pre_entry_target_topic_work_count,
          trailing_3y_target_topic_work_count,
          focal_confirmed_eligible_target_topic_work_count,
          CASE
            WHEN focal_confirmed_eligible_target_topic_work_count = 0
              THEN 'NO_FOCAL_ENTRY'
            WHEN pre_entry_target_topic_work_count = 0
              THEN 'NEW_TOPIC_ENTRY'
            WHEN trailing_3y_target_topic_work_count = 0
              THEN 'RETURN'
            ELSE 'CONTINUATION'
          END AS observable_history_pivot_state,
          CAST(
            focal_confirmed_eligible_target_topic_work_count > 0
            AND pre_entry_target_topic_work_count = 0
            AS BOOLEAN
          ) AS observable_history_new_topic_entry
        FROM read_parquet('{source}')
        ORDER BY
          certified_episode_index,
          field_id,
          topic_id,
          entry_year,
          author_numeric_id
    """
    table = connection.execute(query).fetch_arrow_table().cast(OUTPUT_ARROW_SCHEMA)
    require(table.num_rows == ROWS, "classification rows")
    partial = output_path.with_name(f".{output_path.name}.partial")
    pq.write_table(
        table,
        partial,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=262_144,
    )
    os.chmod(partial, 0o600)
    partial.rename(output_path)
    return table.num_rows


def _classification_accounting(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> dict[str, Any]:
    escaped = _sql_path(path)
    connection.execute(
        f"CREATE TEMP VIEW classified AS SELECT * FROM read_parquet('{escaped}')"
    )
    require(_scalar(connection, "SELECT count(*) FROM classified") == ROWS, "output rows")
    require(
        _scalar(
            connection,
            """
            SELECT count(*) - count(
              DISTINCT row(
                certified_episode_index,
                field_id,
                topic_id,
                entry_year,
                author_numeric_id
              )
            )
            FROM classified
            """,
        )
        == 0,
        "output duplicate key",
    )
    state_rule_mismatches = _scalar(
        connection,
        """
        SELECT count(*) FROM classified
        WHERE observable_history_pivot_state <> CASE
          WHEN focal_confirmed_eligible_target_topic_work_count = 0
            THEN 'NO_FOCAL_ENTRY'
          WHEN pre_entry_target_topic_work_count = 0
            THEN 'NEW_TOPIC_ENTRY'
          WHEN trailing_3y_target_topic_work_count = 0
            THEN 'RETURN'
          ELSE 'CONTINUATION'
        END
        """,
    )
    boolean_mismatches = _scalar(
        connection,
        """
        SELECT count(*) FROM classified
        WHERE observable_history_new_topic_entry
          <> (observable_history_pivot_state = 'NEW_TOPIC_ENTRY')
        """,
    )
    require(state_rule_mismatches == boolean_mismatches == 0, "classifier mismatch")
    raw = connection.execute(
        """
        SELECT
          certified_episode_index,
          field_id,
          topic_id,
          entry_year,
          count(*) AS author_rows,
          count(*) FILTER (
            WHERE observable_history_pivot_state = 'NO_FOCAL_ENTRY'
          ) AS no_focal_entry_rows,
          count(*) FILTER (
            WHERE observable_history_pivot_state = 'NEW_TOPIC_ENTRY'
          ) AS new_topic_entry_rows,
          count(*) FILTER (
            WHERE observable_history_pivot_state = 'RETURN'
          ) AS return_rows,
          count(*) FILTER (
            WHERE observable_history_pivot_state = 'CONTINUATION'
          ) AS continuation_rows
        FROM classified
        GROUP BY ALL
        ORDER BY certified_episode_index
        """
    ).fetchall()
    names = (
        "certified_episode_index",
        "field_id",
        "topic_id",
        "entry_year",
        "author_rows",
        "no_focal_entry_rows",
        "new_topic_entry_rows",
        "return_rows",
        "continuation_rows",
    )
    by_episode = [dict(zip(names, map(int, row), strict=True)) for row in raw]
    require(len(by_episode) == 4, "episode classification rows")
    for row in by_episode:
        require(
            row["author_rows"]
            == sum(
                row[name]
                for name in (
                    "no_focal_entry_rows",
                    "new_topic_entry_rows",
                    "return_rows",
                    "continuation_rows",
                )
            ),
            "episode state closure",
        )
    totals = {
        name: sum(row[name] for row in by_episode)
        for name in (
            "author_rows",
            "no_focal_entry_rows",
            "new_topic_entry_rows",
            "return_rows",
            "continuation_rows",
        )
    }
    require(totals["author_rows"] == ROWS, "total state closure")
    require(
        totals["author_rows"]
        == sum(totals[f"{state.lower()}_rows"] for state in STATES),
        "four-state partition closure",
    )
    return {
        "by_episode": by_episode,
        "totals": totals,
        "state_rule_mismatches": state_rule_mismatches,
        "primary_boolean_mismatches": boolean_mismatches,
        "source_rows_filtered": 0,
        "source_feature_value_mutations": 0,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    totals = report["classification_accounting"]["totals"]
    return f"""# Production observable-history pivot classification

Status: `{report['status']}`

The deterministic threshold-free four-state classifier preserves all
{totals['author_rows']:,} verified author-episode rows:

- `NO_FOCAL_ENTRY`: {totals['no_focal_entry_rows']:,}
- `NEW_TOPIC_ENTRY`: {totals['new_topic_entry_rows']:,}
- `RETURN`: {totals['return_rows']:,}
- `CONTINUATION`: {totals['continuation_rows']:,}

`NEW_TOPIC_ENTRY` is an observable-history state under the current-snapshot
topic and author-token ABI. It is not a true lifetime-first claim.

No raw work or author-event partition, post-entry persistence or impact, API,
model, or G4 result was read. Independent all-row state replay is required
before any successor decision.
"""


def build(
    *,
    config_path: Path,
    operator_base: Path,
    source_root: Path,
    output: Path,
    threads: int,
) -> dict[str, Any]:
    started = time.monotonic()
    config_path = config_path.resolve(strict=True)
    operator_base = operator_base.resolve(strict=True)
    output = output.resolve()
    require(output.parent == operator_base, "output parent")
    config = _read_yaml(config_path, "config")
    repository = _validate_repository(config)
    classification = _mapping(
        config.get("classification_contract"),
        "classification contract",
    )
    require(tuple(classification.get("exact_states", ())) == STATES, "states")
    require(
        classification.get("minimum_history_threshold") == "NONE",
        "history threshold",
    )
    require(classification.get("graph_team_cap") == "NONE", "graph cap")
    require(classification.get("post_entry_input_allowed") is False, "post-entry input")
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden inputs and outputs",
    )
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary")
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output.name == output_contract.get("root_basename"), "output basename")
    require(not output.exists(), "output already exists")
    require(
        tuple(output_contract.get("required_columns", ())) == OUTPUT_COLUMNS,
        "output columns",
    )
    source = _validate_input(
        config=config,
        operator_base=operator_base,
        source_root=source_root,
    )
    source_stat = source["feature_path"].stat()
    source_identity = (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=operator_base)
    )
    os.chmod(temporary, 0o700)
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        connection = duckdb.connect(database=":memory:")
        connection.execute(f"PRAGMA threads={max(1, threads)}")
        source_path = source["feature_path"]
        escaped_source = _sql_path(source_path)
        require(
            _scalar(
                connection,
                f"SELECT count(*) FROM read_parquet('{escaped_source}')",
            )
            == ROWS,
            "source row count",
        )
        invariant_violations = {
            "trailing_count_exceeds_pre_entry_count": _scalar(
                connection,
                f"""
                SELECT count(*) FROM read_parquet('{escaped_source}')
                WHERE trailing_3y_target_topic_work_count
                    > pre_entry_target_topic_work_count
                """,
            ),
            "zero_pre_entry_positive_trailing": _scalar(
                connection,
                f"""
                SELECT count(*) FROM read_parquet('{escaped_source}')
                WHERE pre_entry_target_topic_work_count = 0
                  AND trailing_3y_target_topic_work_count > 0
                """,
            ),
        }
        require(
            all(value == 0 for value in invariant_violations.values()),
            "source invariant",
        )
        artifact_path = temporary / str(output_contract["artifact_basename"])
        _write_classification(
            connection,
            source_path=source_path,
            output_path=artifact_path,
        )
        accounting = _classification_accounting(connection, artifact_path)
        current = source_path.stat()
        require(
            source_identity
            == (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ),
            "source changed during classification",
        )
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": STATUS,
            "mode": config["mode"],
            "config_sha256": _sha256(config_path),
            "repository_bindings": repository,
            "verified_source_binding": source["binding"],
            "scope": {
                "exact_certified_episode_rows": 4,
                "registered_author_episode_rows": ROWS,
                "source_rows_filtered": 0,
                "source_feature_mutations": 0,
                "raw_work_partition_rows_read": 0,
                "raw_author_event_partition_rows_read": 0,
                "post_entry_offsets_read": [],
                "persistence_or_impact_rows_read": False,
                "api_requests_created": 0,
                "model_fitted": False,
                "g4_open": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "definition": {
                "unit": classification["unit"],
                "states": list(STATES),
                "primary_pivot_state": "NEW_TOPIC_ENTRY",
                "minimum_history_threshold": "NONE",
                "graph_team_cap": "NONE",
                "observable_history_claim_only": True,
                "true_lifetime_first_claim_allowed": False,
            },
            "source_invariant_violations": invariant_violations,
            "classification_accounting": accounting,
            "authorization": {
                "independent_state_replay": "REQUIRED_NEXT",
                "discovery_outcome_opening": "NOT_AUTHORIZED",
                "final_candidate_eligibility": "NOT_AUTHORIZED",
                "g0_measurement_freeze": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
            },
            "execution": {
                "workload_class": "CPU_IO",
                "duckdb_threads": max(1, threads),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "builder_relative_path": Path(__file__)
                .resolve()
                .relative_to(REPO_ROOT)
                .as_posix(),
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "gpu_used": False,
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / str(output_contract["report_basename"])
        _write_json(report_path, report)
        markdown_path = temporary / str(output_contract["markdown_basename"])
        _write_text(markdown_path, _markdown(report))
        connection.close()
        connection = None

        artifacts: list[dict[str, Any]] = []
        for path in (artifact_path, report_path, markdown_path):
            os.chmod(path, 0o600)
            item: dict[str, Any] = {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            if path.suffix == ".parquet":
                item["rows"] = pq.ParquetFile(path).metadata.num_rows
            artifacts.append(item)
        artifacts.sort(key=lambda item: str(item["relative_path"]))
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "root_basename": output.name,
            "config_sha256": _sha256(config_path),
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes(
                    [
                        (item["relative_path"], item["bytes"], item["sha256"])
                        for item in artifacts
                    ]
                )
            ).hexdigest(),
            "classification_rows": ROWS,
            "classification_columns": len(OUTPUT_COLUMNS),
            "four_state_partition_complete": True,
            "new_topic_entry_rows": accounting["totals"]["new_topic_entry_rows"],
            "source_rows_filtered": 0,
            "source_feature_mutations": 0,
            "raw_work_partition_rows_read": 0,
            "raw_author_event_partition_rows_read": 0,
            "post_entry_rows_read": False,
            "outcomes_read": False,
            "api_requests_created": 0,
            "independent_verification_complete": False,
            "g0_measurement_freeze_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "receipt_written_last": True,
        }
        receipt_path = temporary / str(output_contract["receipt_basename"])
        _write_json(receipt_path, receipt)
        require(
            {path.name for path in temporary.iterdir()} == EXPECTED_FILES,
            "unexpected output files",
        )
        os.replace(temporary, output)
        return report
    finally:
        if connection is not None:
            connection.close()
        if temporary.exists():
            require(
                temporary.parent == operator_base
                and temporary.name.startswith(f".{output.name}.partial-"),
                "unsafe temporary cleanup target",
            )
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build(
        config_path=args.config,
        operator_base=args.operator_base,
        source_root=args.source_root,
        output=args.output,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
