#!/usr/bin/env python3
"""Independently replay the D0 descriptive persistence-calibration readout."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_persistence_calibration_readout_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_d0_persistence_calibration_readout_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_persistence_calibration_readout_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_persistence_calibration_readout_outputs.v1"
SOURCE_SCHEMA = "pivot_feedback_d0_persistence_outcome_source.v1"
SOURCE_RECEIPT_SCHEMA = "pivot_feedback_d0_persistence_outcome_source_outputs.v1"
SOURCE_VERIFY_SCHEMA = (
    "pivot_feedback_d0_persistence_outcome_source_verification.v1"
)
STATUS = (
    "PASS_INDEPENDENT_EXACT_4301_DESCRIPTIVE_PERSISTENCE_CALIBRATION_ALL_VALUE_REPLAY"
)
PRODUCER_STATUS = "PASS_EXACT_4301_DESCRIPTIVE_PERSISTENCE_CALIBRATION_READOUT"
ROWS = 4_301
EXPECTED_FILES = {
    "PERSISTENCE_CALIBRATION_READOUT.md",
    "persistence_calibration_readout.json",
    "persistence_calibration_readout.outputs.json",
}
COUNT_BINS = ("0", "1", "2", "3", "4", "5_PLUS")
FIRST_OFFSETS = ("NONE", "T_PLUS_1", "T_PLUS_2", "T_PLUS_3")
PRESENCE_PATTERNS = ("000", "001", "010", "011", "100", "101", "110", "111")
EPISODES = (
    (1, 17, 14430, 2003),
    (2, 17, 10579, 2012),
    (3, 27, 10151, 1993),
    (4, 27, 12421, 1953),
)
SOURCE_COLUMNS = (
    "certified_episode_index",
    "field_id",
    "topic_id",
    "entry_year",
    "author_numeric_id",
    "observable_history_pivot_state",
    "persistence_followup_complete",
    "target_topic_work_count_t_plus_1",
    "target_topic_work_count_t_plus_2",
    "target_topic_work_count_t_plus_3",
    "target_topic_work_count_t_plus_1_through_t_plus_3",
    "three_year_target_topic_persistence_state",
    "three_year_target_topic_persistent",
)
SOURCE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("observable_history_pivot_state", pa.string(), nullable=False),
        pa.field("persistence_followup_complete", pa.bool_(), nullable=False),
        pa.field("target_topic_work_count_t_plus_1", pa.uint32()),
        pa.field("target_topic_work_count_t_plus_2", pa.uint32()),
        pa.field("target_topic_work_count_t_plus_3", pa.uint32()),
        pa.field("target_topic_work_count_t_plus_1_through_t_plus_3", pa.uint32()),
        pa.field(
            "three_year_target_topic_persistence_state",
            pa.string(),
            nullable=False,
        ),
        pa.field("three_year_target_topic_persistent", pa.bool_()),
    ],
    metadata={b"schema_version": SOURCE_SCHEMA.encode()},
)


class PersistenceCalibrationVerificationError(RuntimeError):
    """Fail-closed independent readout verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistenceCalibrationVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


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
        raise PersistenceCalibrationVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PersistenceCalibrationVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "verification output already exists")
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


def summarize_rows(rows: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Independently summarize `(t1, t2, t3)` integer rows."""
    count_bins: Counter[str] = Counter()
    first_offsets: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    positive = 0
    pairs = 0
    for raw in rows:
        require(len(raw) == 3, "count row width")
        t1, t2, t3 = (int(value) for value in raw)
        require(t1 >= 0 and t2 >= 0 and t3 >= 0, "negative count")
        total = t1 + t2 + t3
        pairs += total
        positive += int(total > 0)
        count_bins[str(total) if total <= 4 else "5_PLUS"] += 1
        if t1 > 0:
            first_offsets["T_PLUS_1"] += 1
        elif t2 > 0:
            first_offsets["T_PLUS_2"] += 1
        elif t3 > 0:
            first_offsets["T_PLUS_3"] += 1
        else:
            first_offsets["NONE"] += 1
        patterns[f"{int(t1 > 0)}{int(t2 > 0)}{int(t3 > 0)}"] += 1
    row_count = len(rows)
    return {
        "rows": row_count,
        "matured_rows": row_count,
        "matured_persistence_rows": positive,
        "matured_no_persistence_rows": row_count - positive,
        "right_censored_rows": 0,
        "descriptive_persistence_rate": positive / row_count if row_count else None,
        "qualifying_author_work_pairs": pairs,
        "target_topic_work_count_distribution": {
            key: count_bins[key] for key in COUNT_BINS
        },
        "first_observed_persistence_offset_distribution": {
            key: first_offsets[key] for key in FIRST_OFFSETS
        },
        "yearly_presence_pattern_distribution": {
            key: patterns[key] for key in PRESENCE_PATTERNS
        },
    }


def _validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_yaml(config_path, "config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V47",
        "config authorization",
    )
    binding = _mapping(
        _mapping(config.get("repository_bindings"), "repository bindings").get(
            "gate_state"
        ),
        "gate binding",
    )
    gate_path = _safe_repo_path(binding.get("relative_path"), "gate path")
    require(_sha256(gate_path) == binding.get("sha256"), "gate hash")
    gate = _read_yaml(gate_path, "gate")
    require(gate.get("schema_version") == binding.get("required_schema_version"), "gate schema")
    require(gate.get("status") == binding.get("required_status"), "gate status")
    for dotted, expected in _mapping(
        binding.get("required_fields"),
        "gate required fields",
    ).items():
        require(_nested(gate, dotted) == expected, f"gate field {dotted}")
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden inputs and outputs",
    )
    require(not any(forbidden.values()), "forbidden scope must remain false")
    return config, {
        "relative_path": str(gate_path.relative_to(REPO_ROOT)),
        "sha256": binding["sha256"],
        "schema_version": gate["schema_version"],
    }


def _source_readout(
    config: Mapping[str, Any],
    operator_base: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _mapping(config.get("operator_input"), "operator input")
    root = operator_base / str(source.get("root_basename"))
    require(root.is_dir() and not root.is_symlink(), "source root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "source root mode")
    receipt_path = root / str(source.get("receipt_basename"))
    artifact_path = root / str(source.get("artifact_basename"))
    verification_path = operator_base / str(
        source.get("external_verification_basename")
    )
    require(_sha256(receipt_path) == source.get("receipt_sha256"), "source receipt hash")
    require(_sha256(artifact_path) == source.get("artifact_sha256"), "source artifact hash")
    require(
        _sha256(verification_path) == source.get("external_verification_sha256"),
        "source verification hash",
    )
    receipt = _read_json(receipt_path, "source receipt")
    verification = _read_json(verification_path, "source verification")
    require(receipt.get("schema_version") == SOURCE_RECEIPT_SCHEMA, "source receipt schema")
    require(receipt.get("status") == source.get("required_status"), "source status")
    require(
        verification.get("schema_version") == SOURCE_VERIFY_SCHEMA,
        "source verification schema",
    )
    require(
        verification.get("status") == source.get("external_verification_status"),
        "source verification status",
    )
    require(verification.get("output_receipt_sha256") == _sha256(receipt_path), "verified receipt")
    require(verification.get("output_artifact_sha256") == _sha256(artifact_path), "verified artifact")
    table = pq.read_table(artifact_path, columns=list(SOURCE_COLUMNS))
    require(table.num_rows == ROWS, "source rows")
    require(table.schema.equals(SOURCE_ARROW_SCHEMA, check_metadata=True), "source schema")
    values = table.to_pydict()
    keys: set[tuple[int, int, int, int, int]] = set()
    episode_rows: dict[int, list[tuple[int, int, int]]] = {
        item[0]: [] for item in EPISODES
    }
    all_rows: list[tuple[int, int, int]] = []
    for index in range(ROWS):
        episode = int(values["certified_episode_index"][index])
        key = (
            episode,
            int(values["field_id"][index]),
            int(values["topic_id"][index]),
            int(values["entry_year"][index]),
            int(values["author_numeric_id"][index]),
        )
        require(key not in keys, "duplicate source key")
        keys.add(key)
        require(
            values["observable_history_pivot_state"][index] == "NEW_TOPIC_ENTRY",
            "non-primary pivot row",
        )
        require(values["persistence_followup_complete"][index] is True, "incomplete row")
        raw_counts = (
            values["target_topic_work_count_t_plus_1"][index],
            values["target_topic_work_count_t_plus_2"][index],
            values["target_topic_work_count_t_plus_3"][index],
        )
        require(all(value is not None for value in raw_counts), "null count")
        counts = tuple(int(value) for value in raw_counts)
        total = sum(counts)
        require(
            values["target_topic_work_count_t_plus_1_through_t_plus_3"][index]
            == total,
            "source total mismatch",
        )
        expected_positive = total > 0
        require(
            values["three_year_target_topic_persistent"][index]
            is expected_positive,
            "source boolean mismatch",
        )
        require(
            values["three_year_target_topic_persistence_state"][index]
            == (
                "MATURED_PERSISTENCE"
                if expected_positive
                else "MATURED_NO_PERSISTENCE"
            ),
            "source state mismatch",
        )
        require(episode in episode_rows, "unexpected episode")
        episode_rows[episode].append(counts)
        all_rows.append(counts)

    by_episode: list[dict[str, Any]] = []
    for episode, field, topic, year in EPISODES:
        indices = [
            index
            for index, value in enumerate(values["certified_episode_index"])
            if value == episode
        ]
        require(indices, f"empty episode {episode}")
        require(all(values["field_id"][index] == field for index in indices), "episode field")
        require(all(values["topic_id"][index] == topic for index in indices), "episode topic")
        require(all(values["entry_year"][index] == year for index in indices), "episode year")
        by_episode.append(
            {
                "certified_episode_index": episode,
                "field_id": field,
                "topic_id": topic,
                "entry_year": year,
                **summarize_rows(episode_rows[episode]),
            }
        )
    readout = {
        "overall": summarize_rows(all_rows),
        "by_episode": by_episode,
        "invariants": {
            "source_rows": ROWS,
            "duplicate_key_rows": 0,
            "non_new_topic_entry_rows": 0,
            "incomplete_followup_rows": 0,
            "null_count_state_or_boolean_rows": 0,
            "total_count_mismatch_rows": 0,
            "persistence_state_mismatch_rows": 0,
            "persistence_boolean_mismatch_rows": 0,
            "distribution_partition_mismatch_count": 0,
        },
    }
    return readout, {
        "root_basename": root.name,
        "receipt_sha256": _sha256(receipt_path),
        "artifact_sha256": _sha256(artifact_path),
        "verification_sha256": _sha256(verification_path),
        "rows": table.num_rows,
        "columns": table.num_columns,
    }


def _validate_output(
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == output.get("root_basename"), "output basename")
    require(output_root.is_dir() and not output_root.is_symlink(), "output root")
    require(stat.S_IMODE(output_root.stat().st_mode) == 0o700, "output root mode")
    require({path.name for path in output_root.iterdir()} == EXPECTED_FILES, "output files")
    report_path = output_root / str(output.get("report_basename"))
    markdown_path = output_root / str(output.get("markdown_basename"))
    receipt_path = output_root / str(output.get("receipt_basename"))
    for path in (report_path, markdown_path, receipt_path):
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"file mode {path.name}")
    report = _read_json(report_path, "producer report")
    receipt = _read_json(receipt_path, "producer receipt")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema")
    require(report.get("status") == PRODUCER_STATUS, "report status")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(receipt.get("status") == PRODUCER_STATUS, "receipt status")
    require(receipt.get("source_rows_read") == ROWS, "receipt source rows")
    require(receipt.get("aggregate_only_no_author_ids") is True, "aggregate boundary")
    require(receipt.get("association_or_effect_computed") is False, "effect boundary")
    require(receipt.get("api_requests_created") == 0, "API boundary")
    require(receipt.get("receipt_written_last") is True, "receipt written last")
    artifacts = {
        item["relative_path"]: item
        for item in receipt.get("artifacts", [])
        if isinstance(item, Mapping)
    }
    require(set(artifacts) == {report_path.name, markdown_path.name}, "receipt artifacts")
    for path in (report_path, markdown_path):
        item = artifacts[path.name]
        require(item.get("bytes") == path.stat().st_size, f"bytes {path.name}")
        require(item.get("sha256") == _sha256(path), f"hash {path.name}")
    tree = hashlib.sha256(
        _canonical_bytes(
            [
                (item["relative_path"], item["bytes"], item["sha256"])
                for item in sorted(artifacts.values(), key=lambda value: value["relative_path"])
            ]
        )
    ).hexdigest()
    require(receipt.get("artifact_tree_sha256") == tree, "artifact tree")
    require("author_numeric_id" not in report_path.read_text(encoding="utf-8"), "author id field leak")
    require("author_numeric_id" not in markdown_path.read_text(encoding="utf-8"), "author id markdown leak")
    return report, {
        "receipt_sha256": _sha256(receipt_path),
        "report_sha256": _sha256(report_path),
        "markdown_sha256": _sha256(markdown_path),
        "artifact_tree_sha256": tree,
    }


def run(
    config_path: Path,
    operator_base: Path,
    output_root: Path,
    verification_output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, gate_binding = _validate_config(config_path)
    expected, source_binding = _source_readout(config, operator_base)
    report, output_hashes = _validate_output(config, output_root)
    observed = report.get("calibration_readout")
    require(observed == expected, "derived readout mismatch")
    result = {
        "schema_version": (
            "pivot_feedback_d0_persistence_calibration_readout_verification.v1"
        ),
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "output_root_basename": output_root.name,
        "repository_bindings": {"gate_state": gate_binding},
        "source_binding": source_binding,
        "output_hashes": output_hashes,
        "readout_replay": expected,
        "mismatches": {
            "overall_value_mismatches": 0,
            "episode_value_mismatches": 0,
            "distribution_value_mismatches": 0,
            "invariant_value_mismatches": 0,
        },
        "verifier": {
            "strategy": "PYARROW_PYTHON_ROW_WISE_ALL_SOURCE_AND_DERIVED_VALUE_REPLAY",
            "producer_strategy": "PYARROW_NUMPY_VECTOR_AGGREGATION",
            "imports_producer_module": False,
            "uses_producer_aggregation_implementation": False,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        },
        "scope": {
            "source_rows_read": ROWS,
            "raw_work_or_author_event_partition_rows_read": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "joined_external_rows": 0,
            "api_requests_created_or_submitted": 0,
            "adjusted_comparison_or_inferential_effect_computed": False,
            "outcome_based_ranking_or_selection_performed": False,
            "model_fitted": False,
            "g4_open": False,
        },
        "authorization": {
            "descriptive_calibration_readout_independently_verified": True,
            "d0_complete": False,
            "g0_measurement_freeze_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "This independent row-wise replay verifies only deterministic "
            "aggregate summaries of the bound 4,301-row persistence source. "
            "It establishes no adjusted comparison, association, causality, "
            "final eligibility, G0 freeze, model result, G4 result, or "
            "scientific effect."
        ),
    }
    _write_json(verification_output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.config.resolve(strict=True),
        args.operator_base.resolve(strict=True),
        args.output_root.resolve(strict=True),
        args.verification_output.resolve(strict=False),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
