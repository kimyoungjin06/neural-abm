#!/usr/bin/env python3
"""Build the bounded D0 descriptive persistence-calibration readout."""

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
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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
STATUS = "PASS_EXACT_4301_DESCRIPTIVE_PERSISTENCE_CALIBRATION_READOUT"
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


class PersistenceCalibrationReadoutError(RuntimeError):
    """Fail-closed descriptive readout error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PersistenceCalibrationReadoutError(message)


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
        raise PersistenceCalibrationReadoutError(
            f"invalid {label}: {error}"
        ) from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PersistenceCalibrationReadoutError(
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


def _numpy(table: pa.Table, name: str, dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    array = table[name].combine_chunks()
    require(array.null_count == 0, f"null source column {name}")
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def summarize_counts(
    t1: np.ndarray[Any, Any],
    t2: np.ndarray[Any, Any],
    t3: np.ndarray[Any, Any],
) -> dict[str, Any]:
    """Return the exact bounded descriptive distributions."""
    require(t1.shape == t2.shape == t3.shape, "count vector shape mismatch")
    total = t1.astype(np.uint64) + t2.astype(np.uint64) + t3.astype(np.uint64)
    persistent = total > 0
    rows = int(total.size)
    positive = int(np.count_nonzero(persistent))
    count_distribution = {
        "0": int(np.count_nonzero(total == 0)),
        "1": int(np.count_nonzero(total == 1)),
        "2": int(np.count_nonzero(total == 2)),
        "3": int(np.count_nonzero(total == 3)),
        "4": int(np.count_nonzero(total == 4)),
        "5_PLUS": int(np.count_nonzero(total >= 5)),
    }
    first_offsets = {
        "NONE": int(np.count_nonzero(~persistent)),
        "T_PLUS_1": int(np.count_nonzero(t1 > 0)),
        "T_PLUS_2": int(np.count_nonzero((t1 == 0) & (t2 > 0))),
        "T_PLUS_3": int(np.count_nonzero((t1 == 0) & (t2 == 0) & (t3 > 0))),
    }
    codes = (
        (t1 > 0).astype(np.uint8) * 4
        + (t2 > 0).astype(np.uint8) * 2
        + (t3 > 0).astype(np.uint8)
    )
    patterns = {
        pattern: int(np.count_nonzero(codes == code))
        for code, pattern in enumerate(PRESENCE_PATTERNS)
    }
    require(sum(count_distribution.values()) == rows, "count bins partition")
    require(sum(first_offsets.values()) == rows, "first offsets partition")
    require(sum(patterns.values()) == rows, "presence patterns partition")
    return {
        "rows": rows,
        "matured_rows": rows,
        "matured_persistence_rows": positive,
        "matured_no_persistence_rows": rows - positive,
        "right_censored_rows": 0,
        "descriptive_persistence_rate": positive / rows if rows else None,
        "qualifying_author_work_pairs": int(total.sum()),
        "target_topic_work_count_distribution": count_distribution,
        "first_observed_persistence_offset_distribution": first_offsets,
        "yearly_presence_pattern_distribution": patterns,
    }


def _validate_repository(config: Mapping[str, Any]) -> dict[str, Any]:
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
    required_fields = _mapping(binding.get("required_fields"), "gate required fields")
    for dotted, expected in required_fields.items():
        require(_nested(gate, dotted) == expected, f"gate field {dotted}")
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden inputs and outputs",
    )
    require(not any(forbidden.values()), "forbidden scope must remain false")
    return {
        "relative_path": str(gate_path.relative_to(REPO_ROOT)),
        "sha256": binding["sha256"],
        "schema_version": gate["schema_version"],
    }


def _validate_source(
    config: Mapping[str, Any],
    operator_base: Path,
) -> tuple[pa.Table, dict[str, Any]]:
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
    require(receipt.get("persistence_source_rows") == ROWS, "source receipt rows")
    require(receipt.get("denominator_rows_filtered") == 0, "source filtered rows")
    require(receipt.get("denominator_rows_mutated") == 0, "source mutated rows")
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
    return table, {
        "root_basename": root.name,
        "receipt_sha256": _sha256(receipt_path),
        "artifact_sha256": _sha256(artifact_path),
        "verification_sha256": _sha256(verification_path),
        "rows": table.num_rows,
        "columns": table.num_columns,
    }


def _build_readout(table: pa.Table) -> dict[str, Any]:
    t1 = _numpy(table, "target_topic_work_count_t_plus_1", np.dtype("<u4"))
    t2 = _numpy(table, "target_topic_work_count_t_plus_2", np.dtype("<u4"))
    t3 = _numpy(table, "target_topic_work_count_t_plus_3", np.dtype("<u4"))
    total = _numpy(
        table,
        "target_topic_work_count_t_plus_1_through_t_plus_3",
        np.dtype("<u4"),
    )
    episode = _numpy(table, "certified_episode_index", np.dtype("u1"))
    field = _numpy(table, "field_id", np.dtype("u1"))
    topic = _numpy(table, "topic_id", np.dtype("<u4"))
    year = _numpy(table, "entry_year", np.dtype("<u2"))
    author = _numpy(table, "author_numeric_id", np.dtype("<u8"))
    pivot = table["observable_history_pivot_state"].combine_chunks()
    complete = _numpy(table, "persistence_followup_complete", np.dtype("?"))
    outcome_state = table[
        "three_year_target_topic_persistence_state"
    ].combine_chunks()
    persistent = _numpy(
        table,
        "three_year_target_topic_persistent",
        np.dtype("?"),
    )
    derived_total = (
        t1.astype(np.uint64) + t2.astype(np.uint64) + t3.astype(np.uint64)
    )
    derived_persistent = derived_total > 0
    require(np.array_equal(total.astype(np.uint64), derived_total), "source total mismatch")
    require(np.all(complete), "incomplete source row")
    require(np.array_equal(persistent, derived_persistent), "source boolean mismatch")
    require(
        pc.all(pc.equal(pivot, "NEW_TOPIC_ENTRY")).as_py() is True,
        "non-primary pivot row",
    )
    expected_state = np.where(
        derived_persistent,
        "MATURED_PERSISTENCE",
        "MATURED_NO_PERSISTENCE",
    )
    observed_state = np.asarray(outcome_state.to_pylist(), dtype=object)
    require(np.array_equal(observed_state, expected_state), "source state mismatch")
    keys = {
        (
            int(episode[index]),
            int(field[index]),
            int(topic[index]),
            int(year[index]),
            int(author[index]),
        )
        for index in range(ROWS)
    }
    require(len(keys) == ROWS, "duplicate source key")

    by_episode: list[dict[str, Any]] = []
    for expected_episode, expected_field, expected_topic, expected_year in EPISODES:
        mask = episode == expected_episode
        require(np.count_nonzero(mask) > 0, f"empty episode {expected_episode}")
        require(np.all(field[mask] == expected_field), "episode field mismatch")
        require(np.all(topic[mask] == expected_topic), "episode topic mismatch")
        require(np.all(year[mask] == expected_year), "episode year mismatch")
        by_episode.append(
            {
                "certified_episode_index": expected_episode,
                "field_id": expected_field,
                "topic_id": expected_topic,
                "entry_year": expected_year,
                **summarize_counts(t1[mask], t2[mask], t3[mask]),
            }
        )
    require(sum(row["rows"] for row in by_episode) == ROWS, "episode partition")
    return {
        "overall": summarize_counts(t1, t2, t3),
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


def _render_markdown(readout: Mapping[str, Any]) -> str:
    overall = _mapping(readout.get("overall"), "overall")
    lines = [
        "# Persistence calibration readout",
        "",
        f"Status: `{STATUS}`",
        "",
        "| Episode | Rows | Persistent | No persistence | Rate | Work pairs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row_value in readout["by_episode"]:
        row = _mapping(row_value, "episode row")
        lines.append(
            f"| {row['certified_episode_index']} | {row['rows']:,} | "
            f"{row['matured_persistence_rows']:,} | "
            f"{row['matured_no_persistence_rows']:,} | "
            f"{row['descriptive_persistence_rate']:.2%} | "
            f"{row['qualifying_author_work_pairs']:,} |"
        )
    lines.extend(
        [
            f"| Total | {overall['rows']:,} | "
            f"{overall['matured_persistence_rows']:,} | "
            f"{overall['matured_no_persistence_rows']:,} | "
            f"{overall['descriptive_persistence_rate']:.2%} | "
            f"{overall['qualifying_author_work_pairs']:,} |",
            "",
            "This aggregate-only readout uses only the independently verified "
            "4,301-row source. Rates are unadjusted descriptions, not comparable "
            "field effects, associations, causal estimates, or scientific effects.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_output(
    *,
    output: Path,
    config_path: Path,
    gate_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    readout: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    require(not output.exists(), "output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": STATUS,
            "mode": "EXISTING_VERIFIED_4301_ROW_ARTIFACT_ONLY_DESCRIPTIVE_MEASUREMENT_CALIBRATION",
            "scope": {
                "source_rows_read": ROWS,
                "raw_work_or_author_event_partition_rows_read": 0,
                "confirmatory_or_holdout_rows_read": 0,
                "exposure_peer_citation_impact_or_affiliation_rows_joined": 0,
                "api_requests_created_or_submitted": 0,
                "adjusted_comparison_or_inferential_effect_computed": False,
                "outcome_based_ranking_or_selection_performed": False,
                "author_level_rows_written": 0,
                "model_fitted": False,
                "g4_open": False,
            },
            "repository_bindings": {"gate_state": dict(gate_binding)},
            "source_binding": dict(source_binding),
            "calibration_readout": dict(readout),
            "execution": {
                "builder_relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "config_sha256": _sha256(config_path),
                "elapsed_seconds": elapsed_seconds,
                "gpu_used": False,
                "workload_class": "CPU_SMALL_AGGREGATE",
            },
            "authorization": {
                "descriptive_calibration_readout": "COMPLETE_PENDING_INDEPENDENT_REPLAY",
                "exposure_or_outcome_association_analysis": "NOT_AUTHORIZED",
                "outcome_based_selection": "NOT_AUTHORIZED",
                "final_candidate_eligibility": "NOT_AUTHORIZED",
                "g0_measurement_freeze": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "claim_boundary": (
                "This aggregate-only readout deterministically summarizes the "
                "independently verified 4,301-row current-snapshot-annotated "
                "persistence source. It establishes no adjusted episode "
                "comparison, exposure association, causality, final eligibility, "
                "G0 freeze, model result, G4 result, or scientific effect."
            ),
        }
        report_path = temporary / "persistence_calibration_readout.json"
        markdown_path = temporary / "PERSISTENCE_CALIBRATION_READOUT.md"
        _write_json(report_path, report)
        _write_text(markdown_path, _render_markdown(readout))
        artifacts = [
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (markdown_path, report_path)
        ]
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "root_basename": output.name,
            "config_sha256": _sha256(config_path),
            "source_receipt_sha256": source_binding["receipt_sha256"],
            "source_artifact_sha256": source_binding["artifact_sha256"],
            "source_verification_sha256": source_binding["verification_sha256"],
            "source_rows_read": ROWS,
            "aggregate_only_no_author_ids": True,
            "report_rows": len(readout["by_episode"]) + 1,
            "matured_persistence_rows": readout["overall"]["matured_persistence_rows"],
            "matured_no_persistence_rows": readout["overall"][
                "matured_no_persistence_rows"
            ],
            "right_censored_rows": readout["overall"]["right_censored_rows"],
            "association_or_effect_computed": False,
            "api_requests_created": 0,
            "model_fitted": False,
            "g4_open": False,
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes(
                    [
                        (item["relative_path"], item["bytes"], item["sha256"])
                        for item in sorted(artifacts, key=lambda value: value["relative_path"])
                    ]
                )
            ).hexdigest(),
            "artifacts": artifacts,
            "independent_verification_complete": False,
            "receipt_written_last": True,
        }
        _write_json(
            temporary / "persistence_calibration_readout.outputs.json",
            receipt,
        )
        require({path.name for path in temporary.iterdir()} == EXPECTED_FILES, "output files")
        for path in temporary.iterdir():
            os.chmod(path, 0o600)
        temporary.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(config_path: Path, operator_base: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = _read_yaml(config_path, "config")
    gate_binding = _validate_repository(config)
    expected_output = _mapping(config.get("output_contract"), "output contract")
    require(output.name == expected_output.get("root_basename"), "output basename")
    table, source_binding = _validate_source(config, operator_base)
    readout = _build_readout(table)
    return _write_output(
        output=output,
        config_path=config_path,
        gate_binding=gate_binding,
        source_binding=source_binding,
        readout=readout,
        elapsed_seconds=round(time.perf_counter() - started, 6),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.config.resolve(strict=True),
        args.operator_base.resolve(strict=True),
        args.output.resolve(strict=False),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
