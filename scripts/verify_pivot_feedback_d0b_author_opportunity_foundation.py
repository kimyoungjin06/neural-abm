#!/usr/bin/env python3
"""Independently replay the D0-B pre-entry author-opportunity foundation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "pivot_feedback_d0b_author_opportunity_foundation_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0b_author_opportunity_foundation_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0b_author_opportunity_foundation_outputs.v1"
WORK_SCHEMA_VERSION = "pivot_feedback_d0b_selected_predecision_work.v1"
AUTHOR_SCHEMA_VERSION = "pivot_feedback_d0b_pre_entry_author_measurement_base.v1"
VERIFY_SCHEMA = (
    "pivot_feedback_d0b_author_opportunity_foundation_verification.v1"
)
OUTPUT_STATUS = "PASS_D0B_OUTCOME_BLIND_PRE_ENTRY_AUTHOR_OPPORTUNITY_FOUNDATION"
VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0B_OUTCOME_BLIND_PRE_ENTRY_AUTHOR_OPPORTUNITY_"
    "FOUNDATION_D0B_AND_D0_STILL_PARTIAL"
)
PARTITIONS = 4096
MAX_CONTROL_BYTES = 32 * 1024 * 1024
EXPECTED_FILES = {
    "d0b_author_opportunity_foundation.json",
    "d0b_author_opportunity_foundation.md",
    "d0b_author_opportunity_foundation.outputs.json",
    "episode_selected_predecision_works.parquet",
    "pre_entry_author_measurement_base.parquet",
}
FORBIDDEN_COLUMNS = {
    "entry_effect_size",
    "persistence_effect_size",
    "overshoot_timing_gap",
    "predictive_score",
    "realized_citations",
    "final_candidate_eligibility",
    "g4_outcome",
}
WORK_INPUT_SCHEMA = pa.schema(
    [
        pa.field("work_id_numeric", pa.uint64(), nullable=False),
        pa.field("publication_day", pa.int32(), nullable=True),
        pa.field("publication_year", pa.uint16(), nullable=True),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=False),
        pa.field("is_paratext", pa.bool_(), nullable=False),
        pa.field("is_retracted", pa.bool_(), nullable=True),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("primary_topic_id", pa.uint32(), nullable=True),
        pa.field("primary_field_id", pa.uint8(), nullable=True),
        pa.field("primary_domain_id", pa.uint8(), nullable=True),
    ]
)
EVENT_INPUT_SCHEMA = pa.schema(
    [
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
        pa.field("declared_authors_count", pa.uint32(), nullable=False),
        pa.field("primary_field_id", pa.uint8(), nullable=False),
    ]
)
WORK_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("incumbency_qualifying_work", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": WORK_SCHEMA_VERSION.encode()},
)


def _author_schema() -> pa.Schema:
    fields = [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
    ]
    for stem in (
        "pre_entry_all_field_work_count",
        "pre_entry_focal_field_work_count",
        "portfolio_window_all_field_work_count",
        "portfolio_window_focal_field_work_count",
    ):
        for suffix in ("cap_10", "cap_50", "cap_100", "all"):
            fields.append(pa.field(f"{stem}_{suffix}", pa.uint32(), nullable=False))
    fields.extend(
        [
            pa.field("pre_entry_target_topic_work_count", pa.uint32(), nullable=False),
            pa.field("pre_signal_target_topic_work_count", pa.uint32(), nullable=False),
            pa.field("first_target_topic_year", pa.uint16(), nullable=True),
            pa.field("observed_no_prior_target_topic", pa.bool_(), nullable=False),
            pa.field("incumbent_before_signal_window", pa.bool_(), nullable=False),
        ]
    )
    return pa.schema(
        fields,
        metadata={b"schema_version": AUTHOR_SCHEMA_VERSION.encode()},
    )


AUTHOR_OUTPUT_SCHEMA = _author_schema()


class D0BAuthorFoundationVerificationError(RuntimeError):
    """Fail-closed independent-verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise D0BAuthorFoundationVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing field {dotted_path}",
        )
        current = current[part]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0BAuthorFoundationVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise D0BAuthorFoundationVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / value).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _write_report(path: Path, result: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve(strict=True)
    target = path.parent.resolve(strict=True) / path.name
    require(
        target != root and not target.is_relative_to(root),
        "verification report must be outside output root",
    )
    payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, 0o600, follow_symlinks=False)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
        relative = str(item.get("relative_path"))
        require(relative not in result, "duplicate receipt artifact")
        result[relative] = item
    return result


def _validate_output_identity(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid output root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "output root mode mismatch")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set mismatch")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"output mode mismatch: {path.name}")

    receipt_path = root / "d0b_author_opportunity_foundation.outputs.json"
    receipt = _read_json(receipt_path, "D0-B output receipt")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema mismatch")
    require(receipt.get("status") == OUTPUT_STATUS, "receipt status mismatch")
    require(receipt.get("artifact_count") == 4, "receipt artifact count mismatch")
    require(receipt.get("d0b_author_foundation_built") is True, "foundation not built")
    for key in (
        "d0b_complete",
        "d0_complete",
        "discovery_outcome_opening_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe receipt state: {key}")
    require(receipt.get("receipt_written_last") is True, "receipt order not declared")

    expected_artifacts = EXPECTED_FILES - {receipt_path.name}
    observed = []
    for relative, item in _artifact_map(receipt).items():
        require(
            "/" not in relative and relative in expected_artifacts,
            "unexpected output artifact",
        )
        path = root / relative
        actual: dict[str, Any] = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if "rows" in item:
            parquet = pq.ParquetFile(path)
            actual["rows"] = parquet.metadata.num_rows
            actual["row_groups"] = parquet.metadata.num_row_groups
        require(actual == item, f"output artifact mismatch: {relative}")
        observed.append(actual)
    observed.sort(key=lambda item: str(item["relative_path"]))
    require(
        {item["relative_path"] for item in observed} == expected_artifacts,
        "output artifact set mismatch",
    )
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
        "output artifact tree mismatch",
    )

    report = _read_json(
        root / "d0b_author_opportunity_foundation.json",
        "D0-B producer report",
    )
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == OUTPUT_STATUS, "report status mismatch")
    scope = _mapping(report.get("scope"), "report scope")
    require(scope.get("d0b_author_foundation_built") is True, "report foundation absent")
    for key in (
        "d0b_complete",
        "d0_complete",
        "entry_or_post_entry_target_works_read",
        "persistence_rows_read",
        "realized_citation_rows_read",
        "predictive_scores_computed",
        "threshold_tuning_performed",
        "final_candidate_eligibility_established",
        "model_fitting_performed",
        "g4_rows_read",
        "gpu_used",
    ):
        require(scope.get(key) is False, f"unsafe report scope: {key}")
    require(scope.get("scientific_effect") == "NOT_ESTABLISHED", "effect overstated")
    authorization = _mapping(report.get("authorization"), "authorization")
    require(
        authorization.get("remaining_d0b_measurement_components") == "AUTHORIZED",
        "remaining measurement components not authorized",
    )
    for key in (
        "discovery_outcome_opening",
        "g0_measurement_freeze",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "effect authorized")
    return receipt, report, observed


def _validate_config(
    path: Path,
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    path = path.resolve(strict=True)
    config = _read_yaml(path, "D0-B config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("mode") == "OUTCOME_BLIND_PRE_ENTRY_AUTHOR_MEASUREMENT_FOUNDATION",
        "config mode mismatch",
    )
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_D0A_GATE_STATE_V4",
        "config authorization mismatch",
    )
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden boundary",
    )
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary opened")
    episodes = [
        _mapping(row, "frozen episode")
        for row in _list(config.get("frozen_episodes"), "frozen episodes")
    ]
    require(
        [
            (
                int(row["episode_index"]),
                int(row["field_id"]),
                int(row["topic_id"]),
                int(row["entry_year"]),
            )
            for row in episodes
        ]
        == [
            (1, 17, 14430, 2003),
            (2, 17, 10579, 2012),
            (3, 27, 11361, 2007),
            (4, 27, 12505, 1991),
        ],
        "episode freeze mismatch",
    )
    for row in episodes:
        entry = int(row["entry_year"])
        require(row["signal_window"] == [entry - 3, entry - 1], "signal window mismatch")
        require(row["reference_window"] == [entry - 6, entry - 4], "reference window mismatch")
        require(
            row["incumbent_entry_must_precede_year"] == entry - 6,
            "incumbent boundary mismatch",
        )

    reported = _mapping(report.get("repository_bindings"), "reported bindings")
    validated = {}
    d0a_document = None
    for label, yaml_document in (("gate_state", True), ("d0a_freeze", False)):
        binding = _mapping(
            config["repository_bindings"][label],
            f"{label} binding",
        )
        bound_path = _safe_repo_path(binding.get("relative_path"), f"{label} path")
        observed_sha = _sha256(bound_path)
        require(observed_sha == binding.get("sha256"), f"{label} SHA mismatch")
        document = (
            _read_yaml(bound_path, label)
            if yaml_document
            else _read_json(bound_path, label)
        )
        require(
            document.get("schema_version") == binding.get("required_schema_version"),
            f"{label} schema mismatch",
        )
        require(
            document.get("status") == binding.get("required_status"),
            f"{label} status mismatch",
        )
        for dotted_path, expected in _mapping(
            binding.get("required_fields", {}),
            f"{label} required fields",
        ).items():
            observed_value = _nested(document, dotted_path)
            require(
                type(observed_value) is type(expected)
                and observed_value == expected,
                f"{label} field mismatch: {dotted_path}",
            )
        validated[label] = {
            "relative_path": binding["relative_path"],
            "sha256": observed_sha,
            "status": document["status"],
        }
        require(reported.get(label) == validated[label], f"reported {label} differs")
        if label == "d0a_freeze":
            d0a_document = document
    assert d0a_document is not None
    require(
        [
            (row["field_id"], row["topic_id"], row["entry_year"])
            for row in d0a_document["selected_pilot_episodes"]
        ]
        == [
            (row["field_id"], row["topic_id"], row["entry_year"])
            for row in episodes
        ],
        "D0-A selected episode mismatch",
    )
    execution = _mapping(report.get("execution"), "execution")
    require(
        execution.get("config_relative_path") == str(path.relative_to(REPO_ROOT)),
        "reported config path mismatch",
    )
    require(execution.get("config_sha256") == _sha256(path), "reported config SHA mismatch")
    builder = _safe_repo_path(execution.get("builder_relative_path"), "reported builder")
    require(execution.get("builder_sha256") == _sha256(builder), "reported builder SHA mismatch")
    require(execution.get("workload_class") == "CPU_IO", "workload class mismatch")
    return config, episodes, validated


def _validate_partitioned_input(
    root: Path,
    binding: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[list[Path], dict[str, Any], list[tuple[int, int, int, int, int]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid normalized input root")
    require(root.name == binding.get("expected_root_basename"), "input basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding.get("receipt_sha256"), "input receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "input receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "input status mismatch")
    prefix = str(binding["artifact_prefix"])
    items = [
        item
        for relative, item in _artifact_map(receipt).items()
        if relative.startswith(prefix)
    ]
    items.sort(key=lambda item: str(item["relative_path"]))
    require(len(items) == binding.get("artifact_count") == PARTITIONS, "input partition count mismatch")
    paths = []
    pre_stats = []
    for index, item in enumerate(items):
        relative = f"{prefix}part-{index:04d}.parquet"
        require(item.get("relative_path") == relative, "input partition sequence mismatch")
        artifact = (root / relative).resolve(strict=True)
        require(artifact.is_relative_to(root), "input artifact escapes root")
        require(artifact.is_file() and not artifact.is_symlink(), "invalid input artifact")
        current = artifact.stat()
        require(current.st_size == item.get("bytes"), "input artifact byte mismatch")
        paths.append(artifact)
        pre_stats.append(
            (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
        )

    def rehash(pair: tuple[Path, Mapping[str, Any]]) -> tuple[str, int, str]:
        artifact, item = pair
        digest = _sha256(artifact)
        require(digest == item.get("sha256"), f"input SHA mismatch: {artifact.name}")
        return str(item["relative_path"]), int(item["bytes"]), digest

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(rehash, zip(paths, items, strict=True)))
    expected_bytes = binding.get("artifact_bytes")
    if expected_bytes is not None:
        require(
            sum(item[1] for item in verified) == expected_bytes,
            "input artifact byte total mismatch",
        )
    return paths, {
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_prefix": prefix,
        "artifacts_rehashed": len(verified),
        "artifact_bytes": sum(item[1] for item in verified),
        "selected_tree_sha256": hashlib.sha256(
            _canonical_bytes(verified)
        ).hexdigest(),
    }, pre_stats


def _validate_unchanged(
    paths: Sequence[Path],
    pre_stats: Sequence[tuple[int, int, int, int, int]],
) -> None:
    observed = []
    for path in paths:
        current = path.stat()
        observed.append(
            (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
        )
    require(observed == list(pre_stats), "input changed during verification")


def _replay_selected_works(
    paths: Sequence[Path],
    episodes: Sequence[Mapping[str, Any]],
    observed_path: Path,
) -> tuple[dict[int, pa.Array], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scanned = 0
    columns = [
        "work_id_numeric",
        "publication_year",
        "temporal_disposition",
        "declared_authors_count",
        "primary_topic_id",
        "primary_field_id",
    ]
    for index, path in enumerate(paths):
        require(pq.ParquetFile(path).schema_arrow == WORK_INPUT_SCHEMA, f"work schema mismatch: {index}")
        table = pq.read_table(path, columns=columns)
        scanned += len(table)
        work_ids = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
        require(bool(np.all(work_ids % PARTITIONS == index)), f"work partition mismatch: {index}")
        years = table["publication_year"].combine_chunks().fill_null(0).to_numpy(zero_copy_only=False)
        dispositions = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
        teams = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
        topics = table["primary_topic_id"].combine_chunks().fill_null(0).to_numpy(zero_copy_only=False)
        fields = table["primary_field_id"].combine_chunks().fill_null(0).to_numpy(zero_copy_only=False)
        usable = np.isin(dispositions, np.asarray([2, 3], dtype=np.uint8))
        for episode in episodes:
            entry = int(episode["entry_year"])
            keep = (
                usable
                & (fields == int(episode["field_id"]))
                & (topics == int(episode["topic_id"]))
                & (years > 0)
                & (years < entry)
            )
            for work_id, year, team in zip(
                work_ids[keep].tolist(),
                years[keep].tolist(),
                teams[keep].tolist(),
                strict=True,
            ):
                rows.append(
                    {
                        "episode_index": int(episode["episode_index"]),
                        "field_id": int(episode["field_id"]),
                        "topic_id": int(episode["topic_id"]),
                        "entry_year": entry,
                        "work_numeric_id": int(work_id),
                        "publication_year": int(year),
                        "declared_authors_count": int(team),
                        "incumbency_qualifying_work": int(year) < entry - 6,
                    }
                )
    rows.sort(key=lambda row: (row["episode_index"], row["work_numeric_id"]))
    require(
        len({(row["episode_index"], row["work_numeric_id"]) for row in rows})
        == len(rows),
        "duplicate replayed selected work",
    )
    expected = pa.Table.from_pylist(rows, schema=WORK_OUTPUT_SCHEMA)
    observed = pq.read_table(observed_path)
    require(observed.schema == WORK_OUTPUT_SCHEMA, "selected-work output schema mismatch")
    require(expected.equals(observed, check_metadata=True), "selected-work replay differs")
    by_episode: dict[str, dict[str, int]] = {}
    selected_sets = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        entry = int(episode["entry_year"])
        selected = [row for row in rows if row["episode_index"] == episode_index]
        by_episode[str(episode_index)] = {
            "selected_work_rows": len(selected),
            "incumbency_qualifying_work_rows": sum(
                bool(row["incumbency_qualifying_work"]) for row in selected
            ),
            "reference_window_work_rows": sum(
                entry - 6 <= row["publication_year"] <= entry - 4
                for row in selected
            ),
            "signal_window_work_rows": sum(
                entry - 3 <= row["publication_year"] <= entry - 1
                for row in selected
            ),
        }
        selected_sets[episode_index] = pa.array(
            [row["work_numeric_id"] for row in selected],
            type=pa.uint64(),
        )
    return selected_sets, {
        "source_rows_replayed": scanned,
        "selected_work_rows": len(rows),
        "by_episode": by_episode,
        "exact_output_match": True,
    }


def _bool_and(*values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    result = values[0]
    for value in values[1:]:
        result = pc.and_(result, value)
    return result


def _indicator(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return pc.cast(mask, pa.uint8())


def _independent_author_rows(
    table: pa.Table,
    episode: Mapping[str, Any],
    selected_work_ids: pa.Array,
) -> pa.Table:
    """Aggregate with Arrow group_by, independently of the NumPy producer."""
    episode_index = int(episode["episode_index"])
    field_id = int(episode["field_id"])
    entry = int(episode["entry_year"])
    author = table["author_numeric_id"].combine_chunks()
    works = table["work_numeric_id"].combine_chunks()
    years = table["publication_year"].combine_chunks()
    disposition = table["temporal_disposition"].combine_chunks()
    teams = table["declared_authors_count"].combine_chunks()
    fields = table["primary_field_id"].combine_chunks()
    usable = _bool_and(
        pc.is_in(
            disposition,
            value_set=pa.array([2, 3], type=pa.uint8()),
        ),
        pc.greater(years, pa.scalar(0, type=pa.uint16())),
    )
    pre_entry = _bool_and(
        usable,
        pc.less_equal(years, pa.scalar(entry - 1, type=pa.uint16())),
    )
    portfolio = _bool_and(
        pre_entry,
        pc.greater_equal(years, pa.scalar(entry - 3, type=pa.uint16())),
    )
    focal = pc.equal(fields, pa.scalar(field_id, type=pa.uint8()))
    target_match = pc.is_in(works, value_set=selected_work_ids)
    invalid_target = _bool_and(
        target_match,
        pc.invert(
            _bool_and(
                usable,
                pc.less_equal(years, pa.scalar(entry - 1, type=pa.uint16())),
                focal,
            )
        ),
    )
    require(
        pc.any(invalid_target).as_py() is not True,
        f"selected target work disagrees with event surface: {episode_index}",
    )
    target = _bool_and(target_match, usable, pre_entry, focal)
    pre_signal_target = _bool_and(
        target,
        pc.less(years, pa.scalar(entry - 6, type=pa.uint16())),
    )

    indicators: dict[str, pa.Array | pa.ChunkedArray] = {}
    for stem, base in (
        ("pre_entry_all_field_work_count", pre_entry),
        ("pre_entry_focal_field_work_count", _bool_and(pre_entry, focal)),
        ("portfolio_window_all_field_work_count", portfolio),
        ("portfolio_window_focal_field_work_count", _bool_and(portfolio, focal)),
    ):
        for suffix, cap in (("cap_10", 10), ("cap_50", 50), ("cap_100", 100)):
            indicators[f"{stem}_{suffix}"] = _indicator(
                _bool_and(
                    base,
                    pc.less_equal(teams, pa.scalar(cap, type=pa.uint32())),
                )
            )
        indicators[f"{stem}_all"] = _indicator(base)
    indicators["pre_entry_target_topic_work_count"] = _indicator(target)
    indicators["pre_signal_target_topic_work_count"] = _indicator(
        pre_signal_target
    )
    first_target = pc.if_else(
        target,
        years,
        pa.scalar(None, type=pa.uint16()),
    )
    aggregate_input = pa.table(
        {
            "author_numeric_id": author,
            **indicators,
            "first_target_topic_year": first_target,
        }
    )
    aggregate_specs = [(name, "sum") for name in indicators]
    aggregate_specs.append(("first_target_topic_year", "min"))
    grouped = aggregate_input.group_by(
        "author_numeric_id",
        use_threads=True,
    ).aggregate(aggregate_specs)
    candidate_name = "portfolio_window_focal_field_work_count_all_sum"
    grouped = grouped.filter(
        pc.greater(grouped[candidate_name], pa.scalar(0, type=pa.uint64()))
    ).sort_by("author_numeric_id")

    data: dict[str, pa.Array | pa.ChunkedArray] = {
        "episode_index": pa.array(
            np.full(len(grouped), episode_index, dtype=np.uint8),
            type=pa.uint8(),
        ),
        "field_id": pa.array(
            np.full(len(grouped), field_id, dtype=np.uint8),
            type=pa.uint8(),
        ),
        "topic_id": pa.array(
            np.full(len(grouped), int(episode["topic_id"]), dtype=np.uint32),
            type=pa.uint32(),
        ),
        "entry_year": pa.array(
            np.full(len(grouped), entry, dtype=np.uint16),
            type=pa.uint16(),
        ),
        "author_numeric_id": grouped["author_numeric_id"],
    }
    for name in AUTHOR_OUTPUT_SCHEMA.names[5:23]:
        values = grouped[f"{name}_sum"]
        maximum = pc.max(values).as_py()
        require(
            maximum is None or maximum <= np.iinfo(np.uint32).max,
            f"replayed count overflow: {name}",
        )
        data[name] = pc.cast(values, pa.uint32())
    data["first_target_topic_year"] = pc.cast(
        grouped["first_target_topic_year_min"],
        pa.uint16(),
    )
    data["observed_no_prior_target_topic"] = pc.equal(
        data["pre_entry_target_topic_work_count"],
        pa.scalar(0, type=pa.uint32()),
    )
    data["incumbent_before_signal_window"] = pc.greater(
        data["pre_signal_target_topic_work_count"],
        pa.scalar(0, type=pa.uint32()),
    )
    return pa.Table.from_arrays(
        [data[name] for name in AUTHOR_OUTPUT_SCHEMA.names],
        schema=AUTHOR_OUTPUT_SCHEMA,
    )


def _replay_author_base(
    paths: Sequence[Path],
    episodes: Sequence[Mapping[str, Any]],
    selected_work_ids: Mapping[int, pa.Array],
    observed_path: Path,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(observed_path)
    require(parquet.schema_arrow == AUTHOR_OUTPUT_SCHEMA, "author output schema mismatch")
    require(FORBIDDEN_COLUMNS.isdisjoint(parquet.schema_arrow.names), "forbidden author column")
    row_group = 0
    source_rows = 0
    maximum_batch = 0
    totals = {
        str(episode["episode_index"]): {
            "author_rows": 0,
            "observed_no_prior_target_topic_rows": 0,
            "prior_target_topic_rows": 0,
            "incumbent_before_signal_window_rows": 0,
            "portfolio_focal_work_events_all": 0,
        }
        for episode in episodes
    }
    for partition_index, path in enumerate(paths):
        require(pq.ParquetFile(path).schema_arrow == EVENT_INPUT_SCHEMA, f"event schema mismatch: {partition_index}")
        table = pq.read_table(path)
        source_rows += len(table)
        authors = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        require(
            bool(np.all(authors % PARTITIONS == partition_index)),
            f"author partition mismatch: {partition_index}",
        )
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            expected = _independent_author_rows(
                table,
                episode,
                selected_work_ids[episode_index],
            )
            maximum_batch = max(maximum_batch, len(expected))
            if len(expected):
                require(row_group < parquet.metadata.num_row_groups, "author row group missing")
                observed = parquet.read_row_group(row_group)
                require(
                    expected.equals(observed, check_metadata=True),
                    f"author replay differs: partition {partition_index}, episode {episode_index}",
                )
                row_group += 1
            rows = expected.to_pydict()
            target = totals[str(episode_index)]
            target["author_rows"] += len(expected)
            target["observed_no_prior_target_topic_rows"] += sum(
                rows["observed_no_prior_target_topic"]
            )
            target["prior_target_topic_rows"] += len(expected) - sum(
                rows["observed_no_prior_target_topic"]
            )
            target["incumbent_before_signal_window_rows"] += sum(
                rows["incumbent_before_signal_window"]
            )
            target["portfolio_focal_work_events_all"] += sum(
                rows["portfolio_window_focal_field_work_count_all"]
            )
    require(row_group == parquet.metadata.num_row_groups, "extra author row groups")
    require(
        sum(item["author_rows"] for item in totals.values())
        == parquet.metadata.num_rows,
        "author output row closure failed",
    )
    return {
        "source_rows_replayed": source_rows,
        "author_rows": parquet.metadata.num_rows,
        "row_groups_replayed": row_group,
        "maximum_partition_episode_batch_rows": maximum_batch,
        "by_episode": totals,
        "aggregation_engine": "PYARROW_GROUP_BY_INDEPENDENT_OF_NUMPY_PRODUCER",
        "exact_output_match": True,
    }


def verify(
    config_path: Path,
    *,
    cutoff_root: Path,
    author_history_root: Path,
    output_root: Path,
    report_path: Path,
    workers: int,
) -> dict[str, Any]:
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    receipt, producer_report, artifacts = _validate_output_identity(output_root)
    config, episodes, repository_bindings = _validate_config(
        config_path,
        producer_report,
    )
    inputs = _mapping(config.get("normalized_inputs"), "normalized inputs")
    cutoff_paths, cutoff_binding, cutoff_stats = _validate_partitioned_input(
        cutoff_root,
        _mapping(inputs["cutoff_work_metadata"], "cutoff input"),
        workers=workers,
    )
    event_paths, event_binding, event_stats = _validate_partitioned_input(
        author_history_root,
        _mapping(inputs["author_history_events"], "event input"),
        workers=workers,
    )
    reported_inputs = _mapping(
        producer_report.get("normalized_input_bindings"),
        "reported input bindings",
    )
    require(
        reported_inputs.get("cutoff_work_metadata") == cutoff_binding,
        "reported cutoff binding differs",
    )
    require(
        reported_inputs.get("author_history_events") == event_binding,
        "reported event binding differs",
    )
    output_root = output_root.resolve(strict=True)
    selected_sets, work_replay = _replay_selected_works(
        cutoff_paths,
        episodes,
        output_root / "episode_selected_predecision_works.parquet",
    )
    require(
        work_replay["source_rows_replayed"]
        == inputs["cutoff_work_metadata"]["rows"],
        "cutoff replay row closure failed",
    )
    author_replay = _replay_author_base(
        event_paths,
        episodes,
        selected_sets,
        output_root / "pre_entry_author_measurement_base.parquet",
    )
    require(
        author_replay["source_rows_replayed"]
        == inputs["author_history_events"]["rows"],
        "event replay row closure failed",
    )
    reported_work = _mapping(
        producer_report.get("selected_work_accounting"),
        "reported work accounting",
    )
    for key in ("selected_work_rows", "by_episode"):
        require(reported_work.get(key) == work_replay[key], f"reported work {key} differs")
    reported_author = _mapping(
        producer_report.get("author_measurement_accounting"),
        "reported author accounting",
    )
    for key in ("maximum_partition_episode_batch_rows", "by_episode"):
        require(reported_author.get(key) == author_replay[key], f"reported author {key} differs")
    require(reported_author.get("team_cap_selected") is False, "team cap was selected")
    require(
        reported_author.get("nonranked_team_cap_profiles") == [10, 50, 100, "ALL"],
        "team cap profiles differ",
    )
    _validate_unchanged(cutoff_paths, cutoff_stats)
    _validate_unchanged(event_paths, event_stats)

    result = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "output_root": str(output_root),
        "output_receipt": {
            "basename": "d0b_author_opportunity_foundation.outputs.json",
            "sha256": _sha256(
                output_root / "d0b_author_opportunity_foundation.outputs.json"
            ),
            "artifact_tree_sha256": receipt["artifact_tree_sha256"],
            "artifacts_rehashed": len(artifacts),
        },
        "repository_bindings": repository_bindings,
        "normalized_input_bindings": {
            "cutoff_work_metadata": cutoff_binding,
            "author_history_events": event_binding,
        },
        "replay": {
            "selected_works": work_replay,
            "author_measurement_base": author_replay,
            "source_rows_replayed_total": (
                work_replay["source_rows_replayed"]
                + author_replay["source_rows_replayed"]
            ),
            "producer_module_imported": False,
            "forbidden_columns_absent": True,
            "entry_or_post_entry_target_works_read": False,
            "persistence_rows_read": False,
            "realized_citation_rows_read": False,
        },
        "scope": {
            "d0b_author_foundation_independently_verified": True,
            "d0b_complete": False,
            "d0_complete": False,
            "discovery_outcome_opening_authorized": False,
            "g0_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "gpu_used": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": {
            "remaining_d0b_measurement_components": "AUTHORIZED",
            "discovery_outcome_opening": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
        },
        "verifier": {
            "relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "claim_boundary": config["claim_boundary"],
    }
    _write_report(report_path, result, root=output_root)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        args.config,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output_root=args.output_root,
        report_path=args.report,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_rows_replayed_total": result["replay"][
                    "source_rows_replayed_total"
                ],
                "author_rows": result["replay"]["author_measurement_base"][
                    "author_rows"
                ],
                "report": str(args.report.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
