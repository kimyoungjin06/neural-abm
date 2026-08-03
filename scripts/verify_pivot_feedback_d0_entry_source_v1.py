#!/usr/bin/env python3
"""Independently replay and verify the production three-count pivot source."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_entry_source_build_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_d0_entry_source_build_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_entry_source_build_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_entry_source_build_outputs.v1"
FEATURE_SCHEMA_VERSION = "pivot_feedback_d0_entry_source_features.v1"
VERIFY_SCHEMA = "pivot_feedback_d0_entry_source_verification.v1"
PRODUCER_STATUS = (
    "PASS_EXACT_FOUR_THREE_COUNT_PRODUCTION_FEATURE_SOURCE_NO_STATE_LABELS"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_FOUR_THREE_COUNT_SOURCE_REPLAY_NO_STATE_LABELS"
)
PARTITIONS = 4096
ROWS = 3_256_271
KEY_COLUMNS = (
    "certified_episode_index",
    "field_id",
    "topic_id",
    "entry_year",
    "author_numeric_id",
)
FEATURE_COLUMNS = (
    *KEY_COLUMNS,
    "pre_entry_target_topic_work_count",
    "trailing_3y_target_topic_work_count",
    "focal_confirmed_eligible_target_topic_work_count",
)
EXPECTED_FILES = {
    "PIVOT_ENTRY_SOURCE_BUILD.md",
    "pivot_entry_source_build.json",
    "pivot_entry_source_build.outputs.json",
    "pivot_entry_source_features.parquet",
}
WORK_SCHEMA = pa.schema(
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
EVENT_SCHEMA = pa.schema(
    [
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
        pa.field("declared_authors_count", pa.uint32(), nullable=False),
        pa.field("primary_field_id", pa.uint8(), nullable=False),
    ]
)
OUTPUT_SCHEMA = pa.schema(
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
    metadata={b"schema_version": FEATURE_SCHEMA_VERSION.encode()},
)
WORK_COLUMNS = [
    "work_id_numeric",
    "publication_year",
    "temporal_disposition",
    "work_type_code",
    "is_paratext",
    "is_retracted",
    "declared_authors_count",
    "primary_topic_id",
    "primary_field_id",
]
EVENT_COLUMNS = list(EVENT_SCHEMA.names)
ELIGIBLE_TYPES = np.array([1, 2, 3, 4, 7, 16, 18, 21], dtype=np.uint8)


class EntrySourceVerificationError(RuntimeError):
    """Fail-closed independent entry-source verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EntrySourceVerificationError(message)


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
        raise EntrySourceVerificationError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EntrySourceVerificationError(f"invalid {label}: {error}") from error


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
            require(relative not in result, "duplicate receipt artifact")
            result[relative] = item
    return result


def _validate_repository(config: Mapping[str, Any]) -> dict[str, Any]:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V43",
        "v43 authorization",
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


def _partition_paths(
    *,
    operator_base: Path,
    binding_value: Any,
    label: str,
    schema: pa.Schema,
    workers: int,
) -> tuple[list[Path], dict[str, Any]]:
    binding = _mapping(binding_value, label)
    root = (operator_base / str(binding.get("root_basename"))).resolve(strict=True)
    require(
        root.parent == operator_base
        and root.is_dir()
        and not root.is_symlink(),
        f"{label} root",
    )
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, f"{label} root mode")
    receipt_path = root / str(binding.get("receipt_basename"))
    require(_sha256(receipt_path) == binding.get("receipt_sha256"), f"{label} receipt SHA")
    receipt = _read_json(receipt_path, f"{label} receipt")
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        f"{label} receipt schema",
    )
    require(receipt.get("status") == binding.get("required_status"), f"{label} status")
    prefix = str(binding.get("artifact_prefix"))
    artifacts = [
        (relative, item)
        for relative, item in _artifact_items(receipt).items()
        if relative.startswith(prefix)
    ]
    artifacts.sort(key=lambda pair: pair[0])
    require(
        len(artifacts) == binding.get("artifact_count") == PARTITIONS,
        f"{label} partition count",
    )
    paths: list[Path] = []
    rows = 0
    for index, (relative, item) in enumerate(artifacts):
        require(
            relative == f"{prefix}part-{index:04d}.parquet",
            f"{label} partition sequence",
        )
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), f"{label} partition escape")
        require(path.is_file() and not path.is_symlink(), f"{label} partition")
        require(path.stat().st_size == item.get("bytes"), f"{label} bytes")
        parquet = pq.ParquetFile(path)
        require(parquet.schema_arrow == schema, f"{label} schema")
        rows += parquet.metadata.num_rows
        paths.append(path)
    require(rows == binding.get("rows"), f"{label} source rows")

    def rehash(pair: tuple[Path, tuple[str, Mapping[str, Any]]]) -> tuple[str, int, str]:
        path, (relative, item) = pair
        observed = _sha256(path)
        require(observed == item.get("sha256"), f"{label} SHA {path.name}")
        return relative, path.stat().st_size, observed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        verified = list(
            executor.map(rehash, zip(paths, artifacts, strict=True))
        )
    require(
        sum(item[1] for item in verified) == binding.get("artifact_bytes"),
        f"{label} byte closure",
    )
    return paths, {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifacts_rehashed": len(verified),
        "artifact_bytes": sum(item[1] for item in verified),
        "rows": rows,
        "selected_tree_sha256": hashlib.sha256(
            _canonical_bytes(verified)
        ).hexdigest(),
    }


def _validate_small_inputs(
    *,
    operator_base: Path,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    selection_spec = _mapping(
        inputs.get("certified_episode_selection"),
        "selection",
    )
    selection_root = operator_base / str(selection_spec.get("root_basename"))
    selection_receipt_path = selection_root / str(
        selection_spec.get("receipt_basename")
    )
    require(
        _sha256(selection_receipt_path) == selection_spec.get("receipt_sha256"),
        "selection receipt SHA",
    )
    selection_receipt = _read_json(selection_receipt_path, "selection receipt")
    require(
        selection_receipt.get("status") == selection_spec.get("required_status"),
        "selection status",
    )
    selection_path = selection_root / str(selection_spec.get("artifact_basename"))
    require(
        _sha256(selection_path) == selection_spec.get("artifact_sha256"),
        "selection SHA",
    )
    selection = pq.read_table(
        selection_path,
        columns=[
            "certified_episode_index",
            "field_id",
            "topic_id",
            "entry_year",
            "content_state",
        ],
    )
    rows = [
        [
            int(row["certified_episode_index"]),
            int(row["field_id"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        ]
        for row in selection.to_pylist()
    ]
    require(rows == selection_spec.get("exact_rows"), "selection exact rows")
    require(
        all(
            value.as_py() == selection_spec.get("required_content_state")
            for value in selection["content_state"]
        ),
        "selection content state",
    )

    certificate_spec = _mapping(
        inputs.get("content_selection_certificate"),
        "certificate",
    )
    certificate_root = operator_base / str(
        certificate_spec.get("root_basename")
    )
    certificate_receipt_path = certificate_root / str(
        certificate_spec.get("receipt_basename")
    )
    require(
        _sha256(certificate_receipt_path)
        == certificate_spec.get("receipt_sha256"),
        "certificate receipt SHA",
    )
    certificate_receipt = _read_json(
        certificate_receipt_path,
        "certificate receipt",
    )
    require(
        certificate_receipt.get("status") == certificate_spec.get("required_status"),
        "certificate receipt status",
    )
    certificate_path = certificate_root / str(
        certificate_spec.get("artifact_basename")
    )
    require(
        _sha256(certificate_path) == certificate_spec.get("artifact_sha256"),
        "certificate SHA",
    )
    certificate = _read_json(certificate_path, "certificate")
    require(
        certificate.get("status") == certificate_spec.get("artifact_status"),
        "certificate status",
    )
    require(certificate.get("global_selection_certificate_issued") is True, "global certificate")
    require(certificate.get("provider_labels_are_ground_truth") is False, "provider boundary")

    pre_spec = _mapping(
        inputs.get("predecision_author_measurement"),
        "predecision",
    )
    pre_root = operator_base / str(pre_spec.get("root_basename"))
    pre_receipt_path = pre_root / str(pre_spec.get("receipt_basename"))
    require(
        _sha256(pre_receipt_path) == pre_spec.get("receipt_sha256"),
        "predecision receipt SHA",
    )
    pre_receipt = _read_json(pre_receipt_path, "predecision receipt")
    require(
        pre_receipt.get("status") == pre_spec.get("required_status"),
        "predecision status",
    )
    pre_path = pre_root / str(pre_spec.get("artifact_basename"))
    require(_sha256(pre_path) == pre_spec.get("artifact_sha256"), "predecision SHA")
    parquet = pq.ParquetFile(pre_path)
    require(parquet.metadata.num_rows == pre_spec.get("rows") == ROWS, "predecision rows")
    require(
        (parquet.schema_arrow.metadata or {}).get(b"schema_version", b"").decode()
        == pre_spec.get("artifact_schema_version"),
        "predecision schema",
    )
    return {
        "episodes": [
            {
                "certified_episode_index": row[0],
                "field_id": row[1],
                "topic_id": row[2],
                "entry_year": row[3],
            }
            for row in rows
        ],
        "predecision_path": pre_path,
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
    receipt_path = root / "pivot_entry_source_build.outputs.json"
    report_path = root / "pivot_entry_source_build.json"
    feature_path = root / "pivot_entry_source_features.parquet"
    receipt = _read_json(receipt_path, "output receipt")
    report = _read_json(report_path, "producer report")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(receipt.get("status") == PRODUCER_STATUS, "receipt status")
    require(receipt.get("config_sha256") == _sha256(config_path), "receipt config")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema")
    require(report.get("status") == PRODUCER_STATUS, "report status")
    require(report.get("config_sha256") == _sha256(config_path), "report config")
    artifacts = _list(receipt.get("artifacts"), "output artifacts")
    observed: list[dict[str, Any]] = []
    for raw in artifacts:
        item = _mapping(raw, "output artifact")
        path = root / str(item.get("relative_path"))
        actual: dict[str, Any] = {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".parquet":
            actual["rows"] = pq.ParquetFile(path).metadata.num_rows
        require(actual == item, f"output artifact {path.name}")
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
    parquet = pq.ParquetFile(feature_path)
    require(parquet.metadata.num_rows == ROWS, "feature rows")
    require(
        parquet.schema_arrow.equals(OUTPUT_SCHEMA, check_metadata=True),
        "feature schema",
    )
    for key in (
        "production_classifier_state_materialized",
        "pivot_label_materialized",
        "post_entry_rows_read",
        "outcomes_read",
        "independent_verification_complete",
        "production_pivot_classification_authorized",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag {key}")
    require(receipt.get("api_requests_created") == 0, "receipt API requests")
    require(receipt.get("receipt_written_last") is True, "receipt order")
    return {
        "receipt": receipt,
        "report": report,
        "feature_path": feature_path,
        "receipt_path": receipt_path,
        "artifacts": observed,
    }


def _nullable_numpy(column: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    valid = pc.is_valid(column).to_numpy(zero_copy_only=False)
    fill_value: bool | int = False if pa.types.is_boolean(column.type) else 0
    values = (
        column.combine_chunks()
        .fill_null(pa.scalar(fill_value, type=column.type))
        .to_numpy(zero_copy_only=False)
    )
    return values, valid


def _scan_work_partition(
    pair: tuple[int, Path],
    episodes: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path, columns=WORK_COLUMNS)
    work = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(work % PARTITIONS == index)), f"work partition {index}")
    year, year_valid = _nullable_numpy(table["publication_year"])
    topic, topic_valid = _nullable_numpy(table["primary_topic_id"])
    field, field_valid = _nullable_numpy(table["primary_field_id"])
    disposition = (
        table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    work_type = (
        table["work_type_code"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    paratext = table["is_paratext"].combine_chunks().to_numpy(zero_copy_only=False)
    retracted, retracted_valid = _nullable_numpy(table["is_retracted"])
    declared = (
        table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    accounting: list[dict[str, int]] = []
    selected: list[np.ndarray] = []
    for episode in episodes:
        episode_index = int(episode["certified_episode_index"])
        entry_year = int(episode["entry_year"])
        target_topic = int(episode["topic_id"])
        target_field = int(episode["field_id"])
        window = year_valid & (year >= entry_year - 3) & (year <= entry_year)
        missing = window & (~topic_valid | ~field_valid)
        wrong = (
            window
            & topic_valid
            & field_valid
            & ((topic != target_topic) | (field != target_field))
        )
        target = (
            window
            & topic_valid
            & field_valid
            & (topic == target_topic)
            & (field == target_field)
        )
        invalid_temporal = target & (disposition != 2)
        temporal_ok = target & (disposition == 2)
        excluded_type = temporal_ok & (
            ~np.isin(work_type, ELIGIBLE_TYPES) | paratext
        )
        type_ok = temporal_ok & np.isin(work_type, ELIGIBLE_TYPES) & ~paratext
        retracted_rows = type_ok & retracted_valid & retracted
        eligible = type_ok & (~retracted_valid | ~retracted)
        retained_null = eligible & ~retracted_valid
        counts = {
            "certified_episode_index": episode_index,
            "field_id": target_field,
            "topic_id": target_topic,
            "entry_year": entry_year,
            "candidate_window_rows": int(np.count_nonzero(window)),
            "missing_primary_topic_or_field_rows": int(np.count_nonzero(missing)),
            "wrong_primary_topic_or_field_rows": int(np.count_nonzero(wrong)),
            "invalid_temporal_disposition_rows": int(
                np.count_nonzero(invalid_temporal)
            ),
            "excluded_work_type_or_paratext_rows": int(
                np.count_nonzero(excluded_type)
            ),
            "retracted_rows": int(np.count_nonzero(retracted_rows)),
            "eligible_prelink_rows": int(np.count_nonzero(eligible)),
            "retained_null_retraction_rows": int(np.count_nonzero(retained_null)),
        }
        require(
            counts["candidate_window_rows"]
            == sum(
                counts[name]
                for name in (
                    "missing_primary_topic_or_field_rows",
                    "wrong_primary_topic_or_field_rows",
                    "invalid_temporal_disposition_rows",
                    "excluded_work_type_or_paratext_rows",
                    "retracted_rows",
                    "eligible_prelink_rows",
                )
            ),
            f"work accounting partition {index}",
        )
        accounting.append(counts)
        positions = np.flatnonzero(eligible)
        if len(positions):
            values = np.empty(
                len(positions),
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
            values["work_id"] = work[positions]
            values["episode"] = episode_index
            values["field"] = target_field
            values["entry_year"] = entry_year
            values["publication_year"] = year[positions]
            values["disposition"] = disposition[positions]
            values["declared"] = declared[positions]
            values["is_focal"] = year[positions] == entry_year
            selected.append(values)
    return {
        "source_rows": table.num_rows,
        "accounting": accounting,
        "eligible": selected,
    }


def _replay_work(
    *,
    paths: Sequence[Path],
    episodes: Sequence[Mapping[str, int]],
    workers: int,
) -> tuple[np.ndarray, list[dict[str, int]], dict[str, int]]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        parts = list(
            executor.map(
                lambda pair: _scan_work_partition(pair, episodes),
                enumerate(paths),
            )
        )
    require(
        sum(int(part["source_rows"]) for part in parts) == 492_293_206,
        "work replay source rows",
    )
    per_episode: list[dict[str, int]] = []
    for episode in episodes:
        identity = {
            "certified_episode_index": int(episode["certified_episode_index"]),
            "field_id": int(episode["field_id"]),
            "topic_id": int(episode["topic_id"]),
            "entry_year": int(episode["entry_year"]),
        }
        totals = {
            name: 0
            for name in (
                "candidate_window_rows",
                "missing_primary_topic_or_field_rows",
                "wrong_primary_topic_or_field_rows",
                "invalid_temporal_disposition_rows",
                "excluded_work_type_or_paratext_rows",
                "retracted_rows",
                "eligible_prelink_rows",
                "retained_null_retraction_rows",
            )
        }
        for part in parts:
            row = part["accounting"][identity["certified_episode_index"] - 1]
            for name in totals:
                totals[name] += int(row[name])
        per_episode.append({**identity, **totals})
    selected = [
        values
        for part in parts
        for values in part["eligible"]
    ]
    require(selected, "no eligible work")
    eligible = np.concatenate(selected)
    order = np.argsort(eligible["work_id"], kind="stable")
    eligible = eligible[order]
    require(bool(np.all(eligible["work_id"][1:] > eligible["work_id"][:-1])), "eligible work uniqueness")
    totals = {
        name: sum(row[name] for row in per_episode)
        for name in per_episode[0]
        if name not in KEY_COLUMNS[:4]
    }
    totals["outside_window_rows_scanned"] = 0
    totals["forbidden_post_entry_offset_scan_count"] = 0
    return eligible, per_episode, totals


def _scan_event_partition(
    pair: tuple[int, Path],
    eligible: np.ndarray,
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path, columns=EVENT_COLUMNS)
    author = (
        table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    require(bool(np.all(author % PARTITIONS == index)), f"event partition {index}")
    work = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    positions = np.searchsorted(eligible["work_id"], work)
    matched = positions < len(eligible)
    valid_positions = positions[matched]
    matched_indices = np.flatnonzero(matched)
    exact_work = np.zeros(len(work), dtype=np.bool_)
    exact_work[matched_indices] = (
        eligible["work_id"][valid_positions] == work[matched_indices]
    )
    selected_rows = np.flatnonzero(exact_work)
    selected_positions = positions[selected_rows]
    year = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)
    disposition = (
        table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    declared = (
        table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    field = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)
    year_match = year[selected_rows] == eligible["publication_year"][selected_positions]
    disposition_match = (
        disposition[selected_rows] == eligible["disposition"][selected_positions]
    )
    declared_match = (
        declared[selected_rows] == eligible["declared"][selected_positions]
    )
    field_match = field[selected_rows] == eligible["field"][selected_positions]
    exact = year_match & disposition_match & declared_match & field_match
    exact_rows = selected_rows[exact]
    exact_positions = selected_positions[exact]
    values = np.empty(
        len(exact_rows),
        dtype=[
            ("episode", "u1"),
            ("author", "<u8"),
            ("work", "<u8"),
            ("is_focal", "?"),
        ],
    )
    values["episode"] = eligible["episode"][exact_positions]
    values["author"] = author[exact_rows]
    values["work"] = work[exact_rows]
    values["is_focal"] = eligible["is_focal"][exact_positions]
    return {
        "source_rows": table.num_rows,
        "joined_rows": len(selected_rows),
        "publication_year_mismatch_rows": int(np.count_nonzero(~year_match)),
        "temporal_disposition_mismatch_rows": int(
            np.count_nonzero(~disposition_match)
        ),
        "declared_authors_count_mismatch_rows": int(
            np.count_nonzero(~declared_match)
        ),
        "primary_field_id_mismatch_rows": int(np.count_nonzero(~field_match)),
        "valid": values,
    }


def _replay_events(
    *,
    paths: Sequence[Path],
    eligible: np.ndarray,
    workers: int,
) -> tuple[np.ndarray, dict[str, int]]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        parts = list(
            executor.map(
                lambda pair: _scan_event_partition(pair, eligible),
                enumerate(paths),
            )
        )
    source_rows = sum(int(part["source_rows"]) for part in parts)
    require(source_rows == 779_630_258, "event replay source rows")
    joined_rows = sum(int(part["joined_rows"]) for part in parts)
    mismatch_names = (
        "publication_year_mismatch_rows",
        "temporal_disposition_mismatch_rows",
        "declared_authors_count_mismatch_rows",
        "primary_field_id_mismatch_rows",
    )
    mismatches = {
        name: sum(int(part[name]) for part in parts)
        for name in mismatch_names
    }
    require(all(value == 0 for value in mismatches.values()), "event metadata mismatch")
    valid_arrays = [part["valid"] for part in parts if len(part["valid"])]
    valid = np.concatenate(valid_arrays)
    require(len(valid) == joined_rows, "valid event row closure")
    order = np.lexsort((valid["work"], valid["author"], valid["episode"]))
    valid = valid[order]
    keep = np.ones(len(valid), dtype=np.bool_)
    keep[1:] = (
        (valid["episode"][1:] != valid["episode"][:-1])
        | (valid["author"][1:] != valid["author"][:-1])
        | (valid["work"][1:] != valid["work"][:-1])
    )
    distinct = valid[keep]
    require(
        bool(
            np.all(
                distinct["is_focal"]
                == valid["is_focal"][np.flatnonzero(keep)]
            )
        ),
        "duplicate focal consistency",
    )
    linked_key = np.empty(
        len(distinct),
        dtype=[("episode", "u1"), ("work", "<u8")],
    )
    linked_key["episode"] = distinct["episode"]
    linked_key["work"] = distinct["work"]
    linked_work_rows = len(np.unique(linked_key))
    accounting = {
        "source_event_rows": source_rows,
        "eligible_work_joined_event_rows": joined_rows,
        **mismatches,
        "valid_event_rows": len(valid),
        "distinct_author_work_pairs": len(distinct),
        "duplicate_author_work_event_excess_rows": len(valid) - len(distinct),
        "linked_work_rows": linked_work_rows,
        "zero_valid_author_token_work_rows": len(eligible) - linked_work_rows,
    }
    return distinct, accounting


def _strictly_sorted_unique(table: pa.Table) -> bool:
    arrays = [
        table[column].combine_chunks().to_numpy(zero_copy_only=False)
        for column in KEY_COLUMNS
    ]
    greater = np.zeros(table.num_rows - 1, dtype=np.bool_)
    equal_prefix = np.ones(table.num_rows - 1, dtype=np.bool_)
    for values in arrays:
        greater |= equal_prefix & (values[1:] > values[:-1])
        equal_prefix &= values[1:] == values[:-1]
    return bool(np.all(greater))


def _array_equals(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> bool:
    left_array = left.combine_chunks() if isinstance(left, pa.ChunkedArray) else left
    right_array = (
        right.combine_chunks() if isinstance(right, pa.ChunkedArray) else right
    )
    return left_array.equals(right_array)


def _feature_replay(
    *,
    predecision_path: Path,
    feature_path: Path,
    pairs: np.ndarray,
    event_accounting: dict[str, int],
) -> dict[str, Any]:
    predecision = pq.read_table(
        predecision_path,
        columns=[*KEY_COLUMNS, "pre_entry_target_topic_work_count"],
    )
    output = pq.read_table(feature_path)
    require(predecision.num_rows == output.num_rows == ROWS, "feature replay rows")
    require(_strictly_sorted_unique(output), "output key sort")
    indices = pc.sort_indices(
        predecision.select(KEY_COLUMNS),
        sort_keys=[(column, "ascending") for column in KEY_COLUMNS],
    )
    for name in KEY_COLUMNS:
        expected = pc.take(predecision[name], indices)
        require(
            _array_equals(expected, output[name]),
            f"output key replay {name}",
        )
    expected_pre = pc.take(
        predecision["pre_entry_target_topic_work_count"],
        indices,
    )
    require(
        _array_equals(
            expected_pre,
            output["pre_entry_target_topic_work_count"],
        ),
        "pre-entry feature replay",
    )
    episode = output["certified_episode_index"].combine_chunks().to_numpy()
    author = output["author_numeric_id"].combine_chunks().to_numpy()
    expected_trailing = np.zeros(ROWS, dtype=np.uint32)
    expected_focal = np.zeros(ROWS, dtype=np.uint32)
    registered_pairs = 0
    unregistered_pairs = 0
    for episode_index in (1, 2, 3, 4):
        output_positions = np.flatnonzero(episode == episode_index)
        episode_authors = author[output_positions]
        episode_pairs = pairs[pairs["episode"] == episode_index]
        positions = np.searchsorted(episode_authors, episode_pairs["author"])
        matched = positions < len(episode_authors)
        matched_indices = np.flatnonzero(matched)
        exact = np.zeros(len(episode_pairs), dtype=np.bool_)
        exact[matched_indices] = (
            episode_authors[positions[matched_indices]]
            == episode_pairs["author"][matched_indices]
        )
        registered_pairs += int(np.count_nonzero(exact))
        unregistered_pairs += int(np.count_nonzero(~exact))
        global_positions = output_positions[positions[exact]]
        focal = episode_pairs["is_focal"][exact]
        np.add.at(expected_trailing, global_positions[~focal], 1)
        np.add.at(expected_focal, global_positions[focal], 1)
    event_accounting["registered_distinct_author_work_pairs"] = registered_pairs
    event_accounting["unregistered_distinct_author_work_pairs"] = unregistered_pairs
    require(
        registered_pairs + unregistered_pairs
        == event_accounting["distinct_author_work_pairs"],
        "registered pair closure",
    )
    observed_trailing = (
        output["trailing_3y_target_topic_work_count"]
        .combine_chunks()
        .to_numpy()
    )
    observed_focal = (
        output["focal_confirmed_eligible_target_topic_work_count"]
        .combine_chunks()
        .to_numpy()
    )
    trailing_mismatches = int(
        np.count_nonzero(observed_trailing != expected_trailing)
    )
    focal_mismatches = int(np.count_nonzero(observed_focal != expected_focal))
    require(trailing_mismatches == focal_mismatches == 0, "feature count replay")
    pre_values = (
        output["pre_entry_target_topic_work_count"].combine_chunks().to_numpy()
    )
    invariants = {
        "trailing_count_exceeds_pre_entry_count": int(
            np.count_nonzero(observed_trailing > pre_values)
        ),
        "zero_pre_entry_positive_trailing": int(
            np.count_nonzero((pre_values == 0) & (observed_trailing > 0))
        ),
        "negative_or_null_feature_rows": 0,
    }
    require(all(value == 0 for value in invariants.values()), "feature invariant")
    by_episode: list[dict[str, int]] = []
    field = output["field_id"].combine_chunks().to_numpy()
    topic = output["topic_id"].combine_chunks().to_numpy()
    entry_year = output["entry_year"].combine_chunks().to_numpy()
    for episode_index in (1, 2, 3, 4):
        mask = episode == episode_index
        positions = np.flatnonzero(mask)
        require(len(np.unique(field[mask])) == 1, "episode field")
        require(len(np.unique(topic[mask])) == 1, "episode topic")
        require(len(np.unique(entry_year[mask])) == 1, "episode year")
        by_episode.append(
            {
                "certified_episode_index": episode_index,
                "field_id": int(field[positions[0]]),
                "topic_id": int(topic[positions[0]]),
                "entry_year": int(entry_year[positions[0]]),
                "author_rows": len(positions),
                "trailing_pair_rows": int(expected_trailing[mask].sum()),
                "focal_pair_rows": int(expected_focal[mask].sum()),
                "trailing_positive_author_rows": int(
                    np.count_nonzero(expected_trailing[mask] > 0)
                ),
                "focal_positive_author_rows": int(
                    np.count_nonzero(expected_focal[mask] > 0)
                ),
            }
        )
    return {
        "rows": ROWS,
        "columns": len(FEATURE_COLUMNS),
        "output_keys_globally_strict_sorted_unique": True,
        "pre_entry_value_mismatches": 0,
        "trailing_value_mismatches": trailing_mismatches,
        "focal_value_mismatches": focal_mismatches,
        "by_episode": by_episode,
        "invariant_violations": invariants,
        "production_classifier_state_columns": 0,
        "pivot_label_columns": 0,
        "forbidden_post_entry_offset_scan_count": 0,
    }


def verify(
    *,
    config_path: Path,
    operator_base: Path,
    producer_root: Path,
    verification_path: Path,
    workers: int,
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
    inputs = _mapping(config.get("operator_inputs"), "operator inputs")
    work_paths, work_binding = _partition_paths(
        operator_base=operator_base,
        binding_value=inputs.get("cutoff_work_metadata"),
        label="cutoff work metadata",
        schema=WORK_SCHEMA,
        workers=workers,
    )
    event_paths, event_binding = _partition_paths(
        operator_base=operator_base,
        binding_value=inputs.get("author_work_events"),
        label="author work events",
        schema=EVENT_SCHEMA,
        workers=workers,
    )
    small = _validate_small_inputs(operator_base=operator_base, inputs=inputs)
    output = _validate_output(root=producer_root, config_path=config_path)
    builder_path = _safe_repo_path(
        output["report"]["execution"]["builder_relative_path"],
        "reported builder",
    )
    require(
        output["report"]["execution"]["builder_sha256"] == _sha256(builder_path),
        "reported builder SHA",
    )
    verifier_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(verifier_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(name=node.module or "")]
        )
    }
    require(
        "build_pivot_feedback_d0_entry_source_v1" not in imported_modules,
        "verifier imports producer",
    )
    eligible, work_by_episode, work_totals = _replay_work(
        paths=work_paths,
        episodes=small["episodes"],
        workers=workers,
    )
    pairs, event_accounting = _replay_events(
        paths=event_paths,
        eligible=eligible,
        workers=workers,
    )
    feature_replay = _feature_replay(
        predecision_path=small["predecision_path"],
        feature_path=output["feature_path"],
        pairs=pairs,
        event_accounting=event_accounting,
    )
    expected_work = {
        "source_work_rows": work_binding["rows"],
        "by_episode": work_by_episode,
        "totals": work_totals,
    }
    require(output["report"]["work_accounting"] == expected_work, "work report replay")
    require(
        output["report"]["author_event_accounting"] == event_accounting,
        "event report replay",
    )
    report_output = output["report"]["output_accounting"]
    for key in (
        "rows",
        "columns",
        "by_episode",
        "invariant_violations",
        "production_classifier_state_columns",
        "pivot_label_columns",
        "forbidden_post_entry_offset_scan_count",
    ):
        require(report_output.get(key) == feature_replay.get(key), f"output report {key}")
    verification = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "verifier": {
            "strategy": "PYARROW_NUMPY_FULL_PARTITION_REPLAY_AND_EXACT_FEATURE_COMPARISON",
            "imports_producer_module": False,
            "uses_producer_join_engine": False,
            "producer_join_engine": "DUCKDB",
            "verification_engine": "PYARROW_NUMPY",
            "workers": max(1, workers),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        },
        "repository_bindings": repository,
        "source_bindings": {
            "cutoff_work_metadata": work_binding,
            "author_work_events": event_binding,
        },
        "output_root_basename": producer_root.name,
        "output_receipt_sha256": _sha256(output["receipt_path"]),
        "output_artifacts": output["artifacts"],
        "scope": {
            "exact_certified_episode_rows": 4,
            "registered_author_episode_rows": ROWS,
            "work_source_rows_replayed": work_binding["rows"],
            "author_event_source_rows_replayed": event_binding["rows"],
            "content_confirmation_unit": (
                "CERTIFIED_EPISODE_TOPIC_NOT_INDIVIDUAL_WORK"
            ),
            "production_classifier_state_materialized": False,
            "pivot_label_materialized": False,
            "post_entry_offsets_read": [],
            "outcomes_read": False,
            "api_requests_created": 0,
        },
        "work_replay": {
            "by_episode": work_by_episode,
            "totals": work_totals,
            "eligible_work_rows": len(eligible),
        },
        "author_event_replay": event_accounting,
        "feature_replay": feature_replay,
        "authorization": {
            "production_source_independently_verified": True,
            "successor_gate_may_authorize_production_classifier_states": True,
            "production_pivot_classification_authorized_by_verifier_alone": False,
            "discovery_outcome_opening_authorized": False,
            "new_api_request_creation_authorized": False,
            "g0_measurement_freeze_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        },
        "claim_boundary": (
            "This independent PyArrow and NumPy replay establishes exact "
            "preservation of all registered author-episode keys, the verified "
            "pre-entry count, and independently reconstructed trailing and "
            "focal counts from every bound work and author-event partition. "
            "It materializes or reads no production classifier state, pivot "
            "label, post-entry outcome, API result, model, or G4 result and "
            "does not establish an association, causal mechanism, or "
            "scientific effect."
        ),
    }
    _write_json_new(verification_path, verification)
    os.chmod(verification_path, 0o600)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--verification-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        operator_base=args.operator_base,
        producer_root=args.producer_root,
        verification_path=args.verification_path,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
