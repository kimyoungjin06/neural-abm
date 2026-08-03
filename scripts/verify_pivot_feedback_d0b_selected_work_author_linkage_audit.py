#!/usr/bin/env python3
"""Independently replay the D0-B selected-work author-linkage audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "pivot_feedback_d0b_selected_work_author_linkage_audit_config.v1"
OUTPUT_REPORT_SCHEMA = "pivot_feedback_d0b_selected_work_author_linkage_audit_report.v1"
OUTPUT_RECEIPT_SCHEMA = "pivot_feedback_d0b_selected_work_author_linkage_audit_outputs.v1"
VERIFY_SCHEMA = "pivot_feedback_d0b_selected_work_author_linkage_audit_verification.v1"
OUTPUT_STATUS = (
    "PASS_D0B_OUTCOME_BLIND_SELECTED_WORK_AUTHOR_LINKAGE_AUDIT_"
    "NO_EPISODE_DECISION"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0B_SELECTED_WORK_AUTHOR_LINKAGE_AUDIT_"
    "SEVERE_HETEROGENEOUS_LINKAGE_GAP_REQUIRES_MEASUREMENT_REVIEW"
)
PARTITIONS = 4096
MAX_CONTROL_BYTES = 32 * 1024 * 1024
EXPECTED_FILES = {
    "selected_work_author_linkage.parquet",
    "selected_work_valid_author_incidence.parquet",
    "selected_work_author_linkage_profile.parquet",
    "selected_work_author_linkage_audit.json",
    "selected_work_author_linkage_audit.md",
    "selected_work_author_linkage_audit.outputs.json",
}
FORBIDDEN_COLUMNS = {
    "entry_effect_size",
    "persistence_effect_size",
    "predictive_score",
    "final_candidate_eligibility",
    "episode_excluded",
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
SELECTED_WORK_SCHEMA = pa.schema(
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
    metadata={b"schema_version": b"pivot_feedback_d0b_selected_predecision_work.v1"},
)
WORK_LINKAGE_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=False),
        pa.field("work_type_name", pa.string(), nullable=False),
        pa.field("is_paratext", pa.bool_(), nullable=False),
        pa.field("is_retracted", pa.bool_(), nullable=True),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("valid_distinct_author_token_count", pa.uint32(), nullable=False),
        pa.field("declared_minus_valid_distinct_token_count", pa.uint32(), nullable=False),
        pa.field("has_valid_author_token", pa.bool_(), nullable=False),
        pa.field("pre_signal_incumbency_qualifying_work", pa.bool_(), nullable=False),
        pa.field("incumbent_valid_author_token_count", pa.uint32(), nullable=False),
        pa.field("has_pre_signal_incumbent_author", pa.bool_(), nullable=False),
        pa.field("all_linked_authors_pre_signal_incumbents", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_d0b_selected_work_author_linkage.v1"},
)
INCIDENCE_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("pre_signal_incumbent_author", pa.bool_(), nullable=False),
    ],
    metadata={
        b"schema_version": b"pivot_feedback_d0b_selected_work_valid_author_incidence.v1"
    },
)
PROFILE_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("aggregation_level", pa.string(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=True),
        pa.field("work_type_name", pa.string(), nullable=False),
        pa.field("team_cap", pa.string(), nullable=False),
        pa.field("calendar_window", pa.string(), nullable=False),
        pa.field("selected_work_rows", pa.uint64(), nullable=False),
        pa.field("linked_work_rows", pa.uint64(), nullable=False),
        pa.field("zero_valid_author_token_work_rows", pa.uint64(), nullable=False),
        pa.field("declared_author_slots", pa.uint64(), nullable=False),
        pa.field("valid_distinct_author_token_rows", pa.uint64(), nullable=False),
        pa.field("declared_minus_valid_distinct_token_rows", pa.uint64(), nullable=False),
        pa.field("any_pre_signal_incumbent_work_rows", pa.uint64(), nullable=False),
        pa.field("all_linked_authors_incumbent_work_rows", pa.uint64(), nullable=False),
    ],
    metadata={
        b"schema_version": b"pivot_feedback_d0b_selected_work_author_linkage_profile.v1"
    },
)


class SelectedWorkAuthorLinkageVerificationError(RuntimeError):
    """Fail-closed independent-linkage-verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectedWorkAuthorLinkageVerificationError(message)


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
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} too large")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectedWorkAuthorLinkageVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise SelectedWorkAuthorLinkageVerificationError(
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


def _validate_output(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid output root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "output root mode mismatch")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set mismatch")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"output mode mismatch: {path.name}")
    receipt_path = root / "selected_work_author_linkage_audit.outputs.json"
    receipt = _read_json(receipt_path, "linkage output receipt")
    require(receipt.get("schema_version") == OUTPUT_RECEIPT_SCHEMA, "receipt schema mismatch")
    require(receipt.get("status") == OUTPUT_STATUS, "receipt status mismatch")
    require(receipt.get("artifact_count") == 5, "receipt artifact count mismatch")
    require(receipt.get("selected_work_author_linkage_audited") is True, "audit absent")
    for key in (
        "episode_exclusion_decision_made",
        "linkage_threshold_selected",
        "d0b_complete",
        "d0_complete",
        "discovery_outcome_opening_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag: {key}")
    expected = EXPECTED_FILES - {receipt_path.name}
    observed = []
    for relative, item in _artifact_map(receipt).items():
        require("/" not in relative and relative in expected, "unexpected artifact")
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
        require(actual == item, f"artifact identity differs: {relative}")
        observed.append(actual)
    observed.sort(key=lambda row: str(row["relative_path"]))
    require(
        {row["relative_path"] for row in observed} == expected,
        "artifact set differs",
    )
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(
            _canonical_bytes(
                [
                    (row["relative_path"], row["bytes"], row["sha256"])
                    for row in observed
                ]
            )
        ).hexdigest(),
        "artifact tree differs",
    )
    report = _read_json(
        root / "selected_work_author_linkage_audit.json",
        "producer report",
    )
    require(report.get("schema_version") == OUTPUT_REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == OUTPUT_STATUS, "report status mismatch")
    scope = _mapping(report.get("scope"), "report scope")
    require(scope.get("selected_work_author_linkage_audited") is True, "scope audit absent")
    for key in (
        "episode_exclusion_decision_made",
        "linkage_threshold_selected",
        "incumbent_attention_definition_validated",
        "focal_or_network_leaveout_built",
        "d0b_complete",
        "d0_complete",
        "entry_or_post_entry_target_works_read",
        "persistence_rows_read",
        "realized_citation_rows_read",
        "predictive_scores_computed",
        "model_fitting_performed",
        "g4_rows_read",
        "gpu_used",
    ):
        require(scope.get(key) is False, f"unsafe report scope: {key}")
    require(scope.get("scientific_effect") == "NOT_ESTABLISHED", "effect overstated")
    return receipt, report, observed


def _validate_repository_bindings(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    reported = _mapping(report.get("repository_bindings"), "reported bindings")
    for label, yaml_document in (
        ("gate_state", True),
        ("author_foundation_summary", False),
    ):
        binding = _mapping(config["repository_bindings"][label], f"{label} binding")
        path = _safe_repo_path(binding.get("relative_path"), f"{label} path")
        digest = _sha256(path)
        require(digest == binding.get("sha256"), f"{label} SHA mismatch")
        document = (
            _read_yaml(path, label)
            if yaml_document
            else _read_json(path, label)
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
            f"{label} fields",
        ).items():
            observed = _nested(document, dotted_path)
            require(
                type(observed) is type(expected) and observed == expected,
                f"{label} field mismatch: {dotted_path}",
            )
        result[label] = {
            "relative_path": binding["relative_path"],
            "sha256": digest,
            "status": document["status"],
        }
        require(reported.get(label) == result[label], f"reported {label} differs")
    return result


def _validate_foundation(
    root: Path,
    binding: Mapping[str, Any],
    reported: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid foundation root")
    require(root.name == binding.get("expected_root_basename"), "foundation basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    digest = _sha256(receipt_path)
    require(digest == binding.get("receipt_sha256"), "foundation receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "foundation receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "foundation status mismatch")
    require(
        receipt.get("artifact_tree_sha256") == binding.get("artifact_tree_sha256"),
        "foundation tree mismatch",
    )
    artifact = _mapping(binding.get("selected_work_artifact"), "selected artifact")
    path = (root / str(artifact["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(root), "selected artifact escapes root")
    require(_sha256(path) == artifact.get("sha256"), "selected artifact SHA mismatch")
    table = pq.read_table(path)
    require(table.schema == SELECTED_WORK_SCHEMA, "selected artifact schema mismatch")
    require(len(table) == artifact.get("rows"), "selected artifact rows differ")
    result = {
        "root_basename": root.name,
        "receipt_sha256": digest,
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "selected_work_artifact_sha256": artifact["sha256"],
        "selected_work_rows": artifact["rows"],
    }
    require(dict(reported) == result, "reported foundation binding differs")
    return table, result


def _validate_partitioned_input(
    root: Path,
    binding: Mapping[str, Any],
    reported: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[list[Path], dict[str, Any], list[tuple[int, int, int, int, int]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid input root")
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
    items.sort(key=lambda row: str(row["relative_path"]))
    require(len(items) == binding.get("artifact_count") == PARTITIONS, "partition count mismatch")
    paths = []
    stats = []
    for index, item in enumerate(items):
        relative = f"{prefix}part-{index:04d}.parquet"
        require(item.get("relative_path") == relative, "partition sequence mismatch")
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), "partition escapes root")
        require(path.is_file() and not path.is_symlink(), "invalid partition")
        current = path.stat()
        require(current.st_size == item.get("bytes"), "partition bytes differ")
        paths.append(path)
        stats.append(
            (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
        )

    def rehash(pair: tuple[Path, Mapping[str, Any]]) -> tuple[str, int, str]:
        path, item = pair
        digest = _sha256(path)
        require(digest == item.get("sha256"), f"partition SHA differs: {path.name}")
        return str(item["relative_path"]), int(item["bytes"]), digest

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(rehash, zip(paths, items, strict=True)))
    expected_bytes = binding.get("artifact_bytes")
    if expected_bytes is not None:
        require(sum(row[1] for row in verified) == expected_bytes, "input byte total differs")
    result = {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_prefix": prefix,
        "artifacts_rehashed": len(verified),
        "artifact_bytes": sum(row[1] for row in verified),
        "selected_tree_sha256": hashlib.sha256(_canonical_bytes(verified)).hexdigest(),
    }
    require(dict(reported) == result, "reported input binding differs")
    return paths, result, stats


def _validate_unchanged(
    paths: Sequence[Path],
    stats: Sequence[tuple[int, int, int, int, int]],
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
    require(observed == list(stats), "input changed during verification")


def _selected_records(
    table: pa.Table,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[int, int]]:
    rows = table.to_pylist()
    require(len(rows) == 83_565, "selected work count differs")
    work_to_index = {}
    for index, row in enumerate(rows):
        work_id = int(row["work_numeric_id"])
        require(work_id not in work_to_index, "duplicate selected work ID")
        require(0 < int(row["publication_year"]) < int(row["entry_year"]), "decision boundary crossed")
        work_to_index[work_id] = index
    sorted_ids = np.asarray(sorted(work_to_index), dtype=np.uint64)
    return rows, sorted_ids, work_to_index


def _replay_metadata(
    paths: Sequence[Path],
    selected_rows: Sequence[Mapping[str, Any]],
    sorted_ids: np.ndarray,
    work_to_index: Mapping[int, int],
    linkage_rows: Sequence[Mapping[str, Any]],
    type_names: Mapping[int, str],
) -> int:
    observed_by_work = {
        int(row["work_numeric_id"]): row
        for row in linkage_rows
    }
    require(len(observed_by_work) == len(selected_rows), "linkage work identity differs")
    seen = set()
    scanned = 0
    for partition_index, path in enumerate(paths):
        require(pq.ParquetFile(path).schema_arrow == WORK_INPUT_SCHEMA, f"work schema mismatch: {partition_index}")
        table = pq.read_table(path)
        scanned += len(table)
        work = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
        require(bool(np.all(work % PARTITIONS == partition_index)), f"work partition mismatch: {partition_index}")
        positions = np.searchsorted(sorted_ids, work)
        in_range = positions < len(sorted_ids)
        matched = np.zeros(len(work), dtype=bool)
        matched[in_range] = sorted_ids[positions[in_range]] == work[in_range]
        for source in table.filter(pa.array(matched)).to_pylist():
            work_id = int(source["work_id_numeric"])
            require(work_id not in seen, "duplicate selected metadata")
            seen.add(work_id)
            selected = selected_rows[work_to_index[work_id]]
            observed = observed_by_work[work_id]
            require(source["publication_year"] == selected["publication_year"], "metadata year differs")
            require(source["temporal_disposition"] in (2, 3), "metadata disposition differs")
            require(source["declared_authors_count"] == selected["declared_authors_count"], "metadata team differs")
            require(source["primary_topic_id"] == selected["topic_id"], "metadata topic differs")
            require(source["primary_field_id"] == selected["field_id"], "metadata field differs")
            require(observed["episode_index"] == selected["episode_index"], "linkage episode differs")
            require(observed["field_id"] == selected["field_id"], "linkage field differs")
            require(observed["topic_id"] == selected["topic_id"], "linkage topic differs")
            require(observed["entry_year"] == selected["entry_year"], "linkage entry differs")
            require(observed["publication_year"] == selected["publication_year"], "linkage year differs")
            require(observed["work_type_code"] == source["work_type_code"], "linkage work type differs")
            require(
                observed["work_type_name"] == type_names[int(source["work_type_code"])],
                "linkage work type name differs",
            )
            require(observed["is_paratext"] == source["is_paratext"], "paratext flag differs")
            require(observed["is_retracted"] == source["is_retracted"], "retracted flag differs")
            require(
                observed["pre_signal_incumbency_qualifying_work"]
                == selected["incumbency_qualifying_work"],
                "incumbency work flag differs",
            )
    require(len(seen) == len(selected_rows), "selected metadata coverage incomplete")
    return scanned


def _replay_incidence(
    paths: Sequence[Path],
    selected_rows: Sequence[Mapping[str, Any]],
    sorted_ids: np.ndarray,
    work_to_index: Mapping[int, int],
) -> tuple[list[dict[str, Any]], dict[int, set[int]], int]:
    raw = []
    scanned = 0
    for partition_index, path in enumerate(paths):
        require(pq.ParquetFile(path).schema_arrow == EVENT_INPUT_SCHEMA, f"event schema mismatch: {partition_index}")
        table = pq.read_table(path)
        scanned += len(table)
        author = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        require(bool(np.all(author % PARTITIONS == partition_index)), f"author partition mismatch: {partition_index}")
        work = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        positions = np.searchsorted(sorted_ids, work)
        in_range = positions < len(sorted_ids)
        matched = np.zeros(len(work), dtype=bool)
        matched[in_range] = sorted_ids[positions[in_range]] == work[in_range]
        for source in table.filter(pa.array(matched)).to_pylist():
            selected = selected_rows[work_to_index[int(source["work_numeric_id"])]]
            require(source["publication_year"] == selected["publication_year"], "event year differs")
            require(source["temporal_disposition"] in (2, 3), "event disposition differs")
            require(source["declared_authors_count"] == selected["declared_authors_count"], "event team differs")
            require(source["primary_field_id"] == selected["field_id"], "event field differs")
            raw.append(
                {
                    "episode_index": selected["episode_index"],
                    "field_id": selected["field_id"],
                    "topic_id": selected["topic_id"],
                    "entry_year": selected["entry_year"],
                    "work_numeric_id": selected["work_numeric_id"],
                    "publication_year": selected["publication_year"],
                    "declared_authors_count": selected["declared_authors_count"],
                    "author_numeric_id": source["author_numeric_id"],
                }
            )
    raw.sort(
        key=lambda row: (
            row["episode_index"],
            row["work_numeric_id"],
            row["author_numeric_id"],
        )
    )
    require(
        len({(row["work_numeric_id"], row["author_numeric_id"]) for row in raw})
        == len(raw),
        "duplicate replayed author-work incidence",
    )
    incumbents: dict[int, set[int]] = {}
    for episode_index in range(1, 5):
        episode_rows = [row for row in raw if row["episode_index"] == episode_index]
        entry = int(episode_rows[0]["entry_year"])
        incumbents[episode_index] = {
            int(row["author_numeric_id"])
            for row in episode_rows
            if int(row["publication_year"]) < entry - 6
        }
    for row in raw:
        row["pre_signal_incumbent_author"] = (
            int(row["author_numeric_id"]) in incumbents[int(row["episode_index"])]
        )
    return raw, incumbents, scanned


def _verify_linkage_counts(
    selected_rows: Sequence[Mapping[str, Any]],
    linkage_rows: Sequence[Mapping[str, Any]],
    incidence_rows: Sequence[Mapping[str, Any]],
    incumbents: Mapping[int, set[int]],
) -> dict[str, Any]:
    by_work: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in incidence_rows:
        by_work[int(row["work_numeric_id"])].append(dict(row))
    linkage_by_work = {
        int(row["work_numeric_id"]): row
        for row in linkage_rows
    }
    totals = {
        "selected_work_rows": len(selected_rows),
        "valid_author_incidence_rows": len(incidence_rows),
        "linked_work_rows": 0,
        "zero_valid_author_token_work_rows": 0,
        "declared_author_slots": 0,
        "valid_distinct_author_token_rows": 0,
        "declared_minus_valid_distinct_token_rows": 0,
        "incumbent_authors_by_episode": {
            str(index): len(incumbents[index])
            for index in range(1, 5)
        },
    }
    for selected in selected_rows:
        work_id = int(selected["work_numeric_id"])
        observed = linkage_by_work[work_id]
        events = by_work.get(work_id, [])
        valid = len(events)
        declared = int(selected["declared_authors_count"])
        require(valid <= declared, "valid author tokens exceed declared count")
        incumbent_count = sum(row["pre_signal_incumbent_author"] for row in events)
        expected = {
            "valid_distinct_author_token_count": valid,
            "declared_minus_valid_distinct_token_count": declared - valid,
            "has_valid_author_token": valid > 0,
            "incumbent_valid_author_token_count": incumbent_count,
            "has_pre_signal_incumbent_author": incumbent_count > 0,
            "all_linked_authors_pre_signal_incumbents": (
                valid > 0 and incumbent_count == valid
            ),
        }
        for key, value in expected.items():
            require(observed[key] == value, f"linkage count differs: {work_id} {key}")
        totals["linked_work_rows"] += int(valid > 0)
        totals["zero_valid_author_token_work_rows"] += int(valid == 0)
        totals["declared_author_slots"] += declared
        totals["valid_distinct_author_token_rows"] += valid
        totals["declared_minus_valid_distinct_token_rows"] += declared - valid
    return totals


def _profile_rows(
    linkage_rows: Sequence[Mapping[str, Any]],
    *,
    episodes: Sequence[Mapping[str, Any]],
    type_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        entry = int(episode["entry_year"])
        for level, type_code, type_name in [
            ("ALL_TYPES", None, "ALL_TYPES"),
            *[
                ("WORK_TYPE", code, name)
                for code, name in sorted(type_names.items())
            ],
        ]:
            for cap_label, cap in (("10", 10), ("50", 50), ("100", 100), ("ALL", None)):
                for window, low, high in (
                    ("ALL_PREDECISION", None, entry - 1),
                    ("REFERENCE", entry - 6, entry - 4),
                    ("SIGNAL", entry - 3, entry - 1),
                ):
                    selected = []
                    for row in linkage_rows:
                        if int(row["episode_index"]) != episode_index:
                            continue
                        if type_code is not None and int(row["work_type_code"]) != type_code:
                            continue
                        if cap is not None and int(row["declared_authors_count"]) > cap:
                            continue
                        year = int(row["publication_year"])
                        if year > high or (low is not None and year < low):
                            continue
                        selected.append(row)
                    rows.append(
                        {
                            "episode_index": episode_index,
                            "field_id": int(episode["field_id"]),
                            "topic_id": int(episode["topic_id"]),
                            "entry_year": entry,
                            "aggregation_level": level,
                            "work_type_code": type_code,
                            "work_type_name": type_name,
                            "team_cap": cap_label,
                            "calendar_window": window,
                            "selected_work_rows": len(selected),
                            "linked_work_rows": sum(
                                row["has_valid_author_token"] for row in selected
                            ),
                            "zero_valid_author_token_work_rows": sum(
                                not row["has_valid_author_token"] for row in selected
                            ),
                            "declared_author_slots": sum(
                                int(row["declared_authors_count"]) for row in selected
                            ),
                            "valid_distinct_author_token_rows": sum(
                                int(row["valid_distinct_author_token_count"])
                                for row in selected
                            ),
                            "declared_minus_valid_distinct_token_rows": sum(
                                int(row["declared_minus_valid_distinct_token_count"])
                                for row in selected
                            ),
                            "any_pre_signal_incumbent_work_rows": sum(
                                row["has_pre_signal_incumbent_author"]
                                for row in selected
                            ),
                            "all_linked_authors_incumbent_work_rows": sum(
                                row["all_linked_authors_pre_signal_incumbents"]
                                for row in selected
                            ),
                        }
                    )
    return rows


def verify(
    config_path: Path,
    *,
    foundation_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    output_root: Path,
    report_path: Path,
    workers: int,
) -> dict[str, Any]:
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    receipt, producer_report, artifacts = _validate_output(output_root)
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, "linkage config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("mode")
        == "OUTCOME_BLIND_SELECTED_WORK_TO_VALID_AUTHOR_TOKEN_LINKAGE_AUDIT",
        "config mode mismatch",
    )
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden boundary")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary opened")
    repository_bindings = _validate_repository_bindings(config, producer_report)
    selected_table, foundation_binding = _validate_foundation(
        foundation_root,
        _mapping(config["author_foundation_output"], "foundation binding"),
        _mapping(producer_report["foundation_output_binding"], "reported foundation"),
    )
    inputs = _mapping(config.get("normalized_inputs"), "normalized inputs")
    reported_inputs = _mapping(
        producer_report.get("normalized_input_bindings"),
        "reported input bindings",
    )
    cutoff_paths, cutoff_binding, cutoff_stats = _validate_partitioned_input(
        cutoff_root,
        _mapping(inputs["cutoff_work_metadata"], "cutoff input"),
        _mapping(reported_inputs["cutoff_work_metadata"], "reported cutoff"),
        workers=workers,
    )
    event_paths, event_binding, event_stats = _validate_partitioned_input(
        author_history_root,
        _mapping(inputs["author_history_events"], "event input"),
        _mapping(reported_inputs["author_history_events"], "reported events"),
        workers=workers,
    )
    output_root = output_root.resolve(strict=True)
    work_table = pq.read_table(output_root / "selected_work_author_linkage.parquet")
    incidence_table = pq.read_table(
        output_root / "selected_work_valid_author_incidence.parquet"
    )
    profile_table = pq.read_table(
        output_root / "selected_work_author_linkage_profile.parquet"
    )
    require(work_table.schema == WORK_LINKAGE_SCHEMA, "work linkage schema mismatch")
    require(incidence_table.schema == INCIDENCE_SCHEMA, "incidence schema mismatch")
    require(profile_table.schema == PROFILE_SCHEMA, "profile schema mismatch")
    for schema in (work_table.schema, incidence_table.schema, profile_table.schema):
        require(FORBIDDEN_COLUMNS.isdisjoint(schema.names), "forbidden output column")
    selected_rows, sorted_ids, work_to_index = _selected_records(selected_table)
    linkage_rows = work_table.to_pylist()
    type_names = {
        int(code): str(name)
        for code, name in _mapping(config.get("work_type_dictionary"), "work types").items()
    }
    cutoff_rows = _replay_metadata(
        cutoff_paths,
        selected_rows,
        sorted_ids,
        work_to_index,
        linkage_rows,
        type_names,
    )
    incidence_rows, incumbents, event_rows = _replay_incidence(
        event_paths,
        selected_rows,
        sorted_ids,
        work_to_index,
    )
    expected_incidence = pa.Table.from_pylist(incidence_rows, schema=INCIDENCE_SCHEMA)
    require(
        expected_incidence.equals(incidence_table, check_metadata=True),
        "author incidence replay differs",
    )
    accounting = _verify_linkage_counts(
        selected_rows,
        linkage_rows,
        incidence_rows,
        incumbents,
    )
    require(
        accounting == producer_report.get("linkage_accounting"),
        "producer linkage accounting differs",
    )
    episodes = [
        _mapping(row, "episode")
        for row in _list(config.get("frozen_episodes"), "episodes")
    ]
    expected_profile = pa.Table.from_pylist(
        _profile_rows(linkage_rows, episodes=episodes, type_names=type_names),
        schema=PROFILE_SCHEMA,
    )
    require(
        expected_profile.equals(profile_table, check_metadata=True),
        "profile replay differs",
    )
    require(cutoff_rows == inputs["cutoff_work_metadata"]["rows"], "cutoff row closure failed")
    require(event_rows == inputs["author_history_events"]["rows"], "event row closure failed")
    _validate_unchanged(cutoff_paths, cutoff_stats)
    _validate_unchanged(event_paths, event_stats)
    episode_three_signal = producer_report["profile_summary"]["3"]["SIGNAL"]
    require(
        episode_three_signal["selected_work_rows"] == 23_073
        and episode_three_signal["linked_work_rows"] == 1_790
        and episode_three_signal["zero_valid_author_token_work_rows"] == 21_283,
        "episode 3 signal linkage margin differs",
    )
    episode_three_other = next(
        row
        for row in profile_table.to_pylist()
        if row["episode_index"] == 3
        and row["aggregation_level"] == "WORK_TYPE"
        and row["work_type_name"] == "other"
        and row["team_cap"] == "ALL"
        and row["calendar_window"] == "ALL_PREDECISION"
    )
    require(
        episode_three_other["selected_work_rows"] == 22_451
        and episode_three_other["linked_work_rows"] == 14,
        "episode 3 other-type linkage margin differs",
    )
    execution = _mapping(producer_report.get("execution"), "producer execution")
    require(execution.get("config_sha256") == _sha256(config_path), "producer config SHA differs")
    builder_path = _safe_repo_path(execution.get("builder_relative_path"), "producer builder")
    require(execution.get("builder_sha256") == _sha256(builder_path), "producer builder SHA differs")
    result = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "output_root": str(output_root),
        "output_receipt": {
            "basename": "selected_work_author_linkage_audit.outputs.json",
            "sha256": _sha256(
                output_root / "selected_work_author_linkage_audit.outputs.json"
            ),
            "artifact_tree_sha256": receipt["artifact_tree_sha256"],
            "artifacts_rehashed": len(artifacts),
        },
        "repository_bindings": repository_bindings,
        "foundation_output_binding": foundation_binding,
        "normalized_input_bindings": {
            "cutoff_work_metadata": cutoff_binding,
            "author_history_events": event_binding,
        },
        "replay": {
            "cutoff_work_rows_replayed": cutoff_rows,
            "author_event_rows_replayed": event_rows,
            "source_rows_replayed_total": cutoff_rows + event_rows,
            "selected_work_rows": len(selected_rows),
            "valid_author_incidence_rows": len(incidence_rows),
            "work_metadata_exact_match": True,
            "author_incidence_exact_match": True,
            "work_linkage_count_exact_match": True,
            "profile_exact_match": True,
            "producer_module_imported": False,
            "forbidden_columns_absent": True,
        },
        "critical_readout": {
            "overall_selected_work_rows": accounting["selected_work_rows"],
            "overall_linked_work_rows": accounting["linked_work_rows"],
            "overall_zero_valid_author_token_work_rows": accounting[
                "zero_valid_author_token_work_rows"
            ],
            "episode_3_signal_selected_work_rows": 23_073,
            "episode_3_signal_linked_work_rows": 1_790,
            "episode_3_signal_zero_valid_author_token_work_rows": 21_283,
            "episode_3_other_type_selected_work_rows": 22_451,
            "episode_3_other_type_linked_work_rows": 14,
            "interpretation": (
                "SEVERE_HETEROGENEOUS_WORK_TYPE_AND_AUTHOR_LINKAGE_GAP_"
                "BLOCKS_UNQUALIFIED_INCUMBENT_ATTENTION_USE"
            ),
        },
        "scope": {
            "linkage_audit_independently_verified": True,
            "measurement_review_required": True,
            "episode_exclusion_decision_made": False,
            "threshold_selected": False,
            "incumbent_attention_validated": False,
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
            "measurement_contract_review": "REQUIRED_BEFORE_LEAVEOUT_BUILD",
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
    parser.add_argument("--foundation-root", type=Path, required=True)
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
        foundation_root=args.foundation_root,
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
                "valid_author_incidence_rows": result["replay"][
                    "valid_author_incidence_rows"
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
