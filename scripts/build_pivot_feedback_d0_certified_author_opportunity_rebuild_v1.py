#!/usr/bin/env python3
"""Build the exact certified-episode author-opportunity base under the frozen ABI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
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
    "pivot_feedback_d0_certified_author_opportunity_rebuild_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_d0_certified_author_opportunity_rebuild_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_certified_author_opportunity_rebuild_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_certified_author_opportunity_rebuild_outputs.v1"
WORK_SCHEMA_VERSION = "pivot_feedback_d0_certified_selected_predecision_work.v1"
AUTHOR_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_pre_entry_author_opportunity_base.v1"
)
STATUS = "PASS_EXACT_FOUR_OUTCOME_BLIND_MEASUREMENT_ABI_AUTHOR_OPPORTUNITY_REBUILD"
PARTITIONS = 4096
MAX_CONTROL_BYTES = 32 * 1024 * 1024
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
EVENT_WORK_DTYPE = np.dtype(
    [
        ("work_numeric_id", "<u8"),
        ("publication_year", "<u2"),
        ("field_id", "u1"),
        ("temporal_disposition", "u1"),
    ],
    align=False,
)
LINKED_WORK_DTYPE = np.dtype(
    [
        ("work_numeric_id", "<u8"),
        ("publication_year", "<u2"),
        ("field_id", "u1"),
        ("topic_id", "<u4"),
        ("work_type_code", "u1"),
        ("declared_authors_count", "<u2"),
        ("is_retracted_null", "u1"),
        ("temporal_disposition", "u1"),
    ],
    align=False,
)
WORK_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=False),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("is_retracted_null", pa.bool_(), nullable=False),
        pa.field("window_code", pa.uint8(), nullable=False),
    ],
    metadata={b"schema_version": WORK_SCHEMA_VERSION.encode()},
)


def _author_schema() -> pa.Schema:
    fields = [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
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


class CertifiedAuthorOpportunityRebuildError(RuntimeError):
    """Fail-closed certified author-opportunity rebuild error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertifiedAuthorOpportunityRebuildError(message)


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
        raise CertifiedAuthorOpportunityRebuildError(
            f"invalid {label}: {error}"
        ) from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise CertifiedAuthorOpportunityRebuildError(
            f"invalid {label}: {error}"
        ) from error


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _validate_repo_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _safe_repo_path(binding.get("relative_path"), "repository binding")
    observed_sha = _sha256(path)
    require(observed_sha == binding.get("sha256"), f"binding SHA mismatch: {path.name}")
    document = _read_yaml(path, path.name)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        f"binding schema mismatch: {path.name}",
    )
    require(
        document.get("status") == binding.get("required_status"),
        f"binding status mismatch: {path.name}",
    )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        "binding required fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"binding field mismatch: {dotted_path}",
        )
    return {
        "relative_path": binding["relative_path"],
        "sha256": observed_sha,
        "status": document["status"],
    }


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    raw_artifacts = receipt.get("artifacts")
    if isinstance(raw_artifacts, Mapping):
        for label, raw in raw_artifacts.items():
            item = _mapping(raw, f"receipt artifact {label}")
            basename = item.get("basename")
            require(isinstance(basename, str) and basename, "bad artifact basename")
            require(basename not in result, "duplicate artifact basename")
            result[basename] = item
    else:
        for raw in _list(raw_artifacts, "receipt artifacts"):
            item = _mapping(raw, "receipt artifact")
            relative = item.get("relative_path")
            require(isinstance(relative, str) and relative, "bad artifact path")
            require(relative not in result, "duplicate artifact path")
            result[relative] = item
    return result


def _validate_partitioned_input(
    root: Path,
    binding: Mapping[str, Any],
    *,
    schema: pa.Schema,
    workers: int,
) -> tuple[list[Path], dict[str, Any], list[tuple[int, int, int, int, int]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid partitioned input root")
    require(root.name == binding.get("expected_root_basename"), "input basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(
        _sha256(receipt_path) == binding.get("receipt_sha256"),
        "input receipt SHA mismatch",
    )
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
    require(
        len(items) == binding.get("artifact_count") == PARTITIONS,
        "input partition count mismatch",
    )
    paths: list[Path] = []
    pre_stats: list[tuple[int, int, int, int, int]] = []
    for index, item in enumerate(items):
        relative = f"{prefix}part-{index:04d}.parquet"
        require(item.get("relative_path") == relative, "input partition sequence mismatch")
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), "input partition escapes root")
        require(path.is_file() and not path.is_symlink(), "invalid input partition")
        require(pq.ParquetFile(path).schema_arrow == schema, "input schema mismatch")
        current = path.stat()
        require(current.st_size == item.get("bytes"), "input byte mismatch")
        paths.append(path)
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
        path, item = pair
        observed = _sha256(path)
        require(observed == item.get("sha256"), f"partition SHA mismatch: {path.name}")
        return str(item["relative_path"]), int(item["bytes"]), observed

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
    require(observed == list(pre_stats), "input changed during execution")


def _load_episodes(
    root: Path,
    binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid certified selection root")
    require(root.name == binding.get("expected_root_basename"), "selection root mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding.get("receipt_sha256"), "selection receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "selection receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "selection status mismatch")
    selection_path = root / str(binding["selection_basename"])
    require(_sha256(selection_path) == binding.get("selection_sha256"), "selection SHA mismatch")
    artifact = _artifact_map(receipt).get(selection_path.name)
    require(artifact is not None, "selection absent from receipt")
    require(artifact.get("sha256") == binding.get("selection_sha256"), "selection receipt differs")
    table = pq.read_table(selection_path)
    metadata = table.schema.metadata or {}
    require(
        metadata.get(b"schema_version", b"").decode()
        == binding.get("selection_schema_version"),
        "selection schema version mismatch",
    )
    require(len(table) == binding.get("selection_rows") == 4, "selection row count mismatch")
    rows = table.to_pylist()
    observed = [
        {
            key: int(row[key])
            for key in (
                "certified_episode_index",
                "field_id",
                "topic_id",
                "entry_year",
                "field_candidate_rank",
                "subfield_id",
            )
        }
        for row in rows
    ]
    expected = [
        {key: int(value) for key, value in _mapping(raw, "exact episode").items()}
        for raw in _list(binding.get("exact_rows"), "exact episodes")
    ]
    require(observed == expected, "certified episode rows differ")
    episodes = []
    for row in rows:
        episodes.append(
            {
                "certified_episode_index": int(row["certified_episode_index"]),
                "field_id": int(row["field_id"]),
                "topic_id": int(row["topic_id"]),
                "topic_name": str(row["topic_name"]),
                "subfield_id": int(row["subfield_id"]),
                "subfield_name": str(row["subfield_name"]),
                "entry_year": int(row["entry_year"]),
                "field_candidate_rank": int(row["field_candidate_rank"]),
                "signal_window": [
                    int(row["entry_year"]) - 3,
                    int(row["entry_year"]) - 1,
                ],
                "reference_window": [
                    int(row["entry_year"]) - 6,
                    int(row["entry_year"]) - 4,
                ],
                "incumbent_entry_must_precede_year": int(row["entry_year"]) - 6,
            }
        )
    return episodes, {
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": binding["receipt_sha256"],
        "selection_basename": selection_path.name,
        "selection_sha256": binding["selection_sha256"],
        "selection_rows": len(table),
    }


def _scan_event_partition(
    pair: tuple[int, Path],
    *,
    fields: np.ndarray,
    dispositions: np.ndarray,
    latest_year: int,
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path)
    authors = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    works = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    years = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)
    observed_dispositions = (
        table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    observed_fields = (
        table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    require(bool(np.all(authors % PARTITIONS == index)), f"author partition mismatch: {index}")
    keep = (
        np.isin(observed_fields, fields)
        & np.isin(observed_dispositions, dispositions)
        & (years > 0)
        & (years <= latest_year)
    )
    selected_authors = authors[keep]
    selected_works = works[keep]
    if len(selected_works):
        order = np.lexsort((selected_works, selected_authors))
        duplicate_pairs = int(
            (
                (selected_authors[order][1:] == selected_authors[order][:-1])
                & (selected_works[order][1:] == selected_works[order][:-1])
            ).sum()
        )
    else:
        duplicate_pairs = 0
    records = np.empty(int(keep.sum()), dtype=EVENT_WORK_DTYPE)
    records["work_numeric_id"] = selected_works
    records["publication_year"] = years[keep]
    records["field_id"] = observed_fields[keep]
    records["temporal_disposition"] = observed_dispositions[keep]
    return {
        "rows": len(table),
        "selected_rows": len(records),
        "duplicate_pairs": duplicate_pairs,
        "records": records,
    }


def _event_work_index(
    paths: Sequence[Path],
    *,
    fields: Sequence[int],
    dispositions: Sequence[int],
    latest_year: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    field_array = np.asarray(fields, dtype=np.uint8)
    disposition_array = np.asarray(dispositions, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(
            executor.map(
                lambda pair: _scan_event_partition(
                    pair,
                    fields=field_array,
                    dispositions=disposition_array,
                    latest_year=latest_year,
                ),
                enumerate(paths),
            )
        )
    accounting = {
        "event_rows_scanned": sum(int(item["rows"]) for item in chunks),
        "abi_temporal_field_event_rows": sum(int(item["selected_rows"]) for item in chunks),
        "duplicate_author_work_event_rows": sum(int(item["duplicate_pairs"]) for item in chunks),
    }
    records = np.concatenate([item["records"] for item in chunks], dtype=EVENT_WORK_DTYPE)
    del chunks
    unique = np.unique(records)
    del records
    work_ids = unique["work_numeric_id"]
    if len(work_ids) > 1:
        require(
            not bool(np.any(work_ids[1:] == work_ids[:-1])),
            "author events disagree on work metadata",
        )
    accounting["distinct_temporally_valid_author_linked_work_rows"] = len(unique)
    metadata = {
        "publication_year": unique["publication_year"],
        "field_id": unique["field_id"],
        "temporal_disposition": unique["temporal_disposition"],
    }
    return work_ids, metadata, accounting


def _scan_work_partition(
    pair: tuple[int, Path],
    *,
    fields: np.ndarray,
    dispositions: np.ndarray,
    included_types: np.ndarray,
    latest_year: int,
    valid_work_ids: np.ndarray,
    event_metadata: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path)
    works = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(works % PARTITIONS == index)), f"work partition mismatch: {index}")
    year_array = table["publication_year"].combine_chunks()
    year_valid = year_array.is_valid().to_numpy(zero_copy_only=False)
    years = year_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint16, copy=False)
    observed_dispositions = (
        table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    types = table["work_type_code"].combine_chunks().to_numpy(zero_copy_only=False)
    paratext = table["is_paratext"].combine_chunks().to_numpy(zero_copy_only=False)
    retracted_array = table["is_retracted"].combine_chunks()
    retracted_valid = retracted_array.is_valid().to_numpy(zero_copy_only=False)
    retracted = retracted_array.fill_null(False).to_numpy(zero_copy_only=False)
    teams = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
    topic_array = table["primary_topic_id"].combine_chunks()
    topic_valid = topic_array.is_valid().to_numpy(zero_copy_only=False)
    topics = topic_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
    field_array = table["primary_field_id"].combine_chunks()
    field_valid = field_array.is_valid().to_numpy(zero_copy_only=False)
    observed_fields = field_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
    scope = (
        year_valid
        & topic_valid
        & field_valid
        & np.isin(observed_fields, fields)
        & np.isin(observed_dispositions, dispositions)
        & (years > 0)
        & (years <= latest_year)
        & (topics > 0)
    )
    included = scope & np.isin(types, included_types)
    retracted_true = included & ~paratext & retracted_valid & retracted
    eligible = included & ~paratext & ~retracted_true
    eligible_indices = np.flatnonzero(eligible)
    eligible_works = works[eligible_indices]
    positions = np.searchsorted(valid_work_ids, eligible_works)
    in_range = positions < len(valid_work_ids)
    matched = np.zeros(len(eligible_indices), dtype=bool)
    matched[in_range] = valid_work_ids[positions[in_range]] == eligible_works[in_range]
    if bool(np.any(matched)):
        source = eligible_indices[matched]
        event = positions[matched]
        require(
            bool(np.all(event_metadata["publication_year"][event] == years[source])),
            f"event/work year mismatch: {index}",
        )
        require(
            bool(np.all(event_metadata["field_id"][event] == observed_fields[source])),
            f"event/work field mismatch: {index}",
        )
        require(
            bool(
                np.all(
                    event_metadata["temporal_disposition"][event]
                    == observed_dispositions[source]
                )
            ),
            f"event/work disposition mismatch: {index}",
        )
    linked_indices = eligible_indices[matched]
    linked = np.empty(len(linked_indices), dtype=LINKED_WORK_DTYPE)
    linked["work_numeric_id"] = works[linked_indices]
    linked["publication_year"] = years[linked_indices]
    linked["field_id"] = observed_fields[linked_indices]
    linked["topic_id"] = topics[linked_indices]
    linked["work_type_code"] = types[linked_indices]
    linked["declared_authors_count"] = teams[linked_indices]
    linked["is_retracted_null"] = (~retracted_valid[linked_indices]).astype(np.uint8)
    linked["temporal_disposition"] = observed_dispositions[linked_indices]
    return {
        "rows": len(table),
        "scope_rows": int(scope.sum()),
        "included_type_rows": int(included.sum()),
        "paratext_excluded_rows": int((included & paratext).sum()),
        "retracted_true_excluded_rows": int(retracted_true.sum()),
        "eligible_prelink_rows": len(eligible_indices),
        "linked_rows": len(linked_indices),
        "zero_valid_author_token_rows": len(eligible_indices) - len(linked_indices),
        "is_retracted_null_linked_rows": int((~retracted_valid[linked_indices]).sum()),
        "linked": linked,
    }


def _linked_work_universe(
    paths: Sequence[Path],
    *,
    fields: Sequence[int],
    dispositions: Sequence[int],
    included_types: Sequence[int],
    latest_year: int,
    valid_work_ids: np.ndarray,
    event_metadata: Mapping[str, np.ndarray],
    workers: int,
) -> tuple[np.ndarray, dict[str, int]]:
    field_array = np.asarray(fields, dtype=np.uint8)
    disposition_array = np.asarray(dispositions, dtype=np.uint8)
    type_array = np.asarray(included_types, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(
            executor.map(
                lambda pair: _scan_work_partition(
                    pair,
                    fields=field_array,
                    dispositions=disposition_array,
                    included_types=type_array,
                    latest_year=latest_year,
                    valid_work_ids=valid_work_ids,
                    event_metadata=event_metadata,
                ),
                enumerate(paths),
            )
        )
    names = (
        "rows",
        "scope_rows",
        "included_type_rows",
        "paratext_excluded_rows",
        "retracted_true_excluded_rows",
        "eligible_prelink_rows",
        "linked_rows",
        "zero_valid_author_token_rows",
        "is_retracted_null_linked_rows",
    )
    accounting = {name: sum(int(item[name]) for item in chunks) for name in names}
    linked = np.concatenate([item["linked"] for item in chunks], dtype=LINKED_WORK_DTYPE)
    del chunks
    order = np.argsort(linked["work_numeric_id"], kind="stable")
    linked = linked[order]
    if len(linked) > 1:
        require(
            not bool(
                np.any(
                    linked["work_numeric_id"][1:] == linked["work_numeric_id"][:-1]
                )
            ),
            "linked work duplicated",
        )
    return linked, accounting


def _selected_work_table(
    linked: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[pa.Table, dict[int, np.ndarray], dict[str, Any]]:
    chunks: list[pa.Table] = []
    selected_ids: dict[int, np.ndarray] = {}
    accounting: dict[str, Any] = {"by_episode": {}}
    for episode in episodes:
        episode_index = int(episode["certified_episode_index"])
        entry = int(episode["entry_year"])
        keep = (
            (linked["field_id"] == int(episode["field_id"]))
            & (linked["topic_id"] == int(episode["topic_id"]))
            & (linked["publication_year"] <= entry - 1)
        )
        values = linked[keep]
        years = values["publication_year"]
        window = np.where(
            years <= entry - 7,
            0,
            np.where(years <= entry - 4, 1, 2),
        ).astype(np.uint8)
        selected_ids[episode_index] = values["work_numeric_id"].copy()
        chunks.append(
            pa.Table.from_arrays(
                [
                    pa.array(np.full(len(values), episode_index), type=pa.uint8()),
                    pa.array(values["field_id"], type=pa.uint8()),
                    pa.array(values["topic_id"], type=pa.uint32()),
                    pa.array(np.full(len(values), entry), type=pa.uint16()),
                    pa.array(values["work_numeric_id"], type=pa.uint64()),
                    pa.array(values["publication_year"], type=pa.uint16()),
                    pa.array(values["work_type_code"], type=pa.uint8()),
                    pa.array(values["declared_authors_count"], type=pa.uint16()),
                    pa.array(values["is_retracted_null"].astype(bool), type=pa.bool_()),
                    pa.array(window, type=pa.uint8()),
                ],
                schema=WORK_OUTPUT_SCHEMA,
            )
        )
        accounting["by_episode"][str(episode_index)] = {
            "selected_work_rows": len(values),
            "pre_signal_work_rows": int((window == 0).sum()),
            "reference_window_work_rows": int((window == 1).sum()),
            "signal_window_work_rows": int((window == 2).sum()),
            "is_retracted_null_rows": int(values["is_retracted_null"].sum()),
        }
    table = pa.concat_tables(chunks)
    table = table.replace_schema_metadata(WORK_OUTPUT_SCHEMA.metadata)
    accounting["selected_work_rows"] = len(table)
    return table, selected_ids, accounting


def _sorted_membership(
    values: np.ndarray,
    sorted_candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(sorted_candidates):
        return np.zeros(len(values), dtype=bool), np.zeros(len(values), dtype=np.int64)
    positions = np.searchsorted(sorted_candidates, values)
    in_range = positions < len(sorted_candidates)
    result = np.zeros(len(values), dtype=bool)
    result[in_range] = sorted_candidates[positions[in_range]] == values[in_range]
    return result, positions


def _counts_by_author(
    inverse: np.ndarray,
    mask: np.ndarray,
    size: int,
) -> np.ndarray:
    values = np.bincount(inverse[mask], minlength=size)
    require(bool(np.all(values <= np.iinfo(np.uint32).max)), "author count overflow")
    return values.astype(np.uint32, copy=False)


def _cap_profile_counts(
    inverse: np.ndarray,
    base_mask: np.ndarray,
    teams: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_inverse = inverse[base_mask]
    selected_teams = teams[base_mask]
    bucket = np.full(len(selected_teams), 3, dtype=np.uint8)
    bucket[selected_teams <= 100] = 2
    bucket[selected_teams <= 50] = 1
    bucket[selected_teams <= 10] = 0
    flat = selected_inverse.astype(np.int64, copy=False) * 4 + bucket
    exact = np.bincount(flat, minlength=size * 4).reshape(size, 4)
    cumulative = np.cumsum(exact, axis=1, dtype=np.uint64)
    require(
        bool(np.all(cumulative <= np.iinfo(np.uint32).max)),
        "author cap count overflow",
    )
    cumulative = cumulative.astype(np.uint32, copy=False)
    return (
        cumulative[:, 0],
        cumulative[:, 1],
        cumulative[:, 2],
        cumulative[:, 3],
    )


def _author_batch(
    table: pa.Table,
    *,
    partition_index: int,
    linked: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    selected_work_ids: Mapping[int, np.ndarray],
) -> tuple[list[pa.Table], dict[int, dict[str, int]], int]:
    authors = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    require(
        bool(np.all(authors % PARTITIONS == partition_index)),
        f"author partition mismatch: {partition_index}",
    )
    works = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    eligible, positions = _sorted_membership(works, linked["work_numeric_id"])
    if bool(np.any(eligible)):
        source = np.flatnonzero(eligible)
        matched = positions[eligible]
        years_all = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)
        fields_all = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)
        dispositions_all = (
            table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
        )
        teams_all = (
            table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
        )
        require(
            bool(np.all(years_all[source] == linked["publication_year"][matched])),
            f"eligible event/work year mismatch: {partition_index}",
        )
        require(
            bool(np.all(fields_all[source] == linked["field_id"][matched])),
            f"eligible event/work field mismatch: {partition_index}",
        )
        require(
            bool(
                np.all(
                    dispositions_all[source]
                    == linked["temporal_disposition"][matched]
                )
            ),
            f"eligible event/work disposition mismatch: {partition_index}",
        )
        require(
            bool(
                np.all(
                    teams_all[source]
                    == linked["declared_authors_count"][matched]
                )
            ),
            f"eligible event/work team mismatch: {partition_index}",
        )
    authors = authors[eligible]
    works = works[eligible]
    years = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)[eligible]
    teams = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)[eligible]
    fields = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)[eligible]
    unique_authors, inverse = np.unique(authors, return_inverse=True)
    size = len(unique_authors)
    result: list[pa.Table] = []
    accounting: dict[int, dict[str, int]] = {}
    for episode in episodes:
        episode_index = int(episode["certified_episode_index"])
        field_id = int(episode["field_id"])
        entry = int(episode["entry_year"])
        pre_entry = years <= entry - 1
        portfolio = pre_entry & (years >= entry - 3)
        focal = fields == field_id
        metrics: dict[str, np.ndarray] = {}
        for stem, mask in (
            ("pre_entry_all_field_work_count", pre_entry),
            ("pre_entry_focal_field_work_count", pre_entry & focal),
            ("portfolio_window_all_field_work_count", portfolio),
            ("portfolio_window_focal_field_work_count", portfolio & focal),
        ):
            profiles = _cap_profile_counts(inverse, mask, teams, size)
            for suffix, values in zip(
                ("cap_10", "cap_50", "cap_100", "all"),
                profiles,
                strict=True,
            ):
                metrics[f"{stem}_{suffix}"] = values
        target_match, _ = _sorted_membership(works, selected_work_ids[episode_index])
        require(
            not bool(np.any(target_match & (~pre_entry | ~focal))),
            f"selected target event violates episode boundary: {episode_index}",
        )
        target_counts = _counts_by_author(inverse, target_match, size)
        pre_signal = target_match & (years < entry - 6)
        pre_signal_counts = _counts_by_author(inverse, pre_signal, size)
        first_target = np.full(size, np.iinfo(np.uint16).max, dtype=np.uint16)
        np.minimum.at(
            first_target,
            inverse[target_match],
            years[target_match].astype(np.uint16, copy=False),
        )
        candidate = metrics["portfolio_window_focal_field_work_count_all"] > 0
        candidate_indices = np.flatnonzero(candidate)
        first_values = first_target[candidate_indices]
        arrays: dict[str, pa.Array] = {
            "certified_episode_index": pa.array(
                np.full(len(candidate_indices), episode_index),
                type=pa.uint8(),
            ),
            "field_id": pa.array(
                np.full(len(candidate_indices), field_id),
                type=pa.uint8(),
            ),
            "topic_id": pa.array(
                np.full(len(candidate_indices), int(episode["topic_id"])),
                type=pa.uint32(),
            ),
            "entry_year": pa.array(
                np.full(len(candidate_indices), entry),
                type=pa.uint16(),
            ),
            "author_numeric_id": pa.array(
                unique_authors[candidate_indices],
                type=pa.uint64(),
            ),
        }
        for name in AUTHOR_OUTPUT_SCHEMA.names[5:21]:
            arrays[name] = pa.array(metrics[name][candidate_indices], type=pa.uint32())
        arrays["pre_entry_target_topic_work_count"] = pa.array(
            target_counts[candidate_indices],
            type=pa.uint32(),
        )
        arrays["pre_signal_target_topic_work_count"] = pa.array(
            pre_signal_counts[candidate_indices],
            type=pa.uint32(),
        )
        arrays["first_target_topic_year"] = pa.array(
            first_values,
            mask=first_values == np.iinfo(np.uint16).max,
            type=pa.uint16(),
        )
        arrays["observed_no_prior_target_topic"] = pa.array(
            target_counts[candidate_indices] == 0,
            type=pa.bool_(),
        )
        arrays["incumbent_before_signal_window"] = pa.array(
            pre_signal_counts[candidate_indices] > 0,
            type=pa.bool_(),
        )
        batch = pa.Table.from_arrays(
            [arrays[name] for name in AUTHOR_OUTPUT_SCHEMA.names],
            schema=AUTHOR_OUTPUT_SCHEMA,
        )
        result.append(batch)
        accounting[episode_index] = {
            "author_rows": len(batch),
            "observed_no_prior_target_topic_rows": int(
                (target_counts[candidate_indices] == 0).sum()
            ),
            "prior_target_topic_rows": int(
                (target_counts[candidate_indices] > 0).sum()
            ),
            "incumbent_before_signal_window_rows": int(
                (pre_signal_counts[candidate_indices] > 0).sum()
            ),
            "portfolio_focal_work_events_all": int(
                metrics["portfolio_window_focal_field_work_count_all"][
                    candidate_indices
                ].sum(dtype=np.uint64)
            ),
        }
    return result, accounting, int(eligible.sum())


def _write_author_base(
    paths: Sequence[Path],
    output: Path,
    *,
    linked: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    selected_work_ids: Mapping[int, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    writer = pq.ParquetWriter(
        output,
        AUTHOR_OUTPUT_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    totals = {
        int(episode["certified_episode_index"]): {
            "author_rows": 0,
            "observed_no_prior_target_topic_rows": 0,
            "prior_target_topic_rows": 0,
            "incumbent_before_signal_window_rows": 0,
            "portfolio_focal_work_events_all": 0,
        }
        for episode in episodes
    }
    source_rows = 0
    eligible_event_rows = 0
    maximum_batch_rows = 0
    try:
        for index, path in enumerate(paths):
            table = pq.read_table(path)
            source_rows += len(table)
            batches, accounting, eligible_rows = _author_batch(
                table,
                partition_index=index,
                linked=linked,
                episodes=episodes,
                selected_work_ids=selected_work_ids,
            )
            eligible_event_rows += eligible_rows
            for batch in batches:
                maximum_batch_rows = max(maximum_batch_rows, len(batch))
                if len(batch):
                    writer.write_table(batch)
            for episode_index, values in accounting.items():
                for key, value in values.items():
                    totals[episode_index][key] += int(value)
    finally:
        writer.close()
    os.chmod(output, 0o600)
    parquet = pq.ParquetFile(output)
    require(
        parquet.metadata.num_rows
        == sum(values["author_rows"] for values in totals.values()),
        "author output row closure failed",
    )
    return {
        "relative_path": output.name,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
    }, {
        "source_rows_scanned": source_rows,
        "eligible_author_work_event_rows": eligible_event_rows,
        "maximum_partition_episode_batch_rows": maximum_batch_rows,
        "by_episode": {str(key): value for key, value in totals.items()},
    }


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    os.chmod(path, 0o600)
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": len(table),
        "row_groups": pq.ParquetFile(path).metadata.num_row_groups,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Certified D0 author-opportunity rebuild",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Episode | Field | Topic | Entry | Eligible linked target works | Focal-field-active authors |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for episode in report["episodes"]:
        index = str(episode["certified_episode_index"])
        work = report["selected_work_accounting"]["by_episode"][index]
        author = report["author_opportunity_accounting"]["by_episode"][index]
        lines.append(
            f"| {index} | {episode['field_id']} | {episode['topic_id']} | "
            f"{episode['entry_year']} | {work['selected_work_rows']:,} | "
            f"{author['author_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "All counts are predecision and outcome-blind. Independent replay is",
            "required before leave-out, network, or bridge construction.",
            "",
        ]
    )
    return "\n".join(lines)


def _cleanup_temporary(temporary: Path | None, *, output: Path) -> None:
    if temporary is None:
        return
    require(temporary.is_absolute(), "temporary path must be absolute")
    require(temporary.parent == output.parent, "temporary path parent mismatch")
    require(
        temporary.name.startswith(f".{output.name}."),
        "temporary path prefix mismatch",
    )
    if temporary.exists():
        shutil.rmtree(temporary)


def build(
    *,
    config_path: Path,
    certified_selection_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, config_path.name)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V37",
        "rebuild not authorized",
    )
    repository = _mapping(config.get("repository_bindings"), "repository bindings")
    gate_binding = _validate_repo_binding(_mapping(repository["gate_state"], "gate binding"))
    abi_binding = _validate_repo_binding(
        _mapping(repository["measurement_abi"], "measurement ABI binding")
    )
    abi_config = _read_yaml(
        _safe_repo_path(repository["measurement_abi"]["relative_path"], "measurement ABI"),
        "measurement ABI",
    )
    episodes, selection_binding = _load_episodes(
        certified_selection_root,
        _mapping(config.get("certified_episode_selection"), "certified selection"),
    )
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden boundary")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary opened")
    projection = _mapping(config.get("measurement_projection"), "measurement projection")
    included_types = [int(value) for value in projection["inherited_work_type_codes"]]
    abi_dictionary = {
        int(code): str(name)
        for code, name in _mapping(
            abi_config["work_type_dictionary"],
            "ABI work type dictionary",
        ).items()
    }
    abi_included = _mapping(
        abi_config["work_type_eligibility"]["included"],
        "ABI included work types",
    )
    require(
        included_types
        == sorted(code for code, name in abi_dictionary.items() if name in abi_included),
        "projected included work types differ from frozen ABI",
    )
    dispositions = [int(value) for value in projection["allowed_temporal_dispositions"]]
    require(
        dispositions == abi_config["measurement_abi"]["allowed_temporal_dispositions"],
        "projected temporal dispositions differ from frozen ABI",
    )
    latest_year = int(projection["latest_predecision_year"])
    require(
        latest_year == abi_config["measurement_abi"]["latest_attention_year"],
        "projected latest year differs from frozen ABI",
    )
    fields = [int(value) for value in projection["all_field_metric_field_ids"]]
    require(fields == list(range(11, 37)), "all-field projection closure failed")
    require(
        [int(value) for value in projection["focal_field_ids"]] == [17, 27],
        "focal field projection differs",
    )
    inputs = _mapping(config.get("normalized_inputs"), "normalized inputs")
    work_paths, work_binding, work_stats = _validate_partitioned_input(
        cutoff_root,
        _mapping(inputs["cutoff_work_metadata"], "cutoff input"),
        schema=WORK_INPUT_SCHEMA,
        workers=workers,
    )
    event_paths, event_binding, event_stats = _validate_partitioned_input(
        author_history_root,
        _mapping(inputs["author_history_events"], "author input"),
        schema=EVENT_INPUT_SCHEMA,
        workers=workers,
    )
    output = output.resolve()
    require(output.name == config["output_contract"]["root_basename"], "output basename mismatch")
    require(not output.exists(), "output directory already exists")
    require(output.parent.is_dir(), "output parent missing")
    for input_root in (
        certified_selection_root.resolve(),
        cutoff_root.resolve(),
        author_history_root.resolve(),
    ):
        require(
            output != input_root and not output.is_relative_to(input_root),
            "output lies inside input root",
        )

    valid_work_ids, event_metadata, event_accounting = _event_work_index(
        event_paths,
        fields=fields,
        dispositions=dispositions,
        latest_year=latest_year,
        workers=workers,
    )
    require(
        event_accounting["event_rows_scanned"] == inputs["author_history_events"]["rows"],
        "author event row closure failed",
    )
    require(
        event_accounting["duplicate_author_work_event_rows"] == 0,
        "duplicate author-work events require an explicit deduplication successor",
    )
    linked, work_accounting = _linked_work_universe(
        work_paths,
        fields=fields,
        dispositions=dispositions,
        included_types=included_types,
        latest_year=latest_year,
        valid_work_ids=valid_work_ids,
        event_metadata=event_metadata,
        workers=workers,
    )
    del valid_work_ids, event_metadata
    require(
        work_accounting["rows"] == inputs["cutoff_work_metadata"]["rows"],
        "cutoff work row closure failed",
    )
    require(work_accounting["linked_rows"] == len(linked), "linked work closure failed")
    selected_work, selected_work_ids, selected_accounting = _selected_work_table(
        linked,
        episodes,
    )
    require(
        all(values["selected_work_rows"] > 0 for values in selected_accounting["by_episode"].values()),
        "certified episode has no eligible linked predecision work",
    )

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        selected_artifact = _write_parquet(
            selected_work,
            temporary / "certified_episode_selected_predecision_works.parquet",
        )
        author_artifact, author_accounting = _write_author_base(
            event_paths,
            temporary / "certified_pre_entry_author_opportunity_base.parquet",
            linked=linked,
            episodes=episodes,
            selected_work_ids=selected_work_ids,
        )
        require(
            author_accounting["source_rows_scanned"]
            == inputs["author_history_events"]["rows"],
            "author panel source row closure failed",
        )
        _validate_unchanged(work_paths, work_stats)
        _validate_unchanged(event_paths, event_stats)
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "label": config["label"],
            "status": STATUS,
            "scope": {
                "exact_certified_episode_rows": 4,
                "selected_work_manifest_built": True,
                "author_opportunity_base_built": True,
                "outcomes_read": False,
                "entry_or_persistence_rows_read": False,
                "realized_citation_rows_read": False,
                "final_candidate_eligibility_established": False,
                "leaveout_network_bridge_built": False,
                "independent_verification_complete": False,
                "d0_complete": False,
                "model_fitting_performed": False,
                "g4_rows_read": False,
                "api_requests_created": 0,
                "gpu_used": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "repository_bindings": {
                "gate_state": gate_binding,
                "measurement_abi": abi_binding,
            },
            "certified_selection_binding": selection_binding,
            "normalized_input_bindings": {
                "cutoff_work_metadata": work_binding,
                "author_history_events": event_binding,
            },
            "measurement_projection": {
                "included_work_type_codes": included_types,
                "included_work_type_names": [abi_dictionary[code] for code in included_types],
                "allowed_temporal_dispositions": dispositions,
                "latest_predecision_year": latest_year,
                "focal_field_ids": projection["focal_field_ids"],
                "all_field_metric_field_ids": fields,
                "team_cap_profiles": [10, 50, 100, "ALL"],
                "team_cap_selected": False,
            },
            "episodes": episodes,
            "author_event_index_accounting": event_accounting,
            "work_abi_accounting": work_accounting,
            "selected_work_accounting": {
                **selected_accounting,
                "artifact": selected_artifact,
            },
            "author_opportunity_accounting": {
                **author_accounting,
                "artifact": author_artifact,
            },
            "authorization": {
                "independent_verification": "REQUIRED_NEXT",
                "leaveout_network_bridge_build": "PAUSED_PENDING_INDEPENDENT_VERIFICATION",
                "production_pivot_classification": "NOT_AUTHORIZED",
                "discovery_outcome_opening": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "execution": {
                "workload_class": "CPU_IO",
                "workers": workers,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                "builder_sha256": _sha256(Path(__file__).resolve()),
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "certified_author_opportunity_rebuild.json"
        _write_json(report_path, report)
        markdown_path = temporary / "CERTIFIED_AUTHOR_OPPORTUNITY_REBUILD.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts = [
            selected_artifact,
            author_artifact,
            {
                "relative_path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            },
            {
                "relative_path": markdown_path.name,
                "bytes": markdown_path.stat().st_size,
                "sha256": _sha256(markdown_path),
            },
        ]
        artifacts.sort(key=lambda item: str(item["relative_path"]))
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
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
            "exact_certified_episode_rows": 4,
            "selected_work_manifest_built": True,
            "author_opportunity_base_built": True,
            "independent_verification_complete": False,
            "leaveout_network_bridge_authorized": False,
            "discovery_outcome_opening_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "outcomes_read": False,
            "gpu_used": False,
            "receipt_written_last": True,
        }
        _write_json(
            temporary / "certified_author_opportunity_rebuild.outputs.json",
            receipt,
        )
        os.replace(temporary, output)
        temporary = None
        return report
    finally:
        _cleanup_temporary(temporary, output=output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--certified-selection-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        config_path=args.config,
        certified_selection_root=args.certified_selection_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output=args.output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_work_accounting": report["selected_work_accounting"],
                "author_opportunity_accounting": report[
                    "author_opportunity_accounting"
                ],
                "execution": report["execution"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
