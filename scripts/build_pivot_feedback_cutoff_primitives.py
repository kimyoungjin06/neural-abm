#!/usr/bin/env python3
"""Build the normalized, receipt-bound cutoff-primitives component.

The output is intentionally a relational ABI, not a thresholded analytic risk
set.  It adds an outcome-blind work-metadata index to four already verified
production parents and freezes the exact predicates needed to construct a view
at an entry year ``t`` without selecting any G0 measurement values.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

import audit_openalex_canonical_work_time_spine_v3 as canonical
import audit_openalex_work_side_author_history_v3 as author


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/pivot_feedback_cutoff_primitives_v0.yaml"
CONFIG_SHA256 = "bdc1365d87a56d3571559ef21fdd56d21be570a8e712081d30748361f22bf764"
CONFIG_SCHEMA = "pivot_feedback_cutoff_primitives_config.v1"
PLAN_SCHEMA = "pivot_feedback_cutoff_primitives_plan.v1"
REPORT_SCHEMA = "pivot_feedback_cutoff_primitives_report.v1"
MANIFEST_SCHEMA = "pivot_feedback_cutoff_primitives_manifest.v1"
OUTPUTS_SCHEMA = "pivot_feedback_cutoff_primitives_outputs.v1"
STATE_SCHEMA = "pivot_feedback_cutoff_primitives_state.v1"

PARTITION_COUNT = 4096
FULL_SCOPE = "FULL_CURRENT_SNAPSHOT_PRODUCTION"
PILOT_SCOPE = "SOURCE_PREFIX_PILOT_PERMANENTLY_NONPROMOTABLE"
WORK_PATTERN = r"^https://openalex[.]org/W(0|[1-9][0-9]*)$"
TOPIC_PATTERN = r"^https://openalex[.]org/T(0|[1-9][0-9]*)$"
FIELD_PATTERN = r"^https://openalex[.]org/fields/(0|[1-9][0-9]*)$"
DOMAIN_PATTERN = r"^https://openalex[.]org/domains/(0|[1-9][0-9]*)$"
W_MINUS_ONE = "https://openalex.org/W-1"

RECORD_DTYPE = np.dtype(
    [
        ("work", "<u8"),
        ("type", "u1"),
        ("paratext", "u1"),
        ("retracted", "u1"),  # 0 false, 1 true, 2 null
        ("authors", "<u2"),
        ("topic", "<u4"),  # zero is the null sentinel
        ("field", "u1"),
        ("domain", "u1"),
    ],
    align=False,
)

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
CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("work_id_numeric", pa.uint64(), nullable=False),
        pa.field("publication_day", pa.int32(), nullable=True),
        pa.field("publication_year", pa.uint16(), nullable=True),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
    ]
)


class CutoffPrimitivesError(RuntimeError):
    """Controlled fail-closed error."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Strict loader for the trusted repository config."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CutoffPrimitivesError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be an array")
    return list(value)


def _integer(value: Any, field: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be int")
    return int(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return canonical._sha256_file(path)


def _atomic_json(path: Path, value: Any) -> None:
    canonical._atomic_bytes(path, _canonical_json_bytes(value))


def _atomic_text(path: Path, value: str) -> None:
    canonical._atomic_bytes(path, value.encode("utf-8"))


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _safe_relative(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value, f"{field} must be a path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe {field}")
    require(path.as_posix() == value, f"noncanonical {field}")
    return path


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    require(resolved == DEFAULT_CONFIG.resolve(strict=True), "only the frozen v0 config is accepted")
    payload = canonical._read_regular_file_bytes(resolved, field="config")
    digest = _sha256_bytes(payload)
    require(digest == CONFIG_SHA256, "config SHA-256 mismatch")
    try:
        parsed = yaml.load(payload.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CutoffPrimitivesError("invalid strict YAML config") from exc
    config = _mapping(parsed, "config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    dependencies = _mapping(config["implementation_dependencies"], "dependencies")
    for prefix in ("canonical_helper", "author_helper"):
        relative = _safe_relative(dependencies[f"{prefix}_path"], f"{prefix} path")
        require(
            _sha256_file(REPO_ROOT / relative) == dependencies[f"{prefix}_sha256"],
            f"{prefix} SHA-256 mismatch",
        )
    return config, digest


@dataclass(frozen=True)
class Parents:
    content_root: Path
    canonical_root: Path
    author_root: Path
    relation_root: Path
    citation_root: Path
    receipts: dict[str, dict[str, Any]]
    main_tasks: list[author.SourceTask]


def _load_receipt(
    root: Path, expected: Mapping[str, Any], label: str, *, verify_artifacts: bool
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    basename = str(expected["receipt_basename"])
    receipt_path = root / basename
    require(receipt_path.is_file(), f"missing {label} receipt")
    require(_sha256_file(receipt_path) == expected["receipt_sha256"], f"{label} receipt SHA mismatch")
    receipt = canonical._read_json(receipt_path, f"{label} receipt")
    require(isinstance(receipt, dict), f"{label} receipt must be an object")
    require(receipt.get("schema_version") == expected["receipt_schema_version"], f"{label} schema mismatch")
    require(receipt.get("status") == expected["required_status"], f"{label} status mismatch")
    for key in ("task_fingerprint_sha256", "result_fingerprint_sha256"):
        wanted = expected.get(key)
        if wanted is not None:
            require(receipt.get(key) == wanted, f"{label} {key} mismatch")
    wanted_count = expected.get("artifact_count")
    if wanted_count is not None:
        require(receipt.get("artifact_count") == wanted_count, f"{label} artifact count mismatch")
    if verify_artifacts:
        author._validate_receipt_artifacts(root, receipt, label)
    return dict(receipt)


def _schema_equal(path: Path, expected: pa.Schema, label: str) -> int:
    parquet = pq.ParquetFile(path)
    require(parquet.schema_arrow == expected, f"{label} schema mismatch")
    return int(parquet.metadata.num_rows)


def _validate_parent_surfaces(config: Mapping[str, Any], roots: Mapping[str, Path]) -> None:
    parents = _mapping(config["parents"], "parents")
    canonical_cfg = _mapping(parents["canonical_work_time_spine"], "canonical parent")
    author_cfg = _mapping(parents["author_history"], "author parent")
    relation_cfg = _mapping(parents["work_topic_relation"], "relation parent")
    citation_cfg = _mapping(parents["citation_chronology"], "citation parent")

    canonical_schema = pa.schema(
        [
            pa.field("work_id_numeric", pa.uint64(), nullable=False),
            pa.field("publication_day", pa.int32(), nullable=True),
            pa.field("publication_year", pa.uint16(), nullable=True),
            pa.field("temporal_disposition", pa.uint8(), nullable=False),
        ]
    )
    event_schema = author.EVENT_SCHEMA
    citation_schema = pa.schema(
        [
            pa.field("cited_work_id", pa.uint64(), nullable=False),
            pa.field("citing_work_id", pa.uint64(), nullable=False),
            pa.field("available_at", pa.date32(), nullable=False),
            pa.field("cited_publication_date", pa.date32(), nullable=False),
            pa.field("signed_lag_days", pa.int32(), nullable=False),
            pa.field("raw_pair_multiplicity", pa.uint32(), nullable=False),
        ]
    )
    canonical_rows = author_rows = citation_rows = 0
    for partition in range(PARTITION_COUNT):
        canonical_rows += _schema_equal(
            roots["canonical"] / "spine" / f"part-{partition:04d}.parquet",
            canonical_schema,
            f"canonical partition {partition}",
        )
        author_rows += _schema_equal(
            roots["author"] / "events" / f"part-{partition:04d}.parquet",
            event_schema,
            f"author-event partition {partition}",
        )
        citation_rows += _schema_equal(
            roots["citation"] / "chronology_safe_edges_by_cited_work" / f"part-{partition:04d}.parquet",
            citation_schema,
            f"citation partition {partition}",
        )
    require(canonical_rows == canonical_cfg["rows"], "canonical surface row mismatch")
    require(author_rows == author_cfg["event_rows"], "author-event surface row mismatch")
    require(citation_rows == citation_cfg["chronology_safe_rows"], "citation surface row mismatch")

    relation = roots["relation"] / _safe_relative(
        relation_cfg["relation_relative_path"], "relation path"
    )
    require(relation.stat().st_size == relation_cfg["relation_bytes"], "relation byte count mismatch")
    require(relation.stat().st_size % 16 == 0, "relation record alignment mismatch")
    require(relation.stat().st_size // 16 == relation_cfg["relation_rows"], "relation row count mismatch")
    require(_sha256_file(relation) == relation_cfg["relation_sha256"], "relation SHA mismatch")

    manifest = roots["citation"] / "chronology_safe_edge_manifest.json"
    manifest_value = canonical._read_json(manifest, "citation safe manifest")
    require(manifest_value.get("partition_count") == PARTITION_COUNT, "citation manifest partition mismatch")
    require(manifest_value.get("rows") == citation_cfg["chronology_safe_rows"], "citation manifest row mismatch")
    require(
        manifest_value.get("partition_tree_sha256")
        == citation_cfg["chronology_safe_partition_tree_sha256"],
        "citation manifest tree mismatch",
    )


def validate_parents(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    content_root: Path,
    canonical_root: Path,
    author_root: Path,
    relation_root: Path,
    citation_root: Path,
    verify_artifacts: bool,
) -> Parents:
    roots = {
        "content": content_root.resolve(strict=True),
        "canonical": canonical_root.resolve(strict=True),
        "author": author_root.resolve(strict=True),
        "relation": relation_root.resolve(strict=True),
        "citation": citation_root.resolve(strict=True),
    }
    parents_cfg = _mapping(config["parents"], "parents")
    receipts = {
        key: _load_receipt(roots[root_key], _mapping(parents_cfg[key], key), key, verify_artifacts=verify_artifacts)
        for key, root_key in (
            ("current_content_baseline", "content"),
            ("canonical_work_time_spine", "canonical"),
            ("author_history", "author"),
            ("work_topic_relation", "relation"),
            ("citation_chronology", "citation"),
        )
    }
    content_cfg = _mapping(parents_cfg["current_content_baseline"], "content parent")
    content_plan = canonical._read_json(roots["content"] / "audit_plan.json", "content plan")
    require(content_plan.get("plan_sha256") == content_cfg["plan_sha256"], "content plan binding mismatch")
    source_cfg = _mapping(config["source"], "source")
    source_root = source_root.resolve(strict=True)
    require(source_root.name == source_cfg["root_id"], "source root ID mismatch")
    tasks = [
        author._task_from_content(_mapping(item, "content task"), source_cfg["table"])
        for item in _list(content_plan.get("tasks"), "content tasks")
        if isinstance(item, dict) and item.get("table_name") == source_cfg["table"]
    ]
    tasks.sort(key=lambda task: task.relative_path)
    require(len(tasks) == source_cfg["files"], "main source file count mismatch")
    require(sum(task.source_bytes for task in tasks) == source_cfg["source_bytes"], "main source bytes mismatch")
    _validate_parent_surfaces(config, roots)
    return Parents(
        content_root=roots["content"],
        canonical_root=roots["canonical"],
        author_root=roots["author"],
        relation_root=roots["relation"],
        citation_root=roots["citation"],
        receipts=receipts,
        main_tasks=tasks,
    )


def _bool_array(value: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return pc.fill_null(value, False).to_numpy(zero_copy_only=False).astype(bool, copy=False)


def annual_cutoff_day(entry_year: int) -> int:
    """Return the Arrow date32 day for December 31 of ``entry_year - 1``."""

    require(1 < entry_year <= 9999, "entry year outside supported Gregorian range")
    epoch = dt.date(1970, 1, 1)
    return (dt.date(entry_year - 1, 12, 31) - epoch).days


def precutoff_work_mask(
    publication_year: np.ndarray,
    year_is_null: np.ndarray,
    temporal_disposition: np.ndarray,
    entry_year: int,
) -> np.ndarray:
    """Mechanically implement the frozen annual pre-cutoff work predicate."""

    require(
        len(publication_year) == len(year_is_null) == len(temporal_disposition),
        "pre-cutoff work array length mismatch",
    )
    return (
        ~np.asarray(year_is_null, dtype=bool)
        & np.isin(temporal_disposition, np.array([2, 3], dtype="u1"))
        & (publication_year <= entry_year - 1)
    )


def citation_pre_cutoff_mask(available_at_day: np.ndarray, entry_year: int) -> np.ndarray:
    """Implement the proxy citation predicate on non-null Arrow date32 days."""

    return np.asarray(available_at_day, dtype=np.int64) <= annual_cutoff_day(entry_year)


def _nullable_numeric_id(
    values: pa.Array, *, pattern: str, prefix_length: int, dtype: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    nulls = _bool_array(pc.is_null(values))
    result = np.zeros(len(values), dtype=dtype)
    indices = np.flatnonzero(~nulls)
    if len(indices):
        selected = values.take(pa.array(indices, type=pa.int64()))
        valid = _bool_array(pc.match_substring_regex(selected, pattern=pattern))
        require(bool(np.all(valid)), f"malformed {field}")
        parsed = author._parse_numeric_strings(
            selected,
            active=np.arange(len(selected), dtype=np.int64),
            prefix_length=prefix_length,
            field=field,
        )
        require(bool(np.all(parsed > 0)), f"{field} zero is reserved for null")
        require(bool(np.all(parsed <= np.iinfo(np.dtype(dtype)).max)), f"{field} outside {dtype}")
        result[indices] = parsed.astype(dtype, copy=False)
    return result, nulls


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    _private_dir(path.parent)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    require(not temporary.exists(), "stale parquet temporary file")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=6,
            row_group_size=262_144,
            use_dictionary=True,
            write_statistics=True,
        )
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        canonical._fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _canonical_partition(root: Path, partition: int) -> pa.Table:
    table = pq.read_table(root / "spine" / f"part-{partition:04d}.parquet")
    require(table.schema == CANONICAL_SCHEMA, "canonical partition schema mismatch")
    return table.combine_chunks()


def _scan_sources(
    config: Mapping[str, Any],
    *,
    source_root: Path,
    tasks: Sequence[author.SourceTask],
    spool_root: Path,
) -> dict[str, Any]:
    source_cfg = _mapping(config["source"], "source")
    profile = _mapping(config["input_profile"], "input profile")
    raw_dictionary = _mapping(profile["work_type_dictionary"], "work types")
    type_by_name = {str(name): int(code) for code, name in raw_dictionary.items()}
    type_names = list(type_by_name)
    projected_columns = [str(value) for value in _list(source_cfg["projected_columns"], "projected columns")]
    expected_schema = {
        "id": "string",
        "type": "string",
        "is_paratext": "bool",
        "is_retracted": "bool",
        "authors_count": "int64",
        "primary_topic__id": "string",
        "primary_topic__domain__id": "string",
        "primary_topic__field__id": "string",
    }
    metrics: dict[str, Any] = {
        "files": 0,
        "source_bytes": 0,
        "raw_rows": 0,
        "quarantined_w_minus_1_rows": 0,
        "retained_rows": 0,
        "null_counts": {key: 0 for key in expected_schema if key != "id"},
        "work_type_counts": Counter(),
        "authors_count_minimum": None,
        "authors_count_maximum": None,
    }
    with author.SpoolWriter(spool_root, RECORD_DTYPE, prefix="work") as writer:
        for task_index, task in enumerate(tasks, start=1):
            if task_index == 1 or task_index % 32 == 0 or task_index == len(tasks):
                print(f"source scan {task_index}/{len(tasks)}", file=sys.stderr, flush=True)
            source = author._source_path(source_root, task)
            parquet = pq.ParquetFile(source)
            require(author._projected_schema(parquet, projected_columns) == expected_schema, "projected source schema mismatch")
            file_rows = 0
            for batch in parquet.iter_batches(batch_size=262_144, columns=projected_columns):
                ids, types, paratexts, retracted, authors_count, topics, domains, fields = [
                    batch.column(index) for index in range(len(projected_columns))
                ]
                require(ids.null_count == 0, "null work ID")
                minus = _bool_array(pc.equal(ids, W_MINUS_ONE))
                valid = _bool_array(pc.match_substring_regex(ids, pattern=WORK_PATTERN))
                require(bool(np.all(valid | minus)), "malformed work ID")
                retained = np.flatnonzero(~minus)
                work = author._parse_numeric_strings(
                    ids,
                    active=retained,
                    prefix_length=len("https://openalex.org/W"),
                    field="work ID",
                )

                require(types.null_count == 0, "null work type")
                type_indices = pc.index_in(types, value_set=pa.array(type_names, type=pa.string()))
                require(type_indices.null_count == 0, "unknown work type")
                type_zero = np.asarray(type_indices.to_numpy(zero_copy_only=False), dtype=np.int16)
                require(bool(np.all(type_zero >= 0)), "unknown work type")
                type_codes_all = np.array([type_by_name[type_names[index]] for index in type_zero], dtype="u1")

                require(paratexts.null_count == 0, "null is_paratext")
                paratext_all = np.asarray(paratexts.to_numpy(zero_copy_only=False), dtype=bool)
                retract_null = _bool_array(pc.is_null(retracted))
                retract_all = _bool_array(retracted).astype("u1")
                retract_all[retract_null] = 2
                require(authors_count.null_count == 0, "null authors_count")
                authors_all = np.asarray(authors_count.to_numpy(zero_copy_only=False), dtype=np.int64)
                require(bool(np.all((authors_all >= 0) & (authors_all <= np.iinfo(np.uint16).max))), "authors_count outside uint16")

                topic_all, topic_null = _nullable_numeric_id(
                    topics,
                    pattern=TOPIC_PATTERN,
                    prefix_length=len("https://openalex.org/T"),
                    dtype="<u4",
                    field="primary topic ID",
                )
                domain_all, domain_null = _nullable_numeric_id(
                    domains,
                    pattern=DOMAIN_PATTERN,
                    prefix_length=len("https://openalex.org/domains/"),
                    dtype="u1",
                    field="primary domain ID",
                )
                field_all, field_null = _nullable_numeric_id(
                    fields,
                    pattern=FIELD_PATTERN,
                    prefix_length=len("https://openalex.org/fields/"),
                    dtype="u1",
                    field="primary field ID",
                )
                require(np.array_equal(topic_null, domain_null), "topic/domain null-mask mismatch")
                require(np.array_equal(topic_null, field_null), "topic/field null-mask mismatch")

                block = np.empty(len(retained), dtype=RECORD_DTYPE)
                block["work"] = work
                block["type"] = type_codes_all[retained]
                block["paratext"] = paratext_all[retained].astype("u1", copy=False)
                block["retracted"] = retract_all[retained]
                block["authors"] = authors_all[retained].astype("<u2", copy=False)
                block["topic"] = topic_all[retained]
                block["field"] = field_all[retained]
                block["domain"] = domain_all[retained]
                writer.write(block, (work % PARTITION_COUNT).astype("<u2"))

                raw = len(ids)
                file_rows += raw
                metrics["raw_rows"] += raw
                metrics["quarantined_w_minus_1_rows"] += int(np.count_nonzero(minus))
                metrics["retained_rows"] += len(retained)
                for name, array in zip(
                    ("type", "is_paratext", "is_retracted", "authors_count", "primary_topic__id", "primary_topic__domain__id", "primary_topic__field__id"),
                    (types, paratexts, retracted, authors_count, topics, domains, fields),
                    strict=True,
                ):
                    metrics["null_counts"][name] += array.null_count
                metrics["work_type_counts"].update(types.to_pylist())
                batch_min = int(authors_all.min()) if len(authors_all) else None
                batch_max = int(authors_all.max()) if len(authors_all) else None
                if batch_min is not None:
                    metrics["authors_count_minimum"] = batch_min if metrics["authors_count_minimum"] is None else min(metrics["authors_count_minimum"], batch_min)
                    metrics["authors_count_maximum"] = batch_max if metrics["authors_count_maximum"] is None else max(metrics["authors_count_maximum"], batch_max)
            require(file_rows == parquet.metadata.num_rows, "source footer row mismatch")
            metrics["files"] += 1
            metrics["source_bytes"] += task.source_bytes
    metrics["work_type_counts"] = dict(sorted(metrics["work_type_counts"].items()))
    return metrics


def _work_table(records: np.ndarray, canonical_table: pa.Table, selected: np.ndarray) -> pa.Table:
    publication_day = canonical_table["publication_day"].take(pa.array(selected, type=pa.int64()))
    publication_year = canonical_table["publication_year"].take(pa.array(selected, type=pa.int64()))
    disposition = canonical_table["temporal_disposition"].take(pa.array(selected, type=pa.int64()))
    retract_state = records["retracted"]
    topic_null = records["topic"] == 0
    return pa.Table.from_arrays(
        [
            pa.array(records["work"], type=pa.uint64()),
            publication_day,
            publication_year,
            disposition,
            pa.array(records["type"], type=pa.uint8()),
            pa.array(records["paratext"].astype(bool), type=pa.bool_()),
            pa.array(retract_state == 1, mask=retract_state == 2, type=pa.bool_()),
            pa.array(records["authors"], type=pa.uint16()),
            pa.array(records["topic"], mask=topic_null, type=pa.uint32()),
            pa.array(records["field"], mask=topic_null, type=pa.uint8()),
            pa.array(records["domain"], mask=topic_null, type=pa.uint8()),
        ],
        schema=WORK_SCHEMA,
    )


def _materialize_partitions(
    *, output: Path, canonical_root: Path, spool_root: Path, full_scope: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parts: list[dict[str, Any]] = []
    total_rows = 0
    for partition in range(PARTITION_COUNT):
        if partition % 128 == 0:
            print(f"work partitions {partition}/{PARTITION_COUNT}", file=sys.stderr, flush=True)
        spool = spool_root / f"work-{partition:04d}.bin"
        require(spool.stat().st_size % RECORD_DTYPE.itemsize == 0, "spool alignment mismatch")
        records = np.fromfile(spool, dtype=RECORD_DTYPE)
        if len(records):
            records = records[np.argsort(records["work"], kind="stable")]
            require(bool(np.all(records["work"][1:] > records["work"][:-1])), "duplicate work ID")
            require(bool(np.all(records["work"] % PARTITION_COUNT == partition)), "work partition mismatch")
        canonical_table = _canonical_partition(canonical_root, partition)
        canonical_work = canonical_table["work_id_numeric"].to_numpy(zero_copy_only=False)
        if full_scope:
            require(np.array_equal(records["work"], canonical_work), f"full source/canonical closure failed in partition {partition}")
            selected = np.arange(len(canonical_work), dtype=np.int64)
        else:
            selected = np.searchsorted(canonical_work, records["work"])
            require(bool(np.all(selected < len(canonical_work))), "pilot work outside canonical partition")
            require(np.array_equal(canonical_work[selected], records["work"]), "pilot work missing from canonical partition")
        table = _work_table(records, canonical_table, selected)
        target = output / "work_metadata" / f"part-{partition:04d}.parquet"
        _write_parquet_atomic(target, table)
        item = {
            "partition_id": partition,
            "relative_path": target.relative_to(output).as_posix(),
            "rows": len(records),
            "bytes": target.stat().st_size,
            "sha256": _sha256_file(target),
            "first_work_id": int(records["work"][0]) if len(records) else None,
            "last_work_id": int(records["work"][-1]) if len(records) else None,
        }
        parts.append(item)
        total_rows += len(records)
    tree = canonical._tree_digest(parts)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "partition_count": PARTITION_COUNT,
        "partition_formula": "work_id_numeric_mod_4096",
        "rows": total_rows,
        "work_metadata_tree_sha256": tree,
        "partitions": parts,
    }
    return manifest, parts


def _profile_matches(config: Mapping[str, Any], metrics: Mapping[str, Any], full_scope: bool) -> None:
    source = _mapping(config["source"], "source")
    profile = _mapping(config["input_profile"], "profile")
    if not full_scope:
        return
    for key in ("files", "source_bytes", "raw_rows", "quarantined_w_minus_1_rows", "retained_rows"):
        require(metrics[key] == source[key], f"full source {key} mismatch")
    require(metrics["null_counts"] == profile["null_counts"], "full null profile mismatch")
    require(metrics["work_type_counts"] == profile["exact_work_type_counts"], "full work-type profile mismatch")
    require(metrics["authors_count_minimum"] == profile["authors_count_minimum"], "authors_count minimum mismatch")
    require(metrics["authors_count_maximum"] == profile["authors_count_maximum"], "authors_count maximum mismatch")


def _artifact_contract(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _result_fingerprint(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("result_fingerprint_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["source_accounting"]
    return f"""# Pivot-feedback normalized cutoff primitives

- Status: `{report['status']}`
- Scope: `{report['scope']}`
- Work rows: `{metrics['retained_rows']:,}`
- Work partitions: `{report['work_metadata']['partition_count']:,}`
- Work-metadata tree: `{report['work_metadata']['work_metadata_tree_sha256']}`
- GPU used: `false`

This is a normalized cutoff-primitives component, not a final analytic risk
set. Topic membership and citation availability remain current-snapshot
retrospective proxies. Candidate eligibility, allowed work types, team caps,
same-day policy, fixed outcome horizon, sampling, and weights remain pending in
a versioned G0 successor.
"""


def _terminal_reentry(output: Path) -> dict[str, Any]:
    receipt = canonical._read_json(output / "pivot_feedback_cutoff_panel.outputs.json", "receipt")
    require(receipt.get("schema_version") == OUTPUTS_SCHEMA, "terminal receipt schema mismatch")
    author._validate_receipt_artifacts(output, receipt, "terminal output")
    return dict(receipt)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha = load_config(args.config)
    output = args.output.resolve()
    if (output / "pivot_feedback_cutoff_panel.outputs.json").is_file():
        return _terminal_reentry(output)
    require(not output.exists(), "output must be fresh or terminal")
    source_root = args.source_root.resolve(strict=True)
    full_scope = args.max_source_files is None
    if full_scope:
        require(args.authorize_full, "full execution requires --authorize-full")
    else:
        require(args.max_source_files > 0, "--max-source-files must be positive")
        require(not args.authorize_full, "pilot cannot use --authorize-full")

    execution = _mapping(config["execution"], "execution")
    available_memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:", 1)[1].split()[0]) * 1024
    require(available_memory >= execution["minimum_available_ram_bytes"], "insufficient available memory")
    output_parent = output.parent.resolve(strict=True)
    free_disk = shutil.disk_usage(output_parent).free
    require(free_disk >= execution["minimum_free_disk_bytes"], "insufficient free output disk")
    if execution["require_source_and_output_distinct_devices"]:
        require(source_root.stat().st_dev != output_parent.stat().st_dev, "source and output must use distinct devices")

    verify_parent_artifacts = bool(full_scope and execution["verify_every_parent_receipt_artifact_before_full"])
    parents = validate_parents(
        config,
        source_root=source_root,
        content_root=args.content_root,
        canonical_root=args.canonical_root,
        author_root=args.author_root,
        relation_root=args.relation_root,
        citation_root=args.citation_root,
        verify_artifacts=verify_parent_artifacts,
    )
    tasks = parents.main_tasks if full_scope else parents.main_tasks[: args.max_source_files]
    scope = FULL_SCOPE if full_scope else PILOT_SCOPE
    task_payload = {
        "config_sha256": config_sha,
        "scope": scope,
        "source_tasks": [
            {"relative_path": task.relative_path, "bytes": task.source_bytes, "source_stat": task.source_stat}
            for task in tasks
        ],
        "parent_receipt_sha256": {
            key: _mapping(config["parents"], "parents")[key]["receipt_sha256"]
            for key in parents.receipts
        },
    }
    task_fingerprint = _sha256_bytes(_canonical_json_bytes(task_payload))
    _private_dir(output)
    _private_dir(output / ".cutoff_primitives_tmp" / "work_spools")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "scope": scope,
        "config_sha256": config_sha,
        "task_fingerprint_sha256": task_fingerprint,
        "source_root_id": source_root.name,
        "source_tasks": task_payload["source_tasks"],
        "parent_receipt_sha256": task_payload["parent_receipt_sha256"],
        "verify_every_parent_receipt_artifact": verify_parent_artifacts,
        "normalized_relational_abi": config["normalized_relational_abi"],
        "cutoff_predicate_abi": config["cutoff_predicate_abi"],
    }
    _atomic_json(output / "audit_plan.json", plan)
    metrics = _scan_sources(
        config,
        source_root=source_root,
        tasks=tasks,
        spool_root=output / ".cutoff_primitives_tmp" / "work_spools",
    )
    _profile_matches(config, metrics, full_scope)
    manifest, _ = _materialize_partitions(
        output=output,
        canonical_root=parents.canonical_root,
        spool_root=output / ".cutoff_primitives_tmp" / "work_spools",
        full_scope=full_scope,
    )
    require(manifest["rows"] == metrics["retained_rows"], "manifest/source row closure failed")
    _atomic_json(output / "work_metadata_manifest.json", manifest)
    shutil.rmtree(output / ".cutoff_primitives_tmp")
    scope_cfg = _mapping(config["scope"], "scope")
    status = scope_cfg["full_status"] if full_scope else scope_cfg["pilot_status"]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "scope": scope,
        "production_component_pass": full_scope,
        "pilot_promotion_allowed": False,
        "gpu_used": False,
        "source_accounting": metrics,
        "work_metadata": {
            "partition_count": manifest["partition_count"],
            "rows": manifest["rows"],
            "work_metadata_tree_sha256": manifest["work_metadata_tree_sha256"],
            "full_source_canonical_closure": full_scope,
        },
        "parent_evidence": {
            key: {
                "receipt_sha256": _mapping(config["parents"], "parents")[key]["receipt_sha256"],
                "result_fingerprint_sha256": receipt.get("result_fingerprint_sha256"),
                "all_receipt_artifacts_rehashed": verify_parent_artifacts,
            }
            for key, receipt in parents.receipts.items()
        },
        "normalized_relational_abi": config["normalized_relational_abi"],
        "cutoff_predicate_abi": config["cutoff_predicate_abi"],
        "gate_state": config["gate_state_after_component_pass"],
        "claim_boundary": config["claim_boundary"],
        "provenance": {
            "config_sha256": config_sha,
            "tool_sha256": _sha256_file(Path(__file__)),
            "task_fingerprint_sha256": task_fingerprint,
        },
    }
    report["result_fingerprint_sha256"] = _result_fingerprint(report)
    _atomic_json(output / "pivot_feedback_cutoff_panel.json", report)
    _atomic_text(output / "pivot_feedback_cutoff_panel.md", _render_markdown(report))
    state = {
        "schema_version": STATE_SCHEMA,
        "state": "COMPLETE",
        "status": status,
        "scope": scope,
        "task_fingerprint_sha256": task_fingerprint,
        "result_fingerprint_sha256": report["result_fingerprint_sha256"],
    }
    _atomic_json(output / "run_state.json", state)

    artifact_paths = sorted(
        [path for path in output.rglob("*") if path.is_file()],
        key=lambda path: path.relative_to(output).as_posix(),
    )
    artifacts = [_artifact_contract(path, output) for path in artifact_paths]
    artifact_tree = canonical._tree_digest(artifacts)
    receipt = {
        "schema_version": OUTPUTS_SCHEMA,
        "status": status,
        "scope": scope,
        "production_component_pass": full_scope,
        "completion_contract": scope_cfg["completion_contract"],
        "operator_paths_redacted": True,
        "artifact_count": len(artifacts),
        "artifact_tree_sha256": artifact_tree,
        "work_metadata_tree_sha256": manifest["work_metadata_tree_sha256"],
        "work_metadata_rows": manifest["rows"],
        "task_fingerprint_sha256": task_fingerprint,
        "result_fingerprint_sha256": report["result_fingerprint_sha256"],
        "artifacts": artifacts,
    }
    _atomic_json(output / "pivot_feedback_cutoff_panel.outputs.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--relation-root", type=Path, required=True)
    parser.add_argument("--citation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-files", type=int)
    parser.add_argument("--authorize-full", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(args)
    except (CutoffPrimitivesError, author.AuthorHistoryError, canonical.SpineAuditError, OSError, pa.ArrowException, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED_NORMALIZED_CUTOFF_PRIMITIVES", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
