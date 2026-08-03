#!/usr/bin/env python3
"""Build the receipt-bound S0 ORCID and affiliation projection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "pivot_feedback_s0_authorship_orcid_affiliation_config.v1"
REPORT_SCHEMA = "pivot_feedback_s0_authorship_orcid_affiliation_report.v1"
FIELD_YEAR_SCHEMA = (
    "pivot_feedback_s0_authorship_orcid_affiliation_field_year.v1"
)
FULL_STATUS = (
    "PASS_S0_FULL_AUTHORSHIP_ORCID_AND_AFFILIATION_PROJECTION_"
    "WITH_TOKEN_AND_CURRENT_SNAPSHOT_LIMITATIONS"
)
PILOT_STATUS = "PILOT_S0_AUTHORSHIP_ORCID_AND_AFFILIATION_NONPROMOTABLE"
CHECKPOINT_SCHEMA = "openalex_g1_content_freeze_checkpoint.v1"
PARTITION_COUNT = 256
WORK_PARTITIONS = 4096
FIELD_MIN = 11
FIELD_MAX = 36
YEAR_MULTIPLIER = 65_536
WORK_PREFIX = "https://openalex.org/W"
AUTHOR_PREFIX = "https://openalex.org/A"
INSTITUTION_PREFIX = "https://openalex.org/I"
ORCID_PREFIX = "https://orcid.org/"
AUTHOR_ORCID_DTYPE = np.dtype(
    [("author_numeric_id", "<u8"), ("orcid_code", "<u8")], align=False
)
INSTITUTION_DTYPE = np.dtype(
    [("field_year_key", "<u4"), ("institution_numeric_id", "<u8")],
    align=False,
)
STABLE_AUTHOR_DTYPE = AUTHOR_ORCID_DTYPE
SOURCE_COLUMNS = [
    "id",
    "authorships__raw_affiliation_strings",
    "authorships__institutions",
    "authorships__author__id",
    "authorships__author__orcid",
]
FIELD_YEAR_METRICS = (
    "raw_authorship_rows",
    "grammar_valid_author_id_rows",
    "nonnull_orcid_rows",
    "syntax_valid_orcid_rows",
    "checksum_valid_orcid_rows",
    "author_work_affiliation_rows",
    "author_work_institution_rows",
    "stable_orcid_author_work_pairs",
    "stable_orcid_author_tokens",
    "distinct_institutions",
)


class ProjectionError(RuntimeError):
    """Fail-closed projection error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProjectionError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(isinstance(current, Mapping) and part in current, f"missing {dotted_path}")
        current = current[part]
    return current


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a string")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    require(str(pure) == value, f"{label} must be normalized POSIX")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectionError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ProjectionError(f"invalid {label}: {error}") from error


def _stat_identity(path: Path) -> dict[str, int]:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), f"invalid source file {path}")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(stat.S_IMODE(info.st_mode)),
        "bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _validate_repository_binding(
    raw: Mapping[str, Any],
    *,
    yaml_document: bool,
) -> dict[str, Any]:
    item = dict(raw)
    relative = _safe_relative(item.get("relative_path"), "repository binding path")
    path = (REPO_ROOT / relative).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes root")
    require(_sha256_file(path) == item.get("sha256"), f"repository SHA mismatch: {relative}")
    document = _read_yaml(path, relative) if yaml_document else _read_json(path, relative)
    require(
        document.get("schema_version") == item.get("required_schema_version"),
        f"repository schema mismatch: {relative}",
    )
    require(
        document.get("status") == item.get("required_status"),
        f"repository status mismatch: {relative}",
    )
    for dotted_path, expected in _mapping(
        item.get("required_fields"), "repository required fields"
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"repository field mismatch: {dotted_path}",
        )
    return {"relative_path": relative, "sha256": item["sha256"], "status": document["status"]}


def _validate_receipt_root(
    root: Path,
    binding: Mapping[str, Any],
    *,
    plan_required: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid receipt root")
    require(root.name == binding.get("expected_root_basename"), "receipt root basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256_file(receipt_path) == binding.get("receipt_sha256"), "receipt SHA mismatch")
    receipt = _read_json(receipt_path, "operator receipt")
    require(receipt.get("schema_version") == binding.get("receipt_schema_version"), "receipt schema mismatch")
    require(receipt.get("status") == binding.get("required_status"), "receipt status mismatch")
    plan = None
    if plan_required:
        plan_path = root / str(binding["plan_basename"])
        require(_sha256_file(plan_path) == binding.get("plan_sha256"), "operator plan SHA mismatch")
        plan = _read_json(plan_path, "operator plan")
        require(plan.get("schema_version") == binding.get("plan_schema_version"), "operator plan schema mismatch")
    return receipt, plan


def _validate_content_root(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    receipt, _ = _validate_receipt_root(root, binding)
    require(receipt.get("task_fingerprint_sha256") == binding["task_fingerprint_sha256"], "content task fingerprint mismatch")
    require(receipt.get("plan_sha256") == binding["plan_sha256"], "content plan mismatch")
    require(receipt.get("content_tree_sha256") == binding["content_tree_sha256"], "content tree mismatch")
    tree = _mapping(receipt.get("checkpoint_tree"), "content checkpoint tree")
    require(tree.get("files") == binding["checkpoint_files"], "content checkpoint count mismatch")
    require(tree.get("tree_sha256") == binding["checkpoint_tree_sha256"], "checkpoint tree mismatch")
    return {
        "receipt_sha256": binding["receipt_sha256"],
        "task_fingerprint_sha256": binding["task_fingerprint_sha256"],
        "plan_sha256": binding["plan_sha256"],
        "content_tree_sha256": binding["content_tree_sha256"],
        "checkpoint_tree_sha256": binding["checkpoint_tree_sha256"],
    }


def _validate_source_tasks_and_checkpoints(
    source_root: Path,
    content_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    content_binding: Mapping[str, Any],
    *,
    table_name: str,
) -> dict[str, Any]:
    total_bytes = 0
    checkpoint_bytes = 0
    task_indices: set[int] = set()
    content_commitments: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(tasks):
        task = dict(raw)
        relative = _safe_relative(task.get("relative_path"), "source task path")
        path = (source_root / relative).resolve(strict=True)
        require(path.is_relative_to(source_root), "source task escapes root")
        current = _stat_identity(path)
        require(current == task.get("source_stat"), f"source stat mismatch: {relative}")
        digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        checkpoint_path = content_root / "checkpoints" / digest[:2] / f"{digest}.json"
        checkpoint_raw = checkpoint_path.read_bytes()
        checkpoint = _mapping(json.loads(checkpoint_raw.decode("utf-8")), "content checkpoint")
        require(checkpoint.get("schema_version") == CHECKPOINT_SCHEMA, "checkpoint schema mismatch")
        require(checkpoint.get("task_fingerprint_sha256") == content_binding["task_fingerprint_sha256"], "checkpoint task mismatch")
        require(checkpoint.get("plan_sha256") == content_binding["plan_sha256"], "checkpoint plan mismatch")
        require(checkpoint.get("relative_path") == relative, "checkpoint path mismatch")
        require(checkpoint.get("table_name") == table_name, "checkpoint table mismatch")
        task_index = checkpoint.get("task_index")
        require(
            isinstance(task_index, int)
            and not isinstance(task_index, bool)
            and task_index >= 0
            and task_index not in task_indices,
            "checkpoint task index invalid or duplicated",
        )
        task_indices.add(task_index)
        require(checkpoint.get("source_identity_before") == current, "checkpoint pre-stat mismatch")
        require(checkpoint.get("source_identity_after") == current, "checkpoint post-stat mismatch")
        require(
            checkpoint.get("expected_parent_identity")
            == {"bytes": current["bytes"], "mtime_ns": current["mtime_ns"]},
            "checkpoint expected parent identity mismatch",
        )
        require(checkpoint.get("bytes_hashed") == current["bytes"], "checkpoint byte count mismatch")
        require(isinstance(checkpoint.get("content_sha256"), str) and len(checkpoint["content_sha256"]) == 64, "checkpoint content digest missing")
        claimed = checkpoint.get("checkpoint_sha256")
        core = dict(checkpoint)
        del core["checkpoint_sha256"]
        require(
            claimed == hashlib.sha256(_canonical_bytes(core)).hexdigest(),
            "checkpoint self-digest mismatch",
        )
        total_bytes += current["bytes"]
        checkpoint_bytes += len(checkpoint_raw)
        content_commitments.append(
            {
                "ordinal": ordinal,
                "task_index": task_index,
                "relative_path": relative,
                "source_bytes": current["bytes"],
                "content_sha256": checkpoint["content_sha256"],
                "checkpoint_file_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
            }
        )
    return {
        "selected_source_files": len(tasks),
        "selected_source_bytes": total_bytes,
        "selected_checkpoints_verified": len(tasks),
        "selected_checkpoint_bytes": checkpoint_bytes,
        "selected_content_commitment_tree_sha256": hashlib.sha256(
            _canonical_bytes(content_commitments)
        ).hexdigest(),
        "current_exact_stat_verified_for_every_source_file": True,
        "source_content_rehashed_in_this_run": False,
    }


def _receipt_artifacts(
    root: Path,
    receipt: Mapping[str, Any],
    prefix: str,
    expected: int,
) -> list[tuple[Path, dict[str, Any]]]:
    selected = [
        _mapping(item, "receipt artifact")
        for item in _list(receipt.get("artifacts"), "receipt artifacts")
        if str(_mapping(item, "receipt artifact").get("relative_path", "")).startswith(prefix)
    ]
    selected.sort(key=lambda item: str(item["relative_path"]))
    require(len(selected) == expected, f"{prefix} artifact count mismatch")
    result = []
    for index, item in enumerate(selected):
        relative = f"{prefix}part-{index:04d}.parquet"
        require(item.get("relative_path") == relative, f"{prefix} partition sequence mismatch")
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(), "invalid normalized artifact")
        require(path.stat().st_size == item.get("bytes"), "normalized artifact size mismatch")
        result.append((path, item))
    return result


def _load_work_cell_lookup(
    cutoff_root: Path,
    receipt: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    require(receipt.get("work_metadata_tree_sha256") == binding["work_metadata_tree_sha256"], "work tree mismatch")
    artifacts = _receipt_artifacts(
        cutoff_root, receipt, "work_metadata/", int(binding["expected_work_artifacts"])
    )
    maximum_id = 0
    for path, _ in artifacts:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(row_group).column(0).statistics
            maximum_id = max(maximum_id, int(stats.max))
    lookup = np.zeros(maximum_id + 1, dtype=np.uint32)
    rows = 0
    digest_rows: list[tuple[str, int, str]] = []
    for index, (path, item) in enumerate(artifacts):
        observed_sha = _sha256_file(path)
        require(observed_sha == item["sha256"], f"work artifact SHA mismatch: {index}")
        table = pq.read_table(
            path,
            columns=["work_id_numeric", "publication_year", "primary_field_id"],
        )
        work_ids = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
        require(bool(np.all(work_ids % WORK_PARTITIONS == index)), "work partition formula mismatch")
        years = (
            table["publication_year"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint32, copy=False)
        )
        fields = (
            table["primary_field_id"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint32, copy=False)
        )
        require(bool(np.all(lookup[work_ids] == 0)), "duplicate work ID in lookup")
        lookup[work_ids] = ((fields + 1) << 16) | years
        rows += len(table)
        digest_rows.append((item["relative_path"], int(item["bytes"]), observed_sha))
    require(rows == binding["expected_work_rows"], "work lookup row closure failed")
    return lookup, {
        "artifacts_rehashed": len(artifacts),
        "artifact_bytes": sum(int(item["bytes"]) for _, item in artifacts),
        "artifact_tree_sha256": hashlib.sha256(_canonical_bytes(digest_rows)).hexdigest(),
        "rows": rows,
        "maximum_work_numeric_id": maximum_id,
        "lookup_bytes": int(lookup.nbytes),
    }


def _regex_mask(array: pa.Array, pattern: str) -> np.ndarray:
    return (
        pc.fill_null(pc.match_substring_regex(array, pattern), False)
        .to_numpy(zero_copy_only=False)
        .astype(bool, copy=False)
    )


def _parse_openalex_ids(
    array: pa.Array,
    *,
    pattern: str,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    valid = _regex_mask(array, pattern)
    safe = pc.if_else(pa.array(valid), array, pa.scalar(f"{prefix}0"))
    suffix = pc.utf8_slice_codeunits(safe, start=len(prefix))
    numeric = pc.cast(suffix, pa.uint64()).to_numpy(zero_copy_only=False)
    return numeric.astype(np.uint64, copy=False), valid


def _parse_orcid(
    array: pa.Array,
    pattern: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    syntax = _regex_mask(array, pattern)
    compact = pc.replace_substring(array, "-", "")
    safe = pc.if_else(
        pa.array(syntax),
        compact,
        pa.scalar(f"{ORCID_PREFIX}0000000000000000"),
    )
    prefix_text = pc.utf8_slice_codeunits(safe, start=len(ORCID_PREFIX), stop=len(ORCID_PREFIX) + 15)
    prefix_number = pc.cast(prefix_text, pa.uint64()).to_numpy(zero_copy_only=False)
    check_text = pc.utf8_slice_codeunits(
        safe, start=len(ORCID_PREFIX) + 15, stop=len(ORCID_PREFIX) + 16
    )
    is_x = (
        pc.fill_null(pc.equal(check_text, "X"), False)
        .to_numpy(zero_copy_only=False)
        .astype(bool, copy=False)
    )
    safe_check = pc.if_else(pa.array(is_x), pa.scalar("0"), check_text)
    check_digit = pc.cast(safe_check, pa.uint8()).to_numpy(zero_copy_only=False).astype(np.uint8)
    check_digit[is_x] = 10
    total = np.zeros(len(prefix_number), dtype=np.uint64)
    divisor = 10**14
    for _ in range(15):
        digit = (prefix_number // divisor) % 10
        total = (total + digit) * 2
        divisor //= 10
    expected = ((12 - (total % 11)) % 11).astype(np.uint8)
    valid = syntax & (expected == check_digit)
    code = prefix_number * np.uint64(11) + check_digit.astype(np.uint64) + np.uint64(1)
    code[~valid] = 0
    return code, syntax, valid


def _nonempty_list_row_mask(array: pa.Array) -> np.ndarray:
    flags = np.zeros(len(array), dtype=bool)
    flat = pc.list_flatten(array)
    if len(flat) == 0:
        return flags
    parents = pc.list_parent_indices(array).to_numpy(zero_copy_only=False)
    nonempty = (
        pc.fill_null(pc.greater(pc.utf8_length(flat), 0), False)
        .to_numpy(zero_copy_only=False)
        .astype(bool, copy=False)
    )
    flags[parents[nonempty]] = True
    return flags


def _institution_projection(
    array: pa.Array,
    pattern: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_flags = np.zeros(len(array), dtype=bool)
    flat = pc.list_flatten(array)
    if len(flat) == 0:
        return row_flags, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64)
    parents = pc.list_parent_indices(array).to_numpy(zero_copy_only=False).astype(np.int64)
    ids = flat.field("id")
    numeric, valid = _parse_openalex_ids(ids, pattern=pattern, prefix=INSTITUTION_PREFIX)
    valid_parents = parents[valid]
    row_flags[valid_parents] = True
    return row_flags, valid_parents, numeric[valid]


def _add_grouped(
    cells: dict[int, dict[str, int]],
    keys: np.ndarray,
    metric: str,
) -> None:
    if len(keys) == 0:
        return
    unique, counts = np.unique(keys, return_counts=True)
    for key, count in zip(unique.tolist(), counts.tolist(), strict=True):
        cell = cells.setdefault(int(key), {})
        cell[metric] = cell.get(metric, 0) + int(count)


class PartitionedSpool:
    def __init__(self, root: Path, prefix: str, dtype: np.dtype) -> None:
        self.root = root
        self.prefix = prefix
        self.dtype = dtype
        root.mkdir(parents=True, exist_ok=False)
        self.paths = [root / f"{prefix}-{index:03d}.bin" for index in range(PARTITION_COUNT)]
        self.handles = [path.open("wb", buffering=1024 * 1024) for path in self.paths]
        for path in self.paths:
            os.chmod(path, 0o600)
        self.rows = 0

    def append(self, records: np.ndarray, buckets: np.ndarray) -> None:
        require(records.dtype == self.dtype, "spool dtype mismatch")
        require(len(records) == len(buckets), "spool bucket length mismatch")
        if len(records) == 0:
            return
        order = np.argsort(buckets, kind="stable")
        sorted_buckets = buckets[order]
        sorted_records = records[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_buckets[1:] != sorted_buckets[:-1], True])
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            bucket = int(sorted_buckets[start])
            self.handles[bucket].write(sorted_records[start:stop].tobytes())
        self.rows += len(records)

    def close(self) -> None:
        for handle in self.handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()


def _scan_authorship_source(
    source_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    work_lookup: np.ndarray,
    config: Mapping[str, Any],
    scratch: Path,
) -> tuple[dict[int, dict[str, int]], dict[str, Any], list[Path], list[Path]]:
    source_cfg = _mapping(config["source"], "source config")
    orcid_cfg = _mapping(config["orcid_contract"], "ORCID config")
    cells: dict[int, dict[str, int]] = {}
    totals: dict[str, int] = defaultdict(int)
    author_spool = PartitionedSpool(scratch / "author_orcid", "author-orcid", AUTHOR_ORCID_DTYPE)
    institution_spool = PartitionedSpool(
        scratch / "institution", "institution", INSTITUTION_DTYPE
    )
    started = time.monotonic()
    try:
        for file_index, task in enumerate(tasks):
            relative = str(task["relative_path"])
            path = source_root / relative
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            require(all(column in schema.names for column in SOURCE_COLUMNS), f"source schema missing column: {relative}")
            projected_rows = 0
            for batch in parquet.iter_batches(
                batch_size=int(config["execution"]["source_batch_rows"]),
                columns=SOURCE_COLUMNS,
            ):
                projected_rows += len(batch)
                work_values = batch.column(0)
                raw_affiliations = batch.column(1)
                institutions = batch.column(2)
                author_values = batch.column(3)
                orcid_values = batch.column(4)
                work_ids, valid_work = _parse_openalex_ids(
                    work_values,
                    pattern=source_cfg["work_id_pattern"],
                    prefix=WORK_PREFIX,
                )
                w_minus_1 = (
                    pc.fill_null(pc.equal(work_values, source_cfg["exact_w_minus_1"]), False)
                    .to_numpy(zero_copy_only=False)
                    .astype(bool, copy=False)
                )
                require(bool(np.all(valid_work | w_minus_1)), "unexpected malformed work ID")
                encoded = np.zeros(len(batch), dtype=np.uint32)
                in_range = valid_work & (work_ids < len(work_lookup))
                encoded[in_range] = work_lookup[work_ids[in_range]]
                unmatched = valid_work & (encoded == 0)
                totals["unmatched_valid_work_rows"] += int(unmatched.sum())
                field_ids = (encoded >> 16).astype(np.int64) - 1
                years = encoded & np.uint32(0xFFFF)
                classified = (field_ids >= FIELD_MIN) & (field_ids <= FIELD_MAX)
                cell_keys = field_ids[classified].astype(np.uint64) * YEAR_MULTIPLIER + years[classified]

                author_ids, valid_author = _parse_openalex_ids(
                    author_values,
                    pattern=source_cfg["author_id_pattern"],
                    prefix=AUTHOR_PREFIX,
                )
                orcid_code, orcid_syntax, orcid_valid = _parse_orcid(
                    orcid_values, orcid_cfg["accepted_form"]
                )
                orcid_nonnull = ~orcid_values.is_null().to_numpy(zero_copy_only=False)
                raw_affiliation_present = _nonempty_list_row_mask(raw_affiliations)
                (
                    institution_present,
                    institution_parents,
                    institution_ids,
                ) = _institution_projection(
                    institutions, source_cfg["institution_id_pattern"]
                )
                affiliation_present = raw_affiliation_present | institution_present

                _add_grouped(cells, cell_keys, "raw_authorship_rows")
                totals["classified_field_year_raw_authorship_rows"] += int(
                    classified.sum()
                )
                valid_author_classified = valid_author & classified
                _add_grouped(
                    cells,
                    (
                        field_ids[valid_author_classified].astype(np.uint64)
                        * YEAR_MULTIPLIER
                    )
                    + years[valid_author_classified],
                    "grammar_valid_author_id_rows",
                )
                totals[
                    "classified_field_year_grammar_valid_author_id_rows"
                ] += int(valid_author_classified.sum())
                for mask, metric in (
                    (orcid_nonnull, "nonnull_orcid_rows"),
                    (orcid_syntax, "syntax_valid_orcid_rows"),
                    (orcid_valid, "checksum_valid_orcid_rows"),
                    (valid_author & affiliation_present, "author_work_affiliation_rows"),
                    (valid_author & institution_present, "author_work_institution_rows"),
                ):
                    selected = mask & classified
                    totals[f"classified_field_year_{metric}"] += int(
                        selected.sum()
                    )
                    _add_grouped(
                        cells,
                        field_ids[selected].astype(np.uint64) * YEAR_MULTIPLIER
                        + years[selected],
                        metric,
                    )

                author_orcid_mask = valid_author & orcid_valid
                author_records = np.empty(int(author_orcid_mask.sum()), dtype=AUTHOR_ORCID_DTYPE)
                author_records["author_numeric_id"] = author_ids[author_orcid_mask]
                author_records["orcid_code"] = orcid_code[author_orcid_mask]
                author_spool.append(
                    author_records,
                    (author_records["author_numeric_id"] % PARTITION_COUNT).astype(np.uint16),
                )

                valid_institution_parent = (
                    valid_author[institution_parents]
                    & classified[institution_parents]
                )
                parent_rows = institution_parents[valid_institution_parent]
                projected_institutions = institution_ids[valid_institution_parent]
                institution_records = np.empty(
                    len(parent_rows), dtype=INSTITUTION_DTYPE
                )
                institution_records["field_year_key"] = (
                    field_ids[parent_rows].astype(np.uint64) * YEAR_MULTIPLIER
                    + years[parent_rows]
                ).astype(np.uint32)
                institution_records["institution_numeric_id"] = projected_institutions
                institution_spool.append(
                    institution_records,
                    (institution_records["field_year_key"] % PARTITION_COUNT).astype(np.uint16),
                )

                totals["raw_rows"] += len(batch)
                totals["w_minus_1_rows"] += int(w_minus_1.sum())
                totals["retained_rows"] += int(valid_work.sum())
                totals["nonnull_author_id_rows"] += int((~author_values.is_null().to_numpy(zero_copy_only=False)).sum())
                totals["grammar_valid_author_id_rows"] += int(valid_author.sum())
                totals["null_author_id_rows"] += int(author_values.null_count)
                totals["nonnull_orcid_rows"] += int(orcid_nonnull.sum())
                totals["syntax_valid_orcid_rows"] += int(orcid_syntax.sum())
                totals["checksum_valid_orcid_rows"] += int(orcid_valid.sum())
                totals["raw_affiliation_present_rows"] += int(raw_affiliation_present.sum())
                totals["institution_present_rows"] += int(institution_present.sum())
                totals["author_work_affiliation_rows"] += int((valid_author & affiliation_present).sum())
                totals["author_work_institution_rows"] += int((valid_author & institution_present).sum())
                totals["grammar_valid_institution_occurrences"] += len(institution_ids)
            require(projected_rows == parquet.metadata.num_rows, f"source file row closure failed: {relative}")
            if (file_index + 1) % 25 == 0 or file_index + 1 == len(tasks):
                elapsed = time.monotonic() - started
                print(
                    f"S0_AUTHORSHIP_SCAN files={file_index + 1}/{len(tasks)} "
                    f"rows={totals['raw_rows']} elapsed_seconds={elapsed:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        author_spool.close()
        institution_spool.close()
    totals["elapsed_seconds_milli"] = int((time.monotonic() - started) * 1000)
    totals["author_orcid_spool_rows"] = author_spool.rows
    totals["institution_spool_rows"] = institution_spool.rows
    return cells, dict(totals), author_spool.paths, institution_spool.paths


def _reduce_author_orcid(
    paths: Sequence[Path],
    scratch: Path,
) -> tuple[list[Path], dict[str, int]]:
    stable_root = scratch / "stable_author"
    stable_root.mkdir()
    stable_paths = [stable_root / f"stable-author-{index:03d}.bin" for index in range(PARTITION_COUNT)]
    orcid_spool = PartitionedSpool(scratch / "stable_orcid", "stable-orcid", AUTHOR_ORCID_DTYPE)
    totals: dict[str, int] = defaultdict(int)
    try:
        for bucket, path in enumerate(paths):
            raw = np.fromfile(path, dtype=AUTHOR_ORCID_DTYPE)
            if len(raw) == 0:
                stable_paths[bucket].write_bytes(b"")
                os.chmod(stable_paths[bucket], 0o600)
                continue
            order = np.lexsort((raw["orcid_code"], raw["author_numeric_id"]))
            ordered = raw[order]
            keep = np.ones(len(ordered), dtype=bool)
            keep[1:] = (
                (ordered["author_numeric_id"][1:] != ordered["author_numeric_id"][:-1])
                | (ordered["orcid_code"][1:] != ordered["orcid_code"][:-1])
            )
            pairs = ordered[keep]
            authors, first, counts = np.unique(
                pairs["author_numeric_id"], return_index=True, return_counts=True
            )
            stable_mask = counts == 1
            stable = np.empty(int(stable_mask.sum()), dtype=STABLE_AUTHOR_DTYPE)
            stable["author_numeric_id"] = authors[stable_mask]
            stable["orcid_code"] = pairs["orcid_code"][first[stable_mask]]
            stable.tofile(stable_paths[bucket])
            os.chmod(stable_paths[bucket], 0o600)
            orcid_spool.append(
                stable,
                (stable["orcid_code"] % PARTITION_COUNT).astype(np.uint16),
            )
            totals["distinct_author_orcid_pairs"] += len(pairs)
            totals["author_tokens_with_valid_orcid"] += len(authors)
            totals["stable_single_valid_orcid_author_tokens"] += len(stable)
            totals["multiple_valid_orcid_author_tokens"] += int((counts > 1).sum())
    finally:
        orcid_spool.close()

    for path in orcid_spool.paths:
        raw = np.fromfile(path, dtype=AUTHOR_ORCID_DTYPE)
        if len(raw) == 0:
            continue
        order = np.lexsort((raw["author_numeric_id"], raw["orcid_code"]))
        pairs = raw[order]
        keep = np.ones(len(pairs), dtype=bool)
        keep[1:] = (
            (pairs["orcid_code"][1:] != pairs["orcid_code"][:-1])
            | (pairs["author_numeric_id"][1:] != pairs["author_numeric_id"][:-1])
        )
        pairs = pairs[keep]
        codes, counts = np.unique(pairs["orcid_code"], return_counts=True)
        totals["distinct_valid_orcids_on_stable_tokens"] += len(codes)
        totals["shared_valid_orcids_across_stable_tokens"] += int((counts > 1).sum())
        totals["stable_author_tokens_on_shared_valid_orcids"] += int(
            counts[counts > 1].sum()
        )
        totals["maximum_stable_author_tokens_per_valid_orcid"] = max(
            totals["maximum_stable_author_tokens_per_valid_orcid"],
            int(counts.max()),
        )
    return stable_paths, dict(totals)


def _reduce_institutions(
    paths: Sequence[Path],
    cells: dict[int, dict[str, int]],
) -> dict[str, int]:
    all_institutions: set[int] = set()
    field_year_pairs = 0
    for path in paths:
        raw = np.fromfile(path, dtype=INSTITUTION_DTYPE)
        if len(raw) == 0:
            continue
        order = np.lexsort((raw["institution_numeric_id"], raw["field_year_key"]))
        pairs = raw[order]
        keep = np.ones(len(pairs), dtype=bool)
        keep[1:] = (
            (pairs["field_year_key"][1:] != pairs["field_year_key"][:-1])
            | (pairs["institution_numeric_id"][1:] != pairs["institution_numeric_id"][:-1])
        )
        pairs = pairs[keep]
        keys, counts = np.unique(pairs["field_year_key"], return_counts=True)
        for key, count in zip(keys.tolist(), counts.tolist(), strict=True):
            cell = cells.setdefault(int(key), {})
            cell["distinct_institutions"] = cell.get("distinct_institutions", 0) + int(count)
        all_institutions.update(
            int(value) for value in np.unique(pairs["institution_numeric_id"]).tolist()
        )
        field_year_pairs += len(pairs)
    return {
        "distinct_institution_tokens": len(all_institutions),
        "distinct_institution_field_year_pairs": field_year_pairs,
    }


def _distinct_author_cell_keys(
    authors: np.ndarray,
    cell_keys: np.ndarray,
) -> np.ndarray:
    if len(authors) == 0:
        return np.empty(0, dtype=np.uint64)
    order = np.lexsort((cell_keys, authors))
    authors = authors[order]
    cell_keys = cell_keys[order]
    keep = np.ones(len(authors), dtype=bool)
    keep[1:] = (authors[1:] != authors[:-1]) | (cell_keys[1:] != cell_keys[:-1])
    return cell_keys[keep]


def _scan_stable_events(
    author_root: Path,
    receipt: Mapping[str, Any],
    binding: Mapping[str, Any],
    stable_paths: Sequence[Path],
    cells: dict[int, dict[str, int]],
    *,
    maximum_event_files: int | None,
) -> dict[str, Any]:
    artifacts = _receipt_artifacts(
        author_root, receipt, "events/", int(binding["expected_event_artifacts"])
    )
    if maximum_event_files is not None:
        artifacts = artifacts[:maximum_event_files]
    totals: dict[str, int] = defaultdict(int)
    digest_rows: list[tuple[str, int, str]] = []
    for bucket in range(PARTITION_COUNT):
        stable = np.fromfile(stable_paths[bucket], dtype=STABLE_AUTHOR_DTYPE)
        stable_authors = stable["author_numeric_id"]
        for index in range(bucket, len(artifacts), PARTITION_COUNT):
            path, item = artifacts[index]
            observed_sha = _sha256_file(path)
            require(observed_sha == item["sha256"], f"event artifact SHA mismatch: {index}")
            table = pq.read_table(
                path,
                columns=[
                    "author_numeric_id",
                    "publication_year",
                    "primary_field_id",
                ],
            )
            authors = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
            require(bool(np.all(authors % WORK_PARTITIONS == index)), "event partition mismatch")
            if len(stable_authors):
                positions = np.searchsorted(stable_authors, authors)
                matched = positions < len(stable_authors)
                matched[matched] &= stable_authors[positions[matched]] == authors[matched]
            else:
                matched = np.zeros(len(authors), dtype=bool)
            fields = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)
            years = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)
            classified = (fields >= FIELD_MIN) & (fields <= FIELD_MAX)
            selected = matched & classified
            keys = fields[selected].astype(np.uint64) * YEAR_MULTIPLIER + years[selected]
            _add_grouped(cells, keys, "stable_orcid_author_work_pairs")
            distinct_keys = _distinct_author_cell_keys(authors[selected], keys)
            _add_grouped(cells, distinct_keys, "stable_orcid_author_tokens")
            totals["event_rows_scanned"] += len(table)
            totals["stable_orcid_author_work_pairs"] += int(selected.sum())
            totals["stable_orcid_author_tokens_summed_within_field_year"] += len(
                distinct_keys
            )
            digest_rows.append((item["relative_path"], int(item["bytes"]), observed_sha))
    if maximum_event_files is None:
        require(
            totals["event_rows_scanned"] == binding["expected_event_rows"],
            "full event row closure failed",
        )
    totals["event_artifacts_rehashed"] = len(artifacts)
    totals["event_artifact_tree_sha256"] = hashlib.sha256(
        _canonical_bytes(digest_rows)
    ).hexdigest()
    return dict(totals)


def _field_year_table(cells: Mapping[int, Mapping[str, int]]) -> pa.Table:
    rows = []
    for key in sorted(cells):
        field_id, year = divmod(int(key), YEAR_MULTIPLIER)
        require(FIELD_MIN <= field_id <= FIELD_MAX, "invalid field-year key")
        row: dict[str, Any] = {
            "field_id": field_id,
            "publication_year": None if year == 0 else year,
        }
        for metric in FIELD_YEAR_METRICS:
            row[metric] = int(cells[key].get(metric, 0))
        require(row["checksum_valid_orcid_rows"] <= row["syntax_valid_orcid_rows"] <= row["nonnull_orcid_rows"] <= row["raw_authorship_rows"], "ORCID cell monotonicity failed")
        require(row["author_work_institution_rows"] <= row["author_work_affiliation_rows"] <= row["grammar_valid_author_id_rows"], "affiliation cell monotonicity failed")
        rows.append(row)
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=True),
            *[
                pa.field(metric, pa.uint64(), nullable=False)
                for metric in FIELD_YEAR_METRICS
            ],
        ],
        metadata={b"schema_version": FIELD_YEAR_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.chmod(path, 0o600)
    _fsync_file(path)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_parquet(path: Path, table: pa.Table) -> dict[str, Any]:
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
    _fsync_file(path)
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "rows": len(table),
    }


def _available_memory_bytes(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProjectionError(f"cannot read memory availability: {error}") from error
    for line in lines:
        if line.startswith("MemAvailable:"):
            parts = line.split()
            require(
                len(parts) == 3
                and parts[1].isdigit()
                and parts[2] == "kB",
                "invalid MemAvailable record",
            )
            return int(parts[1]) * 1024
    raise ProjectionError("MemAvailable is missing")


def _resource_preflight(
    source_root: Path,
    output_parent: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    resource = _mapping(config["resource_envelope"], "resource envelope")
    memory = _available_memory_bytes()
    disk = shutil.disk_usage(output_parent).free
    source_device = source_root.stat().st_dev
    output_device = output_parent.stat().st_dev
    require(memory >= resource["minimum_available_memory_bytes"], "insufficient available memory")
    require(disk >= resource["minimum_free_output_bytes"], "insufficient output disk")
    if resource["source_and_output_distinct_devices_required"]:
        require(source_device != output_device, "source and output share a device")
    return {
        "available_memory_bytes": memory,
        "free_output_bytes": disk,
        "source_device": source_device,
        "output_device": output_device,
        "distinct_devices": source_device != output_device,
        "gpu_used": False,
    }


def build(
    *,
    config_path: Path,
    source_root: Path,
    content_root: Path,
    author_root: Path,
    cutoff_root: Path,
    output: Path,
    maximum_source_files: int | None,
    maximum_event_files: int | None,
) -> dict[str, Any]:
    started = time.monotonic()
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, "projection config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(config.get("label") == "pivot_feedback_s0_authorship_orcid_affiliation_v1", "config label mismatch")
    require(config.get("status_before_execution") == "AUTHORIZED_S0_REMAINING_COMPONENT", "projection is not authorized")
    gate_parent = _validate_repository_binding(
        _mapping(config["gate_parent"], "gate parent"), yaml_document=True
    )
    core_parent = _validate_repository_binding(
        _mapping(config["s0_core_parent"], "S0 core parent"), yaml_document=False
    )
    source_root = source_root.resolve(strict=True)
    content_root = content_root.resolve(strict=True)
    author_root = author_root.resolve(strict=True)
    cutoff_root = cutoff_root.resolve(strict=True)
    source_cfg = _mapping(config["source"], "source")
    require(source_root.name == source_cfg["expected_root_basename"], "source root basename mismatch")
    output = output.resolve()
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent does not exist")
    for root in (source_root, content_root, author_root, cutoff_root):
        require(not output.is_relative_to(root), "output may not be inside an input root")
    preflight = _resource_preflight(source_root, output.parent, config)

    content_binding = _mapping(config["content_freeze_parent"], "content parent")
    content_parent = _validate_content_root(content_root, content_binding)
    author_binding = _mapping(config["author_history_parent"], "author parent")
    author_receipt, author_plan = _validate_receipt_root(
        author_root, author_binding, plan_required=True
    )
    assert author_plan is not None
    tasks = [
        _mapping(item, "authorship task")
        for item in _list(author_plan.get("selected_authorship_tasks"), "authorship tasks")
    ]
    require(len(tasks) == author_binding["expected_authorship_tasks"], "authorship task count mismatch")
    require(len(tasks) == source_cfg["expected_files"], "source file count mismatch")
    require(sum(int(item["bytes"]) for item in tasks) == source_cfg["expected_bytes"], "authorship task byte closure failed")
    tasks.sort(key=lambda item: str(item["relative_path"]))
    full_scope = maximum_source_files is None and maximum_event_files is None
    if maximum_source_files is not None:
        require(1 <= maximum_source_files <= len(tasks), "invalid source-file pilot limit")
        tasks = tasks[:maximum_source_files]
    source_binding = _validate_source_tasks_and_checkpoints(
        source_root,
        content_root,
        tasks,
        content_binding,
        table_name=source_cfg["table"],
    )
    cutoff_binding = _mapping(config["cutoff_parent"], "cutoff parent")
    cutoff_receipt, _ = _validate_receipt_root(cutoff_root, cutoff_binding)
    work_lookup, work_binding = _load_work_cell_lookup(
        cutoff_root, cutoff_receipt, cutoff_binding
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    scratch = temporary / "scratch"
    scratch.mkdir(mode=0o700)
    try:
        cells, source_counts, author_spools, institution_spools = _scan_authorship_source(
            source_root, tasks, work_lookup, config, scratch
        )
        del work_lookup
        if full_scope:
            require(source_counts["raw_rows"] == source_cfg["expected_raw_rows"], "source raw-row closure failed")
            require(source_counts["w_minus_1_rows"] == source_cfg["expected_w_minus_1_rows"], "W-1 closure failed")
            require(source_counts["retained_rows"] == source_cfg["expected_retained_rows"], "retained row closure failed")
            require(source_counts["nonnull_author_id_rows"] == source_cfg["expected_nonnull_author_id_rows"], "nonnull author footer closure failed")
            require(source_counts["null_author_id_rows"] == source_cfg["expected_null_author_id_rows"], "null author closure failed")
            require(source_counts["nonnull_orcid_rows"] == source_cfg["expected_nonnull_orcid_rows"], "nonnull ORCID footer closure failed")
            require(source_counts["unmatched_valid_work_rows"] == 0, "valid authorship work failed lookup")
        stable_paths, orcid_diagnostics = _reduce_author_orcid(
            author_spools, scratch
        )
        institution_diagnostics = _reduce_institutions(institution_spools, cells)
        event_diagnostics = _scan_stable_events(
            author_root,
            author_receipt,
            author_binding,
            stable_paths,
            cells,
            maximum_event_files=maximum_event_files,
        )
        field_year = _field_year_table(cells)
        field_year_artifact = _write_parquet(
            temporary / "field_year_orcid_affiliation.parquet", field_year
        )
        sums = {
            metric: int(sum(field_year[metric].to_pylist()))
            for metric in FIELD_YEAR_METRICS
        }
        for metric in FIELD_YEAR_METRICS[:7]:
            require(
                sums[metric]
                == source_counts[f"classified_field_year_{metric}"],
                f"field-year source margin mismatch: {metric}",
            )
        require(
            sums["stable_orcid_author_work_pairs"]
            == event_diagnostics["stable_orcid_author_work_pairs"],
            "stable ORCID event margin mismatch",
        )
        require(
            sums["stable_orcid_author_tokens"]
            == event_diagnostics[
                "stable_orcid_author_tokens_summed_within_field_year"
            ],
            "stable ORCID token margin mismatch",
        )
        require(
            sums["distinct_institutions"]
            == institution_diagnostics["distinct_institution_field_year_pairs"],
            "institution field-year margin mismatch",
        )
        status = FULL_STATUS if full_scope else PILOT_STATUS
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "status": status,
            "scope": {
                "full_scope": full_scope,
                "source_files": len(tasks),
                "event_files": event_diagnostics["event_artifacts_rehashed"],
                "component_pass": full_scope,
                "pilot_promotion_allowed": False,
                "s0_gate_pass": False,
                "d0_authorized": False,
                "model_fitting_authorized": False,
                "gpu_used": False,
            },
            "gate_parent": gate_parent,
            "s0_core_parent": core_parent,
            "preflight": preflight,
            "source_binding": {**content_parent, **source_binding},
            "work_lookup_binding": work_binding,
            "source_accounting": source_counts,
            "orcid_token_diagnostics": orcid_diagnostics,
            "institution_diagnostics": institution_diagnostics,
            "stable_event_diagnostics": event_diagnostics,
            "field_year_inventory": {
                "schema_version": FIELD_YEAR_SCHEMA,
                "rows": len(field_year),
                "metric_sums": sums,
                "artifact": field_year_artifact,
            },
            "remaining_s0_component": "DISTINCT_COAUTHOR_GRAPH_FIELD_YEAR",
            "gate_status": {
                "g_minus_1a": "PASS_BOUNDED",
                "s0_threshold_free_inventory": (
                    "PARTIAL_ORCID_AFFILIATION_PASS_COAUTHOR_GRAPH_NOT_RUN"
                    if full_scope
                    else "PARTIAL_CORE_PASS_PILOT_ONLY"
                ),
                "d0_discovery_pilot": "NOT_AUTHORIZED",
                "g0_measurement_freeze": "PENDING",
                "empirical_g1_signature": "NOT_RUN",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "provenance": {
                "config_relative_path": str(config_path.relative_to(REPO_ROOT)),
                "config_sha256": _sha256_file(config_path),
                "runner_relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "runner_sha256": _sha256_file(Path(__file__).resolve()),
            },
            "execution": {
                "elapsed_seconds": time.monotonic() - started,
                "scratch_removed_before_completion": True,
            },
            "claim_boundary": config["claim_boundary"],
        }
        report["provenance"]["result_fingerprint_sha256"] = hashlib.sha256(
            _canonical_bytes(report)
        ).hexdigest()
        shutil.rmtree(scratch)
        _write_json(temporary / "authorship_orcid_affiliation_report.json", report)
        markdown = "\n".join(
            [
                "# S0 authorship ORCID and affiliation projection",
                "",
                f"Status: `{status}`",
                "",
                f"- source rows: {source_counts['raw_rows']:,}",
                f"- checksum-valid ORCID rows: {source_counts['checksum_valid_orcid_rows']:,}",
                f"- stable single-ORCID author tokens: {orcid_diagnostics['stable_single_valid_orcid_author_tokens']:,}",
                f"- distinct institution tokens: {institution_diagnostics['distinct_institution_tokens']:,}",
                f"- field-year rows: {len(field_year):,}",
                "",
                "This component does not establish natural-person identity or",
                "historical affiliation observability. The distinct coauthor graph",
                "component remains not run, so S0 and D0 remain closed.",
                "",
            ]
        )
        md_path = temporary / "authorship_orcid_affiliation_report.md"
        md_path.write_text(markdown, encoding="utf-8")
        os.chmod(md_path, 0o600)
        _fsync_file(md_path)
        artifacts = []
        for path in sorted(temporary.iterdir()):
            artifacts.append(
                {
                    "relative_path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        receipt = {
            "schema_version": "pivot_feedback_s0_authorship_orcid_affiliation_outputs.v1",
            "status": status,
            "full_scope": full_scope,
            "component_pass": full_scope,
            "s0_gate_pass": False,
            "d0_authorized": False,
            "completion_contract": "SCRATCH_REMOVED_RECEIPT_WRITTEN_LAST_ATOMIC_DIRECTORY_RENAME",
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": hashlib.sha256(_canonical_bytes(artifacts)).hexdigest(),
            "artifacts": artifacts,
            "result_fingerprint_sha256": report["provenance"][
                "result_fingerprint_sha256"
            ],
        }
        _write_json(temporary / "authorship_orcid_affiliation_report.outputs.json", receipt)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/configs/pivot_feedback_s0_authorship_orcid_affiliation_v1.yaml",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-source-files", type=int)
    parser.add_argument("--maximum-event-files", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build(
            config_path=args.config,
            source_root=args.source_root,
            content_root=args.content_root,
            author_root=args.author_root,
            cutoff_root=args.cutoff_root,
            output=args.output,
            maximum_source_files=args.maximum_source_files,
            maximum_event_files=args.maximum_event_files,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    except (ProjectionError, OSError, ValueError, pa.ArrowException) as error:
        print(json.dumps({"status": "BLOCKED_S0_AUTHORSHIP_ORCID_AFFILIATION", "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
