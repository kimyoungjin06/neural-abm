#!/usr/bin/env python3
"""Audit an OpenAlex derived parquet root without scanning column bodies.

This command is the first G-1 gate for the prospective researcher-pivot study.
It verifies physical inventory, parquet footers, schema consistency, and
historical footer-row expectations. It does not establish distinct IDs, join
coverage, author validity, temporal leakage safety, or empirical readiness.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/openalex_g1_source_inventory_v0.yaml"
TOOL_SCHEMA_VERSION = "openalex_source_inventory.v1"
CHECK_STATUSES = {"PASS", "WARN", "BLOCK", "PENDING"}


@dataclass(frozen=True)
class FileSnapshot:
    relative_path: str
    bytes: int
    mtime_ns: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("source inventory config must be a mapping")
    if raw.get("schema_version") != "openalex_g1_source_inventory_config.v1":
        raise ValueError("unsupported source inventory config schema_version")
    if not isinstance(raw.get("tables"), dict) or not raw["tables"]:
        raise ValueError("source inventory config must declare tables")
    return raw


def resolve_root(config: dict[str, Any], explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        root = explicit_root
    else:
        env_name = str(config["root"]["env"])
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(f"set {env_name} or pass --root")
        root = Path(value)
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"OpenAlex root is not a directory: {resolved}")
    return resolved


def git_metadata() -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status_error": status.stderr.strip() if status.returncode != 0 else None,
    }


def check(
    name: str,
    status: str,
    *,
    observed: Any = None,
    expected: Any = None,
    note: str | None = None,
) -> dict[str, Any]:
    if status not in CHECK_STATUSES:
        raise ValueError(f"invalid check status: {status}")
    result: dict[str, Any] = {"name": name, "status": status}
    if observed is not None:
        result["observed"] = observed
    if expected is not None:
        result["expected"] = expected
    if note is not None:
        result["note"] = note
    return result


def snapshot_files(root: Path, table_names: Iterable[str]) -> list[FileSnapshot]:
    rows: list[FileSnapshot] = []
    for table_name in sorted(table_names):
        table_dir = root / table_name
        if not table_dir.is_dir():
            continue
        for path in sorted(table_dir.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file() or path.suffix != ".parquet":
                continue
            stat = path.stat()
            rows.append(
                FileSnapshot(
                    relative_path=path.relative_to(root).as_posix(),
                    bytes=int(stat.st_size),
                    mtime_ns=int(stat.st_mtime_ns),
                )
            )
    return rows


def snapshot_fingerprint(rows: Iterable[FileSnapshot], *, include_mtime: bool) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.relative_path):
        parts = [row.relative_path, str(row.bytes)]
        if include_mtime:
            parts.append(str(row.mtime_ns))
        digest.update("\0".join(parts).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def physical_schema_fingerprint(parquet: pq.ParquetFile) -> str:
    schema = parquet.schema_arrow.remove_metadata()
    return sha256_bytes(schema.serialize().to_pybytes())


def canonical_schema_fingerprint(parquet: pq.ParquetFile) -> str:
    fields = sorted(
        (field.name, str(field.type), bool(field.nullable))
        for field in parquet.schema_arrow.remove_metadata()
    )
    return sha256_bytes(json.dumps(fields, separators=(",", ":")).encode("utf-8"))


def schema_fields(parquet: pq.ParquetFile) -> dict[str, dict[str, Any]]:
    return {
        field.name: {"type": str(field.type), "nullable": bool(field.nullable)}
        for field in parquet.schema_arrow.remove_metadata()
    }


def footer_null_statistics(
    metadata: pq.FileMetaData,
    columns: Iterable[str],
) -> dict[str, dict[str, int | None]]:
    requested = sorted(set(columns))
    results: dict[str, dict[str, int | None]] = {
        column: {
            "null_count": 0,
            "row_groups_with_statistics": 0,
            "row_groups_total": int(metadata.num_row_groups),
        }
        for column in requested
    }
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        indices = {
            row_group.column(index).path_in_schema: index
            for index in range(row_group.num_columns)
        }
        for column in requested:
            index = indices.get(column)
            if index is None:
                results[column]["null_count"] = None
                continue
            statistics = row_group.column(index).statistics
            if statistics is None or not statistics.has_null_count:
                results[column]["null_count"] = None
                continue
            results[column]["row_groups_with_statistics"] = (
                int(results[column]["row_groups_with_statistics"] or 0) + 1
            )
            if results[column]["null_count"] is not None:
                previous = int(results[column]["null_count"] or 0)
                results[column]["null_count"] = previous + int(statistics.null_count)
    return results


def footer_record(
    path: Path,
    *,
    root: Path,
    content_sha256: bool,
    footer_stat_columns: Iterable[str],
) -> dict[str, Any]:
    stat = path.stat()
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    compressions: set[str] = set()
    if metadata.num_row_groups:
        row_group = metadata.row_group(0)
        for column_index in range(row_group.num_columns):
            compressions.add(str(row_group.column(column_index).compression))
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "footer_rows": int(metadata.num_rows),
        "row_groups": int(metadata.num_row_groups),
        "footer_metadata_bytes": int(metadata.serialized_size),
        "columns": list(parquet.schema_arrow.names),
        "schema_fields": schema_fields(parquet),
        "canonical_schema_fingerprint_sha256": canonical_schema_fingerprint(parquet),
        "physical_schema_fingerprint_sha256": physical_schema_fingerprint(parquet),
        "footer_null_statistics": footer_null_statistics(metadata, footer_stat_columns),
        "compression": sorted(compressions),
        "format_version": str(metadata.format_version),
        "created_by": metadata.created_by,
        "content_sha256": sha256_file(path) if content_sha256 else None,
    }


def safe_footer_record(
    path: Path,
    *,
    root: Path,
    content_sha256: bool,
    footer_stat_columns: Iterable[str],
) -> dict[str, Any]:
    try:
        return footer_record(
            path,
            root=root,
            content_sha256=content_sha256,
            footer_stat_columns=footer_stat_columns,
        )
    except Exception as exc:
        try:
            stat = path.stat()
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
        except OSError:
            size = None
            mtime_ns = None
        return {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": size,
            "mtime_ns": mtime_ns,
            "error": f"{type(exc).__name__}: {exc}",
            "content_sha256": None,
        }


def inventory_record_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["relative_path"])):
        values = (
            record.get("relative_path"),
            record.get("bytes"),
            record.get("footer_rows"),
            record.get("canonical_schema_fingerprint_sha256"),
            record.get("physical_schema_fingerprint_sha256"),
            record.get("content_sha256"),
            record.get("error"),
        )
        digest.update(json.dumps(values, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_table(
    root: Path,
    table_name: str,
    spec: dict[str, Any],
    *,
    filename_pattern: re.Pattern[str],
    workers: int,
    content_sha256: bool,
    content_sha256_required: bool,
    reject_unexpected_entries: bool,
) -> dict[str, Any]:
    table_dir = root / table_name
    checks: list[dict[str, Any]] = []
    if not table_dir.is_dir():
        return {
            "name": table_name,
            "role": spec.get("role"),
            "checks": [check("table_directory_exists", "BLOCK", observed=False)],
            "files": [],
        }

    entries = sorted(table_dir.iterdir(), key=lambda item: item.name)
    parquet_files = [
        path
        for path in entries
        if not path.is_symlink() and path.is_file() and path.suffix == ".parquet"
    ]
    unexpected_entries = [
        path.name
        for path in entries
        if path.is_symlink() or not (path.is_file() and path.suffix == ".parquet")
    ]
    checks.append(
        check(
            "table_directory_exists",
            "BLOCK" if table_dir.is_symlink() else "PASS",
            observed=True,
            expected="regular directory, not a symlink",
        )
    )
    checks.append(
        check(
            "unexpected_table_entries",
            "BLOCK" if unexpected_entries and reject_unexpected_entries else "PASS",
            observed=unexpected_entries,
            expected=[],
        )
    )

    expected_file_count = int(spec["expected_file_count"])
    checks.append(
        check(
            "parquet_file_count",
            "PASS" if len(parquet_files) == expected_file_count else "BLOCK",
            observed=len(parquet_files),
            expected=expected_file_count,
        )
    )
    bad_names = [
        path.name for path in parquet_files if not filename_pattern.fullmatch(path.name)
    ]
    checks.append(
        check(
            "parquet_filename_pattern",
            "PASS" if not bad_names else "BLOCK",
            observed=bad_names,
            expected=filename_pattern.pattern,
        )
    )
    prefix_numbers: dict[str, list[int]] = {}
    noncanonical_index_names: list[str] = []
    for path in parquet_files:
        match = filename_pattern.fullmatch(path.name)
        if match is not None:
            prefix = match.group("prefix")
            number = int(match.group("index"))
            prefix_numbers.setdefault(prefix, []).append(number)
            if path.name != f"{prefix}-{number}.parquet":
                noncanonical_index_names.append(path.name)
    default_prefixes = (
        ["part"] if spec.get("role") == "main_works" else ["base", "delta", "repair"]
    )
    allowed_prefixes = set(
        str(value) for value in spec.get("allowed_prefixes", default_prefixes)
    )
    unexpected_prefixes = sorted(set(prefix_numbers) - allowed_prefixes)
    checks.append(
        check(
            "allowed_filename_prefixes",
            "PASS" if not unexpected_prefixes else "BLOCK",
            observed=sorted(prefix_numbers),
            expected=sorted(allowed_prefixes),
        )
    )
    duplicate_indices = {
        prefix: sorted(number for number in set(numbers) if numbers.count(number) > 1)
        for prefix, numbers in prefix_numbers.items()
        if len(numbers) != len(set(numbers))
    }
    checks.append(
        check(
            "unique_canonical_filename_indices",
            (
                "PASS"
                if not duplicate_indices and not noncanonical_index_names
                else "BLOCK"
            ),
            observed={
                "duplicate_indices": duplicate_indices,
                "noncanonical_names": sorted(noncanonical_index_names),
            },
            expected={"duplicate_indices": {}, "noncanonical_names": []},
        )
    )
    sequence_gaps: dict[str, list[int]] = {}
    for prefix, numbers in prefix_numbers.items():
        observed_numbers = set(numbers)
        missing = sorted(set(range(min(numbers), max(numbers) + 1)) - observed_numbers)
        if missing:
            sequence_gaps[prefix] = missing
    observed_prefix_contract = {
        prefix: {
            "count": len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "gaps": sequence_gaps.get(prefix, []),
        }
        for prefix, numbers in sorted(prefix_numbers.items())
    }
    expected_prefix_contract = {
        str(prefix): {
            "count": int(values["count"]),
            "min": int(values["min"]),
            "max": int(values["max"]),
            "gaps": sorted(int(value) for value in values.get("gaps", [])),
        }
        for prefix, values in dict(spec.get("expected_prefix_contract", {})).items()
    }
    checks.append(
        check(
            "filename_prefix_contract",
            (
                "PASS"
                if not expected_prefix_contract
                or observed_prefix_contract == expected_prefix_contract
                else "BLOCK"
            ),
            observed=observed_prefix_contract,
            expected=(expected_prefix_contract or "capture-only"),
        )
    )
    expected_sequence_gaps = {
        str(prefix): sorted(int(value) for value in values)
        for prefix, values in dict(spec.get("expected_sequence_gaps", {})).items()
    }
    checks.append(
        check(
            "filename_sequence_gaps",
            "PASS" if sequence_gaps == expected_sequence_gaps else "BLOCK",
            observed=sequence_gaps,
            expected=expected_sequence_gaps,
            note=str(
                spec.get(
                    "lineage_evidence_id",
                    "unexpected gaps require a frozen lineage replacement declaration",
                )
            ),
        )
    )
    lineage_status = spec.get("lineage_status")
    if lineage_status is not None:
        checks.append(
            check(
                "lineage_provenance",
                str(lineage_status),
                observed=spec.get("lineage_evidence"),
                expected="frozen replacement declaration with cryptographic provenance",
                note=spec.get("lineage_note"),
            )
        )
    required_lineage_files = sorted(
        str(value) for value in spec.get("required_lineage_files", [])
    )
    missing_lineage_files = sorted(
        set(required_lineage_files) - {path.name for path in parquet_files}
    )
    checks.append(
        check(
            "required_lineage_files",
            "PASS" if not missing_lineage_files else "BLOCK",
            observed=missing_lineage_files,
            expected=required_lineage_files,
        )
    )
    zero_byte_files = [path.name for path in parquet_files if path.stat().st_size == 0]
    checks.append(
        check(
            "zero_byte_files",
            "PASS" if not zero_byte_files else "BLOCK",
            observed=zero_byte_files,
            expected=[],
        )
    )

    footer_null_checks = dict(spec.get("footer_null_checks", {}))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(
            executor.map(
                lambda path: safe_footer_record(
                    path,
                    root=root,
                    content_sha256=content_sha256,
                    footer_stat_columns=footer_null_checks,
                ),
                parquet_files,
            )
        )
    records.sort(key=lambda item: str(item["relative_path"]))
    errors = [record for record in records if record.get("error")]
    checks.append(
        check(
            "readable_parquet_footers",
            "PASS" if not errors else "BLOCK",
            observed=[record["relative_path"] for record in errors],
            expected=[],
        )
    )

    readable = [record for record in records if not record.get("error")]
    zero_row_files = [
        record["relative_path"]
        for record in readable
        if int(record["footer_rows"]) == 0
    ]
    checks.append(
        check(
            "zero_row_files",
            "PASS" if not zero_row_files else "BLOCK",
            observed=zero_row_files,
            expected=[],
        )
    )
    footer_rows = sum(int(record["footer_rows"]) for record in readable)
    expected_footer_rows = spec.get("expected_footer_rows")
    if expected_footer_rows is None:
        checks.append(
            check(
                "historical_footer_rows",
                "WARN",
                observed=footer_rows,
                note="no historical expectation is frozen for this table",
            )
        )
    else:
        expected_rows = int(expected_footer_rows)
        checks.append(
            check(
                "historical_footer_rows",
                "PASS" if footer_rows == expected_rows else "BLOCK",
                observed=footer_rows,
                expected=expected_rows,
                note="footer rows do not establish distinct IDs or orphan freedom",
            )
        )

    canonical_schema_counts: dict[str, int] = {}
    physical_schema_counts: dict[str, int] = {}
    schema_columns: dict[str, list[str]] = {}
    for record in readable:
        canonical = str(record["canonical_schema_fingerprint_sha256"])
        physical = str(record["physical_schema_fingerprint_sha256"])
        canonical_schema_counts[canonical] = (
            canonical_schema_counts.get(canonical, 0) + 1
        )
        physical_schema_counts[physical] = physical_schema_counts.get(physical, 0) + 1
        schema_columns.setdefault(canonical, list(record["columns"]))
    checks.append(
        check(
            "single_canonical_schema_fingerprint",
            "PASS" if len(canonical_schema_counts) == 1 else "BLOCK",
            observed=canonical_schema_counts,
            expected="one order-insensitive names/types/nullability fingerprint",
        )
    )
    checks.append(
        check(
            "physical_schema_order_variants",
            "WARN" if len(physical_schema_counts) > 1 else "PASS",
            observed=physical_schema_counts,
            expected="one physical schema ordering",
            note="canonical compatibility can pass while base/delta column order differs",
        )
    )
    required_columns = set(str(value) for value in spec.get("required_columns", []))
    missing_by_schema = {
        fingerprint: sorted(required_columns - set(columns))
        for fingerprint, columns in schema_columns.items()
        if required_columns - set(columns)
    }
    checks.append(
        check(
            "required_columns",
            "PASS" if not missing_by_schema else "BLOCK",
            observed=missing_by_schema,
            expected=sorted(required_columns),
        )
    )

    required_column_types = {"id": "string"} if "id" in required_columns else {}
    required_column_types.update(
        {
            str(name): str(value)
            for name, value in dict(spec.get("required_column_types", {})).items()
        }
    )
    type_mismatches: dict[str, dict[str, dict[str, str | None]]] = {}
    for record in readable:
        fingerprint = str(record["canonical_schema_fingerprint_sha256"])
        if fingerprint in type_mismatches:
            continue
        fields = dict(record["schema_fields"])
        mismatches = {
            name: {
                "observed": (str(fields[name]["type"]) if name in fields else None),
                "expected": expected_type,
            }
            for name, expected_type in required_column_types.items()
            if name not in fields or str(fields[name]["type"]) != expected_type
        }
        if mismatches:
            type_mismatches[fingerprint] = mismatches
    checks.append(
        check(
            "required_column_types",
            "PASS" if not type_mismatches else "BLOCK",
            observed=type_mismatches,
            expected=required_column_types,
        )
    )

    aggregate_null_statistics: dict[str, dict[str, Any]] = {}
    for column, raw_policy in footer_null_checks.items():
        policy = dict(raw_policy)
        file_statistics = [
            dict(record.get("footer_null_statistics", {}).get(column, {}))
            for record in readable
        ]
        complete = bool(readable) and all(
            stats
            and stats.get("null_count") is not None
            and stats.get("row_groups_with_statistics") == stats.get("row_groups_total")
            for stats in file_statistics
        )
        null_count = (
            sum(int(stats["null_count"]) for stats in file_statistics)
            if complete
            else None
        )
        expected_null_count = policy.get("expected_null_count")
        if not complete:
            status = str(policy.get("missing_statistics_status", "BLOCK"))
        elif expected_null_count is not None and null_count != int(expected_null_count):
            status = str(policy.get("mismatch_status", "BLOCK"))
        elif null_count:
            status = str(policy.get("nonzero_status", "WARN"))
        else:
            status = "PASS"
        aggregate_null_statistics[str(column)] = {
            "null_count": null_count,
            "footer_rows": footer_rows,
            "null_fraction": (
                null_count / footer_rows
                if null_count is not None and footer_rows
                else None
            ),
            "statistics_complete": complete,
        }
        checks.append(
            check(
                f"footer_null_count:{column}",
                status,
                observed=aggregate_null_statistics[str(column)],
                expected=(
                    int(expected_null_count)
                    if expected_null_count is not None
                    else "footer statistics available"
                ),
                note=policy.get("note"),
            )
        )

    checks.append(
        check(
            "content_sha256",
            (
                "PASS"
                if content_sha256 and not errors
                else "PENDING"
                if content_sha256_required
                else "PASS"
            ),
            observed="complete" if content_sha256 and not errors else "not computed",
            expected=(
                "complete before PASS_METADATA"
                if content_sha256_required
                else "optional"
            ),
        )
    )

    return {
        "name": table_name,
        "role": spec.get("role"),
        "primary_key_candidate": spec.get("primary_key_candidate"),
        "foreign_key_candidate": spec.get("foreign_key_candidate"),
        "file_count": len(parquet_files),
        "bytes": sum(int(record.get("bytes") or 0) for record in records),
        "footer_rows": footer_rows,
        "row_groups": sum(int(record.get("row_groups", 0)) for record in readable),
        "footer_metadata_bytes": sum(
            int(record.get("footer_metadata_bytes", 0)) for record in readable
        ),
        "canonical_schema_fingerprints": canonical_schema_counts,
        "physical_schema_fingerprints": physical_schema_counts,
        "footer_null_statistics": aggregate_null_statistics,
        "lineage_evidence_id": spec.get("lineage_evidence_id"),
        "lineage_evidence": spec.get("lineage_evidence"),
        "inventory_record_fingerprint_sha256": inventory_record_fingerprint(records),
        "checks": checks,
        "files": records,
    }


def audit_source(
    *,
    root: Path,
    config: dict[str, Any],
    config_path: Path,
    workers: int,
    content_sha256: bool,
) -> dict[str, Any]:
    started = utc_now()
    table_specs = config["tables"]
    table_names = list(table_specs)
    inventory_config = config["inventory"]
    content_sha256_required = bool(
        inventory_config.get("content_sha256_required_for_pass_metadata", False)
    )
    checks: list[dict[str, Any]] = []

    root_id = str(config["root"]["id"])
    checks.append(
        check(
            "root_id",
            "PASS" if root.name == root_id else "BLOCK",
            observed=root.name,
            expected=root_id,
        )
    )
    observed_table_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    expected_table_dirs = sorted(table_names)
    unexpected_table_dirs = sorted(set(observed_table_dirs) - set(expected_table_dirs))
    missing_table_dirs = sorted(set(expected_table_dirs) - set(observed_table_dirs))
    expected_table_count = int(inventory_config["expected_table_count"])
    checks.extend(
        [
            check(
                "table_directory_count",
                "PASS" if len(observed_table_dirs) == expected_table_count else "BLOCK",
                observed=len(observed_table_dirs),
                expected=expected_table_count,
            ),
            check(
                "missing_table_directories",
                "PASS" if not missing_table_dirs else "BLOCK",
                observed=missing_table_dirs,
                expected=[],
            ),
            check(
                "unexpected_table_directories",
                (
                    "BLOCK"
                    if unexpected_table_dirs
                    and bool(inventory_config["reject_unexpected_table_directories"])
                    else "PASS"
                ),
                observed=unexpected_table_dirs,
                expected=[],
            ),
        ]
    )
    symlink_table_dirs = sorted(
        path.name for path in root.iterdir() if path.is_symlink() and path.is_dir()
    )
    checks.append(
        check(
            "symlink_table_directories",
            "PASS" if not symlink_table_dirs else "BLOCK",
            observed=symlink_table_dirs,
            expected=[],
        )
    )
    root_files = sorted(path.name for path in root.iterdir() if path.is_file())
    checks.append(
        check(
            "root_level_files",
            "WARN" if root_files else "PASS",
            observed=root_files,
            expected=[],
            note="root metadata files are recorded but not treated as parquet tables",
        )
    )

    before = snapshot_files(root, table_names)
    filename_pattern = re.compile(str(inventory_config["allowed_filename_pattern"]))
    if set(filename_pattern.groupindex) != {"prefix", "index"}:
        raise ValueError(
            "allowed_filename_pattern must define named prefix and index groups"
        )
    tables = [
        audit_table(
            root,
            table_name,
            dict(table_specs[table_name]),
            filename_pattern=filename_pattern,
            workers=workers,
            content_sha256=content_sha256,
            content_sha256_required=content_sha256_required,
            reject_unexpected_entries=bool(
                inventory_config["reject_unexpected_table_entries"]
            ),
        )
        for table_name in table_names
    ]
    after = snapshot_files(root, table_names)
    inventory_fingerprint = snapshot_fingerprint(after, include_mtime=False)
    before_stability = snapshot_fingerprint(before, include_mtime=True)
    after_stability = snapshot_fingerprint(after, include_mtime=True)
    expected_inventory_fingerprint = inventory_config.get(
        "expected_inventory_fingerprint_sha256"
    )
    checks.append(
        check(
            "frozen_inventory_fingerprint",
            (
                "PASS"
                if expected_inventory_fingerprint is None
                or inventory_fingerprint == str(expected_inventory_fingerprint)
                else "BLOCK"
            ),
            observed=inventory_fingerprint,
            expected=(
                str(expected_inventory_fingerprint)
                if expected_inventory_fingerprint is not None
                else "capture-only"
            ),
            note="This path-and-size identity is not a content checksum.",
        )
    )
    checks.append(
        check(
            "inventory_stable_during_audit",
            "PASS" if before_stability == after_stability else "BLOCK",
            observed={"before": before_stability, "after": after_stability},
            expected="matching fingerprints",
        )
    )

    all_checks = checks + [item for table in tables for item in table["checks"]]
    blocks = [item for item in all_checks if item["status"] == "BLOCK"]
    warnings = [item for item in all_checks if item["status"] == "WARN"]
    pending = [item for item in all_checks if item["status"] == "PENDING"]
    if blocks:
        status = "BLOCKED"
    elif content_sha256 or not content_sha256_required:
        status = "PASS_METADATA"
    else:
        status = "INVENTORY_ONLY"

    config_bytes = config_path.read_bytes()
    try:
        config_label = config_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        config_label = config_path.name
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "manifest_id": str(config["label"]),
        "status": status,
        "pass_metadata": status == "PASS_METADATA",
        "pass_g1": False,
        "claim_boundary": (
            "Footer totals and schema consistency do not establish distinct IDs, "
            "join coverage, author validity, leakage freedom, or empirical readiness."
        ),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "root": {
            "id": root_id,
            "env": str(config["root"]["env"]),
            "snapshot_date": str(config["root"]["snapshot_date"]),
            "lineage_id": str(config["root"]["lineage_id"]),
            "coverage": str(config["root"]["coverage"]),
            "historical_reference_id": str(config["root"]["historical_reference_id"]),
            "table_count": len(observed_table_dirs),
            "parquet_file_count": len(after),
            "bytes": sum(row.bytes for row in after),
            "footer_rows": sum(int(table.get("footer_rows", 0)) for table in tables),
            "row_groups": sum(int(table.get("row_groups", 0)) for table in tables),
            "footer_metadata_bytes": sum(
                int(table.get("footer_metadata_bytes", 0)) for table in tables
            ),
            "inventory_fingerprint_sha256": inventory_fingerprint,
            "stability_fingerprint_sha256": after_stability,
        },
        "creator": {
            **git_metadata(),
            "tool": "scripts/audit_openalex_source_inventory.py",
            "tool_sha256": sha256_file(Path(__file__)),
            "config": config_label,
            "config_sha256": sha256_bytes(config_bytes),
        },
        "execution": {
            "footer_workers": workers,
            "content_sha256": content_sha256,
            "content_sha256_required_for_pass_metadata": content_sha256_required,
            "path_redacted": True,
        },
        "feature_roles": config.get("feature_roles", {}),
        "checks": checks,
        "tables": tables,
        "summary": {
            "block_count": len(blocks),
            "warning_count": len(warnings),
            "pending_count": len(pending),
        },
        "pending_full_scan_gates": list(config.get("pending_full_scan_gates", [])),
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    root = manifest["root"]
    summary = manifest["summary"]
    lines = [
        "# OpenAlex G-1 Source Inventory Audit",
        "",
        f"- Status: `{manifest['status']}`",
        f"- PASS_METADATA: `{str(manifest['pass_metadata']).lower()}`",
        f"- PASS_G1: `{str(manifest['pass_g1']).lower()}`",
        f"- Root ID: `{root['id']}`",
        f"- Snapshot date: `{root['snapshot_date']}`",
        f"- Tables: `{root['table_count']}`",
        f"- Parquet files: `{root['parquet_file_count']}`",
        f"- Bytes: `{root['bytes']}`",
        f"- Inventory fingerprint: `{root['inventory_fingerprint_sha256']}`",
        f"- Blocks / warnings / pending: `{summary['block_count']} / "
        f"{summary['warning_count']} / {summary['pending_count']}`",
        "",
        "## Claim Boundary",
        "",
        manifest["claim_boundary"],
        "",
        "## Table Inventory",
        "",
        "| Table | Role | Files | Footer rows | Bytes | Schema variants |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for table in manifest["tables"]:
        lines.append(
            f"| `{table['name']}` | {table.get('role') or ''} | "
            f"{table.get('file_count', 0)} | {table.get('footer_rows', 0)} | "
            f"{table.get('bytes', 0)} | "
            f"{len(table.get('canonical_schema_fingerprints', {}))} |"
        )
    lines.extend(
        [
            "",
            "## Pending Full-Scan Gates",
            "",
            *[f"- `{name}`" for name in manifest["pending_full_scan_gates"]],
            "",
            "A metadata inventory authorizes only the next full-scan validation step.",
            "It does not authorize effect estimation or model fitting.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--content-sha256", action="store_true")
    parser.add_argument("--require-inventory-pass", action="store_true")
    parser.add_argument("--require-pass-metadata", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    root = resolve_root(config, args.root)
    output_paths = [args.output]
    if args.markdown_output is not None:
        output_paths.append(args.markdown_output)
    unsafe_outputs = [
        str(path)
        for path in output_paths
        if path.expanduser().resolve().is_relative_to(root)
    ]
    if unsafe_outputs:
        print(
            "audit outputs must not be written inside the source root: "
            + ", ".join(unsafe_outputs),
            file=sys.stderr,
        )
        return 2
    workers = 1 if args.content_sha256 else args.workers
    manifest = audit_source(
        root=root,
        config=config,
        config_path=config_path,
        workers=workers,
        content_sha256=bool(args.content_sha256),
    )
    atomic_write(
        args.output,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    if args.markdown_output is not None:
        atomic_write(args.markdown_output, render_markdown(manifest))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "pass_metadata": manifest["pass_metadata"],
                "pass_g1": manifest["pass_g1"],
                "tables": manifest["root"]["table_count"],
                "parquet_files": manifest["root"]["parquet_file_count"],
                "blocks": manifest["summary"]["block_count"],
                "warnings": manifest["summary"]["warning_count"],
                "pending": manifest["summary"]["pending_count"],
            },
            sort_keys=True,
        )
    )
    if args.require_pass_metadata and not manifest["pass_metadata"]:
        return 2
    if args.require_inventory_pass and manifest["summary"]["block_count"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
