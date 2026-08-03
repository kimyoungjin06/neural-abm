#!/usr/bin/env python3
"""Audit exact W-1/W0 incidence in a frozen OpenAlex Parquet inventory.

This is a deliberately narrow audit. It applies Parquet predicate pushdown to
the source ``id`` column of every frozen table and, separately, to declared
work-target columns. It reports targeted match counts and affected files only;
it neither reads nor claims conservation of all rows.
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
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_sentinel_incidence_v0.yaml"
)
DEFAULT_PARENT_MANIFEST = (
    REPO_ROOT
    / "experiments/runs/20260716_openalex_g1_source_inventory/source_manifest.json"
)
CONFIG_SCHEMA_VERSION = "openalex_sentinel_incidence_config.v1"
REPORT_SCHEMA_VERSION = "openalex_sentinel_incidence_report.v1"
SENTINEL_KEYS = ("w_minus_1", "w_zero")


class SentinelAuditError(RuntimeError):
    """Raised when the targeted audit cannot honor its frozen contract."""


@dataclass(frozen=True)
class FileSnapshot:
    relative_path: str
    bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class ScanTask:
    scope: str
    table: str
    column: str
    relative_path: str


@dataclass(frozen=True)
class ScanResult:
    task: ScanTask
    counts: dict[str, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SentinelAuditError("sentinel incidence config must be a mapping")
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SentinelAuditError("unsupported sentinel incidence config schema_version")
    parent = value.get("parent_inventory")
    settings = value.get("targeted_incidence")
    if not isinstance(parent, dict) or not isinstance(settings, dict):
        raise SentinelAuditError(
            "config must define parent_inventory and targeted_incidence mappings"
        )
    required_parent = {
        "schema_version",
        "manifest_id",
        "manifest_sha256",
        "source_root_env",
        "source_root_id",
        "expected_table_count",
        "inventory_fingerprint_sha256",
        "stability_fingerprint_sha256",
    }
    missing_parent = sorted(required_parent - set(parent))
    if missing_parent:
        raise SentinelAuditError(
            f"parent_inventory is missing fields: {', '.join(missing_parent)}"
        )
    sentinels = settings.get("sentinels")
    if not isinstance(sentinels, dict) or tuple(sentinels) != SENTINEL_KEYS:
        raise SentinelAuditError(
            "targeted_incidence.sentinels must declare w_minus_1 then w_zero"
        )
    expected_sentinels = {
        "w_minus_1": (
            "https://openalex.org/W-1",
            "quarantine_candidate",
        ),
        "w_zero": ("https://openalex.org/W0", "unresolved_diagnostic"),
    }
    for key, (expected_value, expected_disposition) in expected_sentinels.items():
        record = sentinels.get(key)
        if not isinstance(record, dict):
            raise SentinelAuditError(f"sentinel {key} must be a mapping")
        if record.get("value") != expected_value:
            raise SentinelAuditError(f"sentinel {key} has an unexpected exact value")
        if record.get("disposition") != expected_disposition:
            raise SentinelAuditError(f"sentinel {key} has an unexpected disposition")
    targets = settings.get("work_target_columns")
    if not isinstance(targets, dict) or not targets:
        raise SentinelAuditError("work_target_columns must be a non-empty mapping")
    if not str(settings.get("source_id_column", "")):
        raise SentinelAuditError("source_id_column must be non-empty")
    if int(settings.get("batch_size", 0)) < 1:
        raise SentinelAuditError("batch_size must be positive")
    if int(settings.get("workers", 0)) < 1:
        raise SentinelAuditError("workers must be positive")
    if settings.get("predicate_pushdown_required") is not True:
        raise SentinelAuditError("predicate_pushdown_required must be true")
    if int(parent.get("expected_table_count", 0)) < 1:
        raise SentinelAuditError("expected_table_count must be positive")
    for field in (
        "manifest_sha256",
        "inventory_fingerprint_sha256",
        "stability_fingerprint_sha256",
    ):
        raw = str(parent[field])
        if len(raw) != 64 or any(
            character not in "0123456789abcdef" for character in raw
        ):
            raise SentinelAuditError(
                f"parent_inventory.{field} must be lowercase SHA-256"
            )
    return value


def resolve_root(config: dict[str, Any], explicit_root: Path | None) -> Path:
    frozen = config["parent_inventory"]
    if explicit_root is None:
        env_name = str(frozen["source_root_env"])
        raw = os.environ.get(env_name)
        if not raw:
            raise SentinelAuditError(f"set {env_name} or pass --root")
        root = Path(raw)
    else:
        root = explicit_root
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise SentinelAuditError("source root must be a regular directory")
    if root.name != str(frozen["source_root_id"]):
        raise SentinelAuditError("source root basename does not match frozen contract")
    return root


def safe_relative_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SentinelAuditError(f"unsafe relative path in parent inventory: {raw}")
    return relative


def inventory_fingerprint(rows: Iterable[FileSnapshot]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.relative_path):
        digest.update(f"{row.relative_path}\0{row.bytes}\n".encode("utf-8"))
    return digest.hexdigest()


def stability_fingerprint(rows: Iterable[FileSnapshot]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.relative_path):
        digest.update(
            f"{row.relative_path}\0{row.bytes}\0{row.mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _regular_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def snapshot_live_root(root: Path, table_names: Iterable[str]) -> list[FileSnapshot]:
    expected_tables = sorted(set(table_names))
    observed_tables = sorted(
        entry.name for entry in root.iterdir() if entry.is_dir() or entry.is_symlink()
    )
    if observed_tables != expected_tables:
        raise SentinelAuditError(
            "live table directory set differs from the parent inventory"
        )
    rows: list[FileSnapshot] = []
    for table_name in expected_tables:
        table_dir = root / table_name
        if not _regular_directory(table_dir):
            raise SentinelAuditError(
                f"table path is not a regular directory: {table_name}"
            )
        for entry in sorted(table_dir.iterdir(), key=lambda path: path.name):
            try:
                mode = entry.lstat().st_mode
            except FileNotFoundError as exc:
                raise SentinelAuditError(
                    f"input disappeared during snapshot: {table_name}/{entry.name}"
                ) from exc
            if not stat.S_ISREG(mode) or entry.suffix != ".parquet":
                raise SentinelAuditError(
                    f"unexpected live table entry: {table_name}/{entry.name}"
                )
            file_stat = entry.stat()
            rows.append(
                FileSnapshot(
                    relative_path=entry.relative_to(root).as_posix(),
                    bytes=int(file_stat.st_size),
                    mtime_ns=int(file_stat.st_mtime_ns),
                )
            )
    return rows


def load_parent_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SentinelAuditError("parent inventory manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise SentinelAuditError("parent inventory manifest must be an object")
    return manifest, sha256_bytes(raw)


def validate_parent_and_live_root(
    *,
    root: Path,
    manifest_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], str, list[FileSnapshot]]:
    manifest, manifest_sha256 = load_parent_manifest(manifest_path)
    frozen = config["parent_inventory"]
    if manifest_sha256 != str(frozen["manifest_sha256"]):
        raise SentinelAuditError("parent inventory manifest SHA-256 mismatch")
    if manifest.get("schema_version") != frozen["schema_version"]:
        raise SentinelAuditError("parent inventory schema_version mismatch")
    if manifest.get("manifest_id") != frozen["manifest_id"]:
        raise SentinelAuditError("parent inventory manifest_id mismatch")
    root_record = manifest.get("root")
    if not isinstance(root_record, dict):
        raise SentinelAuditError("parent inventory root record is missing")
    if root_record.get("id") != frozen["source_root_id"]:
        raise SentinelAuditError("parent inventory root ID mismatch")
    expected_inventory = str(frozen["inventory_fingerprint_sha256"])
    expected_stability = str(frozen["stability_fingerprint_sha256"])
    if root_record.get("inventory_fingerprint_sha256") != expected_inventory:
        raise SentinelAuditError("parent inventory fingerprint/config mismatch")
    if root_record.get("stability_fingerprint_sha256") != expected_stability:
        raise SentinelAuditError("parent stability fingerprint/config mismatch")
    if int(manifest.get("summary", {}).get("block_count", 1)) != 0:
        raise SentinelAuditError("parent inventory contains blocking checks")

    raw_tables = manifest.get("tables")
    if not isinstance(raw_tables, list):
        raise SentinelAuditError("parent inventory tables must be a list")
    expected_table_count = int(frozen["expected_table_count"])
    if len(raw_tables) != expected_table_count:
        raise SentinelAuditError("parent inventory table count mismatch")
    table_names = [str(table.get("name", "")) for table in raw_tables]
    if any(not name for name in table_names) or len(table_names) != len(
        set(table_names)
    ):
        raise SentinelAuditError(
            "parent inventory table names are missing or duplicated"
        )
    if int(root_record.get("table_count", -1)) != expected_table_count:
        raise SentinelAuditError("parent root table count mismatch")

    source_column = str(config["targeted_incidence"]["source_id_column"])
    target_columns = {
        str(table): str(column)
        for table, column in config["targeted_incidence"]["work_target_columns"].items()
    }
    unknown_targets = sorted(set(target_columns) - set(table_names))
    if unknown_targets:
        raise SentinelAuditError(
            f"work-target tables absent from parent: {', '.join(unknown_targets)}"
        )

    manifest_snapshots: list[FileSnapshot] = []
    seen_paths: set[str] = set()
    for table in raw_tables:
        table_name = str(table["name"])
        records = table.get("files")
        if not isinstance(records, list) or not records:
            raise SentinelAuditError(f"parent table has no files: {table_name}")
        if int(table.get("file_count", -1)) != len(records):
            raise SentinelAuditError(f"parent table file count mismatch: {table_name}")
        required_columns = {source_column}
        if table_name in target_columns:
            required_columns.add(target_columns[table_name])
        for record in records:
            if not isinstance(record, dict) or record.get("error"):
                raise SentinelAuditError(
                    f"parent file record is unreadable: {table_name}"
                )
            relative = safe_relative_path(str(record.get("relative_path", "")))
            if relative.parts[0] != table_name:
                raise SentinelAuditError(
                    f"parent file path escapes declared table: {relative.as_posix()}"
                )
            relative_text = relative.as_posix()
            if relative_text in seen_paths:
                raise SentinelAuditError(f"duplicate parent file path: {relative_text}")
            seen_paths.add(relative_text)
            columns = record.get("columns")
            if not isinstance(columns, list) or not required_columns.issubset(columns):
                missing = sorted(required_columns - set(columns or []))
                raise SentinelAuditError(
                    f"required targeted columns missing in {relative_text}: {missing}"
                )
            manifest_snapshots.append(
                FileSnapshot(
                    relative_path=relative_text,
                    bytes=int(record["bytes"]),
                    mtime_ns=int(record["mtime_ns"]),
                )
            )
    if int(root_record.get("parquet_file_count", -1)) != len(manifest_snapshots):
        raise SentinelAuditError("parent root Parquet file count mismatch")
    if inventory_fingerprint(manifest_snapshots) != expected_inventory:
        raise SentinelAuditError(
            "parent file records do not reproduce inventory fingerprint"
        )
    if stability_fingerprint(manifest_snapshots) != expected_stability:
        raise SentinelAuditError(
            "parent file records do not reproduce stability fingerprint"
        )

    live = snapshot_live_root(root, table_names)
    if live != sorted(manifest_snapshots, key=lambda item: item.relative_path):
        raise SentinelAuditError("live path/size/mtime inventory differs from parent")
    if inventory_fingerprint(live) != expected_inventory:
        raise SentinelAuditError("live path-and-size inventory fingerprint mismatch")
    if stability_fingerprint(live) != expected_stability:
        raise SentinelAuditError("live stability fingerprint mismatch")
    return manifest, manifest_sha256, live


def sentinel_values(config: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(config["targeted_incidence"]["sentinels"][key]["value"])
        for key in SENTINEL_KEYS
    }


def scan_targeted_file(
    *,
    root: Path,
    task: ScanTask,
    sentinels: dict[str, str],
    batch_size: int,
) -> ScanResult:
    """Count exact sentinel matches using a pushed-down ``isin`` predicate."""

    path = root / safe_relative_path(task.relative_path)
    dataset = ds.dataset(path, format="parquet")
    values = [sentinels[key] for key in SENTINEL_KEYS]
    scanner = dataset.scanner(
        columns=[task.column],
        filter=ds.field(task.column).isin(values),
        batch_size=batch_size,
        use_threads=False,
    )
    counts = {key: 0 for key in SENTINEL_KEYS}
    matched_rows = 0
    for batch in scanner.to_batches():
        if batch.schema.names != [task.column]:
            raise SentinelAuditError(
                f"scanner returned an unexpected schema for {task.relative_path}"
            )
        array = batch.column(0)
        matched_rows += batch.num_rows
        for key in SENTINEL_KEYS:
            matches = pc.sum(
                pc.cast(pc.equal(array, pa.scalar(sentinels[key])), pa.int64())
            ).as_py()
            counts[key] += int(matches or 0)
    if sum(counts.values()) != matched_rows:
        raise SentinelAuditError(
            f"predicate returned non-sentinel rows for {task.relative_path}"
        )
    return ScanResult(task=task, counts=counts)


def build_scan_tasks(
    manifest: dict[str, Any], config: dict[str, Any]
) -> list[ScanTask]:
    settings = config["targeted_incidence"]
    source_column = str(settings["source_id_column"])
    targets = {
        str(table): str(column)
        for table, column in settings["work_target_columns"].items()
    }
    tasks: list[ScanTask] = []
    for table in sorted(manifest["tables"], key=lambda item: str(item["name"])):
        table_name = str(table["name"])
        records = sorted(table["files"], key=lambda item: str(item["relative_path"]))
        for record in records:
            relative_path = str(record["relative_path"])
            tasks.append(
                ScanTask(
                    scope="source_id",
                    table=table_name,
                    column=source_column,
                    relative_path=relative_path,
                )
            )
            if table_name in targets:
                tasks.append(
                    ScanTask(
                        scope="work_target",
                        table=table_name,
                        column=targets[table_name],
                        relative_path=relative_path,
                    )
                )
    return sorted(
        tasks,
        key=lambda item: (item.scope, item.table, item.column, item.relative_path),
    )


def execute_tasks(
    *,
    root: Path,
    tasks: list[ScanTask],
    sentinels: dict[str, str],
    batch_size: int,
    workers: int,
    progress: Callable[[int, int], None] | None = None,
) -> list[ScanResult]:
    def execute(task: ScanTask) -> ScanResult:
        try:
            return scan_targeted_file(
                root=root,
                task=task,
                sentinels=sentinels,
                batch_size=batch_size,
            )
        except SentinelAuditError:
            raise
        except Exception as exc:
            raise SentinelAuditError(
                f"targeted predicate scan failed for {task.relative_path} "
                f"column {task.column}: {type(exc).__name__}: {exc}"
            ) from exc

    results: list[ScanResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, result in enumerate(executor.map(execute, tasks), start=1):
            results.append(result)
            if progress is not None and (
                completed == len(tasks) or completed % 1000 == 0
            ):
                progress(completed, len(tasks))
    return results


def _empty_incidence(sentinels: dict[str, str]) -> dict[str, Any]:
    return {
        key: {
            "value": sentinels[key],
            "exact_row_count": 0,
            "affected_file_count": 0,
            "affected_files": [],
        }
        for key in SENTINEL_KEYS
    }


def aggregate_results(
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    results: list[ScanResult],
) -> dict[str, Any]:
    sentinels = sentinel_values(config)
    settings = config["targeted_incidence"]
    source_column = str(settings["source_id_column"])
    target_columns = {
        str(table): str(column)
        for table, column in settings["work_target_columns"].items()
    }
    source_tables: dict[str, dict[str, Any]] = {}
    target_tables: dict[str, dict[str, Any]] = {}
    for table in sorted(manifest["tables"], key=lambda item: str(item["name"])):
        name = str(table["name"])
        source_tables[name] = {
            "table": name,
            "column": source_column,
            "files_evaluated": int(table["file_count"]),
            "sentinels": _empty_incidence(sentinels),
        }
        if name in target_columns:
            target_tables[name] = {
                "table": name,
                "column": target_columns[name],
                "files_evaluated": int(table["file_count"]),
                "sentinels": _empty_incidence(sentinels),
            }

    for result in results:
        containers = (
            source_tables if result.task.scope == "source_id" else target_tables
        )
        record = containers[result.task.table]
        for key in SENTINEL_KEYS:
            count = int(result.counts[key])
            record["sentinels"][key]["exact_row_count"] += count
            if count:
                record["sentinels"][key]["affected_files"].append(
                    {
                        "relative_path": result.task.relative_path,
                        "exact_row_count": count,
                    }
                )

    for record in [*source_tables.values(), *target_tables.values()]:
        for key in SENTINEL_KEYS:
            affected = sorted(
                record["sentinels"][key]["affected_files"],
                key=lambda item: item["relative_path"],
            )
            record["sentinels"][key]["affected_files"] = affected
            record["sentinels"][key]["affected_file_count"] = len(affected)

    def scope_totals(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = list(records)
        return {
            key: {
                "value": sentinels[key],
                "exact_row_count": sum(
                    int(record["sentinels"][key]["exact_row_count"]) for record in rows
                ),
                "affected_file_count": sum(
                    int(record["sentinels"][key]["affected_file_count"])
                    for record in rows
                ),
            }
            for key in SENTINEL_KEYS
        }

    source_rows = list(source_tables.values())
    target_rows = list(target_tables.values())
    return {
        "source_id": {
            "column": source_column,
            "table_count": len(source_rows),
            "files_evaluated": sum(item["files_evaluated"] for item in source_rows),
            "totals": scope_totals(source_rows),
            "tables": source_rows,
        },
        "work_targets": {
            "table_column_count": len(target_rows),
            "files_evaluated": sum(item["files_evaluated"] for item in target_rows),
            "totals": scope_totals(target_rows),
            "tables": target_rows,
        },
    }


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


def relative_or_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def audit_sentinel_incidence(
    *,
    root: Path,
    config: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    started_clock = time.monotonic()
    config_sha256 = sha256_file(config_path)
    tool_path = Path(__file__).resolve()
    tool_sha256 = sha256_file(tool_path)
    manifest, manifest_sha256, start_snapshot = validate_parent_and_live_root(
        root=root,
        manifest_path=manifest_path,
        config=config,
    )
    start_stability = stability_fingerprint(start_snapshot)
    settings = config["targeted_incidence"]
    effective_workers = int(workers or settings["workers"])
    if effective_workers < 1:
        raise SentinelAuditError("workers must be positive")
    tasks = build_scan_tasks(manifest, config)
    results = execute_tasks(
        root=root,
        tasks=tasks,
        sentinels=sentinel_values(config),
        batch_size=int(settings["batch_size"]),
        workers=effective_workers,
        progress=progress,
    )
    incidence = aggregate_results(manifest=manifest, config=config, results=results)

    end_manifest, end_manifest_sha256, end_snapshot = validate_parent_and_live_root(
        root=root,
        manifest_path=manifest_path,
        config=config,
    )
    if end_manifest != manifest or end_manifest_sha256 != manifest_sha256:
        raise SentinelAuditError("parent inventory manifest changed during audit")
    end_stability = stability_fingerprint(end_snapshot)
    if end_snapshot != start_snapshot or end_stability != start_stability:
        raise SentinelAuditError("source inventory changed during audit")
    if sha256_file(config_path) != config_sha256:
        raise SentinelAuditError("sentinel incidence config changed during audit")
    if sha256_file(tool_path) != tool_sha256:
        raise SentinelAuditError("sentinel incidence tool changed during audit")

    sentinels = {
        key: {
            "value": str(settings["sentinels"][key]["value"]),
            "disposition": str(settings["sentinels"][key]["disposition"]),
        }
        for key in SENTINEL_KEYS
    }
    deterministic_payload = {
        "parent_manifest_sha256": manifest_sha256,
        "parent_inventory_fingerprint_sha256": config["parent_inventory"][
            "inventory_fingerprint_sha256"
        ],
        "parent_stability_fingerprint_sha256": start_stability,
        "config_sha256": config_sha256,
        "tool_sha256": tool_sha256,
        "sentinels": sentinels,
        "incidence": incidence,
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": str(config["label"]),
        "status": "COMPLETE_TARGETED_INCIDENCE_AUDIT",
        "audit_complete": True,
        "pass_g1": False,
        "analysis_view_quarantine_applied": False,
        "claim_boundary": str(config["claim_boundary"]),
        "scope": {
            "kind": "targeted_incidence_only",
            "row_conservation_claimed": False,
            "all_non_sentinel_identifiers_validated": False,
        },
        "sentinels": sentinels,
        "source": {
            "source_root_id": str(config["parent_inventory"]["source_root_id"]),
            "source_root_env": str(config["parent_inventory"]["source_root_env"]),
            "parent_inventory_fingerprint_sha256": str(
                config["parent_inventory"]["inventory_fingerprint_sha256"]
            ),
            "parent_stability_fingerprint_sha256": start_stability,
            "parent_stable_after_audit": True,
        },
        "provenance": {
            "tool": relative_or_name(tool_path),
            "tool_sha256": tool_sha256,
            "config": relative_or_name(config_path),
            "config_sha256": config_sha256,
            "parent_manifest": relative_or_name(manifest_path),
            "parent_manifest_sha256": manifest_sha256,
            "git": git_metadata(),
            "result_fingerprint_sha256": sha256_bytes(
                canonical_json_bytes(deterministic_payload)
            ),
            "path_redacted": True,
        },
        "execution": {
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started_clock,
            "workers": effective_workers,
            "batch_size": int(settings["batch_size"]),
            "predicate_pushdown": {
                "required": True,
                "applied": True,
                "engine": "pyarrow.dataset.Scanner",
                "expression": "field.isin([W-1, W0])",
            },
            "scan_task_count": len(tasks),
        },
        "incidence": incidence,
    }
    encoded = json.dumps(report, sort_keys=True)
    if str(root) in encoded:
        raise SentinelAuditError("absolute source path leaked into report")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    source = report["incidence"]["source_id"]
    targets = report["incidence"]["work_targets"]
    lines = [
        "# OpenAlex Sentinel Targeted-Incidence Audit",
        "",
        f"- Status: `{report['status']}`",
        "- `pass_g1=false`; this audit does not run the empirical G1 gate.",
        "- Scope: exact targeted W-1/W0 matches only; no row-conservation claim.",
        "- W-1: quarantine candidate; this report does not apply the quarantine.",
        "- W0: separate unresolved diagnostic; it is not folded into W-1.",
        "",
        "## Source-ID incidence across all frozen tables",
        "",
        "| Table | Files | W-1 rows | W-1 files | W0 rows | W0 files |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for table in source["tables"]:
        minus = table["sentinels"]["w_minus_1"]
        zero = table["sentinels"]["w_zero"]
        lines.append(
            f"| `{table['table']}` | {table['files_evaluated']:,} | "
            f"{minus['exact_row_count']:,} | {minus['affected_file_count']:,} | "
            f"{zero['exact_row_count']:,} | {zero['affected_file_count']:,} |"
        )
    source_minus = source["totals"]["w_minus_1"]
    source_zero = source["totals"]["w_zero"]
    lines.append(
        f"| **Targeted total** | {source['files_evaluated']:,} | "
        f"**{source_minus['exact_row_count']:,}** | "
        f"**{source_minus['affected_file_count']:,}** | "
        f"**{source_zero['exact_row_count']:,}** | "
        f"**{source_zero['affected_file_count']:,}** |"
    )
    lines.extend(
        [
            "",
            "## Work-target incidence",
            "",
            "| Table.column | Files | W-1 rows | W-1 files | W0 rows | W0 files |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for table in targets["tables"]:
        minus = table["sentinels"]["w_minus_1"]
        zero = table["sentinels"]["w_zero"]
        lines.append(
            f"| `{table['table']}.{table['column']}` | "
            f"{table['files_evaluated']:,} | {minus['exact_row_count']:,} | "
            f"{minus['affected_file_count']:,} | {zero['exact_row_count']:,} | "
            f"{zero['affected_file_count']:,} |"
        )
    target_minus = targets["totals"]["w_minus_1"]
    target_zero = targets["totals"]["w_zero"]
    lines.append(
        f"| **Targeted total** | {targets['files_evaluated']:,} | "
        f"**{target_minus['exact_row_count']:,}** | "
        f"**{target_minus['affected_file_count']:,}** | "
        f"**{target_zero['exact_row_count']:,}** | "
        f"**{target_zero['affected_file_count']:,}** |"
    )
    lines.extend(["", "## Affected files", ""])
    affected_any = False
    for scope_name, scope in (("source_id", source), ("work_target", targets)):
        for table in scope["tables"]:
            for key in SENTINEL_KEYS:
                affected = table["sentinels"][key]["affected_files"]
                if not affected:
                    continue
                affected_any = True
                lines.append(
                    f"### {scope_name}: `{table['table']}.{table['column']}` / `{key}`"
                )
                lines.append("")
                for item in affected:
                    lines.append(
                        f"- `{item['relative_path']}`: "
                        f"{item['exact_row_count']:,} exact matches"
                    )
                lines.append("")
    if not affected_any:
        lines.extend(["No targeted sentinel matches were found.", ""])
    lines.extend(
        [
            "## Provenance",
            "",
            f"- Parent manifest SHA-256: `{report['provenance']['parent_manifest_sha256']}`",
            f"- Config SHA-256: `{report['provenance']['config_sha256']}`",
            f"- Tool SHA-256: `{report['provenance']['tool_sha256']}`",
            f"- Result fingerprint: `{report['provenance']['result_fingerprint_sha256']}`",
            "- Parent path/size/mtime inventory matched at both start and end.",
            "- All reported file paths are relative to the frozen source root.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    return markdown


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def validate_output_paths(root: Path, json_path: Path, markdown_path: Path) -> None:
    outputs = [lexical_absolute(json_path), lexical_absolute(markdown_path)]
    if outputs[0] == outputs[1]:
        raise SentinelAuditError("JSON and Markdown output paths must differ")
    source = lexical_absolute(root)
    resolved_source = root.resolve()
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise SentinelAuditError(
                f"refusing to replace existing output: {output.name}"
            )
        if output.is_relative_to(source) or output.resolve().is_relative_to(
            resolved_source
        ):
            raise SentinelAuditError("audit outputs must not be inside the source root")


def _write_fsynced(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def write_output_pair(
    *, json_path: Path, markdown_path: Path, report: dict[str, Any]
) -> None:
    json_path = lexical_absolute(json_path)
    markdown_path = lexical_absolute(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_partial = json_path.with_name(f".{json_path.name}.{os.getpid()}.partial")
    markdown_partial = markdown_path.with_name(
        f".{markdown_path.name}.{os.getpid()}.partial"
    )
    for partial in (json_partial, markdown_partial):
        if partial.exists() or partial.is_symlink():
            raise SentinelAuditError(f"temporary output already exists: {partial.name}")
    try:
        _write_fsynced(
            json_partial,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write_fsynced(markdown_partial, render_markdown(report))
        json_partial.replace(json_path)
        markdown_partial.replace(markdown_path)
        for directory in {json_path.parent, markdown_path.parent}:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        json_partial.unlink(missing_ok=True)
        markdown_partial.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--inventory-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--workers", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        manifest_path = args.inventory_manifest.expanduser().resolve()
        config = load_config(config_path)
        root = resolve_root(config, args.root)
        output_json = lexical_absolute(args.output_json)
        output_md = lexical_absolute(
            args.output_md
            if args.output_md is not None
            else args.output_json.with_suffix(".md")
        )
        validate_output_paths(root, output_json, output_md)

        def progress(completed: int, total: int) -> None:
            print(
                f"targeted predicate scans: {completed:,}/{total:,}",
                file=sys.stderr,
                flush=True,
            )

        report = audit_sentinel_incidence(
            root=root,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            workers=args.workers,
            progress=progress,
        )
        if str(root) in render_markdown(report):
            raise SentinelAuditError("absolute source path leaked into Markdown")
        write_output_pair(
            json_path=output_json,
            markdown_path=output_md,
            report=report,
        )
    except (OSError, SentinelAuditError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass_g1": report["pass_g1"],
                "result_fingerprint_sha256": report["provenance"][
                    "result_fingerprint_sha256"
                ],
                "output_json": output_json.name,
                "output_md": output_md.name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
