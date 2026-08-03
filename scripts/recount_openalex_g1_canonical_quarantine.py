#!/usr/bin/env python3
"""Recount grammar-valid OpenAlex work IDs with one frozen malformed quarantine.

The output view excludes only malformed rows that match the frozen quarantine
contract exactly. Grammar-valid W0 is retained and reported as an unresolved
entity-validity diagnostic. Passing this command is not PASS_G1 and does not
establish canonical or entity-valid OpenAlex identifiers.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import recount_openalex_g1 as base  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_recount_canonical_quarantine_v1.yaml"
)
BASE_DEPENDENCY_PATH = REPO_ROOT / "scripts/recount_openalex_g1.py"
CONFIG_SCHEMA_VERSION = "openalex_g1_grammar_quarantine_recount_config.v1"
STATE_SCHEMA_VERSION = "openalex_g1_grammar_quarantine_recount_state.v1"
REPORT_SCHEMA_VERSION = "openalex_g1_grammar_quarantine_recount_report.v1"
PLAN_SCHEMA_VERSION = "openalex_g1_grammar_quarantine_recount_plan.v1"
QUARANTINE_SCHEMA_VERSION = "openalex_g1_malformed_quarantine_observation.v1"
UINT64_BYTES = np.dtype("<u8").itemsize
SPLITMIX64_BUCKETS = base.SPLITMIX64_BUCKETS
SPLITMIX64_SEED = base.SPLITMIX64_SEED
ZERO_WARNING_CODE = "WARN_ZERO_ID_UNRESOLVED"

RecountError = base.RecountError
canonical_json_bytes = base.canonical_json_bytes
sha256_bytes = base.sha256_bytes
sha256_file = base.sha256_file
atomic_write_json = base.atomic_write_json
atomic_write_text = base.atomic_write_text
safe_relative_path = base.safe_relative_path
file_record_sha256 = base.file_record_sha256
splitmix64_bucket_histogram = base.splitmix64_bucket_histogram


def _validate_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise RecountError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _repo_artifact(relative_raw: str, *, field: str) -> Path:
    relative = safe_relative_path(relative_raw)
    path = REPO_ROOT / relative
    mode = base.lstat_mode(path)
    if mode is None or path.is_symlink() or not path.is_file():
        raise RecountError(f"{field} must name a regular non-symlink file")
    return path


def _contract_entry(entry: dict[str, Any], *, field: str) -> dict[str, Any]:
    required = (
        "raw_value",
        "expected_count",
        "expected_relative_path",
        "expected_row_group",
        "expected_file_row",
    )
    missing = [name for name in required if name not in entry]
    if missing:
        raise RecountError(f"{field} is missing fields: {missing}")
    count = int(entry["expected_count"])
    if count != 1:
        raise RecountError(
            f"{field}.expected_count must be 1 when an exact row location is frozen"
        )
    relative = safe_relative_path(str(entry["expected_relative_path"]))
    row_group = int(entry["expected_row_group"])
    file_row = int(entry["expected_file_row"])
    if row_group < 0 or file_row < 0:
        raise RecountError(f"{field} row locations must be non-negative")
    raw_value = entry["raw_value"]
    if not isinstance(raw_value, str) or not raw_value:
        raise RecountError(f"{field}.raw_value must be a non-empty string")
    return {
        "raw_value": raw_value,
        "relative_path": relative.as_posix(),
        "row_group": row_group,
        "file_row": file_row,
        "count": count,
    }


def _entry_key(entry: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(entry["raw_value"]),
        str(entry["relative_path"]),
        int(entry["row_group"]),
        int(entry["file_row"]),
    )


def _sorted_multiset(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, int, int], int] = {}
    for entry in entries:
        key = _entry_key(entry)
        counts[key] = counts.get(key, 0) + int(entry.get("count", 1))
    return [
        {
            "raw_value": key[0],
            "relative_path": key[1],
            "row_group": key[2],
            "file_row": key[3],
            "count": counts[key],
        }
        for key in sorted(counts)
    ]


def expected_quarantine_multiset(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _contract_entry(entry, field=f"quarantine.entries[{index}]")
        for index, entry in enumerate(config["quarantine"]["entries"])
    ]


def expected_zero_multiset(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _contract_entry(
            config["zero_id_diagnostic"]["entry"],
            field="zero_id_diagnostic.entry",
        )
    ]


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecountError("recount config must be a mapping")
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise RecountError("unsupported grammar-quarantine config schema_version")
    settings = value.get("main_id_recount")
    if not isinstance(settings, dict):
        raise RecountError("config must define main_id_recount")
    if settings.get("spool_dtype") != "uint64":
        raise RecountError("main_id_recount.spool_dtype must be uint64")
    for field in (
        "expected_files",
        "expected_body_rows",
        "expected_grammar_valid_rows",
        "batch_size",
        "duplicate_scan_chunk_rows",
    ):
        if int(settings.get(field, 0)) < 1:
            raise RecountError(f"main_id_recount.{field} must be positive")
    for field in (
        "expected_null_rows",
        "expected_grammar_valid_duplicate_row_excess",
        "invalid_sample_limit",
    ):
        if int(settings.get(field, -1)) < 0:
            raise RecountError(f"main_id_recount.{field} must be non-negative")
    if int(settings.get("expected_grammar_valid_distinct_ids", 0)) < 1:
        raise RecountError(
            "main_id_recount.expected_grammar_valid_distinct_ids must be positive"
        )

    quarantine = value.get("quarantine")
    if not isinstance(quarantine, dict):
        raise RecountError("config must define quarantine")
    if quarantine.get("policy") != "exact_raw_value_and_location_allowlist.v1":
        raise RecountError("unsupported quarantine policy")
    if quarantine.get("require_no_unlisted_invalid") is not True:
        raise RecountError("quarantine must require no unlisted malformed IDs")
    entries = quarantine.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RecountError("quarantine.entries must be a non-empty list")
    expected_quarantine = expected_quarantine_multiset(value)
    if len({_entry_key(entry) for entry in expected_quarantine}) != len(
        expected_quarantine
    ):
        raise RecountError("quarantine exact entries must be unique")
    if int(quarantine.get("expected_rows", -1)) != sum(
        int(entry["count"]) for entry in expected_quarantine
    ):
        raise RecountError("quarantine.expected_rows does not match its entries")

    zero = value.get("zero_id_diagnostic")
    if not isinstance(zero, dict):
        raise RecountError("config must define zero_id_diagnostic")
    if zero.get("warning_code") != ZERO_WARNING_CODE:
        raise RecountError(
            f"zero_id_diagnostic.warning_code must be {ZERO_WARNING_CODE}"
        )
    if zero.get("retained_in_grammar_valid_spool") is not True:
        raise RecountError("W0 must be retained in the grammar-valid spool")
    if zero.get("entity_validity") != "UNRESOLVED":
        raise RecountError("W0 entity validity must remain UNRESOLVED")
    if zero.get("api_404_is_exclusion_evidence") is not False:
        raise RecountError("an API 404 must not authorize W0 exclusion")
    expected_zero = expected_zero_multiset(value)
    if expected_zero[0]["raw_value"] != f"{settings['grammar_prefix']}0":
        raise RecountError("zero-ID diagnostic must freeze the grammar-valid W0 value")

    expected_body = int(settings["expected_body_rows"])
    expected_valid = int(settings["expected_grammar_valid_rows"])
    expected_null = int(settings["expected_null_rows"])
    expected_quarantine_rows = int(quarantine["expected_rows"])
    if expected_body != expected_valid + expected_null + expected_quarantine_rows:
        raise RecountError("full body/grammar-valid/quarantine counts do not conserve")

    dependency = value.get("tool_dependency")
    if not isinstance(dependency, dict):
        raise RecountError("config must freeze tool_dependency")
    dependency_relative = str(dependency.get("relative_path", ""))
    dependency_path = _repo_artifact(
        dependency_relative,
        field="tool_dependency.relative_path",
    )
    expected_dependency_sha = _validate_sha256(
        dependency.get("sha256"), field="tool_dependency.sha256"
    )
    if dependency_path != BASE_DEPENDENCY_PATH:
        raise RecountError("tool_dependency must freeze scripts/recount_openalex_g1.py")
    if sha256_file(dependency_path) != expected_dependency_sha:
        raise RecountError("frozen base dependency SHA-256 mismatch")

    evidence = value.get("parent_evidence", {}).get("blocked_zero_invalid_recount")
    if not isinstance(evidence, dict):
        raise RecountError(
            "config must freeze the blocked zero-invalid parent evidence"
        )
    if evidence.get("status") != "BLOCKED":
        raise RecountError("parent blocked recount evidence must remain BLOCKED")
    evidence_path = _repo_artifact(
        str(evidence.get("relative_path", "")),
        field="parent_evidence.blocked_zero_invalid_recount.relative_path",
    )
    evidence_sha = _validate_sha256(
        evidence.get("sha256"),
        field="parent_evidence.blocked_zero_invalid_recount.sha256",
    )
    if sha256_file(evidence_path) != evidence_sha:
        raise RecountError("frozen blocked parent evidence SHA-256 mismatch")
    raw_report_sha = _validate_sha256(
        evidence.get("raw_report_sha256"),
        field="parent_evidence.blocked_zero_invalid_recount.raw_report_sha256",
    )
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence_payload.get("provenance", {}).get("raw_report_sha256")
        != raw_report_sha
    ):
        raise RecountError("frozen original blocked report SHA-256 mismatch")
    if not isinstance(value.get("claim_boundary"), str):
        raise RecountError("config must define claim_boundary")
    if not isinstance(value.get("pending_after_grammar_valid_main_id_recount"), list):
        raise RecountError(
            "config must define pending_after_grammar_valid_main_id_recount"
        )
    return value


def resolve_root(config: dict[str, Any], explicit_root: Path | None) -> Path:
    return base.resolve_root(config, explicit_root)


def load_and_validate_parent(
    *, path: Path, root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    adapted = deepcopy(config)
    adapted["main_id_recount"]["expected_rows"] = int(
        config["main_id_recount"]["expected_body_rows"]
    )
    return base.load_and_validate_parent(path=path, root=root, config=adapted)


def _python_parse_grammar_ids(
    values: pa.Array, *, prefix: str
) -> tuple[np.ndarray, int, list[tuple[int, str]], list[int]]:
    grammar = re.compile(rf"{re.escape(prefix)}(0|[1-9][0-9]*)")
    maximum = int(np.iinfo(np.uint64).max)
    numeric: list[int] = []
    invalid: list[tuple[int, str]] = []
    zero_indices: list[int] = []
    null_count = 0
    for index, value in enumerate(values.to_pylist()):
        if value is None:
            null_count += 1
            continue
        match = grammar.fullmatch(value)
        if match is None:
            invalid.append((index, str(value)))
            continue
        parsed = int(match.group(1))
        if parsed > maximum:
            invalid.append((index, str(value)))
            continue
        numeric.append(parsed)
        if parsed == 0:
            zero_indices.append(index)
    return np.asarray(numeric, dtype="<u8"), null_count, invalid, zero_indices


def parse_grammar_valid_openalex_ids_compact(
    values: pa.Array, *, prefix: str
) -> tuple[np.ndarray, int, list[tuple[int, str]], list[int]]:
    """Return only grammar-valid uint64 IDs plus exact malformed row positions."""

    suffix = pc.utf8_slice_codeunits(values, start=len(prefix))
    try:
        numeric_arrow = pc.cast(suffix, pa.uint64(), safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return _python_parse_grammar_ids(values, prefix=prefix)

    prefix_ok = pc.fill_null(pc.starts_with(values, pattern=prefix), False)
    nonempty = pc.fill_null(pc.greater(pc.utf8_length(suffix), 0), False)
    leading_zero = pc.fill_null(
        pc.equal(pc.utf8_slice_codeunits(suffix, 0, 1), "0"), True
    )
    grammar_zero = pc.fill_null(pc.equal(suffix, "0"), False)
    decimal_normal_form = pc.or_(grammar_zero, pc.invert(leading_zero))
    shape_ok = pc.and_(prefix_ok, pc.and_(nonempty, decimal_normal_form))
    reconstructed = pc.cast(numeric_arrow, pa.string())
    exact_numeric_text = pc.fill_null(pc.equal(suffix, reconstructed), False)
    grammar_ok = pc.fill_null(
        pc.and_(shape_ok, pc.and_(pc.is_valid(numeric_arrow), exact_numeric_text)),
        False,
    )
    grammar_mask = np.asarray(grammar_ok.to_numpy(zero_copy_only=False), dtype=bool)
    numeric_all = np.asarray(
        pc.fill_null(numeric_arrow, 0).to_numpy(zero_copy_only=False), dtype="<u8"
    )
    compact = np.asarray(numeric_all[grammar_mask], dtype="<u8")
    invalid_mask_arrow = pc.and_(pc.is_valid(values), pc.invert(grammar_ok))
    invalid_mask = np.asarray(
        invalid_mask_arrow.to_numpy(zero_copy_only=False), dtype=bool
    )
    invalid_indices = np.flatnonzero(invalid_mask)
    invalid_values = values.take(pa.array(invalid_indices, type=pa.int64())).to_pylist()
    invalid = [
        (int(index), str(value))
        for index, value in zip(invalid_indices.tolist(), invalid_values, strict=True)
    ]
    zero_indices = np.flatnonzero(grammar_mask & (numeric_all == 0)).tolist()
    return compact, int(values.null_count), invalid, zero_indices


def file_scope_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    return base.file_scope_fingerprint(records)


def _frozen_repo_record(config: dict[str, Any], key: str) -> dict[str, str]:
    record = config[key]
    return {
        "relative_path": str(record["relative_path"]),
        "sha256": str(record["sha256"]),
    }


def plan_payload(
    *,
    config: dict[str, Any],
    config_path: Path,
    parent_sha256: str,
    parent_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    max_files: int | None,
    batch_size: int,
) -> dict[str, Any]:
    full_scope = max_files is None
    selected = records if full_scope else records[:max_files]
    selected_body_rows = sum(int(record["footer_rows"]) for record in selected)
    evidence = config["parent_evidence"]["blocked_zero_invalid_recount"]
    dependency = config["tool_dependency"]
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "label": str(config["label"]),
        "scope": (
            "full_grammar_valid_main_id_recount"
            if full_scope
            else "engineering_file_pilot"
        ),
        "selected_file_count": len(selected),
        "selected_body_footer_rows": selected_body_rows,
        "selected_file_fingerprint_sha256": file_scope_fingerprint(selected),
        "parent_manifest_sha256": parent_sha256,
        "parent_inventory_fingerprint_sha256": parent_manifest["root"][
            "inventory_fingerprint_sha256"
        ],
        "parent_stability_fingerprint_sha256": parent_manifest["root"][
            "stability_fingerprint_sha256"
        ],
        "blocked_zero_invalid_parent_evidence": {
            "relative_path": str(evidence["relative_path"]),
            "sha256": str(evidence["sha256"]),
            "raw_report_sha256": str(evidence["raw_report_sha256"]),
            "status": "BLOCKED",
        },
        "config_sha256": sha256_file(config_path),
        "tool_sha256": sha256_file(Path(__file__)),
        "base_dependency": {
            "relative_path": str(dependency["relative_path"]),
            "sha256": sha256_file(BASE_DEPENDENCY_PATH),
        },
        "quarantine_contract_sha256": sha256_bytes(
            canonical_json_bytes(config["quarantine"])
        ),
        "zero_id_diagnostic_contract_sha256": sha256_bytes(
            canonical_json_bytes(config["zero_id_diagnostic"])
        ),
        "table": str(config["main_id_recount"]["table"]),
        "column": str(config["main_id_recount"]["column"]),
        "id_codec": "openalex_work_id_decimal_normal_form_uint64_compact.v1",
        "batch_size": batch_size,
        "bucket_algorithm": {
            "name": "splitmix64-v1",
            "seed": SPLITMIX64_SEED,
            "bucket_count": SPLITMIX64_BUCKETS,
            "input_universe": "grammar_valid_rows_only_including_w0",
            "equality_key": "full_uint64_not_hash",
            "test_vectors": {
                "0": "e220a8397b1dcdaf",
                "1": "910a2dec89025cc1",
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
        },
    }
    payload["plan_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def checkpoint_id(*, plan_sha256: str, record: dict[str, Any], column: str) -> str:
    payload = {
        "plan_sha256": plan_sha256,
        "stage": "materialize_grammar_valid_main_id",
        "table_relative_path": record["relative_path"],
        "frozen_file_record_sha256": file_record_sha256(record),
        "column_projection": [column],
        "id_codec": "openalex_work_id_decimal_normal_form_uint64_compact.v1",
        "malformed_treatment": "exact_frozen_quarantine_no_numeric_sentinel.v1",
        "bucket_algorithm": "splitmix64-v1/256/seed-0/grammar-valid-only",
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _metric_sha256(metric: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in metric.items()
        if key not in {"metric_sha256", "elapsed_seconds"}
    }
    return sha256_bytes(canonical_json_bytes(stable))


def _validate_metric_multiset(
    entries: Any, *, relative_path: str, field: str
) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        raise RecountError(f"checkpoint {field} must be a list")
    normalized = _sorted_multiset(entries)
    if normalized != entries:
        raise RecountError(f"checkpoint {field} is not a deterministic multiset")
    if any(entry["relative_path"] != relative_path for entry in entries):
        raise RecountError(f"checkpoint {field} path does not match its source file")
    return normalized


def validate_state(state: dict[str, Any], plan: dict[str, Any]) -> None:
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RecountError("unsupported grammar-quarantine recount state schema")
    for key in (
        "plan_sha256",
        "parent_manifest_sha256",
        "config_sha256",
        "tool_sha256",
        "selected_file_fingerprint_sha256",
        "quarantine_contract_sha256",
        "zero_id_diagnostic_contract_sha256",
    ):
        if state.get(key) != plan.get(key):
            raise RecountError(f"resume refused because {key} changed")
    if state.get("base_dependency") != plan.get("base_dependency"):
        raise RecountError("resume refused because the base dependency changed")
    if state.get("blocked_zero_invalid_parent_evidence") != plan.get(
        "blocked_zero_invalid_parent_evidence"
    ):
        raise RecountError("resume refused because blocked parent evidence changed")


def validate_checkpoint_metrics(
    *,
    records: list[dict[str, Any]],
    state: dict[str, Any],
    plan: dict[str, Any],
    config: dict[str, Any],
) -> None:
    metrics = list(state.get("file_metrics", []))
    completed_files = int(state.get("completed_files", -1))
    if completed_files != len(metrics) or not 0 <= completed_files <= len(records):
        raise RecountError("checkpoint file count is inconsistent")
    body_offset = 0
    valid_offset = 0
    null_rows = 0
    quarantined_rows = 0
    unexpected_rows = 0
    zero_rows = 0
    compressed_bytes = 0
    histogram = np.zeros(SPLITMIX64_BUCKETS, dtype=np.uint64)
    quarantine_entries: list[dict[str, Any]] = []
    zero_entries: list[dict[str, Any]] = []
    expected_quarantine_keys = {
        _entry_key(entry) for entry in expected_quarantine_multiset(config)
    }
    column = str(config["main_id_recount"]["column"])
    for index, metric in enumerate(metrics):
        record = records[index]
        relative_path = str(record["relative_path"])
        expected_checkpoint = checkpoint_id(
            plan_sha256=plan["plan_sha256"], record=record, column=column
        )
        conditions = (
            int(metric.get("file_index", -1)) == index,
            metric.get("relative_path") == relative_path,
            metric.get("checkpoint_id") == expected_checkpoint,
            metric.get("frozen_file_record_sha256") == file_record_sha256(record),
            int(metric.get("body_start_row", -1)) == body_offset,
            int(metric.get("body_rows", -1)) == int(record["footer_rows"]),
            int(metric.get("grammar_valid_start_row", -1)) == valid_offset,
            int(metric.get("grammar_valid_rows", -1)) >= 0,
            metric.get("metric_sha256") == _metric_sha256(metric),
        )
        if not all(conditions):
            raise RecountError(f"checkpoint metadata mismatch: {relative_path}")
        file_histogram = list(metric.get("grammar_valid_bucket_histogram", []))
        file_valid_rows = int(metric["grammar_valid_rows"])
        if len(file_histogram) != SPLITMIX64_BUCKETS:
            raise RecountError(
                f"checkpoint grammar-valid histogram missing: {relative_path}"
            )
        if sum(int(value) for value in file_histogram) != file_valid_rows:
            raise RecountError(
                f"checkpoint grammar-valid histogram does not conserve: {relative_path}"
            )
        file_quarantine = _validate_metric_multiset(
            metric.get("quarantine_multiset"),
            relative_path=relative_path,
            field="quarantine_multiset",
        )
        if any(
            _entry_key(entry) not in expected_quarantine_keys
            for entry in file_quarantine
        ):
            raise RecountError(
                "checkpoint quarantine contains an unlisted malformed row"
            )
        file_zero = _validate_metric_multiset(
            metric.get("zero_id_diagnostic_multiset"),
            relative_path=relative_path,
            field="zero_id_diagnostic_multiset",
        )
        file_nulls = int(metric.get("null_rows", -1))
        file_quarantined = int(metric.get("quarantined_malformed_rows", -1))
        file_unexpected = int(metric.get("unexpected_malformed_rows", -1))
        file_zero_rows = int(metric.get("zero_id_rows", -1))
        file_body_rows = int(metric["body_rows"])
        if min(file_nulls, file_quarantined, file_unexpected, file_zero_rows) < 0:
            raise RecountError(f"checkpoint counters are negative: {relative_path}")
        if file_quarantined != sum(int(item["count"]) for item in file_quarantine):
            raise RecountError(f"checkpoint quarantine count mismatch: {relative_path}")
        if file_zero_rows != sum(int(item["count"]) for item in file_zero):
            raise RecountError(f"checkpoint W0 count mismatch: {relative_path}")
        if file_body_rows != (
            file_valid_rows + file_nulls + file_quarantined + file_unexpected
        ):
            raise RecountError(f"checkpoint row conservation mismatch: {relative_path}")
        body_offset += file_body_rows
        valid_offset += file_valid_rows
        null_rows += file_nulls
        quarantined_rows += file_quarantined
        unexpected_rows += file_unexpected
        zero_rows += file_zero_rows
        compressed_bytes += int(metric["id_column_compressed_bytes"])
        histogram += np.asarray(file_histogram, dtype=np.uint64)
        quarantine_entries.extend(file_quarantine)
        zero_entries.extend(file_zero)
    aggregate_conditions = (
        int(state.get("body_rows_scanned", -1)) == body_offset,
        int(state.get("grammar_valid_rows_written", -1)) == valid_offset,
        int(state.get("null_rows", -1)) == null_rows,
        int(state.get("quarantined_malformed_rows", -1)) == quarantined_rows,
        int(state.get("unexpected_malformed_rows", -1)) == unexpected_rows,
        int(state.get("zero_id_rows", -1)) == zero_rows,
        int(state.get("id_column_compressed_bytes", -1)) == compressed_bytes,
        list(state.get("grammar_valid_bucket_histogram", []))
        == [int(value) for value in histogram],
        list(state.get("quarantine_multiset", []))
        == _sorted_multiset(quarantine_entries),
        list(state.get("zero_id_diagnostic_multiset", []))
        == _sorted_multiset(zero_entries),
        int(state.get("spool_capacity_rows", -1))
        == int(plan["selected_body_footer_rows"]),
    )
    if not all(aggregate_conditions):
        raise RecountError("checkpoint aggregate counters are inconsistent")
    if body_offset != valid_offset + null_rows + quarantined_rows + unexpected_rows:
        raise RecountError("checkpoint aggregate row conservation failed")
    if int(histogram.sum()) != valid_offset:
        raise RecountError("checkpoint grammar-valid histogram conservation failed")


def _segment_sha256_from_file(path: Path, *, start_row: int, rows: int) -> str:
    digest = hashlib.sha256()
    remaining = rows * UINT64_BYTES
    with path.open("rb") as handle:
        handle.seek(start_row * UINT64_BYTES)
        while remaining:
            block = handle.read(min(16 * 1024 * 1024, remaining))
            if not block:
                raise RecountError("grammar-valid spool segment is truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def validate_completed_segments(
    *, spool_path: Path, file_metrics: list[dict[str, Any]]
) -> None:
    for metric in file_metrics:
        observed = _segment_sha256_from_file(
            spool_path,
            start_row=int(metric["grammar_valid_start_row"]),
            rows=int(metric["grammar_valid_rows"]),
        )
        if observed != metric["grammar_valid_id_sha256"]:
            raise RecountError(
                "resume refused because a completed grammar-valid spool segment "
                f"checksum changed: {metric['relative_path']}"
            )


def create_state(
    plan: dict[str, Any], capacity: dict[str, int | None]
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "phase": "materialize",
        "status": "RUNNING",
        "started_at_utc": base.utc_now(),
        "updated_at_utc": base.utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "parent_manifest_sha256": plan["parent_manifest_sha256"],
        "config_sha256": plan["config_sha256"],
        "tool_sha256": plan["tool_sha256"],
        "base_dependency": plan["base_dependency"],
        "blocked_zero_invalid_parent_evidence": plan[
            "blocked_zero_invalid_parent_evidence"
        ],
        "selected_file_fingerprint_sha256": plan["selected_file_fingerprint_sha256"],
        "quarantine_contract_sha256": plan["quarantine_contract_sha256"],
        "zero_id_diagnostic_contract_sha256": plan[
            "zero_id_diagnostic_contract_sha256"
        ],
        "completed_files": 0,
        "body_rows_scanned": 0,
        "grammar_valid_rows_written": 0,
        "spool_capacity_rows": int(plan["selected_body_footer_rows"]),
        "null_rows": 0,
        "quarantined_malformed_rows": 0,
        "unexpected_malformed_rows": 0,
        "unexpected_malformed_samples": [],
        "zero_id_rows": 0,
        "quarantine_multiset": [],
        "zero_id_diagnostic_multiset": [],
        "id_column_compressed_bytes": 0,
        "grammar_valid_bucket_histogram": [0] * SPLITMIX64_BUCKETS,
        "file_metrics": [],
        "capacity": capacity,
    }


def update_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = base.utc_now()
    atomic_write_json(path, state)


def _observation(
    *, raw_value: str, relative_path: str, row_group: int, file_row: int
) -> dict[str, Any]:
    return {
        "raw_value": raw_value,
        "relative_path": relative_path,
        "row_group": row_group,
        "file_row": file_row,
        "count": 1,
    }


def materialize_ids(
    *,
    root: Path,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    spool_path: Path,
    pause_after_new_files: int | None,
) -> bool:
    capacity_rows = int(state["spool_capacity_rows"])
    spool = np.memmap(spool_path, dtype="<u8", mode="r+", shape=(capacity_rows,))
    validate_checkpoint_metrics(records=records, state=state, plan=plan, config=config)
    validate_completed_segments(
        spool_path=spool_path, file_metrics=state["file_metrics"]
    )
    body_offset = int(state["body_rows_scanned"])
    valid_offset = int(state["grammar_valid_rows_written"])
    completed_at_start = int(state["completed_files"])
    settings = config["main_id_recount"]
    column = str(settings["column"])
    prefix = str(settings["grammar_prefix"])
    batch_size = int(plan["batch_size"])
    sample_limit = int(settings["invalid_sample_limit"])
    histogram = np.asarray(state["grammar_valid_bucket_histogram"], dtype=np.uint64)
    expected_quarantine_keys = {
        _entry_key(entry) for entry in expected_quarantine_multiset(config)
    }

    for file_index in range(completed_at_start, len(records)):
        record = records[file_index]
        relative_path = str(record["relative_path"])
        source = root / safe_relative_path(relative_path)
        file_started = time.perf_counter()
        parquet = pq.ParquetFile(source)
        compressed_bytes = base.compressed_column_bytes(parquet, column)
        file_body_rows = 0
        file_valid_rows = 0
        file_nulls = 0
        file_quarantined = 0
        file_unexpected = 0
        file_zero_rows = 0
        file_unexpected_samples: list[str] = []
        file_quarantine_entries: list[dict[str, Any]] = []
        file_zero_entries: list[dict[str, Any]] = []
        file_digest = hashlib.sha256()
        file_histogram = np.zeros(SPLITMIX64_BUCKETS, dtype=np.uint64)
        body_start = body_offset
        valid_start = valid_offset

        for row_group in range(parquet.metadata.num_row_groups):
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group],
                columns=[column],
                use_threads=True,
            ):
                values = batch.column(0)
                batch_body_start = file_body_rows
                numeric, null_count, malformed, zero_indices = (
                    parse_grammar_valid_openalex_ids_compact(values, prefix=prefix)
                )
                if valid_offset + len(numeric) > capacity_rows:
                    raise RecountError("grammar-valid spool exceeds body-row capacity")
                spool[valid_offset : valid_offset + len(numeric)] = numeric
                file_digest.update(np.ascontiguousarray(numeric).view(np.uint8))
                file_histogram += splitmix64_bucket_histogram(numeric)
                for local_index, raw_value in malformed:
                    event = _observation(
                        raw_value=raw_value,
                        relative_path=relative_path,
                        row_group=row_group,
                        file_row=batch_body_start + local_index,
                    )
                    if _entry_key(event) in expected_quarantine_keys:
                        file_quarantine_entries.append(event)
                        file_quarantined += 1
                    else:
                        file_unexpected += 1
                        if len(file_unexpected_samples) < sample_limit:
                            file_unexpected_samples.append(
                                json.dumps(event, sort_keys=True)
                            )
                for local_index in zero_indices:
                    file_zero_entries.append(
                        _observation(
                            raw_value=f"{prefix}0",
                            relative_path=relative_path,
                            row_group=row_group,
                            file_row=batch_body_start + int(local_index),
                        )
                    )
                    file_zero_rows += 1
                batch_rows = len(values)
                body_offset += batch_rows
                valid_offset += len(numeric)
                file_body_rows += batch_rows
                file_valid_rows += len(numeric)
                file_nulls += null_count

        expected_file_rows = int(record["footer_rows"])
        if file_body_rows != expected_file_rows:
            raise RecountError(
                f"data-page rows differ from footer rows for {relative_path}: "
                f"{file_body_rows} != {expected_file_rows}"
            )
        if file_body_rows != (
            file_valid_rows + file_nulls + file_quarantined + file_unexpected
        ):
            raise RecountError(f"file row conservation failed: {relative_path}")
        if file_valid_rows:
            base.flush_memmap_range(
                spool,
                start_row=valid_start,
                end_row=valid_offset,
                path=spool_path,
            )
        else:
            base.fsync_file(spool_path)
        file_quarantine_multiset = _sorted_multiset(file_quarantine_entries)
        file_zero_multiset = _sorted_multiset(file_zero_entries)
        metric = {
            "file_index": file_index,
            "relative_path": relative_path,
            "checkpoint_id": checkpoint_id(
                plan_sha256=plan["plan_sha256"], record=record, column=column
            ),
            "frozen_file_record_sha256": file_record_sha256(record),
            "body_start_row": body_start,
            "body_rows": file_body_rows,
            "grammar_valid_start_row": valid_start,
            "grammar_valid_rows": file_valid_rows,
            "null_rows": file_nulls,
            "quarantined_malformed_rows": file_quarantined,
            "quarantine_multiset": file_quarantine_multiset,
            "unexpected_malformed_rows": file_unexpected,
            "unexpected_malformed_samples": file_unexpected_samples,
            "zero_id_rows": file_zero_rows,
            "zero_id_diagnostic_multiset": file_zero_multiset,
            "id_column_compressed_bytes": compressed_bytes,
            "grammar_valid_bucket_histogram": [int(value) for value in file_histogram],
            "grammar_valid_id_sha256": file_digest.hexdigest(),
            "elapsed_seconds": time.perf_counter() - file_started,
        }
        metric["metric_sha256"] = _metric_sha256(metric)
        histogram += file_histogram
        state["file_metrics"].append(metric)
        state["completed_files"] = file_index + 1
        state["body_rows_scanned"] = body_offset
        state["grammar_valid_rows_written"] = valid_offset
        state["null_rows"] = int(state["null_rows"]) + file_nulls
        state["quarantined_malformed_rows"] = (
            int(state["quarantined_malformed_rows"]) + file_quarantined
        )
        state["unexpected_malformed_rows"] = (
            int(state["unexpected_malformed_rows"]) + file_unexpected
        )
        state["unexpected_malformed_samples"] = (
            list(state["unexpected_malformed_samples"]) + file_unexpected_samples
        )[:sample_limit]
        state["zero_id_rows"] = int(state["zero_id_rows"]) + file_zero_rows
        state["quarantine_multiset"] = _sorted_multiset(
            [*state["quarantine_multiset"], *file_quarantine_multiset]
        )
        state["zero_id_diagnostic_multiset"] = _sorted_multiset(
            [*state["zero_id_diagnostic_multiset"], *file_zero_multiset]
        )
        state["id_column_compressed_bytes"] = (
            int(state["id_column_compressed_bytes"]) + compressed_bytes
        )
        state["grammar_valid_bucket_histogram"] = [int(value) for value in histogram]
        update_state(state_path, state)

        if pause_after_new_files is not None:
            newly_completed = int(state["completed_files"]) - completed_at_start
            if newly_completed >= pause_after_new_files and file_index + 1 < len(
                records
            ):
                state["status"] = "PAUSED"
                update_state(state_path, state)
                del spool
                return False

    if body_offset != int(plan["selected_body_footer_rows"]):
        raise RecountError(
            "data-page body total differs from the selected footer total"
        )
    if body_offset != (
        valid_offset
        + int(state["null_rows"])
        + int(state["quarantined_malformed_rows"])
        + int(state["unexpected_malformed_rows"])
    ):
        raise RecountError("aggregate body/grammar-valid row conservation failed")
    if int(histogram.sum()) != valid_offset:
        raise RecountError("grammar-valid SplitMix64 histogram does not conserve rows")
    del spool
    state["phase"] = "compact"
    state["status"] = "RUNNING"
    update_state(state_path, state)
    return True


def compact_spool(
    *,
    state: dict[str, Any],
    state_path: Path,
    spool_path: Path,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    config: dict[str, Any],
) -> None:
    validate_checkpoint_metrics(records=records, state=state, plan=plan, config=config)
    validate_completed_segments(
        spool_path=spool_path, file_metrics=state["file_metrics"]
    )
    capacity_bytes = int(state["spool_capacity_rows"]) * UINT64_BYTES
    valid_bytes = int(state["grammar_valid_rows_written"]) * UINT64_BYTES
    observed_bytes = int(spool_path.stat().st_size)
    if observed_bytes not in {capacity_bytes, valid_bytes}:
        raise RecountError("grammar-valid spool has an invalid pre-compaction size")
    if observed_bytes != valid_bytes:
        with spool_path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    state["phase"] = "sort"
    update_state(state_path, state)


def exact_grammar_valid_statistics(
    sorted_ids: np.ndarray, *, chunk_rows: int
) -> dict[str, int | None]:
    statistics = base.exact_duplicate_statistics(sorted_ids, chunk_rows=chunk_rows)
    return {
        "grammar_valid_distinct_ids": statistics["distinct_ids"],
        "grammar_valid_duplicate_row_excess": statistics["duplicate_row_excess"],
        "grammar_valid_duplicate_id_groups": statistics["duplicate_id_groups"],
        "grammar_valid_max_multiplicity": statistics["max_multiplicity"],
        "grammar_valid_min_numeric_id": statistics["min_numeric_id"],
        "grammar_valid_max_numeric_id": statistics["max_numeric_id"],
    }


def _contract_satisfied(
    *, config: dict[str, Any], state: dict[str, Any], full_scope: bool
) -> bool | None:
    if not full_scope:
        return None
    return (
        state["quarantine_multiset"] == expected_quarantine_multiset(config)
        and int(state["quarantined_malformed_rows"])
        == int(config["quarantine"]["expected_rows"])
        and int(state["unexpected_malformed_rows"]) == 0
        and int(state["null_rows"])
        == int(config["main_id_recount"]["expected_null_rows"])
    )


def quarantine_artifact_payload(
    *, config: dict[str, Any], plan: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    full_scope = plan["scope"] == "full_grammar_valid_main_id_recount"
    return {
        "schema_version": QUARANTINE_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "scope": plan["scope"],
        "policy": config["quarantine"]["policy"],
        "source_was_modified": False,
        "numeric_sentinel_used": False,
        "expected_malformed_quarantine_multiset": expected_quarantine_multiset(config),
        "observed_malformed_quarantine_multiset": state["quarantine_multiset"],
        "observed_quarantined_malformed_rows": int(state["quarantined_malformed_rows"]),
        "unexpected_malformed_rows": int(state["unexpected_malformed_rows"]),
        "unexpected_malformed_samples": list(state["unexpected_malformed_samples"]),
        "full_scope_contract_satisfied": _contract_satisfied(
            config=config, state=state, full_scope=full_scope
        ),
        "zero_id_diagnostic": {
            "warning_code": ZERO_WARNING_CODE,
            "entity_validity": "UNRESOLVED",
            "retained_in_grammar_valid_spool": True,
            "api_404_is_exclusion_evidence": False,
            "expected_multiset": expected_zero_multiset(config),
            "observed_multiset": state["zero_id_diagnostic_multiset"],
            "observed_rows": int(state["zero_id_rows"]),
        },
    }


def sort_and_count(
    *,
    state: dict[str, Any],
    state_path: Path,
    raw_path: Path,
    sorted_path: Path,
    quarantine_path: Path,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, int | None], dict[str, str], float]:
    started = time.perf_counter()
    validate_checkpoint_metrics(records=records, state=state, plan=plan, config=config)
    validate_completed_segments(spool_path=raw_path, file_metrics=state["file_metrics"])
    rows = int(state["grammar_valid_rows_written"])
    if raw_path.stat().st_size != rows * UINT64_BYTES:
        raise RecountError("grammar-valid compact spool size mismatch")
    if rows:
        raw = np.memmap(raw_path, dtype="<u8", mode="r", shape=(rows,))
        values = np.array(raw, dtype="<u8", copy=True)
        del raw
    else:
        values = np.empty(0, dtype="<u8")
    values.sort(kind="quicksort")
    base.preallocate(sorted_path, rows * UINT64_BYTES)
    if rows:
        sorted_ids = np.memmap(sorted_path, dtype="<u8", mode="r+", shape=(rows,))
        sorted_ids[:] = values
        sorted_ids.flush()
        base.fsync_file(sorted_path)
        statistics = exact_grammar_valid_statistics(
            sorted_ids,
            chunk_rows=int(config["main_id_recount"]["duplicate_scan_chunk_rows"]),
        )
        del sorted_ids
    else:
        base.fsync_file(sorted_path)
        statistics = exact_grammar_valid_statistics(
            values,
            chunk_rows=int(config["main_id_recount"]["duplicate_scan_chunk_rows"]),
        )
    del values
    atomic_write_json(
        quarantine_path,
        quarantine_artifact_payload(config=config, plan=plan, state=state),
    )
    artifact_hashes = {
        raw_path.name: sha256_file(raw_path),
        sorted_path.name: sha256_file(sorted_path),
        quarantine_path.name: sha256_file(quarantine_path),
    }
    elapsed = time.perf_counter() - started
    state["phase"] = "report"
    state["artifact_sha256"] = artifact_hashes
    state["exact_grammar_valid_statistics"] = statistics
    state["sort_count_quarantine_and_checksum_seconds"] = elapsed
    update_state(state_path, state)
    return statistics, artifact_hashes, elapsed


def validate_artifact_hashes(
    *, state: dict[str, Any], artifact_paths: dict[str, Path]
) -> None:
    hashes = dict(state.get("artifact_sha256", {}))
    if set(hashes) != set(artifact_paths):
        raise RecountError("complete artifact commitment set is incomplete or unknown")
    for name, expected_sha256 in hashes.items():
        path = artifact_paths[name]
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise RecountError(f"recount artifact checksum mismatch: {name}")


def validate_complete_outputs(
    *,
    state: dict[str, Any],
    report_path: Path,
    markdown_path: Path,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    missing = [
        path.name
        for path in (report_path, markdown_path, *artifact_paths.values())
        if not path.is_file()
    ]
    if missing:
        raise RecountError(f"complete recount outputs are missing: {missing}")
    if sha256_file(report_path) != state.get("report_sha256"):
        raise RecountError("complete recount report checksum mismatch")
    if sha256_file(markdown_path) != state.get("markdown_sha256"):
        raise RecountError("complete recount Markdown checksum mismatch")
    validate_artifact_hashes(state=state, artifact_paths=artifact_paths)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("plan_sha256") != state.get("plan_sha256"):
        raise RecountError("complete recount report plan mismatch")
    if report.get("status") != state.get("status"):
        raise RecountError("complete recount report status mismatch")
    quarantine = json.loads(
        artifact_paths["malformed_quarantine.json"].read_text(encoding="utf-8")
    )
    if quarantine.get("plan_sha256") != state.get("plan_sha256"):
        raise RecountError("complete quarantine artifact plan mismatch")
    if quarantine.get("observed_malformed_quarantine_multiset") != state.get(
        "quarantine_multiset"
    ):
        raise RecountError("complete quarantine artifact observation mismatch")
    return report


def evaluate_checks(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
    statistics: dict[str, int | None],
) -> list[dict[str, Any]]:
    settings = config["main_id_recount"]
    full_scope = plan["scope"] == "full_grammar_valid_main_id_recount"
    checks: list[dict[str, Any]] = []

    def add(
        name: str, observed: Any, expected: Any, *, full_only: bool = False
    ) -> None:
        status = (
            "PENDING"
            if full_only and not full_scope
            else "PASS"
            if observed == expected
            else "BLOCK"
        )
        checks.append(
            {"name": name, "status": status, "observed": observed, "expected": expected}
        )

    add(
        "body_rows_scanned",
        int(state["body_rows_scanned"]),
        (
            int(settings["expected_body_rows"])
            if full_scope
            else int(plan["selected_body_footer_rows"])
        ),
    )
    add(
        "null_main_work_ids",
        int(state["null_rows"]),
        int(settings["expected_null_rows"]),
    )
    add(
        "unexpected_malformed_main_work_ids",
        int(state["unexpected_malformed_rows"]),
        0,
    )
    add(
        "body_to_grammar_valid_row_conservation",
        int(state["body_rows_scanned"]),
        int(state["grammar_valid_rows_written"])
        + int(state["null_rows"])
        + int(state["quarantined_malformed_rows"])
        + int(state["unexpected_malformed_rows"]),
    )
    add(
        "frozen_malformed_quarantine_row_count",
        int(state["quarantined_malformed_rows"]),
        int(config["quarantine"]["expected_rows"]),
        full_only=True,
    )
    add(
        "frozen_malformed_quarantine_exact_multiset",
        state["quarantine_multiset"],
        expected_quarantine_multiset(config),
        full_only=True,
    )
    add(
        "grammar_valid_rows",
        int(state["grammar_valid_rows_written"]),
        int(settings["expected_grammar_valid_rows"]),
        full_only=True,
    )
    add(
        "exact_distinct_grammar_valid_main_work_ids",
        statistics["grammar_valid_distinct_ids"],
        int(settings["expected_grammar_valid_distinct_ids"]),
        full_only=True,
    )
    add(
        "duplicate_grammar_valid_main_work_row_excess",
        statistics["grammar_valid_duplicate_row_excess"],
        int(settings["expected_grammar_valid_duplicate_row_excess"]),
        full_only=True,
    )
    add(
        "frozen_w0_diagnostic_row_count",
        int(state["zero_id_rows"]),
        sum(int(item["count"]) for item in expected_zero_multiset(config)),
        full_only=True,
    )
    add(
        "frozen_w0_diagnostic_exact_multiset",
        state["zero_id_diagnostic_multiset"],
        expected_zero_multiset(config),
        full_only=True,
    )
    add(
        "grammar_valid_bucket_histogram_conserves_rows",
        sum(int(value) for value in state["grammar_valid_bucket_histogram"]),
        int(state["grammar_valid_rows_written"]),
    )
    return checks


def _check_pass(checks: list[dict[str, Any]], name: str) -> bool:
    return next(item for item in checks if item["name"] == name)["status"] == "PASS"


def build_report(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
    statistics: dict[str, int | None],
    artifact_hashes: dict[str, str],
    sort_seconds: float,
    parent_stable_after: bool,
    invocation_elapsed_seconds: float,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    checks = evaluate_checks(
        config=config, plan=plan, state=state, statistics=statistics
    )
    checks.append(
        {
            "name": "parent_inventory_stable_after_recount",
            "status": "PASS" if parent_stable_after else "BLOCK",
            "observed": parent_stable_after,
            "expected": True,
        }
    )
    blocks = [item for item in checks if item["status"] == "BLOCK"]
    full_scope = plan["scope"] == "full_grammar_valid_main_id_recount"
    pass_body = full_scope and all(
        _check_pass(checks, name)
        for name in (
            "body_rows_scanned",
            "body_to_grammar_valid_row_conservation",
            "parent_inventory_stable_after_recount",
        )
    )
    quarantine_satisfied = full_scope and all(
        _check_pass(checks, name)
        for name in (
            "null_main_work_ids",
            "unexpected_malformed_main_work_ids",
            "frozen_malformed_quarantine_row_count",
            "frozen_malformed_quarantine_exact_multiset",
        )
    )
    pass_grammar = full_scope and not blocks
    status = (
        "BLOCKED"
        if blocks
        else "PASS_GRAMMAR_VALID_MAIN_ID_RECOUNT_WITH_FROZEN_QUARANTINE"
        if full_scope
        else "PASS_ENGINEERING_PILOT"
    )
    pilot_bucket = int(plan["parent_inventory_fingerprint_sha256"][:16], 16) % 256
    usage = resource.getrusage(resource.RUSAGE_SELF)
    artifact_bytes = {
        name: int(path.stat().st_size)
        for name, path in artifact_paths.items()
        if name in artifact_hashes
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": str(config["label"]),
        "status": status,
        "scope": plan["scope"],
        "pass_body_recount": pass_body,
        "quarantine_contract_satisfied": quarantine_satisfied,
        "pass_grammar_valid_main_id_recount": pass_grammar,
        "source_key_integrity_clean": False,
        "entity_validity_established": False,
        "parent_content_identity_verified": False,
        "g_minus_1_data_readiness": "PENDING",
        "g1_empirical_signature": "NOT_RUN",
        "pass_g1": False,
        "source_root_id": str(config["parent_inventory"]["source_root_id"]),
        "parent_manifest_sha256": plan["parent_manifest_sha256"],
        "parent_inventory_fingerprint_sha256": plan[
            "parent_inventory_fingerprint_sha256"
        ],
        "parent_stability_fingerprint_sha256": plan[
            "parent_stability_fingerprint_sha256"
        ],
        "blocked_zero_invalid_parent_evidence": plan[
            "blocked_zero_invalid_parent_evidence"
        ],
        "plan_sha256": plan["plan_sha256"],
        "creator": {
            **base.git_metadata(),
            "tool_sha256": plan["tool_sha256"],
            "base_dependency": plan["base_dependency"],
        },
        "started_at_utc": state["started_at_utc"],
        "finished_at_utc": base.utc_now(),
        "execution": {
            "selected_files": int(plan["selected_file_count"]),
            "batch_size": int(plan["batch_size"]),
            "materialize_file_seconds": sum(
                float(item["elapsed_seconds"]) for item in state["file_metrics"]
            ),
            "sort_count_quarantine_and_checksum_seconds": sort_seconds,
            "current_invocation_elapsed_seconds": invocation_elapsed_seconds,
            "id_column_compressed_bytes": int(state["id_column_compressed_bytes"]),
            "spool_dtype": "uint64-little-endian-compact-grammar-valid",
            "artifact_bytes": artifact_bytes,
            "artifact_bytes_total": sum(artifact_bytes.values()),
            "peak_rss_kib": int(usage.ru_maxrss),
            "process_user_cpu_seconds": float(usage.ru_utime),
            "process_system_cpu_seconds": float(usage.ru_stime),
            "process_input_blocks": int(usage.ru_inblock),
            "process_output_blocks": int(usage.ru_oublock),
            "runtime": dict(plan["runtime"]),
            "path_redacted": True,
        },
        "counts": {
            "body_rows_scanned": int(state["body_rows_scanned"]),
            "grammar_valid_rows": int(state["grammar_valid_rows_written"]),
            "null_rows": int(state["null_rows"]),
            "quarantined_malformed_rows": int(state["quarantined_malformed_rows"]),
            "unexpected_malformed_rows": int(state["unexpected_malformed_rows"]),
            "zero_id_rows_retained": int(state["zero_id_rows"]),
            **statistics,
        },
        "diagnostic_buckets": {
            "algorithm": "splitmix64-v1",
            "input_universe": "grammar_valid_rows_only_including_w0",
            "seed": SPLITMIX64_SEED,
            "bucket_count": SPLITMIX64_BUCKETS,
            "histogram": [
                int(value) for value in state["grammar_valid_bucket_histogram"]
            ],
            "pilot_bucket": pilot_bucket,
            "pilot_bucket_rows": int(
                state["grammar_valid_bucket_histogram"][pilot_bucket]
            ),
            "hash_is_equality_key": False,
        },
        "malformed_quarantine": {
            "policy": config["quarantine"]["policy"],
            "expected_multiset": expected_quarantine_multiset(config),
            "observed_multiset": state["quarantine_multiset"],
            "source_was_modified": False,
            "numeric_sentinel_used": False,
        },
        "zero_id_diagnostic": {
            "warning_code": ZERO_WARNING_CODE,
            "status": "WARN",
            "raw_value": expected_zero_multiset(config)[0]["raw_value"],
            "expected_multiset": expected_zero_multiset(config),
            "observed_multiset": state["zero_id_diagnostic_multiset"],
            "retained_in_grammar_valid_spool": True,
            "entity_validity": "UNRESOLVED",
            "api_404_is_exclusion_evidence": False,
            "sensitivity_status": "PENDING",
        },
        "warnings": [
            {
                "code": ZERO_WARNING_CODE,
                "status": "WARN",
                "message": (
                    "W0 is grammar-valid and retained, but its entity validity is "
                    "unresolved; an API 404 alone does not authorize exclusion."
                ),
            },
            {
                "code": "WARN_PARENT_CONTENT_SHA256_PENDING",
                "status": "WARN",
                "message": (
                    "The frozen parent currently binds paths, sizes, and mtimes; "
                    "full-root content SHA-256 remains pending, so inventory-era "
                    "byte identity is not established."
                ),
            },
        ],
        "unexpected_malformed_samples": list(state["unexpected_malformed_samples"]),
        "artifact_sha256": artifact_hashes,
        "checks": checks,
        "pending_after_grammar_valid_main_id_recount": list(
            config["pending_after_grammar_valid_main_id_recount"]
        ),
        "claim_boundary": str(config["claim_boundary"]),
    }


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# OpenAlex G-1 Grammar-Valid Main-ID Recount",
        "",
        f"- Status: `{report['status']}`",
        f"- Scope: `{report['scope']}`",
        f"- PASS body recount: `{str(report['pass_body_recount']).lower()}`",
        "- Quarantine contract satisfied: "
        f"`{str(report['quarantine_contract_satisfied']).lower()}`",
        "- PASS grammar-valid main-ID recount: "
        f"`{str(report['pass_grammar_valid_main_id_recount']).lower()}`",
        f"- Source key integrity clean: `{str(report['source_key_integrity_clean']).lower()}`",
        f"- Entity validity established: `{str(report['entity_validity_established']).lower()}`",
        "- Parent content identity verified: "
        f"`{str(report['parent_content_identity_verified']).lower()}`",
        f"- G-1 data readiness: `{report['g_minus_1_data_readiness']}`",
        f"- Empirical G1 signature: `{report['g1_empirical_signature']}`",
        f"- PASS empirical G1: `{str(report['pass_g1']).lower()}`",
        f"- Warning: `{ZERO_WARNING_CODE}`",
        "- Warning: `WARN_PARENT_CONTENT_SHA256_PENDING`",
        "",
        "## Exact Counts",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Body rows scanned | {counts['body_rows_scanned']} |",
        f"| Grammar-valid rows | {counts['grammar_valid_rows']} |",
        f"| Quarantined malformed rows | {counts['quarantined_malformed_rows']} |",
        f"| Unexpected malformed rows | {counts['unexpected_malformed_rows']} |",
        f"| W0 rows retained | {counts['zero_id_rows_retained']} |",
        f"| Grammar-valid distinct IDs | {counts['grammar_valid_distinct_ids']} |",
        "| Grammar-valid duplicate row excess | "
        f"{counts['grammar_valid_duplicate_row_excess']} |",
        "",
        "## Checks",
        "",
        "| Check | Status | Observed | Expected |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["checks"]:
        observed = json.dumps(item["observed"], sort_keys=True)
        expected = json.dumps(item["expected"], sort_keys=True)
        lines.append(
            f"| `{item['name']}` | {item['status']} | `{observed}` | `{expected}` |"
        )
    lines.extend(
        [
            "",
            "## W0 Diagnostic",
            "",
            "`W0` remains in the grammar-valid spool. Its entity validity and "
            "exclusion sensitivity remain pending; an API 404 is not treated as "
            "exclusion evidence.",
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def _expected_spool_sizes(state: dict[str, Any]) -> set[int]:
    capacity = int(state["spool_capacity_rows"]) * UINT64_BYTES
    compact = int(state["grammar_valid_rows_written"]) * UINT64_BYTES
    if state["phase"] == "materialize":
        return {capacity}
    if state["phase"] == "compact":
        return {capacity, compact}
    return {compact}


def run_recount(
    *,
    config_path: Path,
    parent_path: Path,
    root: Path,
    output_dir: Path,
    max_files: int | None,
    batch_size: int | None,
    pause_after_new_files: int | None,
) -> dict[str, Any]:
    invocation_started = time.perf_counter()
    config = load_config(config_path)
    parent, parent_sha256, all_records = load_and_validate_parent(
        path=parent_path, root=root, config=config
    )
    if max_files is not None and not 1 <= max_files < len(all_records):
        raise RecountError("--max-files must select at least one but not all files")
    records = all_records if max_files is None else all_records[:max_files]
    selected_batch_size = int(batch_size or config["main_id_recount"]["batch_size"])
    if selected_batch_size < 1:
        raise RecountError("batch size must be positive")
    plan = plan_payload(
        config=config,
        config_path=config_path,
        parent_sha256=parent_sha256,
        parent_manifest=parent,
        records=all_records,
        max_files=max_files,
        batch_size=selected_batch_size,
    )
    base.validate_output_directory_path(output_dir)
    base.validate_output_containment(root=root, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base.validate_output_directory_path(output_dir)

    lock_path = output_dir / "recount.lock"
    plan_path = output_dir / "recount_plan.json"
    state_path = output_dir / "recount_state.json"
    raw_path = output_dir / "main_ids.grammar_valid.unsorted.u64"
    sorted_path = output_dir / "main_ids.grammar_valid.sorted.u64"
    quarantine_path = output_dir / "malformed_quarantine.json"
    report_path = output_dir / "grammar_valid_main_id_recount.json"
    markdown_path = output_dir / "grammar_valid_main_id_recount.md"
    artifact_paths = {
        raw_path.name: raw_path,
        sorted_path.name: sorted_path,
        quarantine_path.name: quarantine_path,
    }
    atomic_targets = (
        plan_path,
        state_path,
        quarantine_path,
        report_path,
        markdown_path,
    )
    managed_paths = (
        lock_path,
        plan_path,
        state_path,
        raw_path,
        sorted_path,
        quarantine_path,
        report_path,
        markdown_path,
        *(path.with_suffix(path.suffix + ".partial") for path in atomic_targets),
    )
    for path in managed_paths:
        base.validate_managed_output_path(path)

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecountError("another recount process holds the output lock") from exc

        if plan_path.exists():
            frozen_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            claimed_sha = frozen_plan.pop("plan_sha256", None)
            observed_sha = sha256_bytes(canonical_json_bytes(frozen_plan))
            frozen_plan["plan_sha256"] = claimed_sha
            if claimed_sha != observed_sha or frozen_plan != plan:
                raise RecountError("output directory contains a different recount plan")
        else:
            atomic_write_json(plan_path, plan)

        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            validate_state(state, plan)
            validate_checkpoint_metrics(
                records=records, state=state, plan=plan, config=config
            )
            if state.get("phase") == "complete":
                return validate_complete_outputs(
                    state=state,
                    report_path=report_path,
                    markdown_path=markdown_path,
                    artifact_paths=artifact_paths,
                )
            if (
                not raw_path.is_file()
                or raw_path.stat().st_size not in _expected_spool_sizes(state)
            ):
                raise RecountError("resume spool is missing or has the wrong size")
            validate_completed_segments(
                spool_path=raw_path, file_metrics=state["file_metrics"]
            )
            state["status"] = "RUNNING"
            update_state(state_path, state)
        else:
            stale = [
                path.name
                for path in (*artifact_paths.values(), report_path, markdown_path)
                if path.exists()
            ]
            if stale:
                raise RecountError(
                    f"state is missing but recount outputs already exist: {stale}"
                )
            selected_body_rows = int(plan["selected_body_footer_rows"])
            capacity = base.ensure_capacity(output_dir, selected_body_rows)
            base.preallocate(raw_path, selected_body_rows * UINT64_BYTES)
            state = create_state(plan, capacity)
            update_state(state_path, state)

        if state["phase"] == "materialize":
            finished = materialize_ids(
                root=root,
                records=records,
                plan=plan,
                config=config,
                state=state,
                state_path=state_path,
                spool_path=raw_path,
                pause_after_new_files=pause_after_new_files,
            )
            if not finished:
                return {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "status": "PAUSED",
                    "scope": plan["scope"],
                    "pass_body_recount": False,
                    "quarantine_contract_satisfied": False,
                    "pass_grammar_valid_main_id_recount": False,
                    "source_key_integrity_clean": False,
                    "entity_validity_established": False,
                    "parent_content_identity_verified": False,
                    "g_minus_1_data_readiness": "PENDING",
                    "g1_empirical_signature": "NOT_RUN",
                    "pass_g1": False,
                    "completed_files": int(state["completed_files"]),
                    "selected_files": len(records),
                    "body_rows_scanned": int(state["body_rows_scanned"]),
                    "grammar_valid_rows_written": int(
                        state["grammar_valid_rows_written"]
                    ),
                    "claim_boundary": str(config["claim_boundary"]),
                }

        if state["phase"] == "compact":
            compact_spool(
                state=state,
                state_path=state_path,
                spool_path=raw_path,
                records=records,
                plan=plan,
                config=config,
            )

        validate_checkpoint_metrics(
            records=records, state=state, plan=plan, config=config
        )
        if state["phase"] == "sort":
            statistics, artifact_hashes, sort_seconds = sort_and_count(
                state=state,
                state_path=state_path,
                raw_path=raw_path,
                sorted_path=sorted_path,
                quarantine_path=quarantine_path,
                records=records,
                plan=plan,
                config=config,
            )
        elif state["phase"] == "report":
            statistics = dict(state["exact_grammar_valid_statistics"])
            artifact_hashes = dict(state["artifact_sha256"])
            sort_seconds = float(state["sort_count_quarantine_and_checksum_seconds"])
            validate_artifact_hashes(state=state, artifact_paths=artifact_paths)
        else:
            raise RecountError(f"unsupported recount phase: {state['phase']}")

        try:
            end_parent, end_parent_sha256, _ = load_and_validate_parent(
                path=parent_path, root=root, config=config
            )
            parent_stable_after = (
                end_parent_sha256 == plan["parent_manifest_sha256"]
                and end_parent["root"]["stability_fingerprint_sha256"]
                == plan["parent_stability_fingerprint_sha256"]
            )
        except RecountError:
            parent_stable_after = False
        report = build_report(
            config=config,
            plan=plan,
            state=state,
            statistics=statistics,
            artifact_hashes=artifact_hashes,
            sort_seconds=sort_seconds,
            parent_stable_after=parent_stable_after,
            invocation_elapsed_seconds=time.perf_counter() - invocation_started,
            artifact_paths=artifact_paths,
        )
        atomic_write_json(report_path, report)
        atomic_write_text(markdown_path, render_markdown(report))
        state["phase"] = "complete"
        state["status"] = report["status"]
        state["report_sha256"] = sha256_file(report_path)
        state["markdown_sha256"] = sha256_file(markdown_path)
        update_state(state_path, state)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pause-after-new-files", type=int)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    for name in ("max_files", "batch_size", "pause_after_new_files"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        root = resolve_root(config, args.root)
        report = run_recount(
            config_path=config_path,
            parent_path=args.inventory_manifest.expanduser().resolve(),
            root=root,
            output_dir=args.output_dir.expanduser(),
            max_files=args.max_files,
            batch_size=args.batch_size,
            pause_after_new_files=args.pause_after_new_files,
        )
    except (OSError, ValueError, RecountError, json.JSONDecodeError) as exc:
        print(f"recount failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "pass_grammar_valid_main_id_recount": report[
                    "pass_grammar_valid_main_id_recount"
                ],
                "pass_g1": report["pass_g1"],
                "body_rows": report.get("counts", {}).get(
                    "body_rows_scanned", report.get("body_rows_scanned")
                ),
                "grammar_valid_rows": report.get("counts", {}).get(
                    "grammar_valid_rows", report.get("grammar_valid_rows_written")
                ),
            },
            sort_keys=True,
        )
    )
    if args.require_pass and not report["pass_grammar_valid_main_id_recount"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
