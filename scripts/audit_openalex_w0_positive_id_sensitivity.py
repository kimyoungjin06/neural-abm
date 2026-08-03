#!/usr/bin/env python3
"""Quantify the frozen positive-work-ID sensitivity for the retained W0 row.

This bounded component revalidates the current W0 projection in the main-work
and authorship files and propagates its one-row exclusion through already
frozen profile margins.  It does not decide whether W0 is a valid entity and
does not pass G-1, select fields, or estimate an effect.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, BinaryIO, Iterator, Mapping

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/openalex_g1_w0_positive_id_sensitivity_v0.yaml"
)
CONFIG_SCHEMA_VERSION = "openalex_g1_w0_positive_id_sensitivity_config.v1"
REPORT_SCHEMA_VERSION = "openalex_g1_w0_positive_id_sensitivity_report.v1"
OUTPUTS_SCHEMA_VERSION = "openalex_g1_w0_positive_id_sensitivity_outputs.v1"
SUCCESS_STATUS = "PASS_W0_POSITIVE_ID_EXCLUSION_SENSITIVITY_COMPONENT"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
W0 = "https://openalex.org/W0"
REQUIRED_GATE_STATUS = {
    "positive_id_exclusion_sensitivity": "PASS",
    "nonnull_field_margins": "UNCHANGED",
    "current_main_w0_projection": "PASS",
    "current_topic_w0_incidence": "PASS_REVALIDATED",
    "frozen_source_propagation_ledger": "PASS_REVALIDATED",
    "w_zero_entity_validity": "UNRESOLVED",
    "parent_content_identity": "UNVERIFIED",
    "author_entity_validity": "PENDING",
    "candidate_eligibility": "PENDING",
    "publication_date_validity": "PENDING",
    "g_minus_1_data_readiness": "PENDING",
    "g0_measurement_freeze": "PENDING",
    "empirical_g1_signature": "NOT_RUN",
    "pass_g1": False,
}
CLAIM_BOUNDARY = (
    "A pass quantifies the predeclared positive-work-ID sensitivity for the "
    "single retained W0 record and propagates its one-row removal through the "
    "already frozen cross-field profile margins. It does not decide whether W0 "
    "denotes a real or invalid entity, establish historical or current full-root "
    "byte identity, validate publication dates or authors, select fields or "
    "candidate researchers, pass G-1 or G0, run empirical G1, or establish an "
    "effect."
)
REQUIRED_POSITIVE_ID_SENSITIVITY = {
    "retained_work_id_regex": r"^https://openalex[.]org/W[1-9][0-9]*$",
    "excluded_identifier": W0,
    "expected_current_main_rows": 1,
    "expected_current_authorship_rows": 1,
    "expected_main_projection": {
        "ids__openalex": W0,
        "ids__mag": "0",
        "publication_date": "2024-01-01",
        "publication_year": 2024.0,
        "authors_count": 1,
        "primary_topic__id": None,
        "primary_topic__field__id": None,
        "primary_topic__domain__id": None,
    },
    "expected_authorship_projection": {
        "authorships__author_position": "first",
        "authorships__author__id": None,
        "authorships__author__orcid": None,
    },
    "expected_work_profile_status": "none_usable",
    "expected_declared_authors_band": "1",
    "expected_temporal_window": "followup_diagnostic_2023_2025",
    "require_all_nonnull_field_counts_unchanged": True,
    "require_null_field_delta": -1,
    "require_master_work_delta": -1,
}


class W0SensitivityError(RuntimeError):
    """Raised when a frozen W0 sensitivity input violates its contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise W0SensitivityError(message)


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be a mapping")
    return value


def require_list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be a list")
    return value


def require_int(value: Any, field: str) -> int:
    require(type(value) is int, f"{field} must be an integer")
    return value


def require_sha256(value: Any, field: str) -> str:
    text = str(value)
    require(SHA256_PATTERN.fullmatch(text) is not None, f"{field} is not SHA-256")
    return text


def safe_relative_path(value: Any, field: str) -> Path:
    path = Path(str(value))
    require(not path.is_absolute(), f"{field} must be repository-relative")
    require(path.parts and path != Path("."), f"{field} may not be empty")
    require(".." not in path.parts, f"{field} may not contain '..'")
    return path


def _reject_symlink_components(path: Path, boundary: Path, field: str) -> None:
    require(path.is_relative_to(boundary), f"{field} escapes its boundary")
    current = boundary
    for part in path.relative_to(boundary).parts:
        current = current / part
        require(not current.is_symlink(), f"{field} contains a symlink component")


def read_json(path: Path, field: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{field} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise W0SensitivityError(f"cannot read {field}: {exc}") from exc
    return require_mapping(value, field)


def tracked_json(binding: Mapping[str, Any], field: str) -> tuple[Path, dict[str, Any]]:
    relative = safe_relative_path(binding.get("relative_path"), f"{field}.relative_path")
    candidate = REPO_ROOT / relative
    _reject_symlink_components(candidate, REPO_ROOT, field)
    value = read_json(candidate, field)
    require(
        sha256_file(candidate) == require_sha256(binding.get("sha256"), f"{field}.sha256"),
        f"{field} SHA-256 mismatch",
    )
    expected_schema = binding.get("schema_version")
    if expected_schema is not None:
        require(value.get("schema_version") == expected_schema, f"{field} schema mismatch")
    expected_status = binding.get("required_status")
    if expected_status is not None:
        require(value.get("status") == expected_status, f"{field} status mismatch")
    expected_fingerprint = binding.get("result_fingerprint_sha256")
    if expected_fingerprint is not None:
        provenance = require_mapping(value.get("provenance"), f"{field}.provenance")
        observed = provenance.get("result_fingerprint_sha256")
        require(observed == expected_fingerprint, f"{field} result fingerprint mismatch")
    return candidate, value


def read_config(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "config is not a regular file")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise W0SensitivityError(f"cannot read config: {exc}") from exc
    config = require_mapping(value, "config")
    require(config.get("schema_version") == CONFIG_SCHEMA_VERSION, "config schema mismatch")
    success = require_mapping(config.get("success_contract"), "success_contract")
    require(success.get("status") == SUCCESS_STATUS, "success status mismatch")
    require(
        require_mapping(success.get("gate_status"), "success_contract.gate_status")
        == REQUIRED_GATE_STATUS,
        "success gate status drift",
    )
    require(
        require_mapping(
            config.get("positive_id_sensitivity"), "positive_id_sensitivity"
        )
        == REQUIRED_POSITIVE_ID_SENSITIVITY,
        "positive-ID sensitivity policy drift",
    )
    require(config.get("claim_boundary") == CLAIM_BOUNDARY, "claim boundary drift")
    return config


def resolve_absolute_path(
    override: Path | None,
    env_name: str,
    field: str,
) -> Path:
    if override is not None:
        path = override
    else:
        raw = os.environ.get(env_name)
        require(raw is not None and raw != "", f"{env_name} is not set")
        path = Path(raw)
    require(path.is_absolute(), f"{field} must be an absolute operator path")
    return path


def validate_inventory(
    contract: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    path, manifest = tracked_json(contract, "inventory manifest")
    require(manifest.get("schema_version") == contract.get("schema_version"), "inventory schema mismatch")
    root = require_mapping(manifest.get("root"), "inventory.root")
    require(
        root.get("inventory_fingerprint_sha256")
        == contract.get("inventory_fingerprint_sha256"),
        "inventory path-size fingerprint mismatch",
    )
    require(
        root.get("stability_fingerprint_sha256")
        == contract.get("stability_fingerprint_sha256"),
        "inventory stability fingerprint mismatch",
    )
    files: dict[str, dict[str, Any]] = {}
    for table in require_list(manifest.get("tables"), "inventory.tables"):
        table_map = require_mapping(table, "inventory table")
        for record in require_list(table_map.get("files"), "inventory table files"):
            item = require_mapping(record, "inventory file")
            relative = str(item.get("relative_path", ""))
            require(relative and relative not in files, "inventory file path is missing or duplicated")
            files[relative] = item
    return path, manifest, files


def validate_sentinel(
    value: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    source_totals = require_mapping(value.get("source_id_totals"), "sentinel source totals")
    w0 = require_mapping(source_totals.get("w_zero"), "sentinel W0 source totals")
    require(
        w0.get("exact_row_count") == contract.get("required_w_zero_source_rows"),
        "frozen W0 source-row total mismatch",
    )
    require(
        w0.get("affected_file_count") == contract.get("required_w_zero_source_files"),
        "frozen W0 source-file total mismatch",
    )
    targets = require_mapping(value.get("work_target_totals"), "sentinel target totals")
    require(
        targets.get("w_zero_rows") == contract.get("required_w_zero_work_target_rows"),
        "frozen W0 work-target total mismatch",
    )
    observed_tables = {
        str(row["table"])
        for row in require_list(value.get("nonzero_source_tables"), "sentinel source tables")
        if require_mapping(row, "sentinel source table").get("w_zero_rows", 0)
    }
    declared_tables = [
        str(item)
        for item in require_list(
            contract.get("affected_source_tables"), "affected source tables"
        )
    ]
    require(
        len(declared_tables) == len(set(declared_tables)),
        "affected source tables are duplicated",
    )
    require(
        observed_tables == set(declared_tables),
        "frozen W0 propagation table set mismatch",
    )
    return {
        "source_rows": int(w0["exact_row_count"]),
        "source_files": int(w0["affected_file_count"]),
        "source_tables": len(observed_tables),
        "work_target_rows": int(targets["w_zero_rows"]),
        "affected_source_tables": sorted(declared_tables),
    }


def validate_taxonomy_reference(
    value: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, int]:
    tables = require_mapping(
        require_mapping(value.get("accounting"), "taxonomy accounting").get("tables"),
        "taxonomy tables",
    )
    main_metrics = require_mapping(
        require_mapping(tables.get("main"), "taxonomy main").get("metrics"),
        "taxonomy main metrics",
    )
    topic_metrics = require_mapping(
        require_mapping(tables.get("topics"), "taxonomy topics").get("metrics"),
        "taxonomy topic metrics",
    )
    main_w0 = require_int(main_metrics.get("source_w_zero_rows"), "main W0 rows")
    topic_w0 = require_int(topic_metrics.get("source_w_zero_rows"), "topic W0 rows")
    require(main_w0 == contract.get("required_main_w_zero_rows"), "taxonomy main W0 mismatch")
    require(topic_w0 == contract.get("required_topic_w_zero_rows"), "taxonomy topic W0 mismatch")
    return {"main_rows": main_w0, "topic_rows": topic_w0}


def resolve_full_report(
    override: Path | None,
    contract: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    env_name = str(contract.get("path_env", ""))
    require(env_name != "", "work-source full report path alias is missing")
    path = resolve_absolute_path(override, env_name, "work-source full report")
    require(path.name == contract.get("expected_basename"), "work-source full basename mismatch")
    report = read_json(path, "work-source full report")
    require(
        sha256_file(path) == require_sha256(contract.get("sha256"), "work-source full SHA-256"),
        "work-source full report SHA-256 mismatch",
    )
    require(report.get("status") == contract.get("required_status"), "work-source full status mismatch")
    provenance = require_mapping(
        report.get("provenance"), "work-source full report provenance"
    )
    require(
        provenance.get("result_fingerprint_sha256")
        == contract.get("result_fingerprint_sha256"),
        "work-source full result fingerprint mismatch",
    )
    return path, report


def validate_work_source_w0(
    report: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    profile = require_mapping(report.get("author_work_profile"), "author work profile")
    w0 = require_mapping(profile.get("w0"), "author work profile W0")
    expected = {
        "observed_authorship_slots": 1,
        "valid_author_ids": 0,
        "invalid_author_ids": 0,
        "null_author_ids": 1,
        "status": policy.get("expected_work_profile_status"),
    }
    require(w0 == expected, "frozen W0 authorship profile mismatch")
    matching_cells = [
        row
        for row in require_list(profile.get("joint_sparse_cube"), "joint sparse cube")
        if require_mapping(row, "joint sparse cube cell").get("primary_field_id") is None
        and row.get("publication_year") == policy.get("expected_main_projection", {}).get("publication_year")
        and row.get("author_id_status") == policy.get("expected_work_profile_status")
        and str(row.get("declared_authors_band")) == str(policy.get("expected_declared_authors_band"))
    ]
    require(len(matching_cells) == 1, "W0 profile stratum is not unique")
    require(require_int(matching_cells[0].get("works"), "W0 stratum works") >= 1, "W0 profile stratum is empty")
    return {"w0": w0, "joint_stratum_before": dict(matching_cells[0])}


@contextmanager
def open_source_file(root: Path, relative: Path) -> Iterator[BinaryIO]:
    directory_fds: list[int] = []
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        current_fd = os.open(root, directory_flags)
        directory_fds.append(current_fd)
        for part in relative.parts[:-1]:
            current_fd = os.open(part, directory_flags, dir_fd=current_fd)
            directory_fds.append(current_fd)
        file_fd = os.open(relative.name, file_flags, dir_fd=current_fd)
        with os.fdopen(file_fd, "rb") as handle:
            yield handle
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def scan_current_w0_projection(
    root: Path,
    spec: Mapping[str, Any],
    inventory_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    relative = safe_relative_path(spec.get("relative_path"), "selected file path")
    path = root / relative
    _reject_symlink_components(path, root, "selected source file")
    locator_before = path.stat(follow_symlinks=False)
    require(
        stat.S_ISREG(locator_before.st_mode),
        "selected source file is not regular",
    )
    inventory = require_mapping(inventory_files.get(relative.as_posix()), "selected inventory file")
    expected = {
        "bytes": require_int(spec.get("expected_bytes"), "selected expected bytes"),
        "mtime_ns": require_int(spec.get("expected_mtime_ns"), "selected expected mtime"),
        "footer_rows": require_int(spec.get("expected_rows"), "selected expected rows"),
    }
    for key, value in expected.items():
        require(inventory.get(key) == value, f"selected {key} differs from inventory")
    for field in (
        "physical_schema_fingerprint_sha256",
        "canonical_schema_fingerprint_sha256",
    ):
        require(
            inventory.get(field) == require_sha256(spec.get(field), f"selected {field}"),
            f"selected {field} differs from inventory",
        )
    selected_columns = [str(item) for item in require_list(spec.get("selected_columns"), "selected columns")]
    require("id" in selected_columns, "selected projection must include id")
    selected_schema = require_mapping(spec.get("selected_schema"), "selected schema")
    require(set(selected_schema) == set(selected_columns), "selected schema columns mismatch")
    inventory_schema = require_mapping(inventory.get("schema_fields"), "inventory schema fields")
    require(
        {column: inventory_schema.get(column) for column in selected_columns}
        == selected_schema,
        "selected schema differs from inventory",
    )
    rows: list[dict[str, Any]] = []
    projected_digest = hashlib.sha256()
    projected_rows = 0
    with open_source_file(root, relative) as handle:
        before = os.fstat(handle.fileno())
        require(
            (before.st_dev, before.st_ino)
            == (locator_before.st_dev, locator_before.st_ino),
            "selected source locator changed before open",
        )
        require(before.st_size == expected["bytes"], "selected source size drift")
        require(
            before.st_mtime_ns == expected["mtime_ns"],
            "selected source mtime drift",
        )
        parquet = pq.ParquetFile(handle)
        require(
            parquet.metadata.num_rows == expected["footer_rows"],
            "selected source row drift",
        )
        schema_names = set(parquet.schema_arrow.names)
        require(
            set(selected_columns) <= schema_names,
            "selected projection column missing",
        )
        live_selected_schema = {
            column: {
                "nullable": bool(parquet.schema_arrow.field(column).nullable),
                "type": str(parquet.schema_arrow.field(column).type),
            }
            for column in selected_columns
        }
        require(live_selected_schema == selected_schema, "live selected schema drift")
        for batch in parquet.iter_batches(columns=selected_columns, batch_size=131_072):
            projected_rows += batch.num_rows
            table = batch.to_pydict()
            for values in zip(
                *(table[column] for column in selected_columns), strict=True
            ):
                row = dict(zip(selected_columns, values, strict=True))
                projected_digest.update(canonical_json_bytes(row))
                projected_digest.update(b"\n")
                if row["id"] == W0:
                    rows.append(row)
        after = os.fstat(handle.fileno())
    locator_after = path.stat(follow_symlinks=False)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        all(getattr(before, field) == getattr(after, field) for field in identity_fields),
        "selected source changed during scan",
    )
    require(
        all(
            getattr(after, field) == getattr(locator_after, field)
            for field in identity_fields
        ),
        "selected source locator changed after scan",
    )
    require(projected_rows == expected["footer_rows"], "selected projected row conservation failed")
    return {
        "role": str(spec.get("role")),
        "relative_path": relative.as_posix(),
        "bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "projected_rows": projected_rows,
        "selected_schema_sha256": sha256_bytes(
            canonical_json_bytes(selected_schema)
        ),
        "source_identity_stable_during_scan": True,
        "projected_payload_sha256": projected_digest.hexdigest(),
        "w0_rows": rows,
        "w0_selected_values_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def validate_current_rows(
    scans: list[dict[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    by_role = {item["role"]: item for item in scans}
    require(set(by_role) == {"main_work", "authorship"}, "selected source roles mismatch")
    main_rows = by_role["main_work"]["w0_rows"]
    authorship_rows = by_role["authorship"]["w0_rows"]
    expected_main_rows = require_int(
        policy.get("expected_current_main_rows"), "expected current main W0 rows"
    )
    expected_authorship_rows = require_int(
        policy.get("expected_current_authorship_rows"),
        "expected current authorship W0 rows",
    )
    require(expected_main_rows == 1, "current main W0 contract must select one row")
    require(
        expected_authorship_rows == 1,
        "current authorship W0 contract must select one row",
    )
    require(len(main_rows) == expected_main_rows, "current main W0 count mismatch")
    require(
        len(authorship_rows) == expected_authorship_rows,
        "current authorship W0 count mismatch",
    )
    main = main_rows[0]
    authorship = authorship_rows[0]
    require(main.get("id") == W0 and authorship.get("id") == W0, "selected row is not W0")
    for key, expected in require_mapping(policy.get("expected_main_projection"), "expected main projection").items():
        observed = main.get(key)
        require(
            observed is None
            if expected is None
            else type(observed) is type(expected) and observed == expected,
            f"current main W0 projection mismatch: {key}",
        )
    for key, expected in require_mapping(policy.get("expected_authorship_projection"), "expected authorship projection").items():
        observed = authorship.get(key)
        require(
            observed is None
            if expected is None
            else type(observed) is type(expected) and observed == expected,
            f"current authorship W0 projection mismatch: {key}",
        )
    return {
        "main_selected_values_sha256": by_role["main_work"]["w0_selected_values_sha256"],
        "authorship_selected_values_sha256": by_role["authorship"]["w0_selected_values_sha256"],
        "main_projected_payload_sha256": by_role["main_work"]["projected_payload_sha256"],
        "authorship_projected_payload_sha256": by_role["authorship"]["projected_payload_sha256"],
    }


def derive_positive_id_profile(
    cross_field: Mapping[str, Any],
    work_source_w0: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    profile = require_mapping(cross_field.get("profile"), "cross-field profile")
    master_before = require_int(profile.get("master_work_rows"), "master work rows")
    classified_before = require_int(profile.get("classified_primary_field_works"), "classified works")
    null_before = require_int(profile.get("null_primary_field_works"), "null-field works")
    master_delta = require_int(
        policy.get("require_master_work_delta"), "required master-work delta"
    )
    null_delta = require_int(
        policy.get("require_null_field_delta"), "required null-field delta"
    )
    require(master_delta == -1, "positive-ID view must remove exactly one work")
    require(null_delta == -1, "positive-ID view must remove one null-field work")
    master_after = master_before + master_delta
    null_after = null_before + null_delta
    require(master_after > 0 and null_after >= 0, "positive-ID counts are invalid")
    require(master_before == classified_before + null_before, "baseline field conservation failed")
    require(master_after == classified_before + null_after, "positive-ID field conservation failed")

    fields: list[dict[str, Any]] = []
    for domain in require_list(cross_field.get("domains"), "cross-field domains"):
        for field in require_list(require_mapping(domain, "cross-field domain").get("fields"), "cross-field fields"):
            item = require_mapping(field, "cross-field field")
            fields.append({"field_id": int(item["field_id"]), "works": int(item["works"])})
    fields.sort(key=lambda item: item["field_id"])
    require(len(fields) == 26, "cross-field profile does not contain 26 fields")
    require(sum(item["works"] for item in fields) == classified_before, "nonnull field counts do not conserve")

    null_profile = require_mapping(profile.get("null_primary_field_profile"), "null-field profile")
    null_auth_before = {
        key: require_int(value, f"null authorship {key}")
        for key, value in require_mapping(null_profile.get("authorship_status_counts"), "null authorship counts").items()
    }
    expected_status = str(policy.get("expected_work_profile_status"))
    require(expected_status in null_auth_before, "W0 authorship status missing from null profile")
    null_auth_after = dict(null_auth_before)
    null_auth_after[expected_status] -= 1
    require(sum(null_auth_before.values()) == null_before, "baseline null authorship conservation failed")
    require(sum(null_auth_after.values()) == null_after, "positive-ID null authorship conservation failed")

    global_auth_before = {key: 0 for key in null_auth_before}
    for domain in require_list(cross_field.get("domains"), "cross-field domains"):
        counts = require_mapping(require_mapping(domain, "cross-field domain").get("authorship_status_counts"), "domain authorship counts")
        for key in global_auth_before:
            global_auth_before[key] += require_int(counts.get(key), f"domain authorship {key}")
    for key, value in null_auth_before.items():
        global_auth_before[key] += value
    global_auth_after = dict(global_auth_before)
    global_auth_after[expected_status] -= 1
    require(sum(global_auth_before.values()) == master_before, "baseline global authorship conservation failed")
    require(sum(global_auth_after.values()) == master_after, "positive-ID global authorship conservation failed")

    window = str(policy.get("expected_temporal_window"))
    global_windows_before = {
        key: require_int(value, f"global temporal {key}")
        for key, value in require_mapping(profile.get("global_temporal_window_counts"), "global temporal counts").items()
    }
    null_windows_before = {
        key: require_int(value, f"null temporal {key}")
        for key, value in require_mapping(null_profile.get("temporal_window_counts"), "null temporal counts").items()
    }
    require(window in global_windows_before and window in null_windows_before, "W0 temporal window is missing")
    global_windows_after = dict(global_windows_before)
    null_windows_after = dict(null_windows_before)
    global_windows_after[window] -= 1
    null_windows_after[window] -= 1

    joint_before = require_mapping(work_source_w0.get("joint_stratum_before"), "W0 joint stratum")
    joint_after = dict(joint_before)
    joint_after["works"] = require_int(joint_before.get("works"), "W0 stratum works") - 1

    return {
        "baseline_grammar_valid_view": {
            "master_work_rows": master_before,
            "classified_primary_field_works": classified_before,
            "null_primary_field_works": null_before,
            "classified_primary_field_rate": classified_before / master_before,
            "null_primary_field_rate": null_before / master_before,
            "global_authorship_status_counts": global_auth_before,
            "null_field_authorship_status_counts": null_auth_before,
            "global_temporal_window_counts": global_windows_before,
            "null_field_temporal_window_counts": null_windows_before,
            "w0_joint_stratum": joint_before,
        },
        "positive_id_view": {
            "master_work_rows": master_after,
            "classified_primary_field_works": classified_before,
            "null_primary_field_works": null_after,
            "classified_primary_field_rate": classified_before / master_after,
            "null_primary_field_rate": null_after / master_after,
            "global_authorship_status_counts": global_auth_after,
            "null_field_authorship_status_counts": null_auth_after,
            "global_temporal_window_counts": global_windows_after,
            "null_field_temporal_window_counts": null_windows_after,
            "w0_joint_stratum": joint_after,
        },
        "delta": {
            "master_work_rows": master_delta,
            "classified_primary_field_works": 0,
            "null_primary_field_works": null_delta,
            "nonnull_field_count_deltas": {str(item["field_id"]): 0 for item in fields},
            "global_authorship_status_counts": {key: global_auth_after[key] - global_auth_before[key] for key in global_auth_before},
            "null_field_authorship_status_counts": {key: null_auth_after[key] - null_auth_before[key] for key in null_auth_before},
            "global_temporal_window_counts": {key: global_windows_after[key] - global_windows_before[key] for key in global_windows_before},
            "null_field_temporal_window_counts": {key: null_windows_after[key] - null_windows_before[key] for key in null_windows_before},
        },
        "checks": {
            "baseline_field_conservation": True,
            "positive_id_field_conservation": True,
            "baseline_authorship_conservation": True,
            "positive_id_authorship_conservation": True,
            "all_26_nonnull_field_counts_unchanged": True,
            "only_null_field_margin_changes": True,
            "only_expected_authorship_status_changes": True,
            "only_expected_temporal_window_changes": True,
        },
    }


def build_report(
    *,
    config_path: Path,
    source_root: Path,
    work_source_full_report: Path,
) -> dict[str, Any]:
    config = read_config(config_path)
    started = utc_now()
    source_contract = require_mapping(config.get("source"), "source")
    require(source_root.is_dir() and not source_root.is_symlink(), "source root is not a regular directory")
    require(source_root.name == source_contract.get("root_id"), "source root ID mismatch")

    inventory_path, inventory, inventory_files = validate_inventory(
        require_mapping(source_contract.get("inventory_manifest"), "inventory contract")
    )
    require(
        require_mapping(inventory.get("root"), "inventory root").get("id") == source_contract.get("root_id"),
        "inventory root ID mismatch",
    )

    frozen = require_mapping(config.get("frozen_evidence"), "frozen evidence")
    sentinel_path, sentinel = tracked_json(
        require_mapping(frozen.get("sentinel_incidence"), "sentinel contract"),
        "sentinel incidence",
    )
    sentinel_summary = validate_sentinel(
        sentinel,
        require_mapping(frozen.get("sentinel_incidence"), "sentinel contract"),
    )
    compact_path, _compact = tracked_json(
        require_mapping(frozen.get("work_source_compact"), "work-source compact contract"),
        "work-source compact",
    )
    full_path, full_report = resolve_full_report(
        work_source_full_report,
        require_mapping(frozen.get("work_source_full"), "work-source full contract"),
    )
    cross_path, cross_field = tracked_json(
        require_mapping(frozen.get("cross_field_profile"), "cross-field contract"),
        "cross-field profile",
    )
    taxonomy_path, taxonomy = tracked_json(
        require_mapping(frozen.get("taxonomy_reference"), "taxonomy contract"),
        "taxonomy reference",
    )
    taxonomy_w0 = validate_taxonomy_reference(
        taxonomy,
        require_mapping(frozen.get("taxonomy_reference"), "taxonomy contract"),
    )

    policy = require_mapping(config.get("positive_id_sensitivity"), "positive-ID policy")
    pattern = re.compile(str(policy.get("retained_work_id_regex")))
    require(pattern.fullmatch("https://openalex.org/W1") is not None, "positive-ID regex rejects W1")
    require(pattern.fullmatch(W0) is None, "positive-ID regex retains W0")
    require(policy.get("excluded_identifier") == W0, "excluded identifier is not W0")

    scans = [
        scan_current_w0_projection(source_root, require_mapping(spec, "selected file spec"), inventory_files)
        for spec in require_list(source_contract.get("current_selected_files"), "selected files")
    ]
    current_projection = validate_current_rows(scans, policy)
    work_source_w0 = validate_work_source_w0(full_report, policy)
    sensitivity = derive_positive_id_profile(cross_field, work_source_w0, policy)
    require(
        policy.get("require_all_nonnull_field_counts_unchanged") is True
        and sensitivity["checks"]["all_26_nonnull_field_counts_unchanged"],
        "nonnull field sensitivity contract failed",
    )

    success = require_mapping(config.get("success_contract"), "success contract")
    gate_status = require_mapping(success.get("gate_status"), "success gate status")
    finished = utc_now()
    stable = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": str(config.get("label")),
        "status": SUCCESS_STATUS,
        "scope": {
            "work_parquet_files_read": len(scans),
            "work_parquet_rows_scanned": sum(item["projected_rows"] for item in scans),
            "full_root_rescan_performed": False,
            "positive_id_sensitivity_only": True,
            "w0_entity_validity_decided": False,
            "field_selection_performed": False,
            "scientific_effect_estimation_performed": False,
            "gpu_used": False,
        },
        "frozen_w0_incidence": sentinel_summary,
        "current_main_topics_w0": taxonomy_w0,
        "current_selected_file_scans": [
            {key: value for key, value in item.items() if key != "w0_rows"}
            for item in scans
        ],
        "current_w0_projection": current_projection,
        "sensitivity": sensitivity,
        "gate_status": gate_status,
        "claim_boundary": str(config.get("claim_boundary")),
        "provenance": {
            "path_redacted": True,
            "config_sha256": sha256_file(config_path),
            "tool_sha256": sha256_file(Path(__file__)),
            "inventory_manifest_sha256": sha256_file(inventory_path),
            "sentinel_summary_sha256": sha256_file(sentinel_path),
            "work_source_compact_sha256": sha256_file(compact_path),
            "work_source_full_sha256": sha256_file(full_path),
            "cross_field_profile_sha256": sha256_file(cross_path),
            "taxonomy_reference_sha256": sha256_file(taxonomy_path),
            "source_root_id": str(source_contract.get("root_id")),
            "source_inventory_fingerprint_sha256": inventory["root"]["inventory_fingerprint_sha256"],
            "source_stability_fingerprint_sha256": inventory["root"]["stability_fingerprint_sha256"],
        },
    }
    stable["provenance"]["result_fingerprint_sha256"] = sha256_bytes(
        canonical_json_bytes(stable)
    )
    return {"generated_at_utc": finished, "execution": {"started_at_utc": started, "finished_at_utc": finished}, **stable}


def render_markdown(report: Mapping[str, Any]) -> str:
    baseline = report["sensitivity"]["baseline_grammar_valid_view"]
    positive = report["sensitivity"]["positive_id_view"]
    delta = report["sensitivity"]["delta"]
    gates = report["gate_status"]
    lines = [
        "# OpenAlex W0 Positive-ID Sensitivity",
        "",
        f"- Status: `{report['status']}`",
        f"- Work files read: `{report['scope']['work_parquet_files_read']}`",
        f"- Rows projected: `{report['scope']['work_parquet_rows_scanned']:,}`",
        "- GPU used: `false`",
        "- Scientific effect estimated: `false`",
        "",
        "## Profile Delta",
        "",
        "| Quantity | Grammar-valid view | Positive-ID view | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Master works | {baseline['master_work_rows']:,} | {positive['master_work_rows']:,} | {delta['master_work_rows']:,} |",
        f"| Classified primary field | {baseline['classified_primary_field_works']:,} | {positive['classified_primary_field_works']:,} | {delta['classified_primary_field_works']:,} |",
        f"| Null primary field | {baseline['null_primary_field_works']:,} | {positive['null_primary_field_works']:,} | {delta['null_primary_field_works']:,} |",
        "",
        "All 26 non-null field counts are unchanged. The excluded W0 row is in",
        "the null-primary-field, 2024, one-declared-author, no-usable-author-ID stratum.",
        "",
        "## Gate Status",
        "",
        "| Gate | Status |",
        "| --- | --- |",
        *[f"| `{key}` | `{str(value).lower() if isinstance(value, bool) else value}` |" for key, value in sorted(gates.items())],
        "",
        "## Claim Boundary",
        "",
        str(report["claim_boundary"]),
        "",
        f"- Result fingerprint: `{report['provenance']['result_fingerprint_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def _absolute_lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path.expanduser())))


def _require_no_symlink_components(path: Path, field: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        require(not current.is_symlink(), f"{field} contains a symlink component")


def prepare_output_paths(
    output_json: Path,
    output_markdown: Path,
    source_root: Path,
) -> tuple[Path, Path, Path]:
    json_path = _absolute_lexical_path(output_json)
    markdown_path = _absolute_lexical_path(output_markdown)
    require(json_path.suffix == ".json", "JSON output must end in .json")
    require(markdown_path.suffix == ".md", "Markdown output must end in .md")
    require(json_path.parent == markdown_path.parent, "outputs must share one directory")
    outputs_path = json_path.with_suffix(".outputs.json")
    require(
        len({json_path, markdown_path, outputs_path}) == 3,
        "output paths collide",
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_components(json_path.parent, "output parent")
    require(
        json_path.parent.is_dir() and not json_path.parent.is_symlink(),
        "output parent is not a regular directory",
    )
    for path in (json_path, markdown_path, outputs_path):
        _require_no_symlink_components(path, "output path")
        require(
            not path.is_relative_to(source_root),
            "output may not be inside source root",
        )
        require(not path.exists() and not path.is_symlink(), "output already exists")
        temporary = path.with_suffix(path.suffix + ".tmp")
        require(
            not temporary.exists() and not temporary.is_symlink(),
            "temporary output already exists",
        )
    return json_path, markdown_path, outputs_path


def atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not path.exists() and not path.is_symlink(), "output already exists")
    require(
        not temporary.exists() and not temporary.is_symlink(),
        "temporary output already exists",
    )
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    temporary.replace(path)


def write_outputs(
    report: dict[str, Any],
    output_json: Path,
    output_markdown: Path,
    outputs_path: Path,
) -> dict[str, Any]:
    json_payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_payload = render_markdown(report)
    atomic_write(output_json, json_payload)
    atomic_write(output_markdown, markdown_payload)
    outputs = {
        "schema_version": OUTPUTS_SCHEMA_VERSION,
        "status": report["status"],
        "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
        "artifacts": [
            {"role": "report_json", "relative_name": output_json.name, "bytes": len(json_payload.encode("utf-8")), "sha256": sha256_bytes(json_payload.encode("utf-8"))},
            {"role": "report_markdown", "relative_name": output_markdown.name, "bytes": len(markdown_payload.encode("utf-8")), "sha256": sha256_bytes(markdown_payload.encode("utf-8"))},
        ],
    }
    atomic_write(outputs_path, json.dumps(outputs, indent=2, sort_keys=True) + "\n")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--work-source-full-report", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        config = read_config(config_path)
        source = require_mapping(config.get("source"), "source")
        root = resolve_absolute_path(args.root, str(source.get("root_env", "")), "source root").resolve()
        full_contract = require_mapping(
            require_mapping(config.get("frozen_evidence"), "frozen evidence").get("work_source_full"),
            "work-source full contract",
        )
        full_report = resolve_absolute_path(
            args.work_source_full_report,
            str(full_contract.get("path_env", "")),
            "work-source full report",
        ).resolve()
        output_json, output_markdown, outputs_path = prepare_output_paths(
            args.output_json,
            args.output_markdown,
            root,
        )
        report = build_report(
            config_path=config_path,
            source_root=root,
            work_source_full_report=full_report,
        )
        write_outputs(
            report,
            output_json,
            output_markdown,
            outputs_path,
        )
    except (OSError, ValueError, W0SensitivityError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
                "outputs_manifest_sha256": sha256_file(outputs_path),
                "master_work_delta": report["sensitivity"]["delta"]["master_work_rows"],
                "nonnull_field_counts_unchanged": report["sensitivity"]["checks"]["all_26_nonnull_field_counts_unchanged"],
            },
            sort_keys=True,
        )
    )
    if args.require_pass and report["status"] != SUCCESS_STATUS:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
