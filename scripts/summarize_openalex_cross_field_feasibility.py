#!/usr/bin/env python3
"""Summarize frozen OpenAlex cross-field profile feasibility without rescanning.

The command validates a small domain/field/subfield/topic taxonomy graph and
independently reprojects the joint sparse cube from the frozen full
work-source report.  It does not read any OpenAlex work Parquet data and does
not establish field-entity validity, candidate eligibility, G-1, G0, or
empirical G1.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_cross_field_feasibility_v0.yaml"
)
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "experiments/evidence/results/"
    "openalex_g1_cross_field_feasibility_20260718.summary.json"
)
DEFAULT_OUTPUT_MARKDOWN = DEFAULT_OUTPUT_JSON.with_suffix(".md")
REPORT_SCHEMA_VERSION = "openalex_g1_cross_field_feasibility_summary.v1"
SUCCESS_STATUS = "PASS_CROSS_FIELD_PROFILE_FEASIBILITY_ONLY"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ENTITY_ID_PATTERNS = {
    "domains": re.compile(r"https://openalex[.]org/domains/([1-9][0-9]*)"),
    "fields": re.compile(r"https://openalex[.]org/fields/([1-9][0-9]*)"),
    "subfields": re.compile(r"https://openalex[.]org/subfields/([1-9][0-9]*)"),
    "topics": re.compile(r"https://openalex[.]org/T([1-9][0-9]*)"),
}


class CrossFieldFeasibilityError(RuntimeError):
    """Raised when frozen cross-field feasibility inputs violate the contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
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
        raise CrossFieldFeasibilityError(message)


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
    require(not path.is_absolute(), f"{field} must be relative")
    require(".." not in path.parts, f"{field} may not contain '..'")
    require(path.parts and path != Path("."), f"{field} may not be empty")
    return path


def read_json(path: Path, field: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{field} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossFieldFeasibilityError(f"cannot read {field}: {exc}") from exc
    return require_mapping(value, field)


def read_config(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "config is not a regular file")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CrossFieldFeasibilityError(f"cannot read config: {exc}") from exc
    config = require_mapping(value, "config")
    require(
        config.get("schema_version")
        == "openalex_g1_cross_field_feasibility_config.v1",
        "unexpected config schema_version",
    )
    return config


def resolve_env_path(
    override: Path | None,
    contract: Mapping[str, Any],
    *,
    field: str,
) -> Path:
    if override is not None:
        path = override
    else:
        env_name = str(contract.get("path_env") or contract.get("root_env") or "")
        require(env_name != "", f"{field} environment alias is missing")
        raw = os.environ.get(env_name)
        require(raw is not None and raw != "", f"{env_name} is not set")
        path = Path(raw)
    require(path.is_absolute(), f"{field} must resolve to an absolute operator path")
    return path


def _entity_numeric_id(entity: str, value: Any) -> int:
    text = str(value)
    match = ENTITY_ID_PATTERNS[entity].fullmatch(text)
    require(match is not None, f"invalid {entity} entity ID: {text!r}")
    return int(match.group(1))


def _id_name(value: Any, field: str) -> tuple[str, str]:
    item = require_mapping(value, field)
    identifier = str(item.get("id", ""))
    name = str(item.get("display_name", ""))
    require(identifier != "" and name != "", f"{field} needs id and display_name")
    return identifier, name


def _declared_child_ids(
    value: Any,
    field: str,
    child_map: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    declared: set[str] = set()
    for index, raw_child in enumerate(require_list(value, field)):
        child_id, child_name = _id_name(raw_child, f"{field}[{index}]")
        require(child_id in child_map, f"{field} references a missing child")
        require(
            str(child_map[child_id]["display_name"]) == child_name,
            f"{field} child display_name mismatch",
        )
        require(child_id not in declared, f"{field} contains a duplicate child ID")
        declared.add(child_id)
    return declared


def load_and_validate_taxonomy(
    root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), "taxonomy root is not a directory")
    rows_by_entity: dict[str, list[dict[str, Any]]] = {
        "domains": [],
        "fields": [],
        "subfields": [],
        "topics": [],
    }
    canonical_digest = hashlib.sha256()
    file_commitments: list[dict[str, Any]] = []

    for index, raw_spec in enumerate(require_list(contract.get("files"), "taxonomy.files")):
        spec = require_mapping(raw_spec, f"taxonomy.files[{index}]")
        entity = str(spec.get("entity", ""))
        require(entity in rows_by_entity, f"unsupported taxonomy entity: {entity!r}")
        relative_path = safe_relative_path(
            spec.get("relative_path"), f"taxonomy.files[{index}].relative_path"
        )
        path = root / relative_path
        require(path.is_file() and not path.is_symlink(), "taxonomy file is not regular")
        expected_sha256 = require_sha256(
            spec.get("sha256"), f"taxonomy.files[{index}].sha256"
        )
        observed_sha256 = sha256_file(path)
        require(observed_sha256 == expected_sha256, "taxonomy file SHA-256 mismatch")

        file_rows = 0
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    require(line.strip() != "", "taxonomy JSONL contains a blank row")
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise CrossFieldFeasibilityError(
                            f"invalid taxonomy JSON at declared file {index}, "
                            f"line {line_number}: {exc}"
                        ) from exc
                    row = require_mapping(value, "taxonomy JSON row")
                    rows_by_entity[entity].append(row)
                    canonical_digest.update(canonical_json_bytes(row))
                    canonical_digest.update(b"\n")
                    file_rows += 1
        except OSError as exc:
            raise CrossFieldFeasibilityError("cannot decompress taxonomy file") from exc

        expected_rows = require_int(
            spec.get("expected_rows"), f"taxonomy.files[{index}].expected_rows"
        )
        require(file_rows == expected_rows, "taxonomy file row count mismatch")
        file_commitments.append(
            {"entity": entity, "rows": file_rows, "sha256": observed_sha256}
        )

    observed_canonical_sha256 = canonical_digest.hexdigest()
    expected_canonical_sha256 = require_sha256(
        contract.get("canonical_sha256"), "taxonomy.canonical_sha256"
    )
    require(
        observed_canonical_sha256 == expected_canonical_sha256,
        "canonical taxonomy SHA-256 mismatch",
    )

    expected_counts = require_mapping(
        contract.get("expected_entity_counts"), "taxonomy.expected_entity_counts"
    )
    observed_counts = {key: len(value) for key, value in rows_by_entity.items()}
    for entity, count in observed_counts.items():
        require(
            count == require_int(expected_counts.get(entity), f"taxonomy count {entity}"),
            f"taxonomy {entity} count mismatch",
        )

    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for entity, rows in rows_by_entity.items():
        mapping: dict[str, dict[str, Any]] = {}
        for row in rows:
            identifier = str(row.get("id", ""))
            _entity_numeric_id(entity, identifier)
            require(identifier not in mapping, f"duplicate {entity} ID")
            require(str(row.get("display_name", "")) != "", f"empty {entity} name")
            mapping[identifier] = row
        maps[entity] = mapping

    domains = maps["domains"]
    fields = maps["fields"]
    subfields = maps["subfields"]
    topics = maps["topics"]

    fields_by_domain: dict[str, set[str]] = defaultdict(set)
    subfields_by_field: dict[str, set[str]] = defaultdict(set)
    topics_by_subfield: dict[str, set[str]] = defaultdict(set)

    for identifier, row in fields.items():
        parent_id, parent_name = _id_name(row.get("domain"), f"field {identifier}.domain")
        require(parent_id in domains, "field references missing domain")
        require(domains[parent_id]["display_name"] == parent_name, "field domain name mismatch")
        fields_by_domain[parent_id].add(identifier)

    for identifier, row in subfields.items():
        field_id, field_name = _id_name(row.get("field"), f"subfield {identifier}.field")
        domain_id, domain_name = _id_name(
            row.get("domain"), f"subfield {identifier}.domain"
        )
        require(field_id in fields and domain_id in domains, "subfield parent is missing")
        require(fields[field_id]["display_name"] == field_name, "subfield field name mismatch")
        require(domains[domain_id]["display_name"] == domain_name, "subfield domain name mismatch")
        require(fields[field_id]["domain"]["id"] == domain_id, "subfield parent chain mismatch")
        subfields_by_field[field_id].add(identifier)

    for identifier, row in topics.items():
        subfield_id, subfield_name = _id_name(
            row.get("subfield"), f"topic {identifier}.subfield"
        )
        field_id, field_name = _id_name(row.get("field"), f"topic {identifier}.field")
        domain_id, domain_name = _id_name(row.get("domain"), f"topic {identifier}.domain")
        require(
            subfield_id in subfields and field_id in fields and domain_id in domains,
            "topic parent is missing",
        )
        require(
            subfields[subfield_id]["display_name"] == subfield_name,
            "topic subfield name mismatch",
        )
        require(fields[field_id]["display_name"] == field_name, "topic field name mismatch")
        require(domains[domain_id]["display_name"] == domain_name, "topic domain name mismatch")
        require(subfields[subfield_id]["field"]["id"] == field_id, "topic field chain mismatch")
        require(subfields[subfield_id]["domain"]["id"] == domain_id, "topic domain chain mismatch")
        topics_by_subfield[subfield_id].add(identifier)

    for identifier, row in domains.items():
        declared = _declared_child_ids(row.get("fields", []), "domain.fields", fields)
        require(declared == fields_by_domain[identifier], "domain child field set mismatch")
    for identifier, row in fields.items():
        declared = _declared_child_ids(
            row.get("subfields", []), "field.subfields", subfields
        )
        require(declared == subfields_by_field[identifier], "field child subfield set mismatch")
    for identifier, row in subfields.items():
        declared = _declared_child_ids(
            row.get("topics", []), "subfield.topics", topics
        )
        require(declared == topics_by_subfield[identifier], "subfield child topic set mismatch")

    duplicate_names: dict[str, list[dict[str, Any]]] = {}
    distinct_name_counts: dict[str, int] = {}
    for entity, mapping in maps.items():
        by_name: dict[str, list[str]] = defaultdict(list)
        for identifier, row in mapping.items():
            by_name[str(row["display_name"])].append(identifier)
        distinct_name_counts[entity] = len(by_name)
        duplicate_names[entity] = [
            {"display_name": name, "entity_ids": sorted(identifiers)}
            for name, identifiers in sorted(by_name.items())
            if len(identifiers) > 1
        ]

    field_taxonomy: dict[int, dict[str, Any]] = {}
    for identifier, row in fields.items():
        field_id = _entity_numeric_id("fields", identifier)
        domain_identifier = str(row["domain"]["id"])
        descendant_topic_count = sum(
            len(topics_by_subfield[subfield_id])
            for subfield_id in subfields_by_field[identifier]
        )
        field_taxonomy[field_id] = {
            "field_id": field_id,
            "field_name": str(row["display_name"]),
            "domain_id": _entity_numeric_id("domains", domain_identifier),
            "domain_name": str(domains[domain_identifier]["display_name"]),
            "subfield_count": len(subfields_by_field[identifier]),
            "topic_count": descendant_topic_count,
        }

    return {
        "counts": observed_counts,
        "distinct_display_name_counts": distinct_name_counts,
        "duplicate_display_names": duplicate_names,
        "canonical_sha256": observed_canonical_sha256,
        "file_commitments": file_commitments,
        "field_taxonomy": field_taxonomy,
        "internal_graph_checks": {
            "unique_entity_ids": True,
            "parent_ids_resolve": True,
            "parent_names_match": True,
            "parent_chains_match": True,
            "declared_child_sets_match": True,
            "declared_child_names_match": True,
            "declared_child_ids_unique": True,
        },
    }


def validate_success_contract(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    success_contract = require_mapping(
        config.get("success_contract"), "success_contract"
    )
    success_status = str(success_contract.get("success_status", ""))
    require(success_status == SUCCESS_STATUS, "success status contract mismatch")
    required_gate = require_mapping(
        success_contract.get("required_gate_status"),
        "required_gate_status",
    )
    gate_status = dict(required_gate)
    require(gate_status.get("pass_g1") is False, "pass_g1 must remain false")
    require(gate_status.get("field_entity_validity") == "UNVERIFIED", "bad field gate")
    require(
        gate_status.get("candidate_eligibility") == "NOT_ESTABLISHED",
        "bad candidate gate",
    )
    return success_status, gate_status


def validate_parent_reports(
    config: Mapping[str, Any],
    full_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    parent = require_mapping(
        config.get("parent_work_source_profile"), "parent_work_source_profile"
    )
    compact_contract = require_mapping(parent.get("compact_summary"), "compact_summary")
    compact_relative = safe_relative_path(
        compact_contract.get("relative_path"), "compact_summary.relative_path"
    )
    compact_path = REPO_ROOT / compact_relative
    compact_sha256 = sha256_file(compact_path)
    require(
        compact_sha256
        == require_sha256(compact_contract.get("sha256"), "compact_summary.sha256"),
        "compact parent summary SHA-256 mismatch",
    )
    compact = read_json(compact_path, "compact parent summary")

    full_contract = require_mapping(parent.get("full_report"), "full_report")
    require(
        full_report_path.name == str(full_contract.get("expected_basename")),
        "unexpected full report basename",
    )
    full_sha256 = sha256_file(full_report_path)
    require(
        full_sha256 == require_sha256(full_contract.get("sha256"), "full_report.sha256"),
        "full parent report SHA-256 mismatch",
    )
    full = read_json(full_report_path, "full parent report")

    required_status = str(compact_contract.get("required_status", ""))
    required_fingerprint = require_sha256(
        compact_contract.get("result_fingerprint_sha256"),
        "compact_summary.result_fingerprint_sha256",
    )
    require(compact.get("status") == required_status, "compact parent status mismatch")
    require(
        compact.get("provenance", {}).get("result_fingerprint_sha256")
        == required_fingerprint,
        "compact parent fingerprint mismatch",
    )
    require(full.get("status") == full_contract.get("required_status"), "full status mismatch")
    require(
        full.get("provenance", {}).get("result_fingerprint_sha256")
        == full_contract.get("result_fingerprint_sha256"),
        "full parent fingerprint mismatch",
    )
    require(
        compact.get("provenance", {}).get("result_fingerprint_sha256")
        == full.get("provenance", {}).get("result_fingerprint_sha256"),
        "compact/full parent fingerprints differ",
    )
    return compact, full, {
        "compact_summary_sha256": compact_sha256,
        "full_report_sha256": full_sha256,
        "parent_result_fingerprint_sha256": required_fingerprint,
    }


def _empty_counts(keys: Iterable[str]) -> dict[str, int]:
    return {key: 0 for key in keys}


def _window_contains(window: Mapping[str, Any], year: int) -> bool:
    start = window.get("start_year_inclusive")
    end = window.get("end_year_inclusive")
    return (start is None or year >= int(start)) and (end is None or year <= int(end))


def project_profile(
    full_report: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    profile = require_mapping(full_report.get("author_work_profile"), "author_work_profile")
    by_field = require_list(profile.get("by_primary_field"), "by_primary_field")
    cube = require_list(profile.get("joint_sparse_cube"), "joint_sparse_cube")
    parent_contract = require_mapping(
        contract.get("parent_work_source_profile"), "parent_work_source_profile"
    )
    full_contract = require_mapping(parent_contract.get("full_report"), "full_report")
    require(
        len(cube) == require_int(
            full_contract.get("required_joint_sparse_cube_cells"),
            "required_joint_sparse_cube_cells",
        ),
        "joint sparse cube cell count mismatch",
    )
    required_master_rows = require_int(
        full_contract.get("required_main_rows_profiled"), "required_main_rows_profiled"
    )
    require(
        profile.get("master_work_rows_profiled") == required_master_rows,
        "master work row count mismatch",
    )

    profile_contract = require_mapping(contract.get("profile_projection"), "profile_projection")
    statuses = [
        str(value)
        for value in require_list(
            profile_contract.get("expected_author_id_statuses"),
            "expected_author_id_statuses",
        )
    ]
    status_set = set(statuses)
    require(len(status_set) == len(statuses), "duplicate expected author status")
    windows = [
        require_mapping(value, "temporal window")
        for value in require_list(
            profile_contract.get("temporal_windows"), "temporal_windows"
        )
    ]
    window_keys = [str(window.get("key", "")) for window in windows]
    require(all(window_keys) and len(set(window_keys)) == len(window_keys), "bad window keys")
    snapshot_year = require_int(profile_contract.get("snapshot_year"), "snapshot_year")

    margin: dict[int | None, dict[str, int]] = {}
    for index, raw_row in enumerate(by_field):
        row = require_mapping(raw_row, f"by_primary_field[{index}]")
        raw_field_id = row.get("field_id")
        require(
            raw_field_id is None or type(raw_field_id) is int,
            "primary field ID must be integer or null",
        )
        field_id = raw_field_id
        require(field_id not in margin, "duplicate primary-field margin bucket")
        raw_statuses = require_mapping(row.get("statuses"), "primary-field statuses")
        require(set(raw_statuses) == status_set, "primary-field status set mismatch")
        margin[field_id] = {}
        for status in statuses:
            value = require_int(raw_statuses.get(status), f"status {status}")
            require(value >= 0, "negative field status count")
            margin[field_id][status] = value

    expected_field_ids = set(require_mapping(taxonomy.get("field_taxonomy"), "field_taxonomy"))
    require(
        set(margin) == expected_field_ids | {None},
        "profile field IDs do not equal taxonomy fields plus null",
    )
    require(
        len(margin)
        == require_int(
            profile_contract.get("expected_primary_field_buckets_including_null"),
            "expected_primary_field_buckets_including_null",
        ),
        "primary-field bucket count mismatch",
    )

    cube_status: dict[int | None, dict[str, int]] = defaultdict(
        lambda: _empty_counts(statuses)
    )
    cube_windows: dict[int | None, dict[str, int]] = defaultdict(
        lambda: _empty_counts(window_keys)
    )
    cube_diagnostics: dict[int | None, dict[str, int]] = defaultdict(
        lambda: {
            "publication_year_null": 0,
            "snapshot_year": 0,
            "future_after_snapshot_year": 0,
            "pre_1900": 0,
        }
    )
    observed_years: list[int] = []
    total_cube_works = 0

    for index, raw_cell in enumerate(cube):
        cell = require_mapping(raw_cell, f"joint_sparse_cube[{index}]")
        raw_field_id = cell.get("primary_field_id")
        require(
            raw_field_id is None or type(raw_field_id) is int,
            "cube primary field ID must be integer or null",
        )
        field_id = raw_field_id
        require(field_id in margin, "cube contains an undeclared field bucket")
        status = str(cell.get("author_id_status", ""))
        require(status in status_set, "cube contains an unexpected author status")
        works = require_int(cell.get("works"), "cube works")
        require(works > 0, "joint sparse cube must contain positive cells only")
        year = cell.get("publication_year")
        require(year is None or type(year) is int, "cube year must be integer or null")

        cube_status[field_id][status] += works
        total_cube_works += works
        if year is None:
            cube_diagnostics[field_id]["publication_year_null"] += works
            continue
        observed_years.append(year)
        for window, key in zip(windows, window_keys, strict=True):
            if _window_contains(window, year):
                cube_windows[field_id][key] += works
        if year == snapshot_year:
            cube_diagnostics[field_id]["snapshot_year"] += works
        if year > snapshot_year:
            cube_diagnostics[field_id]["future_after_snapshot_year"] += works
        if year < 1900:
            cube_diagnostics[field_id]["pre_1900"] += works

    require(total_cube_works == required_master_rows, "joint cube total mismatch")
    for field_id, statuses_by_field in margin.items():
        require(
            cube_status[field_id] == statuses_by_field,
            "joint cube projection does not reproduce by-primary-field margin",
        )

    taxonomy_fields = require_mapping(taxonomy.get("field_taxonomy"), "field_taxonomy")
    domain_map: dict[int, dict[str, Any]] = {}
    field_records: list[dict[str, Any]] = []
    for field_id in sorted(expected_field_ids):
        metadata = require_mapping(taxonomy_fields[field_id], f"taxonomy field {field_id}")
        status_counts = dict(margin[field_id])
        works = sum(status_counts.values())
        record = {
            **metadata,
            "works": works,
            "authorship_status_counts": status_counts,
            "authorship_status_rates": {
                key: status_counts[key] / works for key in statuses
            },
            "temporal_window_counts": dict(cube_windows[field_id]),
            "temporal_diagnostics": dict(cube_diagnostics[field_id]),
        }
        field_records.append(record)
        domain_id = require_int(metadata.get("domain_id"), "domain_id")
        if domain_id not in domain_map:
            domain_map[domain_id] = {
                "domain_id": domain_id,
                "domain_name": str(metadata["domain_name"]),
                "field_count": 0,
                "subfield_count": 0,
                "topic_count": 0,
                "works": 0,
                "authorship_status_counts": _empty_counts(statuses),
                "temporal_window_counts": _empty_counts(window_keys),
                "temporal_diagnostics": _empty_counts(cube_diagnostics[field_id]),
                "fields": [],
            }
        domain = domain_map[domain_id]
        require(domain["domain_name"] == metadata["domain_name"], "domain name drift")
        domain["field_count"] += 1
        domain["subfield_count"] += metadata["subfield_count"]
        domain["topic_count"] += metadata["topic_count"]
        domain["works"] += works
        for key, value in status_counts.items():
            domain["authorship_status_counts"][key] += value
        for key, value in cube_windows[field_id].items():
            domain["temporal_window_counts"][key] += value
        for key, value in cube_diagnostics[field_id].items():
            domain["temporal_diagnostics"][key] += value
        domain["fields"].append(record)

    domains = [domain_map[key] for key in sorted(domain_map)]
    require(len(domains) == taxonomy["counts"]["domains"], "domain count mismatch")
    classified_works = sum(domain["works"] for domain in domains)
    null_status_counts = dict(margin[None])
    unclassified_works = sum(null_status_counts.values())
    require(
        classified_works + unclassified_works == required_master_rows,
        "classified plus null-field rows do not conserve master works",
    )

    global_diagnostics = {
        key: sum(value[key] for value in cube_diagnostics.values())
        for key in next(iter(cube_diagnostics.values()))
    }
    global_windows = {
        key: sum(value[key] for value in cube_windows.values()) for key in window_keys
    }
    return {
        "master_work_rows": required_master_rows,
        "classified_primary_field_works": classified_works,
        "classified_primary_field_rate": classified_works / required_master_rows,
        "null_primary_field_works": unclassified_works,
        "null_primary_field_rate": unclassified_works / required_master_rows,
        "global_temporal_window_counts": global_windows,
        "global_temporal_diagnostics": {
            **global_diagnostics,
            "observed_nonnull_year_min": min(observed_years),
            "observed_nonnull_year_max": max(observed_years),
        },
        "null_primary_field_profile": {
            "works": unclassified_works,
            "authorship_status_counts": null_status_counts,
            "temporal_window_counts": dict(cube_windows[None]),
            "temporal_diagnostics": dict(cube_diagnostics[None]),
        },
        "domains": domains,
        "field_records": field_records,
        "projection_checks": {
            "joint_cube_cells": len(cube),
            "joint_cube_total_equals_master_rows": True,
            "joint_cube_reproduces_all_27_primary_field_status_margins": True,
            "taxonomy_field_ids_equal_profile_nonnull_field_ids": True,
            "domain_field_work_counts_conserve_classified_works": True,
        },
    }


def build_summary(
    *,
    config_path: Path,
    taxonomy_root: Path,
    full_report_path: Path,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    config = read_config(config_path)
    compact, full_report, parent_hashes = validate_parent_reports(
        config, full_report_path
    )
    taxonomy_contract = require_mapping(config.get("taxonomy"), "taxonomy")
    taxonomy = load_and_validate_taxonomy(taxonomy_root, taxonomy_contract)
    projection = project_profile(full_report, taxonomy, config)

    success_status, gate_status = validate_success_contract(config)

    taxonomy_public = {
        "snapshot_date": taxonomy_contract.get("snapshot_date"),
        "entity_counts": taxonomy["counts"],
        "distinct_display_name_counts": taxonomy["distinct_display_name_counts"],
        "duplicate_display_names": taxonomy["duplicate_display_names"],
        "canonical_sha256": taxonomy["canonical_sha256"],
        "internal_graph_checks": taxonomy["internal_graph_checks"],
    }
    profile_public = {
        key: projection[key]
        for key in (
            "master_work_rows",
            "classified_primary_field_works",
            "classified_primary_field_rate",
            "null_primary_field_works",
            "null_primary_field_rate",
            "global_temporal_window_counts",
            "global_temporal_diagnostics",
            "null_primary_field_profile",
            "projection_checks",
        )
    }
    scope = {
        "work_parquet_files_read": 0,
        "work_parquet_rows_scanned": 0,
        "parent_full_report_reused": True,
        "parent_joint_sparse_cube_cells_reprojected": projection["projection_checks"][
            "joint_cube_cells"
        ],
        "taxonomy_entity_rows_read": sum(taxonomy["counts"].values()),
        "final_field_selection_performed": False,
    }
    field_selection = {
        "status": "PENDING_G0",
        "selected_field_ids": [],
        "selected_domain_ids": [],
        "final_domain_holdout": None,
        "selection_used_scientific_outcomes": False,
    }
    pending = require_mapping(
        config.get("pending_full_taxonomy_reference_audit"),
        "pending_full_taxonomy_reference_audit",
    )
    pending_public = {
        "status": pending.get("status"),
        "workload_class": pending.get("workload_class"),
        "main_files": pending.get("main_files"),
        "topic_files": pending.get("topic_files"),
        "total_files": pending.get("total_files"),
        "main_raw_rows": pending.get("main_raw_rows"),
        "main_retained_rows": pending.get("main_retained_rows"),
        "topic_rows": pending.get("topic_rows"),
        "required_extensions": pending.get("required_extensions"),
    }

    fingerprint_payload = {
        "status": success_status,
        "gate_status": gate_status,
        "scope": scope,
        "taxonomy": taxonomy_public,
        "profile": profile_public,
        "domains": projection["domains"],
        "field_selection": field_selection,
        "pending_full_taxonomy_reference_audit": pending_public,
        "claim_boundary": config.get("claim_boundary"),
        "parent_hashes": parent_hashes,
    }
    result_fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_payload))
    config_relative = config_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    tool_path = Path(__file__).resolve()
    tool_relative = tool_path.relative_to(REPO_ROOT).as_posix()

    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": "openalex_g1_cross_field_feasibility_20260718",
        "generated_at_utc": generated_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "status": success_status,
        "gate_status": gate_status,
        "scope": scope,
        "taxonomy": taxonomy_public,
        "profile": profile_public,
        "domains": projection["domains"],
        "field_selection": field_selection,
        "pending_full_taxonomy_reference_audit": pending_public,
        "claim_boundary": config.get("claim_boundary"),
        "provenance": {
            "config_relative_path": config_relative,
            "config_sha256": sha256_file(config_path),
            "tool_relative_path": tool_relative,
            "tool_sha256": sha256_file(tool_path),
            **parent_hashes,
            "parent_compact_status": compact.get("status"),
            "parent_full_report_status": full_report.get("status"),
            "taxonomy_canonical_sha256": taxonomy["canonical_sha256"],
            "taxonomy_file_commitments": taxonomy["file_commitments"],
            "result_fingerprint_sha256": result_fingerprint,
            "path_redacted": True,
        },
    }
    require("/home/" not in json.dumps(summary), "summary leaked a local home path")
    require("/hdd/" not in json.dumps(summary), "summary leaked a local HDD path")
    require("/ssd/" not in json.dumps(summary), "summary leaked a local SSD path")
    return summary


def _format_count(value: int) -> str:
    return f"{value:,}"


def _format_rate(value: float) -> str:
    return f"{value * 100:.6f}%"


def render_markdown(summary: Mapping[str, Any]) -> str:
    profile = summary["profile"]
    scope = summary["scope"]
    snapshot_year = int(str(summary["taxonomy"]["snapshot_date"]).split("-", 1)[0])
    duplicate_subfield_names = summary["taxonomy"]["duplicate_display_names"][
        "subfields"
    ]
    duplicate_subfield_id_count = sum(
        len(item["entity_ids"]) for item in duplicate_subfield_names
    )
    lines = [
        "# OpenAlex G-1 Cross-Field Profile Feasibility",
        "",
        "## Status and claim boundary",
        "",
        f"- Component status: `{summary['status']}`",
        f"- Work Parquet files read: `{_format_count(scope['work_parquet_files_read'])}`",
        f"- Work Parquet rows rescanned: `{_format_count(scope['work_parquet_rows_scanned'])}`",
        "- This is an exact reprojection of the frozen parent profile plus a",
        "  validation of the small taxonomy graph. It is not field-entity",
        "  validation, candidate selection, G-1, G0, empirical G1, or a",
        "  scientific result.",
        "",
        "## Taxonomy graph",
        "",
        "| Entity | Rows | Distinct display names |",
        "| --- | ---: | ---: |",
    ]
    for entity in ("domains", "fields", "subfields", "topics"):
        lines.append(
            f"| {entity} | {_format_count(summary['taxonomy']['entity_counts'][entity])} "
            f"| {_format_count(summary['taxonomy']['distinct_display_name_counts'][entity])} |"
        )
    lines.extend(
        [
            "",
            "All taxonomy IDs are unique, all parent IDs resolve, all embedded",
            "parent names and chains agree, and all declared child IDs and names",
            "agree with unique, exact child sets.",
            "Display names remain labels rather than join keys; "
            f"{_format_count(len(duplicate_subfield_names))} subfield display names are",
            f"shared across {_format_count(duplicate_subfield_id_count)} distinct subfield IDs.",
            "",
            "## Global profile diagnostics",
            "",
            "| Diagnostic | Exact value |",
            "| --- | ---: |",
            f"| Retained master works | {_format_count(profile['master_work_rows'])} |",
            f"| Non-null primary field | {_format_count(profile['classified_primary_field_works'])} |",
            f"| Non-null primary-field rate | {_format_rate(profile['classified_primary_field_rate'])} |",
            f"| Null primary field | {_format_count(profile['null_primary_field_works'])} |",
            f"| Null primary-field rate | {_format_rate(profile['null_primary_field_rate'])} |",
            f"| Null publication year | {_format_count(profile['global_temporal_diagnostics']['publication_year_null'])} |",
            f"| Snapshot-year {snapshot_year} | {_format_count(profile['global_temporal_diagnostics']['snapshot_year'])} |",
            f"| Publication year after {snapshot_year} | {_format_count(profile['global_temporal_diagnostics']['future_after_snapshot_year'])} |",
            f"| Publication year before 1900 | {_format_count(profile['global_temporal_diagnostics']['pre_1900'])} |",
            f"| Minimum non-null year | {profile['global_temporal_diagnostics']['observed_nonnull_year_min']} |",
            f"| Maximum non-null year | {profile['global_temporal_diagnostics']['observed_nonnull_year_max']} |",
            "",
            "## Domain totals",
            "",
            "| Domain ID | Domain | Fields | Subfields | Topics | Works | Complete | Partial | None usable | Zero authorship |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain in summary["domains"]:
        statuses = domain["authorship_status_counts"]
        lines.append(
            f"| {domain['domain_id']} | {domain['domain_name']} | {domain['field_count']} "
            f"| {domain['subfield_count']} | {domain['topic_count']} "
            f"| {_format_count(domain['works'])} | {_format_count(statuses['complete'])} "
            f"| {_format_count(statuses['partial'])} | {_format_count(statuses['none_usable'])} "
            f"| {_format_count(statuses['zero_authorship_row'])} |"
        )
    lines.extend(
        [
            "",
            "## All 26 fields",
            "",
            "| Domain ID | Domain | Field ID | Field | Subfields | Topics | Works | Complete | Partial | None usable | Zero authorship | Through 2012 | 2013-16 | 2017-19 | 2020-22 | 2023-25 | Null year | After 2026 |",
            "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain in summary["domains"]:
        for field in domain["fields"]:
            statuses = field["authorship_status_counts"]
            windows = field["temporal_window_counts"]
            diagnostics = field["temporal_diagnostics"]
            lines.append(
                f"| {domain['domain_id']} | {domain['domain_name']} | {field['field_id']} "
                f"| {field['field_name']} | {field['subfield_count']} | {field['topic_count']} "
                f"| {_format_count(field['works'])} | {_format_count(statuses['complete'])} "
                f"| {_format_count(statuses['partial'])} | {_format_count(statuses['none_usable'])} "
                f"| {_format_count(statuses['zero_authorship_row'])} "
                f"| {_format_count(windows['discovery_through_2012'])} "
                f"| {_format_count(windows['g1_replication_2013_2016'])} "
                f"| {_format_count(windows['model_selection_2017_2019'])} "
                f"| {_format_count(windows['g4_temporal_2020_2022'])} "
                f"| {_format_count(windows['followup_diagnostic_2023_2025'])} "
                f"| {_format_count(diagnostics['publication_year_null'])} "
                f"| {_format_count(diagnostics['future_after_snapshot_year'])} |"
            )

    null_field = profile["null_primary_field_profile"]
    lines.extend(
        [
            "",
            "## Null primary-field bucket",
            "",
            "| Works | Complete | Partial | None usable | Zero authorship | Null year | After 2026 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {_format_count(null_field['works'])} "
                f"| {_format_count(null_field['authorship_status_counts']['complete'])} "
                f"| {_format_count(null_field['authorship_status_counts']['partial'])} "
                f"| {_format_count(null_field['authorship_status_counts']['none_usable'])} "
                f"| {_format_count(null_field['authorship_status_counts']['zero_authorship_row'])} "
                f"| {_format_count(null_field['temporal_diagnostics']['publication_year_null'])} "
                f"| {_format_count(null_field['temporal_diagnostics']['future_after_snapshot_year'])} |"
            ),
            "",
            "## Gate status",
            "",
            "| Gate | Status |",
            "| --- | --- |",
        ]
    )
    for key, value in summary["gate_status"].items():
        rendered = "false" if value is False else str(value)
        lines.append(f"| `{key}` | `{rendered}` |")
    pending = summary["pending_full_taxonomy_reference_audit"]
    gate_status = summary["gate_status"]
    rendered_pass_g1 = (
        "false" if gate_status["pass_g1"] is False else str(gate_status["pass_g1"])
    )
    lines.extend(
        [
            "",
            "## Taxonomy-reference audit extension",
            "",
            f"- Declared status: `{pending['status']}`",
            f"- Workload class: `{pending['workload_class']}`",
            "",
            f"The declared extension covers exactly "
            f"{_format_count(pending['main_files'])} main files plus "
            f"{_format_count(pending['topic_files'])} topic files "
            f"({_format_count(pending['total_files'])} total). It must validate",
            "work-to-taxonomy references, hierarchy labels, duplicate work-topic",
            "edges, primary-topic inclusion, topic scores, and publication dates.",
            "",
            "No final field or domain has been selected. Field selection remains",
            f"`{summary['field_selection']['status']}`, G-1 remains "
            f"`{gate_status['g_minus_1_data_readiness']}`, G0 remains "
            f"`{gate_status['g0_measurement_freeze']}`, empirical G1 remains",
            f"`{gate_status['empirical_g1_signature']}`, and "
            f"`pass_g1={rendered_pass_g1}`.",
            "",
            "## Provenance",
            "",
            f"- Result fingerprint SHA-256: `{summary['provenance']['result_fingerprint_sha256']}`",
            f"- Config SHA-256: `{summary['provenance']['config_sha256']}`",
            f"- Tool SHA-256: `{summary['provenance']['tool_sha256']}`",
            f"- Parent compact summary SHA-256: `{summary['provenance']['compact_summary_sha256']}`",
            f"- Parent full report SHA-256: `{summary['provenance']['full_report_sha256']}`",
            f"- Parent result fingerprint SHA-256: `{summary['provenance']['parent_result_fingerprint_sha256']}`",
            f"- Taxonomy canonical SHA-256: `{summary['provenance']['taxonomy_canonical_sha256']}`",
            "- Operator-local input paths are redacted.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    require("/home/" not in markdown, "Markdown leaked a local home path")
    require("/hdd/" not in markdown, "Markdown leaked a local HDD path")
    require("/ssd/" not in markdown, "Markdown leaked a local SSD path")
    return markdown


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    require(not temporary.exists() and not temporary.is_symlink(), "temporary output exists")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--taxonomy-root", type=Path)
    parser.add_argument("--full-report", type=Path)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--generated-at-utc")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_config(config_path)
    taxonomy_root = resolve_env_path(
        args.taxonomy_root,
        require_mapping(config.get("taxonomy"), "taxonomy"),
        field="taxonomy root",
    )
    full_report_contract = require_mapping(
        require_mapping(
            config.get("parent_work_source_profile"), "parent_work_source_profile"
        ).get("full_report"),
        "full_report",
    )
    full_report_path = resolve_env_path(
        args.full_report,
        full_report_contract,
        field="full report",
    )
    summary = build_summary(
        config_path=config_path,
        taxonomy_root=taxonomy_root,
        full_report_path=full_report_path,
        generated_at_utc=args.generated_at_utc,
    )
    atomic_write_text(
        args.output_json,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(args.output_markdown, render_markdown(summary))
    print(json.dumps({"status": summary["status"], "gate_status": summary["gate_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
