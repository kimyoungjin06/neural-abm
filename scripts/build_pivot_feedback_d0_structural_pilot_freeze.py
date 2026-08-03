#!/usr/bin/env python3
"""Freeze the outcome-blind D0-A pilot fields and attention episodes.

This component deliberately stops before author-level entry opportunities,
leave-out attention, persistence labels, citation-success proxies, or fitting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "pivot_feedback_d0_structural_pilot_freeze_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_structural_pilot_freeze_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_structural_pilot_freeze_outputs.v1"
FIELD_SCHEMA = "pivot_feedback_d0_field_structural_ranking.v1"
TOPIC_YEAR_SCHEMA = "pivot_feedback_d0_selected_field_topic_year_attention.v1"
CANDIDATE_SCHEMA = "pivot_feedback_d0_episode_candidates.v1"
STATUS = "PASS_D0A_OUTCOME_BLIND_TWO_FIELD_FOUR_EPISODE_STRUCTURAL_FREEZE"
PARTITIONS = 4096
MAX_CONTROL_BYTES = 32 * 1024 * 1024
ID_RE = {
    "topic": re.compile(r"^https://openalex[.]org/T([1-9][0-9]*)$"),
    "field": re.compile(r"^https://openalex[.]org/fields/([1-9][0-9]*)$"),
    "subfield": re.compile(r"^https://openalex[.]org/subfields/([1-9][0-9]*)$"),
}
WORK_COLUMNS = (
    "publication_year",
    "temporal_disposition",
    "primary_topic_id",
    "primary_field_id",
)
FIELD_METRICS = (
    "work_volume",
    "usable_date_rate",
    "history_usable_author_work_rate",
    "stable_orcid_author_work_density",
    "affiliation_row_density",
    "institution_row_density",
    "cap100_graph_nodes",
    "cap100_graph_edges",
)


class D0StructuralFreezeError(RuntimeError):
    """Fail-closed D0-A structural-freeze error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise D0StructuralFreezeError(message)


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
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0StructuralFreezeError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise D0StructuralFreezeError(f"invalid {label}: {error}") from error


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a string")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / value).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _validate_repo_binding(
    binding: Mapping[str, Any],
    *,
    yaml_document: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = dict(binding)
    path = _safe_repo_path(item.get("relative_path"), "repository binding")
    observed_sha = _sha256(path)
    require(observed_sha == item.get("sha256"), f"binding SHA mismatch: {path.name}")
    document = (
        _read_yaml(path, path.name)
        if yaml_document
        else _read_json(path, path.name)
    )
    require(
        document.get("schema_version") == item.get("required_schema_version"),
        f"binding schema mismatch: {path.name}",
    )
    if "required_status" in item:
        require(
            document.get("status") == item.get("required_status"),
            f"binding status mismatch: {path.name}",
        )
    for dotted_path, expected in _mapping(
        item.get("required_fields", {}),
        f"{path.name} required fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"binding field mismatch: {path.name}:{dotted_path}",
        )
    return document, {
        "relative_path": item["relative_path"],
        "sha256": observed_sha,
        "schema_version": document["schema_version"],
        "status": document.get("status"),
    }


def _receipt_artifacts(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
        relative = item.get("relative_path")
        require(isinstance(relative, str) and relative, "bad artifact path")
        require(relative not in result, "duplicate artifact path")
        result[relative] = item
    return result


def _validate_bound_parquet(
    root: Path,
    binding: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid normalized root")
    require(root.name == binding.get("expected_root_basename"), "root basename mismatch")
    receipt_path = root / str(binding.get("receipt_basename"))
    require(_sha256(receipt_path) == binding.get("receipt_sha256"), "receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "receipt status mismatch")
    artifact_contract = _mapping(binding.get("artifact"), "artifact contract")
    relative = str(artifact_contract.get("relative_path"))
    declared = _receipt_artifacts(receipt).get(relative)
    require(declared is not None, f"receipt omits {relative}")
    require(declared.get("sha256") == artifact_contract.get("sha256"), "artifact receipt SHA differs")
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), "artifact escapes normalized root")
    require(path.is_file() and not path.is_symlink(), "invalid normalized artifact")
    require(_sha256(path) == artifact_contract.get("sha256"), "artifact SHA mismatch")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == artifact_contract.get("rows"), "artifact row mismatch")
    return path, {
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_relative_path": relative,
        "artifact_sha256": artifact_contract["sha256"],
        "artifact_rows": parquet.metadata.num_rows,
    }


def _validate_work_metadata(
    root: Path,
    binding: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[list[Path], dict[str, Any], list[tuple[int, int, int, int, int]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid cutoff root")
    require(root.name == binding.get("expected_root_basename"), "cutoff root basename mismatch")
    receipt_path = root / str(binding.get("receipt_basename"))
    require(_sha256(receipt_path) == binding.get("receipt_sha256"), "cutoff receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "cutoff receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "cutoff status mismatch")
    prefix = str(binding.get("artifact_prefix"))
    declared = [
        item
        for relative, item in _receipt_artifacts(receipt).items()
        if relative.startswith(prefix)
    ]
    declared.sort(key=lambda item: str(item["relative_path"]))
    require(len(declared) == binding.get("artifact_count") == PARTITIONS, "work partition count mismatch")
    paths: list[Path] = []
    pre_stats: list[tuple[int, int, int, int, int]] = []
    for index, item in enumerate(declared):
        expected = f"{prefix}part-{index:04d}.parquet"
        require(item.get("relative_path") == expected, "work partition sequence mismatch")
        path = (root / expected).resolve(strict=True)
        require(path.is_relative_to(root), "work partition escapes root")
        require(path.is_file() and not path.is_symlink(), "invalid work partition")
        stat = path.stat()
        require(stat.st_size == item.get("bytes"), "work partition byte mismatch")
        paths.append(path)
        pre_stats.append(
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        )

    def rehash(pair: tuple[Path, Mapping[str, Any]]) -> tuple[str, int, str]:
        path, item = pair
        observed = _sha256(path)
        require(observed == item.get("sha256"), f"work partition SHA mismatch: {path.name}")
        return str(item["relative_path"]), int(item["bytes"]), observed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(rehash, zip(paths, declared, strict=True)))
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
    post_stats = []
    for path in paths:
        stat = path.stat()
        post_stats.append(
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        )
    require(list(pre_stats) == post_stats, "work metadata changed during D0-A scan")


def _parse_id(value: Any, entity: str) -> int:
    require(isinstance(value, str), f"{entity} ID is not text")
    match = ID_RE[entity].fullmatch(value)
    require(match is not None, f"invalid {entity} ID")
    return int(match.group(1))


def _load_taxonomy(
    path: Path,
    binding: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    path = path.resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), "invalid topic taxonomy projection")
    require(path.name == binding.get("expected_basename"), "taxonomy basename mismatch")
    require(_sha256(path) == binding.get("sha256"), "taxonomy projection SHA mismatch")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == binding.get("rows"), "taxonomy row count mismatch")
    schema = parquet.schema_arrow
    for name, expected_type in _mapping(
        binding.get("required_columns"),
        "taxonomy required columns",
    ).items():
        require(name in schema.names, f"taxonomy column missing: {name}")
        require(str(schema.field(name).type) == expected_type, f"taxonomy type mismatch: {name}")
    table = pq.read_table(path, columns=list(binding["required_columns"]))
    topics: dict[int, dict[str, Any]] = {}
    for row in table.to_pylist():
        topic_id = _parse_id(row["id"], "topic")
        field_id = _parse_id(row["field__id"], "field")
        subfield_id = _parse_id(row["subfield__id"], "subfield")
        require(topic_id not in topics, "duplicate taxonomy topic")
        topics[topic_id] = {
            "topic_id": topic_id,
            "topic_name": str(row["display_name"]),
            "field_id": field_id,
            "field_name": str(row["field__display_name"]),
            "subfield_id": subfield_id,
            "subfield_name": str(row["subfield__display_name"]),
        }
    require(len(topics) == binding.get("rows"), "taxonomy unique-topic closure failed")
    return topics, {
        "basename": path.name,
        "sha256": binding["sha256"],
        "rows": len(topics),
        "role": binding["role"],
        "canonical_taxonomy_parent_sha256": binding[
            "canonical_taxonomy_parent_sha256"
        ],
    }


def _candidate_universe(document: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    universe = _mapping(document.get("candidate_universe"), "candidate universe")
    fields: dict[int, dict[str, Any]] = {}
    for raw_domain in _list(universe.get("domains"), "candidate domains"):
        domain = _mapping(raw_domain, "candidate domain")
        for raw_field in _list(domain.get("fields"), "candidate fields"):
            field = _mapping(raw_field, "candidate field")
            field_id = int(field["field_id"])
            require(field_id not in fields, "duplicate candidate field")
            fields[field_id] = {
                "domain_id": str(domain["domain_id"]),
                "domain_name": str(domain["display_name"]),
                "domain_rng_code": int(domain["rng_code"]),
                "field_name": str(field["display_name"]),
                "field_slug": str(field["field_slug"]),
            }
    require(set(fields) == set(range(11, 37)), "candidate field universe mismatch")
    return fields


def _sum_by_field(
    table: pa.Table,
    *,
    entry_year_end: int,
    cap100_only: bool = False,
) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for row in table.to_pylist():
        year = row["publication_year"]
        if year is None or int(year) > entry_year_end:
            continue
        if cap100_only and int(row["declared_team_cap"]) != 100:
            continue
        field_id = int(row["field_id"])
        cell = result.setdefault(field_id, {})
        for name, value in row.items():
            if name in {"field_id", "publication_year", "declared_team_cap"}:
                continue
            if isinstance(value, (int, np.integer)):
                cell[name] = cell.get(name, 0) + int(value)
    return result


def _percentiles(values: Mapping[int, Fraction]) -> dict[int, Fraction]:
    field_ids = sorted(values)
    require(len(field_ids) == 26, "percentile field count mismatch")
    sorted_values = sorted(values.values())
    result: dict[int, Fraction] = {}
    for field_id in field_ids:
        rank = sorted_values.index(values[field_id])
        result[field_id] = Fraction(rank, len(field_ids) - 1)
    return result


def _field_ranking(
    core_path: Path,
    orcid_path: Path,
    graph_path: Path,
    *,
    fields: Mapping[int, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[pa.Table, list[dict[str, Any]], list[int]]:
    entry_year_end = int(contract["entry_year_end_inclusive"])
    core = _sum_by_field(pq.read_table(core_path), entry_year_end=entry_year_end)
    orcid = _sum_by_field(pq.read_table(orcid_path), entry_year_end=entry_year_end)
    graph = _sum_by_field(
        pq.read_table(graph_path),
        entry_year_end=entry_year_end,
        cap100_only=True,
    )
    expected_fields = [int(value) for value in contract["field_ids"]]
    require(expected_fields == list(range(11, 37)), "field-selection universe mismatch")
    require(set(core) == set(orcid) == set(graph) == set(expected_fields), "field input coverage mismatch")

    raw_rows: dict[int, dict[str, Any]] = {}
    fractions: dict[str, dict[int, Fraction]] = {metric: {} for metric in FIELD_METRICS}
    for field_id in expected_fields:
        c, o, g = core[field_id], orcid[field_id], graph[field_id]
        denominators = {
            "total_canonical_works": c["total_canonical_works"],
            "distinct_author_work_pairs": c["distinct_author_work_pairs"],
            "grammar_valid_author_id_rows": o["grammar_valid_author_id_rows"],
        }
        require(all(value > 0 for value in denominators.values()), "zero field denominator")
        metric_values = {
            "work_volume": Fraction(c["total_canonical_works"], 1),
            "usable_date_rate": Fraction(
                c["usable_date_works"],
                c["total_canonical_works"],
            ),
            "history_usable_author_work_rate": Fraction(
                c["history_usable_distinct_author_work_pairs"],
                c["distinct_author_work_pairs"],
            ),
            "stable_orcid_author_work_density": Fraction(
                o["stable_orcid_author_work_pairs"],
                o["grammar_valid_author_id_rows"],
            ),
            "affiliation_row_density": Fraction(
                o["author_work_affiliation_rows"],
                o["grammar_valid_author_id_rows"],
            ),
            "institution_row_density": Fraction(
                o["author_work_institution_rows"],
                o["grammar_valid_author_id_rows"],
            ),
            "cap100_graph_nodes": Fraction(g["graph_node_tokens"], 1),
            "cap100_graph_edges": Fraction(g["distinct_undirected_edges"], 1),
        }
        for metric, value in metric_values.items():
            fractions[metric][field_id] = value
        raw_rows[field_id] = {
            "field_id": field_id,
            **fields[field_id],
            "work_volume": c["total_canonical_works"],
            "usable_date_works": c["usable_date_works"],
            "distinct_author_work_pairs": c["distinct_author_work_pairs"],
            "history_usable_distinct_author_work_pairs": c[
                "history_usable_distinct_author_work_pairs"
            ],
            "grammar_valid_author_id_rows": o["grammar_valid_author_id_rows"],
            "stable_orcid_author_work_pairs": o["stable_orcid_author_work_pairs"],
            "author_work_affiliation_rows": o["author_work_affiliation_rows"],
            "author_work_institution_rows": o["author_work_institution_rows"],
            "cap100_graph_nodes": g["graph_node_tokens"],
            "cap100_graph_edges": g["distinct_undirected_edges"],
        }

    require(tuple(contract["metrics"]) == FIELD_METRICS, "field metric contract mismatch")
    percentiles = {
        metric: _percentiles(fractions[metric])
        for metric in FIELD_METRICS
    }
    scoring: dict[int, tuple[Fraction, Fraction]] = {}
    for field_id in expected_fields:
        scores = [percentiles[metric][field_id] for metric in FIELD_METRICS]
        scoring[field_id] = (min(scores), sum(scores, Fraction(0)) / len(scores))
    ranked = sorted(
        expected_fields,
        key=lambda field_id: (
            scoring[field_id][0],
            scoring[field_id][1],
            -field_id,
        ),
        reverse=True,
    )
    first = ranked[0]
    second = next(
        field_id
        for field_id in ranked[1:]
        if fields[field_id]["domain_id"] != fields[first]["domain_id"]
    )
    selected = [first, second]
    require(len(selected) == int(contract["required_pilot_fields"]), "pilot field count mismatch")

    rows = []
    for global_rank, field_id in enumerate(ranked, start=1):
        raw = raw_rows[field_id]
        row = {
            **raw,
            "usable_date_rate": float(fractions["usable_date_rate"][field_id]),
            "history_usable_author_work_rate": float(
                fractions["history_usable_author_work_rate"][field_id]
            ),
            "stable_orcid_author_work_density": float(
                fractions["stable_orcid_author_work_density"][field_id]
            ),
            "affiliation_row_density": float(
                fractions["affiliation_row_density"][field_id]
            ),
            "institution_row_density": float(
                fractions["institution_row_density"][field_id]
            ),
        }
        for metric in FIELD_METRICS:
            row[f"{metric}_percentile"] = float(percentiles[metric][field_id])
        row["minimum_metric_percentile"] = float(scoring[field_id][0])
        row["mean_metric_percentile"] = float(scoring[field_id][1])
        row["global_rank"] = global_rank
        row["pilot_selected"] = field_id in selected
        row["pilot_selection_order"] = (
            selected.index(field_id) + 1 if field_id in selected else None
        )
        rows.append(row)
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("domain_id", pa.string(), nullable=False),
            pa.field("domain_name", pa.string(), nullable=False),
            pa.field("domain_rng_code", pa.uint8(), nullable=False),
            pa.field("field_name", pa.string(), nullable=False),
            pa.field("field_slug", pa.string(), nullable=False),
            pa.field("work_volume", pa.uint64(), nullable=False),
            pa.field("usable_date_works", pa.uint64(), nullable=False),
            pa.field("distinct_author_work_pairs", pa.uint64(), nullable=False),
            pa.field(
                "history_usable_distinct_author_work_pairs",
                pa.uint64(),
                nullable=False,
            ),
            pa.field("grammar_valid_author_id_rows", pa.uint64(), nullable=False),
            pa.field("stable_orcid_author_work_pairs", pa.uint64(), nullable=False),
            pa.field("author_work_affiliation_rows", pa.uint64(), nullable=False),
            pa.field("author_work_institution_rows", pa.uint64(), nullable=False),
            pa.field("cap100_graph_nodes", pa.uint64(), nullable=False),
            pa.field("cap100_graph_edges", pa.uint64(), nullable=False),
            pa.field("usable_date_rate", pa.float64(), nullable=False),
            pa.field(
                "history_usable_author_work_rate",
                pa.float64(),
                nullable=False,
            ),
            pa.field(
                "stable_orcid_author_work_density",
                pa.float64(),
                nullable=False,
            ),
            pa.field("affiliation_row_density", pa.float64(), nullable=False),
            pa.field("institution_row_density", pa.float64(), nullable=False),
            *[
                pa.field(f"{metric}_percentile", pa.float64(), nullable=False)
                for metric in FIELD_METRICS
            ],
            pa.field("minimum_metric_percentile", pa.float64(), nullable=False),
            pa.field("mean_metric_percentile", pa.float64(), nullable=False),
            pa.field("global_rank", pa.uint8(), nullable=False),
            pa.field("pilot_selected", pa.bool_(), nullable=False),
            pa.field("pilot_selection_order", pa.uint8(), nullable=True),
        ],
        metadata={b"schema_version": FIELD_SCHEMA.encode()},
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    selected_rows = [
        row
        for row in table.to_pylist()
        if row["pilot_selected"]
    ]
    selected_rows.sort(key=lambda row: int(row["pilot_selection_order"]))
    return table, selected_rows, ranked


def _scan_topic_years(
    paths: Sequence[Path],
    *,
    selected_fields: Sequence[int],
    taxonomy: Mapping[int, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, int]]:
    allowed_dispositions = np.asarray(
        contract["allowed_temporal_dispositions"],
        dtype=np.uint8,
    )
    entry_year_end = int(contract["entry_year_end_inclusive"])
    latest_attention_year = entry_year_end - 1
    counts: dict[int, int] = {}
    accounting = {
        "work_rows_scanned": 0,
        "selected_field_rows_with_usable_predecision_year_and_topic": 0,
        "taxonomy_missing_topic_field_pairs": 0,
        "taxonomy_field_mismatch_topic_field_pairs": 0,
    }
    selected_array = np.asarray(selected_fields, dtype=np.uint8)
    for path in paths:
        table = pq.read_table(path, columns=list(WORK_COLUMNS))
        accounting["work_rows_scanned"] += len(table)
        years = (
            table["publication_year"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        dispositions = (
            table["temporal_disposition"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        topics = (
            table["primary_topic_id"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        fields = (
            table["primary_field_id"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        selected = (
            np.isin(fields, selected_array)
            & np.isin(dispositions, allowed_dispositions)
            & (years > 0)
            & (years <= latest_attention_year)
            & (topics > 0)
        )
        selected_topics = topics[selected]
        selected_fields_array = fields[selected]
        taxonomy_pairs = np.unique(
            (selected_fields_array.astype(np.uint64) << np.uint64(32))
            | selected_topics.astype(np.uint64)
        )
        for packed in taxonomy_pairs.tolist():
            field_id = int(packed >> 32)
            topic_id = int(packed & 0xFFFFFFFF)
            item = taxonomy.get(int(topic_id))
            if item is None:
                accounting["taxonomy_missing_topic_field_pairs"] += 1
            elif int(item["field_id"]) != int(field_id):
                accounting["taxonomy_field_mismatch_topic_field_pairs"] += 1
        require(
            accounting["taxonomy_missing_topic_field_pairs"] == 0,
            "selected primary topic missing from taxonomy",
        )
        require(
            accounting["taxonomy_field_mismatch_topic_field_pairs"] == 0,
            "selected primary topic field differs from taxonomy",
        )
        keys = (
            (fields[selected].astype(np.uint64) << np.uint64(48))
            | (topics[selected].astype(np.uint64) << np.uint64(16))
            | years[selected].astype(np.uint64)
        )
        unique, values = np.unique(keys, return_counts=True)
        for key, value in zip(unique.tolist(), values.tolist(), strict=True):
            counts[int(key)] = counts.get(int(key), 0) + int(value)
        accounting[
            "selected_field_rows_with_usable_predecision_year_and_topic"
        ] += int(selected.sum())

    rows = []
    for key in sorted(counts):
        field_id = key >> 48
        topic_id = (key >> 16) & 0xFFFFFFFF
        year = key & 0xFFFF
        metadata = taxonomy[topic_id]
        rows.append(
            {
                "field_id": field_id,
                "topic_id": topic_id,
                "topic_name": metadata["topic_name"],
                "subfield_id": metadata["subfield_id"],
                "subfield_name": metadata["subfield_name"],
                "publication_year": year,
                "usable_primary_topic_works": counts[key],
            }
        )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("subfield_name", pa.string(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=False),
            pa.field("usable_primary_topic_works", pa.uint64(), nullable=False),
        ],
        metadata={b"schema_version": TOPIC_YEAR_SCHEMA.encode()},
    )
    accounting["topic_year_cells"] = len(rows)
    accounting["distinct_topics"] = len(
        {(row["field_id"], row["topic_id"]) for row in rows}
    )
    return pa.Table.from_pylist(rows, schema=schema), accounting


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[Fraction, int, int, int, int]:
    return (
        Fraction(
            int(candidate["standardized_gain_numerator"]),
            int(candidate["standardized_gain_denominator"]),
        ),
        int(candidate["minimum_baseline_recent_count"]),
        int(candidate["absolute_gain"]),
        -int(candidate["topic_id"]),
        -int(candidate["entry_year"]),
    )


def _episode_candidates(
    topic_year: pa.Table,
    *,
    selected_fields: Sequence[int],
    contract: Mapping[str, Any],
) -> tuple[pa.Table, list[dict[str, Any]], dict[str, Any]]:
    baseline_offsets = [int(value) for value in contract["baseline_offsets"]]
    recent_offsets = [int(value) for value in contract["recent_offsets"]]
    entry_year_end = int(contract["entry_year_end_inclusive"])
    minimum_separation = int(
        _mapping(contract["pair_selection"], "pair selection")[
            "minimum_entry_year_separation"
        ]
    )
    require(baseline_offsets == [-6, -5, -4], "baseline offsets mismatch")
    require(recent_offsets == [-3, -2, -1], "recent offsets mismatch")
    by_topic: dict[tuple[int, int], dict[str, Any]] = {}
    for row in topic_year.to_pylist():
        key = (int(row["field_id"]), int(row["topic_id"]))
        item = by_topic.setdefault(
            key,
            {
                "topic_name": row["topic_name"],
                "subfield_id": int(row["subfield_id"]),
                "subfield_name": row["subfield_name"],
                "years": {},
            },
        )
        item["years"][int(row["publication_year"])] = int(
            row["usable_primary_topic_works"]
        )

    candidates_by_field: dict[int, list[dict[str, Any]]] = {}
    for (field_id, topic_id), item in by_topic.items():
        years = item["years"]
        earliest_entry = min(years) + 6
        for entry_year in range(earliest_entry, entry_year_end + 1):
            baseline = sum(years.get(entry_year + offset, 0) for offset in baseline_offsets)
            recent = sum(years.get(entry_year + offset, 0) for offset in recent_offsets)
            if baseline <= 0 or recent <= baseline:
                continue
            delta = recent - baseline
            numerator = delta * delta
            denominator = baseline + 1
            require(numerator <= np.iinfo(np.uint64).max, "episode score numerator overflow")
            candidate = {
                "field_id": field_id,
                "topic_id": topic_id,
                "topic_name": item["topic_name"],
                "subfield_id": item["subfield_id"],
                "subfield_name": item["subfield_name"],
                "entry_year": entry_year,
                "baseline_start_year": entry_year - 6,
                "baseline_end_year": entry_year - 4,
                "recent_start_year": entry_year - 3,
                "recent_end_year": entry_year - 1,
                "reserved_response_end_year": entry_year + 2,
                "baseline_works": baseline,
                "recent_works": recent,
                "absolute_gain": delta,
                "minimum_baseline_recent_count": min(baseline, recent),
                "standardized_gain_numerator": numerator,
                "standardized_gain_denominator": denominator,
                "standardized_gain": delta / math.sqrt(denominator),
                "field_candidate_rank": 0,
                "pilot_episode_selected": False,
                "pilot_episode_order": None,
            }
            candidates_by_field.setdefault(field_id, []).append(candidate)

    selected: list[dict[str, Any]] = []
    for field_id in selected_fields:
        candidates = candidates_by_field.get(field_id, [])
        require(candidates, f"no positive-attention candidate for field {field_id}")
        candidates.sort(key=_candidate_key, reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["field_candidate_rank"] = rank
        first = candidates[0]
        compatible = [
            candidate
            for candidate in candidates[1:]
            if int(candidate["subfield_id"]) != int(first["subfield_id"])
            and abs(int(candidate["entry_year"]) - int(first["entry_year"]))
            >= minimum_separation
        ]
        require(compatible, f"no structurally independent episode pair for field {field_id}")
        second = compatible[0]
        for order, candidate in enumerate((first, second), start=1):
            candidate["pilot_episode_selected"] = True
            candidate["pilot_episode_order"] = order
            selected.append(candidate)

    rows = [
        candidate
        for field_id in selected_fields
        for candidate in candidates_by_field[field_id]
    ]
    rows.sort(
        key=lambda row: (
            int(selected_fields.index(int(row["field_id"]))),
            int(row["field_candidate_rank"]),
        )
    )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("subfield_name", pa.string(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("baseline_start_year", pa.uint16(), nullable=False),
            pa.field("baseline_end_year", pa.uint16(), nullable=False),
            pa.field("recent_start_year", pa.uint16(), nullable=False),
            pa.field("recent_end_year", pa.uint16(), nullable=False),
            pa.field("reserved_response_end_year", pa.uint16(), nullable=False),
            pa.field("baseline_works", pa.uint64(), nullable=False),
            pa.field("recent_works", pa.uint64(), nullable=False),
            pa.field("absolute_gain", pa.uint64(), nullable=False),
            pa.field(
                "minimum_baseline_recent_count",
                pa.uint64(),
                nullable=False,
            ),
            pa.field(
                "standardized_gain_numerator",
                pa.uint64(),
                nullable=False,
            ),
            pa.field(
                "standardized_gain_denominator",
                pa.uint64(),
                nullable=False,
            ),
            pa.field("standardized_gain", pa.float64(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("pilot_episode_selected", pa.bool_(), nullable=False),
            pa.field("pilot_episode_order", pa.uint8(), nullable=True),
        ],
        metadata={b"schema_version": CANDIDATE_SCHEMA.encode()},
    )
    selected.sort(
        key=lambda row: (
            selected_fields.index(int(row["field_id"])),
            int(row["pilot_episode_order"]),
        )
    )
    return pa.Table.from_pylist(rows, schema=schema), selected, {
        "candidate_rows": len(rows),
        "candidate_rows_by_field": {
            str(field_id): len(candidates_by_field[field_id])
            for field_id in selected_fields
        },
        "selected_episode_count": len(selected),
        "minimum_entry_year_separation": minimum_separation,
        "distinct_subfields_within_each_field": all(
            len(
                {
                    int(row["subfield_id"])
                    for row in selected
                    if int(row["field_id"]) == field_id
                }
            )
            == 2
            for field_id in selected_fields
        ),
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
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    os.chmod(path, 0o600)


def _markdown(report: Mapping[str, Any]) -> str:
    fields = report["field_selection"]["selected_pilot_fields"]
    episodes = report["episode_selection"]["selected_pilot_episodes"]
    lines = [
        "# D0-A outcome-blind structural pilot freeze",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Pilot fields",
        "",
    ]
    for item in fields:
        lines.append(
            f"- {item['pilot_selection_order']}. field {item['field_id']} — "
            f"{item['field_name']} ({item['domain_name']})"
        )
    lines.extend(["", "## Provisional attention episodes", ""])
    for item in episodes:
        lines.append(
            f"- field {item['field_id']}, episode {item['pilot_episode_order']}: "
            f"T{item['topic_id']} {item['topic_name']} at entry year "
            f"{item['entry_year']} (subfield {item['subfield_id']}; "
            f"baseline {item['baseline_works']:,}, recent {item['recent_works']:,})"
        )
    lines.extend(
        [
            "",
            "No entry, persistence, realized-citation, predictive, model-selection,",
            "or G4 outcome was read. This is a D0-A structural freeze, not a D0",
            "pass, final G0 field choice, fitted model, or scientific result.",
            "",
        ]
    )
    return "\n".join(lines)


def build(
    config_path: Path,
    *,
    s0_core_root: Path,
    s0_orcid_root: Path,
    s0_graph_root: Path,
    cutoff_root: Path,
    taxonomy_topics: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, "D0-A config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("label") == "pivot_feedback_d0_structural_pilot_freeze_v1",
        "config label mismatch",
    )
    require(
        config.get("mode") == "OUTCOME_BLIND_DISCOVERY_STRUCTURAL_FREEZE",
        "config mode mismatch",
    )
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_S0_COMPLETION",
        "D0-A is not authorized",
    )

    repo_contracts = _mapping(config.get("repository_bindings"), "repository bindings")
    gate_document, gate_binding = _validate_repo_binding(
        _mapping(repo_contracts["gate_state"], "gate binding"),
        yaml_document=True,
    )
    _, s0_binding = _validate_repo_binding(
        _mapping(repo_contracts["s0_completion"], "S0 binding"),
        yaml_document=False,
    )
    universe_document, universe_binding = _validate_repo_binding(
        _mapping(repo_contracts["candidate_universe"], "universe binding"),
        yaml_document=True,
    )
    _, taxonomy_reference_binding = _validate_repo_binding(
        _mapping(repo_contracts["taxonomy_reference"], "taxonomy reference binding"),
        yaml_document=False,
    )
    require(
        _nested(gate_document, "authorization.model_fitting") == "NOT_AUTHORIZED",
        "gate unexpectedly authorizes fitting",
    )
    fields = _candidate_universe(universe_document)

    inputs = _mapping(config.get("normalized_inputs"), "normalized inputs")
    core_path, core_binding = _validate_bound_parquet(
        s0_core_root,
        _mapping(inputs["s0_core"], "S0 core input"),
    )
    orcid_path, orcid_binding = _validate_bound_parquet(
        s0_orcid_root,
        _mapping(inputs["s0_orcid_affiliation"], "S0 ORCID input"),
    )
    graph_path, graph_binding = _validate_bound_parquet(
        s0_graph_root,
        _mapping(inputs["s0_coauthor_graph"], "S0 graph input"),
    )
    work_paths, cutoff_binding, cutoff_stats = _validate_work_metadata(
        cutoff_root,
        _mapping(inputs["cutoff_work_metadata"], "cutoff input"),
        workers=workers,
    )
    taxonomy, taxonomy_binding = _load_taxonomy(
        taxonomy_topics,
        _mapping(inputs["topic_taxonomy_projection"], "topic taxonomy input"),
    )

    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden boundary",
    )
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary opened")
    field_contract = _mapping(config.get("field_selection"), "field selection")
    episode_contract = _mapping(config.get("episode_selection"), "episode selection")
    require(field_contract["entry_year_end_inclusive"] == 2012, "field cutoff differs")
    require(episode_contract["entry_year_end_inclusive"] == 2012, "episode cutoff differs")

    field_table, selected_field_rows, reserve_order = _field_ranking(
        core_path,
        orcid_path,
        graph_path,
        fields=fields,
        contract=field_contract,
    )
    selected_field_ids = [int(row["field_id"]) for row in selected_field_rows]
    topic_year_table, scan_accounting = _scan_topic_years(
        work_paths,
        selected_fields=selected_field_ids,
        taxonomy=taxonomy,
        contract=episode_contract,
    )
    require(
        scan_accounting["work_rows_scanned"]
        == _mapping(inputs["cutoff_work_metadata"], "cutoff input")["rows"],
        "cutoff row closure failed",
    )
    candidate_table, selected_episodes, episode_accounting = _episode_candidates(
        topic_year_table,
        selected_fields=selected_field_ids,
        contract=episode_contract,
    )
    require(len(selected_episodes) == 4, "four-episode freeze failed")
    _validate_unchanged(work_paths, cutoff_stats)

    output = output.resolve()
    input_paths = [
        s0_core_root.resolve(),
        s0_orcid_root.resolve(),
        s0_graph_root.resolve(),
        cutoff_root.resolve(),
        taxonomy_topics.resolve(),
    ]
    for input_path in input_paths:
        if input_path.is_dir():
            require(
                output != input_path and not output.is_relative_to(input_path),
                "output lies inside an input root",
            )
        else:
            require(output != input_path.parent, "output aliases taxonomy parent")
    require(not output.exists(), "output directory already exists")
    require(output.parent.is_dir(), "output parent does not exist")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        artifacts = [
            _write_parquet(
                field_table,
                temporary / "field_structural_ranking.parquet",
            ),
            _write_parquet(
                topic_year_table,
                temporary / "selected_field_topic_year_attention.parquet",
            ),
            _write_parquet(
                candidate_table,
                temporary / "episode_candidates.parquet",
            ),
        ]
        selected_field_report = []
        for row in selected_field_rows:
            selected_field_report.append(
                {
                    "field_id": int(row["field_id"]),
                    "field_name": row["field_name"],
                    "field_slug": row["field_slug"],
                    "domain_id": row["domain_id"],
                    "domain_name": row["domain_name"],
                    "global_rank": int(row["global_rank"]),
                    "pilot_selection_order": int(row["pilot_selection_order"]),
                    "minimum_metric_percentile": row[
                        "minimum_metric_percentile"
                    ],
                    "mean_metric_percentile": row["mean_metric_percentile"],
                }
            )
        selected_episode_report = []
        for row in selected_episodes:
            selected_episode_report.append(
                {
                    key: row[key]
                    for key in (
                        "field_id",
                        "topic_id",
                        "topic_name",
                        "subfield_id",
                        "subfield_name",
                        "entry_year",
                        "baseline_start_year",
                        "baseline_end_year",
                        "recent_start_year",
                        "recent_end_year",
                        "reserved_response_end_year",
                        "baseline_works",
                        "recent_works",
                        "absolute_gain",
                        "standardized_gain_numerator",
                        "standardized_gain_denominator",
                        "standardized_gain",
                        "field_candidate_rank",
                        "pilot_episode_order",
                    )
                }
            )
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "label": "pivot_feedback_d0_structural_pilot_freeze_v1",
            "status": STATUS,
            "scope": {
                "d0a_structural_freeze_complete": True,
                "d0_complete": False,
                "entry_year_end_inclusive": 2012,
                "pilot_fields": 2,
                "pilot_episodes": 4,
                "outcomes_read": False,
                "author_entry_rows_read": False,
                "persistence_rows_read": False,
                "citations_realized_read": False,
                "predictive_scores_computed": False,
                "threshold_tuning_performed": False,
                "model_fitting_performed": False,
                "g4_rows_read": False,
                "gpu_used": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "repository_bindings": {
                "gate_state": gate_binding,
                "s0_completion": s0_binding,
                "candidate_universe": universe_binding,
                "taxonomy_reference": taxonomy_reference_binding,
            },
            "normalized_input_bindings": {
                "s0_core": core_binding,
                "s0_orcid_affiliation": orcid_binding,
                "s0_coauthor_graph": graph_binding,
                "cutoff_work_metadata": cutoff_binding,
                "topic_taxonomy_projection": taxonomy_binding,
            },
            "field_selection": {
                "role": field_contract["role"],
                "metrics": list(FIELD_METRICS),
                "ranking": field_contract["ranking"],
                "selection": field_contract["selection"],
                "require_distinct_domains": True,
                "selected_pilot_fields": selected_field_report,
                "reserve_order_field_ids": reserve_order,
                "artifact": artifacts[0],
            },
            "topic_year_attention": {
                **scan_accounting,
                "artifact": artifacts[1],
            },
            "episode_selection": {
                "role": episode_contract["role"],
                "attention_unit": episode_contract["attention_unit"],
                "family_unit": episode_contract["family_unit"],
                "baseline_offsets": episode_contract["baseline_offsets"],
                "recent_offsets": episode_contract["recent_offsets"],
                "response_offsets_reserved_for_independence_only": episode_contract[
                    "response_offsets_reserved_for_independence_only"
                ],
                "exact_score_order": episode_contract[
                    "standardized_gain_exact_order"
                ],
                "pair_selection": episode_contract["pair_selection"],
                **episode_accounting,
                "selected_pilot_episodes": selected_episode_report,
                "artifact": artifacts[2],
            },
            "authorization": {
                "d0b_outcome_blind_author_panel": "AUTHORIZED",
                "d0_discovery_outcome_opening": "NOT_YET_AUTHORIZED",
                "g0_measurement_freeze": "NOT_AUTHORIZED",
                "final_panel_materialization": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "execution": {
                "workload_class": "CPU_IO",
                "workers": workers,
                "runtime_seconds": round(time.monotonic() - started, 6),
                "config_relative_path": str(
                    config_path.relative_to(REPO_ROOT)
                ),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": str(
                    Path(__file__).resolve().relative_to(REPO_ROOT)
                ),
                "builder_sha256": _sha256(Path(__file__).resolve()),
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "d0_structural_pilot_freeze.json"
        _write_json(report_path, report)
        artifacts.append(
            {
                "relative_path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            }
        )
        markdown_path = temporary / "d0_structural_pilot_freeze.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts.append(
            {
                "relative_path": markdown_path.name,
                "bytes": markdown_path.stat().st_size,
                "sha256": _sha256(markdown_path),
            }
        )
        artifacts.sort(key=lambda item: str(item["relative_path"]))
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes(
                    [
                        (
                            item["relative_path"],
                            item["bytes"],
                            item["sha256"],
                        )
                        for item in artifacts
                    ]
                )
            ).hexdigest(),
            "result_fingerprint_sha256": hashlib.sha256(
                _canonical_bytes(
                    {
                        "status": STATUS,
                        "config_sha256": _sha256(config_path),
                        "selected_field_ids": selected_field_ids,
                        "selected_episodes": [
                            (
                                int(row["field_id"]),
                                int(row["topic_id"]),
                                int(row["entry_year"]),
                            )
                            for row in selected_episodes
                        ],
                        "artifacts": [
                            (
                                item["relative_path"],
                                item["bytes"],
                                item["sha256"],
                            )
                            for item in artifacts
                        ],
                    }
                )
            ).hexdigest(),
            "d0a_structural_freeze_complete": True,
            "d0_complete": False,
            "d0b_outcome_blind_author_panel_authorized": True,
            "g0_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "gpu_used": False,
            "receipt_written_last": True,
        }
        _write_json(
            temporary / "d0_structural_pilot_freeze.outputs.json",
            receipt,
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/configs/pivot_feedback_d0_structural_pilot_freeze_v1.yaml",
    )
    parser.add_argument("--s0-core-root", type=Path, required=True)
    parser.add_argument("--s0-orcid-root", type=Path, required=True)
    parser.add_argument("--s0-graph-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--taxonomy-topics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        args.config,
        s0_core_root=args.s0_core_root,
        s0_orcid_root=args.s0_orcid_root,
        s0_graph_root=args.s0_graph_root,
        cutoff_root=args.cutoff_root,
        taxonomy_topics=args.taxonomy_topics,
        output=args.output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_field_ids": [
                    row["field_id"]
                    for row in report["field_selection"]["selected_pilot_fields"]
                ],
                "selected_episodes": [
                    {
                        "field_id": row["field_id"],
                        "topic_id": row["topic_id"],
                        "entry_year": row["entry_year"],
                    }
                    for row in report["episode_selection"][
                        "selected_pilot_episodes"
                    ]
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
