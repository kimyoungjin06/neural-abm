#!/usr/bin/env python3
"""Build the outcome-blind F-phase field and shock-family assignment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_f_field_episode_assignment_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_f_field_episode_assignment.v1"
PARTITIONS = 4096
PERIODS = (
    ("g1_empirical_replication", 2013, 2016),
    ("model_selection", 2017, 2019),
    ("g4_temporal_holdout", 2020, 2022),
)


def _load_d0_helper() -> Any:
    path = ROOT / "scripts/build_pivot_feedback_d0_certified_author_opportunity_rebuild_v1.py"
    spec = importlib.util.spec_from_file_location("pivot_feedback_d0_author_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load D0 author-opportunity helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d0 = _load_d0_helper()

AUTHOR_FIELD_YEAR_DTYPE = np.dtype(
    [("author_id", "<u8"), ("field_id", "u1"), ("year", "<u2")]
)
AUTHOR_FIELD_DTYPE = np.dtype([("author_id", "<u8"), ("field_id", "u1")])

FIELD_SCHEMA = pa.schema(
    [
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("field_name", pa.string(), nullable=False),
        pa.field("domain_id", pa.string(), nullable=False),
        pa.field("domain_name", pa.string(), nullable=False),
        pa.field("eligible_author_count", pa.uint64(), nullable=False),
        pa.field("matured_entry_opportunity_count", pa.uint64(), nullable=False),
        pa.field("pre_cutoff_work_count", pa.uint64(), nullable=False),
        pa.field("publication_date_completeness_numerator", pa.uint64(), nullable=False),
        pa.field("publication_date_completeness_denominator", pa.uint64(), nullable=False),
        pa.field("publication_date_completeness", pa.float64(), nullable=False),
        pa.field("fixed_horizon_outcome_coverage_numerator", pa.uint64(), nullable=False),
        pa.field("fixed_horizon_outcome_coverage_denominator", pa.uint64(), nullable=False),
        pa.field("fixed_horizon_outcome_coverage", pa.float64(), nullable=False),
        pa.field("pre_entry_t_minus_1_edge_coverage_numerator", pa.uint64(), nullable=False),
        pa.field("pre_entry_t_minus_1_edge_coverage_denominator", pa.uint64(), nullable=False),
        pa.field("pre_entry_t_minus_1_edge_coverage", pa.float64(), nullable=False),
        pa.field("orcid_linked_author_rate_numerator", pa.uint64(), nullable=False),
        pa.field("orcid_linked_author_rate_denominator", pa.uint64(), nullable=False),
        pa.field("orcid_linked_author_rate", pa.float64(), nullable=False),
        pa.field("independent_attention_shock_count", pa.uint32(), nullable=False),
        pa.field("minimum_structural_coverage_numerator", pa.uint8(), nullable=False),
        pa.field("minimum_structural_coverage_denominator", pa.uint8(), nullable=False),
        pa.field("minimum_structural_coverage", pa.float64(), nullable=False),
        pa.field("domain_field_rank", pa.uint8(), nullable=False),
        pa.field("domain_primary", pa.bool_(), nullable=False),
        pa.field("domain_reserve", pa.bool_(), nullable=False),
        pa.field("selected_slot_id", pa.string(), nullable=True),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_field_structural_coverage.v1"},
)

CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("topic_name", pa.string(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
        pa.field("subfield_name", pa.string(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("temporal_partition_id", pa.string(), nullable=False),
        pa.field("reference_count", pa.uint64(), nullable=False),
        pa.field("recent_count", pa.uint64(), nullable=False),
        pa.field("acceleration", pa.float64(), nullable=False),
        pa.field("field_year_median", pa.float64(), nullable=False),
        pa.field("field_year_mad", pa.float64(), nullable=False),
        pa.field("modified_z", pa.float64(), nullable=False),
        pa.field("selected_field", pa.bool_(), nullable=False),
        pa.field("assigned_slot_id", pa.string(), nullable=True),
        pa.field("retained_for_assigned_panel", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_attention_shock_candidate.v1"},
)

SLOT_SCHEMA = pa.schema(
    [
        pa.field("slot_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("temporal_partition_id", pa.string(), nullable=False),
        pa.field("selected_domain_id", pa.string(), nullable=False),
        pa.field("nominal_primary_field_id", pa.uint8(), nullable=False),
        pa.field("selected_field_id", pa.uint8(), nullable=False),
        pa.field("reserve_field_ids", pa.list_(pa.uint8(), 2), nullable=False),
        pa.field("reserve_activated", pa.bool_(), nullable=False),
        pa.field("reserve_activation_reason", pa.string(), nullable=True),
        pa.field("retained_attention_candidate_count", pa.uint32(), nullable=False),
        pa.field("independent_subfield_family_count", pa.uint16(), nullable=False),
        pa.field("sealed", pa.bool_(), nullable=False),
        pa.field("author_cold", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"pivot_feedback_f_slot_assignment.v1"},
)


class AssignmentBuildError(RuntimeError):
    """Raised when an input or assignment violates the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssignmentBuildError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _validate_repo_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    containers = {
        "authorization_parent": config["authorization_parent"],
        **_mapping(config["bound_design_sources"], "bound design sources"),
    }
    observed: dict[str, Any] = {}
    for name, raw in containers.items():
        binding = _mapping(raw, name)
        relative = str(binding["relative_path"])
        path = (ROOT / relative).resolve(strict=True)
        require(path.is_relative_to(ROOT), f"{name} path escape")
        require(_sha256(path) == binding["sha256"], f"{name} SHA mismatch")
        if path.suffix == ".json":
            document = _read_json(path)
        else:
            document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name)
        require(
            document.get("schema_version") == binding["required_schema_version"],
            f"{name} schema mismatch",
        )
        required_status = binding.get("required_status")
        if required_status is not None:
            require(document.get("status") == required_status, f"{name} status mismatch")
        require(binding.get("mutation_allowed") is False, f"{name} mutable binding")
        observed[name] = {"relative_path": relative, "sha256": binding["sha256"]}
    return observed


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["relative_path"]): _mapping(item, "receipt artifact")
        for item in receipt.get("artifacts", [])
    }


def _validate_compact_input(root: Path, binding: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid compact input root")
    require(root.name == binding["expected_root_basename"], "compact root basename")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "compact receipt SHA")
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "compact receipt schema")
    require(receipt.get("status") == binding["required_status"], "compact receipt status")
    relative = str(binding["artifact_relative_path"])
    artifact_path = (root / relative).resolve(strict=True)
    require(artifact_path.is_relative_to(root), "compact artifact path escape")
    require(_sha256(artifact_path) == binding["artifact_sha256"], "compact artifact SHA")
    artifact = _artifact_map(receipt).get(relative)
    require(artifact is not None, "compact artifact absent from receipt")
    require(artifact.get("sha256") == binding["artifact_sha256"], "receipt artifact SHA")
    parquet = pq.ParquetFile(artifact_path)
    require(parquet.metadata.num_rows == binding["artifact_rows"], "compact artifact rows")
    metadata = parquet.schema_arrow.metadata or {}
    require(
        metadata.get(b"schema_version", b"").decode() == binding["artifact_schema_version"],
        "compact artifact schema",
    )
    return artifact_path, {
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_relative_path": relative,
        "artifact_sha256": binding["artifact_sha256"],
        "artifact_rows": binding["artifact_rows"],
    }


def _numeric_suffix(value: str, marker: str) -> int:
    require(marker in value, f"taxonomy identifier missing {marker}")
    suffix = value.rsplit(marker, 1)[1]
    require(suffix.isdigit(), "taxonomy numeric suffix")
    return int(suffix)


def _load_taxonomy(path: Path, binding: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    path = path.resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), "invalid taxonomy input")
    require(path.name == binding["expected_basename"], "taxonomy basename")
    require(_sha256(path) == binding["sha256"], "taxonomy SHA mismatch")
    table = pq.read_table(path, columns=list(binding["required_columns"]))
    require(len(table) == binding["rows"], "taxonomy row count")
    topics: dict[int, dict[str, Any]] = {}
    for row in table.to_pylist():
        topic_id = _numeric_suffix(str(row["id"]), "/T")
        field_id = _numeric_suffix(str(row["field__id"]), "/fields/")
        subfield_id = _numeric_suffix(str(row["subfield__id"]), "/subfields/")
        require(topic_id not in topics, "duplicate taxonomy topic")
        topics[topic_id] = {
            "topic_name": str(row["display_name"]),
            "field_id": field_id,
            "field_name": str(row["field__display_name"]),
            "subfield_id": subfield_id,
            "subfield_name": str(row["subfield__display_name"]),
        }
    require(set(item["field_id"] for item in topics.values()) == set(range(11, 37)), "taxonomy field closure")
    return topics, {"basename": path.name, "sha256": binding["sha256"], "rows": len(table)}


def _period_for_year(year: int) -> str:
    for name, start, end in PERIODS:
        if start <= year <= end:
            return name
    raise AssignmentBuildError(f"decision year outside registered partitions: {year}")


def _topic_year_counts(linked: np.ndarray, *, minimum_year: int) -> dict[tuple[int, int], dict[int, int]]:
    keep = linked["publication_year"] >= minimum_year
    fields = linked["field_id"][keep].astype(np.uint64, copy=False)
    topics = linked["topic_id"][keep].astype(np.uint64, copy=False)
    years = linked["publication_year"][keep].astype(np.uint64, copy=False)
    keys = (fields << np.uint64(56)) | (topics << np.uint64(16)) | years
    unique, counts = np.unique(keys, return_counts=True)
    result: dict[tuple[int, int], dict[int, int]] = {}
    for key, count in zip(unique, counts, strict=True):
        numeric = int(key)
        field_id = numeric >> 56
        topic_id = (numeric >> 16) & 0xFFFFFFFF
        year = numeric & 0xFFFF
        result.setdefault((field_id, topic_id), {})[year] = int(count)
    return result


def _attention_candidates(
    topic_year: Mapping[tuple[int, int], Mapping[int, int]],
    taxonomy: Mapping[int, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    multiplier = float(contract["modified_z_multiplier"])
    threshold = float(contract["modified_z_threshold_inclusive"])
    reference_offsets = [int(value) for value in contract["reference_offsets"]]
    recent_offsets = [int(value) for value in contract["recent_offsets"]]
    candidates: list[dict[str, Any]] = []
    topics_by_field: dict[int, list[int]] = {field_id: [] for field_id in range(11, 37)}
    for field_id, topic_id in topic_year:
        require(topic_id in taxonomy, f"topic {topic_id} absent from taxonomy")
        require(taxonomy[topic_id]["field_id"] == field_id, "topic taxonomy field mismatch")
        topics_by_field[field_id].append(topic_id)
    for field_id in range(11, 37):
        for decision_year in range(2013, 2023):
            rows: list[tuple[int, int, int, float]] = []
            for topic_id in topics_by_field[field_id]:
                annual = topic_year[(field_id, topic_id)]
                reference = sum(annual.get(decision_year + offset, 0) for offset in reference_offsets)
                if reference <= 0:
                    continue
                recent = sum(annual.get(decision_year + offset, 0) for offset in recent_offsets)
                acceleration = math.log1p(recent / 3.0) - math.log1p(reference / 3.0)
                rows.append((topic_id, reference, recent, acceleration))
            if not rows:
                continue
            values = np.asarray([row[3] for row in rows], dtype=np.float64)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if not math.isfinite(mad) or mad <= 0.0:
                continue
            for topic_id, reference, recent, acceleration in rows:
                modified_z = multiplier * (acceleration - median) / mad
                if recent <= reference or acceleration <= 0.0 or modified_z < threshold:
                    continue
                topic = taxonomy[topic_id]
                candidates.append(
                    {
                        "field_id": field_id,
                        "topic_id": topic_id,
                        "topic_name": topic["topic_name"],
                        "subfield_id": topic["subfield_id"],
                        "subfield_name": topic["subfield_name"],
                        "decision_year": decision_year,
                        "temporal_partition_id": _period_for_year(decision_year),
                        "reference_count": reference,
                        "recent_count": recent,
                        "acceleration": acceleration,
                        "field_year_median": median,
                        "field_year_mad": mad,
                        "modified_z": modified_z,
                    }
                )
    candidates.sort(
        key=lambda row: (
            int(row["field_id"]),
            int(row["decision_year"]),
            int(row["subfield_id"]),
            int(row["topic_id"]),
        )
    )
    return candidates


def _scan_author_activity(
    pair: tuple[int, Path],
    *,
    linked: np.ndarray,
    linked_work_ids: np.ndarray,
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path)
    authors = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(authors % PARTITIONS == index)), f"author partition mismatch: {index}")
    works = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    positions = np.searchsorted(linked_work_ids, works)
    in_range = positions < len(linked)
    matched = np.zeros(len(works), dtype=bool)
    matched[in_range] = linked_work_ids[positions[in_range]] == works[in_range]
    source = np.flatnonzero(matched)
    if not len(source):
        return {"source_rows": len(table), "eligible_rows": 0, "periods": {name: {} for name, _, _ in PERIODS}}
    linked_rows = positions[matched]
    years_all = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)
    fields_all = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)
    dispositions_all = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(years_all[source] == linked["publication_year"][linked_rows])), "author/work year mismatch")
    require(bool(np.all(fields_all[source] == linked["field_id"][linked_rows])), "author/work field mismatch")
    require(bool(np.all(dispositions_all[source] == linked["temporal_disposition"][linked_rows])), "author/work disposition mismatch")
    years = years_all[source]
    required = (years >= 2010) & (years <= 2021)
    triples = np.empty(int(required.sum()), dtype=AUTHOR_FIELD_YEAR_DTYPE)
    triples["author_id"] = authors[source][required]
    triples["field_id"] = fields_all[source][required]
    triples["year"] = years[required]
    triples = np.unique(triples)
    pairs = np.empty(len(triples), dtype=AUTHOR_FIELD_DTYPE)
    pairs["author_id"] = triples["author_id"]
    pairs["field_id"] = triples["field_id"]
    unique_pairs, inverse = np.unique(pairs, return_inverse=True)
    bits = np.zeros(len(unique_pairs), dtype=np.uint16)
    offsets = triples["year"].astype(np.int32) - 2010
    np.bitwise_or.at(bits, inverse, np.left_shift(np.uint16(1), offsets.astype(np.uint16)))
    period_counts: dict[str, Any] = {}
    for name, start, end in PERIODS:
        any_eligible = np.zeros(len(unique_pairs), dtype=bool)
        opportunities = np.zeros(26, dtype=np.uint64)
        by_year: dict[str, dict[str, int]] = {}
        for decision_year in range(start, end + 1):
            mask_value = 0
            for activity_year in range(decision_year - 3, decision_year):
                mask_value |= 1 << (activity_year - 2010)
            eligible = (bits & np.uint16(mask_value)) != 0
            any_eligible |= eligible
            counts = np.bincount(
                unique_pairs["field_id"][eligible].astype(np.int64) - 11,
                minlength=26,
            ).astype(np.uint64)
            opportunities += counts
            by_year[str(decision_year)] = {
                str(field_id): int(counts[field_id - 11]) for field_id in range(11, 37)
            }
        author_counts = np.bincount(
            unique_pairs["field_id"][any_eligible].astype(np.int64) - 11,
            minlength=26,
        )
        period_counts[name] = {
            "eligible_author_count": author_counts.astype(np.uint64),
            "matured_entry_opportunity_count": opportunities,
            "by_year": by_year,
        }
    return {"source_rows": len(table), "eligible_rows": int(matched.sum()), "periods": period_counts}


def _author_metrics(paths: Sequence[Path], linked: np.ndarray, workers: int) -> tuple[dict[str, Any], dict[str, int]]:
    linked_work_ids = np.ascontiguousarray(linked["work_numeric_id"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(
            executor.map(
                lambda pair: _scan_author_activity(
                    pair,
                    linked=linked,
                    linked_work_ids=linked_work_ids,
                ),
                enumerate(paths),
            )
        )
    totals: dict[str, Any] = {}
    for name, start, end in PERIODS:
        author = np.zeros(26, dtype=np.uint64)
        opportunity = np.zeros(26, dtype=np.uint64)
        by_year = {str(year): np.zeros(26, dtype=np.uint64) for year in range(start, end + 1)}
        for chunk in chunks:
            period = chunk["periods"][name]
            if not period:
                continue
            author += period["eligible_author_count"]
            opportunity += period["matured_entry_opportunity_count"]
            for year in by_year:
                by_year[year] += np.asarray(
                    [period["by_year"][year][str(field_id)] for field_id in range(11, 37)],
                    dtype=np.uint64,
                )
        totals[name] = {"eligible_author_count": author, "matured_entry_opportunity_count": opportunity, "by_year": by_year}
    return totals, {
        "source_rows_scanned": sum(int(chunk["source_rows"]) for chunk in chunks),
        "eligible_author_work_event_rows": sum(int(chunk["eligible_rows"]) for chunk in chunks),
    }


def _ratio(numerator: int, denominator: int) -> Fraction:
    return Fraction(numerator, denominator) if denominator else Fraction(0, 1)


def _compact_metrics(
    *,
    core_path: Path,
    graph_path: Path,
    orcid_path: Path,
) -> dict[int, dict[str, Any]]:
    core_rows = pq.read_table(core_path).to_pylist()
    graph_rows = pq.read_table(graph_path).to_pylist()
    orcid_rows = pq.read_table(orcid_path).to_pylist()
    core_year: dict[tuple[int, int], Mapping[str, Any]] = {}
    publication = {field_id: [0, 0] for field_id in range(11, 37)}
    for row in core_rows:
        field_id = int(row["field_id"])
        publication[field_id][0] += int(row["usable_date_works"])
        publication[field_id][1] += int(row["total_canonical_works"])
        if row["publication_year"] is not None:
            core_year[(field_id, int(row["publication_year"]))] = row
    graph_year = {
        (int(row["field_id"]), int(row["publication_year"])): row
        for row in graph_rows
        if row["publication_year"] is not None and int(row["declared_team_cap"]) == 100
    }
    orcid_year = {
        (int(row["field_id"]), int(row["publication_year"])): row
        for row in orcid_rows
        if row["publication_year"] is not None
    }
    result: dict[int, dict[str, Any]] = {}
    for field_id in range(11, 37):
        edge_periods: list[tuple[int, int]] = []
        orcid_periods: list[tuple[int, int]] = []
        for _, start, end in PERIODS:
            years = range(start - 1, end)
            denominator = sum(int(core_year.get((field_id, year), {}).get("distinct_author_tokens", 0)) for year in years)
            edge_numerator = sum(int(graph_year.get((field_id, year), {}).get("graph_node_tokens", 0)) for year in years)
            orcid_numerator = sum(int(orcid_year.get((field_id, year), {}).get("stable_orcid_author_tokens", 0)) for year in years)
            edge_periods.append((edge_numerator, denominator))
            orcid_periods.append((orcid_numerator, denominator))
        edge = min(edge_periods, key=lambda pair: _ratio(*pair))
        orcid = min(orcid_periods, key=lambda pair: _ratio(*pair))
        result[field_id] = {
            "publication": tuple(publication[field_id]),
            "edge": edge,
            "orcid": orcid,
        }
    return result


def _cross_field_metadata(config: Mapping[str, Any]) -> tuple[dict[int, dict[str, str]], dict[str, int]]:
    binding = config["bound_design_sources"]["cross_field_matrix"]
    path = ROOT / str(binding["relative_path"])
    matrix = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "cross-field matrix")
    field_meta: dict[int, dict[str, str]] = {}
    domain_codes: dict[str, int] = {}
    for domain in matrix["candidate_universe"]["domains"]:
        domain_id = str(domain["domain_id"])
        domain_codes[domain_id] = int(domain["rng_code"])
        for field in domain["fields"]:
            field_meta[int(field["field_id"])] = {
                "field_name": str(field["display_name"]),
                "domain_id": domain_id,
                "domain_name": str(domain["display_name"]),
            }
    require(set(field_meta) == set(range(11, 37)), "cross-field candidate closure")
    return field_meta, domain_codes


def _rank_and_assign(
    *,
    linked: np.ndarray,
    author: Mapping[str, Any],
    compact: Mapping[int, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    field_meta: Mapping[int, Mapping[str, str]],
    domain_codes: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    shock_counts: dict[tuple[int, str], int] = {}
    for field_id in range(11, 37):
        for name, _, _ in PERIODS:
            shock_counts[(field_id, name)] = len(
                {
                    int(row["subfield_id"])
                    for row in candidates
                    if int(row["field_id"]) == field_id and row["temporal_partition_id"] == name
                }
            )
    raw: dict[int, dict[str, Any]] = {}
    pre_cutoff_mask = linked["publication_year"] <= 2012
    pre_cutoff_counts = np.bincount(
        linked["field_id"][pre_cutoff_mask].astype(np.int64) - 11,
        minlength=26,
    )
    for field_id in range(11, 37):
        eligible = min(int(author[name]["eligible_author_count"][field_id - 11]) for name, _, _ in PERIODS)
        opportunities = min(int(author[name]["matured_entry_opportunity_count"][field_id - 11]) for name, _, _ in PERIODS)
        fixed_periods = [
            (int(author[name]["matured_entry_opportunity_count"][field_id - 11]), int(author[name]["matured_entry_opportunity_count"][field_id - 11]))
            for name, _, _ in PERIODS
        ]
        fixed = min(fixed_periods, key=lambda pair: _ratio(*pair))
        pre_cutoff = int(pre_cutoff_counts[field_id - 11])
        publication = compact[field_id]["publication"]
        edge = compact[field_id]["edge"]
        orcid = compact[field_id]["orcid"]
        shock = min(shock_counts[(field_id, name)] for name, _, _ in PERIODS)
        metrics = {
            "eligible_author_count": Fraction(eligible, 1),
            "matured_entry_opportunity_count": Fraction(opportunities, 1),
            "pre_cutoff_work_count": Fraction(pre_cutoff, 1),
            "publication_date_completeness": _ratio(*publication),
            "fixed_horizon_outcome_coverage": _ratio(*fixed),
            "pre_entry_t_minus_1_edge_coverage": _ratio(*edge),
            "orcid_linked_author_rate": _ratio(*orcid),
            "independent_attention_shock_count": Fraction(shock, 1),
        }
        raw[field_id] = {
            **field_meta[field_id],
            "eligible": eligible,
            "opportunities": opportunities,
            "pre_cutoff": pre_cutoff,
            "publication": publication,
            "fixed": fixed,
            "edge": edge,
            "orcid": orcid,
            "shock": shock,
            "metrics": metrics,
        }
    for field_id, item in raw.items():
        percentiles = [
            Fraction(sum(other["metrics"][name] <= value for other in raw.values()), len(raw))
            for name, value in item["metrics"].items()
        ]
        item["minimum"] = min(percentiles)
        item["rank_key"] = (
            item["minimum"],
            Fraction(item["eligible"], 1),
            _ratio(*item["fixed"]),
            -field_id,
        )
    by_domain: dict[str, list[int]] = {}
    for field_id, item in raw.items():
        by_domain.setdefault(item["domain_id"], []).append(field_id)
    for fields in by_domain.values():
        fields.sort(key=lambda field_id: raw[field_id]["rank_key"], reverse=True)
    domain_order = sorted(
        by_domain,
        key=lambda domain_id: (raw[by_domain[domain_id][0]]["rank_key"], -domain_codes[domain_id]),
        reverse=True,
    )
    selected: dict[int, str] = {}
    slots: list[dict[str, Any]] = []
    for index, domain_id in enumerate(domain_order):
        nominal_primary = by_domain[domain_id][0]
        reserves = by_domain[domain_id][1:3]
        require(len(reserves) == 2, f"domain {domain_id} lacks two frozen reserves")
        if index < 3:
            slot_id = f"g1_domain_slot_{index + 1:02d}"
            role = "g1_replication"
            period = "g1_empirical_replication"
            sealed = False
            author_cold = False
        else:
            slot_id = "g4_domain_author_cold_slot"
            role = "g4_domain_author_cold"
            period = "g4_temporal_holdout"
            sealed = True
            author_cold = True
        selected_field: int | None = None
        period_candidates: list[Mapping[str, Any]] = []
        families: set[int] = set()
        for candidate_field in (nominal_primary, *reserves):
            candidate_rows = [
                row
                for row in candidates
                if int(row["field_id"]) == candidate_field
                and row["temporal_partition_id"] == period
            ]
            candidate_families = {
                int(row["subfield_id"]) for row in candidate_rows
            }
            if len(candidate_families) >= 3:
                selected_field = candidate_field
                period_candidates = candidate_rows
                families = candidate_families
                break
        require(
            selected_field is not None,
            f"domain {domain_id} primary and two reserves have fewer than three shock families",
        )
        reserve_activated = selected_field != nominal_primary
        selected[selected_field] = slot_id
        slots.append(
            {
                "slot_id": slot_id,
                "role": role,
                "temporal_partition_id": period,
                "selected_domain_id": domain_id,
                "nominal_primary_field_id": nominal_primary,
                "selected_field_id": selected_field,
                "reserve_field_ids": reserves,
                "reserve_activated": reserve_activated,
                "reserve_activation_reason": (
                    "NOMINAL_PRIMARY_FEWER_THAN_THREE_INDEPENDENT_SUBFIELD_FAMILIES_IN_ASSIGNED_PERIOD"
                    if reserve_activated
                    else None
                ),
                "retained_attention_candidate_count": len(period_candidates),
                "independent_subfield_family_count": len(families),
                "sealed": sealed,
                "author_cold": author_cold,
            }
        )
    field_rows: list[dict[str, Any]] = []
    domain_ranks = {field_id: rank for fields in by_domain.values() for rank, field_id in enumerate(fields, 1)}
    for field_id in range(11, 37):
        item = raw[field_id]
        minimum: Fraction = item["minimum"]
        publication = item["publication"]
        fixed = item["fixed"]
        edge = item["edge"]
        orcid = item["orcid"]
        field_rows.append(
            {
                "field_id": field_id,
                "field_name": item["field_name"],
                "domain_id": item["domain_id"],
                "domain_name": item["domain_name"],
                "eligible_author_count": item["eligible"],
                "matured_entry_opportunity_count": item["opportunities"],
                "pre_cutoff_work_count": item["pre_cutoff"],
                "publication_date_completeness_numerator": publication[0],
                "publication_date_completeness_denominator": publication[1],
                "publication_date_completeness": float(_ratio(*publication)),
                "fixed_horizon_outcome_coverage_numerator": fixed[0],
                "fixed_horizon_outcome_coverage_denominator": fixed[1],
                "fixed_horizon_outcome_coverage": float(_ratio(*fixed)),
                "pre_entry_t_minus_1_edge_coverage_numerator": edge[0],
                "pre_entry_t_minus_1_edge_coverage_denominator": edge[1],
                "pre_entry_t_minus_1_edge_coverage": float(_ratio(*edge)),
                "orcid_linked_author_rate_numerator": orcid[0],
                "orcid_linked_author_rate_denominator": orcid[1],
                "orcid_linked_author_rate": float(_ratio(*orcid)),
                "independent_attention_shock_count": item["shock"],
                "minimum_structural_coverage_numerator": minimum.numerator,
                "minimum_structural_coverage_denominator": minimum.denominator,
                "minimum_structural_coverage": float(minimum),
                "domain_field_rank": domain_ranks[field_id],
                "domain_primary": domain_ranks[field_id] == 1,
                "domain_reserve": domain_ranks[field_id] in (2, 3),
                "selected_slot_id": selected.get(field_id),
            }
        )
    return field_rows, slots, selected


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", compression_level=9, use_dictionary=True, write_statistics=True, version="2.6")
    os.chmod(path, 0o600)
    return {"relative_path": path.name, "bytes": path.stat().st_size, "rows": len(table), "sha256": _sha256(path)}


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot-feedback F field and episode assignment v1",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Slot | Domain | Field | Reserve | Candidates | Families |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["slot_assignment"]:
        lines.append(
            f"| {row['slot_id']} | {row['selected_domain_id']} | {row['selected_field_id']} | "
            f"{','.join(str(value) for value in row['reserve_field_ids'])} | "
            f"{row['retained_attention_candidate_count']} | "
            f"{row['independent_subfield_family_count']} |"
        )
    lines.extend(
        [
            "",
            "This is an outcome-blind assignment component, not final-panel",
            "conformance and not an F pass. Entry and persistence labels were not",
            "computed; G4 remains unopened.",
            "",
        ]
    )
    return "\n".join(lines)


def _cleanup(temporary: Path | None, output: Path) -> None:
    if temporary is None:
        return
    require(temporary.is_absolute(), "temporary path must be absolute")
    require(temporary.parent == output.parent, "temporary parent mismatch")
    require(temporary.name.startswith(f".{output.name}."), "temporary prefix mismatch")
    if temporary.exists():
        shutil.rmtree(temporary)


def build(
    *,
    config_path: Path,
    cutoff_root: Path,
    author_history_root: Path,
    taxonomy_topics: Path,
    s0_core_root: Path,
    s0_graph_root: Path,
    s0_orcid_root: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    config_path = config_path.resolve(strict=True)
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(config.get("status") == "FROZEN_FIELD_EPISODE_ASSIGNMENT_PENDING_PRODUCTION_EXECUTION_NO_F_PASS", "config status")
    repository_bindings = _validate_repo_bindings(config)
    projection = _mapping(config["operator_projection"], "operator projection")
    require(projection["field_ids"] == list(range(11, 37)), "field projection")
    require(projection["latest_required_predecision_year"] == 2021, "latest predecision year")
    inputs = _mapping(config["normalized_inputs"], "normalized inputs")
    work_paths, work_binding, work_stats = d0._validate_partitioned_input(
        cutoff_root,
        _mapping(inputs["cutoff_work_metadata"], "cutoff binding"),
        schema=d0.WORK_INPUT_SCHEMA,
        workers=workers,
    )
    event_paths, event_binding, event_stats = d0._validate_partitioned_input(
        author_history_root,
        _mapping(inputs["author_history_events"], "author binding"),
        schema=d0.EVENT_INPUT_SCHEMA,
        workers=workers,
    )
    taxonomy, taxonomy_binding = _load_taxonomy(
        taxonomy_topics,
        _mapping(inputs["topic_taxonomy_projection"], "taxonomy binding"),
    )
    core_path, core_binding = _validate_compact_input(s0_core_root, _mapping(inputs["s0_core"], "S0 core"))
    graph_path, graph_binding = _validate_compact_input(s0_graph_root, _mapping(inputs["s0_graph"], "S0 graph"))
    orcid_path, orcid_binding = _validate_compact_input(s0_orcid_root, _mapping(inputs["s0_orcid"], "S0 ORCID"))

    fields = [int(value) for value in projection["field_ids"]]
    dispositions = [int(value) for value in projection["allowed_temporal_dispositions"]]
    included_types = [int(value) for value in projection["work_type_codes"]]
    latest_year = int(projection["latest_required_predecision_year"])
    valid_work_ids, event_metadata, event_accounting = d0._event_work_index(
        event_paths,
        fields=fields,
        dispositions=dispositions,
        latest_year=latest_year,
        workers=workers,
    )
    require(event_accounting["event_rows_scanned"] == inputs["author_history_events"]["rows"], "author event row closure")
    require(event_accounting["duplicate_author_work_event_rows"] == 0, "duplicate author-work pairs")
    linked, work_accounting = d0._linked_work_universe(
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
    require(work_accounting["rows"] == inputs["cutoff_work_metadata"]["rows"], "work row closure")
    topic_year = _topic_year_counts(linked, minimum_year=int(projection["earliest_required_predecision_year"]))
    candidates = _attention_candidates(
        topic_year,
        taxonomy,
        _mapping(config["structural_metric_definitions"]["independent_attention_shock_count"], "shock contract"),
    )
    author_metrics, author_accounting = _author_metrics(event_paths, linked, workers)
    require(author_accounting["source_rows_scanned"] == inputs["author_history_events"]["rows"], "author activity row closure")
    compact = _compact_metrics(core_path=core_path, graph_path=graph_path, orcid_path=orcid_path)
    field_meta, domain_codes = _cross_field_metadata(config)
    field_rows, slots, selected = _rank_and_assign(
        linked=linked,
        author=author_metrics,
        compact=compact,
        candidates=candidates,
        field_meta=field_meta,
        domain_codes=domain_codes,
    )
    slot_by_field = {int(row["selected_field_id"]): row for row in slots}
    candidate_rows: list[dict[str, Any]] = []
    for row in candidates:
        field_id = int(row["field_id"])
        slot = slot_by_field.get(field_id)
        retained = slot is not None and row["temporal_partition_id"] == slot["temporal_partition_id"]
        candidate_rows.append(
            {
                **row,
                "selected_field": slot is not None,
                "assigned_slot_id": None if slot is None else slot["slot_id"],
                "retained_for_assigned_panel": retained,
            }
        )
    d0._validate_unchanged(work_paths, work_stats)
    d0._validate_unchanged(event_paths, event_stats)

    output = output.resolve()
    require(output.name == config["output_contract"]["root_basename"], "output basename")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent missing")
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    try:
        artifacts = [
            _write_parquet(pa.Table.from_pylist(field_rows, schema=FIELD_SCHEMA), temporary / "field_structural_coverage.parquet"),
            _write_parquet(pa.Table.from_pylist(candidate_rows, schema=CANDIDATE_SCHEMA), temporary / "attention_shock_candidates.parquet"),
            _write_parquet(pa.Table.from_pylist(slots, schema=SLOT_SCHEMA), temporary / "slot_assignment.parquet"),
        ]
        report: dict[str, Any] = {
            "schema_version": config["output_contract"]["report_schema_version"],
            "status": config["output_contract"]["passing_status"],
            "repository_bindings": repository_bindings,
            "input_bindings": {
                "cutoff_work_metadata": work_binding,
                "author_history_events": event_binding,
                "topic_taxonomy_projection": taxonomy_binding,
                "s0_core": core_binding,
                "s0_graph": graph_binding,
                "s0_orcid": orcid_binding,
            },
            "scope": {
                "candidate_field_count": 26,
                "candidate_domain_count": 4,
                "g1_selected_field_count": 3,
                "g4_selected_field_count": 1,
                "entry_or_persistence_labels_computed": False,
                "realized_citation_counts_read": False,
                "effect_metrics_read": False,
                "model_fitted": False,
                "g4_open": False,
                "final_panel_conformance_complete": False,
                "pass_f": False,
                "pass_g0": False,
            },
            "source_accounting": {
                "author_event_index": event_accounting,
                "eligible_work": work_accounting,
                "author_activity": author_accounting,
                "topic_year_cells": sum(len(years) for years in topic_year.values()),
                "attention_shock_candidates": len(candidates),
            },
            "field_structural_coverage": field_rows,
            "slot_assignment": slots,
            "selected_field_to_slot": {str(field_id): slot for field_id, slot in sorted(selected.items())},
            "authorization": {
                "final_panel_leaveout_and_support_replay": "AUTHORIZED_NEXT",
                "operator_only_holdout_materialization": "NOT_AUTHORIZED_BEFORE_FINAL_PANEL_CONFORMANCE",
                "confirmatory_analytic_read": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
            },
            "execution": {
                "workers": workers,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "config_relative_path": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "builder_sha256": _sha256(Path(__file__).resolve()),
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "field_episode_assignment.json"
        _write_json(report_path, report)
        markdown_path = temporary / "FIELD_EPISODE_ASSIGNMENT.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts.extend(
            [
                {"relative_path": report_path.name, "bytes": report_path.stat().st_size, "sha256": _sha256(report_path)},
                {"relative_path": markdown_path.name, "bytes": markdown_path.stat().st_size, "sha256": _sha256(markdown_path)},
            ]
        )
        artifacts.sort(key=lambda item: str(item["relative_path"]))
        receipt = {
            "schema_version": config["output_contract"]["receipt_schema_version"],
            "status": report["status"],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes([(item["relative_path"], item["bytes"], item["sha256"]) for item in artifacts])
            ).hexdigest(),
            "selected_field_ids": [int(row["selected_field_id"]) for row in slots],
            "selected_domain_ids": [str(row["selected_domain_id"]) for row in slots],
            "entry_or_persistence_labels_computed": False,
            "final_panel_conformance_complete": False,
            "pass_f": False,
            "pass_g0": False,
            "g4_open": False,
            "receipt_written_last": True,
        }
        _write_json(temporary / "field_episode_assignment.outputs.json", receipt)
        os.replace(temporary, output)
        temporary = None
        return report
    finally:
        _cleanup(temporary, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--taxonomy-topics", type=Path, required=True)
    parser.add_argument("--s0-core-root", type=Path, required=True)
    parser.add_argument("--s0-graph-root", type=Path, required=True)
    parser.add_argument("--s0-orcid-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        config_path=args.config,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        taxonomy_topics=args.taxonomy_topics,
        s0_core_root=args.s0_core_root,
        s0_graph_root=args.s0_graph_root,
        s0_orcid_root=args.s0_orcid_root,
        output=args.output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "slot_assignment": report["slot_assignment"],
                "source_accounting": report["source_accounting"],
                "execution": report["execution"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
