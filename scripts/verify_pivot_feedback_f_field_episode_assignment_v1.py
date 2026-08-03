#!/usr/bin/env python3
"""Independently verify the outcome-blind F field/episode assignment."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_f_field_episode_assignment_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_f_field_episode_assignment.v1"
REPORT_SCHEMA = "pivot_feedback_f_field_episode_assignment_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_f_field_episode_assignment_outputs.v1"
FIELD_SCHEMA = "pivot_feedback_f_field_structural_coverage.v1"
CANDIDATE_SCHEMA = "pivot_feedback_f_attention_shock_candidate.v1"
SLOT_SCHEMA = "pivot_feedback_f_slot_assignment.v1"
STATUS = (
    "PASS_OUTCOME_BLIND_FIELD_AND_EPISODE_ASSIGNMENT_COMPONENT_"
    "FINAL_PANEL_AND_SEAL_PENDING"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_OUTCOME_BLIND_FIELD_AND_EPISODE_ASSIGNMENT_"
    "FINAL_PANEL_AND_SEAL_PENDING"
)
PERIODS = {
    "g1_empirical_replication": (2013, 2016),
    "model_selection": (2017, 2019),
    "g4_temporal_holdout": (2020, 2022),
}
EXPECTED_FILES = {
    "FIELD_EPISODE_ASSIGNMENT.md",
    "attention_shock_candidates.parquet",
    "field_episode_assignment.json",
    "field_episode_assignment.outputs.json",
    "field_structural_coverage.parquet",
    "slot_assignment.parquet",
}
ARTIFACT_FILES = EXPECTED_FILES - {"field_episode_assignment.outputs.json"}
METRICS = (
    "eligible_author_count",
    "matured_entry_opportunity_count",
    "pre_cutoff_work_count",
    "publication_date_completeness",
    "fixed_horizon_outcome_coverage",
    "pre_entry_t_minus_1_edge_coverage",
    "orcid_linked_author_rate",
    "independent_attention_shock_count",
)
FORBIDDEN_COLUMNS = {
    "entry_label",
    "persistence_label",
    "entry_effect",
    "persistence_effect",
    "effect_size",
    "realized_citation_count",
    "model_score",
    "prediction",
    "g4_outcome",
    "author_id",
    "author_numeric_id",
    "work_id",
    "work_numeric_id",
}
FIELD_SIGNATURE = (
    ("field_id", "uint8", False),
    ("field_name", "string", False),
    ("domain_id", "string", False),
    ("domain_name", "string", False),
    ("eligible_author_count", "uint64", False),
    ("matured_entry_opportunity_count", "uint64", False),
    ("pre_cutoff_work_count", "uint64", False),
    ("publication_date_completeness_numerator", "uint64", False),
    ("publication_date_completeness_denominator", "uint64", False),
    ("publication_date_completeness", "double", False),
    ("fixed_horizon_outcome_coverage_numerator", "uint64", False),
    ("fixed_horizon_outcome_coverage_denominator", "uint64", False),
    ("fixed_horizon_outcome_coverage", "double", False),
    ("pre_entry_t_minus_1_edge_coverage_numerator", "uint64", False),
    ("pre_entry_t_minus_1_edge_coverage_denominator", "uint64", False),
    ("pre_entry_t_minus_1_edge_coverage", "double", False),
    ("orcid_linked_author_rate_numerator", "uint64", False),
    ("orcid_linked_author_rate_denominator", "uint64", False),
    ("orcid_linked_author_rate", "double", False),
    ("independent_attention_shock_count", "uint32", False),
    ("minimum_structural_coverage_numerator", "uint8", False),
    ("minimum_structural_coverage_denominator", "uint8", False),
    ("minimum_structural_coverage", "double", False),
    ("domain_field_rank", "uint8", False),
    ("domain_primary", "bool", False),
    ("domain_reserve", "bool", False),
    ("selected_slot_id", "string", True),
)
CANDIDATE_SIGNATURE = (
    ("field_id", "uint8", False),
    ("topic_id", "uint32", False),
    ("topic_name", "string", False),
    ("subfield_id", "uint16", False),
    ("subfield_name", "string", False),
    ("decision_year", "uint16", False),
    ("temporal_partition_id", "string", False),
    ("reference_count", "uint64", False),
    ("recent_count", "uint64", False),
    ("acceleration", "double", False),
    ("field_year_median", "double", False),
    ("field_year_mad", "double", False),
    ("modified_z", "double", False),
    ("selected_field", "bool", False),
    ("assigned_slot_id", "string", True),
    ("retained_for_assigned_panel", "bool", False),
)
SLOT_SIGNATURE = (
    ("slot_id", "string", False),
    ("role", "string", False),
    ("temporal_partition_id", "string", False),
    ("selected_domain_id", "string", False),
    ("nominal_primary_field_id", "uint8", False),
    ("selected_field_id", "uint8", False),
    ("reserve_field_ids", "fixed_size_list<element: uint8>[2]", False),
    ("reserve_activated", "bool", False),
    ("reserve_activation_reason", "string", True),
    ("retained_attention_candidate_count", "uint32", False),
    ("independent_subfield_family_count", "uint16", False),
    ("sealed", "bool", False),
    ("author_cold", "bool", False),
)


class AssignmentVerificationError(RuntimeError):
    """Fail-closed field/episode assignment verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssignmentVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssignmentVerificationError(f"invalid {label}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _schema_version(schema: pa.Schema) -> str | None:
    value = (schema.metadata or {}).get(b"schema_version")
    return None if value is None else value.decode("utf-8")


def _schema_signature(schema: pa.Schema) -> tuple[tuple[str, str, bool], ...]:
    return tuple((field.name, str(field.type), field.nullable) for field in schema)


def _ratio(numerator: int, denominator: int) -> Fraction:
    return Fraction(numerator, denominator) if denominator else Fraction(0, 1)


def _period_for_year(year: int) -> str:
    for name, (start, end) in PERIODS.items():
        if start <= year <= end:
            return name
    raise AssignmentVerificationError(f"candidate decision year out of range: {year}")


def _validate_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve(strict=True)
    require(path.is_file() and path.is_relative_to(ROOT), "invalid config path")
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("mode")
        == "F_PHASE_OUTCOME_BLIND_FIELD_AND_EPISODE_ASSIGNMENT_COMPONENT_NOT_A_GATE",
        "config improperly declares a gate",
    )
    completion = _mapping(config.get("completion_accounting"), "completion accounting")
    require(completion.get("this_component_is_new_gate") is False, "component promoted to gate")
    require(completion.get("this_component_alone_may_pass_f") is False, "component may pass F")
    require(
        completion.get("final_panel_conformance_after_assignment") == "PENDING",
        "final-panel state differs",
    )
    require(completion.get("atomic_partition_and_holdout_byte_seal") == "PENDING", "seal state differs")
    require(completion.get("confirmatory_entry_or_persistence_labels_read") == 0, "label read declared")
    require(completion.get("model_fitted") is False, "model declared fitted")
    require(completion.get("g4_open") is False, "G4 declared open")
    output = _mapping(config.get("output_contract"), "output contract")
    require(output.get("passing_status") == STATUS, "configured status mismatch")
    require(output.get("raw_author_or_work_ids_in_output") is False, "raw IDs allowed")
    return config, _sha256(path)


def _validate_receipt(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = _read_json(
        root / "field_episode_assignment.outputs.json", "assignment output receipt"
    )
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema mismatch")
    require(receipt.get("status") == STATUS, "receipt status mismatch")
    declared = receipt.get("artifacts")
    require(isinstance(declared, list), "receipt artifacts must be a list")
    require(receipt.get("artifact_count") == len(declared) == 5, "artifact count mismatch")
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in declared:
        item = _mapping(raw, "receipt artifact")
        relative = item.get("relative_path")
        require(isinstance(relative, str) and relative in ARTIFACT_FILES, "unexpected artifact path")
        require(relative not in seen, "duplicate receipt artifact")
        seen.add(relative)
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"invalid artifact: {relative}")
        require(item.get("bytes") == path.stat().st_size, f"artifact byte mismatch: {relative}")
        require(item.get("sha256") == _sha256(path), f"artifact SHA mismatch: {relative}")
        if relative.endswith(".parquet"):
            require(
                item.get("rows") == pq.ParquetFile(path).metadata.num_rows,
                f"artifact row mismatch: {relative}",
            )
        observed.append(item)
    require(seen == ARTIFACT_FILES, "receipt artifact set mismatch")
    require(
        [str(item["relative_path"]) for item in observed]
        == sorted(str(item["relative_path"]) for item in observed),
        "receipt artifacts are not sorted",
    )
    tree = [
        (str(item["relative_path"]), int(item["bytes"]), str(item["sha256"]))
        for item in observed
    ]
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(_canonical_bytes(tree)).hexdigest(),
        "artifact tree mismatch",
    )
    require(receipt.get("entry_or_persistence_labels_computed") is False, "receipt declares labels")
    require(receipt.get("final_panel_conformance_complete") is False, "receipt passes final panel")
    require(receipt.get("pass_f") is False, "receipt improperly passes F")
    require(receipt.get("pass_g0") is False, "receipt improperly passes G0")
    require(receipt.get("g4_open") is False, "receipt opens G4")
    require(receipt.get("receipt_written_last") is True, "receipt-last assertion absent")
    return receipt, observed


def _field_metadata(config: Mapping[str, Any]) -> tuple[dict[int, dict[str, str]], dict[str, int]]:
    binding = _mapping(
        _mapping(config["bound_design_sources"], "design sources")["cross_field_matrix"],
        "cross-field binding",
    )
    path = (ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(ROOT), "cross-field path escape")
    require(_sha256(path) == binding["sha256"], "cross-field SHA mismatch")
    matrix = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "cross-field matrix")
    metadata: dict[int, dict[str, str]] = {}
    codes: dict[str, int] = {}
    for raw_domain in matrix["candidate_universe"]["domains"]:
        domain = _mapping(raw_domain, "domain")
        domain_id = str(domain["domain_id"])
        codes[domain_id] = int(domain["rng_code"])
        for raw_field in domain["fields"]:
            field = _mapping(raw_field, "field")
            metadata[int(field["field_id"])] = {
                "field_name": str(field["display_name"]),
                "domain_id": domain_id,
                "domain_name": str(domain["display_name"]),
            }
    require(set(metadata) == set(range(11, 37)), "cross-field closure mismatch")
    require(len(codes) == 4, "domain count mismatch")
    return metadata, codes


def _validate_repository_bindings(config: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    expected_raw = {
        "authorization_parent": config["authorization_parent"],
        **_mapping(config["bound_design_sources"], "bound design sources"),
    }
    expected = {}
    for name, raw in expected_raw.items():
        binding = _mapping(raw, name)
        path = (ROOT / str(binding["relative_path"])).resolve(strict=True)
        require(path.is_relative_to(ROOT), f"repository path escape: {name}")
        require(_sha256(path) == binding["sha256"], f"repository binding SHA mismatch: {name}")
        expected[name] = {
            "relative_path": str(binding["relative_path"]),
            "sha256": str(binding["sha256"]),
        }
    require(report.get("repository_bindings") == expected, "report repository bindings differ")


def _validate_input_bindings(config: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    expected = _mapping(config["normalized_inputs"], "normalized inputs")
    observed = _mapping(report.get("input_bindings"), "report input bindings")
    require(set(observed) == set(expected), "input binding set mismatch")
    for name, raw in expected.items():
        contract = _mapping(raw, f"input contract {name}")
        item = _mapping(observed[name], f"observed input {name}")
        expected_root = contract.get("expected_root_basename")
        if expected_root is not None:
            require(item.get("root_basename") == expected_root, f"input root mismatch: {name}")
            require(item.get("receipt_basename") == contract["receipt_basename"], f"input receipt mismatch: {name}")
            require(item.get("receipt_sha256") == contract["receipt_sha256"], f"input receipt SHA mismatch: {name}")
        if "artifact_prefix" in contract:
            require(item.get("artifact_prefix") == contract["artifact_prefix"], f"input prefix mismatch: {name}")
            require(item.get("artifacts_rehashed") == contract["artifact_count"], f"input rehash count mismatch: {name}")
            if "artifact_bytes" in contract:
                require(item.get("artifact_bytes") == contract["artifact_bytes"], f"input bytes mismatch: {name}")
            tree = item.get("selected_tree_sha256")
            require(isinstance(tree, str) and len(tree) == 64, f"input tree missing: {name}")
        if "artifact_relative_path" in contract:
            require(item.get("artifact_relative_path") == contract["artifact_relative_path"], f"compact path mismatch: {name}")
            require(item.get("artifact_sha256") == contract["artifact_sha256"], f"compact SHA mismatch: {name}")
            require(item.get("artifact_rows") == contract["artifact_rows"], f"compact rows mismatch: {name}")
        if name == "topic_taxonomy_projection":
            require(item.get("basename") == contract["expected_basename"], "taxonomy basename mismatch")
            require(item.get("sha256") == contract["sha256"], "taxonomy SHA mismatch")
            require(item.get("rows") == contract["rows"], "taxonomy rows mismatch")


def _metric_values(row: Mapping[str, Any]) -> dict[str, Fraction]:
    return {
        "eligible_author_count": Fraction(int(row["eligible_author_count"]), 1),
        "matured_entry_opportunity_count": Fraction(
            int(row["matured_entry_opportunity_count"]), 1
        ),
        "pre_cutoff_work_count": Fraction(int(row["pre_cutoff_work_count"]), 1),
        "publication_date_completeness": _ratio(
            int(row["publication_date_completeness_numerator"]),
            int(row["publication_date_completeness_denominator"]),
        ),
        "fixed_horizon_outcome_coverage": _ratio(
            int(row["fixed_horizon_outcome_coverage_numerator"]),
            int(row["fixed_horizon_outcome_coverage_denominator"]),
        ),
        "pre_entry_t_minus_1_edge_coverage": _ratio(
            int(row["pre_entry_t_minus_1_edge_coverage_numerator"]),
            int(row["pre_entry_t_minus_1_edge_coverage_denominator"]),
        ),
        "orcid_linked_author_rate": _ratio(
            int(row["orcid_linked_author_rate_numerator"]),
            int(row["orcid_linked_author_rate_denominator"]),
        ),
        "independent_attention_shock_count": Fraction(
            int(row["independent_attention_shock_count"]), 1
        ),
    }


def _validate_candidates(table: pa.Table) -> tuple[list[dict[str, Any]], dict[tuple[int, str], int]]:
    require(_schema_version(table.schema) == CANDIDATE_SCHEMA, "candidate schema mismatch")
    require(
        _schema_signature(table.schema) == CANDIDATE_SIGNATURE,
        "candidate exact schema mismatch",
    )
    require(FORBIDDEN_COLUMNS.isdisjoint(table.column_names), "forbidden candidate column present")
    rows = table.to_pylist()
    keys: list[tuple[int, int, int, int]] = []
    topic_metadata: dict[int, tuple[Any, ...]] = {}
    families: dict[tuple[int, str], set[int]] = {
        (field_id, period): set()
        for field_id in range(11, 37)
        for period in PERIODS
    }
    for row in rows:
        field_id = int(row["field_id"])
        topic_id = int(row["topic_id"])
        subfield_id = int(row["subfield_id"])
        year = int(row["decision_year"])
        require(11 <= field_id <= 36, "candidate field out of range")
        period = _period_for_year(year)
        require(row["temporal_partition_id"] == period, "candidate period mismatch")
        reference = int(row["reference_count"])
        recent = int(row["recent_count"])
        acceleration = float(row["acceleration"])
        median = float(row["field_year_median"])
        mad = float(row["field_year_mad"])
        modified_z = float(row["modified_z"])
        require(reference > 0 and recent > reference, "candidate count rule failed")
        expected_acceleration = math.log1p(recent / 3.0) - math.log1p(reference / 3.0)
        require(
            math.isclose(acceleration, expected_acceleration, rel_tol=1e-12, abs_tol=1e-12),
            "candidate acceleration mismatch",
        )
        require(math.isfinite(median) and math.isfinite(mad) and mad > 0.0, "candidate MAD invalid")
        expected_z = 0.6744897501960817 * (acceleration - median) / mad
        require(
            math.isclose(modified_z, expected_z, rel_tol=1e-12, abs_tol=1e-12),
            "candidate modified-z mismatch",
        )
        require(modified_z >= 3.0, "candidate below modified-z threshold")
        metadata = (
            str(row["topic_name"]),
            subfield_id,
            str(row["subfield_name"]),
            field_id,
        )
        require(topic_id not in topic_metadata or topic_metadata[topic_id] == metadata, "topic metadata differs")
        topic_metadata[topic_id] = metadata
        keys.append((field_id, year, subfield_id, topic_id))
        families[(field_id, period)].add(subfield_id)
    require(keys == sorted(keys), "candidate rows are not sorted")
    require(len(keys) == len(set((field, year, topic) for field, year, _, topic in keys)), "duplicate candidate topic-year")
    return rows, {key: len(value) for key, value in families.items()}


def _validate_fields(
    table: pa.Table,
    *,
    metadata: Mapping[int, Mapping[str, str]],
    shock_families: Mapping[tuple[int, str], int],
) -> tuple[list[dict[str, Any]], dict[int, tuple[Any, ...]], dict[str, list[int]]]:
    require(_schema_version(table.schema) == FIELD_SCHEMA, "field schema mismatch")
    require(
        _schema_signature(table.schema) == FIELD_SIGNATURE,
        "field exact schema mismatch",
    )
    require(FORBIDDEN_COLUMNS.isdisjoint(table.column_names), "forbidden field column present")
    rows = table.to_pylist()
    require(len(rows) == 26, "field row count mismatch")
    require([int(row["field_id"]) for row in rows] == list(range(11, 37)), "field rows differ")
    values = {int(row["field_id"]): _metric_values(row) for row in rows}
    rank_keys: dict[int, tuple[Any, ...]] = {}
    by_domain: dict[str, list[int]] = {}
    for row in rows:
        field_id = int(row["field_id"])
        expected_meta = metadata[field_id]
        for key in ("field_name", "domain_id", "domain_name"):
            require(row[key] == expected_meta[key], f"field metadata mismatch: {field_id}")
        by_domain.setdefault(str(row["domain_id"]), []).append(field_id)
        for name in (
            "publication_date_completeness",
            "fixed_horizon_outcome_coverage",
            "pre_entry_t_minus_1_edge_coverage",
            "orcid_linked_author_rate",
        ):
            observed = float(row[name])
            require(math.isfinite(observed), f"nonfinite ratio: {field_id} {name}")
            require(
                math.isclose(observed, float(values[field_id][name]), rel_tol=0.0, abs_tol=1e-15),
                f"ratio mismatch: {field_id} {name}",
            )
        require(
            int(row["fixed_horizon_outcome_coverage_denominator"]) > 0
            and values[field_id]["fixed_horizon_outcome_coverage"] == 1,
            f"fixed-horizon coverage differs: {field_id}",
        )
        expected_shock = min(
            int(shock_families[(field_id, period)]) for period in PERIODS
        )
        require(
            int(row["independent_attention_shock_count"]) == expected_shock,
            f"shock-family minimum mismatch: {field_id}",
        )
        percentiles = [
            Fraction(
                sum(values[other][name] <= values[field_id][name] for other in values),
                26,
            )
            for name in METRICS
        ]
        expected_minimum = min(percentiles)
        observed_minimum = _ratio(
            int(row["minimum_structural_coverage_numerator"]),
            int(row["minimum_structural_coverage_denominator"]),
        )
        require(observed_minimum == expected_minimum, f"minimum coverage mismatch: {field_id}")
        require(
            math.isclose(
                float(row["minimum_structural_coverage"]),
                float(expected_minimum),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"minimum coverage float mismatch: {field_id}",
        )
        rank_keys[field_id] = (
            expected_minimum,
            values[field_id]["eligible_author_count"],
            values[field_id]["fixed_horizon_outcome_coverage"],
            -field_id,
        )
    for domain_id, fields in by_domain.items():
        expected_order = sorted(fields, key=lambda field_id: rank_keys[field_id], reverse=True)
        observed_order = sorted(fields, key=lambda field_id: int(rows[field_id - 11]["domain_field_rank"]))
        require(observed_order == expected_order, f"domain field ranking mismatch: {domain_id}")
        for rank, field_id in enumerate(expected_order, 1):
            row = rows[field_id - 11]
            require(int(row["domain_field_rank"]) == rank, f"domain rank differs: {field_id}")
            require(row["domain_primary"] is (rank == 1), f"primary marker differs: {field_id}")
            require(row["domain_reserve"] is (rank in (2, 3)), f"reserve marker differs: {field_id}")
    return rows, rank_keys, by_domain


def _validate_slots(
    table: pa.Table,
    *,
    field_rows: Sequence[Mapping[str, Any]],
    rank_keys: Mapping[int, tuple[Any, ...]],
    by_domain: Mapping[str, Sequence[int]],
    domain_codes: Mapping[str, int],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    require(_schema_version(table.schema) == SLOT_SCHEMA, "slot schema mismatch")
    require(
        _schema_signature(table.schema) == SLOT_SIGNATURE,
        "slot exact schema mismatch",
    )
    require(FORBIDDEN_COLUMNS.isdisjoint(table.column_names), "forbidden slot column present")
    rows = table.to_pylist()
    require(len(rows) == 4, "slot count mismatch")
    domain_order = sorted(
        by_domain,
        key=lambda domain_id: (
            rank_keys[sorted(by_domain[domain_id], key=lambda field_id: rank_keys[field_id], reverse=True)[0]],
            -int(domain_codes[domain_id]),
        ),
        reverse=True,
    )
    selected_fields: set[int] = set()
    for index, (row, domain_id) in enumerate(zip(rows, domain_order, strict=True)):
        ranked = sorted(by_domain[domain_id], key=lambda field_id: rank_keys[field_id], reverse=True)
        nominal_primary = ranked[0]
        reserves = ranked[1:3]
        require(row["selected_domain_id"] == domain_id, "slot domain order mismatch")
        require(
            int(row["nominal_primary_field_id"]) == nominal_primary,
            "slot nominal primary field mismatch",
        )
        require(
            [int(value) for value in row["reserve_field_ids"]] == reserves,
            "slot reserve fields mismatch",
        )
        if index < 3:
            require(row["slot_id"] == f"g1_domain_slot_{index + 1:02d}", "G1 slot ID mismatch")
            require(row["role"] == "g1_replication", "G1 role mismatch")
            require(row["temporal_partition_id"] == "g1_empirical_replication", "G1 period mismatch")
            require(row["sealed"] is False and row["author_cold"] is False, "G1 improperly sealed/cold")
        else:
            require(row["slot_id"] == "g4_domain_author_cold_slot", "G4 slot ID mismatch")
            require(row["role"] == "g4_domain_author_cold", "G4 role mismatch")
            require(row["temporal_partition_id"] == "g4_temporal_holdout", "G4 period mismatch")
            require(row["sealed"] is True, "G4 slot is not sealed")
            require(row["author_cold"] is True, "G4 slot is not author-cold")
        candidate_order = [nominal_primary, *reserves]
        conforming: list[tuple[int, list[Mapping[str, Any]], set[int]]] = []
        for candidate_field in candidate_order:
            candidate_rows = [
                candidate
                for candidate in candidates
                if int(candidate["field_id"]) == candidate_field
                and candidate["temporal_partition_id"]
                == row["temporal_partition_id"]
            ]
            candidate_families = {
                int(candidate["subfield_id"]) for candidate in candidate_rows
            }
            if len(candidate_families) >= 3:
                conforming.append(
                    (candidate_field, candidate_rows, candidate_families)
                )
        require(conforming, "nominal primary and reserves all fail structural conformance")
        selected_field, retained, families = conforming[0]
        require(
            int(row["selected_field_id"]) == selected_field,
            "slot selected field is not first conforming frozen candidate",
        )
        selected_fields.add(selected_field)
        activated = selected_field != nominal_primary
        require(row["reserve_activated"] is activated, "reserve activation marker mismatch")
        expected_reason = (
            "NOMINAL_PRIMARY_FEWER_THAN_THREE_INDEPENDENT_SUBFIELD_FAMILIES_IN_ASSIGNED_PERIOD"
            if activated
            else None
        )
        require(
            row["reserve_activation_reason"] == expected_reason,
            "reserve activation reason mismatch",
        )
        require(len(families) >= 3, "selected slot has fewer than three shock families")
        require(int(row["retained_attention_candidate_count"]) == len(retained), "slot candidate count mismatch")
        require(int(row["independent_subfield_family_count"]) == len(families), "slot family count mismatch")
    require(len(selected_fields) == 4, "selected fields are not unique")
    require(len({str(row["selected_domain_id"]) for row in rows}) == 4, "domains are not disjoint")
    for row in field_rows:
        field_id = int(row["field_id"])
        slot = next((item for item in rows if int(item["selected_field_id"]) == field_id), None)
        require(row["selected_slot_id"] == (None if slot is None else slot["slot_id"]), f"field slot marker mismatch: {field_id}")
    return rows


def _validate_candidate_assignment(
    candidates: Sequence[Mapping[str, Any]], slots: Sequence[Mapping[str, Any]]
) -> None:
    by_field = {int(row["selected_field_id"]): row for row in slots}
    for row in candidates:
        field_id = int(row["field_id"])
        slot = by_field.get(field_id)
        expected_selected = slot is not None
        expected_slot = None if slot is None else slot["slot_id"]
        expected_retained = bool(
            slot is not None
            and row["temporal_partition_id"] == slot["temporal_partition_id"]
        )
        require(row["selected_field"] is expected_selected, "candidate selected-field marker mismatch")
        require(row["assigned_slot_id"] == expected_slot, "candidate assigned-slot mismatch")
        require(row["retained_for_assigned_panel"] is expected_retained, "candidate retention mismatch")


def _validate_report(
    report: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha: str,
    fields: Sequence[Mapping[str, Any]],
    slots: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
    candidate_count: int,
) -> None:
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == STATUS, "report status mismatch")
    scope = _mapping(report.get("scope"), "report scope")
    require(scope.get("candidate_field_count") == 26, "report field count mismatch")
    require(scope.get("candidate_domain_count") == 4, "report domain count mismatch")
    require(scope.get("g1_selected_field_count") == 3, "report G1 count mismatch")
    require(scope.get("g4_selected_field_count") == 1, "report G4 count mismatch")
    for key in (
        "entry_or_persistence_labels_computed",
        "realized_citation_counts_read",
        "effect_metrics_read",
        "model_fitted",
        "g4_open",
        "final_panel_conformance_complete",
        "pass_f",
        "pass_g0",
    ):
        require(scope.get(key) is False, f"report scope improperly promotes {key}")
    require(report.get("field_structural_coverage") == list(fields), "report field table differs")
    require(report.get("slot_assignment") == list(slots), "report slot table differs")
    expected_map = {
        str(int(row["selected_field_id"])): str(row["slot_id"]) for row in slots
    }
    require(report.get("selected_field_to_slot") == expected_map, "selected-field map mismatch")
    require(receipt.get("selected_field_ids") == [int(row["selected_field_id"]) for row in slots], "receipt selected fields differ")
    require(receipt.get("selected_domain_ids") == [str(row["selected_domain_id"]) for row in slots], "receipt selected domains differ")
    source = _mapping(report.get("source_accounting"), "source accounting")
    inputs = _mapping(config["normalized_inputs"], "normalized inputs")
    require(source["author_event_index"]["event_rows_scanned"] == inputs["author_history_events"]["rows"], "event-index row closure failed")
    require(source["author_activity"]["source_rows_scanned"] == inputs["author_history_events"]["rows"], "author-activity row closure failed")
    require(source["eligible_work"]["rows"] == inputs["cutoff_work_metadata"]["rows"], "work row closure failed")
    require(source.get("attention_shock_candidates") == candidate_count, "candidate accounting mismatch")
    require(int(source.get("topic_year_cells", 0)) > 0, "topic-year accounting empty")
    authorization = _mapping(report.get("authorization"), "report authorization")
    require(authorization.get("final_panel_leaveout_and_support_replay") == "AUTHORIZED_NEXT", "next authorization differs")
    require(authorization.get("operator_only_holdout_materialization") == "NOT_AUTHORIZED_BEFORE_FINAL_PANEL_CONFORMANCE", "holdout materialization authorized early")
    require(authorization.get("confirmatory_analytic_read") == "NOT_AUTHORIZED", "analytic read authorized")
    require(authorization.get("model_fitting") == "NOT_AUTHORIZED", "model fitting authorized")
    require(authorization.get("g4_open") == "NOT_AUTHORIZED", "G4 authorized")
    execution = _mapping(report.get("execution"), "report execution")
    require(execution.get("config_relative_path") == DEFAULT_CONFIG.relative_to(ROOT).as_posix(), "config path differs")
    require(execution.get("config_sha256") == config_sha, "config SHA differs")
    builder_path = (ROOT / str(execution.get("builder_relative_path"))).resolve(strict=True)
    require(builder_path.is_relative_to(ROOT), "builder path escape")
    require(_sha256(builder_path) == execution.get("builder_sha256"), "builder SHA differs")
    require(float(execution.get("elapsed_seconds", 0.0)) > 0.0, "elapsed time invalid")
    _validate_repository_bindings(config, report)
    _validate_input_bindings(config, report)
    require(report.get("claim_boundary") == config.get("claim_boundary"), "claim boundary differs")


def verify(root: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, config_sha = _validate_config(config_path)
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "output root is invalid")
    require(root.name == config["output_contract"]["root_basename"], "output root basename mismatch")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set mismatch")
    receipt, artifacts = _validate_receipt(root)
    report = _read_json(root / "field_episode_assignment.json", "assignment report")
    fields_table = pq.read_table(root / "field_structural_coverage.parquet")
    candidates_table = pq.read_table(root / "attention_shock_candidates.parquet")
    slots_table = pq.read_table(root / "slot_assignment.parquet")
    metadata, domain_codes = _field_metadata(config)
    candidates, shock_families = _validate_candidates(candidates_table)
    fields, rank_keys, by_domain = _validate_fields(
        fields_table,
        metadata=metadata,
        shock_families=shock_families,
    )
    slots = _validate_slots(
        slots_table,
        field_rows=fields,
        rank_keys=rank_keys,
        by_domain=by_domain,
        domain_codes=domain_codes,
        candidates=candidates,
    )
    _validate_candidate_assignment(candidates, slots)
    _validate_report(
        report,
        config=config,
        config_sha=config_sha,
        fields=fields,
        slots=slots,
        receipt=receipt,
        candidate_count=len(candidates),
    )
    markdown = (root / "FIELD_EPISODE_ASSIGNMENT.md").read_text(encoding="utf-8")
    require(f"Status: `{STATUS}`" in markdown, "markdown status mismatch")
    require("not an F pass" in markdown, "markdown F boundary missing")
    require("G4 remains unopened" in markdown, "markdown G4 boundary missing")
    return {
        "schema_version": "pivot_feedback_f_field_episode_assignment_verification.v1",
        "status": VERIFY_STATUS,
        "output_root_basename": root.name,
        "output_receipt_sha256": _sha256(
            root / "field_episode_assignment.outputs.json"
        ),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "artifacts_rehashed": len(artifacts) + 1,
        "candidate_rows_verified": len(candidates),
        "field_rows_verified": len(fields),
        "slot_assignment": slots,
        "final_panel_conformance_complete": False,
        "holdout_byte_seal_complete": False,
        "pass_f": False,
        "pass_g0": False,
        "g4_open": False,
        "next_authorized_component": "FINAL_PANEL_LEAVEOUT_AND_SUPPORT_REPLAY",
        "verifier_sha256": _sha256(Path(__file__).resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify(args.output_root, args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        AssignmentVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pa.ArrowException,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_FIELD_EPISODE_ASSIGNMENT_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
