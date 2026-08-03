#!/usr/bin/env python3
"""Independently verify the compact S0 core inventory output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "pivot_feedback_s0_threshold_free_core_inventory_report.v1"
FIELD_YEAR_SCHEMA = "pivot_feedback_s0_field_year_core.v1"
ATTENTION_SCHEMA = "pivot_feedback_s0_attention_trajectory_core.v1"
OUTPUT_SCHEMA = "pivot_feedback_s0_threshold_free_core_inventory_outputs.v1"
STATUS = (
    "PASS_S0_THRESHOLD_FREE_CORE_FIELD_YEAR_INVENTORY_"
    "WITH_ORCID_AFFILIATION_AND_GRAPH_TOPOLOGY_PENDING"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_S0_CORE_OUTPUT_CONSERVATION_AND_ATTENTION_REPLAY_"
    "DEFERRED_COMPONENTS_STILL_PENDING"
)
EXPECTED_FILES = {
    "attention_trajectory_core.parquet",
    "field_year_core.parquet",
    "s0_core_inventory.json",
    "s0_core_inventory.md",
    "s0_core_inventory.outputs.json",
}
FORBIDDEN_COLUMNS = {
    "eligible_author_count",
    "matured_entry_opportunity_count",
    "fixed_horizon_outcome_coverage",
    "independent_attention_shock_count",
    "entry_effect_size",
    "persistence_effect_size",
    "overshoot_timing_gap",
    "predictive_score",
    "selected_best_field",
    "selected_best_threshold",
    "g4_holdout_outcome",
    "cited_by_count",
}


class S0VerificationError(RuntimeError):
    """Fail-closed S0 output verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S0VerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S0VerificationError(f"invalid {label}: {error}") from error


def _schema_version(schema: pa.Schema) -> str | None:
    metadata = schema.metadata or {}
    value = metadata.get(b"schema_version")
    return None if value is None else value.decode("utf-8")


def _validate_receipt(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = _read_json(root / "s0_core_inventory.outputs.json", "output receipt")
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "output receipt schema mismatch")
    require(receipt.get("status") == STATUS, "output receipt status mismatch")
    require(receipt.get("core_component_pass") is True, "core component is not passed")
    require(receipt.get("s0_gate_pass") is False, "receipt improperly passes S0")
    require(receipt.get("d0_authorized") is False, "receipt improperly authorizes D0")
    declared = receipt.get("artifacts")
    require(isinstance(declared, list), "receipt artifacts must be a list")
    require(receipt.get("artifact_count") == len(declared) == 4, "artifact count mismatch")
    observed: list[dict[str, Any]] = []
    for item_raw in declared:
        item = _mapping(item_raw, "receipt artifact")
        relative = item.get("relative_path")
        require(
            isinstance(relative, str)
            and "/" not in relative
            and relative in EXPECTED_FILES - {"s0_core_inventory.outputs.json"},
            "unexpected receipt artifact path",
        )
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"invalid artifact {relative}")
        observed_item = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        require(observed_item == item, f"artifact identity mismatch: {relative}")
        observed.append(observed_item)
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(_canonical_bytes(observed)).hexdigest(),
        "artifact tree mismatch",
    )
    return receipt, observed


def _validate_field_year(table: pa.Table, report: Mapping[str, Any]) -> dict[str, Any]:
    require(_schema_version(table.schema) == FIELD_YEAR_SCHEMA, "field-year schema version mismatch")
    require(len(table) == report["field_year_inventory"]["rows"], "field-year row count mismatch")
    require(FORBIDDEN_COLUMNS.isdisjoint(table.column_names), "forbidden field-year column present")
    fields = table["field_id"].to_pylist()
    years = table["publication_year"].to_pylist()
    keys = list(zip(fields, (-1 if year is None else year for year in years), strict=True))
    require(keys == sorted(keys), "field-year rows are not sorted")
    require(len(keys) == len(set(keys)), "duplicate field-year row")
    require(set(fields) == set(range(11, 37)), "field domain mismatch")
    require(sum(year is None for year in years) == 26, "null-year cell count mismatch")
    observed_years = [int(year) for year in years if year is not None]
    require(min(observed_years) == 1000 and max(observed_years) == 2050, "year range mismatch")

    columns = {
        name: table[name].to_pylist()
        for name in table.column_names
        if name not in {"field_id", "publication_year"}
    }
    for index in range(len(table)):
        total = columns["total_canonical_works"][index]
        require(columns["usable_date_works"][index] <= total, "usable works exceed total")
        require(
            columns["primary_topic_relation_works"][index]
            == columns["all_topic_relation_works"][index],
            "primary/all-topic closure failed",
        )
        require(
            columns["history_usable_distinct_author_work_pairs"][index]
            <= columns["distinct_author_work_pairs"][index],
            "usable author-work pairs exceed total",
        )
        require(
            columns["history_usable_distinct_author_tokens"][index]
            <= columns["distinct_author_tokens"][index],
            "usable author tokens exceed total",
        )
        require(
            columns["zero_declared_author_works"][index]
            + columns["single_declared_author_works"][index]
            + columns["multi_declared_author_works"][index]
            == total,
            "declared-author work partition does not conserve",
        )
        require(
            columns["declared_pair_incidences_cap_10"][index]
            <= columns["declared_pair_incidences_cap_50"][index]
            <= columns["declared_pair_incidences_cap_100"][index]
            <= columns["declared_pair_incidences_all"][index],
            "team-cap pair profile is not monotone",
        )
        require(
            columns["citation_calendar_support_at_least_10y"][index]
            <= columns["citation_calendar_support_at_least_5y"][index]
            <= columns["citation_calendar_support_at_least_3y"][index]
            <= columns["citation_calendar_support_at_least_1y"][index]
            <= total,
            "calendar-support profile is not monotone",
        )

    sums = {name: int(sum(values)) for name, values in columns.items()}
    work = report["work_accounting"]
    author = report["author_event_accounting"]
    require(sums["total_canonical_works"] == work["classified_field_rows"], "classified work margin mismatch")
    require(
        sums["primary_topic_relation_works"] == work["classified_field_rows"],
        "classified topic coverage margin mismatch",
    )
    require(
        sums["distinct_author_work_pairs"] == author["classified_field_rows"],
        "classified author-event margin mismatch",
    )
    require(work["classified_field_rows"] + work["null_field_rows"] == work["rows"], "work field margin does not conserve")
    require(
        author["classified_field_rows"] + author["null_field_rows"] == author["rows"],
        "author-event field margin does not conserve",
    )
    return {
        "rows": len(table),
        "field_count": len(set(fields)),
        "null_year_cells": sum(year is None for year in years),
        "observed_year_min": min(observed_years),
        "observed_year_max": max(observed_years),
        "metric_sums": sums,
    }


def _validate_attention(
    attention: pa.Table,
    field_year: pa.Table,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    require(_schema_version(attention.schema) == ATTENTION_SCHEMA, "attention schema version mismatch")
    require(
        len(attention) == report["attention_trajectory_inventory"]["rows"],
        "attention row count mismatch",
    )
    require(FORBIDDEN_COLUMNS.isdisjoint(attention.column_names), "forbidden attention column present")
    annual = {
        (int(field), int(year)): int(count)
        for field, year, count in zip(
            field_year["field_id"].to_pylist(),
            field_year["publication_year"].to_pylist(),
            field_year["total_canonical_works"].to_pylist(),
            strict=True,
        )
        if year is not None
    }
    observed_rows = attention.to_pylist()
    observed_keys = [
        (int(row["field_id"]), int(row["publication_year"])) for row in observed_rows
    ]
    require(observed_keys == sorted(annual), "attention keys differ from field-year keys")
    for row in observed_rows:
        field_id = int(row["field_id"])
        year = int(row["publication_year"])
        require(
            int(row["annual_canonical_works"]) == annual[(field_id, year)],
            "annual attention mismatch",
        )
        require(
            int(row["trailing_3y_canonical_works"])
            == sum(annual.get((field_id, candidate), 0) for candidate in range(year - 2, year + 1)),
            "trailing-three-year attention mismatch",
        )
        require(
            int(row["trailing_5y_canonical_works"])
            == sum(annual.get((field_id, candidate), 0) for candidate in range(year - 4, year + 1)),
            "trailing-five-year attention mismatch",
        )
    return {
        "rows": len(attention),
        "all_annual_counts_replayed": True,
        "all_trailing_3y_counts_replayed": True,
        "all_trailing_5y_counts_replayed": True,
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "output root is invalid")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set mismatch")
    receipt, artifacts = _validate_receipt(root)
    report = _read_json(root / "s0_core_inventory.json", "S0 core report")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == STATUS, "report status mismatch")
    require(report["scope"]["core_component_pass"] is True, "report core pass missing")
    require(report["scope"]["s0_gate_pass"] is False, "report improperly passes S0")
    require(report["scope"]["d0_authorized"] is False, "report improperly authorizes D0")
    require(report["scope"]["model_fitting_authorized"] is False, "report authorizes model fitting")
    require(report["scope"]["gpu_used"] is False, "report claims GPU use")
    require(report.get("forbidden_metrics_emitted") == [], "report emits forbidden metrics")
    deferred = _mapping(report.get("deferred_required_metrics"), "deferred metrics")
    require(
        set(deferred) == {"orcid_field_year", "affiliation_field_year", "coauthor_graph_field_year"},
        "deferred component set mismatch",
    )
    require(
        all(str(_mapping(value, key).get("state")).startswith("NOT_RUN") for key, value in deferred.items()),
        "deferred component improperly promoted",
    )
    require(report["gate_status"]["d0_discovery_pilot"] == "NOT_AUTHORIZED", "D0 gate differs")
    require(report["gate_status"]["scientific_effect"] == "NOT_ESTABLISHED", "effect is overstated")

    provenance = _mapping(report.get("provenance"), "report provenance")
    config_path = REPO_ROOT / provenance["config_relative_path"]
    runner_path = REPO_ROOT / provenance["runner_relative_path"]
    require(_sha256_file(config_path) == provenance["config_sha256"], "config SHA mismatch")
    require(_sha256_file(runner_path) == provenance["runner_sha256"], "runner SHA mismatch")
    fingerprint_report = json.loads(json.dumps(report))
    expected_fingerprint = fingerprint_report["provenance"].pop(
        "result_fingerprint_sha256"
    )
    require(
        hashlib.sha256(_canonical_bytes(fingerprint_report)).hexdigest()
        == expected_fingerprint
        == receipt["result_fingerprint_sha256"],
        "result fingerprint mismatch",
    )

    field_year = pq.read_table(root / "field_year_core.parquet")
    attention = pq.read_table(root / "attention_trajectory_core.parquet")
    field_result = _validate_field_year(field_year, report)
    attention_result = _validate_attention(attention, field_year, report)
    return {
        "schema_version": "pivot_feedback_s0_threshold_free_core_verification.v1",
        "status": VERIFY_STATUS,
        "output_root_basename": root.name,
        "artifacts_rehashed": len(artifacts) + 1,
        "output_receipt_sha256": _sha256_file(root / "s0_core_inventory.outputs.json"),
        "field_year_verification": field_result,
        "attention_replay": attention_result,
        "deferred_components_still_not_run": sorted(deferred),
        "s0_gate_pass": False,
        "d0_authorized": False,
        "model_fitting_authorized": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "verifier_sha256": _sha256_file(Path(__file__).resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify(args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (S0VerificationError, OSError, ValueError, pa.ArrowException) as error:
        print(json.dumps({"status": "BLOCKED_S0_CORE_VERIFICATION", "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
