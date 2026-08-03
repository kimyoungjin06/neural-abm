#!/usr/bin/env python3
"""Independently verify an S0 coauthor graph profile output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "pivot_feedback_s0_coauthor_graph_profile_report.v1"
FIELD_YEAR_SCHEMA = "pivot_feedback_s0_coauthor_graph_field_year.v1"
OUTPUT_SCHEMA = "pivot_feedback_s0_coauthor_graph_profile_outputs.v1"
FULL_STATUS = (
    "PASS_S0_EXACT_NONRANKED_COAUTHOR_GRAPH_PROFILES_"
    "WITH_CURRENT_SNAPSHOT_AUTHOR_TOKEN_LIMITATIONS"
)
PILOT_STATUS = "PILOT_S0_COAUTHOR_GRAPH_PROFILE_NONPROMOTABLE"
FULL_VERIFY_STATUS = (
    "PASS_INDEPENDENT_S0_COAUTHOR_GRAPH_PROFILE_TOPOLOGY_CONSERVATION_"
    "ORCID_AFFILIATION_RECEIPT_STILL_REQUIRED"
)
PILOT_VERIFY_STATUS = (
    "PASS_INDEPENDENT_NONPROMOTABLE_S0_COAUTHOR_GRAPH_PROFILE_PILOT"
)
EXPECTED_FILES = {
    "coauthor_graph_profile_report.json",
    "coauthor_graph_profile_report.md",
    "coauthor_graph_profile_report.outputs.json",
    "field_year_coauthor_graph.parquet",
}
GRAPH_METRICS = (
    "graph_node_tokens",
    "distinct_undirected_edges",
    "degree_sum",
    "degree_min",
    "degree_p25",
    "degree_p50",
    "degree_p75",
    "degree_p90",
    "degree_p95",
    "degree_p99",
    "degree_max",
    "connected_components",
    "largest_component_nodes",
)
FORBIDDEN_COLUMNS = {
    "author_numeric_id",
    "author_u",
    "author_v",
    "work_numeric_id",
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
}
YEAR_MULTIPLIER = 65_536


class GraphVerificationError(RuntimeError):
    """Fail-closed graph verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


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


def _write_report(path: Path, result: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    require(
        target != root and not target.is_relative_to(root),
        "verification report must be outside output root",
    )
    payload = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, 0o600, follow_symlinks=False)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraphVerificationError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise GraphVerificationError(f"invalid {label}: {error}") from error


def _schema_version(schema: pa.Schema) -> str | None:
    value = (schema.metadata or {}).get(b"schema_version")
    return None if value is None else value.decode("utf-8")


def _validate_receipt(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = _read_json(
        root / "coauthor_graph_profile_report.outputs.json", "output receipt"
    )
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "receipt schema mismatch")
    status = receipt.get("status")
    require(status in {FULL_STATUS, PILOT_STATUS}, "receipt status mismatch")
    full_scope = status == FULL_STATUS
    require(receipt.get("full_scope") is full_scope, "receipt scope mismatch")
    require(receipt.get("component_pass") is full_scope, "receipt pass mismatch")
    require(receipt.get("s0_gate_pass") is False, "receipt improperly passes S0")
    require(receipt.get("d0_authorized") is False, "receipt improperly authorizes D0")
    require(
        receipt.get("completion_contract")
        == "EXACT_INTERMEDIATES_REMOVED_RECEIPT_WRITTEN_LAST_ATOMIC_DIRECTORY_RENAME",
        "completion contract mismatch",
    )
    declared = receipt.get("artifacts")
    require(isinstance(declared, list), "receipt artifacts must be a list")
    require(receipt.get("artifact_count") == len(declared) == 3, "artifact count mismatch")
    expected = EXPECTED_FILES - {"coauthor_graph_profile_report.outputs.json"}
    observed = []
    for raw in declared:
        item = _mapping(raw, "receipt artifact")
        relative = item.get("relative_path")
        require(
            isinstance(relative, str)
            and "/" not in relative
            and relative in expected,
            "unexpected artifact path",
        )
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"invalid artifact: {relative}")
        actual = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        require(item == actual, f"artifact identity mismatch: {relative}")
        observed.append(actual)
    require({item["relative_path"] for item in observed} == expected, "artifact set mismatch")
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(_canonical_bytes(observed)).hexdigest(),
        "artifact tree mismatch",
    )
    return receipt, observed


def _validate_provenance(
    report: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = _mapping(report.get("provenance"), "provenance")
    require(
        provenance.get("config_relative_path")
        == "experiments/configs/pivot_feedback_s0_coauthor_graph_profile_v1.yaml",
        "unexpected config path",
    )
    require(
        provenance.get("runner_relative_path")
        == "scripts/build_pivot_feedback_s0_coauthor_graph_profile.py",
        "unexpected runner path",
    )
    config_path = REPO_ROOT / provenance["config_relative_path"]
    runner_path = REPO_ROOT / provenance["runner_relative_path"]
    require(_sha256_file(config_path) == provenance.get("config_sha256"), "config SHA mismatch")
    require(_sha256_file(runner_path) == provenance.get("runner_sha256"), "runner SHA mismatch")
    config = _read_yaml(config_path, "graph config")
    require(
        provenance.get("duckdb_version")
        == config["execution"]["required_duckdb_version"],
        "DuckDB provenance mismatch",
    )
    require(provenance.get("scipy_connected_components") is True, "SciPy provenance missing")
    replay = json.loads(json.dumps(report))
    expected = replay["provenance"].pop("result_fingerprint_sha256")
    observed = hashlib.sha256(_canonical_bytes(replay)).hexdigest()
    require(
        observed == expected == receipt.get("result_fingerprint_sha256"),
        "result fingerprint mismatch",
    )
    return config


def _core_keys(
    config: Mapping[str, Any],
    core_root: Path | None,
) -> tuple[list[int], dict[str, Any]]:
    binding = _mapping(config.get("s0_core_parent"), "core parent")
    summary_path = REPO_ROOT / str(binding["summary_relative_path"])
    require(_sha256_file(summary_path) == binding.get("summary_sha256"), "core summary SHA mismatch")
    root = (
        Path("/ssd") / str(binding["expected_root_basename"])
        if core_root is None
        else core_root.resolve(strict=True)
    )
    require(root.is_dir(), "bound core root missing")
    require(root.name == binding.get("expected_root_basename"), "core root basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    field_path = root / str(binding["field_year_relative_path"])
    require(_sha256_file(receipt_path) == binding.get("receipt_sha256"), "core receipt SHA mismatch")
    require(_sha256_file(field_path) == binding.get("field_year_sha256"), "core field-year SHA mismatch")
    table = pq.read_table(field_path, columns=["field_id", "publication_year"])
    require(_schema_version(table.schema) == binding.get("field_year_schema_version"), "core schema mismatch")
    keys = [
        int(field) * YEAR_MULTIPLIER + (0 if year is None else int(year))
        for field, year in zip(
            table["field_id"].to_pylist(),
            table["publication_year"].to_pylist(),
            strict=True,
        )
    ]
    require(len(keys) == binding.get("expected_field_year_rows"), "core key row mismatch")
    return keys, {
        "summary_sha256": binding["summary_sha256"],
        "receipt_sha256": binding["receipt_sha256"],
        "field_year_sha256": binding["field_year_sha256"],
        "field_year_rows": len(keys),
    }


def _validate_graph_table(
    table: pa.Table,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    core_keys: Sequence[int],
) -> dict[str, Any]:
    inventory = _mapping(report.get("field_year_inventory"), "field-year inventory")
    require(_schema_version(table.schema) == FIELD_YEAR_SCHEMA, "field-year schema version mismatch")
    expected_schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=True),
            pa.field("declared_team_cap", pa.uint16(), nullable=False),
            *[
                pa.field(metric, pa.uint64(), nullable=False)
                for metric in GRAPH_METRICS
            ],
            pa.field("largest_component_share", pa.float64(), nullable=True),
        ]
    )
    require(table.schema.equals(expected_schema, check_metadata=False), "field-year physical schema mismatch")
    require(FORBIDDEN_COLUMNS.isdisjoint(table.column_names), "forbidden graph column present")
    require(len(table) == inventory.get("rows") == len(core_keys) * 3, "graph row count mismatch")
    rows = table.to_pylist()
    keys = [
        (
            int(row["field_id"]),
            -1 if row["publication_year"] is None else int(row["publication_year"]),
            int(row["declared_team_cap"]),
        )
        for row in rows
    ]
    require(keys == sorted(keys), "graph rows are not sorted")
    require(len(keys) == len(set(keys)), "duplicate graph row")
    observed_cell_caps = {
        (
            int(row["field_id"]) * YEAR_MULTIPLIER
            + (0 if row["publication_year"] is None else int(row["publication_year"])),
            int(row["declared_team_cap"]),
        )
        for row in rows
    }
    require(
        observed_cell_caps
        == {(key, cap) for key in core_keys for cap in (10, 50, 100)},
        "core cell by cap coverage mismatch",
    )
    prior_by_cell: dict[int, tuple[int, int]] = {}
    for row in rows:
        node_count = int(row["graph_node_tokens"])
        edge_count = int(row["distinct_undirected_edges"])
        degree_sum = int(row["degree_sum"])
        components = int(row["connected_components"])
        largest = int(row["largest_component_nodes"])
        degrees = [
            int(row[name])
            for name in (
                "degree_min",
                "degree_p25",
                "degree_p50",
                "degree_p75",
                "degree_p90",
                "degree_p95",
                "degree_p99",
                "degree_max",
            )
        ]
        require(degree_sum == 2 * edge_count, "degree sum does not equal twice edges")
        require(degrees == sorted(degrees), "degree order statistics are not monotone")
        if node_count == 0:
            require(
                edge_count == components == largest == degree_sum == 0,
                "empty graph metrics do not close",
            )
            require(row["largest_component_share"] is None, "empty graph share is not null")
            require(all(value == 0 for value in degrees), "empty graph degree quantiles are nonzero")
        else:
            require(1 <= components <= node_count, "component count outside node domain")
            require(1 <= largest <= node_count, "largest component outside node domain")
            require(components >= node_count - edge_count, "component lower bound failed")
            require(degrees[-1] <= node_count - 1, "maximum degree exceeds simple graph")
            require(
                math.isclose(
                    float(row["largest_component_share"]),
                    largest / node_count,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "largest component share mismatch",
            )
        cell = int(row["field_id"]) * YEAR_MULTIPLIER + (
            0 if row["publication_year"] is None else int(row["publication_year"])
        )
        current = (node_count, edge_count)
        if cell in prior_by_cell:
            prior = prior_by_cell[cell]
            require(
                current[0] >= prior[0] and current[1] >= prior[1],
                "nested graph profile monotonicity failed",
            )
        prior_by_cell[cell] = current
    sums = {
        metric: {
            str(cap): int(
                sum(
                    int(row[metric])
                    for row in rows
                    if int(row["declared_team_cap"]) == cap
                )
            )
            for cap in (10, 50, 100)
        }
        for metric in (
            "graph_node_tokens",
            "distinct_undirected_edges",
            "connected_components",
            "largest_component_nodes",
        )
    }
    require(sums == inventory.get("profile_metric_sums"), "profile metric sums mismatch")
    materialization = _mapping(
        report.get("materialization_accounting"), "materialization accounting"
    )
    graph_contract = _mapping(config.get("graph_contract"), "graph contract")
    for cap in (10, 50, 100):
        require(
            sums["distinct_undirected_edges"][str(cap)]
            <= int(materialization[f"actual_pair_incidences_cap_{cap}"])
            <= int(graph_contract[f"cap_{cap}_classified_declared_pair_incidences"]),
            f"edge incidence margin mismatch: {cap}",
        )
        require(
            sums["graph_node_tokens"][str(cap)]
            == int(materialization[f"materialized_node_tokens_cap_{cap}"]),
            f"materialized node-token closure mismatch: {cap}",
        )
        require(
            sums["distinct_undirected_edges"][str(cap)]
            == int(materialization[f"materialized_distinct_edges_cap_{cap}"]),
            f"materialized distinct-edge closure mismatch: {cap}",
        )
    return {
        "rows": len(table),
        "core_cells": len(core_keys),
        "profile_metric_sums": sums,
        "all_degree_sums_close": True,
        "all_component_bounds_hold": True,
        "all_largest_component_shares_replayed": True,
    }


def verify(
    root: Path,
    *,
    core_root: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "output root is invalid")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "output root mode mismatch")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set mismatch")
    for path in root.iterdir():
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"output file mode mismatch: {path.name}")
    receipt, artifacts = _validate_receipt(root)
    report = _read_json(root / "coauthor_graph_profile_report.json", "graph report")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == receipt["status"], "report status mismatch")
    full_scope = receipt["status"] == FULL_STATUS
    scope = _mapping(report.get("scope"), "scope")
    require(scope.get("full_scope") is full_scope, "report scope mismatch")
    require(scope.get("component_pass") is full_scope, "report component pass mismatch")
    require(scope.get("pilot_promotion_allowed") is False, "pilot promotion allowed")
    require(scope.get("s0_gate_pass") is False, "report improperly passes S0")
    require(scope.get("d0_authorized") is False, "report improperly authorizes D0")
    require(scope.get("model_fitting_authorized") is False, "report authorizes model fitting")
    require(scope.get("gpu_used") is False, "report claims GPU use")
    profiles = _mapping(report.get("profiles"), "profiles")
    require(profiles.get("caps") == [10, 50, 100], "profile caps mismatch")
    require(profiles.get("ranked_or_selected") is False, "profiles are ranked")
    require(profiles.get("all_reported") is True, "profiles are incomplete")
    require(profiles.get("unbounded_all_edge_materialized") is False, "unbounded graph was materialized")
    gate = _mapping(report.get("gate_status"), "gate status")
    require(gate.get("d0_discovery_pilot") == "NOT_AUTHORIZED", "D0 gate mismatch")
    require(gate.get("g0_measurement_freeze") == "PENDING", "G0 gate mismatch")
    require(gate.get("empirical_g1_signature") == "NOT_RUN", "empirical G1 mismatch")
    require(gate.get("scientific_effect") == "NOT_ESTABLISHED", "effect claim overstated")
    execution = _mapping(report.get("execution"), "execution")
    require(
        execution.get("exact_intermediate_materialization_removed_before_completion")
        is True,
        "intermediate removal boundary missing",
    )
    config = _validate_provenance(report, receipt)
    require(report.get("claim_boundary") == config.get("claim_boundary"), "claim boundary mismatch")
    core_keys, core_binding = _core_keys(config, core_root)
    require(report.get("core_binding") == core_binding, "core binding mismatch")
    event = _mapping(report.get("event_binding"), "event binding")
    author = _mapping(config.get("author_history_parent"), "author parent")
    require(event.get("receipt_sha256") == author.get("receipt_sha256"), "event receipt mismatch")
    require(event.get("selected_artifact_stats_unchanged") is True, "event stat boundary missing")
    materialization = _mapping(
        report.get("materialization_accounting"), "materialization accounting"
    )
    require(materialization.get("partition_violations") == 0, "partition violations present")
    require(
        materialization.get("event_rows_cap_10")
        <= materialization.get("event_rows_cap_50")
        <= materialization.get("event_rows_cap_100")
        <= materialization.get("classified_rows")
        <= materialization.get("source_rows"),
        "event profile margins are not monotone",
    )
    if full_scope:
        require(
            event.get("selected_artifacts_rehashed")
            == author.get("expected_artifact_count"),
            "full event artifact count mismatch",
        )
        require(
            event.get("selected_artifact_bytes")
            == author.get("expected_artifact_bytes"),
            "full event byte mismatch",
        )
        require(
            event.get("selected_artifact_tree_sha256")
            == author.get("expected_artifact_tree_sha256"),
            "full event tree mismatch",
        )
        require(
            materialization.get("source_rows") == author.get("expected_rows"),
            "full event row mismatch",
        )
    parquet_path = root / "field_year_coauthor_graph.parquet"
    table = pq.read_table(parquet_path)
    table_result = _validate_graph_table(table, report, config, core_keys)
    artifact = _mapping(
        _mapping(report.get("field_year_inventory"), "field-year inventory").get(
            "artifact"
        ),
        "field-year artifact",
    )
    require(
        artifact
        == {
            "relative_path": parquet_path.name,
            "bytes": parquet_path.stat().st_size,
            "sha256": _sha256_file(parquet_path),
            "rows": len(table),
        },
        "field-year artifact identity mismatch",
    )
    return {
        "schema_version": "pivot_feedback_s0_coauthor_graph_profile_verification.v1",
        "status": FULL_VERIFY_STATUS if full_scope else PILOT_VERIFY_STATUS,
        "output_root_basename": root.name,
        "full_scope": full_scope,
        "component_pass": full_scope,
        "artifacts_rehashed": len(artifacts) + 1,
        "output_receipt_sha256": _sha256_file(
            root / "coauthor_graph_profile_report.outputs.json"
        ),
        "graph_verification": table_result,
        "unbounded_all_graph_still_not_materialized": True,
        "s0_gate_pass": False,
        "d0_authorized": False,
        "model_fitting_authorized": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "verifier_sha256": _sha256_file(Path(__file__).resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="write a new private verification report outside the output root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify(args.output_root, core_root=args.core_root)
        if args.report is not None:
            _write_report(args.report, result, root=args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (
        GraphVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pa.ArrowException,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_S0_COAUTHOR_GRAPH_PROFILE_VERIFICATION",
                    "error": str(error),
                }
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
