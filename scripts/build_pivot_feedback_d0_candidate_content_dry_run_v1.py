#!/usr/bin/env python3
"""Build the zero-cost candidate-level content aggregation dry-run v1."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
from decimal import Decimal, ROUND_UP
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_candidate_content_dry_run_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-candidate-content-dry-run-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_candidate_content_dry_run_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_candidate_content_dry_run_outputs.v1"
CANDIDATE_SCHEMA = "pivot_feedback_d0_candidate_content_dry_run.v1"
SENSITIVITY_SCHEMA = "pivot_feedback_d0_candidate_content_sensitivity.v1"
STATUS = "PASS_ZERO_COST_CANDIDATE_CONTENT_DRY_RUN"
ROUTES = {
    "NO_API_RULE_ACCEPT",
    "NO_API_RULE_REJECT",
    "API_REQUIRED_AMBIGUOUS",
    "QUARANTINE_UNRESOLVED",
}


class CandidateContentDryRunBuildError(RuntimeError):
    """Fail-closed candidate-content dry-run error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateContentDryRunBuildError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise CandidateContentDryRunBuildError(f"invalid YAML: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateContentDryRunBuildError(f"invalid JSON: {error}") from error
    return _mapping(value, path.name)


def _ssd_root(value: Any) -> Path:
    require(
        isinstance(value, str)
        and value
        and "/" not in value
        and value not in {".", ".."},
        "unsafe SSD root basename",
    )
    path = (Path("/ssd") / value).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "SSD root escapes /ssd")
    require(path.is_dir() and not path.is_symlink(), "invalid SSD root")
    return path


def _bound_file(binding: Mapping[str, Any], key: str = "basename") -> Path:
    root = _ssd_root(binding["root_basename"])
    path = root / str(binding[key])
    require(path.is_file() and not path.is_symlink(), f"invalid input: {path.name}")
    require(_sha256(path) == binding["sha256"], f"input SHA: {path.name}")
    return path


def _validate_config(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status")
        == "FROZEN_ZERO_COST_CANDIDATE_CONTENT_DRY_RUN_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "parent escapes repository")
    require(_sha256(parent_path) == parent["sha256"], "parent SHA")
    parent_document = _read_yaml(parent_path)
    require(
        parent_document.get("schema_version") == parent["required_schema_version"],
        "parent schema",
    )
    require(
        parent_document.get("status") == parent["required_status"],
        "parent status",
    )
    rule = _mapping(config.get("candidate_rule"), "candidate rule")
    require(rule["recent_window_codes"] == [1, 2], "recent windows")
    require(rule["recent_title_rows"] == 64, "recent title rows")
    require(rule["rule_is_semantic_ground_truth"] is False, "ground-truth claim")
    require(
        rule["dry_run_state_is_production_disposition"] is False,
        "dry-run production disposition",
    )
    boundary = _mapping(config.get("execution_boundary"), "execution boundary")
    require(all(value is False for value in boundary.values()), "execution opened")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "existing_384_title_candidate_dry_run",
        "sensitivity_report",
        "counterfactual_frontier_cost_estimate",
    ):
        require(authorization.get(key) == "AUTHORIZED", f"safe work closed: {key}")
    for key in (
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "production_content_dispositions",
        "selection_certificate",
        "deterministic_reselection",
        "author_panel_rebuild",
        "pivot_classification",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(
            authorization.get(key) == "NOT_AUTHORIZED",
            f"unsafe authorization: {key}",
        )
    return config, {"parent_sha256": _sha256(parent_path)}


def _load_inputs(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], pa.Table, dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    sample_binding = _mapping(frozen.get("title_sample"), "title sample")
    sample_path = _bound_file(sample_binding)
    sample_parquet = pq.ParquetFile(sample_path)
    require(sample_parquet.metadata.num_rows == sample_binding["rows"], "sample rows")
    metadata = sample_parquet.schema_arrow.metadata or {}
    require(
        metadata.get(b"schema_version", b"").decode()
        == sample_binding["schema_version"],
        "sample schema",
    )
    sample_rows = sample_parquet.read().to_pylist()
    sample_receipt = _ssd_root(sample_binding["root_basename"]) / str(
        sample_binding["receipt_basename"]
    )
    require(
        _sha256(sample_receipt) == sample_binding["receipt_sha256"],
        "sample receipt SHA",
    )

    route_binding = _mapping(frozen.get("local_title_routes"), "title routes")
    route_path = _bound_file(route_binding)
    with route_path.open(newline="", encoding="utf-8") as handle:
        route_rows = list(csv.DictReader(handle))
    require(len(route_rows) == route_binding["rows"], "route rows")
    require(
        route_binding["prior_api_columns_used_for_rule"] is False,
        "API columns opened",
    )
    route_receipt = _ssd_root(route_binding["root_basename"]) / str(
        route_binding["receipt_basename"]
    )
    require(
        _sha256(route_receipt) == route_binding["receipt_sha256"],
        "route receipt SHA",
    )

    scope_binding = _mapping(
        frozen.get("candidate_window_scope"),
        "candidate scope",
    )
    scope_path = _bound_file(scope_binding)
    scope_parquet = pq.ParquetFile(scope_path)
    require(scope_parquet.metadata.num_rows == scope_binding["rows"], "scope rows")
    scope_metadata = scope_parquet.schema_arrow.metadata or {}
    require(
        scope_metadata.get(b"schema_version", b"").decode()
        == scope_binding["schema_version"],
        "scope schema",
    )
    scope_table = scope_parquet.read()
    scope_receipt = _ssd_root(scope_binding["root_basename"]) / str(
        scope_binding["receipt_basename"]
    )
    require(
        _sha256(scope_receipt) == scope_binding["receipt_sha256"],
        "scope receipt SHA",
    )

    candidate_binding = _mapping(frozen.get("candidate_table"), "candidate table")
    candidate_path = _bound_file(candidate_binding)
    require(
        pq.ParquetFile(candidate_path).metadata.num_rows == candidate_binding["rows"],
        "candidate rows",
    )
    return sample_rows, route_rows, scope_table, {
        "sample_sha256": _sha256(sample_path),
        "sample_receipt_sha256": _sha256(sample_receipt),
        "route_sha256": _sha256(route_path),
        "route_receipt_sha256": _sha256(route_receipt),
        "scope_sha256": _sha256(scope_path),
        "scope_receipt_sha256": _sha256(scope_receipt),
        "candidate_sha256": _sha256(candidate_path),
    }


def classify_candidate(
    *,
    recent_accept_fraction: float,
    recent_reject_fraction: float,
    score_drop: float,
    source_share_increase: float,
    accept_threshold: float,
    semantic_reject_threshold: float,
    score_drop_threshold: float,
    source_share_threshold: float,
) -> tuple[str, str]:
    indexing_anomaly = (
        score_drop >= score_drop_threshold
        and source_share_increase >= source_share_threshold
    )
    if indexing_anomaly:
        return (
            "WOULD_CONTENT_FAIL_INDEXING_ANOMALY",
            "JOINT_SCORE_DROP_AND_SOURCE_CONCENTRATION_SHIFT",
        )
    if recent_reject_fraction >= semantic_reject_threshold:
        return (
            "WOULD_CONTENT_FAIL_SEMANTIC_OFF_TOPIC",
            "RECENT_RULE_REJECT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD",
        )
    if recent_accept_fraction >= accept_threshold:
        return (
            "WOULD_CONTENT_PASS",
            "RECENT_RULE_ACCEPT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD",
        )
    return (
        "WOULD_CONTENT_UNRESOLVED",
        "CONSERVATIVE_RECENT_EVIDENCE_NOT_DECISIVE",
    )


def _dominant_source_share(rows: Sequence[Mapping[str, Any]]) -> float:
    counts = Counter(
        "<MISSING_SOURCE>" if row["source_name"] is None else str(row["source_name"])
        for row in rows
    )
    require(bool(counts), "empty source group")
    return max(counts.values()) / len(rows)


def _aggregate(
    sample_rows: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    route_by_id = {row["openalex_id"]: row for row in route_rows}
    require(len(route_by_id) == len(route_rows) == 384, "route IDs")
    require(
        {str(row["openalex_id"]) for row in sample_rows} == set(route_by_id),
        "sample/route ID set",
    )
    joined: list[dict[str, Any]] = []
    for sample in sample_rows:
        route = route_by_id[str(sample["openalex_id"])]
        require(int(route["topic_id"]) == int(sample["topic_id"]), "topic join")
        require(int(route["window_code"]) == int(sample["window_code"]), "window join")
        require(route["local_route"] in ROUTES, "unknown local route")
        joined.append({**sample, "local_route": route["local_route"], "local_reason": route["local_reason"]})

    rule = _mapping(config.get("candidate_rule"), "candidate rule")
    indexing = _mapping(rule.get("indexing_anomaly"), "indexing anomaly")
    primary_accept = float(rule["minimum_recent_rule_accept_lower_bound_fraction"])
    semantic_reject = float(
        rule["minimum_recent_rule_reject_lower_bound_fraction_for_semantic_fail"]
    )
    score_threshold = float(
        indexing["minimum_baseline_to_signal_median_primary_topic_score_drop"]
    )
    source_threshold = float(
        indexing["minimum_baseline_to_signal_dominant_source_share_increase"]
    )
    candidates: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    topic_ids = sorted({int(row["topic_id"]) for row in joined})
    require(topic_ids == [10579, 12421, 12505, 14430], "topic set")
    for topic_id in topic_ids:
        topic_rows = [row for row in joined if int(row["topic_id"]) == topic_id]
        require(len(topic_rows) == 96, f"topic rows: {topic_id}")
        windows = {
            code: [row for row in topic_rows if int(row["window_code"]) == code]
            for code in (0, 1, 2)
        }
        require(all(len(rows) == 32 for rows in windows.values()), "window rows")
        recent = windows[1] + windows[2]
        recent_counts = Counter(str(row["local_route"]) for row in recent)
        accept_fraction = recent_counts["NO_API_RULE_ACCEPT"] / len(recent)
        reject_fraction = recent_counts["NO_API_RULE_REJECT"] / len(recent)
        unknown_rows = (
            recent_counts["API_REQUIRED_AMBIGUOUS"]
            + recent_counts["QUARANTINE_UNRESOLVED"]
        )
        baseline_scores = [
            float(row["primary_topic_score"])
            for row in windows[1]
            if row["primary_topic_score"] is not None
        ]
        signal_scores = [
            float(row["primary_topic_score"])
            for row in windows[2]
            if row["primary_topic_score"] is not None
        ]
        require(len(baseline_scores) == len(signal_scores) == 32, "topic scores")
        baseline_median = float(statistics.median(baseline_scores))
        signal_median = float(statistics.median(signal_scores))
        baseline_source_share = _dominant_source_share(windows[1])
        signal_source_share = _dominant_source_share(windows[2])
        score_drop = baseline_median - signal_median
        source_increase = signal_source_share - baseline_source_share
        dry_state, reason = classify_candidate(
            recent_accept_fraction=accept_fraction,
            recent_reject_fraction=reject_fraction,
            score_drop=score_drop,
            source_share_increase=source_increase,
            accept_threshold=primary_accept,
            semantic_reject_threshold=semantic_reject,
            score_drop_threshold=score_threshold,
            source_share_threshold=source_threshold,
        )
        first = topic_rows[0]
        opaque_signal = sum(
            row["local_reason"] == "OPAQUE_ARCHIVAL_IDENTIFIER"
            for row in windows[2]
        )
        candidate = {
            "episode_index": int(first["episode_index"]),
            "field_id": int(first["field_id"]),
            "field_candidate_rank": 0,
            "topic_id": topic_id,
            "topic_name": str(route_by_id[str(first["openalex_id"])]["target_topic_name"]),
            "entry_year": int(first["entry_year"]),
            "earlier_accept_rows": sum(
                row["local_route"] == "NO_API_RULE_ACCEPT" for row in windows[0]
            ),
            "baseline_accept_rows": sum(
                row["local_route"] == "NO_API_RULE_ACCEPT" for row in windows[1]
            ),
            "signal_accept_rows": sum(
                row["local_route"] == "NO_API_RULE_ACCEPT" for row in windows[2]
            ),
            "recent_accept_rows": recent_counts["NO_API_RULE_ACCEPT"],
            "recent_reject_rows": recent_counts["NO_API_RULE_REJECT"],
            "recent_unknown_rows": unknown_rows,
            "recent_accept_lower_bound_fraction": accept_fraction,
            "recent_reject_lower_bound_fraction": reject_fraction,
            "baseline_median_primary_topic_score": baseline_median,
            "signal_median_primary_topic_score": signal_median,
            "baseline_to_signal_score_drop": score_drop,
            "baseline_dominant_source_share": baseline_source_share,
            "signal_dominant_source_share": signal_source_share,
            "baseline_to_signal_source_share_increase": source_increase,
            "signal_opaque_title_rows": opaque_signal,
            "indexing_anomaly_flag": (
                score_drop >= score_threshold
                and source_increase >= source_threshold
            ),
            "dry_run_state": dry_state,
            "dry_run_reason": reason,
            "production_content_state": "CONTENT_UNRESOLVED",
            "production_disposition_created": False,
        }
        candidates.append(candidate)
        for threshold in rule["sensitivity_accept_thresholds"]:
            state, _ = classify_candidate(
                recent_accept_fraction=accept_fraction,
                recent_reject_fraction=reject_fraction,
                score_drop=score_drop,
                source_share_increase=source_increase,
                accept_threshold=float(threshold),
                semantic_reject_threshold=semantic_reject,
                score_drop_threshold=score_threshold,
                source_share_threshold=source_threshold,
            )
            sensitivity.append(
                {
                    "field_id": int(first["field_id"]),
                    "topic_id": topic_id,
                    "entry_year": int(first["entry_year"]),
                    "accept_threshold": float(threshold),
                    "dry_run_state": state,
                    "production_disposition_created": False,
                }
            )
    candidates.sort(key=lambda row: int(row["episode_index"]))
    rank_by_key = {
        (17, 14430, 2003): 1,
        (17, 10579, 2012): 45,
        (27, 12505, 1991): 1,
        (27, 12421, 1953): 6,
    }
    for row in candidates:
        key = (int(row["field_id"]), int(row["topic_id"]), int(row["entry_year"]))
        require(key in rank_by_key, "candidate key")
        row["field_candidate_rank"] = rank_by_key[key]
    return candidates, sensitivity


def _candidate_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    fields = [
        pa.field("episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("field_candidate_rank", pa.uint32(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("topic_name", pa.string(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
    ]
    fields.extend(
        pa.field(name, pa.uint16(), nullable=False)
        for name in (
            "earlier_accept_rows",
            "baseline_accept_rows",
            "signal_accept_rows",
            "recent_accept_rows",
            "recent_reject_rows",
            "recent_unknown_rows",
        )
    )
    fields.extend(
        pa.field(name, pa.float64(), nullable=False)
        for name in (
            "recent_accept_lower_bound_fraction",
            "recent_reject_lower_bound_fraction",
            "baseline_median_primary_topic_score",
            "signal_median_primary_topic_score",
            "baseline_to_signal_score_drop",
            "baseline_dominant_source_share",
            "signal_dominant_source_share",
            "baseline_to_signal_source_share_increase",
        )
    )
    fields.extend(
        [
            pa.field("signal_opaque_title_rows", pa.uint16(), nullable=False),
            pa.field("indexing_anomaly_flag", pa.bool_(), nullable=False),
            pa.field("dry_run_state", pa.string(), nullable=False),
            pa.field("dry_run_reason", pa.string(), nullable=False),
            pa.field("production_content_state", pa.string(), nullable=False),
            pa.field("production_disposition_created", pa.bool_(), nullable=False),
        ]
    )
    schema = pa.schema(
        fields,
        metadata={b"schema_version": CANDIDATE_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(list(rows), schema=schema)


def _sensitivity_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("accept_threshold", pa.float64(), nullable=False),
            pa.field("dry_run_state", pa.string(), nullable=False),
            pa.field("production_disposition_created", pa.bool_(), nullable=False),
        ],
        metadata={b"schema_version": SENSITIVITY_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(list(rows), schema=schema)


def _closure(
    candidates: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    scope_table: pa.Table,
) -> tuple[dict[str, Any], dict[str, Any]]:
    counts = Counter(str(row["dry_run_state"]) for row in candidates)
    sensitivity_counts: dict[str, dict[str, int]] = {}
    for threshold in config["candidate_rule"]["sensitivity_accept_thresholds"]:
        selected = [
            row for row in sensitivity if float(row["accept_threshold"]) == float(threshold)
        ]
        states = Counter(str(row["dry_run_state"]) for row in selected)
        sensitivity_counts[f"{float(threshold):.2f}"] = {
            "would_pass": states["WOULD_CONTENT_PASS"],
            "would_fail_indexing": states["WOULD_CONTENT_FAIL_INDEXING_ANOMALY"],
            "would_unresolved": states["WOULD_CONTENT_UNRESOLVED"],
        }
    closure = {
        "candidate_rows": len(candidates),
        "would_content_pass_rows": counts["WOULD_CONTENT_PASS"],
        "would_content_fail_indexing_anomaly_rows": counts[
            "WOULD_CONTENT_FAIL_INDEXING_ANOMALY"
        ],
        "would_content_fail_semantic_off_topic_rows": counts[
            "WOULD_CONTENT_FAIL_SEMANTIC_OFF_TOPIC"
        ],
        "would_content_unresolved_rows": counts["WOULD_CONTENT_UNRESOLVED"],
        "indexing_anomaly_topic_ids": sorted(
            int(row["topic_id"])
            for row in candidates
            if row["indexing_anomaly_flag"]
        ),
        "production_content_dispositions_created": 0,
        "production_content_required_rows_resolved": 0,
        "selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
        "sensitivity": sensitivity_counts,
    }
    require(closure == config["expected_dry_run"], "dry-run closure")

    estimate_config = _mapping(
        config.get("counterfactual_frontier_estimate"),
        "frontier estimate",
    )
    target = _mapping(
        estimate_config["if_topic_12505_were_frozen_content_fail"],
        "counterfactual target",
    )
    mask = pc.and_(
        pc.equal(scope_table["field_id"], target["next_required_field_id"]),
        pc.equal(
            scope_table["field_candidate_rank"],
            target["next_required_field_candidate_rank"],
        ),
    )
    target_scope = scope_table.filter(mask)
    memberships = int(pc.sum(target_scope["author_linked_works"]).as_py())
    require(
        memberships == target["recent_six_year_author_linked_work_memberships"],
        "counterfactual membership count",
    )
    cost = _mapping(
        estimate_config["historical_pro_rata_api_cost_reference"],
        "cost reference",
    )
    unit = Decimal(str(cost["successful_openai_and_claude_cost_usd_for_384_titles"])) / Decimal(384)
    minimum = int(cost["observed_ambiguous_rows_per_96_topic_sample_minimum"])
    maximum = int(cost["observed_ambiguous_rows_per_96_topic_sample_maximum"])
    quant = Decimal("0.000001")
    computed_minimum = str((unit * minimum).quantize(quant, rounding=ROUND_UP))
    computed_maximum = str((unit * maximum).quantize(quant, rounding=ROUND_UP))
    require(
        computed_minimum == f"{float(cost['projected_two_provider_cost_usd_minimum']):.6f}",
        "minimum cost estimate",
    )
    require(
        computed_maximum == f"{float(cost['projected_two_provider_cost_usd_maximum']):.6f}",
        "maximum cost estimate",
    )
    estimate = {
        "counterfactual_only": True,
        "dry_run_states_applied_as_production_states": False,
        "next_required_field_id": int(target["next_required_field_id"]),
        "next_required_field_candidate_rank": int(
            target["next_required_field_candidate_rank"]
        ),
        "next_required_topic_id": int(target["next_required_topic_id"]),
        "next_required_entry_year": int(target["next_required_entry_year"]),
        "frozen_topic_card_available": bool(target["frozen_topic_card_available"]),
        "recent_six_year_author_linked_work_memberships": memberships,
        "target_title_sample_rows": int(target["target_title_sample_rows"]),
        "observed_ambiguous_rows_per_96_topic_sample_range": [minimum, maximum],
        "historical_pro_rata_two_provider_cost_usd_range": [
            computed_minimum,
            computed_maximum,
        ],
        "reference_is_current_price_quote": False,
        "raw_title_scan_authorized": False,
        "new_api_request_creation_authorized": False,
        "api_submission_authorized": False,
    }
    return closure, estimate


def _write_private(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    path.chmod(0o600)
    return {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": _sha256(path),
    }


def _markdown(
    candidates: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> str:
    lines = [
        "# Candidate-content zero-cost dry-run v1",
        "",
        f"Status: `{STATUS}`",
        "",
        "| Candidate | Recent accept lower bound | Score drop | Source-share increase | Dry-run state |",
        "|---|---:|---:|---:|---|",
    ]
    for row in candidates:
        lines.append(
            "| F{field}/T{topic}/{year} | {accept:.4f} | {score:.4f} | "
            "{source:.4f} | `{state}` |".format(
                field=row["field_id"],
                topic=row["topic_id"],
                year=row["entry_year"],
                accept=row["recent_accept_lower_bound_fraction"],
                score=row["baseline_to_signal_score_drop"],
                source=row["baseline_to_signal_source_share_increase"],
                state=row["dry_run_state"],
            )
        )
    lines.extend(
        [
            "",
            "All four production states remain `CONTENT_UNRESOLVED`. The dry-run",
            "created no production disposition and issued no selection certificate.",
            "",
            "At accept thresholds 0.50 and 0.60, three candidates would pass and",
            "T12505 would fail the joint indexing-anomaly rule. At 0.70, T12421",
            "would become unresolved.",
            "",
            "If T12505 were later frozen as CONTENT_FAIL, the next required local",
            f"sample would be rank {estimate['next_required_field_candidate_rank']} "
            f"T{estimate['next_required_topic_id']}, with "
            f"{estimate['recent_six_year_author_linked_work_memberships']:,} recent "
            "work memberships and a 96-title target. This is a counterfactual cost",
            "estimate only; neither a raw scan nor an API request is authorized.",
            "",
            f"Closure: {json.dumps(dict(closure), sort_keys=True)}",
            "",
        ]
    )
    return "\n".join(lines)


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(config_path.is_relative_to(REPO_ROOT), "config escapes repository")
    require(not output_root.exists(), "output root exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config, bindings = _validate_config(config_path)
    sample, routes, scope, inputs = _load_inputs(config)
    candidates, sensitivity = _aggregate(sample, routes, config)
    closure, estimate = _closure(candidates, sensitivity, config, scope)
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["candidate"] = _write_parquet(
            _candidate_table(candidates),
            temporary / str(contract["candidate_basename"]),
        )
        artifacts["sensitivity"] = _write_parquet(
            _sensitivity_table(sensitivity),
            temporary / str(contract["sensitivity_basename"]),
        )
        estimate_path = temporary / str(contract["estimate_basename"])
        _write_private(estimate_path, _canonical_json_bytes(estimate))
        artifacts["estimate"] = {
            "basename": estimate_path.name,
            "bytes": estimate_path.stat().st_size,
            "sha256": _sha256(estimate_path),
        }
        markdown_path = temporary / str(contract["markdown_basename"])
        _write_private(
            markdown_path,
            _markdown(candidates, closure, estimate).encode("utf-8"),
        )
        artifacts["markdown"] = {
            "basename": markdown_path.name,
            "bytes": markdown_path.stat().st_size,
            "sha256": _sha256(markdown_path),
        }
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": _sha256(config_path),
            "producer_relative_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "repository_bindings": bindings,
            "input_evidence": inputs,
            "closure": closure,
            "counterfactual_frontier_estimate": estimate,
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "raw_openalex_main_table_read": False,
            "new_titles_read": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "production_content_dispositions_created": 0,
            "production_content_required_rows_resolved": 0,
            "selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        receipt_path = temporary / str(contract["receipt_basename"])
        _write_private(receipt_path, _canonical_json_bytes(receipt))
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(args.config, args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        CandidateContentDryRunBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_CANDIDATE_CONTENT_DRY_RUN_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
