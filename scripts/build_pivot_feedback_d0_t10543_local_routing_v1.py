#!/usr/bin/env python3
"""Build zero-cost T10543 local routing and candidate-v1 aggregation."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_t10543_local_routing_v1.yaml"
)
DEFAULT_SAMPLE_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10543-targeted-title-sample-v1-20260729"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10543-local-routing-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_t10543_local_routing_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t10543_local_routing_outputs.v1"
STATUS = "PASS_ZERO_COST_T10543_LOCAL_ROUTING_AND_CANDIDATE_DRY_RUN"
ROUTES = [
    "NO_API_RULE_ACCEPT",
    "NO_API_RULE_REJECT",
    "API_REQUIRED_AMBIGUOUS",
    "QUARANTINE_UNRESOLVED",
]
ROUTE_COLUMNS = [
    "source_row_number",
    "openalex_id",
    "window_code",
    "publication_year",
    "title",
    "normalized_title",
    "local_route",
    "local_reason",
    "positive_match_count",
    "negative_match_count",
    "positive_matches",
    "negative_matches",
    "primary_topic_score",
    "source_name",
]


class T10543LocalRoutingBuildError(RuntimeError):
    """Fail-closed T10543 local-routing build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T10543LocalRoutingBuildError(message)


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
        raise T10543LocalRoutingBuildError(f"invalid YAML: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise T10543LocalRoutingBuildError(f"invalid JSON: {error}") from error
    return _mapping(value, path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _dotted(document: Mapping[str, Any], key: str) -> Any:
    value: Any = document
    for part in key.split("."):
        value = _mapping(value, part)[part]
    return value


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status") == "FROZEN_T10543_LOCAL_ROUTING_DRY_RUN_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    routing = _mapping(config.get("routing_contract"), "routing")
    require(routing.get("states") == ROUTES, "routing states")
    for key in (
        "primary_topic_score_used_for_semantic_acceptance",
        "source_name_used_for_semantic_acceptance",
        "api_labels_used_for_rule_execution",
        "unresolved_may_be_forced_to_decisive_label",
    ):
        require(routing.get(key) is False, f"unsafe routing: {key}")
    card = _mapping(config.get("topic_card"), "topic card")
    require(card.get("topic_id") == 10543, "topic card ID")
    for key in ("positive_patterns", "negative_patterns"):
        patterns = card.get(key)
        require(isinstance(patterns, list) and patterns, f"topic card {key}")
        for pattern in patterns:
            try:
                re.compile(str(pattern))
            except re.error as error:
                raise T10543LocalRoutingBuildError(
                    f"invalid topic-card regex: {error}"
                ) from error
    rule = _mapping(config.get("candidate_rule_v1"), "candidate rule")
    provenance = _mapping(rule.get("provenance"), "rule provenance")
    frozen_rule = _read_yaml(_repo_file(provenance))
    source_rule = _mapping(frozen_rule.get("candidate_rule"), "source rule")
    for key in (
        "recent_window_codes",
        "recent_title_rows",
        "minimum_recent_rule_accept_lower_bound_fraction",
        "minimum_recent_rule_reject_lower_bound_fraction_for_semantic_fail",
        "unresolved_and_quarantine_count_as_accept",
        "unresolved_and_quarantine_count_as_reject",
        "indexing_anomaly",
        "sensitivity_accept_thresholds",
    ):
        require(rule[key] == source_rule[key], f"candidate rule changed: {key}")
    boundary = _mapping(config.get("execution_boundary"), "execution boundary")
    for key, value in boundary.items():
        if key == "sample_title_rows_read":
            require(value == 96, "sample read boundary")
        else:
            require(value is False, f"unsafe boundary: {key}")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "production_content_disposition",
        "candidate_rule_amendment",
        "selection_certificate",
        "deterministic_reselection",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config


def normalize_title(value: Any) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return " ".join(re.findall(r"[a-z0-9]+", text))


def route_title(
    title: Any,
    card: Mapping[str, Any],
    *,
    opaque_patterns: Sequence[re.Pattern[str]],
    minimum_tokens: int,
) -> dict[str, Any]:
    normalized = normalize_title(title)
    if len(normalized.split()) < minimum_tokens:
        return {
            "normalized_title": normalized,
            "local_route": "QUARANTINE_UNRESOLVED",
            "local_reason": "UNINFORMATIVE_TITLE",
            "positive_matches": [],
            "negative_matches": [],
        }
    if any(pattern.search(normalized) for pattern in opaque_patterns):
        return {
            "normalized_title": normalized,
            "local_route": "QUARANTINE_UNRESOLVED",
            "local_reason": "OPAQUE_ARCHIVAL_IDENTIFIER",
            "positive_matches": [],
            "negative_matches": [],
        }
    positive = [
        str(pattern)
        for pattern in card["positive_patterns"]
        if re.search(str(pattern), normalized)
    ]
    negative = [
        str(pattern)
        for pattern in card["negative_patterns"]
        if re.search(str(pattern), normalized)
    ]
    if positive and not negative:
        route = "NO_API_RULE_ACCEPT"
        reason = "POSITIVE_CARD_MATCH_WITHOUT_NEGATIVE"
    elif negative and not positive:
        route = "NO_API_RULE_REJECT"
        reason = "NEGATIVE_CARD_MATCH_WITHOUT_POSITIVE"
    else:
        route = "API_REQUIRED_AMBIGUOUS"
        reason = "MIXED_OR_UNMATCHED_TITLE"
    return {
        "normalized_title": normalized,
        "local_route": route,
        "local_reason": reason,
        "positive_matches": positive,
        "negative_matches": negative,
    }


def _load_sample(
    config: Mapping[str, Any],
    sample_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    binding = _mapping(config.get("frozen_input"), "frozen input")
    root = sample_root.resolve(strict=True)
    require(root.name == binding["root_basename"], "sample root basename")
    require(root.is_dir() and not root.is_symlink(), "sample root")
    sample_path = root / str(binding["sample_basename"])
    receipt_path = root / str(binding["receipt_basename"])
    verification_path = Path("/ssd") / str(binding["verification_basename"])
    require(_sha256(sample_path) == binding["sample_sha256"], "sample SHA")
    require(_sha256(receipt_path) == binding["receipt_sha256"], "sample receipt SHA")
    require(
        _sha256(verification_path) == binding["verification_sha256"],
        "sample verification SHA",
    )
    receipt = _read_json(receipt_path)
    verification = _read_json(verification_path)
    require(receipt.get("status") == binding["receipt_status"], "sample status")
    require(
        verification.get("status") == binding["verification_status"],
        "sample verification status",
    )
    parquet = pq.ParquetFile(sample_path)
    require(parquet.metadata.num_rows == binding["sample_rows"], "sample rows")
    require(
        (parquet.schema_arrow.metadata or {}).get(b"schema_version", b"").decode()
        == binding["sample_schema_version"],
        "sample schema",
    )
    rows = parquet.read().to_pylist()
    require(
        {(row["field_id"], row["topic_id"], row["entry_year"]) for row in rows}
        == {(27, 10543, 1993)},
        "sample target",
    )
    require(
        Counter(row["window_code"] for row in rows) == {0: 32, 1: 32, 2: 32},
        "sample windows",
    )
    return rows, {
        "sample_sha256": _sha256(sample_path),
        "sample_receipt_sha256": _sha256(receipt_path),
        "sample_verification_sha256": _sha256(verification_path),
    }


def _route_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    routing = _mapping(config.get("routing_contract"), "routing")
    card = _mapping(config.get("topic_card"), "topic card")
    opaque = [
        re.compile(str(pattern))
        for pattern in routing["global_opaque_title_patterns"]
    ]
    rows: list[dict[str, Any]] = []
    for source_row, sample in enumerate(sample_rows, start=1):
        decision = route_title(
            sample["title"],
            card,
            opaque_patterns=opaque,
            minimum_tokens=int(routing["minimum_informative_token_count"]),
        )
        rows.append(
            {
                "source_row_number": source_row,
                "openalex_id": str(sample["openalex_id"]),
                "window_code": int(sample["window_code"]),
                "publication_year": int(sample["publication_year"]),
                "title": "" if sample["title"] is None else str(sample["title"]),
                "normalized_title": decision["normalized_title"],
                "local_route": decision["local_route"],
                "local_reason": decision["local_reason"],
                "positive_match_count": len(decision["positive_matches"]),
                "negative_match_count": len(decision["negative_matches"]),
                "positive_matches": "|".join(decision["positive_matches"]),
                "negative_matches": "|".join(decision["negative_matches"]),
                "primary_topic_score": float(sample["primary_topic_score"]),
                "source_name": ""
                if sample["source_name"] is None
                else str(sample["source_name"]),
            }
        )
    return rows


def _dominant_source(rows: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    counts = Counter(
        "<MISSING_SOURCE>" if row["source_name"] is None else str(row["source_name"])
        for row in rows
    )
    source, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return source, count / len(rows)


def _classify(
    accept: float,
    reject: float,
    score_drop: float,
    source_increase: float,
    *,
    accept_threshold: float,
    reject_threshold: float,
    score_threshold: float,
    source_threshold: float,
) -> str:
    if score_drop >= score_threshold and source_increase >= source_threshold:
        return "WOULD_CONTENT_FAIL_INDEXING_ANOMALY"
    if reject >= reject_threshold:
        return "WOULD_CONTENT_FAIL_SEMANTIC_OFF_TOPIC"
    if accept >= accept_threshold:
        return "WOULD_CONTENT_PASS"
    return "WOULD_CONTENT_UNRESOLVED"


def _candidate_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route_by_id = {str(row["openalex_id"]): row for row in route_rows}
    require(len(route_by_id) == 96, "route IDs")
    combined = [{**row, **route_by_id[str(row["openalex_id"])]} for row in sample_rows]
    windows = {
        window: [row for row in combined if int(row["window_code"]) == window]
        for window in (0, 1, 2)
    }
    require(all(len(rows) == 32 for rows in windows.values()), "candidate windows")
    recent = windows[1] + windows[2]
    route_counts = Counter(str(row["local_route"]) for row in recent)
    recent_accept = route_counts["NO_API_RULE_ACCEPT"]
    recent_reject = route_counts["NO_API_RULE_REJECT"]
    recent_unknown = (
        route_counts["API_REQUIRED_AMBIGUOUS"]
        + route_counts["QUARANTINE_UNRESOLVED"]
    )
    accept_fraction = recent_accept / 64
    reject_fraction = recent_reject / 64
    baseline_score = float(
        statistics.median(float(row["primary_topic_score"]) for row in windows[1])
    )
    signal_score = float(
        statistics.median(float(row["primary_topic_score"]) for row in windows[2])
    )
    baseline_source, baseline_source_share = _dominant_source(windows[1])
    signal_source, signal_source_share = _dominant_source(windows[2])
    score_drop = baseline_score - signal_score
    source_increase = signal_source_share - baseline_source_share
    rule = _mapping(config.get("candidate_rule_v1"), "candidate rule")
    indexing = _mapping(rule.get("indexing_anomaly"), "indexing")
    state = _classify(
        accept_fraction,
        reject_fraction,
        score_drop,
        source_increase,
        accept_threshold=float(
            rule["minimum_recent_rule_accept_lower_bound_fraction"]
        ),
        reject_threshold=float(
            rule["minimum_recent_rule_reject_lower_bound_fraction_for_semantic_fail"]
        ),
        score_threshold=float(
            indexing["minimum_baseline_to_signal_median_primary_topic_score_drop"]
        ),
        source_threshold=float(
            indexing["minimum_baseline_to_signal_dominant_source_share_increase"]
        ),
    )
    reasons = {
        "WOULD_CONTENT_FAIL_INDEXING_ANOMALY": (
            "JOINT_SCORE_DROP_AND_SOURCE_CONCENTRATION_SHIFT"
        ),
        "WOULD_CONTENT_FAIL_SEMANTIC_OFF_TOPIC": (
            "RECENT_RULE_REJECT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
        ),
        "WOULD_CONTENT_PASS": (
            "RECENT_RULE_ACCEPT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
        ),
        "WOULD_CONTENT_UNRESOLVED": "CONSERVATIVE_RECENT_EVIDENCE_NOT_DECISIVE",
    }
    opaque = [
        row
        for row in recent
        if row["local_reason"] == "OPAQUE_ARCHIVAL_IDENTIFIER"
    ]
    opaque_source, opaque_source_share = _dominant_source(opaque)
    candidate = {
        "field_id": 27,
        "field_candidate_rank": 2,
        "topic_id": 10543,
        "topic_name": "Prostate Cancer Treatment and Research",
        "entry_year": 1993,
        "earlier_accept_rows": sum(
            row["local_route"] == "NO_API_RULE_ACCEPT" for row in windows[0]
        ),
        "baseline_accept_rows": sum(
            row["local_route"] == "NO_API_RULE_ACCEPT" for row in windows[1]
        ),
        "signal_accept_rows": sum(
            row["local_route"] == "NO_API_RULE_ACCEPT" for row in windows[2]
        ),
        "recent_accept_rows": recent_accept,
        "recent_reject_rows": recent_reject,
        "recent_ambiguous_rows": route_counts["API_REQUIRED_AMBIGUOUS"],
        "recent_quarantine_rows": route_counts["QUARANTINE_UNRESOLVED"],
        "recent_unknown_rows": recent_unknown,
        "recent_accept_lower_bound_fraction": accept_fraction,
        "recent_reject_lower_bound_fraction": reject_fraction,
        "baseline_median_primary_topic_score": baseline_score,
        "signal_median_primary_topic_score": signal_score,
        "baseline_to_signal_score_drop": score_drop,
        "baseline_dominant_source": baseline_source,
        "baseline_dominant_source_share": baseline_source_share,
        "signal_dominant_source": signal_source,
        "signal_dominant_source_share": signal_source_share,
        "baseline_to_signal_source_share_increase": source_increase,
        "signal_opaque_title_rows": sum(
            row["local_reason"] == "OPAQUE_ARCHIVAL_IDENTIFIER"
            for row in windows[2]
        ),
        "recent_opaque_title_rows": len(opaque),
        "recent_opaque_title_fraction": len(opaque) / 64,
        "recent_opaque_dominant_source": opaque_source,
        "recent_opaque_dominant_source_share": opaque_source_share,
        "recent_opaque_median_primary_topic_score": float(
            statistics.median(float(row["primary_topic_score"]) for row in opaque)
        ),
        "indexing_anomaly_v1_flag": (
            score_drop
            >= float(
                indexing["minimum_baseline_to_signal_median_primary_topic_score_drop"]
            )
            and source_increase
            >= float(
                indexing[
                    "minimum_baseline_to_signal_dominant_source_share_increase"
                ]
            )
        ),
        "dry_run_state": state,
        "dry_run_reason": reasons[state],
        "production_content_state": "CONTENT_UNRESOLVED",
        "production_disposition_created": False,
    }
    sensitivity = [
        {
            "field_id": 27,
            "topic_id": 10543,
            "entry_year": 1993,
            "accept_threshold": float(threshold),
            "dry_run_state": _classify(
                accept_fraction,
                reject_fraction,
                score_drop,
                source_increase,
                accept_threshold=float(threshold),
                reject_threshold=float(
                    rule[
                        "minimum_recent_rule_reject_lower_bound_fraction_for_semantic_fail"
                    ]
                ),
                score_threshold=float(
                    indexing[
                        "minimum_baseline_to_signal_median_primary_topic_score_drop"
                    ]
                ),
                source_threshold=float(
                    indexing[
                        "minimum_baseline_to_signal_dominant_source_share_increase"
                    ]
                ),
            ),
            "production_disposition_created": False,
        }
        for threshold in rule["sensitivity_accept_thresholds"]
    ]
    route_summary = {
        str(window): dict(
            Counter(str(row["local_route"]) for row in windows[window])
        )
        for window in (0, 1, 2)
    }
    return [candidate], sensitivity, route_summary


def _candidate_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("earlier_accept_rows", pa.uint8(), nullable=False),
            pa.field("baseline_accept_rows", pa.uint8(), nullable=False),
            pa.field("signal_accept_rows", pa.uint8(), nullable=False),
            pa.field("recent_accept_rows", pa.uint8(), nullable=False),
            pa.field("recent_reject_rows", pa.uint8(), nullable=False),
            pa.field("recent_ambiguous_rows", pa.uint8(), nullable=False),
            pa.field("recent_quarantine_rows", pa.uint8(), nullable=False),
            pa.field("recent_unknown_rows", pa.uint8(), nullable=False),
            pa.field("recent_accept_lower_bound_fraction", pa.float64(), nullable=False),
            pa.field("recent_reject_lower_bound_fraction", pa.float64(), nullable=False),
            pa.field("baseline_median_primary_topic_score", pa.float64(), nullable=False),
            pa.field("signal_median_primary_topic_score", pa.float64(), nullable=False),
            pa.field("baseline_to_signal_score_drop", pa.float64(), nullable=False),
            pa.field("baseline_dominant_source", pa.string(), nullable=False),
            pa.field("baseline_dominant_source_share", pa.float64(), nullable=False),
            pa.field("signal_dominant_source", pa.string(), nullable=False),
            pa.field("signal_dominant_source_share", pa.float64(), nullable=False),
            pa.field(
                "baseline_to_signal_source_share_increase",
                pa.float64(),
                nullable=False,
            ),
            pa.field("signal_opaque_title_rows", pa.uint8(), nullable=False),
            pa.field("recent_opaque_title_rows", pa.uint8(), nullable=False),
            pa.field("recent_opaque_title_fraction", pa.float64(), nullable=False),
            pa.field("recent_opaque_dominant_source", pa.string(), nullable=False),
            pa.field(
                "recent_opaque_dominant_source_share",
                pa.float64(),
                nullable=False,
            ),
            pa.field(
                "recent_opaque_median_primary_topic_score",
                pa.float64(),
                nullable=False,
            ),
            pa.field("indexing_anomaly_v1_flag", pa.bool_(), nullable=False),
            pa.field("dry_run_state", pa.string(), nullable=False),
            pa.field("dry_run_reason", pa.string(), nullable=False),
            pa.field("production_content_state", pa.string(), nullable=False),
            pa.field("production_disposition_created", pa.bool_(), nullable=False),
        ],
        metadata={
            b"schema_version": b"pivot_feedback_d0_t10543_candidate_content_dry_run.v1"
        },
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
        metadata={
            b"schema_version": b"pivot_feedback_d0_t10543_candidate_sensitivity.v1"
        },
    )
    return pa.Table.from_pylist(list(rows), schema=schema)


def _write_private(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUTE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)
    return {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "sha256": _sha256(path),
    }


def _write_parquet(path: Path, table: pa.Table) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    path.chmod(0o600)
    return {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": _sha256(path),
    }


def _markdown(candidate: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# T10543 local routing and candidate-v1 dry-run",
            "",
            f"Status: `{STATUS}`",
            "",
            f"- recent accept: {candidate['recent_accept_rows']}/64",
            f"- recent reject: {candidate['recent_reject_rows']}/64",
            f"- recent ambiguous: {candidate['recent_ambiguous_rows']}/64",
            f"- recent opaque quarantine: {candidate['recent_quarantine_rows']}/64",
            f"- candidate-rule-v1 state: `{candidate['dry_run_state']}`",
            f"- recent opaque share: {candidate['recent_opaque_title_fraction']:.6f}",
            (
                "- opaque dominant source: "
                f"{candidate['recent_opaque_dominant_source']} "
                f"({candidate['recent_opaque_dominant_source_share']:.6f})"
            ),
            (
                "- opaque median current-snapshot topic score: "
                f"{candidate['recent_opaque_median_primary_topic_score']:.6f}"
            ),
            "",
            "The frozen v1 rule remains unresolved because quarantine is neither",
            "accept nor reject and its joint baseline-to-signal anomaly test does",
            "not treat already-present baseline contamination as a decisive fail.",
            "Opaque identifiers were not sent to an API. No production disposition,",
            "certificate, reselection, outcome read, or model fit occurred.",
            "",
        ]
    )


def build(
    *,
    config_path: Path = DEFAULT_CONFIG,
    sample_root: Path = DEFAULT_SAMPLE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(not output_root.exists(), "output root exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config = _load_config(config_path)
    sample_rows, input_evidence = _load_sample(config, sample_root)
    route_rows = _route_rows(sample_rows, config)
    candidates, sensitivity, route_summary = _candidate_rows(
        sample_rows,
        route_rows,
        config,
    )
    candidate = candidates[0]
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["routes"] = _write_csv(
            temporary / str(contract["routes_basename"]),
            route_rows,
        )
        artifacts["candidate"] = _write_parquet(
            temporary / str(contract["candidate_basename"]),
            _candidate_table(candidates),
        )
        artifacts["sensitivity"] = _write_parquet(
            temporary / str(contract["sensitivity_basename"]),
            _sensitivity_table(sensitivity),
        )
        report_path = temporary / str(contract["report_basename"])
        _write_private(report_path, _markdown(candidate).encode())
        artifacts["report"] = {
            "basename": report_path.name,
            "bytes": report_path.stat().st_size,
            "sha256": _sha256(report_path),
        }
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": _sha256(config_path),
            "producer_relative_path": (
                Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
            ),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "input_evidence": input_evidence,
            "route_summary": route_summary,
            "candidate": candidate,
            "sensitivity": sensitivity,
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "sample_title_rows_read": 96,
            "raw_openalex_main_table_read": False,
            "new_titles_read": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "production_content_dispositions_created": 0,
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
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(
            config_path=args.config,
            sample_root=args.sample_root,
            output_root=args.output_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        T10543LocalRoutingBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_T10543_LOCAL_ROUTING_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
