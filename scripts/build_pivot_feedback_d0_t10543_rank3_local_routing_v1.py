#!/usr/bin/env python3
"""Build zero-cost rank-3 T10543 routing and rule-v2 dry-run."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import build_pivot_feedback_d0_t10543_local_routing_v1 as base


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10543_rank3_local_routing_v1.yaml"
)
DEFAULT_SAMPLE_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10543-rank3-targeted-title-sample-v1-20260729"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10543-rank3-local-routing-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_t10543_rank3_local_routing_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t10543_rank3_local_routing_outputs.v1"
STATUS = "PASS_ZERO_COST_T10543_RANK3_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN"


class T10543Rank3LocalRoutingBuildError(RuntimeError):
    """Fail-closed rank-3 local-routing build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T10543Rank3LocalRoutingBuildError(message)


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
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


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


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = _repo_file(parent)
    parent_document = _read_yaml(parent_path)
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    card_source = _mapping(
        _mapping(config.get("topic_card"), "topic card").get("source"),
        "card source",
    )
    card_path = _repo_file(card_source)
    card_config = _read_yaml(card_path)
    require(card_source.get("reuse_without_change") is True, "card reuse")
    rule_source = _mapping(
        _mapping(config.get("production_rule_v2"), "rule").get("source"),
        "rule source",
    )
    rule_path = _repo_file(rule_source)
    require(rule_source.get("reuse_without_change") is True, "rule reuse")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "production_content_disposition",
        "selection_certificate",
        "deterministic_reselection",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config, card_config, {
        "parent_sha256": _sha256(parent_path),
        "topic_card_config_sha256": _sha256(card_path),
        "production_rule_v2_config_sha256": _sha256(rule_path),
    }


def _load_sample(
    config: Mapping[str, Any],
    sample_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    binding = _mapping(config.get("frozen_input"), "frozen input")
    root = sample_root.resolve(strict=True)
    require(root.name == binding["root_basename"], "sample root")
    sample_path = root / str(binding["sample_basename"])
    receipt_path = root / str(binding["receipt_basename"])
    verification_path = Path("/ssd") / str(binding["verification_basename"])
    require(_sha256(sample_path) == binding["sample_sha256"], "sample SHA")
    require(_sha256(receipt_path) == binding["receipt_sha256"], "receipt SHA")
    require(
        _sha256(verification_path) == binding["verification_sha256"],
        "verification SHA",
    )
    require(_read_json(receipt_path).get("status") == binding["receipt_status"], "sample status")
    require(
        _read_json(verification_path).get("status") == binding["verification_status"],
        "verification status",
    )
    parquet = pq.ParquetFile(sample_path)
    require(parquet.metadata.num_rows == binding["sample_rows"], "sample rows")
    require(
        (parquet.schema_arrow.metadata or {}).get(b"schema_version", b"").decode()
        == binding["sample_schema_version"],
        "sample schema",
    )
    rows = parquet.read().to_pylist()
    return rows, {
        "sample_sha256": _sha256(sample_path),
        "sample_receipt_sha256": _sha256(receipt_path),
        "sample_verification_sha256": _sha256(verification_path),
    }


def _dominant_share(rows: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    counts = Counter(
        "<MISSING_SOURCE>" if row["source_name"] is None else str(row["source_name"])
        for row in rows
    )
    source, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return source, count / len(rows)


def _candidate(
    samples: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    route_by_id = {str(row["openalex_id"]): row for row in routes}
    rows = [{**row, **route_by_id[str(row["openalex_id"])]} for row in samples]
    windows = {
        window: [row for row in rows if int(row["window_code"]) == window]
        for window in (0, 1, 2)
    }
    recent = windows[1] + windows[2]
    counts = Counter(str(row["local_route"]) for row in recent)
    baseline_score = float(
        statistics.median(float(row["primary_topic_score"]) for row in windows[1])
    )
    signal_score = float(
        statistics.median(float(row["primary_topic_score"]) for row in windows[2])
    )
    baseline_source, baseline_share = _dominant_share(windows[1])
    signal_source, signal_share = _dominant_share(windows[2])
    opaque = sum(
        row["local_reason"] == "OPAQUE_ARCHIVAL_IDENTIFIER" for row in recent
    )
    opaque_fraction = opaque / 64
    score_drop = baseline_score - signal_score
    source_increase = signal_share - baseline_share
    rule = _mapping(config.get("production_rule_v2"), "rule")
    if opaque_fraction >= float(rule["majority_opaque_threshold"]):
        state = "WOULD_CONTENT_FAIL_INDEXING_ANOMALY"
        reason = "RECENT_SAMPLE_MAJORITY_OPAQUE_INDEXING_ARTIFACTS"
        branch = "RECENT_MAJORITY_OPAQUE_INDEXING_ANOMALY"
    elif (
        score_drop >= float(rule["joint_indexing_score_drop_threshold"])
        and source_increase
        >= float(rule["joint_indexing_source_share_increase_threshold"])
    ):
        state = "WOULD_CONTENT_FAIL_INDEXING_ANOMALY"
        reason = "JOINT_SCORE_DROP_AND_SOURCE_CONCENTRATION_SHIFT"
        branch = "JOINT_SCORE_AND_SOURCE_SHIFT_INDEXING_ANOMALY"
    elif counts["NO_API_RULE_REJECT"] / 64 >= float(rule["semantic_reject_threshold"]):
        state = "WOULD_CONTENT_FAIL_SEMANTIC_OFF_TOPIC"
        reason = "RECENT_RULE_REJECT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
        branch = "SEMANTIC_OFF_TOPIC"
    elif counts["NO_API_RULE_ACCEPT"] / 64 >= float(rule["semantic_accept_threshold"]):
        state = "WOULD_CONTENT_PASS"
        reason = "RECENT_RULE_ACCEPT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
        branch = "CONTENT_PASS"
    else:
        state = "WOULD_CONTENT_UNRESOLVED"
        reason = "CONSERVATIVE_RECENT_EVIDENCE_NOT_DECISIVE"
        branch = "CONTENT_UNRESOLVED"
    return {
        "field_id": 27,
        "field_candidate_rank": 3,
        "topic_id": 10543,
        "topic_name": "Prostate Cancer Treatment and Research",
        "entry_year": 1986,
        "recent_title_rows": 64,
        "recent_accept_rows": counts["NO_API_RULE_ACCEPT"],
        "recent_reject_rows": counts["NO_API_RULE_REJECT"],
        "recent_ambiguous_rows": counts["API_REQUIRED_AMBIGUOUS"],
        "recent_opaque_quarantine_rows": opaque,
        "recent_opaque_quarantine_fraction": opaque_fraction,
        "baseline_median_primary_topic_score": baseline_score,
        "signal_median_primary_topic_score": signal_score,
        "baseline_to_signal_score_drop": score_drop,
        "baseline_dominant_source": baseline_source,
        "baseline_dominant_source_share": baseline_share,
        "signal_dominant_source": signal_source,
        "signal_dominant_source_share": signal_share,
        "baseline_to_signal_source_share_increase": source_increase,
        "candidate_rule_v2_branch": branch,
        "candidate_rule_v2_state": state,
        "candidate_rule_v2_reason": reason,
        "production_content_state": "CONTENT_UNRESOLVED",
        "production_content_disposition_created": False,
    }


def _table(row: Mapping[str, Any]) -> pa.Table:
    fields: list[pa.Field] = []
    for key, value in row.items():
        if isinstance(value, bool):
            data_type = pa.bool_()
        elif isinstance(value, int):
            data_type = pa.int64()
        elif isinstance(value, float):
            data_type = pa.float64()
        else:
            data_type = pa.string()
        fields.append(pa.field(key, data_type, nullable=False))
    schema = pa.schema(
        fields,
        metadata={
            b"schema_version": b"pivot_feedback_d0_t10543_rank3_rule_v2_dry_run.v1"
        },
    )
    return pa.Table.from_pylist([dict(row)], schema=schema)


def _markdown(candidate: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# T10543 Medicine rank-3 local routing and rule-v2 dry-run",
            "",
            f"Status: `{STATUS}`",
            "",
            f"- recent accept: {candidate['recent_accept_rows']}/64",
            f"- recent reject: {candidate['recent_reject_rows']}/64",
            f"- recent ambiguous: {candidate['recent_ambiguous_rows']}/64",
            f"- recent opaque quarantine: {candidate['recent_opaque_quarantine_rows']}/64",
            f"- score drop: {candidate['baseline_to_signal_score_drop']:.6f}",
            f"- source-share increase: {candidate['baseline_to_signal_source_share_increase']:.6f}",
            f"- rule-v2 branch: `{candidate['candidate_rule_v2_branch']}`",
            f"- rule-v2 state: `{candidate['candidate_rule_v2_state']}`",
            "",
            "No API request, production disposition, certificate, reselection,",
            "outcome read, or model fit occurred.",
            "",
        ]
    )


def build(
    config_path: Path = DEFAULT_CONFIG,
    sample_root: Path = DEFAULT_SAMPLE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(not output_root.exists(), "output root exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config, card_config, repository_bindings = _load_config(config_path)
    samples, input_evidence = _load_sample(config, sample_root)
    routes = base._route_rows(samples, card_config)
    candidate = _candidate(samples, routes, config)
    expected = _mapping(config.get("expected_dry_run"), "expected")
    for key, value in expected.items():
        require(candidate[key] == value, f"expected dry-run: {key}")
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["routes"] = base._write_csv(
            temporary / str(contract["routes_basename"]),
            routes,
        )
        artifacts["candidate"] = base._write_parquet(
            temporary / str(contract["candidate_basename"]),
            _table(candidate),
        )
        report_path = temporary / str(contract["report_basename"])
        base._write_private(report_path, _markdown(candidate).encode())
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
            "repository_bindings": repository_bindings,
            "input_evidence": input_evidence,
            "candidate": candidate,
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
        base._write_private(receipt_path, _canonical_json_bytes(receipt))
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
        print(json.dumps(build(args.config, args.sample_root, args.output_root), indent=2, sort_keys=True))
        return 0
    except (
        T10543Rank3LocalRoutingBuildError,
        base.T10543LocalRoutingBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_T10543_RANK3_LOCAL_ROUTING_V1", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
