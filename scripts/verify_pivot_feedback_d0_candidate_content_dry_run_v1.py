#!/usr/bin/env python3
"""Independently verify the candidate-level content dry-run v1."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

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
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-candidate-content-dry-run-v1-20260729.verification-v1.json"
)
CONFIG_SCHEMA = "pivot_feedback_d0_candidate_content_dry_run_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_candidate_content_dry_run_outputs.v1"
VERIFY_SCHEMA = "pivot_feedback_d0_candidate_content_dry_run_verification.v1"
VERIFY_STATUS = "PASS_INDEPENDENT_ZERO_COST_CANDIDATE_CONTENT_DRY_RUN"


class CandidateContentDryRunVerificationError(RuntimeError):
    """Fail-closed candidate-content verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateContentDryRunVerificationError(message)


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
        raise CandidateContentDryRunVerificationError(
            f"invalid YAML: {error}"
        ) from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateContentDryRunVerificationError(
            f"invalid JSON: {error}"
        ) from error
    return _mapping(value, path.name)


def _ssd_root(basename: Any) -> Path:
    require(
        isinstance(basename, str)
        and basename
        and "/" not in basename
        and basename not in {".", ".."},
        "unsafe SSD root basename",
    )
    path = (Path("/ssd") / basename).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "SSD root escapes /ssd")
    require(path.is_dir() and not path.is_symlink(), "invalid SSD root")
    return path


def _input_path(binding: Mapping[str, Any], key: str = "basename") -> Path:
    path = _ssd_root(binding["root_basename"]) / str(binding[key])
    require(path.is_file() and not path.is_symlink(), "invalid frozen input")
    require(_sha256(path) == binding["sha256"], f"input SHA: {path.name}")
    return path


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


def _source_share(rows: Sequence[Mapping[str, Any]]) -> float:
    counts = Counter(
        "<MISSING_SOURCE>" if row["source_name"] is None else str(row["source_name"])
        for row in rows
    )
    return max(counts.values()) / len(rows)


def _independent_rows(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    sample_binding = _mapping(frozen.get("title_sample"), "sample")
    sample_path = _input_path(sample_binding)
    sample_file = pq.ParquetFile(sample_path)
    require(sample_file.metadata.num_rows == 384, "sample rows")
    sample_rows = sample_file.read().to_pylist()
    route_binding = _mapping(frozen.get("local_title_routes"), "routes")
    route_path = _input_path(route_binding)
    with route_path.open(newline="", encoding="utf-8") as handle:
        route_rows = list(csv.DictReader(handle))
    require(len(route_rows) == 384, "route rows")
    route_by_id = {row["openalex_id"]: row for row in route_rows}
    require(len(route_by_id) == 384, "route IDs")

    rule = _mapping(config.get("candidate_rule"), "rule")
    indexing = _mapping(rule.get("indexing_anomaly"), "indexing")
    accept_threshold = float(
        rule["minimum_recent_rule_accept_lower_bound_fraction"]
    )
    reject_threshold = float(
        rule["minimum_recent_rule_reject_lower_bound_fraction_for_semantic_fail"]
    )
    score_threshold = float(
        indexing["minimum_baseline_to_signal_median_primary_topic_score_drop"]
    )
    source_threshold = float(
        indexing["minimum_baseline_to_signal_dominant_source_share_increase"]
    )
    rank = {
        (17, 14430, 2003): 1,
        (17, 10579, 2012): 45,
        (27, 12505, 1991): 1,
        (27, 12421, 1953): 6,
    }
    output: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for topic_id in (10579, 12421, 12505, 14430):
        topic = [
            {**row, **{"local_route": route_by_id[str(row["openalex_id"])]["local_route"],
                      "local_reason": route_by_id[str(row["openalex_id"])]["local_reason"]}}
            for row in sample_rows
            if int(row["topic_id"]) == topic_id
        ]
        require(len(topic) == 96, f"topic rows: {topic_id}")
        windows = {
            code: [row for row in topic if int(row["window_code"]) == code]
            for code in (0, 1, 2)
        }
        require(all(len(rows) == 32 for rows in windows.values()), "window rows")
        recent = windows[1] + windows[2]
        routes = Counter(str(row["local_route"]) for row in recent)
        accept = routes["NO_API_RULE_ACCEPT"] / 64
        reject = routes["NO_API_RULE_REJECT"] / 64
        unknown = (
            routes["API_REQUIRED_AMBIGUOUS"] + routes["QUARANTINE_UNRESOLVED"]
        )
        baseline_median = float(
            statistics.median(float(row["primary_topic_score"]) for row in windows[1])
        )
        signal_median = float(
            statistics.median(float(row["primary_topic_score"]) for row in windows[2])
        )
        baseline_source = _source_share(windows[1])
        signal_source = _source_share(windows[2])
        score_drop = baseline_median - signal_median
        source_increase = signal_source - baseline_source
        state = _classify(
            accept,
            reject,
            score_drop,
            source_increase,
            accept_threshold=accept_threshold,
            reject_threshold=reject_threshold,
            score_threshold=score_threshold,
            source_threshold=source_threshold,
        )
        first = topic[0]
        field_id = int(first["field_id"])
        entry_year = int(first["entry_year"])
        reason = {
            "WOULD_CONTENT_FAIL_INDEXING_ANOMALY": (
                "JOINT_SCORE_DROP_AND_SOURCE_CONCENTRATION_SHIFT"
            ),
            "WOULD_CONTENT_FAIL_SEMANTIC_OFF_TOPIC": (
                "RECENT_RULE_REJECT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
            ),
            "WOULD_CONTENT_PASS": (
                "RECENT_RULE_ACCEPT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
            ),
            "WOULD_CONTENT_UNRESOLVED": (
                "CONSERVATIVE_RECENT_EVIDENCE_NOT_DECISIVE"
            ),
        }[state]
        output.append(
            {
                "episode_index": int(first["episode_index"]),
                "field_id": field_id,
                "field_candidate_rank": rank[(field_id, topic_id, entry_year)],
                "topic_id": topic_id,
                "topic_name": route_by_id[str(first["openalex_id"])][
                    "target_topic_name"
                ],
                "entry_year": entry_year,
                "earlier_accept_rows": sum(
                    row["local_route"] == "NO_API_RULE_ACCEPT"
                    for row in windows[0]
                ),
                "baseline_accept_rows": sum(
                    row["local_route"] == "NO_API_RULE_ACCEPT"
                    for row in windows[1]
                ),
                "signal_accept_rows": sum(
                    row["local_route"] == "NO_API_RULE_ACCEPT"
                    for row in windows[2]
                ),
                "recent_accept_rows": routes["NO_API_RULE_ACCEPT"],
                "recent_reject_rows": routes["NO_API_RULE_REJECT"],
                "recent_unknown_rows": unknown,
                "recent_accept_lower_bound_fraction": accept,
                "recent_reject_lower_bound_fraction": reject,
                "baseline_median_primary_topic_score": baseline_median,
                "signal_median_primary_topic_score": signal_median,
                "baseline_to_signal_score_drop": score_drop,
                "baseline_dominant_source_share": baseline_source,
                "signal_dominant_source_share": signal_source,
                "baseline_to_signal_source_share_increase": source_increase,
                "signal_opaque_title_rows": sum(
                    row["local_reason"] == "OPAQUE_ARCHIVAL_IDENTIFIER"
                    for row in windows[2]
                ),
                "indexing_anomaly_flag": (
                    score_drop >= score_threshold
                    and source_increase >= source_threshold
                ),
                "dry_run_state": state,
                "dry_run_reason": reason,
                "production_content_state": "CONTENT_UNRESOLVED",
                "production_disposition_created": False,
            }
        )
        for threshold in rule["sensitivity_accept_thresholds"]:
            sensitivity.append(
                {
                    "field_id": field_id,
                    "topic_id": topic_id,
                    "entry_year": entry_year,
                    "accept_threshold": float(threshold),
                    "dry_run_state": _classify(
                        accept,
                        reject,
                        score_drop,
                        source_increase,
                        accept_threshold=float(threshold),
                        reject_threshold=reject_threshold,
                        score_threshold=score_threshold,
                        source_threshold=source_threshold,
                    ),
                    "production_disposition_created": False,
                }
            )
    output.sort(key=lambda row: int(row["episode_index"]))
    evidence = {
        "sample_sha256": _sha256(sample_path),
        "route_sha256": _sha256(route_path),
    }
    return output, sensitivity, evidence


def _verify_artifacts(
    output_root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    require(output_root.is_dir() and not output_root.is_symlink(), "output root")
    require(output_root.stat().st_mode & 0o777 == 0o700, "output root mode")
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")
    receipt_path = output_root / str(contract["receipt_basename"])
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "receipt schema")
    require(receipt.get("status") == "PASS_ZERO_COST_CANDIDATE_CONTENT_DRY_RUN", "receipt status")
    paths = {
        "candidate": output_root / str(contract["candidate_basename"]),
        "sensitivity": output_root / str(contract["sensitivity_basename"]),
        "estimate": output_root / str(contract["estimate_basename"]),
        "markdown": output_root / str(contract["markdown_basename"]),
    }
    require(
        set(path.name for path in output_root.iterdir())
        == {path.name for path in paths.values()} | {receipt_path.name},
        "artifact set",
    )
    artifacts = _mapping(receipt.get("artifacts"), "receipt artifacts")
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"artifact: {key}")
        require(path.stat().st_mode & 0o777 == 0o600, f"artifact mode: {key}")
        require(
            _sha256(path) == artifacts[key]["sha256"],
            f"artifact SHA: {key}",
        )
    require(receipt_path.stat().st_mode & 0o777 == 0o600, "receipt mode")
    require(
        hashlib.sha256(_canonical_json_bytes(artifacts)).hexdigest()
        == receipt["artifact_tree_sha256"],
        "artifact tree",
    )
    return receipt, paths


def validate(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(_sha256(parent_path) == parent["sha256"], "parent SHA")
    boundary = _mapping(config.get("execution_boundary"), "execution boundary")
    require(all(value is False for value in boundary.values()), "unsafe boundary")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "production_content_dispositions",
        "selection_certificate",
        "deterministic_reselection",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(
            authorization.get(key) == "NOT_AUTHORIZED",
            f"unsafe authorization: {key}",
        )
    receipt, paths = _verify_artifacts(output_root, config)
    expected_candidates, expected_sensitivity, evidence = _independent_rows(config)
    observed_candidates = pq.read_table(paths["candidate"]).to_pylist()
    observed_sensitivity = pq.read_table(paths["sensitivity"]).to_pylist()
    require(observed_candidates == expected_candidates, "candidate replay")
    require(observed_sensitivity == expected_sensitivity, "sensitivity replay")
    require(receipt["closure"] == config["expected_dry_run"], "closure")
    require(
        receipt["input_evidence"]["sample_sha256"] == evidence["sample_sha256"],
        "sample evidence",
    )
    require(
        receipt["input_evidence"]["route_sha256"] == evidence["route_sha256"],
        "route evidence",
    )
    estimate = _read_json(paths["estimate"])
    require(estimate["counterfactual_only"] is True, "counterfactual flag")
    require(
        estimate["dry_run_states_applied_as_production_states"] is False,
        "dry-run state applied",
    )
    for key in (
        "raw_title_scan_authorized",
        "new_api_request_creation_authorized",
        "api_submission_authorized",
    ):
        require(estimate[key] is False, f"unsafe estimate flag: {key}")
    for key in (
        "raw_openalex_main_table_read",
        "new_titles_read",
        "network_accessed",
        "credentials_accessed",
        "outcomes_read",
        "g4_open",
    ):
        require(receipt[key] is False, f"unsafe receipt flag: {key}")
    require(receipt["api_requests_created"] == 0, "API requests created")
    require(receipt["api_requests_submitted"] == 0, "API requests submitted")
    require(
        receipt["production_content_dispositions_created"] == 0,
        "production disposition",
    )
    require(receipt["selection_certificate_issued"] is False, "certificate")
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": _sha256(config_path),
        "producer_sha256": receipt["producer_sha256"],
        "receipt_sha256": _sha256(
            output_root / str(config["output_contract"]["receipt_basename"])
        ),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "closure": receipt["closure"],
        "counterfactual_frontier_estimate": estimate,
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
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.config, args.output_root)
        report_path = args.report.absolute()
        require(not report_path.exists(), "verification report exists")
        require(report_path.parent == Path("/ssd"), "verification report parent")
        temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(result))
        temporary.chmod(0o600)
        os.replace(temporary, report_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        CandidateContentDryRunVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_CANDIDATE_CONTENT_DRY_RUN_VERIFICATION_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
