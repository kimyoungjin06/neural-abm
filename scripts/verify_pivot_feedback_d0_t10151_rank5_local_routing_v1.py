#!/usr/bin/env python3
"""Independently verify rank-5 T10151 routing and rule-v2 dry-run."""

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

try:
    from scripts import verify_pivot_feedback_d0_t10543_local_routing_v1 as base
except ModuleNotFoundError:
    import verify_pivot_feedback_d0_t10543_local_routing_v1 as base


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_local_routing_v1.yaml"
)
DEFAULT_SAMPLE_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-targeted-title-sample-v1-20260729"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-local-routing-v1-20260729"
)
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-local-routing-v1-20260729."
    "verification-v1.json"
)
CONFIG_SCHEMA = "pivot_feedback_d0_t10151_rank5_local_routing_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t10151_rank5_local_routing_outputs.v1"
VERIFY_SCHEMA = "pivot_feedback_d0_t10151_rank5_local_routing_verification.v1"
VERIFY_STATUS = "PASS_INDEPENDENT_T10151_RANK5_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN"


class T10151Rank5LocalRoutingVerificationError(RuntimeError):
    """Fail-closed rank-5 local-routing verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T10151Rank5LocalRoutingVerificationError(message)


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


def _card_config(source: Mapping[str, Any]) -> dict[str, Any]:
    card = _mapping(source.get("frozen_topic_card"), "frozen card")
    require(card.get("freeze_timing") == "BEFORE_T10151_TITLE_SAMPLE_READ", "freeze timing")
    return {
        "routing_contract": {
            "states": [
                "NO_API_RULE_ACCEPT",
                "NO_API_RULE_REJECT",
                "API_REQUIRED_AMBIGUOUS",
                "QUARANTINE_UNRESOLVED",
            ],
            "precedence": card["precedence"],
            "title_normalization": card["title_normalization"],
            "minimum_informative_token_count": card[
                "minimum_informative_token_count"
            ],
            "global_opaque_title_patterns": card["global_opaque_title_patterns"],
            "default_state": "API_REQUIRED_AMBIGUOUS",
        },
        "topic_card": {
            "topic_id": 10151,
            "display_name": "Hepatitis C virus research",
            "reference_id": "https://openalex.org/T10151",
            "positive_patterns": card["positive_patterns"],
            "negative_patterns": card["negative_patterns"],
        },
    }


def _validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status") == "FROZEN_T10151_RANK5_LOCAL_ROUTING_DRY_RUN_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    card_binding = _mapping(
        _mapping(config.get("topic_card"), "topic card").get("source"),
        "card source",
    )
    require(card_binding["topic_id"] == 10151, "card topic")
    require(card_binding["reuse_without_change"] is True, "card reuse")
    card_config = _card_config(_read_yaml(_repo_file(card_binding)))
    rule_binding = _mapping(
        _mapping(config.get("production_rule_v2"), "rule").get("source"),
        "rule source",
    )
    _repo_file(rule_binding)
    require(rule_binding["reuse_without_change"] is True, "rule reuse")
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
    return config, card_config


def _verify_output(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = _mapping(config.get("output_contract"), "output contract")
    root = output_root.resolve(strict=True)
    require(root.name == contract["root_basename"], "output basename")
    require(root.is_dir() and not root.is_symlink(), "output root")
    require(root.stat().st_mode & 0o777 == 0o700, "output mode")
    paths = {
        "routes": root / str(contract["routes_basename"]),
        "candidate": root / str(contract["candidate_basename"]),
        "report": root / str(contract["report_basename"]),
    }
    receipt_path = root / str(contract["receipt_basename"])
    require(
        {path.name for path in root.iterdir()}
        == {path.name for path in paths.values()} | {receipt_path.name},
        "artifact set",
    )
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "output schema")
    require(
        receipt.get("status")
        == "PASS_ZERO_COST_T10151_RANK5_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN",
        "output status",
    )
    require(receipt.get("config_sha256") == config_sha256, "config SHA")
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"artifact: {key}")
        require(path.stat().st_mode & 0o777 == 0o600, f"artifact mode: {key}")
        artifact = _mapping(receipt["artifacts"][key], f"artifact: {key}")
        require(path.stat().st_size == artifact["bytes"], f"artifact bytes: {key}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {key}")
    require(receipt_path.stat().st_mode & 0o777 == 0o600, "receipt mode")
    require(
        hashlib.sha256(_canonical_json_bytes(receipt["artifacts"])).hexdigest()
        == receipt["artifact_tree_sha256"],
        "artifact tree",
    )
    return receipt, paths


def _load_sample(
    config: Mapping[str, Any],
    sample_root: Path,
) -> list[dict[str, Any]]:
    binding = _mapping(config.get("frozen_input"), "frozen input")
    root = sample_root.resolve(strict=True)
    require(root.name == binding["root_basename"], "sample root")
    sample = root / str(binding["sample_basename"])
    receipt = root / str(binding["receipt_basename"])
    verification = Path("/ssd") / str(binding["verification_basename"])
    require(_sha256(sample) == binding["sample_sha256"], "sample SHA")
    require(_sha256(receipt) == binding["receipt_sha256"], "sample receipt SHA")
    require(_sha256(verification) == binding["verification_sha256"], "verification SHA")
    rows = pq.read_table(sample).to_pylist()
    require(len(rows) == 96, "sample rows")
    require(
        {
            (
                row["field_id"],
                row["field_candidate_rank"],
                row["topic_id"],
                row["entry_year"],
            )
            for row in rows
        }
        == {(27, 5, 10151, 1993)},
        "sample target",
    )
    require(
        Counter(int(row["window_code"]) for row in rows)
        == {0: 32, 1: 32, 2: 32},
        "sample windows",
    )
    return rows


def _replay_routes(
    samples: Sequence[Mapping[str, Any]],
    path: Path,
    card_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        observed = list(csv.DictReader(handle))
    require(len(observed) == len(samples) == 96, "route rows")
    output: list[dict[str, Any]] = []
    for source_row, (sample, row) in enumerate(
        zip(samples, observed, strict=True),
        start=1,
    ):
        expected = base._route(sample["title"], card_config)
        require(int(row["source_row_number"]) == source_row, "route source row")
        require(row["openalex_id"] == sample["openalex_id"], "route ID")
        require(int(row["window_code"]) == sample["window_code"], "route window")
        require(row["title"] == (sample["title"] or ""), "route title")
        for key in ("normalized_title", "local_route", "local_reason"):
            require(row[key] == expected[key], f"route replay: {key}")
        require(
            row["positive_matches"] == "|".join(expected["positive_matches"]),
            "positive matches",
        )
        require(
            row["negative_matches"] == "|".join(expected["negative_matches"]),
            "negative matches",
        )
        output.append({**sample, **expected})
    return output


def _dominant(rows: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    counts = Counter(
        "<MISSING_SOURCE>" if row["source_name"] is None else str(row["source_name"])
        for row in rows
    )
    source, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return source, count / len(rows)


def _rule_state(
    *,
    routes: Counter[str],
    opaque_fraction: float,
    score_drop: float,
    source_increase: float,
    rule: Mapping[str, Any],
) -> tuple[str, str, str]:
    if opaque_fraction >= float(rule["majority_opaque_threshold"]):
        return (
            "RECENT_MAJORITY_OPAQUE_INDEXING_ANOMALY",
            "WOULD_CONTENT_FAIL_INDEXING_ANOMALY",
            "RECENT_SAMPLE_MAJORITY_OPAQUE_INDEXING_ARTIFACTS",
        )
    if (
        score_drop >= float(rule["joint_indexing_score_drop_threshold"])
        and source_increase
        >= float(rule["joint_indexing_source_share_increase_threshold"])
    ):
        return (
            "JOINT_SCORE_AND_SOURCE_SHIFT_INDEXING_ANOMALY",
            "WOULD_CONTENT_FAIL_INDEXING_ANOMALY",
            "JOINT_SCORE_DROP_AND_SOURCE_CONCENTRATION_SHIFT",
        )
    if routes["NO_API_RULE_REJECT"] / 64 >= float(rule["semantic_reject_threshold"]):
        return (
            "SEMANTIC_OFF_TOPIC",
            "WOULD_CONTENT_FAIL_SEMANTIC_OFF_TOPIC",
            "RECENT_RULE_REJECT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD",
        )
    if routes["NO_API_RULE_ACCEPT"] / 64 >= float(rule["semantic_accept_threshold"]):
        return (
            "CONTENT_PASS",
            "WOULD_CONTENT_PASS",
            "RECENT_RULE_ACCEPT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD",
        )
    return (
        "CONTENT_UNRESOLVED",
        "WOULD_CONTENT_UNRESOLVED",
        "CONSERVATIVE_RECENT_EVIDENCE_NOT_DECISIVE",
    )


def _verify_candidate(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    windows = {
        window: [row for row in rows if int(row["window_code"]) == window]
        for window in (0, 1, 2)
    }
    recent = windows[1] + windows[2]
    routes = Counter(str(row["local_route"]) for row in recent)
    baseline_score = float(
        statistics.median(float(row["primary_topic_score"]) for row in windows[1])
    )
    signal_score = float(
        statistics.median(float(row["primary_topic_score"]) for row in windows[2])
    )
    baseline_source, baseline_share = _dominant(windows[1])
    signal_source, signal_share = _dominant(windows[2])
    opaque = sum(
        row["local_reason"] == "OPAQUE_ARCHIVAL_IDENTIFIER" for row in recent
    )
    opaque_fraction = opaque / 64
    score_drop = baseline_score - signal_score
    source_increase = signal_share - baseline_share
    branch, state, reason = _rule_state(
        routes=routes,
        opaque_fraction=opaque_fraction,
        score_drop=score_drop,
        source_increase=source_increase,
        rule=_mapping(config.get("production_rule_v2"), "rule"),
    )
    expected = {
        "field_id": 27,
        "field_candidate_rank": 5,
        "topic_id": 10151,
        "topic_name": "Hepatitis C virus research",
        "entry_year": 1993,
        "recent_title_rows": 64,
        "recent_accept_rows": routes["NO_API_RULE_ACCEPT"],
        "recent_reject_rows": routes["NO_API_RULE_REJECT"],
        "recent_ambiguous_rows": routes["API_REQUIRED_AMBIGUOUS"],
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
    observed_rows = pq.read_table(path).to_pylist()
    require(len(observed_rows) == 1, "candidate rows")
    observed = observed_rows[0]
    require(observed == expected, "candidate replay")
    require(receipt["candidate"] == observed, "candidate receipt")
    for key, value in _mapping(config.get("expected_dry_run"), "expected").items():
        require(observed[key] == value, f"expected dry-run: {key}")
    return observed


def validate(
    config_path: Path = DEFAULT_CONFIG,
    sample_root: Path = DEFAULT_SAMPLE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config_sha256 = _sha256(config_path)
    config, card_config = _validate_config(config_path)
    receipt, paths = _verify_output(
        config,
        output_root,
        config_sha256=config_sha256,
    )
    samples = _load_sample(config, sample_root)
    routed = _replay_routes(samples, paths["routes"], card_config)
    candidate = _verify_candidate(
        routed,
        paths["candidate"],
        receipt,
        config,
    )
    for key in (
        "raw_openalex_main_table_read",
        "new_titles_read",
        "network_accessed",
        "credentials_accessed",
        "selection_certificate_issued",
        "deterministic_reselection_executed",
        "outcomes_read",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt[key] is False, f"unsafe receipt flag: {key}")
    require(receipt["api_requests_created"] == 0, "API requests created")
    require(receipt["api_requests_submitted"] == 0, "API requests submitted")
    require(
        receipt["production_content_dispositions_created"] == 0,
        "production disposition",
    )
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "producer_sha256": receipt["producer_sha256"],
        "receipt_sha256": _sha256(
            output_root / str(config["output_contract"]["receipt_basename"])
        ),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "candidate": candidate,
        "api_requests_created": 0,
        "api_requests_submitted": 0,
        "api_cost_usd": "0.000000",
        "production_content_dispositions_created": 0,
        "selection_certificate_issued": False,
        "outcomes_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.config, args.sample_root, args.output_root)
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
        T10151Rank5LocalRoutingVerificationError,
        base.T10543LocalRoutingVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_T10151_RANK5_ROUTING_VERIFICATION", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
