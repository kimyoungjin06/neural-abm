#!/usr/bin/env python3
"""Independently verify the zero-cost API-only topic-routing dry-run v1."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_api_only_topic_routing_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-api-only-topic-routing-v1-dry-run-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_api_only_topic_routing_config.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_api_only_topic_routing_outputs.v1"
RESULT_STATUS = "PASS_ZERO_COST_API_ONLY_TOPIC_ROUTING_DRY_RUN"
VERIFY_STATUS = "PASS_INDEPENDENT_ZERO_COST_API_ONLY_TOPIC_ROUTING_DRY_RUN"
ROUTES = [
    "NO_API_RULE_ACCEPT",
    "NO_API_RULE_REJECT",
    "API_REQUIRED_AMBIGUOUS",
    "QUARANTINE_UNRESOLVED",
]


class TopicRoutingVerificationError(RuntimeError):
    """Fail-closed topic-routing verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TopicRoutingVerificationError(message)


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
        raise TopicRoutingVerificationError(f"invalid YAML {path.name}: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TopicRoutingVerificationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"missing CSV header: {path.name}")
        return list(reader.fieldnames), list(reader)


def _ssd_root(basename: Any) -> Path:
    require(
        isinstance(basename, str)
        and basename
        and "/" not in basename
        and basename not in {".", ".."},
        "unsafe SSD root basename",
    )
    root = (Path("/ssd") / basename).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "SSD root escapes /ssd")
    require(root.is_dir() and not root.is_symlink(), "invalid SSD root")
    return root


def _normalize(value: Any) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _route(
    title: Any,
    card: Mapping[str, Any],
    *,
    opaque_patterns: Sequence[re.Pattern[str]],
    minimum_tokens: int,
) -> tuple[str, str, str, list[str], list[str]]:
    normalized = _normalize(title)
    if len(normalized.split()) < minimum_tokens:
        return (
            normalized,
            "QUARANTINE_UNRESOLVED",
            "UNINFORMATIVE_TITLE",
            [],
            [],
        )
    if any(pattern.search(normalized) for pattern in opaque_patterns):
        return (
            normalized,
            "QUARANTINE_UNRESOLVED",
            "OPAQUE_ARCHIVAL_IDENTIFIER",
            [],
            [],
        )
    positive = [
        pattern
        for pattern in card["positive_patterns"]
        if re.search(pattern, normalized)
    ]
    negative = [
        pattern
        for pattern in card["negative_patterns"]
        if re.search(pattern, normalized)
    ]
    if positive and not negative:
        return (
            normalized,
            "NO_API_RULE_ACCEPT",
            "POSITIVE_CARD_MATCH_WITHOUT_NEGATIVE",
            positive,
            negative,
        )
    if negative and not positive:
        return (
            normalized,
            "NO_API_RULE_REJECT",
            "NEGATIVE_CARD_MATCH_WITHOUT_POSITIVE",
            positive,
            negative,
        )
    return (
        normalized,
        "API_REQUIRED_AMBIGUOUS",
        "MIXED_OR_UNMATCHED_TITLE",
        positive,
        negative,
    )


def _validate_config(
    config_path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("status") == "FROZEN_ZERO_COST_TOPIC_CARD_ROUTING_DRY_RUN_AUTHORIZED",
        "config status mismatch",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = (repo_root / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(repo_root), "parent escapes repository")
    require(_sha256(parent_path) == parent["sha256"], "parent SHA mismatch")
    parent_document = _read_yaml(parent_path)
    require(
        parent_document.get("schema_version") == parent["required_schema_version"],
        "parent schema mismatch",
    )
    require(
        parent_document.get("status") == parent["required_status"],
        "parent status mismatch",
    )

    reference = _mapping(config.get("topic_reference"), "reference")
    reference_path = (repo_root / str(reference["relative_path"])).resolve(strict=True)
    require(reference_path.is_relative_to(repo_root), "reference escapes repository")
    require(_sha256(reference_path) == reference["sha256"], "reference SHA mismatch")
    reference_document = _read_json(reference_path)
    require(
        reference_document.get("schema_version") == reference["required_schema_version"],
        "reference schema mismatch",
    )
    require(reference.get("reference_is_ground_truth") is False, "reference ground truth")

    routing = _mapping(config.get("routing_contract"), "routing")
    require(routing.get("states") == ROUTES, "routes mismatch")
    require(
        routing.get("precedence")
        == [
            "QUARANTINE_IF_OPAQUE_OR_UNINFORMATIVE",
            "ACCEPT_IF_POSITIVE_AND_NOT_NEGATIVE",
            "REJECT_IF_NEGATIVE_AND_NOT_POSITIVE",
            "API_REQUIRED_OTHERWISE",
        ],
        "routing precedence mismatch",
    )
    for key in (
        "primary_topic_score_used_for_acceptance",
        "source_name_used_for_acceptance",
        "api_labels_used_for_rule_execution",
        "unresolved_may_be_forced_to_decisive_label",
    ):
        require(routing.get(key) is False, f"unsafe routing flag: {key}")
    cards = _mapping(config.get("topic_cards"), "cards")
    require(set(cards) == {10579, 12421, 12505, 14430}, "card set mismatch")
    expected_reference = {
        int(str(topic["id"]).rsplit("T", 1)[1]): topic["display_name"]
        for topic in reference_document["topics"]
    }
    require(
        expected_reference
        == {int(key): value["display_name"] for key, value in cards.items()},
        "card/reference mismatch",
    )
    for topic_id, card in cards.items():
        for key in ("positive_patterns", "negative_patterns"):
            patterns = card[key]
            require(isinstance(patterns, list) and patterns, f"patterns missing: {topic_id}")
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as error:
                    raise TopicRoutingVerificationError(
                        f"invalid regex for topic {topic_id}: {error}"
                    ) from error

    dry_run = _mapping(config.get("dry_run_contract"), "dry-run")
    for key in (
        "network_access",
        "credential_access",
        "api_requests_created",
        "api_requests_submitted",
        "accuracy_claim_allowed",
        "error_rate_claim_allowed",
        "calibration_claim_allowed",
    ):
        require(dry_run.get(key) is False, f"unsafe dry-run flag: {key}")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("zero_cost_dry_run") == "AUTHORIZED", "dry-run closed")
    for key in (
        "complete_candidate_work_manifest",
        "paid_api_submission",
        "production_rule_accept_or_reject",
        "episode_reselection",
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
    return config, {
        "parent_sha256": _sha256(parent_path),
        "reference_sha256": _sha256(reference_path),
    }


def _load_expected_rows(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    content = _mapping(frozen.get("content_audit"), "content")
    content_root = _ssd_root(content["root_basename"])
    sample_path = content_root / str(content["sample_basename"])
    content_receipt_path = content_root / str(content["receipt_basename"])
    require(_sha256(sample_path) == content["sample_sha256"], "sample SHA mismatch")
    require(
        _sha256(content_receipt_path) == content["receipt_sha256"],
        "content receipt SHA mismatch",
    )
    parquet = pq.ParquetFile(sample_path)
    require(parquet.metadata.num_rows == 384, "sample rows mismatch")
    sample_rows = parquet.read(
        columns=["episode_index", "topic_id", "window_code", "openalex_id", "title"]
    ).to_pylist()

    prior = _mapping(frozen.get("prior_api_diagnostic"), "prior API")
    api_root = _ssd_root(prior["root_basename"])
    api_path = api_root / str(prior["decisions_basename"])
    api_receipt_path = api_root / str(prior["receipt_basename"])
    require(_sha256(api_path) == prior["decisions_sha256"], "API decisions SHA")
    require(_sha256(api_receipt_path) == prior["receipt_sha256"], "API receipt SHA")
    require(prior.get("diagnostic_labels_are_ground_truth") is False, "API ground truth")
    require(prior.get("use_for_rule_execution") is False, "API executes rules")
    _, api_rows = _read_csv(api_path)
    api_by_id = {row["openalex_id"]: row for row in api_rows}
    require(len(api_by_id) == len(api_rows) == 384, "API input rows")
    require(
        {row["openalex_id"] for row in sample_rows} == set(api_by_id),
        "sample/API ID mismatch",
    )

    routing = _mapping(config.get("routing_contract"), "routing")
    opaque = [
        re.compile(pattern) for pattern in routing["global_opaque_title_patterns"]
    ]
    minimum_tokens = int(routing["minimum_informative_token_count"])
    cards = _mapping(config.get("topic_cards"), "cards")
    expected: list[dict[str, str]] = []
    for sample in sample_rows:
        topic_id = int(sample["topic_id"])
        card = _mapping(cards[topic_id], f"card {topic_id}")
        normalized, route, reason, positive, negative = _route(
            sample["title"],
            card,
            opaque_patterns=opaque,
            minimum_tokens=minimum_tokens,
        )
        api = api_by_id[str(sample["openalex_id"])]
        require(int(api["episode_index"]) == int(sample["episode_index"]), "episode")
        require(int(api["window_code"]) == int(sample["window_code"]), "window")
        require(api["target_topic_name"] == card["display_name"], "topic name")
        expected.append(
            {
                "source_row_number": api["source_row_number"],
                "openalex_id": str(sample["openalex_id"]),
                "episode_index": str(int(sample["episode_index"])),
                "window_code": str(int(sample["window_code"])),
                "topic_id": str(topic_id),
                "target_topic_name": str(card["display_name"]),
                "title": "" if sample["title"] is None else str(sample["title"]),
                "normalized_title": normalized,
                "local_route": route,
                "local_reason": reason,
                "positive_match_count": str(len(positive)),
                "negative_match_count": str(len(negative)),
                "positive_matches": "|".join(positive),
                "negative_matches": "|".join(negative),
                "prior_openai_label": api["openai_label"],
                "prior_openai_confidence": api["openai_confidence"],
                "prior_claude_label": api["claude_label"],
                "prior_claude_confidence": api["claude_confidence"],
                "prior_label_agreement": api["label_agreement"],
                "prior_dual_high_confidence_on_topic": api[
                    "dual_high_confidence_on_topic"
                ],
            }
        )
    expected.sort(key=lambda row: int(row["source_row_number"]))
    require(
        [int(row["source_row_number"]) for row in expected] == list(range(1, 385)),
        "source row sequence",
    )
    evidence = {
        "sample_sha256": _sha256(sample_path),
        "content_receipt_sha256": _sha256(content_receipt_path),
        "prior_api_decisions_sha256": _sha256(api_path),
        "prior_api_receipt_sha256": _sha256(api_receipt_path),
    }
    return expected, evidence


def _summary_metrics(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    groups: list[tuple[str, str, list[Mapping[str, str]]]] = [
        ("ALL", "ALL_TOPICS", list(rows))
    ]
    for topic_id in sorted({int(row["topic_id"]) for row in rows}):
        selected = [row for row in rows if int(row["topic_id"]) == topic_id]
        groups.append((str(topic_id), selected[0]["target_topic_name"], selected))
    result: list[dict[str, str]] = []
    for topic_id, name, selected in groups:
        accepts = [row for row in selected if row["local_route"] == ROUTES[0]]
        rejects = [row for row in selected if row["local_route"] == ROUTES[1]]
        api_required = [row for row in selected if row["local_route"] == ROUTES[2]]
        quarantine = [row for row in selected if row["local_route"] == ROUTES[3]]
        avoided = len(accepts) + len(rejects) + len(quarantine)
        result.append(
            {
                "topic_id": topic_id,
                "target_topic_name": name,
                "sample_rows": str(len(selected)),
                "no_api_rule_accept_rows": str(len(accepts)),
                "no_api_rule_reject_rows": str(len(rejects)),
                "api_required_ambiguous_rows": str(len(api_required)),
                "quarantine_unresolved_rows": str(len(quarantine)),
                "avoided_api_request_rows": str(avoided),
                "counterfactual_api_request_rows": str(len(api_required)),
                "api_request_avoidance_fraction": f"{avoided / len(selected):.6f}",
                "rule_accept_dual_high_api_rows": str(
                    sum(
                        row["prior_dual_high_confidence_on_topic"] == "true"
                        for row in accepts
                    )
                ),
                "rule_accept_either_api_off_or_unreviewable_rows": str(
                    sum(
                        row["prior_openai_label"] in {"OFF_TOPIC", "UNREVIEWABLE"}
                        or row["prior_claude_label"] in {"OFF_TOPIC", "UNREVIEWABLE"}
                        for row in accepts
                    )
                ),
                "rule_reject_both_api_off_topic_rows": str(
                    sum(
                        row["prior_openai_label"] == "OFF_TOPIC"
                        and row["prior_claude_label"] == "OFF_TOPIC"
                        for row in rejects
                    )
                ),
                "quarantine_both_api_unreviewable_rows": str(
                    sum(
                        row["prior_openai_label"] == "UNREVIEWABLE"
                        and row["prior_claude_label"] == "UNREVIEWABLE"
                        for row in quarantine
                    )
                ),
                "api_required_dual_high_api_rows": str(
                    sum(
                        row["prior_dual_high_confidence_on_topic"] == "true"
                        for row in api_required
                    )
                ),
            }
        )
    return result


def validate(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    require(output_root.is_dir() and not output_root.is_symlink(), "output root invalid")
    require(output_root.stat().st_mode & 0o777 == 0o700, "output root mode")

    config, bindings = _validate_config(config_path, repo_root=repo_root)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == output_contract["root_basename"], "output basename")
    expected_names = {
        output_contract["decisions_basename"],
        output_contract["topic_summary_basename"],
        output_contract["report_basename"],
        output_contract["receipt_basename"],
    }
    actual_names = {path.name for path in output_root.iterdir()}
    require(actual_names == expected_names, "output file set mismatch")
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"output mode: {path.name}")

    receipt_path = output_root / str(output_contract["receipt_basename"])
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(receipt.get("status") == RESULT_STATUS, "receipt status")
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    producer_path = repo_root / str(receipt["producer_relative_path"])
    require(producer_path.is_file() and not producer_path.is_symlink(), "producer path")
    require(receipt.get("producer_sha256") == _sha256(producer_path), "producer SHA")
    verifier_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    producer_module = ".".join(
        Path(str(receipt["producer_relative_path"])).with_suffix("").parts
    )
    imported_modules = {
        alias.name
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    require(
        producer_module not in imported_modules,
        "producer module reference found in verifier",
    )

    artifacts = _mapping(receipt.get("artifacts"), "artifacts")
    require(set(artifacts) == {"decisions", "topic_summary", "report"}, "artifact set")
    for artifact in artifacts.values():
        item = _mapping(artifact, "artifact")
        path = output_root / str(item["basename"])
        require(_sha256(path) == item["sha256"], f"artifact SHA: {path.name}")
        require(path.stat().st_size == item["bytes"], f"artifact size: {path.name}")
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(_canonical_json_bytes(artifacts)).hexdigest(),
        "artifact tree SHA",
    )

    expected_rows, evidence = _load_expected_rows(config)
    decision_header, observed_rows = _read_csv(
        output_root / str(output_contract["decisions_basename"])
    )
    require(
        decision_header == list(expected_rows[0]),
        "decision header mismatch",
    )
    require(observed_rows == expected_rows, "decision replay mismatch")
    expected_summary = _summary_metrics(expected_rows)
    summary_header, observed_summary = _read_csv(
        output_root / str(output_contract["topic_summary_basename"])
    )
    require(summary_header == list(expected_summary[0]), "summary header mismatch")
    require(observed_summary == expected_summary, "summary replay mismatch")

    overall = expected_summary[0]
    route_counts = {
        route: sum(row["local_route"] == route for row in expected_rows)
        for route in ROUTES
    }
    require(receipt.get("sample_rows") == 384, "receipt sample rows")
    require(receipt.get("topic_card_count") == 4, "topic card count")
    require(receipt.get("route_counts") == route_counts, "receipt route counts")
    require(
        receipt.get("avoided_api_request_rows")
        == int(overall["avoided_api_request_rows"]),
        "avoided request rows",
    )
    require(
        receipt.get("counterfactual_api_request_rows")
        == int(overall["counterfactual_api_request_rows"]),
        "counterfactual requests",
    )
    require(
        receipt.get("api_request_avoidance_fraction")
        == overall["api_request_avoidance_fraction"],
        "avoidance fraction",
    )
    require(receipt.get("repository_bindings") == bindings, "binding receipt")
    require(receipt.get("input_evidence") == evidence, "input evidence receipt")
    require(receipt.get("network_accessed") is False, "network accessed")
    require(receipt.get("credentials_accessed") is False, "credentials accessed")
    require(receipt.get("api_requests_created") == 0, "API requests created")
    require(receipt.get("api_requests_submitted") == 0, "API requests submitted")
    require(receipt.get("api_cost_usd") == "0.000000", "API cost differs")
    for key in (
        "prior_api_labels_are_ground_truth",
        "accuracy_established",
        "candidate_universe_screen_complete",
        "production_dispositions_authorized",
        "episode_reselection_authorized",
        "pivot_classification_authorized",
        "outcomes_read",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag: {key}")

    return {
        "schema_version": "pivot_feedback_d0_api_only_topic_routing_verification.v1",
        "status": VERIFY_STATUS,
        "config_sha256": _sha256(config_path),
        "producer_sha256": _sha256(producer_path),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "receipt_sha256": _sha256(receipt_path),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "sample_rows": 384,
        "topic_card_count": 4,
        "route_counts": route_counts,
        "avoided_api_request_rows": int(overall["avoided_api_request_rows"]),
        "counterfactual_api_request_rows": int(
            overall["counterfactual_api_request_rows"]
        ),
        "api_request_avoidance_fraction": overall[
            "api_request_avoidance_fraction"
        ],
        "rule_accept_dual_high_api_rows": int(
            overall["rule_accept_dual_high_api_rows"]
        ),
        "rule_accept_either_api_off_or_unreviewable_rows": int(
            overall["rule_accept_either_api_off_or_unreviewable_rows"]
        ),
        "rule_reject_both_api_off_topic_rows": int(
            overall["rule_reject_both_api_off_topic_rows"]
        ),
        "quarantine_both_api_unreviewable_rows": int(
            overall["quarantine_both_api_unreviewable_rows"]
        ),
        "producer_module_imported": False,
        "network_accessed": False,
        "credentials_accessed": False,
        "api_requests_submitted": 0,
        "api_cost_usd": "0.000000",
        "prior_api_labels_are_ground_truth": False,
        "accuracy_established": False,
        "candidate_universe_screen_complete": False,
        "production_dispositions_authorized": False,
        "episode_reselection_authorized": False,
        "pivot_classification_authorized": False,
        "outcomes_read": False,
        "model_fitting_authorized": False,
        "g4_open": False,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    require(not path.exists(), "verification report already exists")
    require(path.parent.is_dir(), "verification report parent missing")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(result))
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.config, args.output_root)
        if args.report is not None:
            _write_report(args.report.absolute(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        TopicRoutingVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_API_ONLY_TOPIC_ROUTING_V1_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
