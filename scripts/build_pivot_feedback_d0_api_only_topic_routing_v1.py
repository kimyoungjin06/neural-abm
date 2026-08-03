#!/usr/bin/env python3
"""Build the zero-cost API-only topic-routing dry-run v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
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
OUTPUT_SCHEMA = "pivot_feedback_d0_api_only_topic_routing_outputs.v1"
STATUS = "PASS_ZERO_COST_API_ONLY_TOPIC_ROUTING_DRY_RUN"
ROUTES = [
    "NO_API_RULE_ACCEPT",
    "NO_API_RULE_REJECT",
    "API_REQUIRED_AMBIGUOUS",
    "QUARANTINE_UNRESOLVED",
]
DECISION_COLUMNS = [
    "source_row_number",
    "openalex_id",
    "episode_index",
    "window_code",
    "topic_id",
    "target_topic_name",
    "title",
    "normalized_title",
    "local_route",
    "local_reason",
    "positive_match_count",
    "negative_match_count",
    "positive_matches",
    "negative_matches",
    "prior_openai_label",
    "prior_openai_confidence",
    "prior_claude_label",
    "prior_claude_confidence",
    "prior_label_agreement",
    "prior_dual_high_confidence_on_topic",
]
SUMMARY_COLUMNS = [
    "topic_id",
    "target_topic_name",
    "sample_rows",
    "no_api_rule_accept_rows",
    "no_api_rule_reject_rows",
    "api_required_ambiguous_rows",
    "quarantine_unresolved_rows",
    "avoided_api_request_rows",
    "counterfactual_api_request_rows",
    "api_request_avoidance_fraction",
    "rule_accept_dual_high_api_rows",
    "rule_accept_either_api_off_or_unreviewable_rows",
    "rule_reject_both_api_off_topic_rows",
    "quarantine_both_api_unreviewable_rows",
    "api_required_dual_high_api_rows",
]


class TopicRoutingBuildError(RuntimeError):
    """Fail-closed topic-routing dry-run error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TopicRoutingBuildError(message)


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


def _write_private(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise TopicRoutingBuildError(f"invalid config: {error}") from error
    return _mapping(value, "config")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TopicRoutingBuildError(f"invalid JSON {path.name}: {error}") from error
    return _mapping(value, path.name)


def _resolve_ssd_root(basename: Any) -> Path:
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
    tokens = normalized.split()
    if len(tokens) < minimum_tokens:
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


def _load_and_validate_config(path: Path) -> dict[str, Any]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("status") == "FROZEN_ZERO_COST_TOPIC_CARD_ROUTING_DRY_RUN_AUTHORIZED",
        "config status mismatch",
    )
    routing = _mapping(config.get("routing_contract"), "routing contract")
    require(routing.get("states") == ROUTES, "route state mismatch")
    require(
        routing.get("primary_topic_score_used_for_acceptance") is False,
        "topic score acceptance opened",
    )
    require(
        routing.get("source_name_used_for_acceptance") is False,
        "source acceptance opened",
    )
    require(
        routing.get("api_labels_used_for_rule_execution") is False,
        "API labels execute rules",
    )
    cards = _mapping(config.get("topic_cards"), "topic cards")
    require(set(cards) == {10579, 12421, 12505, 14430}, "topic card set mismatch")
    for topic_id, raw in cards.items():
        card = _mapping(raw, f"topic card {topic_id}")
        for key in ("positive_patterns", "negative_patterns"):
            patterns = card.get(key)
            require(isinstance(patterns, list) and patterns, f"{key} missing")
            for pattern in patterns:
                require(isinstance(pattern, str) and pattern, f"invalid {key}")
                try:
                    re.compile(pattern)
                except re.error as error:
                    raise TopicRoutingBuildError(
                        f"invalid regex for topic {topic_id}: {error}"
                    ) from error
    dry_run = _mapping(config.get("dry_run_contract"), "dry-run contract")
    require(dry_run.get("network_access") is False, "network access opened")
    require(dry_run.get("credential_access") is False, "credential access opened")
    require(dry_run.get("api_requests_created") is False, "API requests created")
    require(dry_run.get("api_requests_submitted") is False, "API requests submitted")
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
    return config


def _validate_repository_bindings(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    parent = _mapping(config.get("repository_parent"), "repository parent")
    parent_path = (repo_root / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(repo_root), "parent escapes repository")
    require(_sha256(parent_path) == parent.get("sha256"), "parent SHA mismatch")
    parent_document = _read_yaml(parent_path)
    require(
        parent_document.get("schema_version") == parent.get("required_schema_version"),
        "parent schema mismatch",
    )
    require(
        parent_document.get("status") == parent.get("required_status"),
        "parent status mismatch",
    )

    reference = _mapping(config.get("topic_reference"), "topic reference")
    reference_path = (repo_root / str(reference["relative_path"])).resolve(strict=True)
    require(reference_path.is_relative_to(repo_root), "reference escapes repository")
    require(_sha256(reference_path) == reference.get("sha256"), "reference SHA mismatch")
    reference_document = _read_json(reference_path)
    require(
        reference_document.get("schema_version")
        == reference.get("required_schema_version"),
        "reference schema mismatch",
    )
    require(reference.get("reference_is_ground_truth") is False, "reference ground truth")
    expected = {
        int(str(topic["id"]).rsplit("T", 1)[1]): topic["display_name"]
        for topic in reference_document["topics"]
    }
    cards = _mapping(config.get("topic_cards"), "topic cards")
    require(
        expected == {int(key): value["display_name"] for key, value in cards.items()},
        "topic reference/card mismatch",
    )
    return {
        "parent_sha256": _sha256(parent_path),
        "reference_sha256": _sha256(reference_path),
    }


def _load_inputs(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    content = _mapping(frozen.get("content_audit"), "content input")
    content_root = _resolve_ssd_root(content["root_basename"])
    sample_path = content_root / str(content["sample_basename"])
    receipt_path = content_root / str(content["receipt_basename"])
    require(_sha256(sample_path) == content.get("sample_sha256"), "sample SHA mismatch")
    require(_sha256(receipt_path) == content.get("receipt_sha256"), "content receipt SHA")
    sample_file = pq.ParquetFile(sample_path)
    require(sample_file.metadata.num_rows == content.get("sample_rows"), "sample rows")
    schema_metadata = sample_file.schema_arrow.metadata or {}
    require(
        schema_metadata.get(b"schema_version", b"").decode()
        == content.get("sample_schema_version"),
        "sample schema mismatch",
    )
    sample_rows = sample_file.read(
        columns=[
            "episode_index",
            "topic_id",
            "window_code",
            "openalex_id",
            "title",
        ]
    ).to_pylist()

    prior = _mapping(frozen.get("prior_api_diagnostic"), "prior API input")
    api_root = _resolve_ssd_root(prior["root_basename"])
    decisions_path = api_root / str(prior["decisions_basename"])
    api_receipt_path = api_root / str(prior["receipt_basename"])
    require(
        _sha256(decisions_path) == prior.get("decisions_sha256"),
        "prior decisions SHA",
    )
    require(
        _sha256(api_receipt_path) == prior.get("receipt_sha256"),
        "prior receipt SHA",
    )
    require(prior.get("diagnostic_labels_are_ground_truth") is False, "API ground truth")
    require(prior.get("use_for_rule_execution") is False, "API executes rule")
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        api_rows = list(csv.DictReader(handle))
    require(len(api_rows) == len(sample_rows) == 384, "input row count mismatch")
    api_by_id = {row["openalex_id"]: row for row in api_rows}
    require(len(api_by_id) == len(api_rows), "duplicate API OpenAlex IDs")
    require(
        {row["openalex_id"] for row in sample_rows} == set(api_by_id),
        "sample/API ID set mismatch",
    )
    evidence = {
        "sample_sha256": _sha256(sample_path),
        "content_receipt_sha256": _sha256(receipt_path),
        "prior_api_decisions_sha256": _sha256(decisions_path),
        "prior_api_receipt_sha256": _sha256(api_receipt_path),
    }
    return sample_rows, api_by_id, evidence


def _decision_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    api_by_id: Mapping[str, Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    routing = _mapping(config.get("routing_contract"), "routing")
    cards = _mapping(config.get("topic_cards"), "cards")
    opaque = [
        re.compile(pattern)
        for pattern in routing["global_opaque_title_patterns"]
    ]
    minimum_tokens = int(routing["minimum_informative_token_count"])
    output: list[dict[str, Any]] = []
    for sample in sample_rows:
        topic_id = int(sample["topic_id"])
        card = _mapping(cards[topic_id], f"card {topic_id}")
        routed = route_title(
            sample["title"],
            card,
            opaque_patterns=opaque,
            minimum_tokens=minimum_tokens,
        )
        api = api_by_id[str(sample["openalex_id"])]
        require(int(api["episode_index"]) == int(sample["episode_index"]), "episode mismatch")
        require(int(api["window_code"]) == int(sample["window_code"]), "window mismatch")
        require(api["target_topic_name"] == card["display_name"], "topic name mismatch")
        output.append(
            {
                "source_row_number": int(api["source_row_number"]),
                "openalex_id": sample["openalex_id"],
                "episode_index": int(sample["episode_index"]),
                "window_code": int(sample["window_code"]),
                "topic_id": topic_id,
                "target_topic_name": card["display_name"],
                "title": "" if sample["title"] is None else str(sample["title"]),
                "normalized_title": routed["normalized_title"],
                "local_route": routed["local_route"],
                "local_reason": routed["local_reason"],
                "positive_match_count": len(routed["positive_matches"]),
                "negative_match_count": len(routed["negative_matches"]),
                "positive_matches": "|".join(routed["positive_matches"]),
                "negative_matches": "|".join(routed["negative_matches"]),
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
    require(len(output) == 384, "decision row count mismatch")
    output.sort(key=lambda row: int(row["source_row_number"]))
    require(
        [row["source_row_number"] for row in output] == list(range(1, 385)),
        "source row number sequence mismatch",
    )
    return output


def _summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[Mapping[str, Any]]]] = [
        ("ALL", "ALL_TOPICS", list(rows))
    ]
    for topic_id in sorted({int(row["topic_id"]) for row in rows}):
        selected = [row for row in rows if int(row["topic_id"]) == topic_id]
        groups.append((str(topic_id), str(selected[0]["target_topic_name"]), selected))
    output: list[dict[str, Any]] = []
    for topic_id, name, selected in groups:
        counts = {
            route: sum(row["local_route"] == route for row in selected)
            for route in ROUTES
        }
        accepts = [
            row for row in selected if row["local_route"] == "NO_API_RULE_ACCEPT"
        ]
        rejects = [
            row for row in selected if row["local_route"] == "NO_API_RULE_REJECT"
        ]
        quarantined = [
            row for row in selected if row["local_route"] == "QUARANTINE_UNRESOLVED"
        ]
        api_required = [
            row for row in selected if row["local_route"] == "API_REQUIRED_AMBIGUOUS"
        ]
        avoided = len(accepts) + len(rejects) + len(quarantined)
        output.append(
            {
                "topic_id": topic_id,
                "target_topic_name": name,
                "sample_rows": len(selected),
                "no_api_rule_accept_rows": counts["NO_API_RULE_ACCEPT"],
                "no_api_rule_reject_rows": counts["NO_API_RULE_REJECT"],
                "api_required_ambiguous_rows": counts["API_REQUIRED_AMBIGUOUS"],
                "quarantine_unresolved_rows": counts["QUARANTINE_UNRESOLVED"],
                "avoided_api_request_rows": avoided,
                "counterfactual_api_request_rows": len(api_required),
                "api_request_avoidance_fraction": f"{avoided / len(selected):.6f}",
                "rule_accept_dual_high_api_rows": sum(
                    row["prior_dual_high_confidence_on_topic"] == "true"
                    for row in accepts
                ),
                "rule_accept_either_api_off_or_unreviewable_rows": sum(
                    row["prior_openai_label"] in {"OFF_TOPIC", "UNREVIEWABLE"}
                    or row["prior_claude_label"] in {"OFF_TOPIC", "UNREVIEWABLE"}
                    for row in accepts
                ),
                "rule_reject_both_api_off_topic_rows": sum(
                    row["prior_openai_label"] == "OFF_TOPIC"
                    and row["prior_claude_label"] == "OFF_TOPIC"
                    for row in rejects
                ),
                "quarantine_both_api_unreviewable_rows": sum(
                    row["prior_openai_label"] == "UNREVIEWABLE"
                    and row["prior_claude_label"] == "UNREVIEWABLE"
                    for row in quarantined
                ),
                "api_required_dual_high_api_rows": sum(
                    row["prior_dual_high_confidence_on_topic"] == "true"
                    for row in api_required
                ),
            }
        )
    return output


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _report(summary: Sequence[Mapping[str, Any]]) -> bytes:
    overall = summary[0]
    lines = [
        "# API-only topic-routing zero-cost dry-run",
        "",
        f"Status: `{STATUS}`",
        "",
        "No network request, credential read, or API submission occurred.",
        "",
        "## Routing result",
        "",
        "| Topic | Rows | Rule accept | Rule reject | API required | Quarantine | Avoided requests |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {name} | {sample_rows} | {accept} | {reject} | {api} | "
            "{quarantine} | {avoided} ({fraction}) |".format(
                name=row["target_topic_name"],
                sample_rows=row["sample_rows"],
                accept=row["no_api_rule_accept_rows"],
                reject=row["no_api_rule_reject_rows"],
                api=row["api_required_ambiguous_rows"],
                quarantine=row["quarantine_unresolved_rows"],
                avoided=row["avoided_api_request_rows"],
                fraction=row["api_request_avoidance_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Descriptive prior-API comparison",
            "",
            f"- Rule-accept rows also dual-HIGH API `ON_TOPIC`: "
            f"{overall['rule_accept_dual_high_api_rows']}",
            f"- Rule-accept rows where either API was `OFF_TOPIC` or "
            f"`UNREVIEWABLE`: "
            f"{overall['rule_accept_either_api_off_or_unreviewable_rows']}",
            f"- Rule-reject rows where both APIs were `OFF_TOPIC`: "
            f"{overall['rule_reject_both_api_off_topic_rows']}",
            f"- Quarantined rows where both APIs were `UNREVIEWABLE`: "
            f"{overall['quarantine_both_api_unreviewable_rows']}",
            "",
            "These are descriptive engineering comparisons. The prior API labels "
            "are not ground truth, and the fractions are not accuracy or error-rate "
            "estimates.",
            "",
            "## Boundary",
            "",
            "The 384 rows are a diagnostic title sample. The 70,277 existing "
            "candidate rows are topic-year candidates, not a complete title-bearing "
            "candidate-work manifest. Paid API execution, production dispositions, "
            "episode reselection, pivot classification, outcomes, fitting, and G4 "
            "remain unauthorized.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output root must be a direct /ssd child")
    require(not output_root.exists(), "output root already exists")
    config = _load_and_validate_config(config_path)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == output_contract["root_basename"], "output basename")
    bindings = _validate_repository_bindings(config, repo_root=repo_root)
    sample_rows, api_by_id, evidence = _load_inputs(config)
    decisions = _decision_rows(sample_rows, api_by_id, config)
    summary = _summarize(decisions)
    overall = summary[0]

    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    temp_root.chmod(0o700)
    try:
        names = {
            "decisions": output_contract["decisions_basename"],
            "topic_summary": output_contract["topic_summary_basename"],
            "report": output_contract["report_basename"],
        }
        payloads = {
            "decisions": _csv_bytes(decisions, DECISION_COLUMNS),
            "topic_summary": _csv_bytes(summary, SUMMARY_COLUMNS),
            "report": _report(summary),
        }
        artifacts: dict[str, dict[str, Any]] = {}
        for key in ("decisions", "topic_summary", "report"):
            artifact_path = temp_root / str(names[key])
            _write_private(artifact_path, payloads[key])
            artifacts[key] = {
                "basename": artifact_path.name,
                "sha256": _sha256(artifact_path),
                "bytes": artifact_path.stat().st_size,
            }
        artifact_tree_sha256 = hashlib.sha256(
            _canonical_json_bytes(artifacts)
        ).hexdigest()
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "config_relative_path": str(config_path.relative_to(repo_root)),
            "config_sha256": _sha256(config_path),
            "producer_relative_path": str(Path(__file__).resolve().relative_to(repo_root)),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "repository_bindings": bindings,
            "input_evidence": evidence,
            "sample_rows": 384,
            "topic_card_count": 4,
            "route_counts": {
                route: sum(row["local_route"] == route for row in decisions)
                for route in ROUTES
            },
            "avoided_api_request_rows": overall["avoided_api_request_rows"],
            "counterfactual_api_request_rows": overall[
                "counterfactual_api_request_rows"
            ],
            "api_request_avoidance_fraction": overall[
                "api_request_avoidance_fraction"
            ],
            "descriptive_prior_api_comparison": {
                key: overall[key]
                for key in (
                    "rule_accept_dual_high_api_rows",
                    "rule_accept_either_api_off_or_unreviewable_rows",
                    "rule_reject_both_api_off_topic_rows",
                    "quarantine_both_api_unreviewable_rows",
                    "api_required_dual_high_api_rows",
                )
            },
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
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
            "artifacts": artifacts,
            "artifact_tree_sha256": artifact_tree_sha256,
        }
        receipt_path = temp_root / str(output_contract["receipt_basename"])
        _write_private(receipt_path, _canonical_json_bytes(receipt))
        os.replace(temp_root, output_root)
        return receipt
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build(args.config, args.output_root)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (
        TopicRoutingBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_API_ONLY_TOPIC_ROUTING_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
