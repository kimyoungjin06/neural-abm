#!/usr/bin/env python3
"""Reduce the verified rank-5 API pair with the frozen dual-HIGH rule."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_result_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v2-results-20260730"
)
STATUS = "PASS_RANK5_CROSS_PROVIDER_RESULT_REDUCED_DISPOSITION_PENDING"
CORRECTED_STATUS = (
    "PASS_RANK5_HCV_CROSS_PROVIDER_RESULT_REDUCED_DISPOSITION_PENDING"
)
LABELS = {"ON_TOPIC", "AMBIGUOUS", "OFF_TOPIC", "UNREVIEWABLE"}
CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
DECISION_FIELDS = [
    "item_id",
    "openalex_id",
    "title",
    "openai_label",
    "openai_confidence",
    "openai_rationale",
    "claude_label",
    "claude_confidence",
    "claude_rationale",
    "consensus_route",
    "consensus_reason",
]


class Rank5CrossProviderResultBuildError(RuntimeError):
    """Fail-closed result reduction error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5CrossProviderResultBuildError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _nested(document: Mapping[str, Any], dotted: str) -> Any:
    value: Any = document
    for part in dotted.split("."):
        require(isinstance(value, Mapping) and part in value, f"missing {dotted}")
        value = value[part]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _yaml(path.resolve(strict=True))
    schema = config.get("schema_version")
    require(
        schema
        in {
            "pivot_feedback_d0_t10151_rank5_cross_provider_result_config.v1",
            "pivot_feedback_d0_t10151_rank5_cross_provider_result_config.v2",
        },
        "config schema",
    )
    require(
        config.get("status")
        == (
            "FROZEN_VERIFIED_INPUT_BINDINGS_RESULT_REDUCTION_AUTHORIZED"
            if str(schema).endswith(".v1")
            else "FROZEN_CORRECT_HCV_INPUT_BINDINGS_RESULT_REDUCTION_AUTHORIZED"
        ),
        "config status",
    )
    parent_binding = _mapping(config.get("contract_parent"), "contract parent")
    successor = _yaml(_repo_file(parent_binding))
    require(
        successor.get("schema_version") == parent_binding["required_schema_version"],
        "successor schema",
    )
    require(successor.get("status") == parent_binding["required_status"], "successor status")
    inherited_binding = _mapping(successor.get("contract_parent"), "inherited parent")
    inherited = _yaml(_repo_file(inherited_binding))
    require(
        inherited.get("schema_version") == inherited_binding["required_schema_version"],
        "inherited schema",
    )
    effective = dict(inherited)
    if str(schema).endswith(".v2"):
        override = _mapping(successor.get("review_contract_override"), "review override")
        review = dict(_mapping(effective.get("review_contract"), "inherited review"))
        for key in ("request_seed", "target_topic_name", "target_definition"):
            review[key] = override[key]
        effective["review_contract"] = review
    required_key = (
        "required_inherited_fields"
        if str(schema).endswith(".v1")
        else "required_effective_fields"
    )
    for dotted, expected in _mapping(
        parent_binding.get(required_key),
        "required contract fields",
    ).items():
        observed = _nested(effective, dotted)
        require(type(observed) is type(expected) and observed == expected, f"contract field: {dotted}")
    reduction = _mapping(config.get("reduction_contract"), "reduction")
    require(reduction.get("result_can_create_production_disposition") is False, "production disposition")
    require(reduction.get("result_can_issue_certificate") is False, "certificate")
    require(reduction.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    if str(schema).endswith(".v2"):
        predecessor = _mapping(config.get("invalid_predecessor"), "invalid predecessor")
        _repo_file(
            {
                "relative_path": predecessor["result_config_relative_path"],
                "sha256": predecessor["result_config_sha256"],
            }
        )
        require(predecessor.get("may_enter_current_result") is False, "invalid predecessor reuse")
    return config, effective


def _input_path(root_basename: str, basename: str, sha256: str) -> Path:
    root = (Path("/ssd") / root_basename).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "input root")
    path = root / basename
    require(path.is_file() and not path.is_symlink(), f"input invalid: {path.name}")
    require(_sha256(path) == sha256, f"input SHA: {path.name}")
    return path


def _inputs(config: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prepared = _mapping(frozen.get("prepared"), "prepared")
    execution = _mapping(frozen.get("execution"), "execution")
    key_path = _input_path(
        str(prepared["root_basename"]),
        str(prepared["request_key_basename"]),
        str(prepared["request_key_sha256"]),
    )
    preparation_receipt = _input_path(
        str(prepared["root_basename"]),
        str(prepared["receipt_basename"]),
        str(prepared["receipt_sha256"]),
    )
    verification_path = Path("/ssd") / str(prepared["verification_basename"])
    require(_sha256(verification_path) == prepared["verification_sha256"], "preparation verification SHA")
    with key_path.open(newline="", encoding="utf-8") as handle:
        request_rows = list(csv.DictReader(handle))
    require(len(request_rows) == prepared["rows"] == 25, "request key rows")
    require(len({row["item_id"] for row in request_rows}) == 25, "request item IDs")
    openai_path = _input_path(
        str(execution["root_basename"]),
        str(execution["openai_response_basename"]),
        str(execution["openai_response_sha256"]),
    )
    claude_path = _input_path(
        str(execution["root_basename"]),
        str(execution["claude_response_basename"]),
        str(execution["claude_response_sha256"]),
    )
    execution_receipt_path = _input_path(
        str(execution["root_basename"]),
        str(execution["receipt_basename"]),
        str(execution["receipt_sha256"]),
    )
    execution_receipt = _json(execution_receipt_path)
    require(execution_receipt.get("status") == execution["receipt_status"], "execution status")
    require(execution_receipt.get("provider_calls") == execution["provider_calls"] == 2, "provider calls")
    return request_rows, _json(openai_path), _json(claude_path), {
        "request_key_sha256": _sha256(key_path),
        "preparation_receipt_sha256": _sha256(preparation_receipt),
        "preparation_verification_sha256": _sha256(verification_path),
        "openai_response_sha256": _sha256(openai_path),
        "claude_response_sha256": _sha256(claude_path),
        "execution_receipt_sha256": _sha256(execution_receipt_path),
    }


def _validate_decisions(value: Any, provider: str) -> dict[str, dict[str, str]]:
    body = _mapping(value, f"{provider} structured output")
    require(set(body) == {"decisions"}, f"{provider} output fields")
    rows = body["decisions"]
    require(isinstance(rows, list) and len(rows) == 25, f"{provider} response rows")
    decisions: dict[str, dict[str, str]] = {}
    for raw in rows:
        row = _mapping(raw, f"{provider} decision")
        require(
            set(row) == {"item_id", "label", "confidence", "rationale"},
            f"{provider} decision fields",
        )
        item_id = row["item_id"]
        require(isinstance(item_id, str) and len(item_id) == 64, f"{provider} item ID")
        require(item_id not in decisions, f"{provider} duplicate item")
        require(row["label"] in LABELS, f"{provider} label")
        require(row["confidence"] in CONFIDENCES, f"{provider} confidence")
        require(
            isinstance(row["rationale"], str) and bool(row["rationale"].strip()),
            f"{provider} rationale",
        )
        decisions[item_id] = {
            "label": str(row["label"]),
            "confidence": str(row["confidence"]),
            "rationale": str(row["rationale"]).strip(),
        }
    return decisions


def _openai_decisions(wrapper: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    require(wrapper.get("provider") == "openai" and wrapper.get("http_status") == 200, "OpenAI HTTP")
    body = _mapping(wrapper.get("response"), "OpenAI response")
    require(body.get("status") == "completed" and not body.get("incomplete_details"), "OpenAI completion")
    texts = [
        item.get("text")
        for output in body.get("output", [])
        if isinstance(output, Mapping) and output.get("type") == "message"
        for item in output.get("content", [])
        if isinstance(item, Mapping) and item.get("type") == "output_text"
    ]
    require(len(texts) == 1 and isinstance(texts[0], str), "OpenAI output text")
    return _validate_decisions(json.loads(str(texts[0])), "OpenAI")


def _claude_decisions(wrapper: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    require(wrapper.get("provider") == "claude" and wrapper.get("http_status") == 200, "Claude HTTP")
    body = _mapping(wrapper.get("response"), "Claude response")
    require(body.get("stop_reason") == "end_turn", "Claude completion")
    content = body.get("content")
    require(isinstance(content, list) and len(content) == 1, "Claude content")
    item = _mapping(content[0], "Claude item")
    require(item.get("type") == "text" and isinstance(item.get("text"), str), "Claude text")
    return _validate_decisions(json.loads(str(item["text"])), "Claude")


def _consensus(openai: Mapping[str, str], claude: Mapping[str, str]) -> tuple[str, str]:
    if (
        openai["label"] == claude["label"] == "ON_TOPIC"
        and openai["confidence"] == claude["confidence"] == "HIGH"
    ):
        return "API_DUAL_HIGH_ACCEPT", "EXACT_DUAL_HIGH_ON_TOPIC"
    if (
        openai["label"] == claude["label"] == "OFF_TOPIC"
        and openai["confidence"] == claude["confidence"] == "HIGH"
    ):
        return "API_DUAL_HIGH_REJECT", "EXACT_DUAL_HIGH_OFF_TOPIC"
    if openai["label"] != claude["label"]:
        return "API_REMAINS_UNRESOLVED", "LABEL_DISAGREEMENT"
    if openai["label"] in {"AMBIGUOUS", "UNREVIEWABLE"}:
        return "API_REMAINS_UNRESOLVED", "AGREED_NON_DECISIVE_LABEL"
    return "API_REMAINS_UNRESOLVED", "NON_DUAL_HIGH_CONFIDENCE"


def _reduce(
    request_rows: Sequence[Mapping[str, str]],
    openai: Mapping[str, Mapping[str, str]],
    claude: Mapping[str, Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    expected_ids = {row["item_id"] for row in request_rows}
    require(set(openai) == expected_ids, "OpenAI item set")
    require(set(claude) == expected_ids, "Claude item set")
    reduced = []
    for source in request_rows:
        item_id = source["item_id"]
        oa = openai[item_id]
        cl = claude[item_id]
        route, reason = _consensus(oa, cl)
        reduced.append(
            {
                "item_id": item_id,
                "openalex_id": source["openalex_id"],
                "title": source["title"],
                "openai_label": oa["label"],
                "openai_confidence": oa["confidence"],
                "openai_rationale": oa["rationale"],
                "claude_label": cl["label"],
                "claude_confidence": cl["confidence"],
                "claude_rationale": cl["rationale"],
                "consensus_route": route,
                "consensus_reason": reason,
            }
        )
    route_counts = Counter(row["consensus_route"] for row in reduced)
    reason_counts = Counter(row["consensus_reason"] for row in reduced)
    reduction = _mapping(config.get("reduction_contract"), "reduction")
    api_accept = route_counts["API_DUAL_HIGH_ACCEPT"]
    api_reject = route_counts["API_DUAL_HIGH_REJECT"]
    api_unresolved = route_counts["API_REMAINS_UNRESOLVED"]
    accept = int(reduction["local_accept_rows"]) + api_accept
    reject = int(reduction["local_reject_rows"]) + api_reject
    unresolved = int(reduction["local_unresolved_rows"]) - api_accept - api_reject
    require(accept + reject + unresolved == reduction["denominator_rows"] == 64, "augmented denominator")
    if bool(reduction["indexing_anomaly_preexisting_flag"]):
        state = "CONTENT_FAIL"
        reason = "PREEXISTING_INDEXING_ANOMALY"
    elif reject >= int(reduction["reject_threshold_rows"]):
        state = "CONTENT_FAIL"
        reason = "AUGMENTED_REJECT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
    elif accept >= int(reduction["accept_threshold_rows"]):
        state = "CONTENT_PASS"
        reason = "AUGMENTED_ACCEPT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
    else:
        state = "CONTENT_UNRESOLVED"
        reason = "AUGMENTED_CONSERVATIVE_EVIDENCE_NOT_DECISIVE"
    candidate = {
        "field_id": 27,
        "field_candidate_rank": 5,
        "topic_id": 10151,
        "entry_year": 1993,
        "recent_denominator_rows": 64,
        "local_accept_rows": int(reduction["local_accept_rows"]),
        "local_reject_rows": int(reduction["local_reject_rows"]),
        "local_unresolved_rows": int(reduction["local_unresolved_rows"]),
        "api_dual_high_accept_rows": api_accept,
        "api_dual_high_reject_rows": api_reject,
        "api_remains_unresolved_rows": api_unresolved,
        "augmented_accept_rows": accept,
        "augmented_reject_rows": reject,
        "augmented_unresolved_rows": unresolved,
        "augmented_accept_fraction": accept / 64,
        "augmented_reject_fraction": reject / 64,
        "candidate_content_state": state,
        "candidate_reason": reason,
        "production_disposition_created": False,
        "certificate_issued": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "api_labels_are_ground_truth": False,
    }
    summary = {
        "schema_version": "pivot_feedback_d0_t10151_rank5_cross_provider_result_summary.v1",
        "request_rows": len(reduced),
        "openai_label_counts": dict(sorted(Counter(row["openai_label"] for row in reduced).items())),
        "openai_confidence_counts": dict(sorted(Counter(row["openai_confidence"] for row in reduced).items())),
        "claude_label_counts": dict(sorted(Counter(row["claude_label"] for row in reduced).items())),
        "claude_confidence_counts": dict(sorted(Counter(row["claude_confidence"] for row in reduced).items())),
        "exact_label_agreement_rows": sum(row["openai_label"] == row["claude_label"] for row in reduced),
        "consensus_route_counts": dict(sorted(route_counts.items())),
        "unresolved_reason_counts": dict(sorted(reason_counts.items())),
        "candidate": candidate,
        "human_review": "NOT_RUN_BY_DESIGN",
    }
    return reduced, candidate, summary


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def build(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config, _ = _load_config(config_path)
    status = (
        CORRECTED_STATUS
        if str(config.get("schema_version")).endswith(".v2")
        else STATUS
    )
    contract = _mapping(config.get("output_contract"), "output")
    output = output.absolute()
    require(output.parent == Path("/ssd"), "output parent")
    require(output.name == contract["root_basename"], "output basename")
    require(not output.exists(), "output exists")
    request_rows, openai_raw, claude_raw, input_hashes = _inputs(config)
    decisions, candidate, summary = _reduce(
        request_rows,
        _openai_decisions(openai_raw),
        _claude_decisions(claude_raw),
        config,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    temporary.chmod(0o700)
    try:
        decisions_path = temporary / str(contract["decisions_basename"])
        with decisions_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DECISION_FIELDS)
            writer.writeheader()
            writer.writerows(decisions)
        decisions_path.chmod(0o600)
        _write_json(temporary / str(contract["candidate_basename"]), candidate)
        _write_json(temporary / str(contract["summary_basename"]), summary)
        report = "\n".join(
            [
                "# T10151 rank-5 cross-provider result",
                "",
                f"Status: `{status}`",
                "",
                f"- exact label agreement: {summary['exact_label_agreement_rows']}/25",
                f"- dual-HIGH accept: {candidate['api_dual_high_accept_rows']}/25",
                f"- dual-HIGH reject: {candidate['api_dual_high_reject_rows']}/25",
                f"- API unresolved: {candidate['api_remains_unresolved_rows']}/25",
                f"- augmented accept: {candidate['augmented_accept_rows']}/64",
                f"- augmented reject: {candidate['augmented_reject_rows']}/64",
                f"- augmented unresolved: {candidate['augmented_unresolved_rows']}/64",
                f"- candidate state: `{candidate['candidate_content_state']}`",
                "",
                "Human review was not run by design. API labels are bounded",
                "title-only evidence and are not expert ground truth. No production",
                "disposition, certificate, outcome read, or model fit occurred.",
                "",
            ]
        ).encode("utf-8")
        report_path = temporary / str(contract["report_basename"])
        with report_path.open("xb") as handle:
            handle.write(report)
        report_path.chmod(0o600)
        artifacts = [
            _artifact(decisions_path, rows=25),
            _artifact(temporary / str(contract["candidate_basename"]), rows=1),
            _artifact(temporary / str(contract["summary_basename"])),
            _artifact(report_path),
        ]
        manifest = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_cross_provider_result_manifest.v1",
            "config_sha256": _sha256(config_path),
            "input_hashes": input_hashes,
            "artifacts": artifacts,
        }
        _write_json(temporary / str(contract["manifest_basename"]), manifest)
        artifacts.append(_artifact(temporary / str(contract["manifest_basename"])))
        receipt = {
            "schema_version": (
                "pivot_feedback_d0_t10151_rank5_cross_provider_result_outputs.v2"
                if str(config.get("schema_version")).endswith(".v2")
                else "pivot_feedback_d0_t10151_rank5_cross_provider_result_outputs.v1"
            ),
            "status": status,
            "config_sha256": _sha256(config_path),
            "input_hashes": input_hashes,
            "candidate": candidate,
            "result_rows": 25,
            "human_review": "NOT_RUN_BY_DESIGN",
            "api_labels_are_ground_truth": False,
            "production_disposition_created": False,
            "certificate_issued": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "artifacts": artifacts,
        }
        _write_json(temporary / str(contract["receipt_basename"]), receipt)
        os.replace(temporary, output)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            require(temporary.parent == output.parent, "temporary parent")
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(config_path=args.config, output=args.output), indent=2, sort_keys=True))
        return 0
    except (Rank5CrossProviderResultBuildError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_CROSS_PROVIDER_RESULT_BUILD", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
