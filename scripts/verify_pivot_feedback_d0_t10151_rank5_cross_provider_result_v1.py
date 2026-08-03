#!/usr/bin/env python3
"""Independently verify rank-5 cross-provider result reduction."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
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
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v2-results-20260730."
    "verification-v1.json"
)
RESULT_STATUS = "PASS_RANK5_CROSS_PROVIDER_RESULT_REDUCED_DISPOSITION_PENDING"
CORRECTED_RESULT_STATUS = (
    "PASS_RANK5_HCV_CROSS_PROVIDER_RESULT_REDUCED_DISPOSITION_PENDING"
)
VERIFY_STATUS = "PASS_INDEPENDENT_RANK5_CROSS_PROVIDER_RESULT_VERIFICATION"
LABELS = {"ON_TOPIC", "AMBIGUOUS", "OFF_TOPIC", "UNREVIEWABLE"}
CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
OUTPUT_FILES = {
    "rank5_cross_provider_decisions.csv",
    "rank5_augmented_candidate.json",
    "rank5_cross_provider_result_summary.json",
    "API_RESULT_SUMMARY.md",
    "result_manifest.json",
    "cross_provider_result.outputs.json",
}


class Rank5CrossProviderResultVerificationError(RuntimeError):
    """Fail-closed result verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5CrossProviderResultVerificationError(message)


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


def _repo_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return _yaml(path)


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
    parent = _mapping(config.get("contract_parent"), "contract parent")
    successor = _repo_binding(parent)
    require(successor.get("schema_version") == parent["required_schema_version"], "successor schema")
    require(successor.get("status") == parent["required_status"], "successor status")
    inherited_binding = _mapping(successor.get("contract_parent"), "inherited binding")
    inherited = _repo_binding(inherited_binding)
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
        parent.get(required_key),
        "required fields",
    ).items():
        observed = _nested(effective, dotted)
        require(type(observed) is type(expected) and observed == expected, f"contract field: {dotted}")
    if str(schema).endswith(".v2"):
        predecessor = _mapping(config.get("invalid_predecessor"), "invalid predecessor")
        _repo_binding(
            {
                "relative_path": predecessor["result_config_relative_path"],
                "sha256": predecessor["result_config_sha256"],
            }
        )
        require(predecessor.get("may_enter_current_result") is False, "invalid predecessor reuse")
    return config, effective


def _bound_file(root: str, basename: str, digest: str) -> Path:
    parent = (Path("/ssd") / root).resolve(strict=True)
    require(parent.parent == Path("/ssd").resolve(), "input root")
    path = parent / basename
    require(path.is_file() and not path.is_symlink(), f"input invalid: {basename}")
    require(_sha256(path) == digest, f"input SHA: {basename}")
    return path


def _parse_structured(value: Any, provider: str) -> dict[str, dict[str, str]]:
    document = _mapping(value, f"{provider} output")
    require(set(document) == {"decisions"}, f"{provider} output keys")
    raw_rows = document["decisions"]
    require(isinstance(raw_rows, list) and len(raw_rows) == 25, f"{provider} rows")
    rows: dict[str, dict[str, str]] = {}
    for raw in raw_rows:
        item = _mapping(raw, f"{provider} decision")
        require(set(item) == {"item_id", "label", "confidence", "rationale"}, f"{provider} fields")
        item_id = item["item_id"]
        require(isinstance(item_id, str) and len(item_id) == 64 and item_id not in rows, f"{provider} item ID")
        require(item["label"] in LABELS, f"{provider} label")
        require(item["confidence"] in CONFIDENCES, f"{provider} confidence")
        require(isinstance(item["rationale"], str) and bool(item["rationale"].strip()), f"{provider} rationale")
        rows[item_id] = {
            "label": str(item["label"]),
            "confidence": str(item["confidence"]),
            "rationale": str(item["rationale"]).strip(),
        }
    return rows


def _provider_rows(path: Path, provider: str) -> dict[str, dict[str, str]]:
    wrapper = _json(path)
    require(wrapper.get("provider") == provider and wrapper.get("http_status") == 200, f"{provider} HTTP")
    body = _mapping(wrapper.get("response"), f"{provider} body")
    if provider == "openai":
        require(body.get("status") == "completed" and not body.get("incomplete_details"), "OpenAI status")
        texts = [
            content.get("text")
            for output in body.get("output", [])
            if isinstance(output, Mapping) and output.get("type") == "message"
            for content in output.get("content", [])
            if isinstance(content, Mapping) and content.get("type") == "output_text"
        ]
    else:
        require(body.get("stop_reason") == "end_turn", "Claude status")
        content = body.get("content")
        require(isinstance(content, list) and len(content) == 1, "Claude content")
        item = _mapping(content[0], "Claude item")
        require(item.get("type") == "text", "Claude content type")
        texts = [item.get("text")]
    require(len(texts) == 1 and isinstance(texts[0], str), f"{provider} text")
    return _parse_structured(json.loads(str(texts[0])), provider)


def _expected(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, str]]:
    inputs = _mapping(config.get("frozen_inputs"), "inputs")
    prepared = _mapping(inputs.get("prepared"), "prepared")
    execution = _mapping(inputs.get("execution"), "execution")
    key_path = _bound_file(
        str(prepared["root_basename"]),
        str(prepared["request_key_basename"]),
        str(prepared["request_key_sha256"]),
    )
    preparation_receipt = _bound_file(
        str(prepared["root_basename"]),
        str(prepared["receipt_basename"]),
        str(prepared["receipt_sha256"]),
    )
    verification = Path("/ssd") / str(prepared["verification_basename"])
    require(_sha256(verification) == prepared["verification_sha256"], "preparation verification SHA")
    openai_path = _bound_file(
        str(execution["root_basename"]),
        str(execution["openai_response_basename"]),
        str(execution["openai_response_sha256"]),
    )
    claude_path = _bound_file(
        str(execution["root_basename"]),
        str(execution["claude_response_basename"]),
        str(execution["claude_response_sha256"]),
    )
    execution_receipt = _bound_file(
        str(execution["root_basename"]),
        str(execution["receipt_basename"]),
        str(execution["receipt_sha256"]),
    )
    require(_json(execution_receipt).get("status") == execution["receipt_status"], "execution status")
    with key_path.open(newline="", encoding="utf-8") as handle:
        keys = list(csv.DictReader(handle))
    require(len(keys) == 25 and len({row["item_id"] for row in keys}) == 25, "request key")
    openai = _provider_rows(openai_path, "openai")
    claude = _provider_rows(claude_path, "claude")
    ids = {row["item_id"] for row in keys}
    require(set(openai) == set(claude) == ids, "provider item sets")
    rows = []
    for source in keys:
        item_id = source["item_id"]
        oa = openai[item_id]
        cl = claude[item_id]
        if oa["label"] == cl["label"] == "ON_TOPIC" and oa["confidence"] == cl["confidence"] == "HIGH":
            route, reason = "API_DUAL_HIGH_ACCEPT", "EXACT_DUAL_HIGH_ON_TOPIC"
        elif oa["label"] == cl["label"] == "OFF_TOPIC" and oa["confidence"] == cl["confidence"] == "HIGH":
            route, reason = "API_DUAL_HIGH_REJECT", "EXACT_DUAL_HIGH_OFF_TOPIC"
        elif oa["label"] != cl["label"]:
            route, reason = "API_REMAINS_UNRESOLVED", "LABEL_DISAGREEMENT"
        elif oa["label"] in {"AMBIGUOUS", "UNREVIEWABLE"}:
            route, reason = "API_REMAINS_UNRESOLVED", "AGREED_NON_DECISIVE_LABEL"
        else:
            route, reason = "API_REMAINS_UNRESOLVED", "NON_DUAL_HIGH_CONFIDENCE"
        rows.append(
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
    routes = Counter(row["consensus_route"] for row in rows)
    reasons = Counter(row["consensus_reason"] for row in rows)
    accept = 27 + routes["API_DUAL_HIGH_ACCEPT"]
    reject = 12 + routes["API_DUAL_HIGH_REJECT"]
    unresolved = 25 - routes["API_DUAL_HIGH_ACCEPT"] - routes["API_DUAL_HIGH_REJECT"]
    require(accept + reject + unresolved == 64, "denominator")
    if reject >= 32:
        state, state_reason = "CONTENT_FAIL", "AUGMENTED_REJECT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
    elif accept >= 32:
        state, state_reason = "CONTENT_PASS", "AUGMENTED_ACCEPT_LOWER_BOUND_AT_OR_ABOVE_THRESHOLD"
    else:
        state, state_reason = "CONTENT_UNRESOLVED", "AUGMENTED_CONSERVATIVE_EVIDENCE_NOT_DECISIVE"
    candidate = {
        "field_id": 27,
        "field_candidate_rank": 5,
        "topic_id": 10151,
        "entry_year": 1993,
        "recent_denominator_rows": 64,
        "local_accept_rows": 27,
        "local_reject_rows": 12,
        "local_unresolved_rows": 25,
        "api_dual_high_accept_rows": routes["API_DUAL_HIGH_ACCEPT"],
        "api_dual_high_reject_rows": routes["API_DUAL_HIGH_REJECT"],
        "api_remains_unresolved_rows": routes["API_REMAINS_UNRESOLVED"],
        "augmented_accept_rows": accept,
        "augmented_reject_rows": reject,
        "augmented_unresolved_rows": unresolved,
        "augmented_accept_fraction": accept / 64,
        "augmented_reject_fraction": reject / 64,
        "candidate_content_state": state,
        "candidate_reason": state_reason,
        "production_disposition_created": False,
        "certificate_issued": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "api_labels_are_ground_truth": False,
    }
    summary = {
        "schema_version": "pivot_feedback_d0_t10151_rank5_cross_provider_result_summary.v1",
        "request_rows": 25,
        "openai_label_counts": dict(sorted(Counter(row["openai_label"] for row in rows).items())),
        "openai_confidence_counts": dict(sorted(Counter(row["openai_confidence"] for row in rows).items())),
        "claude_label_counts": dict(sorted(Counter(row["claude_label"] for row in rows).items())),
        "claude_confidence_counts": dict(sorted(Counter(row["claude_confidence"] for row in rows).items())),
        "exact_label_agreement_rows": sum(row["openai_label"] == row["claude_label"] for row in rows),
        "consensus_route_counts": dict(sorted(routes.items())),
        "unresolved_reason_counts": dict(sorted(reasons.items())),
        "candidate": candidate,
        "human_review": "NOT_RUN_BY_DESIGN",
    }
    hashes = {
        "request_key_sha256": _sha256(key_path),
        "preparation_receipt_sha256": _sha256(preparation_receipt),
        "preparation_verification_sha256": _sha256(verification),
        "openai_response_sha256": _sha256(openai_path),
        "claude_response_sha256": _sha256(claude_path),
        "execution_receipt_sha256": _sha256(execution_receipt),
    }
    return rows, candidate, summary, hashes


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def validate(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config, _ = _load_config(config_path)
    output = output.resolve(strict=True)
    require(output.name == _mapping(config["output_contract"], "output")["root_basename"], "output basename")
    require({path.name for path in output.iterdir()} == OUTPUT_FILES, "output files")
    for path in output.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"output mode: {path.name}")
    expected_rows, expected_candidate, expected_summary, expected_hashes = _expected(config)
    with (output / "rank5_cross_provider_decisions.csv").open(newline="", encoding="utf-8") as handle:
        observed_rows = list(csv.DictReader(handle))
    require(observed_rows == expected_rows, "decision rows")
    require(_json(output / "rank5_augmented_candidate.json") == expected_candidate, "candidate")
    require(_json(output / "rank5_cross_provider_result_summary.json") == expected_summary, "summary")
    receipt = _json(output / "cross_provider_result.outputs.json")
    require(
        receipt.get("status")
        == (
            CORRECTED_RESULT_STATUS
            if str(config.get("schema_version")).endswith(".v2")
            else RESULT_STATUS
        ),
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "receipt config SHA")
    require(receipt.get("input_hashes") == expected_hashes, "receipt input hashes")
    require(receipt.get("candidate") == expected_candidate, "receipt candidate")
    require(receipt.get("result_rows") == 25, "receipt rows")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "receipt human review")
    require(receipt.get("api_labels_are_ground_truth") is False, "receipt ground truth")
    require(receipt.get("production_disposition_created") is False, "receipt production disposition")
    require(receipt.get("certificate_issued") is False, "receipt certificate")
    require(receipt.get("outcomes_read") is False, "receipt outcomes")
    manifest = _json(output / "result_manifest.json")
    require(manifest.get("config_sha256") == _sha256(config_path), "manifest config SHA")
    require(manifest.get("input_hashes") == expected_hashes, "manifest input hashes")
    listed = {item["basename"]: item for item in manifest.get("artifacts", [])}
    for name, rows in (
        ("rank5_cross_provider_decisions.csv", 25),
        ("rank5_augmented_candidate.json", 1),
        ("rank5_cross_provider_result_summary.json", None),
        ("API_RESULT_SUMMARY.md", None),
    ):
        actual = _artifact(output / name, rows=rows)
        require(listed.get(name) == actual, f"manifest artifact: {name}")
    receipt_artifacts = {item["basename"]: item for item in receipt.get("artifacts", [])}
    for name in OUTPUT_FILES - {"cross_provider_result.outputs.json"}:
        require(name in receipt_artifacts, f"receipt artifact: {name}")
        require(receipt_artifacts[name]["sha256"] == _sha256(output / name), f"receipt artifact SHA: {name}")
    return {
        "schema_version": "pivot_feedback_d0_t10151_rank5_cross_provider_result_verification.v1",
        "status": VERIFY_STATUS,
        "config_sha256": _sha256(config_path),
        "output_root_basename": output.name,
        "decision_rows_verified": 25,
        "exact_label_agreement_rows": expected_summary["exact_label_agreement_rows"],
        "consensus_route_counts": expected_summary["consensus_route_counts"],
        "unresolved_reason_counts": expected_summary["unresolved_reason_counts"],
        "candidate": expected_candidate,
        "human_review": "NOT_RUN_BY_DESIGN",
        "production_disposition_authorized": False,
        "certificate_authorized": False,
        "outcomes_read": False,
        "g4_open": False,
        "artifacts": {
            name: _artifact(output / name)
            for name in sorted(OUTPUT_FILES)
        },
    }


def verify(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    result = validate(config_path=config_path, output=output)
    report = report.absolute()
    require(not report.exists(), "verification report exists")
    with report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report.chmod(0o600)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = (
            validate(config_path=args.config, output=args.output)
            if args.check_only
            else verify(config_path=args.config, output=args.output, report=args.report)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (Rank5CrossProviderResultVerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_CROSS_PROVIDER_RESULT_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
