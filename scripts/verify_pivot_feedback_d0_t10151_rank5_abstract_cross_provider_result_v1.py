#!/usr/bin/env python3
"""Independently verify the five-row abstract result reduction."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

import execute_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1 as execution


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-abstract-cross-provider-v1-results-20260730"
)
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-abstract-cross-provider-v1-results-20260730."
    "verification-v1.json"
)
STATUS = "PASS_INDEPENDENT_RANK5_ABSTRACT_CROSS_PROVIDER_RESULT_VERIFICATION"


class AbstractCrossProviderResultVerificationError(RuntimeError):
    """Fail-closed abstract result verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AbstractCrossProviderResultVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def validate(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_config.v1",
        "config schema",
    )
    require(
        config.get("status") == "FROZEN_EXACT_FIVE_ABSTRACT_RESULT_REDUCTION_AUTHORIZED",
        "config status",
    )
    request = _mapping(config.get("request_contract"), "request")
    request_path = (REPO_ROOT / str(request["relative_path"])).resolve(strict=True)
    require(request_path.is_relative_to(REPO_ROOT), "request path")
    require(_sha256(request_path) == request["sha256"], "request SHA")
    request_config = _yaml(request_path)

    output = _mapping(config.get("output_contract"), "output")
    output_root = output_root.resolve(strict=True)
    require(output_root.parent == Path("/ssd").resolve(), "output parent")
    require(output_root.name == output["root_basename"], "output root")
    receipt_path = output_root / str(output["receipt_basename"])
    receipt = _json(receipt_path)
    require(
        receipt.get("status")
        == "PASS_RANK5_ABSTRACT_CROSS_PROVIDER_RESULT_REDUCED_DISPOSITION_PENDING",
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    artifacts = {
        str(value["basename"]): _mapping(value, "artifact")
        for value in receipt.get("artifacts", [])
    }
    expected_artifacts = {
        output["decisions_basename"],
        output["candidate_basename"],
        output["summary_basename"],
        output["report_basename"],
    }
    require(set(artifacts) == expected_artifacts, "artifact set")
    for basename, artifact in artifacts.items():
        path = output_root / basename
        require(path.is_file() and not path.is_symlink(), f"artifact: {basename}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {basename}")
        require(path.stat().st_mode & 0o077 == 0, f"artifact mode: {basename}")

    prepared = _mapping(config.get("prepared_input"), "prepared")
    prepared_root = (Path("/ssd") / str(prepared["root_basename"])).resolve(strict=True)
    with (prepared_root / str(prepared["key_basename"])).open(newline="", encoding="utf-8") as handle:
        key_rows = list(csv.DictReader(handle))
    require(len(key_rows) == 5, "key rows")
    key_by_item = {row["item_id"]: row for row in key_rows}
    execution_input = _mapping(config.get("execution_input"), "execution")
    execution_root = (Path("/ssd") / str(execution_input["root_basename"])).resolve(strict=True)
    expected_ids = set(key_by_item)
    provider_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for provider, parser in (
        ("openai", execution.transport._openai_text),
        ("claude", execution.transport._claude_text),
    ):
        path = execution_root / str(execution_input[f"{provider}_response_basename"])
        require(_sha256(path) == execution_input[f"{provider}_response_sha256"], f"{provider} SHA")
        response = _json(path)
        text = parser(_mapping(response["response"], f"{provider} response"))
        parsed = execution.decisions(
            text,
            provider=provider,
            expected_ids=expected_ids,
            config=request_config,
        )
        provider_rows[provider] = {row["item_id"]: row for row in parsed}

    expected_rows: list[dict[str, Any]] = []
    routes: Counter[str] = Counter()
    exact_label_agreement = 0
    for item_id in sorted(expected_ids):
        openai = provider_rows["openai"][item_id]
        claude = provider_rows["claude"][item_id]
        exact_label_agreement += openai["label"] == claude["label"]
        if (
            openai["label"] == claude["label"] == "ON_TOPIC"
            and openai["confidence"] == claude["confidence"] == "HIGH"
        ):
            route = "ABSTRACT_DUAL_HIGH_ACCEPT"
            reason = "EXACT_DUAL_HIGH_ON_TOPIC"
        elif (
            openai["label"] == claude["label"] == "OFF_TOPIC"
            and openai["confidence"] == claude["confidence"] == "HIGH"
        ):
            route = "ABSTRACT_DUAL_HIGH_REJECT"
            reason = "EXACT_DUAL_HIGH_OFF_TOPIC"
        else:
            route = "ABSTRACT_REMAINS_UNRESOLVED"
            reason = (
                "LABEL_DISAGREEMENT"
                if openai["label"] != claude["label"]
                else "AGREED_LABEL_NON_DUAL_HIGH_CONFIDENCE"
            )
        routes[route] += 1
        key = key_by_item[item_id]
        expected_rows.append(
            {
                "item_id": item_id,
                "openalex_id": key["openalex_id"],
                "title": key["title"],
                "abstract_sha256": key["abstract_sha256"],
                "openai_label": openai["label"],
                "openai_confidence": openai["confidence"],
                "openai_rationale": openai["rationale"],
                "claude_label": claude["label"],
                "claude_confidence": claude["confidence"],
                "claude_rationale": claude["rationale"],
                "consensus_route": route,
                "consensus_reason": reason,
            }
        )

    decisions_path = output_root / str(output["decisions_basename"])
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        observed_rows = list(csv.DictReader(handle))
    require(observed_rows == expected_rows, "decision rows")
    require(exact_label_agreement == 3, "label agreement")
    require(routes["ABSTRACT_DUAL_HIGH_ACCEPT"] == 1, "accept rows")
    require(routes["ABSTRACT_DUAL_HIGH_REJECT"] == 1, "reject rows")
    require(routes["ABSTRACT_REMAINS_UNRESOLVED"] == 3, "reviewed unresolved")

    candidate_path = output_root / str(output["candidate_basename"])
    candidate = _json(candidate_path)
    require(candidate.get("previous_accept_rows") == 27, "previous accept")
    require(candidate.get("previous_reject_rows") == 21, "previous reject")
    require(candidate.get("previous_unresolved_rows") == 16, "previous unresolved")
    require(candidate.get("abstract_dual_high_accept_rows") == 1, "candidate accept added")
    require(candidate.get("abstract_dual_high_reject_rows") == 1, "candidate reject added")
    require(candidate.get("abstract_reviewed_unresolved_rows") == 3, "candidate reviewed unresolved")
    require(candidate.get("local_abstract_unavailable_or_fragment_rows") == 11, "candidate unavailable")
    require(
        (
            candidate.get("final_accept_rows"),
            candidate.get("final_reject_rows"),
            candidate.get("final_unresolved_rows"),
        )
        == (28, 22, 14),
        "final counts",
    )
    require(sum((28, 22, 14)) == candidate.get("recent_denominator_rows") == 64, "denominator")
    require(candidate.get("candidate_content_state") == "CONTENT_UNRESOLVED", "content state")
    require(candidate.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(candidate.get("provider_labels_are_ground_truth") is False, "ground truth")
    require(candidate.get("production_disposition_created") is False, "production")
    require(receipt.get("candidate") == candidate, "receipt candidate")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "receipt human")
    require(receipt.get("production_disposition_created") is False, "receipt production")
    require(receipt.get("outcomes_read") is False, "outcomes")
    return {
        "schema_version": "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "result_receipt_sha256": _sha256(receipt_path),
        "decisions_sha256": _sha256(decisions_path),
        "candidate_sha256": _sha256(candidate_path),
        "exact_label_agreement_rows": 3,
        "dual_high_accept_rows": 1,
        "dual_high_reject_rows": 1,
        "reviewed_unresolved_rows": 3,
        "unavailable_or_fragment_rows": 11,
        "final_accept_rows": 28,
        "final_reject_rows": 22,
        "final_unresolved_rows": 14,
        "final_content_state": "CONTENT_UNRESOLVED",
        "human_review": "NOT_RUN_BY_DESIGN",
        "provider_labels_are_ground_truth": False,
        "production_disposition_authorized": False,
        "outcomes_read": False,
        "g4_open": False,
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verification-output", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate(config_path=args.config, output_root=args.output)
        if not args.check_only:
            verification = args.verification_output.absolute()
            require(verification.parent == Path("/ssd"), "verification parent")
            require(not verification.exists(), "verification exists")
            _write(verification, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        AbstractCrossProviderResultVerificationError,
        execution.AbstractCrossProviderExecutionError,
        execution.transport.Rank5CrossProviderExecutionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_ABSTRACT_CROSS_PROVIDER_RESULT_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
