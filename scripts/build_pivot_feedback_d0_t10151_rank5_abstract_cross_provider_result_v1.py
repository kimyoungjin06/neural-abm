#!/usr/bin/env python3
"""Reduce the five-row abstract provider pair under the frozen dual-HIGH rule."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
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
STATUS = "PASS_RANK5_ABSTRACT_CROSS_PROVIDER_RESULT_REDUCED_DISPOSITION_PENDING"


class AbstractCrossProviderResultError(RuntimeError):
    """Fail-closed abstract result reduction error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AbstractCrossProviderResultError(message)


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


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _load_inputs(config: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, list[dict[str, Any]]]]:
    request = _mapping(config.get("request_contract"), "request contract")
    request_path = _repo(request)
    request_config = _yaml(request_path)
    require(request_config.get("schema_version") == request["required_schema_version"], "request schema")
    require(request_config.get("status") == request["required_status"], "request status")
    for key in ("prepare_script", "request_verifier", "execution_script"):
        _repo(_mapping(request.get(key), key))

    prepared = _mapping(config.get("prepared_input"), "prepared")
    prepared_root = (Path("/ssd") / str(prepared["root_basename"])).resolve(strict=True)
    require(prepared_root.parent == Path("/ssd").resolve(), "prepared root")
    prep_receipt = prepared_root / str(prepared["preparation_receipt_basename"])
    verification = Path("/ssd") / str(prepared["verification_basename"])
    require(_sha256(prep_receipt) == prepared["preparation_receipt_sha256"], "preparation receipt SHA")
    require(_sha256(verification) == prepared["verification_sha256"], "verification SHA")
    require(_json(prep_receipt).get("status") == prepared["preparation_receipt_status"], "preparation status")
    require(_json(verification).get("status") == prepared["verification_status"], "verification status")
    with (prepared_root / str(prepared["key_basename"])).open(newline="", encoding="utf-8") as handle:
        key_rows = list(csv.DictReader(handle))
    require(len(key_rows) == prepared["rows"] == 5, "key rows")

    execution_input = _mapping(config.get("execution_input"), "execution")
    execution_root = (Path("/ssd") / str(execution_input["root_basename"])).resolve(strict=True)
    require(execution_root.parent == Path("/ssd").resolve(), "execution root")
    receipt_path = execution_root / str(execution_input["receipt_basename"])
    require(_sha256(receipt_path) == execution_input["receipt_sha256"], "execution receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("status") == execution_input["receipt_status"], "execution status")
    require(receipt.get("provider_calls") == execution_input["provider_calls"] == 2, "provider calls")
    require(
        str(_mapping(receipt.get("actual_cost"), "actual cost")["estimated_actual_pair_usd"])
        == str(execution_input["estimated_actual_pair_usd"]),
        "actual pair cost",
    )
    responses: dict[str, list[dict[str, Any]]] = {}
    expected_ids = {row["item_id"] for row in key_rows}
    request_config = execution._yaml(execution.DEFAULT_CONFIG)
    for provider, parser in (
        ("openai", execution.transport._openai_text),
        ("claude", execution.transport._claude_text),
    ):
        basename = execution_input[f"{provider}_response_basename"]
        path = execution_root / str(basename)
        require(_sha256(path) == execution_input[f"{provider}_response_sha256"], f"{provider} response SHA")
        response = _json(path)
        text = parser(_mapping(response["response"], f"{provider} response body"))
        responses[provider] = execution.decisions(
            text,
            provider=provider,
            expected_ids=expected_ids,
            config=request_config,
        )
        require(len(responses[provider]) == execution_input["rows_per_provider"] == 5, f"{provider} rows")
    return key_rows, responses


def reduce(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    key_rows, responses = _load_inputs(config)
    key_by_item = {row["item_id"]: row for row in key_rows}
    by_provider = {
        provider: {row["item_id"]: row for row in rows}
        for provider, rows in responses.items()
    }
    rows: list[dict[str, Any]] = []
    for item_id in sorted(key_by_item):
        key = key_by_item[item_id]
        openai = by_provider["openai"][item_id]
        claude = by_provider["claude"][item_id]
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
        rows.append(
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
    routes = Counter(row["consensus_route"] for row in rows)
    exact_label_agreement = sum(
        row["openai_label"] == row["claude_label"] for row in rows
    )
    reduction = _mapping(config.get("reduction"), "reduction")
    current = _mapping(reduction.get("current"), "current")
    expected = _mapping(reduction.get("expected"), "expected")
    accept_added = routes["ABSTRACT_DUAL_HIGH_ACCEPT"]
    reject_added = routes["ABSTRACT_DUAL_HIGH_REJECT"]
    reviewed_unresolved = routes["ABSTRACT_REMAINS_UNRESOLVED"]
    unavailable = int(expected["unavailable_or_fragment_rows"])
    final_accept = int(current["accept_rows"]) + accept_added
    final_reject = int(current["reject_rows"]) + reject_added
    final_unresolved = reviewed_unresolved + unavailable
    state = (
        "CONTENT_PASS"
        if final_accept >= int(current["pass_threshold_rows"])
        else "CONTENT_FAIL"
        if final_reject >= int(current["fail_threshold_rows"])
        else "CONTENT_UNRESOLVED"
    )
    observed = {
        "exact_label_agreement_rows": exact_label_agreement,
        "dual_high_accept_rows": accept_added,
        "dual_high_reject_rows": reject_added,
        "reviewed_remains_unresolved_rows": reviewed_unresolved,
        "unavailable_or_fragment_rows": unavailable,
        "total_previous_unresolved_rows": int(current["unresolved_rows"]),
        "final_accept_rows": final_accept,
        "final_reject_rows": final_reject,
        "final_unresolved_rows": final_unresolved,
        "final_content_state": state,
    }
    for key, value in observed.items():
        require(value == expected[key], f"expected reduction: {key}")
    require(final_accept + final_reject + final_unresolved == current["recent_denominator_rows"], "denominator")
    candidate = {
        "field_id": 27,
        "field_candidate_rank": 5,
        "topic_id": 10151,
        "entry_year": 1993,
        "recent_denominator_rows": int(current["recent_denominator_rows"]),
        "previous_accept_rows": int(current["accept_rows"]),
        "previous_reject_rows": int(current["reject_rows"]),
        "previous_unresolved_rows": int(current["unresolved_rows"]),
        "abstract_dual_high_accept_rows": accept_added,
        "abstract_dual_high_reject_rows": reject_added,
        "abstract_reviewed_unresolved_rows": reviewed_unresolved,
        "local_abstract_unavailable_or_fragment_rows": unavailable,
        "final_accept_rows": final_accept,
        "final_reject_rows": final_reject,
        "final_unresolved_rows": final_unresolved,
        "candidate_content_state": state,
        "candidate_reason": "LOCAL_ABSTRACT_DUAL_HIGH_EVIDENCE_NOT_DECISIVE",
        "human_review": "NOT_RUN_BY_DESIGN",
        "provider_labels_are_ground_truth": False,
        "production_disposition_created": False,
        "certificate_issued": False,
    }
    summary = {
        **observed,
        "openai_label_counts": dict(Counter(row["openai_label"] for row in rows)),
        "claude_label_counts": dict(Counter(row["claude_label"] for row in rows)),
        "openai_confidence_counts": dict(Counter(row["openai_confidence"] for row in rows)),
        "claude_confidence_counts": dict(Counter(row["claude_confidence"] for row in rows)),
        "consensus_route_counts": dict(routes),
        "human_review": "NOT_RUN_BY_DESIGN",
        "provider_labels_are_ground_truth": False,
    }
    return rows, candidate, summary


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"basename": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if rows is not None:
        value["rows"] = rows
    return value


def build(
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
    output = _mapping(config.get("output_contract"), "output")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == output["root_basename"], "output name")
    require(not output_root.exists(), "output exists")
    rows, candidate, summary = reduce(config)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    temporary.chmod(0o700)
    try:
        decisions_path = temporary / str(output["decisions_basename"])
        with decisions_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        decisions_path.chmod(0o600)
        candidate_path = temporary / str(output["candidate_basename"])
        summary_path = temporary / str(output["summary_basename"])
        _write(candidate_path, _json_bytes(candidate))
        _write(summary_path, _json_bytes(summary))
        report_path = temporary / str(output["report_basename"])
        _write(
            report_path,
            "\n".join(
                [
                    "# Rank-5 local-abstract cross-provider result",
                    "",
                    f"Status: `{STATUS}`",
                    "",
                    "- dual-HIGH accept added: 1",
                    "- dual-HIGH reject added: 1",
                    "- reviewed but unresolved: 3",
                    "- unavailable or fragment: 11",
                    "- final counts: 28 accept / 22 reject / 14 unresolved",
                    "- final state: `CONTENT_UNRESOLVED`",
                    "- human review: `NOT_RUN_BY_DESIGN`",
                    "",
                ]
            ).encode("utf-8"),
        )
        artifacts = [
            _artifact(decisions_path, rows=5),
            _artifact(candidate_path),
            _artifact(summary_path),
            _artifact(report_path),
        ]
        receipt = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "candidate": candidate,
            "summary": summary,
            "human_review": "NOT_RUN_BY_DESIGN",
            "provider_labels_are_ground_truth": False,
            "production_disposition_created": False,
            "selection_certificate_issued": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "artifacts": artifacts,
        }
        _write(temporary / str(output["receipt_basename"]), _json_bytes(receipt))
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            require(temporary.parent == output_root.parent, "temporary parent")
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(config_path=args.config, output_root=args.output), indent=2, sort_keys=True))
        return 0
    except (
        AbstractCrossProviderResultError,
        execution.AbstractCrossProviderExecutionError,
        execution.transport.Rank5CrossProviderExecutionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_ABSTRACT_CROSS_PROVIDER_RESULT", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
