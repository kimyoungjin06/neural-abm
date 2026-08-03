#!/usr/bin/env python3
"""Finalize preserved no-retry external provider responses after transport normalization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

import execute_pivot_feedback_d0_t10151_rank5_cross_provider_v1 as transport
import execute_pivot_feedback_d0_t10151_rank5_external_cross_provider_v1 as execution


DEFAULT_CONFIG = execution.DEFAULT_CONFIG
DEFAULT_OUTPUT = execution.DEFAULT_OUTPUT
STATUS = "PASS_EXACT_11_EXTERNAL_EVIDENCE_CROSS_PROVIDER_PRESERVED_RESPONSES_FINALIZED"


class ExternalCrossProviderFinalizationError(RuntimeError):
    """Fail-closed preserved-response finalization error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalCrossProviderFinalizationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def finalize(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = execution.validate_config(config_path)
    items, _ = execution.load_packet(config)
    expected_requests = execution.request_bodies(items, config)
    upper = execution.cost_upper_bound(expected_requests, config)
    output = _mapping(config.get("output_contract"), "output")
    root = output_root.resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "output parent")
    require(root.name == _mapping(config.get("execution"), "execution")["root_basename"], "output name")
    failure_path = root / "external_cross_provider_failure.outputs.json"
    require(failure_path.is_file(), "preserved failure receipt")
    failure = execution._json(failure_path)
    require(
        failure.get("status") == "BLOCKED_OR_PARTIAL_EXTERNAL_CROSS_PROVIDER_EXECUTION",
        "failure status",
    )
    require(failure.get("responses_preserved") == ["claude", "openai"], "preserved providers")
    for provider in ("openai", "claude"):
        request_path = root / str(output[f"{provider}_request_basename"])
        require(execution._json(request_path) == expected_requests[provider], f"{provider} request")
    responses = {
        provider: execution._json(root / str(output[f"{provider}_response_basename"]))
        for provider in ("openai", "claude")
    }
    expected_ids = {str(item["item_id"]) for item in items}
    parsed = {
        "openai": execution.parse_decisions(
            transport._openai_text(_mapping(responses["openai"]["response"], "OpenAI response")),
            provider="openai",
            expected_ids=expected_ids,
            config=config,
        ),
        "claude": execution.parse_decisions(
            transport._claude_text(_mapping(responses["claude"]["response"], "Claude response")),
            provider="claude",
            expected_ids=expected_ids,
            config=config,
        ),
    }
    cost = execution.actual_cost(responses, config)
    rows, candidate = execution.reduce_results(items, parsed, config)
    truncations = {
        provider: sum(bool(row["rationale_transport_truncated"]) for row in parsed[provider])
        for provider in ("openai", "claude")
    }
    candidate["rationale_transport_truncations"] = truncations
    decisions_path = root / str(output["decisions_basename"])
    candidate_path = root / str(output["candidate_basename"])
    report_path = root / str(output["report_basename"])
    receipt_path = root / str(output["receipt_basename"])
    for path in (decisions_path, candidate_path, report_path, receipt_path):
        require(not path.exists(), f"final artifact exists: {path.name}")
    with decisions_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decisions_path.chmod(0o600)
    execution._write_json(candidate_path, candidate)
    report_path.write_text(
        "\n".join(
            [
                "# Rank-5 external-evidence cross-provider result",
                "",
                f"Status: `{STATUS}`",
                "",
                "- provider calls: 2 completed before transport validation",
                "- provider retries: 0",
                (
                    "- deterministic rationale prefix normalization: "
                    f"OpenAI {truncations['openai']} / Claude {truncations['claude']}"
                ),
                f"- dual-HIGH accept added: {candidate['external_dual_high_accept_rows']}",
                f"- dual-HIGH reject added: {candidate['external_dual_high_reject_rows']}",
                f"- reviewed unresolved: {candidate['external_reviewed_unresolved_rows']}",
                f"- no-new-evidence holdouts: {candidate['external_holdout_rows']}",
                (
                    "- final counts: "
                    f"{candidate['final_accept_rows']} accept / "
                    f"{candidate['final_reject_rows']} reject / "
                    f"{candidate['final_unresolved_rows']} unresolved"
                ),
                f"- candidate state: `{candidate['candidate_content_state']}`",
                "- labels and confidence were not changed by transport normalization",
                "- raw provider responses remain preserved",
                "- human review: `NOT_RUN_BY_DESIGN`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report_path.chmod(0o600)
    artifacts = []
    for basename, rows_count in (
        (output["key_basename"], 11),
        (output["openai_request_basename"], None),
        (output["claude_request_basename"], None),
        (output["openai_response_basename"], 11),
        (output["claude_response_basename"], 11),
        (failure_path.name, None),
        (decisions_path.name, 11),
        (candidate_path.name, None),
        (report_path.name, None),
    ):
        artifacts.append(
            execution._artifact(root / str(basename), rows=rows_count)
        )
    external_binding = _mapping(
        _mapping(config.get("verified_evidence"), "evidence").get("operator_local"),
        "operator",
    )
    receipt = {
        "schema_version": "pivot_feedback_d0_t10151_rank5_external_cross_provider_outputs.v1",
        "status": STATUS,
        "config_sha256": execution._sha256(config_path),
        "execution_script_sha256": execution._sha256(Path(execution.__file__).resolve()),
        "finalizer_sha256": execution._sha256(Path(__file__).resolve()),
        "evidence_verification_status": execution.evidence_verifier.STATUS,
        "review_packet_sha256": external_binding["packet_sha256"],
        "request_rows": 11,
        "holdout_rows": 3,
        "provider_calls": 2,
        "retries": 0,
        "preserved_response_finalization": True,
        "rationale_transport": {
            "normalization": "DETERMINISTIC_UNICODE_PREFIX_TO_MAXIMUM_RATIONALE_CHARACTERS",
            "truncations": truncations,
            "semantic_fields_unchanged": ["item_id", "label", "confidence"],
            "raw_responses_preserved": True,
        },
        "cost_upper_bound": upper,
        "actual_cost": cost,
        "candidate": candidate,
        "artifacts": artifacts,
        "credential_values_persisted": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "provider_labels_are_ground_truth": False,
        "production_disposition_created": False,
        "selection_certificate_issued": False,
        "outcomes_read": False,
        "model_fitting_authorized": False,
        "g4_open": False,
    }
    execution._write_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                finalize(config_path=args.config, output_root=args.output),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        ExternalCrossProviderFinalizationError,
        execution.ExternalCrossProviderError,
        execution.evidence_verifier.ExternalEvidenceVerificationError,
        transport.Rank5CrossProviderExecutionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_EXTERNAL_CROSS_PROVIDER_FINALIZATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
