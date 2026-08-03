#!/usr/bin/env python3
"""Independently verify preserved external provider responses and reduction."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

import execute_pivot_feedback_d0_t10151_rank5_cross_provider_v1 as transport


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_external_cross_provider_v1.yaml"
)
DEFAULT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-external-cross-provider-v1-20260730"
)
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-external-cross-provider-v1-20260730.verification-v1.json"
)
STATUS = "PASS_INDEPENDENT_EXACT_11_EXTERNAL_CROSS_PROVIDER_RESULT_VERIFICATION"


class ExternalCrossProviderVerificationError(RuntimeError):
    """Fail-closed external provider result verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalCrossProviderVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


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


def _provider_text(response: Mapping[str, Any], provider: str) -> str:
    if provider == "openai":
        return transport._openai_text(_mapping(response.get("response"), "OpenAI body"))
    return transport._claude_text(_mapping(response.get("response"), "Claude body"))


def verify(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_t10151_rank5_external_cross_provider_config.v1",
        "config schema",
    )
    output = _mapping(config.get("output_contract"), "output")
    root = output_root.resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "output parent")
    require(root.name == _mapping(config.get("execution"), "execution")["root_basename"], "output name")
    require(root.stat().st_mode & 0o077 == 0, "root mode")
    receipt_path = root / str(output["receipt_basename"])
    receipt = _json(receipt_path)
    require(
        receipt.get("status")
        == "PASS_EXACT_11_EXTERNAL_EVIDENCE_CROSS_PROVIDER_PRESERVED_RESPONSES_FINALIZED",
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    require(receipt.get("provider_calls") == 2 and receipt.get("retries") == 0, "provider calls")
    require(receipt.get("preserved_response_finalization") is True, "preserved finalization")
    require(receipt.get("credential_values_persisted") is False, "credentials")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(receipt.get("provider_labels_are_ground_truth") is False, "ground truth")
    require(receipt.get("production_disposition_created") is False, "production")
    for artifact_value in receipt.get("artifacts", []):
        artifact = _mapping(artifact_value, "artifact")
        path = root / str(artifact["basename"])
        require(path.is_file() and not path.is_symlink(), f"artifact: {path.name}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {path.name}")
        require(path.stat().st_mode & 0o077 == 0, f"artifact mode: {path.name}")

    with (root / str(output["key_basename"])).open(newline="", encoding="utf-8") as handle:
        key_rows = list(csv.DictReader(handle))
    require(len(key_rows) == 11, "key rows")
    expected_ids = {row["item_id"] for row in key_rows}
    require(len(expected_ids) == 11, "key item IDs")
    request_documents = {
        provider: _json(root / str(output[f"{provider}_request_basename"]))
        for provider in ("openai", "claude")
    }
    openai_text = request_documents["openai"]["input"][0]["content"][0]["text"]
    claude_text = request_documents["claude"]["messages"][0]["content"][0]["text"]
    require(openai_text == claude_text, "visible request equality")
    visible = _mapping(json.loads(openai_text), "visible request")
    items = visible.get("items")
    require(isinstance(items, list) and len(items) == 11, "visible items")
    require({str(item["item_id"]) for item in items} == expected_ids, "visible IDs")
    require("openalex.org/" not in openai_text, "OpenAlex ID leak")
    require('"pmid"' not in openai_text.lower(), "PMID key leak")

    contract = _mapping(config.get("review_contract"), "review")
    rationale_transport = _mapping(contract.get("rationale_transport"), "rationale transport")
    maximum = int(contract["maximum_rationale_characters"])
    raw_maximum = int(rationale_transport["maximum_accepted_raw_characters"])
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    observed_truncations: dict[str, int] = {}
    response_documents = {
        provider: _json(root / str(output[f"{provider}_response_basename"]))
        for provider in ("openai", "claude")
    }
    for provider in ("openai", "claude"):
        raw_output = _mapping(
            json.loads(_provider_text(response_documents[provider], provider)),
            f"{provider} output",
        )
        values = raw_output.get("decisions")
        require(isinstance(values, list) and len(values) == 11, f"{provider} decision rows")
        by_id: dict[str, dict[str, Any]] = {}
        truncations = 0
        for value in values:
            decision = _mapping(value, f"{provider} decision")
            require(set(decision) == {"item_id", "label", "confidence", "rationale"}, f"{provider} fields")
            item_id = str(decision["item_id"])
            require(item_id in expected_ids and item_id not in by_id, f"{provider} item ID")
            require(decision["label"] in contract["labels"], f"{provider} label")
            require(decision["confidence"] in contract["confidence_levels"], f"{provider} confidence")
            raw_rationale = str(decision["rationale"])
            require(1 <= len(raw_rationale) <= raw_maximum, f"{provider} rationale")
            truncations += len(raw_rationale) > maximum
            by_id[item_id] = {
                "label": str(decision["label"]),
                "confidence": str(decision["confidence"]),
                "rationale": raw_rationale[:maximum],
                "raw_characters": len(raw_rationale),
                "truncated": len(raw_rationale) > maximum,
            }
        require(set(by_id) == expected_ids, f"{provider} complete IDs")
        parsed[provider] = by_id
        observed_truncations[provider] = truncations
    require(
        receipt["rationale_transport"]["truncations"] == observed_truncations,
        "rationale truncations",
    )

    decisions_path = root / str(output["decisions_basename"])
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 11, "reduced decision rows")
    routes = Counter()
    for row in rows:
        item_id = row["item_id"]
        require(item_id in expected_ids, "reduced item ID")
        for provider in ("openai", "claude"):
            raw = parsed[provider][item_id]
            require(row[f"{provider}_label"] == raw["label"], f"{provider} reduced label")
            require(row[f"{provider}_confidence"] == raw["confidence"], f"{provider} reduced confidence")
            require(row[f"{provider}_rationale"] == raw["rationale"], f"{provider} reduced rationale")
            require(int(row[f"{provider}_rationale_original_characters"]) == raw["raw_characters"], f"{provider} rationale length")
            require(
                (row[f"{provider}_rationale_transport_truncated"] == "True")
                == raw["truncated"],
                f"{provider} truncation flag",
            )
        openai = parsed["openai"][item_id]
        claude = parsed["claude"][item_id]
        expected_route = (
            "EXTERNAL_DUAL_HIGH_ACCEPT"
            if openai["label"] == claude["label"] == "ON_TOPIC"
            and openai["confidence"] == claude["confidence"] == "HIGH"
            else "EXTERNAL_DUAL_HIGH_REJECT"
            if openai["label"] == claude["label"] == "OFF_TOPIC"
            and openai["confidence"] == claude["confidence"] == "HIGH"
            else "EXTERNAL_REMAINS_UNRESOLVED"
        )
        require(row["consensus_route"] == expected_route, "consensus route")
        routes[expected_route] += 1
    require(
        routes
        == Counter(
            {
                "EXTERNAL_DUAL_HIGH_ACCEPT": 2,
                "EXTERNAL_DUAL_HIGH_REJECT": 3,
                "EXTERNAL_REMAINS_UNRESOLVED": 6,
            }
        ),
        "route counts",
    )
    candidate_path = root / str(output["candidate_basename"])
    candidate = _json(candidate_path)
    expected_candidate = {
        "external_dual_high_accept_rows": 2,
        "external_dual_high_reject_rows": 3,
        "external_reviewed_unresolved_rows": 6,
        "external_holdout_rows": 3,
        "final_accept_rows": 30,
        "final_reject_rows": 25,
        "final_unresolved_rows": 9,
        "candidate_content_state": "CONTENT_UNRESOLVED",
    }
    for key, value in expected_candidate.items():
        require(candidate.get(key) == value, f"candidate: {key}")
    require(candidate.get("rationale_transport_truncations") == observed_truncations, "candidate truncations")
    require(candidate.get("production_disposition_created") is False, "candidate production")

    pricing = _mapping(_mapping(config.get("cost_control"), "cost").get("pricing"), "pricing")
    total = Decimal("0")
    for provider in ("openai", "claude"):
        usage = _mapping(
            _mapping(response_documents[provider].get("response"), f"{provider} response").get("usage"),
            f"{provider} usage",
        )
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
        rates = _mapping(pricing[provider], f"{provider} pricing")
        total += (
            Decimal(input_tokens) * Decimal(str(rates["input_usd_per_million_tokens"]))
            + Decimal(output_tokens) * Decimal(str(rates["output_usd_per_million_tokens"]))
        ) / Decimal(1_000_000)
    actual = _mapping(receipt.get("actual_cost"), "actual cost")
    require(Decimal(str(actual["estimated_actual_pair_usd"])) == total, "actual pair cost")
    return {
        "schema_version": "pivot_feedback_d0_t10151_rank5_external_cross_provider_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "receipt_sha256": _sha256(receipt_path),
        "decisions_sha256": _sha256(decisions_path),
        "candidate_sha256": _sha256(candidate_path),
        "provider_calls": 2,
        "retries": 0,
        "request_rows": 11,
        "holdout_rows": 3,
        "dual_high_accept_rows": 2,
        "dual_high_reject_rows": 3,
        "reviewed_unresolved_rows": 6,
        "final_accept_rows": 30,
        "final_reject_rows": 25,
        "final_unresolved_rows": 9,
        "candidate_content_state": "CONTENT_UNRESOLVED",
        "rationale_transport_truncations": observed_truncations,
        "estimated_actual_pair_usd": str(total),
        "human_review": "NOT_RUN_BY_DESIGN",
        "provider_labels_are_ground_truth": False,
        "production_disposition_created": False,
        "selection_certificate_issued": False,
        "outcomes_read": False,
        "g4_open": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--verification-output", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args(argv)
    try:
        result = verify(config_path=args.config, output_root=args.output_root)
        output = args.verification_output.absolute()
        require(output.parent == Path("/ssd"), "verification parent")
        require(not output.exists(), "verification exists")
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.chmod(0o600)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        ExternalCrossProviderVerificationError,
        transport.Rank5CrossProviderExecutionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_EXTERNAL_CROSS_PROVIDER_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
