#!/usr/bin/env python3
"""Review exact external-evidence packets with OpenAI and Claude and reduce them."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from decimal import Decimal, ROUND_UP
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

import execute_pivot_feedback_d0_t10151_rank5_cross_provider_v1 as transport
import verify_pivot_feedback_d0_t10151_rank5_external_evidence_v1 as evidence_verifier


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_external_cross_provider_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-external-cross-provider-v1-20260730"
)
DEFAULT_CREDENTIALS = Path(
    "/home/kimyoungjin06/Desktop/Workspace/1.2.8.TwinPaper/.env"
)
STATUS = "PASS_EXACT_11_EXTERNAL_EVIDENCE_CROSS_PROVIDER_RESULT_REDUCED"


class ExternalCrossProviderError(RuntimeError):
    """Fail-closed external-evidence cross-provider error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalCrossProviderError(message)


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


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def validate_config(path: Path) -> dict[str, Any]:
    config = _yaml(path.resolve(strict=True))
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_t10151_rank5_external_cross_provider_config.v1",
        "config schema",
    )
    require(
        config.get("status") == "FROZEN_EXACT_11_EXTERNAL_EVIDENCE_REVIEW_AUTHORIZED",
        "config status",
    )
    gate = _mapping(config.get("gate_parent"), "gate")
    gate_doc = _yaml(_repo(gate))
    require(gate_doc.get("schema_version") == gate["required_schema_version"], "gate schema")
    require(gate_doc.get("status") == gate["required_status"], "gate status")
    evidence = _mapping(config.get("verified_evidence"), "evidence")
    for key in ("config", "producer", "verifier"):
        _repo(_mapping(evidence.get(key), key))
    contract = _mapping(config.get("review_contract"), "review")
    require(contract.get("exact_items") == 11, "exact items")
    require(contract.get("holdout_rows") == 3, "holdout rows")
    require(contract.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(contract.get("provider_labels_are_ground_truth") is False, "ground truth")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("provider_calls") == 2, "provider calls")
    require(authorization.get("retries") == 0, "retries")
    require(
        authorization.get("submit_exact_requests")
        == "AUTHORIZED_AFTER_LIVE_PACKET_VERIFICATION_AND_COST_PREFLIGHT",
        "submission",
    )
    require(str(authorization.get("production_disposition")).startswith("NOT_AUTHORIZED"), "production")
    require(authorization.get("human_review") == "NOT_RUN_BY_DESIGN", "authorization human")
    return config


def load_packet(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    binding = _mapping(_mapping(config.get("verified_evidence"), "evidence").get("operator_local"), "operator")
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "evidence root")
    packet_path = root / str(binding["packet_basename"])
    receipt_path = root / str(binding["receipt_basename"])
    verification_path = Path("/ssd") / str(binding["verification_basename"])
    require(_sha256(packet_path) == binding["packet_sha256"], "packet SHA")
    require(_sha256(receipt_path) == binding["receipt_sha256"], "evidence receipt SHA")
    require(_sha256(verification_path) == binding["verification_sha256"], "evidence verification SHA")
    require(_json(receipt_path).get("status") == binding["receipt_status"], "evidence receipt")
    stored_verification = _json(verification_path)
    require(stored_verification.get("status") == binding["verification_status"], "evidence verification")
    evidence_config = _repo(_mapping(_mapping(config["verified_evidence"], "evidence")["config"], "config"))
    live = evidence_verifier.verify(config_path=evidence_config, output_root=root)
    require(live == stored_verification, "stored/live evidence verification")
    packet = _json(packet_path)
    items = packet.get("items")
    require(isinstance(items, list) and len(items) == binding["packet_rows"] == 11, "packet rows")
    require(len({str(item["item_id"]) for item in items}) == 11, "packet item IDs")
    return [_mapping(item, "packet item") for item in items], packet


def response_schema(item_ids: Sequence[str], config: Mapping[str, Any]) -> dict[str, Any]:
    contract = _mapping(config.get("review_contract"), "review")
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 11,
                "maxItems": 11,
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string", "enum": list(item_ids)},
                        "label": {"type": "string", "enum": list(contract["labels"])},
                        "confidence": {
                            "type": "string",
                            "enum": list(contract["confidence_levels"]),
                        },
                        "rationale": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": int(contract["maximum_rationale_characters"]),
                        },
                    },
                    "required": ["item_id", "label", "confidence", "rationale"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def request_bodies(
    items: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    contract = _mapping(config.get("review_contract"), "review")
    providers = _mapping(config.get("providers"), "providers")
    visible = {
        "target_topic_name": contract["target_topic_name"],
        "target_definition": contract["target_definition"],
        "items": list(items),
    }
    user_text = json.dumps(visible, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    item_ids = [str(item["item_id"]) for item in items]
    schema = response_schema(item_ids, config)
    claude_schema = json.loads(json.dumps(schema))
    claude = _mapping(providers.get("claude"), "Claude")
    for keyword in claude["remove_array_schema_keywords"]:
        claude_schema["properties"]["decisions"].pop(str(keyword), None)
    openai = _mapping(providers.get("openai"), "OpenAI")
    return {
        "openai": {
            "model": openai["model"],
            "store": False,
            "max_output_tokens": openai["maximum_output_tokens"],
            "reasoning": {"effort": openai["reasoning_effort"]},
            "instructions": contract["prompt_text"],
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "rank5_external_evidence_decisions",
                    "strict": True,
                    "schema": schema,
                }
            },
        },
        "claude": {
            "model": claude["model"],
            "max_tokens": claude["maximum_output_tokens"],
            "system": contract["prompt_text"],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_text}]}
            ],
            "output_config": {
                "format": {"type": "json_schema", "schema": claude_schema}
            },
        },
    }


def cost_upper_bound(
    requests: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cost = _mapping(config.get("cost_control"), "cost")
    providers = _mapping(config.get("providers"), "providers")
    pricing = _mapping(cost.get("pricing"), "pricing")
    total = Decimal("0")
    details: dict[str, Any] = {}
    for provider in ("openai", "claude"):
        request_bytes = len(
            json.dumps(
                requests[provider],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        output_tokens = int(_mapping(providers[provider], provider)["maximum_output_tokens"])
        rates = _mapping(pricing[provider], f"{provider} rates")
        amount = (
            Decimal(request_bytes) * Decimal(str(rates["input_usd_per_million_tokens"]))
            + Decimal(output_tokens) * Decimal(str(rates["output_usd_per_million_tokens"]))
        ) / Decimal(1_000_000)
        total += amount
        details[provider] = {
            "serialized_request_bytes_as_input_token_upper_bound": request_bytes,
            "maximum_output_tokens": output_tokens,
            "upper_bound_usd": str(amount.quantize(Decimal("0.000001"), rounding=ROUND_UP)),
        }
    pair = total.quantize(Decimal("0.000001"), rounding=ROUND_UP)
    prior = Decimal(str(cost["prior_conservative_cumulative_upper_bound_usd"]))
    cumulative = (prior + total).quantize(Decimal("0.000001"), rounding=ROUND_UP)
    require(pair <= Decimal(str(cost["maximum_authorized_pair_usd"])), "pair preflight")
    require(cumulative <= Decimal(str(cost["cumulative_authorized_cap_usd"])), "cumulative preflight")
    return {
        "method": cost["estimate_method"],
        "providers": details,
        "pair_upper_bound_usd": str(pair),
        "maximum_authorized_pair_usd": str(cost["maximum_authorized_pair_usd"]),
        "prior_conservative_cumulative_upper_bound_usd": str(prior),
        "cumulative_upper_bound_usd": str(cumulative),
        "cumulative_authorized_cap_usd": str(cost["cumulative_authorized_cap_usd"]),
        "within_authorized_caps": True,
    }


def parse_decisions(
    text: str,
    *,
    provider: str,
    expected_ids: set[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    document = _mapping(json.loads(text), f"{provider} output")
    values = document.get("decisions")
    require(isinstance(values, list) and len(values) == 11, f"{provider} rows")
    contract = _mapping(config.get("review_contract"), "review")
    rationale_transport = _mapping(contract.get("rationale_transport"), "rationale transport")
    persisted_maximum = int(contract["maximum_rationale_characters"])
    raw_maximum = int(rationale_transport["maximum_accepted_raw_characters"])
    require(
        rationale_transport.get("persisted_normalization")
        == "DETERMINISTIC_UNICODE_PREFIX_TO_MAXIMUM_RATIONALE_CHARACTERS",
        "rationale normalization",
    )
    output = []
    for raw in values:
        decision = _mapping(raw, f"{provider} decision")
        require(set(decision) == {"item_id", "label", "confidence", "rationale"}, f"{provider} fields")
        require(decision["item_id"] in expected_ids, f"{provider} item ID")
        require(decision["label"] in contract["labels"], f"{provider} label")
        require(decision["confidence"] in contract["confidence_levels"], f"{provider} confidence")
        raw_rationale = decision["rationale"]
        require(
            isinstance(raw_rationale, str) and 1 <= len(raw_rationale) <= raw_maximum,
            f"{provider} rationale",
        )
        output.append(
            {
                **decision,
                "rationale": raw_rationale[:persisted_maximum],
                "rationale_original_characters": len(raw_rationale),
                "rationale_transport_truncated": len(raw_rationale) > persisted_maximum,
            }
        )
    require({row["item_id"] for row in output} == expected_ids, f"{provider} complete IDs")
    return output


def actual_cost(
    responses: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cost = _mapping(config.get("cost_control"), "cost")
    pricing = _mapping(cost.get("pricing"), "pricing")
    details: dict[str, Any] = {}
    total = Decimal("0")
    for provider in ("openai", "claude"):
        usage = _mapping(
            _mapping(responses[provider]["response"], f"{provider} response").get("usage"),
            f"{provider} usage",
        )
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        require(input_tokens > 0 and output_tokens > 0, f"{provider} usage")
        rates = _mapping(pricing[provider], f"{provider} rates")
        amount = (
            Decimal(input_tokens) * Decimal(str(rates["input_usd_per_million_tokens"]))
            + Decimal(output_tokens) * Decimal(str(rates["output_usd_per_million_tokens"]))
        ) / Decimal(1_000_000)
        total += amount
        details[provider] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_actual_usd": str(amount),
        }
    prior = Decimal(str(cost["prior_conservative_cumulative_upper_bound_usd"]))
    cumulative = prior + total
    require(total <= Decimal(str(cost["maximum_authorized_pair_usd"])), "actual pair cost")
    require(cumulative <= Decimal(str(cost["cumulative_authorized_cap_usd"])), "actual cumulative cost")
    return {
        "pricing_basis": "PROVIDER_REPORTED_USAGE_TIMES_FROZEN_STANDARD_RATES",
        "providers": details,
        "estimated_actual_pair_usd": str(total),
        "prior_conservative_cumulative_upper_bound_usd": str(prior),
        "cumulative_actual_plus_prior_upper_bound_usd": str(cumulative),
        "cumulative_authorized_cap_usd": str(cost["cumulative_authorized_cap_usd"]),
        "within_authorized_caps": True,
    }


def reduce_results(
    items: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item_by_id = {str(item["item_id"]): item for item in items}
    by_provider = {
        provider: {str(row["item_id"]): row for row in rows}
        for provider, rows in responses.items()
    }
    rows = []
    for item_id in sorted(item_by_id):
        item = item_by_id[item_id]
        openai = by_provider["openai"][item_id]
        claude = by_provider["claude"][item_id]
        if (
            openai["label"] == claude["label"] == "ON_TOPIC"
            and openai["confidence"] == claude["confidence"] == "HIGH"
        ):
            route = "EXTERNAL_DUAL_HIGH_ACCEPT"
            reason = "EXACT_DUAL_HIGH_ON_TOPIC"
        elif (
            openai["label"] == claude["label"] == "OFF_TOPIC"
            and openai["confidence"] == claude["confidence"] == "HIGH"
        ):
            route = "EXTERNAL_DUAL_HIGH_REJECT"
            reason = "EXACT_DUAL_HIGH_OFF_TOPIC"
        else:
            route = "EXTERNAL_REMAINS_UNRESOLVED"
            reason = (
                "LABEL_DISAGREEMENT"
                if openai["label"] != claude["label"]
                else "AGREED_LABEL_NON_DUAL_HIGH_CONFIDENCE"
            )
        rows.append(
            {
                "item_id": item_id,
                "title": str(item["title"]),
                "evidence_sha256": hashlib.sha256(
                    json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "openai_label": openai["label"],
                "openai_confidence": openai["confidence"],
                "openai_rationale": openai["rationale"],
                "openai_rationale_original_characters": openai[
                    "rationale_original_characters"
                ],
                "openai_rationale_transport_truncated": openai[
                    "rationale_transport_truncated"
                ],
                "claude_label": claude["label"],
                "claude_confidence": claude["confidence"],
                "claude_rationale": claude["rationale"],
                "claude_rationale_original_characters": claude[
                    "rationale_original_characters"
                ],
                "claude_rationale_transport_truncated": claude[
                    "rationale_transport_truncated"
                ],
                "consensus_route": route,
                "consensus_reason": reason,
            }
        )
    counts = Counter(row["consensus_route"] for row in rows)
    contract = _mapping(config.get("review_contract"), "review")
    accept_added = counts["EXTERNAL_DUAL_HIGH_ACCEPT"]
    reject_added = counts["EXTERNAL_DUAL_HIGH_REJECT"]
    reviewed_unresolved = counts["EXTERNAL_REMAINS_UNRESOLVED"]
    holdout = int(contract["holdout_rows"])
    final_accept = int(contract["current_accept_rows"]) + accept_added
    final_reject = int(contract["current_reject_rows"]) + reject_added
    final_unresolved = holdout + reviewed_unresolved
    require(final_accept + final_reject + final_unresolved == contract["recent_denominator_rows"], "denominator")
    state = (
        "CONTENT_PASS"
        if final_accept >= int(contract["pass_lower_bound_rows"])
        else "CONTENT_FAIL"
        if final_reject >= int(contract["fail_lower_bound_rows"])
        else "CONTENT_UNRESOLVED"
    )
    candidate = {
        "schema_version": "pivot_feedback_d0_t10151_rank5_external_augmented_candidate.v1",
        "field_id": 27,
        "field_candidate_rank": 5,
        "topic_id": "T10151",
        "entry_year": 1993,
        "previous_accept_rows": int(contract["current_accept_rows"]),
        "previous_reject_rows": int(contract["current_reject_rows"]),
        "previous_unresolved_rows": int(contract["current_unresolved_rows"]),
        "external_review_rows": 11,
        "external_holdout_rows": holdout,
        "external_dual_high_accept_rows": accept_added,
        "external_dual_high_reject_rows": reject_added,
        "external_reviewed_unresolved_rows": reviewed_unresolved,
        "final_accept_rows": final_accept,
        "final_reject_rows": final_reject,
        "final_unresolved_rows": final_unresolved,
        "candidate_content_state": state,
        "pass_threshold_rows": int(contract["pass_lower_bound_rows"]),
        "fail_threshold_rows": int(contract["fail_lower_bound_rows"]),
        "human_review": "NOT_RUN_BY_DESIGN",
        "provider_labels_are_ground_truth": False,
        "production_disposition_created": False,
        "selection_certificate_issued": False,
    }
    return rows, candidate


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


def execute(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
    credentials_path: Path = DEFAULT_CREDENTIALS,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = validate_config(config_path)
    items, packet = load_packet(config)
    requests = request_bodies(items, config)
    upper = cost_upper_bound(requests, config)
    output_contract = _mapping(config.get("output_contract"), "output")
    execution = _mapping(config.get("execution"), "execution")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == execution["root_basename"], "output name")
    require(not output_root.exists(), "output exists")
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    responses: dict[str, dict[str, Any]] = {}
    try:
        key_path = output_root / str(output_contract["key_basename"])
        packet_evidence = {
            str(item["item_id"]): hashlib.sha256(
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for item in items
        }
        external_root = Path("/ssd") / str(
            _mapping(_mapping(config["verified_evidence"], "evidence")["operator_local"], "operator")[
                "root_basename"
            ]
        )
        evidence_table = __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(
            external_root / "rank5_external_evidence.parquet"
        )
        evidence_by_item = {
            str(row["item_id"]): row for row in evidence_table.to_pylist() if row["review_eligible"]
        }
        with key_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["item_id", "openalex_id", "title", "evidence_sha256"],
            )
            writer.writeheader()
            for item in sorted(items, key=lambda value: str(value["item_id"])):
                item_id = str(item["item_id"])
                writer.writerow(
                    {
                        "item_id": item_id,
                        "openalex_id": evidence_by_item[item_id]["openalex_id"],
                        "title": item["title"],
                        "evidence_sha256": packet_evidence[item_id],
                    }
                )
        key_path.chmod(0o600)
        for provider in ("openai", "claude"):
            _write_json(
                output_root / str(output_contract[f"{provider}_request_basename"]),
                requests[provider],
            )
        credentials = transport._read_credentials(credentials_path)
        providers = _mapping(config.get("providers"), "providers")
        calls = {
            "openai": {
                "url": _mapping(providers["openai"], "OpenAI")["endpoint"],
                "headers": {"Authorization": f"Bearer {credentials['openai']}"},
            },
            "claude": {
                "url": _mapping(providers["claude"], "Claude")["endpoint"],
                "headers": {
                    "x-api-key": credentials["claude"],
                    "anthropic-version": "2023-06-01",
                },
            },
        }
        failures = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(
                    transport._request,
                    provider=provider,
                    url=str(calls[provider]["url"]),
                    headers=_mapping(calls[provider]["headers"], f"{provider} headers"),
                    payload=requests[provider],
                    timeout=int(execution["timeout_seconds_per_provider"]),
                ): provider
                for provider in ("openai", "claude")
            }
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    responses[provider] = future.result()
                    _write_json(
                        output_root / str(output_contract[f"{provider}_response_basename"]),
                        responses[provider],
                    )
                except Exception as error:
                    failures.append(error)
        if failures:
            raise failures[0]
        require(set(responses) == {"openai", "claude"}, "provider responses")
        expected_ids = {str(item["item_id"]) for item in items}
        parsed = {
            "openai": parse_decisions(
                transport._openai_text(_mapping(responses["openai"]["response"], "OpenAI response")),
                provider="openai",
                expected_ids=expected_ids,
                config=config,
            ),
            "claude": parse_decisions(
                transport._claude_text(_mapping(responses["claude"]["response"], "Claude response")),
                provider="claude",
                expected_ids=expected_ids,
                config=config,
            ),
        }
        cost = actual_cost(responses, config)
        rows, candidate = reduce_results(items, parsed, config)
        decisions_path = output_root / str(output_contract["decisions_basename"])
        with decisions_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        decisions_path.chmod(0o600)
        candidate_path = output_root / str(output_contract["candidate_basename"])
        _write_json(candidate_path, candidate)
        report_path = output_root / str(output_contract["report_basename"])
        report_path.write_text(
            "\n".join(
                [
                    "# Rank-5 external-evidence cross-provider result",
                    "",
                    f"Status: `{STATUS}`",
                    "",
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
                    "- human review: `NOT_RUN_BY_DESIGN`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        report_path.chmod(0o600)
        artifacts = [
            _artifact(key_path, rows=11),
            _artifact(output_root / str(output_contract["openai_request_basename"])),
            _artifact(output_root / str(output_contract["claude_request_basename"])),
            _artifact(output_root / str(output_contract["openai_response_basename"]), rows=11),
            _artifact(output_root / str(output_contract["claude_response_basename"]), rows=11),
            _artifact(decisions_path, rows=11),
            _artifact(candidate_path),
            _artifact(report_path),
        ]
        receipt = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_external_cross_provider_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "evidence_verification_status": evidence_verifier.STATUS,
            "review_packet_sha256": _sha256(
                external_root
                / str(
                    _mapping(
                        _mapping(config["verified_evidence"], "evidence")["operator_local"],
                        "operator",
                    )["packet_basename"]
                )
            ),
            "request_rows": 11,
            "holdout_rows": 3,
            "provider_calls": 2,
            "retries": 0,
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
        _write_json(output_root / str(output_contract["receipt_basename"]), receipt)
        return receipt
    except Exception as error:
        failure_path = output_root / "external_cross_provider_failure.outputs.json"
        if not failure_path.exists():
            _write_json(
                failure_path,
                {
                    "schema_version": "pivot_feedback_d0_t10151_rank5_external_cross_provider_failure.v1",
                    "status": "BLOCKED_OR_PARTIAL_EXTERNAL_CROSS_PROVIDER_EXECUTION",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "responses_preserved": sorted(responses),
                    "credential_values_persisted": False,
                    "human_review": "NOT_RUN_BY_DESIGN",
                    "production_disposition_created": False,
                    "outcomes_read": False,
                },
            )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                execute(
                    config_path=args.config,
                    output_root=args.output,
                    credentials_path=args.credentials,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        ExternalCrossProviderError,
        evidence_verifier.ExternalEvidenceVerificationError,
        transport.Rank5CrossProviderExecutionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_EXTERNAL_CROSS_PROVIDER", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
