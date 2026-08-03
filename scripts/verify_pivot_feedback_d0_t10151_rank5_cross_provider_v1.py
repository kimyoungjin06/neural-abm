#!/usr/bin/env python3
"""Independently verify the prepared rank-5 cross-provider requests."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v1-prepared-20260730"
)
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v1-prepared-20260730."
    "verification-v1.json"
)
PREPARATION_STATUS = "PASS_RANK5_25_ROW_CROSS_PROVIDER_REQUESTS_PREPARED_NOT_SUBMITTED"
SUCCESSOR_PREPARATION_STATUS = (
    "PASS_RANK5_TRANSPORT_SUCCESSOR_REQUESTS_PREPARED_NOT_SUBMITTED"
)
CORRECTED_PREPARATION_STATUS = (
    "PASS_RANK5_HCV_TARGET_CORRECTED_REQUESTS_PREPARED_NOT_SUBMITTED"
)
VERIFY_STATUS = "PASS_INDEPENDENT_RANK5_25_ROW_CROSS_PROVIDER_REQUEST_VERIFICATION"
OUTPUT_FILES = {
    "request_key.csv",
    "openai_request.json",
    "claude_request.json",
    "preparation_manifest.json",
    "cross_provider_preparation.outputs.json",
}


class Rank5CrossProviderVerificationError(RuntimeError):
    """Fail-closed request verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5CrossProviderVerificationError(message)


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


def _source_file(binding: Mapping[str, Any]) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "source root")
    path = root / str(binding["basename"])
    require(path.is_file() and not path.is_symlink(), f"source invalid: {path.name}")
    require(_sha256(path) == binding["sha256"], f"source SHA: {path.name}")
    return path


def _expected_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    sources = _mapping(config.get("source_inputs"), "sources")
    sample_binding = _mapping(sources.get("sample"), "sample")
    routes_binding = _mapping(sources.get("local_routes"), "routes")
    sample = pq.read_table(_source_file(sample_binding))
    require(sample.num_rows == 96, "sample rows")
    sample_by_id = {str(row["openalex_id"]): row for row in sample.to_pylist()}
    with _source_file(routes_binding).open(newline="", encoding="utf-8") as handle:
        route_rows = list(csv.DictReader(handle))
    require(len(route_rows) == 96, "route rows")
    selected = [
        row
        for row in route_rows
        if int(row["window_code"]) in {1, 2}
        and row["local_route"] == "API_REQUIRED_AMBIGUOUS"
    ]
    require(len(selected) == 25, "selected rows")
    seed = str(_mapping(config.get("review_contract"), "contract")["request_seed"])
    expected = []
    for route in selected:
        source = sample_by_id.get(route["openalex_id"])
        require(source is not None, "source join")
        require(str(source["title"]) == route["title"], "source title")
        expected.append(
            {
                "item_id": hashlib.sha256(
                    f"{seed}\0item\0{route['openalex_id']}".encode()
                ).hexdigest(),
                "openalex_id": route["openalex_id"],
                "title": str(source["title"]),
                "language": str(source["language"] or ""),
                "work_type": str(source["work_type_name"]),
            }
        )
    expected.sort(key=lambda row: row["item_id"])
    require(len({row["item_id"] for row in expected}) == 25, "item IDs")
    return expected


def _validate_config(config_path: Path) -> dict[str, Any]:
    raw = _yaml(config_path.resolve(strict=True))
    schema = raw.get("schema_version")
    if schema == "pivot_feedback_d0_t10151_rank5_cross_provider_config.v1":
        require(
            raw.get("status")
            == "FROZEN_RANK5_25_ROW_CROSS_PROVIDER_CONTRACT_NOT_SUBMITTED",
            "config status",
        )
        config = raw
    elif schema in {
        "pivot_feedback_d0_t10151_rank5_cross_provider_config.v2",
        "pivot_feedback_d0_t10151_rank5_cross_provider_config.v3",
    }:
        require(
            raw.get("status")
            == (
                "FROZEN_RANK5_TRANSPORT_ONLY_SUCCESSOR_NOT_SUBMITTED"
                if schema.endswith(".v2")
                else "FROZEN_RANK5_HCV_TARGET_CORRECTED_SUCCESSOR_NOT_SUBMITTED"
            ),
            "config status",
        )
        contract_parent = _mapping(raw.get("contract_parent"), "contract parent")
        parent_path = (
            REPO_ROOT / str(contract_parent["relative_path"])
        ).resolve(strict=True)
        require(parent_path.is_relative_to(REPO_ROOT), "contract parent escapes")
        require(_sha256(parent_path) == contract_parent["sha256"], "contract parent SHA")
        inherited = _yaml(parent_path)
        require(
            inherited.get("schema_version")
            == contract_parent["required_schema_version"],
            "contract parent schema",
        )
        require(
            inherited.get("status") == contract_parent["required_status"],
            "contract parent status",
        )
        config = dict(inherited)
        for key in (
            "schema_version",
            "status",
            "gate_parent",
            "execution",
            "transport_amendment",
            "cost_control_successor",
        ):
            config[key] = raw[key]
        if schema.endswith(".v3"):
            override = _mapping(raw.get("review_contract_override"), "review override")
            review = dict(_mapping(config.get("review_contract"), "inherited review"))
            for key in ("request_seed", "target_topic_name", "target_definition"):
                review[key] = override[key]
            config["review_contract"] = review
    else:
        raise Rank5CrossProviderVerificationError("config schema")
    parent = _mapping(config.get("gate_parent"), "gate")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "gate escapes")
    require(_sha256(parent_path) == parent["sha256"], "gate SHA")
    gate = _yaml(parent_path)
    require(gate.get("schema_version") == parent["required_schema_version"], "gate schema")
    require(gate.get("status") == parent["required_status"], "gate status")
    for dotted, expected in _mapping(parent.get("required_fields"), "gate fields").items():
        observed = _nested(gate, dotted)
        require(type(observed) is type(expected) and observed == expected, f"gate field: {dotted}")
    contract = _mapping(config.get("review_contract"), "contract")
    require(contract.get("exact_items") == 25, "exact items")
    require(contract.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(contract.get("api_labels_are_ground_truth") is False, "ground truth")
    require(contract.get("pass_lower_bound_rows") == 32, "pass bound")
    require(contract.get("fail_lower_bound_rows") == 32, "fail bound")
    if schema in {
        "pivot_feedback_d0_t10151_rank5_cross_provider_config.v2",
        "pivot_feedback_d0_t10151_rank5_cross_provider_config.v3",
    }:
        amendment = _mapping(config.get("transport_amendment"), "transport amendment")
        require(
            amendment.get("remove_from_decisions_array_schema")
            == ["minItems", "maxItems"],
            "transport correction",
        )
        require(
            amendment.get("semantic_contract_unchanged") is True
            or amendment.get(
                "semantic_contract_unchanged_except_corrected_target_identity"
            )
            is True,
            "semantic change",
        )
        if schema.endswith(".v3"):
            contract = _mapping(config.get("review_contract"), "corrected review")
            require(contract.get("target_topic_name") == "Hepatitis C virus research", "corrected target")
    return config


def _request_text(body: Mapping[str, Any], provider: str) -> str:
    if provider == "openai":
        inputs = body.get("input")
        require(isinstance(inputs, list) and len(inputs) == 1, "OpenAI input")
        content = _mapping(inputs[0], "OpenAI input").get("content")
        require(isinstance(content, list) and len(content) == 1, "OpenAI content")
        item = _mapping(content[0], "OpenAI text")
        require(item.get("type") == "input_text", "OpenAI input type")
    else:
        messages = body.get("messages")
        require(isinstance(messages, list) and len(messages) == 1, "Claude messages")
        content = _mapping(messages[0], "Claude message").get("content")
        require(isinstance(content, list) and len(content) == 1, "Claude content")
        item = _mapping(content[0], "Claude text")
        require(item.get("type") == "text", "Claude input type")
    require(isinstance(item.get("text"), str), f"{provider} text")
    return str(item["text"])


def _schema(body: Mapping[str, Any], provider: str) -> dict[str, Any]:
    if provider == "openai":
        text = _mapping(body.get("text"), "OpenAI text format")
        fmt = _mapping(text.get("format"), "OpenAI format")
        require(fmt.get("type") == "json_schema" and fmt.get("strict") is True, "OpenAI schema mode")
        return _mapping(fmt.get("schema"), "OpenAI schema")
    output = _mapping(body.get("output_config"), "Claude output config")
    fmt = _mapping(output.get("format"), "Claude format")
    require(fmt.get("type") == "json_schema", "Claude schema mode")
    return _mapping(fmt.get("schema"), "Claude schema")


def _verify_schema(
    schema: Mapping[str, Any],
    item_ids: Sequence[str],
    config: Mapping[str, Any],
    provider: str,
) -> None:
    require(schema.get("type") == "object" and schema.get("additionalProperties") is False, "response object schema")
    require(schema.get("required") == ["decisions"], "response required")
    decisions = _mapping(_mapping(schema.get("properties"), "properties").get("decisions"), "decisions schema")
    require(decisions.get("type") == "array", "decision array")
    successor = config.get("schema_version") in {
        "pivot_feedback_d0_t10151_rank5_cross_provider_config.v2",
        "pivot_feedback_d0_t10151_rank5_cross_provider_config.v3",
    }
    if successor and provider == "claude":
        require("minItems" not in decisions and "maxItems" not in decisions, "Claude cardinality correction")
    else:
        require(
            decisions.get("minItems") == decisions.get("maxItems") == 25,
            "decision count schema",
        )
    item = _mapping(decisions.get("items"), "decision item")
    require(item.get("additionalProperties") is False, "decision extra properties")
    properties = _mapping(item.get("properties"), "decision properties")
    require(set(properties) == {"item_id", "label", "confidence", "rationale"}, "decision fields")
    require(_mapping(properties["item_id"], "item ID").get("enum") == list(item_ids), "item ID enum")
    contract = _mapping(config.get("review_contract"), "contract")
    require(_mapping(properties["label"], "label").get("enum") == contract["labels"], "label enum")
    require(
        _mapping(properties["confidence"], "confidence").get("enum")
        == contract["confidence_levels"],
        "confidence enum",
    )
    require(
        _mapping(properties["rationale"], "rationale").get("maxLength")
        == contract["maximum_rationale_characters"],
        "rationale length",
    )


def _artifacts(output: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {
            "basename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in output.iterdir()
        if path.name != "cross_provider_preparation.outputs.json"
    }


def validate(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _validate_config(config_path)
    output = output.resolve(strict=True)
    require(output.is_dir() and not output.is_symlink(), "output root")
    require(output.name == _mapping(config["execution"], "execution")["prepared_root_basename"], "output basename")
    require({path.name for path in output.iterdir()} == OUTPUT_FILES, "output files")
    for path in output.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid artifact: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"artifact mode: {path.name}")

    expected = _expected_rows(config)
    key_path = output / "request_key.csv"
    with key_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames == ["item_id", "openalex_id", "title", "language", "work_type"],
            "request key fields",
        )
        observed_key = list(reader)
    require(observed_key == expected, "request key rows")

    openai = _json(output / "openai_request.json")
    claude = _json(output / "claude_request.json")
    providers = _mapping(config.get("providers"), "providers")
    contract = _mapping(config.get("review_contract"), "contract")
    require(openai.get("model") == _mapping(providers["openai"], "OpenAI")["model"], "OpenAI model")
    require(openai.get("store") is False, "OpenAI store")
    require(openai.get("max_output_tokens") == 4096, "OpenAI max tokens")
    require(_mapping(openai.get("reasoning"), "reasoning").get("effort") == "minimal", "OpenAI reasoning")
    require(openai.get("instructions") == contract["prompt_text"], "OpenAI prompt")
    require(claude.get("model") == _mapping(providers["claude"], "Claude")["model"], "Claude model")
    require(claude.get("max_tokens") == 4096, "Claude max tokens")
    require(claude.get("system") == contract["prompt_text"], "Claude prompt")
    visible_expected = {
        "target_topic_name": contract["target_topic_name"],
        "target_definition": contract["target_definition"],
        "items": [
            {
                "item_id": row["item_id"],
                "title": row["title"],
                "language": row["language"],
                "work_type": row["work_type"],
            }
            for row in expected
        ],
    }
    openai_visible = _mapping(json.loads(_request_text(openai, "openai")), "OpenAI visible")
    claude_visible = _mapping(json.loads(_request_text(claude, "claude")), "Claude visible")
    require(openai_visible == visible_expected, "OpenAI visible payload")
    require(claude_visible == visible_expected, "Claude visible payload")
    forbidden = set(contract["forbidden_request_fields"])
    require(not (set(openai_visible) & forbidden), "OpenAI forbidden top-level field")
    require(not (set(claude_visible) & forbidden), "Claude forbidden top-level field")
    allowed_item_fields = {"item_id", "title", "language", "work_type"}
    require(
        all(set(item) == allowed_item_fields for item in openai_visible["items"]),
        "OpenAI item fields",
    )
    require(
        all(set(item) == allowed_item_fields for item in claude_visible["items"]),
        "Claude item fields",
    )
    item_ids = [row["item_id"] for row in expected]
    _verify_schema(_schema(openai, "openai"), item_ids, config, "openai")
    _verify_schema(_schema(claude, "claude"), item_ids, config, "claude")

    cost = _mapping(config.get("cost_control"), "cost")
    pricing = _mapping(cost.get("pricing"), "pricing")
    total = Decimal("0")
    for provider, body in (("openai", openai), ("claude", claude)):
        body_bytes = len(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        rates = _mapping(pricing[provider], f"{provider} pricing")
        provider_cost = (
            Decimal(body_bytes) * Decimal(str(rates["input_usd_per_million_tokens"]))
            + Decimal(4096) * Decimal(str(rates["output_usd_per_million_tokens"]))
        ) / Decimal(1_000_000)
        total += provider_cost
    require(total <= Decimal(str(cost["maximum_authorized_total_usd"])), "cost cap")
    cumulative_total: Decimal | None = None
    if "cost_control_successor" in config:
        successor_cost = _mapping(config["cost_control_successor"], "successor cost")
        cumulative_total = (
            Decimal(str(successor_cost["prior_attempt_upper_bound_usd"])) + total
        )
        require(
            cumulative_total
            <= Decimal(str(successor_cost["cumulative_authorized_cap_usd"])),
            "cumulative cost cap",
        )

    receipt = _json(output / "cross_provider_preparation.outputs.json")
    schema = str(config.get("schema_version"))
    successor = schema.endswith(".v2")
    corrected = schema.endswith(".v3")
    require(
        receipt.get("status")
        == (
            CORRECTED_PREPARATION_STATUS
            if corrected
            else
            SUCCESSOR_PREPARATION_STATUS
            if successor
            else PREPARATION_STATUS
        ),
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "receipt config SHA")
    require(receipt.get("request_rows") == 25, "receipt rows")
    require(receipt.get("provider_calls_prepared") == 2, "receipt provider calls")
    require(receipt.get("api_submitted") is False, "receipt submission")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "receipt human review")
    require(receipt.get("outcomes_read") is False, "receipt outcomes")
    manifest = _json(output / "preparation_manifest.json")
    require(manifest.get("config_sha256") == _sha256(config_path), "manifest config SHA")
    listed = {item["basename"]: item for item in manifest.get("artifacts", [])}
    actual = _artifacts(output)
    for name in ("request_key.csv", "openai_request.json", "claude_request.json"):
        require(name in listed and listed[name]["sha256"] == actual[name]["sha256"], f"manifest SHA: {name}")
        require(listed[name]["bytes"] == actual[name]["bytes"], f"manifest bytes: {name}")
    receipt_artifacts = {item["basename"]: item for item in receipt.get("artifacts", [])}
    for name in actual:
        require(name in receipt_artifacts, f"receipt artifact missing: {name}")
        require(receipt_artifacts[name]["sha256"] == actual[name]["sha256"], f"receipt SHA: {name}")
    result = {
        "schema_version": "pivot_feedback_d0_t10151_rank5_cross_provider_preparation_verification.v1",
        "status": VERIFY_STATUS,
        "config_sha256": _sha256(config_path),
        "prepared_root_basename": output.name,
        "request_rows_verified": 25,
        "provider_calls_verified": 2,
        "cost_upper_bound_usd": str(total),
        "cost_cap_usd": str(cost["maximum_authorized_total_usd"]),
        "within_cost_cap": True,
        "human_review": "NOT_RUN_BY_DESIGN",
        "api_submitted": False,
        "outcomes_read": False,
        "g4_open": False,
        "artifacts": actual,
    }
    if cumulative_total is not None:
        result.update(
            {
                "prior_attempt_results_reusable": False,
                "cumulative_cost_upper_bound_usd": str(cumulative_total),
                "cumulative_cost_cap_usd": str(
                    _mapping(config["cost_control_successor"], "successor cost")[
                        "cumulative_authorized_cap_usd"
                    ]
                ),
            }
        )
    return result


def verify(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    result = validate(config_path=config_path, output=output)
    report = report.absolute()
    require(not report.exists(), "verification report already exists")
    data = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with report.open("xb") as handle:
        handle.write(data)
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
    except (Rank5CrossProviderVerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_CROSS_PROVIDER_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
