#!/usr/bin/env python3
"""Independently verify the exact five-row abstract request pair."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_UP
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-abstract-cross-provider-v1-prepared-20260730"
)
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-abstract-cross-provider-v1-prepared-20260730."
    "verification-v1.json"
)
STATUS = "PASS_INDEPENDENT_EXACT_FIVE_LOCAL_ABSTRACT_REQUEST_VERIFICATION"


class AbstractCrossProviderVerificationError(RuntimeError):
    """Fail-closed abstract request verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AbstractCrossProviderVerificationError(message)


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


def _request_user_text(body: Mapping[str, Any], provider: str) -> str:
    if provider == "openai":
        inputs = body.get("input")
        require(isinstance(inputs, list) and len(inputs) == 1, "OpenAI input")
        content = _mapping(inputs[0], "OpenAI input").get("content")
    else:
        messages = body.get("messages")
        require(isinstance(messages, list) and len(messages) == 1, "Claude messages")
        content = _mapping(messages[0], "Claude message").get("content")
    require(isinstance(content, list) and len(content) == 1, f"{provider} content")
    item = _mapping(content[0], f"{provider} content item")
    require(isinstance(item.get("text"), str), f"{provider} user text")
    return str(item["text"])


def validate(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_config.v1",
        "config schema",
    )
    require(
        config.get("status") == "FROZEN_EXACT_FIVE_LOCAL_ABSTRACT_REVIEW_NOT_SUBMITTED",
        "config status",
    )
    parent = _mapping(config.get("gate_parent"), "parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")

    execution = _mapping(config.get("execution"), "execution")
    output_root = output_root.resolve(strict=True)
    require(output_root.parent == Path("/ssd").resolve(), "output parent")
    require(output_root.name == execution["prepared_root_basename"], "output root")
    require(output_root.stat().st_mode & 0o077 == 0, "output permissions")
    expected_files = {
        "request_key.csv",
        "openai_request.json",
        "claude_request.json",
        "preparation_manifest.json",
        "abstract_cross_provider_preparation.outputs.json",
    }
    require({path.name for path in output_root.iterdir()} == expected_files, "prepared file set")
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"prepared artifact: {path.name}")
        require(path.stat().st_mode & 0o077 == 0, f"artifact permissions: {path.name}")
    receipt_path = output_root / "abstract_cross_provider_preparation.outputs.json"
    receipt = _json(receipt_path)
    require(
        receipt.get("status")
        == "PASS_EXACT_FIVE_LOCAL_ABSTRACT_CROSS_PROVIDER_REQUESTS_PREPARED",
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    require(receipt.get("request_rows") == 5, "receipt rows")
    require(receipt.get("provider_calls_prepared") == 2, "provider calls")
    require(receipt.get("api_submitted") is False, "API submitted")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    artifacts = {
        str(item["basename"]): _mapping(item, "artifact")
        for item in receipt.get("artifacts", [])
    }
    require(
        set(artifacts)
        == {
            "request_key.csv",
            "openai_request.json",
            "claude_request.json",
            "preparation_manifest.json",
        },
        "receipt artifacts",
    )
    for basename, artifact in artifacts.items():
        require(_sha256(output_root / basename) == artifact["sha256"], f"artifact SHA: {basename}")

    source = _mapping(config.get("source_input"), "source")
    recovery_root = (Path("/ssd") / str(source["root_basename"])).resolve(strict=True)
    recovery_path = recovery_root / str(source["parquet_basename"])
    receipt_source = recovery_root / str(source["receipt_basename"])
    verification_source = Path("/ssd") / str(source["verification_basename"])
    require(_sha256(recovery_path) == source["parquet_sha256"], "recovery SHA")
    require(_sha256(receipt_source) == source["receipt_sha256"], "recovery receipt SHA")
    require(_sha256(verification_source) == source["verification_sha256"], "recovery verification SHA")
    recovery = [
        row
        for row in pq.read_table(recovery_path).to_pylist()
        if row["eligibility_state"] == source["selected_state"]
    ]
    require(len(recovery) == source["selected_rows"] == 5, "recovery eligible rows")
    recovery_by_id = {str(row["openalex_id"]): row for row in recovery}

    with (output_root / "request_key.csv").open(newline="", encoding="utf-8") as handle:
        key_rows = list(csv.DictReader(handle))
    require(len(key_rows) == 5, "key rows")
    require(len({row["item_id"] for row in key_rows}) == 5, "key item IDs")
    require({row["openalex_id"] for row in key_rows} == set(recovery_by_id), "key OpenAlex IDs")
    seed = str(_mapping(config.get("review_contract"), "review")["request_seed"])
    key_by_item: dict[str, dict[str, str]] = {}
    for row in key_rows:
        source_row = recovery_by_id[row["openalex_id"]]
        expected_item = hashlib.sha256(
            f"{seed}\0item\0{row['openalex_id']}".encode()
        ).hexdigest()
        require(row["item_id"] == expected_item, "opaque item ID")
        for key in ("title", "language", "work_type", "abstract_sha256"):
            require(row[key] == str(source_row[key]), f"key field: {key}")
        key_by_item[row["item_id"]] = row

    requests = {
        provider: _json(output_root / f"{provider}_request.json")
        for provider in ("openai", "claude")
    }
    user_texts = {
        provider: _request_user_text(requests[provider], provider)
        for provider in ("openai", "claude")
    }
    require(user_texts["openai"] == user_texts["claude"], "provider visible payload")
    visible = _mapping(json.loads(user_texts["openai"]), "visible payload")
    contract = _mapping(config.get("review_contract"), "review")
    require(visible.get("target_topic_name") == contract["target_topic_name"], "target name")
    require(visible.get("target_definition") == contract["target_definition"], "target definition")
    items = visible.get("items")
    require(isinstance(items, list) and len(items) == 5, "visible items")
    require(
        all(
            set(_mapping(item, "visible item"))
            == {"item_id", "title", "language", "work_type", "abstract"}
            for item in items
        ),
        "visible fields",
    )
    require({item["item_id"] for item in items} == set(key_by_item), "visible item IDs")
    for item_value in items:
        item = _mapping(item_value, "item")
        key = key_by_item[str(item["item_id"])]
        source_row = recovery_by_id[key["openalex_id"]]
        for field in ("title", "language", "work_type", "abstract"):
            require(item[field] == str(source_row[field]), f"visible content: {field}")
    serialized_visible = user_texts["openai"]
    for row in key_rows:
        require(row["openalex_id"] not in serialized_visible, "OpenAlex ID exposed")
    for forbidden in ("source_file", "source_row_group", "primary_topic_score", "publication_year"):
        require(forbidden not in serialized_visible, f"forbidden visible field: {forbidden}")

    providers = _mapping(config.get("providers"), "providers")
    openai = requests["openai"]
    claude = requests["claude"]
    require(openai.get("model") == _mapping(providers["openai"], "OpenAI")["model"], "OpenAI model")
    require(openai.get("store") is False and "tools" not in openai, "OpenAI boundary")
    require(claude.get("model") == _mapping(providers["claude"], "Claude")["model"], "Claude model")
    require("tools" not in claude, "Claude tools")
    require(openai.get("instructions") == contract["prompt_text"], "OpenAI prompt")
    require(claude.get("system") == contract["prompt_text"], "Claude prompt")
    openai_schema = _mapping(_mapping(_mapping(openai["text"], "text")["format"], "format")["schema"], "OpenAI schema")
    claude_schema = _mapping(_mapping(_mapping(claude["output_config"], "output")["format"], "format")["schema"], "Claude schema")
    for provider, schema in (("openai", openai_schema), ("claude", claude_schema)):
        decisions = _mapping(_mapping(schema["properties"], "properties")["decisions"], "decisions")
        item_schema = _mapping(decisions["items"], "decision item")
        properties = _mapping(item_schema["properties"], "decision properties")
        require(set(properties["item_id"]["enum"]) == set(key_by_item), f"{provider} schema IDs")
        require(properties["label"]["enum"] == contract["labels"], f"{provider} labels")
        require(properties["confidence"]["enum"] == contract["confidence_levels"], f"{provider} confidence")
        require(item_schema["additionalProperties"] is False, f"{provider} additional fields")
    openai_decisions = openai_schema["properties"]["decisions"]
    claude_decisions = claude_schema["properties"]["decisions"]
    require(openai_decisions.get("minItems") == openai_decisions.get("maxItems") == 5, "OpenAI cardinality")
    require("minItems" not in claude_decisions and "maxItems" not in claude_decisions, "Claude transport schema")

    cost = _mapping(config.get("cost_control"), "cost")
    pricing = _mapping(cost.get("pricing"), "pricing")
    total = Decimal("0")
    for provider in ("openai", "claude"):
        request_bytes = len(
            json.dumps(
                requests[provider],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        maximum_output = int(_mapping(providers[provider], provider)["maximum_output_tokens"])
        rates = _mapping(pricing[provider], f"{provider} rates")
        total += (
            Decimal(request_bytes) * Decimal(str(rates["input_usd_per_million_tokens"]))
            + Decimal(maximum_output) * Decimal(str(rates["output_usd_per_million_tokens"]))
        ) / Decimal(1_000_000)
    pair_upper = total.quantize(Decimal("0.000001"), rounding=ROUND_UP)
    prior = Decimal(str(cost["prior_conservative_cumulative_upper_bound_usd"]))
    cumulative = (prior + total).quantize(Decimal("0.000001"), rounding=ROUND_UP)
    require(pair_upper <= Decimal(str(cost["maximum_authorized_pair_usd"])), "pair cost")
    require(cumulative <= Decimal(str(cost["cumulative_authorized_cap_usd"])), "cumulative cost")
    receipt_cost = _mapping(receipt.get("cost_upper_bound"), "receipt cost")
    require(Decimal(str(receipt_cost["pair_upper_bound_usd"])) == pair_upper, "receipt pair cost")
    require(Decimal(str(receipt_cost["cumulative_upper_bound_usd"])) == cumulative, "receipt cumulative cost")
    return {
        "schema_version": "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "preparation_receipt_sha256": _sha256(receipt_path),
        "request_rows": 5,
        "provider_calls": 2,
        "visible_payload_identical": True,
        "forbidden_fields_absent": True,
        "pair_upper_bound_usd": str(pair_upper),
        "cumulative_upper_bound_usd": str(cumulative),
        "within_authorized_caps": True,
        "human_review": "NOT_RUN_BY_DESIGN",
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
            path = args.verification_output.absolute()
            require(path.parent == Path("/ssd"), "verification parent")
            require(not path.exists(), "verification exists")
            _write(path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (AbstractCrossProviderVerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_ABSTRACT_CROSS_PROVIDER_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
