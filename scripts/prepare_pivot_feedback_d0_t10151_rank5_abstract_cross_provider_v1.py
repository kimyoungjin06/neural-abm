#!/usr/bin/env python3
"""Prepare the exact five-row local-abstract OpenAI and Claude request pair."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_UP
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
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
CONFIG_SCHEMA = "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_config.v1"
STATUS = "PASS_EXACT_FIVE_LOCAL_ABSTRACT_CROSS_PROVIDER_REQUESTS_PREPARED"
FILES = {
    "request_key.csv",
    "openai_request.json",
    "claude_request.json",
    "preparation_manifest.json",
    "abstract_cross_provider_preparation.outputs.json",
}


class AbstractCrossProviderPreparationError(RuntimeError):
    """Fail-closed abstract request preparation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AbstractCrossProviderPreparationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _nested(document: Mapping[str, Any], dotted: str) -> Any:
    value: Any = document
    for part in dotted.split("."):
        require(isinstance(value, Mapping) and part in value, f"missing {dotted}")
        value = value[part]
    return value


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def validate_config(path: Path) -> dict[str, Any]:
    config = _yaml(path.resolve(strict=True))
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status") == "FROZEN_EXACT_FIVE_LOCAL_ABSTRACT_REVIEW_NOT_SUBMITTED",
        "config status",
    )
    parent = _mapping(config.get("gate_parent"), "gate parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "gate schema")
    require(parent_doc.get("status") == parent["required_status"], "gate status")
    for dotted, expected in _mapping(parent.get("required_fields"), "gate fields").items():
        observed = _nested(parent_doc, dotted)
        require(type(observed) is type(expected) and observed == expected, f"gate field: {dotted}")
    source = _mapping(config.get("source_input"), "source")
    for key in ("producer", "verifier"):
        _repo(_mapping(source.get(key), key))
    contract = _mapping(config.get("review_contract"), "review")
    require(contract.get("exact_items") == 5, "exact items")
    require(contract.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(contract.get("provider_labels_are_ground_truth") is False, "ground truth")
    require(contract.get("consensus_accept") == "BOTH_LABEL_ON_TOPIC_AND_CONFIDENCE_HIGH", "accept")
    require(contract.get("consensus_reject") == "BOTH_LABEL_OFF_TOPIC_AND_CONFIDENCE_HIGH", "reject")
    execution = _mapping(config.get("execution"), "execution")
    require(execution.get("retries") == 0, "retries")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("submit_exact_requests") == "AUTHORIZED_AFTER_VERIFICATION_AND_COST_PREFLIGHT", "submission")
    require(authorization.get("human_review") == "NOT_RUN_BY_DESIGN", "authorization human")
    require(str(authorization.get("production_disposition")).startswith("NOT_AUTHORIZED"), "production")
    require(authorization.get("outcomes_read") is False, "outcomes")
    return config


def selected_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    source = _mapping(config.get("source_input"), "source")
    root = (Path("/ssd") / str(source["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "source root")
    parquet_path = root / str(source["parquet_basename"])
    receipt_path = root / str(source["receipt_basename"])
    verification_path = Path("/ssd") / str(source["verification_basename"])
    require(_sha256(parquet_path) == source["parquet_sha256"], "recovery parquet SHA")
    require(_sha256(receipt_path) == source["receipt_sha256"], "recovery receipt SHA")
    require(_sha256(verification_path) == source["verification_sha256"], "recovery verification SHA")
    require(_json(receipt_path).get("status") == source["receipt_status"], "recovery receipt")
    require(_json(verification_path).get("status") == source["verification_status"], "recovery verification")
    table = pq.read_table(parquet_path)
    require(table.num_rows == source["rows"] == 16, "recovery rows")
    require(
        (table.schema.metadata or {}).get(b"schema_version", b"").decode()
        == source["parquet_schema_version"],
        "recovery schema",
    )
    eligible = [
        row
        for row in table.to_pylist()
        if row["eligibility_state"] == source["selected_state"]
    ]
    require(len(eligible) == source["selected_rows"] == 5, "eligible rows")
    seed = str(_mapping(config.get("review_contract"), "review")["request_seed"])
    output: list[dict[str, str]] = []
    for row in eligible:
        openalex_id = str(row["openalex_id"])
        output.append(
            {
                "item_id": hashlib.sha256(
                    f"{seed}\0item\0{openalex_id}".encode()
                ).hexdigest(),
                "openalex_id": openalex_id,
                "title": str(row["title"]),
                "language": str(row["language"]),
                "work_type": str(row["work_type"]),
                "abstract": str(row["abstract"]),
                "abstract_sha256": str(row["abstract_sha256"]),
            }
        )
    output.sort(key=lambda row: row["item_id"])
    require(len({row["item_id"] for row in output}) == 5, "opaque IDs")
    return output


def response_schema(item_ids: Sequence[str], contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
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
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _mapping(config.get("review_contract"), "review")
    providers = _mapping(config.get("providers"), "providers")
    visible = {
        "target_topic_name": contract["target_topic_name"],
        "target_definition": contract["target_definition"],
        "items": [
            {
                "item_id": row["item_id"],
                "title": row["title"],
                "language": row["language"],
                "work_type": row["work_type"],
                "abstract": row["abstract"],
            }
            for row in rows
        ],
    }
    user_text = json.dumps(visible, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    schema = response_schema([row["item_id"] for row in rows], contract)
    claude_schema = json.loads(json.dumps(schema))
    claude_config = _mapping(providers.get("claude"), "Claude")
    for key in claude_config["remove_array_schema_keywords"]:
        claude_schema["properties"]["decisions"].pop(str(key), None)
    openai = _mapping(providers.get("openai"), "OpenAI")
    openai_body = {
        "model": openai["model"],
        "store": False,
        "max_output_tokens": openai["maximum_output_tokens"],
        "reasoning": {"effort": openai["reasoning_effort"]},
        "instructions": contract["prompt_text"],
        "input": [{"role": "user", "content": [{"type": "input_text", "text": user_text}]}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rank5_abstract_decisions",
                "strict": True,
                "schema": schema,
            }
        },
    }
    claude_body = {
        "model": claude_config["model"],
        "max_tokens": claude_config["maximum_output_tokens"],
        "system": contract["prompt_text"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        "output_config": {"format": {"type": "json_schema", "schema": claude_schema}},
    }
    return openai_body, claude_body


def cost_upper_bound(
    requests: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cost = _mapping(config.get("cost_control"), "cost")
    providers = _mapping(config.get("providers"), "providers")
    pricing = _mapping(cost.get("pricing"), "pricing")
    total = Decimal("0")
    estimates: dict[str, Any] = {}
    for provider in ("openai", "claude"):
        request_bytes = len(
            json.dumps(
                requests[provider],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        maximum_output_tokens = int(_mapping(providers[provider], provider)["maximum_output_tokens"])
        rates = _mapping(pricing[provider], f"{provider} pricing")
        amount = (
            Decimal(request_bytes) * Decimal(str(rates["input_usd_per_million_tokens"]))
            + Decimal(maximum_output_tokens)
            * Decimal(str(rates["output_usd_per_million_tokens"]))
        ) / Decimal(1_000_000)
        total += amount
        estimates[provider] = {
            "serialized_request_bytes_as_input_token_upper_bound": request_bytes,
            "maximum_output_tokens": maximum_output_tokens,
            "upper_bound_usd": str(amount.quantize(Decimal("0.000001"), rounding=ROUND_UP)),
        }
    pair_cap = Decimal(str(cost["maximum_authorized_pair_usd"]))
    require(total <= pair_cap, "pair cost upper bound")
    prior = Decimal(str(cost["prior_conservative_cumulative_upper_bound_usd"]))
    cumulative = prior + total
    cumulative_cap = Decimal(str(cost["cumulative_authorized_cap_usd"]))
    require(cumulative <= cumulative_cap, "cumulative cost upper bound")
    return {
        "method": cost["estimate_method"],
        "providers": estimates,
        "pair_upper_bound_usd": str(total.quantize(Decimal("0.000001"), rounding=ROUND_UP)),
        "maximum_authorized_pair_usd": str(pair_cap),
        "prior_conservative_cumulative_upper_bound_usd": str(prior),
        "cumulative_upper_bound_usd": str(
            cumulative.quantize(Decimal("0.000001"), rounding=ROUND_UP)
        ),
        "cumulative_authorized_cap_usd": str(cumulative_cap),
        "within_authorized_caps": True,
    }


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
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
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = validate_config(config_path)
    execution = _mapping(config.get("execution"), "execution")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == execution["prepared_root_basename"], "output name")
    require(not output_root.exists(), "output exists")
    rows = selected_rows(config)
    openai, claude = request_bodies(rows, config)
    requests = {"openai": openai, "claude": claude}
    cost = cost_upper_bound(requests, config)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    temporary.chmod(0o700)
    try:
        key_path = temporary / "request_key.csv"
        fields = [
            "item_id",
            "openalex_id",
            "title",
            "language",
            "work_type",
            "abstract_sha256",
        ]
        with key_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
        key_path.chmod(0o600)
        _write(temporary / "openai_request.json", _json_bytes(openai))
        _write(temporary / "claude_request.json", _json_bytes(claude))
        artifacts = [
            _artifact(key_path, rows=5),
            _artifact(temporary / "openai_request.json", rows=5),
            _artifact(temporary / "claude_request.json", rows=5),
        ]
        manifest = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_manifest.v1",
            "config_sha256": _sha256(config_path),
            "artifacts": artifacts,
        }
        _write(temporary / "preparation_manifest.json", _json_bytes(manifest))
        artifacts.append(_artifact(temporary / "preparation_manifest.json"))
        receipt = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_preparation_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "request_rows": 5,
            "provider_calls_prepared": 2,
            "api_submitted": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "provider_labels_are_ground_truth": False,
            "outcomes_read": False,
            "production_disposition_authorized": False,
            "cost_upper_bound": cost,
            "artifacts": artifacts,
        }
        _write(
            temporary / "abstract_cross_provider_preparation.outputs.json",
            _json_bytes(receipt),
        )
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
    except (AbstractCrossProviderPreparationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_ABSTRACT_CROSS_PROVIDER_PREPARATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
