#!/usr/bin/env python3
"""Prepare the exact 25-row rank-5 OpenAI and Claude request pair."""

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
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v1-prepared-20260730"
)
STATUS = "PASS_RANK5_25_ROW_CROSS_PROVIDER_REQUESTS_PREPARED_NOT_SUBMITTED"
SUCCESSOR_STATUS = (
    "PASS_RANK5_TRANSPORT_SUCCESSOR_REQUESTS_PREPARED_NOT_SUBMITTED"
)
CORRECTED_STATUS = (
    "PASS_RANK5_HCV_TARGET_CORRECTED_REQUESTS_PREPARED_NOT_SUBMITTED"
)
FILES = {
    "request_key.csv",
    "openai_request.json",
    "claude_request.json",
    "preparation_manifest.json",
    "cross_provider_preparation.outputs.json",
}


class Rank5CrossProviderPreparationError(RuntimeError):
    """Fail-closed request preparation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5CrossProviderPreparationError(message)


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


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _validate_config(path: Path) -> dict[str, Any]:
    raw = _load_yaml(path.resolve(strict=True))
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
        inherited = _load_yaml(parent_path)
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
        raise Rank5CrossProviderPreparationError("config schema")
    parent = _mapping(config.get("gate_parent"), "gate parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "gate parent escapes")
    require(_sha256(parent_path) == parent["sha256"], "gate parent SHA")
    parent_doc = _load_yaml(parent_path)
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "gate schema")
    require(parent_doc.get("status") == parent["required_status"], "gate status")
    for dotted, expected in _mapping(parent.get("required_fields"), "gate fields").items():
        observed = _nested(parent_doc, dotted)
        require(type(observed) is type(expected) and observed == expected, f"gate field: {dotted}")
    contract = _mapping(config.get("review_contract"), "review contract")
    require(contract.get("exact_items") == 25, "exact items")
    require(contract.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(contract.get("api_labels_are_ground_truth") is False, "ground truth")
    require(contract.get("consensus_accept") == "BOTH_LABEL_ON_TOPIC_AND_CONFIDENCE_HIGH", "accept rule")
    require(contract.get("consensus_reject") == "BOTH_LABEL_OFF_TOPIC_AND_CONFIDENCE_HIGH", "reject rule")
    cost = _mapping(config.get("cost_control"), "cost")
    require(Decimal(str(cost.get("maximum_authorized_total_usd"))) == Decimal("0.10"), "cost cap")
    require(cost.get("maximum_provider_calls") == 2, "provider calls")
    if schema in {
        "pivot_feedback_d0_t10151_rank5_cross_provider_config.v2",
        "pivot_feedback_d0_t10151_rank5_cross_provider_config.v3",
    }:
        amendment = _mapping(config.get("transport_amendment"), "transport amendment")
        require(amendment.get("provider") == "claude", "transport provider")
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
            require(
                contract.get("target_topic_name")
                == "Hepatitis C virus research",
                "corrected target",
            )
    return config


def _ssd_input(binding: Mapping[str, Any]) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "SSD root")
    path = root / str(binding["basename"])
    require(path.is_file() and not path.is_symlink(), f"invalid input: {path.name}")
    require(_sha256(path) == binding["sha256"], f"input SHA: {path.name}")
    return path


def _selected_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    sources = _mapping(config.get("source_inputs"), "sources")
    sample_binding = _mapping(sources.get("sample"), "sample")
    routes_binding = _mapping(sources.get("local_routes"), "routes")
    sample_path = _ssd_input(sample_binding)
    routes_path = _ssd_input(routes_binding)
    sample_table = pq.read_table(sample_path)
    require(sample_table.num_rows == sample_binding["rows"] == 96, "sample rows")
    metadata = sample_table.schema.metadata or {}
    require(
        metadata.get(b"schema_version", b"").decode()
        == sample_binding["schema_version"],
        "sample schema",
    )
    sample_rows = {str(row["openalex_id"]): row for row in sample_table.to_pylist()}
    require(len(sample_rows) == 96, "sample IDs")
    with routes_path.open(newline="", encoding="utf-8") as handle:
        routes = list(csv.DictReader(handle))
    require(len(routes) == routes_binding["rows"] == 96, "route rows")
    windows = {int(value) for value in routes_binding["selected_window_codes"]}
    selected_routes = [
        row
        for row in routes
        if int(row["window_code"]) in windows
        and row["local_route"] == routes_binding["selected_local_route"]
    ]
    require(len(selected_routes) == routes_binding["selected_rows"] == 25, "selected rows")
    contract = _mapping(config.get("review_contract"), "contract")
    seed = str(contract["request_seed"])
    selected: list[dict[str, str]] = []
    for route in selected_routes:
        source = sample_rows.get(route["openalex_id"])
        require(source is not None, "route/sample join")
        require(str(source["title"]) == route["title"], "route/sample title")
        require(int(source["window_code"]) == int(route["window_code"]), "route/sample window")
        openalex_id = route["openalex_id"]
        item_id = hashlib.sha256(f"{seed}\0item\0{openalex_id}".encode()).hexdigest()
        selected.append(
            {
                "item_id": item_id,
                "openalex_id": openalex_id,
                "title": str(source["title"]),
                "language": str(source["language"] or ""),
                "work_type": str(source["work_type_name"]),
            }
        )
    selected.sort(key=lambda row: row["item_id"])
    require(len({row["item_id"] for row in selected}) == 25, "opaque item IDs")
    return selected


def _response_schema(
    item_ids: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": 25,
                "maxItems": 25,
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


def _request_bodies(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _mapping(config.get("review_contract"), "contract")
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
            }
            for row in rows
        ],
    }
    user_text = json.dumps(
        visible,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    schema = _response_schema([row["item_id"] for row in rows], contract)
    claude_schema = json.loads(json.dumps(schema))
    amendment = config.get("transport_amendment")
    if amendment is not None:
        decisions = claude_schema["properties"]["decisions"]
        for key in _mapping(amendment, "transport amendment")[
            "remove_from_decisions_array_schema"
        ]:
            decisions.pop(str(key), None)
    openai = _mapping(providers.get("openai"), "OpenAI")
    claude = _mapping(providers.get("claude"), "Claude")
    openai_body = {
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
                "name": "rank5_title_decisions",
                "strict": True,
                "schema": schema,
            }
        },
    }
    claude_body = {
        "model": claude["model"],
        "max_tokens": claude["maximum_output_tokens"],
        "system": contract["prompt_text"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        "output_config": {
            "format": {"type": "json_schema", "schema": claude_schema}
        },
    }
    return openai_body, claude_body


def _cost_upper_bound(
    requests: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cost = _mapping(config.get("cost_control"), "cost")
    providers = _mapping(config.get("providers"), "providers")
    pricing = _mapping(cost.get("pricing"), "pricing")
    estimates: dict[str, Any] = {}
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
        rates = _mapping(pricing.get(provider), f"{provider} pricing")
        maximum_output_tokens = int(_mapping(providers[provider], provider)["maximum_output_tokens"])
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
    cap = Decimal(str(cost["maximum_authorized_total_usd"]))
    require(total <= cap, f"cost upper bound {total} exceeds {cap}")
    result = {
        "method": cost["estimate_method"],
        "providers": estimates,
        "total_upper_bound_usd": str(total.quantize(Decimal("0.000001"), rounding=ROUND_UP)),
        "maximum_authorized_total_usd": str(cap),
        "within_authorized_cap": True,
    }
    if "cost_control_successor" in config:
        successor = _mapping(config["cost_control_successor"], "successor cost")
        prior = Decimal(str(successor["prior_attempt_upper_bound_usd"]))
        cumulative = prior + total
        cumulative_cap = Decimal(str(successor["cumulative_authorized_cap_usd"]))
        require(cumulative <= cumulative_cap, "cumulative cost upper bound")
        result.update(
            {
                "prior_attempt_upper_bound_usd": str(prior),
                "cumulative_upper_bound_usd": str(
                    cumulative.quantize(Decimal("0.000001"), rounding=ROUND_UP)
                ),
                "cumulative_authorized_cap_usd": str(cumulative_cap),
                "cumulative_within_authorized_cap": True,
            }
        )
    return result


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
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _validate_config(config_path)
    output = output.absolute()
    require(output.parent == Path("/ssd"), "output must be a direct /ssd child")
    require(output.name == _mapping(config["execution"], "execution")["prepared_root_basename"], "output basename")
    require(not output.exists(), "output already exists")
    rows = _selected_rows(config)
    openai_body, claude_body = _request_bodies(rows, config)
    requests = {"openai": openai_body, "claude": claude_body}
    cost = _cost_upper_bound(requests, config)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        key_path = temporary / "request_key.csv"
        with key_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["item_id", "openalex_id", "title", "language", "work_type"],
            )
            writer.writeheader()
            writer.writerows(rows)
        key_path.chmod(0o600)
        _write(temporary / "openai_request.json", _canonical_json(openai_body))
        _write(temporary / "claude_request.json", _canonical_json(claude_body))
        artifacts = [
            _artifact(key_path, rows=25),
            _artifact(temporary / "openai_request.json", rows=25),
            _artifact(temporary / "claude_request.json", rows=25),
        ]
        manifest = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_cross_provider_manifest.v1",
            "config_sha256": _sha256(config_path),
            "artifacts": artifacts,
        }
        _write(temporary / "preparation_manifest.json", _canonical_json(manifest))
        artifacts.append(_artifact(temporary / "preparation_manifest.json"))
        schema = str(config.get("schema_version"))
        successor = schema.endswith(".v2")
        corrected = schema.endswith(".v3")
        receipt = {
            "schema_version": (
                "pivot_feedback_d0_t10151_rank5_cross_provider_preparation_outputs.v3"
                if corrected
                else
                "pivot_feedback_d0_t10151_rank5_cross_provider_preparation_outputs.v2"
                if successor
                else "pivot_feedback_d0_t10151_rank5_cross_provider_preparation_outputs.v1"
            ),
            "status": (
                CORRECTED_STATUS
                if corrected
                else SUCCESSOR_STATUS
                if successor
                else STATUS
            ),
            "config_sha256": _sha256(config_path),
            "request_rows": 25,
            "provider_calls_prepared": 2,
            "api_submitted": False,
            "api_responses_received": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "cost_upper_bound": cost,
            "artifacts": artifacts,
        }
        _write(
            temporary / "cross_provider_preparation.outputs.json",
            _canonical_json(receipt),
        )
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
    except (Rank5CrossProviderPreparationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_CROSS_PROVIDER_PREPARATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
