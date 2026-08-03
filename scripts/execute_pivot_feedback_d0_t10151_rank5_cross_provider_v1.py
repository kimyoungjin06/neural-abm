#!/usr/bin/env python3
"""Execute the verified two-call rank-5 OpenAI/Claude request contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_v1.yaml"
)
DEFAULT_PREPARED = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v1-prepared-20260730"
)
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v1-prepared-20260730."
    "verification-v1.json"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v1-execution-20260730"
)
DEFAULT_CREDENTIALS = Path(
    "/home/kimyoungjin06/Desktop/Workspace/1.2.8.TwinPaper/.env"
)
VERIFIER = (
    REPO_ROOT
    / "scripts/verify_pivot_feedback_d0_t10151_rank5_cross_provider_v1.py"
)
VERIFY_STATUS = "PASS_INDEPENDENT_RANK5_25_ROW_CROSS_PROVIDER_REQUEST_VERIFICATION"
STATUS = "PASS_RANK5_CROSS_PROVIDER_TWO_CALL_EXECUTION_COMPLETE"
SUCCESSOR_STATUS = (
    "PASS_RANK5_CROSS_PROVIDER_TRANSPORT_SUCCESSOR_EXECUTION_COMPLETE"
)
CORRECTED_STATUS = (
    "PASS_RANK5_CROSS_PROVIDER_HCV_TARGET_CORRECTED_EXECUTION_COMPLETE"
)


class Rank5CrossProviderExecutionError(RuntimeError):
    """Fail-closed execution error."""


class ProviderError(Rank5CrossProviderExecutionError):
    """Provider HTTP or transport failure."""

    def __init__(self, provider: str, status: int | None, body: Any) -> None:
        super().__init__(f"{provider} request failed with HTTP {status}")
        self.provider = provider
        self.status = status
        self.body = body


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5CrossProviderExecutionError(message)


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


def _effective_config(path: Path) -> dict[str, Any]:
    raw = _yaml(path)
    schema = raw.get("schema_version")
    if schema == "pivot_feedback_d0_t10151_rank5_cross_provider_config.v1":
        return raw
    require(
        schema
        in {
            "pivot_feedback_d0_t10151_rank5_cross_provider_config.v2",
            "pivot_feedback_d0_t10151_rank5_cross_provider_config.v3",
        },
        "config schema",
    )
    parent = _mapping(raw.get("contract_parent"), "contract parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "contract parent escapes")
    require(_sha256(parent_path) == parent["sha256"], "contract parent SHA")
    inherited = _yaml(parent_path)
    require(inherited.get("schema_version") == parent["required_schema_version"], "contract parent schema")
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
    if str(schema).endswith(".v3"):
        override = _mapping(raw.get("review_contract_override"), "review override")
        review = dict(_mapping(config.get("review_contract"), "inherited review"))
        for key in ("request_seed", "target_topic_name", "target_definition"):
            review[key] = override[key]
        config["review_contract"] = review
    return config


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _read_credentials(path: Path) -> dict[str, str]:
    raw = path.absolute()
    require(not raw.is_symlink(), "credential file symlinked")
    path = path.resolve(strict=True)
    require(path.is_file(), "credential file invalid")
    require(path.stat().st_mode & 0o077 == 0, "credential permissions too broad")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key in {"OPENAI_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_API_KEY"}:
            values[key] = value
    require(bool(values.get("OPENAI_API_KEY", "").strip()), "OpenAI credential missing")
    claude = values.get("CLAUDE_API_KEY") or values.get("ANTHROPIC_API_KEY")
    require(bool(claude and claude.strip()), "Claude credential missing")
    return {"openai": values["OPENAI_API_KEY"], "claude": str(claude)}


def _live_verify(
    *,
    config: Path,
    prepared: Path,
    verification: Path,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--config",
            str(config),
            "--output",
            str(prepared),
            "--check-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, "live request verification failed")
    live = _mapping(json.loads(result.stdout), "live verification")
    require(live.get("status") == VERIFY_STATUS, "live verification status")
    stored = _json(verification.resolve(strict=True))
    require(stored == live, "stored verification differs from live verification")
    return live


def _request(
    *,
    provider: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: int,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {
                "provider": provider,
                "http_status": response.status,
                "request_id": response.headers.get("x-request-id")
                or response.headers.get("request-id"),
                "response": _mapping(body, f"{provider} response"),
            }
    except HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            body: Any = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": {"type": "non_json_error_response"}}
        raise ProviderError(provider, error.code, body) from error
    except (URLError, TimeoutError) as error:
        raise ProviderError(provider, None, {"error": {"type": type(error).__name__}}) from error


def _openai_text(body: Mapping[str, Any]) -> str:
    require(body.get("status") == "completed", "OpenAI completion status")
    require(not body.get("incomplete_details"), "OpenAI incomplete response")
    texts = [
        content.get("text")
        for output in body.get("output", [])
        if isinstance(output, Mapping) and output.get("type") == "message"
        for content in output.get("content", [])
        if isinstance(content, Mapping) and content.get("type") == "output_text"
    ]
    require(len(texts) == 1 and isinstance(texts[0], str), "OpenAI structured text")
    return str(texts[0])


def _claude_text(body: Mapping[str, Any]) -> str:
    require(body.get("stop_reason") == "end_turn", "Claude stop reason")
    content = body.get("content")
    require(isinstance(content, list) and len(content) == 1, "Claude content")
    item = _mapping(content[0], "Claude content item")
    require(item.get("type") == "text" and isinstance(item.get("text"), str), "Claude text")
    return str(item["text"])


def _basic_decisions(text: str, provider: str) -> list[dict[str, Any]]:
    value = _mapping(json.loads(text), f"{provider} structured output")
    decisions = value.get("decisions")
    require(isinstance(decisions, list) and len(decisions) == 25, f"{provider} decision rows")
    ids = []
    for decision_value in decisions:
        decision = _mapping(decision_value, f"{provider} decision")
        require(
            set(decision) == {"item_id", "label", "confidence", "rationale"},
            f"{provider} decision fields",
        )
        ids.append(decision["item_id"])
    require(len(set(ids)) == 25, f"{provider} decision IDs")
    return [dict(value) for value in decisions]


def _actual_cost(
    responses: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    pricing = _mapping(_mapping(config.get("cost_control"), "cost")["pricing"], "pricing")
    usage_values: dict[str, dict[str, Any]] = {}
    total = Decimal("0")
    for provider in ("openai", "claude"):
        body = _mapping(responses[provider]["response"], f"{provider} body")
        usage = _mapping(body.get("usage"), f"{provider} usage")
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        require(input_tokens > 0 and output_tokens > 0, f"{provider} usage")
        rates = _mapping(pricing[provider], f"{provider} pricing")
        amount = (
            Decimal(input_tokens) * Decimal(str(rates["input_usd_per_million_tokens"]))
            + Decimal(output_tokens) * Decimal(str(rates["output_usd_per_million_tokens"]))
        ) / Decimal(1_000_000)
        total += amount
        usage_values[provider] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_actual_usd": str(amount),
        }
    cap = Decimal(str(_mapping(config["cost_control"], "cost")["maximum_authorized_total_usd"]))
    require(total <= cap, "actual cost exceeds cap")
    result = {
        "pricing_basis": "PROVIDER_REPORTED_USAGE_TIMES_FROZEN_STANDARD_RATES",
        "providers": usage_values,
        "estimated_actual_total_usd": str(total),
        "authorized_cap_usd": str(cap),
        "within_authorized_cap": True,
    }
    if "cost_control_successor" in config:
        successor = _mapping(config["cost_control_successor"], "successor cost")
        prior = Decimal(str(successor["prior_attempt_upper_bound_usd"]))
        cumulative = prior + total
        cumulative_cap = Decimal(str(successor["cumulative_authorized_cap_usd"]))
        require(cumulative <= cumulative_cap, "cumulative actual-plus-prior cost")
        result.update(
            {
                "prior_attempt_conservative_upper_bound_usd": str(prior),
                "cumulative_actual_plus_prior_upper_bound_usd": str(cumulative),
                "cumulative_authorized_cap_usd": str(cumulative_cap),
                "cumulative_within_authorized_cap": True,
            }
        )
    return result


def execute(
    *,
    config_path: Path = DEFAULT_CONFIG,
    prepared: Path = DEFAULT_PREPARED,
    verification: Path = DEFAULT_VERIFICATION,
    output: Path = DEFAULT_OUTPUT,
    credentials_path: Path = DEFAULT_CREDENTIALS,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _effective_config(config_path)
    execution = _mapping(config.get("execution"), "execution")
    require(execution.get("retries") == 0, "retry contract")
    prepared = prepared.resolve(strict=True)
    require(prepared.name == execution["prepared_root_basename"], "prepared root")
    output = output.absolute()
    require(output.parent == Path("/ssd"), "execution output parent")
    require(output.name == execution["execution_root_basename"], "execution output basename")
    require(not output.exists(), "execution output already exists")
    live = _live_verify(config=config_path, prepared=prepared, verification=verification)
    preparation_receipt = _json(prepared / "cross_provider_preparation.outputs.json")
    upper = Decimal(
        str(
            _mapping(preparation_receipt.get("cost_upper_bound"), "cost upper")[
                "total_upper_bound_usd"
            ]
        )
    )
    cap = Decimal(str(_mapping(config.get("cost_control"), "cost")["maximum_authorized_total_usd"]))
    require(upper <= cap, "preflight cost upper bound")
    if "cost_control_successor" in config:
        successor_cost = _mapping(config["cost_control_successor"], "successor cost")
        cumulative_upper = (
            Decimal(str(successor_cost["prior_attempt_upper_bound_usd"])) + upper
        )
        require(
            cumulative_upper
            <= Decimal(str(successor_cost["cumulative_authorized_cap_usd"])),
            "cumulative preflight cost",
        )
    credentials = _read_credentials(credentials_path)
    providers = _mapping(config.get("providers"), "providers")
    payloads = {
        "openai": _json(prepared / "openai_request.json"),
        "claude": _json(prepared / "claude_request.json"),
    }
    timeout = int(execution["timeout_seconds_per_provider"])
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
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    responses: dict[str, dict[str, Any]] = {}
    try:
        failures: list[Exception] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(
                    _request,
                    provider=provider,
                    url=str(calls[provider]["url"]),
                    headers=_mapping(calls[provider]["headers"], f"{provider} headers"),
                    payload=payloads[provider],
                    timeout=timeout,
                ): provider
                for provider in ("openai", "claude")
            }
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    responses[provider] = future.result()
                    _write_json(
                        output / f"{provider}_response.json",
                        responses[provider],
                    )
                except Exception as error:
                    failures.append(error)
        if failures:
            raise failures[0]
        require(set(responses) == {"openai", "claude"}, "provider responses")
        openai_decisions = _basic_decisions(
            _openai_text(_mapping(responses["openai"]["response"], "OpenAI body")),
            "openai",
        )
        claude_decisions = _basic_decisions(
            _claude_text(_mapping(responses["claude"]["response"], "Claude body")),
            "claude",
        )
        actual_cost = _actual_cost(responses, config)
        artifacts = [
            {
                "basename": f"{provider}_response.json",
                "bytes": (output / f"{provider}_response.json").stat().st_size,
                "sha256": _sha256(output / f"{provider}_response.json"),
                "decision_rows": len(openai_decisions if provider == "openai" else claude_decisions),
            }
            for provider in ("openai", "claude")
        ]
        manifest = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_cross_provider_execution_manifest.v1",
            "config_sha256": _sha256(config_path),
            "prepared_verification_sha256": _sha256(verification),
            "artifacts": artifacts,
        }
        _write_json(output / "execution_manifest.json", manifest)
        artifacts.append(
            {
                "basename": "execution_manifest.json",
                "bytes": (output / "execution_manifest.json").stat().st_size,
                "sha256": _sha256(output / "execution_manifest.json"),
            }
        )
        schema = str(config.get("schema_version"))
        successor = schema.endswith(".v2")
        corrected = schema.endswith(".v3")
        receipt = {
            "schema_version": (
                "pivot_feedback_d0_t10151_rank5_cross_provider_execution_outputs.v3"
                if corrected
                else
                "pivot_feedback_d0_t10151_rank5_cross_provider_execution_outputs.v2"
                if successor
                else "pivot_feedback_d0_t10151_rank5_cross_provider_execution_outputs.v1"
            ),
            "status": (
                CORRECTED_STATUS
                if corrected
                else SUCCESSOR_STATUS
                if successor
                else STATUS
            ),
            "config_sha256": _sha256(config_path),
            "prepared_root_basename": prepared.name,
            "preparation_receipt_sha256": _sha256(
                prepared / "cross_provider_preparation.outputs.json"
            ),
            "preparation_verification_sha256": _sha256(verification),
            "live_verification_status": live["status"],
            "request_rows": 25,
            "provider_calls": 2,
            "provider_responses_complete": True,
            "actual_cost": actual_cost,
            "credential_values_persisted": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "api_labels_are_ground_truth": False,
            "augmented_disposition_authorized": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "artifacts": artifacts,
        }
        _write_json(output / "cross_provider_execution.outputs.json", receipt)
        return receipt
    except Exception as error:
        for provider, response in responses.items():
            path = output / f"{provider}_response.json"
            if not path.exists():
                _write_json(path, response)
        failure = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_cross_provider_execution_failure.v1",
            "status": "BLOCKED_OR_PARTIAL_RANK5_CROSS_PROVIDER_EXECUTION",
            "error_type": type(error).__name__,
            "error": str(error),
            "provider": error.provider if isinstance(error, ProviderError) else None,
            "http_status": error.status if isinstance(error, ProviderError) else None,
            "provider_error_body": error.body if isinstance(error, ProviderError) else None,
            "responses_preserved": sorted(responses),
            "credential_values_persisted": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "augmented_disposition_authorized": False,
            "outcomes_read": False,
        }
        _write_json(output / "execution_failure.outputs.json", failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    args = parser.parse_args(argv)
    try:
        result = execute(
            config_path=args.config,
            prepared=args.prepared,
            verification=args.verification,
            output=args.output,
            credentials_path=args.credentials,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (Rank5CrossProviderExecutionError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_CROSS_PROVIDER_EXECUTION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
