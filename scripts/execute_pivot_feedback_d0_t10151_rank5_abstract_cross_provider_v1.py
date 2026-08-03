#!/usr/bin/env python3
"""Execute the verified five-row local-abstract provider pair."""

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

import yaml

import execute_pivot_feedback_d0_t10151_rank5_cross_provider_v1 as transport


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.yaml"
)
DEFAULT_PREPARED = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-abstract-cross-provider-v1-prepared-20260730"
)
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-abstract-cross-provider-v1-prepared-20260730."
    "verification-v1.json"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-abstract-cross-provider-v1-execution-20260730"
)
DEFAULT_CREDENTIALS = Path(
    "/home/kimyoungjin06/Desktop/Workspace/1.2.8.TwinPaper/.env"
)
VERIFIER = (
    REPO_ROOT
    / "scripts/verify_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.py"
)
VERIFY_STATUS = "PASS_INDEPENDENT_EXACT_FIVE_LOCAL_ABSTRACT_REQUEST_VERIFICATION"
STATUS = "PASS_EXACT_FIVE_LOCAL_ABSTRACT_CROSS_PROVIDER_EXECUTION_COMPLETE"


class AbstractCrossProviderExecutionError(RuntimeError):
    """Fail-closed abstract cross-provider execution error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AbstractCrossProviderExecutionError(message)


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


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def decisions(
    text: str,
    *,
    provider: str,
    expected_ids: set[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    document = _mapping(json.loads(text), f"{provider} structured output")
    values = document.get("decisions")
    require(isinstance(values, list) and len(values) == 5, f"{provider} decision rows")
    contract = _mapping(config.get("review_contract"), "review")
    output = []
    for value in values:
        decision = _mapping(value, f"{provider} decision")
        require(
            set(decision) == {"item_id", "label", "confidence", "rationale"},
            f"{provider} decision fields",
        )
        require(decision["item_id"] in expected_ids, f"{provider} item ID")
        require(decision["label"] in contract["labels"], f"{provider} label")
        require(decision["confidence"] in contract["confidence_levels"], f"{provider} confidence")
        rationale = decision["rationale"]
        require(isinstance(rationale, str) and 1 <= len(rationale) <= int(contract["maximum_rationale_characters"]), f"{provider} rationale")
        output.append(dict(decision))
    require({value["item_id"] for value in output} == expected_ids, f"{provider} complete IDs")
    return output


def _actual_cost(
    responses: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cost = _mapping(config.get("cost_control"), "cost")
    pricing = _mapping(cost.get("pricing"), "pricing")
    values: dict[str, Any] = {}
    total = Decimal("0")
    for provider in ("openai", "claude"):
        body = _mapping(responses[provider]["response"], f"{provider} response")
        usage = _mapping(body.get("usage"), f"{provider} usage")
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        require(input_tokens > 0 and output_tokens > 0, f"{provider} usage")
        rates = _mapping(pricing[provider], f"{provider} rates")
        amount = (
            Decimal(input_tokens) * Decimal(str(rates["input_usd_per_million_tokens"]))
            + Decimal(output_tokens) * Decimal(str(rates["output_usd_per_million_tokens"]))
        ) / Decimal(1_000_000)
        total += amount
        values[provider] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_actual_usd": str(amount),
        }
    pair_cap = Decimal(str(cost["maximum_authorized_pair_usd"]))
    require(total <= pair_cap, "actual pair cost")
    prior = Decimal(str(cost["prior_conservative_cumulative_upper_bound_usd"]))
    cumulative = prior + total
    cumulative_cap = Decimal(str(cost["cumulative_authorized_cap_usd"]))
    require(cumulative <= cumulative_cap, "actual cumulative cost")
    return {
        "pricing_basis": "PROVIDER_REPORTED_USAGE_TIMES_FROZEN_STANDARD_RATES",
        "providers": values,
        "estimated_actual_pair_usd": str(total),
        "maximum_authorized_pair_usd": str(pair_cap),
        "prior_conservative_cumulative_upper_bound_usd": str(prior),
        "cumulative_actual_plus_prior_upper_bound_usd": str(cumulative),
        "cumulative_authorized_cap_usd": str(cumulative_cap),
        "within_authorized_caps": True,
    }


def execute(
    *,
    config_path: Path = DEFAULT_CONFIG,
    prepared_root: Path = DEFAULT_PREPARED,
    verification_path: Path = DEFAULT_VERIFICATION,
    output_root: Path = DEFAULT_OUTPUT,
    credentials_path: Path = DEFAULT_CREDENTIALS,
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
    execution = _mapping(config.get("execution"), "execution")
    require(execution.get("retries") == 0, "retry contract")
    prepared_root = prepared_root.resolve(strict=True)
    require(prepared_root.name == execution["prepared_root_basename"], "prepared root")
    verification_path = verification_path.resolve(strict=True)
    require(verification_path.name == execution["verification_basename"], "verification name")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == execution["execution_root_basename"], "output name")
    require(not output_root.exists(), "output exists")

    live_process = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--config",
            str(config_path),
            "--output",
            str(prepared_root),
            "--check-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(live_process.returncode == 0, "live verification")
    live = _mapping(json.loads(live_process.stdout), "live verification")
    stored = _json(verification_path)
    require(live.get("status") == VERIFY_STATUS, "live verification status")
    require(stored == live, "stored/live verification")
    preparation_receipt = _json(
        prepared_root / "abstract_cross_provider_preparation.outputs.json"
    )
    upper = _mapping(preparation_receipt.get("cost_upper_bound"), "cost upper")
    require(Decimal(str(upper["pair_upper_bound_usd"])) <= Decimal("0.040000"), "pair preflight")
    require(Decimal(str(upper["cumulative_upper_bound_usd"])) <= Decimal("0.10"), "cumulative preflight")

    with (prepared_root / "request_key.csv").open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
        item_index = header.index("item_id")
        expected_ids = {
            line.rstrip("\n").split(",")[item_index]
            for line in handle
            if line.strip()
        }
    require(len(expected_ids) == 5, "request key IDs")
    credentials = transport._read_credentials(credentials_path)
    providers = _mapping(config.get("providers"), "providers")
    payloads = {
        provider: _json(prepared_root / f"{provider}_request.json")
        for provider in ("openai", "claude")
    }
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
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    responses: dict[str, dict[str, Any]] = {}
    try:
        failures: list[Exception] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(
                    transport._request,
                    provider=provider,
                    url=str(calls[provider]["url"]),
                    headers=_mapping(calls[provider]["headers"], f"{provider} headers"),
                    payload=payloads[provider],
                    timeout=int(execution["timeout_seconds_per_provider"]),
                ): provider
                for provider in ("openai", "claude")
            }
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    responses[provider] = future.result()
                    _write(output_root / f"{provider}_response.json", responses[provider])
                except Exception as error:
                    failures.append(error)
        if failures:
            raise failures[0]
        require(set(responses) == {"openai", "claude"}, "provider responses")
        parsed = {
            "openai": decisions(
                transport._openai_text(
                    _mapping(responses["openai"]["response"], "OpenAI response")
                ),
                provider="openai",
                expected_ids=expected_ids,
                config=config,
            ),
            "claude": decisions(
                transport._claude_text(
                    _mapping(responses["claude"]["response"], "Claude response")
                ),
                provider="claude",
                expected_ids=expected_ids,
                config=config,
            ),
        }
        actual_cost = _actual_cost(responses, config)
        artifacts = []
        for provider in ("openai", "claude"):
            path = output_root / f"{provider}_response.json"
            artifacts.append(
                {
                    "basename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "decision_rows": len(parsed[provider]),
                }
            )
        manifest = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_execution_manifest.v1",
            "config_sha256": _sha256(config_path),
            "preparation_verification_sha256": _sha256(verification_path),
            "artifacts": artifacts,
        }
        _write(output_root / "execution_manifest.json", manifest)
        manifest_path = output_root / "execution_manifest.json"
        artifacts.append(
            {
                "basename": manifest_path.name,
                "bytes": manifest_path.stat().st_size,
                "sha256": _sha256(manifest_path),
            }
        )
        receipt = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_execution_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "prepared_root_basename": prepared_root.name,
            "preparation_receipt_sha256": _sha256(
                prepared_root / "abstract_cross_provider_preparation.outputs.json"
            ),
            "preparation_verification_sha256": _sha256(verification_path),
            "live_verification_status": live["status"],
            "request_rows": 5,
            "provider_calls": 2,
            "provider_responses_complete": True,
            "actual_cost": actual_cost,
            "credential_values_persisted": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "provider_labels_are_ground_truth": False,
            "production_disposition_authorized": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "artifacts": artifacts,
        }
        _write(output_root / "abstract_cross_provider_execution.outputs.json", receipt)
        return receipt
    except Exception as error:
        failure = {
            "schema_version": "pivot_feedback_d0_t10151_rank5_abstract_cross_provider_execution_failure.v1",
            "status": "BLOCKED_OR_PARTIAL_ABSTRACT_CROSS_PROVIDER_EXECUTION",
            "error_type": type(error).__name__,
            "error": str(error),
            "provider": (
                error.provider
                if isinstance(error, transport.ProviderError)
                else None
            ),
            "http_status": (
                error.status
                if isinstance(error, transport.ProviderError)
                else None
            ),
            "responses_preserved": sorted(responses),
            "credential_values_persisted": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "production_disposition_authorized": False,
            "outcomes_read": False,
        }
        _write(output_root / "execution_failure.outputs.json", failure)
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
            prepared_root=args.prepared,
            verification_path=args.verification,
            output_root=args.output,
            credentials_path=args.credentials,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        AbstractCrossProviderExecutionError,
        transport.Rank5CrossProviderExecutionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_ABSTRACT_CROSS_PROVIDER_EXECUTION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
