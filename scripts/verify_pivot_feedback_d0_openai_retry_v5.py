#!/usr/bin/env python3
"""Independently verify the fail-closed v5 OpenAI-only retry package."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_UP
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v5.yaml"
)
DEFAULT_V4_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
DEFAULT_V4_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
    ".verification-v4.json"
)
DEFAULT_V4_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-execution-20260729"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729"
)
V4_VERIFIER = (
    REPO_ROOT / "scripts/verify_pivot_feedback_d0_cross_provider_api_review_v4.py"
)
V4_VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0_CROSS_PROVIDER_CUSTOM_ID_REPAIR_NOT_SUBMITTED"
)
PREPARATION_STATUS = "PASS_D0_OPENAI_RETRY_REQUESTS_PREPARED_NOT_SUBMITTED"
VERIFY_STATUS = "PASS_INDEPENDENT_D0_OPENAI_RETRY_VERIFICATION_NOT_SUBMITTED"
OUTPUT_FILES = {
    "API_RUN_INSTRUCTIONS.md",
    "openai_retry_requests.jsonl",
    "preparation_manifest.json",
    "openai_retry_preparation.outputs.json",
}


class OpenAIRetryV5VerificationError(RuntimeError):
    """Fail-closed v5 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenAIRetryV5VerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenAIRetryV5VerificationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise OpenAIRetryV5VerificationError(
            f"invalid YAML {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _safe_repo_file(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            rows.append(_mapping(json.loads(line), f"{path.name}:{line_number}"))
        except json.JSONDecodeError as error:
            raise OpenAIRetryV5VerificationError(
                f"invalid JSONL {path.name}:{line_number}: {error}"
            ) from error
    return rows


def _artifact_tree(artifacts: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(artifacts, key=lambda value: str(value["relative_path"])):
        digest.update(str(item["relative_path"]).encode())
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode())
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _audit_openai_output(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    http = Counter()
    statuses = Counter()
    incomplete = Counter()
    identifiers = set()
    valid = 0
    missing = 0
    invalid_json = 0
    input_tokens = 0
    output_tokens = 0
    for row in rows:
        identifiers.add(row.get("custom_id"))
        response = _mapping(row.get("response"), "OpenAI response")
        http[response.get("status_code")] += 1
        body = _mapping(response.get("body"), "OpenAI response body")
        statuses[body.get("status")] += 1
        detail = body.get("incomplete_details") or {}
        if detail:
            incomplete[_mapping(detail, "incomplete details").get("reason")] += 1
        usage = _mapping(body.get("usage"), "OpenAI usage")
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        text: str | None = None
        for item in body.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text = content.get("text")
        if text is None:
            missing += 1
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            invalid_json += 1
            continue
        if set(value) == {"label", "rationale", "confidence"}:
            valid += 1
    return {
        "rows": len(rows),
        "unique_custom_ids": len(identifiers),
        "http_200_rows": http[200],
        "provider_failed_rows": sum(
            1 for row in rows if row.get("error") is not None
        ),
        "response_completed_rows": statuses["completed"],
        "response_incomplete_rows": statuses["incomplete"],
        "incomplete_max_output_token_rows": incomplete["max_output_tokens"],
        "valid_structured_rows": valid,
        "missing_output_text_rows": missing,
        "invalid_json_rows": invalid_json,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _expected_cost(
    request_path: Path,
    cost_config: Mapping[str, Any],
) -> dict[str, str]:
    input_upper = request_path.stat().st_size + (
        int(cost_config["retry_input_token_safety_overhead_per_request"]) * 384
    )
    output_upper = int(cost_config["retry_max_output_tokens_per_request"]) * 384
    rates = _mapping(
        cost_config["openai_batch_price_per_million"],
        "OpenAI rates",
    )
    retry_max = (
        Decimal(input_upper) * Decimal(str(rates["input"]))
        + Decimal(output_upper) * Decimal(str(rates["output"]))
    ) / Decimal(1_000_000)
    v4_actual = Decimal(str(cost_config["v4_actual_total_cost_usd"]))
    cumulative = v4_actual + retry_max
    quant = Decimal("0.000001")
    return {
        "v4_actual_total_cost_usd": str(
            v4_actual.quantize(quant, rounding=ROUND_UP)
        ),
        "retry_input_token_upper_bound": str(input_upper),
        "retry_output_token_upper_bound": str(output_upper),
        "retry_maximum_usd": str(retry_max.quantize(quant, rounding=ROUND_UP)),
        "cumulative_actual_plus_retry_maximum_usd": str(
            cumulative.quantize(quant, rounding=ROUND_UP)
        ),
        "cumulative_authorized_maximum_usd": str(
            Decimal(
                str(cost_config["cumulative_authorized_maximum_usd"])
            ).quantize(Decimal("0.00"))
        ),
    }


def validate(
    *,
    config_path: Path,
    v4_prepared_root: Path,
    v4_report: Path,
    v4_execution_root: Path,
    output_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = _load_yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_cross_provider_api_review_config.v5",
        "config schema differs",
    )
    require(
        config.get("status") == "FROZEN_OPENAI_ONLY_RETRY_CONTRACT_NOT_SUBMITTED",
        "config status differs",
    )
    parent = _mapping(config.get("contract_parent"), "contract parent")
    parent_path = _safe_repo_file(
        repo_root,
        parent.get("relative_path"),
        "parent config",
    )
    require(_sha256(parent_path) == parent.get("sha256"), "parent SHA differs")
    parent_config = _load_yaml(parent_path)
    require(
        parent_config.get("schema_version") == parent.get("required_schema_version"),
        "parent schema differs",
    )
    require(
        parent_config.get("status") == parent.get("required_status"),
        "parent status differs",
    )

    v4_prepared_root = v4_prepared_root.resolve(strict=True)
    v4_report = v4_report.resolve(strict=True)
    v4_execution_root = v4_execution_root.resolve(strict=True)
    v4_preparation = _mapping(config.get("v4_preparation"), "v4 preparation")
    v4_execution = _mapping(config.get("v4_execution"), "v4 execution")
    require(
        v4_prepared_root.name == v4_preparation.get("root_basename"),
        "v4 prepared root differs",
    )
    live = subprocess.run(
        [
            sys.executable,
            str(V4_VERIFIER),
            "--output-root",
            str(v4_prepared_root),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    require(live.returncode == 0, "live v4 verification failed")
    live_v4 = _mapping(json.loads(live.stdout), "live v4 verification")
    require(live_v4.get("status") == V4_VERIFY_STATUS, "v4 status differs")
    require(_load_json(v4_report) == live_v4, "v4 report differs from live result")
    v4_receipt = v4_prepared_root / "cross_provider_preparation.outputs.json"
    require(
        _sha256(v4_receipt) == v4_preparation.get("receipt_sha256"),
        "v4 receipt SHA differs",
    )
    require(
        _sha256(v4_report) == v4_preparation.get("verification_report_sha256"),
        "v4 report SHA differs",
    )
    require(
        live_v4.get("artifact_tree_sha256")
        == v4_preparation.get("artifact_tree_sha256"),
        "v4 artifact tree differs",
    )
    require(
        v4_execution_root.name == v4_execution.get("root_basename"),
        "v4 execution root differs",
    )
    v4_submission = v4_execution_root / "cross_provider_submission.outputs.json"
    require(
        _sha256(v4_submission) == v4_execution.get("submission_receipt_sha256"),
        "v4 submission receipt differs",
    )
    openai_output = v4_execution_root / "openai_output.jsonl"
    claude_results = v4_execution_root / "claude_results.jsonl"
    require(
        _sha256(openai_output) == v4_execution.get("openai_output_sha256"),
        "v4 OpenAI output differs",
    )
    require(
        _sha256(claude_results) == v4_execution.get("claude_results_sha256"),
        "v4 Claude output differs",
    )
    expected_audit = {
        "rows": 384,
        "unique_custom_ids": 384,
        "http_200_rows": 384,
        "provider_failed_rows": 0,
        "response_completed_rows": 143,
        "response_incomplete_rows": 241,
        "incomplete_max_output_token_rows": 241,
        "valid_structured_rows": 144,
        "missing_output_text_rows": 124,
        "invalid_json_rows": 116,
        "input_tokens": 120891,
        "output_tokens": 91417,
    }
    audit = _audit_openai_output(openai_output)
    require(audit == expected_audit, "v4 OpenAI audit differs")
    for key, value in expected_audit.items():
        config_key = f"openai_{key}"
        if config_key in v4_execution:
            require(v4_execution[config_key] == value, f"v4 {config_key} differs")

    raw_output = output_root.absolute()
    require(not raw_output.is_symlink(), "output root symlinked")
    output_root = output_root.resolve(strict=True)
    contract = _mapping(config.get("output_contract"), "output contract")
    require(
        output_root.name == contract.get("prepared_root_basename"),
        "output basename differs",
    )
    require(output_root.stat().st_mode & 0o777 == 0o700, "output mode differs")
    require(
        {path.name for path in output_root.iterdir()} == OUTPUT_FILES,
        "output file set differs",
    )
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid file: {path.name}")
        require(
            path.stat().st_mode & 0o777 == 0o600,
            f"file mode differs: {path.name}",
        )

    retry = _mapping(config.get("openai_retry"), "OpenAI retry")
    source_path = v4_prepared_root / str(retry["source_request_basename"])
    require(
        _sha256(source_path) == retry.get("source_request_sha256"),
        "source request SHA differs",
    )
    source_rows = _read_jsonl(source_path)
    retry_path = output_root / "openai_retry_requests.jsonl"
    retry_rows = _read_jsonl(retry_path)
    require(len(source_rows) == len(retry_rows) == 384, "request rows differ")
    retry_ids = set()
    for source_raw, repaired_raw in zip(source_rows, retry_rows, strict=True):
        source = dict(source_raw)
        repaired = dict(repaired_raw)
        require(source.get("custom_id") == repaired.get("custom_id"), "custom ID changed")
        retry_ids.add(repaired.get("custom_id"))
        source_body = _mapping(source.pop("body"), "source body")
        repaired_body = _mapping(repaired.pop("body"), "retry body")
        require(source == repaired, "request envelope changed")
        require(source_body.get("max_output_tokens") == 256, "source cap differs")
        require("reasoning" not in source_body, "source reasoning differs")
        expected_body = dict(source_body)
        expected_body["max_output_tokens"] = 1024
        expected_body["reasoning"] = {"effort": "minimal"}
        require(repaired_body == expected_body, "retry body changes differ")
    require(len(retry_ids) == 384 and None not in retry_ids, "retry IDs differ")

    manifest = _load_json(output_root / "preparation_manifest.json")
    receipt_path = output_root / "openai_retry_preparation.outputs.json"
    receipt = _load_json(receipt_path)
    require(
        manifest.get("schema_version")
        == "pivot_feedback_d0_openai_retry_manifest.v5",
        "manifest schema differs",
    )
    require(manifest.get("status") == PREPARATION_STATUS, "manifest status differs")
    require(manifest.get("config_sha256") == _sha256(config_path), "config SHA differs")
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_openai_retry_outputs.v5",
        "receipt schema differs",
    )
    require(receipt.get("status") == PREPARATION_STATUS, "receipt status differs")
    require(receipt.get("retry_request_rows") == 384, "receipt row count differs")
    require(receipt.get("retry_custom_ids_unchanged") is True, "receipt IDs differ")
    require(
        receipt.get("retry_only_reasoning_and_output_cap_changed") is True,
        "receipt change scope differs",
    )
    require(receipt.get("reasoning_effort") == "minimal", "reasoning differs")
    require(receipt.get("max_output_tokens") == 1024, "output cap differs")
    require(manifest.get("v4_openai_audit") == audit, "manifest audit differs")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list), "artifacts missing")
    require(receipt.get("artifact_count") == len(artifacts) == 3, "artifact count differs")
    for raw in artifacts:
        artifact = _mapping(raw, "artifact")
        path = output_root / str(artifact["relative_path"])
        require(path.stat().st_size == artifact.get("bytes"), f"size differs: {path.name}")
        require(_sha256(path) == artifact.get("sha256"), f"SHA differs: {path.name}")
    require(
        receipt.get("artifact_tree_sha256") == _artifact_tree(artifacts),
        "artifact tree differs",
    )
    expected_cost = _expected_cost(
        retry_path,
        _mapping(config.get("cost_control"), "cost control"),
    )
    require(receipt.get("cost_control") == expected_cost, "receipt cost differs")
    require(manifest.get("cost_control") == expected_cost, "manifest cost differs")
    require(
        Decimal(expected_cost["cumulative_actual_plus_retry_maximum_usd"])
        <= Decimal(expected_cost["cumulative_authorized_maximum_usd"])
        == Decimal("1.00"),
        "cumulative cost authorization exceeded",
    )
    for key in (
        "api_submitted",
        "api_responses_received",
        "api_review_complete",
        "human_calibration_complete",
        "candidate_universe_screen_complete",
        "d0_complete",
        "outcomes_read",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag: {key}")
    return {
        "schema_version": "pivot_feedback_d0_openai_retry_verification.v5",
        "status": VERIFY_STATUS,
        "verifier_relative_path": Path(__file__).resolve().relative_to(repo_root).as_posix(),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_path),
        "v4_verification_report_sha256": _sha256(v4_report),
        "v4_submission_receipt_sha256": _sha256(v4_submission),
        "v4_openai_output_sha256": _sha256(openai_output),
        "v4_claude_results_sha256": _sha256(claude_results),
        "v4_openai_audit_exact_match": True,
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(receipt_path),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "retry_request_rows": 384,
        "unique_expected_custom_ids": 384,
        "only_reasoning_and_output_cap_changed": True,
        "cost_control_exact_match": True,
        "retry_maximum_usd": expected_cost["retry_maximum_usd"],
        "cumulative_actual_plus_retry_maximum_usd": expected_cost[
            "cumulative_actual_plus_retry_maximum_usd"
        ],
        "authorized_maximum_total_usd": expected_cost[
            "cumulative_authorized_maximum_usd"
        ],
        "api_submitted": False,
        "api_responses_received": False,
        "human_calibration_complete": False,
        "d0_complete": False,
        "outcomes_read": False,
        "producer_module_imported": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--v4-prepared-root",
        type=Path,
        default=DEFAULT_V4_PREPARED_ROOT,
    )
    parser.add_argument("--v4-report", type=Path, default=DEFAULT_V4_REPORT)
    parser.add_argument(
        "--v4-execution-root",
        type=Path,
        default=DEFAULT_V4_EXECUTION_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            config_path=args.config,
            v4_prepared_root=args.v4_prepared_root,
            v4_report=args.v4_report,
            v4_execution_root=args.v4_execution_root,
            output_root=args.output_root,
        )
        if args.report is not None:
            report = args.report.resolve()
            require(report.parent.is_dir(), "report parent missing")
            require(not report.exists(), "report already exists")
            with report.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.write("\n")
            report.chmod(0o600)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        OpenAIRetryV5VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_OPENAI_RETRY_V5_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
