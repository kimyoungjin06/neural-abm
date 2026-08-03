#!/usr/bin/env python3
"""Prepare the fail-closed OpenAI retry after v4 incomplete responses."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_UP
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
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
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729"
)
STATUS = "PASS_D0_OPENAI_RETRY_REQUESTS_PREPARED_NOT_SUBMITTED"


class OpenAIRetryPreparationError(RuntimeError):
    """Fail-closed OpenAI retry preparation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenAIRetryPreparationError(message)


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
        raise OpenAIRetryPreparationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise OpenAIRetryPreparationError(
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
            raise OpenAIRetryPreparationError(
                f"invalid JSONL {path.name}:{line_number}: {error}"
            ) from error
    return rows


def _audit_openai_output(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    require(len(rows) == 384, "v4 OpenAI output rows differ")
    http = Counter()
    response_status = Counter()
    incomplete = Counter()
    valid = 0
    missing = 0
    invalid_json = 0
    input_tokens = 0
    output_tokens = 0
    ids = set()
    for row in rows:
        ids.add(row.get("custom_id"))
        response = _mapping(row.get("response"), "OpenAI response")
        http[response.get("status_code")] += 1
        body = _mapping(response.get("body"), "OpenAI response body")
        response_status[body.get("status")] += 1
        detail = body.get("incomplete_details") or {}
        if detail:
            incomplete[detail.get("reason")] += 1
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
    result = {
        "rows": len(rows),
        "unique_custom_ids": len(ids),
        "http_200_rows": http[200],
        "provider_failed_rows": sum(
            1 for row in rows if row.get("error") is not None
        ),
        "response_completed_rows": response_status["completed"],
        "response_incomplete_rows": response_status["incomplete"],
        "incomplete_max_output_token_rows": incomplete["max_output_tokens"],
        "valid_structured_rows": valid,
        "missing_output_text_rows": missing,
        "invalid_json_rows": invalid_json,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    require(
        result
        == {
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
        },
        "v4 OpenAI audit differs",
    )
    return result


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    path.chmod(0o600)


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


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


def _cleanup_temporary_directory(
    temporary: Path | None,
    *,
    output: Path,
) -> None:
    if temporary is None:
        return
    require(temporary.is_absolute(), "temporary path must be absolute")
    require(output.is_absolute(), "output path must be absolute")
    require(temporary != output, "temporary path equals output")
    require(temporary.parent == output.parent, "temporary path not output sibling")
    require(
        temporary.name.startswith(f".{output.name}.tmp-"),
        "temporary prefix differs",
    )
    if temporary.exists():
        require(
            temporary.is_dir() and not temporary.is_symlink(),
            "temporary path invalid",
        )
        shutil.rmtree(temporary)


def prepare(
    *,
    config_path: Path,
    v4_prepared_root: Path,
    v4_report: Path,
    v4_execution_root: Path,
    output: Path,
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
    parent = _mapping(config.get("contract_parent"), "parent")
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

    v4_prepared_root = v4_prepared_root.resolve(strict=True)
    v4_report = v4_report.resolve(strict=True)
    v4_execution_root = v4_execution_root.resolve(strict=True)
    preparation = _mapping(config.get("v4_preparation"), "v4 preparation")
    execution = _mapping(config.get("v4_execution"), "v4 execution")
    require(v4_prepared_root.name == preparation.get("root_basename"), "v4 root differs")
    require(
        _sha256(v4_prepared_root / "cross_provider_preparation.outputs.json")
        == preparation.get("receipt_sha256"),
        "v4 receipt SHA differs",
    )
    require(_sha256(v4_report) == preparation.get("verification_report_sha256"), "v4 report SHA differs")
    require(v4_execution_root.name == execution.get("root_basename"), "v4 execution root differs")
    require(
        _sha256(v4_execution_root / "cross_provider_submission.outputs.json")
        == execution.get("submission_receipt_sha256"),
        "v4 submission receipt differs",
    )
    openai_output = v4_execution_root / "openai_output.jsonl"
    claude_results = v4_execution_root / "claude_results.jsonl"
    require(_sha256(openai_output) == execution.get("openai_output_sha256"), "v4 OpenAI output differs")
    require(_sha256(claude_results) == execution.get("claude_results_sha256"), "v4 Claude output differs")
    audit = _audit_openai_output(openai_output)

    retry = _mapping(config.get("openai_retry"), "retry")
    source_path = v4_prepared_root / str(retry["source_request_basename"])
    require(_sha256(source_path) == retry.get("source_request_sha256"), "source request SHA differs")
    rows = _read_jsonl(source_path)
    require(len(rows) == retry.get("request_rows") == 384, "source rows differ")
    repaired = []
    for raw in rows:
        row = dict(raw)
        body = _mapping(row.get("body"), "request body")
        require(body.get("max_output_tokens") == 256, "parent output cap differs")
        require("reasoning" not in body, "parent reasoning already set")
        body["max_output_tokens"] = 1024
        body["reasoning"] = {"effort": "minimal"}
        row["body"] = body
        repaired.append(row)
    require(len({row["custom_id"] for row in repaired}) == 384, "retry IDs not unique")

    output = output.resolve(strict=False)
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent missing")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    temporary.chmod(0o700)
    try:
        request_path = temporary / "openai_retry_requests.jsonl"
        _write_jsonl(request_path, repaired)
        cost = _mapping(config.get("cost_control"), "cost")
        overhead = int(cost["retry_input_token_safety_overhead_per_request"]) * 384
        input_upper = request_path.stat().st_size + overhead
        output_upper = 384 * int(cost["retry_max_output_tokens_per_request"])
        rates = _mapping(cost["openai_batch_price_per_million"], "rates")
        retry_max = (
            Decimal(input_upper) * Decimal(str(rates["input"]))
            + Decimal(output_upper) * Decimal(str(rates["output"]))
        ) / Decimal(1_000_000)
        v4_actual = Decimal(str(cost["v4_actual_total_cost_usd"]))
        cumulative = v4_actual + retry_max
        quant = Decimal("0.000001")
        estimate = {
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
                Decimal(str(cost["cumulative_authorized_maximum_usd"])).quantize(
                    Decimal("0.00")
                )
            ),
        }
        require(
            Decimal(estimate["cumulative_actual_plus_retry_maximum_usd"])
            <= Decimal(estimate["cumulative_authorized_maximum_usd"]),
            "cumulative cost exceeds authorization",
        )
        instructions_path = temporary / "API_RUN_INSTRUCTIONS.md"
        instructions_path.write_text(
            "# D0 OpenAI retry v5\n\n"
            "Status: `NOT_SUBMITTED`\n\n"
            "Submit only `openai_retry_requests.jsonl`. Claude already completed "
            "with 384 valid structured responses and must not be resubmitted. "
            "Every retry request preserves its v4 ID, visible input, prompt, "
            "schema, and model; only `reasoning.effort=minimal` and "
            "`max_output_tokens=1024` differ.\n\n"
            f"Cumulative actual-plus-retry maximum: "
            f"`${estimate['cumulative_actual_plus_retry_maximum_usd']}` under "
            f"`${estimate['cumulative_authorized_maximum_usd']}`.\n",
            encoding="utf-8",
        )
        instructions_path.chmod(0o600)
        primary = [
            _artifact(instructions_path),
            _artifact(request_path, rows=384),
        ]
        manifest_path = temporary / "preparation_manifest.json"
        manifest = {
            "schema_version": "pivot_feedback_d0_openai_retry_manifest.v5",
            "status": STATUS,
            "config_relative_path": config_path.relative_to(repo_root).as_posix(),
            "config_sha256": _sha256(config_path),
            "v4_openai_audit": audit,
            "retry_request_rows": 384,
            "cost_control": estimate,
            "artifacts": primary,
            "api_submitted": False,
            "api_responses_received": False,
            "human_calibration_complete": False,
            "outcomes_read": False,
        }
        _write_json(manifest_path, manifest)
        artifacts = [*primary, _artifact(manifest_path)]
        receipt = {
            "schema_version": "pivot_feedback_d0_openai_retry_outputs.v5",
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": _artifact_tree(artifacts),
            "artifacts": artifacts,
            "v4_openai_output_sha256": _sha256(openai_output),
            "v4_claude_results_sha256": _sha256(claude_results),
            "v4_claude_valid_structured_rows": 384,
            "v4_openai_valid_structured_rows": audit["valid_structured_rows"],
            "retry_request_rows": 384,
            "retry_custom_ids_unchanged": True,
            "retry_only_reasoning_and_output_cap_changed": True,
            "reasoning_effort": "minimal",
            "max_output_tokens": 1024,
            "cost_control": estimate,
            "api_submitted": False,
            "api_responses_received": False,
            "api_review_complete": False,
            "human_calibration_complete": False,
            "candidate_universe_screen_complete": False,
            "episode_3_quarantined": True,
            "d0_complete": False,
            "outcomes_read": False,
            "g0_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        }
        _write_json(temporary / "openai_retry_preparation.outputs.json", receipt)
        require(
            all(path.stat().st_mode & 0o777 == 0o600 for path in temporary.iterdir()),
            "file mode differs",
        )
        os.replace(temporary, output)
        temporary = None
        return receipt
    finally:
        _cleanup_temporary_directory(temporary, output=output)


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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(
            config_path=args.config,
            v4_prepared_root=args.v4_prepared_root,
            v4_report=args.v4_report,
            v4_execution_root=args.v4_execution_root,
            output=args.output,
        )
        print(result["status"])
        print(f"output={args.output}")
        print(
            "cumulative_maximum_usd="
            f"{result['cost_control']['cumulative_actual_plus_retry_maximum_usd']}"
        )
        print("api_submitted=false")
        return 0
    except (
        OpenAIRetryPreparationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
