#!/usr/bin/env python3
"""Submit independently verified repaired v4 OpenAI and Claude batches."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v4.yaml"
)
DEFAULT_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
DEFAULT_VERIFICATION_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
    ".verification-v4.json"
)
DEFAULT_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-execution-20260729"
)
DEFAULT_CREDENTIAL_FILE = Path(
    "/home/kimyoungjin06/Desktop/Workspace/1.2.8.TwinPaper/.env"
)
VERIFIER = (
    REPO_ROOT / "scripts/verify_pivot_feedback_d0_cross_provider_api_review_v4.py"
)
BASE_SUBMITTER = (
    REPO_ROOT / "scripts/submit_pivot_feedback_d0_cross_provider_api_review_v3.py"
)
VERIFICATION_STATUS = (
    "PASS_INDEPENDENT_D0_CROSS_PROVIDER_CUSTOM_ID_REPAIR_NOT_SUBMITTED"
)
PREPARATION_STATUS = (
    "PASS_D0_CROSS_PROVIDER_CUSTOM_ID_REPAIR_PREPARED_NOT_SUBMITTED"
)
SUBMISSION_STATUS = "SUBMITTED_D0_CROSS_PROVIDER_V4_BATCHES_RESPONSES_PENDING"


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_submission_v3_base",
        BASE_SUBMITTER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base submitter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
CrossProviderV4SubmissionError = BASE.CrossProviderSubmissionError
ProviderAPIError = BASE.ProviderAPIError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossProviderV4SubmissionError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _validate_preparation(
    *,
    config_path: Path,
    prepared_root: Path,
    verification_report: Path,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--config",
            str(config_path),
            "--output-root",
            str(prepared_root),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, "independent v4 verification failed")
    live = _mapping(json.loads(result.stdout), "live v4 verification")
    require(live.get("status") == VERIFICATION_STATUS, "v4 verification status differs")
    report = BASE._load_json(verification_report.resolve(strict=True))
    require(report == live, "v4 verification report differs from live verification")
    return live


def _preflight(
    *,
    config_path: Path,
    prepared_root: Path,
    verification_report: Path,
    operator_cap_usd: Decimal,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = BASE._load_yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_cross_provider_api_review_config.v4",
        "config schema differs",
    )
    require(
        config.get("status")
        == "FROZEN_CROSS_PROVIDER_RETRY_CONTRACT_BATCH_REQUESTS_NOT_SUBMITTED",
        "config status differs",
    )
    prepared_root = prepared_root.resolve(strict=True)
    verification = _validate_preparation(
        config_path=config_path,
        prepared_root=prepared_root,
        verification_report=verification_report,
    )
    receipt_path = prepared_root / "cross_provider_preparation.outputs.json"
    receipt = BASE._load_json(receipt_path)
    require(receipt.get("status") == PREPARATION_STATUS, "preparation status differs")
    require(receipt.get("custom_id_characters") == 64, "custom ID length differs")
    require(receipt.get("custom_ids_unique_per_provider") is True, "custom IDs not unique")
    estimate = _mapping(receipt.get("cost_control"), "cost control")
    estimated = Decimal(str(estimate["total_maximum_usd"]))
    authorized = Decimal(str(estimate["authorized_maximum_usd"]))
    require(operator_cap_usd <= authorized, "operator cap exceeds frozen authorization")
    require(estimated <= operator_cap_usd, "estimated maximum exceeds operator cap")
    require(verification.get("api_submitted") is False, "verification says submitted")
    return {
        "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": BASE._sha256(config_path),
        "prepared_root_basename": prepared_root.name,
        "preparation_receipt_sha256": BASE._sha256(receipt_path),
        "verification_report_basename": verification_report.name,
        "verification_report_sha256": BASE._sha256(verification_report),
        "verifier_sha256": BASE._sha256(VERIFIER),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "openai_requests_sha256": BASE._sha256(
            prepared_root / "openai_requests.jsonl"
        ),
        "claude_batch_request_sha256": BASE._sha256(
            prepared_root / "claude_batch_request.json"
        ),
        "custom_id_characters": 64,
        "estimated_maximum_total_usd": str(estimated),
        "operator_cap_usd": str(operator_cap_usd.quantize(Decimal("0.00"))),
        "frozen_authorized_maximum_usd": str(authorized),
        "credential_values_persisted": False,
        "api_submitted": False,
        "outcomes_read": False,
    }


def submit(
    *,
    config_path: Path,
    prepared_root: Path,
    verification_report: Path,
    execution_root: Path,
    credential_file: Path,
    operator_cap_usd: Decimal,
) -> dict[str, Any]:
    preflight = _preflight(
        config_path=config_path,
        prepared_root=prepared_root,
        verification_report=verification_report,
        operator_cap_usd=operator_cap_usd,
    )
    credentials = BASE._read_credentials(credential_file)
    prepared_root = prepared_root.resolve(strict=True)
    execution_root = execution_root.resolve(strict=False)
    require(not execution_root.exists(), "execution root already exists")
    require(execution_root.parent.is_dir(), "execution parent missing")
    execution_root.mkdir(mode=0o700)
    execution_root.chmod(0o700)
    BASE._write_json(
        execution_root / "submission_context.json",
        {
            "schema_version": "pivot_feedback_d0_cross_provider_submission_context.v4",
            "status": "PREFLIGHT_PASS_SUBMISSION_STARTED",
            **preflight,
        },
    )

    openai_batch_id: str | None = None
    try:
        upload = BASE._upload_openai_file(
            api_key=credentials["OPENAI_API_KEY"],
            path=prepared_root / "openai_requests.jsonl",
        )
        BASE._write_json(
            execution_root / "openai_file_upload.response.json",
            upload,
        )
        upload_body = _mapping(upload["response"], "OpenAI upload response")
        file_id = str(upload_body.get("id", ""))
        require(file_id.startswith("file-"), "OpenAI file ID missing")
        require(upload_body.get("purpose") == "batch", "OpenAI purpose differs")

        openai_batch = BASE._post_json(
            provider="OpenAI batch creation",
            url="https://api.openai.com/v1/batches",
            headers={"Authorization": f"Bearer {credentials['OPENAI_API_KEY']}"},
            payload={
                "input_file_id": file_id,
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": {
                    "study": "pivot-feedback-d0",
                    "contract": "cross-provider-v4",
                    "judge": "openai",
                },
            },
        )
        BASE._write_json(
            execution_root / "openai_batch_create.response.json",
            openai_batch,
        )
        openai_body = _mapping(openai_batch["response"], "OpenAI batch response")
        openai_batch_id = str(openai_body.get("id", ""))
        require(openai_batch_id.startswith("batch_"), "OpenAI batch ID missing")

        claude_payload = BASE._load_json(
            prepared_root / "claude_batch_request.json"
        )
        claude_batch = BASE._post_json(
            provider="Claude batch creation",
            url="https://api.anthropic.com/v1/messages/batches",
            headers={
                "x-api-key": credentials["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            payload=claude_payload,
        )
        BASE._write_json(
            execution_root / "claude_batch_create.response.json",
            claude_batch,
        )
        claude_body = _mapping(claude_batch["response"], "Claude batch response")
        claude_batch_id = str(claude_body.get("id", ""))
        require(claude_batch_id.startswith("msgbatch_"), "Claude batch ID missing")
        require(
            claude_body.get("processing_status") == "in_progress",
            "Claude batch not in progress",
        )

        receipt = {
            "schema_version": "pivot_feedback_d0_cross_provider_submission_outputs.v4",
            "status": SUBMISSION_STATUS,
            **preflight,
            "openai": {
                "input_file_id": file_id,
                "batch_id": openai_batch_id,
                "status": openai_body.get("status"),
                "request_id": openai_batch.get("request_id"),
            },
            "claude": {
                "batch_id": claude_batch_id,
                "processing_status": claude_body.get("processing_status"),
                "request_id": claude_batch.get("request_id"),
            },
            "credential_values_persisted": False,
            "api_submitted": True,
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
        BASE._write_json(
            execution_root / "cross_provider_submission.outputs.json",
            receipt,
        )
        return receipt
    except Exception as error:
        cancellation: dict[str, Any] | None = None
        if openai_batch_id is not None:
            try:
                cancellation = BASE._cancel_openai_batch(
                    api_key=credentials["OPENAI_API_KEY"],
                    batch_id=openai_batch_id,
                )
                BASE._write_json(
                    execution_root / "openai_batch_cancel.response.json",
                    cancellation,
                )
            except Exception as cancel_error:
                cancellation = {
                    "status": "CANCELLATION_FAILED",
                    "error_type": type(cancel_error).__name__,
                }
                BASE._write_json(
                    execution_root / "openai_batch_cancel_failure.json",
                    cancellation,
                )
        failure = {
            "schema_version": "pivot_feedback_d0_cross_provider_submission_failure.v4",
            "status": "BLOCKED_OR_PARTIAL_CROSS_PROVIDER_V4_SUBMISSION",
            **preflight,
            "error_type": type(error).__name__,
            "error": str(error),
            "provider_response": (
                error.response if isinstance(error, ProviderAPIError) else None
            ),
            "openai_batch_id": openai_batch_id,
            "openai_cancellation": cancellation,
            "credential_values_persisted": False,
            "api_submission_complete": False,
            "api_responses_received": False,
            "human_calibration_complete": False,
            "d0_complete": False,
            "outcomes_read": False,
        }
        BASE._write_json(execution_root / "submission_failure.outputs.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument(
        "--verification-report",
        type=Path,
        default=DEFAULT_VERIFICATION_REPORT,
    )
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    parser.add_argument("--credential-file", type=Path, default=DEFAULT_CREDENTIAL_FILE)
    parser.add_argument("--max-total-usd", type=Decimal, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        require(args.max_total_usd > 0, "operator cap must be positive")
        if not args.execute:
            result = _preflight(
                config_path=args.config,
                prepared_root=args.prepared_root,
                verification_report=args.verification_report,
                operator_cap_usd=args.max_total_usd,
            )
            print(json.dumps({"status": "PREFLIGHT_PASS_NOT_SUBMITTED", **result}, indent=2))
            return 0
        result = submit(
            config_path=args.config,
            prepared_root=args.prepared_root,
            verification_report=args.verification_report,
            execution_root=args.execution_root,
            credential_file=args.credential_file,
            operator_cap_usd=args.max_total_usd,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        CrossProviderV4SubmissionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_CROSS_PROVIDER_V4_SUBMISSION",
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
