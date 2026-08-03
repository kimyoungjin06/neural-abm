#!/usr/bin/env python3
"""Submit the independently verified v5 OpenAI-only retry batch."""

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
    / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v5.yaml"
)
DEFAULT_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729"
)
DEFAULT_VERIFICATION_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729.verification-v5.json"
)
DEFAULT_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "execution-20260729"
)
DEFAULT_CREDENTIAL_FILE = Path(
    "/home/kimyoungjin06/Desktop/Workspace/1.2.8.TwinPaper/.env"
)
VERIFIER = REPO_ROOT / "scripts/verify_pivot_feedback_d0_openai_retry_v5.py"
BASE_SUBMITTER = (
    REPO_ROOT / "scripts/submit_pivot_feedback_d0_cross_provider_api_review_v3.py"
)
VERIFICATION_STATUS = (
    "PASS_INDEPENDENT_D0_OPENAI_RETRY_VERIFICATION_NOT_SUBMITTED"
)
PREPARATION_STATUS = "PASS_D0_OPENAI_RETRY_REQUESTS_PREPARED_NOT_SUBMITTED"
SUBMISSION_STATUS = "SUBMITTED_D0_OPENAI_RETRY_V5_RESPONSES_PENDING"


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_openai_retry_submission_v5_base",
        BASE_SUBMITTER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base submitter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
OpenAIRetryV5SubmissionError = BASE.CrossProviderSubmissionError
ProviderAPIError = BASE.ProviderAPIError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenAIRetryV5SubmissionError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _read_openai_credential(path: Path) -> str:
    raw_path = path.absolute()
    require(not raw_path.is_symlink(), "credential file symlinked")
    path = path.resolve(strict=True)
    require(path.is_file(), "credential file invalid")
    require(path.stat().st_mode & 0o077 == 0, "credential file permissions too broad")
    value = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, candidate = line.split("=", 1)
        if key.strip() != "OPENAI_API_KEY":
            continue
        candidate = candidate.strip()
        if (
            len(candidate) >= 2
            and candidate[0] == candidate[-1]
            and candidate[0] in {'"', "'"}
        ):
            candidate = candidate[1:-1]
        value = candidate
    require(bool(value.strip()), "OpenAI credential missing")
    return value


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
    require(result.returncode == 0, "independent v5 verification failed")
    live = _mapping(json.loads(result.stdout), "live v5 verification")
    require(live.get("status") == VERIFICATION_STATUS, "v5 verification status differs")
    report = BASE._load_json(verification_report.resolve(strict=True))
    require(report == live, "v5 verification report differs from live verification")
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
        == "pivot_feedback_d0_cross_provider_api_review_config.v5",
        "config schema differs",
    )
    require(
        config.get("status") == "FROZEN_OPENAI_ONLY_RETRY_CONTRACT_NOT_SUBMITTED",
        "config status differs",
    )
    prepared_root = prepared_root.resolve(strict=True)
    verification = _validate_preparation(
        config_path=config_path,
        prepared_root=prepared_root,
        verification_report=verification_report,
    )
    receipt_path = prepared_root / "openai_retry_preparation.outputs.json"
    receipt = BASE._load_json(receipt_path)
    require(receipt.get("status") == PREPARATION_STATUS, "preparation status differs")
    require(receipt.get("retry_request_rows") == 384, "retry rows differ")
    require(receipt.get("retry_custom_ids_unchanged") is True, "retry IDs differ")
    estimate = _mapping(receipt.get("cost_control"), "cost control")
    cumulative = Decimal(
        str(estimate["cumulative_actual_plus_retry_maximum_usd"])
    )
    authorized = Decimal(str(estimate["cumulative_authorized_maximum_usd"]))
    require(operator_cap_usd <= authorized, "operator cap exceeds frozen authorization")
    require(cumulative <= operator_cap_usd, "cumulative maximum exceeds operator cap")
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
        "openai_retry_requests_sha256": BASE._sha256(
            prepared_root / "openai_retry_requests.jsonl"
        ),
        "retry_request_rows": 384,
        "v4_actual_total_cost_usd": estimate["v4_actual_total_cost_usd"],
        "retry_maximum_usd": estimate["retry_maximum_usd"],
        "cumulative_actual_plus_retry_maximum_usd": str(cumulative),
        "operator_cap_usd": str(operator_cap_usd.quantize(Decimal("0.00"))),
        "frozen_authorized_maximum_usd": str(authorized),
        "claude_resubmitted": False,
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
    api_key = _read_openai_credential(credential_file)
    prepared_root = prepared_root.resolve(strict=True)
    execution_root = execution_root.resolve(strict=False)
    require(not execution_root.exists(), "execution root already exists")
    require(execution_root.parent.is_dir(), "execution parent missing")
    execution_root.mkdir(mode=0o700)
    execution_root.chmod(0o700)
    BASE._write_json(
        execution_root / "submission_context.json",
        {
            "schema_version": "pivot_feedback_d0_openai_retry_submission_context.v5",
            "status": "PREFLIGHT_PASS_SUBMISSION_STARTED",
            **preflight,
        },
    )

    batch_id: str | None = None
    try:
        upload = BASE._upload_openai_file(
            api_key=api_key,
            path=prepared_root / "openai_retry_requests.jsonl",
        )
        BASE._write_json(
            execution_root / "openai_file_upload.response.json",
            upload,
        )
        upload_body = _mapping(upload["response"], "OpenAI upload response")
        file_id = str(upload_body.get("id", ""))
        require(file_id.startswith("file-"), "OpenAI file ID missing")
        require(upload_body.get("purpose") == "batch", "OpenAI purpose differs")

        batch = BASE._post_json(
            provider="OpenAI retry batch creation",
            url="https://api.openai.com/v1/batches",
            headers={"Authorization": f"Bearer {api_key}"},
            payload={
                "input_file_id": file_id,
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": {
                    "study": "pivot-feedback-d0",
                    "contract": "openai-retry-v5",
                    "judge": "openai",
                },
            },
        )
        BASE._write_json(
            execution_root / "openai_batch_create.response.json",
            batch,
        )
        batch_body = _mapping(batch["response"], "OpenAI batch response")
        batch_id = str(batch_body.get("id", ""))
        require(batch_id.startswith("batch_"), "OpenAI batch ID missing")
        receipt = {
            "schema_version": "pivot_feedback_d0_openai_retry_submission_outputs.v5",
            "status": SUBMISSION_STATUS,
            **preflight,
            "openai": {
                "input_file_id": file_id,
                "batch_id": batch_id,
                "status": batch_body.get("status"),
                "request_id": batch.get("request_id"),
            },
            "claude": {
                "resubmitted": False,
                "accepted_v4_result_rows": 384,
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
            execution_root / "openai_retry_submission.outputs.json",
            receipt,
        )
        return receipt
    except Exception as error:
        cancellation: dict[str, Any] | None = None
        if batch_id is not None:
            try:
                cancellation = BASE._cancel_openai_batch(
                    api_key=api_key,
                    batch_id=batch_id,
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
            "schema_version": "pivot_feedback_d0_openai_retry_submission_failure.v5",
            "status": "BLOCKED_OR_PARTIAL_D0_OPENAI_RETRY_V5_SUBMISSION",
            **preflight,
            "error_type": type(error).__name__,
            "error": str(error),
            "provider_response": (
                error.response if isinstance(error, ProviderAPIError) else None
            ),
            "openai_batch_id": batch_id,
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
        OpenAIRetryV5SubmissionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_OPENAI_RETRY_V5_SUBMISSION",
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
