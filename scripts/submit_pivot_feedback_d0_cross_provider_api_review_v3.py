#!/usr/bin/env python3
"""Submit independently verified OpenAI and Claude D0 review batches."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import mimetypes
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v3.yaml"
)
DEFAULT_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
)
DEFAULT_VERIFICATION_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
    ".verification-v3.json"
)
DEFAULT_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-execution-20260729"
)
DEFAULT_CREDENTIAL_FILE = Path(
    "/home/kimyoungjin06/Desktop/Workspace/1.2.8.TwinPaper/.env"
)
VERIFIER = (
    REPO_ROOT / "scripts/verify_pivot_feedback_d0_cross_provider_api_review_v3.py"
)
PREPARATION_STATUS = "PASS_D0_CROSS_PROVIDER_BATCH_REQUESTS_PREPARED_NOT_SUBMITTED"
VERIFICATION_STATUS = (
    "PASS_INDEPENDENT_D0_CROSS_PROVIDER_INPUT_VERIFICATION_NOT_SUBMITTED"
)
SUBMISSION_STATUS = "SUBMITTED_D0_CROSS_PROVIDER_BATCHES_RESPONSES_PENDING"


class CrossProviderSubmissionError(RuntimeError):
    """Fail-closed cross-provider submission error."""


class ProviderAPIError(CrossProviderSubmissionError):
    """Provider API request failed."""

    def __init__(
        self,
        *,
        provider: str,
        status: int | None,
        response: Mapping[str, Any],
    ) -> None:
        super().__init__(f"{provider} API request failed with HTTP {status}")
        self.provider = provider
        self.status = status
        self.response = dict(response)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossProviderSubmissionError(message)


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
        raise CrossProviderSubmissionError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise CrossProviderSubmissionError(
            f"invalid YAML {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _read_credentials(path: Path) -> dict[str, str]:
    raw_path = path.absolute()
    require(not raw_path.is_symlink(), "credential file symlinked")
    path = path.resolve(strict=True)
    require(path.is_file(), "credential file invalid")
    require(path.stat().st_mode & 0o077 == 0, "credential file permissions too broad")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        if key in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY"}:
            values[key] = value
    require(values.get("OPENAI_API_KEY", "").strip(), "OpenAI credential missing")
    claude = values.get("ANTHROPIC_API_KEY") or values.get("CLAUDE_API_KEY")
    require(bool(claude and claude.strip()), "Claude credential missing")
    return {
        "OPENAI_API_KEY": values["OPENAI_API_KEY"],
        "ANTHROPIC_API_KEY": str(claude),
    }


def _provider_response(
    *,
    provider: str,
    request: Request,
    timeout: int = 60,
) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {
                "http_status": response.status,
                "request_id": (
                    response.headers.get("x-request-id")
                    or response.headers.get("request-id")
                ),
                "response": _mapping(body, f"{provider} response"),
            }
    except HTTPError as error:
        try:
            body = json.loads(error.read().decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            body = {"error": {"type": "non_json_error_response"}}
        raise ProviderAPIError(
            provider=provider,
            status=error.code,
            response={
                "http_status": error.code,
                "request_id": (
                    error.headers.get("x-request-id")
                    or error.headers.get("request-id")
                ),
                "response": body,
            },
        ) from error
    except URLError as error:
        raise ProviderAPIError(
            provider=provider,
            status=None,
            response={
                "http_status": None,
                "request_id": None,
                "response": {"error": {"type": "network_error"}},
            },
        ) from error


def _post_json(
    *,
    provider: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode()
    request = Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    return _provider_response(provider=provider, request=request)


def _upload_openai_file(
    *,
    api_key: str,
    path: Path,
) -> dict[str, Any]:
    boundary = f"----neural-abm-{secrets.token_hex(16)}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="purpose"\r\n\r\n')
    body.extend(b"batch\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            'Content-Disposition: form-data; name="file"; '
            f'filename="{path.name}"\r\n'
        ).encode()
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.extend(path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    request = Request(
        "https://api.openai.com/v1/files",
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    return _provider_response(provider="OpenAI file upload", request=request)


def _cancel_openai_batch(
    *,
    api_key: str,
    batch_id: str,
) -> dict[str, Any]:
    return _post_json(
        provider="OpenAI batch cancellation",
        url=f"https://api.openai.com/v1/batches/{batch_id}/cancel",
        headers={"Authorization": f"Bearer {api_key}"},
        payload={},
    )


def _validate_preparation(
    *,
    config_path: Path,
    prepared_root: Path,
    verification_report: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    require(result.returncode == 0, "independent preparation verification failed")
    current = _mapping(json.loads(result.stdout), "live verification")
    require(current.get("status") == VERIFICATION_STATUS, "verification status differs")
    report = _load_json(verification_report.resolve(strict=True))
    require(report == current, "verification report differs from live verification")
    return current, report


def _preflight(
    *,
    config_path: Path,
    prepared_root: Path,
    verification_report: Path,
    operator_cap_usd: Decimal,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _load_yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_cross_provider_api_review_config.v3",
        "config schema differs",
    )
    prepared_root = prepared_root.resolve(strict=True)
    require(prepared_root.is_dir(), "prepared root invalid")
    verification, _ = _validate_preparation(
        config_path=config_path,
        prepared_root=prepared_root,
        verification_report=verification_report,
    )
    receipt_path = prepared_root / "cross_provider_preparation.outputs.json"
    receipt = _load_json(receipt_path)
    require(receipt.get("status") == PREPARATION_STATUS, "preparation status differs")
    estimate = _mapping(receipt.get("cost_control"), "cost control")
    estimated = Decimal(str(estimate["total_maximum_usd"]))
    authorized = Decimal(str(estimate["authorized_maximum_usd"]))
    require(operator_cap_usd <= authorized, "operator cap exceeds frozen authorization")
    require(estimated <= operator_cap_usd, "estimated maximum exceeds operator cap")
    require(verification.get("api_submitted") is False, "verification says submitted")
    return {
        "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "prepared_root_basename": prepared_root.name,
        "preparation_receipt_sha256": _sha256(receipt_path),
        "verification_report_basename": verification_report.name,
        "verification_report_sha256": _sha256(verification_report),
        "verifier_sha256": _sha256(VERIFIER),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "openai_requests_sha256": _sha256(
            prepared_root / "openai_requests.jsonl"
        ),
        "claude_batch_request_sha256": _sha256(
            prepared_root / "claude_batch_request.json"
        ),
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
    credentials = _read_credentials(credential_file)
    prepared_root = prepared_root.resolve(strict=True)
    execution_root = execution_root.resolve(strict=False)
    require(not execution_root.exists(), "execution root already exists")
    require(execution_root.parent.is_dir(), "execution parent missing")
    execution_root.mkdir(mode=0o700)
    execution_root.chmod(0o700)
    _write_json(
        execution_root / "submission_context.json",
        {
            "schema_version": "pivot_feedback_d0_cross_provider_submission_context.v3",
            "status": "PREFLIGHT_PASS_SUBMISSION_STARTED",
            **preflight,
        },
    )

    openai_batch_id: str | None = None
    try:
        upload = _upload_openai_file(
            api_key=credentials["OPENAI_API_KEY"],
            path=prepared_root / "openai_requests.jsonl",
        )
        _write_json(
            execution_root / "openai_file_upload.response.json",
            upload,
        )
        upload_body = _mapping(upload["response"], "OpenAI upload response")
        file_id = str(upload_body.get("id", ""))
        require(file_id.startswith("file-"), "OpenAI file ID missing")
        require(upload_body.get("purpose") == "batch", "OpenAI file purpose differs")

        openai_batch = _post_json(
            provider="OpenAI batch creation",
            url="https://api.openai.com/v1/batches",
            headers={"Authorization": f"Bearer {credentials['OPENAI_API_KEY']}"},
            payload={
                "input_file_id": file_id,
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": {
                    "study": "pivot-feedback-d0",
                    "contract": "cross-provider-v3",
                    "judge": "openai",
                },
            },
        )
        _write_json(
            execution_root / "openai_batch_create.response.json",
            openai_batch,
        )
        openai_body = _mapping(openai_batch["response"], "OpenAI batch response")
        openai_batch_id = str(openai_body.get("id", ""))
        require(openai_batch_id.startswith("batch_"), "OpenAI batch ID missing")
        require(openai_body.get("endpoint") == "/v1/responses", "OpenAI endpoint differs")

        claude_payload = _load_json(
            prepared_root / "claude_batch_request.json"
        )
        claude_batch = _post_json(
            provider="Claude batch creation",
            url="https://api.anthropic.com/v1/messages/batches",
            headers={
                "x-api-key": credentials["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
            payload=claude_payload,
        )
        _write_json(
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
            "schema_version": "pivot_feedback_d0_cross_provider_submission_outputs.v3",
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
        _write_json(
            execution_root / "cross_provider_submission.outputs.json",
            receipt,
        )
        return receipt
    except Exception as error:
        cancellation: dict[str, Any] | None = None
        if openai_batch_id is not None:
            try:
                cancellation = _cancel_openai_batch(
                    api_key=credentials["OPENAI_API_KEY"],
                    batch_id=openai_batch_id,
                )
                _write_json(
                    execution_root / "openai_batch_cancel.response.json",
                    cancellation,
                )
            except Exception as cancel_error:
                cancellation = {
                    "status": "CANCELLATION_FAILED",
                    "error_type": type(cancel_error).__name__,
                }
                _write_json(
                    execution_root / "openai_batch_cancel_failure.json",
                    cancellation,
                )
        failure = {
            "schema_version": "pivot_feedback_d0_cross_provider_submission_failure.v3",
            "status": "BLOCKED_OR_PARTIAL_CROSS_PROVIDER_SUBMISSION",
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
        _write_json(execution_root / "submission_failure.outputs.json", failure)
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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement for paid external submission.",
    )
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
        CrossProviderSubmissionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_CROSS_PROVIDER_SUBMISSION",
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
