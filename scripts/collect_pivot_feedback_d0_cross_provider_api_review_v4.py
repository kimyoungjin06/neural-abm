#!/usr/bin/env python3
"""Poll v4 provider batches and download terminal result files create-only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-execution-20260729"
)
DEFAULT_CREDENTIAL_FILE = Path(
    "/home/kimyoungjin06/Desktop/Workspace/1.2.8.TwinPaper/.env"
)
BASE_SUBMITTER = (
    REPO_ROOT / "scripts/submit_pivot_feedback_d0_cross_provider_api_review_v3.py"
)
TERMINAL_OPENAI = {"completed", "failed", "expired", "cancelled"}


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_collection_v3_base",
        BASE_SUBMITTER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load credential helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


class CrossProviderCollectionError(RuntimeError):
    """Fail-closed cross-provider collection error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossProviderCollectionError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get(
    *,
    provider: str,
    url: str,
    headers: Mapping[str, str],
    expect_json: bool,
) -> tuple[bytes, str | None]:
    try:
        with urlopen(Request(url, headers=dict(headers)), timeout=60) as response:
            body = response.read()
            if expect_json:
                _mapping(json.loads(body), f"{provider} response")
            return body, (
                response.headers.get("x-request-id")
                or response.headers.get("request-id")
            )
    except HTTPError as error:
        raise CrossProviderCollectionError(
            f"{provider} returned HTTP {error.code}"
        ) from error
    except URLError as error:
        raise CrossProviderCollectionError(
            f"{provider} network request failed"
        ) from error


def _download_create_only(
    *,
    path: Path,
    url: str,
    headers: Mapping[str, str],
    provider: str,
) -> dict[str, Any]:
    body, request_id = _get(
        provider=provider,
        url=url,
        headers=headers,
        expect_json=False,
    )
    digest = hashlib.sha256(body).hexdigest()
    if path.exists():
        require(path.is_file() and not path.is_symlink(), f"invalid existing {path.name}")
        require(_sha256(path) == digest, f"existing {path.name} differs")
    else:
        with path.open("xb") as handle:
            handle.write(body)
        path.chmod(0o600)
    return {
        "relative_path": path.name,
        "bytes": len(body),
        "sha256": digest,
        "request_id": request_id,
    }


def collect(
    *,
    execution_root: Path,
    credential_file: Path,
) -> dict[str, Any]:
    raw_root = execution_root.absolute()
    require(not raw_root.is_symlink(), "execution root symlinked")
    execution_root = execution_root.resolve(strict=True)
    require(execution_root.stat().st_mode & 0o777 == 0o700, "execution mode differs")
    receipt_path = execution_root / "cross_provider_submission.outputs.json"
    receipt = BASE._load_json(receipt_path)
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_cross_provider_submission_outputs.v4",
        "submission receipt schema differs",
    )
    require(receipt.get("api_submitted") is True, "batches not submitted")
    credentials = BASE._read_credentials(credential_file)
    openai_headers = {
        "Authorization": f"Bearer {credentials['OPENAI_API_KEY']}"
    }
    claude_headers = {
        "x-api-key": credentials["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    }

    openai_id = str(_mapping(receipt["openai"], "OpenAI receipt")["batch_id"])
    openai_raw, openai_request_id = _get(
        provider="OpenAI batch status",
        url=f"https://api.openai.com/v1/batches/{openai_id}",
        headers=openai_headers,
        expect_json=True,
    )
    openai = _mapping(json.loads(openai_raw), "OpenAI status")
    require(openai.get("id") == openai_id, "OpenAI batch ID differs")

    claude_id = str(_mapping(receipt["claude"], "Claude receipt")["batch_id"])
    claude_raw, claude_request_id = _get(
        provider="Claude batch status",
        url=f"https://api.anthropic.com/v1/messages/batches/{claude_id}",
        headers=claude_headers,
        expect_json=True,
    )
    claude = _mapping(json.loads(claude_raw), "Claude status")
    require(claude.get("id") == claude_id, "Claude batch ID differs")

    downloads: dict[str, Any] = {}
    openai_status = str(openai.get("status"))
    if openai_status in TERMINAL_OPENAI:
        for kind in ("output", "error"):
            file_id = openai.get(f"{kind}_file_id")
            if file_id:
                downloads[f"openai_{kind}"] = _download_create_only(
                    path=execution_root / f"openai_{kind}.jsonl",
                    url=f"https://api.openai.com/v1/files/{file_id}/content",
                    headers=openai_headers,
                    provider=f"OpenAI {kind} file",
                )

    claude_status = str(claude.get("processing_status"))
    if claude_status == "ended":
        results_url = claude.get("results_url")
        require(isinstance(results_url, str) and results_url, "Claude results URL missing")
        downloads["claude_results"] = _download_create_only(
            path=execution_root / "claude_results.jsonl",
            url=results_url,
            headers=claude_headers,
            provider="Claude results",
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot_dir = execution_root / "status_snapshots"
    if not snapshot_dir.exists():
        snapshot_dir.mkdir(mode=0o700)
    require(
        snapshot_dir.is_dir()
        and not snapshot_dir.is_symlink()
        and snapshot_dir.stat().st_mode & 0o777 == 0o700,
        "snapshot directory invalid",
    )
    snapshot = {
        "schema_version": "pivot_feedback_d0_cross_provider_status_snapshot.v4",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "openai": {
            "batch_id": openai_id,
            "status": openai_status,
            "request_counts": openai.get("request_counts"),
            "output_file_id": openai.get("output_file_id"),
            "error_file_id": openai.get("error_file_id"),
            "request_id": openai_request_id,
        },
        "claude": {
            "batch_id": claude_id,
            "processing_status": claude_status,
            "request_counts": claude.get("request_counts"),
            "results_url_present": bool(claude.get("results_url")),
            "request_id": claude_request_id,
        },
        "downloads": downloads,
        "credential_values_persisted": False,
        "both_terminal": (
            openai_status in TERMINAL_OPENAI and claude_status == "ended"
        ),
        "api_review_complete": False,
        "human_calibration_complete": False,
        "d0_complete": False,
        "outcomes_read": False,
    }
    snapshot_path = snapshot_dir / f"status-{timestamp}.json"
    BASE._write_json(snapshot_path, snapshot)
    snapshot["snapshot_relative_path"] = snapshot_path.relative_to(execution_root).as_posix()
    snapshot["snapshot_sha256"] = _sha256(snapshot_path)
    return snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    parser.add_argument("--credential-file", type=Path, default=DEFAULT_CREDENTIAL_FILE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = collect(
            execution_root=args.execution_root,
            credential_file=args.credential_file,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        CrossProviderCollectionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_CROSS_PROVIDER_V4_COLLECTION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
