#!/usr/bin/env python3
"""Poll the v5 OpenAI retry batch and download terminal files create-only."""

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
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "execution-20260729"
)
DEFAULT_CREDENTIAL_FILE = Path(
    "/home/kimyoungjin06/Desktop/Workspace/1.2.8.TwinPaper/.env"
)
SUBMITTER = REPO_ROOT / "scripts/submit_pivot_feedback_d0_openai_retry_v5.py"
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _load_submitter():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_openai_retry_collection_v5_submitter",
        SUBMITTER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load OpenAI credential helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUBMIT = _load_submitter()
BASE = SUBMIT.BASE


class OpenAIRetryV5CollectionError(RuntimeError):
    """Fail-closed OpenAI retry collection error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenAIRetryV5CollectionError(message)


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
        raise OpenAIRetryV5CollectionError(
            f"{provider} returned HTTP {error.code}"
        ) from error
    except URLError as error:
        raise OpenAIRetryV5CollectionError(
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
    receipt_path = execution_root / "openai_retry_submission.outputs.json"
    receipt = BASE._load_json(receipt_path)
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_openai_retry_submission_outputs.v5",
        "submission receipt schema differs",
    )
    require(receipt.get("api_submitted") is True, "batch not submitted")
    api_key = SUBMIT._read_openai_credential(credential_file)
    headers = {"Authorization": f"Bearer {api_key}"}
    batch_id = str(_mapping(receipt["openai"], "OpenAI receipt")["batch_id"])
    raw, request_id = _get(
        provider="OpenAI retry batch status",
        url=f"https://api.openai.com/v1/batches/{batch_id}",
        headers=headers,
        expect_json=True,
    )
    status_body = _mapping(json.loads(raw), "OpenAI status")
    require(status_body.get("id") == batch_id, "OpenAI batch ID differs")

    downloads: dict[str, Any] = {}
    status = str(status_body.get("status"))
    if status in TERMINAL:
        for kind in ("output", "error"):
            file_id = status_body.get(f"{kind}_file_id")
            if file_id:
                downloads[f"openai_retry_{kind}"] = _download_create_only(
                    path=execution_root / f"openai_retry_{kind}.jsonl",
                    url=f"https://api.openai.com/v1/files/{file_id}/content",
                    headers=headers,
                    provider=f"OpenAI retry {kind} file",
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
        "schema_version": "pivot_feedback_d0_openai_retry_status_snapshot.v5",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "openai": {
            "batch_id": batch_id,
            "status": status,
            "request_counts": status_body.get("request_counts"),
            "output_file_id": status_body.get("output_file_id"),
            "error_file_id": status_body.get("error_file_id"),
            "request_id": request_id,
        },
        "downloads": downloads,
        "credential_values_persisted": False,
        "terminal": status in TERMINAL,
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
        OpenAIRetryV5CollectionError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0_OPENAI_RETRY_V5_COLLECTION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
