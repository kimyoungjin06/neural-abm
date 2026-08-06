#!/usr/bin/env python3
"""Download the receipt-bound OpenAlex taxonomy dry-run with verified resume."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, BinaryIO, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RECEIPT = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_openalex_source_intake_v1_20260806.summary.json"
)
SOURCE_RECEIPT_SHA256 = "9c6def04d23755b8d35ae6ce90be282c45f6938afd39df4287545151af8c7946"
FOOTER_PREFLIGHT = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_openalex_footer_preflight_v1_20260806.summary.json"
)
FOOTER_PREFLIGHT_SHA256 = "51787c153673a5f74f419268b02f52ffda5222e89a1e14aada7891b3f07cc246"
MANIFEST_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-source-intake-v1-20260806")
DEFAULT_OUTPUT_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-taxonomy-dry-run-v1-20260806")
TAXONOMY_ENTITIES = ("topics", "subfields", "fields")
PLAN_NAME = "download_plan.json"
STATE_NAME = "download_state.json"
RECEIPT_NAME = "download_receipt.json"
LOCK_NAME = "download.lock"
CHUNK_BYTES = 1024 * 1024


class OpenAlexDownloadError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenAlexDownloadError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path, *, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as handle:
        while True:
            size = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
            if size == 0:
                break
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if limit is not None:
        require(remaining == 0, f"file shorter than hash limit: {path}")
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_create_only(path: Path, payload: Mapping[str, Any], mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_private_dir(path: Path, *, create: bool) -> None:
    if create and not path.exists():
        path.mkdir(mode=0o700)
    require(path.is_dir() and not path.is_symlink(), f"invalid directory: {path}")
    require(stat.S_IMODE(path.stat().st_mode) == 0o700, f"directory mode is not 0700: {path}")


def _ensure_relative_parent(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe relative path")
    target = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts[:-1]:
        current = current / part
        _require_private_dir(current, create=True)
    return target


def _s3_binding(url: str, entity: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    require(parsed.scheme == "s3" and parsed.netloc == "openalex", f"unexpected source: {url}")
    require(not parsed.query and not parsed.fragment, f"source URL has query or fragment: {url}")
    pure = PurePosixPath(parsed.path.lstrip("/"))
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe source path: {url}")
    expected_prefix = PurePosixPath("data") / "parquet" / entity
    require(tuple(pure.parts[:3]) == tuple(expected_prefix.parts), f"entity prefix mismatch: {url}")
    require(pure.suffix == ".parquet", f"non-Parquet source: {url}")
    http_url = f"https://openalex.s3.amazonaws.com/{pure.as_posix()}"
    return pure.as_posix(), http_url


def build_taxonomy_plan() -> dict[str, Any]:
    require(SOURCE_RECEIPT.is_file() and _sha256(SOURCE_RECEIPT) == SOURCE_RECEIPT_SHA256, "source receipt drift")
    require(FOOTER_PREFLIGHT.is_file() and _sha256(FOOTER_PREFLIGHT) == FOOTER_PREFLIGHT_SHA256, "footer preflight drift")
    receipt = _load_json(SOURCE_RECEIPT)
    preflight = _load_json(FOOTER_PREFLIGHT)
    require(preflight["authorization"]["manifest_driven_downloader_implementation_and_dry_run"] == "AUTHORIZED_NEXT", "dry-run authorization missing")
    require(preflight["authorization"]["full_source_data_download"] == "NOT_AUTHORIZED_BY_THIS_PREFLIGHT", "full transfer unexpectedly authorized")
    require(preflight["dependency_disposition"]["initial_download_entities"] == ["works", "topics", "subfields", "fields"], "core entity drift")
    bindings = {item["label"]: item for item in receipt["captured_manifests"]}
    files: list[dict[str, Any]] = []
    manifest_bindings: list[dict[str, Any]] = []
    for entity in TAXONOMY_ENTITIES:
        binding = bindings[entity]
        manifest_path = MANIFEST_ROOT / str(binding["relative_path"])
        require(manifest_path.is_file() and not manifest_path.is_symlink(), f"invalid manifest: {entity}")
        require(manifest_path.stat().st_size == binding["bytes"], f"manifest size drift: {entity}")
        require(_sha256(manifest_path) == binding["sha256"], f"manifest hash drift: {entity}")
        manifest = _load_json(manifest_path)
        require(manifest["date"] == "2026-06-26" and manifest["entity"] == entity, f"manifest identity drift: {entity}")
        manifest_bindings.append(
            {
                "entity": entity,
                "relative_path": str(binding["relative_path"]),
                "bytes": int(binding["bytes"]),
                "sha256": str(binding["sha256"]),
                "record_count": int(manifest["record_count"]),
                "content_length_bytes": int(manifest["content_length"]),
                "file_count": len(manifest["files"]),
            }
        )
        for item in manifest["files"]:
            relative_path, http_url = _s3_binding(str(item["url"]), entity)
            files.append(
                {
                    "entity": entity,
                    "s3_url": str(item["url"]),
                    "http_url": http_url,
                    "relative_path": relative_path,
                    "expected_bytes": int(item["meta"]["content_length"]),
                    "expected_records": int(item["meta"]["record_count"]),
                }
            )
    files.sort(key=lambda item: str(item["relative_path"]))
    require(len(files) == 8, "taxonomy dry-run file-count drift")
    require(sum(int(item["expected_bytes"]) for item in files) == 2_783_017, "taxonomy byte-count drift")
    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_download_plan.v1",
        "status": "BOUND_TAXONOMY_DRY_RUN_ONLY_FULL_CORE_TRANSFER_NOT_AUTHORIZED",
        "scope": "TOPICS_SUBFIELDS_FIELDS_ONLY",
        "snapshot_manifest_date": "2026-06-26",
        "source_receipt": {
            "relative_path": SOURCE_RECEIPT.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_RECEIPT_SHA256,
        },
        "footer_preflight": {
            "relative_path": FOOTER_PREFLIGHT.relative_to(ROOT).as_posix(),
            "sha256": FOOTER_PREFLIGHT_SHA256,
        },
        "captured_manifest_root": str(MANIFEST_ROOT),
        "manifest_bindings": manifest_bindings,
        "file_count": len(files),
        "expected_bytes": sum(int(item["expected_bytes"]) for item in files),
        "files": files,
        "authorization": {
            "taxonomy_dry_run": "AUTHORIZED",
            "works_transfer": "NOT_AUTHORIZED",
            "source_row_decode": "NOT_AUTHORIZED",
            "confirmatory_execution": "NOT_AUTHORIZED",
        },
    }


def _remote_metadata(http_url: str) -> dict[str, Any]:
    request = Request(http_url, method="HEAD", headers={"Accept-Encoding": "identity"})
    try:
        with urlopen(request, timeout=60) as response:
            return {
                "content_length_bytes": int(response.headers["Content-Length"]),
                "etag": str(response.headers.get("ETag", "")),
                "last_modified": str(response.headers.get("Last-Modified", "")),
                "version_id": str(response.headers.get("x-amz-version-id", "")),
                "accept_ranges": str(response.headers.get("Accept-Ranges", "")),
            }
    except (HTTPError, URLError, TimeoutError) as exc:
        raise OpenAlexDownloadError(f"remote metadata failed: {http_url}: {exc}") from exc


def _open_range(http_url: str, start: int, end: int) -> BinaryIO:
    request = Request(
        http_url,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
    )
    try:
        response = urlopen(request, timeout=120)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise OpenAlexDownloadError(f"range request failed: {http_url}: {exc}") from exc
    expected_length = end - start + 1
    content_range = str(response.headers.get("Content-Range", ""))
    require(getattr(response, "status", None) == 206, f"range response is not 206: {http_url}")
    require(content_range.startswith(f"bytes {start}-{end}/"), f"content-range drift: {http_url}")
    require(int(response.headers["Content-Length"]) == expected_length, f"range length drift: {http_url}")
    return response


def _new_state(plan_sha256: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_download_state.v1",
        "status": "PENDING",
        "plan_sha256": plan_sha256,
        "scope": plan["scope"],
        "run_count": 0,
        "intentional_pause_count": 0,
        "http_head_requests": 0,
        "http_range_requests": 0,
        "remote_bytes_transferred": 0,
        "resumed_partial_file_count": 0,
        "files": {
            str(item["relative_path"]): {
                "status": "PENDING",
                "expected_bytes": int(item["expected_bytes"]),
                "downloaded_bytes": 0,
                "checkpoint_sha256": hashlib.sha256(b"").hexdigest(),
                "remote": None,
                "final_sha256": None,
            }
            for item in plan["files"]
        },
    }


def _validate_existing_state(root: Path, plan: Mapping[str, Any], state: Mapping[str, Any], plan_sha256: str) -> None:
    require(state["schema_version"] == "pivot_feedback_lean_h3_openalex_download_state.v1", "state schema drift")
    require(state["plan_sha256"] == plan_sha256 and state["scope"] == plan["scope"], "state plan drift")
    expected_paths = {str(item["relative_path"]) for item in plan["files"]}
    require(set(state["files"]) == expected_paths, "state file surface drift")
    for item in plan["files"]:
        relative = str(item["relative_path"])
        entry = state["files"][relative]
        target = root.joinpath(*PurePosixPath(relative).parts)
        part = target.with_name(f"{target.name}.part")
        if entry["status"] == "COMPLETE":
            require(target.is_file() and not target.is_symlink() and not part.exists(), f"invalid complete artifact: {relative}")
            require(stat.S_IMODE(target.stat().st_mode) == 0o600, f"complete artifact mode drift: {relative}")
            require(target.stat().st_size == entry["expected_bytes"] == item["expected_bytes"], f"complete size drift: {relative}")
            require(_sha256(target) == entry["final_sha256"], f"complete hash drift: {relative}")
        else:
            require(not target.exists(), f"unregistered final artifact: {relative}")
            downloaded = int(entry["downloaded_bytes"])
            if downloaded:
                require(part.is_file() and not part.is_symlink(), f"missing partial artifact: {relative}")
                require(stat.S_IMODE(part.stat().st_mode) == 0o600, f"partial mode drift: {relative}")
                require(part.stat().st_size == downloaded, f"partial size drift: {relative}")
                require(_sha256(part) == entry["checkpoint_sha256"], f"partial hash drift: {relative}")
            else:
                require(not part.exists(), f"unexpected empty partial artifact: {relative}")


def _receipt(
    root: Path,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    plan_sha256: str,
    state_sha256: str,
    *,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    for item in plan["files"]:
        relative = str(item["relative_path"])
        entry = state["files"][relative]
        require(entry["status"] == "COMPLETE", f"receipt requested before completion: {relative}")
        inventory.append(
            {
                "entity": item["entity"],
                "s3_url": item["s3_url"],
                "relative_path": relative,
                "bytes": int(entry["downloaded_bytes"]),
                "expected_records_from_manifest": int(item["expected_records"]),
                "sha256": entry["final_sha256"],
                "remote": entry["remote"],
            }
        )
    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_taxonomy_dry_run_receipt.v1",
        "status": "PASS_EXACT_MANIFEST_TAXONOMY_RANGE_RESUME_LOCAL_HASH_INVENTORY_NO_ROW_DECODE",
        "completed_at_utc": completed_at_utc or _utc_now(),
        "mode": "BOUNDED_TAXONOMY_TRANSFER_DRY_RUN_NOT_FULL_CORE_DOWNLOAD_NOT_SCIENTIFIC_GATE",
        "output_root": str(root),
        "plan_sha256": plan_sha256,
        "terminal_state_sha256": state_sha256,
        "file_count": len(inventory),
        "downloaded_bytes": sum(int(item["bytes"]) for item in inventory),
        "inventory": inventory,
        "resume_evidence": {
            "run_count": int(state["run_count"]),
            "intentional_pause_count": int(state["intentional_pause_count"]),
            "resumed_partial_file_count": int(state["resumed_partial_file_count"]),
            "http_head_requests": int(state["http_head_requests"]),
            "http_range_requests": int(state["http_range_requests"]),
            "remote_bytes_transferred": int(state["remote_bytes_transferred"]),
        },
        "access_accounting": {
            "taxonomy_source_files_materialized_locally": len(inventory),
            "taxonomy_source_bytes_materialized_locally": sum(int(item["bytes"]) for item in inventory),
            "works_source_files_materialized_locally": 0,
            "works_source_bytes_materialized_locally": 0,
            "source_row_values_decoded": 0,
            "current_g1_label_rows_read": 0,
            "model_selection_rows_read": 0,
            "holdout_or_private_g4_rows_read": 0,
            "seal_open_events": 0,
            "g4_open": False,
            "models_fitted": 0,
        },
        "authorization": {
            "taxonomy_dry_run": "COMPLETE",
            "works_or_full_core_transfer": "NOT_AUTHORIZED_BY_THIS_RECEIPT",
            "source_row_decode": "NOT_AUTHORIZED",
            "confirmatory_successor_execution": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "This receipt establishes only exact-manifest selection, HTTP range resume, content-length "
            "closure, and local SHA-256 inventory for the eight taxonomy dry-run files. It does not "
            "authorize or establish the 724.97 GB Works transfer, row decoding, historical completeness, "
            "H3 maturity, eligible cohorts, precision, an association, null, mechanism, causal effect, "
            "or scientific effect."
        ),
    }


def run_taxonomy_dry_run(*, output_root: Path, resume: bool, max_new_bytes: int | None) -> dict[str, Any]:
    require(output_root.is_absolute(), "output root must be absolute")
    require(output_root == DEFAULT_OUTPUT_ROOT, "CLI authorization binds one exact output root")
    require(max_new_bytes is None or max_new_bytes > 0, "max-new-bytes must be positive")
    existed = output_root.exists()
    _require_private_dir(output_root, create=True)
    lock_path = output_root / LOCK_NAME
    lock_handle = lock_path.open("a+b")
    os.chmod(lock_path, 0o600)
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise OpenAlexDownloadError("download root is locked") from exc
    try:
        plan = build_taxonomy_plan()
        plan_path = output_root / PLAN_NAME
        if plan_path.exists():
            require(_load_json(plan_path) == plan, "download plan drift")
        else:
            require(not existed, "existing root is missing its download plan")
            _write_create_only(plan_path, plan)
        require(stat.S_IMODE(plan_path.stat().st_mode) == 0o600, "plan mode drift")
        plan_sha256 = _sha256(plan_path)

        state_path = output_root / STATE_NAME
        receipt_path = output_root / RECEIPT_NAME
        if state_path.exists():
            require(resume, "existing state requires --resume")
            state = _load_json(state_path)
            _validate_existing_state(output_root, plan, state, plan_sha256)
        else:
            require(not resume and not receipt_path.exists(), "resume requested without state")
            state = _new_state(plan_sha256, plan)
            _atomic_write_json(state_path, state)
        require(stat.S_IMODE(state_path.stat().st_mode) == 0o600, "state mode drift")

        if receipt_path.exists():
            require(resume, "terminal receipt requires --resume")
            receipt = _load_json(receipt_path)
            require(
                receipt
                == _receipt(
                    output_root,
                    plan,
                    state,
                    plan_sha256,
                    _sha256(state_path),
                    completed_at_utc=str(receipt["completed_at_utc"]),
                ),
                "terminal receipt drift",
            )
            return {
                "status": "PASS_TERMINAL_LOCAL_REVALIDATION_NO_REMOTE_ACCESS",
                "receipt_sha256": _sha256(receipt_path),
                "file_count": receipt["file_count"],
                "downloaded_bytes": receipt["downloaded_bytes"],
            }

        state["run_count"] = int(state["run_count"]) + 1
        remaining_budget = max_new_bytes
        resumed_paths: set[str] = set()
        for item in plan["files"]:
            relative = str(item["relative_path"])
            entry = state["files"][relative]
            if entry["status"] == "COMPLETE":
                continue
            target = _ensure_relative_parent(output_root, relative)
            part = target.with_name(f"{target.name}.part")
            offset = int(entry["downloaded_bytes"])
            if offset:
                resumed_paths.add(relative)
            remote = _remote_metadata(str(item["http_url"]))
            state["http_head_requests"] = int(state["http_head_requests"]) + 1
            require(remote["content_length_bytes"] == item["expected_bytes"], f"remote size drift: {relative}")
            require(remote["accept_ranges"].lower() == "bytes", f"range support missing: {relative}")
            if entry["remote"] is not None:
                require(entry["remote"] == remote, f"remote identity drift during resume: {relative}")
            else:
                entry["remote"] = remote
            if remaining_budget == 0:
                break
            expected = int(item["expected_bytes"])
            request_bytes = expected - offset
            if remaining_budget is not None:
                request_bytes = min(request_bytes, remaining_budget)
            end = offset + request_bytes - 1
            response = _open_range(str(item["http_url"]), offset, end)
            state["http_range_requests"] = int(state["http_range_requests"]) + 1
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            descriptor = os.open(part, flags, 0o600)
            received = 0
            try:
                with os.fdopen(descriptor, "ab", closefd=True) as handle, response:
                    while received < request_bytes:
                        chunk = response.read(min(CHUNK_BYTES, request_bytes - received))
                        require(bool(chunk), f"truncated range response: {relative}")
                        handle.write(chunk)
                        received += len(chunk)
                    require(response.read(1) == b"", f"range response exceeded request: {relative}")
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                if not part.exists():
                    raise
                entry["downloaded_bytes"] = part.stat().st_size
                entry["checkpoint_sha256"] = _sha256(part)
                _atomic_write_json(state_path, state)
                raise
            require(received == request_bytes, f"range byte count drift: {relative}")
            offset += received
            entry["downloaded_bytes"] = offset
            entry["checkpoint_sha256"] = _sha256(part)
            state["remote_bytes_transferred"] = int(state["remote_bytes_transferred"]) + received
            if remaining_budget is not None:
                remaining_budget -= received
            if offset == expected:
                require(part.stat().st_size == expected, f"complete partial size drift: {relative}")
                final_sha256 = _sha256(part)
                os.replace(part, target)
                entry["status"] = "COMPLETE"
                entry["final_sha256"] = final_sha256
            _atomic_write_json(state_path, state)
            if remaining_budget == 0:
                break

        state["resumed_partial_file_count"] = int(state["resumed_partial_file_count"]) + len(resumed_paths)
        complete_count = sum(entry["status"] == "COMPLETE" for entry in state["files"].values())
        if complete_count != len(plan["files"]):
            state["status"] = "INTENTIONALLY_PAUSED_MAX_NEW_BYTES"
            state["intentional_pause_count"] = int(state["intentional_pause_count"]) + 1
            _atomic_write_json(state_path, state)
            return {
                "status": state["status"],
                "complete_files": complete_count,
                "total_files": len(plan["files"]),
                "remote_bytes_transferred": state["remote_bytes_transferred"],
            }

        state["status"] = "COMPLETE"
        _atomic_write_json(state_path, state)
        receipt = _receipt(output_root, plan, state, plan_sha256, _sha256(state_path))
        _write_create_only(receipt_path, receipt)
        return {
            "status": receipt["status"],
            "receipt_sha256": _sha256(receipt_path),
            "file_count": receipt["file_count"],
            "downloaded_bytes": receipt["downloaded_bytes"],
        }
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("taxonomy-dry-run",), default="taxonomy-dry-run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-bytes", type=int)
    args = parser.parse_args()
    result = run_taxonomy_dry_run(
        output_root=DEFAULT_OUTPUT_ROOT,
        resume=args.resume,
        max_new_bytes=args.max_new_bytes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
