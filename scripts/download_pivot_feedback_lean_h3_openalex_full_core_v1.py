#!/usr/bin/env python3
"""Transfer the authorized exact-manifest OpenAlex lean-H3 full core."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from types import ModuleType
from typing import Any, BinaryIO, Mapping


ROOT = Path(__file__).resolve().parents[1]
DRY_RUN_SCRIPT = ROOT / "scripts/download_pivot_feedback_lean_h3_openalex_v1.py"
DRY_RUN_SCRIPT_SHA256 = "20fbcdd971691597f452473a735fbd85944c18cc425f926ccd4bcf54c58a3ece"
AUTHORIZATION = ROOT / (
    "experiments/configs/"
    "pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v1.json"
)
AUTHORIZATION_SHA256 = "b9b1519587e16c20632871601db5e04419f038be41d043c5fe2100057b92c817"
SOURCE_RECEIPT = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_openalex_source_intake_v1_20260806.summary.json"
)
SOURCE_RECEIPT_SHA256 = "9c6def04d23755b8d35ae6ce90be282c45f6938afd39df4287545151af8c7946"
MANIFEST_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-source-intake-v1-20260806")
OUTPUT_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-full-core-v1-20260806")
ENTITIES = ("works", "topics", "subfields", "fields")
PLAN_NAME = "download_plan.json"
STATE_NAME = "download_state.json"
CANARY_RECEIPT_NAME = "canary_receipt.json"
TERMINAL_RECEIPT_NAME = "download_receipt.json"
LOCK_NAME = "download.lock"
EXPECTED_FILE_COUNT = 2_454
EXPECTED_BYTES = 724_973_106_144
WORKING_RESERVE_BYTES = 1_087_459_659_216
TOTAL_RESERVED_BYTES = 1_812_432_765_360
SEGMENT_BYTES = 268_435_456
CANARY_NEW_BYTE_CAP = 67_108_864
INITIAL_DNS_INTERRUPTED_DOWNLOADER_SHA256 = (
    "bab9e2fc78f16be766080494eac27e6ba5bb6c21611b495bff7fbf422e48f1c8"
)
STREAM_CHUNK_BYTES = 4 * 1024 * 1024


class FullCoreTransferError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FullCoreTransferError(message)


def _load_dry_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lean_h3_openalex_dry_run_v1", DRY_RUN_SCRIPT)
    require(spec is not None and spec.loader is not None, "cannot load bound dry-run downloader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dry = _load_dry_run_module()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(path: Path, start: int, length: int) -> str:
    require(start >= 0 and length >= 0, "invalid local hash range")
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            require(bool(chunk), f"local file ended inside hash range: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stat_binding(path: Path) -> dict[str, int]:
    info = path.stat()
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "mode": stat.S_IMODE(info.st_mode),
    }


def _validate_stat_binding(path: Path, binding: Mapping[str, Any]) -> None:
    require(_stat_binding(path) == dict(binding), f"completed file identity drift: {path}")


def _repository_binding(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    require(_sha256(path) == expected_sha256, f"{label} hash drift")
    return _load_json(path)


def build_full_core_plan() -> dict[str, Any]:
    require(_sha256(DRY_RUN_SCRIPT) == DRY_RUN_SCRIPT_SHA256, "bound dry-run downloader drift")
    authorization = _repository_binding(AUTHORIZATION, AUTHORIZATION_SHA256, "authorization")
    source_receipt = _repository_binding(SOURCE_RECEIPT, SOURCE_RECEIPT_SHA256, "source receipt")
    require(
        authorization["status"]
        == "AUTHORIZED_EXACT_MANIFEST_FULL_CORE_TRANSFER_OPERATIONAL_ONLY_NO_ROW_DECODE",
        "authorization status drift",
    )
    require(authorization["authorization"]["full_core_source_byte_transfer"] == "AUTHORIZED", "full-core transfer not authorized")
    require(authorization["authorization"]["source_row_decode"] == "NOT_AUTHORIZED", "row decode boundary drift")
    require(authorization["transfer_scope"]["entities"] == list(ENTITIES), "entity scope drift")
    require(authorization["transfer_scope"]["authors_entity_included"] is False, "Authors unexpectedly included")
    require(authorization["capacity_contract"]["total_reserved_budget_bytes"] == TOTAL_RESERVED_BYTES, "total reserve drift")
    require(authorization["execution_contract"]["segment_bytes"] == SEGMENT_BYTES, "segment size drift")
    for label in (
        "source_intake_summary",
        "footer_preflight_summary",
        "taxonomy_dry_run_verification",
        "taxonomy_dry_run_downloader",
    ):
        binding = authorization["source_bindings"][label]
        path = ROOT / str(binding["relative_path"])
        require(path.is_file() and not path.is_symlink(), f"invalid authorized binding: {label}")
        require(_sha256(path) == binding["sha256"], f"authorized binding hash drift: {label}")

    receipt_bindings = {item["label"]: item for item in source_receipt["captured_manifests"]}
    expected_manifest_hashes = authorization["source_bindings"]["entity_manifest_sha256"]
    files: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for entity in ENTITIES:
        binding = receipt_bindings[entity]
        path = MANIFEST_ROOT / str(binding["relative_path"])
        require(path.is_file() and not path.is_symlink(), f"invalid {entity} manifest")
        require(path.stat().st_size == binding["bytes"], f"{entity} manifest size drift")
        manifest_sha256 = _sha256(path)
        require(manifest_sha256 == binding["sha256"] == expected_manifest_hashes[entity], f"{entity} manifest hash drift")
        manifest = _load_json(path)
        require(manifest["date"] == "2026-06-26" and manifest["entity"] == entity, f"{entity} manifest identity drift")
        manifests.append(
            {
                "entity": entity,
                "relative_path": str(binding["relative_path"]),
                "bytes": int(binding["bytes"]),
                "sha256": manifest_sha256,
                "file_count": len(manifest["files"]),
                "record_count": int(manifest["record_count"]),
                "content_length_bytes": int(manifest["content_length"]),
            }
        )
        for item in manifest["files"]:
            relative_path, http_url = dry._s3_binding(str(item["url"]), entity)
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
    require(len(files) == EXPECTED_FILE_COUNT, "full-core file-count drift")
    require(sum(int(item["expected_bytes"]) for item in files) == EXPECTED_BYTES, "full-core byte-count drift")
    require(sum(item["entity"] == "works" for item in files) == 2_446, "Works file-count drift")
    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_full_core_download_plan.v1",
        "status": "AUTHORIZED_EXACT_MANIFEST_FULL_CORE_TRANSFER_NO_ROW_DECODE",
        "scope": "WORKS_TOPICS_SUBFIELDS_FIELDS",
        "authorization": {
            "relative_path": AUTHORIZATION.relative_to(ROOT).as_posix(),
            "sha256": AUTHORIZATION_SHA256,
        },
        "source_receipt": {
            "relative_path": SOURCE_RECEIPT.relative_to(ROOT).as_posix(),
            "sha256": SOURCE_RECEIPT_SHA256,
        },
        "dry_run_downloader_sha256": DRY_RUN_SCRIPT_SHA256,
        "captured_manifest_root": str(MANIFEST_ROOT),
        "snapshot_manifest_date": "2026-06-26",
        "manifest_bindings": manifests,
        "file_count": len(files),
        "expected_bytes": EXPECTED_BYTES,
        "working_reserve_bytes": WORKING_RESERVE_BYTES,
        "total_reserved_budget_bytes": TOTAL_RESERVED_BYTES,
        "segment_bytes": SEGMENT_BYTES,
        "files": files,
        "prohibitions": {
            "authors_transfer": True,
            "source_row_decode": True,
            "parquet_row_group_read": True,
            "mutable_prefix_sync": True,
            "confirmatory_execution": True,
        },
    }


def _new_state(plan: Mapping[str, Any], plan_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_full_core_download_state.v1",
        "status": "PENDING",
        "plan_sha256": plan_sha256,
        "run_count": 0,
        "intentional_pause_count": 0,
        "http_head_requests": 0,
        "http_range_requests": 0,
        "remote_new_bytes_transferred": 0,
        "remote_recovery_validation_bytes": 0,
        "recovered_uncheckpointed_tail_count": 0,
        "minimum_capacity_margin_bytes": None,
        "last_capacity_check": None,
        "files": {
            str(item["relative_path"]): {
                "status": "PENDING",
                "expected_bytes": int(item["expected_bytes"]),
                "downloaded_bytes": 0,
                "segments": [],
                "remote": None,
                "final_sha256": None,
                "final_stat": None,
            }
            for item in plan["files"]
        },
    }


def _downloaded_bytes(state: Mapping[str, Any]) -> int:
    return sum(int(item["downloaded_bytes"]) for item in state["files"].values())


def _complete_files(state: Mapping[str, Any]) -> int:
    return sum(item["status"] == "COMPLETE" for item in state["files"].values())


def _capacity_check(state: dict[str, Any]) -> dict[str, int]:
    available = shutil.disk_usage(OUTPUT_ROOT).free
    downloaded = _downloaded_bytes(state)
    remaining = EXPECTED_BYTES - downloaded
    required = remaining + WORKING_RESERVE_BYTES
    margin = available - required
    snapshot = {
        "available_bytes": available,
        "downloaded_bytes": downloaded,
        "remaining_source_bytes": remaining,
        "working_reserve_bytes": WORKING_RESERVE_BYTES,
        "required_available_bytes": required,
        "margin_bytes": margin,
    }
    state["last_capacity_check"] = snapshot
    previous = state["minimum_capacity_margin_bytes"]
    state["minimum_capacity_margin_bytes"] = margin if previous is None else min(int(previous), margin)
    require(margin >= 0, "capacity invariant failed: available < remaining source + working reserve")
    return snapshot


def _validate_segment_surface(part: Path, entry: Mapping[str, Any]) -> None:
    cursor = 0
    for segment in entry["segments"]:
        start = int(segment["start"])
        length = int(segment["bytes"])
        require(start == cursor and length > 0, f"segment continuity drift: {part}")
        require(_sha256_range(part, start, length) == segment["sha256"], f"segment hash drift: {part}")
        cursor += length
    require(cursor == int(entry["downloaded_bytes"]), f"segment total drift: {part}")


def _validate_existing_state(plan: Mapping[str, Any], state: Mapping[str, Any], plan_sha256: str) -> None:
    require(state["schema_version"] == "pivot_feedback_lean_h3_openalex_full_core_download_state.v1", "state schema drift")
    require(state["plan_sha256"] == plan_sha256, "state plan drift")
    plan_by_path = {str(item["relative_path"]): item for item in plan["files"]}
    require(set(state["files"]) == set(plan_by_path), "state file surface drift")
    partial_count = 0
    for relative, item in plan_by_path.items():
        entry = state["files"][relative]
        require(entry["expected_bytes"] == item["expected_bytes"], f"state expected-byte drift: {relative}")
        target = OUTPUT_ROOT.joinpath(*PurePosixPath(relative).parts)
        part = target.with_name(f"{target.name}.part")
        if entry["status"] == "COMPLETE":
            require(target.is_file() and not target.is_symlink() and not part.exists(), f"invalid complete file: {relative}")
            _validate_stat_binding(target, entry["final_stat"])
            require(entry["downloaded_bytes"] == item["expected_bytes"], f"complete byte drift: {relative}")
        else:
            require(not target.exists(), f"unregistered final file: {relative}")
            downloaded = int(entry["downloaded_bytes"])
            if part.exists():
                require(part.is_file() and not part.is_symlink(), f"invalid partial file: {relative}")
                require(stat.S_IMODE(part.stat().st_mode) == 0o600, f"partial mode drift: {relative}")
                require(part.stat().st_size >= downloaded, f"partial file is behind checkpoint: {relative}")
                _validate_segment_surface(part, entry)
                partial_count += 1
            else:
                require(downloaded == 0 and not entry["segments"], f"missing partial file: {relative}")
    require(partial_count <= 1, "more than one partial file")


def _remote_range_sha256(http_url: str, start: int, end: int) -> str:
    response: BinaryIO = dry._open_range(http_url, start, end)
    digest = hashlib.sha256()
    remaining = end - start + 1
    with response:
        while remaining:
            chunk = response.read(min(STREAM_CHUNK_BYTES, remaining))
            require(bool(chunk), "truncated recovery validation range")
            digest.update(chunk)
            remaining -= len(chunk)
        require(response.read(1) == b"", "recovery validation range overflow")
    return digest.hexdigest()


def _recover_uncheckpointed_tail(
    *,
    part: Path,
    entry: dict[str, Any],
    http_url: str,
    state: dict[str, Any],
) -> None:
    checkpoint = int(entry["downloaded_bytes"])
    actual = part.stat().st_size
    if actual == checkpoint:
        return
    require(actual > checkpoint, "partial file is behind checkpoint")
    length = actual - checkpoint
    local_sha256 = _sha256_range(part, checkpoint, length)
    remote_sha256 = _remote_range_sha256(http_url, checkpoint, actual - 1)
    state["http_range_requests"] = int(state["http_range_requests"]) + 1
    state["remote_recovery_validation_bytes"] = int(state["remote_recovery_validation_bytes"]) + length
    require(local_sha256 == remote_sha256, "uncheckpointed tail differs from bound remote object")
    entry["segments"].append(
        {
            "start": checkpoint,
            "bytes": length,
            "sha256": local_sha256,
            "disposition": "RECOVERED_AFTER_REMOTE_RANGE_HASH_MATCH",
        }
    )
    entry["downloaded_bytes"] = actual
    state["recovered_uncheckpointed_tail_count"] = int(state["recovered_uncheckpointed_tail_count"]) + 1


def _append_range(
    *,
    part: Path,
    http_url: str,
    start: int,
    length: int,
) -> str:
    end = start + length - 1
    response: BinaryIO = dry._open_range(http_url, start, end)
    digest = hashlib.sha256()
    received = 0
    descriptor = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=True) as handle, response:
            while received < length:
                chunk = response.read(min(STREAM_CHUNK_BYTES, length - received))
                require(bool(chunk), f"truncated transfer range: {http_url}")
                handle.write(chunk)
                digest.update(chunk)
                received += len(chunk)
            require(response.read(1) == b"", f"transfer range overflow: {http_url}")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    require(received == length, f"transfer byte-count drift: {http_url}")
    return digest.hexdigest()


def _canary_receipt(plan_sha256: str, state_path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    materialized: list[dict[str, Any]] = []
    works_bytes = 0
    for relative, entry in sorted(state["files"].items()):
        downloaded = int(entry["downloaded_bytes"])
        if downloaded == 0:
            continue
        entity = relative.split("/")[2]
        works_bytes += downloaded if entity == "works" else 0
        path = OUTPUT_ROOT.joinpath(*PurePosixPath(relative).parts)
        if entry["status"] != "COMPLETE":
            path = path.with_name(f"{path.name}.part")
        materialized.append(
            {
                "entity": entity,
                "relative_path": relative,
                "downloaded_bytes": downloaded,
                "prefix_sha256": _sha256_range(path, 0, downloaded),
                "status": entry["status"],
                "remote": entry["remote"],
            }
        )
    require(works_bytes > 0, "canary did not reach Works bytes")
    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_full_core_canary_receipt.v1",
        "status": "PASS_FULL_CORE_CANARY_INTENTIONAL_PAUSE_WITH_WORKS_RANGE_BYTES_NO_ROW_DECODE",
        "captured_at_utc": _utc_now(),
        "plan_sha256": plan_sha256,
        "paused_state_sha256": _sha256(state_path),
        "downloader_lineage": {
            "initial_dns_interrupted_attempt_sha256": INITIAL_DNS_INTERRUPTED_DOWNLOADER_SHA256,
            "verified_cumulative_cap_resume_sha256": _sha256(Path(__file__)),
            "initial_interrupted_bytes": 12_576_018,
            "initial_interrupted_complete_files": 70,
            "initial_interruption": "TEMPORARY_DNS_NAME_RESOLUTION_FAILURE_BEFORE_REGISTERED_CANARY_PAUSE",
        },
        "new_bytes_transferred": int(state["remote_new_bytes_transferred"]),
        "works_bytes_materialized": works_bytes,
        "materialized_prefixes": materialized,
        "capacity_check": state["last_capacity_check"],
        "authorization": {
            "verified_background_resume": "AUTHORIZED_NEXT",
            "source_row_decode": "NOT_AUTHORIZED",
            "confirmatory_execution": "NOT_AUTHORIZED",
        },
        "claim_boundary": (
            "This canary receipt establishes only exact-manifest range transfer, partial checkpointing, "
            "and capacity enforcement through the first authorized Works bytes. It is not full download "
            "completion, row decoding, historical completeness, H3 maturity, cohort eligibility, an "
            "association, null, mechanism, causal effect, or scientific effect."
        ),
    }


def _terminal_receipt(plan_sha256: str, state_path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    inventory = [
        {
            "relative_path": relative,
            "bytes": int(entry["downloaded_bytes"]),
            "sha256": entry["final_sha256"],
            "remote": entry["remote"],
        }
        for relative, entry in sorted(state["files"].items())
    ]
    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_full_core_download_receipt.v1",
        "status": "PASS_EXACT_MANIFEST_FULL_CORE_TRANSFER_COMPLETE_LOCAL_SHA256_NO_ROW_DECODE",
        "completed_at_utc": _utc_now(),
        "plan_sha256": plan_sha256,
        "terminal_state_sha256": _sha256(state_path),
        "file_count": len(inventory),
        "downloaded_bytes": sum(item["bytes"] for item in inventory),
        "inventory": inventory,
        "transfer_accounting": {
            "run_count": state["run_count"],
            "http_head_requests": state["http_head_requests"],
            "http_range_requests": state["http_range_requests"],
            "remote_new_bytes_transferred": state["remote_new_bytes_transferred"],
            "remote_recovery_validation_bytes": state["remote_recovery_validation_bytes"],
            "minimum_capacity_margin_bytes": state["minimum_capacity_margin_bytes"],
        },
        "access_accounting": {
            "source_files_materialized_locally": len(inventory),
            "source_bytes_materialized_locally": sum(item["bytes"] for item in inventory),
            "source_row_values_decoded": 0,
            "current_g1_label_rows_read": 0,
            "model_selection_rows_read": 0,
            "holdout_or_private_g4_rows_read": 0,
            "seal_open_events": 0,
            "g4_open": False,
            "models_fitted": 0,
        },
        "authorization": {
            "source_transfer": "COMPLETE",
            "source_row_decode": "NOT_AUTHORIZED",
            "confirmatory_successor_execution": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
    }


def _lock_root() -> tuple[int, Any]:
    lock_path = OUTPUT_ROOT / LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    handle = os.fdopen(descriptor, "a+b", closefd=True)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise FullCoreTransferError("full-core download root is locked") from exc
    return handle.fileno(), handle


def run_full_core(*, auto_resume: bool, max_new_bytes: int | None) -> dict[str, Any]:
    require(max_new_bytes is None or max_new_bytes > 0, "max-new-bytes must be positive")
    root_existed = OUTPUT_ROOT.exists()
    dry._require_private_dir(OUTPUT_ROOT, create=True)
    _, lock_handle = _lock_root()
    try:
        plan = build_full_core_plan()
        plan_path = OUTPUT_ROOT / PLAN_NAME
        if plan_path.exists():
            require(_load_json(plan_path) == plan, "full-core plan drift")
        else:
            require(not root_existed, "existing root is missing its plan")
            dry._write_create_only(plan_path, plan)
        require(stat.S_IMODE(plan_path.stat().st_mode) == 0o600, "plan mode drift")
        plan_sha256 = _sha256(plan_path)
        state_path = OUTPUT_ROOT / STATE_NAME
        if state_path.exists():
            require(auto_resume, "existing state requires --auto-resume")
            state = _load_json(state_path)
            _validate_existing_state(plan, state, plan_sha256)
        else:
            require(not auto_resume or not root_existed, "resume requested for incomplete root surface")
            state = _new_state(plan, plan_sha256)
            _capacity_check(state)
            dry._atomic_write_json(state_path, state)

        terminal_path = OUTPUT_ROOT / TERMINAL_RECEIPT_NAME
        if terminal_path.exists():
            require(state["status"] == "COMPLETE", "terminal receipt with nonterminal state")
            receipt = _load_json(terminal_path)
            require(receipt["plan_sha256"] == plan_sha256, "terminal receipt plan drift")
            require(receipt["terminal_state_sha256"] == _sha256(state_path), "terminal receipt state drift")
            return {
                "status": "PASS_TERMINAL_LOCAL_STATE_REVALIDATION",
                "file_count": receipt["file_count"],
                "downloaded_bytes": receipt["downloaded_bytes"],
            }

        canary_path = OUTPUT_ROOT / CANARY_RECEIPT_NAME
        if not canary_path.exists():
            require(
                max_new_bytes == CANARY_NEW_BYTE_CAP,
                "uncapped resume is forbidden before the registered canary receipt",
            )
            run_budget = CANARY_NEW_BYTE_CAP - _downloaded_bytes(state)
            require(run_budget >= 0, "pre-canary bytes exceed the cumulative canary cap")
        else:
            run_budget = max_new_bytes
        state["run_count"] = int(state["run_count"]) + 1
        state["status"] = "RUNNING"
        dry._atomic_write_json(state_path, state)
        plan_by_path = {str(item["relative_path"]): item for item in plan["files"]}
        for relative, item in plan_by_path.items():
            entry = state["files"][relative]
            if entry["status"] == "COMPLETE":
                continue
            target = dry._ensure_relative_parent(OUTPUT_ROOT, relative)
            part = target.with_name(f"{target.name}.part")
            remote = dry._remote_metadata(str(item["http_url"]))
            state["http_head_requests"] = int(state["http_head_requests"]) + 1
            require(remote["content_length_bytes"] == item["expected_bytes"], f"remote size drift: {relative}")
            require(remote["accept_ranges"].lower() == "bytes", f"range support missing: {relative}")
            if entry["remote"] is None:
                entry["remote"] = remote
            else:
                require(entry["remote"] == remote, f"remote identity drift: {relative}")
            if part.exists():
                _recover_uncheckpointed_tail(
                    part=part,
                    entry=entry,
                    http_url=str(item["http_url"]),
                    state=state,
                )
                dry._atomic_write_json(state_path, state)
            expected = int(item["expected_bytes"])
            while int(entry["downloaded_bytes"]) < expected:
                if run_budget == 0:
                    break
                capacity = _capacity_check(state)
                offset = int(entry["downloaded_bytes"])
                length = min(SEGMENT_BYTES, expected - offset)
                if run_budget is not None:
                    length = min(length, run_budget)
                segment_sha256 = _append_range(
                    part=part,
                    http_url=str(item["http_url"]),
                    start=offset,
                    length=length,
                )
                state["http_range_requests"] = int(state["http_range_requests"]) + 1
                state["remote_new_bytes_transferred"] = int(state["remote_new_bytes_transferred"]) + length
                entry["segments"].append(
                    {
                        "start": offset,
                        "bytes": length,
                        "sha256": segment_sha256,
                        "disposition": "TRANSFERRED_AND_CHECKPOINTED",
                    }
                )
                entry["downloaded_bytes"] = offset + length
                if run_budget is not None:
                    run_budget -= length
                if int(entry["downloaded_bytes"]) == expected:
                    require(part.stat().st_size == expected, f"completed partial size drift: {relative}")
                    entry["final_sha256"] = _sha256(part)
                    os.replace(part, target)
                    entry["status"] = "COMPLETE"
                    entry["final_stat"] = _stat_binding(target)
                dry._atomic_write_json(state_path, state)
                print(
                    json.dumps(
                        {
                            "status": "RUNNING",
                            "complete_files": _complete_files(state),
                            "total_files": EXPECTED_FILE_COUNT,
                            "downloaded_bytes": _downloaded_bytes(state),
                            "total_bytes": EXPECTED_BYTES,
                            "active_relative_path": relative,
                            "capacity_margin_bytes_before_segment": capacity["margin_bytes"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if run_budget == 0:
                break

        complete = _complete_files(state)
        if complete != EXPECTED_FILE_COUNT:
            require(max_new_bytes is not None, "transfer stopped before completion without a byte cap")
            state["status"] = "INTENTIONALLY_PAUSED_MAX_NEW_BYTES"
            state["intentional_pause_count"] = int(state["intentional_pause_count"]) + 1
            _capacity_check(state)
            dry._atomic_write_json(state_path, state)
            if not canary_path.exists():
                require(max_new_bytes == CANARY_NEW_BYTE_CAP, "first pause is not the registered canary cap")
                require(_downloaded_bytes(state) == CANARY_NEW_BYTE_CAP, "cumulative canary byte cap drift")
                dry._write_create_only(canary_path, _canary_receipt(plan_sha256, state_path, state))
            return {
                "status": state["status"],
                "complete_files": complete,
                "total_files": EXPECTED_FILE_COUNT,
                "downloaded_bytes": _downloaded_bytes(state),
                "total_bytes": EXPECTED_BYTES,
                "canary_receipt_sha256": _sha256(canary_path),
            }

        state["status"] = "COMPLETE"
        _capacity_check(state)
        dry._atomic_write_json(state_path, state)
        receipt = _terminal_receipt(plan_sha256, state_path, state)
        dry._write_create_only(terminal_path, receipt)
        return {
            "status": receipt["status"],
            "file_count": receipt["file_count"],
            "downloaded_bytes": receipt["downloaded_bytes"],
            "receipt_sha256": _sha256(terminal_path),
        }
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def plan_only() -> dict[str, Any]:
    plan = build_full_core_plan()
    available = shutil.disk_usage("/ssd").free
    return {
        "status": "PASS_FULL_CORE_PLAN_ONLY_NO_TRANSFER",
        "file_count": plan["file_count"],
        "expected_bytes": plan["expected_bytes"],
        "available_bytes": available,
        "required_prestart_bytes": TOTAL_RESERVED_BYTES,
        "margin_bytes": available - TOTAL_RESERVED_BYTES,
        "prestart_capacity_pass": available >= TOTAL_RESERVED_BYTES,
        "output_root_exists": OUTPUT_ROOT.exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--max-new-bytes", type=int)
    args = parser.parse_args()
    try:
        result = plan_only() if args.plan_only else run_full_core(
            auto_resume=args.auto_resume,
            max_new_bytes=args.max_new_bytes,
        )
    except (FullCoreTransferError, dry.OpenAlexDownloadError) as exc:
        print(json.dumps({"status": "STOPPED_FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 75
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
