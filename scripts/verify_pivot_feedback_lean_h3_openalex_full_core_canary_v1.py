#!/usr/bin/env python3
"""Independently verify the exact-manifest OpenAlex full-core transfer canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION = ROOT / (
    "experiments/configs/"
    "pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v1.json"
)
AUTHORIZATION_SHA256 = "b9b1519587e16c20632871601db5e04419f038be41d043c5fe2100057b92c817"
DOWNLOADER = ROOT / "scripts/download_pivot_feedback_lean_h3_openalex_full_core_v1.py"
DOWNLOADER_SHA256 = "009221e32193e99dd00fdceefe3db6829bc9d0ec7b4f99b0427a2aa259a1394d"
INITIAL_ATTEMPT_SHA256 = "bab9e2fc78f16be766080494eac27e6ba5bb6c21611b495bff7fbf422e48f1c8"
SOURCE_RECEIPT = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_openalex_source_intake_v1_20260806.summary.json"
)
SOURCE_RECEIPT_SHA256 = "9c6def04d23755b8d35ae6ce90be282c45f6938afd39df4287545151af8c7946"
MANIFEST_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-source-intake-v1-20260806")
DOWNLOAD_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-full-core-v1-20260806")
PLAN_SHA256 = "182d1d4e679c4e7a01811822e0384c06e84795b48f175f65a6f20baa560c9031"
PAUSED_STATE_SHA256 = "270e449eba613219b03adfdc3727b4807e6ec445d7f4cc3c47a7bb1fbba6f4d3"
CANARY_RECEIPT_SHA256 = "450630c2212f936be2ada9b3f8252a52fe077773152f8671586e8d07ca12207f"
EXPECTED_TOTAL_FILES = 2_454
EXPECTED_TOTAL_BYTES = 724_973_106_144
EXPECTED_CANARY_BYTES = 67_108_864
EXPECTED_CANARY_WORKS_BYTES = 64_325_847
STATUS = "PASS_INDEPENDENT_FULL_CORE_CANARY_MANIFEST_PREFIX_FOOTER_AND_CAPACITY_CLOSURE_NO_ROW_DECODE"


class FullCoreCanaryVerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FullCoreCanaryVerificationError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_prefix(path: Path, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            require(bool(chunk), f"file shorter than canary prefix: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_inventory(source_receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bindings = {item["label"]: item for item in source_receipt["captured_manifests"]}
    inventory: dict[str, dict[str, Any]] = {}
    for entity in ("works", "topics", "subfields", "fields"):
        binding = bindings[entity]
        path = MANIFEST_ROOT / str(binding["relative_path"])
        require(path.is_file() and not path.is_symlink(), f"invalid manifest: {entity}")
        require(path.stat().st_size == binding["bytes"], f"manifest size drift: {entity}")
        require(_sha256(path) == binding["sha256"], f"manifest hash drift: {entity}")
        manifest = _load_json(path)
        require(manifest["date"] == "2026-06-26" and manifest["entity"] == entity, f"manifest identity drift: {entity}")
        for item in manifest["files"]:
            s3_url = str(item["url"])
            require(s3_url.startswith(f"s3://openalex/data/parquet/{entity}/"), f"manifest URL drift: {s3_url}")
            relative = s3_url.removeprefix("s3://openalex/")
            inventory[relative] = {
                "entity": entity,
                "s3_url": s3_url,
                "http_url": f"https://openalex.s3.amazonaws.com/{relative}",
                "expected_bytes": int(item["meta"]["content_length"]),
                "expected_records": int(item["meta"]["record_count"]),
            }
    require(len(inventory) == EXPECTED_TOTAL_FILES, "manifest file-count drift")
    require(sum(item["expected_bytes"] for item in inventory.values()) == EXPECTED_TOTAL_BYTES, "manifest byte-count drift")
    return inventory


def verify() -> dict[str, Any]:
    for path, expected_hash, label in (
        (AUTHORIZATION, AUTHORIZATION_SHA256, "authorization"),
        (DOWNLOADER, DOWNLOADER_SHA256, "downloader"),
        (SOURCE_RECEIPT, SOURCE_RECEIPT_SHA256, "source receipt"),
    ):
        require(path.is_file() and not path.is_symlink(), f"invalid {label}")
        require(_sha256(path) == expected_hash, f"{label} hash drift")
    authority = _load_json(AUTHORIZATION)
    require(authority["authorization"]["full_core_source_byte_transfer"] == "AUTHORIZED", "transfer authority drift")
    require(authority["authorization"]["source_row_decode"] == "NOT_AUTHORIZED", "row decode boundary drift")
    manifest = _manifest_inventory(_load_json(SOURCE_RECEIPT))

    plan_path = DOWNLOAD_ROOT / "download_plan.json"
    state_path = DOWNLOAD_ROOT / "download_state.json"
    canary_path = DOWNLOAD_ROOT / "canary_receipt.json"
    require(_sha256(plan_path) == PLAN_SHA256, "plan hash drift")
    require(_sha256(canary_path) == CANARY_RECEIPT_SHA256, "canary receipt hash drift")
    plan = _load_json(plan_path)
    canary = _load_json(canary_path)
    require(plan["file_count"] == EXPECTED_TOTAL_FILES and plan["expected_bytes"] == EXPECTED_TOTAL_BYTES, "plan totals drift")
    require(plan["authorization"]["sha256"] == AUTHORIZATION_SHA256, "plan authorization drift")
    plan_by_path = {str(item["relative_path"]): item for item in plan["files"]}
    require(set(plan_by_path) == set(manifest), "plan path surface drift")
    for relative, expected in manifest.items():
        item = plan_by_path[relative]
        require(item["entity"] == expected["entity"], f"plan entity drift: {relative}")
        require(item["s3_url"] == expected["s3_url"], f"plan S3 URL drift: {relative}")
        require(item["http_url"] == expected["http_url"], f"plan HTTP URL drift: {relative}")
        require(item["expected_bytes"] == expected["expected_bytes"], f"plan byte drift: {relative}")
        require(item["expected_records"] == expected["expected_records"], f"plan row-count drift: {relative}")

    require(
        canary["status"]
        == "PASS_FULL_CORE_CANARY_INTENTIONAL_PAUSE_WITH_WORKS_RANGE_BYTES_NO_ROW_DECODE",
        "canary status drift",
    )
    require(canary["plan_sha256"] == PLAN_SHA256, "canary plan drift")
    require(canary["paused_state_sha256"] == PAUSED_STATE_SHA256, "canary paused-state drift")
    lineage = canary["downloader_lineage"]
    require(lineage["initial_dns_interrupted_attempt_sha256"] == INITIAL_ATTEMPT_SHA256, "initial attempt lineage drift")
    require(lineage["verified_cumulative_cap_resume_sha256"] == DOWNLOADER_SHA256, "resumed downloader lineage drift")
    require(lineage["initial_interrupted_bytes"] == 12_576_018, "initial interrupted byte drift")
    require(lineage["initial_interrupted_complete_files"] == 70, "initial interrupted file drift")
    require(canary["new_bytes_transferred"] == EXPECTED_CANARY_BYTES, "canary cumulative cap drift")
    require(canary["works_bytes_materialized"] == EXPECTED_CANARY_WORKS_BYTES, "canary Works byte drift")
    prefixes = canary["materialized_prefixes"]
    require(len(prefixes) == 209, "canary prefix-count drift")
    require(sum(int(item["downloaded_bytes"]) for item in prefixes) == EXPECTED_CANARY_BYTES, "canary prefix total drift")
    require(sum(int(item["downloaded_bytes"]) for item in prefixes if item["entity"] == "works") == EXPECTED_CANARY_WORKS_BYTES, "canary Works prefix total drift")

    verified_prefixes: list[dict[str, Any]] = []
    footer_rows = 0
    complete_files = 0
    partial_files = 0
    for prefix in prefixes:
        relative = str(prefix["relative_path"])
        require(relative in manifest, f"canary path absent from manifest: {relative}")
        expected = manifest[relative]
        require(prefix["entity"] == expected["entity"], f"canary entity drift: {relative}")
        length = int(prefix["downloaded_bytes"])
        target = DOWNLOAD_ROOT.joinpath(*PurePosixPath(relative).parts)
        part = target.with_name(f"{target.name}.part")
        candidate = target if target.exists() else part
        require(candidate.is_file() and not candidate.is_symlink(), f"missing canary prefix file: {relative}")
        require(stat.S_IMODE(candidate.stat().st_mode) == 0o600, f"canary file mode drift: {relative}")
        require(candidate.stat().st_size >= length, f"canary prefix truncated: {relative}")
        require(_sha256_prefix(candidate, length) == prefix["prefix_sha256"], f"canary prefix hash drift: {relative}")
        if prefix["status"] == "COMPLETE":
            require(length == expected["expected_bytes"] and target.is_file(), f"canary complete-file drift: {relative}")
            parquet = pq.ParquetFile(target)
            require(parquet.metadata.num_rows == expected["expected_records"], f"canary footer row drift: {relative}")
            footer_rows += parquet.metadata.num_rows
            complete_files += 1
        else:
            require(prefix["status"] == "PENDING" and length < expected["expected_bytes"], f"canary partial disposition drift: {relative}")
            partial_files += 1
        verified_prefixes.append(
            {
                "entity": prefix["entity"],
                "relative_path": relative,
                "canary_bytes": length,
                "prefix_sha256": prefix["prefix_sha256"],
                "canary_status": prefix["status"],
            }
        )
    require(complete_files == 208 and partial_files == 1, "canary complete/partial split drift")
    capacity = canary["capacity_check"]
    require(capacity["margin_bytes"] >= 0, "canary capacity margin failed")
    require(capacity["downloaded_bytes"] == EXPECTED_CANARY_BYTES, "canary capacity byte drift")
    prefix_inventory_sha256 = hashlib.sha256(
        json.dumps(verified_prefixes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    entity_prefix_counts = {
        entity: sum(item["entity"] == entity for item in verified_prefixes)
        for entity in ("works", "topics", "subfields", "fields")
    }
    entity_prefix_bytes = {
        entity: sum(item["canary_bytes"] for item in verified_prefixes if item["entity"] == entity)
        for entity in ("works", "topics", "subfields", "fields")
    }
    partial_prefix = next(item for item in verified_prefixes if item["canary_status"] == "PENDING")

    state_relation = "PAUSED_STATE_STILL_CURRENT"
    if _sha256(state_path) != PAUSED_STATE_SHA256:
        state_relation = "LIVE_STATE_ADVANCED_AFTER_IMMUTABLE_CANARY_PREFIX_RECEIPT"
    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_full_core_canary_verification_summary.v1",
        "status": STATUS,
        "verified_at_utc": _utc_now(),
        "mode": "INDEPENDENT_EXACT_MANIFEST_PREFIX_HASH_AND_COMPLETE_FILE_FOOTER_REPLAY_NO_ROW_DECODE",
        "bindings": {
            "authorization_sha256": AUTHORIZATION_SHA256,
            "downloader_sha256": DOWNLOADER_SHA256,
            "initial_attempt_downloader_sha256": INITIAL_ATTEMPT_SHA256,
            "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
            "download_plan_sha256": PLAN_SHA256,
            "paused_state_sha256": PAUSED_STATE_SHA256,
            "canary_receipt_sha256": CANARY_RECEIPT_SHA256,
        },
        "canary_replay": {
            "total_plan_files": EXPECTED_TOTAL_FILES,
            "total_plan_bytes": EXPECTED_TOTAL_BYTES,
            "canary_prefix_count": len(prefixes),
            "canary_complete_files": complete_files,
            "canary_partial_files": partial_files,
            "canary_bytes": EXPECTED_CANARY_BYTES,
            "canary_works_bytes": EXPECTED_CANARY_WORKS_BYTES,
            "complete_file_footer_rows": footer_rows,
            "capacity_margin_bytes": capacity["margin_bytes"],
            "live_state_relation": state_relation,
            "prefix_inventory_sha256": prefix_inventory_sha256,
            "entity_prefix_counts": entity_prefix_counts,
            "entity_prefix_bytes": entity_prefix_bytes,
            "partial_prefix": partial_prefix,
        },
        "access_accounting": {
            "source_prefix_files_hashed": len(prefixes),
            "complete_parquet_footers_opened": complete_files,
            "source_row_values_decoded": 0,
            "current_g1_label_rows_read": 0,
            "model_selection_rows_read": 0,
            "holdout_or_private_g4_rows_read": 0,
            "seal_open_events": 0,
            "g4_open": False,
            "models_fitted": 0,
        },
        "authorization": {
            "background_full_core_transfer_resume": "AUTHORIZED_NEXT",
            "source_row_decode": "NOT_AUTHORIZED",
            "confirmatory_successor_execution": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": canary["claim_boundary"],
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify()
    if args.output is not None:
        _write_create_only(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
