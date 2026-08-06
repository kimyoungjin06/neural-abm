#!/usr/bin/env python3
"""Independently verify the bounded OpenAlex taxonomy download dry-run."""

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
DOWNLOAD_SCRIPT = ROOT / "scripts/download_pivot_feedback_lean_h3_openalex_v1.py"
DOWNLOAD_SCRIPT_SHA256 = "20fbcdd971691597f452473a735fbd85944c18cc425f926ccd4bcf54c58a3ece"
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
DOWNLOAD_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-taxonomy-dry-run-v1-20260806")
PLAN_SHA256 = "33a25c5a237011484b5b72d937243b29c0bca1415291811dce8bbf7a0a886103"
STATE_SHA256 = "b0a9387fec1842c6909567c82d1fd557aa200c36c82abeed806a27d509a69616"
RECEIPT_SHA256 = "839715c7bf20048815781914bea852ca5b6bdac8ab4fb162dcc4569ba5048c37"
ENTITIES = ("topics", "subfields", "fields")
STATUS = "PASS_INDEPENDENT_TAXONOMY_RANGE_RESUME_HASH_AND_FOOTER_CLOSURE_NO_ROW_DECODE"


class TaxonomyDryRunVerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TaxonomyDryRunVerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _manifest_inventory(source_receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bindings = {item["label"]: item for item in source_receipt["captured_manifests"]}
    inventory: dict[str, dict[str, Any]] = {}
    for entity in ENTITIES:
        binding = bindings[entity]
        path = MANIFEST_ROOT / str(binding["relative_path"])
        require(path.is_file() and not path.is_symlink(), f"invalid manifest: {entity}")
        require(path.stat().st_size == binding["bytes"], f"manifest size drift: {entity}")
        require(_sha256(path) == binding["sha256"], f"manifest hash drift: {entity}")
        manifest = _load_json(path)
        require(manifest["date"] == "2026-06-26" and manifest["entity"] == entity, f"manifest identity drift: {entity}")
        for item in manifest["files"]:
            s3_url = str(item["url"])
            prefix = f"s3://openalex/data/parquet/{entity}/"
            require(s3_url.startswith(prefix), f"manifest URL prefix drift: {s3_url}")
            relative = s3_url.removeprefix("s3://openalex/")
            pure = PurePosixPath(relative)
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe manifest path: {relative}")
            inventory[relative] = {
                "entity": entity,
                "s3_url": s3_url,
                "http_url": f"https://openalex.s3.amazonaws.com/{relative}",
                "expected_bytes": int(item["meta"]["content_length"]),
                "expected_records": int(item["meta"]["record_count"]),
            }
    require(len(inventory) == 8, "manifest inventory file-count drift")
    require(sum(item["expected_bytes"] for item in inventory.values()) == 2_783_017, "manifest inventory byte drift")
    return inventory


def verify() -> dict[str, Any]:
    repository_bindings = (
        (DOWNLOAD_SCRIPT, DOWNLOAD_SCRIPT_SHA256),
        (SOURCE_RECEIPT, SOURCE_RECEIPT_SHA256),
        (FOOTER_PREFLIGHT, FOOTER_PREFLIGHT_SHA256),
    )
    for path, expected_hash in repository_bindings:
        require(path.is_file() and not path.is_symlink(), f"invalid repository binding: {path.name}")
        require(_sha256(path) == expected_hash, f"repository binding drift: {path.name}")
    preflight = _load_json(FOOTER_PREFLIGHT)
    require(preflight["authorization"]["manifest_driven_downloader_implementation_and_dry_run"] == "AUTHORIZED_NEXT", "dry-run authority drift")
    require(preflight["authorization"]["full_source_data_download"] == "NOT_AUTHORIZED_BY_THIS_PREFLIGHT", "full transfer authority drift")
    source_receipt = _load_json(SOURCE_RECEIPT)
    expected = _manifest_inventory(source_receipt)

    require(DOWNLOAD_ROOT.is_dir() and not DOWNLOAD_ROOT.is_symlink(), "invalid download root")
    directories = [DOWNLOAD_ROOT, *(path for path in DOWNLOAD_ROOT.rglob("*") if path.is_dir())]
    require(all(not path.is_symlink() for path in directories), "symlinked download directory")
    require(all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories), "download directory mode drift")

    plan_path = DOWNLOAD_ROOT / "download_plan.json"
    state_path = DOWNLOAD_ROOT / "download_state.json"
    receipt_path = DOWNLOAD_ROOT / "download_receipt.json"
    lock_path = DOWNLOAD_ROOT / "download.lock"
    require(_sha256(plan_path) == PLAN_SHA256, "plan hash drift")
    require(_sha256(state_path) == STATE_SHA256, "state hash drift")
    require(_sha256(receipt_path) == RECEIPT_SHA256, "receipt hash drift")
    control_files = {plan_path, state_path, receipt_path, lock_path}
    data_files = {path for path in DOWNLOAD_ROOT.rglob("*.parquet") if path.is_file()}
    all_files = {path for path in DOWNLOAD_ROOT.rglob("*") if path.is_file()}
    require(all_files == control_files | data_files, "download root file surface drift")
    require(not list(DOWNLOAD_ROOT.rglob("*.part")), "partial files remain after completion")
    require(all(not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o600 for path in all_files), "download file mode drift")

    plan = _load_json(plan_path)
    state = _load_json(state_path)
    receipt = _load_json(receipt_path)
    require(plan["schema_version"] == "pivot_feedback_lean_h3_openalex_download_plan.v1", "plan schema drift")
    require(plan["status"] == "BOUND_TAXONOMY_DRY_RUN_ONLY_FULL_CORE_TRANSFER_NOT_AUTHORIZED", "plan status drift")
    require(plan["scope"] == "TOPICS_SUBFIELDS_FIELDS_ONLY", "plan scope drift")
    require(plan["file_count"] == 8 and plan["expected_bytes"] == 2_783_017, "plan totals drift")
    plan_by_path = {str(item["relative_path"]): item for item in plan["files"]}
    require(set(plan_by_path) == set(expected), "plan path surface drift")
    for relative, manifest_item in expected.items():
        plan_item = plan_by_path[relative]
        require(plan_item["entity"] == manifest_item["entity"], f"plan entity drift: {relative}")
        require(plan_item["s3_url"] == manifest_item["s3_url"], f"plan S3 URL drift: {relative}")
        require(plan_item["http_url"] == manifest_item["http_url"], f"plan HTTP URL drift: {relative}")
        require(plan_item["expected_bytes"] == manifest_item["expected_bytes"], f"plan byte drift: {relative}")
        require(plan_item["expected_records"] == manifest_item["expected_records"], f"plan row-count drift: {relative}")

    require(state["status"] == "COMPLETE", "download state is not complete")
    require(state["plan_sha256"] == PLAN_SHA256, "state plan hash drift")
    require(state["run_count"] == 2, "run-count drift")
    require(state["intentional_pause_count"] == 1, "pause-count drift")
    require(state["resumed_partial_file_count"] == 1, "resume-count drift")
    require(state["http_head_requests"] == 9, "HEAD request-count drift")
    require(state["http_range_requests"] == 9, "range request-count drift")
    require(state["remote_bytes_transferred"] == 2_783_017, "transferred byte-count drift")

    require(receipt["status"] == "PASS_EXACT_MANIFEST_TAXONOMY_RANGE_RESUME_LOCAL_HASH_INVENTORY_NO_ROW_DECODE", "receipt status drift")
    require(receipt["plan_sha256"] == PLAN_SHA256 and receipt["terminal_state_sha256"] == STATE_SHA256, "receipt control binding drift")
    require(receipt["file_count"] == 8 and receipt["downloaded_bytes"] == 2_783_017, "receipt totals drift")
    receipt_by_path = {str(item["relative_path"]): item for item in receipt["inventory"]}
    require(set(receipt_by_path) == set(expected), "receipt path surface drift")

    verified_inventory: list[dict[str, Any]] = []
    schema_hashes: dict[str, set[str]] = {entity: set() for entity in ENTITIES}
    entity_rows: dict[str, int] = {entity: 0 for entity in ENTITIES}
    for relative, manifest_item in sorted(expected.items()):
        path = DOWNLOAD_ROOT.joinpath(*PurePosixPath(relative).parts)
        receipt_item = receipt_by_path[relative]
        state_item = state["files"][relative]
        require(path in data_files and path.stat().st_size == manifest_item["expected_bytes"], f"data size drift: {relative}")
        content_sha256 = _sha256(path)
        require(receipt_item["sha256"] == state_item["final_sha256"] == content_sha256, f"data hash drift: {relative}")
        require(receipt_item["s3_url"] == manifest_item["s3_url"], f"receipt URL drift: {relative}")
        parquet = pq.ParquetFile(path)
        require(parquet.metadata.num_rows == manifest_item["expected_records"], f"footer row-count drift: {relative}")
        schema_sha256 = hashlib.sha256(parquet.schema_arrow.remove_metadata().serialize().to_pybytes()).hexdigest()
        entity = str(manifest_item["entity"])
        schema_hashes[entity].add(schema_sha256)
        entity_rows[entity] += parquet.metadata.num_rows
        verified_inventory.append(
            {
                "entity": entity,
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": content_sha256,
                "footer_rows": parquet.metadata.num_rows,
                "footer_row_groups": parquet.metadata.num_row_groups,
                "schema_sha256": schema_sha256,
            }
        )
    require(entity_rows == {"topics": 4516, "subfields": 252, "fields": 26}, "entity footer row totals drift")
    access = receipt["access_accounting"]
    require(access["taxonomy_source_files_materialized_locally"] == 8, "taxonomy materialization count drift")
    require(access["taxonomy_source_bytes_materialized_locally"] == 2_783_017, "taxonomy materialization bytes drift")
    require(access["works_source_files_materialized_locally"] == 0, "Works file access drift")
    require(access["source_row_values_decoded"] == 0, "row decode drift")
    require(access["holdout_or_private_g4_rows_read"] == 0 and access["g4_open"] is False, "holdout boundary drift")

    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_taxonomy_dry_run_verification_summary.v1",
        "status": STATUS,
        "verified_at_utc": _utc_now(),
        "mode": "INDEPENDENT_LOCAL_BYTE_HASH_AND_PARQUET_FOOTER_REPLAY_NO_ROW_DECODE_NOT_SCIENTIFIC_GATE",
        "bindings": {
            "download_script_sha256": DOWNLOAD_SCRIPT_SHA256,
            "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
            "footer_preflight_sha256": FOOTER_PREFLIGHT_SHA256,
            "download_plan_sha256": PLAN_SHA256,
            "download_state_sha256": STATE_SHA256,
            "download_receipt_sha256": RECEIPT_SHA256,
        },
        "download_replay": {
            "scope": plan["scope"],
            "file_count": len(verified_inventory),
            "bytes": sum(item["bytes"] for item in verified_inventory),
            "entity_footer_rows": entity_rows,
            "entity_schema_sha256": {entity: sorted(values) for entity, values in schema_hashes.items()},
            "inventory": verified_inventory,
        },
        "resume_replay": receipt["resume_evidence"],
        "access_accounting": {
            **access,
            "local_parquet_footer_files_opened": len(verified_inventory),
            "local_parquet_row_values_decoded": 0,
        },
        "authorization": receipt["authorization"],
        "claim_boundary": receipt["claim_boundary"],
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
