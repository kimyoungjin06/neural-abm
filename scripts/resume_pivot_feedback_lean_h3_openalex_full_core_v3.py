#!/usr/bin/env python3
"""Resume the exact-manifest OpenAlex lean-H3 full-core transfer from bound S3 object versions.

The 2026-09-23 OpenAlex snapshot placed delete markers over the 2026-06-26
objects, so an unversioned URL now returns 404. The bucket keeps versions, and
the exact 2026-06-26 objects remain retrievable by version id. The v3
authorization binds one version per pending file. This script loads the v2
resume unchanged, verified by hash, which in turn loads the v1 downloader with
the v2 capacity rule. It then routes the downloader's two remote calls, HEAD
and range GET, to the bound version of each pending file. Every HEAD must
return the bound version id, ETag, size, and last-modified date, and any
unbound URL is refused. The plan, state, segment and tail-recovery controls,
and row-decode prohibition are the v1 ones. After the transfer, each bound
file's S3 multipart ETag is recomputed locally and must equal the bound ETag.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, BinaryIO, Callable
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
RESUME_V2 = ROOT / "scripts" / "resume_pivot_feedback_lean_h3_openalex_full_core_v2.py"
RESUME_V2_SHA256 = "b8594a642e4166be179e0d65a57865f3e1915872154ad2774af9820c8f189f2b"
AUTHORIZATION_V3 = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v3.json"
)
AUTHORIZATION_V3_SHA256 = "468895a4980307a4131674d5268eb6e4432b476ed97cd945c401813c55cdf375"
RESUME_RECEIPT_SUFFIX = ".resume-v3-receipt.json"
IDENTITY_FIELDS = ("version_id", "etag", "content_length_bytes", "last_modified")


class ResumeAuthorizationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResumeAuthorizationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resume_v2() -> ModuleType:
    require(_sha256(RESUME_V2) == RESUME_V2_SHA256, "v2 resume hash drift")
    spec = importlib.util.spec_from_file_location("resume_v2", RESUME_V2)
    require(spec is not None and spec.loader is not None, "cannot load the v2 resume")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_authorization(resume_v2: ModuleType) -> dict[str, Any]:
    require(_sha256(AUTHORIZATION_V3) == AUTHORIZATION_V3_SHA256, "v3 authorization hash drift")
    authorization = json.loads(AUTHORIZATION_V3.read_text(encoding="utf-8"))
    amends = authorization["amends"]
    require(amends["changed"] == ["REMOTE_OBJECT_ADDRESSING_BY_BOUND_S3_VERSION_ID"], "v3 changes more than addressing")
    require(amends["authorization_v2"]["sha256"] == resume_v2.AUTHORIZATION_V2_SHA256, "v2 authorization binding drift")
    require(amends["resume_v2"]["sha256"] == RESUME_V2_SHA256, "v2 resume binding drift")
    grants = authorization["authorization"]
    require(grants["resume_of_the_full_core_source_byte_transfer"] == "AUTHORIZED", "resume not authorized")
    for key in ("source_row_decode", "transform_or_materialization", "authors_transfer", "confirmatory_successor_execution"):
        require(grants[key] == "NOT_AUTHORIZED", f"{key} must stay unauthorized")
    bindings = authorization["version_bindings"]
    paths = [row["relative_path"] for row in bindings]
    require(len(paths) == len(set(paths)) == authorization["stall_record"]["pending_files"], "version binding surface drift")
    for row in bindings:
        require(set(row) == {"relative_path", *IDENTITY_FIELDS}, f"version binding fields drift: {row['relative_path']}")
    return authorization


def versioned_url(http_url: str, version_id: str) -> str:
    require("?" not in http_url and bool(version_id), f"cannot version this url: {http_url}")
    return f"{http_url}?versionId={quote(version_id, safe='')}"


def bind_versions(module: ModuleType, plan: dict[str, Any], bindings: list[dict[str, Any]]) -> None:
    """Route the downloader's HEAD and range calls to the bound object versions only."""

    by_path = {row["relative_path"]: row for row in bindings}
    by_url = {str(item["http_url"]): by_path[item["relative_path"]] for item in plan["files"] if item["relative_path"] in by_path}
    require(len(by_url) == len(by_path), "a version binding has no plan file")
    head: Callable[[str], dict[str, Any]] = module.dry._remote_metadata
    open_range: Callable[[str, int, int], BinaryIO] = module.dry._open_range

    def bound(http_url: str) -> dict[str, Any]:
        row = by_url.get(http_url)
        module.require(row is not None, f"no bound version for this url: {http_url}")
        return row

    def versioned_head(http_url: str) -> dict[str, Any]:
        row = bound(http_url)
        remote = head(versioned_url(http_url, row["version_id"]))
        for field in IDENTITY_FIELDS:
            module.require(remote[field] == row[field], f"bound version identity drift ({field}): {row['relative_path']}")
        return remote

    def versioned_range(http_url: str, start: int, end: int) -> BinaryIO:
        row = bound(http_url)
        return open_range(versioned_url(http_url, row["version_id"]), start, end)

    module.dry._remote_metadata = versioned_head
    module.dry._open_range = versioned_range


def multipart_etag(path: Path, part_bytes: int) -> str:
    """The S3 multipart ETag of a local file: the MD5 of its parts' MD5s, with the part count."""

    digests = []
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(part_bytes), b""):
            digests.append(hashlib.md5(chunk).digest())
    if len(digests) == 1:
        return f'"{digests[0].hex()}"'
    return f'"{hashlib.md5(b"".join(digests)).hexdigest()}-{len(digests)}"'


def check_bound_etags(module: ModuleType, authorization: dict[str, Any]) -> dict[str, Any]:
    part_bytes = int(authorization["post_transfer_check"]["multipart_part_bytes"])
    state = module._load_json(module.OUTPUT_ROOT / module.STATE_NAME)
    mismatches = []
    for row in authorization["version_bindings"]:
        entry = state["files"][row["relative_path"]]
        require(entry["status"] == "COMPLETE", f"bound file is not complete: {row['relative_path']}")
        require({field: entry["remote"][field] for field in IDENTITY_FIELDS} == {field: row[field] for field in IDENTITY_FIELDS},
                f"recorded remote identity drift: {row['relative_path']}")
        etag = row["etag"].strip('"')
        parts = int(etag.split("-")[1]) if "-" in etag else 1
        require(math.ceil(row["content_length_bytes"] / part_bytes) == parts, f"etag part count drift: {row['relative_path']}")
        target = module.OUTPUT_ROOT.joinpath(*PurePosixPath(row["relative_path"]).parts)
        if multipart_etag(target, part_bytes) != row["etag"]:
            mismatches.append(row["relative_path"])
    return {"checked_files": len(authorization["version_bindings"]), "etag_mismatches": mismatches}


def resume_receipt_path(module: ModuleType) -> Path:
    return module.OUTPUT_ROOT.with_name(module.OUTPUT_ROOT.name + RESUME_RECEIPT_SUFFIX)


def preflight(module: ModuleType, resume_v2: ModuleType, authorization_v2: dict[str, Any],
              authorization: dict[str, Any]) -> dict[str, Any]:
    """The v2 network-free preflight, plus the version bindings against the state."""

    result = resume_v2.preflight(module, authorization_v2)
    listing = authorization["version_evidence"]["listing"]
    require(_sha256(Path(listing["path"])) == listing["sha256"], "version listing hash drift")
    state = module._load_json(module.OUTPUT_ROOT / module.STATE_NAME)
    pending = {relative for relative, entry in state["files"].items() if entry["status"] != "COMPLETE"}
    bound = {row["relative_path"] for row in authorization["version_bindings"]}
    require(pending == bound, "the version bindings do not cover exactly the pending files")
    plan = module._load_json(module.OUTPUT_ROOT / module.PLAN_NAME)
    expected = {item["relative_path"]: int(item["expected_bytes"]) for item in plan["files"]}
    for row in authorization["version_bindings"]:
        require(row["content_length_bytes"] == expected[row["relative_path"]], f"bound size drift: {row['relative_path']}")
    remaining = sum(row["content_length_bytes"] for row in authorization["version_bindings"])
    require(remaining == result["remaining_source_bytes"], "bound bytes differ from the remaining source bytes")
    return {**result, "status": "PASS_V3_RESUME_PREFLIGHT_NO_NETWORK", "bound_pending_files": len(bound)}


def resume(module: ModuleType, authorization: dict[str, Any]) -> dict[str, Any]:
    receipt_path = resume_receipt_path(module)
    require(not receipt_path.exists(), "a v3 resume receipt already exists")
    plan = module._load_json(module.OUTPUT_ROOT / module.PLAN_NAME)
    bind_versions(module, plan, authorization["version_bindings"])
    started = datetime.now(timezone.utc).isoformat()
    before = module._sha256(module.OUTPUT_ROOT / module.STATE_NAME)
    result = module.run_full_core(auto_resume=True, max_new_bytes=None)
    require(str(result["status"]).startswith("PASS"), "transfer did not complete")
    etags = check_bound_etags(module, authorization)
    require(etags["etag_mismatches"] == [], f"bound etag mismatch: {etags['etag_mismatches']}")
    terminal = module.OUTPUT_ROOT / module.TERMINAL_RECEIPT_NAME
    receipt = {
        "schema_version": "pivot_feedback_lean_h3_openalex_full_core_resume_receipt.v3",
        "status": "PASS_V3_RESUME_COMPLETED_THE_EXACT_MANIFEST_TRANSFER_FROM_BOUND_VERSIONS_NO_ROW_DECODE",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_v3_sha256": AUTHORIZATION_V3_SHA256,
        "resume_v2_sha256": RESUME_V2_SHA256,
        "runtime_reserve_bytes": int(module.V2_RUNTIME_RESERVE_BYTES),
        "state_sha256_before_resume": before,
        "terminal_receipt_sha256": module._sha256(terminal),
        "terminal_state_sha256": module._sha256(module.OUTPUT_ROOT / module.STATE_NAME),
        "result": result,
        "bound_etag_check": etags,
        "authorization": {"source_row_decode": "NOT_AUTHORIZED", "transform_or_materialization": "NOT_AUTHORIZED"},
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_path.chmod(0o600)
    return {**receipt, "receipt_path": str(receipt_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "resume"))
    args = parser.parse_args()
    try:
        resume_v2 = load_resume_v2()
        authorization_v2 = resume_v2.load_authorization()
        authorization = load_authorization(resume_v2)
        module = resume_v2.load_downloader_v1(int(authorization_v2["capacity_contract"]["source_phase_safety_reserve_bytes"]))
        result = preflight(module, resume_v2, authorization_v2, authorization)
        if args.command == "resume":
            result = resume(module, authorization)
    except (ResumeAuthorizationError, RuntimeError) as exc:
        print(json.dumps({"status": "STOPPED_FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 75
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
