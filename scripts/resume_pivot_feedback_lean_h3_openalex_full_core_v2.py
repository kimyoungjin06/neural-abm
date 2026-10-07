#!/usr/bin/env python3
"""Resume the exact-manifest OpenAlex lean-H3 full-core transfer under the v2 capacity rule.

The v1 downloader is loaded unchanged and verified by hash. Only its runtime
capacity reserve is replaced, as the v2 authorization registers: the source
transfer needs the remaining source bytes plus a safety reserve, and the
transform-and-replay reserve is deferred to the stage that needs it. The plan,
state, resume controls, and row-decode prohibition are the v1 ones.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER_V1 = ROOT / "scripts" / "download_pivot_feedback_lean_h3_openalex_full_core_v1.py"
DOWNLOADER_V1_SHA256 = "7c16af6e38569889969f7528dbbd2cb1bccef87ab407fc5ae4cd23bd62f59e5f"
AUTHORIZATION_V2 = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v2.json"
)
AUTHORIZATION_V2_SHA256 = "8f48356dc7db8514645c6c864bc3f192aba5fc645f74ddc2faa7c4e2818e2176"
RESUME_RECEIPT_SUFFIX = ".resume-v2-receipt.json"


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


def load_authorization() -> dict[str, Any]:
    require(_sha256(AUTHORIZATION_V2) == AUTHORIZATION_V2_SHA256, "v2 authorization hash drift")
    authorization = json.loads(AUTHORIZATION_V2.read_text(encoding="utf-8"))
    amends = authorization["amends"]
    require(amends["changed"] == ["RUNTIME_CAPACITY_RULE_ONLY"], "v2 changes more than the capacity rule")
    v1 = ROOT / amends["authorization_v1"]["relative_path"]
    require(_sha256(v1) == amends["authorization_v1"]["sha256"], "v1 authorization hash drift")
    require(amends["downloader_v1"]["sha256"] == DOWNLOADER_V1_SHA256, "v1 downloader binding drift")
    grants = authorization["authorization"]
    require(grants["resume_of_the_full_core_source_byte_transfer"] == "AUTHORIZED", "resume not authorized")
    for key in ("source_row_decode", "transform_or_materialization", "authors_transfer", "confirmatory_successor_execution"):
        require(grants[key] == "NOT_AUTHORIZED", f"{key} must stay unauthorized")
    reserve = int(authorization["capacity_contract"]["source_phase_safety_reserve_bytes"])
    require(0 < reserve < int(authorization["capacity_contract"]["v1_working_reserve_bytes"]), "safety reserve out of range")
    return authorization


def load_downloader_v1(reserve_bytes: int) -> ModuleType:
    """Load the v1 downloader unchanged and replace only its runtime capacity reserve."""

    require(_sha256(DOWNLOADER_V1) == DOWNLOADER_V1_SHA256, "v1 downloader hash drift")
    spec = importlib.util.spec_from_file_location("lean_h3_openalex_full_core_v1", DOWNLOADER_V1)
    require(spec is not None and spec.loader is not None, "cannot load the v1 downloader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The plan records the v1 reserve, so the constant stays untouched and the
    # plan byte-identical. Only the runtime check the run loop calls is replaced.
    module._capacity_check = _source_phase_capacity_check(module, int(reserve_bytes))
    module.V2_RUNTIME_RESERVE_BYTES = int(reserve_bytes)
    return module


def _source_phase_capacity_check(module: ModuleType, reserve_bytes: int) -> Any:
    """The v1 capacity check with the registered source-phase safety reserve."""

    def check(state: dict[str, Any]) -> dict[str, int]:
        available = module.shutil.disk_usage(module.OUTPUT_ROOT).free
        downloaded = module._downloaded_bytes(state)
        remaining = module.EXPECTED_BYTES - downloaded
        required = remaining + reserve_bytes
        margin = available - required
        snapshot = {
            "available_bytes": available,
            "downloaded_bytes": downloaded,
            "remaining_source_bytes": remaining,
            "working_reserve_bytes": reserve_bytes,
            "required_available_bytes": required,
            "margin_bytes": margin,
            "capacity_rule": "V2_SOURCE_PHASE_REMAINING_PLUS_SAFETY_RESERVE",
        }
        state["last_capacity_check"] = snapshot
        previous = state["minimum_capacity_margin_bytes"]
        state["minimum_capacity_margin_bytes"] = margin if previous is None else min(int(previous), margin)
        module.require(margin >= 0, "capacity invariant failed: available < remaining source + v2 safety reserve")
        return snapshot

    return check


def resume_receipt_path(module: ModuleType) -> Path:
    return module.OUTPUT_ROOT.with_name(module.OUTPUT_ROOT.name + RESUME_RECEIPT_SUFFIX)


def preflight(module: ModuleType, authorization: dict[str, Any]) -> dict[str, Any]:
    """Validate the v1 plan and state without any network access."""

    plan = module.build_full_core_plan()
    plan_path = module.OUTPUT_ROOT / module.PLAN_NAME
    require(module._load_json(plan_path) == plan, "full-core plan drift")
    plan_sha256 = module._sha256(plan_path)
    require(plan_sha256 == authorization["stall_record"]["plan_sha256"], "plan differs from the stall record")
    state = module._load_json(module.OUTPUT_ROOT / module.STATE_NAME)
    module._validate_existing_state(plan, state, plan_sha256)
    available = module.shutil.disk_usage(module.OUTPUT_ROOT).free
    remaining = module.EXPECTED_BYTES - module._downloaded_bytes(state)
    reserve = int(authorization["capacity_contract"]["source_phase_safety_reserve_bytes"])
    return {
        "status": "PASS_V2_RESUME_PREFLIGHT_NO_NETWORK",
        "complete_files": module._complete_files(state),
        "total_files": module.EXPECTED_FILE_COUNT,
        "remaining_source_bytes": remaining,
        "available_bytes": available,
        "required_available_bytes": remaining + reserve,
        "margin_bytes": available - remaining - reserve,
        "v1_rule_margin_bytes": available - remaining - int(authorization["capacity_contract"]["v1_working_reserve_bytes"]),
    }


def resume(module: ModuleType, authorization: dict[str, Any]) -> dict[str, Any]:
    receipt_path = resume_receipt_path(module)
    require(not receipt_path.exists(), "a v2 resume receipt already exists")
    started = datetime.now(timezone.utc).isoformat()
    before = module._sha256(module.OUTPUT_ROOT / module.STATE_NAME)
    result = module.run_full_core(auto_resume=True, max_new_bytes=None)
    terminal = module.OUTPUT_ROOT / module.TERMINAL_RECEIPT_NAME
    receipt = {
        "schema_version": "pivot_feedback_lean_h3_openalex_full_core_resume_receipt.v2",
        "status": "PASS_V2_RESUME_COMPLETED_THE_EXACT_MANIFEST_TRANSFER_NO_ROW_DECODE",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_v2_sha256": AUTHORIZATION_V2_SHA256,
        "downloader_v1_sha256": DOWNLOADER_V1_SHA256,
        "runtime_reserve_bytes": int(module.V2_RUNTIME_RESERVE_BYTES),
        "state_sha256_before_resume": before,
        "terminal_receipt_sha256": module._sha256(terminal),
        "terminal_state_sha256": module._sha256(module.OUTPUT_ROOT / module.STATE_NAME),
        "result": result,
        "authorization": {"source_row_decode": "NOT_AUTHORIZED", "transform_or_materialization": "NOT_AUTHORIZED"},
    }
    require(receipt["status"].startswith("PASS") and str(result["status"]).startswith("PASS"), "transfer did not complete")
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_path.chmod(0o600)
    return {**receipt, "receipt_path": str(receipt_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "resume"))
    args = parser.parse_args()
    try:
        authorization = load_authorization()
        module = load_downloader_v1(int(authorization["capacity_contract"]["source_phase_safety_reserve_bytes"]))
        result = preflight(module, authorization) if args.command == "preflight" else resume(module, authorization)
    except (ResumeAuthorizationError, RuntimeError) as exc:
        print(json.dumps({"status": "STOPPED_FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 75
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
