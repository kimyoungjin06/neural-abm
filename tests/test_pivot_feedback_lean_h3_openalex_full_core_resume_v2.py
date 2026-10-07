from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resume_pivot_feedback_lean_h3_openalex_full_core_v2.py"
AUTHORIZATION = ROOT / "experiments" / "configs" / "pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v2.json"
OUTPUT_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-full-core-v1-20260806")
needs_root = pytest.mark.skipif(not OUTPUT_ROOT.exists(), reason="the transfer root is machine-local")


def _resume_module():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("resume_v2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_authorization_changes_only_the_runtime_capacity_rule() -> None:
    resume = _resume_module()
    authorization = resume.load_authorization()
    assert authorization["amends"]["changed"] == ["RUNTIME_CAPACITY_RULE_ONLY"]
    grants = authorization["authorization"]
    assert grants["resume_of_the_full_core_source_byte_transfer"] == "AUTHORIZED"
    for key in ("source_row_decode", "transform_or_materialization", "authors_transfer", "current_g4_or_holdout_access"):
        assert grants[key] == "NOT_AUTHORIZED"
    contract = authorization["capacity_contract"]
    assert contract["deferred"]["working_transformed_and_replay_cap_bytes"] == contract["v1_working_reserve_bytes"]
    assert contract["deferred"]["authorized_here"] is False


def test_tampered_authorization_is_refused(tmp_path: Path) -> None:
    resume = _resume_module()
    copy = tmp_path / "authorization.json"
    copy.write_text(AUTHORIZATION.read_text(encoding="utf-8").replace("AUTHORIZED_RESUME", "AUTHORIZED_ANYTHING"), encoding="utf-8")
    resume.AUTHORIZATION_V2 = copy
    with pytest.raises(resume.ResumeAuthorizationError, match="hash drift"):
        resume.load_authorization()


def test_the_v1_downloader_is_loaded_unchanged_except_the_runtime_check() -> None:
    resume = _resume_module()
    module = resume.load_downloader_v1(107374182400)
    assert module.WORKING_RESERVE_BYTES == 1_087_459_659_216
    assert module.V2_RUNTIME_RESERVE_BYTES == 107374182400


def test_the_source_phase_check_uses_the_v2_reserve_and_fails_closed() -> None:
    resume = _resume_module()
    module = resume.load_downloader_v1(100)
    module.EXPECTED_BYTES = 1_000
    module.shutil = SimpleNamespace(disk_usage=lambda _root: SimpleNamespace(free=400))
    state = {"files": {"a": {"downloaded_bytes": 700}}, "minimum_capacity_margin_bytes": None, "last_capacity_check": None}
    snapshot = module._capacity_check(state)
    assert (snapshot["remaining_source_bytes"], snapshot["required_available_bytes"], snapshot["margin_bytes"]) == (300, 400, 0)
    state["files"]["a"]["downloaded_bytes"] = 600
    with pytest.raises(module.FullCoreTransferError, match="v2 safety reserve"):
        module._capacity_check(state)


def test_the_resume_script_has_no_row_decode_surface() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"pyarrow", "pandas", "polars", "duckdb"}


@needs_root
def test_preflight_rebuilds_the_identical_plan_and_validates_the_state() -> None:
    resume = _resume_module()
    authorization = resume.load_authorization()
    module = resume.load_downloader_v1(int(authorization["capacity_contract"]["source_phase_safety_reserve_bytes"]))
    result = resume.preflight(module, authorization)
    assert result["status"] == "PASS_V2_RESUME_PREFLIGHT_NO_NETWORK"
    assert result["total_files"] == 2454
    state = json.loads((OUTPUT_ROOT / "download_state.json").read_text(encoding="utf-8"))
    assert state["plan_sha256"] == authorization["stall_record"]["plan_sha256"]
