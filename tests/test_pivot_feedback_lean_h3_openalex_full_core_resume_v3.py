from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resume_pivot_feedback_lean_h3_openalex_full_core_v3.py"
AUTHORIZATION = ROOT / "experiments" / "configs" / "pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v3.json"
OUTPUT_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-full-core-v1-20260806")
needs_root = pytest.mark.skipif(not OUTPUT_ROOT.exists(), reason="the transfer root is machine-local")


def _resume_module():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("resume_v3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _authorization() -> dict:
    resume = _resume_module()
    return resume.load_authorization(resume.load_resume_v2())


def test_v3_changes_only_the_remote_addressing_and_keeps_every_prohibition() -> None:
    authorization = _authorization()
    assert authorization["amends"]["changed"] == ["REMOTE_OBJECT_ADDRESSING_BY_BOUND_S3_VERSION_ID"]
    assert "V2_RUNTIME_CAPACITY_RULE" in authorization["amends"]["unchanged"]
    grants = authorization["authorization"]
    for key in ("source_row_decode", "transform_or_materialization", "authors_transfer", "current_g4_or_holdout_access",
                "model_fitting", "confirmatory_successor_execution"):
        assert grants[key] == "NOT_AUTHORIZED"
    assert authorization["human_authorization"]["decision"] == "FINISH_THE_2026_06_26_SNAPSHOT_OPTION_A"


def test_the_bindings_are_the_71_pending_2026_06_26_works_parts() -> None:
    authorization = _authorization()
    bindings = authorization["version_bindings"]
    expected = [f"data/parquet/works/updated_date=2026-06-26/part_{index:04d}.parquet" for index in range(89, 160)]
    assert [row["relative_path"] for row in bindings] == expected
    assert sum(row["content_length_bytes"] for row in bindings) == 35_183_477_118
    assert len({row["version_id"] for row in bindings}) == 71
    assert all(" Jun 2026 " in row["last_modified"] for row in bindings)
    validation = authorization["version_evidence"]["rule_validation"]
    assert validation["rule_reproduces_the_v1_recorded_version_etag_size_and_last_modified"] == validation[
        "completed_works_2026_06_26_files"
    ] == 89


def test_tampered_authorization_is_refused(tmp_path: Path) -> None:
    resume = _resume_module()
    copy = tmp_path / "authorization.json"
    copy.write_text(AUTHORIZATION.read_text(encoding="utf-8").replace("PACFN1j", "PACFN1k"), encoding="utf-8")
    resume.AUTHORIZATION_V3 = copy
    with pytest.raises(resume.ResumeAuthorizationError, match="hash drift"):
        resume.load_authorization(resume.load_resume_v2())


def test_versioned_url_is_quoted_and_refuses_an_existing_query() -> None:
    resume = _resume_module()
    assert resume.versioned_url("https://h/k.parquet", "a.b_c/d") == "https://h/k.parquet?versionId=a.b_c%2Fd"
    with pytest.raises(resume.ResumeAuthorizationError):
        resume.versioned_url("https://h/k.parquet?x=1", "v")
    with pytest.raises(resume.ResumeAuthorizationError):
        resume.versioned_url("https://h/k.parquet", "")


class _Refused(RuntimeError):
    pass


def _fake_downloader(remote: dict) -> tuple[SimpleNamespace, list]:
    calls: list = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise _Refused(message)

    def head(url: str) -> dict:
        calls.append(("HEAD", url))
        return dict(remote)

    def open_range(url: str, start: int, end: int) -> str:
        calls.append(("GET", url, start, end))
        return "stream"

    return SimpleNamespace(require=require, dry=SimpleNamespace(_remote_metadata=head, _open_range=open_range)), calls


BINDING = {
    "relative_path": "data/k.parquet",
    "version_id": "V1",
    "etag": '"e-2"',
    "content_length_bytes": 10,
    "last_modified": "Sat, 27 Jun 2026 03:02:03 GMT",
}
PLAN = {"files": [{"relative_path": "data/k.parquet", "http_url": "https://h/data/k.parquet"},
                  {"relative_path": "data/done.parquet", "http_url": "https://h/data/done.parquet"}]}


def test_bound_calls_go_to_the_bound_version_and_unbound_urls_are_refused() -> None:
    resume = _resume_module()
    remote = {**{key: BINDING[key] for key in resume.IDENTITY_FIELDS}, "accept_ranges": "bytes"}
    module, calls = _fake_downloader(remote)
    resume.bind_versions(module, PLAN, [BINDING])
    assert module.dry._remote_metadata("https://h/data/k.parquet") == remote
    assert module.dry._open_range("https://h/data/k.parquet", 0, 9) == "stream"
    assert calls == [("HEAD", "https://h/data/k.parquet?versionId=V1"), ("GET", "https://h/data/k.parquet?versionId=V1", 0, 9)]
    with pytest.raises(_Refused, match="no bound version"):
        module.dry._open_range("https://h/data/done.parquet", 0, 9)


@pytest.mark.parametrize("field", ["version_id", "etag", "content_length_bytes", "last_modified"])
def test_a_head_that_differs_from_the_binding_fails_closed(field: str) -> None:
    resume = _resume_module()
    remote = {**{key: BINDING[key] for key in resume.IDENTITY_FIELDS}, "accept_ranges": "bytes"}
    remote[field] = 11 if field == "content_length_bytes" else "other"
    module, _calls = _fake_downloader(remote)
    resume.bind_versions(module, PLAN, [BINDING])
    with pytest.raises(_Refused, match=f"identity drift \\({field}\\)"):
        module.dry._remote_metadata("https://h/data/k.parquet")


def test_multipart_etag_matches_the_s3_construction(tmp_path: Path) -> None:
    resume = _resume_module()
    path = tmp_path / "object"
    data = bytes(range(256)) * 5
    path.write_bytes(data)
    parts = [hashlib.md5(data[start:start + 512]).digest() for start in range(0, len(data), 512)]
    assert resume.multipart_etag(path, 512) == f'"{hashlib.md5(b"".join(parts)).hexdigest()}-3"'
    assert resume.multipart_etag(path, 4096) == f'"{hashlib.md5(data).hexdigest()}"'


def test_the_resume_script_has_no_row_decode_surface() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"pyarrow", "pandas", "polars", "duckdb"}


@needs_root
def test_preflight_binds_exactly_the_pending_files_of_the_unchanged_plan() -> None:
    resume = _resume_module()
    resume_v2 = resume.load_resume_v2()
    authorization_v2 = resume_v2.load_authorization()
    authorization = resume.load_authorization(resume_v2)
    module = resume_v2.load_downloader_v1(int(authorization_v2["capacity_contract"]["source_phase_safety_reserve_bytes"]))
    state = json.loads((OUTPUT_ROOT / "download_state.json").read_text(encoding="utf-8"))
    if state["status"] == "COMPLETE":
        pytest.skip("the transfer has completed; the bindings are checked by the v3 receipt")
    result = resume.preflight(module, resume_v2, authorization_v2, authorization)
    assert result["status"] == "PASS_V3_RESUME_PREFLIGHT_NO_NETWORK"
    assert result["bound_pending_files"] == 71
    assert state["plan_sha256"] == authorization["stall_record"]["plan_sha256"]


RESULT = ROOT / "experiments" / "evidence" / "results" / (
    "pivot_feedback_lean_h3_openalex_full_core_resume_v3_20261007.summary.json"
)


def test_the_recorded_result_is_the_complete_exact_manifest_with_no_row_decode() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    authorization = _authorization()
    assert result["status"] == "PASS_THE_EXACT_2026_06_26_FULL_CORE_TRANSFER_IS_COMPLETE_NO_ROW_DECODE"
    assert result["authorization_v3"]["sha256"] == _resume_module().AUTHORIZATION_V3_SHA256
    assert result["pending_at_start"]["state_sha256"] == authorization["stall_record"]["state_sha256_at_amendment"]
    terminal = result["terminal"]
    assert (terminal["file_count"], terminal["downloaded_bytes"]) == (2_454, 724_973_106_144)
    assert terminal["plan_sha256"] == authorization["stall_record"]["plan_sha256"]
    controls = result["controls"]
    assert controls["bound_etag_check"]["checked_files"] == 71
    assert controls["bound_etag_check"]["etag_mismatches"] == []
    assert controls["new_bytes"] + controls["uncheckpointed_tail"]["bytes"] == result["pending_at_start"]["bytes"]
    assert all(value in (0, False) for value in terminal["access_accounting"].values())
    assert result["corrects"]["commit"] == "e7ae454"
    assert result["authorization"]["source_row_decode"] == "NOT_AUTHORIZED"


@needs_root
def test_the_receipts_on_disk_match_the_record() -> None:
    terminal = json.loads(RESULT.read_text(encoding="utf-8"))["terminal"]
    for receipt in (terminal["download_receipt"], terminal["resume_v3_receipt"]):
        path = Path(receipt["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == receipt["status"]
    state = OUTPUT_ROOT / "download_state.json"
    assert hashlib.sha256(state.read_bytes()).hexdigest() == terminal["state_sha256"]
