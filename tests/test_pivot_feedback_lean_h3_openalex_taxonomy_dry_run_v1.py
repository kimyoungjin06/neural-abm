from __future__ import annotations

import json
from pathlib import Path
import runpy
import stat

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER = ROOT / "scripts/download_pivot_feedback_lean_h3_openalex_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_openalex_taxonomy_dry_run_v1.py"
SUMMARY = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_openalex_taxonomy_dry_run_v1_20260806.summary.json"
)


def test_taxonomy_plan_is_exact_manifest_bound_and_has_no_works_scope() -> None:
    module = runpy.run_path(str(DOWNLOADER))
    plan = module["build_taxonomy_plan"]()
    assert plan["status"] == "BOUND_TAXONOMY_DRY_RUN_ONLY_FULL_CORE_TRANSFER_NOT_AUTHORIZED"
    assert plan["scope"] == "TOPICS_SUBFIELDS_FIELDS_ONLY"
    assert plan["file_count"] == 8
    assert plan["expected_bytes"] == 2_783_017
    assert {item["entity"] for item in plan["files"]} == {"topics", "subfields", "fields"}
    assert all(item["s3_url"].startswith("s3://openalex/data/parquet/") for item in plan["files"])
    assert all(item["http_url"].startswith("https://openalex.s3.amazonaws.com/data/parquet/") for item in plan["files"])
    assert plan["authorization"] == {
        "taxonomy_dry_run": "AUTHORIZED",
        "works_transfer": "NOT_AUTHORIZED",
        "source_row_decode": "NOT_AUTHORIZED",
        "confirmatory_execution": "NOT_AUTHORIZED",
    }


def test_terminal_resume_is_local_only_and_output_root_is_exact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = runpy.run_path(str(DOWNLOADER))

    def forbidden_remote(*args: object, **kwargs: object) -> None:
        raise AssertionError("terminal resume attempted remote access")

    monkeypatch.setitem(module["run_taxonomy_dry_run"].__globals__, "_remote_metadata", forbidden_remote)
    monkeypatch.setitem(module["run_taxonomy_dry_run"].__globals__, "_open_range", forbidden_remote)
    result = module["run_taxonomy_dry_run"](
        output_root=module["DEFAULT_OUTPUT_ROOT"],
        resume=True,
        max_new_bytes=None,
    )
    assert result["status"] == "PASS_TERMINAL_LOCAL_REVALIDATION_NO_REMOTE_ACCESS"
    with pytest.raises(module["OpenAlexDownloadError"], match="exact output root"):
        module["run_taxonomy_dry_run"](
            output_root=tmp_path.resolve(),
            resume=False,
            max_new_bytes=1,
        )


def test_completed_download_receipt_records_real_partial_resume() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == (
        "PASS_INDEPENDENT_TAXONOMY_RANGE_RESUME_HASH_AND_FOOTER_CLOSURE_NO_ROW_DECODE"
    )
    assert summary["download_replay"]["file_count"] == 8
    assert summary["download_replay"]["bytes"] == 2_783_017
    assert summary["download_replay"]["entity_footer_rows"] == {
        "fields": 26,
        "subfields": 252,
        "topics": 4516,
    }
    resume = summary["resume_replay"]
    assert resume == {
        "http_head_requests": 9,
        "http_range_requests": 9,
        "intentional_pause_count": 1,
        "remote_bytes_transferred": 2_783_017,
        "resumed_partial_file_count": 1,
        "run_count": 2,
    }
    access = summary["access_accounting"]
    assert access["taxonomy_source_files_materialized_locally"] == 8
    assert access["taxonomy_source_bytes_materialized_locally"] == 2_783_017
    assert access["works_source_files_materialized_locally"] == 0
    assert access["source_row_values_decoded"] == 0
    assert access["holdout_or_private_g4_rows_read"] == 0
    assert access["g4_open"] is False


def test_independent_verifier_replays_saved_summary_except_timestamp() -> None:
    module = runpy.run_path(str(VERIFIER))
    replay = module["verify"]()
    saved = json.loads(SUMMARY.read_text(encoding="utf-8"))
    replay.pop("verified_at_utc")
    saved.pop("verified_at_utc")
    assert replay == saved


def test_download_root_is_private_terminal_and_verifier_never_decodes_rows() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    download_root = Path("/ssd/pivot-feedback-lean-h3-openalex-taxonomy-dry-run-v1-20260806")
    assert stat.S_IMODE(download_root.stat().st_mode) == 0o700
    assert not list(download_root.rglob("*.part"))
    assert len(list(download_root.rglob("*.parquet"))) == 8
    assert all(
        not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in download_root.rglob("*")
        if path.is_file()
    )
    source = VERIFIER.read_text(encoding="utf-8")
    assert "pq.ParquetFile(" in source
    for forbidden in ("pq.read_table(", ".read_row_group(", ".iter_batches("):
        assert forbidden not in source
    assert summary["authorization"]["works_or_full_core_transfer"] == (
        "NOT_AUTHORIZED_BY_THIS_RECEIPT"
    )
