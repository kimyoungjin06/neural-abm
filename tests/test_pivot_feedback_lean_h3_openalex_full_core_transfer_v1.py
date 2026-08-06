from __future__ import annotations

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION = ROOT / (
    "experiments/configs/"
    "pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v1.json"
)
DOWNLOADER = ROOT / "scripts/download_pivot_feedback_lean_h3_openalex_full_core_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_openalex_full_core_canary_v1.py"
CANARY_SUMMARY = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_openalex_full_core_canary_v1_20260806.summary.json"
)


def test_authorization_is_exact_transfer_only_and_retains_scientific_boundary() -> None:
    authority = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    assert authority["status"] == (
        "AUTHORIZED_EXACT_MANIFEST_FULL_CORE_TRANSFER_OPERATIONAL_ONLY_NO_ROW_DECODE"
    )
    scope = authority["transfer_scope"]
    assert scope["entities"] == ["works", "topics", "subfields", "fields"]
    assert scope["exact_manifest_file_count"] == 2_454
    assert scope["exact_manifest_content_length_bytes"] == 724_973_106_144
    assert scope["authors_entity_included"] is False
    assert scope["mutable_prefix_sync_allowed"] is False
    capacity = authority["capacity_contract"]
    assert capacity["raw_source_cap_bytes"] == 724_973_106_144
    assert capacity["working_transformed_and_replay_cap_bytes"] == 1_087_459_659_216
    assert capacity["total_reserved_budget_bytes"] == 1_812_432_765_360
    assert capacity["observed_margin_after_total_reserve_bytes"] == 1_211_318_423_120
    assert authority["authorization"]["full_core_source_byte_transfer"] == "AUTHORIZED"
    assert authority["authorization"]["source_row_decode"] == "NOT_AUTHORIZED"
    assert authority["authorization"]["confirmatory_successor_execution"] == "NOT_AUTHORIZED"
    assert authority["authorization"]["scientific_effect"] == "NOT_ESTABLISHED"


def test_full_core_plan_reconstructs_all_exact_manifest_files() -> None:
    module = runpy.run_path(str(DOWNLOADER))
    plan = module["build_full_core_plan"]()
    assert plan["status"] == "AUTHORIZED_EXACT_MANIFEST_FULL_CORE_TRANSFER_NO_ROW_DECODE"
    assert plan["scope"] == "WORKS_TOPICS_SUBFIELDS_FIELDS"
    assert plan["file_count"] == 2_454
    assert plan["expected_bytes"] == 724_973_106_144
    assert plan["working_reserve_bytes"] == 1_087_459_659_216
    assert plan["total_reserved_budget_bytes"] == 1_812_432_765_360
    assert plan["segment_bytes"] == 268_435_456
    entity_counts = {
        entity: sum(item["entity"] == entity for item in plan["files"])
        for entity in ("works", "topics", "subfields", "fields")
    }
    assert entity_counts == {"works": 2446, "topics": 6, "subfields": 1, "fields": 1}
    assert "authors" not in {item["entity"] for item in plan["files"]}
    assert len({item["relative_path"] for item in plan["files"]}) == 2_454
    assert all(item["s3_url"].startswith("s3://openalex/data/parquet/") for item in plan["files"])
    assert all(item["http_url"].startswith("https://openalex.s3.amazonaws.com/data/parquet/") for item in plan["files"])
    assert all(plan["prohibitions"].values())


def test_capacity_invariant_fails_closed_below_remaining_plus_working_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path(str(DOWNLOADER))
    state = {
        "files": {
            "only": {"downloaded_bytes": 0},
        },
        "minimum_capacity_margin_bytes": None,
        "last_capacity_check": None,
    }
    required = module["EXPECTED_BYTES"] + module["WORKING_RESERVE_BYTES"]
    fake = SimpleNamespace(free=required - 1)
    monkeypatch.setattr(module["shutil"], "disk_usage", lambda _: fake)
    with pytest.raises(module["FullCoreTransferError"], match="capacity invariant failed"):
        module["_capacity_check"](state)
    assert state["last_capacity_check"]["margin_bytes"] == -1


def test_plan_only_passes_live_capacity_without_creating_output() -> None:
    module = runpy.run_path(str(DOWNLOADER))
    before = module["OUTPUT_ROOT"].exists()
    result = module["plan_only"]()
    assert result["status"] == "PASS_FULL_CORE_PLAN_ONLY_NO_TRANSFER"
    assert result["file_count"] == 2_454
    assert result["expected_bytes"] == 724_973_106_144
    assert result["prestart_capacity_pass"] is True
    assert result["margin_bytes"] >= 0
    assert module["OUTPUT_ROOT"].exists() is before


def test_full_core_downloader_contains_no_row_decode_surface() -> None:
    source = DOWNLOADER.read_text(encoding="utf-8")
    for forbidden in (
        "pyarrow",
        "pq.read_table(",
        ".read_row_group(",
        ".iter_batches(",
        "authors.manifest",
    ):
        assert forbidden not in source
    assert 'ENTITIES = ("works", "topics", "subfields", "fields")' in source
    assert 'OUTPUT_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-full-core-v1-20260806")' in source


def test_independent_canary_summary_closes_cumulative_cap_and_works_prefix() -> None:
    summary = json.loads(CANARY_SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == (
        "PASS_INDEPENDENT_FULL_CORE_CANARY_MANIFEST_PREFIX_FOOTER_AND_CAPACITY_CLOSURE_NO_ROW_DECODE"
    )
    replay = summary["canary_replay"]
    assert replay["total_plan_files"] == 2_454
    assert replay["total_plan_bytes"] == 724_973_106_144
    assert replay["canary_prefix_count"] == 209
    assert replay["canary_complete_files"] == 208
    assert replay["canary_partial_files"] == 1
    assert replay["canary_bytes"] == 67_108_864
    assert replay["canary_works_bytes"] == 64_325_847
    assert replay["entity_prefix_counts"] == {
        "fields": 1,
        "subfields": 1,
        "topics": 6,
        "works": 201,
    }
    assert replay["entity_prefix_bytes"] == {
        "fields": 19_238,
        "subfields": 174_417,
        "topics": 2_589_362,
        "works": 64_325_847,
    }
    assert replay["capacity_margin_bytes"] >= 0
    assert replay["partial_prefix"]["canary_bytes"] == 11_730_808
    assert summary["access_accounting"]["source_row_values_decoded"] == 0
    assert summary["authorization"]["background_full_core_transfer_resume"] == "AUTHORIZED_NEXT"


def test_canary_verifier_replays_prefix_receipt_after_live_state_advances() -> None:
    module = runpy.run_path(str(VERIFIER))
    replay = module["verify"]()
    saved = json.loads(CANARY_SUMMARY.read_text(encoding="utf-8"))
    for payload in (replay, saved):
        payload.pop("verified_at_utc")
        payload["canary_replay"].pop("live_state_relation")
    assert replay == saved


def test_canary_verifier_uses_footer_metadata_without_row_decode() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    assert "pq.ParquetFile(" in source
    for forbidden in ("pq.read_table(", ".read_row_group(", ".iter_batches("):
        assert forbidden not in source
