from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.yaml"
BUILDER = SCRIPTS / "build_pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.py"
VERIFIER = SCRIPTS / "verify_pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.py"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1_20260804.summary.json"
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v6.md"
PRODUCTION = Path("/ssd/pivot-feedback-f-aligned-strict-incumbent-rescreen-v1-20260804")
VERIFICATION = Path("/ssd/pivot-feedback-f-aligned-strict-incumbent-rescreen-v1-20260804.verification.json")
CUTOFF = Path("/ssd/pivot-feedback-cutoff-primitives-v0-full-20260723")
AUTHORS = Path("/ssd/openalex-g1-author-history-v3-full-20260722")
TAXONOMY = Path("/home/kimyoungjin06/Desktop/SSD/science-atlas-runtime-20260719/science_atlas_openalex_work_taxonomy_v0_20260526/tables/work_openalex_taxonomy_primary.parquet")


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _load(BUILDER, "f_aligned_strict_rescreen_builder_test")
VERIFY = _load(VERIFIER, "f_aligned_strict_rescreen_verifier_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_keeps_estimand_threshold_family_gate_and_frozen_order() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert contract["schema_version"] == BUILD.CONFIG_SCHEMA
    assert contract["primary_rescreen"]["primary_work_inclusion"] == "EVERY_LINKED_AUTHOR_IS_PRE_SIGNAL_INCUMBENT"
    assert contract["primary_rescreen"]["candidate_universe"].startswith("ALL_ELIGIBLE")
    assert contract["primary_rescreen"]["aggregate_count_or_normalization_reuse"] == "FORBIDDEN"
    assert contract["primary_rescreen"]["modified_z_threshold_inclusive"] == 3.0
    assert contract["primary_rescreen"]["minimum_distinct_current_snapshot_subfield_families"] == 3
    assert contract["primary_rescreen"]["threshold_relaxation"] is False
    assert [row["field_order"] for row in contract["frozen_field_attempts"]["expected"]] == [
        [25, 17, 16], [32, 18, 20], [28, 11, 24], [36, 29, 27]
    ]
    assert contract["completion_accounting"]["pass_f"] is False
    assert contract["completion_accounting"]["pass_g0"] is False


def test_production_recomputes_aligned_surface_and_selects_first_conforming_fields() -> None:
    report = json.loads((PRODUCTION / "aligned_strict_incumbent_rescreen.json").read_text(encoding="utf-8"))

    assert report["status"].startswith("PASS_OUTCOME_BLIND_STRICT_INCUMBENT_ALIGNED_ALL_TOPIC")
    accounting = report["replay_accounting"]
    assert accounting["strict_surface_rows"] == 7099
    assert accounting["normalization_group_count"] == 45
    assert accounting["mad_zero_or_nonfinite_group_count"] == 0
    assert accounting["candidate_rows"] == 99
    assert accounting["candidate_topics"] == 70
    assert accounting["predecessor_exact_count_rows"] == 194
    assert accounting["selected_candidate_episode_count"] == 65
    assert [row["aligned_selected_field_id"] for row in report["slot_assignment"]] == [16, 32, 11, 27]
    assert [row["aligned_subfield_family_count"] for row in report["slot_assignment"]] == [3, 3, 7, 16]
    assert report["authorization"]["aligned_strict_incumbent_final_panel_leaveout_replay"] == "AUTHORIZED_NEXT"
    assert report["scope"]["pass_f"] is False
    assert report["scope"]["pass_g0"] is False


def test_exact_production_passes_independent_surface_and_taxonomy_verification(tmp_path: Path) -> None:
    result = VERIFY.verify(
        config_path=CONFIG,
        output_root=PRODUCTION,
        cutoff_root=CUTOFF,
        author_history_root=AUTHORS,
        taxonomy_path=TAXONOMY,
        verification_output=tmp_path / "verification.json",
    )

    assert result["status"] == VERIFY.STATUS
    assert result["surface_verification"] == {
        "normalization_group_count": 45,
        "bad_mad_count": 0,
        "surface_rows": 7099,
        "candidate_rows": 99,
    }
    assert result["taxonomy_verification"]["taxonomy_topics"] == 4516
    assert result["taxonomy_verification"]["surface_topics"] == 1989
    assert result["predecessor_closure_verification"]["exact_rows"] == 194
    assert result["selection_verification"]["selected_fields"] == {
        "g1_domain_slot_01": 16,
        "g1_domain_slot_02": 32,
        "g1_domain_slot_03": 11,
        "g4_domain_author_cold_slot": 27,
    }
    assert result["pass_f"] is False
    assert result["pass_g0"] is False


def test_verifier_is_independent_of_builder() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert "build_pivot_feedback_f_aligned_strict_incumbent_rescreen_v1" not in source
    assert "statistics.median" in source
    assert "surface taxonomy mismatch" in source
    assert "predecessor strict-count mismatch" in source


def test_verifier_rejects_receipt_that_improperly_passes_f(tmp_path: Path) -> None:
    output = tmp_path / PRODUCTION.name
    shutil.copytree(PRODUCTION, output)
    receipt_path = output / "aligned_strict_incumbent_rescreen.outputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["pass_f"] = True
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(VERIFY.RescreenVerificationError, match="unsafe receipt state: pass_f"):
        VERIFY.verify(
            config_path=CONFIG,
            output_root=output,
            cutoff_root=CUTOFF,
            author_history_root=AUTHORS,
            taxonomy_path=TAXONOMY,
            verification_output=tmp_path / "should-not-exist.json",
        )


def test_repository_summary_and_human_ledger_bind_fail_closed_result() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")

    assert summary["production_artifact"]["output_receipt_sha256"] == _sha(PRODUCTION / "aligned_strict_incumbent_rescreen.outputs.json")
    assert summary["repository_bindings"]["config_sha256"] == _sha(CONFIG)
    assert summary["repository_bindings"]["builder_sha256"] == _sha(BUILDER)
    assert summary["repository_bindings"]["verifier_sha256"] == _sha(VERIFIER)
    assert summary["independent_verification"]["verification_sha256"] == _sha(VERIFICATION)
    assert verification["status"] == VERIFY.STATUS
    assert summary["fail_closed_state"]["pass_f"] is False
    assert "65" in ledger and "fields 16, 32, 11, and 27" in ledger
    assert "`pass_f=false` and `pass_g0=false`" in ledger
