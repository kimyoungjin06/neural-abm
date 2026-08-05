from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CURRENT_AUTHORITY = ROOT / "experiments/configs/pivot_feedback_current_state_v1.yaml"
PREFLIGHT = Path(
    "/ssd/pivot-feedback-f-g4-private-operator-control-v1/"
    "pivot_feedback_holdout_access_seal_preflight_v1.json"
)
PREFLIGHT_VERIFICATION = Path(
    "/ssd/pivot-feedback-f-g4-private-operator-control-v1/"
    "pivot_feedback_f_g4_private_explicit_list_preflight_v1.verification.json"
)
OPERATOR_ROOT = Path("/ssd/pivot-feedback-f-g4-private-operator-control-v1")
SEAL_ROOT = OPERATOR_ROOT / "pivot-feedback/holdout-seal-v1"
MANIFEST = SEAL_ROOT / "manifest.json"
RECEIPT = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_holdout_access_seal_receipt_v1.summary.json"
)
SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_holdout_access_seal_receipt_verification_v1_20260805.summary.json"
)
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v13.md"
CREATOR_PATH = ROOT / "scripts/create_pivot_feedback_holdout_access_seal_v1.py"
VERIFIER_PATH = (
    ROOT / "scripts/verify_pivot_feedback_holdout_access_seal_receipt_v1.py"
)
CURRENT_STATE_SUMMARY = (
    ROOT / "experiments/evidence/results/pivot_feedback_current_state_v1_20260803.summary.json"
)
SECTIONS_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_pre_g1_scientific_sections_v1_20260803.summary.json"
)
PANEL_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_aligned_final_panel_conformance_v1_20260804.summary.json"
)
PARTITION_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_atomic_partition_assignment_v1_20260804.summary.json"
)

SUMMARY_STATUS = (
    "PASS_PRODUCTION_PRIVATE_G4_BYTE_SEAL_RECEIPT_INDEPENDENTLY_VERIFIED_"
    "UNOPENED_NO_F_PASS"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CREATOR = _load(CREATOR_PATH, "production_holdout_seal_creator_test_module")
VERIFIER = _load(VERIFIER_PATH, "production_holdout_seal_verifier_test_module")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_production_receipt_binds_exact_verified_preflight_and_unopened_seal() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    preflight_verification = json.loads(
        PREFLIGHT_VERIFICATION.read_text(encoding="utf-8")
    )

    assert receipt["schema_version"] == "pivot_feedback_holdout_access_seal_outputs.v1"
    assert receipt["status"] == "PASS_CONFIRMATORY_PARTITIONS_SEALED_AND_UNOPENED"
    assert receipt["preflight_sha256"] == _sha(PREFLIGHT)
    assert receipt["preflight_sha256"] == preflight_verification["preflight_sha256"]
    assert receipt["artifact_count"] == 153
    assert receipt["artifact_bytes_total"] == 3_067_046_872
    assert receipt["receipt_written_last"] is True
    assert receipt["seal_opened"] is False
    assert receipt["prior_open_event_count"] == 0
    assert receipt["confirmatory_or_holdout_rows_read"] == 0
    assert receipt["independent_verification_required"] is True
    assert receipt["independent_verification_complete"] is False
    assert receipt["g0_measurement_freeze_authorized"] is False
    assert receipt["model_fitting_authorized"] is False
    assert receipt["g4_open"] is False
    assert receipt["scientific_effect"] == "NOT_ESTABLISHED"
    assert receipt["creator_sha256"] == _sha(CREATOR_PATH)
    assert _mode(RECEIPT) == 0o444


def test_manifest_is_exact_preflight_projection_and_private_tree_is_closed() -> None:
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected_entries = [
        {
            "canonical_relative_path": row["canonical_relative_path"],
            "sealed_relative_path": f'artifacts/{row["canonical_relative_path"]}',
            "byte_size": row["expected_byte_size"],
            "sha256": row["expected_sha256"],
            "regular_file": True,
            "symlink": False,
        }
        for row in preflight["artifact_entries"]
    ]
    core = dict(manifest)
    fingerprint = core.pop("manifest_fingerprint_sha256")

    assert manifest["schema_version"] == (
        "pivot_feedback_holdout_access_seal_manifest.v1"
    )
    assert manifest["status"] == "SEALED_ARTIFACT_MANIFEST_FINALIZED_UNOPENED"
    assert manifest["preflight_sha256"] == _sha(PREFLIGHT)
    assert manifest["artifact_entries"] == expected_entries
    assert manifest["artifact_count"] == 153
    assert manifest["artifact_bytes_total"] == 3_067_046_872
    assert fingerprint == _canonical_sha(core)
    assert receipt["manifest_sha256"] == _sha(MANIFEST)
    assert receipt["manifest_fingerprint_sha256"] == fingerprint
    assert manifest["seal_opened"] is False
    assert manifest["prior_open_event_count"] == 0
    assert manifest["confirmatory_or_holdout_rows_read"] == 0

    entries = list(SEAL_ROOT.rglob("*"))
    directories = [SEAL_ROOT, *[path for path in entries if path.is_dir()]]
    regular_files = [path for path in entries if path.is_file()]
    assert len(directories) == 7
    assert len(regular_files) == 154
    assert not any(path.is_symlink() for path in entries)
    assert all(_mode(path) == 0o700 for path in directories)
    assert all(_mode(path) == 0o400 for path in regular_files)


def test_read_only_production_verifier_rehashes_entire_unopened_seal() -> None:
    result = VERIFIER.verify(receipt_path=RECEIPT, operator_root=OPERATOR_ROOT)

    assert result["schema_version"] == "pivot_feedback_holdout_access_seal_verification.v1"
    assert result["status"] == (
        "PASS_READ_ONLY_INDEPENDENT_HOLDOUT_SEAL_VERIFICATION_UNOPENED"
    )
    assert result["receipt_sha256"] == _sha(RECEIPT)
    assert result["manifest_sha256"] == _sha(MANIFEST)
    assert result["artifact_replay"] == {
        "artifact_count": 153,
        "artifact_bytes_total": 3_067_046_872,
        "all_artifacts_rehashed": True,
        "sealed_tree_exact": True,
        "manifest_fingerprint_valid": True,
    }
    access = result["access_accounting"]
    assert access["operator_artifact_files_rehashed"] == 153
    assert access["operator_artifact_bytes_rehashed"] == 3_067_046_872
    assert access["operator_parquet_files_parsed"] == 0
    assert access["production_row_level_files_opened"] == 0
    assert access["confirmatory_or_holdout_rows_read"] == 0
    assert result["authorization"]["holdout_seal_receipt"] == (
        "COMPLETE_INDEPENDENTLY_VERIFIED"
    )
    assert result["authorization"]["seal_opening"] == "NOT_AUTHORIZED"
    assert result["authorization"]["g4_open"] == "NOT_AUTHORIZED"


def test_production_creator_is_create_only_after_finalization() -> None:
    receipt_before = RECEIPT.read_bytes()
    manifest_before = MANIFEST.read_bytes()

    with pytest.raises(
        CREATOR.HoldoutSealCreationError,
        match="final seal path already exists",
    ):
        CREATOR.create_seal(
            preflight_path=PREFLIGHT,
            source_root=Path(
                "/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804"
            ),
            operator_root=OPERATOR_ROOT,
            receipt_path=RECEIPT,
            seal_created_at_utc="2026-08-05T02:39:45.443250Z",
            receipt_written_at_utc="2026-08-05T02:39:45.443259Z",
        )

    assert RECEIPT.read_bytes() == receipt_before
    assert MANIFEST.read_bytes() == manifest_before
    assert not list((OPERATOR_ROOT / "pivot-feedback").glob(".holdout-seal-v1.staging-*"))


def test_summary_and_ledger_preserve_atomic_f_closure_boundary() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")

    assert summary["status"] == SUMMARY_STATUS
    assert summary["active_authority"]["sha256"] == _sha(CURRENT_AUTHORITY)
    assert summary["production_receipt"]["sha256"] == _sha(RECEIPT)
    assert summary["manifest"]["sha256"] == _sha(MANIFEST)
    assert summary["repository_bindings"]["creator_sha256"] == _sha(CREATOR_PATH)
    assert summary["repository_bindings"]["independent_verifier_sha256"] == _sha(
        VERIFIER_PATH
    )
    bindings = summary["f_phase_transition_receipt_bindings"]
    assert bindings["pre_g1_decision_registry"]["sha256"] == _sha(
        CURRENT_STATE_SUMMARY
    )
    assert bindings["thirteen_pre_g1_required_sections"]["sha256"] == _sha(
        SECTIONS_SUMMARY
    )
    assert bindings["outcome_blind_final_panel_conformance"]["sha256"] == _sha(
        PANEL_SUMMARY
    )
    atomic = bindings["atomic_partition_and_holdout_byte_seal"]
    assert atomic["atomic_partition_sha256"] == _sha(PARTITION_SUMMARY)
    assert atomic["byte_seal_receipt_sha256"] == _sha(RECEIPT)
    assert atomic["byte_seal_independently_verified"] is True
    transition = summary["f_phase_transition_state"]
    assert transition["all_four_transition_receipts_available_for_atomic_reconciliation"] is True
    assert transition["f_closure_reconciliation_complete"] is False
    assert transition["pass_f"] is False
    assert transition["pass_g0"] is False
    assert summary["fail_closed_state"]["holdout_byte_seal_complete"] is True
    assert summary["fail_closed_state"]["holdout_seal_independently_verified"] is True
    assert summary["fail_closed_state"]["seal_opened"] is False
    assert summary["authorization"]["f_phase_atomic_closure_reconciliation"] == (
        "AUTHORIZED_NEXT_FROM_ALL_F_RECEIPTS_ONLY"
    )
    assert "not another micro-gate" in ledger
    assert "single aggregate F-closure successor" in ledger
    assert "does not silently convert" in ledger
    assert "`pass_f=false`," in ledger
