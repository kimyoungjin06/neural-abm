from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_g4_private_root_artifact_binding_v1.yaml"
BINDER = ROOT / "scripts/bind_pivot_feedback_f_g4_private_root_artifact_list_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_f_g4_private_root_artifact_binding_v1.py"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_f_g4_private_root_artifact_binding_v1_20260805.summary.json"
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v11.md"
SOURCE_ROOT = Path("/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804")
SOURCE_RECEIPT = SOURCE_ROOT / "g4_private_feature_source_materialization.outputs.json"
SOURCE_VERIFICATION = Path(
    "/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804.verification.json"
)
BINDING = Path(
    "/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804."
    "root-artifact-binding-v1.json"
)
BINDING_VERIFICATION = Path(
    "/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804."
    "root-artifact-binding-v1.verification.json"
)

BINDING_STATUS = (
    "PASS_OPERATOR_ONLY_EXACT_PRIVATE_G4_ROOT_AND_COMPLETE_153_ARTIFACT_LIST_"
    "BOUND_NO_PREFLIGHT_NO_F_PASS"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_PRIVATE_G4_ROOT_AND_COMPLETE_153_ARTIFACT_LIST_"
    "BOUND_NO_PREFLIGHT_NO_F_PASS"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BIND_MODULE = _load(BINDER, "g4_private_root_artifact_binder_test_module")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _expected_entries(receipt: dict) -> list[dict]:
    return [
        {
            "canonical_relative_path": row["relative_path"],
            "expected_byte_size": row["bytes"],
            "expected_sha256": row["sha256"],
            "expected_rows": row.get("rows"),
        }
        for row in receipt["artifacts"]
    ]


def test_contract_authorizes_only_binding_then_explicit_list_preflight() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    root = contract["bound_private_source_package"]["exact_source_root"]
    receipt = contract["bound_private_source_package"]["production_receipt"]
    binding = contract["explicit_binding_contract"]
    completion = contract["completion_accounting"]
    authorization = contract["authorization_after_pass"]

    assert contract["schema_version"] == (
        "pivot_feedback_f_g4_private_root_artifact_binding.v1"
    )
    assert contract["status"] == (
        "FROZEN_EXACT_PRIVATE_G4_ROOT_AND_153_ARTIFACT_LIST_"
        "PENDING_BINDING_NO_F_PASS"
    )
    assert root["absolute_path"] == SOURCE_ROOT.as_posix()
    assert root["resolved_path_sha256"] == hashlib.sha256(
        SOURCE_ROOT.as_posix().encode("utf-8")
    ).hexdigest()
    assert root["expected_directory_count_including_root"] == 6
    assert root["expected_regular_file_count_including_receipt"] == 154
    assert root["expected_symlink_count"] == 0
    assert receipt["artifact_count"] == 153
    assert receipt["artifact_bytes_total"] == 3_067_046_872
    assert binding["list_source"] == (
        "PRODUCTION_RECEIPT_ARTIFACT_ARRAY_EXACT_PROJECTION"
    )
    assert binding["source_artifact_byte_hashing_in_this_component"] is False
    assert binding["parquet_or_row_parser_in_this_component"] is False
    assert binding["preflight_written_in_this_component"] is False
    assert binding["byte_seal_written_in_this_component"] is False
    assert completion["this_component_is_new_gate"] is False
    assert completion["exact_private_source_root_bound"] is False
    assert completion["complete_explicit_artifact_list_bound"] is False
    assert completion["holdout_preflight_complete"] is False
    assert completion["holdout_byte_seal_complete"] is False
    assert completion["pass_f"] is False
    assert completion["pass_g0"] is False
    assert authorization["explicit_file_list_preflight"] == (
        "AUTHORIZED_NEXT_FROM_BOUND_ROOT_AND_LIST_ONLY"
    )
    assert authorization["preflight_must_rehash_every_bound_artifact_byte"] is True
    assert authorization["byte_seal_creation"] == (
        "NOT_AUTHORIZED_WITHOUT_SUCCESSFUL_PREFLIGHT"
    )
    assert authorization["g4_open"] == "NOT_AUTHORIZED"


def test_binding_is_exact_sorted_receipt_projection_and_root_identity() -> None:
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    entries = _expected_entries(receipt)
    root_metadata = SOURCE_ROOT.stat()

    assert binding["schema_version"] == (
        "pivot_feedback_f_g4_private_root_artifact_binding_output.v1"
    )
    assert binding["status"] == BINDING_STATUS
    assert binding["artifact_entries"] == entries
    assert entries == sorted(entries, key=lambda row: row["canonical_relative_path"])
    assert len({row["canonical_relative_path"] for row in entries}) == 153
    assert binding["artifact_count"] == len(entries) == 153
    assert binding["artifact_bytes_total"] == sum(
        row["expected_byte_size"] for row in entries
    ) == 3_067_046_872
    assert binding["source_root_binding"] == {
        "basename": SOURCE_ROOT.name,
        "device": root_metadata.st_dev,
        "directory_count_including_root": 6,
        "inode": root_metadata.st_ino,
        "mode": "0700",
        "regular_file_count_including_receipt": 154,
        "resolved_absolute_path": SOURCE_ROOT.as_posix(),
        "resolved_path_sha256": hashlib.sha256(
            SOURCE_ROOT.as_posix().encode("utf-8")
        ).hexdigest(),
        "symlink_count": 0,
    }
    assert binding["binding_assertions"] == {
        "actual_tree_closes_to_explicit_list_plus_receipt": True,
        "complete_explicit_artifact_list_bound": True,
        "exact_private_source_root_bound": True,
        "expected_sha_values_copied_without_refresh_from_verified_receipt": True,
        "explicit_list_exact_receipt_projection": True,
        "explicit_list_sorted_and_unique": True,
        "listed_paths_regular_and_size_matched": True,
        "parquet_or_row_parser_used": False,
        "source_artifact_content_hashed_in_this_component": False,
        "source_tree_symlinks_absent": True,
    }


def test_actual_tree_closes_without_opening_source_artifact_content() -> None:
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    expected_paths = {
        row["canonical_relative_path"] for row in binding["artifact_entries"]
    }
    actual_files = {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in SOURCE_ROOT.rglob("*")
        if path.is_file()
    }
    actual_directories = [SOURCE_ROOT, *[path for path in SOURCE_ROOT.rglob("*") if path.is_dir()]]

    assert actual_files == expected_paths | {SOURCE_RECEIPT.name}
    assert len(actual_files) == 154
    assert len(actual_directories) == 6
    assert not any(path.is_symlink() for path in SOURCE_ROOT.rglob("*"))
    assert all(_mode(path) == 0o700 for path in actual_directories)
    assert all(_mode(SOURCE_ROOT / relative) == 0o600 for relative in actual_files)
    for row in receipt["artifacts"]:
        assert (SOURCE_ROOT / row["relative_path"]).stat().st_size == row["bytes"]
    assert _mode(BINDING) == 0o400
    assert _mode(BINDING_VERIFICATION) == 0o400
    assert binding["access_accounting"]["source_artifact_content_files_opened"] == 0
    assert binding["access_accounting"]["source_artifact_bytes_hashed"] == 0
    assert binding["access_accounting"]["operator_parquet_files_opened"] == 0


def test_independent_verification_binds_metadata_only_fail_closed_result() -> None:
    verification = json.loads(BINDING_VERIFICATION.read_text(encoding="utf-8"))

    assert verification == {
        "actual_tree_closure_verified": True,
        "artifact_bytes_total_verified": 3_067_046_872,
        "artifact_count_verified": 153,
        "artifact_tree_sha256_verified": (
            "62d6978c776c32464c4c862df12a52a580a6066dda45d4fb0dfe75910a562084"
        ),
        "authorization": {
            "byte_seal_creation": "NOT_AUTHORIZED_WITHOUT_SUCCESSFUL_PREFLIGHT",
            "explicit_file_list_preflight": "AUTHORIZED_NEXT_FROM_BOUND_ROOT_AND_LIST_ONLY",
            "future_earlier_partition_anti_join_application": "REQUIRED_WHEN_ROWS_ARE_MATERIALIZED",
            "g4_open": "NOT_AUTHORIZED",
            "label_derivation_or_analytic_read": "NOT_AUTHORIZED_BEFORE_F_PASS_SUCCESSOR",
            "model_fitting": "NOT_AUTHORIZED",
            "model_selection_score_read": "NOT_AUTHORIZED_BEFORE_E_PASS_AND_PRE_G2_LOCK",
            "preflight_must_rehash_every_bound_artifact_byte": True,
        },
        "binding_basename": BINDING.name,
        "binding_sha256": _sha(BINDING),
        "complete_explicit_artifact_list_bound": True,
        "complete_private_g4_artifact_set": True,
        "confirmatory_or_holdout_analytic_rows_read": 0,
        "directory_count_including_root_verified": 6,
        "earlier_partition_guard_application_complete": False,
        "entry_or_persistence_labels_materialized": False,
        "exact_private_source_root_bound": True,
        "expected_sha_values_inherited_without_refresh": True,
        "explicit_list_exact_receipt_projection_verified": True,
        "explicit_list_sorted_and_unique_verified": True,
        "g4_open": False,
        "holdout_byte_seal_complete": False,
        "holdout_preflight_complete": False,
        "listed_paths_regular_and_size_matched_verified": True,
        "model_fitted": False,
        "model_score_read": False,
        "operator_parquet_files_opened": 0,
        "pass_f": False,
        "pass_g0": False,
        "regular_file_count_including_receipt_verified": 154,
        "schema_version": (
            "pivot_feedback_f_g4_private_root_artifact_binding_verification.v1"
        ),
        "source_artifact_bytes_hashed": 0,
        "source_artifact_content_files_opened": 0,
        "source_root_basename": SOURCE_ROOT.name,
        "source_root_device": SOURCE_ROOT.stat().st_dev,
        "source_root_identity_verified": True,
        "source_root_inode": SOURCE_ROOT.stat().st_ino,
        "source_root_resolved_path_sha256": hashlib.sha256(
            SOURCE_ROOT.as_posix().encode("utf-8")
        ).hexdigest(),
        "source_tree_modes_verified": True,
        "source_tree_symlinks_absent": True,
        "status": VERIFY_STATUS,
    }


def test_binder_is_create_only_and_tools_import_no_row_data_libraries() -> None:
    for source_path in (BINDER, VERIFIER):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})

    before = BINDING.read_bytes()
    with pytest.raises(
        BIND_MODULE.G4PrivateRootArtifactBindingError,
        match="binding output exists",
    ):
        BIND_MODULE.bind(
            config_path=CONFIG,
            source_root=SOURCE_ROOT,
            source_verification=SOURCE_VERIFICATION,
            output_path=BINDING,
            bound_at_utc="2026-08-05T01:40:30.447037Z",
        )
    assert BINDING.read_bytes() == before
    binder_source = BINDER.read_text(encoding="utf-8")
    verifier_source = VERIFIER.read_text(encoding="utf-8")
    assert "source_artifact_bytes_hashed\": 0" in binder_source
    assert "source_artifact_bytes_hashed\": 0" in verifier_source
    assert "parquet_or_row_parser_used\": False" in binder_source
    assert "bind_pivot_feedback_f_g4_private_root_artifact_list_v1" not in verifier_source


def test_summary_and_ledger_bind_hashes_next_step_and_fail_closed_state() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")

    assert summary["status"] == BINDING_STATUS
    assert summary["binding_artifact"]["sha256"] == _sha(BINDING)
    assert summary["binding_artifact"]["mode"] == "0400"
    assert summary["independent_verification"]["sha256"] == _sha(
        BINDING_VERIFICATION
    )
    assert summary["repository_bindings"]["config_sha256"] == _sha(CONFIG)
    assert summary["repository_bindings"]["builder_sha256"] == _sha(BINDER)
    assert summary["repository_bindings"]["verifier_sha256"] == _sha(VERIFIER)
    assert summary["explicit_artifact_list"]["artifact_count"] == 153
    assert summary["access_accounting"]["source_artifact_bytes_hashed"] == 0
    assert summary["authorization"]["explicit_file_list_preflight"] == (
        "AUTHORIZED_NEXT_FROM_BOUND_ROOT_AND_LIST_ONLY"
    )
    assert summary["fail_closed_state"]["holdout_preflight_complete"] is False
    assert summary["fail_closed_state"]["holdout_byte_seal_complete"] is False
    assert summary["fail_closed_state"]["pass_f"] is False
    assert summary["fail_closed_state"]["pass_g0"] is False
    assert "exact sorted projection of all 153" in ledger
    assert "did not open or hash source-artifact" in ledger
    assert "preflight must independently rehash" in ledger
    assert "Byte-seal creation remains unauthorized" in ledger
    assert "`pass_f=false`," in ledger
