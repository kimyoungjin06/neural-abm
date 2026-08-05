from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_g4_private_explicit_list_preflight_v1.yaml"
AUTHOR = ROOT / "scripts/prepare_pivot_feedback_f_g4_private_explicit_list_preflight_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_f_g4_private_explicit_list_preflight_v1.py"
SEAL_CREATOR = ROOT / "scripts/create_pivot_feedback_holdout_access_seal_v1.py"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_f_g4_private_explicit_list_preflight_v1_20260805.summary.json"
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v12.md"
SOURCE_ROOT = Path("/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804")
BINDING = Path(
    "/ssd/pivot-feedback-f-g4-private-feature-source-v1-20260804."
    "root-artifact-binding-v1.json"
)
OPERATOR_ROOT = Path("/ssd/pivot-feedback-f-g4-private-operator-control-v1")
PREFLIGHT = OPERATOR_ROOT / "pivot_feedback_holdout_access_seal_preflight_v1.json"
VERIFICATION = (
    OPERATOR_ROOT
    / "pivot_feedback_f_g4_private_explicit_list_preflight_v1.verification.json"
)

SUMMARY_STATUS = (
    "PASS_OPERATOR_ONLY_AND_INDEPENDENT_BOUND_153_ARTIFACT_BYTES_REHASHED_"
    "PREFLIGHT_NO_SEAL_NO_F_PASS"
)
PREFLIGHT_STATUS = "PASS_EXPLICIT_ALLOWLIST_HASHES_BOUND_ZERO_PRIOR_OPEN_EVENTS"
COMPONENT_STATUS = (
    "PASS_OPERATOR_ONLY_BOUND_153_ARTIFACT_BYTES_REHASHED_NO_SEAL_NO_F_PASS"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_BOUND_153_ARTIFACT_BYTES_REHASHED_PREFLIGHT_NO_SEAL_NO_F_PASS"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_contract_freezes_byte_only_preflight_and_only_seal_creation_next() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    execution = contract["preflight_execution_contract"]
    before = contract["completion_accounting"]
    after = contract["authorization_after_independent_verification_pass"]

    assert contract["schema_version"] == (
        "pivot_feedback_f_g4_private_explicit_list_preflight.v1"
    )
    assert contract["status"] == (
        "FROZEN_BOUND_153_ARTIFACT_FULL_BYTE_PREFLIGHT_PENDING_EXECUTION_NO_F_PASS"
    )
    assert execution["artifact_list_source"] == (
        "BOUND_ROOT_ARTIFACT_BINDING_EXACT_153_ENTRY_PROJECTION"
    )
    assert execution["implicit_artifact_discovery_forbidden"] is True
    assert execution["every_bound_artifact_byte_must_be_hashed"] is True
    assert execution["row_or_parquet_parser_forbidden"] is True
    assert execution["artifact_count"] == 153
    assert execution["artifact_bytes_total"] == 3_067_046_872
    assert execution["independent_full_byte_verification_required"] is True
    assert before["holdout_preflight_complete"] is False
    assert before["holdout_byte_seal_complete"] is False
    assert before["pass_f"] is False
    assert before["pass_g0"] is False
    assert after["byte_seal_creation"] == (
        "AUTHORIZED_NEXT_FROM_EXACT_VERIFIED_PREFLIGHT_ONLY"
    )
    assert after["seal_opening"] == "NOT_AUTHORIZED"
    assert after["g4_open"] == "NOT_AUTHORIZED"


def test_preflight_is_exact_bound_projection_with_valid_fingerprint() -> None:
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    expected_entries = [
        {
            "canonical_relative_path": row["canonical_relative_path"],
            "expected_byte_size": row["expected_byte_size"],
            "expected_sha256": row["expected_sha256"],
        }
        for row in binding["artifact_entries"]
    ]
    fingerprint = preflight["preflight_fingerprint_sha256"]
    core = dict(preflight)
    del core["preflight_fingerprint_sha256"]

    assert preflight["schema_version"] == (
        "pivot_feedback_holdout_access_seal_preflight.v1"
    )
    assert preflight["status"] == PREFLIGHT_STATUS
    assert preflight["component_status"] == COMPONENT_STATUS
    assert preflight["artifact_entries"] == expected_entries
    assert preflight["artifact_count"] == len(expected_entries) == 153
    assert preflight["artifact_bytes_total"] == 3_067_046_872
    assert fingerprint == _canonical_sha(core)
    assert preflight["root_list_binding"]["sha256"] == _sha(BINDING)
    assert preflight["preflight_contract_binding"]["sha256"] == _sha(CONFIG)
    assert preflight["preflight_author_binding"]["sha256"] == _sha(AUTHOR)
    assert _mode(OPERATOR_ROOT) == 0o700
    assert _mode(PREFLIGHT) == 0o400

    access = preflight["access_accounting"]
    assert access["bound_artifact_paths_consumed"] == 153
    assert access["source_artifact_content_files_opened"] == 153
    assert access["source_artifact_bytes_hashed"] == 3_067_046_872
    assert access["operator_parquet_files_parsed"] == 0
    assert access["confirmatory_or_holdout_analytic_rows_read"] == 0
    assert access["byte_seal_created"] is False
    completion = preflight["completion"]
    assert completion["this_component_is_new_gate"] is False
    assert completion["holdout_preflight_complete"] is True
    assert completion["independent_preflight_verification_complete"] is False
    assert completion["holdout_byte_seal_complete"] is False
    assert completion["pass_f"] is False
    assert completion["pass_g0"] is False
    assert completion["g4_open"] is False


def test_independent_verification_records_second_full_byte_pass() -> None:
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))

    assert verification["schema_version"] == (
        "pivot_feedback_f_g4_private_explicit_list_preflight_verification.v1"
    )
    assert verification["status"] == VERIFY_STATUS
    assert verification["preflight_sha256"] == _sha(PREFLIGHT)
    assert verification["preflight_fingerprint_sha256"] == (
        preflight["preflight_fingerprint_sha256"]
    )
    assert verification["artifact_count_verified"] == 153
    assert verification["artifact_bytes_total_verified"] == 3_067_046_872
    assert verification["source_artifact_content_files_opened"] == 153
    assert verification["source_artifact_bytes_hashed"] == 3_067_046_872
    assert verification["every_bound_artifact_opened_and_rehashed"] is True
    assert verification["artifact_size_mismatch_count"] == 0
    assert verification["artifact_sha256_mismatch_count"] == 0
    assert verification["artifact_changed_during_hash_count"] == 0
    assert verification["operator_parquet_files_parsed"] == 0
    assert verification["confirmatory_or_holdout_analytic_rows_read"] == 0
    assert verification["seal_creator_preflight_schema_compatible"] is True
    assert verification["holdout_preflight_complete"] is True
    assert verification["independent_preflight_verification_complete"] is True
    assert verification["holdout_byte_seal_complete"] is False
    assert verification["pass_f"] is False
    assert verification["pass_g0"] is False
    assert verification["g4_open"] is False
    assert verification["authorization"]["byte_seal_creation"] == (
        "AUTHORIZED_NEXT_FROM_EXACT_VERIFIED_PREFLIGHT_ONLY"
    )
    assert verification["authorization"]["seal_opening"] == "NOT_AUTHORIZED"
    assert _mode(VERIFICATION) == 0o400


def test_preflight_tools_are_byte_only_and_independently_implemented() -> None:
    forbidden = {"duckdb", "numpy", "pandas", "pyarrow"}
    for source_path in (AUTHOR, VERIFIER):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported.update(
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imported.isdisjoint(forbidden)

    author_source = AUTHOR.read_text(encoding="utf-8")
    verifier_source = VERIFIER.read_text(encoding="utf-8")
    assert 'getattr(os, "O_NOFOLLOW", 0)' in author_source
    assert 'getattr(os, "O_NOFOLLOW", 0)' in verifier_source
    assert "prepare_pivot_feedback_f_g4_private_explicit_list_preflight_v1" not in verifier_source


def test_existing_seal_creator_accepts_exact_preflight_without_creating_seal() -> None:
    seal_creator = _load(SEAL_CREATOR, "g4_private_preflight_seal_compatibility_test")
    preflight, entries = seal_creator._validate_preflight(PREFLIGHT)

    assert preflight["status"] == PREFLIGHT_STATUS
    assert len(entries) == 153
    assert sum(row["expected_byte_size"] for row in entries) == 3_067_046_872
    assert not any(OPERATOR_ROOT.glob("*.seal.json"))
    assert not any(OPERATOR_ROOT.glob("*.seal-receipt.json"))


def test_summary_and_ledger_bind_evidence_and_fail_closed_successor() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")

    assert summary["status"] == SUMMARY_STATUS
    assert summary["production_preflight"]["sha256"] == _sha(PREFLIGHT)
    assert summary["independent_verification"]["sha256"] == _sha(VERIFICATION)
    assert summary["repository_bindings"]["config_sha256"] == _sha(CONFIG)
    assert summary["repository_bindings"]["preflight_author_sha256"] == _sha(AUTHOR)
    assert summary["repository_bindings"]["independent_verifier_sha256"] == _sha(VERIFIER)
    assert summary["production_execution"]["source_artifact_bytes_hashed"] == 3_067_046_872
    assert summary["independent_verification"]["source_artifact_bytes_hashed"] == 3_067_046_872
    assert summary["fail_closed_state"]["holdout_preflight_complete"] is True
    assert summary["fail_closed_state"]["holdout_byte_seal_complete"] is False
    assert summary["fail_closed_state"]["pass_f"] is False
    assert summary["fail_closed_state"]["pass_g0"] is False
    assert "does not add another empirical threshold or decision gate" in ledger
    assert "Repeating more audits of this same" in ledger
    assert "stage would add little" in ledger
    assert "create the byte seal from the exact" in ledger
    assert "Seal opening" in ledger
    assert "`pass_f=false`," in ledger
