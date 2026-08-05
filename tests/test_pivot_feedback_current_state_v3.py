from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "experiments/configs/pivot_feedback_current_state_v3.yaml"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.json"
BUILDER_PATH = ROOT / "scripts/build_pivot_feedback_current_state_v3.py"
VERIFIER_PATH = ROOT / "scripts/verify_pivot_feedback_current_state_v3.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_PATH, "pivot_feedback_current_state_v3_verifier_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_state_v3_records_terminal_block_without_route_promotion() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    phases = {row["phase_id"]: row for row in authority["active_routing"]["phases"]}

    assert authority["status"] == (
        "ACTIVE_E_PHASE_BLOCKED_INPUT_AUTHORIZATION_CONFLICT_G1_NOT_RUN_HOLDOUT_SEALED_UNOPENED"
    )
    assert authority["active_routing"]["active_phase"] == "E"
    assert phases["E"]["state"] == "BLOCKED_INPUT_AUTHORIZATION_CONFLICT_NO_CONFIRMATORY_READ"
    assert phases["E"]["complete"] is False
    assert phases["E"]["may_read_confirmatory_or_holdout_rows"] is False
    assert phases["M"]["state"] == "NOT_RUN"
    assert phases["H"]["state"] == "BYTE_SEALED_UNOPENED"
    assert authority["lifecycle"]["pass_f"] is True
    assert authority["lifecycle"]["pass_g0"] is True
    assert authority["lifecycle"]["empirical_g1_status"] == "BLOCKED"
    assert authority["lifecycle"]["confirmatory_or_holdout_rows_read"] == 0
    assert authority["lifecycle"]["g4_open"] is False


def test_current_state_v3_independent_summary_binds_tools_and_authority() -> None:
    replay = VERIFIER.verify(AUTHORITY)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert replay["status"] == summary["status"] == (
        "PASS_ACTIVE_E_TERMINAL_BLOCKED_INPUT_AUTHORIZATION_CONFLICT_NO_CONFIRMATORY_READ"
    )
    assert summary["authority"]["sha256"] == _sha(AUTHORITY)
    assert summary["bound_tooling"]["builder_sha256"] == _sha(BUILDER_PATH)
    assert summary["bound_tooling"]["independent_verifier_sha256"] == _sha(VERIFIER_PATH)
    assert summary["execution_result_replay"]["confirmatory_rows_read"] == 0
    assert summary["execution_result_replay"]["private_g4_artifacts_read"] == 0
    assert summary["routing_replay"]["pass_e"] is False
    assert summary["routing_replay"]["pass_g1"] is False


def test_current_state_v3_requires_new_explicit_conflict_resolution_authority() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    attempt = authority["e_phase_execution_attempt"]
    authorization = authority["authorization"]

    assert attempt["resolution_requires_new_explicit_authority"] is True
    assert attempt["scope_or_threshold_changed"] is False
    assert attempt["holdout_seal_opened"] is False
    assert authorization["e_g1_execution"] == (
        "BLOCKED_REQUIRES_NEW_EXPLICIT_CONFLICT_RESOLUTION_AUTHORITY"
    )
    assert authorization["confirmatory_analytic_access"] == (
        "NOT_AUTHORIZED_UNTIL_NEW_EXPLICIT_CONFLICT_RESOLUTION_AUTHORITY"
    )
    assert authorization["g2_or_g3_choice_resolution"] == "NOT_AUTHORIZED_E_NOT_PASSED"
