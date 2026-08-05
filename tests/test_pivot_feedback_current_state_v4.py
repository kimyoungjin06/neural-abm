from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "experiments/configs/pivot_feedback_current_state_v4.yaml"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_current_state_v4_20260805.summary.json"
BUILDER_PATH = ROOT / "scripts/build_pivot_feedback_current_state_v4.py"
VERIFIER_PATH = ROOT / "scripts/verify_pivot_feedback_current_state_v4.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_PATH, "pivot_feedback_current_state_v4_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_state_v4_records_support_terminal_after_confirmatory_materialization() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    phases = {row["phase_id"]: row for row in authority["active_routing"]["phases"]}

    assert authority["active_routing"]["active_phase"] == "E"
    assert phases["E"]["state"] == (
        "TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_AFTER_GUARD_FIRST_LABEL_MATERIALIZATION"
    )
    assert phases["E"]["complete"] is False
    assert phases["E"]["may_read_confirmatory_or_holdout_rows"] is False
    assert phases["M"]["state"] == "NOT_RUN"
    assert phases["H"]["state"] == "BYTE_SEALED_UNOPENED"
    lifecycle = authority["lifecycle"]
    assert lifecycle["pass_f"] is True
    assert lifecycle["pass_g0"] is True
    assert lifecycle["empirical_g1_status"] == "BLOCKED"
    assert lifecycle["confirmatory_or_holdout_rows_read"] == 72_715_221
    assert lifecycle["g1_entry_rows_materialized"] == 17_411
    assert lifecycle["g1_support_failed_unit_count"] == 2
    assert lifecycle["g1_estimators_run"] == 0
    assert lifecycle["g4_open"] is False


def test_current_state_v4_separates_resolved_guard_block_from_new_support_block() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    transition = authority["guard_only_exception_transition"]
    attempt = authority["e_phase_execution_attempt"]

    assert transition["resolved_parent_block_id"] == (
        "G1_AUTHOR_COLD_GUARD_REQUIRED_BUT_PRIVATE_G4_ACCESS_FORBIDDEN"
    )
    assert transition["resolution_scope"] == (
        "EXACT_AUTHOR_COLD_GUARD_PROJECTION_AND_PRE_LABEL_ANTI_JOIN_ONLY"
    )
    assert attempt["attempt_count"] == 2
    assert attempt["block_id"] == (
        "G1_FROZEN_EPISODE_COHORT_SUPPORT_FLOOR_FAILED_AFTER_GUARD_FIRST_MATERIALIZATION"
    )
    assert attempt["support"]["failed_unit_count"] == 2
    assert [row["episode_id"] for row in attempt["support"]["failed_units"]] == [11, 25]
    assert attempt["access_accounting"]["estimators_run"] == 0
    assert attempt["access_accounting"]["holdout_outcome_rows_read"] == 0


def test_current_state_v4_forbids_same_cohort_retry_and_requires_disjoint_successor() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    authorization = authority["authorization"]

    assert authorization["e_g1_execution"] == (
        "TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_NO_ROUTE_PROMOTION"
    )
    assert authorization["scope_or_threshold_relaxation_on_read_cohort"] == (
        "NOT_AUTHORIZED"
    )
    assert authorization["same_cohort_g1_retry"] == "NOT_AUTHORIZED"
    assert authorization["disjoint_g1_successor"] == (
        "REQUIRES_NEW_OUTCOME_BLIND_PREAUTHORIZATION_AND_FRESH_FREEZE"
    )
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"


def test_current_state_v4_independent_summary_replays_exact_transition() -> None:
    replay = VERIFIER.verify(AUTHORITY)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert replay == summary
    assert summary["status"] == (
        "PASS_ACTIVE_E_TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_AFTER_GUARD_FIRST_"
        "CONFIRMATORY_MATERIALIZATION_NO_ESTIMATION"
    )
    assert summary["authority"]["sha256"] == _sha(AUTHORITY)
    assert summary["bound_tooling"]["builder_sha256"] == _sha(BUILDER_PATH)
    assert summary["bound_tooling"]["independent_verifier_sha256"] == _sha(VERIFIER_PATH)
    assert summary["execution_result_replay"]["guarded_risk_rows"] == 72_715_221
    assert summary["execution_result_replay"]["support_failed_episode_ids"] == [11, 25]
    assert summary["execution_result_replay"]["estimators_run"] == 0
    assert summary["routing_replay"]["pass_e"] is False
    assert summary["routing_replay"]["pass_g1"] is False
