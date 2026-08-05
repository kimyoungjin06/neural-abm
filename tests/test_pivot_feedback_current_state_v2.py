from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Callable

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/configs/pivot_feedback_current_state_v1.yaml"
AUTHORITY = ROOT / "experiments/configs/pivot_feedback_current_state_v2.yaml"
BUILDER_PATH = ROOT / "scripts/build_pivot_feedback_current_state_v2.py"
VERIFIER_PATH = ROOT / "scripts/verify_pivot_feedback_current_state_v2.py"
SUMMARY = (
    ROOT
    / "experiments/evidence/results/pivot_feedback_current_state_v2_20260805.summary.json"
)
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v14.md"

AUTHORITY_SHA = "ca431afb63da4cf2e09d0563c0509e3685efac29dc8679df0f9cc7f1ae7851ab"
SUMMARY_STATUS = (
    "PASS_ATOMIC_F_CLOSURE_G0_COMPATIBILITY_PASS_E_ENTRY_READY_FAIL_CLOSED"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load(BUILDER_PATH, "current_state_v2_builder_test_module")
VERIFIER = _load(VERIFIER_PATH, "current_state_v2_verifier_test_module")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority() -> dict:
    return yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))


def _verify_mutation(
    tmp_path: Path,
    mutate: Callable[[dict], None],
) -> str:
    document = _authority()
    mutate(document)
    path = tmp_path / "mutated_current_state_v2.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(VERIFIER.CurrentStateV2VerificationError) as error:
        VERIFIER.verify(path)
    return str(error.value)


def test_v2_is_additive_exact_four_receipt_f_closure() -> None:
    authority = _authority()

    assert authority["schema_version"] == "pivot_feedback_current_state.v2"
    assert authority["status"] == (
        "ACTIVE_E_PHASE_F_ATOMICALLY_CLOSED_G0_COMPATIBILITY_PASS_"
        "HOLDOUT_SEALED_UNOPENED"
    )
    assert _sha(AUTHORITY) == AUTHORITY_SHA
    assert authority["additive_parent"] == {
        "relative_path": "experiments/configs/pivot_feedback_current_state_v1.yaml",
        "sha256": _sha(PARENT),
        "required_schema_version": "pivot_feedback_current_state.v1",
        "required_status": (
            "ACTIVE_F_PHASE_PRE_G1_REGISTRY_BOUND_THIRTEEN_SECTIONS_"
            "PANEL_AND_SEAL_PENDING"
        ),
        "mutation_allowed": False,
    }
    bindings = authority["f_phase_transition_receipt_bindings"]
    assert set(bindings) == set(BUILDER.RECEIPTS)
    for name, expected in BUILDER.RECEIPTS.items():
        assert bindings[name] == {**expected, "mutation_allowed": False}
        assert _sha(ROOT / expected["relative_path"]) == expected["sha256"]


def test_v2_preserves_unrelated_authority_subtrees_exactly() -> None:
    parent = yaml.safe_load(PARENT.read_text(encoding="utf-8"))
    authority = _authority()

    for key in (
        "authority_transition",
        "bound_freeze_inputs",
        "staged_decision_registry",
        "irreversible_boundaries",
    ):
        assert authority[key] == parent[key]
    expected_sections = parent["staged_section_registry"]
    expected_sections["before_e_confirmatory_access"]["state"] = (
        "COMPLETE_OUTCOME_BLIND_HASH_BOUND"
    )
    expected_sections["before_e_confirmatory_access"][
        "freeze_ready_section_count"
    ] = 13
    assert authority["staged_section_registry"] == expected_sections


def test_atomic_closure_moves_both_f_and_legacy_g0_compatibility_once() -> None:
    authority = _authority()
    routing = authority["active_routing"]
    phases = {phase["phase_id"]: phase for phase in routing["phases"]}
    closure = authority["f_phase_closure"]
    lifecycle = authority["lifecycle"]

    assert routing["active_phase"] == "E"
    assert phases["F"]["complete"] is True
    assert phases["F"]["state"] == (
        "COMPLETE_FREEZE_PANEL_CONFORMANCE_AND_BYTE_SEAL"
    )
    assert phases["E"]["complete"] is False
    assert phases["E"]["state"] == "READY_NOT_RUN"
    assert phases["H"]["state"] == "BYTE_SEALED_UNOPENED"
    assert closure["completed_receipts"] == BUILDER.F_RECEIPT_IDS
    assert closure["pending_receipts"] == []
    assert closure["completion_is_one_atomic_milestone"] is True
    assert closure["closure_is_new_scientific_gate"] is False
    assert closure["confirmatory_or_holdout_rows_read"] == 0
    assert closure["pass_f_now"] is True
    assert closure["pass_g0_now"] is True
    assert lifecycle["pass_f"] is True
    assert lifecycle["pass_g0"] is True
    assert lifecycle["empirical_g1_status"] == "NOT_RUN"
    assert lifecycle["holdout_seal_opened"] is False
    assert lifecycle["confirmatory_or_holdout_rows_read"] == 0


def test_independent_verifier_exactly_replays_frozen_summary() -> None:
    observed = VERIFIER.verify(AUTHORITY)
    expected = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert observed == expected
    assert observed["status"] == SUMMARY_STATUS
    assert observed["f_transition_receipt_replay"]["receipt_count"] == 4
    assert observed["f_transition_receipt_replay"]["all_hashes_exact"] is True
    assert observed["f_closure_replay"]["micro_gate_added"] is False
    assert observed["f_closure_replay"]["pass_f"] is True
    assert observed["f_closure_replay"]["pass_g0_compatibility"] is True
    assert observed["fail_closed_replay"]["empirical_g1_status"] == "NOT_RUN"
    assert observed["fail_closed_replay"]["seal_opened"] is False
    assert observed["access_accounting"]["operator_or_row_level_files_read"] == 0


def test_closure_rejects_receipt_drift_and_unsafe_state(tmp_path: Path) -> None:
    def drift(document: dict) -> None:
        document["f_phase_transition_receipt_bindings"][
            "thirteen_pre_g1_required_sections"
        ]["sha256"] = "0" * 64

    assert "thirteen_pre_g1_required_sections binding" in _verify_mutation(
        tmp_path,
        drift,
    )

    def unsafe(document: dict) -> None:
        document["lifecycle"]["holdout_seal_opened"] = True

    assert "lifecycle holdout_seal_opened" in _verify_mutation(
        tmp_path,
        unsafe,
    )


def test_closure_tools_are_metadata_only_independent_and_create_only() -> None:
    forbidden = {"duckdb", "numpy", "pandas", "pyarrow"}
    for path in (BUILDER_PATH, VERIFIER_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imports.update(
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imports.isdisjoint(forbidden)
    verifier_tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    verifier_imports = {
        node.module
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    verifier_imports.update(
        alias.name
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        module.endswith("build_pivot_feedback_current_state_v2")
        for module in verifier_imports
    )

    before = AUTHORITY.read_bytes()
    with pytest.raises(BUILDER.CurrentStateV2BuildError, match="output exists"):
        BUILDER._write_yaml_create_only(AUTHORITY, _authority())
    assert AUTHORITY.read_bytes() == before


def test_summary_ledger_and_authorization_stop_before_e_execution() -> None:
    authority = _authority()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")

    assert summary["repository_artifacts"]["active_authority"]["sha256"] == _sha(
        AUTHORITY
    )
    assert summary["repository_artifacts"]["builder"]["sha256"] == _sha(
        BUILDER_PATH
    )
    assert summary["repository_artifacts"]["independent_verifier"][
        "sha256"
    ] == _sha(VERIFIER_PATH)
    authorization = authority["authorization"]
    assert authorization["e_g1_execution"] == (
        "AUTHORIZED_NEXT_EXACT_REGISTERED_SCOPE_REQUIRES_SEPARATE_EXECUTION_ARTIFACT"
    )
    assert authorization["confirmatory_or_holdout_analytic_access"] == (
        "NOT_AUTHORIZED_BY_THIS_CLOSURE_ARTIFACT"
    )
    assert authorization["holdout_analytic_access"] == (
        "NOT_AUTHORIZED_BEFORE_M_RESOLVED_AND_SCORING_CONTRACT_LOCKED"
    )
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
    assert "single atomic F milestone" in ledger
    assert "`pass_f=true`" in ledger
    assert "`pass_g0=true`" in ledger
    assert "E/G1 execution input preflight" in ledger
    assert "`empirical_g1_status=NOT_RUN`" in ledger
