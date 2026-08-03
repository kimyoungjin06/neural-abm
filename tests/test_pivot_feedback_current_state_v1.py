from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Callable

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "experiments/configs/pivot_feedback_current_state_v1.yaml"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_current_state_v1.py"
SUMMARY = (
    ROOT
    / "experiments/evidence/results/pivot_feedback_current_state_v1_20260803.summary.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("current_state_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_mutation(
    tmp_path: Path,
    mutate: Callable[[dict], None],
) -> str:
    verifier = _load_verifier()
    document = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "mutated_current_state.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(verifier.CurrentStateVerificationError) as error:
        verifier.verify(path.resolve())
    return str(error.value)


def test_current_state_activates_one_six_phase_authority() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))

    assert authority["schema_version"] == "pivot_feedback_current_state.v1"
    assert authority["authority_transition"]["active_routing_uses_this_authority"]
    assert authority["active_routing"]["phase_order"] == ["D", "C", "F", "E", "M", "H"]
    assert authority["active_routing"]["active_phase"] == "F"
    assert _sha256(AUTHORITY) == (
        "52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd"
    )
    for container_name in ("authority_transition", "bound_freeze_inputs"):
        for binding in authority[container_name].values():
            if not isinstance(binding, dict) or "relative_path" not in binding:
                continue
            assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])


def test_pre_g1_registry_is_complete_and_later_choices_are_deferred() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    decisions = authority["staged_decision_registry"]
    sections = authority["staged_section_registry"]

    assert decisions["before_e_confirmatory_access"]["choice_count"] == 13
    assert decisions["before_e_confirmatory_access"]["state"] == (
        "COMPLETE_OUTCOME_BLIND_HASH_BOUND"
    )
    assert decisions["before_m_g2_model_score_access"]["choice_count"] == 3
    assert decisions["before_m_g3_simulation_execution"]["choice_count"] == 3
    assert sections["before_e_confirmatory_access"]["section_count"] == 13
    assert sections["before_e_confirmatory_access"]["freeze_ready_section_count"] == 0
    assert sections["before_m_g2_model_score_access"]["section_ids"] == [
        "g2_predictive_gate"
    ]
    assert sections["before_m_g3_simulation_execution"]["section_ids"] == [
        "g3_simulation_gate"
    ]


def test_f_is_one_incomplete_milestone_not_three_new_gates() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    closure = authority["f_phase_closure"]

    assert closure["completed_receipts"] == ["PRE_G1_DECISION_REGISTRY_COMPLETE"]
    assert len(closure["pending_receipts"]) == 3
    assert closure["completion_is_one_atomic_milestone"] is True
    assert closure["intermediate_receipts_may_record_provenance_but_are_not_new_gates"]
    assert closure["pass_f_now"] is False
    assert closure["pass_g0_now"] is False


def test_verifier_matches_frozen_receipt() -> None:
    completed = subprocess.run(
        [sys.executable, str(VERIFIER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    expected = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert observed == expected
    assert observed["status"] == (
        "PASS_ACTIVE_SIX_PHASE_AUTHORITY_F_IN_PROGRESS_FAIL_CLOSED"
    )
    assert observed["f_phase_replay"]["micro_gate_added"] is False
    assert observed["fail_closed_replay"]["pass_g0"] is False


def test_premature_f_pass_is_rejected(tmp_path: Path) -> None:
    def mutate(document: dict) -> None:
        document["lifecycle"]["pass_f"] = True

    assert "pass_f" in _verify_mutation(tmp_path, mutate)


def test_later_choice_moved_back_to_f_is_rejected(tmp_path: Path) -> None:
    def mutate(document: dict) -> None:
        document["staged_decision_registry"]["before_e_confirmatory_access"][
            "choice_ids"
        ].append("g2_score_margin")

    assert "decision uniqueness" in _verify_mutation(tmp_path, mutate)


def test_confirmatory_access_authorization_is_rejected(tmp_path: Path) -> None:
    def mutate(document: dict) -> None:
        document["authorization"]["confirmatory_or_holdout_analytic_access"] = (
            "AUTHORIZED"
        )

    assert "denial confirmatory_or_holdout_analytic_access" in _verify_mutation(
        tmp_path, mutate
    )
