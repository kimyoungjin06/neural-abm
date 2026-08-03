from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/configs/pivot_feedback_gate_sequence_v1.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_sequence_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_sequence_v1_test_module", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _contract() -> dict:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "gate_sequence.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_exact_contract_is_acyclic_and_fail_closed() -> None:
    result = MODULE.validate(CONTRACT)
    assert result["status"] == MODULE.STATUS
    assert result["topological_order"] == MODULE.EXPECTED_SEQUENCE
    assert result["node_count"] == 10
    assert result["edge_count"] == 9
    assert result["cycle_present"] is False
    assert result["final_cutoff_panel_owner"] == "G0_5"
    assert result["legacy_v0_parents_unchanged"] is True
    assert result["execution_authorized"] is False
    assert result["scientific_effect"] == "NOT_ESTABLISHED"
    assert result["contract_sha256"] == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()


def test_final_panel_is_post_freeze_not_g_minus_1a_or_g0_prerequisite() -> None:
    contract = _contract()
    rules = contract["separation_rules"]
    assert rules["final_cutoff_panel_owner"] == "G0_5"
    assert rules["final_cutoff_panel_is_g_minus_1a_prerequisite"] is False
    assert rules["final_cutoff_panel_is_g0_prerequisite"] is False
    assert rules["g0_freezes_definitions_before_final_panel_materialization"] is True
    nodes = {node["gate_id"]: node for node in contract["gate_nodes"]}
    assert nodes["G0"]["requires"] == ["D0"]
    assert nodes["G0_5"]["requires"] == ["G0"]
    assert nodes["G1"]["requires"] == ["G0_5"]


def test_s0_is_threshold_free_and_d0_cannot_select_by_effect() -> None:
    contract = _contract()
    rules = contract["separation_rules"]
    assert {
        "eligible_author_count",
        "matured_entry_opportunity_count",
        "fixed_horizon_outcome_coverage",
        "independent_attention_shock_count",
        "entry_effect_size",
        "persistence_effect_size",
        "overshoot_timing_gap",
        "predictive_score",
    } <= set(rules["s0_forbidden_metrics"])
    assert rules["d0_entry_end_year"] == 2012
    assert rules["d0_may_select_final_fields_by_effect_size"] is False
    assert rules["d0_may_change_confirmatory_years"] is False


def test_backward_g0_dependency_is_rejected(tmp_path: Path) -> None:
    contract = _contract()
    nodes = {node["gate_id"]: node for node in contract["gate_nodes"]}
    nodes["G0"]["requires"] = ["G0_5"]
    with pytest.raises(
        MODULE.GateSequenceError,
        match="backward or same-order dependency G0_5->G0",
    ):
        MODULE.validate(_write(tmp_path, contract))


def test_second_final_panel_owner_is_rejected(tmp_path: Path) -> None:
    contract = _contract()
    nodes = {node["gate_id"]: node for node in contract["gate_nodes"]}
    nodes["G0"]["owns_final_cutoff_panel"] = True
    with pytest.raises(
        MODULE.GateSequenceError,
        match="final cutoff panel must have one G0.5 owner",
    ):
        MODULE.validate(_write(tmp_path, contract))


def test_v0_parent_hash_drift_is_rejected(tmp_path: Path) -> None:
    contract = _contract()
    contract["immutable_parents"]["pending_g0_overlay"]["sha256"] = "0" * 64
    with pytest.raises(
        MODULE.GateSequenceError,
        match="parent SHA mismatch",
    ):
        MODULE.validate(_write(tmp_path, contract))
