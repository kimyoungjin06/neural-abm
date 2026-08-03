from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v3.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v3_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def _states(*, s0_complete: bool, d0_state: str) -> dict:
    return {
        "G_MINUS_1A": {
            "state": "PASS_BOUNDED",
            "complete": True,
            "authorizes": ["S0"],
        },
        "S0": (
            {
                "state": "PASS_THRESHOLD_FREE_INVENTORY",
                "complete": True,
                "authorizes": ["D0"],
            }
            if s0_complete
            else {
                "state": "PARTIAL_CORE_PASS_DEFERRED_COMPONENTS_NOT_RUN",
                "complete": False,
            }
        ),
        "D0": (
            {
                "state": "AUTHORIZED_NOT_RUN",
                "complete": False,
                "requires": ["S0"],
                "entry_year_end": 2012,
            }
            if d0_state == "AUTHORIZED_NOT_RUN"
            else {
                "state": "NOT_AUTHORIZED",
                "complete": False,
                "requires": ["S0"],
            }
        ),
        "G0": {"state": "PENDING", "complete": False},
        "G0_5": {"state": "NOT_RUN", "complete": False},
        "G1": {"state": "NOT_RUN", "complete": False},
        "G2": {"state": "NOT_RUN", "complete": False},
        "G3": {"state": "NOT_RUN", "complete": False},
        "G4": {"state": "SEALED_NOT_OPENED", "complete": False},
        "G5": {"state": "NOT_RUN_OPTIONAL", "complete": False},
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    for relative_path in [
        *MODULE.S0_VERIFIER_PATHS.values(),
        MODULE.S0_COMPLETION_VERIFIER_PATH,
    ]:
        source = ROOT / relative_path
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    parent_path = repo / "experiments/configs/pivot_feedback_gate_state_v2.yaml"
    _write_yaml(
        parent_path,
        {
            "schema_version": "pivot_feedback_gate_state.v2",
            "status": "VALID_G_MINUS_1A_PASS_S0_PARTIAL_D0_NOT_AUTHORIZED",
            "gate_state": _states(s0_complete=False, d0_state="NOT_AUTHORIZED"),
        },
    )
    g1_path = (
        repo
        / "experiments/evidence/results/"
        "pivot_feedback_g_minus_1a_receipt_20260723.summary.json"
    )
    _write_json(
        g1_path,
        {
            "schema_version": "pivot_feedback_g_minus_1a_receipt.v1",
            "status": (
                "PASS_G_MINUS_1A_BOUNDED_PRIMITIVE_AND_SOURCE_FIDELITY_"
                "WITH_CURRENT_SNAPSHOT_PROXY_LIMITATIONS"
            ),
            "authorization": {
                "s0_threshold_free_inventory": "AUTHORIZED",
                "d0_discovery_pilot": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
        },
    )
    s0_path = (
        repo
        / "experiments/evidence/results/"
        "pivot_feedback_s0_completion_full_20260724.summary.json"
    )
    _write_json(
        s0_path,
        {
            "schema_version": "pivot_feedback_s0_completion_receipt.v1",
            "status": MODULE.S0_RECEIPT_STATUS,
            "scope": {
                "s0_gate_pass": True,
                "threshold_free_only": True,
                "outcomes_read": False,
                "d0_discovery_pilot_authorized": True,
                "d0_entry_end_year": 2012,
                "g0_measurement_freeze_authorized": False,
                "final_panel_materialization_authorized": False,
                "model_fitting_authorized": False,
                "g4_open": False,
                "gpu_used": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "authorization": {
                "d0_discovery_only_pre2013": "AUTHORIZED",
                "g0_measurement_freeze": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "component_receipts": {
                label: {
                    "verifier_sha256": _sha(repo / verifier_path)
                }
                for label, verifier_path in MODULE.S0_VERIFIER_PATHS.items()
            },
            "next_required_implementation": {
                "gate": "D0",
                "entry_year_end": 2012,
                "gpu_required": False,
            },
            "completion_verifier_sha256": _sha(
                repo / MODULE.S0_COMPLETION_VERIFIER_PATH
            ),
        },
    )
    ledger_path = repo / "experiments/configs/pivot_feedback_gate_state_v3.yaml"
    _write_yaml(
        ledger_path,
        {
            "schema_version": MODULE.SCHEMA,
            "label": "pivot_feedback_gate_state_v3",
            "mode": "ADDITIVE_S0_COMPLETION_GATE_LEDGER",
            "status": MODULE.LEDGER_STATUS,
            "additive_parent": {
                "relative_path": (
                    "experiments/configs/pivot_feedback_gate_state_v2.yaml"
                ),
                "sha256": _sha(parent_path),
                "required_schema_version": "pivot_feedback_gate_state.v2",
                "required_status": (
                    "VALID_G_MINUS_1A_PASS_S0_PARTIAL_D0_NOT_AUTHORIZED"
                ),
                "required_fields": {},
                "mutation_allowed": False,
            },
            "execution_evidence": {
                "g_minus_1a_receipt": {
                    "relative_path": (
                        "experiments/evidence/results/"
                        "pivot_feedback_g_minus_1a_receipt_20260723.summary.json"
                    ),
                    "sha256": _sha(g1_path),
                    "required_schema_version": (
                        "pivot_feedback_g_minus_1a_receipt.v1"
                    ),
                    "required_status": (
                        "PASS_G_MINUS_1A_BOUNDED_PRIMITIVE_AND_SOURCE_"
                        "FIDELITY_WITH_CURRENT_SNAPSHOT_PROXY_LIMITATIONS"
                    ),
                    "required_fields": {
                        "authorization.s0_threshold_free_inventory": "AUTHORIZED",
                        "authorization.d0_discovery_pilot": "NOT_AUTHORIZED",
                    },
                },
                "s0_completion_receipt": {
                    "relative_path": (
                        "experiments/evidence/results/"
                        "pivot_feedback_s0_completion_full_20260724.summary.json"
                    ),
                    "sha256": _sha(s0_path),
                    "required_schema_version": (
                        "pivot_feedback_s0_completion_receipt.v1"
                    ),
                    "required_status": MODULE.S0_RECEIPT_STATUS,
                    "required_fields": {
                        "scope.s0_gate_pass": True,
                        "scope.threshold_free_only": True,
                        "scope.outcomes_read": False,
                        "scope.d0_discovery_pilot_authorized": True,
                        "scope.d0_entry_end_year": 2012,
                        "scope.model_fitting_authorized": False,
                        "scope.gpu_used": False,
                        "scope.scientific_effect": "NOT_ESTABLISHED",
                    },
                },
            },
            "gate_state": _states(
                s0_complete=True,
                d0_state="AUTHORIZED_NOT_RUN",
            ),
            "authorization": {
                "d0_discovery_pilot": (
                    "AUTHORIZED_DISCOVERY_ONLY_ENTRY_YEAR_AT_MOST_2012"
                ),
                "g0_measurement_freeze": "NOT_AUTHORIZED",
                "final_panel_materialization": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "next_required_implementation": {
                "gate": "D0",
                "entry_year_end": 2012,
                "final_field_choice_by_effect_size_allowed": False,
                "confirmatory_year_changes_allowed": False,
                "workload_class": "CPU_IO",
                "gpu_required": False,
            },
        },
    )
    return repo, ledger_path


def test_s0_completion_authorizes_only_d0(tmp_path: Path) -> None:
    repo, ledger = _fixture(tmp_path)

    result = MODULE.validate(ledger, repo_root=repo)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["s0"] == "PASS_THRESHOLD_FREE_INVENTORY"
    assert result["d0"] == "AUTHORIZED_NOT_RUN"
    assert result["g0_authorized"] is False
    assert result["model_fitting_authorized"] is False
    assert result["g4_open"] is False


def test_s0_evidence_mutation_fails_closed(tmp_path: Path) -> None:
    repo, ledger = _fixture(tmp_path)
    s0 = (
        repo
        / "experiments/evidence/results/"
        "pivot_feedback_s0_completion_full_20260724.summary.json"
    )
    value = json.loads(s0.read_text(encoding="utf-8"))
    value["scope"]["model_fitting_authorized"] = True
    _write_json(s0, value)

    with pytest.raises(MODULE.GateStateV3Error, match="SHA mismatch"):
        MODULE.validate(ledger, repo_root=repo)


def test_ledger_cannot_authorize_model_fitting(tmp_path: Path) -> None:
    repo, ledger = _fixture(tmp_path)
    value = yaml.safe_load(ledger.read_text(encoding="utf-8"))
    value["authorization"]["model_fitting"] = "AUTHORIZED"
    _write_yaml(ledger, value)

    with pytest.raises(MODULE.GateStateV3Error, match="unsafe authorization"):
        MODULE.validate(ledger, repo_root=repo)


def test_ledger_cannot_open_g4(tmp_path: Path) -> None:
    repo, ledger = _fixture(tmp_path)
    value = yaml.safe_load(ledger.read_text(encoding="utf-8"))
    value["gate_state"]["G4"]["state"] = "OPEN"
    _write_yaml(ledger, value)

    with pytest.raises(MODULE.GateStateV3Error, match="G4 is not sealed"):
        MODULE.validate(ledger, repo_root=repo)


def test_parent_must_remain_partial_and_fail_closed(tmp_path: Path) -> None:
    repo, ledger = _fixture(tmp_path)
    parent = repo / "experiments/configs/pivot_feedback_gate_state_v2.yaml"
    value = yaml.safe_load(parent.read_text(encoding="utf-8"))
    value["gate_state"]["S0"]["complete"] = True
    _write_yaml(parent, value)
    ledger_value = yaml.safe_load(ledger.read_text(encoding="utf-8"))
    ledger_value["additive_parent"]["sha256"] = _sha(parent)
    _write_yaml(ledger, ledger_value)

    with pytest.raises(
        MODULE.GateStateV3Error,
        match="parent no longer records partial S0",
    ):
        MODULE.validate(ledger, repo_root=repo)
