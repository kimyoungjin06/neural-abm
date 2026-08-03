from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v2.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v2.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v2_test_module", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _ledger() -> dict:
    value = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "gate_state.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_exact_gate_state_is_fail_closed_after_s0_core() -> None:
    result = MODULE.validate(LEDGER)

    assert result["status"] == MODULE.STATUS
    assert result["g_minus_1a"] == "PASS_BOUNDED"
    assert result["s0"] == "PARTIAL_CORE_PASS_DEFERRED_COMPONENTS_NOT_RUN"
    assert result["s0_missing_component_count"] == 3
    assert result["d0_authorized"] is False
    assert result["model_fitting_authorized"] is False
    assert result["g4_open"] is False
    assert result["scientific_effect"] == "NOT_ESTABLISHED"
    assert result["next_workload_class"] == "CPU_IO"
    assert result["gpu_required"] is False


def test_s0_cannot_be_marked_complete_without_deferred_evidence(tmp_path: Path) -> None:
    ledger = _ledger()
    ledger["gate_state"]["S0"]["complete"] = True

    with pytest.raises(MODULE.GateStateError, match="S0 improperly complete"):
        MODULE.validate(_write(tmp_path, ledger))


def test_d0_cannot_be_authorized_while_s0_is_partial(tmp_path: Path) -> None:
    ledger = _ledger()
    ledger["authorization"]["d0_discovery_pilot"] = "AUTHORIZED"

    with pytest.raises(MODULE.GateStateError, match="unsafe authorization"):
        MODULE.validate(_write(tmp_path, ledger))


def test_s0_summary_hash_drift_is_rejected(tmp_path: Path) -> None:
    ledger = _ledger()
    ledger["execution_evidence"]["s0_core_inventory"]["sha256"] = "0" * 64

    with pytest.raises(MODULE.GateStateError, match="bound evidence SHA mismatch"):
        MODULE.validate(_write(tmp_path, ledger))
