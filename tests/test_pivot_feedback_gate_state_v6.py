from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v6.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v6.yaml"
OUTPUT = Path(
    "/ssd/pivot-feedback-d0b-selected-work-author-linkage-audit-v1-full-20260724"
)
VERIFICATION = Path(
    "/ssd/pivot-feedback-d0b-selected-work-author-linkage-audit-v1-full-20260724."
    "verification-v1.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v6_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _write_yaml(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_current_linkage_review_gate_passes_with_live_receipts() -> None:
    result = MODULE.validate(
        LEDGER,
        output_root=OUTPUT,
        verification_report=VERIFICATION,
    )

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["measurement_review_required"] is True
    assert result["measurement_abi_redesign_authorized"] is True
    assert result["episode_reselection_successor_authorized_after_abi"] is True
    assert result["leaveout_network_bridge_build_authorized"] is False
    assert result["d0b_complete"] is False
    assert result["d0_complete"] is False
    assert result["d0_discovery_outcome_opening_authorized"] is False
    assert result["g0_authorized"] is False
    assert result["model_fitting_authorized"] is False
    assert result["g4_open"] is False


def test_ledger_cannot_resume_leaveout_before_reselection(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["authorization"]["d0b_leaveout_network_bridge_build"] = "AUTHORIZED"
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(MODULE.GateStateV6Error, match="leaveout build not paused"):
        MODULE.validate(
            path,
            output_root=OUTPUT,
            verification_report=VERIFICATION,
        )


def test_ledger_cannot_claim_episode_exclusion(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["measurement_diagnosis"]["episode_exclusion_decision_made"] = True
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV6Error,
        match="unsafe diagnosis state: episode_exclusion_decision_made",
    ):
        MODULE.validate(
            path,
            output_root=OUTPUT,
            verification_report=VERIFICATION,
        )


def test_ledger_cannot_open_discovery_outcomes(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["authorization"]["d0_discovery_outcome_opening"] = "AUTHORIZED"
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(MODULE.GateStateV6Error, match="D0 outcomes opened"):
        MODULE.validate(
            path,
            output_root=OUTPUT,
            verification_report=VERIFICATION,
        )


def test_v6_verifier_does_not_import_linkage_producer_or_v5_verifier() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert (
        "build_pivot_feedback_d0b_selected_work_author_linkage_audit"
        not in source
    )
    assert "verify_pivot_feedback_gate_state_v5" not in source
    assert "leaveout_network_bridge_build_authorized" in source
