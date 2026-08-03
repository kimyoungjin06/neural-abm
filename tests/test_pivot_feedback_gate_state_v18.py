from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v18.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v18.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_gate_state_v18_test")


def test_current_gate_state_v18_passes() -> None:
    result = VERIFIER.validate(GATE)

    assert result["status"] == VERIFIER.STATUS
    assert result["candidate_rule_v2_production_freeze_authorized"] is True
    assert result["five_production_dispositions_authorized"] is True
    assert result["adaptive_frontier_v3_authorized"] is True
    assert result["selection_certificate_conditioned_on_lazy_closure"] is True
    assert result["new_api_request_creation_authorized"] is False
    assert result["outcomes_read_authorized"] is False


def test_unconditional_certificate_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["authorization"]["d0_selection_certificate_from_frontier_v3"] = (
        "AUTHORIZED"
    )
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV18VerificationError,
        match="certificate authorization",
    ):
        VERIFIER.validate(mutated)


def test_outcomes_opening_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["authorization"]["d0_discovery_outcome_opening"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV18VerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated)
