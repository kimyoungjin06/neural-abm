from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v20.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v20.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_gate_state_v20_test")


def test_current_gate_state_v20_passes() -> None:
    result = VERIFIER.validate(GATE)

    assert result["status"] == VERIFIER.STATUS
    assert result["rank3_production_content_fail_authorized"] is True
    assert result["cumulative_six_dispositions_authorized"] is True
    assert result["adaptive_frontier_v4_authorized"] is True
    assert result["selection_certificate_conditioned_on_lazy_closure"] is True
    assert result["new_api_request_creation_authorized"] is False


def test_unconditional_certificate_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["authorization"]["d0_selection_certificate_from_frontier_v4"] = (
        "AUTHORIZED"
    )
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV20VerificationError,
        match="certificate condition",
    ):
        VERIFIER.validate(mutated)


def test_outcomes_opening_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["authorization"]["d0_discovery_outcome_opening"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV20VerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated)
