from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v17.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v17.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_gate_state_v17_test")


def test_current_gate_state_v17_passes() -> None:
    result = VERIFIER.validate(GATE)

    assert result["status"] == VERIFIER.STATUS
    assert result["candidate_rule_v1_state"] == "WOULD_CONTENT_UNRESOLVED"
    assert result["recent_opaque_quarantine_rows"] == 52
    assert result["candidate_rule_v2_replay_authorized"] is True
    assert result["new_api_request_creation_authorized"] is False
    assert result["production_disposition_authorized"] is False
    assert result["selection_certificate_authorized"] is False


def test_api_opening_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["authorization"]["d0_new_api_request_creation"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV17VerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated)


def test_production_use_requires_retrospective_replay(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["rule_gap"]["simplest_candidate_rule_v2_proposal"][
        "retrospective_replay_required_before_production_use"
    ] = False
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV17VerificationError,
        match="replay requirement",
    ):
        VERIFIER.validate(mutated)
