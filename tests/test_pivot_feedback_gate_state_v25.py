from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v25.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v25.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_gate_state_v25_test")


def test_current_gate_state_v25_passes() -> None:
    result = VERIFIER.validate(GATE)

    assert result["status"] == VERIFIER.STATUS
    assert result["rank5_content_unresolved_authorized"] is True
    assert result["terminal_frontier_v6_authorized"] is True
    assert result["selection_certificate_authorized"] is False
    assert result["new_api_request_creation_authorized"] is False


def test_pass_disposition_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["authorization"]["d0_rank5_production_content_disposition"] = (
        "AUTHORIZED_CONTENT_PASS"
    )
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV25VerificationError,
        match="disposition authorization",
    ):
        VERIFIER.validate(mutated)


def test_certificate_opening_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["authorization"]["d0_selection_certificate"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV25VerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated)
