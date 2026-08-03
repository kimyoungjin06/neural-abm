from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v23.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v23.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_gate_state_v23_test")


def test_current_gate_state_v23_passes() -> None:
    result = VERIFIER.validate(GATE)

    assert result["status"] == VERIFIER.STATUS
    assert result["rank5_topic_id"] == 10151
    assert result["topic_card_frozen_before_sample_read"] is True
    assert result["exact_rank5_sample_authorized"] is True
    assert result["local_routing_authorized"] is False
    assert result["new_api_request_creation_authorized"] is False


def test_pre_sample_freeze_is_required(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["frozen_topic_card"]["freeze_timing"] = "AFTER_SAMPLE_READ"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV23VerificationError,
        match="card freeze timing",
    ):
        VERIFIER.validate(mutated)


def test_local_routing_and_outcomes_remain_closed(tmp_path: Path) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["authorization"]["d0_rank5_local_routing"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.GateStateV23VerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated)
