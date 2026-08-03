from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v28.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v28.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_gate_state_v28_test")


def test_current_gate_state_v28_passes() -> None:
    result = VERIFIER.validate(GATE)
    assert result["status"] == VERIFIER.STATUS
    assert result["corrected_target_topic_name"] == "Hepatitis C virus research"
    assert result["invalid_prior_results_reusable"] is False
    assert result["cumulative_cost_upper_bound_usd"] == "0.06975"
    assert result["augmented_disposition_authorized"] is False


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("source_row_set_change_allowed", True, "scope expansion"),
        ("models_change_allowed", True, "scope expansion"),
        ("additional_provider_calls_authorized", 3, "provider calls"),
    ],
)
def test_correction_scope_expansion_is_rejected(
    tmp_path: Path,
    key: str,
    value: object,
    match: str,
) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document["corrected_successor"][key] = value
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(VERIFIER.GateStateV28VerificationError, match=match):
        VERIFIER.validate(mutated)
