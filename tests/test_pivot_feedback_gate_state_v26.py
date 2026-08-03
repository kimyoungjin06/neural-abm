from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v26.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v26.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_gate_state_v26_test")


def test_current_gate_state_v26_passes() -> None:
    result = VERIFIER.validate(GATE)
    assert result["status"] == VERIFIER.STATUS
    assert result["api_request_rows_authorized"] == 25
    assert result["provider_calls_authorized"] == 2
    assert result["maximum_authorized_total_usd"] == 0.10
    assert result["human_review_required"] is False
    assert result["augmented_disposition_authorized"] is False


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("frozen_exception", "exact_request_rows"), 26, "request rows"),
        (("frozen_exception", "single_provider_promotion"), True, "single provider"),
        (("authorization", "d0_selection_certificate"), "AUTHORIZED", "unsafe authorization"),
    ],
)
def test_scope_expansion_is_rejected(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    match: str,
) -> None:
    document = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    document[path[0]][path[1]] = value
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(VERIFIER.GateStateV26VerificationError, match=match):
        VERIFIER.validate(mutated)
