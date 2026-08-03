from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v10.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v10.yaml"
OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v6-results-20260729"
)
VERIFICATION_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v6-results-20260729"
    ".verification-v6.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v10_test_module",
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


def _validate(path: Path):
    return MODULE.validate(
        path,
        output_root=OUTPUT_ROOT,
        verification_report=VERIFICATION_REPORT,
    )


def test_current_cross_provider_gate_passes_with_live_receipts() -> None:
    result = _validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["api_submitted"] is True
    assert result["api_responses_received"] is True
    assert result["api_review_complete_for_frozen_set"] is True
    assert result["openai_valid_structured_rows"] == 384
    assert result["claude_valid_structured_rows"] == 384
    assert result["human_queue_rows_per_reviewer"] == 233
    assert result["human_calibration_complete"] is False
    assert result["candidate_universe_screen_authorized"] is False
    assert result["d0_complete"] is False
    assert result["g4_open"] is False


def test_ledger_cannot_claim_human_completion(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["api_review_state"]["human_calibration_complete"] = True
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(MODULE.GateStateV10Error, match="unsafe API review state"):
        _validate(path)


def test_ledger_cannot_authorize_candidate_universe_screen(
    tmp_path: Path,
) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["authorization"]["d0_candidate_universe_api_screen"] = "AUTHORIZED"
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV10Error,
        match="candidate screen prematurely authorized",
    ):
        _validate(path)


def test_ledger_cannot_expose_api_outputs_to_human_reviewers(
    tmp_path: Path,
) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["next_required_implementation"][
        "api_outputs_visible_to_human_reviewers"
    ] = True
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(MODULE.GateStateV10Error, match="unsafe next-step flag"):
        _validate(path)


def test_v10_verifier_does_not_import_producer_or_prior_verifier() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_pivot_feedback_d0_cross_provider_results_v6" not in source
    assert "verify_pivot_feedback_gate_state_v9" not in source
    assert "api_review_complete_for_frozen_set" in source
