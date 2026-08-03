from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v9.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v9.yaml"
OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-api-review-v2-prepared-20260729"
)
VERIFICATION_REPORT = Path(
    "/ssd/pivot-feedback-d0-api-review-v2-prepared-20260729.verification-v2.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v9_test_module",
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


def test_current_api_preparation_gate_passes_with_live_receipts() -> None:
    result = _validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["batch_requests_prepared"] is True
    assert result["batch_requests_independently_verified"] is True
    assert result["api_submitted"] is False
    assert result["api_responses_received"] is False
    assert result["human_calibration_complete"] is False
    assert result["episode_3_quarantined"] is True
    assert result["candidate_universe_screen_authorized"] is False
    assert result["d0_complete"] is False
    assert result["g4_open"] is False


def test_ledger_cannot_claim_api_submission(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["api_review_state"]["api_submitted"] = True
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV9Error,
        match="execution completion overstated",
    ):
        _validate(path)


def test_ledger_cannot_allow_artificial_reviewer_substitution(
    tmp_path: Path,
) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["api_review_state"][
        "artificial_agent_may_substitute_for_human_reviewer"
    ] = True
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV9Error,
        match="artificial reviewer substitution allowed",
    ):
        _validate(path)


def test_ledger_cannot_authorize_candidate_universe_screen(
    tmp_path: Path,
) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["authorization"]["d0_candidate_universe_api_screen"] = "AUTHORIZED"
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV9Error,
        match="candidate screen prematurely authorized",
    ):
        _validate(path)


def test_v9_verifier_does_not_import_producer_or_prior_verifier() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "prepare_pivot_feedback_d0_api_review_v2" not in source
    assert "verify_pivot_feedback_d0_api_review_preparation_v2" not in source
    assert "verify_pivot_feedback_gate_state_v8" not in source
    assert "batch_requests_independently_verified" in source
