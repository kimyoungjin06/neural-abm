from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v8.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v8.yaml"
OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-review-packets-v1-20260729"
)
VERIFICATION_REPORT = Path(
    "/ssd/pivot-feedback-d0-content-review-packets-v1-20260729.verification-v1.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v8_test_module",
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


def test_current_packet_preparation_gate_passes_with_live_receipts() -> None:
    result = _validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["packets_prepared"] is True
    assert result["packets_independently_verified"] is True
    assert result["reviewer_1_completed_rows"] == 0
    assert result["reviewer_2_completed_rows"] == 0
    assert result["adjudicated_rows"] == 0
    assert result["human_review_complete"] is False
    assert result["episode_3_quarantined"] is True
    assert result["remaining_episodes_promoted"] is False
    assert result["old_author_panel_current"] is False
    assert result["d0_complete"] is False
    assert result["d0_discovery_outcome_opening_authorized"] is False
    assert result["g4_open"] is False


def test_ledger_cannot_claim_reviewer_completion(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["review_packet_state"]["reviewer_1_completed_rows"] = 384
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV8Error,
        match="review completion overstated",
    ):
        _validate(path)


def test_ledger_cannot_allow_artificial_reviewer_substitution(
    tmp_path: Path,
) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["review_packet_state"][
        "artificial_agent_may_substitute_for_human_reviewer"
    ] = True
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV8Error,
        match="artificial reviewer substitution allowed",
    ):
        _validate(path)


def test_ledger_cannot_open_discovery_outcomes(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["authorization"]["d0_discovery_outcome_opening"] = "AUTHORIZED"
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(MODULE.GateStateV8Error, match="unsafe authorization"):
        _validate(path)


def test_v8_verifier_does_not_import_packet_producer_or_prior_verifier() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_pivot_feedback_d0_content_review_packets" not in source
    assert "verify_pivot_feedback_d0_content_review_packets" not in source
    assert "verify_pivot_feedback_gate_state_v7" not in source
    assert "packets_independently_verified" in source
