from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v7.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v7.yaml"
RESELECTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-measurement-abi-reselection-v1-full-20260724"
)
CONTENT_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v7_test_module",
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
        reselection_root=RESELECTION_ROOT,
        content_root=CONTENT_ROOT,
    )


def test_current_content_review_gate_passes_with_live_receipts() -> None:
    result = _validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["human_double_review_required"] is True
    assert result["human_double_review_complete"] is False
    assert result["episode_3_quarantined"] is True
    assert result["remaining_episodes_promoted"] is False
    assert result["old_author_panel_current"] is False
    assert result["leaveout_network_bridge_build_authorized"] is False
    assert result["d0_complete"] is False
    assert result["d0_discovery_outcome_opening_authorized"] is False
    assert result["g0_authorized"] is False
    assert result["model_fitting_authorized"] is False
    assert result["g4_open"] is False


def test_ledger_cannot_claim_human_review_completion(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["human_review_state"]["reviewer_1_completed_rows"] = 384
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV7Error,
        match="review completion overstated",
    ):
        _validate(path)


def test_ledger_cannot_promote_remaining_episodes(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["current_provisional_scope"][
        "remaining_episodes_automatically_promoted"
    ] = True
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV7Error,
        match="unsafe provisional scope",
    ):
        _validate(path)


def test_ledger_cannot_reuse_old_author_panel(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["current_provisional_scope"]["prior_author_episode_rows_are_current_panel"] = (
        True
    )
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(
        MODULE.GateStateV7Error,
        match="unsafe provisional scope",
    ):
        _validate(path)


def test_ledger_cannot_open_discovery_outcomes(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["authorization"]["d0_discovery_outcome_opening"] = "AUTHORIZED"
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(MODULE.GateStateV7Error, match="unsafe authorization"):
        _validate(path)


def test_v7_verifier_does_not_import_producers_or_v6_verifier() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_pivot_feedback_d0_measurement_abi_reselection" not in source
    assert "audit_pivot_feedback_d0_content_indexing" not in source
    assert "verify_pivot_feedback_gate_state_v6" not in source
    assert "human_double_review_complete" in source
