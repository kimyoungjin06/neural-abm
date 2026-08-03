from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v5.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v5.yaml"
OUTPUT = Path(
    "/ssd/pivot-feedback-d0b-author-opportunity-foundation-v1-full-20260724"
)
VERIFICATION = Path(
    "/ssd/pivot-feedback-d0b-author-opportunity-foundation-v1-full-20260724."
    "verification-v1.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v5_test_module",
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


def test_current_d0b_foundation_partial_gate_passes_with_live_receipts() -> None:
    result = MODULE.validate(
        LEDGER,
        output_root=OUTPUT,
        verification_report=VERIFICATION,
    )

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["d0a_complete"] is True
    assert result["d0b_author_opportunity_foundation_complete"] is True
    assert result["d0b_complete"] is False
    assert result["d0_complete"] is False
    assert result["d0b_remaining_outcome_blind_measurements_authorized"] is True
    assert result["d0_discovery_outcome_opening_authorized"] is False
    assert result["g0_authorized"] is False
    assert result["model_fitting_authorized"] is False
    assert result["g4_open"] is False


def test_ledger_cannot_open_discovery_outcomes(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["authorization"]["d0_discovery_outcome_opening"] = "AUTHORIZED"
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(MODULE.GateStateV5Error, match="D0 outcomes opened"):
        MODULE.validate(
            path,
            output_root=OUTPUT,
            verification_report=VERIFICATION,
        )


def test_ledger_cannot_relabel_frozen_episode(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["frozen_scope"]["episode_topic_years"][0][1] += 1
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(MODULE.GateStateV5Error, match="frozen episodes differ"):
        MODULE.validate(
            path,
            output_root=OUTPUT,
            verification_report=VERIFICATION,
        )


def test_ledger_cannot_select_team_cap(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    ledger["d0b_author_foundation"]["team_cap_selected"] = True
    path = _write_yaml(tmp_path / "gate.yaml", ledger)

    with pytest.raises(MODULE.GateStateV5Error, match="team cap selected"):
        MODULE.validate(
            path,
            output_root=OUTPUT,
            verification_report=VERIFICATION,
        )


def test_v5_verifier_does_not_import_producer_or_v4_verifier() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_pivot_feedback_d0b_author_opportunity_foundation" not in source
    assert "verify_pivot_feedback_gate_state_v4" not in source
    assert "source_rows_replayed_total" in source
