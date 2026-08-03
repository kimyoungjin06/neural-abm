from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v11.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v11.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v11_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _write(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _document() -> dict:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


def test_current_api_only_gate_passes() -> None:
    result = MODULE.validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
    assert result["human_accuracy"] == "NOT_AVAILABLE"
    assert result["human_cohen_kappa"] == "NOT_AVAILABLE"
    assert result["strict_api_consensus_rows"] == 197
    assert result["unresolved_quarantined_rows"] == 187
    assert result["candidate_universe_screen_complete"] is False
    assert result["paid_candidate_screen_authorized"] is False
    assert result["production_pivot_classification_authorized"] is False
    assert result["d0_complete"] is False
    assert result["outcomes_read"] is False
    assert result["g4_open"] is False


def test_human_review_cannot_be_claimed_complete(tmp_path: Path) -> None:
    document = _document()
    document["api_only_branch"]["human_review"] = "COMPLETE"

    with pytest.raises(MODULE.GateStateV11Error, match="human_review"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_api_consensus_cannot_be_called_ground_truth(tmp_path: Path) -> None:
    document = _document()
    document["api_only_branch"]["api_consensus_is_ground_truth"] = True

    with pytest.raises(
        MODULE.GateStateV11Error,
        match="api_consensus_is_ground_truth",
    ):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_paid_full_screen_cannot_be_authorized(tmp_path: Path) -> None:
    document = _document()
    document["authorization"][
        "d0_api_only_paid_candidate_universe_submission"
    ] = "AUTHORIZED"

    with pytest.raises(MODULE.GateStateV11Error, match="unsafe authorization"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_current_sample_cannot_be_promoted_to_candidate_universe(
    tmp_path: Path,
) -> None:
    document = _document()
    document["next_required_implementation"][
        "current_384_sample_may_be_treated_as_candidate_universe"
    ] = True

    with pytest.raises(MODULE.GateStateV11Error, match="unsafe next-step"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_persistence_cannot_enter_primary_event(tmp_path: Path) -> None:
    document = _document()
    document["next_required_implementation"][
        "persistence_may_enter_primary_event"
    ] = True

    with pytest.raises(MODULE.GateStateV11Error, match="unsafe next-step"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_v11_verifier_does_not_import_prior_gate_verifier() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "verify_pivot_feedback_gate_state_v10" not in source
    assert "build_pivot_feedback_d0_cross_provider_results_v6" not in source
