from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v12.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v12.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v12_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _document() -> dict:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_current_zero_cost_routing_gate_passes() -> None:
    result = MODULE.validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["route_counts"] == MODULE.EXPECTED_ROUTES
    assert result["avoided_api_request_rows"] == 290
    assert result["api_exception_rows"] == 94
    assert result["api_required_for_every_candidate"] is False
    assert result["complete_candidate_work_manifest_materialized"] is False
    assert result["candidate_universe_screen_complete"] is False
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
    assert result["paid_api_submission_authorized"] is False
    assert result["production_pivot_classification_authorized"] is False
    assert result["d0_complete"] is False
    assert result["outcomes_read"] is False
    assert result["g4_open"] is False


def test_sample_cannot_be_called_accurate(tmp_path: Path) -> None:
    document = _document()
    document["routing_result"]["accuracy_established"] = True

    with pytest.raises(MODULE.GateStateV12Error, match="routing result differs"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_paid_api_submission_cannot_be_authorized(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["d0_paid_api_exception_submission"] = "AUTHORIZED"

    with pytest.raises(MODULE.GateStateV12Error, match="unsafe authorization"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_api_cannot_be_required_for_every_candidate(tmp_path: Path) -> None:
    document = _document()
    document["next_required_implementation"]["api_required_for_every_candidate"] = True

    with pytest.raises(MODULE.GateStateV12Error, match="unsafe next-step"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_topic_year_rows_cannot_be_called_candidate_works(tmp_path: Path) -> None:
    document = _document()
    document["next_required_implementation"][
        "existing_70277_rows_are_topic_year_candidates_not_candidate_works"
    ] = False

    with pytest.raises(MODULE.GateStateV12Error, match="70,277-row boundary"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_current_sample_cannot_be_promoted_to_complete_universe(
    tmp_path: Path,
) -> None:
    document = _document()
    document["next_required_implementation"][
        "current_384_sample_may_be_treated_as_candidate_universe"
    ] = True

    with pytest.raises(MODULE.GateStateV12Error, match="unsafe next-step"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_v12_verifier_does_not_import_prior_gate_or_routing_verifier() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "verify_pivot_feedback_gate_state_v11" not in source
    assert "from scripts.verify_pivot_feedback_d0_api_only_topic_routing_v1" not in source
    assert "import scripts.verify_pivot_feedback_d0_api_only_topic_routing_v1" not in source
