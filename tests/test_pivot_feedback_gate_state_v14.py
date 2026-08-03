from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v14.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v14.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v14_test_module",
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


def test_current_frontier_proof_gate_passes() -> None:
    result = MODULE.validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["bounded_fixture_cases"] == 299_584
    assert result["bounded_eager_pair_cases"] == 118_404
    assert result["initial_frontier_rows"] == 51
    assert result["structurally_incompatible_rows"] == 47
    assert result["content_required_rows"] == 4
    assert result["content_required_rows_resolved"] == 0
    assert result["general_formal_proof_claimed"] is False
    assert result["selection_certificate_issued"] is False
    assert result["raw_title_scan_authorized"] is False
    assert result["new_api_request_creation_authorized"] is False
    assert result["paid_api_submission_authorized"] is False
    assert result["deterministic_reselection_authorized"] is False


def test_general_formal_proof_cannot_be_claimed(tmp_path: Path) -> None:
    document = _document()
    document["proof_result"]["general_formal_proof_claimed"] = True

    with pytest.raises(MODULE.GateStateV14Error, match="proof result"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_required_content_rows_cannot_be_called_resolved(tmp_path: Path) -> None:
    document = _document()
    document["frontier_result"]["content_required_rows_resolved"] = 4

    with pytest.raises(MODULE.GateStateV14Error, match="frontier result"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_dry_run_cannot_create_production_dispositions(tmp_path: Path) -> None:
    document = _document()
    document["next_required_implementation"][
        "dry_run_may_create_production_content_dispositions"
    ] = True

    with pytest.raises(MODULE.GateStateV14Error, match="unsafe next-step"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_raw_scan_or_new_api_cannot_be_authorized(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["d0_raw_openalex_title_scan"] = "AUTHORIZED"
    document["authorization"]["d0_new_api_request_creation"] = "AUTHORIZED"

    with pytest.raises(MODULE.GateStateV14Error, match="unsafe authorization"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_v14_verifier_does_not_import_parent_or_frontier_verifiers() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "verify_pivot_feedback_gate_state_v13" not in source
    assert (
        "from scripts.verify_pivot_feedback_d0_adaptive_candidate_frontier_v1"
        not in source
    )
    assert (
        "import scripts.verify_pivot_feedback_d0_adaptive_candidate_frontier_v1"
        not in source
    )
