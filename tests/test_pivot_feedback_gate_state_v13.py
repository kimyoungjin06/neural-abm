from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v13.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v13.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v13_test_module",
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


def test_current_scope_corrected_gate_passes() -> None:
    result = MODULE.validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["candidate_rows"] == 70_277
    assert result["candidate_topics"] == 994
    assert result["candidate_window_rows"] == 421_662
    assert result["unique_candidate_window_linked_work_rows"] == 22_437_851
    assert result["candidate_window_linked_work_memberships"] == 102_900_039
    assert result["router_covered_topics"] == 4
    assert result["router_covered_candidate_rows"] == 236
    assert result["complete_candidate_work_manifest_materialized"] is False
    assert result["adaptive_frontier_equivalence_required"] is True
    assert result["adaptive_frontier_equivalence_proven"] is False
    assert result["raw_title_scan_authorized"] is False
    assert result["api_required_for_every_candidate"] is False
    assert result["paid_api_submission_authorized"] is False
    assert result["deterministic_reselection_authorized"] is False


def test_frontier_proof_cannot_be_claimed_complete(tmp_path: Path) -> None:
    document = _document()
    document["corrected_strategy"]["eager_lazy_equivalence_proven"] = True

    with pytest.raises(MODULE.GateStateV13Error, match="strategy flag differs"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_full_title_manifest_cannot_remain_immediate(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["d0_complete_title_bearing_work_manifest"] = "AUTHORIZED"

    with pytest.raises(MODULE.GateStateV13Error, match="full manifest"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_raw_title_scan_cannot_be_authorized(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["d0_raw_openalex_title_scan"] = "AUTHORIZED"

    with pytest.raises(MODULE.GateStateV13Error, match="unsafe authorization"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_unresolved_candidate_cannot_be_skipped(tmp_path: Path) -> None:
    document = _document()
    document["next_required_implementation"][
        "unresolved_candidate_may_be_skipped"
    ] = True

    with pytest.raises(MODULE.GateStateV13Error, match="unsafe next-step"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_api_cannot_be_required_for_every_candidate(tmp_path: Path) -> None:
    document = _document()
    document["next_required_implementation"]["api_required_for_every_candidate"] = True

    with pytest.raises(MODULE.GateStateV13Error, match="unsafe next-step"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_v13_verifier_does_not_import_parent_or_scope_verifiers() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "verify_pivot_feedback_gate_state_v12" not in source
    assert (
        "from scripts.verify_pivot_feedback_d0_candidate_screen_scope_audit_v1"
        not in source
    )
    assert (
        "import scripts.verify_pivot_feedback_d0_candidate_screen_scope_audit_v1"
        not in source
    )
