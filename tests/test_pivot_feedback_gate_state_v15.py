from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v15.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v15.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v15_test_module",
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


def test_current_candidate_content_gate_passes() -> None:
    result = MODULE.validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["candidate_rows"] == 4
    assert result["would_content_pass_rows"] == 3
    assert result["would_content_fail_indexing_anomaly_rows"] == 1
    assert result["production_content_dispositions_created"] == 0
    assert result["production_content_required_rows_resolved"] == 0
    assert result["selection_certificate_issued"] is False
    assert result["next_target_topic_id_if_rule_later_frozen"] == 10543
    assert result["next_target_title_sample_rows"] == 96
    assert result["raw_title_scan_authorized"] is False
    assert result["new_api_request_creation_authorized"] is False
    assert result["deterministic_reselection_authorized"] is False


def test_dry_run_cannot_resolve_production_rows(tmp_path: Path) -> None:
    document = _document()
    document["dry_run_result"]["production_content_required_rows_resolved"] = 4

    with pytest.raises(MODULE.GateStateV15Error, match="dry-run result"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_counterfactual_cannot_be_applied_to_production(tmp_path: Path) -> None:
    document = _document()
    document["counterfactual_next_frontier"][
        "applied_to_production_states"
    ] = True

    with pytest.raises(MODULE.GateStateV15Error, match="state applied"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_raw_scan_api_or_reselection_cannot_be_authorized(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["d0_raw_openalex_title_scan"] = "AUTHORIZED"
    document["authorization"]["d0_new_api_request_creation"] = "AUTHORIZED"
    document["authorization"]["d0_episode_reselection_successor"] = "AUTHORIZED"

    with pytest.raises(MODULE.GateStateV15Error, match="unsafe authorization"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_v15_verifier_does_not_import_parent_or_dry_run_verifiers() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "verify_pivot_feedback_gate_state_v14" not in source
    assert "verify_pivot_feedback_d0_candidate_content_dry_run_v1" not in source
