from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v16.py"
LEDGER = ROOT / "experiments/configs/pivot_feedback_gate_state_v16.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_gate_state_v16_test_module",
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


def test_current_exact_target_authorization_passes() -> None:
    result = MODULE.validate(LEDGER)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["production_candidate_dispositions"] == 4
    assert result["medicine_required_topic_id"] == 10543
    assert result["exact_target_title_rows_authorized"] == 96
    assert result["general_raw_scan_authorized"] is False
    assert result["new_api_request_creation_authorized"] is False
    assert result["global_selection_certificate_issued"] is False
    assert result["outcomes_read"] is False


def test_target_cannot_be_broadened(tmp_path: Path) -> None:
    document = _document()
    document["exact_targeted_sample_authorization"]["total_rows"] = 192

    with pytest.raises(MODULE.GateStateV16Error, match="target"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_api_or_reselection_cannot_be_authorized(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["d0_new_api_request_creation"] = "AUTHORIZED"
    document["authorization"]["d0_episode_reselection_successor"] = "AUTHORIZED"

    with pytest.raises(MODULE.GateStateV16Error, match="unsafe authorization"):
        MODULE.validate(_write(tmp_path / "ledger.yaml", document))


def test_v16_verifier_does_not_import_parent_or_producer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "verify_pivot_feedback_gate_state_v15" not in source
    assert "build_pivot_feedback_d0_production_candidate_rule_v1" not in source
