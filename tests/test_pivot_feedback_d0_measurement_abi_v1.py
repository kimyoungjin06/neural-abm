from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_measurement_abi_v1.py"
ABI = ROOT / "experiments/configs/pivot_feedback_d0_measurement_abi_v1.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_measurement_abi_v1_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _document() -> dict:
    return yaml.safe_load(ABI.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_current_measurement_abi_passes() -> None:
    result = MODULE.validate(ABI)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["included_work_type_names"] == [
        "article",
        "book",
        "book-chapter",
        "book-section",
        "dissertation",
        "preprint",
        "report",
        "review",
    ]
    assert result["selected_field_ids"] == [17, 27]
    assert result["valid_author_linkage_required"] is True
    assert result["outcomes_read"] is False
    assert result["episode_reselection_authorized"] is True
    assert result["leaveout_network_bridge_authorized"] is False


def test_work_type_cannot_be_included_and_excluded(tmp_path: Path) -> None:
    document = _document()
    document["work_type_eligibility"]["excluded"]["article"] = "CONFLICT"

    with pytest.raises(MODULE.MeasurementABIError, match="excluded work type set"):
        MODULE.validate(_write(tmp_path / "abi.yaml", document))


def test_author_linkage_requirement_cannot_be_weakened(tmp_path: Path) -> None:
    document = _document()
    document["measurement_abi"]["valid_author_linkage_rule"] = "OPTIONAL"

    with pytest.raises(MODULE.MeasurementABIError, match="valid_author_linkage_rule"):
        MODULE.validate(_write(tmp_path / "abi.yaml", document))


def test_historical_episode_preservation_cannot_become_objective(
    tmp_path: Path,
) -> None:
    document = _document()
    document["episode_reselection"][
        "historical_episode_ids_are_not_selection_inputs"
    ] = False

    with pytest.raises(
        MODULE.MeasurementABIError,
        match="historical episodes used",
    ):
        MODULE.validate(_write(tmp_path / "abi.yaml", document))


def test_outcomes_cannot_be_opened(tmp_path: Path) -> None:
    document = _document()
    document["forbidden_inputs_and_outputs"]["entry_effect_size"] = True

    with pytest.raises(MODULE.MeasurementABIError, match="forbidden state opened"):
        MODULE.validate(_write(tmp_path / "abi.yaml", document))


def test_validator_does_not_import_reselection_producer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_pivot_feedback_d0_measurement_abi_reselection" not in source
