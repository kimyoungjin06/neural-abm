from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_g_minus_1a_receipt_v1.yaml"
SCRIPT = ROOT / "scripts/verify_pivot_feedback_g_minus_1a_receipt_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_g_minus_1a_receipt_v1_test_module", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _config() -> dict:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "g_minus_1a.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_exact_repository_evidence_issues_bounded_g_minus_1a_receipt() -> None:
    result = MODULE.verify(CONFIG)

    assert result["status"] == MODULE.PASS_STATUS
    assert result["gate_id"] == "G_MINUS_1A"
    assert result["scope"] == {
        "full_repository_evidence_set": True,
        "source_payloads_opened": 0,
        "production_scans_started": 0,
        "production_outputs_mutated": False,
        "gpu_used": False,
    }
    assert result["authorization"] == {
        "s0_threshold_free_inventory": "AUTHORIZED",
        "d0_discovery_pilot": "NOT_AUTHORIZED",
        "model_fitting": "NOT_AUTHORIZED",
        "legacy_g_minus_1": "NOT_PASSED",
        "g0_measurement_freeze": "PENDING",
        "scientific_effect": "NOT_ESTABLISHED",
    }
    assert [item["component_id"] for item in result["repository_evidence"]] == (
        MODULE.EXPECTED_COMPONENTS
    )
    assert all(
        item["verification"] == "PASS_EXACT_REPOSITORY_EVIDENCE"
        for item in result["repository_evidence"]
    )
    assert result["cross_component_closures"]["retained_work_rows"] == 492293206
    assert result["unresolved_boundaries"] == MODULE.EXPECTED_UNRESOLVED
    assert result["provenance"]["config_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()


def test_evidence_hash_drift_is_rejected(tmp_path: Path) -> None:
    config = _config()
    config["repository_evidence"][0]["sha256"] = "0" * 64

    with pytest.raises(MODULE.GMinus1AError, match="SHA-256 mismatch"):
        MODULE.verify(_write(tmp_path, config))


def test_required_source_replay_equality_cannot_be_weakened(tmp_path: Path) -> None:
    config = _config()
    replay = config["repository_evidence"][-1]
    replay["required_fields"][
        "row_level_verification.all_projected_values_and_null_masks_exact"
    ] = False

    with pytest.raises(MODULE.GMinus1AError, match="field mismatch"):
        MODULE.verify(_write(tmp_path, config))


def test_cross_component_receipt_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config()
    config["cross_component_closures"]["author_history_receipt_sha256"] = "0" * 64

    with pytest.raises(
        MODULE.GMinus1AError, match="declared cross-component closures differ"
    ):
        MODULE.verify(_write(tmp_path, config))


def test_unresolved_author_entity_cannot_be_promoted(tmp_path: Path) -> None:
    config = _config()
    config["unresolved_boundaries"]["author_entity_validity"] = "PASS"

    with pytest.raises(MODULE.GMinus1AError, match="unresolved-boundary registry differs"):
        MODULE.verify(_write(tmp_path, config))


def test_g_minus_1a_cannot_authorize_d0_or_model_fitting(tmp_path: Path) -> None:
    config = _config()
    config["gate_definition"]["may_authorize_d0"] = True

    with pytest.raises(MODULE.GMinus1AError, match="unsafe gate permission"):
        MODULE.verify(_write(tmp_path, config))


def test_verifier_is_metadata_only_and_imports_no_producer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "build_pivot_feedback_cutoff_primitives" not in source
    assert "audit_openalex_" not in source
    assert "pyarrow" not in source
    assert "source_payloads_opened" in source


def test_receipt_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(MODULE.GMinus1AError, match="output already exists"):
        MODULE._write_new(output, b"replacement\n")
    assert output.read_text(encoding="utf-8") == "existing"
