from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_structural_pilot_freeze.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_structural_pilot_freeze_verifier_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _output_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "output"
    root.mkdir(mode=0o700)
    parquet_specs = {
        "field_structural_ranking.parquet": MODULE.FIELD_SCHEMA,
        "selected_field_topic_year_attention.parquet": MODULE.TOPIC_YEAR_SCHEMA,
        "episode_candidates.parquet": MODULE.CANDIDATE_SCHEMA,
    }
    for name, schema_version in parquet_specs.items():
        table = pa.table({"value": pa.array([1], type=pa.uint8())})
        table = table.replace_schema_metadata(
            {b"schema_version": schema_version.encode()}
        )
        path = root / name
        pq.write_table(table, path)
        os.chmod(path, 0o600)
    report = {
        "schema_version": MODULE.REPORT_SCHEMA,
        "status": MODULE.OUTPUT_STATUS,
        "scope": {
            "d0a_structural_freeze_complete": True,
            "d0_complete": False,
            "entry_year_end_inclusive": 2012,
            "outcomes_read": False,
            "author_entry_rows_read": False,
            "persistence_rows_read": False,
            "citations_realized_read": False,
            "predictive_scores_computed": False,
            "threshold_tuning_performed": False,
            "model_fitting_performed": False,
            "g4_rows_read": False,
            "gpu_used": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
    }
    _write_json(root / "d0_structural_pilot_freeze.json", report)
    markdown = root / "d0_structural_pilot_freeze.md"
    markdown.write_text("# D0-A\n", encoding="utf-8")
    os.chmod(markdown, 0o600)
    artifacts = []
    for name in sorted(
        MODULE.EXPECTED_FILES - {"d0_structural_pilot_freeze.outputs.json"}
    ):
        path = root / name
        item = {
            "relative_path": name,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        if name.endswith(".parquet"):
            item["rows"] = 1
        artifacts.append(item)
    receipt = {
        "schema_version": MODULE.RECEIPT_SCHEMA,
        "status": MODULE.OUTPUT_STATUS,
        "artifact_count": 5,
        "artifacts": artifacts,
        "artifact_tree_sha256": hashlib.sha256(
            _canonical_bytes(
                [
                    (item["relative_path"], item["bytes"], item["sha256"])
                    for item in artifacts
                ]
            )
        ).hexdigest(),
        "d0a_structural_freeze_complete": True,
        "d0_complete": False,
        "d0b_outcome_blind_author_panel_authorized": True,
        "g0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "gpu_used": False,
    }
    _write_json(root / "d0_structural_pilot_freeze.outputs.json", receipt)
    return root


def test_output_identity_rehashes_all_artifacts_and_preserves_gate_boundary(
    tmp_path: Path,
):
    root = _output_fixture(tmp_path)

    receipt, report, artifacts = MODULE._validate_output_identity(root)

    assert receipt["d0_complete"] is False
    assert report["scope"]["outcomes_read"] is False
    assert len(artifacts) == 5


def test_rebound_report_cannot_claim_that_d0_is_complete(tmp_path: Path):
    root = _output_fixture(tmp_path)
    report_path = root / "d0_structural_pilot_freeze.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["scope"]["d0_complete"] = True
    _write_json(report_path, report)
    receipt_path = root / "d0_structural_pilot_freeze.outputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for item in receipt["artifacts"]:
        if item["relative_path"] == report_path.name:
            item["bytes"] = report_path.stat().st_size
            item["sha256"] = _sha(report_path)
    receipt["artifact_tree_sha256"] = hashlib.sha256(
        _canonical_bytes(
            [
                (item["relative_path"], item["bytes"], item["sha256"])
                for item in receipt["artifacts"]
            ]
        )
    ).hexdigest()
    _write_json(receipt_path, receipt)

    with pytest.raises(
        MODULE.D0StructuralVerificationError,
        match="unsafe report scope: d0_complete",
    ):
        MODULE._validate_output_identity(root)


def test_external_report_is_private_create_only(tmp_path: Path):
    root = _output_fixture(tmp_path)
    report = tmp_path / "verification.json"
    payload = {"status": MODULE.VERIFY_STATUS}

    MODULE._write_report(report, payload, root=root)

    assert json.loads(report.read_text(encoding="utf-8")) == payload
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        MODULE._write_report(report, payload, root=root)


def test_exact_episode_order_survives_float_collision():
    lower = {
        "standardized_gain_numerator": 2**53,
        "standardized_gain_denominator": 2**53,
        "minimum_baseline_recent_count": 1,
        "absolute_gain": 1,
        "topic_id": 2,
        "entry_year": 2000,
    }
    higher = {
        **lower,
        "standardized_gain_numerator": 2**53 + 1,
        "topic_id": 3,
    }
    assert float(Fraction(2**53, 2**53)) == float(
        Fraction(2**53 + 1, 2**53)
    )
    assert MODULE._episode_key(higher) > MODULE._episode_key(lower)


def test_verifier_does_not_import_builder():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "import build_pivot_feedback_d0_structural_pilot_freeze" not in source
    assert "FORBIDDEN_COLUMNS" in source
    assert "INTEGER_CROSS_MULTIPLICATION" not in source
