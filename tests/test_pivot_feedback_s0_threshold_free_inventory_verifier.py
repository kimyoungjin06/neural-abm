from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_s0_threshold_free_inventory.py"
OUTPUT = Path("/ssd/pivot-feedback-s0-threshold-free-core-v1-full-20260723")
SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_s0_threshold_free_core_full_20260723.summary.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_s0_threshold_free_verifier_test_module", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _copy_output(tmp_path: Path) -> Path:
    target = tmp_path / "output"
    shutil.copytree(OUTPUT, target)
    return target


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _rebind_receipt(root: Path) -> None:
    path = root / "s0_core_inventory.outputs.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    for item in receipt["artifacts"]:
        artifact = root / item["relative_path"]
        item["bytes"] = artifact.stat().st_size
        item["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt["artifact_tree_sha256"] = hashlib.sha256(
        _canonical_bytes(receipt["artifacts"])
    ).hexdigest()
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_exact_full_output_passes_independent_verification() -> None:
    result = MODULE.verify(OUTPUT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["artifacts_rehashed"] == 5
    assert result["field_year_verification"]["rows"] == 18544
    assert result["field_year_verification"]["field_count"] == 26
    assert result["field_year_verification"]["metric_sums"][
        "total_canonical_works"
    ] == 354145248
    assert result["field_year_verification"]["metric_sums"][
        "distinct_author_work_pairs"
    ] == 675475922
    assert result["attention_replay"]["all_trailing_5y_counts_replayed"] is True
    assert result["s0_gate_pass"] is False
    assert result["d0_authorized"] is False


def test_compact_repository_summary_preserves_gate_and_hash_boundaries() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["status"] == MODULE.STATUS
    assert summary["scope"]["core_component_pass"] is True
    assert summary["scope"]["s0_gate_pass"] is False
    assert summary["scope"]["d0_authorized"] is False
    assert summary["field_year_inventory"]["rows"] == 18544
    assert summary["field_year_inventory"]["field_year_parquet_sha256"] == (
        hashlib.sha256((OUTPUT / "field_year_core.parquet").read_bytes()).hexdigest()
    )
    assert summary["attention_trajectory_inventory"][
        "attention_parquet_sha256"
    ] == hashlib.sha256(
        (OUTPUT / "attention_trajectory_core.parquet").read_bytes()
    ).hexdigest()
    assert summary["independent_verification"]["verification_report_sha256"] == (
        hashlib.sha256(
            Path(
                "/ssd/pivot-feedback-s0-threshold-free-core-v1-full-20260723.verification.json"
            ).read_bytes()
        ).hexdigest()
    )
    assert set(summary["deferred_required_components"]) == {
        "orcid_field_year",
        "affiliation_field_year",
        "coauthor_graph_field_year",
    }


def test_report_cannot_promote_s0_gate(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    report_path = output / "s0_core_inventory.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["scope"]["s0_gate_pass"] = True
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rebind_receipt(output)

    with pytest.raises(MODULE.S0VerificationError, match="improperly passes S0"):
        MODULE.verify(output)


def test_attention_replay_detects_mutated_trailing_count(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    path = output / "attention_trajectory_core.parquet"
    table = pq.read_table(path)
    values = table["trailing_3y_canonical_works"].to_pylist()
    values[0] += 1
    index = table.column_names.index("trailing_3y_canonical_works")
    table = table.set_column(
        index,
        "trailing_3y_canonical_works",
        pa.array(values, type=pa.uint64()),
    )
    table = table.replace_schema_metadata(
        {b"schema_version": MODULE.ATTENTION_SCHEMA.encode()}
    )
    pq.write_table(table, path)
    _rebind_receipt(output)

    with pytest.raises(
        MODULE.S0VerificationError, match="trailing-three-year attention mismatch"
    ):
        MODULE.verify(output)


def test_forbidden_metric_column_is_rejected(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    path = output / "field_year_core.parquet"
    table = pq.read_table(path)
    table = table.append_column(
        "predictive_score", pa.array([0.0] * len(table), type=pa.float64())
    )
    table = table.replace_schema_metadata(
        {b"schema_version": MODULE.FIELD_YEAR_SCHEMA.encode()}
    )
    pq.write_table(table, path)
    _rebind_receipt(output)

    with pytest.raises(MODULE.S0VerificationError, match="forbidden field-year column"):
        MODULE.verify(output)


def test_deferred_component_cannot_be_silently_promoted(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    report_path = output / "s0_core_inventory.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["deferred_required_metrics"]["orcid_field_year"]["state"] = "PASS"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rebind_receipt(output)

    with pytest.raises(MODULE.S0VerificationError, match="improperly promoted"):
        MODULE.verify(output)


def test_verifier_does_not_import_builder() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "build_pivot_feedback_s0_threshold_free_inventory" not in source
    assert "entry_effect_size" in source
    assert "FORBIDDEN_COLUMNS" in source
