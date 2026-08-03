from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_s0_coauthor_graph_profile.py"
OUTPUT = Path("/ssd/pivot-feedback-s0-coauthor-graph-profile-v1-pilot4-20260724")


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_s0_coauthor_graph_profile_verifier_test_module", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(path, 0o600)


def _copy_output(tmp_path: Path) -> Path:
    target = tmp_path / "output"
    shutil.copytree(OUTPUT, target)
    return target


def _rebind(root: Path) -> None:
    report_path = root / "coauthor_graph_profile_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provenance"].pop("result_fingerprint_sha256", None)
    report["provenance"]["result_fingerprint_sha256"] = hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    _write_json(report_path, report)
    receipt_path = root / "coauthor_graph_profile_report.outputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifacts = []
    for relative in sorted(
        MODULE.EXPECTED_FILES - {"coauthor_graph_profile_report.outputs.json"}
    ):
        path = root / relative
        artifacts.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    receipt["artifacts"] = artifacts
    receipt["artifact_count"] = len(artifacts)
    receipt["artifact_tree_sha256"] = hashlib.sha256(
        _canonical_bytes(artifacts)
    ).hexdigest()
    receipt["result_fingerprint_sha256"] = report["provenance"][
        "result_fingerprint_sha256"
    ]
    _write_json(receipt_path, receipt)


def _refresh_parquet_artifact(root: Path) -> None:
    path = root / "field_year_coauthor_graph.parquet"
    report_path = root / "coauthor_graph_profile_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["field_year_inventory"]["artifact"].update(
        {"bytes": path.stat().st_size, "sha256": _sha(path)}
    )
    _write_json(report_path, report)
    _rebind(root)


def test_exact_real_pilot_passes_independent_verification() -> None:
    result = MODULE.verify(OUTPUT)

    assert result["status"] == MODULE.PILOT_VERIFY_STATUS
    assert result["full_scope"] is False
    assert result["component_pass"] is False
    assert result["graph_verification"]["rows"] == 55632
    assert result["graph_verification"]["all_degree_sums_close"] is True
    assert result["graph_verification"]["all_component_bounds_hold"] is True
    assert result["s0_gate_pass"] is False
    assert result["d0_authorized"] is False


def test_rebound_report_cannot_promote_s0(tmp_path: Path) -> None:
    root = _copy_output(tmp_path)
    path = root / "coauthor_graph_profile_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["scope"]["s0_gate_pass"] = True
    _write_json(path, report)
    _rebind(root)

    with pytest.raises(MODULE.GraphVerificationError, match="improperly passes S0"):
        MODULE.verify(root)


def test_rebound_degree_sum_break_is_rejected(tmp_path: Path) -> None:
    root = _copy_output(tmp_path)
    path = root / "field_year_coauthor_graph.parquet"
    table = pq.read_table(path)
    edges = table["distinct_undirected_edges"].to_pylist()
    index = next(i for i, value in enumerate(edges) if value > 0)
    values = table["degree_sum"].to_pylist()
    values[index] += 1
    column = table.column_names.index("degree_sum")
    table = table.set_column(
        column,
        table.schema.field(column),
        pa.array(values, type=pa.uint64()),
    )
    table = table.replace_schema_metadata(
        {b"schema_version": MODULE.FIELD_YEAR_SCHEMA.encode()}
    )
    pq.write_table(table, path)
    os.chmod(path, 0o600)
    _refresh_parquet_artifact(root)

    with pytest.raises(
        MODULE.GraphVerificationError,
        match="degree sum does not equal twice edges",
    ):
        MODULE.verify(root)


def test_rebound_largest_component_share_break_is_rejected(
    tmp_path: Path,
) -> None:
    root = _copy_output(tmp_path)
    path = root / "field_year_coauthor_graph.parquet"
    table = pq.read_table(path)
    nodes = table["graph_node_tokens"].to_pylist()
    index = next(i for i, value in enumerate(nodes) if value > 0)
    shares = table["largest_component_share"].to_pylist()
    shares[index] += 0.01
    column = table.column_names.index("largest_component_share")
    table = table.set_column(
        column,
        table.schema.field(column),
        pa.array(shares, type=pa.float64()),
    )
    table = table.replace_schema_metadata(
        {b"schema_version": MODULE.FIELD_YEAR_SCHEMA.encode()}
    )
    pq.write_table(table, path)
    os.chmod(path, 0o600)
    _refresh_parquet_artifact(root)

    with pytest.raises(
        MODULE.GraphVerificationError, match="largest component share mismatch"
    ):
        MODULE.verify(root)


def test_rebound_materialized_node_margin_break_is_rejected(
    tmp_path: Path,
) -> None:
    root = _copy_output(tmp_path)
    path = root / "coauthor_graph_profile_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["materialization_accounting"][
        "materialized_node_tokens_cap_10"
    ] += 1
    _write_json(path, report)
    _rebind(root)

    with pytest.raises(
        MODULE.GraphVerificationError,
        match="materialized node-token closure mismatch",
    ):
        MODULE.verify(root)


def test_compact_output_contains_no_raw_graph_ids() -> None:
    table = pq.read_table(OUTPUT / "field_year_coauthor_graph.parquet")

    assert MODULE.FORBIDDEN_COLUMNS.isdisjoint(table.column_names)
    assert set(table["declared_team_cap"].to_pylist()) == {10, 50, 100}
    null_year_rows = [
        row
        for row in table.to_pylist()
        if row["publication_year"] is None
    ]
    assert len(null_year_rows) == 26 * 3
    assert sum(row["graph_node_tokens"] for row in null_year_rows) > 0


def test_cli_writes_new_private_external_report(tmp_path: Path) -> None:
    root = _copy_output(tmp_path)
    report = tmp_path / "verification.json"

    assert (
        MODULE.main(
            [
                "--output-root",
                str(root),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == (
        MODULE.PILOT_VERIFY_STATUS
    )
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert (
        MODULE.main(
            [
                "--output-root",
                str(root),
                "--report",
                str(report),
            ]
        )
        == 2
    )


def test_verifier_does_not_import_builder() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "import build_pivot_feedback_s0_coauthor_graph_profile" not in source
    assert "FORBIDDEN_COLUMNS" in source
    assert "degree sum does not equal twice edges" in source
