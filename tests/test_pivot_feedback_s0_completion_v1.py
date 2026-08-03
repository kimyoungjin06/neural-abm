from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_s0_completion_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_s0_completion_v1_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _write_artifact(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    os.chmod(path, 0o600)


def _receipt(
    root: Path,
    *,
    contract,
    extra: dict,
) -> Path:
    artifacts = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name == contract["receipt"]:
            continue
        artifacts.append(
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    receipt = {
        "schema_version": contract["receipt_schema"],
        "status": contract["receipt_status"],
        "artifact_count": len(artifacts),
        "artifact_tree_sha256": hashlib.sha256(
            _canonical_bytes(artifacts)
        ).hexdigest(),
        "artifacts": artifacts,
        "s0_gate_pass": False,
        "d0_authorized": False,
        **extra,
    }
    path = root / contract["receipt"]
    _write_json(path, receipt)
    return path


def _verification(
    path: Path,
    *,
    root: Path,
    receipt: Path,
    contract,
    extra: dict,
) -> None:
    value = {
        "schema_version": contract["verification_schema"],
        "status": contract["verification_status"],
        "output_root_basename": root.name,
        "output_receipt_sha256": _sha(receipt),
        "s0_gate_pass": False,
        "d0_authorized": False,
        "model_fitting_authorized": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "verifier_sha256": _sha(ROOT / contract["verifier"]),
        **extra,
    }
    _write_json(path, value)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    core = tmp_path / "core"
    core.mkdir(mode=0o700)
    _write_artifact(core / "attention_trajectory_core.parquet", b"attention")
    _write_artifact(core / "field_year_core.parquet", b"core")
    _write_json(
        core / "s0_core_inventory.json",
        {
            "scope": {"gpu_used": False},
            "provenance": {"config_sha256": "a" * 64},
            "work_accounting": {"rows": 100},
            "author_event_accounting": {"rows": 200},
        },
    )
    _write_artifact(core / "s0_core_inventory.md", b"core\n")
    core_receipt = _receipt(
        core,
        contract=MODULE.CORE,
        extra={"core_component_pass": True},
    )
    core_verification = tmp_path / "core.verification.json"
    _verification(
        core_verification,
        root=core,
        receipt=core_receipt,
        contract=MODULE.CORE,
        extra={
            "field_year_verification": {
                "rows": 10,
                "field_count": 26,
                "null_year_cells": 2,
            },
            "attention_replay": {"rows": 9},
        },
    )

    orcid = tmp_path / "orcid"
    orcid.mkdir(mode=0o700)
    _write_json(
        orcid / "authorship_orcid_affiliation_report.json",
        {
            "scope": {"gpu_used": False},
            "provenance": {"config_sha256": "b" * 64},
            "source_accounting": {
                "raw_rows": 1_000,
                "retained_rows": 999,
                "nonnull_orcid_rows": 400,
                "syntax_valid_orcid_rows": 390,
                "checksum_valid_orcid_rows": 380,
                "author_work_affiliation_rows": 500,
                "author_work_institution_rows": 300,
            },
            "orcid_token_diagnostics": {
                "stable_single_valid_orcid_author_tokens": 200,
                "multiple_valid_orcid_author_tokens": 2,
                "shared_valid_orcids_across_stable_tokens": 3,
            },
            "institution_diagnostics": {"distinct_institution_tokens": 50},
        },
    )
    _write_artifact(
        orcid / "authorship_orcid_affiliation_report.md",
        b"orcid\n",
    )
    _write_artifact(
        orcid / "field_year_orcid_affiliation.parquet",
        b"orcid-table",
    )
    orcid_receipt = _receipt(
        orcid,
        contract=MODULE.ORCID,
        extra={"full_scope": True, "component_pass": True},
    )
    orcid_verification = tmp_path / "orcid.verification.json"
    _verification(
        orcid_verification,
        root=orcid,
        receipt=orcid_receipt,
        contract=MODULE.ORCID,
        extra={
            "full_scope": True,
            "component_pass": True,
            "field_year_verification": {
                "rows": 8,
                "field_count": 26,
                "null_year_cells": 1,
            },
        },
    )

    graph = tmp_path / "graph"
    graph.mkdir(mode=0o700)
    _write_json(
        graph / "coauthor_graph_profile_report.json",
        {
            "scope": {"gpu_used": False},
            "core_binding": {
                "receipt_sha256": _sha(core_receipt),
                "field_year_rows": 10,
            },
            "materialization_accounting": {"source_rows": 200},
            "field_year_inventory": {
                "profile_metric_sums": {
                    "graph_node_tokens": {"10": 10, "50": 20, "100": 30},
                    "distinct_undirected_edges": {"10": 5, "50": 10, "100": 15},
                    "connected_components": {"10": 5, "50": 10, "100": 15},
                    "largest_component_nodes": {"10": 2, "50": 3, "100": 4},
                }
            },
        },
    )
    _write_artifact(graph / "coauthor_graph_profile_report.md", b"graph\n")
    _write_artifact(graph / "field_year_coauthor_graph.parquet", b"graph-table")
    graph_receipt = _receipt(
        graph,
        contract=MODULE.GRAPH,
        extra={"full_scope": True, "component_pass": True},
    )
    graph_verification = tmp_path / "graph.verification.json"
    _verification(
        graph_verification,
        root=graph,
        receipt=graph_receipt,
        contract=MODULE.GRAPH,
        extra={
            "full_scope": True,
            "component_pass": True,
            "graph_verification": {
                "rows": 30,
                "core_cells": 10,
                "all_degree_sums_close": True,
                "all_component_bounds_hold": True,
                "all_largest_component_shares_replayed": True,
            },
        },
    )
    return {
        "core_root": core,
        "core_verification": core_verification,
        "orcid_root": orcid,
        "orcid_verification": orcid_verification,
        "graph_root": graph,
        "graph_verification": graph_verification,
    }


def test_three_full_components_pass_and_authorize_only_d0(tmp_path: Path) -> None:
    result = MODULE.verify(**_fixture(tmp_path))

    assert result["status"] == MODULE.PASS_STATUS
    assert result["scope"]["s0_gate_pass"] is True
    assert result["scope"]["d0_discovery_pilot_authorized"] is True
    assert result["scope"]["g0_measurement_freeze_authorized"] is False
    assert result["scope"]["model_fitting_authorized"] is False
    assert result["authorization"]["d0_discovery_only_pre2013"] == "AUTHORIZED"
    assert result["next_required_implementation"]["gate"] == "D0"


def test_changed_artifact_after_verification_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["orcid_root"] / "field_year_orcid_affiliation.parquet").write_bytes(
        b"changed"
    )

    with pytest.raises(MODULE.S0CompletionError, match="artifact size mismatch"):
        MODULE.verify(**paths)


def test_pilot_component_cannot_complete_s0(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt_path = (
        paths["orcid_root"]
        / "authorship_orcid_affiliation_report.outputs.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = (
        "PILOT_S0_AUTHORSHIP_ORCID_AND_AFFILIATION_NONPROMOTABLE"
    )
    receipt["full_scope"] = False
    receipt["component_pass"] = False
    _write_json(receipt_path, receipt)

    with pytest.raises(MODULE.S0CompletionError, match="receipt status mismatch"):
        MODULE.verify(**paths)


def test_threshold_dependent_metric_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["orcid_root"]
    report_path = root / "authorship_orcid_affiliation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["entry_effect_size"] = 0.1
    _write_json(report_path, report)
    receipt_path = _receipt(
        root,
        contract=MODULE.ORCID,
        extra={"full_scope": True, "component_pass": True},
    )
    _verification(
        paths["orcid_verification"],
        root=root,
        receipt=receipt_path,
        contract=MODULE.ORCID,
        extra={
            "full_scope": True,
            "component_pass": True,
            "field_year_verification": {
                "rows": 8,
                "field_count": 26,
                "null_year_cells": 1,
            },
        },
    )

    with pytest.raises(
        MODULE.S0CompletionError,
        match="threshold-dependent metric",
    ):
        MODULE.verify(**paths)


def test_cli_report_is_new_and_compact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report = tmp_path / "summary.json"
    arguments = []
    for name, path in paths.items():
        arguments.extend([f"--{name.replace('_', '-')}", str(path)])
    arguments.extend(["--report", str(report)])

    assert MODULE.main(arguments) == 0
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["status"] == MODULE.PASS_STATUS
    assert stat.S_IMODE(report.stat().st_mode) == 0o644
    assert "/tmp/" not in report.read_text(encoding="utf-8")
    assert MODULE.main(arguments) == 2


def test_completion_verifier_does_not_import_producers() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "import build_pivot_feedback" not in source
    assert "FORBIDDEN_METRIC_KEYS" in source
    assert "D0_DISCOVERY_ONLY_AUTHORIZED" in source
