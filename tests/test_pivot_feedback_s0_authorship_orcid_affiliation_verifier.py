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
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_pivot_feedback_s0_authorship_orcid_affiliation.py"
VERIFIER = (
    ROOT / "scripts/verify_pivot_feedback_s0_authorship_orcid_affiliation.py"
)
CONFIG = (
    ROOT
    / "experiments/configs/pivot_feedback_s0_authorship_orcid_affiliation_v1.yaml"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_s0_authorship_orcid_affiliation_verifier_test_module",
        VERIFIER,
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


def _fixture_table() -> pa.Table:
    rows = []
    for field_id in range(11, 37):
        row = {
            "field_id": field_id,
            "publication_year": 2020,
            **{metric: 0 for metric in MODULE.FIELD_YEAR_METRICS},
        }
        if field_id == 11:
            row.update(
                {
                    "raw_authorship_rows": 2,
                    "grammar_valid_author_id_rows": 2,
                    "nonnull_orcid_rows": 2,
                    "syntax_valid_orcid_rows": 2,
                    "checksum_valid_orcid_rows": 1,
                    "author_work_affiliation_rows": 1,
                    "author_work_institution_rows": 1,
                    "stable_orcid_author_work_pairs": 1,
                    "stable_orcid_author_tokens": 1,
                    "distinct_institutions": 1,
                }
            )
        rows.append(row)
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=True),
            *[
                pa.field(metric, pa.uint64(), nullable=False)
                for metric in MODULE.FIELD_YEAR_METRICS
            ],
        ],
        metadata={b"schema_version": MODULE.FIELD_YEAR_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _metric_sums(table: pa.Table) -> dict[str, int]:
    return {
        metric: int(sum(table[metric].to_pylist()))
        for metric in MODULE.FIELD_YEAR_METRICS
    }


def _make_pilot_output(tmp_path: Path) -> Path:
    root = tmp_path / "pilot-output"
    root.mkdir(mode=0o700)
    table = _fixture_table()
    parquet = root / "field_year_orcid_affiliation.parquet"
    pq.write_table(table, parquet)
    os.chmod(parquet, 0o600)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sums = _metric_sums(table)
    report = {
        "schema_version": MODULE.REPORT_SCHEMA,
        "status": MODULE.PILOT_STATUS,
        "scope": {
            "full_scope": False,
            "source_files": 1,
            "event_files": 1,
            "component_pass": False,
            "pilot_promotion_allowed": False,
            "s0_gate_pass": False,
            "d0_authorized": False,
            "model_fitting_authorized": False,
            "gpu_used": False,
        },
        "gate_parent": {},
        "s0_core_parent": {},
        "preflight": {"gpu_used": False},
        "source_binding": {
            "selected_source_files": 1,
            "selected_source_bytes": 7,
            "selected_checkpoints_verified": 1,
            "current_exact_stat_verified_for_every_source_file": True,
            "source_content_rehashed_in_this_run": False,
        },
        "work_lookup_binding": {
            "artifacts_rehashed": 4096,
            "rows": 492293206,
        },
        "source_accounting": {
            "raw_rows": 3,
            "w_minus_1_rows": 1,
            "retained_rows": 2,
            "nonnull_author_id_rows": 2,
            "null_author_id_rows": 1,
            "grammar_valid_author_id_rows": 2,
            "nonnull_orcid_rows": 2,
            "syntax_valid_orcid_rows": 2,
            "checksum_valid_orcid_rows": 1,
            "raw_affiliation_present_rows": 1,
            "institution_present_rows": 1,
            "author_work_affiliation_rows": 1,
            "author_work_institution_rows": 1,
            "grammar_valid_institution_occurrences": 1,
            "author_orcid_spool_rows": 1,
            "institution_spool_rows": 1,
            "unmatched_valid_work_rows": 0,
            **{
                f"classified_field_year_{metric}": sums[metric]
                for metric in MODULE.FIELD_YEAR_METRICS[:7]
            },
        },
        "orcid_token_diagnostics": {
            "distinct_author_orcid_pairs": 1,
            "author_tokens_with_valid_orcid": 1,
            "stable_single_valid_orcid_author_tokens": 1,
            "multiple_valid_orcid_author_tokens": 0,
            "distinct_valid_orcids_on_stable_tokens": 1,
            "shared_valid_orcids_across_stable_tokens": 0,
            "stable_author_tokens_on_shared_valid_orcids": 0,
            "maximum_stable_author_tokens_per_valid_orcid": 1,
        },
        "institution_diagnostics": {
            "distinct_institution_tokens": 1,
            "distinct_institution_field_year_pairs": 1,
        },
        "stable_event_diagnostics": {
            "event_rows_scanned": 1,
            "stable_orcid_author_work_pairs": 1,
            "stable_orcid_author_tokens_summed_within_field_year": 1,
            "event_artifacts_rehashed": 1,
        },
        "field_year_inventory": {
            "schema_version": MODULE.FIELD_YEAR_SCHEMA,
            "rows": len(table),
            "metric_sums": sums,
            "artifact": {
                "relative_path": parquet.name,
                "bytes": parquet.stat().st_size,
                "sha256": _sha(parquet),
                "rows": len(table),
            },
        },
        "remaining_s0_component": "DISTINCT_COAUTHOR_GRAPH_FIELD_YEAR",
        "gate_status": {
            "g_minus_1a": "PASS_BOUNDED",
            "s0_threshold_free_inventory": "PARTIAL_CORE_PASS_PILOT_ONLY",
            "d0_discovery_pilot": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "PENDING",
            "empirical_g1_signature": "NOT_RUN",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "provenance": {
            "config_relative_path": str(CONFIG.relative_to(ROOT)),
            "config_sha256": _sha(CONFIG),
            "runner_relative_path": str(BUILDER.relative_to(ROOT)),
            "runner_sha256": _sha(BUILDER),
        },
        "execution": {"scratch_removed_before_completion": True},
        "claim_boundary": config["claim_boundary"],
    }
    report["provenance"]["result_fingerprint_sha256"] = hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    _write_json(root / "authorship_orcid_affiliation_report.json", report)
    markdown = root / "authorship_orcid_affiliation_report.md"
    markdown.write_text("# fixture\n", encoding="utf-8")
    os.chmod(markdown, 0o600)
    _rebind_receipt(root)
    return root


def _rebind_receipt(root: Path) -> None:
    report_path = root / "authorship_orcid_affiliation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provenance"].pop("result_fingerprint_sha256", None)
    report["provenance"]["result_fingerprint_sha256"] = hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    _write_json(report_path, report)
    paths = sorted(
        path
        for path in root.iterdir()
        if path.name != "authorship_orcid_affiliation_report.outputs.json"
    )
    artifacts = [
        {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in paths
    ]
    receipt = {
        "schema_version": MODULE.OUTPUT_SCHEMA,
        "status": MODULE.PILOT_STATUS,
        "full_scope": False,
        "component_pass": False,
        "s0_gate_pass": False,
        "d0_authorized": False,
        "completion_contract": (
            "SCRATCH_REMOVED_RECEIPT_WRITTEN_LAST_ATOMIC_DIRECTORY_RENAME"
        ),
        "artifact_count": len(artifacts),
        "artifact_tree_sha256": hashlib.sha256(
            _canonical_bytes(artifacts)
        ).hexdigest(),
        "artifacts": artifacts,
        "result_fingerprint_sha256": report["provenance"][
            "result_fingerprint_sha256"
        ],
    }
    _write_json(
        root / "authorship_orcid_affiliation_report.outputs.json", receipt
    )


def test_exact_pilot_fixture_passes_without_promotion(tmp_path: Path) -> None:
    root = _make_pilot_output(tmp_path)

    result = MODULE.verify(root)

    assert result["status"] == MODULE.PILOT_VERIFY_STATUS
    assert result["full_scope"] is False
    assert result["component_pass"] is False
    assert result["field_year_verification"]["field_count"] == 26
    assert result["s0_gate_pass"] is False
    assert result["d0_authorized"] is False


def test_rebound_report_cannot_promote_s0(tmp_path: Path) -> None:
    root = _make_pilot_output(tmp_path)
    path = root / "authorship_orcid_affiliation_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["scope"]["s0_gate_pass"] = True
    _write_json(path, report)
    _rebind_receipt(root)

    with pytest.raises(MODULE.ProjectionVerificationError, match="improperly passes S0"):
        MODULE.verify(root)


def test_rebound_forbidden_metric_column_is_rejected(tmp_path: Path) -> None:
    root = _make_pilot_output(tmp_path)
    path = root / "field_year_orcid_affiliation.parquet"
    table = pq.read_table(path).append_column(
        "predictive_score", pa.array([0.0] * 26, type=pa.float64())
    )
    table = table.replace_schema_metadata(
        {b"schema_version": MODULE.FIELD_YEAR_SCHEMA.encode()}
    )
    pq.write_table(table, path)
    os.chmod(path, 0o600)
    report_path = root / "authorship_orcid_affiliation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["field_year_inventory"]["artifact"].update(
        {"bytes": path.stat().st_size, "sha256": _sha(path)}
    )
    _write_json(report_path, report)
    _rebind_receipt(root)

    with pytest.raises(
        MODULE.ProjectionVerificationError,
        match="physical schema mismatch|forbidden field-year column",
    ):
        MODULE.verify(root)


def test_rebound_field_year_cell_break_is_rejected(tmp_path: Path) -> None:
    root = _make_pilot_output(tmp_path)
    path = root / "field_year_orcid_affiliation.parquet"
    table = pq.read_table(path)
    values = table["checksum_valid_orcid_rows"].to_pylist()
    values[0] = 3
    index = table.column_names.index("checksum_valid_orcid_rows")
    table = table.set_column(
        index,
        table.schema.field(index),
        pa.array(values, type=pa.uint64()),
    )
    table = table.replace_schema_metadata(
        {b"schema_version": MODULE.FIELD_YEAR_SCHEMA.encode()}
    )
    pq.write_table(table, path)
    os.chmod(path, 0o600)
    report_path = root / "authorship_orcid_affiliation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["field_year_inventory"]["metric_sums"][
        "checksum_valid_orcid_rows"
    ] = 3
    report["source_accounting"][
        "classified_field_year_checksum_valid_orcid_rows"
    ] = 3
    report["field_year_inventory"]["artifact"].update(
        {"bytes": path.stat().st_size, "sha256": _sha(path)}
    )
    _write_json(report_path, report)
    _rebind_receipt(root)

    with pytest.raises(
        MODULE.ProjectionVerificationError, match="ORCID cell monotonicity"
    ):
        MODULE.verify(root)


def test_output_copy_preserves_required_private_modes(tmp_path: Path) -> None:
    root = _make_pilot_output(tmp_path)
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)

    assert MODULE.verify(copied)["status"] == MODULE.PILOT_VERIFY_STATUS


def test_cli_writes_new_private_external_report(tmp_path: Path) -> None:
    root = _make_pilot_output(tmp_path)
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
    source = VERIFIER.read_text(encoding="utf-8")

    assert "import build_pivot_feedback_s0_authorship_orcid_affiliation" not in source
    assert "FORBIDDEN_COLUMNS" in source
    assert "entry_effect_size" in source
