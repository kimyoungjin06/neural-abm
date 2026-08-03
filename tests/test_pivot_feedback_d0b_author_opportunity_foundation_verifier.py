from __future__ import annotations

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
SCRIPT = (
    ROOT
    / "scripts/verify_pivot_feedback_d0b_author_opportunity_foundation.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0b_author_opportunity_foundation_verifier_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _output_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "output"
    root.mkdir(mode=0o700)
    selected = pa.Table.from_pylist(
        [
            {
                "episode_index": 1,
                "field_id": 17,
                "topic_id": 14430,
                "entry_year": 2003,
                "work_numeric_id": 100,
                "publication_year": 1995,
                "declared_authors_count": 1,
                "incumbency_qualifying_work": True,
            }
        ],
        schema=MODULE.WORK_OUTPUT_SCHEMA,
    )
    author = pa.Table.from_arrays(
        [
            pa.array(
                [1],
                type=field.type,
            )
            if not pa.types.is_boolean(field.type)
            else pa.array([False], type=field.type)
            for field in MODULE.AUTHOR_OUTPUT_SCHEMA
        ],
        schema=MODULE.AUTHOR_OUTPUT_SCHEMA,
    )
    for name, table in (
        ("episode_selected_predecision_works.parquet", selected),
        ("pre_entry_author_measurement_base.parquet", author),
    ):
        path = root / name
        pq.write_table(table, path)
        os.chmod(path, 0o600)
    report = {
        "schema_version": MODULE.REPORT_SCHEMA,
        "status": MODULE.OUTPUT_STATUS,
        "scope": {
            "d0b_author_foundation_built": True,
            "d0b_complete": False,
            "d0_complete": False,
            "entry_or_post_entry_target_works_read": False,
            "persistence_rows_read": False,
            "realized_citation_rows_read": False,
            "predictive_scores_computed": False,
            "threshold_tuning_performed": False,
            "final_candidate_eligibility_established": False,
            "model_fitting_performed": False,
            "g4_rows_read": False,
            "gpu_used": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": {
            "remaining_d0b_measurement_components": "AUTHORIZED",
            "discovery_outcome_opening": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
    }
    _write_json(root / "d0b_author_opportunity_foundation.json", report)
    markdown = root / "d0b_author_opportunity_foundation.md"
    markdown.write_text("# D0-B\n", encoding="utf-8")
    os.chmod(markdown, 0o600)
    artifacts = []
    for name in sorted(
        MODULE.EXPECTED_FILES
        - {"d0b_author_opportunity_foundation.outputs.json"}
    ):
        path = root / name
        item = {
            "relative_path": name,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        if name.endswith(".parquet"):
            parquet = pq.ParquetFile(path)
            item["rows"] = parquet.metadata.num_rows
            item["row_groups"] = parquet.metadata.num_row_groups
        artifacts.append(item)
    receipt = {
        "schema_version": MODULE.RECEIPT_SCHEMA,
        "status": MODULE.OUTPUT_STATUS,
        "artifact_count": 4,
        "artifacts": artifacts,
        "artifact_tree_sha256": hashlib.sha256(
            _canonical_bytes(
                [
                    (item["relative_path"], item["bytes"], item["sha256"])
                    for item in artifacts
                ]
            )
        ).hexdigest(),
        "d0b_author_foundation_built": True,
        "d0b_complete": False,
        "d0_complete": False,
        "discovery_outcome_opening_authorized": False,
        "g0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "gpu_used": False,
        "receipt_written_last": True,
    }
    _write_json(root / "d0b_author_opportunity_foundation.outputs.json", receipt)
    return root


def _event(
    author: int,
    work: int,
    year: int,
    disposition: int,
    team: int,
    field: int,
) -> dict:
    return {
        "author_numeric_id": author,
        "work_numeric_id": work,
        "publication_year": year,
        "temporal_disposition": disposition,
        "declared_authors_count": team,
        "primary_field_id": field,
    }


def test_arrow_groupby_replay_has_exact_caps_windows_and_flags() -> None:
    rows = [
        _event(4096, 100, 1995, 2, 5, 17),
        _event(4096, 101, 1998, 2, 20, 99),
        _event(4096, 102, 2000, 2, 5, 17),
        _event(4096, 103, 2001, 2, 25, 17),
        _event(4096, 104, 2002, 2, 60, 17),
        _event(4096, 105, 2003, 2, 5, 17),
        _event(12288, 300, 2002, 2, 150, 17),
    ]
    table = pa.Table.from_pylist(rows, schema=MODULE.EVENT_INPUT_SCHEMA)
    episode = {
        "episode_index": 1,
        "field_id": 17,
        "topic_id": 14430,
        "entry_year": 2003,
    }

    result = MODULE._independent_author_rows(
        table,
        episode,
        pa.array([100], type=pa.uint64()),
    )

    assert result.schema == MODULE.AUTHOR_OUTPUT_SCHEMA
    assert result["author_numeric_id"].to_pylist() == [4096, 12288]
    by_author = {
        row["author_numeric_id"]: row
        for row in result.to_pylist()
    }
    first = by_author[4096]
    assert [
        first[f"pre_entry_all_field_work_count_{suffix}"]
        for suffix in ("cap_10", "cap_50", "cap_100", "all")
    ] == [2, 4, 5, 5]
    assert [
        first[f"portfolio_window_focal_field_work_count_{suffix}"]
        for suffix in ("cap_10", "cap_50", "cap_100", "all")
    ] == [1, 2, 3, 3]
    assert first["first_target_topic_year"] == 1995
    assert first["observed_no_prior_target_topic"] is False
    assert first["incumbent_before_signal_window"] is True
    second = by_author[12288]
    assert second["portfolio_window_focal_field_work_count_cap_100"] == 0
    assert second["portfolio_window_focal_field_work_count_all"] == 1
    assert second["first_target_topic_year"] is None


def test_output_identity_rehashes_artifacts_and_keeps_gates_closed(
    tmp_path: Path,
) -> None:
    root = _output_fixture(tmp_path)

    receipt, report, artifacts = MODULE._validate_output_identity(root)

    assert receipt["d0_complete"] is False
    assert report["scope"]["final_candidate_eligibility_established"] is False
    assert len(artifacts) == 4


def test_external_report_is_private_create_only(tmp_path: Path) -> None:
    root = _output_fixture(tmp_path)
    path = tmp_path / "verification.json"
    payload = {"status": MODULE.VERIFY_STATUS}

    MODULE._write_report(path, payload, root=root)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        MODULE._write_report(path, payload, root=root)


def test_verifier_does_not_import_producer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert (
        "import build_pivot_feedback_d0b_author_opportunity_foundation"
        not in source
    )
    assert "PYARROW_GROUP_BY_INDEPENDENT_OF_NUMPY_PRODUCER" in source
    assert "FORBIDDEN_COLUMNS" in source
