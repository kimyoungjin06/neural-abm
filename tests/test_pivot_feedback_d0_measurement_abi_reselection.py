from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_d0_measurement_abi_reselection.py"


def _load():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_measurement_abi_reselection_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _write_event(path: Path, rows: list[dict]) -> Path:
    pq.write_table(
        pa.Table.from_pylist(rows, schema=MODULE.EVENT_INPUT_SCHEMA),
        path,
    )
    return path


def _write_work(path: Path, rows: list[dict]) -> Path:
    pq.write_table(
        pa.Table.from_pylist(rows, schema=MODULE.WORK_INPUT_SCHEMA),
        path,
    )
    return path


def test_event_index_deduplicates_work_presence_and_reports_pair_duplicates(
    tmp_path: Path,
) -> None:
    path = _write_event(
        tmp_path / "part-0000.parquet",
        [
            {
                "author_numeric_id": 4096,
                "work_numeric_id": 8192,
                "publication_year": 2000,
                "temporal_disposition": 2,
                "declared_authors_count": 2,
                "primary_field_id": 17,
            },
            {
                "author_numeric_id": 4096,
                "work_numeric_id": 8192,
                "publication_year": 2000,
                "temporal_disposition": 2,
                "declared_authors_count": 2,
                "primary_field_id": 17,
            },
            {
                "author_numeric_id": 8192,
                "work_numeric_id": 8192,
                "publication_year": 2000,
                "temporal_disposition": 2,
                "declared_authors_count": 2,
                "primary_field_id": 17,
            },
        ],
    )

    result = MODULE._scan_event_partition(
        (0, path),
        selected_fields=np.asarray([17, 27], dtype=np.uint8),
        allowed_dispositions=np.asarray([2, 3], dtype=np.uint8),
        latest_year=2011,
    )

    assert result["selected_event_rows"] == 3
    assert result["duplicate_author_work_event_rows"] == 1
    assert len(result["work_records"]) == 3


def test_work_abi_excludes_unlinked_and_retracted_but_retains_null_flag(
    tmp_path: Path,
) -> None:
    path = _write_work(
        tmp_path / "part-0000.parquet",
        [
            {
                "work_id_numeric": 4096,
                "publication_day": None,
                "publication_year": 2000,
                "temporal_disposition": 2,
                "work_type_code": 1,
                "is_paratext": False,
                "is_retracted": None,
                "declared_authors_count": 1,
                "primary_topic_id": 100,
                "primary_field_id": 17,
                "primary_domain_id": 3,
            },
            {
                "work_id_numeric": 8192,
                "publication_day": None,
                "publication_year": 2000,
                "temporal_disposition": 2,
                "work_type_code": 1,
                "is_paratext": False,
                "is_retracted": False,
                "declared_authors_count": 1,
                "primary_topic_id": 100,
                "primary_field_id": 17,
                "primary_domain_id": 3,
            },
            {
                "work_id_numeric": 12288,
                "publication_day": None,
                "publication_year": 2000,
                "temporal_disposition": 2,
                "work_type_code": 1,
                "is_paratext": False,
                "is_retracted": True,
                "declared_authors_count": 1,
                "primary_topic_id": 100,
                "primary_field_id": 17,
                "primary_domain_id": 3,
            },
            {
                "work_id_numeric": 16384,
                "publication_day": None,
                "publication_year": 2000,
                "temporal_disposition": 2,
                "work_type_code": 13,
                "is_paratext": False,
                "is_retracted": False,
                "declared_authors_count": 1,
                "primary_topic_id": 100,
                "primary_field_id": 17,
                "primary_domain_id": 3,
            },
        ],
    )
    valid_ids = np.asarray([4096], dtype=np.uint64)
    metadata = {
        "publication_year": np.asarray([2000], dtype=np.uint16),
        "field_id": np.asarray([17], dtype=np.uint8),
        "temporal_disposition": np.asarray([2], dtype=np.uint8),
    }

    result = MODULE._scan_work_partition(
        (0, path),
        selected_fields=np.asarray([17, 27], dtype=np.uint8),
        allowed_dispositions=np.asarray([2, 3], dtype=np.uint8),
        included_types=np.asarray([1, 2, 3, 4, 7, 16, 18, 21], dtype=np.uint8),
        latest_year=2011,
        valid_work_ids=valid_ids,
        event_metadata=metadata,
    )

    assert result["scope_rows"] == 4
    assert result["included_type_rows"] == 3
    assert result["retracted_true_rows"] == 1
    assert result["eligible_prelink_rows"] == 2
    assert result["linked_rows"] == 1
    assert result["zero_valid_author_token_rows"] == 1
    assert result["is_retracted_null_eligible_rows"] == 1
    assert result["linked"]["work_numeric_id"].tolist() == [4096]
    assert result["linked"]["is_retracted_null"].tolist() == [1]


def test_non_paratext_type_with_paratext_boolean_is_excluded(
    tmp_path: Path,
) -> None:
    path = _write_work(
        tmp_path / "part-0000.parquet",
        [
            {
                "work_id_numeric": 4096,
                "publication_day": None,
                "publication_year": 2000,
                "temporal_disposition": 2,
                "work_type_code": 1,
                "is_paratext": True,
                "is_retracted": False,
                "declared_authors_count": 1,
                "primary_topic_id": 100,
                "primary_field_id": 17,
                "primary_domain_id": 3,
            }
        ],
    )

    result = MODULE._scan_work_partition(
        (0, path),
        selected_fields=np.asarray([17, 27], dtype=np.uint8),
        allowed_dispositions=np.asarray([2, 3], dtype=np.uint8),
        included_types=np.asarray([1], dtype=np.uint8),
        latest_year=2011,
        valid_work_ids=np.asarray([4096], dtype=np.uint64),
        event_metadata={
            "publication_year": np.asarray([2000], dtype=np.uint16),
            "field_id": np.asarray([17], dtype=np.uint8),
            "temporal_disposition": np.asarray([2], dtype=np.uint8),
        },
    )

    assert result["paratext_type_boolean_mismatch_rows"] == 1
    assert result["scope_non_paratext_type_boolean_true_rows"] == 1
    assert result["paratext_true_excluded_rows"] == 1
    assert result["eligible_prelink_rows"] == 0
    assert result["linked_rows"] == 0


def test_candidate_ranking_uses_author_linked_counts() -> None:
    rows = []
    for topic_id, baseline, recent, subfield in (
        (100, [1, 1, 1], [10, 10, 10], 1000),
        (200, [2, 2, 2], [12, 12, 12], 2000),
    ):
        for year, linked in zip(range(1994, 2000), baseline + recent, strict=True):
            rows.append(
                {
                    "field_id": 17,
                    "topic_id": topic_id,
                    "topic_name": f"T{topic_id}",
                    "subfield_id": subfield,
                    "subfield_name": f"S{subfield}",
                    "publication_year": year,
                    "eligible_prelink_works": linked + 5,
                    "author_linked_works": linked,
                    "zero_valid_author_token_works": 5,
                    "is_retracted_null_eligible_works": 0,
                    "is_retracted_null_author_linked_works": 0,
                }
            )
    table = pa.Table.from_pylist(
        rows,
        schema=MODULE._topic_year_table(
            {},
            taxonomy={},
        ).schema,
    )
    contract = {
        "baseline_offsets": [-6, -5, -4],
        "recent_offsets": [-3, -2, -1],
        "entry_year_end_inclusive": 2009,
        "pair_selection": {"minimum_entry_year_separation": 9},
    }

    with pytest.raises(
        MODULE.MeasurementABIReselectionError,
        match="no independent episode pair",
    ):
        MODULE._episode_candidates(
            table,
            selected_fields=[17],
            contract=contract,
        )


def test_producer_does_not_reference_entry_or_persistence_artifacts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "entry_effect_size" not in source
    assert "persistence_effect_size" not in source
    assert "g4_outcome" not in source


def test_temporary_cleanup_is_none_after_publish_and_rejects_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[Path] = []
    monkeypatch.setattr(MODULE.shutil, "rmtree", removed.append)
    output = tmp_path / "final"

    MODULE._cleanup_temporary_directory(None, output=output)

    with pytest.raises(
        MODULE.MeasurementABIReselectionError,
        match="relative temporary path",
    ):
        MODULE._cleanup_temporary_directory(Path(), output=output)
    assert removed == []


def test_temporary_cleanup_removes_only_matching_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "final"
    temporary = tmp_path / ".final.partial"
    temporary.mkdir()
    removed: list[Path] = []
    monkeypatch.setattr(MODULE.shutil, "rmtree", removed.append)

    MODULE._cleanup_temporary_directory(temporary, output=output)

    assert removed == [temporary]

    with pytest.raises(
        MODULE.MeasurementABIReselectionError,
        match="expected temporary prefix",
    ):
        MODULE._cleanup_temporary_directory(tmp_path / "unrelated", output=output)
