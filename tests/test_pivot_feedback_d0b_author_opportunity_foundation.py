from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_d0b_author_opportunity_foundation.py"
CONFIG = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_d0b_author_opportunity_foundation_v1.yaml"
)
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v4.yaml"
D0A = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_d0a_structural_pilot_freeze_full_20260724.summary.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0b_author_opportunity_foundation_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _episode() -> dict:
    return {
        "episode_index": 1,
        "field_id": 17,
        "topic_id": 14430,
        "entry_year": 2003,
    }


def test_config_is_parent_bound_and_keeps_outcomes_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == MODULE.CONFIG_SCHEMA
    assert config["repository_bindings"]["gate_state"]["sha256"] == hashlib.sha256(
        GATE.read_bytes()
    ).hexdigest()
    assert config["repository_bindings"]["d0a_freeze"]["sha256"] == hashlib.sha256(
        D0A.read_bytes()
    ).hexdigest()
    assert config["measurement_contract"]["allowed_temporal_dispositions"] == [2, 3]
    assert config["measurement_contract"]["author_token_is_natural_person"] is False
    assert config["output_contract"]["d0b_component_complete"] is False
    assert config["output_contract"]["d0_complete"] is False
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def test_selected_work_surface_excludes_entry_future_and_unusable_rows(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "work_id_numeric": 4096,
            "publication_day": 1,
            "publication_year": 1996,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 5,
            "primary_topic_id": 14430,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 8192,
            "publication_day": 1,
            "publication_year": 2002,
            "temporal_disposition": 3,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 25,
            "primary_topic_id": 14430,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 12288,
            "publication_day": 1,
            "publication_year": 2003,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 5,
            "primary_topic_id": 14430,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 16384,
            "publication_day": 1,
            "publication_year": 2000,
            "temporal_disposition": 1,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 5,
            "primary_topic_id": 14430,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
        {
            "work_id_numeric": 20480,
            "publication_day": 1,
            "publication_year": 2000,
            "temporal_disposition": 2,
            "work_type_code": 1,
            "is_paratext": False,
            "is_retracted": False,
            "declared_authors_count": 5,
            "primary_topic_id": 999,
            "primary_field_id": 17,
            "primary_domain_id": 3,
        },
    ]
    path = tmp_path / "part-0000.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=MODULE.WORK_INPUT_SCHEMA), path)

    table, work_ids, accounting = MODULE._selected_works([path], [_episode()])

    assert table["work_numeric_id"].to_pylist() == [4096, 8192]
    assert table["publication_year"].to_pylist() == [1996, 2002]
    assert table["incumbency_qualifying_work"].to_pylist() == [True, False]
    assert work_ids[1].tolist() == [4096, 8192]
    assert accounting["source_rows_scanned"] == 5
    assert accounting["selected_work_rows"] == 2
    assert accounting["by_episode"]["1"]["incumbency_qualifying_work_rows"] == 1


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


def test_author_foundation_has_exact_windows_monotone_caps_and_flags() -> None:
    rows = [
        _event(4096, 100, 1995, 2, 5, 17),
        _event(4096, 101, 1998, 2, 20, 99),
        _event(4096, 102, 2000, 2, 5, 17),
        _event(4096, 103, 2001, 2, 25, 17),
        _event(4096, 104, 2002, 2, 60, 17),
        _event(4096, 105, 2003, 2, 5, 17),
        _event(4096, 106, 2000, 1, 5, 17),
        _event(8192, 200, 1999, 2, 5, 17),
        _event(8192, 201, 2000, 2, 5, 99),
        _event(12288, 300, 2002, 2, 150, 17),
    ]
    table = pa.Table.from_pylist(rows, schema=MODULE.EVENT_INPUT_SCHEMA)

    batches, accounting = MODULE._author_batch(
        table,
        partition_index=0,
        episodes=[_episode()],
        selected_work_ids={1: np.array([100], dtype=np.uint64)},
    )

    result = batches[0]
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
        first[f"pre_entry_focal_field_work_count_{suffix}"]
        for suffix in ("cap_10", "cap_50", "cap_100", "all")
    ] == [2, 3, 4, 4]
    assert [
        first[f"portfolio_window_focal_field_work_count_{suffix}"]
        for suffix in ("cap_10", "cap_50", "cap_100", "all")
    ] == [1, 2, 3, 3]
    assert first["pre_entry_target_topic_work_count"] == 1
    assert first["pre_signal_target_topic_work_count"] == 1
    assert first["first_target_topic_year"] == 1995
    assert first["observed_no_prior_target_topic"] is False
    assert first["incumbent_before_signal_window"] is True

    second = by_author[12288]
    assert second["portfolio_window_focal_field_work_count_cap_100"] == 0
    assert second["portfolio_window_focal_field_work_count_all"] == 1
    assert second["first_target_topic_year"] is None
    assert second["observed_no_prior_target_topic"] is True
    assert second["incumbent_before_signal_window"] is False
    assert accounting[1]["author_rows"] == 2
    assert accounting[1]["observed_no_prior_target_topic_rows"] == 1
    assert accounting[1]["incumbent_before_signal_window_rows"] == 1

    forbidden = {
        "entry_effect_size",
        "persistence_effect_size",
        "predictive_score",
        "realized_citations",
        "final_candidate_eligibility",
    }
    assert forbidden.isdisjoint(result.schema.names)


def test_target_work_cross_surface_disagreement_fails_closed() -> None:
    table = pa.Table.from_pylist(
        [_event(4096, 100, 2003, 2, 1, 17)],
        schema=MODULE.EVENT_INPUT_SCHEMA,
    )

    with pytest.raises(
        MODULE.D0BAuthorFoundationError,
        match="selected target work disagrees",
    ):
        MODULE._author_batch(
            table,
            partition_index=0,
            episodes=[_episode()],
            selected_work_ids={1: np.array([100], dtype=np.uint64)},
        )
