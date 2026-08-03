from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_s0_threshold_free_inventory_v1.yaml"
SCRIPT = ROOT / "scripts/build_pivot_feedback_s0_threshold_free_inventory.py"
G_MINUS_1A = (
    ROOT
    / "experiments/evidence/results/pivot_feedback_g_minus_1a_receipt_20260723.summary.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_s0_threshold_free_inventory_test_module", SCRIPT
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


def test_contract_is_threshold_free_fail_closed_and_g_minus_1a_bound() -> None:
    config = _config()

    assert config["schema_version"] == MODULE.CONFIG_SCHEMA
    assert config["status_before_execution"] == "AUTHORIZED_BY_G_MINUS_1A"
    assert config["g_minus_1a_parent"]["sha256"] == hashlib.sha256(
        G_MINUS_1A.read_bytes()
    ).hexdigest()
    assert config["structural_contract"]["field_ids"] == list(range(11, 37))
    assert config["structural_contract"][
        "team_cap_profiles_are_ranked_or_selected"
    ] is False
    assert config["structural_contract"][
        "citation_calendar_support_is_outcome_success"
    ] is False
    assert config["structural_contract"][
        "attention_resolution_is_ranked_or_selected"
    ] is False
    assert all(
        item["state"].startswith("NOT_RUN")
        for item in config["deferred_required_metrics"].values()
    )
    assert config["output_contract"]["s0_gate_pass_after_core_component"] is False
    assert config["output_contract"]["d0_authorized_after_core_component"] is False


def test_small_exact_work_and_author_event_aggregation(tmp_path: Path) -> None:
    work_path = tmp_path / "work.parquet"
    work_table = pa.Table.from_arrays(
        [
            pa.array([0, 4096, 8192], type=pa.uint64()),
            pa.array([18262, 19723, None], type=pa.int32()),
            pa.array([2020, 2024, None], type=pa.uint16()),
            pa.array([2, 3, 0], type=pa.uint8()),
            pa.array([1, 1, 1], type=pa.uint8()),
            pa.array([False, False, False], type=pa.bool_()),
            pa.array([False, False, None], type=pa.bool_()),
            pa.array([2, 1, 0], type=pa.uint16()),
            pa.array([10001, 10002, None], type=pa.uint32()),
            pa.array([11, 11, None], type=pa.uint8()),
            pa.array([1, 1, None], type=pa.uint8()),
        ],
        schema=MODULE.WORK_SCHEMA,
    )
    pq.write_table(work_table, work_path)

    event_path = tmp_path / "events.parquet"
    event_table = pa.Table.from_arrays(
        [
            pa.array([4096, 8192, 4096], type=pa.uint64()),
            pa.array([0, 0, 4096], type=pa.uint64()),
            pa.array([2020, 2020, 2024], type=pa.uint16()),
            pa.array([2, 2, 3], type=pa.uint8()),
            pa.array([2, 2, 1], type=pa.uint32()),
            pa.array([11, 11, 11], type=pa.uint8()),
        ],
        schema=MODULE.EVENT_SCHEMA,
    )
    pq.write_table(event_table, event_path)

    structural = _config()["structural_contract"]
    work_cells, work_totals = MODULE._aggregate_work_files([work_path], structural)
    event_cells, event_totals = MODULE._aggregate_event_files(
        [event_path], structural
    )
    table = MODULE._field_year_table(work_cells, event_cells)
    rows = {
        row["publication_year"]: row
        for row in table.to_pylist()
        if row["field_id"] == 11
    }

    assert work_totals == {
        "rows": 3,
        "classified_field_rows": 2,
        "null_field_rows": 1,
        "null_year_rows": 1,
        "usable_date_rows": 2,
    }
    assert event_totals == {
        "rows": 3,
        "history_usable_rows": 3,
        "classified_field_rows": 3,
        "null_field_rows": 0,
    }
    assert rows[2020]["total_canonical_works"] == 1
    assert rows[2020]["declared_author_slots"] == 2
    assert rows[2020]["declared_pair_incidences_cap_10"] == 1
    assert rows[2020]["distinct_author_work_pairs"] == 2
    assert rows[2020]["distinct_author_tokens"] == 2
    assert rows[2024]["distinct_author_work_pairs"] == 1
    assert rows[2024]["distinct_author_tokens"] == 1

    attention = {
        row["publication_year"]: row
        for row in MODULE._attention_table(table).to_pylist()
    }
    assert attention[2020]["annual_canonical_works"] == 1
    assert attention[2024]["trailing_3y_canonical_works"] == 1
    assert attention[2024]["trailing_5y_canonical_works"] == 2


def test_no_forbidden_metric_is_emitted_by_core_schema() -> None:
    config = _config()
    emitted = set(MODULE.WORK_METRICS) | set(MODULE.AUTHOR_METRICS) | {
        "annual_canonical_works",
        "trailing_3y_canonical_works",
        "trailing_5y_canonical_works",
    }

    assert emitted.isdisjoint(config["forbidden_metrics"])
    source = SCRIPT.read_text(encoding="utf-8")
    assert "cited_by_count" not in source
    assert "entry_effect_size" not in emitted
    assert "predictive_score" not in emitted


def test_parent_receipt_pass_authorizes_only_s0_core() -> None:
    parent = MODULE._validate_parent(_config(), ROOT)

    assert parent["status"].startswith("PASS_G_MINUS_1A_BOUNDED")
    assert parent["sha256"] == hashlib.sha256(G_MINUS_1A.read_bytes()).hexdigest()
