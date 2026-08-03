from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (
    ROOT
    / "scripts/build_pivot_feedback_d0_certified_author_opportunity_rebuild_v1.py"
)
VERIFIER = (
    ROOT
    / "scripts/verify_pivot_feedback_d0_certified_author_opportunity_rebuild_v1.py"
)
CONFIG = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_certified_author_opportunity_rebuild_v1.yaml"
)
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v37.yaml"
ABI = ROOT / "experiments/configs/pivot_feedback_d0_measurement_abi_v1.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER_MODULE = _load(BUILDER, "certified_author_opportunity_builder_test")
VERIFIER_MODULE = _load(VERIFIER, "certified_author_opportunity_verifier_test")


def _episode() -> dict:
    return {
        "certified_episode_index": 1,
        "field_id": 17,
        "topic_id": 14430,
        "entry_year": 2003,
    }


def _linked() -> np.ndarray:
    linked = np.empty(5, dtype=BUILDER_MODULE.LINKED_WORK_DTYPE)
    linked["work_numeric_id"] = [100, 101, 102, 103, 104]
    linked["publication_year"] = [1995, 1998, 2000, 2001, 2002]
    linked["field_id"] = [17, 11, 17, 17, 17]
    linked["topic_id"] = [14430, 900, 14430, 901, 902]
    linked["work_type_code"] = [1, 2, 3, 16, 21]
    linked["declared_authors_count"] = [5, 20, 5, 25, 60]
    linked["is_retracted_null"] = [0, 1, 0, 0, 0]
    linked["temporal_disposition"] = [2, 2, 2, 3, 2]
    return linked


def _event(work: int, year: int, field: int, team: int, disposition: int = 2) -> dict:
    return {
        "author_numeric_id": 4096,
        "work_numeric_id": work,
        "publication_year": year,
        "temporal_disposition": disposition,
        "declared_authors_count": team,
        "primary_field_id": field,
    }


def test_config_binds_v37_and_frozen_abi_with_exact_four() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == BUILDER_MODULE.CONFIG_SCHEMA
    assert config["repository_bindings"]["gate_state"]["sha256"] == hashlib.sha256(
        GATE.read_bytes()
    ).hexdigest()
    assert config["repository_bindings"]["measurement_abi"][
        "sha256"
    ] == hashlib.sha256(ABI.read_bytes()).hexdigest()
    assert config["measurement_projection"]["inherited_work_type_codes"] == [
        1,
        2,
        3,
        4,
        7,
        16,
        18,
        21,
    ]
    assert config["measurement_projection"]["all_field_metric_field_ids"] == list(
        range(11, 37)
    )
    assert [
        (
            row["certified_episode_index"],
            row["field_id"],
            row["topic_id"],
            row["entry_year"],
        )
        for row in config["certified_episode_selection"]["exact_rows"]
    ] == [
        (1, 17, 14430, 2003),
        (2, 17, 10579, 2012),
        (3, 27, 10151, 1993),
        (4, 27, 12421, 1953),
    ]
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def test_selected_manifest_is_exact_linked_predecision_surface() -> None:
    table, selected_ids, accounting = BUILDER_MODULE._selected_work_table(
        _linked(),
        [_episode()],
    )

    assert table.schema == BUILDER_MODULE.WORK_OUTPUT_SCHEMA
    assert table["work_numeric_id"].to_pylist() == [100, 102]
    assert table["window_code"].to_pylist() == [0, 2]
    assert selected_ids[1].tolist() == [100, 102]
    assert accounting["by_episode"]["1"] == {
        "selected_work_rows": 2,
        "pre_signal_work_rows": 1,
        "reference_window_work_rows": 0,
        "signal_window_work_rows": 1,
        "is_retracted_null_rows": 0,
    }


def test_author_panel_filters_to_abi_linked_works_and_preserves_all_field() -> None:
    linked = _linked()
    _, selected_ids, _ = BUILDER_MODULE._selected_work_table(
        linked,
        [_episode()],
    )
    table = pa.Table.from_pylist(
        [
            _event(100, 1995, 17, 5),
            _event(101, 1998, 11, 20),
            _event(102, 2000, 17, 5),
            _event(103, 2001, 17, 25, 3),
            _event(104, 2002, 17, 60),
            _event(999, 2002, 17, 1),
        ],
        schema=BUILDER_MODULE.EVENT_INPUT_SCHEMA,
    )

    batches, accounting, eligible_rows = BUILDER_MODULE._author_batch(
        table,
        partition_index=0,
        linked=linked,
        episodes=[_episode()],
        selected_work_ids=selected_ids,
    )

    assert eligible_rows == 5
    result = batches[0]
    assert result.schema == BUILDER_MODULE.AUTHOR_OUTPUT_SCHEMA
    assert len(result) == 1
    row = result.to_pylist()[0]
    assert row["pre_entry_all_field_work_count_all"] == 5
    assert row["pre_entry_focal_field_work_count_all"] == 4
    assert row["portfolio_window_focal_field_work_count_all"] == 3
    assert row["pre_entry_target_topic_work_count"] == 2
    assert row["pre_signal_target_topic_work_count"] == 1
    assert row["first_target_topic_year"] == 1995
    assert row["incumbent_before_signal_window"] is True
    assert accounting[1]["author_rows"] == 1


def test_verifier_is_work_first_and_does_not_import_producer() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert (
        "import build_pivot_feedback_d0_certified_author_opportunity_rebuild_v1"
        not in source
    )
    assert "WORK_FIRST_PRELINK_ELIGIBILITY_THEN_EVENT_LINKAGE" in source
    assert VERIFIER_MODULE.VERIFY_STATUS.startswith("PASS_INDEPENDENT_EXACT_FOUR")
    assert VERIFIER_MODULE.AUTHOR_OUTPUT_SCHEMA == BUILDER_MODULE.AUTHOR_OUTPUT_SCHEMA
    assert VERIFIER_MODULE.WORK_OUTPUT_SCHEMA == BUILDER_MODULE.WORK_OUTPUT_SCHEMA
