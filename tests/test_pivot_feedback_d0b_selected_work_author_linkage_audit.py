from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/build_pivot_feedback_d0b_selected_work_author_linkage_audit.py"
)
CONFIG = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_d0b_selected_work_author_linkage_audit_v1.yaml"
)
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v5.yaml"
FOUNDATION = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_d0b_author_opportunity_foundation_full_20260724.summary.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0b_selected_work_author_linkage_audit_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_contract_is_parent_bound_and_forbids_episode_selection() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == MODULE.CONFIG_SCHEMA
    assert config["repository_bindings"]["gate_state"]["sha256"] == hashlib.sha256(
        GATE.read_bytes()
    ).hexdigest()
    assert config["repository_bindings"]["author_foundation_summary"][
        "sha256"
    ] == hashlib.sha256(FOUNDATION.read_bytes()).hexdigest()
    assert config["audit_contract"]["zero_valid_author_token_works_retained"] is True
    assert (
        config["audit_contract"]["linkage_threshold_or_episode_selection_allowed"]
        is False
    )
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def _selected_table() -> pa.Table:
    rows = [
        {
            "episode_index": 1,
            "field_id": 17,
            "topic_id": 14430,
            "entry_year": 2003,
            "work_numeric_id": work,
            "publication_year": year,
            "declared_authors_count": team,
            "incumbency_qualifying_work": year < 1997,
        }
        for work, year, team in (
            (100, 1995, 2),
            (101, 1998, 2),
            (102, 2000, 3),
            (103, 2002, 2),
        )
    ]
    return pa.Table.from_pylist(rows, schema=MODULE.SELECTED_WORK_SCHEMA)


def test_linkage_tables_retain_zero_token_works_and_freeze_incumbents() -> None:
    selected = _selected_table()
    metadata = {
        "work_type_code": np.array([1, 6, 1, 1], dtype=np.uint8),
        "is_paratext": np.array([False, False, False, False]),
        "is_retracted_value": np.array([False, False, False, False]),
        "is_retracted_null": np.array([False, False, True, False]),
    }
    incidence = {
        "selected_index": np.array([0, 0, 1, 1, 3], dtype=np.int64),
        "author_numeric_id": np.array([1, 2, 1, 3, 3], dtype=np.uint64),
    }
    episodes = [
        {"episode_index": 1, "field_id": 17, "topic_id": 14430, "entry_year": 2003}
    ]

    linkage, author_rows, accounting = MODULE._build_tables(
        selected,
        metadata,
        incidence,
        episodes=episodes,
        type_names={1: "article", 6: "dataset"},
    )

    assert linkage.schema == MODULE.WORK_LINKAGE_SCHEMA
    assert author_rows.schema == MODULE.INCIDENCE_SCHEMA
    rows = linkage.to_pylist()
    assert [row["valid_distinct_author_token_count"] for row in rows] == [2, 2, 0, 1]
    assert [row["declared_minus_valid_distinct_token_count"] for row in rows] == [
        0,
        0,
        3,
        1,
    ]
    assert [row["has_valid_author_token"] for row in rows] == [True, True, False, True]
    assert [row["has_pre_signal_incumbent_author"] for row in rows] == [
        True,
        True,
        False,
        False,
    ]
    assert [row["all_linked_authors_pre_signal_incumbents"] for row in rows] == [
        True,
        False,
        False,
        False,
    ]
    assert rows[2]["is_retracted"] is None
    assert author_rows["pre_signal_incumbent_author"].to_pylist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert accounting["selected_work_rows"] == 4
    assert accounting["linked_work_rows"] == 3
    assert accounting["zero_valid_author_token_work_rows"] == 1
    assert accounting["incumbent_authors_by_episode"] == {"1": 2}


def test_profile_keeps_work_type_caps_and_windows_nonranked() -> None:
    selected = _selected_table()
    linkage, _, _ = MODULE._build_tables(
        selected,
        {
            "work_type_code": np.array([1, 6, 1, 1], dtype=np.uint8),
            "is_paratext": np.zeros(4, dtype=bool),
            "is_retracted_value": np.zeros(4, dtype=bool),
            "is_retracted_null": np.zeros(4, dtype=bool),
        },
        {
            "selected_index": np.array([0, 0, 1, 1, 3], dtype=np.int64),
            "author_numeric_id": np.array([1, 2, 1, 3, 3], dtype=np.uint64),
        },
        episodes=[
            {
                "episode_index": 1,
                "field_id": 17,
                "topic_id": 14430,
                "entry_year": 2003,
            }
        ],
        type_names={1: "article", 6: "dataset"},
    )

    profile, summary = MODULE._profile(
        linkage,
        episodes=[
            {
                "episode_index": 1,
                "field_id": 17,
                "topic_id": 14430,
                "entry_year": 2003,
            }
        ],
        type_names={1: "article", 6: "dataset"},
    )

    assert profile.schema == MODULE.PROFILE_SCHEMA
    assert summary["1"]["ALL_PREDECISION"]["selected_work_rows"] == 4
    assert summary["1"]["ALL_PREDECISION"]["linked_work_rows"] == 3
    assert summary["1"]["REFERENCE"]["selected_work_rows"] == 1
    assert summary["1"]["REFERENCE"]["any_pre_signal_incumbent_work_rows"] == 1
    assert summary["1"]["SIGNAL"]["selected_work_rows"] == 2
    assert summary["1"]["SIGNAL"]["linked_work_rows"] == 1
    assert {
        row["team_cap"]
        for row in profile.to_pylist()
    } == {"10", "50", "100", "ALL"}
