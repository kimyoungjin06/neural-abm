from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/verify_pivot_feedback_d0b_selected_work_author_linkage_audit.py"
)
OUTPUT = Path(
    "/ssd/pivot-feedback-d0b-selected-work-author-linkage-audit-v1-full-20260724"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0b_selected_work_author_linkage_verifier_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _selected_rows() -> list[dict]:
    return [
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


def _incidence_rows() -> list[dict]:
    rows = []
    for work, year, team, author, incumbent in (
        (100, 1995, 2, 1, True),
        (100, 1995, 2, 2, True),
        (101, 1998, 2, 1, True),
        (101, 1998, 2, 3, False),
        (103, 2002, 2, 3, False),
    ):
        rows.append(
            {
                "episode_index": 1,
                "field_id": 17,
                "topic_id": 14430,
                "entry_year": 2003,
                "work_numeric_id": work,
                "publication_year": year,
                "declared_authors_count": team,
                "author_numeric_id": author,
                "pre_signal_incumbent_author": incumbent,
            }
        )
    return rows


def _linkage_rows() -> list[dict]:
    specs = [
        (100, 1995, 2, 2, 2),
        (101, 1998, 2, 2, 1),
        (102, 2000, 3, 0, 0),
        (103, 2002, 2, 1, 0),
    ]
    rows = []
    for work, year, declared, valid, incumbent in specs:
        rows.append(
            {
                "episode_index": 1,
                "field_id": 17,
                "topic_id": 14430,
                "entry_year": 2003,
                "work_numeric_id": work,
                "publication_year": year,
                "work_type_code": 1,
                "work_type_name": "article",
                "is_paratext": False,
                "is_retracted": False,
                "declared_authors_count": declared,
                "valid_distinct_author_token_count": valid,
                "declared_minus_valid_distinct_token_count": declared - valid,
                "has_valid_author_token": valid > 0,
                "pre_signal_incumbency_qualifying_work": year < 1997,
                "incumbent_valid_author_token_count": incumbent,
                "has_pre_signal_incumbent_author": incumbent > 0,
                "all_linked_authors_pre_signal_incumbents": (
                    valid > 0 and incumbent == valid
                ),
            }
        )
    return rows


def test_independent_count_replay_retains_zero_token_work() -> None:
    accounting = MODULE._verify_linkage_counts(
        _selected_rows(),
        _linkage_rows(),
        _incidence_rows(),
        {1: {1, 2}, 2: set(), 3: set(), 4: set()},
    )

    assert accounting["selected_work_rows"] == 4
    assert accounting["linked_work_rows"] == 3
    assert accounting["zero_valid_author_token_work_rows"] == 1
    assert accounting["valid_author_incidence_rows"] == 5
    assert accounting["declared_minus_valid_distinct_token_rows"] == 4


def test_independent_python_profile_replays_windows() -> None:
    rows = MODULE._profile_rows(
        _linkage_rows(),
        episodes=[
            {
                "episode_index": 1,
                "field_id": 17,
                "topic_id": 14430,
                "entry_year": 2003,
            }
        ],
        type_names={1: "article"},
    )
    by_key = {
        (row["aggregation_level"], row["team_cap"], row["calendar_window"]): row
        for row in rows
    }

    all_rows = by_key[("ALL_TYPES", "ALL", "ALL_PREDECISION")]
    reference = by_key[("ALL_TYPES", "ALL", "REFERENCE")]
    signal = by_key[("ALL_TYPES", "ALL", "SIGNAL")]
    assert all_rows["selected_work_rows"] == 4
    assert all_rows["linked_work_rows"] == 3
    assert reference["selected_work_rows"] == 1
    assert reference["any_pre_signal_incumbent_work_rows"] == 1
    assert signal["selected_work_rows"] == 2
    assert signal["zero_valid_author_token_work_rows"] == 1


def test_live_output_identity_is_private_and_hash_closed() -> None:
    receipt, report, artifacts = MODULE._validate_output(OUTPUT)

    assert receipt["d0_complete"] is False
    assert report["scope"]["episode_exclusion_decision_made"] is False
    assert len(artifacts) == 5


def test_external_report_is_private_create_only(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir(mode=0o700)
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
        "import build_pivot_feedback_d0b_selected_work_author_linkage_audit"
        not in source
    )
    assert "SEVERE_HETEROGENEOUS_LINKAGE_GAP" in source
    assert "FORBIDDEN_COLUMNS" in source
