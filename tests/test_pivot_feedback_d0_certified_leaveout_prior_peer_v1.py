from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
BUILDER = SCRIPTS / "build_pivot_feedback_d0_certified_leaveout_prior_peer_v1.py"
VERIFIER = SCRIPTS / "verify_pivot_feedback_d0_certified_leaveout_prior_peer_v1.py"
CONFIG = (
    ROOT
    / "experiments/configs/pivot_feedback_d0_certified_leaveout_prior_peer_v1.yaml"
)
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v38.yaml"
ABI = ROOT / "experiments/configs/pivot_feedback_d0_measurement_abi_v1.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load(BUILDER, "certified_leaveout_prior_peer_builder_test")
VERIFIER_MODULE = _load(
    VERIFIER,
    "certified_leaveout_prior_peer_verifier_test",
)


def _work_author(
    rows: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    values = np.empty(len(rows), dtype=MODULE.WORK_AUTHOR_DTYPE)
    for index, (episode, work, author, year, team) in enumerate(rows):
        values[index] = (episode, work, author, year, team)
    return values


def _author_base() -> pa.Table:
    identities = {
        "certified_episode_index": [1, 1, 2],
        "field_id": [17, 17, 17],
        "topic_id": [14430, 14430, 10579],
        "entry_year": [2003, 2003, 2012],
        "author_numeric_id": [100, 101, 50],
    }
    arrays = []
    for field in MODULE.parent.AUTHOR_OUTPUT_SCHEMA:
        if field.name in identities:
            arrays.append(pa.array(identities[field.name], type=field.type))
        elif pa.types.is_boolean(field.type):
            arrays.append(pa.array([False, False, False], type=field.type))
        else:
            arrays.append(pa.array([0, 0, 0], type=field.type))
    return pa.Table.from_arrays(
        arrays,
        schema=MODULE.parent.AUTHOR_OUTPUT_SCHEMA,
    )


def _target_table() -> pa.Table:
    rows = [
        (1, 17, 14430, 2003, 1000, 1995, 1, 5, 0, 1, 1, True, True),
        (1, 17, 14430, 2003, 1001, 2000, 1, 20, 2, 2, 1, True, False),
        (1, 17, 14430, 2003, 1002, 2002, 1, 60, 2, 2, 2, True, True),
        (2, 17, 10579, 2012, 2000, 2005, 1, 100, 0, 1, 1, True, True),
    ]
    return pa.Table.from_pylist(
        [
            dict(zip(MODULE.TARGET_OUTPUT_SCHEMA.names, row, strict=True))
            for row in rows
        ],
        schema=MODULE.TARGET_OUTPUT_SCHEMA,
    )


def _exclusions(
    rows: list[tuple[int, int, int, int]],
) -> np.ndarray:
    values = np.empty(len(rows), dtype=MODULE.EXCLUSION_DTYPE)
    for index, row in enumerate(rows):
        values[index] = row
    return values


def test_config_binds_v38_and_keeps_outcomes_and_selection_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == MODULE.CONFIG_SCHEMA
    assert config["repository_bindings"]["gate_state"]["sha256"] == hashlib.sha256(
        GATE.read_bytes()
    ).hexdigest()
    assert config["repository_bindings"]["measurement_abi"][
        "sha256"
    ] == hashlib.sha256(ABI.read_bytes()).hexdigest()
    assert config["measurement_contract"]["graph_team_cap_profiles"] == [10, 50, 100]
    assert config["measurement_contract"]["no_team_cap_selected"] is True
    assert config["output_contract"]["independent_verification_required"] is True
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def test_verifier_uses_independent_peer_work_first_sql_replay() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert (
        "import build_pivot_feedback_d0_certified_leaveout_prior_peer_v1"
        not in source
    )
    assert "DUCKDB_TARGET_PEER_WORK_FIRST_SEMIJOIN" in source
    assert "EXCEPT ALL" in source
    assert VERIFIER_MODULE.VERIFY_STATUS.startswith("PASS_INDEPENDENT_EXACT_FOUR")


def test_candidate_positions_respect_episode_then_author_sort_order() -> None:
    candidates = np.array(
        [(1, 100), (1, 200), (2, 50), (2, 300)],
        dtype=[("episode_index", "u1"), ("author_numeric_id", "<u8")],
    )

    positions = MODULE._candidate_positions(
        candidates,
        np.array([2, 1, 2, 1], dtype=np.uint8),
        np.array([300, 100, 50, 200], dtype=np.uint64),
    )

    assert positions.tolist() == [3, 0, 2, 1]


def test_pair_edges_reduce_shared_works_and_exclude_self_pairs() -> None:
    candidate_events = _work_author(
        [
            (1, 10, 100, 1999, 5),
            (1, 11, 100, 2001, 20),
            (1, 12, 101, 2002, 30),
        ]
    )
    peer_events = _work_author(
        [
            (1, 10, 100, 1999, 5),
            (1, 10, 200, 1999, 5),
            (1, 11, 200, 2001, 20),
            (1, 12, 201, 2002, 30),
        ]
    )
    incumbents = {
        1: np.array([200], dtype=np.uint64),
        2: np.empty(0, dtype=np.uint64),
        3: np.empty(0, dtype=np.uint64),
        4: np.empty(0, dtype=np.uint64),
    }

    edges, accounting = MODULE._pair_edges(
        candidate_events,
        peer_events,
        incumbent_authors=incumbents,
    )

    assert [
        (
            int(row["candidate_author_id"]),
            int(row["peer_author_id"]),
            int(row["minimum_declared_team_size"]),
            int(row["first_prior_collaboration_year"]),
            int(row["last_prior_collaboration_year"]),
            int(row["distinct_shared_work_count"]),
            bool(row["peer_pre_signal_incumbent"]),
        )
        for row in edges
    ] == [
        (100, 200, 5, 1999, 2001, 2, True),
        (101, 201, 30, 2002, 2002, 1, False),
    ]
    assert accounting["candidate_target_peer_work_incidences"] == 3
    assert accounting["distinct_candidate_target_peer_edges"] == 2


def test_author_profiles_are_monotone_across_focal_and_network_leaveout() -> None:
    authors = _author_base()
    candidate_values, _, _ = MODULE._candidate_sets(authors)
    edges = np.empty(3, dtype=MODULE.EDGE_DTYPE)
    edges[0] = (1, 100, 200, 5, 1999, 2001, 2, 1)
    edges[1] = (1, 100, 201, 30, 2002, 2002, 1, 0)
    edges[2] = (2, 50, 300, 100, 2010, 2010, 1, 1)
    own = _exclusions(
        [
            (1, 100, 0, 0),
            (1, 101, 1, 0),
        ]
    )
    peer = _exclusions(
        [
            (1, 100, 1, 5),
            (1, 100, 2, 30),
            (2, 50, 3, 100),
        ]
    )
    network = MODULE._reduce_exclusions(
        np.concatenate([own, peer], dtype=MODULE.EXCLUSION_DTYPE)
    )

    result, accounting = MODULE._author_measurement_table(
        candidate_values,
        authors,
        edges,
        peer,
        own,
        network,
        _target_table(),
    )

    rows = result.to_pylist()
    focal = "any_incumbent_focal_leaveout_work_count_cap_50"
    network_name = "any_incumbent_network_leaveout_work_count_cap_50"
    attention = "any_incumbent_attention_work_count_cap_50"
    assert rows[0][attention] == 2
    assert rows[0][focal] == 1
    assert rows[0][network_name] == 0
    assert rows[1][attention] == 2
    assert rows[1][focal] == 1
    assert rows[1][network_name] == 1
    for row in rows:
        for cap in (10, 50, 100):
            assert (
                row[f"any_incumbent_network_leaveout_work_count_cap_{cap}"]
                <= row[f"any_incumbent_focal_leaveout_work_count_cap_{cap}"]
                <= row[f"any_incumbent_attention_work_count_cap_{cap}"]
            )
            assert (
                row[f"all_incumbent_network_leaveout_work_count_cap_{cap}"]
                <= row[f"all_incumbent_focal_leaveout_work_count_cap_{cap}"]
                <= row[f"all_incumbent_attention_work_count_cap_{cap}"]
            )
    assert accounting["by_cap"]["50"][
        "candidate_rows_with_prior_target_active_peer"
    ] == 1
