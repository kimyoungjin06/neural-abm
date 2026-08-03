from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tomllib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

try:
    import duckdb
except ModuleNotFoundError:
    duckdb = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_s0_coauthor_graph_profile.py"
CONFIG = ROOT / "experiments/configs/pivot_feedback_s0_coauthor_graph_profile_v1.yaml"
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v2.yaml"
CORE = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_s0_threshold_free_core_full_20260723.summary.json"
)
DESIGN = ROOT / "docs/research/pivot-feedback-s0-coauthor-graph-profile-v1.md"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_s0_coauthor_graph_profile_test_module", SCRIPT
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


def test_contract_is_exact_nonranked_and_parent_bound() -> None:
    config = _config()
    graph = config["graph_contract"]

    assert config["schema_version"] == MODULE.CONFIG_SCHEMA
    assert config["gate_parent"]["sha256"] == hashlib.sha256(
        GATE.read_bytes()
    ).hexdigest()
    assert config["s0_core_parent"]["summary_sha256"] == hashlib.sha256(
        CORE.read_bytes()
    ).hexdigest()
    assert [item["cap"] for item in graph["profiles"]] == [10, 50, 100]
    assert graph["profiles_are_ranked_or_selected"] is False
    assert graph["all_profiles_must_be_reported"] is True
    assert (
        graph["null_publication_year_cell_encoding"]
        == "ZERO_ONLY_FOR_INTERNAL_BUCKETING_OUTPUT_REMAINS_NULL"
    )
    assert (
        graph["declared_all_team_profile"]["state"]
        == "DIAGNOSTIC_ONLY_NOT_EDGE_MATERIALIZED"
    )
    assert graph["author_entity_claim_allowed"] is False
    assert config["execution"]["gpu_used"] is False
    assert config["output_contract"]["component_may_complete_s0_by_itself"] is False
    assert config["output_contract"]["component_may_authorize_d0_by_itself"] is False
    assert DESIGN.is_file()


def test_duckdb_version_is_locked_for_reproducible_execution() -> None:
    config = _config()

    assert config["execution"]["required_duckdb_version"] == "1.5.5"
    assert "duckdb" not in {
        item.lower()
        for group in (
            "full",
            "research",
        )
        for item in tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["optional-dependencies"][group]
    }


def test_exact_cell_metrics_include_isolates_and_are_nested() -> None:
    quantiles = _config()["graph_contract"]["degree_quantiles"]
    nodes = np.array([1, 2, 3, 4], dtype=np.uint64)
    node_caps = np.array([10, 10, 50, 100], dtype=np.uint8)
    edge_u = np.array([1, 2, 3], dtype=np.uint64)
    edge_v = np.array([2, 3, 4], dtype=np.uint64)
    edge_caps = np.array([10, 50, 100], dtype=np.uint8)

    cap_10 = MODULE._cell_profile_metrics(
        node_authors=nodes,
        node_minimum_caps=node_caps,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_minimum_caps=edge_caps,
        cap=10,
        quantiles=quantiles,
    )
    cap_50 = MODULE._cell_profile_metrics(
        node_authors=nodes,
        node_minimum_caps=node_caps,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_minimum_caps=edge_caps,
        cap=50,
        quantiles=quantiles,
    )
    cap_100 = MODULE._cell_profile_metrics(
        node_authors=nodes,
        node_minimum_caps=node_caps,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_minimum_caps=edge_caps,
        cap=100,
        quantiles=quantiles,
    )

    assert cap_10["graph_node_tokens"] == 2
    assert cap_10["distinct_undirected_edges"] == 1
    assert cap_10["connected_components"] == 1
    assert cap_10["largest_component_nodes"] == 2
    assert cap_10["degree_sum"] == 2
    assert cap_50["graph_node_tokens"] == 3
    assert cap_50["distinct_undirected_edges"] == 2
    assert cap_50["connected_components"] == 1
    assert cap_50["degree_max"] == 2
    assert cap_100["graph_node_tokens"] == 4
    assert cap_100["distinct_undirected_edges"] == 3
    assert cap_100["connected_components"] == 1
    assert cap_100["largest_component_nodes"] == 4


def test_missing_edge_endpoint_is_rejected() -> None:
    with pytest.raises(MODULE.GraphProfileError, match="endpoint missing"):
        MODULE._cell_profile_metrics(
            node_authors=np.array([1], dtype=np.uint64),
            node_minimum_caps=np.array([10], dtype=np.uint8),
            edge_u=np.array([1], dtype=np.uint64),
            edge_v=np.array([2], dtype=np.uint64),
            edge_minimum_caps=np.array([10], dtype=np.uint8),
            cap=10,
            quantiles=_config()["graph_contract"]["degree_quantiles"],
        )


def test_tiny_duckdb_materialization_deduplicates_and_profiles(
    tmp_path: Path,
) -> None:
    if duckdb is None:
        pytest.skip("repository-only DuckDB overlay is not installed")
    event_path = tmp_path / "part-0000.parquet"
    rows = [
        [4096, 100, 2020, 2, 2, 11],
        [8192, 100, 2020, 2, 2, 11],
        [8192, 200, 2020, 2, 20, 11],
        [12288, 200, 2020, 2, 20, 11],
        [16384, 300, 2020, 2, 1, 11],
        [20480, 400, 0, 2, 2, 11],
        [24576, 400, 0, 2, 2, 11],
    ]
    table = pa.Table.from_pylist(
        [
            {
                "author_numeric_id": row[0],
                "work_numeric_id": row[1],
                "publication_year": row[2],
                "temporal_disposition": row[3],
                "declared_authors_count": row[4],
                "primary_field_id": row[5],
            }
            for row in rows
        ],
        schema=MODULE.EVENT_SCHEMA,
    )
    pq.write_table(table, event_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    connection = duckdb.connect(str(scratch / "test.duckdb"))
    try:
        accounting = MODULE._materialize_exact_graph(
            connection, [event_path], scratch, _config()
        )
        assert accounting["source_rows"] == 7
        assert accounting["classified_rows"] == 7
        assert accounting["actual_pair_incidences_cap_10"] == 2
        assert accounting["actual_pair_incidences_cap_50"] == 3
        assert accounting["actual_pair_incidences_cap_100"] == 3
        assert accounting["materialized_node_tokens_cap_10"] == 5
        assert accounting["materialized_node_tokens_cap_50"] == 6
        assert accounting["materialized_node_tokens_cap_100"] == 6
        assert accounting["materialized_distinct_edges_cap_10"] == 2
        assert accounting["materialized_distinct_edges_cap_50"] == 3
        assert accounting["materialized_distinct_edges_cap_100"] == 3

        key = 11 * MODULE.YEAR_MULTIPLIER + 2020
        null_year_key = 11 * MODULE.YEAR_MULTIPLIER
        graph = MODULE._compute_graph_inventory(
            connection,
            core_keys=[null_year_key, key],
            node_root=Path(accounting["node_root"]),
            edge_root=Path(accounting["edge_root"]),
            config=_config(),
        )
    finally:
        connection.close()
    by_key = {
        (row["publication_year"], row["declared_team_cap"]): row
        for row in graph.to_pylist()
    }
    by_cap = {
        cap: by_key[(2020, cap)]
        for cap in (10, 50, 100)
    }

    assert len(graph) == 6
    assert by_cap[10]["graph_node_tokens"] == 3
    assert by_cap[10]["distinct_undirected_edges"] == 1
    assert by_cap[10]["connected_components"] == 2
    assert by_cap[10]["largest_component_nodes"] == 2
    assert by_cap[50]["graph_node_tokens"] == 4
    assert by_cap[50]["distinct_undirected_edges"] == 2
    assert by_cap[50]["connected_components"] == 2
    assert by_cap[50]["largest_component_nodes"] == 3
    assert by_cap[100]["graph_node_tokens"] == by_cap[50]["graph_node_tokens"]
    assert (
        by_cap[100]["distinct_undirected_edges"]
        == by_cap[50]["distinct_undirected_edges"]
    )
    assert by_cap[100]["connected_components"] == by_cap[50]["connected_components"]
    assert by_key[(None, 10)]["graph_node_tokens"] == 2
    assert by_key[(None, 10)]["distinct_undirected_edges"] == 1
    assert by_key[(None, 10)]["connected_components"] == 1
    assert by_key[(None, 10)]["largest_component_nodes"] == 2


def test_graph_schema_excludes_outcome_and_selection_metrics() -> None:
    config = _config()
    emitted = set(MODULE.GRAPH_METRICS) | {"largest_component_share"}

    assert emitted.isdisjoint(config["forbidden_metrics"])
    assert "predictive_score" not in emitted
    assert "selected_best_field" not in emitted
    assert "entry_effect_size" not in emitted
