from __future__ import annotations

from fractions import Fraction
import importlib.util
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_d0_structural_pilot_freeze.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_structural_pilot_freeze_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _field_fixture(tmp_path: Path):
    physical = {15, 16, 17, 19, 21, 22, 23, 25, 26, 31}
    life = {11, 13, 24, 28, 30}
    health = {27, 29, 34, 35, 36}
    fields = {}
    core = []
    orcid = []
    graph = []
    order = [17, 25, 27] + [
        field_id
        for field_id in range(36, 10, -1)
        if field_id not in {17, 25, 27}
    ]
    score = {field_id: len(order) - index for index, field_id in enumerate(order)}
    for field_id in range(11, 37):
        if field_id in physical:
            domain = "physical_sciences"
        elif field_id in life:
            domain = "life_sciences"
        elif field_id in health:
            domain = "health_sciences"
        else:
            domain = "social_sciences"
        fields[field_id] = {
            "domain_id": domain,
            "domain_name": domain.replace("_", " ").title(),
            "domain_rng_code": {
                "life_sciences": 1,
                "social_sciences": 2,
                "physical_sciences": 3,
                "health_sciences": 4,
            }[domain],
            "field_name": f"Field {field_id}",
            "field_slug": f"field_{field_id}",
        }
        value = score[field_id]
        denominator = 10_000
        core.append(
            {
                "field_id": field_id,
                "publication_year": 2012,
                "total_canonical_works": value * 1000,
                "usable_date_works": value * 900,
                "distinct_author_work_pairs": denominator,
                "history_usable_distinct_author_work_pairs": 8_000 + value,
            }
        )
        orcid.append(
            {
                "field_id": field_id,
                "publication_year": 2012,
                "grammar_valid_author_id_rows": denominator,
                "stable_orcid_author_work_pairs": 5_000 + value,
                "author_work_affiliation_rows": 6_000 + value,
                "author_work_institution_rows": 5_500 + value,
            }
        )
        graph.append(
            {
                "field_id": field_id,
                "publication_year": 2012,
                "declared_team_cap": 100,
                "graph_node_tokens": value * 100,
                "distinct_undirected_edges": value * 200,
            }
        )
    return (
        _write_parquet(tmp_path / "core.parquet", core),
        _write_parquet(tmp_path / "orcid.parquet", orcid),
        _write_parquet(tmp_path / "graph.parquet", graph),
        fields,
    )


def test_field_selection_is_outcome_blind_maximin_and_domain_distinct(tmp_path):
    core, orcid, graph, fields = _field_fixture(tmp_path)
    contract = {
        "entry_year_end_inclusive": 2012,
        "field_ids": list(range(11, 37)),
        "required_pilot_fields": 2,
        "metrics": list(MODULE.FIELD_METRICS),
    }
    table, selected, ranked = MODULE._field_ranking(
        core,
        orcid,
        graph,
        fields=fields,
        contract=contract,
    )
    assert len(table) == 26
    assert ranked[:3] == [17, 25, 27]
    assert [row["field_id"] for row in selected] == [17, 27]
    assert selected[0]["domain_id"] != selected[1]["domain_id"]
    assert "entry_effect_size" not in table.schema.names
    assert "predictive_score" not in table.schema.names


def _topic_year_rows() -> list[dict]:
    rows = []
    specs = [
        (17, 100, "A", 1701, "Family A", 1997, 2002, [10, 10, 10, 100, 100, 100]),
        (17, 101, "B", 1702, "Family B", 2006, 2011, [10, 10, 10, 50, 50, 50]),
        (27, 200, "C", 2701, "Family C", 2001, 2006, [10, 10, 10, 90, 90, 90]),
        (27, 201, "D", 2702, "Family D", 1985, 1990, [10, 10, 10, 40, 40, 40]),
    ]
    for field_id, topic_id, name, subfield_id, subfield_name, start, end, counts in specs:
        assert end - start + 1 == len(counts)
        for year, count in zip(range(start, end + 1), counts, strict=True):
            rows.append(
                {
                    "field_id": field_id,
                    "topic_id": topic_id,
                    "topic_name": name,
                    "subfield_id": subfield_id,
                    "subfield_name": subfield_name,
                    "publication_year": year,
                    "usable_primary_topic_works": count,
                }
            )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("subfield_name", pa.string(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=False),
            pa.field("usable_primary_topic_works", pa.uint64(), nullable=False),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def test_episode_pair_requires_distinct_family_and_disjoint_calendar_windows():
    contract = {
        "baseline_offsets": [-6, -5, -4],
        "recent_offsets": [-3, -2, -1],
        "entry_year_end_inclusive": 2012,
        "pair_selection": {"minimum_entry_year_separation": 9},
    }
    table, selected, accounting = MODULE._episode_candidates(
        _topic_year_rows(),
        selected_fields=[17, 27],
        contract=contract,
    )
    assert len(table) > 4
    assert accounting["selected_episode_count"] == 4
    assert accounting["distinct_subfields_within_each_field"] is True
    for field_id in (17, 27):
        rows = [row for row in selected if row["field_id"] == field_id]
        assert len({row["subfield_id"] for row in rows}) == 2
        assert abs(rows[0]["entry_year"] - rows[1]["entry_year"]) >= 9
    assert "entry_effect_size" not in table.schema.names
    assert "persistence_effect_size" not in table.schema.names


def test_exact_rational_score_order_does_not_depend_on_float_rounding():
    lower = {
        "standardized_gain_numerator": 2**53,
        "standardized_gain_denominator": 2**53,
        "minimum_baseline_recent_count": 1,
        "absolute_gain": 1,
        "topic_id": 2,
        "entry_year": 2000,
    }
    higher = {
        **lower,
        "standardized_gain_numerator": 2**53 + 1,
        "topic_id": 3,
    }
    assert float(Fraction(2**53, 2**53)) == float(
        Fraction(2**53 + 1, 2**53)
    )
    assert MODULE._candidate_key(higher) > MODULE._candidate_key(lower)
