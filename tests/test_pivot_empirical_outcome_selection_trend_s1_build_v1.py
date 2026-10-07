from __future__ import annotations

import copy
import datetime as dt
import importlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_pivot_empirical_outcome_selection_trend_s1_v1.py"


def _module():  # noqa: ANN202
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    return importlib.import_module(SCRIPT.stem)


s1 = _module()
CONTRACT = s1.load_contract()
N10 = s1.load_n10(CONTRACT)

AUTHOR = pa.struct([("display_name", pa.string()), ("id", pa.string()), ("orcid", pa.string())])
AUTHORSHIP = pa.struct([("author", AUTHOR), ("author_position", pa.string())])
NAMED = pa.struct([("id", pa.string()), ("display_name", pa.string())])
TOPIC = pa.struct([("id", pa.string()), ("display_name", pa.string()), ("score", pa.float32()), ("subfield", NAMED), ("field", NAMED), ("domain", NAMED)])
SCHEMA = pa.schema([
    ("id", pa.string()), ("publication_year", pa.int32()), ("publication_date", pa.date32()), ("type", pa.string()),
    ("is_paratext", pa.bool_()), ("is_retracted", pa.bool_()), ("is_xpac", pa.bool_()),
    ("authorships", pa.list_(pa.field("element", AUTHORSHIP))), ("primary_topic", TOPIC),
    ("primary_location", pa.struct([("source", NAMED), ("is_oa", pa.bool_())])),
])


def _row(work, year, date, *, field=16, topic=100, subfield=1605, kind="article", paratext=False, retracted=False, xpac=False,
         authors=("https://openalex.org/A1",), source="https://openalex.org/S7", primary=True):  # noqa: ANN001, ANN202
    return {
        "id": work, "publication_year": year, "publication_date": date, "type": kind, "is_paratext": paratext,
        "is_retracted": retracted, "is_xpac": xpac,
        "authorships": [{"author": {"display_name": "x", "id": a, "orcid": None}, "author_position": "middle"} for a in authors],
        "primary_topic": None if not primary else {
            "id": f"https://openalex.org/T{topic}", "display_name": "t", "score": 0.9,
            "subfield": {"id": f"https://openalex.org/subfields/{subfield}", "display_name": "s"},
            "field": {"id": f"https://openalex.org/fields/{field}", "display_name": "f"},
            "domain": {"id": "https://openalex.org/domains/1", "display_name": "d"},
        },
        "primary_location": {"source": None if source is None else {"id": source, "display_name": "j"}, "is_oa": False},
    }


def _task(tmp_path: Path, path: Path) -> dict:
    parts = tmp_path / "parts"
    parts.mkdir(exist_ok=True)
    extract = CONTRACT["extract"]
    return {"index": 0, "path": str(path), "relative_path": "fixture", "part_dir": str(parts), "fields": extract["fields"],
            "latest_year": extract["latest_year"], "type_names": extract["type_names"], "snapshot_date": "2026-06-26",
            "work_quarantine": CONTRACT["identifiers"]["work_quarantine"]}


def test_the_contract_binds_n10_v2_the_guard_and_the_source() -> None:
    assert CONTRACT["implements"]["stage"] == "S1"
    assert CONTRACT["build"]["persistence_labels_computed"] == 0
    assert CONTRACT["extract"]["fields"] == [16, 32, 11]
    assert len(N10.v1.verify_source_bindings(CONTRACT)) == len(CONTRACT["source_bindings"])
    mutated = copy.deepcopy(CONTRACT)
    mutated["extract"]["fields"] = [16, 32, 27]
    with pytest.raises(s1.S1Error, match="fields"):
        s1.validate_contract(mutated)


def test_footer_check_finds_missing_and_mistyped_leaves(tmp_path: Path) -> None:
    good = tmp_path / "good.parquet"
    pq.write_table(pa.Table.from_pylist([_row("https://openalex.org/W1", 2005, dt.date(2005, 5, 3))], schema=SCHEMA), good)
    assert s1.footer_check(good) == {"rows": 1, "missing": [], "mistyped": []}
    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"id": ["x"], "publication_year": pa.array([2005], pa.int64())}), bad)
    result = s1.footer_check(bad)
    assert "author_id" in result["missing"] and result["mistyped"] == ["publication_year"]


def test_extract_file_accounts_for_every_row(tmp_path: Path) -> None:
    rows = [
        _row("https://openalex.org/W1", 2005, dt.date(2005, 5, 3), authors=("https://openalex.org/A1", "https://openalex.org/A2", "https://openalex.org/A1")),
        _row("https://openalex.org/W2", 2005, dt.date(2005, 5, 3), field=17),
        _row("https://openalex.org/W3", 2005, dt.date(2005, 5, 3), kind="editorial"),
        _row("https://openalex.org/W4", 2005, dt.date(2005, 5, 3), paratext=True),
        _row("https://openalex.org/W5", 2005, dt.date(2005, 5, 3), retracted=True),
        _row("https://openalex.org/W6", 2005, dt.date(2005, 5, 3), retracted=None, authors=(None, "https://openalex.org/A3")),
        _row("https://openalex.org/W7", 2018, dt.date(2018, 5, 3)),
        _row("https://openalex.org/W8", 1890, dt.date(1890, 1, 1), field=32),
        _row("https://openalex.org/W-1", 2005, dt.date(2005, 5, 3)),
        _row("https://openalex.org/W10", 2001, dt.date(2001, 1, 1), field=11, xpac=True, source=None),
        _row("https://openalex.org/W11", 2005, dt.date(2005, 5, 3), primary=False),
    ]
    path = tmp_path / "works.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    result = s1.extract_file(_task(tmp_path, path))
    counts = result["counts"]
    assert counts["rows"] == 11
    assert counts["quarantined_work_id"] == 1
    assert sum(counts[k] for k in s1.FAILURE_COUNTS) == 0
    assert counts["registered_field_rows"] == 9
    assert (counts["registered_excluded_by_type"], counts["registered_excluded_paratext"], counts["registered_excluded_retracted"]) == (1, 1, 1)
    assert counts["registered_excluded_by_date_or_year"] == 2
    assert (counts["extract_works"], counts["extract_retracted_null"], counts["extract_xpac"], counts["extract_null_author_ids"]) == (3, 1, 1, 1)
    assert (counts["disposition_1"], counts["disposition_2"]) == (1, 10)
    works = pq.read_table(tmp_path / "parts" / "works_0000.parquet")
    assert works["work_id"].to_pylist() == [1, 6, 10]
    assert works["source"].to_pylist() == [7, 7, None]
    assert works["field"].to_pylist() == [16, 16, 11]
    pairs = pq.read_table(tmp_path / "parts" / "authorships_0000.parquet")
    assert list(zip(pairs["work_id"].to_pylist(), pairs["author_id"].to_pylist())) == [(1, 1), (1, 2), (6, 3), (10, 1)]
    ids = pq.read_table(tmp_path / "parts" / "ids_0000.parquet")["work_id"].to_pylist()
    assert ids == [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]


def test_extract_file_counts_invalid_identifiers_and_dates(tmp_path: Path) -> None:
    rows = [
        _row("https://openalex.org/W1", 2005, dt.date(2006, 1, 2)),
        _row("https://openalex.org/Wx", 2005, dt.date(2005, 5, 3)),
        _row("https://openalex.org/W3", 2005, dt.date(2005, 5, 3), authors=("https://openalex.org/Aq",)),
        _row("https://openalex.org/W4", None, dt.date(2005, 5, 3)),
    ]
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    counts = s1.extract_file(_task(tmp_path, path))["counts"]
    assert counts["publication_date_year_mismatch"] == 1
    assert counts["malformed_work_id"] == 1
    assert counts["registered_malformed_author_id"] == 1
    assert counts["publication_exactly_one_null"] == 1


def _structure_fixture() -> tuple[pa.Table, pa.Table, list[int]]:
    """Ten topics of field 16; topic 1 surges in attention; entrants spread over 2003-2008."""

    works, pairs = [], []
    wid = 1

    def add(year, topic, authors, month=6, day=15, source=None):  # noqa: ANN001, ANN202
        nonlocal wid
        works.append({"work_id": wid, "year": year, "date": dt.date(year, month, day), "topic": topic, "subfield": 1601 if topic <= 5 else 1602,
                      "field": 16, "score": 0.9, "source": source if source is not None else wid % 50, "is_xpac": False, "retracted_null": False})
        pairs.extend((wid, a) for a in authors)
        wid += 1

    for topic in range(1, 11):
        incumbents = [10_000 + 100 * topic + i for i in range(5)]
        for a in incumbents:
            add(1990, topic, [a])
        for year in range(1999, 2005):
            n = 10 if year <= 2001 else (40 if topic in (1, 6) else 10 + topic % 3)
            for i in range(n):
                add(year, topic, [incumbents[i % 5], incumbents[(i + 1) % 5]])
    entrants = []
    for i in range(90):
        author = 50_000 + i
        year = 2003 + i % 6
        month, day = ((1, 1) if i % 7 == 0 else (1 + (i * 5) % 12, 10))
        if i % 5:
            add(year - 1, 6, [author])
        if i % 4 == 0:
            add(year - 3, 9, [author])
        add(year, 1, [author], month=month, day=day)
        entrants.append(author)
    guard = [50_002, 50_003, 99_999]
    schema = pa.schema([("work_id", pa.uint64()), ("year", pa.int16()), ("date", pa.date32()), ("topic", pa.uint32()), ("subfield", pa.uint32()),
                        ("field", pa.uint8()), ("score", pa.float32()), ("source", pa.uint64()), ("is_xpac", pa.bool_()), ("retracted_null", pa.bool_())])
    w = pa.Table.from_pylist(works, schema=schema)
    a = pa.table({"work_id": pa.array([p[0] for p in pairs], pa.uint64()), "author_id": pa.array([p[1] for p in pairs], pa.uint64())})
    return w, a, guard


def _expected_entrants(w: pa.Table, a: pa.Table, guard: list[int], registered: list[dict]) -> set[tuple]:
    work = {r["work_id"]: r for r in w.to_pylist()}
    by_author: dict[int, list[dict]] = {}
    for p in a.to_pylist():
        by_author.setdefault(p["author_id"], []).append(work[p["work_id"]])
    first = {(au, r["topic"]): min(x["year"] for x in rows if x["topic"] == r["topic"]) for au, rows in by_author.items() for r in rows}
    out = set()
    for e in registered:
        for (au, topic), year in first.items():
            if topic != e["topic"] or not e["t"] <= year <= e["t"] + 2:
                continue
            if au in guard or not any(x["field"] == e["field"] and year - 3 <= x["year"] <= year - 1 for x in by_author[au]):
                continue
            date = min(x["date"] for x in by_author[au] if x["topic"] == topic and x["year"] == year)
            out.add((e["episode_id"], au, date))
    discovery_authors = {au for ep, au, _d in out if next(r for r in registered if r["episode_id"] == ep)["partition"] == "discovery"}
    return {row for row in out if next(r for r in registered if r["episode_id"] == row[0])["partition"] == "discovery" or row[1] not in discovery_authors}


def test_build_structure_matches_an_independent_python_implementation() -> None:
    w, a, guard = _structure_fixture()
    built = s1.build_structure(w, a, pa.array(guard, pa.uint64()), CONTRACT, N10, threads=2)
    registered = built["registered"]
    assert {e["topic"] for e in registered} == {1, 6} and {e["field"] for e in registered} == {16}
    assert {e["partition"] for e in registered} == {"discovery", "confirmatory"}
    assert built["counts"]["discovery_author_removed"] > 0
    for topic in (1, 6):
        years = [e["t"] for e in registered if e["topic"] == topic]
        assert all(later - earlier >= 3 for earlier, later in zip(years, years[1:]))
    entrants = built["entrants"].to_pylist()
    assert {(r["episode_id"], r["author_id"], r["entry_date"]) for r in entrants} == _expected_entrants(w, a, guard, registered)
    for r in entrants:
        clock = N10.entry_clock([r["entry_date"]], next(e["t"] for e in registered if e["episode_id"] == r["episode_id"]))
        assert (clock["offset"], clock["precise"]) == (r["offset"], r["precise"])
    assert built["counts"]["g4_guard_removed"] == 2
    assert built["counts"]["raw_entrants"] > built["counts"]["pivot_eligible_entrants"]


def test_attention_counts_only_incumbent_only_works() -> None:
    w, a, _guard = _structure_fixture()
    import duckdb

    con = duckdb.connect()
    con.register("w_arrow", w)
    con.register("a_arrow", a)
    con.execute("CREATE TABLE w AS SELECT * FROM w_arrow")
    con.execute("CREATE TABLE aw AS SELECT a.author_id, w.work_id, w.topic, w.subfield, w.field, w.year, w.date FROM a_arrow a JOIN w USING (work_id)")
    con.execute("CREATE TABLE first_topic AS SELECT author_id, topic, MIN(year) AS first_year FROM aw GROUP BY ALL")
    rows = s1.attention_rows(con, 2005, 2005)[(16, 2005)]
    topic1 = next(r for r in rows if r["topic_id"] == 1)
    assert topic1 == {"topic_id": 1, "recent": [40, 40, 40], "reference": [10, 10, 10]}
    assert next(r for r in rows if r["topic_id"] == 6)["reference"] == [10, 10, 10]


def test_screen_decision() -> None:
    rule = CONTRACT["build"]["indexing_screen"]
    both = {"reference": {"median_score": 0.95, "dominant_share": 0.1}, "recent": {"median_score": 0.7, "dominant_share": 0.5}}
    assert s1.screen_decision(both, rule) == "EXCLUDED_INDEXING_ANOMALY"
    one = {"reference": {"median_score": 0.95, "dominant_share": 0.1}, "recent": {"median_score": 0.7, "dominant_share": 0.2}}
    assert s1.screen_decision(one, rule) == "RETAINED"
    assert s1.screen_decision({"recent": {"median_score": 0.7, "dominant_share": 0.2}}, rule) == "UNDEFINED_NOT_EXCLUDED"


def test_floors_agree_with_the_n10_episode_statistic() -> None:
    validity = s1.plan_validity(CONTRACT)
    rows = []
    for episode, (n, spread) in enumerate([(40, 12), (40, 4), (20, 12), (35, 8)]):
        for i in range(n):
            rows.append({"episode_id": episode, "offset": i % spread, "precise": i % 9 != 0})
    table = pa.Table.from_pylist(rows)
    floors = s1.episode_floors(table, validity)
    for episode in range(4):
        group = [{"offset": r["offset"], "persisted": 0, "precise": r["precise"]} for r in rows if r["episode_id"] == episode]
        assert floors[episode]["meets_floor"] == N10.episode_statistic(group, validity)["meets_floor"]


def test_disposition_requires_four_confirmatory_episodes_per_field() -> None:
    registered = [{"episode_id": i, "field": f, "partition": "confirmatory"} for i, f in enumerate([16] * 4 + [32] * 4 + [11] * 4)]
    floors = {i: {"meets_floor": True} for i in range(12)}
    assert s1.disposition(registered, floors, CONTRACT)["status"] == "S1_PASS"
    floors[11]["meets_floor"] = False
    assert s1.disposition(registered, floors, CONTRACT)["status"] == "STOP_INSUFFICIENT_EPISODES"


def test_authorization_is_checked_before_any_row(tmp_path: Path) -> None:
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps({"schema_version": "x"}), encoding="utf-8")
    with pytest.raises(s1.S1Error, match="schema"):
        s1.load_authorization(path, s1.DEFAULT_CONFIG)
    path.write_text(json.dumps({"schema_version": "pivot_empirical_outcome_selection_trend_s1_authorization.v1", "status": "AUTHORIZED_S1_STRUCTURAL_BUILD",
                                "s1_contract_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(s1.S1Error, match="another S1 contract"):
        s1.load_authorization(path, s1.DEFAULT_CONFIG)


def test_the_builder_computes_no_persistence() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "persisted(" not in source
    assert "n10.persisted" not in source


def _raw_files(tmp_path: Path) -> list[tuple[str, Path]]:
    w, a, _guard = _structure_fixture()
    authors: dict[int, list[str]] = {}
    for p in a.to_pylist():
        authors.setdefault(p["work_id"], []).append(f"https://openalex.org/A{p['author_id']}")
    rows = [
        _row(f"https://openalex.org/W{r['work_id']}", r["year"], r["date"], topic=r["topic"], subfield=r["subfield"], field=r["field"],
             authors=tuple(authors[r["work_id"]]), source=f"https://openalex.org/S{r['source']}")
        for r in w.to_pylist()
    ]
    rows.append(_row("https://openalex.org/W999999", 2005, dt.date(2005, 3, 1), field=17))
    files = []
    for index, chunk in enumerate((rows[::2], rows[1::2])):
        path = tmp_path / f"part_{index:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk, schema=SCHEMA), path)
        files.append((f"fixture/part_{index:04d}.parquet", path))
    return files


def test_execute_end_to_end_is_deterministic(tmp_path: Path) -> None:
    sources = _raw_files(tmp_path)
    _w, _a, guard = _structure_fixture()
    guard_path = tmp_path / "guard.parquet"
    pq.write_table(pa.table({"author_numeric_id": pa.array(guard, pa.uint64())}), guard_path)
    precommit = {"commit_sha": "0" * 40}
    receipts = []
    for name in ("first", "second"):
        receipts.append(s1.execute(CONTRACT, s1.DEFAULT_CONFIG, sources, guard_path, tmp_path / name, precommit, {"status": "FIXTURE"}, "f" * 64,
                                   workers=2, threads=2, started=0.0))
    assert receipts[0] == receipts[1]
    for name in ("s1_summary.json", "private/parts_digest.json", "private/episodes.parquet", "private/entrants.parquet"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    summary = json.loads((tmp_path / "first" / "s1_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "STOP_INSUFFICIENT_EPISODES"
    assert summary["failure_metrics"] == {key: 0 for key in summary["failure_metrics"]}
    assert summary["source_accounting"]["rows"] == summary["source_accounting"]["registered_field_rows"] + 1
    assert summary["entrants"]["g4_guard_removed"] == 2 and summary["entrants"]["discovery_author_removed"] > 0
    assert summary["persistence_labels_computed"] == 0
    assert 0 < summary["entrants"]["precise_share_overall"] < 1
    assert summary["entrants"]["episodes_without_entrants"] >= 0
    assert not (tmp_path / "first" / "private" / "parts").exists()
    assert oct((tmp_path / "first").stat().st_mode & 0o777) == "0o700"
    text = (tmp_path / "first" / "s1_summary.json").read_text(encoding="utf-8")
    assert "https://openalex.org" not in text


def test_execute_stops_on_duplicate_work_ids(tmp_path: Path) -> None:
    rows = [_row("https://openalex.org/W5", 2005, dt.date(2005, 5, 3))]
    files = []
    for index in range(2):
        path = tmp_path / f"dup_{index}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
        files.append((f"dup_{index}", path))
    guard_path = tmp_path / "guard.parquet"
    pq.write_table(pa.table({"author_numeric_id": pa.array([1], pa.uint64())}), guard_path)
    receipt = s1.execute(CONTRACT, s1.DEFAULT_CONFIG, files, guard_path, tmp_path / "out", {"commit_sha": "0" * 40}, {}, "f" * 64,
                         workers=1, threads=1, started=0.0)
    assert receipt["status"] == "STOP_SOURCE_INVALID"
    summary = json.loads((tmp_path / "out" / "s1_summary.json").read_text(encoding="utf-8"))
    assert summary["failure_metrics"]["duplicate_numeric_work_id_rows"] == 1


def test_execute_stops_when_rows_read_differ_from_the_footers(tmp_path: Path) -> None:
    path = tmp_path / "one.parquet"
    pq.write_table(pa.Table.from_pylist([_row("https://openalex.org/W5", 2005, dt.date(2005, 5, 3))], schema=SCHEMA), path)
    guard_path = tmp_path / "guard.parquet"
    pq.write_table(pa.table({"author_numeric_id": pa.array([1], pa.uint64())}), guard_path)
    receipt = s1.execute(CONTRACT, s1.DEFAULT_CONFIG, [("one", path)], guard_path, tmp_path / "out", {"commit_sha": "0" * 40},
                         {"rows_by_footer": 2}, "f" * 64, workers=1, threads=1, started=0.0)
    assert receipt["status"] == "STOP_SOURCE_INVALID"
    summary = json.loads((tmp_path / "out" / "s1_summary.json").read_text(encoding="utf-8"))
    assert summary["failure_metrics"]["rows_read_differ_from_footer_rows"] == 1
