from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_pivot_empirical_ladder_risk_sets_v1.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
n11b = importlib.import_module(SCRIPT.stem)
CONTRACT = n11b.load_contract()
S1 = n11b.load_s1(CONTRACT)
N10 = S1.load_n10(n11b.s1_contract(CONTRACT, S1))
GUARD_COLUMN = n11b.s1_contract(CONTRACT, S1)["build"]["g4_guard"]["column"]

D = dt.date
# work_id: (date, topic, subfield, field, authors). Topic 100 is episode 0's topic, topic 200 episode 1's.
WORKS = {
    10: (D(2003, 3, 10), 300, 1605, 16, [1]),
    11: (D(2004, 6, 1), 400, 1606, 16, [1]),
    12: (D(2005, 5, 10), 100, 1605, 16, [1]),        # A1 enters topic 100, precise, offset 1
    13: (D(2006, 7, 1), 100, 1605, 16, [1, 4]),      # A1's follow-up: must not count anywhere
    20: (D(2004, 2, 2), 300, 1605, 16, [2]),
    21: (D(2006, 1, 1), 100, 1605, 16, [2]),         # A2 enters, imprecise, response year 1
    30: (D(2002, 8, 8), 300, 1605, 16, [3]),
    40: (D(2003, 9, 9), 100, 1605, 16, [4]),         # A4 is a topic-100 incumbent
    41: (D(2004, 4, 4), 300, 1605, 16, [4, 5, 7]),
    42: (D(2005, 7, 1), 100, 1605, 16, [4]),         # incumbent-only topic work, offset 2
    50: (D(2005, 9, 9), 300, 1605, 16, [5]),
    60: (D(2004, 5, 5), 300, 1605, 16, [6]),
    61: (D(2005, 10, 10), 100, 1605, 16, [6]),       # guarded A6 enters, offset 3
    70: (D(2005, 2, 15), 100, 1605, 16, [7]),        # A7 enters, offset 0
    80: (D(2004, 3, 3), 300, 1605, 16, [8, 6]),      # A8's coauthor is the guarded entrant
    90: (D(2005, 11, 11), 300, 1605, 16, [9]),
    91: (D(2006, 3, 3), 300, 1605, 16, [9]),
    92: (D(2001, 1, 1), 500, 3201, 32, [9]),         # career age sees field 32
    94: (D(2006, 1, 1), 300, 1605, 16, []),          # no linked author: counts only in field works
    100: (D(2005, 5, 5), 400, 1606, 16, [10]),
    101: (D(2007, 8, 1), 200, 1606, 16, [10]),       # A10 enters topic 200, offset 6
}
EPISODES = [
    {"episode_id": 0, "field": 16, "topic": 100, "subfield": 1605, "t": 2005, "partition": "discovery", "screen": "RETAINED", "meets_floor": True},
    {"episode_id": 1, "field": 16, "topic": 200, "subfield": 1606, "t": 2006, "partition": "confirmatory", "screen": "RETAINED", "meets_floor": True},
]
ENTRANTS = [
    (0, 1, D(2005, 5, 10), 1, True),
    (0, 2, D(2006, 1, 1), 4, False),
    (0, 7, D(2005, 2, 15), 0, True),
    (1, 10, D(2007, 8, 1), 6, True),
]
GUARDED = [6]


def _fixture(tmp_path: Path, entrants=ENTRANTS) -> tuple[dict, Path]:  # noqa: ANN001
    works = pa.table({
        "work_id": pa.array(list(WORKS), pa.uint64()),
        "year": pa.array([v[0].year for v in WORKS.values()], pa.int16()),
        "date": pa.array([v[0] for v in WORKS.values()], pa.date32()),
        "topic": pa.array([v[1] for v in WORKS.values()], pa.uint32()),
        "subfield": pa.array([v[2] for v in WORKS.values()], pa.uint32()),
        "field": pa.array([v[3] for v in WORKS.values()], pa.uint8()),
    })
    pairs = sorted((w, a) for w, v in WORKS.items() for a in v[4])
    authorships = pa.table({"work_id": pa.array([p[0] for p in pairs], pa.uint64()), "author_id": pa.array([p[1] for p in pairs], pa.uint64())})
    entrants_table = pa.table({
        "episode_id": pa.array([e[0] for e in entrants], pa.int64()), "author_id": pa.array([e[1] for e in entrants], pa.uint64()),
        "entry_date": pa.array([e[2] for e in entrants], pa.date32()), "offset": pa.array([e[3] for e in entrants], pa.int64()),
        "precise": pa.array([e[4] for e in entrants], pa.bool_()),
    })
    paths = {}
    for name, table in (("works", works), ("authorships", authorships), ("episodes", pa.Table.from_pylist(EPISODES)), ("entrants", entrants_table)):
        paths[name] = tmp_path / f"{name}.parquet"
        pq.write_table(table, paths[name])
    guard = tmp_path / "guard.parquet"
    pq.write_table(pa.table({GUARD_COLUMN: pa.array(GUARDED, pa.uint64())}), guard)
    return paths, guard


def _everyone() -> dict:
    contract = copy.deepcopy(CONTRACT)
    contract["sampling"]["floor_authors"] = 1000
    return contract


def _build(tmp_path: Path, contract=None, entrants=ENTRANTS) -> dict:  # noqa: ANN001
    paths, guard = _fixture(tmp_path, entrants)
    return n11b.build(paths, guard, GUARD_COLUMN, contract or _everyone(), N10, threads=2)


def _rows(built: dict, episode: int, author: int) -> list[dict]:
    return [r for r in built["panel"].to_pylist() if r["episode_id"] == episode and r["author_id"] == author]


# -- contract ----------------------------------------------------------------------------


def test_the_contract_binds_n11a_s1_and_the_brief() -> None:
    assert CONTRACT["next_step_id"] == "N11b"
    assert CONTRACT["outcome_boundary"]["persistence_labels_computed"] == 0
    assert CONTRACT["outcome_boundary"]["visible_evidence_computed"] == 0
    assert len(N10.v1.verify_source_bindings(CONTRACT)) == len(CONTRACT["source_bindings"]) == 7
    assert sorted(CONTRACT["inputs"]["files"]) == sorted(n11b.INPUTS.values())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("precondition", "n11a_status"), "IDENTIFICATION_FAILS", "N11a"),
        (("inputs", "s1_root"), "/ssd/elsewhere", "S1 root"),
        (("inputs", "s1_receipt_sha256"), "0" * 64, "S1 receipt"),
        (("sampling", "salt_opened"), [20261012], "fresh"),
        (("outcome_boundary", "persistence_labels_computed"), 1, "persistence"),
        (("authorization", "outcomes"), "AUTHORIZED", "outcomes"),
    ],
)
def test_contract_mutations_are_refused(path: tuple[str, str], value: object, message: str) -> None:
    contract = copy.deepcopy(CONTRACT)
    contract[path[0]][path[1]] = value
    with pytest.raises(n11b.N11bError, match=message):
        n11b.validate_contract(contract)


def test_the_authorization_must_have_the_registered_schema_and_scope(tmp_path: Path) -> None:
    contract_path = n11b.DEFAULT_CONFIG
    good = {
        "schema_version": "pivot_empirical_ladder_risk_sets_authorization.v1", "status": "AUTHORIZED_N11B_RISK_SET_BUILD",
        "n11b_contract_sha256": n11b._sha256(contract_path), "n11b_builder_sha256": n11b._sha256(SCRIPT),
        "human_authorization": {"user_instruction": "x"}, "outcomes": "NOT_AUTHORIZED", "visible_evidence": "NOT_AUTHORIZED",
        "model_fitting": "NOT_AUTHORIZED",
    }
    for change, message in ((("status", "PENDING"), "not authorized"), (("visible_evidence", "AUTHORIZED"), "visible_evidence"),
                            (("n11b_builder_sha256", "0" * 64), "another N11b builder")):
        path = tmp_path / f"{change[0]}.json"
        path.write_text(json.dumps({**good, change[0]: change[1]}), encoding="utf-8")
        with pytest.raises(n11b.N11bError, match=message):
            n11b.load_authorization(path, contract_path, S1)


# -- sampling ----------------------------------------------------------------------------


def test_splitmix64_matches_the_reference_vector() -> None:
    assert int(n11b.splitmix64(np.array([0], np.uint64))[0]) == 0xE220A8397B1DCDAF


def test_the_draw_is_deterministic_uniform_and_episode_specific() -> None:
    authors = np.arange(1, 200_001, dtype=np.uint64)
    first = n11b.uniform_draw(authors, np.zeros(authors.size, np.int64), 20261012)
    assert np.array_equal(first, n11b.uniform_draw(authors, np.zeros(authors.size, np.int64), 20261012))
    assert 0.0 <= first.min() and first.max() < 1.0
    assert abs((first < 0.3).mean() - 0.3) < 0.005
    other = n11b.uniform_draw(authors, np.ones(authors.size, np.int64), 20261012)
    assert abs(np.corrcoef(first, other)[0, 1]) < 0.01


def test_sampling_probability_uses_the_floor_and_the_ratio() -> None:
    assert n11b.sampling_probability(1_000_000, 171, 2000, 20) == pytest.approx(3420 / 1_000_000)
    assert n11b.sampling_probability(1_000_000, 10, 2000, 20) == pytest.approx(2000 / 1_000_000)
    assert n11b.sampling_probability(1500, 10, 2000, 20) == 1.0
    assert n11b.sampling_probability(0, 0, 2000, 20) == 1.0


def test_the_subcohort_follows_the_draw_and_every_entrant_is_kept(tmp_path: Path) -> None:
    contract = copy.deepcopy(CONTRACT)
    contract["sampling"]["floor_authors"] = 1
    contract["sampling"]["ratio_per_entrant"] = 1
    built = _build(tmp_path, contract)
    assert built["status"] == "N11B_PASS"
    sampling = {r["episode_id"]: r for r in built["episodes_sampling"].to_pylist()}
    assert sampling[0]["at_risk_authors_union"] == 8 and sampling[0]["sampling_probability"] == pytest.approx(3 / 8)
    panel_authors = {(r["episode_id"], r["author_id"]): r["in_subcohort"] for r in built["panel"].to_pylist()}
    for episode, members in ((0, [1, 2, 3, 5, 7, 8, 9, 10]), (1, [4, 5, 8, 9, 10])):
        draw = n11b.uniform_draw(np.array(members, np.uint64), np.full(len(members), episode, np.int64), CONTRACT["sampling"]["salt"])
        for author, u in zip(members, draw, strict=True):
            chosen = bool(u < sampling[episode]["sampling_probability"])
            entrant = any(e[0] == episode and e[1] == author for e in ENTRANTS)
            assert ((episode, author) in panel_authors) == (chosen or entrant)
            if (episode, author) in panel_authors:
                assert panel_authors[(episode, author)] == chosen


# -- risk sets, events, features ---------------------------------------------------------


def test_the_risk_set_reproduces_the_s1_entrants_and_their_exits(tmp_path: Path) -> None:
    built = _build(tmp_path)
    assert built["status"] == "N11B_PASS" and not any(built["invariants"].values())
    assert [(r["quarter"], r["event"]) for r in _rows(built, 0, 1)] == [(0, 0), (1, 1)]
    assert [(r["quarter"], r["event"]) for r in _rows(built, 0, 7)] == [(0, 1)]
    assert [(r["quarter"], r["event"]) for r in _rows(built, 0, 2)] == [(q, 0) for q in range(4)] + [(q, 2) for q in range(4, 8)]
    assert [(r["quarter"], r["event"]) for r in _rows(built, 1, 10)] == [(q, 0) for q in range(6)] + [(6, 1)]
    episodes = built["panel"]["episode_id"].to_numpy()
    assert int((episodes == 0).sum()) == 55 and int((episodes == 1).sum()) == 51


def test_eligibility_windows_prior_topic_work_guard_and_discovery_exclusion(tmp_path: Path) -> None:
    built = _build(tmp_path)
    authors = {e: {r["author_id"] for r in built["panel"].to_pylist() if r["episode_id"] == e} for e in (0, 1)}
    assert authors[0] == {1, 2, 3, 5, 7, 8, 9, 10}
    assert authors[1] == {4, 5, 8, 9, 10}  # A1, A2, A7 are discovery entrants; A6 is guarded
    assert [r["quarter"] for r in _rows(built, 0, 3)] == [0, 1, 2, 3]  # active only in 2002
    assert [r["quarter"] for r in _rows(built, 0, 9)] == list(range(4, 12))  # first active in 2005
    assert [r["quarter"] for r in _rows(built, 1, 8)] == list(range(8))  # last active in 2004
    sampling = {r["episode_id"]: r for r in built["episodes_sampling"].to_pylist()}
    assert (sampling[0]["guard_removed_authors"], sampling[1]["guard_removed_authors"]) == (1, 1)
    assert (sampling[0]["discovery_excluded_authors"], sampling[1]["discovery_excluded_authors"]) == (0, 3)
    assert built["counts"]["prior_topic_excluded_author_years"] == 11


def test_features_read_only_the_three_years_before_the_response_year(tmp_path: Path) -> None:
    built = _build(tmp_path)
    a5 = {r["quarter"]: r for r in _rows(built, 0, 5)}
    assert (a5[0]["productivity_works"], a5[0]["subfield_works"], a5[0]["career_age_years"], a5[0]["incumbent_coauthors"]) == (1, 1, 1, 1)
    assert (a5[4]["productivity_works"], a5[4]["subfield_works"]) == (2, 2)  # the 2005 work enters only from 2006
    a9 = {r["quarter"]: r for r in _rows(built, 0, 9)}
    assert (a9[4]["productivity_works"], a9[4]["career_age_years"]) == (1, 5)  # field 32 sets career age
    assert a9[8]["productivity_works"] == 2  # the 2006 work enters only from 2007
    a4 = {r["quarter"]: r for r in _rows(built, 1, 4)}
    assert (a4[0]["productivity_works"], a4[0]["subfield_works"], a4[0]["incumbent_coauthors"]) == (3, 0, 0)


def test_coauthor_entries_become_visible_after_the_entry_quarter(tmp_path: Path) -> None:
    built = _build(tmp_path)
    a5 = [r["coauthor_entries_visible"] for r in _rows(built, 0, 5)]
    assert a5 == [0] + [1] * 11  # A7 entered in quarter 0
    a8 = [r["coauthor_entries_visible"] for r in _rows(built, 0, 8)]
    assert a8 == [0] * 4 + [1] * 8  # the guarded A6 entered in quarter 3 and is still a source


def test_the_environment_counts_incumbent_only_works_and_visible_entries(tmp_path: Path) -> None:
    built = _build(tmp_path)
    env = [r for r in built["environment"].to_pylist() if r["episode_id"] == 0]
    assert [r["incumbent_works_prev_quarter"] for r in env] == [0, 0, 0, 1] + [0] * 8
    assert [r["incumbent_works_prev_year"] for r in env] == [0] * 4 + [1] * 4 + [0] * 4  # A1's 2006 follow-up is excluded
    assert [r["entries_visible"] for r in env] == [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    assert [env[q]["at_risk_authors_full"] for q in (0, 4, 8)] == [6, 5, 4]
    assert (env[0]["field_works_year"], env[0]["field_precise_works_year"]) == (3, 3)  # topic-100 works are excluded
    assert (env[4]["field_works_year"], env[4]["field_precise_works_year"]) == (2, 1)


def test_a_follow_up_topic_work_changes_nothing(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    base = _build(tmp_path / "a")
    saved = dict(WORKS)
    try:
        WORKS[14] = (D(2005, 12, 1), 100, 1605, 16, [1, 7])  # a second topic work by two entrants
        WORKS[15] = (D(2006, 2, 2), 100, 1605, 16, [2])
        (tmp_path / "b").mkdir()
        followed = _build(tmp_path / "b")
    finally:
        WORKS.clear()
        WORKS.update(saved)
    for name in ("panel", "environment", "episodes_sampling"):
        assert followed[name].equals(base[name]), name


# -- invariants and identity -------------------------------------------------------------


def test_an_entrant_mismatch_stops_the_build(tmp_path: Path) -> None:
    missing = _build(tmp_path, entrants=[e for e in ENTRANTS if e[1] != 7])
    assert missing["status"] == "STOP_INVARIANT_FAILED"
    assert missing["invariants"]["risk_set_entrants_not_in_s1"] == 1
    (tmp_path / "b").mkdir()
    shifted = _build(tmp_path / "b", entrants=[(0, 1, D(2005, 5, 11), 1, True), *ENTRANTS[1:]])
    assert shifted["invariants"]["entry_date_mismatches"] == 1


def _identity(output: Path) -> dict[str, str]:
    return {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ["n11b_receipt.json", *n11b.IDENTITY] if (output / name).exists()}


def test_two_runs_are_byte_identical_and_the_summary_has_no_identifier(tmp_path: Path) -> None:
    paths, guard = _fixture(tmp_path)
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        receipt = n11b.execute(_everyone(), n11b.DEFAULT_CONFIG, paths, guard, GUARD_COLUMN, output, {"commit_sha": "0" * 40},
                               {"status": "fixture"}, "a" * 64, S1, N10, threads=2, started=time.monotonic())
        assert receipt["status"] == "N11B_PASS"
        outputs.append(_identity(output))
    assert outputs[0] == outputs[1] and len(outputs[0]) == 5
    summary = json.loads((tmp_path / "first" / "n11b_summary.json").read_text(encoding="utf-8"))
    assert summary["events"]["precise_event_rows"] == 3 and summary["events"]["imprecise_entrants"] == 1
    assert summary["sampling"]["entrants_total"] == 4
    text = json.dumps(summary)
    assert "author_id" not in text and "entrant_vs" not in text
    assert oct((tmp_path / "first" / "private" / "panel.parquet").stat().st_mode & 0o777) == "0o600"
