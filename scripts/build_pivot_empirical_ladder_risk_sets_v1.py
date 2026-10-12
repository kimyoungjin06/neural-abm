#!/usr/bin/env python3
"""Stage N11b of the empirical model ladder: risk sets and pre-entry features.

It reads the private S1 extract only. For every S1 episode it forms the risk
set of field-active authors over the twelve response quarters with the S1
eligibility rule, the G4 guard, and the discovery-author exclusion; draws a
case-cohort sample (a hash subcohort plus every S1 entrant); and writes
integer pre-entry features, coauthor entry exposure, and the shared episode
environment. Every S1 entrant must come back as exactly one event. No
follow-up topic work, persistence label, or visible evidence is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_empirical_ladder_risk_sets_v1.yaml"
S1_BUILDER_ROLE = "S1_BUILDER_REUSED_BY_HASH_BOUND_IMPORT"
S1_CONTRACT_ROLE = "S1_CONTRACT_EXTRACT_ENTRANT_RULES_AND_G4_GUARD_IDENTITY"
S1_RESULT_ROLE = "S1_RECORDED_RESULT_RECEIPT_HASH_AND_BYTE_IDENTITY"
N11A_ROLE = "N11A_V2_IDENTIFICATION_PASSES_PRECONDITION"
INPUTS = {"works": "private/works.parquet", "authorships": "private/authorships.parquet",
          "episodes": "private/episodes.parquet", "entrants": "private/entrants.parquet"}
QUARTERS = 12
PANEL_SCHEMA = pa.schema([
    ("episode_id", pa.int16()), ("author_id", pa.uint64()), ("quarter", pa.int8()), ("response_year", pa.int8()),
    ("in_subcohort", pa.bool_()), ("entrant", pa.bool_()), ("event", pa.int8()),
    ("productivity_works", pa.int32()), ("subfield_works", pa.int32()), ("career_age_years", pa.int16()),
    ("incumbent_coauthors", pa.int32()), ("coauthor_entries_visible", pa.int32()),
])
ENVIRONMENT_SCHEMA = pa.schema([
    ("episode_id", pa.int16()), ("quarter", pa.int8()), ("response_year", pa.int8()),
    ("incumbent_works_prev_quarter", pa.int32()), ("incumbent_works_prev_year", pa.int32()), ("entries_visible", pa.int32()),
    ("at_risk_authors_full", pa.int64()), ("field_works_year", pa.int64()), ("field_precise_works_year", pa.int64()),
])
SAMPLING_SCHEMA = pa.schema([
    ("episode_id", pa.int16()), ("field", pa.int16()), ("topic", pa.int64()), ("subfield", pa.int64()), ("t", pa.int16()),
    ("partition", pa.string()), ("meets_floor", pa.bool_()), ("at_risk_authors_union", pa.int64()), ("s1_entrants", pa.int64()),
    ("sampling_probability", pa.float64()), ("subcohort_authors", pa.int64()), ("subcohort_entrants", pa.int64()),
    ("panel_rows", pa.int64()), ("guard_removed_authors", pa.int64()), ("discovery_excluded_authors", pa.int64()),
])
IDENTITY = ["n11b_summary.json", "private/panel.parquet", "private/environment.parquet", "private/episodes_sampling.parquet"]


class N11bError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise N11bError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(contract: Mapping[str, Any], role: str) -> Path:
    matches = [b for b in contract["source_bindings"] if b["role"] == role]
    require(len(matches) == 1, f"exactly one source binding must have role {role}")
    return ROOT / matches[0]["path"]


# -- contract ----------------------------------------------------------------------------


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(contract, dict), "the N11b contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("schema_version") == "pivot_empirical_ladder_risk_sets.v1", "unexpected N11b schema")
    require(contract.get("status") == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION", "the N11b contract is not a precommit design")
    for binding in contract["source_bindings"]:
        require(_sha256(ROOT / binding["path"]) == binding["sha256"], f"source binding drifted: {binding['path']}")
    precondition = contract["precondition"]
    n11a = json.loads(_binding(contract, N11A_ROLE).read_text(encoding="utf-8"))
    require(n11a["status"] == precondition["n11a_status"], "N11a v2 did not pass identification")
    s1_result = json.loads(_binding(contract, S1_RESULT_ROLE).read_text(encoding="utf-8"))
    require(s1_result["status"] == precondition["s1_status"], "S1 did not pass")
    inputs = contract["inputs"]
    require(s1_result["execution"]["receipt_sha256"] == inputs["s1_receipt_sha256"], "the S1 receipt differs from its record")
    require(s1_result["execution"]["runs"]["development"] == inputs["s1_root"], "the S1 root differs from its record")
    require(sorted(inputs["files"]) == sorted(INPUTS.values()), "the bound S1 files differ from the registered inputs")
    require(all(s1_result["execution"]["byte_identical"][name] for name in inputs["files"]), "an S1 input was not reproduced")
    require(set(inputs["receipt_binds"]) <= set(inputs["files"]), "receipt-bound files must be bound inputs")
    require(contract["clock"]["quarters"] == QUARTERS, "the clock must have twelve quarters")
    sampling = contract["sampling"]
    require(int(sampling["floor_authors"]) >= 1 and float(sampling["ratio_per_entrant"]) > 0, "sampling sizes must be positive")
    require(isinstance(sampling["salt"], int) and sampling["salt_opened"] == "NONE", "the sampling salt must be fresh")
    require(contract["execution"]["identity_artifacts"] == [*IDENTITY[:1], "n11b_receipt.json", *IDENTITY[1:]], "identity artifacts differ")
    require(contract["outcome_boundary"]["persistence_labels_computed"] == 0, "N11b may not compute persistence")
    require(contract["authorization"]["outcomes"] == "NOT_AUTHORIZED", "N11b may not read outcomes")


def load_s1(contract: Mapping[str, Any]) -> ModuleType:
    """The S1 builder by hash-bound import; its contract and N10 functions are checked by S1 itself."""

    spec = importlib.util.spec_from_file_location("s1_builder_for_n11b", _binding(contract, S1_BUILDER_ROLE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def s1_contract(contract: Mapping[str, Any], s1: ModuleType) -> dict[str, Any]:
    return s1.load_contract(_binding(contract, S1_CONTRACT_ROLE))


# -- preflight ---------------------------------------------------------------------------


def preflight(contract: Mapping[str, Any], s1: ModuleType) -> dict[str, Any]:
    """Row-free: input hashes, the S1 receipt binding, guard identity, capacity."""

    inputs = contract["inputs"]
    root = Path(inputs["s1_root"])
    receipt_ok = _sha256(root / "s1_receipt.json") == inputs["s1_receipt_sha256"]
    receipt = json.loads((root / "s1_receipt.json").read_text(encoding="utf-8"))
    receipt_binding_ok = all(receipt["outputs"].get(name) == inputs["files"][name] for name in inputs["receipt_binds"])
    drift = [name for name, digest in sorted(inputs["files"].items()) if _sha256(root / name) != digest]
    guard = s1_contract(contract, s1)["build"]["g4_guard"]
    guard_path = Path(guard["path"])
    guard_ok = _sha256(guard_path) == guard["sha256"] and pq.ParquetFile(guard_path).metadata.num_rows == guard["rows"]
    capacity = contract["capacity"]
    free = shutil.disk_usage("/ssd").free
    required = int(capacity["byte_budget"]) + int(capacity["safety_reserve_bytes"])
    passed = receipt_ok and receipt_binding_ok and not drift and guard_ok and free >= required
    return {
        "status": "PASS_N11B_PREFLIGHT_HASHES_ONLY" if passed else "STOP_INPUT_DRIFT",
        "s1_receipt_identity": receipt_ok,
        "s1_receipt_binds_inputs": receipt_binding_ok,
        "input_drift_files": drift,
        "g4_guard_identity": guard_ok,
        "free_bytes": free,
        "required_free_bytes": required,
    }


# -- sampling ----------------------------------------------------------------------------


def splitmix64(values: np.ndarray) -> np.ndarray:
    """The SplitMix64 finalizer on uint64 values, wrapping modulo 2**64."""

    z = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def uniform_draw(author_ids: np.ndarray, episode_ids: np.ndarray, salt: int) -> np.ndarray:
    """A uniform in [0, 1) from the author id and the episode only: top 53 bits over 2**53."""

    keys = splitmix64(np.asarray(episode_ids, dtype=np.uint64) + np.uint64(salt))
    mixed = splitmix64(np.asarray(author_ids, dtype=np.uint64) ^ keys)
    return (mixed >> np.uint64(11)).astype(np.float64) * 2.0**-53


def sampling_probability(population: int, entrants: int, floor: int, ratio: float) -> float:
    if population <= 0:
        return 1.0
    return min(1.0, max(float(floor), float(ratio) * float(entrants)) / float(population))


# -- build -------------------------------------------------------------------------------


def _offset(column: str, t: str) -> str:
    return f"(4 * year({column}) + (month({column}) - 1) // 3 - 4 * {t})"


def _precise(column: str) -> str:
    return f"(NOT (month({column}) = 1 AND day({column}) = 1))"


def _visible_from(column: str, t: str) -> str:
    """Precise entries are visible from the next quarter, imprecise ones from the next year's first quarter."""

    return f"(CASE WHEN {_precise(column)} THEN {_offset(column, t)} + 1 ELSE 4 * (year({column}) - {t} + 1) END)"


def _scalar(con: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(con.execute(query).fetchone()[0] or 0)


def build(inputs: Mapping[str, Path], guard_path: Path, guard_column: str, contract: Mapping[str, Any], n10: ModuleType, *, threads: int = 8) -> dict[str, Any]:
    """Everything after the checks. Returns the private tables, public counts, and invariant counts."""

    sampling = contract["sampling"]
    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    con.execute("CREATE TABLE w AS SELECT work_id, year, date, topic, subfield, field FROM read_parquet(?)", [str(inputs["works"])])
    con.execute("CREATE TABLE a AS SELECT work_id, author_id FROM read_parquet(?)", [str(inputs["authorships"])])
    con.execute("CREATE TABLE ep AS SELECT episode_id, field, topic, subfield, t, partition, meets_floor FROM read_parquet(?)", [str(inputs["episodes"])])
    con.execute('CREATE TABLE s1e AS SELECT episode_id, author_id, entry_date, "offset", precise FROM read_parquet(?)', [str(inputs["entrants"])])
    con.execute(f'CREATE TABLE guard AS SELECT DISTINCT "{guard_column}" AS author_id FROM read_parquet(?)', [str(guard_path)])

    con.execute("CREATE TABLE aw AS SELECT a.author_id, w.work_id, w.topic, w.subfield, w.field, w.year, w.date FROM a JOIN w USING (work_id)")
    con.execute("CREATE TABLE first_any AS SELECT author_id, MIN(year) AS first_year FROM aw GROUP BY author_id")
    con.execute("CREATE TABLE field_years AS SELECT DISTINCT author_id, field, year FROM aw")
    # Topic works enter only as each author's earliest date in an episode topic (and, below, as incumbent-only counts).
    con.execute("CREATE TABLE ft AS SELECT author_id, topic, MIN(date) AS first_date FROM aw WHERE topic IN (SELECT DISTINCT topic FROM ep) GROUP BY author_id, topic")
    con.execute("CREATE TABLE ep_years AS SELECT episode_id, field, topic, subfield, t, partition, k AS ry, t + k AS y FROM ep, range(3) AS r(k)")
    con.execute("CREATE TABLE discovery_authors AS SELECT DISTINCT s.author_id FROM s1e s JOIN ep e USING (episode_id) WHERE e.partition = 'discovery'")

    # Risk set: the S1 pivot-eligibility rule applied to every author, no prior topic work, guard, exclusion.
    con.execute("CREATE TABLE ep_window AS SELECT episode_id, ry, field, y - d AS year FROM ep_years, range(1, 4) AS r(d)")
    con.execute(
        """
        CREATE TABLE eligible AS
        SELECT DISTINCT ew.episode_id, ew.ry, fy.author_id FROM ep_window ew JOIN field_years fy USING (field, year)
        """
    )
    con.execute(
        """
        CREATE TABLE no_prior AS
        SELECT el.episode_id, el.ry, el.author_id, ft.first_date
        FROM eligible el JOIN ep_years ey USING (episode_id, ry)
        LEFT JOIN ft ON ft.author_id = el.author_id AND ft.topic = ey.topic
        WHERE ft.first_date IS NULL OR year(ft.first_date) >= ey.y
        """
    )
    con.execute("CREATE TABLE after_guard AS SELECT n.* FROM no_prior n ANTI JOIN guard g ON n.author_id = g.author_id")
    con.execute(
        """
        CREATE TABLE risk AS
        SELECT n.* FROM after_guard n JOIN ep e USING (episode_id)
        WHERE NOT (e.partition = 'confirmatory' AND n.author_id IN (SELECT author_id FROM discovery_authors))
        """
    )
    removal = {
        int(row[0]): (int(row[1]), int(row[2]))
        for row in con.execute(
            """
            WITH n AS (SELECT episode_id, COUNT(DISTINCT author_id) AS c FROM no_prior GROUP BY episode_id),
                 g AS (SELECT episode_id, COUNT(DISTINCT author_id) AS c FROM after_guard GROUP BY episode_id),
                 r AS (SELECT episode_id, COUNT(DISTINCT author_id) AS c FROM risk GROUP BY episode_id)
            SELECT e.episode_id, COALESCE(n.c, 0) - COALESCE(g.c, 0), COALESCE(g.c, 0) - COALESCE(r.c, 0)
            FROM ep e LEFT JOIN n USING (episode_id) LEFT JOIN g USING (episode_id) LEFT JOIN r USING (episode_id)
            ORDER BY e.episode_id
            """
        ).fetchall()
    }
    counts: dict[str, Any] = {
        "eligible_author_years": _scalar(con, "SELECT COUNT(*) FROM eligible"),
        "prior_topic_excluded_author_years": _scalar(con, "SELECT (SELECT COUNT(*) FROM eligible) - (SELECT COUNT(*) FROM no_prior)"),
        "guard_removed_author_episodes": sum(v[0] for v in removal.values()),
        "discovery_excluded_author_episodes": sum(v[1] for v in removal.values()),
        "at_risk_author_years_by_response_year": [int(n) for (n,) in con.execute(
            "SELECT COUNT(r.ry) FROM range(3) AS k(ry) LEFT JOIN risk r ON r.ry = k.ry GROUP BY k.ry ORDER BY k.ry").fetchall()],
    }

    # The risk set must reproduce the S1 entrants exactly.
    con.execute(
        """
        CREATE TABLE risk_entrants AS
        SELECT r.episode_id, r.author_id, r.first_date, ey.t FROM risk r JOIN ep_years ey USING (episode_id, ry)
        WHERE r.first_date IS NOT NULL AND year(r.first_date) = ey.y
        """
    )
    invariants = {
        "s1_entrants_missing_from_risk_set": _scalar(con, "SELECT COUNT(*) FROM s1e ANTI JOIN risk_entrants USING (episode_id, author_id)"),
        "risk_set_entrants_not_in_s1": _scalar(con, "SELECT COUNT(*) FROM risk_entrants ANTI JOIN s1e USING (episode_id, author_id)"),
        "entry_date_mismatches": _scalar(con, "SELECT COUNT(*) FROM s1e JOIN risk_entrants USING (episode_id, author_id) WHERE s1e.entry_date <> risk_entrants.first_date"),
        "entry_clock_mismatches": _clock_mismatches(con, n10),
    }

    # Case-cohort sample: a hash subcohort per episode plus every S1 entrant.
    sizes = con.execute(
        """
        SELECT e.episode_id, COALESCE(u.n, 0), COALESCE(s.n, 0) FROM ep e
        LEFT JOIN (SELECT episode_id, COUNT(DISTINCT author_id) AS n FROM risk GROUP BY episode_id) u USING (episode_id)
        LEFT JOIN (SELECT episode_id, COUNT(*) AS n FROM s1e GROUP BY episode_id) s USING (episode_id)
        ORDER BY e.episode_id
        """
    ).fetchall()
    population = {int(e): int(u) for e, u, _n in sizes}
    entrant_counts = {int(e): int(n) for e, _u, n in sizes}
    probability = {e: sampling_probability(population[e], entrant_counts[e], int(sampling["floor_authors"]), float(sampling["ratio_per_entrant"])) for e in population}
    lookup = np.zeros(max(probability) + 1 if probability else 1, dtype=np.float64)
    for episode, value in probability.items():
        lookup[episode] = value
    chosen_episodes, chosen_authors = [], []
    reader = con.execute("SELECT DISTINCT episode_id, author_id FROM risk").to_arrow_reader(4_000_000)
    for batch in reader:
        episodes = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        authors = batch.column(1).to_numpy(zero_copy_only=False).astype(np.uint64)
        keep = uniform_draw(authors, episodes, int(sampling["salt"])) < lookup[episodes]
        chosen_episodes.append(episodes[keep])
        chosen_authors.append(authors[keep])
    subcohort = pa.table({
        "episode_id": pa.array(np.concatenate(chosen_episodes) if chosen_episodes else np.array([], np.int64), pa.int64()),
        "author_id": pa.array(np.concatenate(chosen_authors) if chosen_authors else np.array([], np.uint64), pa.uint64()),
    })
    con.register("subcohort_arrow", subcohort)
    con.execute(
        """
        CREATE TABLE sample AS
        SELECT episode_id, author_id, bool_or(sub) AS in_subcohort, bool_or(ent) AS entrant FROM (
            SELECT episode_id, author_id, TRUE AS sub, FALSE AS ent FROM subcohort_arrow
            UNION ALL SELECT episode_id, author_id, FALSE AS sub, TRUE AS ent FROM s1e
        ) GROUP BY episode_id, author_id
        """
    )
    con.execute(
        f"""
        CREATE TABLE sy AS
        SELECT r.episode_id, r.author_id, r.ry, ey.y, ey.t, ey.field, ey.subfield, ey.topic, s.in_subcohort, s.entrant,
               COALESCE(year(r.first_date) = ey.y, FALSE) AS entry_year,
               {_offset("r.first_date", "ey.t")} AS entry_offset,
               COALESCE({_precise("r.first_date")}, FALSE) AS precise
        FROM risk r JOIN sample s USING (episode_id, author_id) JOIN ep_years ey USING (episode_id, ry)
        """
    )
    con.execute(
        """
        CREATE TABLE rows AS
        SELECT sy.episode_id, sy.author_id, sy.ry, q.q AS quarter, sy.in_subcohort, sy.entrant,
               CASE WHEN sy.entry_year AND sy.precise AND q.q = sy.entry_offset THEN 1
                    WHEN sy.entry_year AND NOT sy.precise THEN 2 ELSE 0 END AS event
        FROM sy, range(12) AS q(q)
        WHERE q.q // 4 = sy.ry AND NOT (sy.entry_year AND sy.precise AND q.q > sy.entry_offset)
        """
    )

    # Features: integer counts from works dated in years y-3..y-1 of the row's response year.
    con.execute(
        """
        CREATE TABLE win AS
        SELECT k.episode_id, k.author_id, k.ry, aw.work_id,
               aw.field = k.field AS in_field, aw.field = k.field AND aw.subfield = k.subfield AS in_subfield
        FROM (SELECT DISTINCT episode_id, author_id, ry, y, field, subfield FROM sy) k
        JOIN aw ON aw.author_id = k.author_id AND aw.year BETWEEN k.y - 3 AND k.y - 1
        """
    )
    con.execute(
        """
        CREATE TABLE coauth AS
        SELECT DISTINCT wn.episode_id, wn.author_id, wn.ry, a.author_id AS coauthor_id, ft.first_date, ey.t
        FROM win wn JOIN ep_years ey USING (episode_id, ry)
        JOIN a ON a.work_id = wn.work_id AND a.author_id <> wn.author_id
        JOIN ft ON ft.author_id = a.author_id AND ft.topic = ey.topic
        """
    )
    con.execute(
        f"""
        CREATE TABLE coauth_ent AS
        SELECT episode_id, author_id, ry, coauthor_id, {_visible_from("first_date", "t")} AS v
        FROM coauth WHERE year(first_date) BETWEEN t AND t + 2
        """
    )
    con.execute(
        """
        CREATE TABLE feat AS
        SELECT k.episode_id, k.author_id, k.ry,
               COALESCE(p.productivity_works, 0) AS productivity_works, COALESCE(p.subfield_works, 0) AS subfield_works,
               k.y - fa.first_year AS career_age_years, COALESCE(c.incumbent_coauthors, 0) AS incumbent_coauthors
        FROM (SELECT DISTINCT episode_id, author_id, ry, y FROM sy) k
        JOIN first_any fa USING (author_id)
        LEFT JOIN (
            SELECT episode_id, author_id, ry,
                   COUNT(DISTINCT work_id) FILTER (WHERE in_field) AS productivity_works,
                   COUNT(DISTINCT work_id) FILTER (WHERE in_subfield) AS subfield_works
            FROM win GROUP BY episode_id, author_id, ry
        ) p USING (episode_id, author_id, ry)
        LEFT JOIN (
            SELECT episode_id, author_id, ry, COUNT(DISTINCT coauthor_id) AS incumbent_coauthors
            FROM coauth WHERE year(first_date) < t GROUP BY episode_id, author_id, ry
        ) c USING (episode_id, author_id, ry)
        """
    )
    con.execute(
        """
        CREATE TABLE expo AS
        SELECT r.episode_id, r.author_id, r.quarter, COUNT(c.coauthor_id) AS coauthor_entries_visible
        FROM rows r LEFT JOIN coauth_ent c
          ON c.episode_id = r.episode_id AND c.author_id = r.author_id AND c.ry = r.ry AND c.v <= r.quarter
        GROUP BY r.episode_id, r.author_id, r.quarter
        """
    )
    panel = con.execute(
        """
        SELECT r.episode_id, r.author_id, r.quarter, r.ry AS response_year, r.in_subcohort, r.entrant, r.event,
               f.productivity_works, f.subfield_works, f.career_age_years, f.incumbent_coauthors, x.coauthor_entries_visible
        FROM rows r JOIN feat f USING (episode_id, author_id, ry) JOIN expo x USING (episode_id, author_id, quarter)
        ORDER BY r.episode_id, r.author_id, r.quarter
        """
    ).to_arrow_table().cast(PANEL_SCHEMA).combine_chunks()

    invariants.update(_panel_invariants(con, panel))
    environment = _environment(con)
    per_episode = con.execute(
        """
        WITH s AS (
            SELECT episode_id, COUNT(*) FILTER (WHERE in_subcohort) AS sub, COUNT(*) FILTER (WHERE in_subcohort AND entrant) AS sub_ent
            FROM sample GROUP BY episode_id
        ), r AS (SELECT episode_id, COUNT(*) AS n FROM rows GROUP BY episode_id)
        SELECT e.episode_id, COALESCE(s.sub, 0), COALESCE(s.sub_ent, 0), COALESCE(r.n, 0)
        FROM ep e LEFT JOIN s USING (episode_id) LEFT JOIN r USING (episode_id) ORDER BY e.episode_id
        """
    ).fetchall()
    episodes = con.execute("SELECT episode_id, field, topic, subfield, t, partition, meets_floor FROM ep ORDER BY episode_id").fetchall()
    sampled = {int(e): (int(a), int(b), int(c)) for e, a, b, c in per_episode}
    episodes_sampling = pa.table({
        "episode_id": [int(e[0]) for e in episodes], "field": [int(e[1]) for e in episodes], "topic": [int(e[2]) for e in episodes],
        "subfield": [int(e[3]) for e in episodes], "t": [int(e[4]) for e in episodes], "partition": [e[5] for e in episodes],
        "meets_floor": [bool(e[6]) for e in episodes],
        "at_risk_authors_union": [population[int(e[0])] for e in episodes], "s1_entrants": [entrant_counts[int(e[0])] for e in episodes],
        "sampling_probability": [probability[int(e[0])] for e in episodes],
        "subcohort_authors": [sampled[int(e[0])][0] for e in episodes], "subcohort_entrants": [sampled[int(e[0])][1] for e in episodes],
        "panel_rows": [sampled[int(e[0])][2] for e in episodes],
        "guard_removed_authors": [removal[int(e[0])][0] for e in episodes], "discovery_excluded_authors": [removal[int(e[0])][1] for e in episodes],
    }).cast(SAMPLING_SCHEMA)
    counts["imprecise_entrants_by_response_year"] = [int(n) for (n,) in con.execute(
        'SELECT COUNT(s."offset") FROM range(3) AS k(ry) LEFT JOIN s1e s ON s."offset" // 4 = k.ry AND NOT s.precise GROUP BY k.ry ORDER BY k.ry').fetchall()]
    con.close()
    status = "N11B_PASS" if not any(invariants.values()) else "STOP_INVARIANT_FAILED"
    return {"status": status, "panel": panel, "environment": environment, "episodes_sampling": episodes_sampling,
            "counts": counts, "invariants": dict(sorted(invariants.items()))}


def _clock_mismatches(con: duckdb.DuckDBPyConnection, n10: ModuleType, sample: int = 2000) -> int:
    """The SQL clock against the hash-bound N10 v2 entry clock, on a deterministic sample of risk-set entrants."""

    rows = con.execute(
        f"""
        SELECT first_date, t, {_offset("first_date", "t")}, {_precise("first_date")}
        FROM risk_entrants ORDER BY episode_id, author_id
        """
    ).fetchall()
    step = max(1, len(rows) // sample)
    mismatches = 0
    for first_date, t, offset, precise in rows[::step]:
        clock = n10.entry_clock([first_date], int(t))
        mismatches += int((clock["offset"], clock["precise"]) != (int(offset), bool(precise)))
    return mismatches


def _panel_invariants(con: duckdb.DuckDBPyConnection, panel: pa.Table) -> dict[str, int]:
    con.register("panel_arrow", panel)
    return {
        "entrants_without_exactly_one_event": _scalar(con, """
            SELECT COUNT(*) FROM s1e LEFT JOIN (
                SELECT episode_id, author_id, COUNT(*) FILTER (WHERE event = 1) AS n1, COUNT(*) FILTER (WHERE event = 2) AS n2
                FROM panel_arrow GROUP BY episode_id, author_id
            ) p USING (episode_id, author_id)
            WHERE NOT COALESCE((s1e.precise AND p.n1 = 1 AND p.n2 = 0) OR (NOT s1e.precise AND p.n1 = 0 AND p.n2 = 4), FALSE)
        """),
        "event_rows_outside_s1_entrants": _scalar(con, "SELECT COUNT(*) FROM panel_arrow p ANTI JOIN s1e USING (episode_id, author_id) WHERE p.event > 0"),
        "entrant_flags_differ_from_s1": _scalar(con, """
            SELECT COUNT(*) FROM (SELECT DISTINCT episode_id, author_id, entrant FROM panel_arrow) p
            LEFT JOIN s1e USING (episode_id, author_id) WHERE p.entrant <> (s1e.author_id IS NOT NULL)
        """),
        "guarded_panel_authors": _scalar(con, "SELECT COUNT(DISTINCT p.author_id) FROM panel_arrow p SEMI JOIN guard g ON p.author_id = g.author_id"),
        "discovery_entrants_in_confirmatory_panels": _scalar(con, """
            SELECT COUNT(DISTINCT p.author_id) FROM panel_arrow p JOIN ep e USING (episode_id)
            WHERE e.partition = 'confirmatory' AND p.author_id IN (SELECT author_id FROM discovery_authors)
        """),
        "rows_after_exit": _scalar(con, """
            SELECT COUNT(*) FROM panel_arrow p JOIN s1e USING (episode_id, author_id)
            WHERE p.quarter > CASE WHEN s1e.precise THEN s1e."offset" ELSE 4 * (s1e."offset" // 4) + 3 END
        """),
        "rows_not_eligible_in_their_response_year": _scalar(con, """
            SELECT COUNT(*) FROM panel_arrow p ANTI JOIN eligible el
              ON el.episode_id = p.episode_id AND el.ry = p.response_year AND el.author_id = p.author_id
        """),
        "rows_without_a_field_work_in_the_window": _scalar(con, "SELECT COUNT(*) FROM panel_arrow WHERE productivity_works < 1"),
    }


def _environment(con: duckdb.DuckDBPyConnection) -> pa.Table:
    con.execute(
        """
        CREATE TABLE tw AS
        SELECT e.episode_id, e.t, w.work_id, w.year, w.date, MAX(year(ft.first_date)) AS m
        FROM ep e JOIN w ON w.topic = e.topic AND w.year BETWEEN e.t - 1 AND e.t + 2
        JOIN a ON a.work_id = w.work_id
        JOIN ft ON ft.author_id = a.author_id AND ft.topic = e.topic
        GROUP BY e.episode_id, e.t, w.work_id, w.year, w.date
        """
    )
    return con.execute(
        f"""
        WITH grid AS (SELECT e.episode_id, e.field, e.topic, e.t, q.q AS quarter FROM ep e, range(12) AS q(q)),
        iq AS (SELECT episode_id, {_offset("date", "t")} AS off, COUNT(*) AS n FROM tw WHERE m < t AND {_precise("date")} GROUP BY ALL),
        iy AS (SELECT episode_id, year - t AS rel, COUNT(*) AS n FROM tw WHERE m < t GROUP BY ALL),
        ev AS (
            SELECT e.episode_id, {_visible_from("ft.first_date", "e.t")} AS v
            FROM ep e JOIN ft ON ft.topic = e.topic AND year(ft.first_date) BETWEEN e.t AND e.t + 2
        ),
        ar AS (SELECT episode_id, ry, COUNT(*) AS n FROM risk GROUP BY ALL),
        fw AS (
            SELECT e.episode_id, w.year - e.t AS ry, COUNT(*) AS n, COUNT(*) FILTER (WHERE {_precise("w.date")}) AS np
            FROM ep e JOIN w ON w.field = e.field AND w.year BETWEEN e.t AND e.t + 2
            WHERE w.topic NOT IN (SELECT DISTINCT topic FROM ep) GROUP BY ALL
        )
        SELECT g.episode_id, g.quarter, g.quarter // 4 AS response_year,
               COALESCE(iq.n, 0) AS incumbent_works_prev_quarter,
               COALESCE(iy.n, 0) AS incumbent_works_prev_year,
               (SELECT COUNT(*) FROM ev WHERE ev.episode_id = g.episode_id AND ev.v <= g.quarter) AS entries_visible,
               COALESCE(ar.n, 0) AS at_risk_authors_full,
               COALESCE(fw.n, 0) AS field_works_year,
               COALESCE(fw.np, 0) AS field_precise_works_year
        FROM grid g
        LEFT JOIN iq ON iq.episode_id = g.episode_id AND iq.off = g.quarter - 1
        LEFT JOIN iy ON iy.episode_id = g.episode_id AND iy.rel = g.quarter // 4 - 1
        LEFT JOIN ar ON ar.episode_id = g.episode_id AND ar.ry = g.quarter // 4
        LEFT JOIN fw ON fw.episode_id = g.episode_id AND fw.ry = g.quarter // 4
        ORDER BY g.episode_id, g.quarter
        """
    ).to_arrow_table().cast(ENVIRONMENT_SCHEMA).combine_chunks()


# -- summary -----------------------------------------------------------------------------


def public_summary(built: Mapping[str, Any], s1: ModuleType) -> dict[str, Any]:
    """Counts and pooled subcohort quantiles only: no identifier and no entrant-versus-non-entrant contrast."""

    panel, sampling = built["panel"], built["episodes_sampling"]
    quarter = panel["quarter"].to_numpy()
    event = panel["event"].to_numpy()
    subcohort = panel["in_subcohort"].to_numpy(zero_copy_only=False)
    first_row = subcohort & (quarter % 4 == 0)
    productivity = panel["productivity_works"].to_numpy()[first_row].astype(float)
    subfield = panel["subfield_works"].to_numpy()[first_row].astype(float)
    incumbent = panel["incumbent_coauthors"].to_numpy()[first_row]
    probabilities = sampling["sampling_probability"].to_pylist()
    return {
        "risk_set": {
            **{k: built["counts"][k] for k in ("eligible_author_years", "prior_topic_excluded_author_years", "guard_removed_author_episodes",
                                                 "discovery_excluded_author_episodes", "at_risk_author_years_by_response_year")},
            "at_risk_authors_union_total": int(sum(sampling["at_risk_authors_union"].to_pylist())),
            "at_risk_authors_union_by_episode": s1._quantiles(sampling["at_risk_authors_union"].to_pylist()),
        },
        "sampling": {
            "sampling_probability_by_episode": s1._quantiles(probabilities),
            "episodes_with_probability_1": sum(1 for p in probabilities if p >= 1.0),
            "subcohort_authors_total": int(sum(sampling["subcohort_authors"].to_pylist())),
            "subcohort_entrants_total": int(sum(sampling["subcohort_entrants"].to_pylist())),
            "entrants_total": int(sum(sampling["s1_entrants"].to_pylist())),
            "panel_rows": panel.num_rows,
            "subcohort_rows": int(subcohort.sum()),
        },
        "events": {
            "precise_event_rows": int((event == 1).sum()),
            "imprecise_entrants": int(((event == 2) & (quarter % 4 == 0)).sum()),
            "precise_events_by_quarter": [int(((event == 1) & (quarter == q)).sum()) for q in range(QUARTERS)],
            "imprecise_entrants_by_response_year": built["counts"]["imprecise_entrants_by_response_year"],
        },
        "subcohort_author_years_pooled": {
            "author_years": int(first_row.sum()),
            "productivity_works": s1._quantiles(productivity.tolist()),
            "subfield_share": s1._quantiles((subfield / np.maximum(productivity, 1.0)).tolist()),
            "career_age_years": s1._quantiles(panel["career_age_years"].to_numpy()[first_row].astype(float).tolist()),
            "incumbent_coauthors": s1._quantiles(incumbent.astype(float).tolist()),
            "share_with_an_incumbent_coauthor": float((incumbent > 0).mean()) if incumbent.size else None,
            "share_of_subcohort_rows_with_a_visible_coauthor_entry": float((panel["coauthor_entries_visible"].to_numpy()[subcohort] > 0).mean()) if subcohort.any() else None,
        },
    }


# -- run ---------------------------------------------------------------------------------


def load_authorization(path: Path, contract_path: Path, s1: ModuleType) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    require(authorization.get("schema_version") == "pivot_empirical_ladder_risk_sets_authorization.v1", "unexpected authorization schema")
    require(authorization.get("status") == "AUTHORIZED_N11B_RISK_SET_BUILD", "N11b is not authorized")
    require(authorization["n11b_contract_sha256"] == _sha256(contract_path), "the authorization names another N11b contract")
    require(authorization["n11b_builder_sha256"] == _sha256(Path(__file__)), "the authorization names another N11b builder")
    require(bool(authorization["human_authorization"]["user_instruction"]), "the authorization does not quote the user")
    for scope in ("outcomes", "visible_evidence", "model_fitting"):
        require(authorization[scope] == "NOT_AUTHORIZED", f"N11b may not authorize {scope}")
    relative = path.resolve().relative_to(ROOT).as_posix()
    require(relative in s1._git("ls-files", "--", relative).split(), "the authorization file is not committed")
    require(not s1._git("status", "--porcelain", "--", relative), "the authorization file is dirty")
    return {**authorization, "_file_sha256": _sha256(path)}


def execute(
    contract: Mapping[str, Any], contract_path: Path, inputs: Mapping[str, Path], guard_path: Path, guard_column: str, output_dir: Path,
    precommit: Mapping[str, Any], preflight_record: Mapping[str, Any], authorization_sha256: str, s1: ModuleType, n10: ModuleType,
    *, threads: int, started: float,
) -> dict[str, Any]:
    """The N11b build after its checks: risk sets, sample, features, environment, outputs."""

    timing: dict[str, float] = {"preflight_seconds": time.monotonic() - started}
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    stage = time.monotonic()
    built = build(inputs, guard_path, guard_column, contract, n10, threads=threads)
    timing["build_seconds"] = time.monotonic() - stage
    summary: dict[str, Any] = {
        "schema_version": "pivot_empirical_ladder_risk_sets_summary.v1",
        "status": built["status"],
        "precommit_sha": precommit["commit_sha"],
        "preflight": dict(preflight_record),
        "invariants": built["invariants"],
        "persistence_labels_computed": 0,
        "visible_evidence_computed": 0,
        "follow_up_counts_computed": 0,
    }
    if built["status"] == "N11B_PASS":
        private = output_dir / "private"
        private.mkdir(mode=0o700)
        s1._write(built["panel"], private / "panel.parquet")
        s1._write(built["environment"], private / "environment.parquet")
        s1._write(built["episodes_sampling"], private / "episodes_sampling.parquet")
        used = s1._dir_bytes(output_dir)
        require(used <= int(contract["capacity"]["byte_budget"]), f"the output exceeds the byte budget: {used}")
        summary.update(public_summary(built, s1))
    else:
        summary["sampling"] = {"entrants_total": int(sum(built["episodes_sampling"]["s1_entrants"].to_pylist()))}
    s1._write_json(output_dir / "n11b_summary.json", summary)
    timing["total_seconds"] = time.monotonic() - started
    s1._write_json(output_dir / "n11b_timing.json", {"note": "NOT_AN_IDENTITY_ARTIFACT", **{k: round(v, 1) for k, v in timing.items()}})
    receipt = {
        "schema_version": "pivot_empirical_ladder_risk_sets_receipt.v1",
        "status": built["status"],
        "contract_sha256": _sha256(contract_path),
        "builder_sha256": _sha256(Path(__file__)),
        "authorization_sha256": authorization_sha256,
        "precommit": dict(precommit),
        "inputs": dict(sorted(contract["inputs"]["files"].items())),
        "outputs": {name: _sha256(output_dir / name) for name in IDENTITY if (output_dir / name).exists()},
        "persistence_labels_computed": 0,
        "visible_evidence_computed": 0,
    }
    s1._write_json(output_dir / "n11b_receipt.json", receipt)
    return receipt


def run(contract: Mapping[str, Any], contract_path: Path, authorization_path: Path, output_dir: Path, precommit: Mapping[str, Any],
        s1: ModuleType, n10: ModuleType, *, threads: int) -> dict[str, Any]:
    authorization = load_authorization(authorization_path, contract_path, s1)
    started = time.monotonic()
    check = preflight(contract, s1)
    require(check["status"] == "PASS_N11B_PREFLIGHT_HASHES_ONLY", f"N11b preflight failed: {check}")
    root = Path(contract["inputs"]["s1_root"])
    guard = s1_contract(contract, s1)["build"]["g4_guard"]
    return execute(contract, contract_path, {key: root / name for key, name in INPUTS.items()}, Path(guard["path"]), guard["column"],
                   output_dir, precommit, {k: v for k, v in check.items() if k != "free_bytes"}, authorization["_file_sha256"],
                   s1, n10, threads=threads, started=started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--precommit-sha")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    contract = load_contract(config_path)
    s1 = load_s1(contract)
    if args.command == "preflight":
        print(json.dumps(preflight(contract, s1), indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or args.authorization is None or args.output_dir is None:
        parser.error("run requires --precommit-sha, --authorization, and --output-dir")
    n10 = s1.load_n10(s1_contract(contract, s1))
    precommit = n10.v1.verify_precommit(args.precommit_sha, contract)
    receipt = run(contract, config_path, args.authorization.resolve(), args.output_dir.resolve(), precommit, s1, n10,
                  threads=int(contract["execution"]["duckdb_threads"]))
    print(json.dumps({"status": receipt["status"], "outputs": receipt["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
