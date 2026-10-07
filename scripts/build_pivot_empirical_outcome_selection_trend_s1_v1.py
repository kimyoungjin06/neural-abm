#!/usr/bin/env python3
"""Stage S1 of the N10 v2 preregistration: the structural build.

It reads column-limited Works rows of the complete 2026-06-26 OpenAlex
snapshot, checks identifiers and publication dates over every row, and keeps
the eligible works of the three registered fields through 2017. From that
extract it computes incumbent-only attention, surges, episodes, the indexing
screen, the discovery and confirmatory partition, and every episode's
entrants after the G4 author guard and the discovery-author exclusion. It
computes no persistence label. The surge, episode, partition, and clock rules
are the hash-bound N10 functions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "pivot_empirical_outcome_selection_trend_s1_build_v1.yaml"
ABI = ROOT / "experiments" / "configs" / "pivot_feedback_d0_measurement_abi_v1.yaml"
G4_SUMMARY = ROOT / "experiments" / "evidence" / "results" / "pivot_feedback_f_g4_private_membership_materialization_v1_20260804.summary.json"
TRANSFER_AUTHORIZATION = ROOT / "experiments" / "configs" / "pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v3.json"
LEAVES = {
    "id": ("id", pa.string()),
    "publication_year": ("publication_year", pa.int32()),
    "publication_date": ("publication_date", pa.date32()),
    "type": ("type", pa.string()),
    "is_paratext": ("is_paratext", pa.bool_()),
    "is_retracted": ("is_retracted", pa.bool_()),
    "is_xpac": ("is_xpac", pa.bool_()),
    "topic_id": ("primary_topic.id", pa.string()),
    "topic_score": ("primary_topic.score", None),
    "subfield_id": ("primary_topic.subfield.id", pa.string()),
    "field_id": ("primary_topic.field.id", pa.string()),
    "source_id": ("primary_location.source.id", pa.string()),
    "author_id": ("authorships.list.element.author.id", pa.string()),
}
PREFIXES = {
    "work": "https://openalex.org/W",
    "author": "https://openalex.org/A",
    "topic": "https://openalex.org/T",
    "subfield": "https://openalex.org/subfields/",
    "field": "https://openalex.org/fields/",
    "source": "https://openalex.org/S",
}
FAILURE_COUNTS = (
    "null_work_id",
    "malformed_work_id",
    "publication_exactly_one_null",
    "publication_date_year_mismatch",
    "malformed_field_id",
    "registered_malformed_topic_id",
    "registered_malformed_subfield_id",
    "registered_malformed_source_id",
    "registered_malformed_author_id",
)


class S1Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise S1Error(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -- contract ----------------------------------------------------------------------------


def load_n10(contract: Mapping[str, Any]) -> ModuleType:
    path = ROOT / contract["implements"]["functions"]
    require(_sha256(path) == contract["implements"]["functions_sha256"], "the N10 v2 functions drifted from their bound hash")
    spec = importlib.util.spec_from_file_location("n10_v2_for_s1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(contract, dict), "the S1 contract must be a mapping")
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("schema_version") == "pivot_empirical_outcome_selection_trend_s1_build.v1", "unexpected S1 schema")
    require(contract.get("status") == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION", "the S1 contract is not a precommit design")
    implements = contract["implements"]
    n10_path = ROOT / implements["contract"]
    require(_sha256(n10_path) == implements["contract_sha256"], "the N10 v2 contract drifted from its bound hash")
    n10 = yaml.safe_load(n10_path.read_text(encoding="utf-8"))
    extract = contract["extract"]
    require(extract["fields"] == n10["fields"]["ids"], "S1 fields differ from N10")
    require(extract["latest_year"] == n10["years"]["latest_outcome_year"], "S1 latest year differs from N10")
    require(contract["build"]["attention"]["decision_years"] == n10["years"]["decision_years"], "S1 decision years differ from N10")
    abi = yaml.safe_load(ABI.read_text(encoding="utf-8"))["work_type_dictionary"]
    require([abi[code] for code in extract["type_codes_from_abi"]] == extract["type_names"], "type names differ from the ABI codes")
    require(extract["type_codes_from_abi"] == n10["work_eligibility"]["type_codes"], "type codes differ from N10")
    transfer = json.loads(TRANSFER_AUTHORIZATION.read_text(encoding="utf-8"))
    require(contract["source"]["plan_sha256"] == transfer["stall_record"]["plan_sha256"], "source plan differs from the transfer record")
    g4 = json.loads(G4_SUMMARY.read_text(encoding="utf-8"))["production_artifact"]
    guard = contract["build"]["g4_guard"]
    require(guard["path"] == f"{g4['root']}/{g4['author_cold_guard_basename']}", "G4 guard path differs from its record")
    require((guard["sha256"], guard["rows"]) == (g4["author_cold_guard_sha256"], g4["author_cold_guard_rows"]), "G4 guard identity differs")
    for column, (leaf, _type) in LEAVES.items():
        require(contract["columns"][column] == leaf, f"column {column} differs from the registered leaf")


def n10_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return yaml.safe_load((ROOT / contract["implements"]["contract"]).read_text(encoding="utf-8"))


# -- source identity and footers ---------------------------------------------------------


def works_files(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = contract["source"]
    root = Path(source["root"])
    require(_sha256(root / "download_plan.json") == source["plan_sha256"], "download plan hash drift")
    require(_sha256(root / "download_state.json") == source["terminal_state_sha256"], "terminal state hash drift")
    require(_sha256(root / "download_receipt.json") == source["terminal_receipt_sha256"], "terminal receipt hash drift")
    state = json.loads((root / "download_state.json").read_text(encoding="utf-8"))
    require(state["status"] == "COMPLETE", "the source transfer is not complete")
    files = [
        {"relative_path": relative, "bytes": int(entry["expected_bytes"]), "final_stat": entry["final_stat"]}
        for relative, entry in sorted(state["files"].items())
        if PurePosixPath(relative).parts[2] == "works"
    ]
    require(len(files) == int(source["works_files"]), "works file count drift")
    return files


def check_file_identity(root: Path, files: Sequence[Mapping[str, Any]]) -> list[str]:
    drift = []
    for item in files:
        status = (root / item["relative_path"]).stat()
        bound = item["final_stat"]
        actual = {"device": status.st_dev, "inode": status.st_ino, "mode": status.st_mode & 0o7777, "mtime_ns": status.st_mtime_ns, "size": status.st_size}
        if any(actual[key] != bound[key] for key in ("device", "inode", "mtime_ns", "size")) or status.st_size != item["bytes"]:
            drift.append(item["relative_path"])
    return drift


def _arrow_leaf_type(schema: pa.Schema, leaf: str) -> pa.DataType | None:
    parts = leaf.split(".")
    if parts[0] not in schema.names:
        return None
    current = schema.field(parts[0]).type
    for name in parts[1:]:
        if name in ("list", "element") and pa.types.is_list(current):
            if name == "element":
                current = current.value_type
            continue
        if not pa.types.is_struct(current) or current.get_field_index(name) < 0:
            return None
        current = current.field(name).type
    return current


def footer_check(path: Path) -> dict[str, Any]:
    """Footer-only: every registered leaf exists with its registered type. No row value is read."""

    parquet = pq.ParquetFile(path)
    paths = {parquet.schema.column(index).path for index in range(len(parquet.schema))}
    missing, mistyped = [], []
    for column, (leaf, expected) in LEAVES.items():
        if leaf not in paths:
            missing.append(column)
            continue
        actual = _arrow_leaf_type(parquet.schema_arrow, leaf)
        ok = actual is not None and (pa.types.is_floating(actual) if expected is None else actual == expected)
        if not ok:
            mistyped.append(column)
    return {"rows": parquet.metadata.num_rows, "missing": missing, "mistyped": mistyped}


def preflight(contract: Mapping[str, Any], *, workers: int = 32) -> dict[str, Any]:
    """Network-free and row-free: source identity, footers, guard identity, capacity."""

    root = Path(contract["source"]["root"])
    files = works_files(contract)
    drift = check_file_identity(root, files)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        footers = list(pool.map(footer_check, [root / item["relative_path"] for item in files]))
    bad = [files[i]["relative_path"] for i, footer in enumerate(footers) if footer["missing"] or footer["mistyped"]]
    guard = contract["build"]["g4_guard"]
    guard_path = Path(guard["path"])
    guard_ok = _sha256(guard_path) == guard["sha256"] and pq.ParquetFile(guard_path).metadata.num_rows == guard["rows"]
    capacity = contract["capacity"]
    free = shutil.disk_usage("/ssd").free
    required = int(capacity["byte_budget"]) + int(capacity["safety_reserve_bytes"])
    passed = not drift and not bad and guard_ok and free >= required
    return {
        "status": "PASS_S1_PREFLIGHT_FOOTERS_ONLY" if passed else "STOP_S1_PREFLIGHT",
        "works_files": len(files),
        "rows_by_footer": sum(footer["rows"] for footer in footers),
        "identity_drift_files": drift,
        "schema_violation_files": bad,
        "g4_guard_identity": guard_ok,
        "free_bytes": free,
        "required_free_bytes": required,
    }


# -- extraction --------------------------------------------------------------------------


def _numeric(values: pa.Array, prefix: str) -> tuple[pa.Array, int]:
    """Validate `prefix` + digits and return uint64 numbers (null where null) and the malformed count."""

    pattern = "^" + prefix.replace(".", "[.]") + "(0|[1-9][0-9]*)$"
    valid = pc.match_substring_regex(values, pattern)
    malformed = int(pc.sum(pc.and_(pc.is_valid(values), pc.invert(pc.fill_null(valid, False)))).as_py() or 0)
    digits = pc.utf8_slice_codeunits(pc.if_else(pc.fill_null(valid, False), values, pa.scalar(None, pa.string())), len(prefix))
    return pc.cast(digits, pa.uint64()), malformed


def _ipc_digest(table: pa.Table) -> str:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def _write(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path, compression="zstd", row_group_size=1_000_000)
    path.chmod(0o600)


def extract_file(task: Mapping[str, Any]) -> dict[str, Any]:
    """One Works file: global accounting over every row, and the registered extract parts."""

    path = Path(task["path"])
    parquet = pq.ParquetFile(path)
    table = parquet.read(columns=[leaf for leaf, _type in LEAVES.values()], use_threads=False)
    rows = table.num_rows
    counts: dict[str, int] = {"rows": rows}

    ids = table["id"].combine_chunks()
    quarantine = pc.is_in(ids, value_set=pa.array(task["work_quarantine"], pa.string()))
    counts["quarantined_work_id"] = int(pc.sum(quarantine).as_py() or 0)
    counts["null_work_id"] = ids.null_count
    kept_ids = pc.if_else(quarantine, pa.scalar(None, pa.string()), ids)
    work, malformed = _numeric(kept_ids, PREFIXES["work"])
    counts["malformed_work_id"] = malformed

    year = table["publication_year"].combine_chunks()
    date = table["publication_date"].combine_chunks()
    year_null, date_null = pc.is_null(year), pc.is_null(date)
    counts["publication_exactly_one_null"] = int(pc.sum(pc.xor(year_null, date_null)).as_py() or 0)
    both = pc.and_(pc.is_valid(year), pc.is_valid(date))
    date_year = pc.year(date)
    mismatch = pc.and_(both, pc.not_equal(pc.cast(date_year, pa.int32()), year))
    counts["publication_date_year_mismatch"] = int(pc.sum(mismatch).as_py() or 0)
    snapshot = dt.date.fromisoformat(task["snapshot_date"])
    pre = pc.and_(both, pc.less(date, pa.scalar(dt.date(1900, 1, 1), pa.date32())))
    usable2 = pc.and_(pc.and_(both, pc.invert(pre)), pc.less(year, snapshot.year))
    usable3 = pc.and_(pc.and_(both, pc.equal(year, snapshot.year)), pc.less_equal(date, pa.scalar(snapshot, pa.date32())))
    counts["disposition_0"] = int(pc.sum(pc.and_(year_null, date_null)).as_py() or 0)
    counts["disposition_1"] = int(pc.sum(pre).as_py() or 0)
    counts["disposition_2"] = int(pc.sum(usable2).as_py() or 0)
    counts["disposition_3"] = int(pc.sum(usable3).as_py() or 0)
    counts["disposition_4"] = int(pc.sum(pc.and_(pc.and_(both, pc.equal(year, snapshot.year)), pc.greater(date, pa.scalar(snapshot, pa.date32())))).as_py() or 0)
    counts["disposition_5"] = int(pc.sum(pc.and_(both, pc.greater(year, snapshot.year))).as_py() or 0)

    topic_struct = table["primary_topic"].combine_chunks()
    field_raw = pc.struct_field(pc.struct_field(topic_struct, "field"), "id")
    field, counts["malformed_field_id"] = _numeric(field_raw, PREFIXES["field"])
    in_field = pc.fill_null(pc.is_in(field, value_set=pa.array(task["fields"], pa.uint64())), False)
    counts["registered_field_rows"] = int(pc.sum(in_field).as_py() or 0)

    usable = pc.fill_null(pc.and_(pc.or_(usable2, usable3), pc.less_equal(year, task["latest_year"])), False)
    types = pc.fill_null(pc.is_in(table["type"].combine_chunks(), value_set=pa.array(task["type_names"], pa.string())), False)
    paratext = table["is_paratext"].combine_chunks()
    retracted = table["is_retracted"].combine_chunks()
    not_paratext = pc.invert(pc.fill_null(paratext, False))
    not_retracted = pc.invert(pc.fill_null(retracted, False))
    retained = pc.and_(pc.is_valid(work), in_field)
    counts["registered_excluded_by_date_or_year"] = int(pc.sum(pc.and_(retained, pc.invert(usable))).as_py() or 0)
    keep = pc.and_(retained, usable)
    counts["registered_excluded_by_type"] = int(pc.sum(pc.and_(keep, pc.invert(types))).as_py() or 0)
    keep = pc.and_(keep, types)
    counts["registered_excluded_paratext"] = int(pc.sum(pc.and_(keep, pc.invert(not_paratext))).as_py() or 0)
    keep = pc.and_(keep, not_paratext)
    counts["registered_excluded_retracted"] = int(pc.sum(pc.and_(keep, pc.invert(not_retracted))).as_py() or 0)
    keep = pc.and_(keep, not_retracted)
    counts["extract_works"] = int(pc.sum(keep).as_py() or 0)
    counts["extract_paratext_null"] = int(pc.sum(pc.and_(keep, pc.is_null(paratext))).as_py() or 0)
    counts["extract_retracted_null"] = int(pc.sum(pc.and_(keep, pc.is_null(retracted))).as_py() or 0)
    xpac = table["is_xpac"].combine_chunks()
    counts["extract_xpac"] = int(pc.sum(pc.and_(keep, pc.fill_null(xpac, False))).as_py() or 0)

    kept = table.filter(keep)
    topic_kept = kept["primary_topic"].combine_chunks()
    topic, counts["registered_malformed_topic_id"] = _numeric(pc.struct_field(topic_kept, "id"), PREFIXES["topic"])
    subfield, counts["registered_malformed_subfield_id"] = _numeric(
        pc.struct_field(pc.struct_field(topic_kept, "subfield"), "id"), PREFIXES["subfield"]
    )
    counts["registered_malformed_topic_id"] += topic.null_count
    counts["registered_malformed_subfield_id"] += subfield.null_count
    source, counts["registered_malformed_source_id"] = _numeric(
        pc.struct_field(pc.struct_field(kept["primary_location"].combine_chunks(), "source"), "id"), PREFIXES["source"]
    )
    works = pa.table(
        {
            "work_id": work.filter(keep),
            "year": pc.cast(year.filter(keep), pa.int16()),
            "date": date.filter(keep),
            "topic": pc.cast(topic, pa.uint32()),
            "subfield": pc.cast(subfield, pa.uint32()),
            "field": pc.cast(field.filter(keep), pa.uint8()),
            "score": pc.cast(pc.struct_field(topic_kept, "score"), pa.float32()),
            "source": source,
            "is_xpac": xpac.filter(keep),
            "retracted_null": pc.is_null(retracted.filter(keep)),
        }
    )
    authorships = kept["authorships"].combine_chunks()
    flat = pc.list_flatten(authorships)
    parents = pc.list_parent_indices(authorships)
    author_raw = pc.struct_field(pc.struct_field(flat, "author"), "id")
    counts["extract_null_author_ids"] = author_raw.null_count
    author, counts["registered_malformed_author_id"] = _numeric(author_raw, PREFIXES["author"])
    pairs = pa.table({"work_id": pc.take(works["work_id"], parents), "author_id": author}).filter(pc.is_valid(author))
    pairs = pairs.group_by(["work_id", "author_id"]).aggregate([]).sort_by([("work_id", "ascending"), ("author_id", "ascending")])
    counts["extract_authorships"] = pairs.num_rows
    works = works.sort_by([("work_id", "ascending")])
    all_ids = pa.table({"work_id": work.filter(pc.is_valid(work))}).sort_by([("work_id", "ascending")])

    part_dir = Path(task["part_dir"])
    stem = f"{int(task['index']):04d}"
    _write(works, part_dir / f"works_{stem}.parquet")
    _write(pairs, part_dir / f"authorships_{stem}.parquet")
    _write(all_ids, part_dir / f"ids_{stem}.parquet")
    return {
        "index": int(task["index"]),
        "relative_path": task["relative_path"],
        "counts": counts,
        "digests": {"works": _ipc_digest(works), "authorships": _ipc_digest(pairs), "ids": _ipc_digest(all_ids)},
    }


# -- build -------------------------------------------------------------------------------


def attention_rows(con: duckdb.DuckDBPyConnection, first: int, last: int) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Incumbent-only work counts per field x decision year, as rows for the N10 surge function."""

    con.execute(
        """
        CREATE OR REPLACE TABLE work_m AS
        SELECT aw.work_id, aw.topic, aw.field, aw.year, MAX(ft.first_year) AS m
        FROM aw JOIN first_topic ft USING (author_id, topic)
        GROUP BY aw.work_id, aw.topic, aw.field, aw.year
        """
    )
    counts = con.execute(
        f"""
        SELECT t.t AS t, field, topic, year - t.t AS rel, COUNT(*) AS n
        FROM work_m, range({first}, {last + 1}) AS t(t)
        WHERE year BETWEEN t.t - 6 AND t.t - 1 AND m <= t.t - 7
        GROUP BY ALL ORDER BY t, field, topic, rel
        """
    ).fetchall()
    grouped: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for t, field, topic, offset, n in counts:
        row = grouped.setdefault((int(field), int(t)), {}).setdefault(int(topic), {"topic_id": int(topic), "recent": [0, 0, 0], "reference": [0, 0, 0]})
        if offset <= -4:
            row["reference"][offset + 6] = int(n)
        else:
            row["recent"][offset + 3] = int(n)
    return {key: [rows[topic] for topic in sorted(rows)] for key, rows in sorted(grouped.items())}


def screen_values(con: duckdb.DuckDBPyConnection) -> dict[tuple[int, int], dict[str, Any]]:
    rows = con.execute(
        """
        WITH scoped AS (
            SELECT e.topic, e.t, CASE WHEN w.year <= e.t - 4 THEN 'reference' ELSE 'recent' END AS win, w.score, w.source
            FROM candidate_episodes e JOIN w ON w.topic = e.topic AND w.year BETWEEN e.t - 6 AND e.t - 1
        ),
        medians AS (
            SELECT topic, t, win, quantile_cont(score, 0.5) AS median_score FROM scoped WHERE score IS NOT NULL GROUP BY ALL
        ),
        sources AS (
            SELECT topic, t, win, source, COUNT(*) AS n FROM scoped WHERE source IS NOT NULL GROUP BY ALL
        ),
        shares AS (
            SELECT topic, t, win, MAX(n)::DOUBLE / SUM(n) AS dominant_share FROM sources GROUP BY ALL
        )
        SELECT COALESCE(m.topic, s.topic), COALESCE(m.t, s.t), COALESCE(m.win, s.win), m.median_score, s.dominant_share
        FROM medians m FULL JOIN shares s ON m.topic = s.topic AND m.t = s.t AND m.win = s.win
        ORDER BY 1, 2, 3
        """
    ).fetchall()
    values: dict[tuple[int, int], dict[str, Any]] = {}
    for topic, t, win, median, share in rows:
        values.setdefault((int(topic), int(t)), {})[win] = {"median_score": median, "dominant_share": share}
    return values


def screen_decision(values: Mapping[str, Any], rule: Mapping[str, Any]) -> str:
    reference, recent = values.get("reference", {}), values.get("recent", {})
    needed = [reference.get("median_score"), recent.get("median_score"), reference.get("dominant_share"), recent.get("dominant_share")]
    if any(value is None for value in needed):
        return "UNDEFINED_NOT_EXCLUDED"
    drop = float(reference["median_score"]) - float(recent["median_score"])
    rise = float(recent["dominant_share"]) - float(reference["dominant_share"])
    return "EXCLUDED_INDEXING_ANOMALY" if drop >= 0.15 and rise >= 0.25 else "RETAINED"


def build_structure(works: pa.Table, authorships: pa.Table, guard_ids: pa.Array, contract: Mapping[str, Any], n10: ModuleType, *, threads: int = 8) -> dict[str, Any]:
    """Everything after the extract. Returns the private episode and entrant tables and the public counts."""

    plan = n10_contract(contract)
    first, last = plan["years"]["decision_years"]
    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    con.register("w_arrow", works)
    con.register("a_arrow", authorships)
    con.register("guard_arrow", pa.table({"author_id": guard_ids}))
    con.execute("CREATE TABLE w AS SELECT * FROM w_arrow")
    con.execute("CREATE TABLE a AS SELECT * FROM a_arrow")
    hierarchy = con.execute("SELECT COUNT(*) FROM (SELECT topic FROM w GROUP BY topic HAVING COUNT(DISTINCT subfield) > 1 OR COUNT(DISTINCT field) > 1)").fetchone()[0]
    require(hierarchy == 0, f"{hierarchy} topics have more than one subfield or field")
    con.execute("CREATE TABLE aw AS SELECT a.author_id, w.work_id, w.topic, w.subfield, w.field, w.year, w.date FROM a JOIN w USING (work_id)")
    con.execute("CREATE TABLE first_topic AS SELECT author_id, topic, MIN(year) AS first_year FROM aw GROUP BY ALL")

    attention = attention_rows(con, first, last)
    surge_years: dict[int, list[int]] = {}
    surge_status: dict[str, int] = {}
    for (_field, t), rows in attention.items():
        result = n10.v1.surge_flags(rows, threshold=float(plan["episodes"]["surge_rule"]["modified_z_threshold_inclusive"]))
        surge_status[result["status"]] = surge_status.get(result["status"], 0) + 1
        for topic in result["surges"]:
            surge_years.setdefault(int(topic), []).append(int(t))
    selected = n10.v1.select_episodes(surge_years, spacing=int(plan["episodes"]["non_overlap"]["minimum_spacing_years"]))
    topic_meta = {int(t): (int(s), int(f)) for t, s, f in con.execute("SELECT topic, ANY_VALUE(subfield), ANY_VALUE(field) FROM w GROUP BY topic").fetchall()}
    candidates = sorted((topic_meta[topic][1], topic, t) for topic, years in selected.items() for t in years)
    for field, _topic, t in candidates:
        require(field in plan["fields"]["ids"] and first <= t <= last, "an episode falls outside the registered fields or years")
    con.register("cand_arrow", pa.table({"field": [c[0] for c in candidates], "topic": [c[1] for c in candidates], "t": [c[2] for c in candidates]}, schema=pa.schema([("field", pa.int64()), ("topic", pa.int64()), ("t", pa.int64())])))
    con.execute("CREATE TABLE candidate_episodes AS SELECT * FROM cand_arrow")
    screens = screen_values(con)
    episodes = []
    for field, topic, t in candidates:
        decision = screen_decision(screens.get((topic, t), {}), contract["build"]["indexing_screen"])
        episodes.append({"field": field, "topic": topic, "subfield": topic_meta[topic][0], "t": t, "screen": decision})
    registered = [e for e in episodes if e["screen"] != "EXCLUDED_INDEXING_ANOMALY"]
    families: dict[int, list[int]] = {}
    for e in registered:
        families.setdefault(e["field"], []).append(e["subfield"])
    assignment = n10.v1.partition_families({f: sorted(set(s)) for f, s in families.items()}, plan["partition"]["seed"])
    for index, e in enumerate(registered):
        e["episode_id"] = index
        e["partition"] = assignment[(e["field"], e["subfield"])]
    reg = pa.table({key: [e[key] for e in registered] for key in ("episode_id", "field", "topic", "subfield", "t")} | {"partition": [e["partition"] for e in registered]})
    con.register("reg_arrow", reg)
    con.execute("CREATE TABLE episodes AS SELECT * FROM reg_arrow")
    con.execute("CREATE TABLE field_years AS SELECT DISTINCT author_id, field, year FROM aw")
    con.execute(
        """
        CREATE TABLE raw_entrants AS
        SELECT e.episode_id, e.partition, e.field, ft.author_id, ft.first_year AS entry_year, MIN(aw.date) AS entry_date, e.t
        FROM episodes e
        JOIN first_topic ft ON ft.topic = e.topic AND ft.first_year BETWEEN e.t AND e.t + 2
        JOIN aw ON aw.author_id = ft.author_id AND aw.topic = e.topic AND aw.year = ft.first_year
        GROUP BY e.episode_id, e.partition, e.field, ft.author_id, ft.first_year, e.t
        """
    )
    con.execute(
        """
        CREATE TABLE eligible_entrants AS
        SELECT r.* FROM raw_entrants r
        WHERE EXISTS (SELECT 1 FROM field_years fy WHERE fy.author_id = r.author_id AND fy.field = r.field AND fy.year BETWEEN r.entry_year - 3 AND r.entry_year - 1)
        """
    )
    raw_count = con.execute("SELECT COUNT(*) FROM raw_entrants").fetchone()[0]
    eligible_count = con.execute("SELECT COUNT(*) FROM eligible_entrants").fetchone()[0]
    con.execute("CREATE TABLE guard AS SELECT DISTINCT author_id FROM guard_arrow")
    guarded = con.execute("SELECT COUNT(*) FROM eligible_entrants WHERE author_id IN (SELECT author_id FROM guard)").fetchone()[0]
    con.execute("CREATE TABLE after_guard AS SELECT * FROM eligible_entrants WHERE author_id NOT IN (SELECT author_id FROM guard)")
    con.execute("CREATE TABLE discovery_authors AS SELECT DISTINCT author_id FROM after_guard WHERE partition = 'discovery'")
    excluded = con.execute("SELECT COUNT(*) FROM after_guard WHERE partition = 'confirmatory' AND author_id IN (SELECT author_id FROM discovery_authors)").fetchone()[0]
    entrants = con.execute(
        """
        SELECT episode_id, author_id, entry_date,
               (4 * year(entry_date) + (month(entry_date) - 1) // 3) - 4 * t AS "offset",
               NOT (month(entry_date) = 1 AND day(entry_date) = 1) AS precise
        FROM after_guard
        WHERE NOT (partition = 'confirmatory' AND author_id IN (SELECT author_id FROM discovery_authors))
        ORDER BY episode_id, author_id
        """
    ).to_arrow_table()
    check_clock(entrants, reg, n10)
    validity = plan_validity(contract)
    floors = episode_floors(entrants, validity)
    con.close()
    return {"episodes_all": episodes, "registered": registered, "entrants": entrants, "floors": floors,
            "counts": {"raw_entrants": int(raw_count), "pivot_eligible_entrants": int(eligible_count), "g4_guard_removed": int(guarded),
                       "discovery_author_removed": int(excluded), "surge_status": dict(sorted(surge_status.items()))}}


def plan_validity(contract: Mapping[str, Any]) -> dict[str, Any]:
    return dict(n10_contract(contract)["episode_validity"])


def check_clock(entrants: pa.Table, episodes: pa.Table, n10: ModuleType, sample: int = 2000) -> None:
    """The SQL clock must equal the hash-bound N10 v2 clock on a deterministic sample."""

    decision = dict(zip(episodes["episode_id"].to_pylist(), episodes["t"].to_pylist(), strict=True))
    rows = entrants.num_rows
    step = max(1, rows // sample)
    for index in range(0, rows, step):
        row = {name: entrants[name][index].as_py() for name in ("episode_id", "entry_date", "offset", "precise")}
        clock = n10.entry_clock([row["entry_date"]], decision[row["episode_id"]])
        require((clock["offset"], clock["precise"]) == (row["offset"], row["precise"]), "the SQL clock differs from the N10 clock")


def episode_floors(entrants: pa.Table, validity: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Per episode, the N10 v2 floors on precise entrants, without persistence."""

    minimum = int(validity["minimum_precise_eligible_entrants"])
    years_needed = int(validity["minimum_nonempty_response_years"])
    floors: dict[int, dict[str, Any]] = {}
    ids = entrants["episode_id"].to_numpy()
    offsets = entrants["offset"].to_numpy()
    precise = entrants["precise"].to_numpy(zero_copy_only=False)
    for episode in np.unique(ids):
        mask = ids == episode
        p = offsets[mask & precise]
        floors[int(episode)] = {
            "entrants": int(mask.sum()),
            "precise_entrants": int(p.size),
            "response_years": int(np.unique(p // 4).size),
            "meets_floor": bool(p.size >= minimum and np.unique(p // 4).size >= years_needed),
        }
    return floors


def disposition(registered: Sequence[Mapping[str, Any]], floors: Mapping[int, Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    plan = n10_contract(contract)
    minimum = int(plan["precision"]["minimum_valid_confirmatory_episodes_per_field"])
    table: dict[str, dict[str, int]] = {}
    for e in registered:
        cell = table.setdefault(str(e["field"]), {"discovery_episodes": 0, "confirmatory_episodes": 0, "discovery_meeting_floors": 0, "confirmatory_meeting_floors": 0})
        cell[f"{e['partition']}_episodes"] += 1
        if floors.get(e["episode_id"], {}).get("meets_floor"):
            cell[f"{e['partition']}_meeting_floors"] += 1
    for field in plan["fields"]["ids"]:
        table.setdefault(str(field), {"discovery_episodes": 0, "confirmatory_episodes": 0, "discovery_meeting_floors": 0, "confirmatory_meeting_floors": 0})
    passed = all(table[str(f)]["confirmatory_meeting_floors"] >= minimum for f in plan["fields"]["ids"])
    return {"status": "S1_PASS" if passed else "STOP_INSUFFICIENT_EPISODES", "by_field": dict(sorted(table.items()))}


# -- run ---------------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def load_authorization(path: Path, contract_path: Path) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    require(authorization.get("schema_version") == "pivot_empirical_outcome_selection_trend_s1_authorization.v1", "unexpected authorization schema")
    require(authorization.get("status") == "AUTHORIZED_S1_STRUCTURAL_BUILD", "S1 is not authorized")
    require(authorization["s1_contract_sha256"] == _sha256(contract_path), "the authorization names another S1 contract")
    require(authorization["s1_builder_sha256"] == _sha256(Path(__file__)), "the authorization names another S1 builder")
    require(bool(authorization["human_authorization"]["user_instruction"]), "the authorization does not quote the user")
    require(authorization["persistence_labels"] == "NOT_AUTHORIZED", "S1 may not compute persistence")
    relative = path.resolve().relative_to(ROOT).as_posix()
    require(relative in _git("ls-files", "--", relative).split(), "the authorization file is not committed")
    require(not _git("status", "--porcelain", "--", relative), "the authorization file is dirty")
    return {**authorization, "_file_sha256": _sha256(path)}


def _summary_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for result in results:
        for key, value in result["counts"].items():
            total[key] = total.get(key, 0) + int(value)
    return dict(sorted(total.items()))


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {name: float(np.quantile(array, q)) for name, q in (("min", 0.0), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("max", 1.0))}


def run(contract: Mapping[str, Any], contract_path: Path, authorization_path: Path, output_dir: Path, precommit: Mapping[str, Any], *, workers: int, threads: int) -> dict[str, Any]:
    authorization = load_authorization(authorization_path, contract_path)
    started = time.monotonic()
    check = preflight(contract)
    require(check["status"] == "PASS_S1_PREFLIGHT_FOOTERS_ONLY", f"S1 preflight failed: {check}")
    root = Path(contract["source"]["root"])
    sources = [(item["relative_path"], root / item["relative_path"]) for item in works_files(contract)]
    return execute(contract, contract_path, sources, Path(contract["build"]["g4_guard"]["path"]), output_dir, precommit,
                   {k: v for k, v in check.items() if k != "free_bytes"}, authorization["_file_sha256"],
                   workers=workers, threads=threads, started=started)


def execute(
    contract: Mapping[str, Any], contract_path: Path, sources: Sequence[tuple[str, Path]], guard_path: Path, output_dir: Path,
    precommit: Mapping[str, Any], preflight_record: Mapping[str, Any], authorization_sha256: str, *, workers: int, threads: int, started: float,
) -> dict[str, Any]:
    """The S1 build after its checks: extraction, global accounting, structure, outputs."""

    n10 = load_n10(contract)
    timing: dict[str, float] = {"preflight_seconds": time.monotonic() - started}
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    private = output_dir / "private"
    parts = private / "parts"
    private.mkdir(mode=0o700)
    parts.mkdir(mode=0o700)
    extract = contract["extract"]
    tasks = [
        {
            "index": index, "path": str(path), "relative_path": relative, "part_dir": str(parts),
            "fields": extract["fields"], "latest_year": extract["latest_year"], "type_names": extract["type_names"],
            "snapshot_date": contract["source"]["snapshot_date"], "work_quarantine": contract["identifiers"]["work_quarantine"],
        }
        for index, (relative, path) in enumerate(sources)
    ]
    stage = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = sorted(pool.map(extract_file, tasks), key=lambda r: r["index"])
    timing["extraction_seconds"] = time.monotonic() - stage
    counts = _summary_counts(results)
    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    duplicates = con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT work_id) FROM read_parquet('{parts}/ids_*.parquet')").fetchone()[0]
    counts["duplicate_numeric_work_id_rows"] = int(duplicates)
    digests = {"schema_version": "pivot_empirical_outcome_selection_trend_s1_parts.v1", "parts": [{"index": r["index"], "relative_path": r["relative_path"], **r["digests"]} for r in results]}
    _write_json(private / "parts_digest.json", digests)
    failures = {key: counts.get(key, 0) for key in FAILURE_COUNTS} | {"duplicate_numeric_work_id_rows": counts["duplicate_numeric_work_id_rows"]}
    if "rows_by_footer" in preflight_record:
        failures["rows_read_differ_from_footer_rows"] = abs(int(counts.get("rows", 0)) - int(preflight_record["rows_by_footer"]))
    summary: dict[str, Any] = {
        "schema_version": "pivot_empirical_outcome_selection_trend_s1_summary.v1",
        "precommit_sha": precommit["commit_sha"],
        "preflight": dict(preflight_record),
        "source_accounting": counts,
        "failure_metrics": failures,
    }
    if any(failures.values()):
        con.close()
        summary["status"] = "STOP_SOURCE_INVALID"
        return _finish(output_dir, summary, precommit, authorization_sha256, contract_path, timing, started)
    stage = time.monotonic()
    works = con.execute(f"SELECT * FROM read_parquet('{parts}/works_*.parquet') ORDER BY work_id").to_arrow_table()
    authorships = con.execute(f"SELECT * FROM read_parquet('{parts}/authorships_*.parquet') ORDER BY work_id, author_id").to_arrow_table()
    con.close()
    shutil.rmtree(parts)
    _write(works, private / "works.parquet")
    _write(authorships, private / "authorships.parquet")
    timing["combine_seconds"] = time.monotonic() - stage
    used = _dir_bytes(output_dir)
    require(used <= int(contract["capacity"]["byte_budget"]), f"the output exceeds the byte budget: {used}")
    guard_ids = pq.read_table(guard_path, columns=[contract["build"]["g4_guard"]["column"]]).column(0).combine_chunks()
    stage = time.monotonic()
    built = build_structure(works, authorships, guard_ids, contract, n10, threads=threads)
    timing["build_seconds"] = time.monotonic() - stage
    episodes_table = pa.table({
        "episode_id": [e["episode_id"] for e in built["registered"]],
        "field": [e["field"] for e in built["registered"]],
        "topic": [e["topic"] for e in built["registered"]],
        "subfield": [e["subfield"] for e in built["registered"]],
        "t": [e["t"] for e in built["registered"]],
        "partition": [e["partition"] for e in built["registered"]],
        "screen": [e["screen"] for e in built["registered"]],
        "meets_floor": [built["floors"].get(e["episode_id"], {}).get("meets_floor", False) for e in built["registered"]],
    })
    _write(episodes_table, private / "episodes.parquet")
    _write(built["entrants"], private / "entrants.parquet")
    floors = built["floors"]
    precise_share = [f["precise_entrants"] / f["entrants"] for f in floors.values() if f["entrants"]]
    screen_mix: dict[str, int] = {}
    for e in built["episodes_all"]:
        screen_mix[e["screen"]] = screen_mix.get(e["screen"], 0) + 1
    summary.update({
        "status": disposition(built["registered"], floors, contract)["status"],
        "disposition": disposition(built["registered"], floors, contract),
        "extract": {"works": works.num_rows, "authorships": authorships.num_rows, "xpac_share": counts["extract_xpac"] / max(1, counts["extract_works"])},
        "episodes": {"candidates": len(built["episodes_all"]), "screen": dict(sorted(screen_mix.items())), "registered": len(built["registered"]),
                     "surge_status_by_field_year": built["counts"]["surge_status"]},
        "entrants": {key: built["counts"][key] for key in ("raw_entrants", "pivot_eligible_entrants", "g4_guard_removed", "discovery_author_removed")}
                    | {"retained": built["entrants"].num_rows,
                       "precise_share_overall": (sum(f["precise_entrants"] for f in floors.values()) / max(1, built["entrants"].num_rows)),
                       "episodes_without_entrants": sum(1 for e in built["registered"] if e["episode_id"] not in floors),
                       "precise_share_by_episode": _quantiles(precise_share),
                       "entrants_by_episode": _quantiles([f["entrants"] for f in floors.values()]),
                       "precise_entrants_by_episode": _quantiles([f["precise_entrants"] for f in floors.values()])},
        "persistence_labels_computed": 0,
    })
    return _finish(output_dir, summary, precommit, authorization_sha256, contract_path, timing, started)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    path.chmod(0o600)


def _finish(output_dir: Path, summary: Mapping[str, Any], precommit: Mapping[str, Any], authorization_sha256: str, contract_path: Path, timing: dict[str, float], started: float) -> dict[str, Any]:
    _write_json(output_dir / "s1_summary.json", summary)
    timing["total_seconds"] = time.monotonic() - started
    _write_json(output_dir / "s1_timing.json", {"note": "NOT_AN_IDENTITY_ARTIFACT", **{k: round(v, 1) for k, v in timing.items()}})
    identity = ["s1_summary.json", "private/parts_digest.json", "private/episodes.parquet", "private/entrants.parquet"]
    receipt = {
        "schema_version": "pivot_empirical_outcome_selection_trend_s1_receipt.v1",
        "status": summary["status"],
        "contract_sha256": _sha256(contract_path),
        "builder_sha256": _sha256(Path(__file__)),
        "authorization_sha256": authorization_sha256,
        "precommit": dict(precommit),
        "outputs": {name: _sha256(output_dir / name) for name in identity if (output_dir / name).exists()},
        "persistence_labels_computed": 0,
    }
    _write_json(output_dir / "s1_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--precommit-sha")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    contract = load_contract(config_path)
    workers = args.workers or int(contract["execution"]["workers"])
    if args.command == "preflight":
        print(json.dumps(preflight(contract), indent=2, sort_keys=True))
        return
    if args.precommit_sha is None or args.authorization is None or args.output_dir is None:
        parser.error("run requires --precommit-sha, --authorization, and --output-dir")
    n10 = load_n10(contract)
    precommit = n10.v1.verify_precommit(args.precommit_sha, contract)
    receipt = run(contract, config_path, args.authorization.resolve(), args.output_dir.resolve(), precommit,
                  workers=workers, threads=int(contract["execution"]["duckdb_threads"]))
    print(json.dumps({"status": receipt["status"], "outputs": receipt["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
