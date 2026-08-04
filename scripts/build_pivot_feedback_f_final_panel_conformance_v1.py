#!/usr/bin/env python3
"""Build the outcome-blind F final-panel leaveout/support conformance artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import build_pivot_feedback_d0_certified_author_opportunity_rebuild_v1 as d0


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_final_panel_conformance_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_final_panel_conformance.v1"
REPORT_SCHEMA = "pivot_feedback_f_final_panel_conformance_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_f_final_panel_conformance_outputs.v1"
EPISODE_SCHEMA_VERSION = "pivot_feedback_f_final_panel_episode_conformance.v1"
SLOT_SCHEMA_VERSION = "pivot_feedback_f_final_panel_slot_support.v1"
STATUS = "PASS_OUTCOME_BLIND_FINAL_PANEL_LEAVEOUT_AND_LABEL_INDEPENDENT_SUPPORT_CONFORMANCE_SEAL_PENDING"

EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("slot_id", pa.string(), nullable=False),
        pa.field("temporal_partition_id", pa.string(), nullable=False),
        pa.field("field_attempt_order", pa.uint8(), nullable=False),
        pa.field("initial_assignment_selected", pa.bool_(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("subfield_id", pa.uint16(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("aggregate_reference_count", pa.uint64(), nullable=False),
        pa.field("aggregate_recent_count", pa.uint64(), nullable=False),
        pa.field("aggregate_modified_z", pa.float64(), nullable=False),
        pa.field("all_incumbent_reference_work_count", pa.uint64(), nullable=False),
        pa.field("all_incumbent_recent_work_count", pa.uint64(), nullable=False),
        pa.field("risk_opportunity_count", pa.uint64(), nullable=False),
        pa.field("risk_with_prior_target_active_peer_count", pa.uint64(), nullable=False),
        pa.field("prior_target_active_peer_relation_count", pa.uint64(), nullable=False),
        pa.field("focal_excluded_attention_work_incidence_count", pa.uint64(), nullable=False),
        pa.field("network_excluded_attention_work_incidence_count", pa.uint64(), nullable=False),
        pa.field("network_leaveout_reference_count_min", pa.uint64(), nullable=False),
        pa.field("network_leaveout_reference_count_median", pa.float64(), nullable=False),
        pa.field("network_leaveout_reference_count_max", pa.uint64(), nullable=False),
        pa.field("network_leaveout_recent_count_min", pa.uint64(), nullable=False),
        pa.field("network_leaveout_recent_count_median", pa.float64(), nullable=False),
        pa.field("network_leaveout_recent_count_max", pa.uint64(), nullable=False),
        pa.field("network_leaveout_surge_opportunity_count", pa.uint64(), nullable=False),
        pa.field("leaveout_surge_present", pa.bool_(), nullable=False),
        pa.field("minimum_total_support_required", pa.uint32(), nullable=False),
        pa.field("label_independent_support_pass", pa.bool_(), nullable=False),
        pa.field("label_dependent_support_status", pa.string(), nullable=False),
        pa.field("sealed_slot", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": EPISODE_SCHEMA_VERSION.encode()},
)

SLOT_SCHEMA = pa.schema(
    [
        pa.field("slot_id", pa.string(), nullable=False),
        pa.field("temporal_partition_id", pa.string(), nullable=False),
        pa.field("initial_selected_field_id", pa.uint8(), nullable=False),
        pa.field("aggregate_screen_attempted_field_ids", pa.string(), nullable=False),
        pa.field("final_selected_field_id", pa.uint8(), nullable=True),
        pa.field("episode_count", pa.uint16(), nullable=False),
        pa.field("subfield_family_count", pa.uint16(), nullable=False),
        pa.field("risk_opportunity_count", pa.uint64(), nullable=False),
        pa.field("episodes_passing_label_independent_support", pa.uint16(), nullable=False),
        pa.field("all_episodes_pass_label_independent_support", pa.bool_(), nullable=False),
        pa.field("leaveout_surge_episode_count", pa.uint16(), nullable=False),
        pa.field("leaveout_conformed_subfield_family_count", pa.uint16(), nullable=False),
        pa.field("final_panel_slot_pass", pa.bool_(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("sealed_slot", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": SLOT_SCHEMA_VERSION.encode()},
)


class FinalPanelBuildError(RuntimeError):
    """Fail-closed final-panel build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalPanelBuildError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, path.name)


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _safe_repo_path(relative: Any) -> Path:
    require(isinstance(relative, str) and relative, "empty repository path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe repository path")
    path = (ROOT / pure).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    require(path.is_file() and not path.is_symlink(), "invalid repository binding")
    return path


def _validate_repo_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _safe_repo_path(binding["relative_path"])
    require(_sha256(path) == binding["sha256"], f"binding SHA mismatch: {path.name}")
    document = _yaml(path) if path.suffix in (".yaml", ".yml") else _json(path)
    require(document.get("schema_version") == binding["required_schema_version"], f"binding schema mismatch: {path.name}")
    if "required_status" in binding:
        require(document.get("status") == binding["required_status"], f"binding status mismatch: {path.name}")
    return {"relative_path": binding["relative_path"], "sha256": binding["sha256"]}


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["relative_path"]): dict(row) for row in receipt.get("artifacts", [])}


def _load_assignment(root: Path, binding: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "assignment root mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "assignment receipt SHA mismatch")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "assignment receipt schema")
    require(receipt.get("status") == binding["required_status"], "assignment receipt status")
    artifacts = _artifact_map(receipt)
    candidate_path = root / str(binding["candidate_basename"])
    slot_path = root / str(binding["slot_basename"])
    for path, sha, schema, rows in (
        (candidate_path, binding["candidate_sha256"], binding["candidate_schema_version"], binding["candidate_rows"]),
        (slot_path, binding["slot_sha256"], binding["slot_schema_version"], binding["slot_rows"]),
    ):
        require(_sha256(path) == sha, f"assignment artifact SHA mismatch: {path.name}")
        require(artifacts[path.name]["sha256"] == sha, f"assignment receipt artifact mismatch: {path.name}")
        table = pq.read_table(path)
        require(len(table) == rows, f"assignment artifact rows: {path.name}")
        require((table.schema.metadata or {}).get(b"schema_version", b"").decode() == schema, f"assignment artifact schema: {path.name}")
    verification = Path(str(binding["verification_path"])).resolve(strict=True)
    require(_sha256(verification) == binding["verification_sha256"], "assignment verification SHA")
    require(_json(verification).get("status") == binding["verification_status"], "assignment verification status")
    candidates = pq.read_table(candidate_path).to_pylist()
    slots = pq.read_table(slot_path).to_pylist()
    retained = [row for row in candidates if row["retained_for_assigned_panel"]]
    require(len(retained) == binding["initially_retained_candidate_rows"] == 107, "initial retained candidate closure")
    require(sum(int(row["retained_attention_candidate_count"]) for row in slots) == len(retained), "slot candidate closure")
    episodes = []
    field_disposition: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        slot_id = str(slot["slot_id"])
        period = str(slot["temporal_partition_id"])
        field_order = [int(slot["nominal_primary_field_id"]), *[int(value) for value in slot["reserve_field_ids"]]]
        dispositions = []
        for order, field_id in enumerate(field_order, 1):
            rows = [
                row for row in candidates
                if int(row["field_id"]) == field_id and str(row["temporal_partition_id"]) == period
            ]
            rows.sort(key=lambda row: (int(row["decision_year"]), int(row["subfield_id"]), int(row["topic_id"])))
            family_count = len({int(row["subfield_id"]) for row in rows})
            eligible = family_count >= 3
            dispositions.append(
                {
                    "field_attempt_order": order,
                    "field_id": field_id,
                    "aggregate_candidate_count": len(rows),
                    "aggregate_subfield_family_count": family_count,
                    "aggregate_screen_eligible": eligible,
                }
            )
            if not eligible:
                continue
            for row in rows:
                episodes.append(
                    {
                        "episode_id": len(episodes) + 1,
                        "slot_id": slot_id,
                        "temporal_partition_id": period,
                        "field_attempt_order": order,
                        "initial_assignment_selected": field_id == int(slot["selected_field_id"]),
                        "field_id": field_id,
                        "topic_id": int(row["topic_id"]),
                        "subfield_id": int(row["subfield_id"]),
                        "decision_year": int(row["decision_year"]),
                        "aggregate_reference_count": int(row["reference_count"]),
                        "aggregate_recent_count": int(row["recent_count"]),
                        "aggregate_modified_z": float(row["modified_z"]),
                        "field_year_median": float(row["field_year_median"]),
                        "field_year_mad": float(row["field_year_mad"]),
                        "sealed_slot": bool(slot["sealed"]),
                    }
                )
        field_disposition[slot_id] = dispositions
    require(len(episodes) == 194, "aggregate-screen-eligible reserve episode closure")
    return episodes, slots, {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "candidate_sha256": binding["candidate_sha256"],
        "slot_sha256": binding["slot_sha256"],
        "verification_sha256": binding["verification_sha256"],
        "aggregate_screen_field_disposition": field_disposition,
    }


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _episode_arrow(episodes: Sequence[Mapping[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(list(episodes))


def _run_sql_replay(
    *,
    database: Path,
    episodes: Sequence[Mapping[str, Any]],
    work_glob: Path,
    event_glob: Path,
    threads: int,
    memory_limit_gb: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    con = duckdb.connect(str(database))
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{_sql_path(database.parent / 'duckdb-spill')}'")
    con.register("episode_input", _episode_arrow(episodes))
    con.execute("CREATE TABLE episode AS SELECT * FROM episode_input")
    work = _sql_path(work_glob)
    event = _sql_path(event_glob)
    eligible = """
        e.temporal_disposition IN (2,3)
        AND w.publication_year=e.publication_year
        AND w.primary_field_id=e.primary_field_id
        AND w.temporal_disposition=e.temporal_disposition
        AND w.primary_topic_id IS NOT NULL
        AND w.work_type_code IN (1,2,3,4,7,16,18,21)
        AND NOT w.is_paratext
        AND coalesce(w.is_retracted,false)=false
    """
    con.execute(
        f"""
        CREATE TABLE target_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id,
               e.work_numeric_id AS work_id,
               e.publication_year,
               e.declared_authors_count,
               e.primary_field_id AS field_id,
               w.primary_topic_id AS topic_id
        FROM read_parquet('{event}') e
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        JOIN (SELECT DISTINCT field_id, topic_id FROM episode) p
          ON p.field_id=e.primary_field_id AND p.topic_id=w.primary_topic_id
        WHERE e.publication_year<=2021 AND {eligible}
        """
    )
    con.execute(
        f"""
        CREATE TABLE modern_event AS
        SELECT e.author_numeric_id AS author_id,
               e.work_numeric_id AS work_id,
               e.publication_year,
               e.declared_authors_count,
               e.primary_field_id AS field_id
        FROM read_parquet('{event}') e
        JOIN read_parquet('{work}') w ON w.work_id_numeric=e.work_numeric_id
        WHERE e.publication_year BETWEEN 2008 AND 2021
          AND e.primary_field_id BETWEEN 11 AND 36
          AND {eligible}
        """
    )
    con.execute("CREATE INDEX modern_author_year ON modern_event(author_id, publication_year)")
    con.execute("CREATE INDEX modern_work ON modern_event(work_id)")
    con.execute("CREATE INDEX target_topic_author ON target_event(field_id, topic_id, author_id)")
    con.execute(
        """
        CREATE TABLE target_first AS
        SELECT ep.episode_id, te.author_id, min(te.publication_year) AS first_target_year
        FROM episode ep JOIN target_event te
          ON te.field_id=ep.field_id AND te.topic_id=ep.topic_id
        WHERE te.publication_year<=ep.decision_year-1
        GROUP BY ep.episode_id, te.author_id
        """
    )
    con.execute(
        """
        CREATE TABLE risk_opportunity AS
        WITH active AS (
          SELECT DISTINCT ep.episode_id, me.author_id
          FROM episode ep JOIN modern_event me ON me.field_id=ep.field_id
          WHERE me.publication_year BETWEEN ep.decision_year-3 AND ep.decision_year-1
        )
        SELECT a.episode_id, a.author_id
        FROM active a LEFT JOIN target_first tf
          ON tf.episode_id=a.episode_id AND tf.author_id=a.author_id
        WHERE tf.author_id IS NULL
        """
    )
    con.execute("CREATE INDEX risk_author ON risk_opportunity(author_id, episode_id)")
    con.execute(
        """
        CREATE TABLE target_window_author AS
        SELECT ep.episode_id, te.work_id, te.author_id, te.publication_year,
               CASE WHEN te.publication_year BETWEEN ep.decision_year-6 AND ep.decision_year-4 THEN 0 ELSE 1 END AS window_code
        FROM episode ep JOIN target_event te
          ON te.field_id=ep.field_id AND te.topic_id=ep.topic_id
        WHERE te.publication_year BETWEEN ep.decision_year-6 AND ep.decision_year-1
        """
    )
    con.execute(
        """
        CREATE TABLE attention_work AS
        SELECT twa.episode_id, twa.work_id, min(twa.window_code) AS window_code
        FROM target_window_author twa
        JOIN target_first tf ON tf.episode_id=twa.episode_id AND tf.author_id=twa.author_id
        JOIN episode ep ON ep.episode_id=twa.episode_id
        GROUP BY twa.episode_id, twa.work_id, ep.decision_year
        HAVING count(*)=count(*) FILTER (WHERE tf.first_target_year<ep.decision_year-6)
        """
    )
    focal_incidence = int(
        con.execute(
            """
            SELECT count(*) FROM (
              SELECT DISTINCT r.episode_id, r.author_id, aw.work_id
              FROM risk_opportunity r
              JOIN target_window_author twa ON twa.episode_id=r.episode_id AND twa.author_id=r.author_id
              JOIN attention_work aw ON aw.episode_id=twa.episode_id AND aw.work_id=twa.work_id
            )
            """
        ).fetchone()[0]
    )
    require(focal_incidence == 0, "risk anti-target rule did not produce zero focal leaveout incidences")
    con.execute(
        """
        CREATE TABLE prior_focal_work AS
        SELECT DISTINCT r.episode_id, r.author_id AS focal_author_id, me.work_id
        FROM risk_opportunity r
        JOIN episode ep ON ep.episode_id=r.episode_id
        JOIN modern_event me ON me.author_id=r.author_id
        WHERE me.publication_year BETWEEN ep.decision_year-5 AND ep.decision_year-1
        """
    )
    team_stats = con.execute(
        """
        SELECT count(*) AS prior_focal_work_relations,
               max(me.declared_authors_count) AS maximum_declared_team_size,
               count(*) FILTER (WHERE me.declared_authors_count>100) AS relations_above_100
        FROM prior_focal_work pf
        JOIN modern_event me ON me.work_id=pf.work_id AND me.author_id=pf.focal_author_id
        """
    ).fetchone()
    con.execute(
        """
        CREATE TABLE prior_target_peer AS
        SELECT DISTINCT pf.episode_id, pf.focal_author_id, co.author_id AS peer_author_id
        FROM prior_focal_work pf
        JOIN modern_event co ON co.work_id=pf.work_id
        JOIN target_first tf ON tf.episode_id=pf.episode_id AND tf.author_id=co.author_id
        WHERE co.author_id<>pf.focal_author_id
        """
    )
    con.execute("CREATE INDEX peer_episode_author ON prior_target_peer(episode_id, peer_author_id)")
    con.execute(
        """
        CREATE TABLE excluded_attention_work AS
        SELECT DISTINCT pp.episode_id, pp.focal_author_id, aw.work_id, aw.window_code
        FROM prior_target_peer pp
        JOIN target_window_author twa
          ON twa.episode_id=pp.episode_id AND twa.author_id=pp.peer_author_id
        JOIN attention_work aw
          ON aw.episode_id=twa.episode_id AND aw.work_id=twa.work_id
        """
    )
    con.execute(
        """
        CREATE TABLE risk_measurement AS
        WITH base AS (
          SELECT episode_id,
                 count(*) FILTER (WHERE window_code=0) AS base_reference,
                 count(*) FILTER (WHERE window_code=1) AS base_recent
          FROM attention_work GROUP BY episode_id
        ), excluded AS (
          SELECT episode_id, focal_author_id,
                 count(*) FILTER (WHERE window_code=0) AS excluded_reference,
                 count(*) FILTER (WHERE window_code=1) AS excluded_recent
          FROM excluded_attention_work GROUP BY episode_id, focal_author_id
        ), peers AS (
          SELECT episode_id, focal_author_id, count(*) AS peer_count
          FROM prior_target_peer GROUP BY episode_id, focal_author_id
        )
        SELECT r.episode_id, r.author_id,
               coalesce(b.base_reference,0)::UBIGINT AS base_reference,
               coalesce(b.base_recent,0)::UBIGINT AS base_recent,
               coalesce(x.excluded_reference,0)::UBIGINT AS excluded_reference,
               coalesce(x.excluded_recent,0)::UBIGINT AS excluded_recent,
               (coalesce(b.base_reference,0)-coalesce(x.excluded_reference,0))::UBIGINT AS adjusted_reference,
               (coalesce(b.base_recent,0)-coalesce(x.excluded_recent,0))::UBIGINT AS adjusted_recent,
               coalesce(p.peer_count,0)::UBIGINT AS peer_count
        FROM risk_opportunity r
        LEFT JOIN base b ON b.episode_id=r.episode_id
        LEFT JOIN excluded x ON x.episode_id=r.episode_id AND x.focal_author_id=r.author_id
        LEFT JOIN peers p ON p.episode_id=r.episode_id AND p.focal_author_id=r.author_id
        """
    )
    bad = int(con.execute("SELECT count(*) FROM risk_measurement WHERE adjusted_reference>base_reference OR adjusted_recent>base_recent").fetchone()[0])
    require(bad == 0, "network leaveout monotonicity failure")
    rows = con.execute(
        """
        SELECT ep.episode_id, ep.slot_id, ep.temporal_partition_id,
               ep.field_attempt_order, ep.initial_assignment_selected,
               ep.field_id, ep.topic_id,
               ep.subfield_id, ep.decision_year, ep.aggregate_reference_count,
               ep.aggregate_recent_count, ep.aggregate_modified_z,
               coalesce(max(rm.base_reference),0)::UBIGINT AS all_incumbent_reference_work_count,
               coalesce(max(rm.base_recent),0)::UBIGINT AS all_incumbent_recent_work_count,
               count(rm.author_id)::UBIGINT AS risk_opportunity_count,
               count(rm.author_id) FILTER (WHERE rm.peer_count>0)::UBIGINT AS risk_with_prior_target_active_peer_count,
               coalesce(sum(rm.peer_count),0)::UBIGINT AS prior_target_active_peer_relation_count,
               0::UBIGINT AS focal_excluded_attention_work_incidence_count,
               coalesce(sum(rm.excluded_reference+rm.excluded_recent),0)::UBIGINT AS network_excluded_attention_work_incidence_count,
               coalesce(min(rm.adjusted_reference),0)::UBIGINT AS network_leaveout_reference_count_min,
               coalesce(median(rm.adjusted_reference),0)::DOUBLE AS network_leaveout_reference_count_median,
               coalesce(max(rm.adjusted_reference),0)::UBIGINT AS network_leaveout_reference_count_max,
               coalesce(min(rm.adjusted_recent),0)::UBIGINT AS network_leaveout_recent_count_min,
               coalesce(median(rm.adjusted_recent),0)::DOUBLE AS network_leaveout_recent_count_median,
               coalesce(max(rm.adjusted_recent),0)::UBIGINT AS network_leaveout_recent_count_max,
               count(rm.author_id) FILTER (
                 WHERE rm.adjusted_recent>rm.adjusted_reference
                   AND ln(1+rm.adjusted_recent/3.0)-ln(1+rm.adjusted_reference/3.0)>0
                   AND 0.6744897501960817*((ln(1+rm.adjusted_recent/3.0)-ln(1+rm.adjusted_reference/3.0))-ep.field_year_median)/ep.field_year_mad>=3.0
               )::UBIGINT AS network_leaveout_surge_opportunity_count,
               ep.sealed_slot
        FROM episode ep LEFT JOIN risk_measurement rm ON rm.episode_id=ep.episode_id
        GROUP BY ALL ORDER BY ep.episode_id
        """
    ).fetch_arrow_table().to_pylist()
    target_event_rows = int(con.execute("SELECT count(*) FROM target_event").fetchone()[0])
    modern_event_rows = int(con.execute("SELECT count(*) FROM modern_event").fetchone()[0])
    con.close()
    accounting = {
        "target_event_rows": target_event_rows,
        "modern_event_rows": modern_event_rows,
        "risk_opportunity_rows": sum(int(row["risk_opportunity_count"]) for row in rows),
        "focal_excluded_attention_work_incidences": focal_incidence,
        "prior_focal_work_relations": int(team_stats[0]),
        "maximum_declared_team_size": int(team_stats[1] or 0),
        "prior_focal_work_relations_above_team_size_100": int(team_stats[2]),
        "team_cap_applied": False,
    }
    return rows, accounting


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", compression_level=9, use_dictionary=True, write_statistics=True, version="2.6")
    os.chmod(path, 0o600)
    return {"relative_path": path.name, "bytes": path.stat().st_size, "rows": len(table), "sha256": _sha256(path)}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    os.chmod(path, 0o600)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Pivot-feedback F final-panel conformance v1",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Slot | Field | Episodes | Families | Risk rows | Label-independent support |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["slot_support"]:
        lines.append(f"| {row['slot_id']} | {row['initial_selected_field_id']} | {row['episode_count']} | {row['leaveout_conformed_subfield_family_count']} | {row['risk_opportunity_count']} | {row['status']} |")
    lines.extend(
        [
            "",
            "The replay used no team cap and emitted no raw author or work IDs.",
            "Entry/persistence labels, case-control sampling, outcome-conditioned",
            "support, model fitting, and G4 analytic opening were not run.",
            "This component does not pass F; the atomic partition and byte seal remain pending.",
            "",
        ]
    )
    return "\n".join(lines)


def _cleanup(path: Path | None, output: Path) -> None:
    if path is None:
        return
    require(path.parent == output.parent and path.name.startswith(f".{output.name}."), "unsafe temporary cleanup")
    if path.exists():
        shutil.rmtree(path)


def build(
    *,
    config_path: Path,
    assignment_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    output: Path,
    threads: int,
    memory_limit_gb: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= threads <= 32, "threads must be in [1,32]")
    require(8 <= memory_limit_gb <= 192, "memory limit must be in [8,192] GB")
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(config.get("status") == "FROZEN_FINAL_PANEL_CONFORMANCE_PENDING_PRODUCTION_EXECUTION_NO_F_PASS", "config status")
    repository_bindings = {"authorization_parent": _validate_repo_binding(_mapping(config["authorization_parent"], "authorization parent"))}
    for name, binding in _mapping(config["bound_design_sources"], "bound sources").items():
        repository_bindings[name] = _validate_repo_binding(_mapping(binding, name))
    panel = _mapping(config["panel_contract"], "panel contract")
    require(panel["team_cap"] == "NONE", "team cap differs")
    require(panel["attention_work_rule"] == "EVERY_LINKED_AUTHOR_IS_PRE_SIGNAL_INCUMBENT", "incumbent rule differs")
    support = _mapping(config["support_replay"], "support replay")
    require(support["label_dependent"]["may_block_f"] is False, "circular label-dependent F gate")
    require(config["completion_accounting"]["this_component_is_new_gate"] is False, "component became gate")
    episodes, slots, assignment_binding = _load_assignment(assignment_root, _mapping(config["verified_assignment_parent"], "assignment parent"))
    inputs = _mapping(config["normalized_inputs"], "normalized inputs")
    work_paths, work_binding, work_stats = d0._validate_partitioned_input(cutoff_root, _mapping(inputs["cutoff_work_metadata"], "cutoff input"), schema=d0.WORK_INPUT_SCHEMA, workers=threads)
    event_paths, event_binding, event_stats = d0._validate_partitioned_input(author_history_root, _mapping(inputs["author_history_events"], "author input"), schema=d0.EVENT_INPUT_SCHEMA, workers=threads)
    output = output.resolve()
    require(output.name == config["output_contract"]["root_basename"], "output basename")
    require(not output.exists() and output.parent.is_dir(), "output must be new under existing parent")
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    try:
        db_path = temporary / "private_replay.duckdb"
        replay_rows, replay_accounting = _run_sql_replay(
            database=db_path,
            episodes=episodes,
            work_glob=cutoff_root / "work_metadata" / "*.parquet",
            event_glob=author_history_root / "events" / "*.parquet",
            threads=threads,
            memory_limit_gb=memory_limit_gb,
        )
        minimum = int(support["label_independent"]["minimum_total_opportunity_rows"])
        episode_rows = []
        for row in replay_rows:
            count = int(row["risk_opportunity_count"])
            require(count > 0, f"zero opportunity episode {row['episode_id']}")
            require(int(row["focal_excluded_attention_work_incidence_count"]) == 0, "focal anti-target mismatch")
            require(int(row["network_leaveout_reference_count_max"]) <= int(row["all_incumbent_reference_work_count"]), "reference leaveout monotonicity")
            require(int(row["network_leaveout_recent_count_max"]) <= int(row["all_incumbent_recent_work_count"]), "recent leaveout monotonicity")
            episode_rows.append(
                {
                    **row,
                    "leaveout_surge_present": int(row["network_leaveout_surge_opportunity_count"]) > 0,
                    "minimum_total_support_required": minimum,
                    "label_independent_support_pass": count >= minimum,
                    "label_dependent_support_status": "DEFERRED_UNTIL_POST_F_AUTHORIZED_LABEL_MATERIALIZATION",
                }
            )
        require(len(episode_rows) == 194, "episode replay closure")
        require(all(row["label_independent_support_pass"] for row in episode_rows), "label-independent support failure")
        slot_rows = []
        for slot in slots:
            selected = [row for row in episode_rows if row["slot_id"] == slot["slot_id"]]
            dispositions = assignment_binding["aggregate_screen_field_disposition"][str(slot["slot_id"])]
            attempted_fields = [int(row["field_id"]) for row in dispositions if row["aggregate_screen_eligible"]]
            final_field: int | None = None
            final_rows: list[dict[str, Any]] = []
            best_rows: list[dict[str, Any]] = []
            best_family_count = -1
            for field_id in attempted_fields:
                candidate = [row for row in selected if int(row["field_id"]) == field_id]
                conformed_families = {
                    int(row["subfield_id"])
                    for row in candidate
                    if int(row["network_leaveout_surge_opportunity_count"]) > 0
                }
                if len(conformed_families) > best_family_count:
                    best_rows = candidate
                    best_family_count = len(conformed_families)
                if len(conformed_families) >= int(panel["final_slot_field_requires_leaveout_surge_subfield_families"]):
                    final_field = field_id
                    final_rows = candidate
                    break
            evaluated = final_rows if final_rows else best_rows
            surge_rows = [row for row in evaluated if int(row["network_leaveout_surge_opportunity_count"]) > 0]
            conformed_families = {int(row["subfield_id"]) for row in surge_rows}
            slot_pass = final_field is not None
            slot_rows.append(
                {
                    "slot_id": str(slot["slot_id"]),
                    "temporal_partition_id": str(slot["temporal_partition_id"]),
                    "initial_selected_field_id": int(slot["selected_field_id"]),
                    "aggregate_screen_attempted_field_ids": ",".join(str(value) for value in attempted_fields),
                    "final_selected_field_id": final_field,
                    "episode_count": len(selected),
                    "subfield_family_count": len({int(row["subfield_id"]) for row in selected}),
                    "risk_opportunity_count": sum(int(row["risk_opportunity_count"]) for row in selected),
                    "episodes_passing_label_independent_support": sum(bool(row["label_independent_support_pass"]) for row in selected),
                    "all_episodes_pass_label_independent_support": all(bool(row["label_independent_support_pass"]) for row in selected),
                    "leaveout_surge_episode_count": len(surge_rows),
                    "leaveout_conformed_subfield_family_count": len(conformed_families),
                    "final_panel_slot_pass": slot_pass,
                    "status": "PASS_LEAVEOUT_SHOCK_FAMILY_SUPPORT" if slot_pass else "BLOCKED_FEWER_THAN_THREE_LEAVEOUT_SURGE_SUBFIELD_FAMILIES",
                    "sealed_slot": bool(slot["sealed"]),
                }
            )
        require(sum(row["episode_count"] for row in slot_rows) == 194, "slot replay closure")
        final_panel_pass = all(bool(row["final_panel_slot_pass"]) for row in slot_rows)
        status = STATUS if final_panel_pass else str(config["output_contract"]["blocking_status"])
        d0._validate_unchanged(work_paths, work_stats)
        d0._validate_unchanged(event_paths, event_stats)
        if db_path.exists():
            db_path.unlink()
        spill = temporary / "duckdb-spill"
        if spill.exists():
            shutil.rmtree(spill)
        artifacts = [
            _write_parquet(pa.Table.from_pylist(episode_rows, schema=EPISODE_SCHEMA), temporary / "final_panel_episode_conformance.parquet"),
            _write_parquet(pa.Table.from_pylist(slot_rows, schema=SLOT_SCHEMA), temporary / "final_panel_slot_support.parquet"),
        ]
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "status": status,
            "repository_bindings": repository_bindings,
            "assignment_binding": assignment_binding,
            "input_bindings": {"cutoff_work_metadata": work_binding, "author_history_events": event_binding},
            "scope": {
                "episode_count": len(episode_rows),
                "slot_count": len(slot_rows),
                "final_panel_conformance_complete": final_panel_pass,
                "raw_author_or_work_ids_emitted": False,
                "entry_or_persistence_labels_read": False,
                "case_control_sampling_applied": False,
                "outcome_conditioned_support_evaluated": False,
                "realized_citation_effect_read": False,
                "model_fitted": False,
                "g4_open": False,
                "pass_f": False,
                "pass_g0": False,
            },
            "replay_accounting": replay_accounting,
            "support_boundary": {
                "label_independent_minimum_total": minimum,
                "episodes_passing": len(episode_rows),
                "label_dependent_support_status": "DEFERRED_UNTIL_POST_F_AUTHORIZED_LABEL_MATERIALIZATION",
                "label_dependent_support_may_block_f": False,
                "reason": "FIRST_CONFIRMATORY_READ_IS_POST_F_SO_OUTCOME_CONDITIONED_COUNTS_CANNOT_BE_A_CIRCULAR_F_PREREQUISITE",
            },
            "slot_support": slot_rows,
            "authorization": config["authorization_after_pass"] if final_panel_pass else config["authorization_after_block"],
            "execution": {
                "config_relative_path": str(config_path.relative_to(ROOT)),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "threads": threads,
                "memory_limit_gb": memory_limit_gb,
                "elapsed_seconds": time.monotonic() - started,
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "final_panel_conformance.json"
        _write_json(report_path, report)
        artifacts.append({"relative_path": report_path.name, "bytes": report_path.stat().st_size, "sha256": _sha256(report_path)})
        markdown_path = temporary / "FINAL_PANEL_CONFORMANCE.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts.append({"relative_path": markdown_path.name, "bytes": markdown_path.stat().st_size, "sha256": _sha256(markdown_path)})
        artifacts.sort(key=lambda row: row["relative_path"])
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": status,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(_canonical_bytes([(row["relative_path"], row["sha256"]) for row in artifacts])).hexdigest(),
            "receipt_written_last": True,
            "independent_verification_complete": False,
            "final_panel_conformance_complete": final_panel_pass,
            "label_independent_support_pass": True,
            "label_dependent_support_deferred_post_f": True,
            "entry_or_persistence_labels_read": False,
            "raw_author_or_work_ids_emitted": False,
            "team_cap_applied": False,
            "holdout_byte_seal_complete": False,
            "pass_f": False,
            "pass_g0": False,
            "model_fitted": False,
            "g4_open": False,
        }
        receipt_path = temporary / "final_panel_conformance.outputs.json"
        _write_json(receipt_path, receipt)
        os.rename(temporary, output)
        temporary = None
        return report
    except BaseException:
        _cleanup(temporary, output)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit-gb", type=int, default=96)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        config_path=args.config,
        assignment_root=args.assignment_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output=args.output,
        threads=args.threads,
        memory_limit_gb=args.memory_limit_gb,
    )
    print(json.dumps({"status": report["status"], "replay_accounting": report["replay_accounting"], "slot_support": report["slot_support"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
