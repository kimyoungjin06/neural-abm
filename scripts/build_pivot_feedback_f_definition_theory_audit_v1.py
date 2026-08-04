#!/usr/bin/env python3
"""Build the outcome-blind post-block definition and theory audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_definition_theory_audit_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_definition_theory_audit.v1"
REPORT_SCHEMA = "pivot_feedback_f_definition_theory_audit_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_f_definition_theory_audit_outputs.v1"
COMPOSITION_SCHEMA_VERSION = "pivot_feedback_f_definition_work_composition.v1"
STATUS = "PASS_OUTCOME_BLIND_DEFINITION_THEORY_AUDIT_ALIGNED_PRIMARY_RESCREEN_AUTHORIZED_NO_F_PASS"

COMPOSITION_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.uint16(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("decision_year", pa.uint16(), nullable=False),
        pa.field("window", pa.string(), nullable=False),
        pa.field("all_target_work_count", pa.uint64(), nullable=False),
        pa.field("aggregate_author_linked_work_count", pa.uint64(), nullable=False),
        pa.field("all_pre_signal_incumbent_work_count", pa.uint64(), nullable=False),
        pa.field("mixed_incumbent_new_work_count", pa.uint64(), nullable=False),
        pa.field("no_pre_signal_incumbent_work_count", pa.uint64(), nullable=False),
        pa.field("no_linked_author_work_count", pa.uint64(), nullable=False),
        pa.field("linked_author_incidence_count", pa.uint64(), nullable=False),
        pa.field("pre_signal_incumbent_author_incidence_count", pa.uint64(), nullable=False),
    ],
    metadata={b"schema_version": COMPOSITION_SCHEMA_VERSION.encode()},
)


class DefinitionAuditBuildError(RuntimeError):
    """Fail-closed definition-audit build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DefinitionAuditBuildError(message)


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
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


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
    require(_sha256(path) == binding["sha256"], f"binding SHA: {path.name}")
    document = _yaml(path) if path.suffix in (".yaml", ".yml") else _json(path)
    require(document.get("schema_version") == binding["required_schema_version"], f"binding schema: {path.name}")
    if "required_status" in binding:
        require(document.get("status") == binding["required_status"], f"binding status: {path.name}")
    return {"relative_path": str(path.relative_to(ROOT)), "sha256": binding["sha256"]}


def _validate_source_root(root: Path, binding: Mapping[str, Any]) -> tuple[Path, int]:
    resolved = root.resolve(strict=True)
    require(resolved.is_dir() and not resolved.is_symlink(), "invalid source root")
    require(resolved.name == binding["expected_root_basename"], "source root basename")
    receipt = resolved / str(binding["receipt_basename"])
    require(receipt.is_file() and _sha256(receipt) == binding["receipt_sha256"], "source receipt SHA")
    prefix = str(binding["artifact_prefix"])
    require(prefix.endswith("/"), "artifact prefix")
    paths = sorted((resolved / prefix[:-1]).glob("*.parquet"))
    require(len(paths) == int(binding["artifact_count"]), "source partition count")
    return resolved / prefix[:-1] / "*.parquet", len(paths)


def _load_parents(config: Mapping[str, Any], final_root: Path, assignment_root: Path) -> tuple[pa.Table, list[dict[str, Any]], dict[str, Any]]:
    parent = _mapping(config["blocked_replay_parent"], "blocked replay parent")
    root = final_root.resolve(strict=True)
    require(root.name == parent["expected_root_basename"], "blocked replay root")
    receipt_path = root / str(parent["receipt_basename"])
    episode_path = root / str(parent["episode_basename"])
    require(_sha256(receipt_path) == parent["receipt_sha256"], "blocked receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == parent["receipt_schema_version"], "blocked receipt schema")
    require(receipt.get("status") == parent["required_status"], "blocked receipt status")
    require(_sha256(episode_path) == parent["episode_sha256"], "blocked episode SHA")
    episode = pq.read_table(episode_path)
    require((episode.schema.metadata or {}).get(b"schema_version", b"").decode() == parent["episode_schema_version"], "blocked episode schema")
    require(len(episode) == int(parent["episode_rows"]), "blocked episode rows")
    verification = Path(str(parent["verification_path"])).resolve(strict=True)
    require(_sha256(verification) == parent["verification_sha256"], "blocked verification SHA")
    require(_json(verification).get("status") == parent["verification_status"], "blocked verification status")

    assignment = _mapping(config["aggregate_assignment_parent"], "assignment parent")
    aroot = assignment_root.resolve(strict=True)
    require(aroot.name == assignment["expected_root_basename"], "assignment root")
    areceipt = aroot / str(assignment["receipt_basename"])
    candidate_path = aroot / str(assignment["candidate_basename"])
    require(_sha256(areceipt) == assignment["receipt_sha256"], "assignment receipt SHA")
    require(_sha256(candidate_path) == assignment["candidate_sha256"], "assignment candidate SHA")
    candidates_table = pq.read_table(candidate_path)
    require((candidates_table.schema.metadata or {}).get(b"schema_version", b"").decode() == assignment["candidate_schema_version"], "assignment candidate schema")
    require(len(candidates_table) == int(assignment["candidate_rows"]), "assignment candidate rows")
    return episode, candidates_table.to_pylist(), {
        "blocked_receipt_sha256": parent["receipt_sha256"],
        "blocked_episode_sha256": parent["episode_sha256"],
        "blocked_verification_sha256": parent["verification_sha256"],
        "assignment_receipt_sha256": assignment["receipt_sha256"],
        "assignment_candidate_sha256": assignment["candidate_sha256"],
    }


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _composition_replay(episodes: pa.Table, work_glob: Path, event_glob: Path, threads: int, memory_limit_gb: int, temp_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{_sql_path(temp_dir)}'")
    episode_input = episodes.select(["episode_id", "field_id", "topic_id", "decision_year"])
    con.register("episode_input", episode_input)
    con.execute("CREATE TABLE episode AS SELECT * FROM episode_input")
    con.execute("CREATE TABLE pair AS SELECT DISTINCT field_id, topic_id FROM episode")
    work = _sql_path(work_glob)
    event = _sql_path(event_glob)
    con.execute(
        f"""
        CREATE TABLE target_work_history AS
        SELECT w.work_id_numeric AS work_id, w.publication_year,
               w.primary_field_id AS field_id, w.primary_topic_id AS topic_id,
               w.temporal_disposition
        FROM read_parquet('{work}') w
        JOIN pair p ON p.field_id=w.primary_field_id AND p.topic_id=w.primary_topic_id
        WHERE w.publication_year<=2021
          AND w.temporal_disposition IN (2,3)
          AND w.primary_topic_id IS NOT NULL
          AND w.work_type_code IN (1,2,3,4,7,16,18,21)
          AND NOT w.is_paratext
          AND coalesce(w.is_retracted,false)=false
        """
    )
    con.execute("CREATE INDEX target_work_id ON target_work_history(work_id)")
    con.execute(
        f"""
        CREATE TABLE target_event AS
        SELECT DISTINCT e.author_numeric_id AS author_id, tw.work_id,
               tw.publication_year, tw.field_id, tw.topic_id
        FROM read_parquet('{event}') e
        JOIN target_work_history tw ON tw.work_id=e.work_numeric_id
        WHERE e.publication_year=tw.publication_year
          AND e.primary_field_id=tw.field_id
          AND e.temporal_disposition=tw.temporal_disposition
        """
    )
    con.execute("CREATE INDEX target_topic_author ON target_event(field_id,topic_id,author_id)")
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
    con.execute("CREATE INDEX target_first_idx ON target_first(episode_id,author_id)")
    con.execute(
        """
        CREATE TABLE target_window_work AS
        SELECT ep.episode_id, tw.work_id,
               CASE WHEN tw.publication_year BETWEEN ep.decision_year-6 AND ep.decision_year-4
                    THEN 'reference' ELSE 'recent' END AS window
        FROM episode ep JOIN target_work_history tw
          ON tw.field_id=ep.field_id AND tw.topic_id=ep.topic_id
        WHERE tw.publication_year BETWEEN ep.decision_year-6 AND ep.decision_year-1
        """
    )
    con.execute("CREATE INDEX target_window_work_idx ON target_window_work(episode_id,work_id)")
    con.execute(
        """
        CREATE TABLE work_author_class AS
        SELECT tww.episode_id, tww.work_id, tww.window,
               count(te.author_id)::UBIGINT AS linked_author_count,
               count(te.author_id) FILTER (WHERE tf.first_target_year<ep.decision_year-6)::UBIGINT AS incumbent_author_count
        FROM target_window_work tww
        JOIN episode ep ON ep.episode_id=tww.episode_id
        LEFT JOIN target_event te ON te.work_id=tww.work_id
        LEFT JOIN target_first tf ON tf.episode_id=tww.episode_id AND tf.author_id=te.author_id
        GROUP BY tww.episode_id, tww.work_id, tww.window
        """
    )
    rows = con.execute(
        """
        SELECT ep.episode_id, ep.field_id, ep.topic_id, ep.decision_year, wac.window,
               count(*)::UBIGINT AS all_target_work_count,
               count(*) FILTER (WHERE linked_author_count>0)::UBIGINT AS aggregate_author_linked_work_count,
               count(*) FILTER (WHERE linked_author_count>0 AND incumbent_author_count=linked_author_count)::UBIGINT AS all_pre_signal_incumbent_work_count,
               count(*) FILTER (WHERE incumbent_author_count>0 AND incumbent_author_count<linked_author_count)::UBIGINT AS mixed_incumbent_new_work_count,
               count(*) FILTER (WHERE linked_author_count>0 AND incumbent_author_count=0)::UBIGINT AS no_pre_signal_incumbent_work_count,
               count(*) FILTER (WHERE linked_author_count=0)::UBIGINT AS no_linked_author_work_count,
               coalesce(sum(linked_author_count),0)::UBIGINT AS linked_author_incidence_count,
               coalesce(sum(incumbent_author_count),0)::UBIGINT AS pre_signal_incumbent_author_incidence_count
        FROM episode ep JOIN work_author_class wac ON wac.episode_id=ep.episode_id
        GROUP BY ALL ORDER BY ep.episode_id, wac.window
        """
    ).fetch_arrow_table().to_pylist()
    accounting = {
        "target_work_history_rows": int(con.execute("SELECT count(*) FROM target_work_history").fetchone()[0]),
        "target_author_event_rows": int(con.execute("SELECT count(*) FROM target_event").fetchone()[0]),
        "target_first_rows": int(con.execute("SELECT count(*) FROM target_first").fetchone()[0]),
        "episode_window_work_rows": int(con.execute("SELECT count(*) FROM target_window_work").fetchone()[0]),
    }
    con.close()
    return rows, accounting


def _validate_composition(rows: Sequence[Mapping[str, Any]], episode_rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == 2 * len(episode_rows) == 388, "composition row closure")
    by_key = {(int(row["episode_id"]), str(row["window"])): row for row in rows}
    require(len(by_key) == len(rows), "duplicate composition row")
    for episode in episode_rows:
        for window, aggregate_key, incumbent_key in (
            ("reference", "aggregate_reference_count", "all_incumbent_reference_work_count"),
            ("recent", "aggregate_recent_count", "all_incumbent_recent_work_count"),
        ):
            row = by_key[(int(episode["episode_id"]), window)]
            aggregate = int(row["all_pre_signal_incumbent_work_count"]) + int(row["mixed_incumbent_new_work_count"]) + int(row["no_pre_signal_incumbent_work_count"])
            require(aggregate == int(row["aggregate_author_linked_work_count"]), "linked category closure")
            require(aggregate == int(episode[aggregate_key]), "aggregate parent closure")
            require(int(row["all_pre_signal_incumbent_work_count"]) == int(episode[incumbent_key]), "incumbent parent closure")
            require(int(row["all_target_work_count"]) == aggregate + int(row["no_linked_author_work_count"]), "all-work closure")
            require(int(row["pre_signal_incumbent_author_incidence_count"]) <= int(row["linked_author_incidence_count"]), "author incidence closure")


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for window in ("reference", "recent"):
        selected = [row for row in rows if row["window"] == window]
        total = sum(int(row["aggregate_author_linked_work_count"]) for row in selected)
        values = {
            "aggregate_author_linked_work_count": total,
            "all_pre_signal_incumbent_work_count": sum(int(row["all_pre_signal_incumbent_work_count"]) for row in selected),
            "mixed_incumbent_new_work_count": sum(int(row["mixed_incumbent_new_work_count"]) for row in selected),
            "no_pre_signal_incumbent_work_count": sum(int(row["no_pre_signal_incumbent_work_count"]) for row in selected),
            "no_linked_author_work_count": sum(int(row["no_linked_author_work_count"]) for row in selected),
            "linked_author_incidence_count": sum(int(row["linked_author_incidence_count"]) for row in selected),
            "pre_signal_incumbent_author_incidence_count": sum(int(row["pre_signal_incumbent_author_incidence_count"]) for row in selected),
        }
        values["all_pre_signal_incumbent_work_share"] = values["all_pre_signal_incumbent_work_count"] / total
        values["mixed_incumbent_new_work_share"] = values["mixed_incumbent_new_work_count"] / total
        values["no_pre_signal_incumbent_work_share"] = values["no_pre_signal_incumbent_work_count"] / total
        values["pre_signal_incumbent_author_incidence_share"] = values["pre_signal_incumbent_author_incidence_count"] / values["linked_author_incidence_count"]
        result[window] = values
    return result


def _stage_and_field_summary(episodes: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate = {(int(row["field_id"]), int(row["topic_id"]), int(row["decision_year"])): row for row in candidates}
    enriched: list[dict[str, Any]] = []
    for row in episodes:
        key = (int(row["field_id"]), int(row["topic_id"]), int(row["decision_year"]))
        require(key in candidate, "episode missing from assignment candidates")
        bound = candidate[key]
        reference = int(row["all_incumbent_reference_work_count"])
        recent = int(row["all_incumbent_recent_work_count"])
        acceleration = math.log1p(recent / 3.0) - math.log1p(reference / 3.0)
        modified_z = 0.6744897501960817 * (acceleration - float(bound["field_year_median"])) / float(bound["field_year_mad"])
        count_up = recent > reference
        base_surge = count_up and acceleration > 0 and modified_z >= 3.0
        final_surge = int(row["network_leaveout_surge_opportunity_count"]) > 0
        enriched.append({**dict(row), "strict_count_up": count_up, "strict_base_surge": base_surge, "final_surge": final_surge})
    stages = {
        "aggregate_screen_episode_count": len(enriched),
        "strict_recent_greater_episode_count": sum(bool(row["strict_count_up"]) for row in enriched),
        "strict_recent_not_greater_episode_count": sum(not bool(row["strict_count_up"]) for row in enriched),
        "strict_with_aggregate_normalization_surge_episode_count": sum(bool(row["strict_base_surge"]) for row in enriched),
        "network_leaveout_surge_episode_count": sum(bool(row["final_surge"]) for row in enriched),
        "peer_leaveout_created_episode_count": sum(not bool(row["strict_base_surge"]) and bool(row["final_surge"]) for row in enriched),
        "peer_leaveout_killed_episode_count": sum(bool(row["strict_base_surge"]) and not bool(row["final_surge"]) for row in enriched),
        "focal_leaveout_incidence_count": sum(int(row["focal_excluded_attention_work_incidence_count"]) for row in enriched),
        "episodes_with_network_exclusion": sum(int(row["network_excluded_attention_work_incidence_count"]) > 0 for row in enriched),
        "episodes_with_changed_network_leaveout_median": sum(
            float(row["network_leaveout_reference_count_median"]) != float(row["all_incumbent_reference_work_count"])
            or float(row["network_leaveout_recent_count_median"]) != float(row["all_incumbent_recent_work_count"])
            for row in enriched
        ),
    }
    fields: list[dict[str, Any]] = []
    for field_id in sorted({int(row["field_id"]) for row in enriched}):
        selected = [row for row in enriched if int(row["field_id"]) == field_id]
        count_up = [row for row in selected if row["strict_count_up"]]
        base = [row for row in selected if row["strict_base_surge"]]
        final = [row for row in selected if row["final_surge"]]
        fields.append(
            {
                "field_id": field_id,
                "episode_count": len(selected),
                "aggregate_reference_count": sum(int(row["aggregate_reference_count"]) for row in selected),
                "aggregate_recent_count": sum(int(row["aggregate_recent_count"]) for row in selected),
                "strict_reference_count": sum(int(row["all_incumbent_reference_work_count"]) for row in selected),
                "strict_recent_count": sum(int(row["all_incumbent_recent_work_count"]) for row in selected),
                "strict_count_up_episode_count": len(count_up),
                "strict_count_up_family_count": len({int(row["subfield_id"]) for row in count_up}),
                "strict_aggregate_normalized_surge_episode_count": len(base),
                "strict_aggregate_normalized_surge_family_count": len({int(row["subfield_id"]) for row in base}),
                "network_leaveout_surge_episode_count": len(final),
                "network_leaveout_surge_family_count": len({int(row["subfield_id"]) for row in final}),
            }
        )
    return stages, fields


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    os.chmod(path, 0o600)


def _markdown(report: Mapping[str, Any]) -> str:
    totals = report["work_composition_totals"]
    lines = [
        "# Pivot-feedback F definition and theory audit v1",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Window | Aggregate linked works | All incumbent | Mixed team | No pre-signal incumbent |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for window in ("reference", "recent"):
        row = totals[window]
        lines.append(
            f"| {window} | {row['aggregate_author_linked_work_count']:,} | "
            f"{row['all_pre_signal_incumbent_work_count']:,} | {row['mixed_incumbent_new_work_count']:,} | "
            f"{row['no_pre_signal_incumbent_work_count']:,} |"
        )
    lines.extend(
        [
            "",
            "The conservative closed-team primary estimand is retained. The next authorized artifact must rescreen all eligible topics and compute field-year median/MAD in that same strict incumbent universe. Aggregate normalization reuse, threshold relaxation, field-order escape, confirmatory reads, F/G0 pass, model fitting, and G4 opening remain unauthorized.",
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


def build(*, config_path: Path, final_root: Path, assignment_root: Path, cutoff_root: Path, author_history_root: Path, output: Path, threads: int, memory_limit_gb: int) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= threads <= 32, "threads must be in [1,32]")
    require(8 <= memory_limit_gb <= 192, "memory limit must be in [8,192] GB")
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(config.get("status") == "FROZEN_DEFINITION_THEORY_AUDIT_PENDING_PRODUCTION_REPLAY_NO_F_PASS", "config status")
    authorization = _mapping(config["authorization_parent"], "authorization parent")
    parent_path = _safe_repo_path(authorization["relative_path"])
    require(_sha256(parent_path) == authorization["sha256"], "authorization parent SHA")
    parent_document = _yaml(parent_path)
    require(parent_document.get("schema_version") == authorization["required_schema_version"], "authorization parent schema")
    require(parent_document["authorization_after_block"].get(authorization["blocked_branch_authorization"]) == authorization["required_value"], "audit not authorized")
    repository_bindings = {
        "authorization_parent": {"relative_path": str(parent_path.relative_to(ROOT)), "sha256": authorization["sha256"]},
        "scientific_parent": _validate_repo_binding(_mapping(config["scientific_parent"], "scientific parent")),
    }
    episode_table, candidates, parent_bindings = _load_parents(config, final_root, assignment_root)
    work_glob, work_partitions = _validate_source_root(cutoff_root, _mapping(config["normalized_inputs"]["cutoff_work_metadata"], "work input"))
    event_glob, event_partitions = _validate_source_root(author_history_root, _mapping(config["normalized_inputs"]["author_history_events"], "event input"))
    primary = _mapping(config["primary_estimand_decision"], "primary estimand")
    aligned = _mapping(config["aligned_primary_replay"], "aligned primary replay")
    require(primary["primary_work_inclusion"] == "EVERY_LINKED_AUTHOR_IS_PRE_SIGNAL_INCUMBENT", "primary work rule")
    require(aligned["aggregate_median_mad_reuse"] == "FORBIDDEN", "aggregate normalization remains allowed")
    require(aligned["threshold_inclusive"] == 3.0 and aligned["threshold_relaxation"] is False, "threshold changed")
    completion = _mapping(config["completion_accounting"], "completion")
    for key in ("realized_citation_effect_read", "model_fitted", "holdout_byte_seal_complete", "g4_analytic_open", "pass_f", "pass_g0"):
        require(completion[key] is False, f"unsafe completion state: {key}")
    require(int(completion["confirmatory_entry_or_persistence_labels_read"]) == 0, "confirmatory labels read")

    output = output.resolve()
    require(output.name == config["output_contract"]["root_basename"], "output basename")
    require(not output.exists() and output.parent.is_dir(), "output must be new")
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    try:
        spill = temporary / "duckdb-spill"
        spill.mkdir(mode=0o700)
        rows, accounting = _composition_replay(episode_table, work_glob, event_glob, threads, memory_limit_gb, spill)
        episode_rows = episode_table.to_pylist()
        _validate_composition(rows, episode_rows)
        totals = _totals(rows)
        stages, fields = _stage_and_field_summary(episode_rows, candidates)
        reference = totals["reference"]
        recent = totals["recent"]
        aggregate_delta = int(recent["aggregate_author_linked_work_count"]) - int(reference["aggregate_author_linked_work_count"])
        contribution = {
            "aggregate_recent_minus_reference": aggregate_delta,
            "all_pre_signal_incumbent_recent_minus_reference": int(recent["all_pre_signal_incumbent_work_count"]) - int(reference["all_pre_signal_incumbent_work_count"]),
            "mixed_incumbent_new_recent_minus_reference": int(recent["mixed_incumbent_new_work_count"]) - int(reference["mixed_incumbent_new_work_count"]),
            "no_pre_signal_incumbent_recent_minus_reference": int(recent["no_pre_signal_incumbent_work_count"]) - int(reference["no_pre_signal_incumbent_work_count"]),
        }
        contribution["no_pre_signal_incumbent_share_of_net_aggregate_growth"] = contribution["no_pre_signal_incumbent_recent_minus_reference"] / aggregate_delta
        contribution["mixed_incumbent_new_share_of_net_aggregate_growth"] = contribution["mixed_incumbent_new_recent_minus_reference"] / aggregate_delta
        contribution["all_pre_signal_incumbent_share_of_net_aggregate_growth"] = contribution["all_pre_signal_incumbent_recent_minus_reference"] / aggregate_delta
        table_path = temporary / "episode_work_composition.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=COMPOSITION_SCHEMA), table_path, compression="zstd", compression_level=9, use_dictionary=True, write_statistics=True, version="2.6")
        os.chmod(table_path, 0o600)
        artifacts: list[dict[str, Any]] = [{"relative_path": table_path.name, "bytes": table_path.stat().st_size, "rows": len(rows), "sha256": _sha256(table_path)}]
        shutil.rmtree(spill)
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": STATUS,
            "repository_bindings": repository_bindings,
            "production_parent_bindings": parent_bindings,
            "input_bindings": {"work_partitions": work_partitions, "event_partitions": event_partitions, "cutoff_receipt_sha256": config["normalized_inputs"]["cutoff_work_metadata"]["receipt_sha256"], "author_history_receipt_sha256": config["normalized_inputs"]["author_history_events"]["receipt_sha256"]},
            "replay_accounting": accounting,
            "work_composition_totals": totals,
            "net_growth_decomposition": contribution,
            "stage_decomposition": stages,
            "field_decomposition": fields,
            "theory_decision": {"primary_estimand_id": primary["estimand_id"], "primary_work_inclusion": primary["primary_work_inclusion"], "mixed_team_primary_action": primary["mixed_incumbent_new_team_action"], "aligned_all_topic_rescreen_required": aligned["required"], "aggregate_median_mad_reuse": aligned["aggregate_median_mad_reuse"], "family_minimum_retained": aligned["minimum_distinct_current_snapshot_subfield_families"], "report_only_sensitivities": [row["id"] for row in config["precommitted_definition_sensitivities"]["definitions"]]},
            "authorization": config["authorization_after_verified_audit"],
            "scope": {"structural_result_already_observed": True, "confirmatory_entry_or_persistence_labels_read": 0, "realized_citation_effect_read": False, "raw_author_or_work_ids_emitted": False, "holdout_byte_seal_complete": False, "model_fitted": False, "g4_open": False, "pass_f": False, "pass_g0": False},
            "execution": {"config_relative_path": str(config_path.relative_to(ROOT)), "config_sha256": _sha256(config_path), "builder_relative_path": str(Path(__file__).resolve().relative_to(ROOT)), "builder_sha256": _sha256(Path(__file__).resolve()), "threads": threads, "memory_limit_gb": memory_limit_gb, "elapsed_seconds": time.monotonic() - started},
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "definition_theory_audit.json"
        _write_json(report_path, report)
        artifacts.append({"relative_path": report_path.name, "bytes": report_path.stat().st_size, "sha256": _sha256(report_path)})
        markdown_path = temporary / "DEFINITION_THEORY_AUDIT.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts.append({"relative_path": markdown_path.name, "bytes": markdown_path.stat().st_size, "sha256": _sha256(markdown_path)})
        artifacts.sort(key=lambda row: row["relative_path"])
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(_canonical_bytes([(row["relative_path"], row["sha256"]) for row in artifacts])).hexdigest(),
            "receipt_written_last": True,
            "independent_verification_complete": False,
            "definition_theory_audit_complete": True,
            "aligned_all_topic_rescreen_authorized_next": True,
            "confirmatory_entry_or_persistence_labels_read": 0,
            "raw_author_or_work_ids_emitted": False,
            "holdout_byte_seal_complete": False,
            "pass_f": False,
            "pass_g0": False,
            "model_fitted": False,
            "g4_open": False,
        }
        _write_json(temporary / "definition_theory_audit.outputs.json", receipt)
        os.rename(temporary, output)
        temporary = None
        return report
    except BaseException:
        _cleanup(temporary, output)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit-gb", type=int, default=96)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(config_path=args.config, final_root=args.final_root, assignment_root=args.assignment_root, cutoff_root=args.cutoff_root, author_history_root=args.author_history_root, output=args.output, threads=args.threads, memory_limit_gb=args.memory_limit_gb)
    print(json.dumps({"status": report["status"], "work_composition_totals": report["work_composition_totals"], "stage_decomposition": report["stage_decomposition"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
