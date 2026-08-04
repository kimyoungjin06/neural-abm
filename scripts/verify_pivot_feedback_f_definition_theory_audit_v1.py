#!/usr/bin/env python3
"""Independently verify the outcome-blind definition/theory audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
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
VERIFY_SCHEMA = "pivot_feedback_f_definition_theory_audit_verification.v1"
STATUS = "PASS_OUTCOME_BLIND_DEFINITION_THEORY_AUDIT_ALIGNED_PRIMARY_RESCREEN_AUTHORIZED_NO_F_PASS"
VERIFY_STATUS = "PASS_INDEPENDENT_DEFINITION_THEORY_AUDIT_RAW_COMPOSITION_REPLAY_NO_F_PASS"


class DefinitionAuditVerificationError(RuntimeError):
    """Fail-closed definition-audit verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DefinitionAuditVerificationError(message)


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


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


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


def _validate_repo_binding(binding: Mapping[str, Any]) -> None:
    path = _safe_repo_path(binding["relative_path"])
    require(_sha256(path) == binding["sha256"], f"binding SHA: {path.name}")
    document = _yaml(path) if path.suffix in (".yaml", ".yml") else _json(path)
    require(document.get("schema_version") == binding["required_schema_version"], f"binding schema: {path.name}")
    if "required_status" in binding:
        require(document.get("status") == binding["required_status"], f"binding status: {path.name}")


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = receipt.get("artifacts")
    require(isinstance(rows, list), "receipt artifacts")
    result = {str(row["relative_path"]): dict(row) for row in rows}
    require(len(result) == len(rows), "duplicate receipt artifact")
    return result


def _validate_output(output_root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], pa.Table, str]:
    root = output_root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid output root")
    require(root.name == config["output_contract"]["root_basename"], "output root basename")
    receipt_path = root / "definition_theory_audit.outputs.json"
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(receipt.get("status") == STATUS, "receipt status")
    require(receipt.get("receipt_written_last") is True, "receipt order")
    require(receipt.get("definition_theory_audit_complete") is True, "audit incomplete")
    require(receipt.get("aligned_all_topic_rescreen_authorized_next") is True, "aligned rescreen not authorized")
    for key in ("raw_author_or_work_ids_emitted", "holdout_byte_seal_complete", "pass_f", "pass_g0", "model_fitted", "g4_open"):
        require(receipt.get(key) is False, f"unsafe receipt state: {key}")
    require(receipt.get("confirmatory_entry_or_persistence_labels_read") == 0, "confirmatory labels read")
    artifacts = _artifact_map(receipt)
    require(set(artifacts) == {"DEFINITION_THEORY_AUDIT.md", "definition_theory_audit.json", "episode_work_composition.parquet"}, "artifact set")
    verified_pairs: list[tuple[str, str]] = []
    for relative, record in sorted(artifacts.items()):
        pure = PurePosixPath(relative)
        require(not pure.is_absolute() and ".." not in pure.parts and len(pure.parts) == 1, "unsafe artifact path")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "invalid artifact")
        require(path.stat().st_size == int(record["bytes"]), f"artifact bytes: {relative}")
        digest = _sha256(path)
        require(digest == record["sha256"], f"artifact SHA: {relative}")
        verified_pairs.append((relative, digest))
    tree = hashlib.sha256(_canonical_bytes(verified_pairs)).hexdigest()
    require(tree == receipt.get("artifact_tree_sha256"), "artifact tree SHA")
    report = _json(root / "definition_theory_audit.json")
    require(report.get("schema_version") == REPORT_SCHEMA and report.get("status") == STATUS, "report identity")
    table = pq.read_table(root / "episode_work_composition.parquet")
    require((table.schema.metadata or {}).get(b"schema_version", b"").decode() == COMPOSITION_SCHEMA_VERSION, "composition schema")
    require(len(table) == 388, "composition rows")
    forbidden = {"author_id", "work_id", "author_numeric_id", "work_numeric_id"}
    require(not forbidden.intersection(table.column_names), "raw ID in composition output")
    return receipt, report, table, _sha256(receipt_path)


def _validate_parent(config: Mapping[str, Any], final_root: Path) -> pa.Table:
    binding = _mapping(config["blocked_replay_parent"], "blocked parent")
    root = final_root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "blocked root")
    receipt_path = root / str(binding["receipt_basename"])
    episode_path = root / str(binding["episode_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "blocked receipt SHA")
    require(_json(receipt_path).get("status") == binding["required_status"], "blocked receipt status")
    require(_sha256(episode_path) == binding["episode_sha256"], "blocked episode SHA")
    table = pq.read_table(episode_path)
    require(len(table) == int(binding["episode_rows"]), "blocked episode rows")
    verification = Path(str(binding["verification_path"])).resolve(strict=True)
    require(_sha256(verification) == binding["verification_sha256"], "blocked verification SHA")
    require(_json(verification).get("status") == binding["verification_status"], "blocked verification status")
    return table


def _source_glob(root: Path, binding: Mapping[str, Any]) -> Path:
    resolved = root.resolve(strict=True)
    require(resolved.name == binding["expected_root_basename"], "source root")
    receipt = resolved / str(binding["receipt_basename"])
    require(_sha256(receipt) == binding["receipt_sha256"], "source receipt SHA")
    prefix = str(binding["artifact_prefix"])
    paths = list((resolved / prefix.rstrip("/")).glob("*.parquet"))
    require(len(paths) == int(binding["artifact_count"]), "source partition count")
    return resolved / prefix.rstrip("/") / "*.parquet"


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _independent_replay(episodes: pa.Table, work_glob: Path, event_glob: Path, threads: int, memory_limit_gb: int, temp_dir: Path) -> list[dict[str, Any]]:
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{_sql_path(temp_dir)}'")
    con.register("episodes_arrow", episodes.select(["episode_id", "field_id", "topic_id", "decision_year"]))
    con.execute("CREATE TABLE ep AS SELECT * FROM episodes_arrow")
    con.execute("CREATE TABLE wanted AS SELECT DISTINCT field_id,topic_id FROM ep")
    con.execute(
        f"""
        CREATE TABLE works AS
        SELECT w.work_id_numeric work_id,w.publication_year,w.primary_field_id field_id,
               w.primary_topic_id topic_id,w.temporal_disposition
        FROM read_parquet('{_sql_path(work_glob)}') w
        SEMI JOIN wanted p ON p.field_id=w.primary_field_id AND p.topic_id=w.primary_topic_id
        WHERE w.publication_year<=2021 AND w.temporal_disposition IN (2,3)
          AND w.work_type_code IN (1,2,3,4,7,16,18,21)
          AND NOT w.is_paratext AND NOT coalesce(w.is_retracted,false)
        """
    )
    con.execute("CREATE INDEX works_id ON works(work_id)")
    con.execute(
        f"""
        CREATE TABLE authorships AS
        SELECT DISTINCT e.author_numeric_id author_id,w.work_id,w.publication_year,w.field_id,w.topic_id
        FROM read_parquet('{_sql_path(event_glob)}') e JOIN works w ON w.work_id=e.work_numeric_id
        WHERE e.publication_year=w.publication_year AND e.primary_field_id=w.field_id
          AND e.temporal_disposition=w.temporal_disposition
        """
    )
    con.execute("CREATE INDEX auth_topic_author ON authorships(field_id,topic_id,author_id)")
    con.execute(
        """
        CREATE TABLE first_seen AS
        SELECT ep.episode_id,a.author_id,min(a.publication_year) first_year
        FROM ep JOIN authorships a USING(field_id,topic_id)
        WHERE a.publication_year<=ep.decision_year-1
        GROUP BY ep.episode_id,a.author_id
        """
    )
    con.execute("CREATE INDEX first_seen_key ON first_seen(episode_id,author_id)")
    con.execute(
        """
        CREATE TABLE window_works AS
        SELECT ep.episode_id,ep.field_id,ep.topic_id,ep.decision_year,w.work_id,
               if(w.publication_year<=ep.decision_year-4,'reference','recent') window_name
        FROM ep JOIN works w USING(field_id,topic_id)
        WHERE w.publication_year BETWEEN ep.decision_year-6 AND ep.decision_year-1
        """
    )
    replay = con.execute(
        """
        WITH classified AS (
          SELECT ww.episode_id,ww.field_id,ww.topic_id,ww.decision_year,ww.work_id,ww.window_name,
                 count(a.author_id) linked,
                 count(a.author_id) FILTER (WHERE fs.first_year<ww.decision_year-6) incumbent
          FROM window_works ww
          LEFT JOIN authorships a ON a.work_id=ww.work_id
          LEFT JOIN first_seen fs ON fs.episode_id=ww.episode_id AND fs.author_id=a.author_id
          GROUP BY ALL
        )
        SELECT episode_id,field_id,topic_id,decision_year,window_name AS window,
               count(*)::UBIGINT all_target_work_count,
               count(*) FILTER (WHERE linked>0)::UBIGINT aggregate_author_linked_work_count,
               count(*) FILTER (WHERE linked>0 AND incumbent=linked)::UBIGINT all_pre_signal_incumbent_work_count,
               count(*) FILTER (WHERE incumbent>0 AND incumbent<linked)::UBIGINT mixed_incumbent_new_work_count,
               count(*) FILTER (WHERE linked>0 AND incumbent=0)::UBIGINT no_pre_signal_incumbent_work_count,
               count(*) FILTER (WHERE linked=0)::UBIGINT no_linked_author_work_count,
               sum(linked)::UBIGINT linked_author_incidence_count,
               sum(incumbent)::UBIGINT pre_signal_incumbent_author_incidence_count
        FROM classified GROUP BY ALL ORDER BY episode_id,window_name
        """
    ).to_arrow_table().to_pylist()
    con.close()
    return replay


def _compare_rows(observed: Sequence[Mapping[str, Any]], replayed: Sequence[Mapping[str, Any]], parent: Sequence[Mapping[str, Any]]) -> None:
    keys = ("episode_id", "window")
    observed_map = {tuple(row[key] for key in keys): dict(row) for row in observed}
    replayed_map = {tuple(row[key] for key in keys): dict(row) for row in replayed}
    require(observed_map == replayed_map, "raw composition replay mismatch")
    parent_map = {int(row["episode_id"]): row for row in parent}
    for (episode_id, window), row in observed_map.items():
        source = parent_map[int(episode_id)]
        aggregate_key = "aggregate_reference_count" if window == "reference" else "aggregate_recent_count"
        incumbent_key = "all_incumbent_reference_work_count" if window == "reference" else "all_incumbent_recent_work_count"
        linked_sum = int(row["all_pre_signal_incumbent_work_count"]) + int(row["mixed_incumbent_new_work_count"]) + int(row["no_pre_signal_incumbent_work_count"])
        require(linked_sum == int(row["aggregate_author_linked_work_count"]) == int(source[aggregate_key]), "aggregate closure")
        require(int(row["all_pre_signal_incumbent_work_count"]) == int(source[incumbent_key]), "incumbent closure")


def verify(*, config_path: Path, final_root: Path, cutoff_root: Path, author_history_root: Path, output_root: Path, verification_path: Path, threads: int = 16, memory_limit_gb: int = 96) -> dict[str, Any]:
    require(1 <= threads <= 32, "threads must be in [1,32]")
    require(8 <= memory_limit_gb <= 192, "memory limit must be in [8,192] GB")
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    authorization = _mapping(config["authorization_parent"], "authorization parent")
    _validate_repo_binding(authorization)
    auth_document = _yaml(_safe_repo_path(authorization["relative_path"]))
    require(auth_document["authorization_after_block"].get(authorization["blocked_branch_authorization"]) == authorization["required_value"], "audit authorization")
    _validate_repo_binding(_mapping(config["scientific_parent"], "scientific parent"))
    primary = _mapping(config["primary_estimand_decision"], "primary")
    aligned = _mapping(config["aligned_primary_replay"], "aligned")
    require(primary["primary_work_inclusion"] == "EVERY_LINKED_AUTHOR_IS_PRE_SIGNAL_INCUMBENT", "primary relaxed")
    require(aligned["aggregate_median_mad_reuse"] == "FORBIDDEN", "aggregate normalization allowed")
    require(aligned["threshold_inclusive"] == 3.0 and aligned["threshold_relaxation"] is False, "threshold relaxed")
    receipt, report, observed_table, receipt_sha = _validate_output(output_root, config)
    parent_table = _validate_parent(config, final_root)
    work_glob = _source_glob(cutoff_root, _mapping(config["normalized_inputs"]["cutoff_work_metadata"], "work input"))
    event_glob = _source_glob(author_history_root, _mapping(config["normalized_inputs"]["author_history_events"], "event input"))
    with tempfile.TemporaryDirectory(prefix=".pivot-definition-verification-", dir=verification_path.parent) as temp:
        replayed = _independent_replay(parent_table, work_glob, event_glob, threads, memory_limit_gb, Path(temp))
    observed = observed_table.to_pylist()
    _compare_rows(observed, replayed, parent_table.to_pylist())
    totals: dict[str, dict[str, int]] = {}
    for window in ("reference", "recent"):
        selected = [row for row in observed if row["window"] == window]
        totals[window] = {name: sum(int(row[name]) for row in selected) for name in ("aggregate_author_linked_work_count", "all_pre_signal_incumbent_work_count", "mixed_incumbent_new_work_count", "no_pre_signal_incumbent_work_count")}
        for name, value in totals[window].items():
            require(value == int(report["work_composition_totals"][window][name]), f"report total: {window}/{name}")
    require(totals["reference"] == {"aggregate_author_linked_work_count": 494218, "all_pre_signal_incumbent_work_count": 39507, "mixed_incumbent_new_work_count": 139434, "no_pre_signal_incumbent_work_count": 315277}, "reference diagnostic closure")
    require(totals["recent"] == {"aggregate_author_linked_work_count": 1141266, "all_pre_signal_incumbent_work_count": 31221, "mixed_incumbent_new_work_count": 243767, "no_pre_signal_incumbent_work_count": 866278}, "recent diagnostic closure")
    stage = report["stage_decomposition"]
    require(stage["strict_recent_not_greater_episode_count"] == 143, "stage reversal count")
    require(stage["strict_with_aggregate_normalization_surge_episode_count"] == 15, "stage z count")
    require(stage["network_leaveout_surge_episode_count"] == 19, "stage leaveout count")
    require(stage["peer_leaveout_created_episode_count"] == 4 and stage["peer_leaveout_killed_episode_count"] == 0, "peer role")
    require(stage["focal_leaveout_incidence_count"] == 0, "focal role")
    authorization_after = report["authorization"]
    require(authorization_after["strict_incumbent_aligned_all_topic_rescreen"] == "AUTHORIZED_NEXT", "next authorization")
    for key in ("aggregate_screen_reuse_as_primary", "threshold_relaxation", "field_outside_frozen_order", "operator_only_private_holdout_materialization", "confirmatory_analytic_read", "model_fitting", "g4_open"):
        require(authorization_after[key] == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    verification = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "config_relative_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "output_root_basename": output_root.resolve().name,
        "output_receipt_sha256": receipt_sha,
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "composition_rows_replayed": len(replayed),
        "raw_source_composition_replay_match": True,
        "aggregate_parent_closure": True,
        "all_incumbent_parent_closure": True,
        "primary_estimand_retained": True,
        "aligned_all_topic_rescreen_authorized_next": True,
        "confirmatory_entry_or_persistence_labels_read": 0,
        "raw_author_or_work_ids_emitted": False,
        "holdout_byte_seal_complete": False,
        "pass_f": False,
        "pass_g0": False,
        "model_fitted": False,
        "g4_open": False,
    }
    verification_path = verification_path.resolve()
    require(not verification_path.exists() and verification_path.parent.is_dir(), "verification output must be new")
    verification_path.write_bytes(json.dumps(verification, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    os.chmod(verification_path, 0o600)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-limit-gb", type=int, default=96)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(config_path=args.config, final_root=args.final_root, cutoff_root=args.cutoff_root, author_history_root=args.author_history_root, output_root=args.output_root, verification_path=args.verification_output, threads=args.threads, memory_limit_gb=args.memory_limit_gb)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
