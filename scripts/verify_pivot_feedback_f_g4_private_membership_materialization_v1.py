#!/usr/bin/env python3
"""Independently verify private G4 risk rows and the author-cold guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_g4_private_membership_materialization_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_g4_private_membership_materialization.v1"
VERIFY_SCHEMA = "pivot_feedback_f_g4_private_membership_materialization_verification.v1"
PASS_STATUS = "PASS_INDEPENDENT_PRIVATE_G4_PREDECISION_MEMBERSHIP_AUTHOR_COLD_GUARD_NO_OUTCOME_READ_NO_F_PASS"
RISK_BASENAME = re.compile(r"episode-(\d{4})\.parquet")

RISK_FIELDS = [
    ("episode_id", "uint16", False),
    ("author_numeric_id", "uint64", False),
    ("field_id", "uint8", False),
    ("topic_id", "uint32", False),
    ("subfield_id", "uint16", False),
    ("decision_year", "uint16", False),
    ("base_reference_count", "uint64", False),
    ("base_recent_count", "uint64", False),
    ("excluded_reference_count", "uint64", False),
    ("excluded_recent_count", "uint64", False),
    ("network_leaveout_reference_count", "uint64", False),
    ("network_leaveout_recent_count", "uint64", False),
    ("prior_target_active_peer_count", "uint64", False),
    ("network_leaveout_surge_present", "bool", False),
]

GUARD_FIELDS = [
    ("author_numeric_id", "uint64", False),
    ("first_g4_decision_year", "uint16", False),
    ("last_g4_decision_year", "uint16", False),
    ("g4_episode_count", "uint16", False),
    ("exclude_from_discovery", "bool", False),
    ("exclude_from_g1", "bool", False),
    ("exclude_from_model_selection", "bool", False),
]

EPISODE_FIELDS = [
    ("episode_id", "uint16", False),
    ("field_id", "uint8", False),
    ("topic_id", "uint32", False),
    ("subfield_id", "uint16", False),
    ("decision_year", "uint16", False),
    ("expected_risk_opportunity_count", "uint64", False),
    ("materialized_risk_opportunity_count", "uint64", False),
    ("expected_network_leaveout_surge_count", "uint64", False),
    ("materialized_network_leaveout_surge_count", "uint64", False),
    ("risk_count_closure", "bool", False),
    ("surge_count_closure", "bool", False),
    ("sealed", "bool", False),
    ("operator_only", "bool", False),
    ("author_cold_guard_bound", "bool", False),
]


class G4PrivateMembershipVerificationError(RuntimeError):
    """Fail-closed private G4 membership verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G4PrivateMembershipVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _safe_repo_path(relative: Any) -> Path:
    require(isinstance(relative, str) and relative, "empty repository path")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe repository path")
    path = (ROOT / pure).resolve(strict=True)
    path.relative_to(ROOT.resolve())
    require(path.is_file() and not path.is_symlink(), "invalid repository path")
    return path


def _assert_schema(path: Path, expected: Sequence[tuple[str, str, bool]], version: str) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    observed = [(field.name, str(field.type), field.nullable) for field in schema]
    require(observed == list(expected), f"schema fields: {path.name}")
    require((schema.metadata or {}).get(b"schema_version", b"").decode() == version, f"schema version: {path.name}")


def _verify_repository_bindings(config: Mapping[str, Any]) -> None:
    authorization = _mapping(config["authorization_parent"], "authorization parent")
    authorization_path = _safe_repo_path(authorization["relative_path"])
    require(_sha256(authorization_path) == authorization["sha256"], "authorization SHA")
    authorization_doc = _yaml(authorization_path)
    require(authorization_doc.get("schema_version") == authorization["required_schema_version"], "authorization schema")
    require(
        authorization_doc["authorization_after_pass"][authorization["required_authorization"]]
        == authorization["required_value"],
        "private materialization authorization",
    )
    sources = config["bound_design_sources"]
    summary_binding = sources["atomic_partition_summary"]
    summary_path = _safe_repo_path(summary_binding["relative_path"])
    require(_sha256(summary_path) == summary_binding["sha256"], "atomic summary SHA")
    summary = _json(summary_path)
    require(summary.get("schema_version") == summary_binding["required_schema_version"], "atomic summary schema")
    require(summary.get("status") == summary_binding["required_status"], "atomic summary status")
    scientific_binding = sources["scientific_sections"]
    scientific_path = _safe_repo_path(scientific_binding["relative_path"])
    require(_sha256(scientific_path) == scientific_binding["sha256"], "scientific SHA")
    scientific = _yaml(scientific_path)
    require(scientific.get("schema_version") == scientific_binding["required_schema_version"], "scientific schema")
    section = scientific["sections"][scientific_binding["required_section"]]
    require(section["atomicity"]["g4_authors_removed_from_all_earlier_fit_or_selection_rows"] is True, "author-cold rule")


def _verify_parent_binding(root: Path, binding: Mapping[str, Any]) -> dict[str, Path]:
    root = root.resolve(strict=True)
    require(root.name == binding["expected_root_basename"], "parent basename")
    require(root.is_dir() and not root.is_symlink(), "parent root")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "parent receipt SHA")
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "parent receipt schema")
    require(receipt.get("status") == binding["required_status"], "parent receipt status")
    for key in ("pass_f", "pass_g0", "g4_open", "model_fitted"):
        require(receipt.get(key) is False, f"unsafe parent state: {key}")
    paths = {}
    for prefix in ("episode", "family", "partition", "slot"):
        if f"{prefix}_basename" not in binding:
            continue
        path = root / str(binding[f"{prefix}_basename"])
        require(_sha256(path) == binding[f"{prefix}_sha256"], f"parent {prefix} SHA")
        require(pq.ParquetFile(path).metadata.num_rows == binding[f"{prefix}_rows"], f"parent {prefix} rows")
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        require(metadata.get(b"schema_version", b"").decode() == binding[f"{prefix}_schema_version"], f"parent {prefix} schema")
        paths[prefix] = path
    if "verification_path" in binding:
        verification_path = Path(str(binding["verification_path"])).resolve(strict=True)
        require(_sha256(verification_path) == binding["verification_sha256"], "parent verification SHA")
        require(_json(verification_path).get("status") == binding["verification_status"], "parent verification status")
    return paths


def _expected_episodes(
    atomic_root: Path,
    final_panel_root: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    atomic_paths = _verify_parent_binding(atomic_root, config["verified_atomic_partition_parent"])
    final_paths = _verify_parent_binding(final_panel_root, config["verified_final_panel_parent"])
    atomic = pq.read_table(atomic_paths["episode"]).to_pylist()
    final = {int(row["episode_id"]): row for row in pq.read_table(final_paths["episode"]).to_pylist()}
    contract = config["private_membership_contract"]
    selected = [row for row in atomic if row["registered_partition_id"] == contract["partition_id"]]
    require(len(selected) == contract["episode_count"] == 37, "expected G4 episodes")
    expected = []
    for row in sorted(selected, key=lambda item: int(item["episode_id"])):
        episode_id = int(row["episode_id"])
        parent = final[episode_id]
        require(parent["sealed_slot"] and parent["leaveout_surge_present"], "expected G4 conformance")
        expected.append(
            {
                "episode_id": episode_id,
                "field_id": int(row["field_id"]),
                "topic_id": int(row["topic_id"]),
                "subfield_id": int(row["subfield_id"]),
                "decision_year": int(row["decision_year"]),
                "field_year_median": float(parent["aligned_field_year_median"]),
                "field_year_mad": float(parent["aligned_field_year_mad"]),
                "base_reference_count": int(parent["aligned_reference_count"]),
                "base_recent_count": int(parent["aligned_recent_count"]),
                "expected_risk_opportunity_count": int(parent["risk_opportunity_count"]),
                "expected_network_leaveout_surge_count": int(parent["network_leaveout_surge_opportunity_count"]),
            }
        )
    require(sum(row["expected_risk_opportunity_count"] for row in expected) == contract["expected_risk_opportunity_rows"], "expected G4 risk total")
    return expected


def _verify_output_tree(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[Path]]:
    require(root.is_dir() and not root.is_symlink(), "output root")
    require((root.stat().st_mode & 0o777) == 0o700, "output root mode")
    directories = [path for path in root.rglob("*") if path.is_dir()]
    require(all(not path.is_symlink() and (path.stat().st_mode & 0o777) == 0o700 for path in directories), "nested directory mode")
    files = [path for path in root.rglob("*") if path.is_file()]
    require(all(not path.is_symlink() and (path.stat().st_mode & 0o777) == 0o600 for path in files), "output file mode")
    receipt_path = root / "g4_private_membership_materialization.outputs.json"
    require(receipt_path in files, "missing receipt")
    receipt = _json(receipt_path)
    output_contract = config["output_contract"]
    require(receipt.get("schema_version") == output_contract["receipt_schema_version"], "receipt schema")
    require(receipt.get("status") == output_contract["passing_status"], "receipt status")
    require(receipt.get("receipt_written_last") is True, "receipt last flag")
    artifacts = {str(row["relative_path"]): dict(row) for row in receipt.get("artifacts", [])}
    actual_relatives = {path.relative_to(root).as_posix() for path in files if path != receipt_path}
    require(set(artifacts) == actual_relatives, "receipt artifact set")
    require(receipt.get("artifact_count") == len(artifacts), "receipt artifact count")
    tree = []
    for relative in sorted(artifacts):
        path = root / relative
        row = artifacts[relative]
        require(path.stat().st_size == row["bytes"], f"artifact bytes: {relative}")
        require(_sha256(path) == row["sha256"], f"artifact SHA: {relative}")
        if path.suffix == ".parquet":
            require(pq.ParquetFile(path).metadata.num_rows == row["rows"], f"artifact rows: {relative}")
        tree.append((relative, row["bytes"], row["sha256"]))
    require(hashlib.sha256(_canonical_bytes(tree)).hexdigest() == receipt["artifact_tree_sha256"], "artifact tree SHA")
    require(receipt_path.stat().st_mtime_ns > max((root / relative).stat().st_mtime_ns for relative in artifacts), "receipt not written last")
    require(receipt["g4_private_predecision_membership_complete"] is True, "membership incomplete")
    require(receipt["g4_author_cold_guard_complete"] is True, "guard incomplete")
    for key in (
        "earlier_partition_guard_application_complete",
        "complete_private_g4_artifact_set",
        "entry_or_persistence_labels_materialized",
        "postdecision_target_source_materialized",
        "holdout_preflight_complete",
        "holdout_byte_seal_complete",
        "model_score_read",
        "model_fitted",
        "g4_open",
        "pass_f",
        "pass_g0",
    ):
        require(receipt[key] is False, f"unsafe receipt state: {key}")
    require(receipt["confirmatory_or_holdout_outcome_rows_read"] == 0, "outcome rows read")
    risk_paths = sorted((root / "private/g4_risk_measurement").glob("*.parquet"))
    require(len(risk_paths) == output_contract["risk_file_count"] == 37, "risk file count")
    return receipt, artifacts, risk_paths


def _episode_arrow(expected: Sequence[Mapping[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist([dict(row) for row in expected])


def _verify_private_rows(
    root: Path,
    risk_paths: Sequence[Path],
    expected: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    threads: int,
    memory_limit_gb: int,
) -> dict[str, Any]:
    expected_ids = {int(row["episode_id"]) for row in expected}
    observed_ids = set()
    for path in risk_paths:
        match = RISK_BASENAME.fullmatch(path.name)
        require(match is not None, f"risk basename: {path.name}")
        observed_ids.add(int(match.group(1)))
        _assert_schema(path, RISK_FIELDS, config["output_contract"]["risk_schema_version"])
    require(observed_ids == expected_ids, "risk episode file set")
    guard_path = root / "private/g4_author_cold_guard.parquet"
    episode_path = root / "private/g4_episode_manifest.parquet"
    _assert_schema(guard_path, GUARD_FIELDS, config["output_contract"]["guard_schema_version"])
    _assert_schema(episode_path, EPISODE_FIELDS, config["output_contract"]["episode_schema_version"])
    forbidden = {
        "work_id",
        "work_numeric_id",
        "entry_label",
        "persistence_label",
        "realized_citation_count",
        "model_score",
    }
    for path in [*risk_paths, guard_path, episode_path]:
        require(not (forbidden & set(pq.ParquetFile(path).schema_arrow.names)), f"forbidden private column: {path.name}")

    episode_manifest = pq.read_table(episode_path).to_pylist()
    require(len(episode_manifest) == len(expected) == 37, "episode manifest rows")
    manifest_by_id = {int(row["episode_id"]): row for row in episode_manifest}
    expected_by_id = {int(row["episode_id"]): row for row in expected}
    for episode_id, parent in expected_by_id.items():
        row = manifest_by_id[episode_id]
        for key in ("field_id", "topic_id", "subfield_id", "decision_year"):
            require(int(row[key]) == int(parent[key]), f"episode manifest parent: {episode_id} {key}")
        require(int(row["expected_risk_opportunity_count"]) == parent["expected_risk_opportunity_count"], "manifest risk expectation")
        require(int(row["expected_network_leaveout_surge_count"]) == parent["expected_network_leaveout_surge_count"], "manifest surge expectation")
        require(row["risk_count_closure"] and row["surge_count_closure"], "manifest count closure")
        require(row["sealed"] and row["operator_only"] and row["author_cold_guard_bound"], "manifest private boundary")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.register("expected_episode", _episode_arrow(expected))
    risk_sql = "read_parquet([" + ",".join(f"'{str(path).replace(chr(39), chr(39)*2)}'" for path in risk_paths) + "])"
    guard_sql = f"read_parquet('{str(guard_path).replace(chr(39), chr(39)*2)}')"
    counts = {
        int(row[0]): {"risk": int(row[1]), "surge": int(row[2])}
        for row in con.execute(
            f"""
            SELECT episode_id, count(*)::UBIGINT,
                   count(*) FILTER (WHERE network_leaveout_surge_present)::UBIGINT
            FROM {risk_sql} GROUP BY episode_id ORDER BY episode_id
            """
        ).fetchall()
    }
    require(set(counts) == expected_ids, "risk count episode set")
    for episode_id, parent in expected_by_id.items():
        require(counts[episode_id]["risk"] == parent["expected_risk_opportunity_count"], f"risk row count: {episode_id}")
        require(counts[episode_id]["surge"] == parent["expected_network_leaveout_surge_count"], f"surge row count: {episode_id}")
        row = manifest_by_id[episode_id]
        require(int(row["materialized_risk_opportunity_count"]) == counts[episode_id]["risk"], "materialized risk manifest")
        require(int(row["materialized_network_leaveout_surge_count"]) == counts[episode_id]["surge"], "materialized surge manifest")
    bad_rows = int(
        con.execute(
            f"""
            SELECT count(*) FROM {risk_sql} r
            JOIN expected_episode e USING (episode_id)
            WHERE r.field_id<>e.field_id OR r.topic_id<>e.topic_id
               OR r.subfield_id<>e.subfield_id OR r.decision_year<>e.decision_year
               OR r.base_reference_count<>e.base_reference_count
               OR r.base_recent_count<>e.base_recent_count
               OR r.excluded_reference_count>r.base_reference_count
               OR r.excluded_recent_count>r.base_recent_count
               OR r.network_leaveout_reference_count<>r.base_reference_count-r.excluded_reference_count
               OR r.network_leaveout_recent_count<>r.base_recent_count-r.excluded_recent_count
               OR r.network_leaveout_surge_present<>(
                 r.network_leaveout_recent_count>r.network_leaveout_reference_count
                 AND ln(1+r.network_leaveout_recent_count/3.0)-ln(1+r.network_leaveout_reference_count/3.0)>0
                 AND 0.6744897501960817*((ln(1+r.network_leaveout_recent_count/3.0)-ln(1+r.network_leaveout_reference_count/3.0))-e.field_year_median)/e.field_year_mad>=3.0
               )
            """
        ).fetchone()[0]
    )
    require(bad_rows == 0, "private risk algebra or parent mismatch")
    duplicate_groups = int(
        con.execute(
            f"""
            SELECT count(*) FROM (
              SELECT episode_id, author_numeric_id, count(*) AS n
              FROM {risk_sql} GROUP BY episode_id, author_numeric_id HAVING n<>1
            )
            """
        ).fetchone()[0]
    )
    require(duplicate_groups == 0, "duplicate private risk key")
    guard_stats = con.execute(
        f"""
        SELECT count(*)::UBIGINT, count(DISTINCT author_numeric_id)::UBIGINT,
               count(*) FILTER (
                 WHERE NOT exclude_from_discovery OR NOT exclude_from_g1
                    OR NOT exclude_from_model_selection
                    OR first_g4_decision_year<2020 OR last_g4_decision_year>2022
                    OR g4_episode_count<1 OR g4_episode_count>37
               )::UBIGINT
        FROM {guard_sql}
        """
    ).fetchone()
    require(int(guard_stats[0]) == int(guard_stats[1]) and int(guard_stats[2]) == 0, "author-cold guard integrity")
    projection_missing = int(
        con.execute(
            f"""
            SELECT count(*) FROM (
              SELECT DISTINCT author_numeric_id FROM {risk_sql}
              EXCEPT SELECT author_numeric_id FROM {guard_sql}
            )
            """
        ).fetchone()[0]
    )
    projection_extra = int(
        con.execute(
            f"""
            SELECT count(*) FROM (
              SELECT author_numeric_id FROM {guard_sql}
              EXCEPT SELECT DISTINCT author_numeric_id FROM {risk_sql}
            )
            """
        ).fetchone()[0]
    )
    require(projection_missing == 0 and projection_extra == 0, "author-cold guard projection mismatch")
    con.close()
    return {
        "risk_opportunity_rows": sum(row["risk"] for row in counts.values()),
        "network_leaveout_surge_rows": sum(row["surge"] for row in counts.values()),
        "author_cold_guard_rows": int(guard_stats[0]),
        "risk_episode_files": len(risk_paths),
        "risk_key_duplicates": duplicate_groups,
        "guard_projection_missing": projection_missing,
        "guard_projection_extra": projection_extra,
    }


def verify(
    *,
    config_path: Path,
    atomic_root: Path,
    final_panel_root: Path,
    output_root: Path,
    verification_output: Path,
    threads: int,
    memory_limit_gb: int,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    _verify_repository_bindings(config)
    expected = _expected_episodes(atomic_root, final_panel_root, config)
    output_root = output_root.resolve(strict=True)
    require(output_root.name == config["output_contract"]["root_basename"], "output basename")
    verification_output = verification_output.resolve()
    require(not verification_output.exists() and verification_output.parent.is_dir(), "verification output must be new")
    receipt, artifacts, risk_paths = _verify_output_tree(output_root, config)
    private_check = _verify_private_rows(
        output_root,
        risk_paths,
        expected,
        config,
        threads=threads,
        memory_limit_gb=memory_limit_gb,
    )
    report = _json(output_root / "g4_private_membership_materialization.json")
    require(report.get("schema_version") == config["output_contract"]["report_schema_version"], "report schema")
    require(report.get("status") == receipt["status"], "report status")
    execution = report["execution"]
    require(execution["config_relative_path"] == config_path.relative_to(ROOT).as_posix(), "executed config path")
    require(execution["config_sha256"] == _sha256(config_path), "executed config SHA")
    builder = _safe_repo_path(execution["builder_relative_path"])
    require(_sha256(builder) == execution["builder_sha256"], "executed builder SHA")
    accounting = report["accounting"]
    for key in ("risk_opportunity_rows", "network_leaveout_surge_rows", "author_cold_guard_rows", "risk_file_count"):
        require(int(accounting[key]) == int(private_check[key if key != "risk_file_count" else "risk_episode_files"]), f"report accounting: {key}")
    require(accounting["raw_work_ids_emitted"] is False, "raw work IDs emitted")
    for key in ("entry_or_persistence_labels_emitted", "postdecision_target_events_emitted", "realized_citation_effects_emitted", "model_scores_emitted"):
        require(accounting[key] is False, f"unsafe accounting: {key}")
    completion = report["completion"]
    require(completion["g4_private_predecision_membership_complete"] is True, "report membership state")
    require(completion["g4_author_cold_guard_complete"] is True, "report guard state")
    for key in (
        "earlier_partition_guard_application_complete",
        "complete_private_g4_artifact_set",
        "entry_or_persistence_labels_materialized",
        "postdecision_target_source_materialized",
        "holdout_preflight_complete",
        "holdout_byte_seal_complete",
        "model_fitted",
        "g4_open",
        "pass_f",
        "pass_g0",
    ):
        require(completion[key] is False, f"unsafe completion state: {key}")
    require(completion["confirmatory_or_holdout_outcome_rows_read"] == 0, "report outcome rows read")
    require(report["authorization"]["explicit_file_list_preflight"] == "NOT_AUTHORIZED_INCOMPLETE_PRIVATE_G4_ARTIFACT_SET", "preflight authorization")
    verification = {
        "schema_version": VERIFY_SCHEMA,
        "status": PASS_STATUS,
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(output_root / "g4_private_membership_materialization.outputs.json"),
        "artifact_count": len(artifacts),
        "risk_episode_files_verified": private_check["risk_episode_files"],
        "risk_opportunity_rows_verified": private_check["risk_opportunity_rows"],
        "network_leaveout_surge_rows_verified": private_check["network_leaveout_surge_rows"],
        "author_cold_guard_rows_verified": private_check["author_cold_guard_rows"],
        "risk_key_duplicates": 0,
        "guard_projection_missing": 0,
        "guard_projection_extra": 0,
        "parent_count_and_row_algebra_verified": True,
        "author_cold_guard_exact_projection_verified": True,
        "forbidden_outcome_and_work_columns_absent": True,
        "g4_private_predecision_membership_complete": True,
        "g4_author_cold_guard_complete": True,
        "earlier_partition_guard_application_complete": False,
        "complete_private_g4_artifact_set": False,
        "holdout_preflight_complete": False,
        "holdout_byte_seal_complete": False,
        "confirmatory_or_holdout_outcome_rows_read": 0,
        "model_score_read": False,
        "model_fitted": False,
        "g4_open": False,
        "pass_f": False,
        "pass_g0": False,
    }
    verification_output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(verification_output, 0o600)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--atomic-root", type=Path, required=True)
    parser.add_argument("--final-panel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--memory-limit-gb", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        atomic_root=args.atomic_root,
        final_panel_root=args.final_panel_root,
        output_root=args.output_root,
        verification_output=args.verification_output,
        threads=args.threads,
        memory_limit_gb=args.memory_limit_gb,
    )
    safe = {
        key: result[key]
        for key in (
            "schema_version",
            "status",
            "output_root_basename",
            "risk_episode_files_verified",
            "risk_opportunity_rows_verified",
            "parent_count_and_row_algebra_verified",
            "author_cold_guard_exact_projection_verified",
            "g4_private_predecision_membership_complete",
            "g4_author_cold_guard_complete",
            "complete_private_g4_artifact_set",
            "holdout_preflight_complete",
            "holdout_byte_seal_complete",
            "confirmatory_or_holdout_outcome_rows_read",
            "pass_f",
            "pass_g0",
            "g4_open",
        )
    }
    print(json.dumps(safe, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
