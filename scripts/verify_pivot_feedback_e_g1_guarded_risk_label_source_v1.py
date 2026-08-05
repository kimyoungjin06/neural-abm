#!/usr/bin/env python3
"""Independently replay the guard-first G1 risk/label materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterable, Mapping

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_e_g1_guarded_risk_label_source_v1.yaml"
)
DEFAULT_OUTPUT = Path("/ssd/pivot-feedback-e-g1-guarded-risk-label-source-v1-20260805")
CONFIG_SHA256 = "c32d355b2c2a94d9891725d5e373b568d06d4de54d9c2ed6d18edef9568bf7b8"
BUILDER = ROOT / "scripts/build_pivot_feedback_e_g1_guarded_risk_label_source_v1.py"
BUILDER_SHA256 = "d46e69933d4a257828ac764b313ceccd848a8f62ed5c9879f0c8c138fc9b8aae"
GUARD = Path(
    "/ssd/pivot-feedback-f-g4-private-membership-v1-20260804/private/"
    "g4_author_cold_guard.parquet"
)
GUARD_SHA256 = "90f79d46dd16e83088a258aeafd795541d918e4afb1915feadd5648523b5ca3a"
GUARD_ROWS = 8_174_698
PASS_STATUS = "PASS_G1_GUARD_FIRST_RISK_AND_LABEL_SOURCE_MATERIALIZED_ESTIMATION_NOT_RUN"
VERIFY_STATUS = (
    "PASS_INDEPENDENT_G1_GUARD_FIRST_RISK_AND_LABEL_SOURCE_ROW_REPLAY_"
    "ESTIMATION_NOT_RUN"
)

COHORT_FIELDS = [
    ("analysis_partition_id", pa.string(), False),
    ("primary_field_id", pa.uint8(), False),
    ("certified_episode_id", pa.uint16(), False),
    ("topic_id", pa.uint32(), False),
    ("subfield_id", pa.uint16(), False),
    ("shock_year", pa.uint16(), False),
    ("response_cohort_offset", pa.int8(), False),
    ("decision_year", pa.uint16(), False),
    ("author_numeric_id", pa.uint64(), False),
    ("entry_outcome", pa.bool_(), False),
    ("persistence_outcome", pa.bool_(), True),
    ("one_off_entry", pa.bool_(), False),
    ("alternative_pivot_entry", pa.bool_(), False),
    ("entry_target_work_count", pa.uint32(), False),
    ("persistence_target_work_count", pa.uint32(), False),
    ("trailing_target_subfield_work_count", pa.uint32(), False),
]
COHORT_SCHEMA = pa.schema(
    [pa.field(name, data_type, nullable=nullable) for name, data_type, nullable in COHORT_FIELDS]
)
MANIFEST_FIELDS = [
    ("certified_episode_id", pa.uint16(), False),
    ("primary_field_id", pa.uint8(), False),
    ("topic_id", pa.uint32(), False),
    ("subfield_id", pa.uint16(), False),
    ("shock_year", pa.uint16(), False),
    ("response_cohort_offset", pa.int8(), False),
    ("decision_year", pa.uint16(), False),
    ("pre_guard_risk_rows", pa.uint64(), False),
    ("guard_excluded_rows", pa.uint64(), False),
    ("guarded_risk_rows", pa.uint64(), False),
    ("entry_rows", pa.uint64(), False),
    ("non_entry_rows", pa.uint64(), False),
    ("persistence_rows", pa.uint64(), False),
    ("one_off_entry_rows", pa.uint64(), False),
    ("alternative_pivot_entry_rows", pa.uint64(), False),
    ("guard_overlap_rows", pa.uint64(), False),
    ("cohort_file_relative_path", pa.string(), False),
]
MANIFEST_SCHEMA = pa.schema(
    [pa.field(name, data_type, nullable=nullable) for name, data_type, nullable in MANIFEST_FIELDS]
)


class GuardedRiskLabelVerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardedRiskLabelVerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _schema_without_metadata(schema: pa.Schema) -> pa.Schema:
    return schema.remove_metadata()


def _schema_version(schema: pa.Schema) -> str:
    metadata = schema.metadata or {}
    return metadata.get(b"schema_version", b"").decode()


def _safe_artifact_path(output: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    require(relative and not pure.is_absolute() and ".." not in pure.parts, "unsafe artifact path")
    path = output.joinpath(*pure.parts)
    require(path.is_file() and not path.is_symlink(), f"invalid artifact path: {relative}")
    require(path.resolve().is_relative_to(output.resolve()), f"artifact escapes output: {relative}")
    return path


def _sum_true(array: pa.Array) -> int:
    value = pc.sum(pc.cast(pc.fill_null(array, False), pa.int64())).as_py()
    return int(value or 0)


def _mismatch_count(left: pa.Array, right: pa.Array) -> int:
    return _sum_true(pc.not_equal(left, right))


def _constant_mismatch(array: pa.Array, value: Any) -> int:
    scalar = pa.scalar(value, type=array.type)
    return _sum_true(pc.not_equal(array, scalar))


def _artifact_rows(path: Path) -> int | None:
    if path.suffix != ".parquet":
        return None
    return int(pq.ParquetFile(path).metadata.num_rows)


def _verify_repo_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    pure = PurePosixPath(str(binding["relative_path"]))
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe repository binding")
    path = ROOT.joinpath(*pure.parts)
    require(path.is_file() and not path.is_symlink(), "invalid repository binding")
    require(_sha256(path) == binding["sha256"], f"repository hash drift: {path.name}")
    payload = _yaml(path) if path.suffix == ".yaml" else _json(path)
    require(payload.get("schema_version") == binding["required_schema_version"], f"repository schema drift: {path.name}")
    if "required_status" in binding:
        require(payload.get("status") == binding["required_status"], f"repository status drift: {path.name}")
    return payload


def _verify_artifact_tree(
    output: Path, receipt_path: Path, receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    artifacts = receipt["artifacts"]
    require(receipt["artifact_count"] == len(artifacts) == 86, "receipt artifact count drift")
    require(
        [row["relative_path"] for row in artifacts]
        == sorted(row["relative_path"] for row in artifacts),
        "artifact order drift",
    )
    require(len({row["relative_path"] for row in artifacts}) == len(artifacts), "duplicate artifact path")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != receipt_path
    }
    require(actual == {row["relative_path"] for row in artifacts}, "artifact tree membership drift")
    replay = []
    newest_artifact_mtime_ns = 0
    for row in artifacts:
        path = _safe_artifact_path(output, str(row["relative_path"]))
        require(_mode(path) == 0o600, f"artifact mode drift: {row['relative_path']}")
        require(path.stat().st_size == row["bytes"], f"artifact bytes drift: {row['relative_path']}")
        require(_sha256(path) == row["sha256"], f"artifact hash drift: {row['relative_path']}")
        observed_rows = _artifact_rows(path)
        if observed_rows is not None:
            require(observed_rows == row["rows"], f"artifact row drift: {row['relative_path']}")
        newest_artifact_mtime_ns = max(newest_artifact_mtime_ns, path.stat().st_mtime_ns)
        replay.append(dict(row))
    expected_tree = hashlib.sha256(
        json.dumps(replay, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(expected_tree == receipt["artifact_tree_sha256"], "artifact tree hash drift")
    require(receipt_path.stat().st_mtime_ns >= newest_artifact_mtime_ns, "receipt was not written last")
    return replay


def _verify_cohort_file(path: Path, manifest: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, int]:
    parquet = pq.ParquetFile(path)
    require(_schema_without_metadata(parquet.schema_arrow) == COHORT_SCHEMA, f"cohort schema drift: {path.name}")
    require(
        _schema_version(parquet.schema_arrow) == config["output_contract"]["cohort_schema_version"],
        f"cohort schema version drift: {path.name}",
    )
    expected_rows = int(manifest["guarded_risk_rows"])
    require(parquet.metadata.num_rows == expected_rows, f"cohort rows drift: {path.name}")
    totals = {
        "rows": 0,
        "entry": 0,
        "non_entry": 0,
        "persistence": 0,
        "one_off": 0,
        "alternative": 0,
        "label_mismatch": 0,
        "constant_mismatch": 0,
        "author_order_or_duplicate": 0,
    }
    previous_author: int | None = None
    for batch in parquet.iter_batches(batch_size=500_000):
        totals["rows"] += batch.num_rows
        columns = {name: batch.column(index) for index, name in enumerate(batch.schema.names)}
        ids = columns["author_numeric_id"].to_numpy(zero_copy_only=False)
        if len(ids):
            if previous_author is not None and int(ids[0]) <= previous_author:
                totals["author_order_or_duplicate"] += 1
            if len(ids) > 1:
                totals["author_order_or_duplicate"] += int(np.count_nonzero(ids[1:] <= ids[:-1]))
            previous_author = int(ids[-1])

        entry = columns["entry_outcome"]
        persistence = columns["persistence_outcome"]
        one_off = columns["one_off_entry"]
        alternative = columns["alternative_pivot_entry"]
        entry_count = columns["entry_target_work_count"]
        persistence_count = columns["persistence_target_work_count"]
        subfield_count = columns["trailing_target_subfield_work_count"]
        expected_entry = pc.greater(entry_count, pa.scalar(0, pa.uint32()))
        expected_persistence = pc.greater(persistence_count, pa.scalar(0, pa.uint32()))
        expected_one_off = pc.and_(entry, pc.equal(persistence_count, pa.scalar(0, pa.uint32())))
        expected_alternative = pc.and_(entry, pc.equal(subfield_count, pa.scalar(0, pa.uint32())))
        totals["label_mismatch"] += _mismatch_count(entry, expected_entry)
        totals["label_mismatch"] += _sum_true(pc.and_(pc.invert(entry), pc.is_valid(persistence)))
        persistence_bad_for_entry = pc.and_(
            entry,
            pc.or_(
                pc.is_null(persistence),
                pc.not_equal(pc.fill_null(persistence, False), expected_persistence),
            ),
        )
        totals["label_mismatch"] += _sum_true(persistence_bad_for_entry)
        totals["label_mismatch"] += _mismatch_count(one_off, expected_one_off)
        totals["label_mismatch"] += _mismatch_count(alternative, expected_alternative)

        totals["entry"] += _sum_true(entry)
        totals["persistence"] += _sum_true(persistence)
        totals["one_off"] += _sum_true(one_off)
        totals["alternative"] += _sum_true(alternative)
        totals["constant_mismatch"] += _constant_mismatch(
            columns["analysis_partition_id"], "g1_empirical_replication"
        )
        for column, key in (
            ("primary_field_id", "primary_field_id"),
            ("certified_episode_id", "certified_episode_id"),
            ("topic_id", "topic_id"),
            ("subfield_id", "subfield_id"),
            ("shock_year", "shock_year"),
            ("response_cohort_offset", "response_cohort_offset"),
            ("decision_year", "decision_year"),
        ):
            totals["constant_mismatch"] += _constant_mismatch(columns[column], int(manifest[key]))
    totals["non_entry"] = totals["rows"] - totals["entry"]
    require(totals["rows"] == expected_rows, f"cohort scan closure: {path.name}")
    require(totals["author_order_or_duplicate"] == 0, f"author order or duplicate drift: {path.name}")
    require(totals["constant_mismatch"] == 0, f"cohort constant drift: {path.name}")
    require(totals["label_mismatch"] == 0, f"label logic drift: {path.name}")
    for observed, key in (
        (totals["entry"], "entry_rows"),
        (totals["non_entry"], "non_entry_rows"),
        (totals["persistence"], "persistence_rows"),
        (totals["one_off"], "one_off_entry_rows"),
        (totals["alternative"], "alternative_pivot_entry_rows"),
    ):
        require(observed == int(manifest[key]), f"manifest count drift: {path.name}: {key}")
    return totals


def _duckdb_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _guard_overlap(paths: Iterable[Path]) -> int:
    explicit = ",".join(f"'{_duckdb_path(path)}'" for path in paths)
    con = duckdb.connect(":memory:")
    try:
        con.execute("PRAGMA threads=16")
        return int(
            con.execute(
                f"""
                SELECT count(*)
                FROM read_parquet([{explicit}]) c
                JOIN read_parquet('{_duckdb_path(GUARD)}') g USING (author_numeric_id)
                WHERE g.exclude_from_g1
                """
            ).fetchone()[0]
        )
    finally:
        con.close()


def verify(config_path: Path, output: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(config_path == DEFAULT_CONFIG.resolve(), "unexpected config path")
    require(_sha256(config_path) == CONFIG_SHA256, "config hash drift")
    require(_sha256(BUILDER) == BUILDER_SHA256, "builder hash drift")
    config = _yaml(config_path)
    require(config.get("schema_version") == "pivot_feedback_e_g1_guarded_risk_label_source.v1", "config schema drift")
    require(config.get("status_before_execution") == "AUTHORIZED_FROM_INDEPENDENT_GUARD_ONLY_EXCEPTION_PASS", "config status drift")
    authorization = _verify_repo_binding(config["authorization_parent"])
    frozen = _verify_repo_binding(config["frozen_scope_parent"])
    require(authorization["authorization"]["e_g1_execution_resume"] == "AUTHORIZED_EXACT_PARENT_V1_PLUS_V2_SCOPE", "guard exception authorization drift")
    require(frozen["registered_execution_scope"]["selected_field_ids_in_registered_slot_order"] == [16, 32, 11], "frozen scope drift")
    scientific_sections = _verify_repo_binding(
        frozen["bound_frozen_contracts"]["thirteen_scientific_sections"]
    )
    support_policy = _verify_repo_binding(
        scientific_sections["bound_design_sources"]["risk_weight_support_policy"]
    )
    require(
        support_policy.get("status")
        == "FROZEN_SIX_RISK_SET_WEIGHT_SUPPORT_CHOICES_PENDING_INDEPENDENT_VALIDATION_NO_G0_PROMOTION",
        "frozen support policy status drift",
    )
    cohort_policy = support_policy["precommitment_choices"]["minimum_cohort_size"]
    require(
        cohort_policy["rule_id"]
        == "FIELD_EPISODE_PARTITION_MINIMUM_500_WITH_100_PER_OUTCOME_CLASS_V1",
        "frozen cohort policy ID drift",
    )
    require(cohort_policy["failed_unit_action"] == "EXCLUDE_UNIT_AND_FAIL_REGISTERED_COVERAGE_REQUIREMENT", "frozen cohort failure action drift")
    require(cohort_policy["pooling_across_fields_to_rescue_failure_allowed"] is False, "field pooling drift")
    require(cohort_policy["pooling_across_partitions_to_rescue_failure_allowed"] is False, "partition pooling drift")
    require(cohort_policy["pooling_across_certified_episodes_to_rescue_failure_allowed"] is False, "episode pooling drift")
    frozen_floor = frozen["label_materialization_contract"]["label_dependent_support"]["cohort_floor"]
    require(
        frozen_floor == {
            "unit": "ANALYSIS_PARTITION_X_FIELD_X_EPISODE",
            "evaluated_before_negative_sampling": True,
            "minimum_total": 500,
            "minimum_entry_events": 100,
            "minimum_non_entry_opportunities": 100,
            "pooling_to_rescue_failure": False,
        },
        "frozen cohort support floor drift",
    )
    require(
        [
            cohort_policy["minimum_total_opportunity_rows"],
            cohort_policy["minimum_entry_events"],
            cohort_policy["minimum_non_entry_opportunities"],
        ]
        == [500, 100, 100],
        "direct frozen cohort thresholds drift",
    )
    require(config["registered_scope"]["response_cohort_count"] == 84, "registered cohort count drift")
    require(config["label_contract"]["guard_anti_join_before_label_logic"] is True, "guard-first contract drift")
    require(config["label_contract"]["outcome_imputation"] == "NONE", "outcome imputation drift")
    require(config["support_contract"]["pooling_to_rescue_failure"] is False, "support pooling drift")

    output = output.resolve(strict=True)
    require(output == Path(config["execution"]["output_root"]).resolve(), "output root drift")
    require(output.is_dir() and not output.is_symlink(), "invalid output root")
    require(_mode(output) == 0o700, "output root mode drift")
    require(all(_mode(path) == 0o700 for path in output.rglob("*") if path.is_dir()), "output directory mode drift")
    require(GUARD.is_file() and not GUARD.is_symlink(), "invalid guard source")
    require(_sha256(GUARD) == GUARD_SHA256 == config["exact_inputs"]["author_cold_guard"]["sha256"], "guard hash drift")
    require(pq.ParquetFile(GUARD).metadata.num_rows == GUARD_ROWS, "guard rows drift")

    receipt_path = output / config["output_contract"]["receipt_basename"]
    report_path = output / config["output_contract"]["report_basename"]
    manifest_path = output / config["output_contract"]["manifest_basename"]
    require(receipt_path.is_file() and report_path.is_file() and manifest_path.is_file(), "required output missing")
    require(_mode(receipt_path) == 0o600, "receipt mode drift")
    receipt = _json(receipt_path)
    report = _json(report_path)
    require(receipt.get("schema_version") == config["output_contract"]["receipt_schema_version"], "receipt schema drift")
    require(report.get("schema_version") == config["output_contract"]["report_schema_version"], "report schema drift")
    require(receipt.get("status") == report.get("status") == PASS_STATUS, "production status drift")
    artifacts = _verify_artifact_tree(output, receipt_path, receipt)

    manifest_file = pq.ParquetFile(manifest_path)
    require(_schema_without_metadata(manifest_file.schema_arrow) == MANIFEST_SCHEMA, "manifest schema drift")
    require(_schema_version(manifest_file.schema_arrow) == config["output_contract"]["manifest_schema_version"], "manifest schema version drift")
    require(manifest_file.metadata.num_rows == 84, "manifest rows drift")
    manifest_rows = pq.read_table(manifest_path).to_pylist()
    keys = [(int(row["certified_episode_id"]), int(row["response_cohort_offset"])) for row in manifest_rows]
    require(len(set(keys)) == len(keys) == 84, "manifest key closure")
    require({key[1] for key in keys} == {0, 1, 2}, "manifest offset drift")
    require(len({key[0] for key in keys}) == 28, "manifest episode count drift")
    require({int(row["primary_field_id"]) for row in manifest_rows} == {16, 32, 11}, "manifest field drift")

    totals = {
        "pre_guard_risk_rows": 0,
        "guard_excluded_rows": 0,
        "guarded_risk_rows": 0,
        "entry_rows": 0,
        "non_entry_rows": 0,
        "persistence_rows": 0,
        "one_off_entry_rows": 0,
        "alternative_pivot_entry_rows": 0,
    }
    cohort_paths = []
    episode_support: dict[int, dict[str, int]] = {}
    episode_identity: dict[int, dict[str, int]] = {}
    for row in manifest_rows:
        require(int(row["pre_guard_risk_rows"]) == int(row["guard_excluded_rows"]) + int(row["guarded_risk_rows"]), "pre/post guard row closure")
        require(int(row["guard_overlap_rows"]) == 0, "manifest guard overlap")
        relative = str(row["cohort_file_relative_path"])
        expected_name = (
            f"private/g1_guarded_risk_labels/episode-{int(row['certified_episode_id']):04d}-"
            f"k{int(row['response_cohort_offset']):+d}.parquet"
        )
        require(relative == expected_name, "cohort path drift")
        path = _safe_artifact_path(output, relative)
        cohort_paths.append(path)
        replay = _verify_cohort_file(path, row, config)
        for target, key in (
            ("pre_guard_risk_rows", "pre_guard_risk_rows"),
            ("guard_excluded_rows", "guard_excluded_rows"),
            ("guarded_risk_rows", "guarded_risk_rows"),
            ("entry_rows", "entry_rows"),
            ("non_entry_rows", "non_entry_rows"),
            ("persistence_rows", "persistence_rows"),
            ("one_off_entry_rows", "one_off_entry_rows"),
            ("alternative_pivot_entry_rows", "alternative_pivot_entry_rows"),
        ):
            totals[target] += int(row[key])
        episode_id = int(row["certified_episode_id"])
        identity = {
            "primary_field_id": int(row["primary_field_id"]),
            "topic_id": int(row["topic_id"]),
            "shock_year": int(row["shock_year"]),
        }
        if episode_id in episode_identity:
            require(episode_identity[episode_id] == identity, "episode identity drift across cohorts")
        else:
            episode_identity[episode_id] = identity
        support = episode_support.setdefault(episode_id, {"total": 0, "entries": 0, "nonentries": 0})
        support["total"] += replay["rows"]
        support["entries"] += replay["entry"]
        support["nonentries"] += replay["non_entry"]

    require(len(cohort_paths) == 84, "cohort file count drift")
    overlap = _guard_overlap(cohort_paths)
    require(overlap == 0, "guard overlap found in materialized cohorts")
    support_rows = []
    floors = config["support_contract"]
    for episode_id, values in sorted(episode_support.items()):
        passed = (
            values["total"] >= int(floors["minimum_total"])
            and values["entries"] >= int(floors["minimum_entry_events"])
            and values["nonentries"] >= int(floors["minimum_non_entry_opportunities"])
        )
        support_rows.append({"episode_id": episode_id, **values, "support_pass": passed})
    require(report["episode_support"] == support_rows, "episode support replay drift")

    require(
        totals["non_entry_rows"]
        == totals["guarded_risk_rows"] - totals["entry_rows"],
        "independent non-entry row closure",
    )
    for key, value in totals.items():
        if key == "non_entry_rows":
            continue
        require(int(report["accounting"][key]) == value, f"report accounting drift: {key}")
    require(int(report["accounting"]["response_cohort_count"]) == 84, "report cohort count drift")
    pass_count = sum(bool(row["support_pass"]) for row in support_rows)
    fail_count = len(support_rows) - pass_count
    failed_units = [
        {
            "episode_id": int(row["episode_id"]),
            **episode_identity[int(row["episode_id"])],
            "total": int(row["total"]),
            "entries": int(row["entries"]),
            "nonentries": int(row["nonentries"]),
            "entry_shortfall_from_frozen_floor": max(
                0, int(frozen_floor["minimum_entry_events"]) - int(row["entries"])
            ),
            "failure_reason": "ENTRY_EVENTS_BELOW_100",
        }
        for row in support_rows
        if not bool(row["support_pass"])
    ]
    require(report["accounting"]["episode_support_pass_count"] == pass_count, "support pass count drift")
    require(report["accounting"]["episode_support_fail_count"] == fail_count, "support fail count drift")
    require(receipt["confirmatory_rows_read"] == totals["guarded_risk_rows"], "receipt confirmatory rows drift")
    require(receipt["holdout_outcome_rows_read"] == 0, "holdout outcome read drift")
    require(receipt["guard_rows_read"] == GUARD_ROWS, "guard read count drift")
    require(receipt["other_private_g4_files_read"] == 0, "other private G4 access reported")
    require(receipt["seal_open_events"] == 0 and receipt["g4_open"] is False, "sealed G4 boundary drift")
    require(receipt["estimators_run"] == 0, "estimator execution drift")
    require(receipt["receipt_written_last"] is True, "receipt-last flag drift")
    require(report["guard"] == {
        "anti_join_before_label_logic": True,
        "guard_overlap_rows": 0,
        "other_private_g4_artifacts_accessed": False,
        "rows": GUARD_ROWS,
        "sealed_root_accessed": False,
        "sha256": GUARD_SHA256,
    }, "report guard boundary drift")
    require(report["access_accounting"]["holdout_outcome_rows_read"] == 0, "report holdout read drift")
    require(report["access_accounting"]["other_private_g4_files_read"] == 0, "report other G4 read drift")
    require(report["access_accounting"]["seal_open_events"] == 0, "report seal open drift")
    require(report["access_accounting"]["g4_open"] is False, "report G4 open drift")
    require(report["access_accounting"]["estimators_run"] == 0, "report estimator drift")
    require(report["access_accounting"]["model_selection_scores_read"] == 0, "model score read drift")

    return {
        "schema_version": "pivot_feedback_e_g1_guarded_risk_label_source_summary.v1",
        "status": VERIFY_STATUS,
        "repository_artifacts": {
            "config": {"relative_path": config_path.relative_to(ROOT).as_posix(), "sha256": _sha256(config_path)},
            "builder": {"relative_path": BUILDER.relative_to(ROOT).as_posix(), "sha256": _sha256(BUILDER)},
            "independent_verifier": {"relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": _sha256(Path(__file__).resolve())},
        },
        "production_artifact": {
            "root": str(output),
            "receipt_sha256": _sha256(receipt_path),
            "report_sha256": _sha256(report_path),
            "manifest_sha256": _sha256(manifest_path),
            "artifact_count_excluding_receipt": len(artifacts),
            "cohort_file_count": len(cohort_paths),
        },
        "independent_row_replay": totals,
        "guard_replay": {
            "guard_sha256": GUARD_SHA256,
            "guard_rows": GUARD_ROWS,
            "materialized_guard_overlap_rows": overlap,
            "anti_join_before_label_logic_contract_bound": True,
            "guard_membership_indicator_emitted": False,
        },
        "label_replay": {
            "cohort_files_replayed": len(cohort_paths),
            "label_logic_mismatches": 0,
            "cohort_constant_mismatches": 0,
            "author_order_or_duplicate_rows": 0,
            "outcome_imputation": "NONE",
        },
        "support_replay": {
            "episode_count": len(support_rows),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pooling_to_rescue_failure": False,
            "episodes": support_rows,
        },
        "registered_execution_disposition": {
            "terminal_status": (
                "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION"
                if fail_count
                else "SUPPORT_PASS_FEATURE_CONTROL_SOURCE_STILL_REQUIRED"
            ),
            "blocking_condition": (
                "FROZEN_EPISODE_COHORT_SUPPORT_FLOOR_FAILED"
                if fail_count
                else None
            ),
            "failed_unit_count": fail_count,
            "failed_units": failed_units,
            "pooling_or_threshold_relaxation_authorized": False,
            "frozen_policy_rule_id": cohort_policy["rule_id"],
            "failed_unit_action": cohort_policy["failed_unit_action"],
            "registered_sampling_weighting_and_estimation": (
                "NOT_RUN_BLOCKED_BY_SUPPORT"
                if fail_count
                else "NOT_AUTHORIZED_UNTIL_COMPLETE_FEATURE_CONTROL_SOURCE"
            ),
            "predecision_feature_and_control_materialization": (
                "NOT_RUN_UNNECESSARY_AFTER_TERMINAL_SUPPORT_FAILURE"
                if fail_count
                else "AUTHORIZED_NEXT_EXACT_G1_ROWS"
            ),
            "pass_e": False,
            "pass_g1": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "access_boundary": {
            "confirmatory_rows_materialized": totals["guarded_risk_rows"],
            "holdout_outcome_rows_read": 0,
            "other_private_g4_files_read": 0,
            "seal_open_events": 0,
            "g4_open": False,
            "estimators_run": 0,
            "model_selection_scores_read": 0,
        },
        "authorization": config["authorization_after_pass"],
        "claim_boundary": config["claim_boundary"],
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise GuardedRiskLabelVerificationError(f"output exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    summary = verify(args.config, args.output_root)
    if args.summary_output is not None:
        _write_create_only(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
