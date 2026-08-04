#!/usr/bin/env python3
"""Independently verify the strict-incumbent aligned all-topic F rescreen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.yaml"
CONFIG_SCHEMA = "pivot_feedback_f_aligned_strict_incumbent_rescreen.v1"
VERIFICATION_SCHEMA = "pivot_feedback_f_aligned_strict_incumbent_rescreen_verification.v1"
STATUS = "PASS_INDEPENDENT_STRICT_INCUMBENT_ALIGNED_ALL_TOPIC_RESCREEN_NO_F_PASS"

EXPECTED_FILES = {
    "ALIGNED_STRICT_INCUMBENT_RESCREEN.md",
    "aligned_all_topic_surface.parquet",
    "aligned_field_attempts.parquet",
    "aligned_slot_assignment.parquet",
    "aligned_strict_candidates.parquet",
    "aligned_strict_incumbent_rescreen.json",
    "aligned_strict_incumbent_rescreen.outputs.json",
    "predecessor_episode_count_closure.parquet",
}

SURFACE_FIELDS = [
    ("slot_id", "string", False),
    ("temporal_partition_id", "string", False),
    ("field_attempt_order", "uint8", False),
    ("field_id", "uint8", False),
    ("topic_id", "uint32", False),
    ("topic_name", "string", False),
    ("subfield_id", "uint16", False),
    ("subfield_name", "string", False),
    ("decision_year", "uint16", False),
    ("reference_count", "uint64", False),
    ("recent_count", "uint64", False),
    ("acceleration", "double", False),
    ("field_year_median", "double", False),
    ("field_year_mad", "double", False),
    ("modified_z", "double", True),
    ("strict_attention_candidate", "bool", False),
]


class RescreenVerificationError(RuntimeError):
    """Fail-closed independent verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RescreenVerificationError(message)


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


def _schema_version(path: Path) -> str:
    return (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"schema_version", b"").decode()


def _assert_schema(path: Path, fields: Sequence[tuple[str, str, bool]], version: str) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    observed = [(field.name, str(field.type), field.nullable) for field in schema]
    require(observed == list(fields), f"schema fields: {path.name}")
    require((schema.metadata or {}).get(b"schema_version", b"").decode() == version, f"schema version: {path.name}")


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["relative_path"]): dict(row) for row in receipt.get("artifacts", [])}


def _verify_receipt(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    contract = _mapping(config["output_contract"], "output contract")
    receipt_path = root / "aligned_strict_incumbent_rescreen.outputs.json"
    receipt = _json(receipt_path)
    require(receipt.get("schema_version") == contract["receipt_schema_version"], "receipt schema")
    require(receipt.get("status") in (contract["passing_status"], contract["blocking_status"]), "receipt status")
    require(receipt.get("receipt_written_last") is True, "receipt ordering flag")
    artifacts = _artifact_map(receipt)
    require(set(artifacts) == EXPECTED_FILES - {receipt_path.name}, "receipt artifact set")
    require(receipt.get("artifact_count") == len(artifacts), "receipt artifact count")
    tree_rows = []
    for relative in sorted(artifacts):
        row = artifacts[relative]
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"artifact path: {relative}")
        require(path.stat().st_size == row["bytes"], f"artifact bytes: {relative}")
        require(_sha256(path) == row["sha256"], f"artifact SHA: {relative}")
        if path.suffix == ".parquet":
            require(pq.ParquetFile(path).metadata.num_rows == row["rows"], f"artifact rows: {relative}")
        tree_rows.append((relative, row["bytes"], row["sha256"]))
    require(hashlib.sha256(_canonical_bytes(tree_rows)).hexdigest() == receipt["artifact_tree_sha256"], "artifact tree SHA")
    return receipt, artifacts


def _verify_modes_and_files(root: Path, verification_output: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), "output root")
    require((root.stat().st_mode & 0o777) == 0o700, "output root mode")
    observed = {path.name for path in root.iterdir()}
    require(observed == EXPECTED_FILES, "unexpected output files")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"non-file output: {path.name}")
        require((path.stat().st_mode & 0o777) == 0o600, f"output file mode: {path.name}")
    require(not verification_output.exists(), "verification output already exists")


def _expected_groups(config: Mapping[str, Any]) -> set[tuple[int, int]]:
    attempts = _mapping(config["frozen_field_attempts"], "frozen attempts")
    periods = _mapping(attempts["periods"], "periods")
    result = set()
    for slot in attempts["expected"]:
        period = _mapping(periods[slot["temporal_partition_id"]], "period")
        for field_id in slot["field_order"]:
            for year in range(period["decision_year_start"], period["decision_year_end"] + 1):
                result.add((int(field_id), int(year)))
    return result


def _verify_surface(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    contract = _mapping(config["output_contract"], "output contract")
    surface_path = root / "aligned_all_topic_surface.parquet"
    candidate_path = root / "aligned_strict_candidates.parquet"
    _assert_schema(surface_path, SURFACE_FIELDS, contract["surface_schema_version"])
    candidate_fields = [*SURFACE_FIELDS, ("aligned_selected_field", "bool", False), ("retained_for_aligned_panel", "bool", False)]
    _assert_schema(candidate_path, candidate_fields, contract["candidate_schema_version"])
    surface = pq.read_table(surface_path).to_pylist()
    candidates = pq.read_table(candidate_path).to_pylist()
    require(len(surface) > 0 and len(candidates) > 0, "empty rescreen tables")
    keys = [(int(row["field_id"]), int(row["topic_id"]), int(row["decision_year"])) for row in surface]
    require(len(keys) == len(set(keys)), "duplicate surface key")
    primary = _mapping(config["primary_rescreen"], "primary rescreen")
    multiplier = float(primary["modified_z_multiplier"])
    threshold = float(primary["modified_z_threshold_inclusive"])
    by_group: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in surface:
        require(int(row["reference_count"]) > 0, "nonpositive reference in normalization surface")
        reference = int(row["reference_count"])
        recent = int(row["recent_count"])
        expected_acceleration = math.log1p(recent / 3.0) - math.log1p(reference / 3.0)
        require(math.isclose(float(row["acceleration"]), expected_acceleration, rel_tol=1e-12, abs_tol=1e-12), "acceleration mismatch")
        by_group.setdefault((int(row["field_id"]), int(row["decision_year"])), []).append(row)
    require(set(by_group) == _expected_groups(config), "field-year normalization group closure")
    bad_mad = 0
    candidate_keys = set()
    for group_rows in by_group.values():
        values = [float(row["acceleration"]) for row in group_rows]
        median = float(statistics.median(values))
        mad = float(statistics.median(abs(value - median) for value in values))
        if not math.isfinite(mad) or mad <= 0:
            bad_mad += 1
        for row in group_rows:
            require(math.isclose(float(row["field_year_median"]), median, rel_tol=1e-12, abs_tol=1e-12), "field-year median mismatch")
            require(math.isclose(float(row["field_year_mad"]), mad, rel_tol=1e-12, abs_tol=1e-12), "field-year MAD mismatch")
            expected_z = None if not math.isfinite(mad) or mad <= 0 else multiplier * (float(row["acceleration"]) - median) / mad
            if expected_z is None:
                require(row["modified_z"] is None, "non-null z for invalid MAD")
                expected_candidate = False
            else:
                require(row["modified_z"] is not None and math.isclose(float(row["modified_z"]), expected_z, rel_tol=1e-12, abs_tol=1e-12), "modified z mismatch")
                expected_candidate = int(row["recent_count"]) > int(row["reference_count"]) and float(row["acceleration"]) > 0 and expected_z >= threshold
            require(bool(row["strict_attention_candidate"]) == expected_candidate, "candidate flag mismatch")
            if expected_candidate:
                candidate_keys.add((int(row["field_id"]), int(row["topic_id"]), int(row["decision_year"])))
    observed_candidate_keys = {(int(row["field_id"]), int(row["topic_id"]), int(row["decision_year"])) for row in candidates}
    require(len(observed_candidate_keys) == len(candidates), "duplicate candidate key")
    require(observed_candidate_keys == candidate_keys, "candidate subset mismatch")
    surface_by_key = {key: row for key, row in zip(keys, surface, strict=True)}
    for row in candidates:
        key = (int(row["field_id"]), int(row["topic_id"]), int(row["decision_year"]))
        expected = surface_by_key[key]
        for name, _, _ in SURFACE_FIELDS:
            left, right = row[name], expected[name]
            if isinstance(left, float):
                require(math.isclose(left, float(right), rel_tol=1e-12, abs_tol=1e-12), f"candidate surface value: {name}")
            else:
                require(left == right, f"candidate surface value: {name}")
        require(row["aligned_selected_field"] == row["retained_for_aligned_panel"], "candidate retention divergence")
    return surface, candidates, {"normalization_group_count": len(by_group), "bad_mad_count": bad_mad}


def _verify_taxonomy(surface_path: Path, taxonomy_path: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    taxonomy_path = taxonomy_path.resolve(strict=True)
    require(taxonomy_path.name == binding["expected_basename"], "taxonomy basename")
    require(_sha256(taxonomy_path) == binding["sha256"], "taxonomy SHA")
    require(pq.ParquetFile(taxonomy_path).metadata.num_rows == binding["rows"], "taxonomy rows")
    con = duckdb.connect()
    result = con.execute(
        f"""
        WITH taxonomy AS (
          SELECT DISTINCT cast(replace(primary_topic_id,'T','') AS UINTEGER) AS topic_id,
                 primary_topic_name AS topic_name,
                 cast(replace(field_id,'fields/','') AS UTINYINT) AS field_id,
                 cast(replace(subfield_id,'subfields/','') AS USMALLINT) AS subfield_id,
                 subfield_name
          FROM read_parquet('{str(taxonomy_path).replace("'", "''")}') WHERE has_primary_topic
        ), observed AS (
          SELECT DISTINCT field_id,topic_id,topic_name,subfield_id,subfield_name
          FROM read_parquet('{str(surface_path).replace("'", "''")}')
        )
        SELECT (SELECT count(*) FROM taxonomy) AS taxonomy_topics,
               count(*) AS observed_topics,
               count(*) FILTER (WHERE t.topic_id IS NOT NULL) AS matched_topics,
               count(*) FILTER (WHERE t.topic_id IS NOT NULL AND t.field_id=o.field_id
                 AND t.topic_name=o.topic_name AND t.subfield_id=o.subfield_id
                 AND t.subfield_name=o.subfield_name) AS exact_topics
        FROM observed o LEFT JOIN taxonomy t USING(topic_id)
        """
    ).fetchone()
    con.close()
    require(result[0] == binding["distinct_topic_rows"], "taxonomy distinct-topic closure")
    require(result[1] == result[2] == result[3], "surface taxonomy mismatch")
    return {"taxonomy_topics": result[0], "surface_topics": result[1], "exact_surface_topics": result[3]}


def _verify_attempts_and_slots(
    root: Path, config: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    contract = _mapping(config["output_contract"], "output contract")
    attempt_path = root / "aligned_field_attempts.parquet"
    slot_path = root / "aligned_slot_assignment.parquet"
    require(_schema_version(attempt_path) == contract["attempt_schema_version"], "attempt schema version")
    require(_schema_version(slot_path) == contract["slot_schema_version"], "slot schema version")
    attempts = pq.read_table(attempt_path).to_pylist()
    slots = pq.read_table(slot_path).to_pylist()
    expected_slots = list(config["frozen_field_attempts"]["expected"])
    require(len(attempts) == 12 and len(slots) == len(expected_slots) == 4, "attempt or slot rows")
    minimum = int(config["primary_rescreen"]["minimum_distinct_current_snapshot_subfield_families"])
    attempt_by_key = {(str(row["slot_id"]), int(row["field_id"])): row for row in attempts}
    selected: dict[str, int | None] = {}
    for expected_slot in expected_slots:
        slot_id = str(expected_slot["slot_id"])
        first = None
        for order, field_id in enumerate(expected_slot["field_order"], 1):
            rows = [row for row in candidates if row["slot_id"] == slot_id and int(row["field_id"]) == int(field_id)]
            families = len({int(row["subfield_id"]) for row in rows})
            attempt = attempt_by_key[(slot_id, int(field_id))]
            require(int(attempt["field_attempt_order"]) == order, "attempt order")
            require(int(attempt["candidate_episode_count"]) == len(rows), "attempt episode count")
            require(int(attempt["candidate_topic_count"]) == len({int(row["topic_id"]) for row in rows}), "attempt topic count")
            require(int(attempt["subfield_family_count"]) == families, "attempt family count")
            eligible = families >= minimum
            require(bool(attempt["aligned_screen_eligible"]) == eligible, "attempt eligibility")
            if first is None and eligible:
                first = int(field_id)
            require(bool(attempt["selected_for_slot"]) == (int(field_id) == first), "attempt selected flag")
        selected[slot_id] = first
    slot_by_id = {str(row["slot_id"]): row for row in slots}
    for expected_slot in expected_slots:
        slot_id = str(expected_slot["slot_id"])
        slot = slot_by_id[slot_id]
        chosen = selected[slot_id]
        require([int(value) for value in slot["frozen_field_order"]] == [int(value) for value in expected_slot["field_order"]], "slot frozen order")
        require(slot["aligned_selected_field_id"] == chosen, "slot selected field")
        require(bool(slot["slot_pass"]) == (chosen is not None), "slot pass")
        retained_rows = [row for row in candidates if row["slot_id"] == slot_id and int(row["field_id"]) == chosen] if chosen is not None else []
        require(int(slot["aligned_candidate_episode_count"]) == len(retained_rows), "slot candidate count")
        require(int(slot["aligned_subfield_family_count"]) == len({int(row["subfield_id"]) for row in retained_rows}), "slot family count")
    retained = [row for row in candidates if row["retained_for_aligned_panel"]]
    expected_retained = [row for row in candidates if selected[str(row["slot_id"])] == int(row["field_id"])]
    require({(row["field_id"], row["topic_id"], row["decision_year"]) for row in retained} == {(row["field_id"], row["topic_id"], row["decision_year"]) for row in expected_retained}, "retained panel candidate set")
    return attempts, slots, {"selected_fields": selected, "retained_candidate_rows": len(retained)}


def _verify_closure(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(config["predecessor_episode_closure"], "predecessor closure")
    parent = Path(str(binding["path"])).resolve(strict=True)
    require(_sha256(parent) == binding["sha256"], "predecessor SHA")
    path = root / "predecessor_episode_count_closure.parquet"
    require(_schema_version(path) == config["output_contract"]["closure_schema_version"], "closure schema version")
    rows = pq.read_table(path).to_pylist()
    parent_rows = {int(row["episode_id"]): row for row in pq.read_table(parent).to_pylist()}
    require(len(rows) == len(parent_rows) == binding["rows"], "predecessor closure rows")
    exact = 0
    for row in rows:
        parent_row = parent_rows[int(row["predecessor_episode_id"])]
        require((int(row["field_id"]), int(row["topic_id"]), int(row["decision_year"])) == (int(parent_row["field_id"]), int(parent_row["topic_id"]), int(parent_row["decision_year"])), "predecessor closure key")
        require(int(row["expected_reference_count"]) == int(parent_row["all_incumbent_reference_work_count"]), "predecessor expected reference")
        require(int(row["expected_recent_count"]) == int(parent_row["all_incumbent_recent_work_count"]), "predecessor expected recent")
        observed_exact = int(row["observed_reference_count"]) == int(row["expected_reference_count"]) and int(row["observed_recent_count"]) == int(row["expected_recent_count"])
        require(bool(row["exact_match"]) == observed_exact, "predecessor exact flag")
        exact += int(observed_exact)
    require(exact == len(rows), "predecessor strict-count mismatch")
    return {"rows": len(rows), "exact_rows": exact}


def verify(
    *,
    config_path: Path,
    output_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    taxonomy_path: Path,
    verification_output: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    output_root = output_root.resolve(strict=True)
    require(output_root.name == config["output_contract"]["root_basename"], "output root basename")
    verification_output = verification_output.resolve()
    _verify_modes_and_files(output_root, verification_output)
    receipt, artifacts = _verify_receipt(output_root, config)
    require(receipt["confirmatory_entry_or_persistence_labels_read"] == 0, "unsafe receipt state: confirmatory labels")
    for key in ("raw_author_or_work_ids_emitted", "holdout_byte_seal_complete", "model_fitted", "g4_open", "pass_f", "pass_g0"):
        require(receipt[key] is False, f"unsafe receipt state: {key}")
    report = _json(output_root / "aligned_strict_incumbent_rescreen.json")
    require(report.get("schema_version") == config["output_contract"]["report_schema_version"], "report schema")
    require(report.get("status") == receipt.get("status"), "report receipt status")
    execution = _mapping(report["execution"], "execution")
    require(_sha256(config_path) == execution["config_sha256"], "executed config SHA")
    builder = _safe_repo_path(execution["builder_relative_path"])
    require(_sha256(builder) == execution["builder_sha256"], "executed builder SHA")
    for root, binding in (
        (cutoff_root.resolve(strict=True), config["normalized_inputs"]["cutoff_work_metadata"]),
        (author_history_root.resolve(strict=True), config["normalized_inputs"]["author_history_events"]),
    ):
        require(root.name == binding["expected_root_basename"], "normalized input root")
        require(_sha256(root / binding["receipt_basename"]) == binding["receipt_sha256"], "normalized input receipt")
    surface, candidates, surface_check = _verify_surface(output_root, config)
    taxonomy_check = _verify_taxonomy(output_root / "aligned_all_topic_surface.parquet", taxonomy_path, _mapping(config["normalized_inputs"]["taxonomy_recovery"], "taxonomy"))
    attempts, slots, selection_check = _verify_attempts_and_slots(output_root, config, candidates)
    closure_check = _verify_closure(output_root, config)
    accounting = _mapping(report["replay_accounting"], "replay accounting")
    require(accounting["strict_surface_rows"] == len(surface), "report surface rows")
    require(accounting["candidate_rows"] == len(candidates), "report candidate rows")
    require(accounting["normalization_group_count"] == surface_check["normalization_group_count"], "report normalization groups")
    require(accounting["mad_zero_or_nonfinite_group_count"] == surface_check["bad_mad_count"], "report bad MAD groups")
    require(accounting["predecessor_exact_count_rows"] == closure_check["exact_rows"], "report predecessor closure")
    require(accounting["selected_candidate_episode_count"] == selection_check["retained_candidate_rows"], "report retained rows")
    require(report["field_attempts"] == attempts and report["slot_assignment"] == slots, "report table summaries")
    all_slots_pass = all(row["slot_pass"] for row in slots)
    require(receipt["all_slots_pass"] == all_slots_pass, "receipt slot pass")
    require(receipt["final_panel_leaveout_replay_authorized_next"] == all_slots_pass, "receipt next authorization")
    expected_status = config["output_contract"]["passing_status"] if all_slots_pass else config["output_contract"]["blocking_status"]
    require(report["status"] == expected_status, "derived status")
    scope = _mapping(report["scope"], "scope")
    require(scope["confirmatory_entry_or_persistence_labels_read"] == 0, "confirmatory labels read")
    for key in ("realized_citation_effect_read", "raw_author_or_work_ids_emitted", "holdout_byte_seal_complete", "model_fitted", "g4_open", "pass_f", "pass_g0"):
        require(scope[key] is False, f"unsafe scope: {key}")
    if all_slots_pass:
        require(report["authorization"]["aligned_strict_incumbent_final_panel_leaveout_replay"] == "AUTHORIZED_NEXT", "next authorization")
    verification = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": STATUS,
        "output_root_basename": output_root.name,
        "output_receipt_sha256": _sha256(output_root / "aligned_strict_incumbent_rescreen.outputs.json"),
        "artifact_count": len(artifacts),
        "surface_verification": {**surface_check, "surface_rows": len(surface), "candidate_rows": len(candidates)},
        "taxonomy_verification": taxonomy_check,
        "selection_verification": selection_check,
        "predecessor_closure_verification": closure_check,
        "all_slots_pass": all_slots_pass,
        "next_authorization": report["authorization"],
        "confirmatory_entry_or_persistence_labels_read": 0,
        "holdout_byte_seal_complete": False,
        "pass_f": False,
        "pass_g0": False,
        "model_fitted": False,
        "g4_open": False,
    }
    verification_output.parent.mkdir(parents=True, exist_ok=True)
    verification_output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(verification_output, 0o600)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        output_root=args.output_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        taxonomy_path=args.taxonomy,
        verification_output=args.verification_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
