#!/usr/bin/env python3
"""Independently verify the exact operator-only G1 author-cold exception."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORITY = ROOT / "experiments/configs/pivot_feedback_e_g1_guard_only_exception_v1.yaml"
GUARD = Path("/ssd/pivot-feedback-f-g4-private-membership-v1-20260804/private/g4_author_cold_guard.parquet")
GUARD_SHA = "90f79d46dd16e83088a258aeafd795541d918e4afb1915feadd5648523b5ca3a"
GUARD_ROWS = 8_174_698
EXPECTED_REPOSITORY = {
    "authority": (
        ROOT / "experiments/configs/pivot_feedback_current_state_v3.yaml",
        "0e16e844a8a7ce8b3321eff47c3763f45582b4eef8f039dc6e18dc092bf94380",
        "pivot_feedback_current_state.v3",
        "ACTIVE_E_PHASE_BLOCKED_INPUT_AUTHORIZATION_CONFLICT_G1_NOT_RUN_HOLDOUT_SEALED_UNOPENED",
    ),
    "authority_verification": (
        ROOT / "experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.json",
        "10c950d2516b3f1e9ea7e61af4c9018f8e4e27a2eeffb82371387e1b3359ba21",
        "pivot_feedback_current_state_v3_summary.v1",
        "PASS_ACTIVE_E_TERMINAL_BLOCKED_INPUT_AUTHORIZATION_CONFLICT_NO_CONFIRMATORY_READ",
    ),
    "blocked_execution_verification": (
        ROOT / "experiments/evidence/results/pivot_feedback_e_g1_execution_v1_20260805.summary.json",
        "75855d8bc688c1c5001f2c19c25cd6c8a99c62355beb975e6cf1c3df425e6e5c",
        "pivot_feedback_e_g1_execution_summary.v1",
        "PASS_INDEPENDENT_VERIFICATION_OF_TERMINAL_BLOCKED_E_G1_NO_CONFIRMATORY_READ",
    ),
    "guard_verification": (
        ROOT / "experiments/evidence/results/pivot_feedback_f_g4_private_membership_materialization_v1_20260804.summary.json",
        "5f733120ec6366be018df254279acd5c378fd85aeadcdd2e9d2f517433edabb4",
        "pivot_feedback_f_g4_private_membership_materialization_summary.v1",
        "PASS_OPERATOR_ONLY_G4_PREDECISION_MEMBERSHIP_AND_AUTHOR_COLD_GUARD_REMAINING_PRIVATE_PACKAGE_PENDING_NO_F_PASS",
    ),
}
EXPECTED_COLUMNS = [
    "author_numeric_id", "first_g4_decision_year", "last_g4_decision_year",
    "g4_episode_count", "exclude_from_discovery", "exclude_from_g1",
    "exclude_from_model_selection",
]


class GuardOnlyExceptionVerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardOnlyExceptionVerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_repo(name: str) -> dict[str, Any]:
    path, sha, schema, status = EXPECTED_REPOSITORY[name]
    require(_sha256(path) == sha, f"repository hash drift: {name}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix == ".yaml" else json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == schema, f"repository schema drift: {name}")
    require(payload.get("status") == status, f"repository status drift: {name}")
    return payload


def verify(authority_path: Path) -> dict[str, Any]:
    authority_path = authority_path.resolve(strict=True)
    authority = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    require(authority.get("schema_version") == "pivot_feedback_e_g1_guard_only_exception.v1", "authority schema drift")
    require(authority.get("status") == "AUTHORIZED_EXACT_GUARD_ONLY_EXCEPTION_PENDING_INDEPENDENT_VERIFICATION", "authority status drift")
    require(authority.get("authorization_basis") == "EXPLICIT_USER_APPROVAL_GUARD_ONLY_EXCEPTION", "explicit approval missing")
    current = _load_repo("authority")
    current_verification = _load_repo("authority_verification")
    blocked = _load_repo("blocked_execution_verification")
    guard_summary = _load_repo("guard_verification")
    require(current["lifecycle"]["empirical_g1_status"] == "BLOCKED", "parent is not blocked")
    require(current_verification["execution_result_replay"]["confirmatory_rows_read"] == 0, "parent confirmatory rows read")
    require(blocked["blocking_condition_replay"]["block_id"] == authority["blocked_state_parent"]["blocked_execution_verification"]["required_block_id"], "block lineage drift")

    for name, binding_name in (
        ("authority", "authority"),
        ("authority_verification", "authority_verification"),
        ("blocked_execution_verification", "blocked_execution_verification"),
    ):
        path, sha, schema, status = EXPECTED_REPOSITORY[name]
        binding = authority["blocked_state_parent"][binding_name]
        require(binding["relative_path"] == path.relative_to(ROOT).as_posix(), f"binding path drift: {name}")
        require(binding["sha256"] == sha and binding["required_schema_version"] == schema and binding["required_status"] == status, f"binding identity drift: {name}")
        require(binding["mutation_allowed"] is False, f"mutable binding: {name}")
    path, sha, schema, status = EXPECTED_REPOSITORY["guard_verification"]
    guard_binding = authority["guard_provenance"]["verification_summary"]
    require(guard_binding["relative_path"] == path.relative_to(ROOT).as_posix(), "guard summary path drift")
    require(guard_binding["sha256"] == sha and guard_binding["required_schema_version"] == schema and guard_binding["required_status"] == status, "guard summary identity drift")
    require(guard_summary["production_artifact"]["author_cold_guard_sha256"] == GUARD_SHA, "guard receipt SHA drift")
    require(guard_summary["production_artifact"]["author_cold_guard_rows"] == GUARD_ROWS, "guard receipt rows drift")

    exact = authority["guard_provenance"]["exact_source"]
    require(exact["path"] == str(GUARD), "guard path drift")
    require(GUARD.is_file() and not GUARD.is_symlink(), "invalid guard file")
    require(GUARD.stat().st_mode & 0o777 == 0o600, "guard mode drift")
    require(GUARD.stat().st_size == exact["bytes"] == 24_368_575, "guard bytes drift")
    require(_sha256(GUARD) == exact["sha256"] == GUARD_SHA, "guard content hash drift")
    parquet = pq.ParquetFile(GUARD)
    metadata = parquet.schema_arrow.metadata or {}
    require(parquet.metadata.num_rows == exact["rows"] == GUARD_ROWS, "guard rows drift")
    require(parquet.metadata.num_row_groups == exact["row_groups"] == 33, "guard row groups drift")
    require(metadata.get(b"schema_version", b"").decode() == exact["schema_version"] == "pivot_feedback_f_g4_author_cold_guard.v1", "guard schema version drift")
    require(parquet.schema_arrow.names == exact["exact_columns"] == EXPECTED_COLUMNS, "guard columns drift")
    projected_rows = 0
    false_or_null = 0
    for batch in parquet.iter_batches(columns=["author_numeric_id", "exclude_from_g1"], batch_size=1_000_000):
        projected_rows += batch.num_rows
        values = batch.column(1)
        false_or_null += values.null_count + int(pc.sum(pc.invert(values)).as_py() or 0)
    require(projected_rows == GUARD_ROWS, "guard projected row closure")
    require(false_or_null == 0, "guard contains non-exclusion row")

    allowed = authority["allowed_operations"]
    require(allowed == [
        "REHASH_EXACT_SOURCE_FILE",
        "READ_EXACT_PARQUET_FOOTER_AND_SCHEMA",
        "PROJECT_AUTHOR_NUMERIC_ID_AND_EXCLUDE_FROM_G1_ONLY",
        "ASSERT_EVERY_PROJECTED_EXCLUDE_FROM_G1_VALUE_TRUE",
        "ANTI_JOIN_BEFORE_ANY_G1_LABEL_CONTROL_OR_ESTIMATOR_LOGIC",
        "RECORD_PRE_GUARD_POST_GUARD_AND_EXCLUDED_ROW_COUNTS",
        "INDEPENDENTLY_VERIFY_ANTI_JOIN_ZERO_OVERLAP",
    ], "allowed operation drift")
    require(set(authority["forbidden_operations"].values()) == {True}, "forbidden operation relaxed")
    semantics = authority["execution_semantics"]
    require(semantics["guard_role"] == "ROW_EXCLUSION_ONLY_NOT_FEATURE_NOT_OUTCOME_NOT_ESTIMAND", "guard role drift")
    require(semantics["anti_join_order"] == "BEFORE_LABEL_CONTROL_SAMPLING_WEIGHT_OR_ESTIMATOR_LOGIC", "anti-join order drift")
    require(semantics["original_holdout_seal_state_after_use"] == "BYTE_SEALED_UNOPENED", "seal state drift")
    require(semantics["g4_open_after_use"] is False and semantics["registered_scope_unchanged"] is True, "execution boundary drift")

    return {
        "schema_version": "pivot_feedback_e_g1_guard_only_exception_summary.v1",
        "status": "PASS_INDEPENDENT_EXACT_GUARD_ONLY_EXCEPTION_G1_RESUME_AUTHORIZED",
        "authority": {"relative_path": authority_path.relative_to(ROOT).as_posix(), "sha256": _sha256(authority_path)},
        "guard_replay": {
            "path": str(GUARD), "sha256": GUARD_SHA, "bytes": 24_368_575,
            "rows": GUARD_ROWS, "row_groups": 33,
            "schema_version": "pivot_feedback_f_g4_author_cold_guard.v1",
            "projected_columns": ["author_numeric_id", "exclude_from_g1"],
            "non_true_exclude_from_g1_rows": 0,
            "source_is_outside_sealed_private_root": True,
        },
        "resolved_block": {
            "block_id": "G1_AUTHOR_COLD_GUARD_REQUIRED_BUT_PRIVATE_G4_ACCESS_FORBIDDEN",
            "resolved_only_for_exact_guard_anti_join": True,
            "scope_or_scientific_rule_changed": False,
        },
        "access_accounting": {
            "guard_rows_read_for_exclude_flag_validation": GUARD_ROWS,
            "confirmatory_rows_read": 0, "holdout_outcome_rows_read": 0,
            "sealed_private_root_files_read": 0, "other_private_g4_files_read": 0,
            "g4_open": False, "seal_open_events": 0,
        },
        "authorization": authority["authorization_after_independent_pass"],
        "claim_boundary": authority["claim_boundary"],
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise GuardOnlyExceptionVerificationError(f"output exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify(args.authority)
    if args.output is not None:
        _write_create_only(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
