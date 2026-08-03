#!/usr/bin/env python3
"""Create an operator-local holdout-seal preflight from explicit file paths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASENAME = "pivot_feedback_holdout_access_seal_preflight_v1.json"
PREFLIGHT_SCHEMA = "pivot_feedback_holdout_access_seal_preflight.v1"
PREFLIGHT_STATUS = (
    "PASS_EXPLICIT_ALLOWLIST_HASHES_BOUND_ZERO_PRIOR_OPEN_EVENTS"
)
EXPECTED_PARTITION = {
    "partition_id": "g4_temporal_holdout",
    "entry_start_year": 2020,
    "entry_end_year": 2022,
    "entry_years": [2020, 2021, 2022],
    "outcome_followup_through_year": 2025,
    "temporal_holdout": True,
    "domain_cold": True,
    "author_cold": True,
    "outcome_blind_definition": True,
}
PREFLIGHT_AUTHORIZATION_GATE = {
    "relative_path": "experiments/configs/pivot_feedback_gate_state_v55.yaml",
    "sha256": "0556eeff318789a2ac32203ede23c63ec0502a7feb9fecf1d367eed8fb53cc73",
    "schema_version": "pivot_feedback_gate_state.v55",
    "status": (
        "VALID_HOLDOUT_SEAL_TOOLING_CLOSED_FIXTURE_PRODUCTION_NOT_RUN_"
        "OPERATOR_PREFLIGHT_REQUIRED"
    ),
}
CREATOR_BINDINGS = {
    "authorization_gate": {
        "relative_path": "experiments/configs/pivot_feedback_gate_state_v54.yaml",
        "sha256": (
            "243145fadeee3586a141dcd5e9b0d34af1d87042b4586f3613bc3ea8d30768c8"
        ),
        "schema_version": "pivot_feedback_gate_state.v54",
        "status": (
            "VALID_HOLDOUT_SEAL_CONTRACT_COMPLETE_G0_NOT_READY_"
            "SEAL_CREATION_ONLY_AUTHORIZED"
        ),
    },
    "contract": {
        "relative_path": (
            "experiments/configs/pivot_feedback_holdout_access_seal_v1.yaml"
        ),
        "sha256": (
            "16054264f703b03726edddc041255de10a541b9339d25f9e8ec8cab302d57568"
        ),
        "schema_version": "pivot_feedback_holdout_access_seal_contract.v1",
        "status": "PENDING_INDEPENDENT_VALIDATION_FUTURE_SEAL_CREATION_ONLY",
    },
    "g0_v2": {
        "relative_path": (
            "experiments/configs/pivot_feedback_g0_measurement_freeze_v2.yaml"
        ),
        "sha256": (
            "a2cb0cbf444a7c128eecc40a815b80d5fe86f617a36849330612491b03934eb3"
        ),
        "schema_version": "pivot_feedback_g0_measurement_freeze.v3",
        "status": "PENDING_G0_V2_G_MINUS_1A_BOUND_THREE_BLOCKERS_NO_SEAL",
    },
    "prerequisite_closure_plan": {
        "relative_path": (
            "experiments/configs/pivot_feedback_g0_prerequisite_"
            "closure_plan_v1.yaml"
        ),
        "sha256": (
            "45db08ea9e51d861270c1be4b8ac675e56e30df7b57268f515fd665708b2c4fc"
        ),
        "schema_version": "pivot_feedback_g0_prerequisite_closure_plan.v1",
        "status": (
            "PENDING_INDEPENDENT_VALIDATION_FOUR_WORKSTREAM_PLAN_NO_EXECUTION"
        ),
    },
}


class HoldoutSealPreflightError(RuntimeError):
    """Fail-closed operator preflight authoring error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutSealPreflightError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise HoldoutSealPreflightError(f"invalid {label}: {error}") from error


def _validate_binding(expected: Mapping[str, Any], label: str) -> None:
    path = REPO_ROOT / str(expected["relative_path"])
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    require(_sha256(path) == expected["sha256"], f"{label} hash drift")
    document = _read_yaml(path, label)
    require(
        document.get("schema_version") == expected["schema_version"],
        f"{label} schema drift",
    )
    require(document.get("status") == expected["status"], f"{label} status drift")


def _validate_repository_authorization() -> None:
    _validate_binding(PREFLIGHT_AUTHORIZATION_GATE, "preflight authorization gate")
    for key, binding in CREATOR_BINDINGS.items():
        _validate_binding(binding, key)
    gate = _read_yaml(
        REPO_ROOT / PREFLIGHT_AUTHORIZATION_GATE["relative_path"],
        "preflight authorization gate",
    )
    scope = _mapping(gate.get("authorized_operator_preflight_scope"), "preflight scope")
    require(
        scope.get("mode")
        == "OPERATOR_LOCAL_EXPLICIT_PATH_SIZE_SHA256_ALLOWLIST_NO_ROW_PARSING",
        "preflight scope mode",
    )
    require(scope.get("may_implicitly_discover_artifacts") is False, "discovery boundary")
    require(
        scope.get("may_use_wildcards_or_recursive_directory_inputs") is False,
        "wildcard boundary",
    )
    require(scope.get("may_parse_or_analyze_holdout_rows") is False, "row parsing boundary")


def _canonical_relative_path(value: Any) -> str:
    require(isinstance(value, str) and value, "artifact relative path")
    require("\\" not in value, f"noncanonical artifact path {value!r}")
    require(not any(token in value for token in ("*", "?", "[", "]", "{", "}")), f"wildcard artifact path {value!r}")
    pure = PurePosixPath(value)
    require(not pure.is_absolute(), f"absolute artifact path {value!r}")
    require("." not in pure.parts and ".." not in pure.parts, f"unsafe artifact path {value!r}")
    require(pure.as_posix() == value, f"noncanonical artifact path {value!r}")
    return value


def _safe_source_file(root: Path, relative: str) -> Path:
    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise HoldoutSealPreflightError(f"missing artifact {relative!r}") from error
        require(not stat.S_ISLNK(metadata.st_mode), f"symlink artifact {relative!r}")
        if index < len(parts) - 1:
            require(stat.S_ISDIR(metadata.st_mode), f"non-directory artifact parent {relative!r}")
        else:
            require(stat.S_ISREG(metadata.st_mode), f"nonregular artifact {relative!r}")
    return current


def _hash_unchanged_file(root: Path, relative: str) -> dict[str, Any]:
    path = _safe_source_file(root, relative)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"nonregular artifact {relative!r}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"artifact changed during preflight {relative!r}",
        )
        require(total == before.st_size, f"artifact size drift {relative!r}")
    finally:
        os.close(descriptor)
    return {
        "canonical_relative_path": relative,
        "expected_byte_size": total,
        "expected_sha256": digest.hexdigest(),
    }


def _validate_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise HoldoutSealPreflightError("invalid authorization timestamp") from error
    require(parsed.tzinfo is not None, "authorization timestamp timezone")
    require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        "authorization timestamp must be UTC",
    )
    return value


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    require(path.name == OUTPUT_BASENAME, "preflight output basename")
    require(path.parent.is_dir(), "preflight output parent")
    require(not path.exists() and not path.is_symlink(), "preflight output exists")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o400,
    )
    try:
        encoded = payload.encode("utf-8")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            require(written > 0, "short preflight write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_preflight(
    *,
    source_root: Path,
    artifact_relative_paths: Sequence[str],
    output_path: Path,
    authorized_at_utc: str,
    attest_complete: bool,
    attest_private_source_root: bool,
    attest_no_row_parsing: bool,
    prior_open_event_count: int,
    confirmatory_or_holdout_rows_read: int,
) -> dict[str, Any]:
    _validate_repository_authorization()
    require(attest_complete is True, "complete artifact-set attestation")
    require(attest_private_source_root is True, "private source-root attestation")
    require(attest_no_row_parsing is True, "no-row-parsing attestation")
    require(prior_open_event_count == 0, "prior open events must be zero")
    require(
        confirmatory_or_holdout_rows_read == 0,
        "confirmatory or holdout row reads must be zero",
    )
    authorized_at_utc = _validate_timestamp(authorized_at_utc)
    require(not source_root.is_symlink(), "source root symlink")
    source_root = source_root.resolve(strict=True)
    require(source_root.is_dir(), "invalid source root")
    require(artifact_relative_paths, "empty explicit artifact list")

    canonical_paths = [_canonical_relative_path(value) for value in artifact_relative_paths]
    require(len(canonical_paths) == len(set(canonical_paths)), "duplicate artifact path")
    canonical_paths.sort()
    entries = [_hash_unchanged_file(source_root, relative) for relative in canonical_paths]
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "authorized_at_utc": authorized_at_utc,
        "preflight_authorization_gate_binding": {
            "relative_path": PREFLIGHT_AUTHORIZATION_GATE["relative_path"],
            "sha256": PREFLIGHT_AUTHORIZATION_GATE["sha256"],
        },
        **{
            f"{key}_binding": {
                "relative_path": value["relative_path"],
                "sha256": value["sha256"],
            }
            for key, value in CREATOR_BINDINGS.items()
        },
        "partition_contract": EXPECTED_PARTITION,
        "source_root_basename": source_root.name,
        "source_root_resolved_path_sha256": hashlib.sha256(
            source_root.as_posix().encode("utf-8")
        ).hexdigest(),
        "artifact_count": len(entries),
        "artifact_bytes_total": sum(item["expected_byte_size"] for item in entries),
        "artifact_entries": entries,
        "attestations": {
            "explicit_artifact_set_complete": True,
            "paths_enumerated_under_v54_authorization": True,
            "hashes_computed_under_v54_authorization": True,
            "source_root_private_and_access_controlled": True,
            "final_operator_path_clear": True,
            "receipt_path_clear": True,
            "row_parsing_or_analysis_performed": False,
            "prior_open_event_count": 0,
            "confirmatory_or_holdout_rows_read": 0,
        },
    }
    output_path = output_path.resolve()
    _write_json_create_only(output_path, payload)
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorized-at-utc", default=None)
    parser.add_argument("--attest-complete", action="store_true")
    parser.add_argument("--attest-private-source-root", action="store_true")
    parser.add_argument("--attest-no-row-parsing", action="store_true")
    parser.add_argument("--prior-open-event-count", type=int, required=True)
    parser.add_argument(
        "--confirmatory-or-holdout-rows-read", type=int, required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare_preflight(
        source_root=args.source_root,
        artifact_relative_paths=args.artifact,
        output_path=args.output,
        authorized_at_utc=args.authorized_at_utc or _utc_now(),
        attest_complete=args.attest_complete,
        attest_private_source_root=args.attest_private_source_root,
        attest_no_row_parsing=args.attest_no_row_parsing,
        prior_open_event_count=args.prior_open_event_count,
        confirmatory_or_holdout_rows_read=args.confirmatory_or_holdout_rows_read,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
