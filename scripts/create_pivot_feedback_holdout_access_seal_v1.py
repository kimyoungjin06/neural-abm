#!/usr/bin/env python3
"""Create a private byte-hash holdout seal from an explicit operator preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = (
    REPO_ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_holdout_access_seal_receipt_v1.summary.json"
)
PREFLIGHT_SCHEMA = "pivot_feedback_holdout_access_seal_preflight.v1"
PREFLIGHT_STATUS = (
    "PASS_EXPLICIT_ALLOWLIST_HASHES_BOUND_ZERO_PRIOR_OPEN_EVENTS"
)
MANIFEST_SCHEMA = "pivot_feedback_holdout_access_seal_manifest.v1"
MANIFEST_STATUS = "SEALED_ARTIFACT_MANIFEST_FINALIZED_UNOPENED"
RECEIPT_SCHEMA = "pivot_feedback_holdout_access_seal_outputs.v1"
RECEIPT_STATUS = "PASS_CONFIRMATORY_PARTITIONS_SEALED_AND_UNOPENED"
FINAL_OPERATOR_RELATIVE_PATH = "pivot-feedback/holdout-seal-v1"
MANIFEST_BASENAME = "manifest.json"
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
REPOSITORY_BINDINGS = {
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


class HoldoutSealCreationError(RuntimeError):
    """Fail-closed holdout seal creation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutSealCreationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HoldoutSealCreationError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise HoldoutSealCreationError(f"invalid {label}: {error}") from error


def _validate_repository_bindings() -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for key, expected in REPOSITORY_BINDINGS.items():
        path = REPO_ROOT / str(expected["relative_path"])
        require(path.is_file() and not path.is_symlink(), f"invalid {key} binding")
        require(_sha256(path) == expected["sha256"], f"{key} hash drift")
        document = _read_yaml(path, key)
        require(
            document.get("schema_version") == expected["schema_version"],
            f"{key} schema drift",
        )
        require(document.get("status") == expected["status"], f"{key} status drift")
        validated[key] = {
            "relative_path": expected["relative_path"],
            "sha256": expected["sha256"],
        }

    gate = _read_yaml(
        REPO_ROOT / REPOSITORY_BINDINGS["authorization_gate"]["relative_path"],
        "authorization gate",
    )
    scope = _mapping(gate.get("authorized_seal_creation_scope"), "creation scope")
    require(scope.get("may_create_or_issue_holdout_seal") is True, "creation not authorized")
    require(scope.get("may_parse_or_analyze_holdout_rows") is False, "row parsing boundary")
    require(scope.get("may_open_holdout_seal") is False, "seal opening boundary")
    require(
        scope.get("explicit_operator_artifact_allowlist_required") is True,
        "allowlist requirement",
    )
    require(scope.get("operator_preflight_required") is True, "preflight requirement")
    return validated


def _canonical_relative_path(value: Any) -> str:
    require(isinstance(value, str) and value, "artifact relative path")
    require("\\" not in value, f"noncanonical artifact path {value!r}")
    pure = PurePosixPath(value)
    require(not pure.is_absolute(), f"absolute artifact path {value!r}")
    require("." not in pure.parts and ".." not in pure.parts, f"unsafe artifact path {value!r}")
    canonical = pure.as_posix()
    require(canonical == value, f"noncanonical artifact path {value!r}")
    return canonical


def _validate_preflight(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(not path.is_symlink(), "operator preflight symlink")
    preflight = _read_json(path.resolve(strict=True), "operator preflight")
    require(preflight.get("schema_version") == PREFLIGHT_SCHEMA, "preflight schema")
    require(preflight.get("status") == PREFLIGHT_STATUS, "preflight status")
    require(preflight.get("partition_contract") == EXPECTED_PARTITION, "preflight partition")
    for key, binding in REPOSITORY_BINDINGS.items():
        declared = _mapping(preflight.get(f"{key}_binding"), f"preflight {key}")
        require(
            declared
            == {
                "relative_path": binding["relative_path"],
                "sha256": binding["sha256"],
            },
            f"preflight {key} binding",
        )

    entries_value = preflight.get("artifact_entries")
    require(isinstance(entries_value, list) and entries_value, "artifact allowlist")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in entries_value:
        entry = _mapping(value, "artifact allowlist entry")
        relative = _canonical_relative_path(entry.get("canonical_relative_path"))
        require(relative not in seen, f"duplicate artifact path {relative!r}")
        seen.add(relative)
        byte_size = entry.get("expected_byte_size")
        digest = entry.get("expected_sha256")
        require(isinstance(byte_size, int) and byte_size >= 0, "expected byte size")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "expected SHA-256",
        )
        require(
            set(entry) == {
                "canonical_relative_path",
                "expected_byte_size",
                "expected_sha256",
            },
            "artifact allowlist entry fields",
        )
        entries.append(
            {
                "canonical_relative_path": relative,
                "expected_byte_size": byte_size,
                "expected_sha256": digest,
            }
        )
    require(
        [item["canonical_relative_path"] for item in entries] == sorted(seen),
        "artifact allowlist must be sorted",
    )
    require(preflight.get("artifact_count") == len(entries), "artifact count")
    require(
        preflight.get("artifact_bytes_total")
        == sum(item["expected_byte_size"] for item in entries),
        "artifact byte total",
    )
    attestations = _mapping(preflight.get("attestations"), "preflight attestations")
    required_true = (
        "explicit_artifact_set_complete",
        "paths_enumerated_under_v54_authorization",
        "hashes_computed_under_v54_authorization",
        "source_root_private_and_access_controlled",
        "final_operator_path_clear",
        "receipt_path_clear",
    )
    for key in required_true:
        require(attestations.get(key) is True, f"preflight attestation {key}")
    require(
        attestations.get("row_parsing_or_analysis_performed") is False,
        "preflight row parsing boundary",
    )
    require(attestations.get("prior_open_event_count") == 0, "prior open events")
    require(
        attestations.get("confirmatory_or_holdout_rows_read") == 0,
        "preflight holdout row reads",
    )
    return preflight, entries


def _validate_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise HoldoutSealCreationError(f"invalid {label}") from error
    require(parsed.tzinfo is not None, f"{label} must include timezone")
    require(parsed.utcoffset() == timezone.utc.utcoffset(parsed), f"{label} must be UTC")
    return parsed


def _assert_source_components_safe(root: Path, relative: str) -> Path:
    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise HoldoutSealCreationError(f"missing artifact {relative!r}") from error
        require(not stat.S_ISLNK(metadata.st_mode), f"symlink artifact {relative!r}")
        if index < len(parts) - 1:
            require(stat.S_ISDIR(metadata.st_mode), f"non-directory artifact parent {relative!r}")
        else:
            require(stat.S_ISREG(metadata.st_mode), f"nonregular artifact {relative!r}")
    return current


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        require(written > 0, "short destination write")
        view = view[written:]


def _copy_and_hash(
    source_root: Path,
    destination_root: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    relative = str(entry["canonical_relative_path"])
    source = _assert_source_components_safe(source_root, relative)
    destination = destination_root / PurePosixPath(relative)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    source_fd = os.open(source, flags)
    destination_fd: int | None = None
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(source_fd)
        require(stat.S_ISREG(before.st_mode), f"nonregular artifact {relative!r}")
        require(before.st_size == entry["expected_byte_size"], f"artifact size drift {relative!r}")
        destination_fd = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
            _write_all(destination_fd, block)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"artifact changed during seal {relative!r}",
        )
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)
    actual_digest = digest.hexdigest()
    require(total == entry["expected_byte_size"], f"artifact size drift {relative!r}")
    require(actual_digest == entry["expected_sha256"], f"artifact hash drift {relative!r}")
    require(_sha256(destination) == actual_digest, f"staged artifact hash drift {relative!r}")
    os.chmod(destination, 0o400)
    return {
        "canonical_relative_path": relative,
        "sealed_relative_path": f"artifacts/{relative}",
        "byte_size": total,
        "sha256": actual_digest,
        "regular_file": True,
        "symlink": False,
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any], mode: int) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        _write_all(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_receipt_atomically_create_only(
    receipt_path: Path, receipt: Mapping[str, Any]
) -> None:
    require(not receipt_path.exists() and not receipt_path.is_symlink(), "receipt already exists")
    require(receipt_path.parent.is_dir(), "receipt parent does not exist")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.partial-", dir=receipt_path.parent
    )
    temporary = Path(temporary_name)
    try:
        payload = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, receipt_path, follow_symlinks=False)
        directory_fd = os.open(receipt_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_seal(
    *,
    preflight_path: Path,
    source_root: Path,
    operator_root: Path,
    receipt_path: Path = DEFAULT_RECEIPT,
    seal_created_at_utc: str,
    receipt_written_at_utc: str,
) -> dict[str, Any]:
    bindings = _validate_repository_bindings()
    preflight, allowlist = _validate_preflight(preflight_path)
    seal_timestamp = _validate_timestamp(seal_created_at_utc, "seal timestamp")
    receipt_timestamp = _validate_timestamp(
        receipt_written_at_utc, "receipt timestamp"
    )
    require(receipt_timestamp >= seal_timestamp, "receipt timestamp precedes seal")

    require(not source_root.is_symlink(), "source root symlink")
    require(not operator_root.is_symlink(), "operator root symlink")
    source_root = source_root.resolve(strict=True)
    operator_root = operator_root.resolve(strict=True)
    receipt_path = receipt_path.resolve()
    require(source_root.is_dir() and not source_root.is_symlink(), "invalid source root")
    require(operator_root.is_dir() and not operator_root.is_symlink(), "invalid operator root")
    require(source_root != operator_root, "source and operator roots must differ")
    require(not operator_root.is_relative_to(source_root), "operator root inside source root")
    require(not source_root.is_relative_to(operator_root), "source root inside operator root")

    final_dir = operator_root / FINAL_OPERATOR_RELATIVE_PATH
    require(not final_dir.exists() and not final_dir.is_symlink(), "final seal path already exists")
    require(not receipt_path.exists() and not receipt_path.is_symlink(), "receipt already exists")
    final_parent = final_dir.parent
    final_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(final_parent.is_dir() and not final_parent.is_symlink(), "invalid final parent")
    os.chmod(final_parent, 0o700)

    temporary = Path(
        tempfile.mkdtemp(prefix=".holdout-seal-v1.staging-", dir=final_parent)
    )
    os.chmod(temporary, 0o700)
    finalized = False
    try:
        artifact_root = temporary / "artifacts"
        artifact_root.mkdir(mode=0o700)
        manifest_entries = [
            _copy_and_hash(source_root, artifact_root, entry) for entry in allowlist
        ]
        manifest_core: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "status": MANIFEST_STATUS,
            "authorization_gate_relative_path": bindings["authorization_gate"]["relative_path"],
            "authorization_gate_sha256": bindings["authorization_gate"]["sha256"],
            "contract_relative_path": bindings["contract"]["relative_path"],
            "contract_sha256": bindings["contract"]["sha256"],
            "g0_v2_relative_path": bindings["g0_v2"]["relative_path"],
            "g0_v2_sha256": bindings["g0_v2"]["sha256"],
            "prerequisite_closure_plan_relative_path": bindings[
                "prerequisite_closure_plan"
            ]["relative_path"],
            "prerequisite_closure_plan_sha256": bindings[
                "prerequisite_closure_plan"
            ]["sha256"],
            "preflight_sha256": _sha256(preflight_path.resolve(strict=True)),
            "partition_contract": EXPECTED_PARTITION,
            "artifact_count": len(manifest_entries),
            "artifact_bytes_total": sum(item["byte_size"] for item in manifest_entries),
            "artifact_entries": manifest_entries,
            "created_at_utc": seal_created_at_utc,
            "seal_opened": False,
            "prior_open_event_count": 0,
            "confirmatory_or_holdout_rows_read": 0,
        }
        manifest = {
            **manifest_core,
            "manifest_fingerprint_sha256": hashlib.sha256(
                _canonical_bytes(manifest_core)
            ).hexdigest(),
        }
        manifest_path = temporary / MANIFEST_BASENAME
        _write_json_create_only(manifest_path, manifest, 0o400)
        manifest_sha256 = _sha256(manifest_path)

        for item in manifest_entries:
            staged = temporary / item["sealed_relative_path"]
            require(staged.is_file() and not staged.is_symlink(), "invalid staged artifact")
            require(staged.stat().st_size == item["byte_size"], "staged byte drift")
            require(_sha256(staged) == item["sha256"], "staged hash drift")
        require(_sha256(manifest_path) == manifest_sha256, "staged manifest drift")
        require(not final_dir.exists(), "final seal path appeared during creation")
        os.rename(temporary, final_dir)
        finalized = True

        final_manifest = final_dir / MANIFEST_BASENAME
        require(_sha256(final_manifest) == manifest_sha256, "final manifest drift")
        for item in manifest_entries:
            final_artifact = final_dir / item["sealed_relative_path"]
            require(final_artifact.is_file() and not final_artifact.is_symlink(), "invalid final artifact")
            require(_sha256(final_artifact) == item["sha256"], "final artifact drift")

        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "authorization_gate_relative_path": bindings["authorization_gate"]["relative_path"],
            "authorization_gate_sha256": bindings["authorization_gate"]["sha256"],
            "contract_relative_path": bindings["contract"]["relative_path"],
            "contract_sha256": bindings["contract"]["sha256"],
            "g0_v2_relative_path": bindings["g0_v2"]["relative_path"],
            "g0_v2_sha256": bindings["g0_v2"]["sha256"],
            "prerequisite_closure_plan_relative_path": bindings[
                "prerequisite_closure_plan"
            ]["relative_path"],
            "prerequisite_closure_plan_sha256": bindings[
                "prerequisite_closure_plan"
            ]["sha256"],
            "creator_relative_path": "scripts/create_pivot_feedback_holdout_access_seal_v1.py",
            "creator_sha256": _sha256(Path(__file__)),
            "preflight_sha256": manifest["preflight_sha256"],
            "manifest_operator_relative_path": (
                f"{FINAL_OPERATOR_RELATIVE_PATH}/{MANIFEST_BASENAME}"
            ),
            "manifest_sha256": manifest_sha256,
            "partition_contract": EXPECTED_PARTITION,
            "artifact_count": manifest["artifact_count"],
            "artifact_bytes_total": manifest["artifact_bytes_total"],
            "manifest_fingerprint_sha256": manifest[
                "manifest_fingerprint_sha256"
            ],
            "seal_created_at_utc": seal_created_at_utc,
            "receipt_written_at_utc": receipt_written_at_utc,
            "receipt_written_last": True,
            "seal_opened": False,
            "prior_open_event_count": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "independent_verification_required": True,
            "independent_verification_complete": False,
            "g0_measurement_freeze_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        }
        _write_receipt_atomically_create_only(receipt_path, receipt)
        return receipt
    finally:
        if not finalized and temporary.exists():
            require(
                temporary.parent == final_parent
                and temporary.name.startswith(".holdout-seal-v1.staging-"),
                "unsafe staging cleanup target",
            )
            shutil.rmtree(temporary)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--seal-created-at-utc", default=None)
    parser.add_argument("--receipt-written-at-utc", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seal_time = args.seal_created_at_utc or _utc_now()
    receipt_time = args.receipt_written_at_utc or _utc_now()
    receipt = create_seal(
        preflight_path=args.preflight,
        source_root=args.source_root,
        operator_root=args.operator_root,
        receipt_path=args.receipt,
        seal_created_at_utc=seal_time,
        receipt_written_at_utc=receipt_time,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
