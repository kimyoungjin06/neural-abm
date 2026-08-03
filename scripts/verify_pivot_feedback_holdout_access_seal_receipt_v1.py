#!/usr/bin/env python3
"""Independently rehash a finalized holdout seal without parsing row data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = (
    REPO_ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_holdout_access_seal_receipt_v1.summary.json"
)
RECEIPT_SCHEMA = "pivot_feedback_holdout_access_seal_outputs.v1"
RECEIPT_STATUS = "PASS_CONFIRMATORY_PARTITIONS_SEALED_AND_UNOPENED"
MANIFEST_SCHEMA = "pivot_feedback_holdout_access_seal_manifest.v1"
MANIFEST_STATUS = "SEALED_ARTIFACT_MANIFEST_FINALIZED_UNOPENED"
VERIFY_SCHEMA = "pivot_feedback_holdout_access_seal_verification.v1"
VERIFY_STATUS = "PASS_READ_ONLY_INDEPENDENT_HOLDOUT_SEAL_VERIFICATION_UNOPENED"
FINAL_OPERATOR_RELATIVE_PATH = "pivot-feedback/holdout-seal-v1"
MANIFEST_RELATIVE_PATH = f"{FINAL_OPERATOR_RELATIVE_PATH}/manifest.json"
CREATOR_RELATIVE_PATH = "scripts/create_pivot_feedback_holdout_access_seal_v1.py"
CREATOR_SHA256 = "e65d8fa3a3c85d6f1c87564dab586fcda0ed163cf4455426caff7e773140cf76"
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


class HoldoutSealVerificationError(RuntimeError):
    """Fail-closed independent holdout-seal verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutSealVerificationError(message)


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
        raise HoldoutSealVerificationError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise HoldoutSealVerificationError(f"invalid {label}: {error}") from error


def _validate_repository_bindings(receipt: Mapping[str, Any]) -> None:
    for key, expected in REPOSITORY_BINDINGS.items():
        path = REPO_ROOT / str(expected["relative_path"])
        require(path.is_file() and not path.is_symlink(), f"invalid {key} binding")
        require(_sha256(path) == expected["sha256"], f"{key} hash drift")
        document = _read_yaml(path, key)
        require(document.get("schema_version") == expected["schema_version"], f"{key} schema")
        require(document.get("status") == expected["status"], f"{key} status")
        require(
            receipt.get(f"{key}_relative_path") == expected["relative_path"],
            f"receipt {key} path",
        )
        require(receipt.get(f"{key}_sha256") == expected["sha256"], f"receipt {key} hash")
    creator = REPO_ROOT / CREATOR_RELATIVE_PATH
    require(creator.is_file() and not creator.is_symlink(), "invalid creator")
    require(_sha256(creator) == CREATOR_SHA256, "creator hash drift")
    require(receipt.get("creator_relative_path") == CREATOR_RELATIVE_PATH, "creator path")
    require(receipt.get("creator_sha256") == CREATOR_SHA256, "receipt creator hash")


def _canonical_relative(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, label)
    require("\\" not in value, f"noncanonical {label}")
    pure = PurePosixPath(value)
    require(not pure.is_absolute(), f"absolute {label}")
    require("." not in pure.parts and ".." not in pure.parts, f"unsafe {label}")
    require(pure.as_posix() == value, f"noncanonical {label}")
    return value


def _safe_operator_file(root: Path, relative: str, label: str) -> Path:
    relative = _canonical_relative(relative, label)
    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise HoldoutSealVerificationError(f"missing {label}") from error
        require(not stat.S_ISLNK(metadata.st_mode), f"symlink {label}")
        if index < len(parts) - 1:
            require(stat.S_ISDIR(metadata.st_mode), f"non-directory {label} parent")
        else:
            require(stat.S_ISREG(metadata.st_mode), f"nonregular {label}")
    return current


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema")
    require(receipt.get("status") == RECEIPT_STATUS, "receipt status")
    require(receipt.get("partition_contract") == EXPECTED_PARTITION, "receipt partition")
    require(receipt.get("manifest_operator_relative_path") == MANIFEST_RELATIVE_PATH, "manifest path")
    require(receipt.get("receipt_written_last") is True, "receipt written last")
    require(receipt.get("seal_opened") is False, "seal opened")
    require(receipt.get("prior_open_event_count") == 0, "prior open events")
    require(receipt.get("confirmatory_or_holdout_rows_read") == 0, "holdout row reads")
    require(receipt.get("independent_verification_required") is True, "verification requirement")
    require(receipt.get("independent_verification_complete") is False, "pre-verification state")
    require(receipt.get("g0_measurement_freeze_authorized") is False, "G0 boundary")
    require(receipt.get("model_fitting_authorized") is False, "model boundary")
    require(receipt.get("g4_open") is False, "G4 boundary")
    require(receipt.get("scientific_effect") == "NOT_ESTABLISHED", "effect boundary")
    for key in ("artifact_count", "artifact_bytes_total"):
        require(isinstance(receipt.get(key), int) and receipt[key] >= 0, f"receipt {key}")
    for key in (
        "manifest_sha256",
        "manifest_fingerprint_sha256",
        "preflight_sha256",
    ):
        value = receipt.get(key)
        require(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            f"receipt {key}",
        )


def _validate_manifest(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any], final_root: Path
) -> dict[str, Any]:
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema")
    require(manifest.get("status") == MANIFEST_STATUS, "manifest status")
    require(manifest.get("partition_contract") == EXPECTED_PARTITION, "manifest partition")
    require(manifest.get("seal_opened") is False, "manifest seal opened")
    require(manifest.get("prior_open_event_count") == 0, "manifest prior opens")
    require(manifest.get("confirmatory_or_holdout_rows_read") == 0, "manifest row reads")
    for key, expected in REPOSITORY_BINDINGS.items():
        require(
            manifest.get(f"{key}_relative_path") == expected["relative_path"],
            f"manifest {key} path",
        )
        require(manifest.get(f"{key}_sha256") == expected["sha256"], f"manifest {key} hash")
    require(manifest.get("preflight_sha256") == receipt["preflight_sha256"], "preflight binding")

    fingerprint = manifest.get("manifest_fingerprint_sha256")
    core = dict(manifest)
    core.pop("manifest_fingerprint_sha256", None)
    require(
        hashlib.sha256(_canonical_bytes(core)).hexdigest() == fingerprint,
        "manifest fingerprint",
    )
    require(fingerprint == receipt["manifest_fingerprint_sha256"], "receipt fingerprint")

    entries_value = manifest.get("artifact_entries")
    require(isinstance(entries_value, list) and entries_value, "manifest entries")
    seen: set[str] = set()
    expected_files = {"manifest.json"}
    total_bytes = 0
    for value in entries_value:
        entry = _mapping(value, "manifest entry")
        relative = _canonical_relative(entry.get("canonical_relative_path"), "artifact path")
        sealed = _canonical_relative(entry.get("sealed_relative_path"), "sealed path")
        require(sealed == f"artifacts/{relative}", "sealed path binding")
        require(relative not in seen, f"duplicate artifact {relative!r}")
        seen.add(relative)
        require(entry.get("regular_file") is True, "regular-file assertion")
        require(entry.get("symlink") is False, "symlink assertion")
        byte_size = entry.get("byte_size")
        digest = entry.get("sha256")
        require(isinstance(byte_size, int) and byte_size >= 0, "artifact byte size")
        require(isinstance(digest, str) and len(digest) == 64, "artifact digest")
        path = _safe_operator_file(final_root, sealed, "sealed artifact")
        require(path.stat().st_size == byte_size, f"artifact size drift {relative!r}")
        require(_sha256(path) == digest, f"artifact hash drift {relative!r}")
        total_bytes += byte_size
        expected_files.add(sealed)
    require(list(seen) != [], "empty artifact set")
    require(
        [item["canonical_relative_path"] for item in entries_value] == sorted(seen),
        "manifest entries not sorted",
    )
    require(manifest.get("artifact_count") == len(entries_value), "manifest artifact count")
    require(manifest.get("artifact_bytes_total") == total_bytes, "manifest byte total")
    require(receipt.get("artifact_count") == len(entries_value), "receipt artifact count")
    require(receipt.get("artifact_bytes_total") == total_bytes, "receipt byte total")

    observed_files: set[str] = set()
    for path in final_root.rglob("*"):
        metadata = path.lstat()
        require(not stat.S_ISLNK(metadata.st_mode), "symlink in sealed tree")
        if stat.S_ISREG(metadata.st_mode):
            observed_files.add(path.relative_to(final_root).as_posix())
        else:
            require(stat.S_ISDIR(metadata.st_mode), "nonregular entry in sealed tree")
    require(observed_files == expected_files, "sealed tree file-set drift")
    return {
        "artifact_count": len(entries_value),
        "artifact_bytes_total": total_bytes,
        "all_artifacts_rehashed": True,
        "sealed_tree_exact": True,
        "manifest_fingerprint_valid": True,
    }


def verify(
    *, receipt_path: Path = DEFAULT_RECEIPT, operator_root: Path
) -> dict[str, Any]:
    require(not receipt_path.is_symlink(), "receipt symlink")
    require(not operator_root.is_symlink(), "operator root symlink")
    receipt_path = receipt_path.resolve(strict=True)
    operator_root = operator_root.resolve(strict=True)
    require(operator_root.is_dir() and not operator_root.is_symlink(), "invalid operator root")
    receipt = _read_json(receipt_path, "holdout seal receipt")
    _validate_receipt(receipt)
    _validate_repository_bindings(receipt)

    manifest_path = _safe_operator_file(
        operator_root, MANIFEST_RELATIVE_PATH, "holdout seal manifest"
    )
    require(_sha256(manifest_path) == receipt["manifest_sha256"], "manifest hash drift")
    manifest = _read_json(manifest_path, "holdout seal manifest")
    final_root = manifest_path.parent
    replay = _validate_manifest(manifest, receipt, final_root)

    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "receipt_relative_path": (
            receipt_path.relative_to(REPO_ROOT).as_posix()
            if receipt_path.is_relative_to(REPO_ROOT)
            else receipt_path.name
        ),
        "receipt_sha256": _sha256(receipt_path),
        "manifest_operator_relative_path": MANIFEST_RELATIVE_PATH,
        "manifest_sha256": receipt["manifest_sha256"],
        "partition_contract": EXPECTED_PARTITION,
        "artifact_replay": replay,
        "access_accounting": {
            "repository_documents_read": 6,
            "operator_json_metadata_files_read": 1,
            "operator_artifact_files_rehashed": replay["artifact_count"],
            "operator_artifact_bytes_rehashed": replay["artifact_bytes_total"],
            "operator_parquet_files_parsed": 0,
            "production_row_level_files_opened": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "api_requests_created_or_submitted": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "authorization": {
            "holdout_seal_receipt": "COMPLETE_INDEPENDENTLY_VERIFIED",
            "seal_opening": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "The read-only verifier independently rehashed the exact private "
            "manifest and every sealed artifact byte without parsing row "
            "data. It confirms only that the declared holdout partition is "
            "hash-closed and unopened. It does not open the seal, pass G0, "
            "authorize model fitting, open G4, or establish a scientific effect."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--operator-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(receipt_path=args.receipt, operator_root=args.operator_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
