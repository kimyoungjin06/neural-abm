#!/usr/bin/env python3
"""Bind the exact private G4 source root and receipt-defined artifact list."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "experiments/configs/pivot_feedback_f_g4_private_root_artifact_binding_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_f_g4_private_root_artifact_binding.v1"
OUTPUT_SCHEMA = "pivot_feedback_f_g4_private_root_artifact_binding_output.v1"
PASS_STATUS = (
    "PASS_OPERATOR_ONLY_EXACT_PRIVATE_G4_ROOT_AND_COMPLETE_153_ARTIFACT_LIST_"
    "BOUND_NO_PREFLIGHT_NO_F_PASS"
)


class G4PrivateRootArtifactBindingError(RuntimeError):
    """Fail-closed private-root and explicit-artifact-list binding error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G4PrivateRootArtifactBindingError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def _json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G4PrivateRootArtifactBindingError(f"invalid {label}: {error}") from error


def _yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise G4PrivateRootArtifactBindingError(f"invalid {label}: {error}") from error


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _canonical_relative_path(value: Any) -> str:
    require(isinstance(value, str) and value, "artifact relative path")
    require("\\" not in value, f"noncanonical artifact path {value!r}")
    require(
        not any(token in value for token in ("*", "?", "[", "]", "{", "}")),
        f"wildcard artifact path {value!r}",
    )
    pure = PurePosixPath(value)
    require(not pure.is_absolute(), f"absolute artifact path {value!r}")
    require("." not in pure.parts and ".." not in pure.parts, f"unsafe artifact path {value!r}")
    require(pure.as_posix() == value, f"noncanonical artifact path {value!r}")
    return value


def _validate_repository_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = _mapping(config.get("authorization_parent"), "authorization parent")
    parent_path = _safe_repo_path(parent.get("relative_path"), "authorization parent")
    require(_sha256(parent_path) == parent.get("sha256"), "authorization parent hash")
    parent_document = _json(parent_path, "authorization parent")
    require(
        parent_document.get("schema_version") == parent.get("required_schema_version"),
        "authorization parent schema",
    )
    require(parent_document.get("status") == parent.get("required_status"), "authorization parent status")
    authorization = _mapping(parent_document.get("authorization"), "parent authorization")
    require(
        authorization.get(parent.get("required_authorization")) == parent.get("required_value"),
        "root/list binding authorization",
    )
    require(parent.get("mutation_allowed") is False, "authorization parent mutation")

    package = _mapping(config.get("bound_private_source_package"), "private package")
    contract = _mapping(package.get("production_contract"), "production contract")
    contract_path = _safe_repo_path(contract.get("relative_path"), "production contract")
    require(_sha256(contract_path) == contract.get("sha256"), "production contract hash")
    contract_document = _yaml(contract_path, "production contract")
    require(
        contract_document.get("schema_version") == contract.get("required_schema_version"),
        "production contract schema",
    )
    require(contract.get("mutation_allowed") is False, "production contract mutation")
    return {
        "authorization_parent_relative_path": parent["relative_path"],
        "authorization_parent_sha256": parent["sha256"],
        "production_contract_relative_path": contract["relative_path"],
        "production_contract_sha256": contract["sha256"],
    }


def _walk_source_tree(root: Path, directory_mode: int, file_mode: int) -> dict[str, Any]:
    directories = [root]
    files: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        require(stat.S_IMODE(directory.lstat().st_mode) == directory_mode, f"directory mode {directory}")
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                metadata = path.lstat()
                require(not stat.S_ISLNK(metadata.st_mode), f"symlink in source tree {path}")
                if stat.S_ISDIR(metadata.st_mode):
                    directories.append(path)
                    pending.append(path)
                else:
                    require(stat.S_ISREG(metadata.st_mode), f"nonregular source path {path}")
                    require(stat.S_IMODE(metadata.st_mode) == file_mode, f"file mode {path}")
                    files.append(path)
    relative_files = sorted(path.relative_to(root).as_posix() for path in files)
    return {
        "directories": directories,
        "files": files,
        "relative_files": relative_files,
    }


def _receipt_entries(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = receipt.get("artifacts")
    require(isinstance(raw, list), "receipt artifact list")
    entries: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        row = _mapping(value, f"receipt artifact {index}")
        relative = _canonical_relative_path(row.get("relative_path"))
        byte_size = row.get("bytes")
        digest = row.get("sha256")
        rows = row.get("rows")
        require(isinstance(byte_size, int) and byte_size >= 0, f"artifact byte size {relative}")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"artifact SHA {relative}",
        )
        require(rows is None or isinstance(rows, int) and rows >= 0, f"artifact rows {relative}")
        entries.append(
            {
                "canonical_relative_path": relative,
                "expected_byte_size": byte_size,
                "expected_sha256": digest,
                "expected_rows": rows,
            }
        )
    require(entries == sorted(entries, key=lambda item: item["canonical_relative_path"]), "receipt artifact order")
    paths = [item["canonical_relative_path"] for item in entries]
    require(len(paths) == len(set(paths)), "duplicate receipt artifact path")
    return entries


def _artifact_tree_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    tree = [
        [item["canonical_relative_path"], item["expected_byte_size"], item["expected_sha256"]]
        for item in entries
    ]
    return hashlib.sha256(_canonical_bytes(tree)).hexdigest()


def _validate_source_package(
    config: Mapping[str, Any],
    source_root_argument: Path,
    source_verification_argument: Path,
) -> dict[str, Any]:
    package = _mapping(config.get("bound_private_source_package"), "private package")
    root_contract = _mapping(package.get("exact_source_root"), "source root contract")
    require(not source_root_argument.is_symlink(), "source root symlink")
    source_root = source_root_argument.resolve(strict=True)
    require(source_root.is_dir(), "invalid source root")
    require(source_root.as_posix() == root_contract.get("absolute_path"), "source root absolute path")
    require(source_root.name == root_contract.get("basename"), "source root basename")
    path_sha = hashlib.sha256(source_root.as_posix().encode("utf-8")).hexdigest()
    require(path_sha == root_contract.get("resolved_path_sha256"), "source root path hash")

    receipt_contract = _mapping(package.get("production_receipt"), "receipt contract")
    receipt_path = source_root / str(receipt_contract.get("basename"))
    require(_sha256(receipt_path) == receipt_contract.get("sha256"), "production receipt hash")
    receipt = _json(receipt_path, "production receipt")
    require(receipt.get("schema_version") == receipt_contract.get("required_schema_version"), "receipt schema")
    require(receipt.get("status") == receipt_contract.get("required_status"), "receipt status")
    require(receipt.get("artifact_count") == receipt_contract.get("artifact_count"), "receipt artifact count")
    require(receipt.get("artifact_tree_sha256") == receipt_contract.get("artifact_tree_sha256"), "receipt tree hash")
    require(receipt.get("complete_private_g4_artifact_set") is True, "complete private package")
    require(receipt.get("holdout_preflight_complete") is False, "preflight boundary")
    require(receipt.get("holdout_byte_seal_complete") is False, "seal boundary")
    require(receipt.get("pass_f") is False and receipt.get("pass_g0") is False, "F/G0 boundary")
    require(receipt.get("g4_open") is False, "G4 boundary")
    entries = _receipt_entries(receipt)
    require(len(entries) == receipt_contract.get("artifact_count"), "explicit list count")
    require(
        sum(item["expected_byte_size"] for item in entries) == receipt_contract.get("artifact_bytes_total"),
        "explicit list byte total",
    )
    require(_artifact_tree_sha256(entries) == receipt_contract.get("artifact_tree_sha256"), "explicit list fingerprint")

    verification_contract = _mapping(
        package.get("independent_source_verification"), "source verification contract"
    )
    require(not source_verification_argument.is_symlink(), "source verification symlink")
    source_verification = source_verification_argument.resolve(strict=True)
    require(
        source_verification.as_posix() == verification_contract.get("absolute_path"),
        "source verification absolute path",
    )
    require(_sha256(source_verification) == verification_contract.get("sha256"), "source verification hash")
    verification = _json(source_verification, "source verification")
    require(
        verification.get("schema_version") == verification_contract.get("required_schema_version"),
        "source verification schema",
    )
    require(verification.get("status") == verification_contract.get("required_status"), "source verification status")
    require(
        verification.get("output_receipt_sha256")
        == verification_contract.get("required_output_receipt_sha256"),
        "source verification receipt binding",
    )
    require(
        verification.get("artifact_count") == verification_contract.get("required_artifact_count"),
        "source verification artifact count",
    )
    require(
        verification.get("source_replay_all_materialized_columns_verified")
        is verification_contract.get("required_source_replay_all_materialized_columns_verified"),
        "source replay boundary",
    )
    require(
        verification.get("confirmatory_or_holdout_analytic_rows_read")
        == verification_contract.get("required_confirmatory_or_holdout_analytic_rows_read"),
        "analytic read boundary",
    )
    require(verification.get("pass_f") is False and verification.get("pass_g0") is False, "verification F/G0 boundary")
    require(verification.get("g4_open") is False, "verification G4 boundary")

    tree = _walk_source_tree(
        source_root,
        int(str(root_contract.get("directory_mode")), 8),
        int(str(root_contract.get("file_mode")), 8),
    )
    require(
        len(tree["directories"]) == root_contract.get("expected_directory_count_including_root"),
        "source directory count",
    )
    require(
        len(tree["files"]) == root_contract.get("expected_regular_file_count_including_receipt"),
        "source regular file count",
    )
    expected_tree_paths = sorted(
        [item["canonical_relative_path"] for item in entries] + [receipt_path.name]
    )
    require(tree["relative_files"] == expected_tree_paths, "source tree closure")
    entry_by_path = {item["canonical_relative_path"]: item for item in entries}
    for relative, item in entry_by_path.items():
        path = source_root / relative
        metadata = path.lstat()
        require(stat.S_ISREG(metadata.st_mode), f"nonregular bound artifact {relative}")
        require(metadata.st_size == item["expected_byte_size"], f"bound artifact size {relative}")

    root_metadata = source_root.lstat()
    return {
        "source_root": source_root,
        "root_metadata": root_metadata,
        "receipt_path": receipt_path,
        "source_verification": source_verification,
        "entries": entries,
        "tree": tree,
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any], mode: int) -> None:
    require(path.parent.is_dir(), "binding output parent")
    require(not path.exists() and not path.is_symlink(), "binding output exists")
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
        encoded = payload.encode("utf-8")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            require(written > 0, "short binding write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def bind(
    *,
    config_path: Path,
    source_root: Path,
    source_verification: Path,
    output_path: Path,
    bound_at_utc: str,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path, "binding config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    repository_bindings = _validate_repository_bindings(config)
    package = _validate_source_package(config, source_root, source_verification)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output_contract.get("binding_schema_version") == OUTPUT_SCHEMA, "output schema contract")
    require(output_contract.get("passing_status") == PASS_STATUS, "output status contract")
    output_path = output_path.resolve()
    require(output_path.name == output_contract.get("binding_basename"), "binding output basename")
    require(not output_path.is_relative_to(package["source_root"]), "binding output inside source root")
    try:
        parsed = datetime.fromisoformat(bound_at_utc.replace("Z", "+00:00"))
    except ValueError as error:
        raise G4PrivateRootArtifactBindingError("invalid bound timestamp") from error
    require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed), "bound timestamp must be UTC")

    root_metadata = package["root_metadata"]
    entries = package["entries"]
    authorization = _mapping(config.get("authorization_after_pass"), "authorization after pass")
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "status": PASS_STATUS,
        "bound_at_utc": bound_at_utc,
        "repository_bindings": {
            "config_relative_path": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": _sha256(config_path),
            "builder_relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "builder_sha256": _sha256(Path(__file__).resolve()),
            **repository_bindings,
        },
        "source_root_binding": {
            "resolved_absolute_path": package["source_root"].as_posix(),
            "basename": package["source_root"].name,
            "resolved_path_sha256": hashlib.sha256(
                package["source_root"].as_posix().encode("utf-8")
            ).hexdigest(),
            "device": root_metadata.st_dev,
            "inode": root_metadata.st_ino,
            "mode": f"{stat.S_IMODE(root_metadata.st_mode):04o}",
            "directory_count_including_root": len(package["tree"]["directories"]),
            "regular_file_count_including_receipt": len(package["tree"]["files"]),
            "symlink_count": 0,
        },
        "production_evidence_binding": {
            "receipt_basename": package["receipt_path"].name,
            "receipt_sha256": _sha256(package["receipt_path"]),
            "source_verification_absolute_path": package["source_verification"].as_posix(),
            "source_verification_sha256": _sha256(package["source_verification"]),
            "artifact_tree_sha256": _artifact_tree_sha256(entries),
        },
        "artifact_count": len(entries),
        "artifact_bytes_total": sum(item["expected_byte_size"] for item in entries),
        "artifact_entries": entries,
        "binding_assertions": {
            "exact_private_source_root_bound": True,
            "complete_explicit_artifact_list_bound": True,
            "explicit_list_sorted_and_unique": True,
            "explicit_list_exact_receipt_projection": True,
            "actual_tree_closes_to_explicit_list_plus_receipt": True,
            "listed_paths_regular_and_size_matched": True,
            "source_tree_symlinks_absent": True,
            "expected_sha_values_copied_without_refresh_from_verified_receipt": True,
            "source_artifact_content_hashed_in_this_component": False,
            "parquet_or_row_parser_used": False,
        },
        "access_accounting": {
            "repository_documents_read": 3,
            "operator_json_control_files_read": 2,
            "source_tree_paths_enumerated": len(package["tree"]["files"]),
            "source_artifact_metadata_entries_checked": len(entries),
            "source_artifact_content_files_opened": 0,
            "source_artifact_bytes_hashed": 0,
            "operator_parquet_files_opened": 0,
            "confirmatory_or_holdout_analytic_rows_read": 0,
            "preflight_written": False,
            "byte_seal_created": False,
        },
        "completion": {
            "this_component_is_new_gate": False,
            "exact_private_source_root_bound": True,
            "complete_explicit_artifact_list_bound": True,
            "complete_private_g4_artifact_set": True,
            "earlier_partition_guard_application_complete": False,
            "holdout_preflight_complete": False,
            "holdout_byte_seal_complete": False,
            "entry_or_persistence_labels_materialized": False,
            "confirmatory_or_holdout_analytic_rows_read": 0,
            "model_score_read": False,
            "model_fitted": False,
            "g4_open": False,
            "pass_f": False,
            "pass_g0": False,
        },
        "authorization": authorization,
    }
    mode = int(str(output_contract.get("output_file_mode")), 8)
    _write_json_create_only(output_path, payload, mode)
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bound-at-utc", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = bind(
        config_path=args.config,
        source_root=args.source_root,
        source_verification=args.source_verification,
        output_path=args.output,
        bound_at_utc=args.bound_at_utc or _utc_now(),
    )
    safe = {
        "schema_version": result["schema_version"],
        "status": result["status"],
        "source_root_basename": result["source_root_binding"]["basename"],
        "artifact_count": result["artifact_count"],
        "artifact_bytes_total": result["artifact_bytes_total"],
        "source_artifact_bytes_hashed": result["access_accounting"]["source_artifact_bytes_hashed"],
        "holdout_preflight_complete": result["completion"]["holdout_preflight_complete"],
        "holdout_byte_seal_complete": result["completion"]["holdout_byte_seal_complete"],
        "pass_f": result["completion"]["pass_f"],
        "pass_g0": result["completion"]["pass_g0"],
        "g4_open": result["completion"]["g4_open"],
    }
    print(json.dumps(safe, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
