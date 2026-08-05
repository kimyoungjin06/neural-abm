#!/usr/bin/env python3
"""Independently verify the private G4 root and explicit artifact-list binding."""

from __future__ import annotations

import argparse
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
BINDING_SCHEMA = "pivot_feedback_f_g4_private_root_artifact_binding_output.v1"
BINDING_STATUS = (
    "PASS_OPERATOR_ONLY_EXACT_PRIVATE_G4_ROOT_AND_COMPLETE_153_ARTIFACT_LIST_"
    "BOUND_NO_PREFLIGHT_NO_F_PASS"
)
VERIFY_SCHEMA = "pivot_feedback_f_g4_private_root_artifact_binding_verification.v1"
VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_PRIVATE_G4_ROOT_AND_COMPLETE_153_ARTIFACT_LIST_"
    "BOUND_NO_PREFLIGHT_NO_F_PASS"
)


class G4PrivateRootArtifactBindingVerificationError(RuntimeError):
    """Fail-closed independent root/list binding verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G4PrivateRootArtifactBindingVerificationError(message)


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
        raise G4PrivateRootArtifactBindingVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise G4PrivateRootArtifactBindingVerificationError(
            f"invalid {label}: {error}"
        ) from error


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
    parent_authorization = _mapping(parent_document.get("authorization"), "parent authorization")
    require(
        parent_authorization.get(parent.get("required_authorization"))
        == parent.get("required_value"),
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
    return {
        "directories": directories,
        "files": files,
        "relative_files": sorted(path.relative_to(root).as_posix() for path in files),
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any], mode: int) -> None:
    require(path.parent.is_dir(), "verification output parent")
    require(not path.exists() and not path.is_symlink(), "verification output exists")
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
            require(written > 0, "short verification write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify(
    *,
    config_path: Path,
    source_root_argument: Path,
    source_verification_argument: Path,
    binding_path: Path,
    verification_output: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path, "binding config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    repository_expected = _validate_repository_bindings(config)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output_contract.get("binding_schema_version") == BINDING_SCHEMA, "binding schema contract")
    require(output_contract.get("passing_status") == BINDING_STATUS, "binding status contract")
    require(output_contract.get("verification_schema_version") == VERIFY_SCHEMA, "verification schema contract")
    require(output_contract.get("verification_passing_status") == VERIFY_STATUS, "verification status contract")

    package = _mapping(config.get("bound_private_source_package"), "private package")
    root_contract = _mapping(package.get("exact_source_root"), "source root contract")
    require(not source_root_argument.is_symlink(), "source root symlink")
    source_root = source_root_argument.resolve(strict=True)
    require(source_root.is_dir(), "invalid source root")
    require(source_root.as_posix() == root_contract.get("absolute_path"), "source root absolute path")
    require(source_root.name == root_contract.get("basename"), "source root basename")
    root_path_sha = hashlib.sha256(source_root.as_posix().encode("utf-8")).hexdigest()
    require(root_path_sha == root_contract.get("resolved_path_sha256"), "source root path hash")

    receipt_contract = _mapping(package.get("production_receipt"), "receipt contract")
    receipt_path = source_root / str(receipt_contract.get("basename"))
    require(_sha256(receipt_path) == receipt_contract.get("sha256"), "production receipt hash")
    receipt = _json(receipt_path, "production receipt")
    require(receipt.get("schema_version") == receipt_contract.get("required_schema_version"), "receipt schema")
    require(receipt.get("status") == receipt_contract.get("required_status"), "receipt status")
    require(receipt.get("artifact_count") == receipt_contract.get("artifact_count"), "receipt artifact count")
    expected_entries = _receipt_entries(receipt)
    require(len(expected_entries) == receipt_contract.get("artifact_count"), "expected artifact count")
    require(
        sum(item["expected_byte_size"] for item in expected_entries)
        == receipt_contract.get("artifact_bytes_total"),
        "expected artifact bytes",
    )
    expected_tree_sha = _artifact_tree_sha256(expected_entries)
    require(expected_tree_sha == receipt_contract.get("artifact_tree_sha256"), "expected artifact tree hash")

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
    source_verification_document = _json(source_verification, "source verification")
    require(
        source_verification_document.get("schema_version")
        == verification_contract.get("required_schema_version"),
        "source verification schema",
    )
    require(
        source_verification_document.get("status") == verification_contract.get("required_status"),
        "source verification status",
    )
    require(
        source_verification_document.get("output_receipt_sha256")
        == verification_contract.get("required_output_receipt_sha256"),
        "source verification receipt binding",
    )
    require(
        source_verification_document.get("source_replay_all_materialized_columns_verified") is True,
        "source replay boundary",
    )
    require(
        source_verification_document.get("confirmatory_or_holdout_analytic_rows_read") == 0,
        "source analytic read boundary",
    )
    require(source_verification_document.get("pass_f") is False, "source F boundary")
    require(source_verification_document.get("pass_g0") is False, "source G0 boundary")
    require(source_verification_document.get("g4_open") is False, "source G4 boundary")

    binding_path = binding_path.resolve(strict=True)
    require(binding_path.name == output_contract.get("binding_basename"), "binding basename")
    require(not binding_path.is_relative_to(source_root), "binding inside source root")
    require(stat.S_IMODE(binding_path.stat().st_mode) == int(str(output_contract.get("output_file_mode")), 8), "binding mode")
    binding = _json(binding_path, "root/list binding")
    require(binding.get("schema_version") == BINDING_SCHEMA, "binding schema")
    require(binding.get("status") == BINDING_STATUS, "binding status")
    require(binding.get("artifact_count") == len(expected_entries), "binding artifact count")
    require(
        binding.get("artifact_bytes_total")
        == sum(item["expected_byte_size"] for item in expected_entries),
        "binding artifact bytes",
    )
    require(binding.get("artifact_entries") == expected_entries, "binding exact receipt projection")

    repository_actual = _mapping(binding.get("repository_bindings"), "binding repository bindings")
    require(
        repository_actual.get("config_relative_path") == config_path.relative_to(ROOT).as_posix(),
        "binding config path",
    )
    require(repository_actual.get("config_sha256") == _sha256(config_path), "binding config hash")
    builder_path = _safe_repo_path(repository_actual.get("builder_relative_path"), "binding builder")
    require(repository_actual.get("builder_sha256") == _sha256(builder_path), "binding builder hash")
    for key, value in repository_expected.items():
        require(repository_actual.get(key) == value, f"binding repository field {key}")

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
        [item["canonical_relative_path"] for item in expected_entries] + [receipt_path.name]
    )
    require(tree["relative_files"] == expected_tree_paths, "independent source tree closure")
    for item in expected_entries:
        relative = item["canonical_relative_path"]
        metadata = (source_root / relative).lstat()
        require(stat.S_ISREG(metadata.st_mode), f"nonregular bound artifact {relative}")
        require(metadata.st_size == item["expected_byte_size"], f"bound artifact size {relative}")

    root_metadata = source_root.lstat()
    root_binding = _mapping(binding.get("source_root_binding"), "source root binding")
    require(root_binding.get("resolved_absolute_path") == source_root.as_posix(), "bound root path")
    require(root_binding.get("basename") == source_root.name, "bound root basename")
    require(root_binding.get("resolved_path_sha256") == root_path_sha, "bound root path hash")
    require(root_binding.get("device") == root_metadata.st_dev, "bound root device")
    require(root_binding.get("inode") == root_metadata.st_ino, "bound root inode")
    require(root_binding.get("mode") == f"{stat.S_IMODE(root_metadata.st_mode):04o}", "bound root mode")
    require(
        root_binding.get("directory_count_including_root") == len(tree["directories"]),
        "bound directory count",
    )
    require(
        root_binding.get("regular_file_count_including_receipt") == len(tree["files"]),
        "bound file count",
    )
    require(root_binding.get("symlink_count") == 0, "bound symlink count")

    evidence = _mapping(binding.get("production_evidence_binding"), "production evidence binding")
    require(evidence.get("receipt_basename") == receipt_path.name, "bound receipt basename")
    require(evidence.get("receipt_sha256") == _sha256(receipt_path), "bound receipt hash")
    require(
        evidence.get("source_verification_absolute_path") == source_verification.as_posix(),
        "bound source verification path",
    )
    require(
        evidence.get("source_verification_sha256") == _sha256(source_verification),
        "bound source verification hash",
    )
    require(evidence.get("artifact_tree_sha256") == expected_tree_sha, "bound tree hash")

    assertions = _mapping(binding.get("binding_assertions"), "binding assertions")
    expected_assertions = {
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
    }
    require(assertions == expected_assertions, "binding assertions")
    access = _mapping(binding.get("access_accounting"), "binding access accounting")
    require(access.get("source_tree_paths_enumerated") == len(tree["files"]), "tree path accounting")
    require(access.get("source_artifact_metadata_entries_checked") == len(expected_entries), "metadata accounting")
    require(access.get("source_artifact_content_files_opened") == 0, "artifact open accounting")
    require(access.get("source_artifact_bytes_hashed") == 0, "artifact hash accounting")
    require(access.get("operator_parquet_files_opened") == 0, "Parquet access accounting")
    require(access.get("confirmatory_or_holdout_analytic_rows_read") == 0, "analytic read accounting")
    require(access.get("preflight_written") is False, "preflight accounting")
    require(access.get("byte_seal_created") is False, "seal accounting")

    completion = _mapping(binding.get("completion"), "binding completion")
    require(completion.get("this_component_is_new_gate") is False, "new gate boundary")
    require(completion.get("exact_private_source_root_bound") is True, "root binding completion")
    require(completion.get("complete_explicit_artifact_list_bound") is True, "list binding completion")
    require(completion.get("complete_private_g4_artifact_set") is True, "private package completion")
    require(completion.get("earlier_partition_guard_application_complete") is False, "guard application boundary")
    require(completion.get("holdout_preflight_complete") is False, "preflight completion boundary")
    require(completion.get("holdout_byte_seal_complete") is False, "seal completion boundary")
    require(completion.get("entry_or_persistence_labels_materialized") is False, "label boundary")
    require(completion.get("confirmatory_or_holdout_analytic_rows_read") == 0, "analytic completion boundary")
    require(completion.get("model_score_read") is False, "score boundary")
    require(completion.get("model_fitted") is False, "model boundary")
    require(completion.get("g4_open") is False, "G4 completion boundary")
    require(completion.get("pass_f") is False and completion.get("pass_g0") is False, "F/G0 completion boundary")
    authorization = _mapping(binding.get("authorization"), "binding authorization")
    require(authorization == config.get("authorization_after_pass"), "binding authorization projection")

    verification_output = verification_output.resolve()
    require(
        verification_output.name == output_contract.get("verification_basename"),
        "verification output basename",
    )
    require(not verification_output.is_relative_to(source_root), "verification output inside source root")
    result = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "binding_basename": binding_path.name,
        "binding_sha256": _sha256(binding_path),
        "source_root_basename": source_root.name,
        "source_root_resolved_path_sha256": root_path_sha,
        "source_root_device": root_metadata.st_dev,
        "source_root_inode": root_metadata.st_ino,
        "artifact_count_verified": len(expected_entries),
        "artifact_bytes_total_verified": sum(
            item["expected_byte_size"] for item in expected_entries
        ),
        "artifact_tree_sha256_verified": expected_tree_sha,
        "directory_count_including_root_verified": len(tree["directories"]),
        "regular_file_count_including_receipt_verified": len(tree["files"]),
        "explicit_list_sorted_and_unique_verified": True,
        "explicit_list_exact_receipt_projection_verified": True,
        "actual_tree_closure_verified": True,
        "listed_paths_regular_and_size_matched_verified": True,
        "source_root_identity_verified": True,
        "source_tree_modes_verified": True,
        "source_tree_symlinks_absent": True,
        "expected_sha_values_inherited_without_refresh": True,
        "source_artifact_content_files_opened": 0,
        "source_artifact_bytes_hashed": 0,
        "operator_parquet_files_opened": 0,
        "confirmatory_or_holdout_analytic_rows_read": 0,
        "exact_private_source_root_bound": True,
        "complete_explicit_artifact_list_bound": True,
        "complete_private_g4_artifact_set": True,
        "earlier_partition_guard_application_complete": False,
        "holdout_preflight_complete": False,
        "holdout_byte_seal_complete": False,
        "entry_or_persistence_labels_materialized": False,
        "model_score_read": False,
        "model_fitted": False,
        "g4_open": False,
        "pass_f": False,
        "pass_g0": False,
        "authorization": authorization,
    }
    mode = int(str(output_contract.get("output_file_mode")), 8)
    _write_json_create_only(verification_output, result, mode)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        source_root_argument=args.source_root,
        source_verification_argument=args.source_verification,
        binding_path=args.binding,
        verification_output=args.verification_output,
    )
    safe = {
        "schema_version": result["schema_version"],
        "status": result["status"],
        "source_root_basename": result["source_root_basename"],
        "artifact_count_verified": result["artifact_count_verified"],
        "artifact_bytes_total_verified": result["artifact_bytes_total_verified"],
        "source_artifact_bytes_hashed": result["source_artifact_bytes_hashed"],
        "holdout_preflight_complete": result["holdout_preflight_complete"],
        "holdout_byte_seal_complete": result["holdout_byte_seal_complete"],
        "pass_f": result["pass_f"],
        "pass_g0": result["pass_g0"],
        "g4_open": result["g4_open"],
    }
    print(json.dumps(safe, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
