#!/usr/bin/env python3
"""Independently rehash and verify the bound private G4 explicit-list preflight."""

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
    / "experiments/configs/pivot_feedback_f_g4_private_explicit_list_preflight_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_f_g4_private_explicit_list_preflight.v1"
PREFLIGHT_SCHEMA = "pivot_feedback_holdout_access_seal_preflight.v1"
PREFLIGHT_STATUS = "PASS_EXPLICIT_ALLOWLIST_HASHES_BOUND_ZERO_PRIOR_OPEN_EVENTS"
COMPONENT_SCHEMA = "pivot_feedback_f_g4_private_explicit_list_preflight_output.v1"
COMPONENT_STATUS = (
    "PASS_OPERATOR_ONLY_BOUND_153_ARTIFACT_BYTES_REHASHED_NO_SEAL_NO_F_PASS"
)
VERIFY_SCHEMA = "pivot_feedback_f_g4_private_explicit_list_preflight_verification.v1"
VERIFY_STATUS = (
    "PASS_INDEPENDENT_BOUND_153_ARTIFACT_BYTES_REHASHED_PREFLIGHT_NO_SEAL_NO_F_PASS"
)


class G4PrivateExplicitListPreflightVerificationError(RuntimeError):
    """Fail-closed independent preflight verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G4PrivateExplicitListPreflightVerificationError(message)


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
        raise G4PrivateExplicitListPreflightVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise G4PrivateExplicitListPreflightVerificationError(
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


def _validate_repository_config(config: Mapping[str, Any]) -> dict[str, Any]:
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
        "preflight authorization",
    )
    require(parent.get("mutation_allowed") is False, "authorization parent mutation")

    bound = _mapping(config.get("bound_root_and_list"), "bound root/list")
    contract = _mapping(bound.get("contract"), "root/list contract")
    contract_path = _safe_repo_path(contract.get("relative_path"), "root/list contract")
    require(_sha256(contract_path) == contract.get("sha256"), "root/list contract hash")
    contract_document = _yaml(contract_path, "root/list contract")
    require(
        contract_document.get("schema_version") == contract.get("required_schema_version"),
        "root/list contract schema",
    )
    require(contract.get("mutation_allowed") is False, "root/list contract mutation")

    compatibility = _mapping(config.get("seal_creator_compatibility"), "seal compatibility")
    repo_bindings = _mapping(compatibility.get("repository_bindings"), "seal bindings")
    validated_repo: dict[str, dict[str, str]] = {}
    for key, raw in repo_bindings.items():
        binding = _mapping(raw, f"seal binding {key}")
        path = _safe_repo_path(binding.get("relative_path"), f"seal binding {key}")
        require(_sha256(path) == binding.get("sha256"), f"seal binding {key} hash")
        document = _yaml(path, f"seal binding {key}")
        require(document.get("schema_version") == binding.get("required_schema_version"), f"seal binding {key} schema")
        require(document.get("status") == binding.get("required_status"), f"seal binding {key} status")
        validated_repo[key] = {
            "relative_path": binding["relative_path"],
            "sha256": binding["sha256"],
        }
    consumers = _mapping(compatibility.get("consumer_scripts"), "consumer scripts")
    validated_consumers: dict[str, dict[str, str]] = {}
    for key, raw in consumers.items():
        binding = _mapping(raw, f"consumer {key}")
        path = _safe_repo_path(binding.get("relative_path"), f"consumer {key}")
        require(_sha256(path) == binding.get("sha256"), f"consumer {key} hash")
        validated_consumers[key] = {
            "relative_path": binding["relative_path"],
            "sha256": binding["sha256"],
        }
    return {
        "parent": {
            "relative_path": parent["relative_path"],
            "sha256": parent["sha256"],
        },
        "root_list_contract": {
            "relative_path": contract["relative_path"],
            "sha256": contract["sha256"],
        },
        "seal_repository_bindings": validated_repo,
        "seal_consumer_bindings": validated_consumers,
    }


def _expected_entries(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = binding.get("artifact_entries")
    require(isinstance(values, list), "bound artifact list")
    entries: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        row = _mapping(value, f"bound artifact {index}")
        relative = _canonical_relative_path(row.get("canonical_relative_path"))
        byte_size = row.get("expected_byte_size")
        digest = row.get("expected_sha256")
        require(isinstance(byte_size, int) and byte_size >= 0, f"bound byte size {relative}")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"bound SHA {relative}",
        )
        entries.append(
            {
                "canonical_relative_path": relative,
                "expected_byte_size": byte_size,
                "expected_sha256": digest,
            }
        )
    paths = [item["canonical_relative_path"] for item in entries]
    require(paths == sorted(paths), "bound artifact order")
    require(len(paths) == len(set(paths)), "duplicate bound artifact path")
    return entries


def _validate_bound_controls(
    config: Mapping[str, Any],
    root_list_binding_argument: Path,
    root_list_binding_verification_argument: Path,
) -> dict[str, Any]:
    bound = _mapping(config.get("bound_root_and_list"), "bound root/list")
    binding_contract = _mapping(bound.get("binding"), "binding contract")
    require(not root_list_binding_argument.is_symlink(), "root/list binding symlink")
    binding_path = root_list_binding_argument.resolve(strict=True)
    require(binding_path.as_posix() == binding_contract.get("absolute_path"), "binding absolute path")
    require(stat.S_IMODE(binding_path.stat().st_mode) == int(str(binding_contract.get("required_mode")), 8), "binding mode")
    require(_sha256(binding_path) == binding_contract.get("sha256"), "binding hash")
    binding = _json(binding_path, "root/list binding")
    require(binding.get("schema_version") == binding_contract.get("required_schema_version"), "binding schema")
    require(binding.get("status") == binding_contract.get("required_status"), "binding status")
    entries = _expected_entries(binding)
    require(len(entries) == binding_contract.get("artifact_count"), "binding artifact count")
    require(
        sum(item["expected_byte_size"] for item in entries)
        == binding_contract.get("artifact_bytes_total"),
        "binding artifact bytes",
    )

    verification_contract = _mapping(
        bound.get("independent_binding_verification"), "binding verification contract"
    )
    require(not root_list_binding_verification_argument.is_symlink(), "binding verification symlink")
    binding_verification_path = root_list_binding_verification_argument.resolve(strict=True)
    require(
        binding_verification_path.as_posix() == verification_contract.get("absolute_path"),
        "binding verification path",
    )
    require(
        stat.S_IMODE(binding_verification_path.stat().st_mode)
        == int(str(verification_contract.get("required_mode")), 8),
        "binding verification mode",
    )
    require(_sha256(binding_verification_path) == verification_contract.get("sha256"), "binding verification hash")
    binding_verification = _json(binding_verification_path, "binding verification")
    require(
        binding_verification.get("schema_version")
        == verification_contract.get("required_schema_version"),
        "binding verification schema",
    )
    require(
        binding_verification.get("status") == verification_contract.get("required_status"),
        "binding verification status",
    )
    require(binding_verification.get("binding_sha256") == binding_contract.get("sha256"), "verified binding hash")
    require(binding_verification.get("artifact_count_verified") == len(entries), "verified binding count")
    require(binding_verification.get("source_artifact_bytes_hashed") == 0, "binding hash boundary")
    require(binding_verification.get("holdout_preflight_complete") is False, "binding preflight boundary")
    require(binding_verification.get("pass_f") is False, "binding F boundary")
    require(binding_verification.get("pass_g0") is False, "binding G0 boundary")
    require(binding_verification.get("g4_open") is False, "binding G4 boundary")
    return {
        "binding_path": binding_path,
        "binding_verification_path": binding_verification_path,
        "binding": binding,
        "entries": entries,
    }


def _safe_source_file(root: Path, relative: str) -> Path:
    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise G4PrivateExplicitListPreflightVerificationError(
                f"missing artifact {relative!r}"
            ) from error
        require(not stat.S_ISLNK(metadata.st_mode), f"symlink artifact {relative!r}")
        if index < len(parts) - 1:
            require(stat.S_ISDIR(metadata.st_mode), f"non-directory artifact parent {relative!r}")
        else:
            require(stat.S_ISREG(metadata.st_mode), f"nonregular artifact {relative!r}")
    return current


def _rehash(root: Path, entry: Mapping[str, Any]) -> int:
    relative = str(entry["canonical_relative_path"])
    path = _safe_source_file(root, relative)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"nonregular opened artifact {relative!r}")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"artifact changed during independent preflight verification {relative!r}",
        )
    finally:
        os.close(descriptor)
    require(total == entry["expected_byte_size"], f"independent size mismatch {relative!r}")
    require(digest.hexdigest() == entry["expected_sha256"], f"independent hash mismatch {relative!r}")
    return total


def _write_json_create_only(path: Path, value: Mapping[str, Any], mode: int) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "invalid operator control root")
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
    root_list_binding_argument: Path,
    root_list_binding_verification_argument: Path,
    preflight_argument: Path,
    verification_output: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path, "preflight config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    repository = _validate_repository_config(config)
    controls = _validate_bound_controls(
        config,
        root_list_binding_argument,
        root_list_binding_verification_argument,
    )
    output_contract = _mapping(config.get("output_contract"), "output contract")
    compatibility = _mapping(config.get("seal_creator_compatibility"), "seal compatibility")

    bound = _mapping(config.get("bound_root_and_list"), "bound root/list")
    root_contract = _mapping(bound.get("exact_source_root"), "source root contract")
    require(not source_root_argument.is_symlink(), "source root symlink")
    source_root = source_root_argument.resolve(strict=True)
    require(source_root.is_dir(), "invalid source root")
    require(source_root.as_posix() == root_contract.get("absolute_path"), "source root absolute path")
    require(source_root.name == root_contract.get("basename"), "source root basename")
    root_path_sha = hashlib.sha256(source_root.as_posix().encode("utf-8")).hexdigest()
    require(root_path_sha == root_contract.get("resolved_path_sha256"), "source root path hash")
    root_metadata = source_root.lstat()
    require(root_metadata.st_dev == root_contract.get("device"), "source root device")
    require(root_metadata.st_ino == root_contract.get("inode"), "source root inode")
    require(f"{stat.S_IMODE(root_metadata.st_mode):04o}" == root_contract.get("mode"), "source root mode")

    require(not preflight_argument.is_symlink(), "preflight symlink")
    preflight_path = preflight_argument.resolve(strict=True)
    require(preflight_path.parent.as_posix() == output_contract.get("operator_control_root"), "preflight operator root")
    require(preflight_path.name == output_contract.get("preflight_basename"), "preflight basename")
    require(stat.S_IMODE(preflight_path.stat().st_mode) == int(str(output_contract.get("output_file_mode")), 8), "preflight mode")
    preflight = _json(preflight_path, "preflight")
    require(preflight.get("schema_version") == PREFLIGHT_SCHEMA, "preflight schema")
    require(preflight.get("status") == PREFLIGHT_STATUS, "preflight status")
    require(preflight.get("component_schema_version") == COMPONENT_SCHEMA, "preflight component schema")
    require(preflight.get("component_status") == COMPONENT_STATUS, "preflight component status")
    require(preflight.get("partition_contract") == compatibility.get("partition_contract"), "preflight partition")
    require(preflight.get("source_root_basename") == source_root.name, "preflight root basename")
    require(preflight.get("source_root_resolved_path_sha256") == root_path_sha, "preflight root path hash")
    require(preflight.get("source_root_device") == root_metadata.st_dev, "preflight root device")
    require(preflight.get("source_root_inode") == root_metadata.st_ino, "preflight root inode")

    contract_binding = _mapping(preflight.get("preflight_contract_binding"), "preflight contract binding")
    require(
        contract_binding
        == {
            "relative_path": config_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(config_path),
        },
        "preflight contract binding",
    )
    author_binding = _mapping(preflight.get("preflight_author_binding"), "preflight author binding")
    author_path = _safe_repo_path(author_binding.get("relative_path"), "preflight author")
    require(author_binding.get("sha256") == _sha256(author_path), "preflight author hash")
    require(preflight.get("authorization_parent_binding") == repository["parent"], "preflight parent binding")
    require(preflight.get("root_list_contract_binding") == repository["root_list_contract"], "preflight root/list contract")
    require(
        preflight.get("root_list_binding")
        == {
            "absolute_path": controls["binding_path"].as_posix(),
            "sha256": _sha256(controls["binding_path"]),
        },
        "preflight root/list binding",
    )
    require(
        preflight.get("root_list_binding_verification")
        == {
            "absolute_path": controls["binding_verification_path"].as_posix(),
            "sha256": _sha256(controls["binding_verification_path"]),
        },
        "preflight root/list verification",
    )
    for key, value in repository["seal_repository_bindings"].items():
        require(preflight.get(f"{key}_binding") == value, f"preflight seal binding {key}")
    require(preflight.get("seal_consumer_bindings") == repository["seal_consumer_bindings"], "preflight consumers")

    preflight_entries = preflight.get("artifact_entries")
    require(isinstance(preflight_entries, list), "preflight artifact entries")
    require(preflight_entries == controls["entries"], "preflight exact bound entry projection")
    require(preflight.get("artifact_count") == len(controls["entries"]), "preflight artifact count")
    expected_bytes = sum(item["expected_byte_size"] for item in controls["entries"])
    require(preflight.get("artifact_bytes_total") == expected_bytes, "preflight artifact bytes")
    binding_contract = _mapping(bound.get("binding"), "binding contract")
    require(preflight.get("artifact_tree_sha256") == binding_contract.get("artifact_tree_sha256"), "preflight tree hash")
    fingerprint = preflight.get("preflight_fingerprint_sha256")
    require(isinstance(fingerprint, str) and len(fingerprint) == 64, "preflight fingerprint")
    core = dict(preflight)
    del core["preflight_fingerprint_sha256"]
    require(hashlib.sha256(_canonical_bytes(core)).hexdigest() == fingerprint, "preflight fingerprint mismatch")

    attestations = _mapping(preflight.get("attestations"), "preflight attestations")
    for key in (
        "explicit_artifact_set_complete",
        "paths_enumerated_under_v54_authorization",
        "hashes_computed_under_v54_authorization",
        "source_root_private_and_access_controlled",
        "final_operator_path_clear",
        "receipt_path_clear",
        "exact_bound_root_and_list_consumed",
        "every_bound_artifact_byte_hashed",
        "all_computed_sizes_and_hashes_match_binding",
    ):
        require(attestations.get(key) is True, f"preflight attestation {key}")
    require(attestations.get("implicit_artifact_discovery_performed") is False, "discovery attestation")
    require(attestations.get("row_parsing_or_analysis_performed") is False, "row parsing attestation")
    require(attestations.get("prior_open_event_count") == 0, "prior open event attestation")
    require(attestations.get("confirmatory_or_holdout_rows_read") == 0, "row read attestation")
    access = _mapping(preflight.get("access_accounting"), "preflight access accounting")
    require(access.get("bound_artifact_paths_consumed") == len(controls["entries"]), "path accounting")
    require(access.get("source_artifact_content_files_opened") == len(controls["entries"]), "file open accounting")
    require(access.get("source_artifact_bytes_hashed") == expected_bytes, "byte hash accounting")
    require(access.get("operator_parquet_files_parsed") == 0, "Parquet parsing accounting")
    require(access.get("confirmatory_or_holdout_analytic_rows_read") == 0, "analytic read accounting")
    require(access.get("preflight_written") is True, "preflight write accounting")
    require(access.get("byte_seal_created") is False, "seal accounting")
    completion = _mapping(preflight.get("completion"), "preflight completion")
    require(completion.get("this_component_is_new_gate") is False, "new gate boundary")
    require(completion.get("holdout_preflight_complete") is True, "preflight completion")
    require(completion.get("independent_preflight_verification_complete") is False, "verification boundary")
    require(completion.get("holdout_byte_seal_complete") is False, "seal completion boundary")
    require(completion.get("entry_or_persistence_labels_materialized") is False, "label boundary")
    require(completion.get("confirmatory_or_holdout_analytic_rows_read") == 0, "analytic boundary")
    require(completion.get("model_score_read") is False and completion.get("model_fitted") is False, "model boundary")
    require(completion.get("pass_f") is False and completion.get("pass_g0") is False, "F/G0 boundary")
    require(completion.get("g4_open") is False, "G4 boundary")
    require(preflight.get("authorization") == config.get("authorization_after_preflight_pass"), "preflight authorization")

    independently_hashed_bytes = sum(_rehash(source_root, entry) for entry in controls["entries"])
    require(independently_hashed_bytes == expected_bytes, "independent hashed byte total")
    verification_output = verification_output.resolve()
    require(verification_output.parent.as_posix() == output_contract.get("operator_control_root"), "verification operator root")
    require(verification_output.name == output_contract.get("verification_basename"), "verification basename")
    result = {
        "schema_version": output_contract["verification_schema_version"],
        "status": output_contract["verification_status"],
        "preflight_basename": preflight_path.name,
        "preflight_sha256": _sha256(preflight_path),
        "preflight_fingerprint_sha256": fingerprint,
        "source_root_basename": source_root.name,
        "source_root_resolved_path_sha256": root_path_sha,
        "source_root_device": root_metadata.st_dev,
        "source_root_inode": root_metadata.st_ino,
        "root_list_binding_sha256": _sha256(controls["binding_path"]),
        "root_list_binding_verification_sha256": _sha256(
            controls["binding_verification_path"]
        ),
        "artifact_count_verified": len(controls["entries"]),
        "artifact_bytes_total_verified": expected_bytes,
        "artifact_tree_sha256_verified": binding_contract["artifact_tree_sha256"],
        "explicit_list_exact_bound_projection_verified": True,
        "every_bound_artifact_opened_and_rehashed": True,
        "artifact_size_mismatch_count": 0,
        "artifact_sha256_mismatch_count": 0,
        "artifact_changed_during_hash_count": 0,
        "source_artifact_content_files_opened": len(controls["entries"]),
        "source_artifact_bytes_hashed": independently_hashed_bytes,
        "operator_parquet_files_parsed": 0,
        "confirmatory_or_holdout_analytic_rows_read": 0,
        "seal_creator_preflight_schema_compatible": True,
        "exact_private_source_root_bound": True,
        "complete_explicit_artifact_list_bound": True,
        "holdout_preflight_complete": True,
        "independent_preflight_verification_complete": True,
        "holdout_byte_seal_complete": False,
        "entry_or_persistence_labels_materialized": False,
        "model_score_read": False,
        "model_fitted": False,
        "g4_open": False,
        "pass_f": False,
        "pass_g0": False,
        "authorization": config["authorization_after_independent_verification_pass"],
    }
    require(result["schema_version"] == VERIFY_SCHEMA, "verification schema contract")
    require(result["status"] == VERIFY_STATUS, "verification status contract")
    mode = int(str(output_contract.get("output_file_mode")), 8)
    _write_json_create_only(verification_output, result, mode)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root-list-binding", type=Path, required=True)
    parser.add_argument("--root-list-binding-verification", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        source_root_argument=args.source_root,
        root_list_binding_argument=args.root_list_binding,
        root_list_binding_verification_argument=args.root_list_binding_verification,
        preflight_argument=args.preflight,
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
