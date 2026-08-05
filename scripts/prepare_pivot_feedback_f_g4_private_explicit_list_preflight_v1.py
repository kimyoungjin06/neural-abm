#!/usr/bin/env python3
"""Hash the bound private G4 artifact list into a seal-compatible preflight."""

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
    / "experiments/configs/pivot_feedback_f_g4_private_explicit_list_preflight_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_f_g4_private_explicit_list_preflight.v1"
PREFLIGHT_SCHEMA = "pivot_feedback_holdout_access_seal_preflight.v1"
PREFLIGHT_STATUS = "PASS_EXPLICIT_ALLOWLIST_HASHES_BOUND_ZERO_PRIOR_OPEN_EVENTS"
COMPONENT_SCHEMA = "pivot_feedback_f_g4_private_explicit_list_preflight_output.v1"
COMPONENT_STATUS = (
    "PASS_OPERATOR_ONLY_BOUND_153_ARTIFACT_BYTES_REHASHED_NO_SEAL_NO_F_PASS"
)


class G4PrivateExplicitListPreflightError(RuntimeError):
    """Fail-closed exact-list full-byte preflight error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G4PrivateExplicitListPreflightError(message)


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
        raise G4PrivateExplicitListPreflightError(f"invalid {label}: {error}") from error


def _yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise G4PrivateExplicitListPreflightError(f"invalid {label}: {error}") from error


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


def _validate_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise G4PrivateExplicitListPreflightError("invalid preflight timestamp") from error
    require(parsed.tzinfo is not None, "preflight timestamp timezone")
    require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        "preflight timestamp must be UTC",
    )
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
    repository_bindings = _mapping(
        compatibility.get("repository_bindings"), "seal repository bindings"
    )
    validated: dict[str, dict[str, str]] = {}
    for key, raw in repository_bindings.items():
        binding = _mapping(raw, f"seal binding {key}")
        path = _safe_repo_path(binding.get("relative_path"), f"seal binding {key}")
        require(_sha256(path) == binding.get("sha256"), f"seal binding {key} hash")
        document = _yaml(path, f"seal binding {key}")
        require(
            document.get("schema_version") == binding.get("required_schema_version"),
            f"seal binding {key} schema",
        )
        require(document.get("status") == binding.get("required_status"), f"seal binding {key} status")
        validated[key] = {
            "relative_path": binding["relative_path"],
            "sha256": binding["sha256"],
        }

    consumers = _mapping(compatibility.get("consumer_scripts"), "consumer scripts")
    consumer_bindings: dict[str, dict[str, str]] = {}
    for key, raw in consumers.items():
        binding = _mapping(raw, f"consumer {key}")
        path = _safe_repo_path(binding.get("relative_path"), f"consumer {key}")
        require(_sha256(path) == binding.get("sha256"), f"consumer {key} hash")
        consumer_bindings[key] = {
            "relative_path": binding["relative_path"],
            "sha256": binding["sha256"],
        }
    return {
        "authorization_parent_relative_path": parent["relative_path"],
        "authorization_parent_sha256": parent["sha256"],
        "root_list_contract_relative_path": contract["relative_path"],
        "root_list_contract_sha256": contract["sha256"],
        "seal_repository_bindings": validated,
        "seal_consumer_bindings": consumer_bindings,
    }


def _binding_entries(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = binding.get("artifact_entries")
    require(isinstance(values, list), "bound artifact list")
    entries: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        row = _mapping(value, f"bound artifact {index}")
        require(
            set(row)
            == {
                "canonical_relative_path",
                "expected_byte_size",
                "expected_sha256",
                "expected_rows",
            },
            "bound artifact fields",
        )
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
    require(paths == sorted(paths), "bound artifact list order")
    require(len(paths) == len(set(paths)), "duplicate bound artifact path")
    return entries


def _validate_bound_inputs(
    config: Mapping[str, Any],
    source_root_argument: Path,
    binding_argument: Path,
    binding_verification_argument: Path,
) -> dict[str, Any]:
    bound = _mapping(config.get("bound_root_and_list"), "bound root/list")
    binding_contract = _mapping(bound.get("binding"), "binding contract")
    require(not binding_argument.is_symlink(), "root/list binding symlink")
    binding_path = binding_argument.resolve(strict=True)
    require(binding_path.as_posix() == binding_contract.get("absolute_path"), "binding absolute path")
    require(stat.S_IMODE(binding_path.stat().st_mode) == int(str(binding_contract.get("required_mode")), 8), "binding mode")
    require(_sha256(binding_path) == binding_contract.get("sha256"), "binding hash")
    binding = _json(binding_path, "root/list binding")
    require(binding.get("schema_version") == binding_contract.get("required_schema_version"), "binding schema")
    require(binding.get("status") == binding_contract.get("required_status"), "binding status")
    require(binding.get("artifact_count") == binding_contract.get("artifact_count"), "binding artifact count")
    require(binding.get("artifact_bytes_total") == binding_contract.get("artifact_bytes_total"), "binding byte total")
    evidence = _mapping(binding.get("production_evidence_binding"), "binding production evidence")
    require(evidence.get("artifact_tree_sha256") == binding_contract.get("artifact_tree_sha256"), "binding tree hash")
    assertions = _mapping(binding.get("binding_assertions"), "binding assertions")
    require(assertions.get("exact_private_source_root_bound") is True, "root bound assertion")
    require(assertions.get("complete_explicit_artifact_list_bound") is True, "list bound assertion")
    require(assertions.get("explicit_list_exact_receipt_projection") is True, "receipt projection assertion")
    entries = _binding_entries(binding)
    require(len(entries) == binding_contract.get("artifact_count"), "bound list count")
    require(
        sum(item["expected_byte_size"] for item in entries)
        == binding_contract.get("artifact_bytes_total"),
        "bound list bytes",
    )

    verification_contract = _mapping(
        bound.get("independent_binding_verification"), "binding verification contract"
    )
    require(not binding_verification_argument.is_symlink(), "binding verification symlink")
    verification_path = binding_verification_argument.resolve(strict=True)
    require(
        verification_path.as_posix() == verification_contract.get("absolute_path"),
        "binding verification absolute path",
    )
    require(
        stat.S_IMODE(verification_path.stat().st_mode)
        == int(str(verification_contract.get("required_mode")), 8),
        "binding verification mode",
    )
    require(_sha256(verification_path) == verification_contract.get("sha256"), "binding verification hash")
    verification = _json(verification_path, "binding verification")
    require(
        verification.get("schema_version") == verification_contract.get("required_schema_version"),
        "binding verification schema",
    )
    require(verification.get("status") == verification_contract.get("required_status"), "binding verification status")
    require(verification.get("binding_sha256") == binding_contract.get("sha256"), "verified binding hash")
    require(
        verification.get("artifact_count_verified")
        == verification_contract.get("required_artifact_count"),
        "verified artifact count",
    )
    require(verification.get("source_artifact_bytes_hashed") == 0, "binding-stage hash boundary")
    require(verification.get("holdout_preflight_complete") is False, "binding-stage preflight boundary")
    require(verification.get("pass_f") is False and verification.get("pass_g0") is False, "binding F/G0 boundary")
    require(verification.get("g4_open") is False, "binding G4 boundary")

    root_contract = _mapping(bound.get("exact_source_root"), "source root contract")
    require(not source_root_argument.is_symlink(), "source root symlink")
    source_root = source_root_argument.resolve(strict=True)
    require(source_root.is_dir(), "invalid source root")
    require(source_root.as_posix() == root_contract.get("absolute_path"), "source root absolute path")
    require(source_root.name == root_contract.get("basename"), "source root basename")
    path_sha = hashlib.sha256(source_root.as_posix().encode("utf-8")).hexdigest()
    require(path_sha == root_contract.get("resolved_path_sha256"), "source root path hash")
    root_metadata = source_root.lstat()
    require(root_metadata.st_dev == root_contract.get("device"), "source root device")
    require(root_metadata.st_ino == root_contract.get("inode"), "source root inode")
    require(f"{stat.S_IMODE(root_metadata.st_mode):04o}" == root_contract.get("mode"), "source root mode")
    binding_root = _mapping(binding.get("source_root_binding"), "binding source root")
    require(binding_root.get("device") == root_metadata.st_dev, "binding root device")
    require(binding_root.get("inode") == root_metadata.st_ino, "binding root inode")
    return {
        "binding_path": binding_path,
        "binding_verification_path": verification_path,
        "source_root": source_root,
        "root_metadata": root_metadata,
        "entries": entries,
        "artifact_tree_sha256": binding_contract["artifact_tree_sha256"],
    }


def _safe_source_file(root: Path, relative: str) -> Path:
    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise G4PrivateExplicitListPreflightError(
                f"missing artifact {relative!r}"
            ) from error
        require(not stat.S_ISLNK(metadata.st_mode), f"symlink artifact {relative!r}")
        if index < len(parts) - 1:
            require(stat.S_ISDIR(metadata.st_mode), f"non-directory artifact parent {relative!r}")
        else:
            require(stat.S_ISREG(metadata.st_mode), f"nonregular artifact {relative!r}")
    return current


def _hash_bound_artifact(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
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
            f"artifact changed during preflight {relative!r}",
        )
    finally:
        os.close(descriptor)
    require(total == entry["expected_byte_size"], f"artifact size drift {relative!r}")
    actual_sha = digest.hexdigest()
    require(actual_sha == entry["expected_sha256"], f"artifact hash drift {relative!r}")
    return dict(entry)


def _write_json_create_only(path: Path, value: Mapping[str, Any], mode: int) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "invalid operator control root")
    require(not path.exists() and not path.is_symlink(), "preflight output exists")
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
            require(written > 0, "short preflight write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare(
    *,
    config_path: Path,
    source_root: Path,
    root_list_binding: Path,
    root_list_binding_verification: Path,
    output_path: Path,
    authorized_at_utc: str,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path, "preflight config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    repository = _validate_repository_bindings(config)
    bound = _validate_bound_inputs(
        config,
        source_root,
        root_list_binding,
        root_list_binding_verification,
    )
    authorized_at_utc = _validate_timestamp(authorized_at_utc)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    output_path = output_path.resolve()
    require(output_path.parent.as_posix() == output_contract.get("operator_control_root"), "operator control root")
    require(output_path.name == output_contract.get("preflight_basename"), "preflight basename")
    require(not output_path.is_relative_to(bound["source_root"]), "preflight inside source root")
    require(
        stat.S_IMODE(output_path.parent.stat().st_mode)
        == int(str(output_contract.get("operator_control_directory_mode")), 8),
        "operator control directory mode",
    )

    entries = [_hash_bound_artifact(bound["source_root"], entry) for entry in bound["entries"]]
    execution = _mapping(config.get("preflight_execution_contract"), "preflight execution")
    require(len(entries) == execution.get("artifact_count"), "preflight artifact count")
    byte_total = sum(item["expected_byte_size"] for item in entries)
    require(byte_total == execution.get("artifact_bytes_total"), "preflight byte total")
    compatibility = _mapping(config.get("seal_creator_compatibility"), "seal compatibility")
    require(compatibility.get("output_schema_version") == PREFLIGHT_SCHEMA, "preflight schema contract")
    require(compatibility.get("output_status") == PREFLIGHT_STATUS, "preflight status contract")
    require(compatibility.get("component_schema_version") == COMPONENT_SCHEMA, "component schema contract")
    require(compatibility.get("component_status") == COMPONENT_STATUS, "component status contract")

    core: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "component_schema_version": COMPONENT_SCHEMA,
        "component_status": COMPONENT_STATUS,
        "authorized_at_utc": authorized_at_utc,
        "preflight_contract_binding": {
            "relative_path": config_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(config_path),
        },
        "preflight_author_binding": {
            "relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "authorization_parent_binding": {
            "relative_path": repository["authorization_parent_relative_path"],
            "sha256": repository["authorization_parent_sha256"],
        },
        "root_list_contract_binding": {
            "relative_path": repository["root_list_contract_relative_path"],
            "sha256": repository["root_list_contract_sha256"],
        },
        "root_list_binding": {
            "absolute_path": bound["binding_path"].as_posix(),
            "sha256": _sha256(bound["binding_path"]),
        },
        "root_list_binding_verification": {
            "absolute_path": bound["binding_verification_path"].as_posix(),
            "sha256": _sha256(bound["binding_verification_path"]),
        },
        **{
            f"{key}_binding": value
            for key, value in repository["seal_repository_bindings"].items()
        },
        "seal_consumer_bindings": repository["seal_consumer_bindings"],
        "partition_contract": compatibility["partition_contract"],
        "source_root_basename": bound["source_root"].name,
        "source_root_resolved_path_sha256": hashlib.sha256(
            bound["source_root"].as_posix().encode("utf-8")
        ).hexdigest(),
        "source_root_device": bound["root_metadata"].st_dev,
        "source_root_inode": bound["root_metadata"].st_ino,
        "artifact_count": len(entries),
        "artifact_bytes_total": byte_total,
        "artifact_tree_sha256": bound["artifact_tree_sha256"],
        "artifact_entries": entries,
        "attestations": {
            "explicit_artifact_set_complete": True,
            "paths_enumerated_under_v54_authorization": True,
            "hashes_computed_under_v54_authorization": True,
            "source_root_private_and_access_controlled": True,
            "final_operator_path_clear": True,
            "receipt_path_clear": True,
            "exact_bound_root_and_list_consumed": True,
            "every_bound_artifact_byte_hashed": True,
            "all_computed_sizes_and_hashes_match_binding": True,
            "implicit_artifact_discovery_performed": False,
            "row_parsing_or_analysis_performed": False,
            "prior_open_event_count": 0,
            "confirmatory_or_holdout_rows_read": 0,
        },
        "access_accounting": {
            "bound_artifact_paths_consumed": len(entries),
            "source_artifact_content_files_opened": len(entries),
            "source_artifact_bytes_hashed": byte_total,
            "operator_parquet_files_parsed": 0,
            "confirmatory_or_holdout_analytic_rows_read": 0,
            "preflight_written": True,
            "byte_seal_created": False,
        },
        "completion": {
            "this_component_is_new_gate": False,
            "exact_private_source_root_bound": True,
            "complete_explicit_artifact_list_bound": True,
            "holdout_preflight_complete": True,
            "independent_preflight_verification_complete": False,
            "holdout_byte_seal_complete": False,
            "entry_or_persistence_labels_materialized": False,
            "confirmatory_or_holdout_analytic_rows_read": 0,
            "model_score_read": False,
            "model_fitted": False,
            "g4_open": False,
            "pass_f": False,
            "pass_g0": False,
        },
        "authorization": config["authorization_after_preflight_pass"],
    }
    payload = {
        **core,
        "preflight_fingerprint_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
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
    parser.add_argument("--root-list-binding", type=Path, required=True)
    parser.add_argument("--root-list-binding-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorized-at-utc", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare(
        config_path=args.config,
        source_root=args.source_root,
        root_list_binding=args.root_list_binding,
        root_list_binding_verification=args.root_list_binding_verification,
        output_path=args.output,
        authorized_at_utc=args.authorized_at_utc or _utc_now(),
    )
    safe = {
        "schema_version": result["schema_version"],
        "status": result["status"],
        "component_status": result["component_status"],
        "source_root_basename": result["source_root_basename"],
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
