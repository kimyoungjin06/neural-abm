#!/usr/bin/env python3
"""Issue the bounded G-1A receipt from exact repository evidence.

This verifier is deliberately metadata-only.  It does not import any producer,
open source payloads, or mutate a production output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "pivot_feedback_g_minus_1a_receipt_config.v1"
RECEIPT_SCHEMA = "pivot_feedback_g_minus_1a_receipt.v1"
PASS_STATUS = (
    "PASS_G_MINUS_1A_BOUNDED_PRIMITIVE_AND_SOURCE_FIDELITY_"
    "WITH_CURRENT_SNAPSHOT_PROXY_LIMITATIONS"
)
EXPECTED_COMPONENTS = [
    "canonical_work_time_spine",
    "author_token_history",
    "work_topic_relation",
    "citation_chronology_proxy",
    "cutoff_primitives",
    "cutoff_source_projection_replay",
]
EXPECTED_UNRESOLVED = {
    "author_entity_validity": "UNRESOLVED_CURRENT_SNAPSHOT_TOKEN_ONLY",
    "historical_author_observability": "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT",
    "historical_topic_observability": "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT",
    "historical_citation_observation_time": "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT",
    "raw_source_id_validity": "BLOCKED_W_MINUS_1_QUARANTINED",
    "w_zero_entity_validity": "UNRESOLVED",
    "candidate_eligibility": "PENDING_G0",
    "allowed_work_types": "PENDING_G0",
    "fixed_outcome_horizon": "PENDING_G0",
    "final_cutoff_panel": "PENDING_G0_5",
    "s0_threshold_free_inventory": "NOT_RUN",
    "d0_discovery_pilot": "NOT_RUN",
    "empirical_g1_signature": "NOT_RUN",
    "scientific_effect": "NOT_ESTABLISHED",
}
MAX_CONFIG_BYTES = 1_048_576
MAX_EVIDENCE_BYTES = 4_194_304


class GMinus1AError(RuntimeError):
    """Fail-closed G-1A validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GMinus1AError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _safe_relative(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a non-empty string")
    pure = PurePosixPath(value)
    require(not pure.is_absolute(), f"{label} must be relative")
    require(".." not in pure.parts, f"{label} may not traverse parents")
    require(str(pure) == value, f"{label} must be normalized POSIX")
    return value


def _repository_file(
    repo_root: Path,
    relative: str,
    *,
    label: str,
    maximum_bytes: int,
) -> Path:
    path = repo_root / _safe_relative(relative, label)
    require(path.is_file(), f"missing {label}: {relative}")
    require(not path.is_symlink(), f"{label} may not be a symlink: {relative}")
    resolved = path.resolve(strict=True)
    root = repo_root.resolve(strict=True)
    require(resolved.is_relative_to(root), f"{label} escapes repository: {relative}")
    require(resolved.stat().st_size <= maximum_bytes, f"{label} is too large: {relative}")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GMinus1AError(f"invalid {label}: {error}") from error
    return _mapping(value, label)


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise GMinus1AError(f"invalid {label}: {error}") from error
    return _mapping(value, label)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(isinstance(current, Mapping) and part in current, f"missing field {dotted_path}")
        current = current[part]
    return current


def _validate_bound_document(
    binding: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
    yaml_document: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = _safe_relative(binding.get("relative_path"), f"{label} path")
    path = _repository_file(
        repo_root,
        relative,
        label=label,
        maximum_bytes=MAX_EVIDENCE_BYTES,
    )
    observed_sha = _sha256_file(path)
    require(observed_sha == binding.get("sha256"), f"{label} SHA-256 mismatch")
    document = _read_yaml(path, label) if yaml_document else _read_json(path, label)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        f"{label} schema mismatch",
    )
    require(document.get("status") == binding.get("required_status"), f"{label} status mismatch")
    required_fields = _mapping(binding.get("required_fields"), f"{label} required fields")
    for dotted_path, expected in required_fields.items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"{label} field mismatch: {dotted_path}",
        )
    return document, {
        "component_id": binding.get("component_id"),
        "relative_path": relative,
        "sha256": observed_sha,
        "schema_version": document["schema_version"],
        "status": document["status"],
        "required_field_count": len(required_fields),
        "verification": "PASS_EXACT_REPOSITORY_EVIDENCE",
    }


def _validate_closures(
    declared: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    canonical = evidence["canonical_work_time_spine"]
    author = evidence["author_token_history"]
    topic = evidence["work_topic_relation"]
    citation = evidence["citation_chronology_proxy"]
    cutoff = evidence["cutoff_primitives"]
    replay = evidence["cutoff_source_projection_replay"]

    observed = {
        "retained_work_rows": canonical["identifier_accounting"]["retained_rows"],
        "canonical_receipt_sha256": canonical["provenance"]["output_receipt_sha256"],
        "author_history_receipt_sha256": author["production_surface"]["output_receipt_sha256"],
        "work_topic_receipt_sha256": topic["provenance"]["output_receipt_sha256"],
        "citation_chronology_receipt_sha256": citation["production_surface"][
            "output_receipt_sha256"
        ],
        "current_content_receipt_sha256": replay["parent_bindings"][
            "content_receipt_sha256"
        ],
        "work_metadata_tree_sha256": replay["output_verification"][
            "work_metadata_tree_sha256"
        ],
        "source_projection_record_sha256": replay["row_level_verification"][
            "source_projection_record_sha256"
        ],
        "source_binding_semantics": (
            "CONTENT_FREEZE_SHA256_PLUS_REPLAY_CURRENT_EXACT_STAT_"
            "WITHOUT_REHASHING_ALL_SOURCE_BYTES_IN_REPLAY"
        ),
    }
    require(dict(declared) == observed, "declared cross-component closures differ")

    retained = observed["retained_work_rows"]
    require(cutoff["source_accounting"]["retained_rows"] == retained, "cutoff row closure failed")
    require(replay["source_accounting"]["retained_rows"] == retained, "replay row closure failed")
    require(replay["row_level_verification"]["rows"] == retained, "replay comparison row closure failed")
    parents = cutoff["normalized_parent_surfaces"]
    require(
        parents["canonical_work_time_spine_receipt_sha256"]
        == observed["canonical_receipt_sha256"],
        "canonical receipt closure failed",
    )
    require(
        parents["author_history_receipt_sha256"]
        == observed["author_history_receipt_sha256"],
        "author receipt closure failed",
    )
    require(
        parents["work_topic_relation_receipt_sha256"]
        == observed["work_topic_receipt_sha256"],
        "topic receipt closure failed",
    )
    require(
        parents["citation_chronology_receipt_sha256"]
        == observed["citation_chronology_receipt_sha256"],
        "citation receipt closure failed",
    )
    require(
        replay["parent_bindings"]["canonical_receipt_sha256"]
        == observed["canonical_receipt_sha256"],
        "replay canonical receipt closure failed",
    )
    require(
        cutoff["work_metadata"]["work_metadata_tree_sha256"]
        == observed["work_metadata_tree_sha256"],
        "work metadata tree closure failed",
    )
    require(
        replay["row_level_verification"]["source_projection_record_sha256"]
        == replay["row_level_verification"]["output_projection_record_sha256"],
        "source/output projection digest closure failed",
    )
    return observed


def verify(config_path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(config_path.is_file() and not config_path.is_symlink(), "config must be a regular file")
    require(config_path.stat().st_size <= MAX_CONFIG_BYTES, "config is too large")
    config = _read_yaml(config_path, "G-1A config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(config.get("label") == "pivot_feedback_g_minus_1a_receipt_v1", "config label mismatch")
    require(config.get("mode") == "READ_ONLY_REPOSITORY_EVIDENCE_BINDING", "config mode mismatch")
    require(
        config.get("status_before_verification") == "READY_TO_VERIFY_BOUNDED_G_MINUS_1A",
        "config pre-verification status mismatch",
    )

    gate = _mapping(config.get("gate_definition"), "gate definition")
    require(gate.get("gate_id") == "G_MINUS_1A", "gate ID mismatch")
    require(gate.get("passing_status") == PASS_STATUS, "passing status mismatch")
    require(gate.get("may_authorize_s0") is True, "S0 is not authorized")
    for field in (
        "may_authorize_d0",
        "may_authorize_model_fitting",
        "may_pass_legacy_g_minus_1",
        "may_pass_g0",
        "may_establish_scientific_effect",
    ):
        require(gate.get(field) is False, f"unsafe gate permission: {field}")

    parent_binding = _mapping(config.get("acyclic_parent"), "acyclic parent")
    parent, parent_result = _validate_bound_document(
        parent_binding,
        repo_root=repo_root,
        label="acyclic parent",
        yaml_document=True,
    )
    nodes = {
        item["gate_id"]: item
        for item in _list(parent.get("gate_nodes"), "parent gate nodes")
    }
    require(nodes["G_MINUS_1A"]["requires"] == [], "G-1A parent dependency mismatch")
    require(nodes["S0"]["requires"] == ["G_MINUS_1A"], "S0 parent dependency mismatch")

    raw_bindings = [
        _mapping(item, "repository evidence binding")
        for item in _list(config.get("repository_evidence"), "repository evidence")
    ]
    component_ids = [str(item.get("component_id")) for item in raw_bindings]
    require(component_ids == EXPECTED_COMPONENTS, "repository evidence component order differs")
    documents: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for binding in raw_bindings:
        component_id = str(binding["component_id"])
        document, result = _validate_bound_document(
            binding,
            repo_root=repo_root,
            label=f"evidence {component_id}",
        )
        documents[component_id] = document
        results.append(result)

    closures = _validate_closures(
        _mapping(config.get("cross_component_closures"), "cross-component closures"),
        documents,
    )
    unresolved = _mapping(config.get("unresolved_boundaries"), "unresolved boundaries")
    require(unresolved == EXPECTED_UNRESOLVED, "unresolved-boundary registry differs")

    receipt_contract = _mapping(config.get("receipt_contract"), "receipt contract")
    require(receipt_contract.get("output_schema_version") == RECEIPT_SCHEMA, "receipt schema contract mismatch")
    require(receipt_contract.get("overwrite_allowed") is False, "receipt overwrite is allowed")
    require(receipt_contract.get("source_payloads_may_be_opened") is False, "source payload access is allowed")
    require(
        receipt_contract.get("production_outputs_may_be_mutated") is False,
        "production output mutation is allowed",
    )
    require(
        receipt_contract.get("g_minus_1a_pass_implies_s0_authorized_only") is True,
        "G-1A authorization boundary differs",
    )

    config_sha = _sha256_file(config_path)
    tool_sha = _sha256_file(Path(__file__).resolve())
    claim_boundary = config.get("claim_boundary")
    require(isinstance(claim_boundary, str) and len(claim_boundary) > 100, "claim boundary missing")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": PASS_STATUS,
        "gate_id": "G_MINUS_1A",
        "scope": {
            "full_repository_evidence_set": True,
            "source_payloads_opened": 0,
            "production_scans_started": 0,
            "production_outputs_mutated": False,
            "gpu_used": False,
        },
        "authorization": {
            "s0_threshold_free_inventory": "AUTHORIZED",
            "d0_discovery_pilot": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "legacy_g_minus_1": "NOT_PASSED",
            "g0_measurement_freeze": "PENDING",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "acyclic_parent": parent_result,
        "repository_evidence": results,
        "cross_component_closures": closures,
        "unresolved_boundaries": unresolved,
        "provenance": {
            "config_relative_path": str(config_path.relative_to(repo_root.resolve())),
            "config_sha256": config_sha,
            "verifier_relative_path": str(Path(__file__).resolve().relative_to(repo_root.resolve())),
            "verifier_sha256": tool_sha,
        },
        "claim_boundary": claim_boundary,
    }
    fingerprint_payload = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    receipt["provenance"]["result_fingerprint_sha256"] = _sha256_bytes(fingerprint_payload)
    return receipt


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"


def _write_new(path: Path, payload: bytes) -> None:
    path = path.resolve()
    require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "experiments/configs/pivot_feedback_g_minus_1a_receipt_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = verify(args.config)
        payload = _canonical_json(receipt)
        if args.output is not None:
            _write_new(args.output, payload)
        sys.stdout.buffer.write(payload)
    except (GMinus1AError, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED_G_MINUS_1A", "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
