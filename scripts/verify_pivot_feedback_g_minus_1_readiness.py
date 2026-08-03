#!/usr/bin/env python
"""Verify the fail-closed pivot-feedback G-1 readiness registry.

The verifier reads bounded repository evidence, optionally checks explicitly
supplied small operator receipts, and re-runs one fixed synthetic author-history
attestation. It has no production source argument and never emits a G-1 pass or
an execution-authorizing decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    REPO_ROOT / "experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml"
)
REGISTRY_SCHEMA = "pivot_feedback_g_minus_1_readiness.v1"
REPORT_SCHEMA = "pivot_feedback_g_minus_1_readiness_verification.v8"
PENDING_STATUS = "G_MINUS_1_PENDING"
HARD_MAX_REGISTRY_BYTES = 1_048_576
AUTHOR_VALIDATOR_TIMEOUT_SECONDS = 30
EXPECTED_REGISTRY_SHA256 = (
    "e5048af2d3cca3b1ff04bbecc1a6cbf5dcc6756bf29db995d3cefbe3a017f2be"
)

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "label",
    "mode",
    "status",
    "gate_state",
    "safety_boundary",
    "passed_repository_evidence",
    "blocking_repository_evidence",
    "passed_operator_control_receipts",
    "missing_required_components",
    "nonblocking_bounded_caveats",
    "claim_boundary",
}
EXPECTED_REPOSITORY_COMPONENTS = {
    "grammar_valid_main_id_view",
    "work_source_membership",
    "taxonomy_reference",
    "main_temporal_summary",
    "w0_positive_id_sensitivity",
    "canonical_full",
    "author_history",
    "citation_chronology",
    "work_topic_relation",
    "cutoff_primitives",
}
EXPECTED_BLOCKING_REPOSITORY_COMPONENTS: set[str] = set()
EXPECTED_OPERATOR_COMPONENTS = {
    "current_content_baseline",
    "main_temporal_v3",
}
EXPECTED_MISSING_COMPONENTS = {
    "cutoff_panel",
    "outcome_blind_field_structure",
    "representative_discovery_pilot",
}
EXPECTED_CAVEATS = {
    "raw_w_minus_1",
    "w0_entity_validity",
    "historical_inventory_byte_identity",
    "author_token_entity_and_past_observability",
    "citation_bibliographic_time_and_past_observability",
    "cutoff_primitives_not_final_panel",
}
EXPECTED_AUTHOR_SYNTHETIC_EVIDENCE = {
    "config_path": "experiments/configs/openalex_g1_work_side_author_history_v2.yaml",
    "config_sha256": "f0359f9b9ab7c2e5a51b1e760f94c70d51e544dc7f594d8e0512a9cc8ff216c2",
    "validator_path": "scripts/verify_openalex_work_side_author_history_fixture_v2.py",
    "validator_sha256": "6a95a633b67604da803f7405ef5ddfd20986b00518b4ebf9bae07dbffb9661de",
    "fixture_path": "tests/fixtures/openalex_work_side_author_history_v2_minimal.json",
    "fixture_sha256": "e77556ac457424193dd33cc9760b00b36aaba06fd207bd8cc1b0fb356cf101ff",
    "report_schema_version": (
        "openalex_g1_work_side_author_history_fixture_validation_report.v2"
    ),
    "pinned_report_status": (
        "PASS_SYNTHETIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_PINNED_FIXTURE_CONTRACT_ONLY"
    ),
    "pinned_evidence_class": "PINNED_CHECKED_FIXTURE",
    "pinned_parent_evidence_class": "CONFIG_PINNED_SYNTHETIC_PARENT_STUBS",
    "generic_report_status": (
        "VALID_UNATTESTED_GENERIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_MECHANICS_ONLY"
    ),
    "generic_evidence_class": "UNATTESTED_GENERIC_FIXTURE",
    "generic_parent_evidence_class": (
        "SELF_DECLARED_UNVERIFIED_SYNTHETIC_PARENT_STUBS"
    ),
    "attested_runtime_fingerprint": {
        "python_implementation": "CPython",
        "python_version": "3.14.0",
        "unicodedata_unidata_version": "16.0.0",
        "canonical_json_abi": (
            "openalex_g1_work_side_author_history_canonical_json.v2"
        ),
    },
    "attested_canonical_stdout_bytes": 6806,
    "attested_canonical_stdout_sha256": (
        "ee2ca45756f09b130f6ad00fe7b56219eaf9083429d80eb657424212c5381ef7"
    ),
    "runtime_mismatch_policy": (
        "NOT_REATTESTED_RUNTIME_MISMATCH_NO_CROSS_RUNTIME_BYTE_CLAIM"
    ),
    "foundation_record_count": 6,
    "production_status": "PRODUCTION_NOT_RUN",
    "production_source_interface_used": False,
    "production_input_accessed": False,
    "pinned_fixture_sha256_match": True,
    "production_parity_established": False,
    "author_entity_validity": "UNRESOLVED_WORK_SIDE_TOKEN_ONLY",
    "g_minus_1_data_readiness": "PENDING",
    "g0_measurement_freeze": "PENDING",
    "pinned_fixture_promotes_component": False,
    "generic_fixture_promotes_component": False,
}


class ReadinessError(RuntimeError):
    """Raised when the registry or bound control evidence fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be a list")
    return value


def _read_regular_bytes(path: Path, *, field: str, maximum_bytes: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReadinessError(f"{field} is unavailable") from exc
    _require(stat.S_ISREG(metadata.st_mode), f"{field} must be a regular file")
    _require(not path.is_symlink(), f"{field} must not be a symlink")
    _require(metadata.st_size <= maximum_bytes, f"{field} exceeds byte cap")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReadinessError(f"{field} cannot be read") from exc
    _require(len(payload) == metadata.st_size, f"{field} changed while read")
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReadinessError("control JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ReadinessError(f"control JSON contains non-finite constant: {value}")


def _json_object(payload: bytes, field: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"{field} is not canonical-readable JSON") from exc
    return _mapping(parsed, field)


def _canonical_json_line(payload: bytes, field: str) -> Mapping[str, Any]:
    _require(payload.endswith(b"\n"), f"{field} lacks one trailing LF")
    _require(payload.count(b"\n") == 1, f"{field} is not one canonical JSON line")
    document = _json_object(payload[:-1], field)
    try:
        canonical = (
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ReadinessError(f"{field} cannot be canonically serialized") from exc
    _require(payload == canonical, f"{field} bytes are not canonical JSON plus LF")
    return document


def _yaml_object(payload: bytes, field: str) -> Mapping[str, Any]:
    try:
        parsed = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReadinessError(f"{field} is invalid YAML") from exc
    return _mapping(parsed, field)


def _current_runtime_fingerprint() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "unicodedata_unidata_version": unicodedata.unidata_version,
        "canonical_json_abi": (
            "openalex_g1_work_side_author_history_canonical_json.v2"
        ),
    }


def _nested_value(document: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = document
    for segment in dotted_path.split("."):
        current = _mapping(current, dotted_path)
        _require(segment in current, f"required field is absent: {dotted_path}")
        current = current[segment]
    return current


def _same_scalar(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _validate_json_binding(
    payload: bytes,
    binding: Mapping[str, Any],
    *,
    field: str,
) -> None:
    expected_sha = binding.get("sha256")
    _require(
        isinstance(expected_sha, str) and len(expected_sha) == 64,
        f"{field} has an invalid SHA-256",
    )
    _require(_sha256(payload) == expected_sha, f"{field} SHA-256 mismatch")
    document = _json_object(payload, field)
    _require(
        document.get("schema_version") == binding.get("required_schema_version"),
        f"{field} schema mismatch",
    )
    _require(
        document.get("status") == binding.get("required_status"),
        f"{field} status mismatch",
    )
    required_fields = _mapping(binding.get("required_fields"), "required_fields")
    for dotted_path, expected in required_fields.items():
        _require(isinstance(dotted_path, str), "required field path must be text")
        actual = _nested_value(document, dotted_path)
        _require(
            _same_scalar(actual, expected),
            f"{field} required field differs: {dotted_path}",
        )


def _safe_repository_path(repo_root: Path, relative_path: Any) -> Path:
    _require(isinstance(relative_path, str), "repository path must be text")
    pure = PurePosixPath(relative_path)
    _require(not pure.is_absolute(), "repository path must be relative")
    _require(
        bool(pure.parts) and all(part not in {"", ".", ".."} for part in pure.parts),
        "repository path must be normalized and contained",
    )
    try:
        root = repo_root.resolve(strict=True)
        path = root.joinpath(*pure.parts)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ReadinessError("repository evidence is unavailable") from exc
    _require(resolved.is_relative_to(root), "repository evidence escapes root")
    return resolved


def _component_ids(items: list[Any], *, id_field: str) -> set[str]:
    identifiers: list[str] = []
    for index, raw in enumerate(items):
        item = _mapping(raw, f"item {index}")
        identifier = item.get(id_field)
        _require(isinstance(identifier, str), f"{id_field} must be text")
        identifiers.append(identifier)
    _require(len(identifiers) == len(set(identifiers)), f"duplicate {id_field}")
    return set(identifiers)


def _load_registry(path: Path) -> tuple[Mapping[str, Any], bytes]:
    payload = _read_regular_bytes(
        path, field="G-1 readiness registry", maximum_bytes=HARD_MAX_REGISTRY_BYTES
    )
    _require(
        _sha256(payload) == EXPECTED_REGISTRY_SHA256,
        "versioned G-1 readiness registry file SHA-256 mismatch",
    )
    try:
        raw = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReadinessError("G-1 readiness registry is invalid YAML") from exc
    registry = _mapping(raw, "G-1 readiness registry")
    _require(set(registry) == EXPECTED_TOP_LEVEL_KEYS, "registry key set differs")
    _require(registry.get("schema_version") == REGISTRY_SCHEMA, "schema differs")
    return registry, payload


def _validate_closed_contract(registry: Mapping[str, Any], registry_bytes: int) -> None:
    _require(registry.get("status") == PENDING_STATUS, "registry is not pending")
    _require(
        registry.get("mode") == "READ_ONLY_FAIL_CLOSED_GATE_REGISTRY",
        "registry mode differs",
    )
    gate = _mapping(registry.get("gate_state"), "gate_state")
    _require(
        gate
        == {
            "g_minus_1_data_readiness": "PENDING",
            "pass_g_minus_1": False,
            "g0_measurement_freeze": "PENDING",
            "empirical_g1_signature": "NOT_RUN",
            "pass_g1": False,
            "execution_authorized": False,
        },
        "gate state is not exactly fail-closed",
    )
    safety = _mapping(registry.get("safety_boundary"), "safety_boundary")
    _require(safety.get("source_payload_access_allowed") is False, "source access open")
    _require(safety.get("production_scan_allowed") is False, "production scan open")
    _require(safety.get("artifact_mutation_allowed") is False, "mutation open")
    _require(
        safety.get("validator_output_surface") == "STDOUT_JSON_ONLY",
        "output surface differs",
    )
    _require(
        safety.get("bounded_author_synthetic_attestation_allowed") is True,
        "bounded author attestation is not explicitly authorized",
    )
    _require(
        safety.get("author_validator_subprocess_mode")
        == "PYTHON_ISOLATED_FIXED_PATHS_SHELL_FALSE",
        "author validator subprocess mode differs",
    )
    _require(
        safety.get("author_validator_timeout_seconds")
        == AUTHOR_VALIDATOR_TIMEOUT_SECONDS,
        "author validator timeout differs",
    )
    maximum_registry_bytes = safety.get("maximum_registry_bytes")
    maximum_control_bytes = safety.get("maximum_control_json_bytes")
    _require(type(maximum_registry_bytes) is int, "registry byte cap must be integer")
    _require(type(maximum_control_bytes) is int, "control byte cap must be integer")
    _require(
        0 < maximum_registry_bytes <= HARD_MAX_REGISTRY_BYTES,
        "registry byte cap is unsafe",
    )
    _require(registry_bytes <= maximum_registry_bytes, "registry exceeds declared cap")
    _require(0 < maximum_control_bytes <= 16_777_216, "control byte cap is unsafe")


def _validate_missing_and_caveats(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    missing = _list(
        registry.get("missing_required_components"), "missing_required_components"
    )
    _require(
        _component_ids(missing, id_field="component_id") == EXPECTED_MISSING_COMPONENTS,
        "missing component set differs",
    )
    blockers: list[dict[str, Any]] = []
    for raw in missing:
        item = _mapping(raw, "missing component")
        _require(
            item.get("production_receipt_sha256") is None, "missing receipt has SHA"
        )
        _require(
            item.get("promotion_allowed_by_this_registry") is False, "promotion open"
        )
        _require(item.get("blocks_g0") is True, "missing component does not block G0")
        state = item.get("current_state")
        _require(
            isinstance(state, str) and state.startswith("MISSING"), "state differs"
        )
        blockers.append(
            {
                "component_id": item["component_id"],
                "current_state": state,
                "expected_basename": item["expected_basename"],
                "receipt_bound": False,
                "blocks_g0": True,
            }
        )

    caveats = _list(
        registry.get("nonblocking_bounded_caveats"),
        "nonblocking_bounded_caveats",
    )
    _require(
        _component_ids(caveats, id_field="caveat_id") == EXPECTED_CAVEATS,
        "bounded caveat set differs",
    )
    for raw in caveats:
        item = _mapping(raw, "bounded caveat")
        _require(
            item.get("blocks_g_minus_1_by_itself") is False,
            "bounded caveat was promoted to a blocker",
        )
    return blockers


def _validate_author_synthetic_evidence(
    registry: Mapping[str, Any],
    *,
    repo_root: Path,
    maximum_bytes: int,
) -> dict[str, Any]:
    passed = _list(
        registry.get("passed_repository_evidence"), "passed_repository_evidence"
    )
    author = next(
        (
            _mapping(item, "author-history repository evidence")
            for item in passed
            if _mapping(item, "repository evidence").get("component_id")
            == "author_history"
        ),
        None,
    )
    _require(author is not None, "author-history repository evidence is absent")
    evidence = _mapping(
        author.get("nonpromoting_synthetic_core_evidence"),
        "author synthetic evidence",
    )
    _require(
        dict(evidence) == EXPECTED_AUTHOR_SYNTHETIC_EVIDENCE,
        "author synthetic evidence contract differs",
    )

    artifacts: list[dict[str, Any]] = []
    artifact_paths: dict[str, Path] = {}
    artifact_payloads: dict[str, bytes] = {}
    for role in ("config", "validator", "fixture"):
        path = _safe_repository_path(repo_root, evidence[f"{role}_path"])
        payload = _read_regular_bytes(
            path,
            field=f"author synthetic {role}",
            maximum_bytes=maximum_bytes,
        )
        expected_sha256 = evidence[f"{role}_sha256"]
        _require(
            _sha256(payload) == expected_sha256,
            f"author synthetic {role} SHA-256 mismatch",
        )
        artifact_paths[role] = path
        artifact_payloads[role] = payload
        artifacts.append(
            {
                "role": role,
                "sha256": expected_sha256,
                "bytes": len(payload),
            }
        )

    author_config = _yaml_object(artifact_payloads["config"], "author synthetic config")
    stdout_contract = _mapping(
        author_config.get("stdout_report_contract"), "author stdout report contract"
    )
    evidence_policy = _mapping(
        author_config.get("fixture_evidence_policy"), "author fixture evidence policy"
    )
    pinned_policy = _mapping(
        evidence_policy.get("pinned_checked_fixture"), "pinned fixture policy"
    )
    generic_policy = _mapping(
        evidence_policy.get("generic_fixture"), "generic fixture policy"
    )
    conditioned_scope = _mapping(
        stdout_contract.get("evidence_conditioned_scope_values"),
        "author evidence-conditioned scope",
    )
    pinned_scope_contract = _mapping(
        conditioned_scope.get("pinned_checked_fixture"), "pinned scope contract"
    )
    generic_scope_contract = _mapping(
        conditioned_scope.get("unattested_generic_fixture"),
        "generic scope contract",
    )
    claim_boundaries = _mapping(
        author_config.get("claim_boundaries"), "author claim boundaries"
    )
    _require(
        pinned_policy.get("status") == evidence["pinned_report_status"]
        and pinned_policy.get("evidence_class") == evidence["pinned_evidence_class"],
        "pinned author evidence policy differs",
    )
    _require(
        generic_policy.get("status") == evidence["generic_report_status"]
        and generic_policy.get("evidence_class") == evidence["generic_evidence_class"],
        "generic author evidence policy differs",
    )
    _require(
        not str(generic_policy.get("status", "")).startswith("PASS"),
        "generic author fixture is incorrectly passing",
    )
    _require(
        pinned_scope_contract.get("parent_evidence_class")
        == evidence["pinned_parent_evidence_class"],
        "pinned parent evidence class differs",
    )
    _require(
        generic_scope_contract.get("parent_evidence_class")
        == evidence["generic_parent_evidence_class"],
        "generic parent evidence class differs",
    )
    _require(
        generic_policy.get("fixture_provenance_verified") is False
        and generic_policy.get("parent_stub_provenance_verified") is False
        and evidence["generic_fixture_promotes_component"] is False,
        "generic fixture nonpromotion boundary differs",
    )

    command = [
        sys.executable,
        "-I",
        str(artifact_paths["validator"]),
        "--config",
        str(artifact_paths["config"]),
        "--synthetic-fixture",
        str(artifact_paths["fixture"]),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root.resolve(strict=True),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=AUTHOR_VALIDATOR_TIMEOUT_SECONDS,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReadinessError("author synthetic validator timed out") from exc
    except OSError as exc:
        raise ReadinessError("author synthetic validator could not execute") from exc
    _require(completed.returncode == 0, "author synthetic validator returned nonzero")
    _require(completed.stderr == b"", "author synthetic validator wrote stderr")
    _require(
        isinstance(completed.stdout, bytes), "author validator stdout is not bytes"
    )
    author_report = _canonical_json_line(
        completed.stdout, "author synthetic validator stdout"
    )

    exact_top_keys = _list(
        stdout_contract.get("exact_top_level_keyset"),
        "author report exact top-level keyset",
    )
    _require(
        len(exact_top_keys) == len(set(exact_top_keys))
        and set(author_report) == set(exact_top_keys),
        "author report top-level keyset differs",
    )
    _require(
        author_report.get("schema_version") == evidence["report_schema_version"],
        "author report schema differs",
    )
    _require(
        author_report.get("surface_kind") == "SYNTHETIC_FIXTURE_ONLY",
        "author report surface differs",
    )
    _require(
        author_report.get("status") == evidence["pinned_report_status"]
        and author_report.get("evidence_class") == evidence["pinned_evidence_class"],
        "author report is not the pinned evidence class",
    )
    _require(
        author_report.get("production_status") == evidence["production_status"],
        "author report production status differs",
    )

    inputs = _mapping(author_report.get("inputs"), "author report inputs")
    expected_input_keys = _list(
        stdout_contract.get("inputs_exact_keyset"), "author input exact keyset"
    )
    _require(
        set(inputs) == set(expected_input_keys), "author report input keyset differs"
    )
    _require(
        inputs.get("config_sha256") == evidence["config_sha256"]
        and inputs.get("fixture_sha256") == evidence["fixture_sha256"],
        "author report input hash differs",
    )
    runtime = _mapping(inputs.get("runtime_fingerprint"), "author runtime fingerprint")
    runtime_keyset = _list(
        stdout_contract.get("runtime_fingerprint_exact_keyset"),
        "author runtime fingerprint keyset",
    )
    _require(set(runtime) == set(runtime_keyset), "author runtime keyset differs")
    current_runtime = _current_runtime_fingerprint()
    _require(
        dict(runtime) == current_runtime, "author report runtime is not current runtime"
    )
    attested_runtime = _mapping(
        evidence["attested_runtime_fingerprint"], "attested author runtime"
    )
    runtime_matches_attestation = current_runtime == dict(attested_runtime)

    scope = _mapping(author_report.get("scope"), "author report scope")
    scope_keys = _list(
        stdout_contract.get("scope_exact_keyset"), "author scope exact keyset"
    )
    _require(set(scope) == set(scope_keys), "author report scope keyset differs")
    minimal_fixture = _mapping(
        author_config.get("minimal_fixture"), "author minimal fixture"
    )
    exact_row_counts = _mapping(
        minimal_fixture.get("exact_row_counts"), "author exact row counts"
    )
    for row_field in ("main_rows", "spine_rows", "authorship_rows"):
        _require(
            _same_scalar(scope.get(row_field), exact_row_counts.get(row_field)),
            f"author report scope differs: {row_field}",
        )
    expected_scope = {
        "gpu_used": False,
        "fixture_evidence_class": evidence["pinned_evidence_class"],
        "parent_evidence_class": evidence["pinned_parent_evidence_class"],
        "production_source_interface_used": evidence[
            "production_source_interface_used"
        ],
        "production_input_accessed": evidence["production_input_accessed"],
        "pinned_fixture_sha256_match": evidence["pinned_fixture_sha256_match"],
        "files_written": 0,
        "scientific_effect_estimation_performed": False,
    }
    for key, expected in expected_scope.items():
        _require(_same_scalar(scope.get(key), expected), f"author scope differs: {key}")

    gate = _mapping(author_report.get("gate_status"), "author gate status")
    expected_gate = {
        "production_execution_authorized": False,
        "production_parity_established": evidence["production_parity_established"],
        "g_minus_1_data_readiness": evidence["g_minus_1_data_readiness"],
        "g0_measurement_freeze": evidence["g0_measurement_freeze"],
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }
    _require(dict(gate) == expected_gate, "author gate status differs")
    _require(
        author_report.get("claim_boundary")
        == claim_boundaries.get("pinned_checked_fixture"),
        "author pinned claim boundary differs",
    )

    foundation_contract = _mapping(
        author_config.get("foundation_runtime_verification"),
        "author foundation verification contract",
    )
    required_foundations = _list(
        foundation_contract.get("required_surfaces"),
        "author required foundation surfaces",
    )
    foundation_records = _list(
        inputs.get("foundation_verification"), "author foundation verification"
    )
    _require(
        len(foundation_records) == evidence["foundation_record_count"] == 6,
        "author foundation record count differs",
    )
    foundation_paths = [
        _mapping(item, "author foundation record").get("repository_relative_path")
        for item in foundation_records
    ]
    _require(
        foundation_paths == sorted(foundation_paths),
        "author foundation records are not sorted",
    )
    actual_foundations: list[dict[str, Any]] = []
    for raw in foundation_records:
        record = _mapping(raw, "author foundation record")
        _require(
            set(record)
            == {
                "repository_relative_path",
                "bytes",
                "sha256",
                "verified_against_config",
            },
            "author foundation record keyset differs",
        )
        _require(
            record.get("verified_against_config") is True,
            "author foundation record is unverified",
        )
        actual_foundations.append(
            {
                "repository_relative_path": record["repository_relative_path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    _require(
        actual_foundations == required_foundations,
        "author foundation records differ from config",
    )

    current_stdout_bytes = len(completed.stdout)
    current_stdout_sha256 = _sha256(completed.stdout)
    if runtime_matches_attestation:
        _require(
            current_stdout_bytes == evidence["attested_canonical_stdout_bytes"],
            "author attested stdout byte count mismatch",
        )
        _require(
            current_stdout_sha256 == evidence["attested_canonical_stdout_sha256"],
            "author attested stdout SHA-256 mismatch",
        )
        verification = "PASS_EXACT_CONFIG_PINNED_NONPROMOTING_SYNTHETIC_ATTESTATION"
        canonical_stdout_reproduced = True
    else:
        _require(
            evidence["runtime_mismatch_policy"]
            == "NOT_REATTESTED_RUNTIME_MISMATCH_NO_CROSS_RUNTIME_BYTE_CLAIM",
            "author runtime mismatch policy differs",
        )
        verification = "NOT_REATTESTED_RUNTIME_MISMATCH"
        canonical_stdout_reproduced = False

    return {
        "component_id": "author_history",
        "verification": verification,
        "artifacts": artifacts,
        "bounded_synthetic_validator_runs": 1,
        "report_status": evidence["pinned_report_status"],
        "fixture_evidence_class": evidence["pinned_evidence_class"],
        "parent_evidence_class": evidence["pinned_parent_evidence_class"],
        "generic_report_status": evidence["generic_report_status"],
        "generic_evidence_class": evidence["generic_evidence_class"],
        "generic_parent_evidence_class": evidence["generic_parent_evidence_class"],
        "generic_fixture_promotes_component": False,
        "canonical_stdout_reproduced": canonical_stdout_reproduced,
        "attested_runtime_fingerprint": dict(attested_runtime),
        "current_runtime_fingerprint": current_runtime,
        "attested_canonical_stdout_bytes": evidence["attested_canonical_stdout_bytes"],
        "attested_canonical_stdout_sha256": evidence[
            "attested_canonical_stdout_sha256"
        ],
        "current_canonical_stdout_bytes": current_stdout_bytes,
        "current_canonical_stdout_sha256": current_stdout_sha256,
        "foundation_verification": foundation_records,
        "claim_boundary": author_report["claim_boundary"],
        "production_status": "PRODUCTION_NOT_RUN",
        "production_source_interface_used": False,
        "production_input_accessed": False,
        "production_parity_established": False,
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "promotes_component": False,
    }


def verify_registry(
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    repo_root: Path = REPO_ROOT,
    receipt_overrides: Mapping[str, Path] | None = None,
    require_operator_receipts: bool = False,
) -> dict[str, Any]:
    """Verify bounded evidence and return an always-pending decision."""

    registry, raw_registry = _load_registry(registry_path)
    _validate_closed_contract(registry, len(raw_registry))
    blockers = _validate_missing_and_caveats(registry)
    safety = _mapping(registry["safety_boundary"], "safety_boundary")
    control_cap = int(safety["maximum_control_json_bytes"])
    author_synthetic_evidence = _validate_author_synthetic_evidence(
        registry,
        repo_root=repo_root,
        maximum_bytes=control_cap,
    )

    repo_bindings = _list(
        registry.get("passed_repository_evidence"), "passed_repository_evidence"
    )
    _require(
        _component_ids(repo_bindings, id_field="component_id")
        == EXPECTED_REPOSITORY_COMPONENTS,
        "repository component set differs",
    )
    repository_results: list[dict[str, Any]] = []
    for raw in repo_bindings:
        binding = _mapping(raw, "repository binding")
        path = _safe_repository_path(repo_root, binding.get("relative_path"))
        payload = _read_regular_bytes(
            path,
            field=f"repository evidence {binding['component_id']}",
            maximum_bytes=control_cap,
        )
        _validate_json_binding(
            payload,
            binding,
            field=f"repository evidence {binding['component_id']}",
        )
        repository_results.append(
            {
                "component_id": binding["component_id"],
                "verification": "PASS_EXACT_REPOSITORY_EVIDENCE",
                "sha256": _sha256(payload),
            }
        )

    blocking_bindings = _list(
        registry.get("blocking_repository_evidence"),
        "blocking_repository_evidence",
    )
    _require(
        _component_ids(blocking_bindings, id_field="component_id")
        == EXPECTED_BLOCKING_REPOSITORY_COMPONENTS,
        "blocking repository component set differs",
    )
    blocking_results: list[dict[str, Any]] = []
    for raw in blocking_bindings:
        binding = _mapping(raw, "blocking repository binding")
        _require(
            binding.get("promotion_allowed_by_this_registry") is False,
            "blocking repository evidence permits promotion",
        )
        _require(
            binding.get("blocks_g0") is True,
            "blocking repository evidence does not block G0",
        )
        state = binding.get("current_state")
        _require(
            isinstance(state, str) and state.startswith("PRODUCTION_RECEIPT_PRESENT_"),
            "blocking repository state differs",
        )
        path = _safe_repository_path(repo_root, binding.get("relative_path"))
        payload = _read_regular_bytes(
            path,
            field=f"blocking repository evidence {binding['component_id']}",
            maximum_bytes=control_cap,
        )
        _validate_json_binding(
            payload,
            binding,
            field=f"blocking repository evidence {binding['component_id']}",
        )
        blocking_results.append(
            {
                "component_id": binding["component_id"],
                "verification": "PASS_EXACT_BLOCKING_REPOSITORY_EVIDENCE",
                "sha256": _sha256(payload),
                "current_state": state,
                "receipt_bound": True,
                "blocks_g0": True,
            }
        )

    overrides = dict(receipt_overrides or {})
    _require(
        set(overrides).issubset(EXPECTED_OPERATOR_COMPONENTS),
        "unknown operator receipt override",
    )
    receipt_bindings = _list(
        registry.get("passed_operator_control_receipts"),
        "passed_operator_control_receipts",
    )
    _require(
        _component_ids(receipt_bindings, id_field="component_id")
        == EXPECTED_OPERATOR_COMPONENTS,
        "operator receipt component set differs",
    )
    operator_results: list[dict[str, Any]] = []
    verified_operator_count = 0
    for raw in receipt_bindings:
        binding = _mapping(raw, "operator receipt binding")
        component_id = str(binding["component_id"])
        env_name = binding.get("path_env")
        _require(isinstance(env_name, str), "receipt path environment name differs")
        selected = overrides.get(component_id)
        if selected is None:
            _require(
                not require_operator_receipts,
                f"operator receipt is required but unavailable: {component_id}",
            )
            operator_results.append(
                {
                    "component_id": component_id,
                    "verification": "NOT_REVERIFIED_NO_EXPLICIT_PATH",
                    "expected_sha256": binding["sha256"],
                }
            )
            continue
        selected = Path(selected)
        _require(selected.is_absolute(), "operator receipt path must be absolute")
        _require(
            selected.name == binding.get("expected_basename"),
            f"operator receipt basename mismatch: {component_id}",
        )
        maximum_bytes = binding.get("maximum_bytes")
        _require(type(maximum_bytes) is int, "receipt byte cap must be integer")
        _require(0 < maximum_bytes <= control_cap, "receipt byte cap is unsafe")
        payload = _read_regular_bytes(
            selected,
            field=f"operator receipt {component_id}",
            maximum_bytes=maximum_bytes,
        )
        _validate_json_binding(
            payload, binding, field=f"operator receipt {component_id}"
        )
        verified_operator_count += 1
        operator_results.append(
            {
                "component_id": component_id,
                "verification": "PASS_EXACT_OPERATOR_CONTROL_RECEIPT",
                "sha256": _sha256(payload),
                "operator_path_redacted": True,
            }
        )

    verification_status = (
        "PASS_FAIL_CLOSED_REGISTRY_AND_ALL_BOUND_CONTROL_EVIDENCE"
        if verified_operator_count == len(receipt_bindings)
        else "PASS_FAIL_CLOSED_REGISTRY_REPOSITORY_EVIDENCE_ONLY"
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "verification_status": verification_status,
        "registry_sha256": _sha256(raw_registry),
        "status": PENDING_STATUS,
        "pass_g_minus_1": False,
        "execution_authorized": False,
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "source_payloads_opened": 0,
        "production_scans_started": 0,
        "repository_evidence": repository_results,
        "blocking_repository_evidence": blocking_results,
        "nonpromoting_synthetic_evidence": author_synthetic_evidence,
        "operator_control_receipts": operator_results,
        "missing_required_components": blockers,
        "missing_required_component_count": len(blockers),
        "blocking_required_component_count": len(blocking_results),
        "total_blocking_component_count": len(blockers) + len(blocking_results),
        "claim_boundary": registry["claim_boundary"],
    }


def _receipt_overrides(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        component_id, separator, raw_path = value.partition("=")
        _require(
            bool(separator and component_id and raw_path), "invalid --receipt value"
        )
        _require(component_id not in result, "duplicate --receipt component")
        result[component_id] = Path(raw_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--receipt",
        action="append",
        default=[],
        metavar="COMPONENT_ID=ABSOLUTE_PATH",
    )
    parser.add_argument("--require-operator-receipts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_registry(
            args.registry,
            repo_root=args.repo_root,
            receipt_overrides=_receipt_overrides(args.receipt),
            require_operator_receipts=args.require_operator_receipts,
        )
    except ReadinessError as exc:
        report = {
            "schema_version": REPORT_SCHEMA,
            "verification_status": "BLOCKED_INVALID_REGISTRY_OR_CONTROL_EVIDENCE",
            "status": PENDING_STATUS,
            "pass_g_minus_1": False,
            "execution_authorized": False,
            "error": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
