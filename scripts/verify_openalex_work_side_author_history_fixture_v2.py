#!/usr/bin/env python3
"""Validate the bounded author-history v2 synthetic fixture.

This program deliberately has no production source, execution, output-directory,
or promotion interface.  A successful invocation writes exactly one canonical
JSON report to stdout and does not create files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import unicodedata
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import Node
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_work_side_author_history_v2.yaml"
)
# This raw-file digest is replaced after every reviewed config change.
EXPECTED_CONFIG_SHA256 = (
    "f0359f9b9ab7c2e5a51b1e760f94c70d51e544dc7f594d8e0512a9cc8ff216c2"
)

SCRIPT_RELATIVE_PATH = "scripts/verify_openalex_work_side_author_history_fixture_v2.py"
CONFIG_RELATIVE_PATH = (
    "experiments/configs/openalex_g1_work_side_author_history_v2.yaml"
)
FIXTURE_SCHEMA = "openalex_g1_work_side_author_history_synthetic_fixture.v2"
REPORT_SCHEMA = "openalex_g1_work_side_author_history_fixture_validation_report.v2"
PINNED_STATUS = (
    "PASS_SYNTHETIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_PINNED_FIXTURE_CONTRACT_ONLY"
)
GENERIC_STATUS = (
    "VALID_UNATTESTED_GENERIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_V2_MECHANICS_ONLY"
)
PINNED_EVIDENCE_CLASS = "PINNED_CHECKED_FIXTURE"
GENERIC_EVIDENCE_CLASS = "UNATTESTED_GENERIC_FIXTURE"
PINNED_PARENT_CLASS = "CONFIG_PINNED_SYNTHETIC_PARENT_STUBS"
GENERIC_PARENT_CLASS = "SELF_DECLARED_UNVERIFIED_SYNTHETIC_PARENT_STUBS"
CANONICAL_JSON_ABI = "openalex_g1_work_side_author_history_canonical_json.v2"
SURFACE_KIND = "SYNTHETIC_FIXTURE_ONLY"
SNAPSHOT_DATE = "2026-03-30"
PRODUCTION_STATUS = "PRODUCTION_NOT_RUN"
PRODUCTION_ROOT_ID = "openalex_works_20260330_repairreplay_20260410_190630"
PINNED_CLAIM_BOUNDARY = (
    "This config-pinned repository synthetic fixture validates bounded "
    "current-snapshot author-token history accounting only; the validator used "
    "no production source interface and accessed no production input, "
    "runtime-dependent diagnostic semantics are bound to the reported runtime "
    "fingerprint and are not asserted byte-identical across runtimes, production "
    "parity is not established, G-1 and G0 remain pending, and no empirical or "
    "scientific effect is established."
)
GENERIC_CLAIM_BOUNDARY = (
    "This unattested generic fixture validates bounded author-token history "
    "mechanics only; fixture provenance and self-declared synthetic parent stubs "
    "are unverified, production-input origin is unknown, the validator exposed "
    "and used no production source interface, runtime-dependent diagnostic "
    "semantics are bound to the reported runtime fingerprint and are not asserted "
    "byte-identical across runtimes, production parity is not established, G-1 "
    "and G0 remain pending, and no empirical or scientific effect is established."
)
PINNED_FIXTURE_RELATIVE_PATH = (
    "tests/fixtures/openalex_work_side_author_history_v2_minimal.json"
)

MAX_CONFIG_BYTES = 1_048_576
MAX_FIXTURE_BYTES = 1_048_576
MAX_FOUNDATION_FILE_BYTES = 4_194_304
MAX_MAIN_ROWS = 10_000
MAX_SPINE_ROWS = 10_000
MAX_AUTHORSHIP_ROWS = 20_000
UINT64_MAX = 2**64 - 1
UINT32_MAX = 2**32 - 1
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
MAX_NESTING_DEPTH = 64
MAX_IDENTIFIER_DECIMAL_DIGITS = 20
MAX_JSON_INTEGER_DECIMAL_DIGITS = 20

WORK_RE = re.compile(r"^https://openalex[.]org/W(0|[1-9][0-9]*)$")
AUTHOR_RE = re.compile(r"^https://openalex[.]org/A(0|[1-9][0-9]*)$")
FIELD_RE = re.compile(r"^https://openalex[.]org/fields/([1-9]|1[0-9]|2[0-6])$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
W_MINUS_1 = "https://openalex.org/W-1"
TEAM_CAPS: tuple[int | str, ...] = (10, 50, 100, "ALL")
THRESHOLDS = (50, 100, 250, 500, 1000)
HISTORY_USABLE = frozenset({2, 3})
TEAM_BANDS = ("1", "2", "3-5", "6-10", "11-50", "51-100", "101+")
FIELD_KEYS = ("null", *(str(value) for value in range(1, 27)))

TOP_LEVEL_KEYS = {
    "schema_version",
    "surface_kind",
    "production_input_access_allowed",
    "root_id",
    "snapshot_date",
    "synthetic_parents",
    "main_rows",
    "spine_rows",
    "authorship_rows",
}
MAIN_KEYS = {"id", "authors_count", "primary_topic__field__id"}
SPINE_KEYS = {
    "work_id_numeric",
    "publication_day",
    "publication_year",
    "temporal_disposition",
}
AUTHORSHIP_KEYS = {
    "id",
    "authorships__author__id",
    "authorships__author__orcid",
    "authorships__raw_author_name",
    "authorships__author__display_name",
    "authorships__author_position",
}
PARENT_KEYS = {
    "current_content_baseline",
    "main_temporal_validity",
    "canonical_work_time_spine",
}
CONTENT_PARENT_KEYS = {
    "schema_version",
    "status",
    "synthetic_content_tree_sha256",
    "synthetic_content_result_fingerprint_sha256",
}
TEMPORAL_PARENT_KEYS = {
    *CONTENT_PARENT_KEYS,
    "synthetic_temporal_result_fingerprint_sha256",
}
SPINE_PARENT_KEYS = {
    *TEMPORAL_PARENT_KEYS,
    "synthetic_spine_result_fingerprint_sha256",
    "synthetic_spine_data_tree_sha256",
    "rows",
}

AGGREGATE_DOMAINS = {
    "primary_pair": "openalex_g1_work_side_author_history.synthetic.primary_pair_diagnostics.v2",
    "usable_pair": "openalex_g1_work_side_author_history.synthetic.history_usable_pair_diagnostics.v2",
    "base_year": "openalex_g1_work_side_author_history.synthetic.base_view_per_year.v2",
    "positive_year": "openalex_g1_work_side_author_history.synthetic.positive_view_per_year.v2",
    "base_token": "openalex_g1_work_side_author_history.synthetic.base_view_per_token.v2",
    "positive_token": "openalex_g1_work_side_author_history.synthetic.positive_view_per_token.v2",
    "base_grid": "openalex_g1_work_side_author_history.synthetic.base_view_team_hyper_grid.v2",
    "positive_grid": "openalex_g1_work_side_author_history.synthetic.positive_view_team_hyper_grid.v2",
}


class ValidationError(Exception):
    """A fail-closed contract violation."""


class _StrictLoader(yaml.SafeLoader):
    """Safe YAML loader rejecting aliases before construction."""

    def compose_node(self, parent: Any, index: Any) -> Node:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise ConstructorError(
                "while composing a node",
                None,
                "YAML aliases are forbidden",
                event.start_mark,
            )
        return super().compose_node(parent, index)


def _construct_unique_mapping(
    loader: _StrictLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    if any(key.value == "<<" for key, _ in node.value):
        raise ConstructorError(
            "while constructing a mapping",
            node.start_mark,
            "YAML merge keys are forbidden",
            node.start_mark,
        )
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "unhashable YAML key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate YAML key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    _require(_is_int(value), f"{field} must be a non-boolean integer")
    _require(minimum <= value <= maximum, f"{field} is outside its allowed range")
    return value


def _require_string_or_null(value: Any, *, field: str) -> str | None:
    _require(value is None or isinstance(value, str), f"{field} must be string or null")
    return value


def _require_exact_keys(
    value: Any, expected: set[str], *, field: str
) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    actual = set(value)
    _require(actual == expected, f"{field} has a missing or extra key")
    _require(
        all(isinstance(key, str) for key in value), f"{field} keys must be strings"
    )
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _domain_hash(domain: str, value: Any) -> str:
    return _sha256(domain.encode("utf-8") + b"\0" + canonical_json_bytes(value))


def _validate_parsed_tree(value: Any, *, field: str) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        _require(depth <= MAX_NESTING_DEPTH, f"{field} exceeds maximum nesting depth")
        if isinstance(current, str):
            _require(
                not any(0xD800 <= ord(character) <= 0xDFFF for character in current),
                f"{field} contains a lone Unicode surrogate",
            )
        elif isinstance(current, dict):
            for key, item in current.items():
                stack.append((key, depth + 1))
                stack.append((item, depth + 1))
        elif isinstance(current, (list, tuple, set, frozenset)):
            stack.extend((item, depth + 1) for item in current)


def _strict_json_loads(payload: bytes, *, field: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"{field} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValidationError(f"{field} contains non-finite JSON constant {value}")

    def reject_float(value: str) -> None:
        raise ValidationError(
            f"{field} contains forbidden floating-point value {value}"
        )

    def parse_integer(value: str) -> int:
        digits = value[1:] if value.startswith("-") else value
        if len(digits) > MAX_JSON_INTEGER_DECIMAL_DIGITS:
            raise ValidationError(f"{field} contains an overlong integer lexeme")
        try:
            return int(value)
        except ValueError as exc:
            raise ValidationError(
                f"{field} contains an invalid integer lexeme"
            ) from exc

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=reject_float,
            parse_int=parse_integer,
        )
    except ValidationError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ValidationError(f"{field} is not strict UTF-8 JSON") from exc
    _validate_parsed_tree(parsed, field=field)
    return parsed


def _strict_yaml_loads(payload: bytes, *, field: str) -> Any:
    try:
        parsed = yaml.load(payload, Loader=_StrictLoader)
    except (UnicodeDecodeError, yaml.YAMLError, RecursionError, ValueError) as exc:
        raise ValidationError(f"{field} is not strict safe YAML") from exc
    _validate_parsed_tree(parsed, field=field)
    return parsed


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _resolve_without_symlink_components(path: Path, *, field: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise ValidationError(f"missing {field}") from exc
        _require(
            not stat.S_ISLNK(info.st_mode),
            f"{field} path component must not be a symlink",
        )
        if index < len(absolute.parts[1:]) - 1:
            _require(
                stat.S_ISDIR(info.st_mode),
                f"{field} parent component must be a directory",
            )
    try:
        return absolute.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"could not resolve {field}") from exc


def _read_bounded_regular(
    path: Path, *, field: str, maximum_bytes: int
) -> tuple[bytes, os.stat_result]:
    _resolve_without_symlink_components(path, field=field)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ValidationError(f"missing {field}") from exc
    _require(stat.S_ISREG(before.st_mode), f"{field} must be a regular file")
    _require(not stat.S_ISLNK(before.st_mode), f"{field} must not be a symlink")
    _require(before.st_nlink == 1, f"{field} must have link count one")
    _require(before.st_size <= maximum_bytes, f"{field} exceeds its hard byte cap")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValidationError(f"could not securely open {field}") from exc
    try:
        opened = os.fstat(descriptor)
        _require(
            _file_identity(opened) == _file_identity(before),
            f"{field} changed before open",
        )
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        _require(len(payload) <= maximum_bytes, f"{field} exceeds its hard byte cap")
        after_fd = os.fstat(descriptor)
        _require(
            _file_identity(after_fd) == _file_identity(before),
            f"{field} changed while read",
        )
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    _require(
        _file_identity(after_path) == _file_identity(before),
        f"{field} changed after read",
    )
    return payload, before


def _validator_contract() -> dict[str, Any]:
    return {
        "script": SCRIPT_RELATIVE_PATH,
        "exact_cli": ["--config", "--synthetic-fixture"],
        "accepted_config_path": CONFIG_RELATIVE_PATH,
        "hard_caps": {
            "fixture_bytes": MAX_FIXTURE_BYTES,
            "main_rows": MAX_MAIN_ROWS,
            "spine_rows": MAX_SPINE_ROWS,
            "authorship_rows": MAX_AUTHORSHIP_ROWS,
        },
        "fixture_schema_version": FIXTURE_SCHEMA,
        "pinned_success_status": PINNED_STATUS,
        "generic_valid_status": GENERIC_STATUS,
        "canonical_json_abi": CANONICAL_JSON_ABI,
        "foundation_file_max_bytes": MAX_FOUNDATION_FILE_BYTES,
        "pinned_fixture_path": PINNED_FIXTURE_RELATIVE_PATH,
        "max_nesting_depth": MAX_NESTING_DEPTH,
        "max_identifier_decimal_digits": MAX_IDENTIFIER_DECIMAL_DIGITS,
        "max_json_integer_decimal_digits": MAX_JSON_INTEGER_DECIMAL_DIGITS,
        "surface_kind": SURFACE_KIND,
        "production_input_access_allowed": False,
        "snapshot_date": SNAPSHOT_DATE,
        "team_caps": ["10", "50", "100", "ALL"],
        "hyper_prolific_thresholds": [50, 100, 250, 500, 1000],
        "history_usable_dispositions": [2, 3],
        "g_minus_1_status": "PENDING",
        "g0_status": "PENDING",
    }


def _repository_path(relative_path: str, *, field: str) -> Path:
    _require(
        isinstance(relative_path, str) and relative_path, f"{field} path is invalid"
    )
    candidate = Path(relative_path)
    _require(
        not candidate.is_absolute()
        and "\\" not in relative_path
        and all(part not in {"", ".", ".."} for part in candidate.parts),
        f"{field} path must be a normalized repository-relative path",
    )
    path = REPO_ROOT / candidate
    resolved = _resolve_without_symlink_components(path, field=field)
    _require(
        resolved.is_relative_to(REPO_ROOT.resolve(strict=True)),
        f"{field} path escapes the repository",
    )
    return path


def _declared_foundation_path_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    foundation = config.get("foundation_binding")
    legacy = config.get("legacy_v0_surfaces")
    canonical = config.get("canonical_production_surface_binding")
    _require(isinstance(foundation, dict), "foundation_binding is missing")
    _require(isinstance(legacy, dict), "legacy_v0_surfaces is missing")
    _require(isinstance(canonical, dict), "canonical production binding is missing")
    legacy_paths = legacy.get("paths_and_sha256")
    _require(isinstance(legacy_paths, dict), "legacy v0 path/hash mapping is missing")
    declared: dict[str, str] = {
        foundation.get("author_v1_config_path"): foundation.get(
            "author_v1_config_sha256"
        ),
        canonical.get("config_path"): canonical.get("config_sha256"),
        canonical.get("runner_path"): canonical.get("runner_sha256"),
        **legacy_paths,
    }
    _require(
        len(declared) == 6, "foundation declarations must identify six unique files"
    )
    for relative_path, digest in declared.items():
        _require(isinstance(relative_path, str), "foundation path must be a string")
        _hex64(digest, field=f"foundation digest for {relative_path}")
    return declared


def _verify_foundation_files(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = config.get("foundation_runtime_verification")
    _require(isinstance(contract, dict), "foundation runtime verification is missing")
    _require(
        contract.get("maximum_bytes_per_file") == MAX_FOUNDATION_FILE_BYTES,
        "foundation byte cap differs from compiled constant",
    )
    required = contract.get("required_surfaces")
    _require(
        isinstance(required, list) and len(required) == 6,
        "foundation surface list mismatch",
    )
    declared = _declared_foundation_path_hashes(config)
    expected_records: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(required):
        record = _require_exact_keys(
            raw,
            {"repository_relative_path", "bytes", "sha256"},
            field=f"foundation required_surfaces[{index}]",
        )
        relative_path = record["repository_relative_path"]
        _require(isinstance(relative_path, str), "foundation path must be a string")
        _require(relative_path not in expected_records, "duplicate foundation path")
        byte_count = _require_int(
            record["bytes"],
            field=f"foundation bytes for {relative_path}",
            minimum=0,
            maximum=MAX_FOUNDATION_FILE_BYTES,
        )
        digest = _hex64(
            record["sha256"], field=f"foundation sha256 for {relative_path}"
        )
        _require(
            declared.get(relative_path) == digest, "foundation declarations disagree"
        )
        expected_records[relative_path] = {"bytes": byte_count, "sha256": digest}
    _require(set(expected_records) == set(declared), "foundation file set mismatch")
    results: list[dict[str, Any]] = []
    for relative_path in sorted(expected_records):
        expected = expected_records[relative_path]
        path = _repository_path(relative_path, field="foundation file")
        payload, _ = _read_bounded_regular(
            path,
            field=f"foundation file {relative_path}",
            maximum_bytes=MAX_FOUNDATION_FILE_BYTES,
        )
        _require(
            len(payload) == expected["bytes"],
            f"foundation byte count mismatch for {relative_path}",
        )
        digest = _sha256(payload)
        _require(
            digest == expected["sha256"],
            f"foundation SHA-256 mismatch for {relative_path}",
        )
        results.append(
            {
                "repository_relative_path": relative_path,
                "bytes": len(payload),
                "sha256": digest,
                "verified_against_config": True,
            }
        )
    return results


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], bytes, os.stat_result, list[dict[str, Any]]]:
    try:
        supplied = path.resolve(strict=True)
        expected = EXPECTED_CONFIG_PATH.resolve(strict=True)
    except OSError as exc:
        raise ValidationError("could not resolve the repository config") from exc
    _require(
        supplied == expected,
        "only the exact audited repository config path is accepted",
    )
    payload, identity = _read_bounded_regular(
        path,
        field="author-history v2 config",
        maximum_bytes=MAX_CONFIG_BYTES,
    )
    _require(
        EXPECTED_CONFIG_SHA256 != "__PATCH_FINAL_AUDITED_CONFIG_SHA256__",
        "validator is not bound to a final audited config SHA-256",
    )
    _require(
        HEX64_RE.fullmatch(EXPECTED_CONFIG_SHA256) is not None,
        "embedded config SHA-256 is malformed",
    )
    _require(
        _sha256(payload) == EXPECTED_CONFIG_SHA256,
        "repository config byte hash mismatch",
    )
    parsed = _strict_yaml_loads(payload, field="author-history v2 config")
    _require(isinstance(parsed, dict), "author-history v2 config must be an object")
    _require(
        parsed.get("synthetic_fixture_validator") == _validator_contract(),
        "synthetic_fixture_validator contract differs from compiled constants",
    )
    _require(
        parsed.get("claim_boundaries")
        == {
            "pinned_checked_fixture": PINNED_CLAIM_BOUNDARY,
            "unattested_generic_fixture": GENERIC_CLAIM_BOUNDARY,
        },
        "config claim boundaries mismatch",
    )
    fixture_contract = parsed.get("fixture_contract")
    _require(isinstance(fixture_contract, dict), "config fixture contract is missing")
    _require(
        fixture_contract.get("production_root_id_forbidden_substring")
        == PRODUCTION_ROOT_ID,
        "config production root ID boundary mismatch",
    )
    evidence_policy = parsed.get("fixture_evidence_policy")
    _require(isinstance(evidence_policy, dict), "fixture evidence policy is missing")
    pinned_policy = evidence_policy.get("pinned_checked_fixture")
    generic_policy = evidence_policy.get("generic_fixture")
    _require(isinstance(pinned_policy, dict), "pinned fixture policy is missing")
    _require(isinstance(generic_policy, dict), "generic fixture policy is missing")
    _require(
        pinned_policy.get("required_repository_relative_path")
        == PINNED_FIXTURE_RELATIVE_PATH
        and pinned_policy.get("evidence_class") == PINNED_EVIDENCE_CLASS
        and pinned_policy.get("status") == PINNED_STATUS,
        "pinned fixture evidence policy mismatch",
    )
    _require(
        generic_policy.get("evidence_class") == GENERIC_EVIDENCE_CLASS
        and generic_policy.get("status") == GENERIC_STATUS,
        "generic fixture evidence policy mismatch",
    )
    foundation_records = _verify_foundation_files(parsed)
    return parsed, payload, identity, foundation_records


def _load_fixture(
    path: Path,
    *,
    config_identity: os.stat_result,
    production_root_id: str,
) -> tuple[dict[str, Any], bytes, Path]:
    resolved = _resolve_without_symlink_components(path, field="synthetic fixture")
    _require(
        not any(production_root_id in part for part in path.absolute().parts),
        "fixture path contains the production root ID",
    )
    _require(
        not any(production_root_id in part for part in resolved.parts),
        "resolved fixture path contains the production root ID",
    )
    payload, identity = _read_bounded_regular(
        path,
        field="synthetic fixture",
        maximum_bytes=MAX_FIXTURE_BYTES,
    )
    _require(
        (identity.st_dev, identity.st_ino)
        != (config_identity.st_dev, config_identity.st_ino),
        "config and fixture must be different files",
    )
    parsed = _strict_json_loads(payload, field="synthetic fixture")
    _require(isinstance(parsed, dict), "synthetic fixture must be an object")
    root_id = parsed.get("root_id")
    _require(isinstance(root_id, str), "fixture root_id must be a string")
    _require(
        root_id.startswith("synthetic-"), "fixture root_id must start with synthetic-"
    )
    _require(
        production_root_id not in root_id,
        "fixture root_id contains the production root ID",
    )
    return parsed, payload, resolved


def _classify_fixture(
    config: Mapping[str, Any],
    *,
    resolved_fixture_path: Path,
    fixture_sha256: str,
) -> tuple[bool, bool]:
    minimal = config.get("minimal_fixture")
    policy = config.get("fixture_evidence_policy", {}).get("pinned_checked_fixture")
    _require(isinstance(minimal, dict), "minimal fixture pin is missing")
    _require(isinstance(policy, dict), "pinned fixture policy is missing")
    pinned_path = minimal.get("path")
    pinned_sha256 = minimal.get("sha256")
    _require(
        pinned_path == PINNED_FIXTURE_RELATIVE_PATH, "minimal fixture path pin mismatch"
    )
    _hex64(pinned_sha256, field="minimal fixture SHA-256")
    _require(
        policy.get("required_repository_relative_path") == pinned_path
        and policy.get("required_sha256") == pinned_sha256,
        "fixture evidence pin cross-binding mismatch",
    )
    expected_path = _repository_path(pinned_path, field="pinned fixture")
    expected_resolved = expected_path.resolve(strict=True)
    sha_match = fixture_sha256 == pinned_sha256
    exact_path = resolved_fixture_path == expected_resolved
    if exact_path:
        _require(
            sha_match,
            "config-pinned fixture path has a SHA-256 mismatch and cannot downgrade to generic",
        )
    return exact_path and sha_match, sha_match


def _runtime_fingerprint() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "unicodedata_unidata_version": unicodedata.unidata_version,
        "canonical_json_abi": CANONICAL_JSON_ABI,
    }


def _hex64(value: Any, *, field: str) -> str:
    _require(
        isinstance(value, str) and HEX64_RE.fullmatch(value) is not None,
        f"{field} must be lowercase 64-hex",
    )
    return value


def _validate_parents(value: Any, *, spine_rows: int) -> dict[str, Any]:
    parents = _require_exact_keys(value, PARENT_KEYS, field="synthetic_parents")
    content = _require_exact_keys(
        parents["current_content_baseline"],
        CONTENT_PARENT_KEYS,
        field="synthetic content parent",
    )
    temporal = _require_exact_keys(
        parents["main_temporal_validity"],
        TEMPORAL_PARENT_KEYS,
        field="synthetic temporal parent",
    )
    spine = _require_exact_keys(
        parents["canonical_work_time_spine"],
        SPINE_PARENT_KEYS,
        field="synthetic spine parent",
    )
    expected_literals = (
        (
            content,
            "openalex_g1_work_side_author_history_synthetic_content_parent.v1",
            "SYNTHETIC_CONTENT_BASELINE_PARENT_ONLY",
        ),
        (
            temporal,
            "openalex_g1_work_side_author_history_synthetic_temporal_parent.v1",
            "SYNTHETIC_MAIN_TEMPORAL_PARENT_ONLY",
        ),
        (
            spine,
            "openalex_g1_work_side_author_history_synthetic_spine_parent.v1",
            "SYNTHETIC_CANONICAL_SPINE_PARENT_ONLY",
        ),
    )
    for parent, schema, status_value in expected_literals:
        _require(parent["schema_version"] == schema, "synthetic parent schema mismatch")
        _require(parent["status"] == status_value, "synthetic parent status mismatch")
        for key, item in parent.items():
            if key.endswith("sha256"):
                _hex64(item, field=f"synthetic parent {key}")
    _require_int(
        spine["rows"],
        field="synthetic spine parent rows",
        minimum=0,
        maximum=MAX_SPINE_ROWS,
    )
    _require(spine["rows"] == spine_rows, "synthetic spine parent row count mismatch")
    for key in (
        "synthetic_content_tree_sha256",
        "synthetic_content_result_fingerprint_sha256",
    ):
        _require(
            content[key] == temporal[key] == spine[key],
            f"parent cross-binding mismatch for {key}",
        )
    temporal_key = "synthetic_temporal_result_fingerprint_sha256"
    _require(
        temporal[temporal_key] == spine[temporal_key],
        "temporal parent result cross-binding mismatch",
    )
    return {
        "synthetic_content_tree_sha256": content["synthetic_content_tree_sha256"],
        "synthetic_content_result_fingerprint_sha256": content[
            "synthetic_content_result_fingerprint_sha256"
        ],
        "synthetic_temporal_result_fingerprint_sha256": temporal[temporal_key],
        "synthetic_spine_result_fingerprint_sha256": spine[
            "synthetic_spine_result_fingerprint_sha256"
        ],
        "synthetic_spine_data_tree_sha256": spine["synthetic_spine_data_tree_sha256"],
        "spine_rows": spine["rows"],
    }


def _parse_numeric_identifier(
    value: str, pattern: re.Pattern[str], *, field: str
) -> int:
    match = pattern.fullmatch(value)
    _require(match is not None, f"{field} has invalid grammar")
    suffix = match.group(1)
    _require(
        len(suffix) <= MAX_IDENTIFIER_DECIMAL_DIGITS,
        f"{field} has an overlong decimal suffix",
    )
    if len(suffix) == MAX_IDENTIFIER_DECIMAL_DIGITS:
        _require(suffix <= str(UINT64_MAX), f"{field} exceeds uint64")
    try:
        numeric = int(suffix)
    except ValueError as exc:
        raise ValidationError(f"{field} has an invalid decimal suffix") from exc
    _require(numeric <= UINT64_MAX, f"{field} exceeds uint64")
    return numeric


def _validate_day(
    disposition: int, day: Any, year: Any, *, field: str
) -> tuple[int | None, int | None]:
    if disposition == 0:
        _require(
            day is None and year is None,
            f"{field} disposition 0 requires null day and year",
        )
        return None, None
    day_value = _require_int(
        day, field=f"{field}.publication_day", minimum=INT32_MIN, maximum=INT32_MAX
    )
    year_value = _require_int(
        year, field=f"{field}.publication_year", minimum=1, maximum=9999
    )
    try:
        publication_date = date(1970, 1, 1) + timedelta(days=day_value)
    except (OverflowError, ValueError) as exc:
        raise ValidationError(
            f"{field} publication day is outside supported calendar"
        ) from exc
    _require(
        publication_date.year == year_value, f"{field} publication day/year mismatch"
    )
    if publication_date < date(1900, 1, 1):
        expected = 1
    elif publication_date.year < 2026:
        expected = 2
    elif publication_date.year == 2026 and publication_date <= date(2026, 3, 30):
        expected = 3
    elif publication_date.year == 2026:
        expected = 4
    else:
        expected = 5
    _require(
        disposition == expected, f"{field} temporal disposition predicate mismatch"
    )
    return day_value, year_value


def _team_band(size: int) -> str:
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    if size <= 50:
        return "11-50"
    if size <= 100:
        return "51-100"
    return "101+"


def _tag(value: str | None) -> dict[str, str]:
    if value is None:
        return {"kind": "null"}
    return {"kind": "string", "value": value}


def _sorted_tags(values: set[str | None]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if None in values:
        result.append(_tag(None))
    result.extend(
        _tag(value)
        for value in sorted(
            (item for item in values if item is not None),
            key=lambda item: item.encode("utf-8"),
        )
    )
    return result


def _normalize_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        " " if character.isspace() else character for character in normalized
    )
    return re.sub(r" +", " ", normalized).strip(" ")


_ORCID_FORMS = (
    re.compile(
        r"^https://orcid[.]org/([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9]{3}[0-9X])$"
    ),
    re.compile(
        r"^http://orcid[.]org/([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9]{3}[0-9Xx])$"
    ),
    re.compile(
        r"^https://www[.]orcid[.]org/([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9]{3}[0-9Xx])$"
    ),
    re.compile(
        r"^http://www[.]orcid[.]org/([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9]{3}[0-9Xx])$"
    ),
    re.compile(r"^([0-9]{4})-([0-9]{4})-([0-9]{4})-([0-9]{3}[0-9Xx])$"),
    re.compile(r"^([0-9]{4})([0-9]{4})([0-9]{4})([0-9]{3}[0-9Xx])$"),
)


def _canonical_orcid(value: str | None) -> str | None:
    if value is None:
        return None
    groups: tuple[str, ...] | None = None
    for pattern in _ORCID_FORMS:
        match = pattern.fullmatch(value)
        if match is not None:
            groups = match.groups()
            break
    if groups is None:
        return None
    digits = "".join(groups)
    digits = digits[:-1] + digits[-1].upper()
    total = 0
    for digit in digits[:15]:
        total = (total + int(digit)) * 2
    check = (12 - (total % 11)) % 11
    expected = "X" if check == 10 else str(check)
    if digits[-1] != expected:
        return None
    return (
        f"https://orcid.org/{digits[0:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:16]}"
    )


def _position_category(value: str | None) -> str:
    if value is None:
        return "null"
    if value in {"first", "middle", "last"}:
        return value
    return "unexpected_non_null"


def _year_and_token_records(
    pairs: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_author_year: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_author_year[(pair["author_numeric_id"], pair["publication_year"])].append(
            pair
        )
    years: list[dict[str, Any]] = []
    for (author, publication_year), events in sorted(by_author_year.items()):
        team_counts = {key: 0 for key in TEAM_BANDS}
        field_counts = {key: 0 for key in FIELD_KEYS}
        for event in events:
            team_counts[_team_band(event["declared_authors_count"])] += 1
            field_counts[event["primary_field_key"]] += 1
        _require(
            sum(team_counts.values()) == len(events),
            "team-band reduction did not close",
        )
        _require(
            sum(field_counts.values()) == len(events), "field reduction did not close"
        )
        years.append(
            {
                "author_numeric_id": author,
                "publication_year": publication_year,
                "unique_work_count": len(events),
                "team_band_unique_work_counts": team_counts,
                "primary_field_unique_work_counts": field_counts,
            }
        )
    by_author: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in years:
        by_author[record["author_numeric_id"]].append(record)
    tokens: list[dict[str, Any]] = []
    for author, records in sorted(by_author.items()):
        observed_years = [record["publication_year"] for record in records]
        annual_counts = [record["unique_work_count"] for record in records]
        first = min(observed_years)
        last = max(observed_years)
        span = last - first + 1
        tokens.append(
            {
                "author_numeric_id": author,
                "first_usable_publication_year": first,
                "last_usable_publication_year": last,
                "active_year_count": len(records),
                "inclusive_span_years": span,
                "inactive_gap_years": span - len(records),
                "total_unique_work_count": sum(annual_counts),
                "maximum_annual_unique_work_count": max(annual_counts),
            }
        )
    return years, tokens


def _cap_grid(
    pairs: Sequence[dict[str, Any]], *, view: str
) -> tuple[dict[str, int], dict[str, int]]:
    pair_counts: dict[str, int] = {}
    cells: dict[str, int] = {}
    for cap in TEAM_CAPS:
        cap_name = str(cap)
        selected = [
            pair
            for pair in pairs
            if cap == "ALL" or pair["declared_authors_count"] <= cap
        ]
        pair_counts[cap_name] = len(selected)
        _, tokens = _year_and_token_records(selected)
        for threshold in THRESHOLDS:
            cells[f"cap_{cap_name}_gt_{threshold}"] = sum(
                record["maximum_annual_unique_work_count"] > threshold
                for record in tokens
            )
    cap_values = [pair_counts[str(cap)] for cap in TEAM_CAPS]
    _require(
        cap_values == sorted(cap_values), f"{view} cap pair counts are not cumulative"
    )
    for cap in TEAM_CAPS:
        values = [cells[f"cap_{cap}_gt_{threshold}"] for threshold in THRESHOLDS]
        _require(
            values == sorted(values, reverse=True),
            f"{view} threshold counts are not monotone",
        )
    for threshold in THRESHOLDS:
        values = [cells[f"cap_{cap}_gt_{threshold}"] for cap in TEAM_CAPS]
        _require(
            values == sorted(values),
            f"{view} cap-conditioned threshold counts are not cumulative",
        )
    return pair_counts, cells


def _validate_and_reduce(
    fixture: dict[str, Any],
    *,
    config_sha256: str,
    fixture_sha256: str,
    pinned_fixture: bool,
    pinned_fixture_sha256_match: bool,
    foundation_verification: list[dict[str, Any]],
    runtime_fingerprint: dict[str, str],
) -> dict[str, Any]:
    top = _require_exact_keys(fixture, TOP_LEVEL_KEYS, field="fixture")
    _require(top["schema_version"] == FIXTURE_SCHEMA, "fixture schema version mismatch")
    _require(top["surface_kind"] == SURFACE_KIND, "fixture surface kind mismatch")
    _require(
        top["production_input_access_allowed"] is False,
        "fixture must prohibit production input access",
    )
    _require(
        isinstance(top["root_id"], str) and top["root_id"].startswith("synthetic-"),
        "fixture root ID mismatch",
    )
    _require(top["snapshot_date"] == SNAPSHOT_DATE, "fixture snapshot date mismatch")
    for key in ("main_rows", "spine_rows", "authorship_rows"):
        _require(isinstance(top[key], list), f"fixture {key} must be an array")
    _require(len(top["main_rows"]) <= MAX_MAIN_ROWS, "fixture main row cap exceeded")
    _require(len(top["spine_rows"]) <= MAX_SPINE_ROWS, "fixture spine row cap exceeded")
    _require(
        len(top["authorship_rows"]) <= MAX_AUTHORSHIP_ROWS,
        "fixture authorship row cap exceeded",
    )
    parents = _validate_parents(
        top["synthetic_parents"], spine_rows=len(top["spine_rows"])
    )

    main_by_id: dict[int, dict[str, Any]] = {}
    main_w_minus_1 = 0
    seen_main_raw: set[str] = set()
    for index, raw in enumerate(top["main_rows"]):
        row = _require_exact_keys(raw, MAIN_KEYS, field=f"main_rows[{index}]")
        identifier = row["id"]
        _require(isinstance(identifier, str), f"main_rows[{index}].id must be a string")
        _require(identifier not in seen_main_raw, "duplicate main work ID")
        seen_main_raw.add(identifier)
        authors_count = _require_int(
            row["authors_count"],
            field=f"main_rows[{index}].authors_count",
            minimum=0,
            maximum=UINT32_MAX,
        )
        field_id = _require_string_or_null(
            row["primary_topic__field__id"],
            field=f"main_rows[{index}].primary_topic__field__id",
        )
        if field_id is not None:
            field_match = FIELD_RE.fullmatch(field_id)
            _require(field_match is not None, "invalid primary field ID")
            field_key = field_match.group(1)
        else:
            field_key = "null"
        if identifier == W_MINUS_1:
            main_w_minus_1 += 1
            continue
        numeric = _parse_numeric_identifier(
            identifier, WORK_RE, field=f"main_rows[{index}].id"
        )
        _require(numeric not in main_by_id, "duplicate retained numeric work ID")
        main_by_id[numeric] = {
            "authors_count": authors_count,
            "primary_field_key": field_key,
        }

    spine_by_id: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(top["spine_rows"]):
        row = _require_exact_keys(raw, SPINE_KEYS, field=f"spine_rows[{index}]")
        numeric = _require_int(
            row["work_id_numeric"],
            field=f"spine_rows[{index}].work_id_numeric",
            minimum=0,
            maximum=UINT64_MAX,
        )
        _require(numeric not in spine_by_id, "duplicate spine numeric work ID")
        disposition = _require_int(
            row["temporal_disposition"],
            field=f"spine_rows[{index}].temporal_disposition",
            minimum=0,
            maximum=5,
        )
        day_value, year_value = _validate_day(
            disposition,
            row["publication_day"],
            row["publication_year"],
            field=f"spine_rows[{index}]",
        )
        spine_by_id[numeric] = {
            "publication_day": day_value,
            "publication_year": year_value,
            "temporal_disposition": disposition,
        }
    _require(
        set(spine_by_id) == set(main_by_id),
        "spine work ID set must equal retained main work ID set",
    )

    observed_slots: dict[int, int] = defaultdict(int)
    author_counts = {
        "null_author_slots": 0,
        "malformed_author_slots": 0,
        "grammar_valid_author_slots": 0,
        "a_zero_slots": 0,
        "positive_author_slots": 0,
    }
    authorship_w_minus_1 = 0
    pair_sources: dict[tuple[int, int], dict[str, Any]] = {}
    for index, raw in enumerate(top["authorship_rows"]):
        row = _require_exact_keys(
            raw, AUTHORSHIP_KEYS, field=f"authorship_rows[{index}]"
        )
        work_id = row["id"]
        _require(
            isinstance(work_id, str), f"authorship_rows[{index}].id must be a string"
        )
        author_id = _require_string_or_null(
            row["authorships__author__id"], field=f"authorship_rows[{index}].author_id"
        )
        orcid = _require_string_or_null(
            row["authorships__author__orcid"], field=f"authorship_rows[{index}].orcid"
        )
        raw_name = _require_string_or_null(
            row["authorships__raw_author_name"],
            field=f"authorship_rows[{index}].raw_name",
        )
        display_name = _require_string_or_null(
            row["authorships__author__display_name"],
            field=f"authorship_rows[{index}].display_name",
        )
        position = _require_string_or_null(
            row["authorships__author_position"],
            field=f"authorship_rows[{index}].position",
        )
        if work_id == W_MINUS_1:
            authorship_w_minus_1 += 1
            continue
        work_numeric = _parse_numeric_identifier(
            work_id, WORK_RE, field=f"authorship_rows[{index}].id"
        )
        _require(
            work_numeric in main_by_id,
            "authorship row does not join a retained main work",
        )
        observed_slots[work_numeric] += 1
        if author_id is None:
            author_counts["null_author_slots"] += 1
            continue
        if AUTHOR_RE.fullmatch(author_id) is None:
            author_counts["malformed_author_slots"] += 1
            raise ValidationError("malformed non-null author ID blocks the component")
        author_numeric = _parse_numeric_identifier(
            author_id,
            AUTHOR_RE,
            field=f"authorship_rows[{index}].author_id",
        )
        author_counts["grammar_valid_author_slots"] += 1
        if author_numeric == 0:
            author_counts["a_zero_slots"] += 1
        else:
            author_counts["positive_author_slots"] += 1
        key = (author_numeric, work_numeric)
        if key not in pair_sources:
            pair_sources[key] = {
                "multiplicity": 0,
                "orcids": set(),
                "raw_names": set(),
                "display_names": set(),
                "positions": set(),
            }
        source = pair_sources[key]
        source["multiplicity"] += 1
        source["orcids"].add(orcid)
        source["raw_names"].add(raw_name)
        source["display_names"].add(display_name)
        source["positions"].add(position)

    declared_slots = sum(row["authors_count"] for row in main_by_id.values())
    mismatches = sum(
        observed_slots[work_id] != row["authors_count"]
        for work_id, row in main_by_id.items()
    )
    _require(
        mismatches == 0, "declared authors_count does not equal observed raw slots"
    )
    _require(
        sum(observed_slots.values())
        == len(top["authorship_rows"]) - authorship_w_minus_1,
        "authorship accounting did not close",
    )

    complete = partial = none_usable = zero_rows = 0
    valid_slots_by_work: dict[int, int] = defaultdict(int)
    # Count source rows, not distinct pairs, for the work class closure.
    for (_, work_numeric), source in pair_sources.items():
        valid_slots_by_work[work_numeric] += source["multiplicity"]
    for work_numeric, row in main_by_id.items():
        raw_slots = observed_slots[work_numeric]
        valid_slots = valid_slots_by_work[work_numeric]
        if raw_slots == 0:
            zero_rows += 1
        elif valid_slots == raw_slots:
            complete += 1
        elif valid_slots == 0:
            none_usable += 1
        else:
            partial += 1

    pair_records: list[dict[str, Any]] = []
    position_category_conflicts = 0
    raw_position_conflicts = 0
    combined_position_conflicts = 0
    for (author_numeric, work_numeric), source in sorted(pair_sources.items()):
        spine = spine_by_id[work_numeric]
        main = main_by_id[work_numeric]
        non_null_positions = {
            value for value in source["positions"] if value is not None
        }
        categories = {_position_category(value) for value in source["positions"]}
        non_null_categories = {value for value in categories if value != "null"}
        category_conflict = len(non_null_categories) > 1
        raw_conflict = len(non_null_positions) > 1
        combined = category_conflict or raw_conflict
        position_category_conflicts += category_conflict
        raw_position_conflicts += raw_conflict
        combined_position_conflicts += combined
        canonical_orcids = sorted(
            {
                canonical
                for value in source["orcids"]
                if (canonical := _canonical_orcid(value)) is not None
            }
        )
        non_null_orcids = {value for value in source["orcids"] if value is not None}
        non_null_raw_names = {
            value for value in source["raw_names"] if value is not None
        }
        non_null_display_names = {
            value for value in source["display_names"] if value is not None
        }
        normalized_raw_names = {_normalize_name(value) for value in source["raw_names"]}
        normalized_display_names = {
            _normalize_name(value) for value in source["display_names"]
        }
        conflicts = {
            "raw_orcid_conflict": len(non_null_orcids) > 1,
            "canonical_orcid_conflict": len(canonical_orcids) > 1,
            "raw_author_name_conflict": len(non_null_raw_names) > 1,
            "display_name_conflict": len(non_null_display_names) > 1,
            "normalized_raw_name_conflict": len(
                {value for value in normalized_raw_names if value is not None}
            )
            > 1,
            "normalized_display_name_conflict": len(
                {value for value in normalized_display_names if value is not None}
            )
            > 1,
            "position_category_conflict": category_conflict,
            "raw_position_conflict": raw_conflict,
            "position_conflict": combined,
        }
        diagnostic = {
            "author_numeric_id": author_numeric,
            "work_numeric_id": work_numeric,
            "raw_multiplicity": source["multiplicity"],
            "temporal_disposition": spine["temporal_disposition"],
            "history_usable": spine["temporal_disposition"] in HISTORY_USABLE,
            "exact_raw_orcid_set": _sorted_tags(source["orcids"]),
            "canonical_valid_orcid_set": canonical_orcids,
            "exact_raw_author_name_set": _sorted_tags(source["raw_names"]),
            "exact_display_name_set": _sorted_tags(source["display_names"]),
            "normalized_raw_author_name_set": _sorted_tags(normalized_raw_names),
            "normalized_display_name_set": _sorted_tags(normalized_display_names),
            "author_position_category_set": sorted(
                categories,
                key=("first", "middle", "last", "null", "unexpected_non_null").index,
            ),
            "exact_raw_position_set": _sorted_tags(source["positions"]),
            "conflict_flags": conflicts,
        }
        pair_records.append(
            {
                **diagnostic,
                "publication_year": spine["publication_year"],
                "declared_authors_count": main["authors_count"],
                "primary_field_key": main["primary_field_key"],
            }
        )

    disposition_counts = {code: 0 for code in range(6)}
    for record in pair_records:
        disposition_counts[record["temporal_disposition"]] += 1
    usable_pairs = [record for record in pair_records if record["history_usable"]]
    diagnostic_keys = {
        "author_numeric_id",
        "work_numeric_id",
        "raw_multiplicity",
        "temporal_disposition",
        "history_usable",
        "exact_raw_orcid_set",
        "canonical_valid_orcid_set",
        "exact_raw_author_name_set",
        "exact_display_name_set",
        "normalized_raw_author_name_set",
        "normalized_display_name_set",
        "author_position_category_set",
        "exact_raw_position_set",
        "conflict_flags",
    }
    primary_diagnostics = [
        {key: record[key] for key in diagnostic_keys} for record in pair_records
    ]
    usable_diagnostics = [
        {key: record[key] for key in diagnostic_keys} for record in usable_pairs
    ]
    _require(
        all(record["publication_year"] is not None for record in usable_pairs),
        "usable history pair lacks publication year",
    )
    base_years, base_tokens = _year_and_token_records(usable_pairs)
    positive_pairs = [
        record for record in usable_pairs if record["author_numeric_id"] > 0
    ]
    positive_years, positive_tokens = _year_and_token_records(positive_pairs)
    _require(
        positive_years
        == [record for record in base_years if record["author_numeric_id"] > 0],
        "positive author-year view is not an exact base projection",
    )
    _require(
        positive_tokens
        == [record for record in base_tokens if record["author_numeric_id"] > 0],
        "positive token view is not an exact base projection",
    )

    base_cap_counts, base_cells = _cap_grid(usable_pairs, view="base")
    positive_cap_counts, positive_cells = _cap_grid(
        positive_pairs, view="positive_author_id"
    )
    for cap in TEAM_CAPS:
        _require(
            positive_cap_counts[str(cap)] <= base_cap_counts[str(cap)],
            "positive cap count exceeds base",
        )
    for key in base_cells:
        _require(
            positive_cells[key] <= base_cells[key], "positive hyper cell exceeds base"
        )
    _require(base_cap_counts["ALL"] == len(usable_pairs), "base ALL cap count mismatch")
    _require(
        positive_cap_counts["ALL"] == len(positive_pairs),
        "positive ALL cap count mismatch",
    )

    distinct_base_tokens = {record["author_numeric_id"] for record in pair_records}
    distinct_positive_tokens = {value for value in distinct_base_tokens if value > 0}
    evidence_class = PINNED_EVIDENCE_CLASS if pinned_fixture else GENERIC_EVIDENCE_CLASS
    parent_evidence_class = (
        PINNED_PARENT_CLASS if pinned_fixture else GENERIC_PARENT_CLASS
    )
    status = PINNED_STATUS if pinned_fixture else GENERIC_STATUS
    claim_boundary = PINNED_CLAIM_BOUNDARY if pinned_fixture else GENERIC_CLAIM_BOUNDARY
    _require_exact_keys(
        runtime_fingerprint,
        {
            "python_implementation",
            "python_version",
            "unicodedata_unidata_version",
            "canonical_json_abi",
        },
        field="runtime fingerprint",
    )
    _require(
        runtime_fingerprint["canonical_json_abi"] == CANONICAL_JSON_ABI,
        "runtime canonical JSON ABI mismatch",
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "surface_kind": SURFACE_KIND,
        "status": status,
        "evidence_class": evidence_class,
        "production_status": PRODUCTION_STATUS,
        "inputs": {
            "config_sha256": config_sha256,
            "fixture_sha256": fixture_sha256,
            "runtime_fingerprint": runtime_fingerprint,
            "foundation_verification": foundation_verification,
        },
        "scope": {
            "main_rows": len(top["main_rows"]),
            "spine_rows": len(top["spine_rows"]),
            "authorship_rows": len(top["authorship_rows"]),
            "gpu_used": False,
            "fixture_evidence_class": evidence_class,
            "parent_evidence_class": parent_evidence_class,
            "production_source_interface_used": False,
            "production_input_accessed": False if pinned_fixture else None,
            "pinned_fixture_sha256_match": pinned_fixture_sha256_match,
            "files_written": 0,
            "scientific_effect_estimation_performed": False,
        },
        "parents": parents,
        "source_accounting": {
            "main_rows": len(top["main_rows"]),
            "spine_rows": len(top["spine_rows"]),
            "authorship_rows": len(top["authorship_rows"]),
            "matched_main_spine_rows": len(main_by_id),
            "duplicate_main_rows": 0,
            "duplicate_spine_rows": 0,
            "matched_authorship_rows": len(top["authorship_rows"])
            - authorship_w_minus_1,
            "unmatched_authorship_rows": 0,
        },
        "work_and_slot_accounting": {
            "retained_work_rows": len(main_by_id),
            "main_w_minus_1_rows": main_w_minus_1,
            "authorship_w_minus_1_rows": authorship_w_minus_1,
            "total_w_minus_1_rows": main_w_minus_1 + authorship_w_minus_1,
            "w_zero_rows": int(0 in main_by_id),
            "declared_slots": declared_slots,
            "observed_slots": sum(observed_slots.values()),
            "authors_count_mismatches": 0,
            "complete_works": complete,
            "partial_works": partial,
            "none_usable_works": none_usable,
            "zero_authorship_row_works": zero_rows,
        },
        "author_id_accounting": {
            **author_counts,
            "distinct_base_tokens": len(distinct_base_tokens),
            "distinct_positive_tokens": len(distinct_positive_tokens),
        },
        "pair_accounting": {
            "grammar_valid_author_slots": author_counts["grammar_valid_author_slots"],
            "distinct_author_work_pairs": len(pair_records),
            "duplicate_pair_row_excess": author_counts["grammar_valid_author_slots"]
            - len(pair_records),
            "position_category_conflict_pairs": position_category_conflicts,
            "raw_position_conflict_pairs": raw_position_conflicts,
            "combined_position_conflict_pairs": combined_position_conflicts,
            "primary_population_aggregate_sha256": _domain_hash(
                AGGREGATE_DOMAINS["primary_pair"], primary_diagnostics
            ),
            "history_usable_population_aggregate_sha256": _domain_hash(
                AGGREGATE_DOMAINS["usable_pair"], usable_diagnostics
            ),
        },
        "temporal_accounting": {
            **{
                f"disposition_{code}_pairs": disposition_counts[code]
                for code in range(6)
            },
            "history_usable_pairs": len(usable_pairs),
            "excluded_pairs": len(pair_records) - len(usable_pairs),
        },
        "history_accounting": {
            "base_view_tokens": len(base_tokens),
            "positive_view_tokens": len(positive_tokens),
            "base_view_unique_pairs": len(usable_pairs),
            "positive_view_unique_pairs": len(positive_pairs),
            "base_view_per_year_aggregate_sha256": _domain_hash(
                AGGREGATE_DOMAINS["base_year"], base_years
            ),
            "positive_view_per_year_aggregate_sha256": _domain_hash(
                AGGREGATE_DOMAINS["positive_year"], positive_years
            ),
            "base_view_per_token_aggregate_sha256": _domain_hash(
                AGGREGATE_DOMAINS["base_token"], base_tokens
            ),
            "positive_view_per_token_aggregate_sha256": _domain_hash(
                AGGREGATE_DOMAINS["positive_token"], positive_tokens
            ),
        },
        "team_and_hyper_sensitivity": {
            "base_view_cap_pair_counts": base_cap_counts,
            "positive_author_id_view_cap_pair_counts": positive_cap_counts,
            "base_view_cap_threshold_token_grid": base_cells,
            "positive_author_id_view_cap_threshold_token_grid": positive_cells,
            "base_view_grid_sha256": _domain_hash(
                AGGREGATE_DOMAINS["base_grid"],
                {
                    "view": "base",
                    "cap_pair_counts": base_cap_counts,
                    "cap_threshold_token_grid": base_cells,
                },
            ),
            "positive_author_id_view_grid_sha256": _domain_hash(
                AGGREGATE_DOMAINS["positive_grid"],
                {
                    "view": "positive_author_id",
                    "cap_pair_counts": positive_cap_counts,
                    "cap_threshold_token_grid": positive_cells,
                },
            ),
        },
        "gate_status": {
            "production_execution_authorized": False,
            "production_parity_established": False,
            "g_minus_1_data_readiness": "PENDING",
            "g0_measurement_freeze": "PENDING",
            "empirical_g1_signature": "NOT_RUN",
            "pass_g1": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": claim_boundary,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--synthetic-fixture", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config, config_payload, config_identity, foundation_verification = _load_config(
            args.config
        )
        fixture, fixture_payload, resolved_fixture_path = _load_fixture(
            args.synthetic_fixture,
            config_identity=config_identity,
            production_root_id=PRODUCTION_ROOT_ID,
        )
        fixture_sha256 = _sha256(fixture_payload)
        pinned_fixture, pinned_fixture_sha256_match = _classify_fixture(
            config,
            resolved_fixture_path=resolved_fixture_path,
            fixture_sha256=fixture_sha256,
        )
        report = _validate_and_reduce(
            fixture,
            config_sha256=_sha256(config_payload),
            fixture_sha256=fixture_sha256,
            pinned_fixture=pinned_fixture,
            pinned_fixture_sha256_match=pinned_fixture_sha256_match,
            foundation_verification=foundation_verification,
            runtime_fingerprint=_runtime_fingerprint(),
        )
    except (OSError, ValidationError) as exc:
        print(f"author-history v2 fixture validation failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
