#!/usr/bin/env python3
"""Run the OpenAlex citation chronology contract on synthetic fixtures only.

This executable is an implementation diagnostic, not the production audit.  It
has no production source-root option and never emits the future production
component status or receipt schema.  The two temporal-input modes, directed-pair
hash ABI, exact deduplication, seven-bucket classifier, accounting closures, and
receipt-last lifecycle are exercised on a closed JSON fixture.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/openalex_g1_citation_available_at_v0.yaml"
)

FIXTURE_SCHEMA = "openalex_g1_citation_edge_chronology_synthetic_fixture.v1"
CONFIG_SCHEMA = "openalex_g1_citation_available_at_design.v1"
PLAN_SCHEMA = "openalex_g1_citation_edge_chronology_synthetic_plan.v1"
CHECKPOINT_SCHEMA = "openalex_g1_citation_edge_chronology_synthetic_checkpoint.v1"
REPORT_SCHEMA = "openalex_g1_citation_edge_chronology_synthetic_report.v1"
STATE_SCHEMA = "openalex_g1_citation_edge_chronology_synthetic_state.v1"
OUTPUTS_SCHEMA = "openalex_g1_citation_edge_chronology_synthetic_outputs.v1"
PRODUCTION_OUTPUTS_SCHEMA = "openalex_g1_citation_edge_chronology_outputs.v1"

SYNTHETIC_STATUS = "SYNTHETIC_CITATION_EDGE_CHRONOLOGY_ONLY"
PRODUCTION_STATUS = "PRODUCTION_NOT_RUN"
SYNTHETIC_SURFACE = "SYNTHETIC_FIXTURE_ONLY"
SYNTHETIC_COMPLETION_CONTRACT = (
    "SYNTHETIC_RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
)
PRODUCTION_COMPLETION_CONTRACT = "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
FULL_STATUS = (
    "PASS_CURRENT_SNAPSHOT_CITATION_EDGE_CHRONOLOGY_ACCOUNTING_WITH_"
    "BIBLIOGRAPHIC_EVENT_TIME_CONVENTION_LIMITATION"
)

DIRECT_MODE = "exact_temporal_reconstruction_from_current_source"
REUSE_MODE = "canonical_work_time_spine_reuse"
ALLOWED_MODES = (DIRECT_MODE, REUSE_MODE)
EXECUTION_ABI = {
    DIRECT_MODE: "CITATION_DIRECT_RECONSTRUCTION_V1",
    REUSE_MODE: "CITATION_CANONICAL_SPINE_REUSE_V1",
}
MODE_TIME_KEY = {
    DIRECT_MODE: "direct_work_times",
    REUSE_MODE: "spine_work_times",
}
FORBIDDEN_MODE_TIME_KEY = {
    DIRECT_MODE: "spine_work_times",
    REUSE_MODE: "direct_work_times",
}
MODE_PARENT_ORDER = {
    DIRECT_MODE: (
        "current_content_baseline",
        "main_temporal_validity",
        "work_source_membership_regression",
    ),
    REUSE_MODE: (
        "current_content_baseline",
        "main_temporal_validity",
        "work_source_membership_regression",
        "canonical_work_time_spine",
    ),
}

PAIR_BUCKETS = (
    "self_loop",
    "source_temporal_unusable",
    "target_external_open_world",
    "target_temporal_unusable",
    "negative_lag_target_after_source",
    "same_day",
    "positive_lag",
)
CROSS_TAB_AXES = {
    "source_temporal_disposition": ("primary_bucket_x_source_temporal_disposition"),
    "target_membership": "primary_bucket_x_target_membership",
    "target_temporal_disposition_or_external_na": (
        "primary_bucket_x_target_temporal_disposition_or_external_na"
    ),
    "self_loop_flag": "primary_bucket_x_self_loop_flag",
    "signed_lag_class_or_na": "primary_bucket_x_signed_lag_class_or_na",
    "pair_multiplicity_class": "primary_bucket_x_pair_multiplicity_class",
}
USABLE_DISPOSITIONS = frozenset({2, 3})
UINT64_MAX = 2**64 - 1
MAX_SYNTHETIC_FIXTURE_BYTES = 1_048_576
MAX_SYNTHETIC_WORK_TIME_ROWS = 10_000
MAX_SYNTHETIC_REFERENCE_PAIR_ROWS = 10_000
SNAPSHOT_DATE = date(2026, 3, 30)

RECEIPT_BASENAME = "citation_edge_chronology_audit.outputs.json"
REPORT_BASENAME = "citation_edge_chronology_audit.json"
MARKDOWN_BASENAME = "citation_edge_chronology_audit.md"
EXPECTED_RECEIPT_ARTIFACTS = (
    "audit_plan.json",
    "run_state.json",
    "checkpoints/synthetic_accounting.json",
    REPORT_BASENAME,
    MARKDOWN_BASENAME,
)
EXPECTED_FINAL_FILES = frozenset((*EXPECTED_RECEIPT_ARTIFACTS, RECEIPT_BASENAME))

CLAIM_BOUNDARY = (
    "This SYNTHETIC_FIXTURE_ONLY report exercises current contract mechanics. "
    "It does not read production OpenAlex inputs, establish any production "
    "parent, emit the future production receipt ABI, establish edge observation "
    "time or a historical graph, choose a maturation policy, establish G-1 or "
    "G0, run empirical G1, or establish an NABM or scientific effect."
)


class CitationSyntheticAuditError(RuntimeError):
    """Raised when the synthetic citation contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CitationSyntheticAuditError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_loads(payload: bytes, *, field: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CitationSyntheticAuditError(
                    f"duplicate JSON key in {field}: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CitationSyntheticAuditError(
            f"non-finite JSON constant in {field}: {value}"
        )

    try:
        return json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CitationSyntheticAuditError(f"invalid JSON for {field}") from exc


def _regular_bytes(path: Path, *, field: str) -> bytes:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise CitationSyntheticAuditError(f"missing {field}") from exc
    require(stat.S_ISREG(info.st_mode), f"{field} must be a regular file")
    require(not stat.S_ISLNK(info.st_mode), f"{field} must not be a symlink")
    with path.open("rb") as handle:
        payload = handle.read()
    after = path.lstat()
    require(
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"{field} changed while read",
    )
    return payload


def _read_json_regular(path: Path, *, field: str) -> dict[str, Any]:
    value = _strict_json_loads(_regular_bytes(path, field=field), field=field)
    require(isinstance(value, dict), f"{field} must be a JSON object")
    return value


def _read_config_snapshot(path: Path) -> tuple[dict[str, Any], bytes, str]:
    payload = _regular_bytes(path, field="citation config")
    try:
        value = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise CitationSyntheticAuditError("invalid citation config YAML") from exc
    require(isinstance(value, dict), "citation config must be a mapping")
    return value, payload, sha256_bytes(payload)


def _load_fixture_payload(path: Path) -> tuple[dict[str, Any], bytes, str]:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise CitationSyntheticAuditError("missing synthetic fixture") from exc
    require(
        stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
        "synthetic fixture must be a regular non-symlink file",
    )
    require(
        info.st_size <= MAX_SYNTHETIC_FIXTURE_BYTES,
        "synthetic fixture exceeds the hard byte cap",
    )
    payload = _regular_bytes(path, field="synthetic fixture")
    value = _strict_json_loads(payload, field="synthetic fixture")
    require(isinstance(value, dict), "synthetic fixture must be a JSON object")
    return value, payload, sha256_bytes(payload)


def _revalidate_execution_inputs(
    *,
    config_path: Path,
    config_sha256: str,
    fixture_path: Path,
    fixture_sha256: str,
    tool_sha256: str,
) -> None:
    require(
        sha256_bytes(_regular_bytes(config_path, field="final citation config"))
        == config_sha256,
        "citation config changed during synthetic execution",
    )
    _, _, final_fixture_sha256 = _load_fixture_payload(fixture_path)
    require(
        final_fixture_sha256 == fixture_sha256,
        "synthetic fixture changed during execution",
    )
    require(
        sha256_bytes(_regular_bytes(Path(__file__), field="final citation tool"))
        == tool_sha256,
        "citation tool changed during synthetic execution",
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    path.chmod(0o600)
    directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _artifact_contract(path: Path, output_dir: Path) -> dict[str, Any]:
    payload = _regular_bytes(path, field="synthetic output artifact")
    return {
        "relative_path": path.relative_to(output_dir).as_posix(),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    require(
        config.get("schema_version") == CONFIG_SCHEMA,
        "citation config schema mismatch",
    )
    require(
        config.get("source", {}).get("snapshot_date") == SNAPSHOT_DATE.isoformat(),
        "citation snapshot date drift",
    )
    lifecycle = config.get("lifecycle")
    require(isinstance(lifecycle, dict), "missing lifecycle contract")
    require(
        lifecycle.get("design_status")
        == "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN",
        "citation implementation lifecycle mismatch",
    )
    require(
        lifecycle.get("implementation_status") == "IMPLEMENTED_SYNTHETICALLY",
        "citation implementation status mismatch",
    )
    require(
        lifecycle.get("production_status") == PRODUCTION_STATUS,
        "production status drift",
    )
    require(lifecycle.get("executable") is True, "synthetic executable flag drift")
    require(
        lifecycle.get("synthetic_fixture_only") is True, "synthetic-only boundary drift"
    )
    require(
        lifecycle.get("production_execution_supported") is False,
        "production execution must remain unsupported",
    )
    require(
        lifecycle.get("production_source_access_supported") is False,
        "production source access must remain unsupported",
    )
    require(
        lifecycle.get("production_result_exists") is False, "production result drift"
    )
    require(
        lifecycle.get("pass_result_allowed_in_design_artifact") is False,
        "full pass must remain forbidden",
    )

    modes = config.get("temporal_input_modes", {}).get("allowed_modes")
    require(set(modes or []) == set(ALLOWED_MODES), "temporal input modes drift")
    require(
        tuple(config.get("exclusive_pair_bucket_policy", {}).get("precedence", []))
        == PAIR_BUCKETS,
        "seven-bucket precedence drift",
    )
    require(
        config.get("exclusive_pair_bucket_policy", {}).get("required_cross_tab_axes")
        == list(CROSS_TAB_AXES.values()),
        "required cross-tab axes drift",
    )
    availability = config.get("available_at_policy", {})
    require(
        availability.get("resolution") == "BIBLIOGRAPHIC_EVENT_TIME_CONVENTION",
        "available_at resolution drift",
    )
    require(
        availability.get("edge_available_at_field") == "citing_work_publication_date",
        "available_at field drift",
    )
    require(
        availability.get("usable_temporal_disposition_codes") == [2, 3],
        "usable temporal dispositions drift",
    )
    require(
        availability.get("openalex_observation_time_established") is False,
        "OpenAlex observation-time boundary drift",
    )
    bucket_contract = config.get("pilot_and_full_rules", {}).get(
        "deterministic_hash_bucket", {}
    )
    require(
        bucket_contract.get("canonical_key_bytes")
        == "uint64_citing_id_big_endian_then_uint64_cited_id_big_endian",
        "pair bucket key ABI drift",
    )
    require(bucket_contract.get("hash_function") == "SHA256", "pair hash ABI drift")
    require(
        bucket_contract.get("bucket_integer")
        == "unsigned_big_endian_first_8_digest_bytes",
        "pair bucket integer ABI drift",
    )
    require(
        config.get("future_success_contract", {}).get("full_component_status")
        == FULL_STATUS,
        "future production component status drift",
    )
    require(
        config.get("completion_contract", {}).get("future_receipt_schema_version")
        == PRODUCTION_OUTPUTS_SCHEMA,
        "future production receipt schema drift",
    )
    synthetic = config.get("synthetic_execution_contract")
    require(isinstance(synthetic, dict), "missing synthetic execution contract")
    require(
        synthetic.get("fixture_schema_version") == FIXTURE_SCHEMA,
        "fixture schema drift",
    )
    require(
        synthetic.get("plan_schema_version") == PLAN_SCHEMA,
        "synthetic plan schema drift",
    )
    require(
        synthetic.get("checkpoint_schema_version") == CHECKPOINT_SCHEMA,
        "synthetic checkpoint schema drift",
    )
    require(
        synthetic.get("report_schema_version") == REPORT_SCHEMA,
        "synthetic report schema drift",
    )
    require(
        synthetic.get("receipt_schema_version") == OUTPUTS_SCHEMA,
        "synthetic receipt schema drift",
    )
    require(
        synthetic.get("receipt_completion_contract") == SYNTHETIC_COMPLETION_CONTRACT,
        "synthetic completion contract drift",
    )
    require(
        synthetic.get("report_status") == SYNTHETIC_STATUS, "synthetic status drift"
    )
    require(
        synthetic.get("full_component_status_allowed") is False,
        "synthetic promotion drift",
    )
    require(
        synthetic.get("production_input_access_allowed") is False,
        "production input boundary drift",
    )
    require(
        tuple(synthetic.get("exact_receipt_artifacts", []))
        == EXPECTED_RECEIPT_ARTIFACTS,
        "synthetic receipt artifact set drift",
    )
    require(
        synthetic.get("hard_max_fixture_bytes") == MAX_SYNTHETIC_FIXTURE_BYTES,
        "synthetic fixture byte cap drift",
    )
    require(
        synthetic.get("hard_max_work_time_rows") == MAX_SYNTHETIC_WORK_TIME_ROWS,
        "synthetic work-time row cap drift",
    )
    require(
        synthetic.get("hard_max_reference_pair_rows")
        == MAX_SYNTHETIC_REFERENCE_PAIR_ROWS,
        "synthetic pair row cap drift",
    )
    require(
        synthetic.get("work_time_publication_day_epoch") == "1970-01-01",
        "synthetic publication-day epoch drift",
    )
    require(
        synthetic.get("work_time_day_year_exact_match_required") is True,
        "synthetic day/year validation drift",
    )
    require(
        synthetic.get("work_time_disposition_must_match_snapshot_classifier") is True,
        "synthetic disposition validation drift",
    )


def _hex64(value: Any, *, field: str) -> str:
    require(isinstance(value, str), f"{field} must be a SHA-256 string")
    require(len(value) == 64, f"{field} must contain 64 hex characters")
    require(
        all(character in "0123456789abcdef" for character in value),
        f"{field} is not lowercase hex",
    )
    return value


def _validate_parent_fixture(mode: str, raw: Any) -> dict[str, Any]:
    require(isinstance(raw, list), "fixture parents must be an ordered list")
    require(
        all(isinstance(item, dict) for item in raw),
        "fixture parent entries must be objects",
    )
    parents = [dict(item) for item in raw]
    names = tuple(parent.get("name") for parent in parents)
    require(names == MODE_PARENT_ORDER[mode], "mode-conditioned parent order mismatch")
    indexed = {str(parent["name"]): parent for parent in parents}

    content = indexed["current_content_baseline"]
    require(
        content.get("synthetic_contract_fixture_valid") is True,
        "content fixture is not validated",
    )
    require(
        content.get("status") == "PASS_CURRENT_CONTENT_BASELINE_COMPONENT",
        "content status mismatch",
    )
    require(content.get("state") == "COMPLETE", "content state mismatch")
    require(
        content.get("completion_contract")
        == "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST",
        "content completion contract mismatch",
    )
    require(content.get("artifact_tree_valid") is True, "content artifact tree invalid")
    require(content.get("full_scope") is True, "content parent must model full scope")
    _hex64(content.get("result_fingerprint_sha256"), field="content result fingerprint")

    temporal = indexed["main_temporal_validity"]
    require(
        temporal.get("synthetic_contract_fixture_valid") is True,
        "temporal fixture is not validated",
    )
    require(
        temporal.get("status")
        == "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_VALIDITY_COMPONENT_WITH_FROZEN_EXCLUSIONS",
        "temporal status mismatch",
    )
    require(
        temporal.get("state") == "COMPLETE_PENDING_RECEIPT", "temporal state mismatch"
    )
    require(
        temporal.get("completion_contract") == PRODUCTION_COMPLETION_CONTRACT,
        "temporal completion contract mismatch",
    )
    require(
        temporal.get("artifact_tree_valid") is True, "temporal artifact tree invalid"
    )
    require(
        temporal.get("receipt_written_last") is True, "temporal receipt order invalid"
    )
    require(
        temporal.get("final_disk_validated") is True,
        "temporal final validation missing",
    )
    require(
        temporal.get("embedded_content_matches") is True,
        "temporal content cross-binding mismatch",
    )
    require(temporal.get("full_scope") is True, "temporal parent must model full scope")
    _hex64(
        temporal.get("result_fingerprint_sha256"), field="temporal result fingerprint"
    )

    membership = indexed["work_source_membership_regression"]
    require(
        membership.get("synthetic_contract_fixture_valid") is True,
        "membership fixture is not validated",
    )
    require(
        membership.get("status")
        == "PASS_WORK_SOURCE_MEMBERSHIP_AND_PROFILE_AUDIT_WITH_FROZEN_QUARANTINE",
        "membership status mismatch",
    )
    require(
        membership.get("artifact_hash_valid") is True,
        "membership artifact hash invalid",
    )
    require(
        membership.get("anchors_reproduced_on_synthetic_current_bytes") is True,
        "membership regression is not reproduced in fixture",
    )
    _hex64(
        membership.get("result_fingerprint_sha256"),
        field="membership result fingerprint",
    )

    if mode == REUSE_MODE:
        spine = indexed["canonical_work_time_spine"]
        require(
            spine.get("synthetic_contract_fixture_valid") is True,
            "spine fixture is not validated",
        )
        require(
            spine.get("status")
            == "PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_INHERITED_TEMPORAL_EXCLUSIONS",
            "spine status mismatch",
        )
        require(
            spine.get("state") == "COMPLETE_PENDING_RECEIPT", "spine state mismatch"
        )
        require(
            spine.get("completion_contract") == PRODUCTION_COMPLETION_CONTRACT,
            "spine completion contract mismatch",
        )
        require(spine.get("artifact_tree_valid") is True, "spine artifact tree invalid")
        require(
            spine.get("receipt_written_last") is True, "spine receipt order invalid"
        )
        require(
            spine.get("final_disk_validated") is True, "spine final validation missing"
        )
        require(
            spine.get("embedded_parents_match") is True,
            "spine parent cross-binding mismatch",
        )
        require(
            spine.get("partition_files") == 4096,
            "spine partition-file contract mismatch",
        )
        require(spine.get("rows") == 492293206, "spine row contract mismatch")
        _hex64(spine.get("result_fingerprint_sha256"), field="spine result fingerprint")
        _hex64(spine.get("data_tree_sha256"), field="spine data tree")

    return {"ordered_names": list(names), "parents": parents}


def _uint64(value: Any, *, field: str, positive: bool = False) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{field} must be an integer",
    )
    minimum = 1 if positive else 0
    require(minimum <= value <= UINT64_MAX, f"{field} is outside uint64 contract")
    return value


def pair_hash_bucket(source_id: int, target_id: int, bucket_count: int) -> int:
    """Return the frozen SHA-256 bucket for an exact directed uint64 pair."""

    source = _uint64(source_id, field="citing work ID")
    target = _uint64(target_id, field="cited work ID")
    require(
        isinstance(bucket_count, int)
        and not isinstance(bucket_count, bool)
        and bucket_count > 1,
        "bucket count must be greater than one",
    )
    digest = hashlib.sha256(struct.pack(">QQ", source, target)).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % bucket_count


def _parse_work_times(raw: Any) -> dict[int, tuple[int | None, int | None, int]]:
    require(isinstance(raw, list), "selected work-time payload must be a list")
    require(
        len(raw) <= MAX_SYNTHETIC_WORK_TIME_ROWS,
        "synthetic work-time payload exceeds the hard row cap",
    )
    result: dict[int, tuple[int | None, int | None, int]] = {}
    for index, item in enumerate(raw):
        require(isinstance(item, dict), f"work-time row {index} must be an object")
        require(
            set(item)
            == {
                "work_id_numeric",
                "publication_day",
                "publication_year",
                "temporal_disposition",
            },
            f"work-time row {index} schema mismatch",
        )
        work_id = _uint64(item["work_id_numeric"], field=f"work-time row {index} ID")
        require(work_id not in result, "work-time IDs must be globally unique")
        disposition = item["temporal_disposition"]
        require(
            isinstance(disposition, int)
            and not isinstance(disposition, bool)
            and 0 <= disposition <= 5,
            f"work-time row {index} disposition mismatch",
        )
        day = item["publication_day"]
        year = item["publication_year"]
        if disposition == 0:
            require(
                day is None and year is None, "disposition 0 requires null day and year"
            )
        else:
            require(
                isinstance(day, int) and not isinstance(day, bool),
                "publication day must be int32",
            )
            require(-(2**31) <= day <= 2**31 - 1, "publication day is outside int32")
            require(
                isinstance(year, int) and not isinstance(year, bool),
                "publication year must be uint16",
            )
            require(1 <= year <= 9999, "publication year is outside calendar range")
            try:
                publication_date = date(1970, 1, 1) + timedelta(days=day)
            except (OverflowError, ValueError) as exc:
                raise CitationSyntheticAuditError(
                    "publication day is outside the supported calendar"
                ) from exc
            require(
                publication_date.year == year,
                "publication day/year exact match failed",
            )
            if publication_date < date(1900, 1, 1):
                expected_disposition = 1
            elif publication_date.year < SNAPSHOT_DATE.year:
                expected_disposition = 2
            elif publication_date.year == SNAPSHOT_DATE.year:
                expected_disposition = 3 if publication_date <= SNAPSHOT_DATE else 4
            else:
                expected_disposition = 5
            require(
                disposition == expected_disposition,
                "publication temporal disposition is inconsistent with snapshot",
            )
        result[work_id] = (day, year, disposition)
    require(result, "selected work-time payload must not be empty")
    return result


def _parse_pairs(raw: Any) -> list[tuple[int, int, str]]:
    require(isinstance(raw, list), "reference_pairs must be a list")
    require(
        len(raw) <= MAX_SYNTHETIC_REFERENCE_PAIR_ROWS,
        "synthetic reference-pair payload exceeds the hard row cap",
    )
    result: list[tuple[int, int, str]] = []
    for index, item in enumerate(raw):
        require(isinstance(item, dict), f"reference pair {index} must be an object")
        require(
            set(item)
            == {"citing_work_id_numeric", "cited_work_id_numeric", "source_task"},
            f"reference pair {index} schema mismatch",
        )
        source = _uint64(
            item["citing_work_id_numeric"],
            field=f"reference pair {index} source",
            positive=True,
        )
        target = _uint64(
            item["cited_work_id_numeric"],
            field=f"reference pair {index} target",
            positive=True,
        )
        source_task = item["source_task"]
        require(
            isinstance(source_task, str) and source_task,
            "source_task must be a nonempty string",
        )
        require(
            "/" not in source_task and "\\" not in source_task,
            "source_task must be a basename token",
        )
        result.append((source, target, source_task))
    require(result, "reference_pairs must not be empty")
    return result


def _pair_attributes(
    source: int,
    target: int,
    multiplicity: int,
    work_times: Mapping[int, tuple[int | None, int | None, int]],
) -> tuple[str, dict[str, str], int | None]:
    require(
        source in work_times, "every citing work must join one selected work-time row"
    )
    source_day, _, source_disposition = work_times[source]
    target_record = work_times.get(target)
    target_membership = (
        "in_snapshot" if target_record is not None else "external_open_world"
    )
    target_disposition: int | None = (
        target_record[2] if target_record is not None else None
    )
    target_day: int | None = target_record[0] if target_record is not None else None
    lag_days = (
        source_day - target_day
        if source_day is not None and target_day is not None
        else None
    )

    if source == target:
        bucket = "self_loop"
    elif source_disposition not in USABLE_DISPOSITIONS:
        bucket = "source_temporal_unusable"
    elif target_record is None:
        bucket = "target_external_open_world"
    elif target_disposition not in USABLE_DISPOSITIONS:
        bucket = "target_temporal_unusable"
    else:
        require(
            source_day is not None and target_day is not None,
            "usable dispositions require publication days",
        )
        if lag_days < 0:
            bucket = "negative_lag_target_after_source"
        elif lag_days == 0:
            bucket = "same_day"
        else:
            bucket = "positive_lag"

    if lag_days is None:
        lag_class = "NA"
    elif lag_days < 0:
        lag_class = "negative"
    elif lag_days == 0:
        lag_class = "same_day"
    else:
        lag_class = "positive"
    if multiplicity == 1:
        multiplicity_class = "one"
    elif multiplicity == 2:
        multiplicity_class = "two"
    else:
        multiplicity_class = "three_or_more"
    attributes = {
        "source_temporal_disposition": f"disposition_{source_disposition}",
        "target_membership": target_membership,
        "target_temporal_disposition_or_external_na": (
            "external_NA"
            if target_disposition is None
            else f"disposition_{target_disposition}"
        ),
        "self_loop_flag": "true" if source == target else "false",
        "signed_lag_class_or_na": lag_class,
        "pair_multiplicity_class": multiplicity_class,
    }
    return bucket, attributes, lag_days


def _plain_nested_counts(
    value: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        axis: {
            bucket: dict(sorted(categories.items()))
            for bucket, categories in sorted(by_bucket.items())
        }
        for axis, by_bucket in sorted(value.items())
    }


def audit_pairs(
    work_times: Mapping[int, tuple[int | None, int | None, int]],
    pairs: Sequence[tuple[int, int, str]],
    *,
    bucket_count: int | None = None,
    bucket_index: int | None = None,
) -> dict[str, Any]:
    """Deduplicate, classify, and close all synthetic pair accounting."""

    require(
        (bucket_count is None) == (bucket_index is None),
        "bucket count and index must be supplied together",
    )
    if bucket_count is not None:
        require(
            isinstance(bucket_count, int)
            and not isinstance(bucket_count, bool)
            and bucket_count > 1,
            "bucket count must be greater than one",
        )
        require(
            isinstance(bucket_index, int)
            and not isinstance(bucket_index, bool)
            and 0 <= bucket_index < bucket_count,
            "bucket index is outside its count",
        )
        selected = [
            pair
            for pair in pairs
            if pair_hash_bucket(pair[0], pair[1], bucket_count) == bucket_index
        ]
    else:
        selected = list(pairs)
    require(selected, "synthetic bucket selected zero pair rows")

    pair_rows = [(source, target) for source, target, _ in selected]
    multiplicities = Counter(pair_rows)
    source_tasks: dict[tuple[int, int], set[str]] = defaultdict(set)
    for source, target, task in selected:
        source_tasks[(source, target)].add(task)

    bucket_counts = Counter({bucket: 0 for bucket in PAIR_BUCKETS})
    cross_tabs: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    attribute_margins: dict[str, Counter[str]] = defaultdict(Counter)
    raw_self_loops = 0
    unique_self_loops = 0
    raw_external = 0
    unique_external = 0
    raw_available_at_non_null = 0
    unique_available_at_non_null = 0
    sensitivity = Counter(
        {
            "primary_zero_day": 0,
            "tolerance_30_days": 0,
            "tolerance_365_days": 0,
            "annual_only": 0,
        }
    )

    for (source, target), multiplicity in sorted(multiplicities.items()):
        bucket, attributes, lag_days = _pair_attributes(
            source,
            target,
            multiplicity,
            work_times,
        )
        bucket_counts[bucket] += 1
        for axis, category in attributes.items():
            contract_axis = CROSS_TAB_AXES[axis]
            cross_tabs[contract_axis][bucket][category] += 1
            attribute_margins[contract_axis][category] += 1
        if source == target:
            raw_self_loops += multiplicity
            unique_self_loops += 1
        if target not in work_times:
            raw_external += multiplicity
            unique_external += 1
        if work_times[source][2] in USABLE_DISPOSITIONS:
            require(
                work_times[source][0] is not None,
                "usable source disposition requires available_at day",
            )
            raw_available_at_non_null += multiplicity
            unique_available_at_non_null += 1

        source_record = work_times[source]
        target_record = work_times.get(target)
        common = (
            source != target
            and source_record[2] in USABLE_DISPOSITIONS
            and target_record is not None
            and target_record[2] in USABLE_DISPOSITIONS
        )
        if common:
            require(lag_days is not None, "common chronology prerequisites require lag")
            source_year = source_record[1]
            target_year = target_record[1]
            require(
                source_year is not None and target_year is not None,
                "usable dispositions require years",
            )
            sensitivity["primary_zero_day"] += int(lag_days >= 0)
            sensitivity["tolerance_30_days"] += int(lag_days >= -30)
            sensitivity["tolerance_365_days"] += int(lag_days >= -365)
            sensitivity["annual_only"] += int(target_year <= source_year)

    raw_rows = len(selected)
    unique_rows = len(multiplicities)
    duplicate_groups = sum(value > 1 for value in multiplicities.values())
    duplicate_excess = raw_rows - unique_rows
    maximum_multiplicity = max(multiplicities.values())
    duplicate_self_loop_excess = raw_self_loops - unique_self_loops
    raw_in_snapshot = raw_rows - raw_external
    unique_in_snapshot = unique_rows - unique_external
    distinct_targets = {target for _, target in multiplicities}
    distinct_external_targets = {
        target for target in distinct_targets if target not in work_times
    }
    duplicate_groups_crossing_tasks = sum(
        multiplicities[pair] > 1 and len(tasks) > 1
        for pair, tasks in source_tasks.items()
    )

    bucket_dict = {bucket: int(bucket_counts[bucket]) for bucket in PAIR_BUCKETS}
    accounting = {
        "raw_pair_rows": raw_rows,
        "sum_pair_multiplicity": sum(multiplicities.values()),
        "unique_pair_rows": unique_rows,
        "duplicate_pair_groups": duplicate_groups,
        "duplicate_pair_row_excess": duplicate_excess,
        "maximum_pair_multiplicity": maximum_multiplicity,
        "duplicate_groups_crossing_source_tasks": duplicate_groups_crossing_tasks,
        "raw_self_loop_rows": raw_self_loops,
        "unique_self_loop_pairs": unique_self_loops,
        "duplicate_self_loop_row_excess": duplicate_self_loop_excess,
        "raw_in_snapshot_target_rows": raw_in_snapshot,
        "raw_external_target_rows": raw_external,
        "unique_in_snapshot_target_pairs": unique_in_snapshot,
        "unique_external_target_pairs": unique_external,
        "distinct_valid_target_ids": len(distinct_targets),
        "distinct_in_snapshot_target_ids": len(
            distinct_targets - distinct_external_targets
        ),
        "distinct_external_target_ids": len(distinct_external_targets),
    }
    availability_accounting = {
        "resolution": "BIBLIOGRAPHIC_EVENT_TIME_CONVENTION",
        "edge_available_at_field": "citing_work_publication_date",
        "synthetic_physical_representation": "publication_day_int32",
        "raw_pair_rows_with_available_at": raw_available_at_non_null,
        "raw_pair_rows_without_available_at": raw_rows - raw_available_at_non_null,
        "unique_pairs_with_available_at": unique_available_at_non_null,
        "unique_pairs_without_available_at": unique_rows - unique_available_at_non_null,
        "openalex_observation_time_established": False,
        "historical_openalex_graph_established": False,
    }
    closures = {
        "raw_equals_sum_multiplicity": raw_rows == sum(multiplicities.values()),
        "raw_equals_unique_plus_duplicate_excess": raw_rows
        == unique_rows + duplicate_excess,
        "duplicate_groups_exact": duplicate_groups
        == sum(value > 1 for value in multiplicities.values()),
        "raw_membership_partition": raw_rows == raw_in_snapshot + raw_external,
        "unique_membership_partition": unique_rows
        == unique_in_snapshot + unique_external,
        "distinct_target_membership_partition": len(distinct_targets)
        == len(distinct_targets - distinct_external_targets)
        + len(distinct_external_targets),
        "self_loop_multiplicity_bridge": raw_self_loops
        == sum(
            multiplicity
            for (source, target), multiplicity in multiplicities.items()
            if source == target
        ),
        "self_loop_duplicate_bridge": raw_self_loops
        == unique_self_loops + duplicate_self_loop_excess,
        "raw_in_snapshot_multiplicity_bridge": raw_in_snapshot
        == sum(
            multiplicity
            for (_, target), multiplicity in multiplicities.items()
            if target in work_times
        ),
        "raw_external_multiplicity_bridge": raw_external
        == sum(
            multiplicity
            for (_, target), multiplicity in multiplicities.items()
            if target not in work_times
        ),
        "seven_bucket_conservation": unique_rows == sum(bucket_dict.values()),
        "self_loop_precedence_bridge": unique_self_loops == bucket_dict["self_loop"],
        "chronology_safe_primary_bridge": sensitivity["primary_zero_day"]
        == bucket_dict["same_day"] + bucket_dict["positive_lag"],
        "raw_available_at_partition": raw_rows
        == raw_available_at_non_null
        + availability_accounting["raw_pair_rows_without_available_at"],
        "unique_available_at_partition": unique_rows
        == unique_available_at_non_null
        + availability_accounting["unique_pairs_without_available_at"],
        "lag_tolerance_sensitivity_monotone": sensitivity["primary_zero_day"]
        <= sensitivity["tolerance_30_days"]
        <= sensitivity["tolerance_365_days"],
    }

    plain_cross_tabs = _plain_nested_counts(cross_tabs)
    plain_margins = {
        axis: dict(sorted(margin.items()))
        for axis, margin in sorted(attribute_margins.items())
    }
    expected_axes = set(CROSS_TAB_AXES.values())
    require(set(plain_cross_tabs) == expected_axes, "cross-tab axis set mismatch")
    cross_tab_closures: dict[str, bool] = {}
    for axis in sorted(expected_axes):
        by_bucket = plain_cross_tabs[axis]
        bucket_margins = {
            bucket: sum(by_bucket.get(bucket, {}).values()) for bucket in PAIR_BUCKETS
        }
        category_margin = Counter()
        for categories in by_bucket.values():
            category_margin.update(categories)
        cross_tab_closures[f"{axis}_primary_bucket_margins"] = (
            bucket_margins == bucket_dict
        )
        cross_tab_closures[f"{axis}_attribute_margins"] = (
            dict(sorted(category_margin.items())) == plain_margins[axis]
        )
        cross_tab_closures[f"{axis}_total"] = (
            sum(category_margin.values()) == unique_rows
        )
    closures["all_required_cross_tab_margins"] = all(cross_tab_closures.values())
    require(all(closures.values()), "synthetic accounting closure failed")

    return {
        "selection": {
            "kind": "all_pairs"
            if bucket_count is None
            else "deterministic_hash_bucket",
            "bucket_count": bucket_count,
            "bucket_index": bucket_index,
            "selected_raw_pair_rows": raw_rows,
            "global_coverage_claim_allowed": False,
        },
        "accounting": accounting,
        "availability_accounting": availability_accounting,
        "exclusive_pair_buckets": bucket_dict,
        "chronology_safe_primary_pairs": (
            bucket_dict["same_day"] + bucket_dict["positive_lag"]
        ),
        "chronology_sensitivity_unique_pairs": dict(sensitivity),
        "cross_tabs": plain_cross_tabs,
        "attribute_margins": plain_margins,
        "closures": closures,
        "cross_tab_closures": cross_tab_closures,
    }


def _report_fingerprint(report: Mapping[str, Any]) -> str:
    selected = {
        key: report[key]
        for key in (
            "schema_version",
            "status",
            "surface_kind",
            "production_status",
            "production_boundary",
            "mode",
            "execution_abi",
            "parent_contract_fixture",
            "selection",
            "accounting",
            "availability_accounting",
            "exclusive_pair_buckets",
            "chronology_safe_primary_pairs",
            "chronology_sensitivity_unique_pairs",
            "cross_tabs",
            "attribute_margins",
            "closures",
            "cross_tab_closures",
            "gate_status",
            "claim_boundary",
            "task_fingerprint_sha256",
            "mode_invariant_result_sha256",
        )
    }
    return sha256_bytes(canonical_json_bytes(selected))


def _build_report(
    plan: Mapping[str, Any],
    parent_context: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": SYNTHETIC_STATUS,
        "surface_kind": SYNTHETIC_SURFACE,
        "production_status": PRODUCTION_STATUS,
        "production_boundary": {
            "synthetic_fixture_opened": True,
            "production_source_roots_opened": False,
            "production_execution_supported": False,
            "production_component_status_emitted": False,
            "production_receipt_schema_emitted": False,
            "production_receipt_schema": PRODUCTION_OUTPUTS_SCHEMA,
            "production_planned_operator_outputs_created": False,
            "canonical_spine_partitions_created": 0,
            "canonical_spine_partitions_consumed_from_disk": 0,
            "synthetic_fixture_payload_only": True,
            "synthetic_fixture_hard_byte_cap": MAX_SYNTHETIC_FIXTURE_BYTES,
            "synthetic_work_time_hard_row_cap": MAX_SYNTHETIC_WORK_TIME_ROWS,
            "synthetic_reference_pair_hard_row_cap": (
                MAX_SYNTHETIC_REFERENCE_PAIR_ROWS
            ),
        },
        "mode": plan["mode"],
        "execution_abi": plan["execution_abi"],
        "parent_contract_fixture": parent_context,
        **result,
        "gate_status": {
            "synthetic_contract_mechanics": "PASS",
            "current_content_baseline_component": "SYNTHETIC_CONTRACT_FIXTURE_ONLY",
            "main_temporal_validity_component": "SYNTHETIC_CONTRACT_FIXTURE_ONLY",
            "citation_edge_accounting_production": "NOT_RUN",
            "bibliographic_available_at_convention_production": "NOT_RUN",
            "openalex_first_observed_time": "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT",
            "historical_graph_reconstruction": "RETROSPECTIVE_PROXY_ONLY_NOT_PRODUCED",
            "historical_content_identity": "UNVERIFIED",
            "raw_source_id_validity": "BLOCKED",
            "w_zero_entity_validity": "UNRESOLVED",
            "fixed_horizon_outcome_maturation": "PENDING_SEPARATE_G0_POLICY",
            "g_minus_1_data_readiness": "PENDING",
            "g0_measurement_freeze": "PENDING",
            "empirical_g1_signature": "NOT_RUN",
            "pass_g1": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "mode_invariant_result_sha256": sha256_bytes(canonical_json_bytes(result)),
        "provenance": {
            "config_sha256": plan["config_sha256"],
            "tool_sha256": plan["tool_sha256"],
            "synthetic_fixture_sha256": plan["synthetic_fixture_sha256"],
            "selected_payload_sha256": plan["selected_payload_sha256"],
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "operator_paths_redacted": True,
        },
    }
    report["provenance"]["result_fingerprint_sha256"] = _report_fingerprint(report)
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# OpenAlex Citation Edge Chronology Synthetic Audit",
            "",
            f"- Status: `{report['status']}`",
            f"- Surface: `{report['surface_kind']}`",
            f"- Production: `{report['production_status']}`",
            f"- Mode: `{report['mode']}`",
            "- Synthetic fixture opened: `true`",
            "- Production source roots opened: `false`",
            "- Production receipt ABI emitted: `false`",
            "- Canonical spine partitions created: `0`",
            "- G-1 / G0 / empirical G1 / scientific effect: `PENDING / PENDING / NOT_RUN / NOT_ESTABLISHED`",
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )


def _state(
    plan: Mapping[str, Any], status: str, *, error: BaseException | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": STATE_SCHEMA,
        "status": status,
        "surface_kind": SYNTHETIC_SURFACE,
        "production_status": PRODUCTION_STATUS,
        "mode": plan["mode"],
        "execution_abi": plan["execution_abi"],
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "production_promotion_allowed": False,
    }
    if error is not None:
        value["error"] = (
            f"{type(error).__name__}: synthetic audit failed; paths and payload details are redacted"
        )
    return value


def _output_manifest(output_dir: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = {
        relative: _artifact_contract(output_dir / relative, output_dir)
        for relative in EXPECTED_RECEIPT_ARTIFACTS
    }
    return {
        "schema_version": OUTPUTS_SCHEMA,
        "status": SYNTHETIC_STATUS,
        "surface_kind": SYNTHETIC_SURFACE,
        "production_status": PRODUCTION_STATUS,
        "production_receipt_schema_satisfied": False,
        "production_receipt_schema": PRODUCTION_OUTPUTS_SCHEMA,
        "production_planned_operator_outputs_created": False,
        "canonical_spine_partition_outputs_created": 0,
        "task_fingerprint_sha256": report["provenance"]["task_fingerprint_sha256"],
        "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
        "completion_contract": SYNTHETIC_COMPLETION_CONTRACT,
        "artifacts": artifacts,
        "receipt_written_last": True,
        "final_disk_validation_required": True,
        "operator_paths_redacted": True,
    }


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        info = output_dir.lstat()
        require(
            stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            "output root must be a real directory",
        )
        require(stat.S_IMODE(info.st_mode) == 0o700, "output root mode must be 0700")
        return
    output_dir.mkdir(parents=True, mode=0o700)
    output_dir.chmod(0o700)


def _relative_output_files(output_dir: Path) -> set[str]:
    result: set[str] = set()
    for path in output_dir.rglob("*"):
        info = path.lstat()
        require(
            not stat.S_ISLNK(info.st_mode), "synthetic output tree contains a symlink"
        )
        relative = path.relative_to(output_dir).as_posix()
        if stat.S_ISDIR(info.st_mode):
            require(
                stat.S_IMODE(info.st_mode) == 0o700,
                "synthetic output directory mode drift",
            )
            require(
                relative == "checkpoints",
                "synthetic output tree contains an undeclared directory",
            )
        elif stat.S_ISREG(info.st_mode):
            require(
                stat.S_IMODE(info.st_mode) == 0o600, "synthetic output file mode drift"
            )
            result.add(relative)
        else:
            raise CitationSyntheticAuditError(
                "synthetic output tree contains an unsafe entry"
            )
    return result


def validate_terminal_publication(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a terminal synthetic receipt and every byte-bound artifact."""

    root_info = output_dir.lstat()
    require(
        stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
        "terminal output root must be a real directory",
    )
    require(
        stat.S_IMODE(root_info.st_mode) == 0o700,
        "terminal output root mode drift",
    )
    receipt_path = output_dir / RECEIPT_BASENAME
    receipt = _read_json_regular(receipt_path, field="synthetic outputs receipt")
    require(
        set(receipt)
        == {
            "schema_version",
            "status",
            "surface_kind",
            "production_status",
            "production_receipt_schema_satisfied",
            "production_receipt_schema",
            "production_planned_operator_outputs_created",
            "canonical_spine_partition_outputs_created",
            "task_fingerprint_sha256",
            "result_fingerprint_sha256",
            "completion_contract",
            "artifacts",
            "receipt_written_last",
            "final_disk_validation_required",
            "operator_paths_redacted",
        },
        "synthetic receipt field set mismatch",
    )
    require(
        receipt.get("schema_version") == OUTPUTS_SCHEMA,
        "synthetic receipt schema mismatch",
    )
    require(
        receipt.get("schema_version") != PRODUCTION_OUTPUTS_SCHEMA,
        "production receipt schema must not be emitted",
    )
    require(
        receipt.get("status") == SYNTHETIC_STATUS, "synthetic receipt status mismatch"
    )
    require(
        receipt.get("status") != FULL_STATUS,
        "synthetic receipt cannot carry production pass",
    )
    require(
        receipt.get("surface_kind") == SYNTHETIC_SURFACE,
        "synthetic receipt surface mismatch",
    )
    require(
        receipt.get("production_status") == PRODUCTION_STATUS,
        "synthetic receipt production status drift",
    )
    require(
        receipt.get("production_receipt_schema_satisfied") is False,
        "synthetic receipt falsely satisfies production ABI",
    )
    require(
        receipt.get("production_receipt_schema") == PRODUCTION_OUTPUTS_SCHEMA,
        "synthetic receipt future-production schema reference mismatch",
    )
    require(
        receipt.get("production_planned_operator_outputs_created") is False,
        "synthetic receipt falsely publishes production outputs",
    )
    require(
        receipt.get("canonical_spine_partition_outputs_created") == 0,
        "synthetic receipt falsely publishes spine partitions",
    )
    require(
        receipt.get("completion_contract") == SYNTHETIC_COMPLETION_CONTRACT,
        "synthetic completion contract mismatch",
    )
    require(
        receipt.get("receipt_written_last") is True,
        "synthetic receipt order marker missing",
    )
    require(
        receipt.get("final_disk_validation_required") is True,
        "synthetic final-validation marker missing",
    )
    require(
        receipt.get("operator_paths_redacted") is True,
        "synthetic receipt path-redaction marker missing",
    )

    artifacts = receipt.get("artifacts")
    require(
        isinstance(artifacts, dict), "synthetic receipt artifacts must be a mapping"
    )
    require(
        set(artifacts) == set(EXPECTED_RECEIPT_ARTIFACTS),
        "synthetic receipt artifact set mismatch",
    )
    require(
        _relative_output_files(output_dir) == EXPECTED_FINAL_FILES,
        "terminal synthetic output file set mismatch",
    )
    for relative in EXPECTED_RECEIPT_ARTIFACTS:
        contract = artifacts[relative]
        require(
            isinstance(contract, dict),
            "synthetic receipt artifact contract must be an object",
        )
        require(
            set(contract) == {"relative_path", "bytes", "sha256"},
            "synthetic receipt artifact field set mismatch",
        )
        require(
            contract.get("relative_path") == relative,
            "synthetic receipt artifact path mismatch",
        )
        path = output_dir / relative
        payload = _regular_bytes(path, field="receipt-bound synthetic artifact")
        require(
            contract.get("bytes") == len(payload),
            "synthetic receipt artifact byte mismatch",
        )
        require(
            contract.get("sha256") == sha256_bytes(payload),
            "synthetic receipt artifact hash mismatch",
        )

    plan = _read_json_regular(
        output_dir / "audit_plan.json", field="terminal synthetic plan"
    )
    state = _read_json_regular(
        output_dir / "run_state.json", field="terminal synthetic state"
    )
    checkpoint = _read_json_regular(
        output_dir / "checkpoints/synthetic_accounting.json",
        field="terminal synthetic checkpoint",
    )
    report = _read_json_regular(
        output_dir / REPORT_BASENAME, field="terminal synthetic report"
    )
    markdown = _regular_bytes(
        output_dir / MARKDOWN_BASENAME, field="terminal synthetic Markdown"
    )
    require(plan.get("schema_version") == PLAN_SCHEMA, "terminal plan schema mismatch")
    require(
        plan.get("task_fingerprint_sha256")
        == sha256_bytes(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in plan.items()
                    if key != "task_fingerprint_sha256"
                }
            )
        ),
        "terminal plan task fingerprint mismatch",
    )
    require(
        state.get("schema_version") == STATE_SCHEMA, "terminal state schema mismatch"
    )
    require(
        set(state)
        == {
            "schema_version",
            "status",
            "surface_kind",
            "production_status",
            "mode",
            "execution_abi",
            "task_fingerprint_sha256",
            "production_promotion_allowed",
            "completion_receipt_relative_path",
            "complete_only_if_synthetic_receipt_exists",
            "result_fingerprint_sha256",
        },
        "terminal state field set mismatch",
    )
    require(
        state.get("status") == "COMPLETE_PENDING_SYNTHETIC_RECEIPT",
        "terminal state lifecycle mismatch",
    )
    require(
        state.get("surface_kind") == SYNTHETIC_SURFACE,
        "terminal state surface mismatch",
    )
    require(
        state.get("production_status") == PRODUCTION_STATUS,
        "terminal state production status mismatch",
    )
    require(
        state.get("production_promotion_allowed") is False,
        "terminal state falsely allows promotion",
    )
    require(
        state.get("completion_receipt_relative_path") == RECEIPT_BASENAME,
        "terminal receipt path mismatch",
    )
    require(
        state.get("complete_only_if_synthetic_receipt_exists") is True,
        "terminal receipt requirement mismatch",
    )
    require(
        checkpoint.get("schema_version") == CHECKPOINT_SCHEMA,
        "terminal checkpoint schema mismatch",
    )
    require(
        set(checkpoint)
        == {
            "schema_version",
            "surface_kind",
            "production_status",
            "task_fingerprint_sha256",
            "selected_payload_sha256",
            "result",
            "result_sha256",
        },
        "terminal checkpoint field set mismatch",
    )
    require(
        checkpoint.get("surface_kind") == SYNTHETIC_SURFACE,
        "terminal checkpoint surface mismatch",
    )
    require(
        checkpoint.get("production_status") == PRODUCTION_STATUS,
        "terminal checkpoint production status mismatch",
    )
    require(
        report.get("schema_version") == REPORT_SCHEMA, "terminal report schema mismatch"
    )
    require(report.get("status") == SYNTHETIC_STATUS, "terminal report status mismatch")
    require(
        report.get("surface_kind") == SYNTHETIC_SURFACE,
        "terminal report surface mismatch",
    )
    require(
        report.get("production_status") == PRODUCTION_STATUS,
        "terminal report production status mismatch",
    )
    require(report.get("mode") == plan.get("mode"), "terminal report mode mismatch")
    require(
        report.get("execution_abi") == plan.get("execution_abi"),
        "terminal report execution ABI mismatch",
    )
    provenance = report.get("provenance")
    require(isinstance(provenance, dict), "report provenance must be a mapping")
    require(
        provenance.get("result_fingerprint_sha256") == _report_fingerprint(report),
        "terminal report fingerprint mismatch",
    )
    require(
        markdown.decode("utf-8") == render_markdown(report),
        "terminal Markdown mismatch",
    )
    task_fingerprint = plan.get("task_fingerprint_sha256")
    result_fingerprint = provenance.get("result_fingerprint_sha256")
    require(
        receipt.get("task_fingerprint_sha256") == task_fingerprint,
        "receipt task fingerprint mismatch",
    )
    require(
        receipt.get("result_fingerprint_sha256") == result_fingerprint,
        "receipt result fingerprint mismatch",
    )
    require(
        report.get("task_fingerprint_sha256") == task_fingerprint,
        "report task fingerprint mismatch",
    )
    require(
        provenance.get("task_fingerprint_sha256") == task_fingerprint,
        "report provenance task fingerprint mismatch",
    )
    require(
        state.get("task_fingerprint_sha256") == task_fingerprint,
        "state task fingerprint mismatch",
    )
    require(state.get("mode") == plan.get("mode"), "state mode mismatch")
    require(
        state.get("execution_abi") == plan.get("execution_abi"),
        "state execution ABI mismatch",
    )
    require(
        state.get("result_fingerprint_sha256") == result_fingerprint,
        "state result fingerprint mismatch",
    )
    require(
        checkpoint.get("task_fingerprint_sha256") == task_fingerprint,
        "checkpoint task fingerprint mismatch",
    )
    require(
        checkpoint.get("selected_payload_sha256")
        == plan.get("selected_payload_sha256"),
        "checkpoint selected-payload commitment mismatch",
    )
    checkpoint_result = checkpoint.get("result")
    require(
        isinstance(checkpoint_result, dict),
        "checkpoint result must be a mapping",
    )
    require(
        checkpoint.get("result_sha256")
        == sha256_bytes(canonical_json_bytes(checkpoint_result)),
        "checkpoint result hash mismatch",
    )
    require(
        checkpoint_result == {key: report[key] for key in checkpoint_result},
        "checkpoint/report result mismatch",
    )
    require(
        report.get("mode_invariant_result_sha256") == checkpoint.get("result_sha256"),
        "mode-invariant result commitment mismatch",
    )
    for report_key, plan_key in (
        ("config_sha256", "config_sha256"),
        ("tool_sha256", "tool_sha256"),
        ("synthetic_fixture_sha256", "synthetic_fixture_sha256"),
        ("selected_payload_sha256", "selected_payload_sha256"),
    ):
        require(
            provenance.get(report_key) == plan.get(plan_key),
            f"report {report_key} commitment mismatch",
        )
    production_boundary = report.get("production_boundary")
    require(
        isinstance(production_boundary, dict),
        "report production boundary must be a mapping",
    )
    require(
        set(production_boundary)
        == {
            "synthetic_fixture_opened",
            "production_source_roots_opened",
            "production_execution_supported",
            "production_component_status_emitted",
            "production_receipt_schema_emitted",
            "production_receipt_schema",
            "production_planned_operator_outputs_created",
            "canonical_spine_partitions_created",
            "canonical_spine_partitions_consumed_from_disk",
            "synthetic_fixture_payload_only",
            "synthetic_fixture_hard_byte_cap",
            "synthetic_work_time_hard_row_cap",
            "synthetic_reference_pair_hard_row_cap",
        },
        "report production-boundary field set mismatch",
    )
    require(
        production_boundary.get("synthetic_fixture_opened") is True,
        "terminal report hides synthetic fixture access",
    )
    require(
        production_boundary.get("production_source_roots_opened") is False,
        "terminal report claims production source-root access",
    )
    require(
        production_boundary.get("production_receipt_schema_emitted") is False,
        "terminal report claims production receipt ABI",
    )
    require(
        production_boundary.get("production_planned_operator_outputs_created") is False,
        "terminal report claims production output publication",
    )
    require(
        production_boundary.get("canonical_spine_partitions_created") == 0,
        "terminal report claims spine creation",
    )
    require(
        production_boundary.get("canonical_spine_partitions_consumed_from_disk") == 0,
        "terminal report claims spine partition consumption",
    )
    gate_status = report.get("gate_status")
    require(isinstance(gate_status, dict), "report gate status must be a mapping")
    require(
        gate_status.get("citation_edge_accounting_production") == "NOT_RUN",
        "terminal report falsely claims production citation accounting",
    )
    require(gate_status.get("pass_g1") is False, "terminal report falsely passes G1")
    require(
        gate_status.get("scientific_effect") == "NOT_ESTABLISHED",
        "terminal report falsely establishes a scientific effect",
    )
    return report, plan


def _unlink_regular(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    require(
        stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode),
        "refusing to remove non-file publication entry",
    )
    path.unlink()


def _invalidate_publication(
    output_dir: Path,
    plan: Mapping[str, Any] | None,
    error: BaseException,
) -> None:
    # Receipt-first invalidation is part of the synthetic terminal ABI.
    for name in (RECEIPT_BASENAME, REPORT_BASENAME, MARKDOWN_BASENAME):
        try:
            _unlink_regular(output_dir / name)
        except (CitationSyntheticAuditError, OSError):
            # An unsafe replacement is not recursively removed, but cleanup of
            # the remaining publication files and failure-state write proceeds.
            continue
    if plan is not None:
        try:
            atomic_write_json(
                output_dir / "run_state.json",
                _state(
                    plan,
                    "FINALIZATION_FAILED_NO_SYNTHETIC_RECEIPT",
                    error=error,
                ),
            )
        except (CitationSyntheticAuditError, OSError):
            pass


def _terminal_receipt_exists(output_dir: Path) -> bool:
    try:
        (output_dir / RECEIPT_BASENAME).lstat()
    except FileNotFoundError:
        return False
    return True


def run_synthetic_audit(
    *,
    config_path: Path,
    synthetic_fixture: Path,
    output_dir: Path,
    mode: str,
    resume: bool = False,
    bucket_count: int | None = None,
    bucket_index: int | None = None,
) -> dict[str, Any]:
    """Execute the closed synthetic fixture without any production I/O path."""

    require(mode in ALLOWED_MODES, "unsupported temporal input mode")
    config, config_payload, config_sha256 = _read_config_snapshot(config_path)
    _validate_config(config)
    tool_sha256 = sha256_bytes(
        _regular_bytes(Path(__file__), field="citation synthetic tool")
    )

    if output_dir.exists() and _terminal_receipt_exists(output_dir):
        try:
            report, plan = validate_terminal_publication(output_dir)
        except BaseException as exc:
            disk_plan: dict[str, Any] | None = None
            try:
                disk_plan = _read_json_regular(
                    output_dir / "audit_plan.json", field="invalid terminal plan"
                )
            except BaseException:
                disk_plan = None
            _invalidate_publication(output_dir, disk_plan, exc)
            raise
        require(
            plan.get("config_sha256") == config_sha256,
            "terminal config commitment mismatch",
        )
        require(
            plan.get("tool_sha256") == tool_sha256, "terminal tool commitment mismatch"
        )
        require(plan.get("mode") == mode, "terminal mode mismatch")
        require(
            plan.get("bucket_count") == bucket_count, "terminal bucket-count mismatch"
        )
        require(
            plan.get("bucket_index") == bucket_index, "terminal bucket-index mismatch"
        )
        require(
            resume,
            "valid terminal synthetic output requires --resume for no-touch return",
        )
        return report

    fixture, fixture_payload, fixture_sha256 = _load_fixture_payload(synthetic_fixture)
    require(
        fixture.get("schema_version") == FIXTURE_SCHEMA,
        "synthetic fixture schema mismatch",
    )
    require(
        fixture.get("surface_kind") == SYNTHETIC_SURFACE,
        "fixture surface must be SYNTHETIC_FIXTURE_ONLY",
    )
    require(
        fixture.get("mode") == mode,
        "fixture mode does not match precommitted invocation mode",
    )
    require(
        fixture.get("production_input") is False,
        "fixture must explicitly deny production input",
    )
    require(
        FORBIDDEN_MODE_TIME_KEY[mode] not in fixture,
        "fixture contains the forbidden other-mode temporal payload",
    )
    require(
        MODE_TIME_KEY[mode] in fixture, "fixture lacks selected mode temporal payload"
    )
    require(
        set(fixture)
        == {
            "schema_version",
            "surface_kind",
            "mode",
            "production_input",
            "parents",
            "reference_pairs",
            MODE_TIME_KEY[mode],
        },
        "synthetic fixture field set mismatch",
    )
    parent_context = _validate_parent_fixture(mode, fixture.get("parents"))
    work_times = _parse_work_times(fixture[MODE_TIME_KEY[mode]])
    pairs = _parse_pairs(fixture.get("reference_pairs"))
    selected_payload = {
        "mode": mode,
        "parents": fixture["parents"],
        MODE_TIME_KEY[mode]: fixture[MODE_TIME_KEY[mode]],
        "reference_pairs": fixture["reference_pairs"],
        "bucket_count": bucket_count,
        "bucket_index": bucket_index,
    }
    selected_payload_sha256 = sha256_bytes(canonical_json_bytes(selected_payload))
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "surface_kind": SYNTHETIC_SURFACE,
        "production_status": PRODUCTION_STATUS,
        "production_input_access_allowed": False,
        "production_promotion_allowed": False,
        "production_receipt_schema": PRODUCTION_OUTPUTS_SCHEMA,
        "synthetic_receipt_schema": OUTPUTS_SCHEMA,
        "config_sha256": config_sha256,
        "config_snapshot_bytes": len(config_payload),
        "tool_sha256": tool_sha256,
        "synthetic_fixture_sha256": fixture_sha256,
        "synthetic_fixture_bytes": len(fixture_payload),
        "selected_payload_sha256": selected_payload_sha256,
        "mode": mode,
        "execution_abi": EXECUTION_ABI[mode],
        "mode_conditioned_parent_order": list(MODE_PARENT_ORDER[mode]),
        "selected_work_time_rows": len(work_times),
        "input_reference_pair_rows": len(pairs),
        "bucket_count": bucket_count,
        "bucket_index": bucket_index,
        "pair_hash_abi": {
            "canonical_key_bytes": "uint64_citing_id_big_endian_then_uint64_cited_id_big_endian",
            "hash_function": "SHA256",
            "bucket_integer": "unsigned_big_endian_first_8_digest_bytes",
            "bucket_formula": "bucket_integer_mod_bucket_count",
        },
    }
    plan["task_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(plan))

    _prepare_output_dir(output_dir)
    existing_files = _relative_output_files(output_dir)
    plan_path = output_dir / "audit_plan.json"
    if existing_files:
        require(resume, "nonempty synthetic output requires --resume")
        require("audit_plan.json" in existing_files, "resume output lacks audit plan")
        disk_plan = _read_json_regular(plan_path, field="resume synthetic plan")
        require(disk_plan == plan, "resume synthetic plan mismatch")
        for name in (RECEIPT_BASENAME, REPORT_BASENAME, MARKDOWN_BASENAME):
            _unlink_regular(output_dir / name)
    else:
        atomic_write_json(plan_path, plan)

    checkpoint_path = output_dir / "checkpoints/synthetic_accounting.json"
    report_path = output_dir / REPORT_BASENAME
    markdown_path = output_dir / MARKDOWN_BASENAME
    receipt_path = output_dir / RECEIPT_BASENAME
    finalization_started = False
    try:
        atomic_write_json(output_dir / "run_state.json", _state(plan, "RUNNING"))
        result = audit_pairs(
            work_times,
            pairs,
            bucket_count=bucket_count,
            bucket_index=bucket_index,
        )
        _revalidate_execution_inputs(
            config_path=config_path,
            config_sha256=config_sha256,
            fixture_path=synthetic_fixture,
            fixture_sha256=fixture_sha256,
            tool_sha256=tool_sha256,
        )
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA,
            "surface_kind": SYNTHETIC_SURFACE,
            "production_status": PRODUCTION_STATUS,
            "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
            "selected_payload_sha256": selected_payload_sha256,
            "result": result,
            "result_sha256": sha256_bytes(canonical_json_bytes(result)),
        }
        atomic_write_json(checkpoint_path, checkpoint)
        report = _build_report(plan, parent_context, result)

        finalization_started = True
        atomic_write_json(report_path, report)
        atomic_write_text(markdown_path, render_markdown(report))
        state = _state(plan, "COMPLETE_PENDING_SYNTHETIC_RECEIPT")
        state.update(
            {
                "completion_receipt_relative_path": RECEIPT_BASENAME,
                "complete_only_if_synthetic_receipt_exists": True,
                "result_fingerprint_sha256": report["provenance"][
                    "result_fingerprint_sha256"
                ],
            }
        )
        atomic_write_json(output_dir / "run_state.json", state)
        outputs = _output_manifest(output_dir, report)
        # The synthetic receipt is intentionally the final publication write.
        atomic_write_json(receipt_path, outputs)
        disk_report, _ = validate_terminal_publication(output_dir)
        require(
            disk_report == report, "final synthetic report differs after publication"
        )
        _revalidate_execution_inputs(
            config_path=config_path,
            config_sha256=config_sha256,
            fixture_path=synthetic_fixture,
            fixture_sha256=fixture_sha256,
            tool_sha256=tool_sha256,
        )
        return report
    except BaseException as exc:
        if finalization_started or _terminal_receipt_exists(output_dir):
            _invalidate_publication(output_dir, plan, exc)
        else:
            atomic_write_json(
                output_dir / "run_state.json",
                _state(plan, "FAILED_SYNTHETIC_NO_RECEIPT", error=exc),
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--synthetic-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=ALLOWED_MODES, required=True)
    parser.add_argument("--hash-bucket-count", type=int)
    parser.add_argument("--hash-bucket-index", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-synthetic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute_synthetic:
        print(
            "refusing execution without --execute-synthetic; production execution is unsupported",
            file=sys.stderr,
        )
        return 2
    try:
        report = run_synthetic_audit(
            config_path=args.config,
            synthetic_fixture=args.synthetic_fixture,
            output_dir=args.output_dir,
            mode=args.mode,
            resume=args.resume,
            bucket_count=args.hash_bucket_count,
            bucket_index=args.hash_bucket_index,
        )
    except CitationSyntheticAuditError as exc:
        print(
            f"{type(exc).__name__}: synthetic citation audit failed; paths and payload details are redacted",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "surface_kind": report["surface_kind"],
                "production_status": report["production_status"],
                "result_fingerprint_sha256": report["provenance"][
                    "result_fingerprint_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
