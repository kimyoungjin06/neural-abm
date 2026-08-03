#!/usr/bin/env python3
"""Build a read-only analytic capacity review for the canonical time spine.

This tool reads only repository files and writes only JSON to stdout.  It does
not import or execute the production runner, open an OpenAlex source file, run
a pilot, benchmark storage, change a gate, or grant production authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml"
)
DEFAULT_ASSUMPTIONS = (
    REPO_ROOT / "experiments/configs/"
    "openalex_g1_canonical_work_time_spine_capacity_review_v0.yaml"
)
INPUT_SCHEMA = "openalex_g1_canonical_work_time_spine_capacity_review_input.v1"
OUTPUT_SCHEMA = "openalex_g1_canonical_work_time_spine_capacity_review.v1"
CONTRACT_SCHEMA = "openalex_g1_canonical_work_time_spine_design.v3"
STATUS = "ANALYTIC_ESTIMATE_ONLY_EMPIRICAL_BENCHMARK_NOT_RUN"
GATE_STATUS = "NOT_PASSED"
PRODUCTION_STATUS = "PRODUCTION_NOT_RUN"
EXPECTED_CONTRACT_SHA256 = (
    "6e8b74ddd12cbef46c8d15d3560f0d54f470b5017685fc1bce2389250cde0484"
)
EXPECTED_RUNNER_SHA256 = (
    "bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280"
)
FRAMED_RECORD_HASH_DOMAIN = b"openalex-work-time-records-v1"
FRAMED_RECORD_HASH_BYTES_PER_RECORD = 24
FRAMED_RECORD_HASH_HEADER_BYTES_PER_DIGEST = 8 + len(FRAMED_RECORD_HASH_DOMAIN) + 8

ASSUMPTION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "gate_status",
        "production_execution_status",
        "frozen_contract_binding",
        "hardware_observation",
        "hypothetical_rate_floor_scenario",
        "review_state",
    }
)
BINDING_KEYS = frozenset(
    {
        "config_path",
        "config_sha256",
        "runner_path",
        "runner_sha256",
        "runner_import_or_execution_allowed",
    }
)
HARDWARE_KEYS = frozenset(
    {
        "observed_at",
        "observation_scope",
        "output_mount_class",
        "output_device_model",
        "output_device_st_dev",
        "output_filesystem_total_bytes",
        "output_filesystem_available_bytes",
        "source_mount_class",
        "source_device_model",
        "source_device_st_dev",
        "source_and_output_device_ids_distinct",
        "mem_total_bytes",
        "mem_available_bytes",
        "nofile_soft_limit",
        "nofile_hard_limit",
        "atomic_rename_fsync_probe_at_observation",
        "source_drive_health_review_at_observation",
        "source_quiet_window_review_at_observation",
        "sequential_throughput_benchmark_at_observation",
        "downstream_pipeline_benchmark_at_observation",
        "snapshot_is_not_execution_preflight",
    }
)
RATE_KEYS = frozenset(
    {
        "scenario_id",
        "status",
        "sequential_read_bytes_per_second",
        "sequential_write_bytes_per_second",
        "sha256_bytes_per_second",
        "demultiplex_records_per_second",
        "stable_sort_records_per_second",
        "worst_case_heap_merge_records_per_second",
        "parquet_encode_records_per_second",
        "parquet_decode_records_per_second",
        "data_file_fsync_operations_per_second",
        "artifact_contracts_per_second",
        "additive_time_safety_multiplier_milli",
        "rates_are_hypothetical_not_benchmarked",
        "conditional_estimates_are_not_unconditional_time_bounds",
    }
)
REVIEW_STATE_KEYS = frozenset(
    {
        "empirical_benchmark_status",
        "operator_acceptance_status",
        "independent_review_status",
        "downstream_capacity_gate_status",
        "full_authorization_status",
        "analytic_output_must_not_mutate_any_gate",
        "analytic_output_must_not_be_used_as_production_receipt_parent",
    }
)


class CapacityReviewError(RuntimeError):
    """Raised when an analytic review input violates its closed contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacityReviewError(message)


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that fails closed on duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be a mapping")
    require(
        all(isinstance(key, str) for key in value),
        f"{field} keys must be strings",
    )
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], field: str) -> None:
    observed = set(value)
    require(
        observed == expected,
        f"{field} key set differs: missing={sorted(expected - observed)!r}, "
        f"extra={sorted(observed - expected)!r}",
    )


def _positive_int(value: Any, field: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{field} must be a positive integer",
    )
    return int(value)


def _nonnegative_int(value: Any, field: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{field} must be a nonnegative integer",
    )
    return int(value)


def _regular_file_snapshot(path: Path, field: str) -> tuple[bytes, str]:
    """Read and hash the same stable regular-file descriptor."""

    lexical = path.lstat()
    require(not stat.S_ISLNK(lexical.st_mode), f"{field} may not be a symlink")
    require(stat.S_ISREG(lexical.st_mode), f"{field} is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode), f"{field} is not a regular file")
        require(
            (before.st_dev, before.st_ino) == (lexical.st_dev, lexical.st_ino),
            f"{field} changed before open",
        )
        while payload := handle.read(1 << 20):
            digest.update(payload)
            chunks.append(payload)
        after = os.fstat(handle.fileno())
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    require(
        all(getattr(before, name) == getattr(after, name) for name in identity_fields),
        f"{field} changed while read",
    )
    payload = b"".join(chunks)
    require(len(payload) == before.st_size, f"{field} size changed while read")
    return payload, digest.hexdigest()


def _repository_file(relative: Any, field: str) -> Path:
    require(isinstance(relative, str) and relative, f"{field} must be a path")
    candidate = Path(relative)
    require(not candidate.is_absolute(), f"{field} must be repository-relative")
    require(".." not in candidate.parts, f"{field} may not traverse parents")
    path = REPO_ROOT / candidate
    resolved = path.resolve(strict=True)
    require(
        resolved.is_relative_to(REPO_ROOT.resolve(strict=True)),
        f"{field} escaped the repository",
    )
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
    require(not path.is_symlink(), f"{field} may not be a symlink")
    return path


def _load_yaml_payload(payload: bytes, field: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CapacityReviewError(f"{field} is not UTF-8") from exc
    value = yaml.load(text, Loader=UniqueKeyLoader)
    return _mapping(value, field)


def _require_default_file(path: Path, default: Path, field: str) -> Path:
    require(
        path.resolve(strict=True) == default.resolve(strict=True),
        f"capacity review is bound to the repository default {field}",
    )
    info = path.lstat()
    require(not stat.S_ISLNK(info.st_mode), f"{field} may not be a symlink")
    require(stat.S_ISREG(info.st_mode), f"{field} is not a regular file")
    return default


def load_inputs(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    assumptions_path: Path = DEFAULT_ASSUMPTIONS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = _require_default_file(
        contract_path, DEFAULT_CONTRACT, "frozen contract"
    )
    assumptions_path = _require_default_file(
        assumptions_path, DEFAULT_ASSUMPTIONS, "assumptions"
    )
    contract_payload, contract_sha256 = _regular_file_snapshot(
        contract_path, "frozen contract"
    )
    assumptions_payload, _ = _regular_file_snapshot(
        assumptions_path, "capacity assumptions"
    )
    contract = _load_yaml_payload(contract_payload, "frozen contract")
    assumptions = _load_yaml_payload(assumptions_payload, "capacity assumptions")
    _exact_keys(assumptions, ASSUMPTION_KEYS, "capacity assumptions")
    binding = _mapping(
        assumptions.get("frozen_contract_binding"), "frozen contract binding"
    )
    _exact_keys(binding, BINDING_KEYS, "frozen contract binding")
    bound_contract = _repository_file(binding.get("config_path"), "config path")
    bound_runner = _repository_file(binding.get("runner_path"), "runner path")
    require(
        bound_contract.resolve(strict=True) == contract_path.resolve(strict=True),
        "bound contract path differs from loaded contract",
    )
    require(
        contract_sha256 == binding.get("config_sha256") == EXPECTED_CONTRACT_SHA256,
        "frozen contract SHA-256 differs",
    )
    _, runner_sha256 = _regular_file_snapshot(bound_runner, "frozen runner")
    require(
        runner_sha256 == binding.get("runner_sha256") == EXPECTED_RUNNER_SHA256,
        "frozen runner SHA-256 differs",
    )
    require(
        binding.get("runner_import_or_execution_allowed") is False,
        "capacity review must not import or execute the runner",
    )
    return contract, assumptions


def _ceil_ratio(numerator: int, denominator: int) -> int:
    require(numerator >= 0, "time component numerator is negative")
    require(denominator > 0, "time component denominator is not positive")
    return (numerator + denominator - 1) // denominator


def _conditional_time(
    components: Mapping[str, tuple[int, int]], multiplier_milli: int
) -> dict[str, Any]:
    component_seconds = {
        name: _ceil_ratio(quantity, rate)
        for name, (quantity, rate) in components.items()
    }
    additive_seconds = sum(component_seconds.values())
    adjusted_seconds = _ceil_ratio(additive_seconds * multiplier_milli, 1000)
    return {
        "status": "HYPOTHETICAL_CONDITIONAL_NOT_BENCHMARKED",
        "component_seconds_rounded_up": component_seconds,
        "additive_seconds_before_safety_multiplier": additive_seconds,
        "safety_multiplier_milli": multiplier_milli,
        "conditional_planning_seconds_rounded_up": adjusted_seconds,
        "conditional_planning_hours_rounded_3dp": round(adjusted_seconds / 3600, 3),
        "is_unconditional_wall_time_bound": False,
        "requires_all_rate_floors_and_unmodeled_costs_to_be_validated": True,
    }


def _validate_review_state(assumptions: Mapping[str, Any]) -> dict[str, Any]:
    state = _mapping(assumptions.get("review_state"), "review state")
    _exact_keys(state, REVIEW_STATE_KEYS, "review state")
    expected = {
        "empirical_benchmark_status": "NOT_RUN",
        "operator_acceptance_status": "NOT_RECORDED",
        "independent_review_status": "PENDING",
        "downstream_capacity_gate_status": GATE_STATUS,
        "full_authorization_status": "NOT_GRANTED",
        "analytic_output_must_not_mutate_any_gate": True,
        "analytic_output_must_not_be_used_as_production_receipt_parent": True,
    }
    for key, value in expected.items():
        require(state.get(key) == value, f"review state drift: {key}")
    return state


def _validate_hardware_observation(assumptions: Mapping[str, Any]) -> dict[str, Any]:
    hardware = _mapping(assumptions.get("hardware_observation"), "hardware observation")
    _exact_keys(hardware, HARDWARE_KEYS, "hardware observation")
    observed_at = hardware.get("observed_at")
    require(isinstance(observed_at, str) and observed_at, "observed_at must be text")
    try:
        parsed_at = datetime.fromisoformat(observed_at)
    except ValueError as exc:
        raise CapacityReviewError("observed_at must be ISO-8601") from exc
    require(
        parsed_at.tzinfo is not None and parsed_at.utcoffset() is not None,
        "observed_at must include a UTC offset",
    )
    require(
        hardware.get("observation_scope")
        == "CONTROL_PLANE_ONLY_NO_SOURCE_PAYLOAD_OPENED",
        "hardware observation scope drift",
    )
    for name in (
        "output_mount_class",
        "output_device_model",
        "source_mount_class",
        "source_device_model",
    ):
        require(
            isinstance(hardware.get(name), str) and bool(hardware.get(name)),
            f"{name} must be nonempty text",
        )
    for name in (
        "atomic_rename_fsync_probe_at_observation",
        "source_drive_health_review_at_observation",
        "source_quiet_window_review_at_observation",
        "sequential_throughput_benchmark_at_observation",
        "downstream_pipeline_benchmark_at_observation",
    ):
        require(hardware.get(name) == "NOT_RUN", f"hardware snapshot drift: {name}")
    require(
        hardware.get("source_and_output_device_ids_distinct") is True,
        "hardware snapshot must preserve distinct-device observation",
    )
    require(
        hardware.get("snapshot_is_not_execution_preflight") is True,
        "hardware snapshot may not claim execution preflight",
    )
    output_device = _positive_int(
        hardware.get("output_device_st_dev"), "output device ID"
    )
    source_device = _positive_int(
        hardware.get("source_device_st_dev"), "source device ID"
    )
    require(output_device != source_device, "observed source/output devices coincide")
    output_total = _positive_int(
        hardware.get("output_filesystem_total_bytes"), "output filesystem total"
    )
    output_available = _positive_int(
        hardware.get("output_filesystem_available_bytes"), "available disk"
    )
    require(output_available <= output_total, "available disk exceeds filesystem total")
    mem_total = _positive_int(hardware.get("mem_total_bytes"), "total RAM")
    mem_available = _positive_int(hardware.get("mem_available_bytes"), "available RAM")
    require(mem_available <= mem_total, "available RAM exceeds total RAM")
    nofile_soft = _positive_int(hardware.get("nofile_soft_limit"), "nofile soft limit")
    nofile_hard = _positive_int(hardware.get("nofile_hard_limit"), "nofile hard limit")
    require(nofile_soft <= nofile_hard, "nofile soft limit exceeds hard limit")
    return hardware


def build_review(
    contract: Mapping[str, Any], assumptions: Mapping[str, Any]
) -> dict[str, Any]:
    assumptions = _mapping(assumptions, "capacity assumptions")
    _exact_keys(assumptions, ASSUMPTION_KEYS, "capacity assumptions")
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema drift")
    require(assumptions.get("schema_version") == INPUT_SCHEMA, "input schema drift")
    require(assumptions.get("status") == STATUS, "input status drift")
    require(assumptions.get("gate_status") == GATE_STATUS, "input gate drift")
    require(
        assumptions.get("production_execution_status") == PRODUCTION_STATUS,
        "production status drift",
    )
    review_state = _validate_review_state(assumptions)

    resources = _mapping(contract.get("resource_envelope"), "resource envelope")
    partitioning = _mapping(
        contract.get("bounded_partition_and_sort_contract"),
        "partition and sort contract",
    )
    spool_contract = _mapping(partitioning.get("spool"), "spool contract")
    run_contract = _mapping(partitioning.get("external_runs"), "external run contract")
    publication = _mapping(
        partitioning.get("partition_publication"), "partition publication"
    )
    completion = _mapping(contract.get("completion_contract"), "completion contract")
    rates = _mapping(
        assumptions.get("hypothetical_rate_floor_scenario"),
        "hypothetical rate floor scenario",
    )
    _exact_keys(rates, RATE_KEYS, "hypothetical rate floor scenario")
    hardware = _validate_hardware_observation(assumptions)
    binding = _mapping(
        assumptions.get("frozen_contract_binding"), "frozen contract binding"
    )
    _exact_keys(binding, BINDING_KEYS, "frozen contract binding")
    require(
        binding.get("config_path") == DEFAULT_CONTRACT.relative_to(REPO_ROOT).as_posix()
        and binding.get("config_sha256") == EXPECTED_CONTRACT_SHA256
        and binding.get("runner_path")
        == (REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v3.py")
        .relative_to(REPO_ROOT)
        .as_posix()
        and binding.get("runner_sha256") == EXPECTED_RUNNER_SHA256
        and binding.get("runner_import_or_execution_allowed") is False,
        "frozen contract binding drift",
    )

    rows = _positive_int(resources.get("declared_retained_rows"), "retained rows")
    record_bytes = _positive_int(resources.get("fixed_record_bytes"), "record bytes")
    input_files = _positive_int(resources.get("declared_input_files"), "input files")
    partitions = _positive_int(partitioning.get("partition_count"), "partitions")
    sort_chunk_rows = _positive_int(run_contract.get("sort_chunk_rows"), "sort rows")
    staging = rows * record_bytes
    parquet_row_bytes = _positive_int(
        resources.get("parquet_bytes_per_row_upper_bound"), "Parquet row bound"
    )
    parquet_fixed_bytes = _positive_int(
        resources.get("parquet_fixed_bytes_per_partition_upper_bound"),
        "Parquet partition bound",
    )
    parquet = rows * parquet_row_bytes + partitions * parquet_fixed_bytes
    control = _positive_int(
        resources.get("control_artifacts_bytes_upper_bound"), "control bound"
    )
    atomic = _positive_int(
        resources.get("largest_next_atomic_artifact_bytes_upper_bound"),
        "atomic artifact bound",
    )
    require(
        atomic == rows * parquet_row_bytes + parquet_fixed_bytes,
        "frozen formula mismatch: atomic artifact",
    )
    working = staging * 3 + parquet + control + atomic
    artifact_count = 7 + 2 * input_files + 2 * partitions
    artifact_sweep = staging + parquet + control
    maximum_runs_in_one_partition = _ceil_ratio(rows, sort_chunk_rows)
    maximum_runs = maximum_runs_in_one_partition + partitions - 1

    exact_checks = {
        "staging": (staging, resources.get("staging_payload_bytes_exact")),
        "spool": (
            staging,
            resources.get("partition_spool_payload_bytes_upper_bound"),
        ),
        "runs": (
            staging,
            resources.get("external_run_payload_bytes_upper_bound"),
        ),
        "parquet": (parquet, resources.get("parquet_payload_bytes_upper_bound")),
        "working": (
            working,
            resources.get("worst_case_remaining_bytes_before_multiplier"),
        ),
        "artifact_count": (
            artifact_count,
            completion.get("exact_receipt_artifact_count_full"),
        ),
        "spool_files": (partitions, spool_contract.get("exact_file_count")),
        "parquet_files": (
            partitions,
            publication.get("exact_parquet_files_full"),
        ),
    }
    for name, (observed, expected) in exact_checks.items():
        require(observed == expected, f"frozen formula mismatch: {name}")
    maximum_open_run_files = _positive_int(
        run_contract.get("maximum_open_run_files"), "run file limit"
    )
    require(
        maximum_runs_in_one_partition
        == run_contract.get("theoretical_maximum_runs_in_one_partition"),
        "frozen formula mismatch: per-partition run files",
    )
    require(
        maximum_runs_in_one_partition <= maximum_open_run_files,
        "per-partition run-file bound exceeds the open-file limit",
    )
    maximum_open_spool_files = _positive_int(
        spool_contract.get("maximum_open_files"), "spool open-file limit"
    )
    require(
        maximum_open_spool_files >= partitions,
        "spool open-file limit does not cover every partition",
    )

    rate_fields = {
        name: _positive_int(rates.get(name), name)
        for name in (
            "sequential_read_bytes_per_second",
            "sequential_write_bytes_per_second",
            "sha256_bytes_per_second",
            "demultiplex_records_per_second",
            "stable_sort_records_per_second",
            "worst_case_heap_merge_records_per_second",
            "parquet_encode_records_per_second",
            "parquet_decode_records_per_second",
            "data_file_fsync_operations_per_second",
            "artifact_contracts_per_second",
            "additive_time_safety_multiplier_milli",
        )
    }
    require(
        rates.get("status") == "UNVALIDATED_ASSUMPTIONS_NOT_MEASURED_FLOORS"
        and rates.get("rates_are_hypothetical_not_benchmarked") is True
        and rates.get("conditional_estimates_are_not_unconditional_time_bounds")
        is True,
        "hypothetical-rate semantics drift",
    )
    multiplier = rate_fields["additive_time_safety_multiplier_milli"]
    require(multiplier >= 1000, "time safety multiplier is below one")

    spool_hash_header_instances = input_files
    spool_hash_bytes = (
        staging * 2
        + rows * FRAMED_RECORD_HASH_BYTES_PER_RECORD
        + spool_hash_header_instances * FRAMED_RECORD_HASH_HEADER_BYTES_PER_DIGEST
    )
    spool_fsyncs = _positive_int(
        spool_contract.get("maximum_close_and_fsync_operations_full"),
        "spool fsyncs",
    )
    spool_time = _conditional_time(
        {
            "payload_read": (
                staging,
                rate_fields["sequential_read_bytes_per_second"],
            ),
            "payload_write": (
                staging,
                rate_fields["sequential_write_bytes_per_second"],
            ),
            "demultiplex_records": (
                rows,
                rate_fields["demultiplex_records_per_second"],
            ),
            "hash_work": (spool_hash_bytes, rate_fields["sha256_bytes_per_second"]),
            "data_file_fsync": (
                spool_fsyncs,
                rate_fields["data_file_fsync_operations_per_second"],
            ),
        },
        multiplier,
    )

    sort_read = 2 * staging + 2 * parquet
    sort_write = staging + parquet
    sort_hash_bytes = (
        3 * staging
        + 2 * rows * FRAMED_RECORD_HASH_BYTES_PER_RECORD
        + parquet
        + 2 * partitions * FRAMED_RECORD_HASH_HEADER_BYTES_PER_DIGEST
    )
    parquet_files = partitions
    sort_data_file_fsyncs = maximum_runs + parquet_files
    sort_time = _conditional_time(
        {
            "payload_read": (
                sort_read,
                rate_fields["sequential_read_bytes_per_second"],
            ),
            "payload_write": (
                sort_write,
                rate_fields["sequential_write_bytes_per_second"],
            ),
            "stable_sort_records": (
                rows,
                rate_fields["stable_sort_records_per_second"],
            ),
            "worst_case_heap_merge_records": (
                rows,
                rate_fields["worst_case_heap_merge_records_per_second"],
            ),
            "parquet_encode_records": (
                rows,
                rate_fields["parquet_encode_records_per_second"],
            ),
            "parquet_decode_records": (
                rows,
                rate_fields["parquet_decode_records_per_second"],
            ),
            "hash_work": (sort_hash_bytes, rate_fields["sha256_bytes_per_second"]),
            "data_file_fsync": (
                sort_data_file_fsyncs,
                rate_fields["data_file_fsync_operations_per_second"],
            ),
        },
        multiplier,
    )

    rehash_time = _conditional_time(
        {
            "payload_read": (
                artifact_sweep,
                rate_fields["sequential_read_bytes_per_second"],
            ),
            "sha256_work": (
                artifact_sweep,
                rate_fields["sha256_bytes_per_second"],
            ),
            "artifact_contracts": (
                artifact_count,
                rate_fields["artifact_contracts_per_second"],
            ),
        },
        multiplier,
    )

    available_disk = _positive_int(
        hardware.get("output_filesystem_available_bytes"), "available disk"
    )
    available_ram = _positive_int(hardware.get("mem_available_bytes"), "available RAM")
    observed_nofile = _positive_int(
        hardware.get("nofile_soft_limit"), "nofile soft limit"
    )
    fresh_disk = _positive_int(
        resources.get("required_free_disk_bytes_full_fresh"), "fresh disk floor"
    )
    required_ram = _positive_int(
        resources.get("minimum_available_ram_bytes_full"), "RAM floor"
    )
    required_nofile = _positive_int(
        resources.get("minimum_nofile_soft_limit"), "nofile floor"
    )
    # Three artifact-contract sweeps plus reconstruction can reread the large
    # staging/Parquet payload twice.  Up to three additional aggregate control
    # passes cover JSON/Markdown reconstruction reads.  This is a logical-read
    # envelope for the receipt/output reconstruction surface, not parent-control
    # revalidation, own-input revalidation, metadata, or physical media IO.
    receipt_surface_validation_read = 5 * staging + 5 * parquet + 6 * control

    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": STATUS,
        "gate_status": GATE_STATUS,
        "production_execution_status": PRODUCTION_STATUS,
        "contract_binding": {
            "config_path": binding["config_path"],
            "config_sha256": binding["config_sha256"],
            "runner_path": binding["runner_path"],
            "runner_sha256": binding["runner_sha256"],
            "runner_imported_or_executed": False,
        },
        "contract_anchors": {
            "retained_rows": rows,
            "record_bytes": record_bytes,
            "partition_count": partitions,
            "sort_chunk_rows": sort_chunk_rows,
            "receipt_artifact_count": artifact_count,
            "staging_payload_bytes_exact": staging,
            "parquet_payload_bytes_upper_bound": parquet,
            "control_artifacts_bytes_upper_bound": control,
            "largest_next_atomic_artifact_bytes_upper_bound": atomic,
            "worst_case_disk_workspace_bytes_before_multiplier": working,
            "framed_record_hash_header_bytes_per_digest": (
                FRAMED_RECORD_HASH_HEADER_BYTES_PER_DIGEST
            ),
        },
        "hardware_snapshot": {
            **hardware,
            "comparison_status": "OBSERVATION_ONLY_NOT_PREFLIGHT_OR_ACCEPTANCE",
            "fresh_disk_requirement_bytes": fresh_disk,
            "disk_headroom_after_fresh_requirement_bytes": available_disk - fresh_disk,
            "available_disk_to_fresh_requirement_ratio_milli": (
                available_disk * 1000 // fresh_disk
            ),
            "required_available_ram_bytes": required_ram,
            "available_ram_to_requirement_ratio_milli": (
                available_ram * 1000 // required_ram
            ),
            "required_nofile_soft_limit": required_nofile,
            "nofile_to_requirement_ratio_milli": (
                observed_nofile * 1000 // required_nofile
            ),
            "snapshot_may_be_stale_at_execution": True,
        },
        "hypothetical_rate_floor_scenario": {
            **rates,
            "all_rates_empirically_validated": False,
        },
        "phase_estimates": {
            "spool_and_numpy_demultiplex": {
                "bound_kind": "EXACT_LOGICAL_PAYLOAD_WITH_ENGINEERING_HASH_MODEL",
                "logical_read_bytes_exact": staging,
                "logical_write_bytes_exact": staging,
                "logical_io_bytes_exact": 2 * staging,
                "hash_input_bytes_engineering_upper_bound": spool_hash_bytes,
                "framed_record_hash_header_instances": spool_hash_header_instances,
                "existing_staging_payload_bytes": staging,
                "incremental_spool_workspace_bytes_exact": staging,
                "combined_staging_and_spool_payload_bytes_exact": 2 * staging,
                "combined_payload_plus_control_bound_bytes": 2 * staging + control,
                "aggregate_spool_buffer_ram_bytes": resources[
                    "spool_buffer_bytes_upper_bound"
                ],
                "demultiplex_working_ram_estimate_bytes": resources[
                    "spool_demultiplex_working_bytes_preflight_estimate"
                ],
                "maximum_open_spool_files": maximum_open_spool_files,
                "maximum_data_file_close_and_fsync_operations": spool_fsyncs,
                "conditional_time": spool_time,
            },
            "external_sort_and_parquet_publication": {
                "bound_kind": "CONSERVATIVE_LOGICAL_IO_UPPER_BOUND",
                "logical_read_bytes_upper_bound": sort_read,
                "logical_write_bytes_upper_bound": sort_write,
                "logical_io_bytes_upper_bound": sort_read + sort_write,
                "hash_input_bytes_engineering_upper_bound": sort_hash_bytes,
                "framed_record_hash_header_instances": 2 * partitions,
                "maximum_external_run_files_global": maximum_runs,
                "maximum_external_run_files_one_partition": (
                    maximum_runs_in_one_partition
                ),
                "maximum_open_run_files_one_partition": maximum_open_run_files,
                "maximum_external_run_files_formula": (
                    "ceil(retained_rows / sort_chunk_rows) + partition_count - 1"
                ),
                "maximum_parquet_files": parquet_files,
                "maximum_data_file_fsync_operations": sort_data_file_fsyncs,
                "worst_case_heap_merge_records_upper_bound": rows,
                "peak_workspace_bytes_upper_bound": working,
                "peak_workspace_includes": [
                    "staging",
                    "spool",
                    "external_runs",
                    "all_parquet_outputs",
                    "control_artifacts",
                    "largest_next_atomic_artifact",
                ],
                "conditional_time": sort_time,
            },
            "one_9185_artifact_final_rehash_sweep": {
                "bound_kind": "CONSERVATIVE_LOGICAL_READ_UPPER_BOUND",
                "artifact_count_exact": artifact_count,
                "logical_read_and_hash_bytes_upper_bound": artifact_sweep,
                "published_payload_and_control_workspace_bytes_upper_bound": (
                    artifact_sweep
                ),
                "incremental_workspace_bytes_for_read_only_rehash": 0,
                "conditional_time": rehash_time,
            },
            "same_invocation_receipt_surface_validation_context": {
                "status": "ADDITIONAL_COST_NOT_COVERED_BY_ONE_SWEEP_LABEL",
                "logical_read_bytes_conservative_upper_bound": (
                    receipt_surface_validation_read
                ),
                "formula": (
                    "5*staging + 5*parquet_upper_bound + 6*control_upper_bound"
                ),
                "semantics": (
                    "three artifact-contract sweeps, two additional large-payload "
                    "reconstruction reads, and three additional aggregate control "
                    "read allowances"
                ),
                "scope": ("RECEIPT_ARTIFACT_AND_OUTPUT_RECONSTRUCTION_READS_ONLY"),
                "excluded_from_this_bound": [
                    "parent_control_revalidation_reads",
                    "own_frozen_input_revalidation_reads",
                    "filesystem_metadata_and_directory_walk_bytes",
                ],
                "physical_device_bytes_may_differ_due_to_cache_and_writeback": True,
            },
        },
        "review_state": review_state,
        "remaining_empirical_requirements": [
            "run a bounded nonpromotable staging-derived 4096-handle demux benchmark",
            "measure peak RSS, open descriptors, aggregate throughput, and fsync tail latency",
            "benchmark both representative and multi-run/adversarial partition shapes",
            "measure external sort, direct Parquet publication, decoded validation, and SHA-256 separately",
            "measure a 9185-entry artifact walk and hash sweep with representative file sizes",
            "repeat under the intended quiet-window process and IO load",
            "replace hypothetical rates only with lower confidence bounds from repeated measurements",
            "obtain independent review and explicit operator acceptance before any gate decision",
        ],
        "claim_boundary": {
            "analytic_estimate_is_benchmark": False,
            "hardware_snapshot_is_live_execution_preflight": False,
            "one_file_pilot_extrapolation_is_sufficient": False,
            "downstream_capacity_gate_passed": False,
            "full_execution_authorized": False,
            "production_receipt_created": False,
        },
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help="Frozen repository contract; only the default path is accepted.",
    )
    parser.add_argument(
        "--assumptions",
        type=Path,
        default=DEFAULT_ASSUMPTIONS,
        help="Repository capacity-review input; only the default path is accepted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    contract, assumptions = load_inputs(
        contract_path=args.contract,
        assumptions_path=args.assumptions,
    )
    report = build_review(contract, assumptions)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
