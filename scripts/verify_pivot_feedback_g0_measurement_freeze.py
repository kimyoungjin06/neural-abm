#!/usr/bin/env python
"""Validate the pending researcher-pivot G0 measurement-freeze overlay.

This verifier has no empirical execution or promotion surface.  The repository
contract is deliberately pending: validation can establish only that its
structure and immutable parent bindings are intact.  Sealing requires a later,
versioned successor with resolved quantitative measurements and external G-1
and holdout-access receipts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Literal, Mapping, Sequence

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml"
)
VALIDATION_SCHEMA_VERSION = "pivot_feedback_g0_measurement_freeze_validation.v1"
INVALID_STATUS = "BLOCKED_INVALID_G0_MEASUREMENT_FREEZE_CONTRACT"
PENDING_SEAL_STATUS = "BLOCKED_PENDING_G0_MEASUREMENT_FREEZE"
BLOCKED_COMMAND_STATUS = "BLOCKED_G0_MEASUREMENT_FREEZE_COMMAND"
BLOCKED_COMMANDS = frozenset({"run", "execute", "authorize", "promote"})
MAX_CONTRACT_BYTES = 1024 * 1024
MAX_CONTRACT_NESTING_DEPTH = 64
ALLOWED_CONTRACT_YAML_TAGS = frozenset(
    {
        "tag:yaml.org,2002:map",
        "tag:yaml.org,2002:seq",
        "tag:yaml.org,2002:str",
        "tag:yaml.org,2002:null",
        "tag:yaml.org,2002:bool",
        "tag:yaml.org,2002:int",
        "tag:yaml.org,2002:float",
    }
)

PEER_PATH = "experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml"
PEER_FILE_SHA256 = "99eeb68b632f9f9a250f50320cfc7a293e3595de81984419d1528e2ee12aa6f8"
PEER_CANONICAL_SHA256 = (
    "82f9b6800023ee7c963727f0447b658d71cfafd9d04c2397aef8452a3e2a1629"
)
MATRIX_PATH = "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml"
MATRIX_FILE_SHA256 = "6fd3e2c7aae7a7aac085ed73ba4bf3d64a91c02d60dce8a9a6a262591a0c934a"
MATRIX_CANONICAL_SHA256 = (
    "067ed8a61f3e6d920287377bf781083264ba661ede1ac779230292006bea28b7"
)

PEER_GATE_STATE = {
    "g_minus_1_data_readiness": "PENDING",
    "g0_measurement_freeze": "PENDING",
    "empirical_g1_signature": "NOT_RUN",
    "external_causal_axis": "NOT_RUN",
    "execution_authorized": False,
}
MATRIX_GATE_STATE = {
    "g_minus_1_data_readiness": "PENDING",
    "g0_measurement_freeze": "PENDING",
    "empirical_g1_signature": "NOT_RUN",
    "pass_g1": False,
    "execution_authorized": False,
}
PEER_DECLARED_PARENT = {
    "relative_path": MATRIX_PATH,
    "required_schema_version": "pivot_feedback_cross_field_matrix.v1",
    "binding_status": "PATH_AND_SCHEMA_ONLY_PENDING_G0_HASH_FREEZE",
}

REQUIRED_RESOLUTION_SECTIONS = (
    "population_and_author_eligibility",
    "pivot_and_robustness_definitions",
    "topic_attention_and_surge",
    "temporal_availability_and_maturation",
    "peer_exposure_and_source_assignment",
    "capacity_bridge_and_entry_mode",
    "risk_set_sampling_and_weights",
    "outcome_horizons_and_h3_timing",
    "negative_control_decision_gates",
    "minimum_support_and_suppression",
    "partition_and_atomic_fold_resolution",
    "estimators_and_uncertainty",
    "hierarchical_multiplicity",
    "g2_predictive_gate",
    "g3_simulation_gate",
)
NEGATIVE_CONTROL_IDS = (
    "lead_attention",
    "same_peer_unrelated_topic_success",
    "post_decision_future_citation_success",
    "future_tie",
    "degree_preserving_shuffled_network",
    "matched_non_neighbor",
    "matched_non_surge_topic",
)
CAPACITY_CELLS = (
    "NEITHER",
    "PUBLIC_SUCCESS_ONLY",
    "CAPACITY_BRIDGE_ONLY",
    "BOTH",
)
ENTRY_MODE_IDS = (
    "chaperoned_entry",
    "independent_entry",
    "unresolved_entry_mode",
)
EXPECTED_TEMPORAL_PARTITIONS = (
    ("discovery_calibration", None, 2012, 2025, False),
    ("g1_empirical_replication", 2013, 2016, 2025, False),
    ("model_selection", 2017, 2019, 2025, False),
    ("g4_temporal_holdout", 2020, 2022, 2025, True),
)


class G0ContractError(RuntimeError):
    """Raised when the pending G0 contract cannot be validated safely."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StrictContractLoader(yaml.SafeLoader):
    """Small YAML subset with unambiguous mappings and no graph features."""

    def compose_node(self, parent: Any, index: Any) -> yaml.Node:
        if self.check_event(yaml.AliasEvent):
            event = self.peek_event()
            raise ConstructorError(
                None,
                None,
                "YAML aliases are forbidden in the G0 contract",
                event.start_mark,
            )
        node = super().compose_node(parent, index)
        if node.tag not in ALLOWED_CONTRACT_YAML_TAGS:
            raise ConstructorError(
                None,
                None,
                f"YAML tag is forbidden in the G0 contract: {node.tag}",
                node.start_mark,
            )
        return node

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                "expected a mapping node",
                node.start_mark,
            )
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise ConstructorError(
                    None,
                    None,
                    "YAML merge keys are forbidden in the G0 contract",
                    key_node.start_mark,
                )
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError as exc:
                raise ConstructorError(
                    None,
                    None,
                    "unhashable YAML mapping key is forbidden in the G0 contract",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    None,
                    None,
                    f"duplicate YAML mapping key is forbidden: {key}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


class Lifecycle(StrictModel):
    design_status: Literal["PROSPECTIVE_PENDING_G0_VALUES"]
    freeze_status: Literal["PENDING_G0_MEASUREMENT_FREEZE"]
    confirmatory_outcomes_inspected: Literal[False]
    g4_holdout_opened: Literal[False]
    pass_g0: Literal[False]
    empirical_g1_signature: Literal["NOT_RUN"]
    pass_g1: Literal[False]
    execution_authorized: Literal[False]
    config_alone_can_authorize_execution: Literal[False]
    successor_required_for_freeze: Literal[True]


class PeerParentBinding(StrictModel):
    relative_path: Literal[PEER_PATH]
    required_file_sha256: Literal[PEER_FILE_SHA256]
    required_canonical_sha256: Literal[PEER_CANONICAL_SHA256]
    required_schema_version: Literal["pivot_feedback_peer_mechanism_registry.v1"]
    required_label: Literal["pivot_feedback_peer_mechanism_registry_v0"]
    required_mode: Literal["PROSPECTIVE_G0_COMPONENT"]
    required_status: Literal["PENDING_G0_MEASUREMENT_FREEZE"]
    parent_mutation_allowed: Literal[False]
    fallback_parent_allowed: Literal[False]


class MatrixParentBinding(StrictModel):
    relative_path: Literal[MATRIX_PATH]
    required_file_sha256: Literal[MATRIX_FILE_SHA256]
    required_canonical_sha256: Literal[MATRIX_CANONICAL_SHA256]
    required_schema_version: Literal["pivot_feedback_cross_field_matrix.v1"]
    required_label: Literal["pivot_feedback_cross_field_matrix_engineering_v0"]
    required_mode: Literal["ENGINEERING_ONLY_PLAN"]
    parent_mutation_allowed: Literal[False]
    fallback_parent_allowed: Literal[False]


class CrossBinding(StrictModel):
    peer_declared_parent_relative_path: Literal[MATRIX_PATH]
    peer_declared_parent_schema_version: Literal["pivot_feedback_cross_field_matrix.v1"]
    peer_declared_binding_status: Literal["PATH_AND_SCHEMA_ONLY_PENDING_G0_HASH_FREEZE"]
    resolution_overlay_only: Literal[True]
    rewrite_parent_allowed: Literal[False]


class FoundationParents(StrictModel):
    peer_mechanism_registry: PeerParentBinding
    cross_field_matrix: MatrixParentBinding
    cross_binding: CrossBinding


class PendingGMinusOneBinding(StrictModel):
    state: Literal["PENDING"]
    receipt_relative_path: None = None
    receipt_sha256: None = None
    required_schema_version: Literal[
        "pivot_feedback_g_minus_1_data_readiness_outputs.v1"
    ]
    required_status: Literal["PASS_G_MINUS_1_DATA_READINESS"]
    result_fingerprint_sha256: None = None


class PendingHoldoutBinding(StrictModel):
    state: Literal["PENDING"]
    receipt_relative_path: None = None
    receipt_sha256: None = None
    required_schema_version: Literal["pivot_feedback_holdout_access_seal_outputs.v1"]
    required_status: Literal["PASS_CONFIRMATORY_PARTITIONS_SEALED_AND_UNOPENED"]
    result_fingerprint_sha256: None = None


class OffsetWindow(StrictModel):
    start_offset: StrictInt
    end_offset: StrictInt
    inclusive: Literal[True]


class AttentionWindows(StrictModel):
    recent_window: OffsetWindow
    reference_window: OffsetWindow


class TemporalClocks(StrictModel):
    resolution: Literal["ANNUAL"]
    portfolio_history: OffsetWindow
    pre_entry_graph: OffsetWindow
    attention_acceleration: AttentionWindows
    persistence_outcome: OffsetWindow
    response_offsets: tuple[StrictInt, ...]
    pretrend_offsets: tuple[StrictInt, ...]
    public_availability_rule: Literal["AVAILABLE_AT_LESS_THAN_OR_EQUAL_TO_T_MINUS_1"]

    @model_validator(mode="after")
    def validate_clocks(self) -> "TemporalClocks":
        observed = (
            (
                self.portfolio_history.start_offset,
                self.portfolio_history.end_offset,
            ),
            (self.pre_entry_graph.start_offset, self.pre_entry_graph.end_offset),
            (
                self.attention_acceleration.recent_window.start_offset,
                self.attention_acceleration.recent_window.end_offset,
            ),
            (
                self.attention_acceleration.reference_window.start_offset,
                self.attention_acceleration.reference_window.end_offset,
            ),
            (
                self.persistence_outcome.start_offset,
                self.persistence_outcome.end_offset,
            ),
        )
        expected = ((-3, -1), (-5, -1), (-3, -1), (-6, -4), (1, 3))
        if observed != expected:
            raise ValueError("fixed annual clock windows differ from the protocol")
        if self.response_offsets != (0, 1, 2):
            raise ValueError("response offsets must remain k=0,1,2")
        if self.pretrend_offsets != (-3, -2, -1):
            raise ValueError("pretrend offsets must remain -3,-2,-1")
        return self


class H3PrimaryChoice(StrictModel):
    endpoint: Literal[
        "EARLIEST_ARGMAX_ADJUSTED_ENTRY_HAZARD_K_0_1_2_MINUS_"
        "EARLIEST_ARGMAX_ADJUSTED_THREE_YEAR_PERSISTENCE_PROBABILITY_K_0_1_2"
    ]
    minimum_point_gap_cohort_years: Literal[1]
    interval_sidedness: Literal["ONE_SIDED"]
    confidence_level: Literal[0.95]
    lower_bound_rule: Literal["GREATER_THAN_ZERO"]
    field_balanced: Literal[True]
    episode_clustered: Literal[True]
    field_weighting: Literal["EQUAL"]
    within_field_episode_weighting: Literal["EQUAL"]
    leave_one_field_out_direction_rule: Literal["ALL_POSITIVE"]
    alternative_pivot_construction_count: Literal[1]
    alternative_pivot_predeclared: Literal[True]
    alternative_pivot_substantively_different: Literal[True]


class MultiplicityChoice(StrictModel):
    first_test: Literal["H3"]
    family_wise_alpha: Literal[0.05]
    conditional_family: tuple[Literal["H1", "H2", "H4"], ...]
    conditional_method: Literal["HOLM"]
    conditional_family_opens_only_if_h3_passes: Literal[True]

    @model_validator(mode="after")
    def validate_family(self) -> "MultiplicityChoice":
        if self.conditional_family != ("H1", "H2", "H4"):
            raise ValueError("conditional family must be ordered H1,H2,H4")
        return self


class FullPanelChoice(StrictModel):
    field_count: Literal[4]
    g1_field_count: Literal[3]
    sealed_g4_field_count: Literal[1]
    g4_domain_and_author_cold: Literal[True]
    minimum_independent_episodes_per_field: Literal[3]


class DiscoveryPilotChoice(StrictModel):
    label: Literal["DISCOVERY_ONLY"]
    field_count: Literal[2]
    independent_shocks_per_field: Literal[2]
    may_pass_g1: Literal[False]
    may_tune_confirmatory_period: Literal[False]


class FixedTemporalPartition(StrictModel):
    partition_id: Literal[
        "discovery_calibration",
        "g1_empirical_replication",
        "model_selection",
        "g4_temporal_holdout",
    ]
    entry_start_year: StrictInt | None
    entry_end_year: StrictInt
    outcome_followup_through_year: StrictInt
    sealed: StrictBool


class PeerMechanismChoices(StrictModel):
    negative_control_ids: tuple[str, ...]
    capacity_cells: tuple[str, ...]
    entry_mode_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_choices(self) -> "PeerMechanismChoices":
        if self.negative_control_ids != NEGATIVE_CONTROL_IDS:
            raise ValueError(
                "all seven negative controls must remain ordered and exact"
            )
        if self.capacity_cells != CAPACITY_CELLS:
            raise ValueError("capacity contrast must retain all four cells")
        if self.entry_mode_ids != ENTRY_MODE_IDS:
            raise ValueError("entry-mode contrast must retain all three modes")
        return self


class EngineeringRNGChoice(StrictModel):
    algorithm: Literal["numpy.random.SeedSequence"]
    base_seed: Literal[20260718]
    replicate_seeds: tuple[StrictInt, ...]
    role: Literal["ENGINEERING_PAIRING_ONLY"]
    monte_carlo_sufficiency_established: Literal[False]
    confirmatory_draw_count_frozen: Literal[False]

    @model_validator(mode="after")
    def validate_replicate_seeds(self) -> "EngineeringRNGChoice":
        if self.replicate_seeds != (1, 2, 3):
            raise ValueError("engineering replicate seeds must remain 1,2,3")
        return self


class FixedDesignChoices(StrictModel):
    temporal_clocks: TemporalClocks
    h3_primary: H3PrimaryChoice
    multiplicity: MultiplicityChoice
    full_panel: FullPanelChoice
    discovery_pilot: DiscoveryPilotChoice
    temporal_partitions: tuple[FixedTemporalPartition, ...]
    peer_mechanism: PeerMechanismChoices
    engineering_rng: EngineeringRNGChoice

    @model_validator(mode="after")
    def validate_partitions(self) -> "FixedDesignChoices":
        observed = tuple(
            (
                partition.partition_id,
                partition.entry_start_year,
                partition.entry_end_year,
                partition.outcome_followup_through_year,
                partition.sealed,
            )
            for partition in self.temporal_partitions
        )
        if observed != EXPECTED_TEMPORAL_PARTITIONS:
            raise ValueError("fixed temporal partitions differ from the protocol")
        return self


class UnresolvedQuantitativeChoices(StrictModel):
    author_minimum_history_works: None = None
    primary_pivot_thresholds: None = None
    attention_surge_threshold: None = None
    public_success_threshold: None = None
    citation_maturation_lag_years: None = None
    negative_control_equivalence_margins: None = None
    negative_sampling_ratio: None = None
    inverse_probability_weight_truncation: None = None
    minimum_cohort_size: None = None
    small_cell_suppression_threshold: None = None
    capacity_cell_support_threshold: None = None
    collinearity_threshold: None = None
    g1_replication_tolerance: None = None
    g2_score_margin: None = None
    g2_maximum_calibration_degradation: None = None
    g2_macro_trajectory_tolerance: None = None
    simulation_draws_or_mcse_stopping_rule: None = None
    parameter_budget: None = None
    tuning_budget: None = None


class PendingResolution(StrictModel):
    state: Literal["PENDING"]
    resolved_measurements: None = None
    successor_required: Literal[True]
    unresolved_quantitative_choices: UnresolvedQuantitativeChoices
    unresolved_required_sections: tuple[str, ...]

    @model_validator(mode="after")
    def validate_required_sections(self) -> "PendingResolution":
        if self.unresolved_required_sections != REQUIRED_RESOLUTION_SECTIONS:
            raise ValueError(
                "pending G0 resolution sections must be complete and ordered"
            )
        return self


class PeerSubtreeDigests(StrictModel):
    exposure_constructs_sha256: Literal[
        "e6c967d329717811789fb405a4d1b0c938f5dfe5d25509cb559c402f0e9741f8"
    ]
    entry_mode_strata_sha256: Literal[
        "51380773712a245d37bac17cbe96558b1479dd8e9f4d7644b7e0b01ff46ef190"
    ]
    negative_controls_sha256: Literal[
        "cbd268699151056df1648858f55277535f27b60c613ac1f4e153ac63569d15f3"
    ]
    claim_levels_sha256: Literal[
        "299495583072eb4d8575276d428351f156045bb5952a99317268d5a7ef00b14e"
    ]
    claim_downgrade_rules_sha256: Literal[
        "774161ea582c3521f4bf2c9d601c320e8d5905f97371a43f7299960a2034e6b9"
    ]


class MatrixSubtreeDigests(StrictModel):
    candidate_universe_sha256: Literal[
        "67a1eff78c23965cfb62ee0f762eb8cb890088094e4b9da91326384d37ad27ad"
    ]
    selection_rules_sha256: Literal[
        "ff3429e7fa58214436403c4c074c02bd49055c083673e22de0fea4ed8be54a11"
    ]
    temporal_partitions_sha256: Literal[
        "8fbb4131d3bdbe09049fcd44dc0ff476d964a5a2009efb6bc2653e13b04449b0"
    ]


class InheritedSubtreeDigests(StrictModel):
    serialization: Literal["JSON_SORTED_KEYS_COMPACT_UTF8_ALLOW_NAN_FALSE"]
    peer_mechanism_registry: PeerSubtreeDigests
    cross_field_matrix: MatrixSubtreeDigests
    override_allowed: Literal[False]


class CompletionContract(StrictModel):
    validation_status: Literal["VALID_PENDING_G0_MEASUREMENT_FREEZE"]
    seal_status_if_complete: Literal["PASS_G0_MEASUREMENT_FREEZE_CONTRACT_ONLY"]
    pending_contract_sealable: Literal[False]
    seal_receipt_required_before_g1: Literal[True]
    seal_does_not_establish_scientific_effect: Literal[True]
    seal_does_not_authorize_execution_by_itself: Literal[True]
    canonical_serialization: Literal["JSON_SORTED_KEYS_COMPACT_UTF8_ALLOW_NAN_FALSE"]


class FailurePolicy(StrictModel):
    invalid_contract_status: Literal["BLOCKED_INVALID_G0_MEASUREMENT_FREEZE_CONTRACT"]
    pending_seal_status: Literal["BLOCKED_PENDING_G0_MEASUREMENT_FREEZE"]
    parent_mismatch_action: Literal["FAIL_CLOSED_NO_RECEIPT"]
    unresolved_value_action: Literal["REQUIRE_VERSIONED_SUCCESSOR"]
    forbidden_commands: tuple[Literal["run", "execute", "authorize", "promote"], ...]
    failure_writes_pass_receipt: Literal[False]

    @model_validator(mode="after")
    def validate_forbidden_commands(self) -> "FailurePolicy":
        if self.forbidden_commands != ("run", "execute", "authorize", "promote"):
            raise ValueError("forbidden G0 command list must be complete and ordered")
        return self


class G0MeasurementFreezeContract(StrictModel):
    schema_version: Literal["pivot_feedback_g0_measurement_freeze.v1"]
    label: Literal["pivot_feedback_g0_measurement_freeze_v0"]
    mode: Literal["PROSPECTIVE_FAIL_CLOSED_G0_RESOLUTION_OVERLAY"]
    lifecycle: Lifecycle
    foundation_parents: FoundationParents
    g_minus_1_binding: PendingGMinusOneBinding
    holdout_access_binding: PendingHoldoutBinding
    fixed_design_choices: FixedDesignChoices
    resolution: PendingResolution
    inherited_subtree_digests: InheritedSubtreeDigests
    completion_contract: CompletionContract
    failure_policy: FailurePolicy


@dataclass(frozen=True)
class LoadedContract:
    value: G0MeasurementFreezeContract
    file_sha256: str
    canonical_sha256: str
    path: Path


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G0ContractError(message)


def _require_scalar_type_fidelity(
    raw: Any,
    normalized: Any,
    *,
    field: str = "contract",
) -> None:
    """Reject Pydantic scalar coercions while allowing YAML lists as tuples."""

    if isinstance(raw, Mapping) and isinstance(normalized, Mapping):
        _require(set(raw) == set(normalized), f"{field} key set differs")
        for key in raw:
            _require_scalar_type_fidelity(
                raw[key],
                normalized[key],
                field=f"{field}.{key}",
            )
        return
    if isinstance(raw, list) and isinstance(normalized, (list, tuple)):
        _require(len(raw) == len(normalized), f"{field} sequence length differs")
        for index, (raw_item, normalized_item) in enumerate(zip(raw, normalized)):
            _require_scalar_type_fidelity(
                raw_item,
                normalized_item,
                field=f"{field}[{index}]",
            )
        return
    _require(type(raw) is type(normalized), f"{field} scalar type differs")


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def read_contract_bytes(path: Path) -> bytes:
    """Read one small unchanged regular file without following a final symlink."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise G0ContractError("cannot stat G0 contract") from exc
    _require(stat.S_ISREG(before.st_mode), "G0 contract must be a regular file")
    _require(
        before.st_size <= MAX_CONTRACT_BYTES,
        "G0 contract exceeds the 1 MiB safety cap",
    )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise G0ContractError("cannot open G0 contract safely") from exc
    try:
        opened = os.fstat(descriptor)
        _require(
            _stat_identity(opened) == _stat_identity(before),
            "G0 contract changed before descriptor read",
        )
        blocks: list[bytes] = []
        total = 0
        while total <= MAX_CONTRACT_BYTES:
            block = os.read(
                descriptor,
                min(64 * 1024, MAX_CONTRACT_BYTES + 1 - total),
            )
            if not block:
                break
            blocks.append(block)
            total += len(block)
        after_descriptor = os.fstat(descriptor)
    except OSError as exc:
        raise G0ContractError("cannot read G0 contract safely") from exc
    finally:
        os.close(descriptor)

    _require(total <= MAX_CONTRACT_BYTES, "G0 contract exceeds the 1 MiB safety cap")
    _require(total == opened.st_size, "G0 contract byte count changed while reading")
    _require(
        _stat_identity(after_descriptor) == _stat_identity(opened),
        "G0 contract changed while reading",
    )
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise G0ContractError("cannot restat G0 contract after reading") from exc
    _require(
        _stat_identity(after_path) == _stat_identity(before),
        "G0 contract path changed while reading",
    )
    return b"".join(blocks)


def validate_contract_nesting(payload: str) -> None:
    """Reject deeply nested YAML before recursive object construction."""

    depth = 0
    try:
        for event in yaml.parse(payload):
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
                _require(
                    depth <= MAX_CONTRACT_NESTING_DEPTH,
                    "G0 contract exceeds the YAML nesting safety cap",
                )
            elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
    except RecursionError as exc:
        raise G0ContractError(
            "G0 contract exceeds the YAML nesting safety cap"
        ) from exc


def _repo_relative_regular_file(relative_path: str) -> Path:
    candidate = Path(relative_path)
    _require(not candidate.is_absolute(), "parent path must be repository-relative")
    _require(".." not in candidate.parts, "parent path may not contain '..'")
    resolved = (REPO_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise G0ContractError("parent path escapes the repository") from exc
    _require(resolved.is_file(), "parent binding must target a regular file")
    _require(
        not (REPO_ROOT / candidate).is_symlink(), "parent binding may not be a symlink"
    )
    return resolved


def _yaml_mapping(path: Path, *, field: str) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, Mapping), f"{field} must be a YAML mapping")
    return dict(raw)


def _validate_parent_identity(
    raw: Mapping[str, Any],
    *,
    schema_version: str,
    label: str,
    mode: str,
    status: str | None,
    gate_state: Mapping[str, Any],
    field: str,
) -> None:
    _require(raw.get("schema_version") == schema_version, f"{field} schema mismatch")
    _require(raw.get("label") == label, f"{field} label mismatch")
    _require(raw.get("mode") == mode, f"{field} mode mismatch")
    if status is not None:
        _require(raw.get("status") == status, f"{field} status mismatch")
    _require(raw.get("gate_state") == dict(gate_state), f"{field} gate-state mismatch")


def _validate_subtree_digest(
    raw: Mapping[str, Any], key: str, expected_sha256: str, *, field: str
) -> None:
    _require(key in raw, f"{field}.{key} is missing")
    _require(
        canonical_sha256(raw[key]) == expected_sha256,
        f"{field}.{key} canonical SHA-256 mismatch",
    )


def validate_parent_bindings(contract: G0MeasurementFreezeContract) -> None:
    peer_binding = contract.foundation_parents.peer_mechanism_registry
    peer_path = _repo_relative_regular_file(peer_binding.relative_path)
    _require(
        sha256_file(peer_path) == peer_binding.required_file_sha256,
        "peer registry file SHA-256 mismatch",
    )
    peer_raw = _yaml_mapping(peer_path, field="peer registry")
    _require(
        canonical_sha256(peer_raw) == peer_binding.required_canonical_sha256,
        "peer registry canonical SHA-256 mismatch",
    )
    _validate_parent_identity(
        peer_raw,
        schema_version=peer_binding.required_schema_version,
        label=peer_binding.required_label,
        mode=peer_binding.required_mode,
        status=peer_binding.required_status,
        gate_state=PEER_GATE_STATE,
        field="peer registry",
    )

    matrix_binding = contract.foundation_parents.cross_field_matrix
    matrix_path = _repo_relative_regular_file(matrix_binding.relative_path)
    _require(
        sha256_file(matrix_path) == matrix_binding.required_file_sha256,
        "cross-field matrix file SHA-256 mismatch",
    )
    matrix_raw = _yaml_mapping(matrix_path, field="cross-field matrix")
    _require(
        canonical_sha256(matrix_raw) == matrix_binding.required_canonical_sha256,
        "cross-field matrix canonical SHA-256 mismatch",
    )
    _validate_parent_identity(
        matrix_raw,
        schema_version=matrix_binding.required_schema_version,
        label=matrix_binding.required_label,
        mode=matrix_binding.required_mode,
        status=None,
        gate_state=MATRIX_GATE_STATE,
        field="cross-field matrix",
    )

    cross_binding = contract.foundation_parents.cross_binding
    expected_peer_parent = {
        "relative_path": cross_binding.peer_declared_parent_relative_path,
        "required_schema_version": (cross_binding.peer_declared_parent_schema_version),
        "binding_status": cross_binding.peer_declared_binding_status,
    }
    _require(
        peer_raw.get("parent_contract") == expected_peer_parent == PEER_DECLARED_PARENT,
        "peer registry to cross-field matrix binding mismatch",
    )

    peer_digests = contract.inherited_subtree_digests.peer_mechanism_registry
    for key, expected in (
        ("exposure_constructs", peer_digests.exposure_constructs_sha256),
        ("entry_mode_strata", peer_digests.entry_mode_strata_sha256),
        ("negative_controls", peer_digests.negative_controls_sha256),
        ("claim_levels", peer_digests.claim_levels_sha256),
        ("claim_downgrade_rules", peer_digests.claim_downgrade_rules_sha256),
    ):
        _validate_subtree_digest(peer_raw, key, expected, field="peer registry")

    matrix_digests = contract.inherited_subtree_digests.cross_field_matrix
    for key, expected in (
        ("candidate_universe", matrix_digests.candidate_universe_sha256),
        ("selection_rules", matrix_digests.selection_rules_sha256),
        ("temporal_partitions", matrix_digests.temporal_partitions_sha256),
    ):
        _validate_subtree_digest(matrix_raw, key, expected, field="cross-field matrix")

    fixed = contract.fixed_design_choices
    fixed_partitions = [
        partition.model_dump(mode="json") for partition in fixed.temporal_partitions
    ]
    _require(
        matrix_raw.get("temporal_partitions") == fixed_partitions,
        "fixed temporal partitions do not equal the matrix parent",
    )

    controls = peer_raw.get("negative_controls")
    _require(isinstance(controls, list), "peer registry controls must be a list")
    _require(len(controls) == 7, "peer registry must retain exactly seven controls")
    _require(
        tuple(control.get("control_id") for control in controls)
        == fixed.peer_mechanism.negative_control_ids,
        "fixed negative controls do not equal the peer parent",
    )
    rules = peer_raw.get("claim_downgrade_rules")
    _require(isinstance(rules, list), "peer claim rules must be a list")
    _require(len(rules) == 10, "peer registry must retain ten claim downgrade rules")
    capacity = peer_raw.get("capacity_bridge_conditional_contrast")
    _require(isinstance(capacity, Mapping), "peer capacity contrast must be a mapping")
    _require(
        tuple(capacity.get("primary_cells", ())) == fixed.peer_mechanism.capacity_cells,
        "fixed capacity cells do not equal the peer parent",
    )
    entry_modes = peer_raw.get("entry_mode_strata")
    _require(isinstance(entry_modes, list), "peer entry modes must be a list")
    _require(
        tuple(entry.get("stratum_id") for entry in entry_modes)
        == fixed.peer_mechanism.entry_mode_ids,
        "fixed entry modes do not equal the peer parent",
    )

    rng = matrix_raw.get("rng")
    _require(isinstance(rng, Mapping), "matrix RNG contract must be a mapping")
    _require(
        rng.get("algorithm") == fixed.engineering_rng.algorithm
        and rng.get("base_seed") == fixed.engineering_rng.base_seed,
        "fixed engineering RNG does not equal the matrix parent",
    )
    _require(
        tuple(rng.get("replicate_seeds", ())) == fixed.engineering_rng.replicate_seeds,
        "fixed engineering replicate seeds do not equal the matrix parent",
    )
    slots = matrix_raw.get("engineering_slots")
    _require(isinstance(slots, list), "matrix engineering slots must be a list")
    g1_slots = [slot for slot in slots if slot.get("role") == "g1_replication"]
    g4_slots = [slot for slot in slots if slot.get("role") == "g4_domain_author_cold"]
    _require(
        len(slots) == fixed.full_panel.field_count
        and len(g1_slots) == fixed.full_panel.g1_field_count
        and len(g4_slots) == fixed.full_panel.sealed_g4_field_count,
        "fixed full-panel counts do not equal the matrix parent",
    )
    _require(
        all(
            slot.get("sealed") is True and slot.get("author_cold") is True
            for slot in g4_slots
        ),
        "matrix G4 slot is not sealed and author-cold",
    )


def load_contract(path: Path = DEFAULT_CONTRACT) -> LoadedContract:
    raw_bytes = read_contract_bytes(path)
    decoded = raw_bytes.decode("utf-8")
    validate_contract_nesting(decoded)
    try:
        raw = yaml.load(decoded, Loader=StrictContractLoader)
    except RecursionError as exc:
        raise G0ContractError(
            "G0 contract exceeds the YAML nesting safety cap"
        ) from exc
    _require(isinstance(raw, Mapping), "G0 contract must be a YAML mapping")
    contract = G0MeasurementFreezeContract.model_validate(raw)
    _require_scalar_type_fidelity(raw, contract.model_dump(mode="python"))
    validate_parent_bindings(contract)
    canonical = canonical_sha256(contract.model_dump(mode="json"))
    return LoadedContract(
        value=contract,
        file_sha256=sha256_bytes(raw_bytes),
        canonical_sha256=canonical,
        path=path,
    )


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def validation_summary(loaded: LoadedContract) -> dict[str, Any]:
    contract = loaded.value
    parents = contract.foundation_parents
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": contract.completion_contract.validation_status,
        "contract": {
            "relative_path": _path_label(loaded.path),
            "file_sha256": loaded.file_sha256,
            "canonical_sha256": loaded.canonical_sha256,
        },
        "parent_bindings": {
            "peer_mechanism_registry": {
                "relative_path": parents.peer_mechanism_registry.relative_path,
                "file_sha256": parents.peer_mechanism_registry.required_file_sha256,
                "canonical_sha256": (
                    parents.peer_mechanism_registry.required_canonical_sha256
                ),
            },
            "cross_field_matrix": {
                "relative_path": parents.cross_field_matrix.relative_path,
                "file_sha256": parents.cross_field_matrix.required_file_sha256,
                "canonical_sha256": (
                    parents.cross_field_matrix.required_canonical_sha256
                ),
            },
            "validated": True,
        },
        "gate_state": {
            "g_minus_1_data_readiness": contract.g_minus_1_binding.state,
            "holdout_access_seal": contract.holdout_access_binding.state,
            "g0_measurement_freeze": "PENDING",
            "empirical_g1_signature": contract.lifecycle.empirical_g1_signature,
            "pass_g0": contract.lifecycle.pass_g0,
            "pass_g1": contract.lifecycle.pass_g1,
            "execution_authorized": contract.lifecycle.execution_authorized,
        },
        "resolution": {
            "state": contract.resolution.state,
            "unresolved_required_sections": list(
                contract.resolution.unresolved_required_sections
            ),
            "successor_required": contract.resolution.successor_required,
        },
        "sealable": contract.completion_contract.pending_contract_sealable,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def _blocked_command_payload(command: str) -> dict[str, Any]:
    return {
        "status": BLOCKED_COMMAND_STATUS,
        "requested_command": command,
        "g0_measurement_freeze": "PENDING",
        "pass_g0": False,
        "execution_authorized": False,
        "reason": "validate and blocked pending seal are the only contract surfaces",
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "seal"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--contract",
            type=Path,
            default=DEFAULT_CONTRACT,
            help="Pending versioned G0 overlay (defaults to the repository v0).",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if resolved_argv and resolved_argv[0].lower() in BLOCKED_COMMANDS:
        print(
            json.dumps(
                _blocked_command_payload(resolved_argv[0].lower()), sort_keys=True
            ),
            file=sys.stderr,
        )
        return 2

    parser = make_parser()
    args = parser.parse_args(resolved_argv)
    try:
        loaded = load_contract(args.contract)
        if args.command == "validate":
            print(json.dumps(validation_summary(loaded), indent=2, sort_keys=True))
            return 0

        contract = loaded.value
        print(
            json.dumps(
                {
                    "status": contract.failure_policy.pending_seal_status,
                    "g0_measurement_freeze": contract.resolution.state,
                    "pass_g0": False,
                    "execution_authorized": False,
                    "successor_required": contract.resolution.successor_required,
                    "reason": (
                        "pending quantitative sections and external receipts require "
                        "a versioned successor"
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except (
        G0ContractError,
        OSError,
        UnicodeError,
        yaml.YAMLError,
        ValidationError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": INVALID_STATUS,
                    "g0_measurement_freeze": "PENDING",
                    "pass_g0": False,
                    "execution_authorized": False,
                    "reason": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
