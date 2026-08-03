#!/usr/bin/env python
"""Validate and plan the engineering-only researcher-pivot field matrix.

This command intentionally has no execution surface.  It freezes a deterministic
cross-field candidate universe and expands the prospective G3 factorial while
G-1, G0, and empirical G1 remain incomplete.  Scientific field selection and
all model fitting stay blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA_VERSION = "pivot_feedback_cross_field_matrix.v1"
PLAN_SCHEMA_VERSION = "pivot_feedback_cross_field_matrix_plan.v1"
BLOCKED_COMMANDS = frozenset({"run", "execute"})
SELECTION_PENDING = "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"

EXPECTED_FIELDS: dict[int, tuple[str, str]] = {
    11: ("agricultural_and_biological_sciences", "Agricultural and Biological Sciences"),
    12: ("arts_and_humanities", "Arts and Humanities"),
    13: (
        "biochemistry_genetics_and_molecular_biology",
        "Biochemistry, Genetics and Molecular Biology",
    ),
    14: (
        "business_management_and_accounting",
        "Business, Management and Accounting",
    ),
    15: ("chemical_engineering", "Chemical Engineering"),
    16: ("chemistry", "Chemistry"),
    17: ("computer_science", "Computer Science"),
    18: ("decision_sciences", "Decision Sciences"),
    19: ("earth_and_planetary_sciences", "Earth and Planetary Sciences"),
    20: (
        "economics_econometrics_and_finance",
        "Economics, Econometrics and Finance",
    ),
    21: ("energy", "Energy"),
    22: ("engineering", "Engineering"),
    23: ("environmental_science", "Environmental Science"),
    24: ("immunology_and_microbiology", "Immunology and Microbiology"),
    25: ("materials_science", "Materials Science"),
    26: ("mathematics", "Mathematics"),
    27: ("medicine", "Medicine"),
    28: ("neuroscience", "Neuroscience"),
    29: ("nursing", "Nursing"),
    30: (
        "pharmacology_toxicology_and_pharmaceutics",
        "Pharmacology, Toxicology and Pharmaceutics",
    ),
    31: ("physics_and_astronomy", "Physics and Astronomy"),
    32: ("psychology", "Psychology"),
    33: ("social_sciences", "Social Sciences"),
    34: ("veterinary", "Veterinary"),
    35: ("dentistry", "Dentistry"),
    36: ("health_professions", "Health Professions"),
}

EXPECTED_DOMAIN_FIELDS: dict[str, frozenset[int]] = {
    "life_sciences": frozenset({11, 13, 24, 28, 30}),
    "social_sciences": frozenset({12, 14, 18, 20, 32, 33}),
    "physical_sciences": frozenset({15, 16, 17, 19, 21, 22, 23, 25, 26, 31}),
    "health_sciences": frozenset({27, 29, 34, 35, 36}),
}

EXPECTED_FACTOR_LEVELS: dict[str, frozenset[str]] = {
    "outcome_disclosure": frozenset(
        {"full", "success_biased", "volume_matched_random"}
    ),
    "revelation_delay": frozenset(
        {"immediate", "empirical_central", "long_delay_stress"}
    ),
    "peer_signal_diffusion": frozenset(
        {"slow", "empirical_central", "fast"}
    ),
    "environment": frozenset(
        {"attention_only", "structural_support_matched_direct_pressure"}
    ),
    "capacity_bridge": frozenset({"absent", "pre_existing"}),
}

EXPECTED_LEARNERS: dict[str, str] = {
    "b2_fixed_rule_abm": "B2",
    "b3_explicit_adaptive_baseline": "B3",
    "b4_adaptive_nabm": "B4",
}

EXPECTED_PARTITIONS: dict[str, tuple[int | None, int]] = {
    "discovery_calibration": (None, 2012),
    "g1_empirical_replication": (2013, 2016),
    "model_selection": (2017, 2019),
    "g4_temporal_holdout": (2020, 2022),
}


class PlannerError(RuntimeError):
    """Raised when a manifest or requested planner action fails closed."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GateState(StrictModel):
    g_minus_1_data_readiness: Literal["PENDING"]
    g0_measurement_freeze: Literal["PENDING"]
    empirical_g1_signature: Literal["NOT_RUN"]
    pass_g1: Literal[False]
    execution_authorized: Literal[False]


class RequiredAuditBoundary(StrictModel):
    raw_source_id_validity: Literal["BLOCKED"]
    w_zero_entity_validity: Literal["UNRESOLVED"]
    author_entity_validity_established: Literal[False]
    parent_content_identity: Literal["UNVERIFIED"]
    analysis_view_materialized: Literal[False]
    g_minus_1_data_readiness: Literal["PENDING"]
    empirical_g1_signature: Literal["NOT_RUN"]
    pass_g1: Literal[False]


class SourceAuditBinding(StrictModel):
    summary_relative_path: str
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_schema_version: str
    required_status: str
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_gate_status: RequiredAuditBoundary


class RequiredTaxonomyReferenceBoundary(StrictModel):
    current_snapshot_taxonomy_reference_component: Literal["PASS"]
    raw_source_id_validity: Literal["BLOCKED"]
    w_zero_entity_validity: Literal["UNRESOLVED"]
    parent_content_identity: Literal["UNVERIFIED"]
    duplicate_work_topic_edges: Literal["PENDING"]
    primary_topic_child_inclusion: Literal["PENDING"]
    publication_date_validity: Literal["PENDING"]
    temporal_available_at: Literal["PENDING"]
    author_entity_validity: Literal["PENDING"]
    candidate_eligibility: Literal["PENDING"]
    g_minus_1_data_readiness: Literal["PENDING"]
    g0_measurement_freeze: Literal["PENDING"]
    empirical_g1_signature: Literal["NOT_RUN"]
    pass_g1: Literal[False]


class TaxonomyReferenceAuditBinding(StrictModel):
    summary_relative_path: Literal[
        "experiments/evidence/results/"
        "openalex_g1_field_taxonomy_reference_20260718.summary.json"
    ]
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_schema_version: Literal[
        "openalex_g1_field_taxonomy_reference_report.v1"
    ]
    required_status: Literal[
        "PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT"
    ]
    result_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_full_scope: Literal[True]
    required_scientific_effect_estimation_performed: Literal[False]
    required_field_status_margin: Literal["PASS"]
    required_gate_status: RequiredTaxonomyReferenceBoundary


class FieldCandidate(StrictModel):
    field_id: StrictInt = Field(ge=11, le=36)
    field_slug: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    rng_code: StrictInt = Field(gt=0)
    scientific_selection_status: Literal[
        "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"
    ]


class DomainCandidate(StrictModel):
    domain_id: Literal[
        "life_sciences",
        "social_sciences",
        "physical_sciences",
        "health_sciences",
    ]
    display_name: str = Field(min_length=1)
    rng_code: StrictInt = Field(gt=0)
    fields: tuple[FieldCandidate, ...]


class CandidateUniverse(StrictModel):
    taxonomy: Literal["openalex_topic_hierarchy_4_domains_26_fields"]
    taxonomy_reference_url: str = Field(min_length=1)
    scientific_field_selection: Literal[
        "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"
    ]
    domains: tuple[DomainCandidate, ...]


class SelectionRules(StrictModel):
    status: Literal["PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"]
    primary_rule: Literal["LEXICOGRAPHIC_MAXIMIN_STRUCTURAL_COVERAGE"]
    eligible_structural_metrics: tuple[str, ...]
    forbidden_outcome_metrics: tuple[str, ...]
    tie_breakers: tuple[str, ...]
    reserves_per_domain: StrictInt = Field(ge=1)
    reserve_rule: Literal["NEXT_RANKED_BY_IDENTICAL_FROZEN_RULE"]
    domain_assignment_rule: Literal[
        "TOP_THREE_STRUCTURAL_DOMAINS_TO_G1_REMAINDER_TO_SEALED_G4"
    ]
    one_slot_per_domain_without_replacement: StrictBool
    g4_domain_disjoint_from_g1: StrictBool
    g4_author_cold: StrictBool


class TemporalPartition(StrictModel):
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

    @model_validator(mode="after")
    def validate_years(self) -> "TemporalPartition":
        if (
            self.entry_start_year is not None
            and self.entry_start_year > self.entry_end_year
        ):
            raise ValueError("entry_start_year must be <= entry_end_year")
        if self.outcome_followup_through_year < self.entry_end_year:
            raise ValueError(
                "outcome_followup_through_year must be >= entry_end_year"
            )
        return self


class EngineeringSlot(StrictModel):
    slot_id: str = Field(min_length=1)
    rng_code: StrictInt = Field(gt=0)
    role: Literal["g1_replication", "g4_domain_author_cold"]
    temporal_partition_id: Literal[
        "g1_empirical_replication", "g4_temporal_holdout"
    ]
    scientific_selection_status: Literal[
        "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"
    ]
    selected_domain_id: None = None
    selected_field_id: None = None
    sealed: StrictBool
    author_cold: StrictBool


class FactorLevel(StrictModel):
    level_id: str = Field(min_length=1)
    rng_code: StrictInt = Field(gt=0)


class LearnerLevel(FactorLevel):
    baseline_level: Literal["B2", "B3", "B4"]
    default_compute: Literal["cpu", "gpu_after_promotion_gate"]


class CoreFactorial(StrictModel):
    outcome_disclosure: tuple[FactorLevel, ...]
    revelation_delay: tuple[FactorLevel, ...]
    peer_signal_diffusion: tuple[FactorLevel, ...]
    learner: tuple[LearnerLevel, ...]
    environment: tuple[FactorLevel, ...]
    capacity_bridge: tuple[FactorLevel, ...]


class RNGComponent(StrictModel):
    component_id: str = Field(min_length=1)
    rng_code: StrictInt = Field(gt=0)
    pairing_scope: Literal[
        "slot_seed",
        "slot_seed_disclosure",
        "slot_seed_delay",
        "slot_seed_diffusion",
        "slot_seed_environment_bridge",
        "slot_seed_learner",
    ]


class RNGContract(StrictModel):
    algorithm: Literal["numpy.random.SeedSequence"]
    base_seed: StrictInt = Field(ge=0)
    replicate_seeds: tuple[StrictInt, ...]
    components: tuple[RNGComponent, ...]


class FieldMatrixManifest(StrictModel):
    schema_version: Literal["pivot_feedback_cross_field_matrix.v1"]
    label: str = Field(min_length=1)
    mode: Literal["ENGINEERING_ONLY_PLAN"]
    gate_state: GateState
    source_audit: SourceAuditBinding
    taxonomy_reference_audit: TaxonomyReferenceAuditBinding
    candidate_universe: CandidateUniverse
    selection_rules: SelectionRules
    temporal_partitions: tuple[TemporalPartition, ...]
    engineering_slots: tuple[EngineeringSlot, ...]
    core_factorial: CoreFactorial
    rng: RNGContract

    @model_validator(mode="after")
    def validate_contract(self) -> "FieldMatrixManifest":
        _validate_candidate_universe(self.candidate_universe)
        _validate_selection_rules(self.selection_rules)
        _validate_temporal_partitions(self.temporal_partitions)
        _validate_engineering_slots(self.engineering_slots)
        _validate_core_factorial(self.core_factorial)
        _validate_rng_contract(self.rng)
        return self


def _validate_candidate_universe(universe: CandidateUniverse) -> None:
    if len(universe.domains) != 4:
        raise ValueError("candidate universe must contain exactly 4 domains")
    domain_ids = [domain.domain_id for domain in universe.domains]
    domain_codes = [domain.rng_code for domain in universe.domains]
    if len(set(domain_ids)) != len(domain_ids):
        raise ValueError("candidate domain IDs must be unique")
    if len(set(domain_codes)) != len(domain_codes):
        raise ValueError("candidate domain rng_codes must be unique")
    if set(domain_ids) != set(EXPECTED_DOMAIN_FIELDS):
        raise ValueError("candidate universe domain IDs are not the frozen four")

    observed_fields: dict[int, tuple[str, str, str, int]] = {}
    field_codes: list[int] = []
    for domain in universe.domains:
        ids = {field.field_id for field in domain.fields}
        if ids != EXPECTED_DOMAIN_FIELDS[domain.domain_id]:
            raise ValueError(
                f"field membership differs for domain {domain.domain_id}"
            )
        for field in domain.fields:
            if field.field_id in observed_fields:
                raise ValueError(f"duplicate candidate field ID: {field.field_id}")
            observed_fields[field.field_id] = (
                field.field_slug,
                field.display_name,
                domain.domain_id,
                field.rng_code,
            )
            field_codes.append(field.rng_code)
    if set(observed_fields) != set(EXPECTED_FIELDS):
        raise ValueError("candidate universe must contain field IDs 11 through 36")
    if len(set(field_codes)) != len(field_codes):
        raise ValueError("candidate field rng_codes must be unique")
    for field_id, (expected_slug, expected_name) in EXPECTED_FIELDS.items():
        slug, name, _domain, _rng_code = observed_fields[field_id]
        if (slug, name) != (expected_slug, expected_name):
            raise ValueError(f"frozen field metadata differs for field {field_id}")


def _validate_selection_rules(rules: SelectionRules) -> None:
    if not rules.eligible_structural_metrics:
        raise ValueError("selection requires structural metrics")
    if not rules.forbidden_outcome_metrics:
        raise ValueError("selection requires forbidden outcome metrics")
    if not rules.tie_breakers:
        raise ValueError("selection requires deterministic tie breakers")
    if len(set(rules.eligible_structural_metrics)) != len(
        rules.eligible_structural_metrics
    ):
        raise ValueError("selection structural metrics must be unique")
    if len(set(rules.tie_breakers)) != len(rules.tie_breakers):
        raise ValueError("selection tie breakers must be unique")
    if not rules.one_slot_per_domain_without_replacement:
        raise ValueError("engineering slots must use domains without replacement")
    if not rules.g4_domain_disjoint_from_g1:
        raise ValueError("G4 domain must remain disjoint from G1")
    if not rules.g4_author_cold:
        raise ValueError("G4 domain slot must remain author-cold")


def _validate_temporal_partitions(
    partitions: Sequence[TemporalPartition],
) -> None:
    if len(partitions) != 4:
        raise ValueError("exactly four temporal partitions are required")
    by_id = {partition.partition_id: partition for partition in partitions}
    if len(by_id) != len(partitions) or set(by_id) != set(EXPECTED_PARTITIONS):
        raise ValueError("temporal partition IDs must be unique and complete")
    for partition_id, expected_range in EXPECTED_PARTITIONS.items():
        partition = by_id[partition_id]
        observed = (partition.entry_start_year, partition.entry_end_year)
        if observed != expected_range:
            raise ValueError(
                f"temporal range differs for {partition_id}: {observed}"
            )
    if by_id["g4_temporal_holdout"].outcome_followup_through_year != 2025:
        raise ValueError("G4 outcome follow-up must remain frozen through 2025")
    if not by_id["g4_temporal_holdout"].sealed:
        raise ValueError("G4 temporal holdout must remain sealed")
    if any(
        partition.sealed
        for key, partition in by_id.items()
        if key != "g4_temporal_holdout"
    ):
        raise ValueError("only the G4 temporal partition may be sealed")

    ranges = sorted(
        (
            -sys.maxsize
            if partition.entry_start_year is None
            else partition.entry_start_year,
            partition.entry_end_year,
            partition.partition_id,
        )
        for partition in partitions
    )
    for left, right in itertools.pairwise(ranges):
        if left[1] >= right[0]:
            raise ValueError(
                f"temporal partitions overlap: {left[2]} and {right[2]}"
            )


def _validate_engineering_slots(slots: Sequence[EngineeringSlot]) -> None:
    if len(slots) != 4:
        raise ValueError("exactly four engineering slots are required")
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise ValueError("engineering slot IDs must be unique")
    if len({slot.rng_code for slot in slots}) != len(slots):
        raise ValueError("engineering slot rng_codes must be unique")
    if any(
        slot.selected_domain_id is not None or slot.selected_field_id is not None
        for slot in slots
    ):
        raise ValueError("scientific domain and field selection must remain pending")
    g1_slots = [slot for slot in slots if slot.role == "g1_replication"]
    g4_slots = [slot for slot in slots if slot.role == "g4_domain_author_cold"]
    if len(g1_slots) != 3 or len(g4_slots) != 1:
        raise ValueError("engineering plan requires three G1 slots and one G4 slot")
    if any(
        slot.temporal_partition_id != "g1_empirical_replication"
        or slot.sealed
        or slot.author_cold
        for slot in g1_slots
    ):
        raise ValueError("G1 slots must be open, non-author-cold G1 placeholders")
    g4 = g4_slots[0]
    if (
        g4.temporal_partition_id != "g4_temporal_holdout"
        or not g4.sealed
        or not g4.author_cold
    ):
        raise ValueError("G4 slot must be sealed and domain/author-cold")


def _level_ids(levels: Sequence[FactorLevel]) -> set[str]:
    return {level.level_id for level in levels}


def _validate_core_factorial(factorial: CoreFactorial) -> None:
    factor_sequences: dict[str, Sequence[FactorLevel]] = {
        "outcome_disclosure": factorial.outcome_disclosure,
        "revelation_delay": factorial.revelation_delay,
        "peer_signal_diffusion": factorial.peer_signal_diffusion,
        "environment": factorial.environment,
        "capacity_bridge": factorial.capacity_bridge,
    }
    all_rng_codes: list[int] = []
    for factor_name, levels in factor_sequences.items():
        if _level_ids(levels) != set(EXPECTED_FACTOR_LEVELS[factor_name]):
            raise ValueError(f"factor levels differ for {factor_name}")
        if len(levels) != len(EXPECTED_FACTOR_LEVELS[factor_name]):
            raise ValueError(f"factor levels must be unique for {factor_name}")
        all_rng_codes.extend(level.rng_code for level in levels)
    learner_by_id = {level.level_id: level for level in factorial.learner}
    if set(learner_by_id) != set(EXPECTED_LEARNERS) or len(learner_by_id) != 3:
        raise ValueError("learner factor must contain exactly B2, B3, and B4")
    for learner_id, baseline_level in EXPECTED_LEARNERS.items():
        if learner_by_id[learner_id].baseline_level != baseline_level:
            raise ValueError(f"baseline ladder level differs for {learner_id}")
    if learner_by_id["b4_adaptive_nabm"].default_compute != "gpu_after_promotion_gate":
        raise ValueError("B4 must remain behind the GPU promotion gate")
    if any(
        learner.default_compute != "cpu"
        for learner_id, learner in learner_by_id.items()
        if learner_id != "b4_adaptive_nabm"
    ):
        raise ValueError("B2 and B3 must default to CPU")
    all_rng_codes.extend(level.rng_code for level in factorial.learner)
    if len(set(all_rng_codes)) != len(all_rng_codes):
        raise ValueError("factor-level rng_codes must be globally unique")
    counts = [len(levels) for levels in factor_sequences.values()]
    cell_count = int(np.prod([*counts, len(factorial.learner)]))
    if cell_count != 324:
        raise ValueError("core factorial must contain exactly 324 cells")


def _validate_rng_contract(contract: RNGContract) -> None:
    if not contract.replicate_seeds:
        raise ValueError("at least one replicate seed is required")
    if len(set(contract.replicate_seeds)) != len(contract.replicate_seeds):
        raise ValueError("replicate seeds must be unique")
    if any(seed < 0 for seed in contract.replicate_seeds):
        raise ValueError("replicate seeds must be non-negative")
    if not contract.components:
        raise ValueError("component-keyed RNG streams are required")
    if len({component.component_id for component in contract.components}) != len(
        contract.components
    ):
        raise ValueError("RNG component IDs must be unique")
    if len({component.rng_code for component in contract.components}) != len(
        contract.components
    ):
        raise ValueError("RNG component rng_codes must be unique")
    required_components = {
        "population",
        "base_network",
        "local_noise",
        "latent_outcome_quantile",
        "outcome_disclosure",
        "revelation_delay",
        "peer_signal_diffusion",
        "environment_mapping",
        "learner_initialization",
        "learner_update",
    }
    if {component.component_id for component in contract.components} != required_components:
        raise ValueError("RNG component contract is incomplete")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_file(relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PlannerError("source audit path must be repository-relative")
    resolved = (REPO_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise PlannerError("source audit path escapes the repository") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise PlannerError("source audit binding must target a regular file")
    return resolved


def load_manifest(path: Path) -> FieldMatrixManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PlannerError("field-matrix manifest must be a YAML mapping")
    manifest = FieldMatrixManifest.model_validate(raw)
    validate_input_bindings(manifest)
    return manifest


def validate_source_audit(binding: SourceAuditBinding) -> dict[str, Any]:
    path = _repo_relative_file(binding.summary_relative_path)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != binding.summary_sha256:
        raise PlannerError("compact work-source summary SHA-256 mismatch")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != binding.summary_schema_version:
        raise PlannerError("compact work-source summary schema version mismatch")
    if raw.get("status") != binding.required_status:
        raise PlannerError("compact work-source summary status mismatch")
    fingerprint = raw.get("provenance", {}).get("result_fingerprint_sha256")
    if fingerprint != binding.result_fingerprint_sha256:
        raise PlannerError("compact work-source result fingerprint mismatch")
    gate_status = raw.get("gate_status")
    if not isinstance(gate_status, Mapping):
        raise PlannerError("compact work-source gate status is missing")
    expected_gate_status = binding.required_gate_status.model_dump(mode="json")
    for key, expected in expected_gate_status.items():
        if gate_status.get(key) != expected:
            raise PlannerError(f"compact work-source gate boundary mismatch: {key}")
    return raw


def validate_taxonomy_reference_payload(
    raw: Mapping[str, Any],
    binding: TaxonomyReferenceAuditBinding,
) -> dict[str, Any]:
    if raw.get("schema_version") != binding.summary_schema_version:
        raise PlannerError("taxonomy-reference summary schema version mismatch")
    if raw.get("status") != binding.required_status:
        raise PlannerError("taxonomy-reference summary status mismatch")

    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PlannerError("taxonomy-reference provenance is missing")
    if (
        provenance.get("result_fingerprint_sha256")
        != binding.result_fingerprint_sha256
    ):
        raise PlannerError("taxonomy-reference result fingerprint mismatch")
    if provenance.get("path_redacted") is not True:
        raise PlannerError("taxonomy-reference path redaction is not established")

    taxonomy = raw.get("taxonomy")
    if not isinstance(taxonomy, Mapping):
        raise PlannerError("taxonomy-reference taxonomy commitment is missing")
    if taxonomy.get("canonical_sha256") != binding.taxonomy_canonical_sha256:
        raise PlannerError("taxonomy-reference canonical taxonomy SHA-256 mismatch")

    scope = raw.get("scope")
    if not isinstance(scope, Mapping):
        raise PlannerError("taxonomy-reference scope is missing")
    if scope.get("full_scope") is not binding.required_full_scope:
        raise PlannerError("taxonomy-reference full-scope requirement mismatch")
    if (
        scope.get("scientific_effect_estimation_performed")
        is not binding.required_scientific_effect_estimation_performed
    ):
        raise PlannerError("taxonomy-reference scientific-effect boundary mismatch")

    field_margin = raw.get("field_status_margin_comparison")
    if not isinstance(field_margin, Mapping):
        raise PlannerError("taxonomy-reference field-margin comparison is missing")
    if field_margin.get("status") != binding.required_field_status_margin:
        raise PlannerError("taxonomy-reference field-margin status mismatch")

    gate_status = raw.get("gate_status")
    if not isinstance(gate_status, Mapping):
        raise PlannerError("taxonomy-reference gate status is missing")
    expected_gate_status = binding.required_gate_status.model_dump(mode="json")
    if set(gate_status) != set(expected_gate_status):
        raise PlannerError("taxonomy-reference gate boundary keys mismatch")
    for key, expected in expected_gate_status.items():
        if gate_status.get(key) != expected:
            raise PlannerError(f"taxonomy-reference gate boundary mismatch: {key}")
    return dict(raw)


def validate_taxonomy_reference_audit(
    binding: TaxonomyReferenceAuditBinding,
) -> dict[str, Any]:
    path = _repo_relative_file(binding.summary_relative_path)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != binding.summary_sha256:
        raise PlannerError("taxonomy-reference summary SHA-256 mismatch")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise PlannerError("taxonomy-reference summary must be a JSON mapping")
    return validate_taxonomy_reference_payload(raw, binding)


def validate_input_bindings(manifest: FieldMatrixManifest) -> None:
    validate_source_audit(manifest.source_audit)
    validate_taxonomy_reference_audit(manifest.taxonomy_reference_audit)


def canonical_manifest_payload(manifest: FieldMatrixManifest) -> dict[str, Any]:
    return manifest.model_dump(mode="json")


def manifest_sha256(manifest: FieldMatrixManifest) -> str:
    return sha256_bytes(canonical_json_bytes(canonical_manifest_payload(manifest)))


def _sorted_levels(levels: Sequence[FactorLevel]) -> list[FactorLevel]:
    return sorted(levels, key=lambda level: (level.rng_code, level.level_id))


def _seed_entropy(
    *,
    manifest: FieldMatrixManifest,
    slot: EngineeringSlot,
    replicate_seed: int,
    component: RNGComponent,
    factors: Mapping[str, FactorLevel],
) -> list[int]:
    entropy = [
        manifest.rng.base_seed,
        slot.rng_code,
        replicate_seed,
        component.rng_code,
    ]
    if component.pairing_scope == "slot_seed":
        return entropy
    if component.pairing_scope == "slot_seed_disclosure":
        return [*entropy, factors["outcome_disclosure"].rng_code]
    if component.pairing_scope == "slot_seed_delay":
        return [*entropy, factors["revelation_delay"].rng_code]
    if component.pairing_scope == "slot_seed_diffusion":
        return [*entropy, factors["peer_signal_diffusion"].rng_code]
    if component.pairing_scope == "slot_seed_environment_bridge":
        return [
            *entropy,
            factors["environment"].rng_code,
            factors["capacity_bridge"].rng_code,
        ]
    if component.pairing_scope == "slot_seed_learner":
        return [*entropy, factors["learner"].rng_code]
    raise PlannerError(f"unsupported RNG pairing scope: {component.pairing_scope}")


def _seed_stream_entry(component: RNGComponent, entropy: list[int]) -> dict[str, Any]:
    identity = {
        "algorithm": "numpy.random.SeedSequence",
        "component_id": component.component_id,
        "component_rng_code": component.rng_code,
        "entropy": entropy,
        "pairing_scope": component.pairing_scope,
    }
    stream_id = sha256_bytes(canonical_json_bytes(identity))
    state = np.random.SeedSequence(entropy).generate_state(4, dtype=np.uint32)
    return {
        "stream_id": stream_id,
        **identity,
        "state_uint32": [int(value) for value in state],
    }


def build_plan(manifest: FieldMatrixManifest) -> dict[str, Any]:
    validate_input_bindings(manifest)
    if manifest.gate_state.execution_authorized:
        raise PlannerError("engineering-only manifest cannot authorize execution")
    resolved_manifest_sha256 = manifest_sha256(manifest)
    slots = sorted(
        manifest.engineering_slots,
        key=lambda slot: (slot.rng_code, slot.slot_id),
    )
    seeds = sorted(manifest.rng.replicate_seeds)
    factor_axes: tuple[tuple[str, Sequence[FactorLevel]], ...] = (
        ("outcome_disclosure", _sorted_levels(manifest.core_factorial.outcome_disclosure)),
        ("revelation_delay", _sorted_levels(manifest.core_factorial.revelation_delay)),
        (
            "peer_signal_diffusion",
            _sorted_levels(manifest.core_factorial.peer_signal_diffusion),
        ),
        ("learner", _sorted_levels(manifest.core_factorial.learner)),
        ("environment", _sorted_levels(manifest.core_factorial.environment)),
        ("capacity_bridge", _sorted_levels(manifest.core_factorial.capacity_bridge)),
    )
    components = sorted(
        manifest.rng.components,
        key=lambda component: (component.rng_code, component.component_id),
    )

    stream_entries: dict[str, dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    learner_counts = {"B2": 0, "B3": 0, "B4": 0}
    for slot in slots:
        for replicate_seed in seeds:
            for combination in itertools.product(
                *(levels for _factor_name, levels in factor_axes)
            ):
                factors = {
                    factor_name: level
                    for (factor_name, _levels), level in zip(
                        factor_axes, combination, strict=True
                    )
                }
                learner = factors["learner"]
                if not isinstance(learner, LearnerLevel):
                    raise PlannerError("learner factor did not resolve to LearnerLevel")
                identity = {
                    "manifest_sha256": resolved_manifest_sha256,
                    "slot_id": slot.slot_id,
                    "replicate_seed": replicate_seed,
                    "factors": {
                        factor_name: factors[factor_name].level_id
                        for factor_name, _levels in factor_axes
                    },
                }
                cell_id = sha256_bytes(canonical_json_bytes(identity))
                stream_ids: dict[str, str] = {}
                for component in components:
                    entropy = _seed_entropy(
                        manifest=manifest,
                        slot=slot,
                        replicate_seed=replicate_seed,
                        component=component,
                        factors=factors,
                    )
                    stream = _seed_stream_entry(component, entropy)
                    stream_entries.setdefault(stream["stream_id"], stream)
                    stream_ids[component.component_id] = stream["stream_id"]
                cells.append(
                    {
                        "cell_id": cell_id,
                        **identity,
                        "baseline_level": learner.baseline_level,
                        "default_compute": learner.default_compute,
                        "seed_stream_ids": stream_ids,
                        "scientific_selection_status": SELECTION_PENDING,
                        "selected_domain_id": None,
                        "selected_field_id": None,
                        "execution_authorized": False,
                        "cell_status": (
                            "SEALED_PLAN_ONLY"
                            if slot.sealed
                            else "PLAN_ONLY_PENDING_GATES"
                        ),
                    }
                )
                learner_counts[learner.baseline_level] += 1

    cells_per_slot_seed = len(cells) // (len(slots) * len(seeds))
    b4_per_slot_seed = learner_counts["B4"] // (len(slots) * len(seeds))
    non_b4_per_slot_seed = (
        learner_counts["B2"] + learner_counts["B3"]
    ) // (len(slots) * len(seeds))
    if (cells_per_slot_seed, b4_per_slot_seed, non_b4_per_slot_seed) != (
        324,
        108,
        216,
    ):
        raise PlannerError("expanded factorial count differs from 324/108/216")

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "label": manifest.label,
        "mode": manifest.mode,
        "manifest_sha256": resolved_manifest_sha256,
        "source_audit": manifest.source_audit.model_dump(mode="json"),
        "taxonomy_reference_audit": (
            manifest.taxonomy_reference_audit.model_dump(mode="json")
        ),
        "gate_state": manifest.gate_state.model_dump(mode="json"),
        "scientific_field_selection": SELECTION_PENDING,
        "execution_authorized": False,
        "candidate_universe": {
            "domain_count": len(manifest.candidate_universe.domains),
            "field_count": sum(
                len(domain.fields) for domain in manifest.candidate_universe.domains
            ),
            "domain_ids": sorted(
                domain.domain_id for domain in manifest.candidate_universe.domains
            ),
            "field_ids": sorted(
                field.field_id
                for domain in manifest.candidate_universe.domains
                for field in domain.fields
            ),
        },
        "selection_rules": manifest.selection_rules.model_dump(mode="json"),
        "temporal_partitions": [
            partition.model_dump(mode="json")
            for partition in sorted(
                manifest.temporal_partitions,
                key=lambda partition: (
                    -sys.maxsize
                    if partition.entry_start_year is None
                    else partition.entry_start_year,
                    partition.partition_id,
                ),
            )
        ],
        "engineering_slots": [
            slot.model_dump(mode="json") for slot in slots
        ],
        "counts": {
            "slot_count": len(slots),
            "replicate_seed_count": len(seeds),
            "cells_per_slot_seed": cells_per_slot_seed,
            "b4_cells_per_slot_seed": b4_per_slot_seed,
            "b2_b3_cells_per_slot_seed": non_b4_per_slot_seed,
            "total_cells": len(cells),
            "total_b2_cells": learner_counts["B2"],
            "total_b3_cells": learner_counts["B3"],
            "total_b4_cells": learner_counts["B4"],
        },
        "seed_sequence_ledger": {
            "algorithm": manifest.rng.algorithm,
            "base_seed": manifest.rng.base_seed,
            "replicate_seeds": seeds,
            "streams": [stream_entries[key] for key in sorted(stream_entries)],
        },
        "cells": cells,
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json_bytes(plan))
    return plan


def validation_summary(
    manifest: FieldMatrixManifest,
    manifest_path: Path,
) -> dict[str, Any]:
    validate_input_bindings(manifest)
    return {
        "status": "VALID_ENGINEERING_ONLY_PLAN",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256(manifest),
        "source_summary_sha256": manifest.source_audit.summary_sha256,
        "source_result_fingerprint_sha256": (
            manifest.source_audit.result_fingerprint_sha256
        ),
        "taxonomy_reference_audit": (
            manifest.taxonomy_reference_audit.model_dump(mode="json")
        ),
        "taxonomy_reference_summary_sha256": (
            manifest.taxonomy_reference_audit.summary_sha256
        ),
        "taxonomy_reference_result_fingerprint_sha256": (
            manifest.taxonomy_reference_audit.result_fingerprint_sha256
        ),
        "taxonomy_reference_status": (
            manifest.taxonomy_reference_audit.required_status
        ),
        "taxonomy_reference_component": (
            manifest.taxonomy_reference_audit.required_gate_status
            .current_snapshot_taxonomy_reference_component
        ),
        "candidate_domain_count": len(manifest.candidate_universe.domains),
        "candidate_field_count": sum(
            len(domain.fields) for domain in manifest.candidate_universe.domains
        ),
        "engineering_slot_count": len(manifest.engineering_slots),
        "scientific_field_selection": SELECTION_PENDING,
        "gate_state": manifest.gate_state.model_dump(mode="json"),
        "execution_authorized": False,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "plan"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--manifest",
            type=Path,
            required=True,
            help="Tracked cross-field engineering manifest.",
        )
        if command == "plan":
            command_parser.add_argument(
                "--output",
                type=Path,
                required=True,
                help="Explicit destination for the generated plan JSON.",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if resolved_argv and resolved_argv[0].lower() in BLOCKED_COMMANDS:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_ENGINEERING_ONLY",
                    "requested_command": resolved_argv[0],
                    "execution_authorized": False,
                    "g_minus_1_data_readiness": "PENDING",
                    "g0_measurement_freeze": "PENDING",
                    "empirical_g1_signature": "NOT_RUN",
                    "pass_g1": False,
                    "reason": "validate and plan are the only authorized commands",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    parser = make_parser()
    args = parser.parse_args(resolved_argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            print(
                json.dumps(
                    validation_summary(manifest, args.manifest),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "plan":
            plan = build_plan(manifest)
            _write_json(args.output, plan)
            print(
                json.dumps(
                    {
                        "status": "PLAN_WRITTEN_ENGINEERING_ONLY",
                        "output": str(args.output),
                        "plan_sha256": plan["plan_sha256"],
                        "total_cells": plan["counts"]["total_cells"],
                        "execution_authorized": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    except (OSError, ValueError, PlannerError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_OR_BLOCKED",
                    "execution_authorized": False,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    raise PlannerError(f"unsupported planner command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
