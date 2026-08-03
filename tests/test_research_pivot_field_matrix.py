from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import plan_research_pivot_field_matrix as planner  # noqa: E402


MANIFEST_PATH = (
    ROOT / "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml"
)
SUMMARY_PATH = (
    ROOT
    / "experiments/evidence/results/"
    "openalex_g1_work_source_membership_20260717.summary.json"
)
TAXONOMY_REFERENCE_SUMMARY_PATH = (
    ROOT
    / "experiments/evidence/results/"
    "openalex_g1_field_taxonomy_reference_20260718.summary.json"
)


def _raw_manifest() -> dict[str, Any]:
    raw = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_manifest(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _matching_cell(
    plan: dict[str, Any],
    *,
    learner: str,
    disclosure: str = "full",
    environment: str = "attention_only",
) -> dict[str, Any]:
    matches = [
        cell
        for cell in plan["cells"]
        if cell["slot_id"] == "g1_domain_slot_01"
        and cell["replicate_seed"] == 1
        and cell["factors"]
        == {
            "outcome_disclosure": disclosure,
            "revelation_delay": "immediate",
            "peer_signal_diffusion": "slow",
            "learner": learner,
            "environment": environment,
            "capacity_bridge": "absent",
        }
    ]
    assert len(matches) == 1
    return matches[0]


def test_manifest_binds_current_audits_and_preserves_closed_gates() -> None:
    manifest = planner.load_manifest(MANIFEST_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    taxonomy_reference = _json(TAXONOMY_REFERENCE_SUMMARY_PATH)

    assert planner.sha256_file(SUMMARY_PATH) == (
        "df626b897bcb53c1bbc3209dbfbec8f69b84361367e09e1a9ffdf39e293d7b83"
    )
    assert manifest.source_audit.summary_sha256 == planner.sha256_file(SUMMARY_PATH)
    assert manifest.source_audit.required_status == summary["status"]
    assert manifest.source_audit.result_fingerprint_sha256 == (
        summary["provenance"]["result_fingerprint_sha256"]
    )
    assert planner.sha256_file(TAXONOMY_REFERENCE_SUMMARY_PATH) == (
        "f8a0d01ee5b04133c1472d4ed7c07f7a578343004c859645f174f1a0f6021f62"
    )
    reference_binding = manifest.taxonomy_reference_audit
    assert reference_binding.summary_sha256 == planner.sha256_file(
        TAXONOMY_REFERENCE_SUMMARY_PATH
    )
    assert reference_binding.summary_schema_version == taxonomy_reference[
        "schema_version"
    ]
    assert reference_binding.required_status == taxonomy_reference["status"]
    assert reference_binding.result_fingerprint_sha256 == (
        taxonomy_reference["provenance"]["result_fingerprint_sha256"]
    )
    assert reference_binding.taxonomy_canonical_sha256 == (
        taxonomy_reference["taxonomy"]["canonical_sha256"]
    )
    assert taxonomy_reference["scope"]["full_scope"] is True
    assert (
        taxonomy_reference["scope"]["scientific_effect_estimation_performed"]
        is False
    )
    assert taxonomy_reference["field_status_margin_comparison"]["status"] == "PASS"
    assert reference_binding.required_gate_status.model_dump(mode="json") == (
        taxonomy_reference["gate_status"]
    )
    assert manifest.gate_state.model_dump() == {
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "execution_authorized": False,
    }

    validation = planner.validation_summary(manifest, MANIFEST_PATH)
    assert validation["taxonomy_reference_audit"] == (
        reference_binding.model_dump(mode="json")
    )
    assert validation["taxonomy_reference_summary_sha256"] == (
        reference_binding.summary_sha256
    )
    assert validation["taxonomy_reference_result_fingerprint_sha256"] == (
        reference_binding.result_fingerprint_sha256
    )
    assert validation["taxonomy_reference_status"] == (
        "PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT"
    )
    assert validation["taxonomy_reference_component"] == "PASS"
    assert validation["scientific_field_selection"] == (
        "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"
    )
    assert validation["execution_authorized"] is False


def test_manifest_keeps_complete_candidate_universe_without_selecting_fields() -> None:
    manifest = planner.load_manifest(MANIFEST_PATH)
    domains = manifest.candidate_universe.domains
    fields = [field for domain in domains for field in domain.fields]

    assert len(domains) == 4
    assert len(fields) == 26
    assert {field.field_id for field in fields} == set(range(11, 37))
    assert len({domain.rng_code for domain in domains}) == 4
    assert len({field.rng_code for field in fields}) == 26
    assert manifest.candidate_universe.scientific_field_selection == (
        "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"
    )
    assert all(
        field.scientific_selection_status
        == "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"
        for field in fields
    )
    assert all(slot.selected_domain_id is None for slot in manifest.engineering_slots)
    assert all(slot.selected_field_id is None for slot in manifest.engineering_slots)


def test_engineering_slots_and_temporal_partitions_are_disjoint_and_sealed() -> None:
    manifest = planner.load_manifest(MANIFEST_PATH)
    g1_slots = [
        slot for slot in manifest.engineering_slots if slot.role == "g1_replication"
    ]
    g4_slots = [
        slot
        for slot in manifest.engineering_slots
        if slot.role == "g4_domain_author_cold"
    ]
    partitions = {
        partition.partition_id: partition
        for partition in manifest.temporal_partitions
    }

    assert len(g1_slots) == 3
    assert len(g4_slots) == 1
    assert not any(slot.sealed or slot.author_cold for slot in g1_slots)
    assert g4_slots[0].sealed is True
    assert g4_slots[0].author_cold is True
    assert manifest.selection_rules.one_slot_per_domain_without_replacement is True
    assert manifest.selection_rules.g4_domain_disjoint_from_g1 is True
    assert partitions["discovery_calibration"].entry_end_year == 2012
    assert (
        partitions["g1_empirical_replication"].entry_start_year,
        partitions["g1_empirical_replication"].entry_end_year,
    ) == (2013, 2016)
    assert (
        partitions["model_selection"].entry_start_year,
        partitions["model_selection"].entry_end_year,
    ) == (2017, 2019)
    assert (
        partitions["g4_temporal_holdout"].entry_start_year,
        partitions["g4_temporal_holdout"].entry_end_year,
        partitions["g4_temporal_holdout"].outcome_followup_through_year,
    ) == (2020, 2022, 2025)
    assert partitions["g4_temporal_holdout"].sealed is True


def test_plan_expands_exact_core_factorial_and_never_authorizes_execution() -> None:
    manifest = planner.load_manifest(MANIFEST_PATH)
    plan = planner.build_plan(manifest)

    assert plan["counts"] == {
        "slot_count": 4,
        "replicate_seed_count": 3,
        "cells_per_slot_seed": 324,
        "b4_cells_per_slot_seed": 108,
        "b2_b3_cells_per_slot_seed": 216,
        "total_cells": 3888,
        "total_b2_cells": 1296,
        "total_b3_cells": 1296,
        "total_b4_cells": 1296,
    }
    assert plan["execution_authorized"] is False
    assert plan["gate_state"]["pass_g1"] is False
    assert plan["taxonomy_reference_audit"] == (
        manifest.taxonomy_reference_audit.model_dump(mode="json")
    )
    assert plan["taxonomy_reference_audit"]["required_status"] == (
        "PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT"
    )
    assert plan["scientific_field_selection"] == (
        "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"
    )
    assert plan["gate_state"]["g_minus_1_data_readiness"] == "PENDING"
    assert plan["gate_state"]["g0_measurement_freeze"] == "PENDING"
    assert plan["gate_state"]["empirical_g1_signature"] == "NOT_RUN"
    assert all(cell["execution_authorized"] is False for cell in plan["cells"])
    assert all(cell["selected_domain_id"] is None for cell in plan["cells"])
    assert all(cell["selected_field_id"] is None for cell in plan["cells"])
    assert len({cell["cell_id"] for cell in plan["cells"]}) == 3888

    unhashed = dict(plan)
    observed_plan_sha256 = unhashed.pop("plan_sha256")
    assert observed_plan_sha256 == planner.sha256_bytes(
        planner.canonical_json_bytes(unhashed)
    )


def test_seed_sequence_ledger_preserves_paired_counterfactual_streams() -> None:
    plan = planner.build_plan(planner.load_manifest(MANIFEST_PATH))
    streams = plan["seed_sequence_ledger"]["streams"]
    assert len(streams) == 276
    assert planner.sha256_bytes(planner.canonical_json_bytes(streams)) == (
        "b0d861a9ae3cd1bf0859009a08f5c8897166cbfa3c124b2f86d154d207734a2c"
    )
    assert planner.sha256_bytes(
        planner.canonical_json_bytes([stream["stream_id"] for stream in streams])
    ) == "e693d84a7cabd62fafdca873ca01c186f4937c52a6d13c54d7d396909ec1a4ab"
    assert planner.sha256_bytes(
        planner.canonical_json_bytes(
            [stream["state_uint32"] for stream in streams]
        )
    ) == "26326856584df20943c093fd1e0ed09974b1ddf882202336b0e124bc206d9338"

    frozen = _matching_cell(plan, learner="b2_fixed_rule_abm")
    explicit = _matching_cell(plan, learner="b3_explicit_adaptive_baseline")
    nabm = _matching_cell(plan, learner="b4_adaptive_nabm")

    shared_components = {
        "population",
        "base_network",
        "local_noise",
        "latent_outcome_quantile",
        "outcome_disclosure",
        "revelation_delay",
        "peer_signal_diffusion",
        "environment_mapping",
    }
    for component in shared_components:
        assert frozen["seed_stream_ids"][component] == explicit["seed_stream_ids"][
            component
        ]
        assert frozen["seed_stream_ids"][component] == nabm["seed_stream_ids"][
            component
        ]
    for component in ("learner_initialization", "learner_update"):
        assert len(
            {
                frozen["seed_stream_ids"][component],
                explicit["seed_stream_ids"][component],
                nabm["seed_stream_ids"][component],
            }
        ) == 3

    disclosure_variant = _matching_cell(
        plan,
        learner="b2_fixed_rule_abm",
        disclosure="success_biased",
    )
    assert frozen["seed_stream_ids"]["latent_outcome_quantile"] == (
        disclosure_variant["seed_stream_ids"]["latent_outcome_quantile"]
    )
    assert frozen["seed_stream_ids"]["outcome_disclosure"] != (
        disclosure_variant["seed_stream_ids"]["outcome_disclosure"]
    )

    environment_variant = _matching_cell(
        plan,
        learner="b2_fixed_rule_abm",
        environment="structural_support_matched_direct_pressure",
    )
    assert frozen["seed_stream_ids"]["latent_outcome_quantile"] == (
        environment_variant["seed_stream_ids"]["latent_outcome_quantile"]
    )
    assert frozen["seed_stream_ids"]["environment_mapping"] != (
        environment_variant["seed_stream_ids"]["environment_mapping"]
    )


def test_manifest_hash_ignores_yaml_mapping_key_order(tmp_path: Path) -> None:
    raw = _raw_manifest()
    reordered_path = tmp_path / "sorted.yaml"
    reordered_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")

    original = planner.load_manifest(MANIFEST_PATH)
    reordered = planner.load_manifest(reordered_path)

    assert planner.manifest_sha256(original) == planner.manifest_sha256(reordered)
    assert planner.build_plan(original)["plan_sha256"] == planner.build_plan(reordered)[
        "plan_sha256"
    ]


def test_plan_is_identical_across_python_hash_seeds(tmp_path: Path) -> None:
    outputs: list[bytes] = []
    plan_hashes: list[str] = []
    for hash_seed in ("1", "987654"):
        output = tmp_path / f"plan-{hash_seed}.json"
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "plan_research_pivot_field_matrix.py"),
                "plan",
                "--manifest",
                str(MANIFEST_PATH),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        outputs.append(output.read_bytes())
        plan_hashes.append(payload["plan_sha256"])

    assert len(set(plan_hashes)) == 1
    assert hashlib.sha256(outputs[0]).hexdigest() == hashlib.sha256(outputs[1]).hexdigest()


@pytest.mark.parametrize("command", ["run", "execute"])
def test_cli_execution_requests_fail_closed(command: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "plan_research_pivot_field_matrix.py"),
            command,
            "--manifest",
            str(MANIFEST_PATH),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stderr)
    assert payload["status"] == "BLOCKED_ENGINEERING_ONLY"
    assert payload["execution_authorized"] is False
    assert payload["g_minus_1_data_readiness"] == "PENDING"
    assert payload["g0_measurement_freeze"] == "PENDING"
    assert payload["empirical_g1_signature"] == "NOT_RUN"
    assert payload["pass_g1"] is False


def test_schema_forbids_unknown_fields_and_gate_promotion() -> None:
    extra = _raw_manifest()
    extra["unexpected"] = "not allowed"
    promoted = _raw_manifest()
    promoted["gate_state"]["execution_authorized"] = True
    missing_reference = _raw_manifest()
    missing_reference.pop("taxonomy_reference_audit")
    pilot_contract = _raw_manifest()
    pilot_contract["taxonomy_reference_audit"]["required_status"] = (
        "PILOT_REFERENCE_ONLY"
    )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        planner.FieldMatrixManifest.model_validate(extra)
    with pytest.raises(ValidationError, match="literal_error"):
        planner.FieldMatrixManifest.model_validate(promoted)
    with pytest.raises(ValidationError, match="missing"):
        planner.FieldMatrixManifest.model_validate(missing_reference)
    with pytest.raises(ValidationError, match="literal_error"):
        planner.FieldMatrixManifest.model_validate(pilot_contract)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate_field_rng", "candidate field rng_codes must be unique"),
        ("open_g4", "G4 slot must be sealed and domain/author-cold"),
        ("overlap_g4", "temporal range differs for g4_temporal_holdout"),
        ("allow_g4_overlap", "G4 domain must remain disjoint from G1"),
    ],
)
def test_manifest_rejects_broken_field_holdout_contracts(
    mutation: str,
    message: str,
) -> None:
    raw = copy.deepcopy(_raw_manifest())
    if mutation == "duplicate_field_rng":
        raw["candidate_universe"]["domains"][0]["fields"][1]["rng_code"] = 11
    elif mutation == "open_g4":
        raw["engineering_slots"][3]["sealed"] = False
    elif mutation == "overlap_g4":
        raw["temporal_partitions"][3]["entry_start_year"] = 2019
    elif mutation == "allow_g4_overlap":
        raw["selection_rules"]["g4_domain_disjoint_from_g1"] = False
    else:  # pragma: no cover - parametrization owns the values
        raise AssertionError(mutation)

    with pytest.raises(ValidationError, match=message):
        planner.FieldMatrixManifest.model_validate(raw)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("schema", "schema version mismatch"),
        ("pilot_status", "summary status mismatch"),
        ("fingerprint", "result fingerprint mismatch"),
        ("path_redaction", "path redaction is not established"),
        ("taxonomy", "canonical taxonomy SHA-256 mismatch"),
        ("full_scope", "full-scope requirement mismatch"),
        ("scientific_effect", "scientific-effect boundary mismatch"),
        ("field_margin", "field-margin status mismatch"),
    ],
)
def test_taxonomy_reference_payload_rejects_nonpassing_mutations(
    mutation: str,
    message: str,
) -> None:
    manifest = planner.load_manifest(MANIFEST_PATH)
    raw = copy.deepcopy(_json(TAXONOMY_REFERENCE_SUMMARY_PATH))
    if mutation == "schema":
        raw["schema_version"] = "openalex_g1_field_taxonomy_reference_report.v0"
    elif mutation == "pilot_status":
        raw["status"] = "PILOT_REFERENCE_ONLY"
    elif mutation == "fingerprint":
        raw["provenance"]["result_fingerprint_sha256"] = "0" * 64
    elif mutation == "path_redaction":
        raw["provenance"]["path_redacted"] = False
    elif mutation == "taxonomy":
        raw["taxonomy"]["canonical_sha256"] = "0" * 64
    elif mutation == "full_scope":
        raw["scope"]["full_scope"] = False
    elif mutation == "scientific_effect":
        raw["scope"]["scientific_effect_estimation_performed"] = True
    elif mutation == "field_margin":
        raw["field_status_margin_comparison"]["status"] = "NOT_RUN_PILOT"
    else:  # pragma: no cover - parametrization owns the values
        raise AssertionError(mutation)

    with pytest.raises(planner.PlannerError, match=message):
        planner.validate_taxonomy_reference_payload(
            raw,
            manifest.taxonomy_reference_audit,
        )


def test_taxonomy_reference_exact_gate_boundary_fails_closed() -> None:
    manifest = planner.load_manifest(MANIFEST_PATH)
    binding = manifest.taxonomy_reference_audit
    expected = binding.required_gate_status.model_dump(mode="json")
    for key in expected:
        raw = copy.deepcopy(_json(TAXONOMY_REFERENCE_SUMMARY_PATH))
        raw["gate_status"][key] = True if key == "pass_g1" else "PROMOTED"
        with pytest.raises(planner.PlannerError, match=key):
            planner.validate_taxonomy_reference_payload(raw, binding)

    missing = copy.deepcopy(_json(TAXONOMY_REFERENCE_SUMMARY_PATH))
    missing["gate_status"].pop("candidate_eligibility")
    with pytest.raises(planner.PlannerError, match="gate boundary keys mismatch"):
        planner.validate_taxonomy_reference_payload(missing, binding)

    extra = copy.deepcopy(_json(TAXONOMY_REFERENCE_SUMMARY_PATH))
    extra["gate_status"]["final_field_selection"] = "PASS"
    with pytest.raises(planner.PlannerError, match="gate boundary keys mismatch"):
        planner.validate_taxonomy_reference_payload(extra, binding)


def test_taxonomy_reference_binding_fails_closed_on_sha_mismatch(
    tmp_path: Path,
) -> None:
    raw = _raw_manifest()
    raw["taxonomy_reference_audit"]["summary_sha256"] = "0" * 64
    path = _write_manifest(tmp_path, raw)

    with pytest.raises(
        planner.PlannerError,
        match="taxonomy-reference summary SHA-256 mismatch",
    ):
        planner.load_manifest(path)


def test_source_binding_fails_closed_on_sha_mismatch(tmp_path: Path) -> None:
    raw = _raw_manifest()
    raw["source_audit"]["summary_sha256"] = "0" * 64
    path = _write_manifest(tmp_path, raw)

    with pytest.raises(
        planner.PlannerError,
        match="compact work-source summary SHA-256 mismatch",
    ):
        planner.load_manifest(path)
