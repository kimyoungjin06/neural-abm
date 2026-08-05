#!/usr/bin/env python3
"""Independently verify the additive pre-read E/G1 input-completeness correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORITY = (
    ROOT / "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v2.yaml"
)
BUILDER = ROOT / "scripts/build_pivot_feedback_e_g1_execution_input_preflight_v2.py"
VERIFIER = Path(__file__).resolve()

PARENT = ROOT / "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v1.yaml"
PARENT_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.json"
)
FIELD_CONFIG = ROOT / "experiments/configs/pivot_feedback_f_field_episode_assignment_v1.yaml"
FIELD_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_field_episode_assignment_v1_20260803.summary.json"
)
S0_CONFIG = ROOT / "experiments/configs/pivot_feedback_s0_authorship_orcid_affiliation_v1.yaml"
S0_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_s0_authorship_orcid_affiliation_full_20260724.summary.json"
)

CITATION_ROOT = Path("/ssd/openalex-g1-citation-chronology-v3-full-20260722")
CITATION_RECEIPT = CITATION_ROOT / "citation_edge_chronology_audit.outputs.json"
TAXONOMY = Path(
    "/hdd/raid_backup_20260624/data/OpenAlex/parquet_exports/"
    "openalex_nonworks_20260330_delta_20260417_100128/"
    "openalex_topics_20260330_delta/openalex_topics_20260330_delta/b000000.parquet"
)
RAW_AUTHORSHIP_ROOT = Path(
    "/hdd/raid_backup_20260624/data/OpenAlex/parquet_exports/"
    "openalex_works_20260330_repairreplay_20260410_190630/"
    "openalex_works_20260330__authorships"
)
CONTENT_FREEZE_RECEIPT = (
    Path("/ssd/openalex-g1-content-freeze-20260719")
    / "content_freeze_audit.outputs.json"
)

REPOSITORY_EXPECTED = {
    PARENT: (
        "9611bdaa1e0144540d82834f743fbf51cc733ef9492771ed9d37f66ec75bf4d2",
        "pivot_feedback_e_g1_execution_input_preflight.v1",
        "E_G1_EXECUTION_INPUT_PREFLIGHT_COMPLETE_PENDING_INDEPENDENT_VALIDATION_NO_G1_RUN",
    ),
    PARENT_SUMMARY: (
        "b44386823c4335aee1e8423ca9ce1d46af51884e69a4e7986e088397556450e1",
        "pivot_feedback_e_g1_execution_input_preflight_summary.v1",
        "PASS_INDEPENDENT_E_G1_EXECUTION_INPUT_PREFLIGHT_EXACT_NON_HOLDOUT_SCOPE_NO_G1_RUN",
    ),
    FIELD_CONFIG: (
        "4a48c46483d3711bffe1d8c13805b4aec6f8aa25d4198dc30372a5da22d0f68b",
        "pivot_feedback_f_field_episode_assignment.v1",
        "FROZEN_FIELD_EPISODE_ASSIGNMENT_PENDING_PRODUCTION_EXECUTION_NO_F_PASS",
    ),
    FIELD_SUMMARY: (
        "1b09be6348e08b551011051b2c4f1377db285985dfac0e9a9a5d5f18e2a8fa08",
        "pivot_feedback_f_field_episode_assignment_summary.v1",
        "PASS_OUTCOME_BLIND_FIELD_AND_EPISODE_ASSIGNMENT_COMPONENT_FINAL_PANEL_AND_SEAL_PENDING",
    ),
    S0_CONFIG: (
        "a792355ab2afec5ecef7cdc10cc97a4721ed744e620a795ec7ddefbef0a30636",
        "pivot_feedback_s0_authorship_orcid_affiliation_config.v1",
        None,
    ),
    S0_SUMMARY: (
        "53547820a617fbd3e222e46214d2a5eb14570ff5370e03ea12015298b6afd78a",
        "pivot_feedback_s0_authorship_orcid_affiliation_full_summary.v1",
        "PASS_S0_FULL_AUTHORSHIP_ORCID_AND_AFFILIATION_PROJECTION_WITH_TOKEN_AND_CURRENT_SNAPSHOT_LIMITATIONS",
    ),
}

CITATION_EXPECTED = {
    "sha256": "fbe753c84ceb3858a06c627f318eb756e997f73577498b06a5e88f72cacabe76",
    "schema_version": "openalex_g1_citation_edge_chronology_outputs.v3",
    "status": (
        "PASS_CURRENT_SNAPSHOT_CITATION_EDGE_CHRONOLOGY_ACCOUNTING_WITH_CANONICAL_"
        "TEMPORAL_V3_PARENT_AND_BIBLIOGRAPHIC_EVENT_TIME_CONVENTION_LIMITATION"
    ),
    "artifact_tree_sha256": "2c26cd80990a270b1fb1fd5780f1eab253a29f2c4f5f9e01ee878f630e13184d",
    "prefix": "chronology_safe_edges_by_cited_work/",
    "count": 4096,
    "bytes": 22_721_388_007,
    "inventory_sha256": "afac933e0fb067d4b69486b8f038cf1cd5fc3d3d1de04fa13f33b165d8dfc98b",
}
TAXONOMY_SHA = "566facdbc02bf4a37783fe64db4c1d104d00b73e250999bd824d560083827ba5"
TAXONOMY_BYTES = 1_194_401
TAXONOMY_ROWS = 4516
TAXONOMY_COLUMNS = [
    "id",
    "display_name",
    "field__id",
    "field__display_name",
    "subfield__id",
    "subfield__display_name",
]
RAW_SOURCE_FILES = 1363
RAW_SOURCE_BYTES = 103_412_497_044
RAW_INVENTORY_SHA = "5ca11e55066c14751a0fd2dc3d8631ef75621109026e8b5fb7250f6ebdaf90a9"
CONTENT_FREEZE_SHA = "082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12"
CONTENT_TREE_SHA = "73bb8b0409c782ce5964d480ac6c6cecb04bab8c09df31999e581000b79ccff6"
CHECKPOINT_TREE_SHA = "e509867989cec9dfd398221ee5d7abd69dc73fa7330acf9302413184d786bfef"

INPUT_GROUP_IDS = [
    "normalized_work_metadata",
    "author_history_events",
    "aligned_final_panel_metadata",
    "atomic_partition_metadata",
    "citation_chronology",
    "topic_taxonomy_projection",
    "raw_authorship_orcid_affiliation",
]
OMITTED_ROLES = [
    "AGE3_CITATION_AVAILABILITY_AND_FUTURE_CITATION_NEGATIVE_CONTROL",
    "FULL_TOPIC_TO_SUBFIELD_ALTERNATIVE_PIVOT_AND_UNRELATED_TOPIC_CONTROL",
    "REQUIRED_ORCID_LINKED_AND_AFFILIATION_SENSITIVITIES",
]
SENSITIVITY_COLUMNS = [
    "sensitivity_id",
    "population_filter",
    "primary_field_id",
    "hypothesis_id",
    "point_estimate",
    "interval_lower",
    "interval_upper",
    "effective_sample_size",
    "missing_proxy_category_included",
    "registered_decision_impact",
    "status",
    "limitation",
]


class G1InputPreflightV2VerificationError(RuntimeError):
    """Raised when independent replay finds drift or an unsafe boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G1InputPreflightV2VerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _canonical_inventory(entries: list[dict[str, Any]]) -> str:
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_repository(path: Path) -> dict[str, Any]:
    expected_sha, expected_schema, expected_status = REPOSITORY_EXPECTED[path]
    _require(path.is_file(), f"missing repository document: {path}")
    _require(_sha256(path) == expected_sha, f"repository hash drift: {path}")
    payload = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.suffix == ".yaml"
        else json.loads(path.read_text(encoding="utf-8"))
    )
    _require(payload.get("schema_version") == expected_schema, f"repository schema drift: {path}")
    if expected_status is not None:
        _require(payload.get("status") == expected_status, f"repository status drift: {path}")
    return payload


def _verify_repository_binding(binding: dict[str, Any], path: Path) -> None:
    expected_sha, expected_schema, expected_status = REPOSITORY_EXPECTED[path]
    _require(binding["relative_path"] == _relative(path), f"binding path drift: {path}")
    _require(binding["sha256"] == expected_sha, f"binding hash drift: {path}")
    _require(binding["required_schema_version"] == expected_schema, f"binding schema drift: {path}")
    if expected_status is None:
        _require("required_status" not in binding, f"unexpected binding status: {path}")
    else:
        _require(binding["required_status"] == expected_status, f"binding status drift: {path}")
    _require(binding["mutation_allowed"] is False, f"mutable binding: {path}")


def _replay_citation(binding: dict[str, Any]) -> dict[str, Any]:
    _require(CITATION_RECEIPT.is_file(), "citation receipt missing")
    _require(_sha256(CITATION_RECEIPT) == CITATION_EXPECTED["sha256"], "citation receipt hash drift")
    receipt = json.loads(CITATION_RECEIPT.read_text(encoding="utf-8"))
    _require(receipt.get("schema_version") == CITATION_EXPECTED["schema_version"], "citation schema drift")
    _require(receipt.get("status") == CITATION_EXPECTED["status"], "citation status drift")
    _require(receipt.get("artifact_tree_sha256") == CITATION_EXPECTED["artifact_tree_sha256"], "citation tree drift")
    entries = sorted(
        [
            entry
            for entry in receipt.get("artifacts", [])
            if entry.get("relative_path", "").startswith(CITATION_EXPECTED["prefix"])
        ],
        key=lambda entry: entry["relative_path"],
    )
    inventory = _canonical_inventory(entries)
    _require(len(entries) == CITATION_EXPECTED["count"], "citation artifact count drift")
    _require(sum(int(entry["bytes"]) for entry in entries) == CITATION_EXPECTED["bytes"], "citation artifact bytes drift")
    _require(inventory == CITATION_EXPECTED["inventory_sha256"], "citation inventory drift")
    expected_binding = {
        "root": str(CITATION_ROOT),
        "root_basename": CITATION_ROOT.name,
        "receipt_basename": CITATION_RECEIPT.name,
        "receipt_sha256": CITATION_EXPECTED["sha256"],
        "receipt_schema_version": CITATION_EXPECTED["schema_version"],
        "required_status": CITATION_EXPECTED["status"],
        "receipt_artifact_tree_sha256": CITATION_EXPECTED["artifact_tree_sha256"],
        "selector": {"kind": "RELATIVE_PATH_PREFIX", "value": CITATION_EXPECTED["prefix"]},
        "selected_artifact_count": CITATION_EXPECTED["count"],
        "selected_artifact_bytes": CITATION_EXPECTED["bytes"],
        "selected_artifact_inventory_sha256": CITATION_EXPECTED["inventory_sha256"],
        "inventory_canonicalization": "SORT_RELATIVE_PATH_JSON_SORT_KEYS_COMPACT_UTF8",
        "first_relative_path": entries[0]["relative_path"],
        "last_relative_path": entries[-1]["relative_path"],
        "analytic_artifact_files_opened_during_preflight": 0,
        "analytic_artifact_bytes_read_during_preflight": 0,
        "mutation_allowed": False,
    }
    _require(binding == expected_binding, "citation binding drift")
    return {
        "artifact_count": len(entries),
        "artifact_bytes": CITATION_EXPECTED["bytes"],
        "inventory_sha256": inventory,
        "receipt_artifact_tree_sha256": receipt["artifact_tree_sha256"],
    }


def _replay_taxonomy(binding: dict[str, Any], field_config: dict[str, Any]) -> dict[str, Any]:
    _require(TAXONOMY.is_file(), "taxonomy projection missing")
    _require(_sha256(TAXONOMY) == TAXONOMY_SHA, "taxonomy hash drift")
    _require(TAXONOMY.stat().st_size == TAXONOMY_BYTES, "taxonomy bytes drift")
    parquet = pq.ParquetFile(TAXONOMY)
    _require(parquet.metadata.num_rows == TAXONOMY_ROWS, "taxonomy row count drift")
    _require(all(column in parquet.schema_arrow.names for column in TAXONOMY_COLUMNS), "taxonomy columns drift")
    frozen = field_config["normalized_inputs"]["topic_taxonomy_projection"]
    _require(frozen["sha256"] == TAXONOMY_SHA, "frozen taxonomy hash drift")
    _require(frozen["rows"] == TAXONOMY_ROWS, "frozen taxonomy rows drift")
    _require(frozen["required_columns"] == TAXONOMY_COLUMNS, "frozen taxonomy columns drift")
    expected_binding = {
        "path": str(TAXONOMY),
        "basename": TAXONOMY.name,
        "sha256": TAXONOMY_SHA,
        "bytes": TAXONOMY_BYTES,
        "rows": TAXONOMY_ROWS,
        "required_columns": TAXONOMY_COLUMNS,
        "current_snapshot_projection": True,
        "analytic_outcome_rows_read_during_preflight": 0,
        "mutation_allowed": False,
    }
    _require(binding == expected_binding, "taxonomy binding drift")
    return {
        "bytes": TAXONOMY_BYTES,
        "rows": TAXONOMY_ROWS,
        "sha256": TAXONOMY_SHA,
        "required_columns": TAXONOMY_COLUMNS,
    }


def _replay_raw_authorship(
    binding: dict[str, Any],
    s0_config: dict[str, Any],
    s0_summary: dict[str, Any],
) -> dict[str, Any]:
    _require(RAW_AUTHORSHIP_ROOT.is_dir(), "raw authorship root missing")
    entries = [
        {"relative_path": path.name, "bytes": path.stat().st_size}
        for path in sorted(RAW_AUTHORSHIP_ROOT.glob("*.parquet"))
    ]
    inventory = _canonical_inventory(entries)
    _require(len(entries) == RAW_SOURCE_FILES, "raw authorship file count drift")
    _require(sum(int(entry["bytes"]) for entry in entries) == RAW_SOURCE_BYTES, "raw authorship bytes drift")
    _require(inventory == RAW_INVENTORY_SHA, "raw authorship inventory drift")

    _require(CONTENT_FREEZE_RECEIPT.is_file(), "content-freeze receipt missing")
    _require(_sha256(CONTENT_FREEZE_RECEIPT) == CONTENT_FREEZE_SHA, "content-freeze receipt hash drift")
    freeze = json.loads(CONTENT_FREEZE_RECEIPT.read_text(encoding="utf-8"))
    _require(freeze.get("schema_version") == "openalex_g1_content_freeze_outputs.v1", "content-freeze schema drift")
    _require(freeze.get("status") == "PASS_CURRENT_CONTENT_BASELINE_COMPONENT", "content-freeze status drift")
    _require(freeze.get("content_tree_sha256") == CONTENT_TREE_SHA, "content tree drift")
    _require(freeze["checkpoint_tree"]["tree_sha256"] == CHECKPOINT_TREE_SHA, "checkpoint tree drift")

    source = s0_config["source"]
    _require(source["expected_files"] == RAW_SOURCE_FILES, "S0 source file count drift")
    _require(source["expected_bytes"] == RAW_SOURCE_BYTES, "S0 source bytes drift")
    _require(s0_config["content_freeze_parent"]["receipt_sha256"] == CONTENT_FREEZE_SHA, "S0 freeze receipt drift")
    _require(s0_config["content_freeze_parent"]["content_tree_sha256"] == CONTENT_TREE_SHA, "S0 content tree drift")
    _require(s0_config["content_freeze_parent"]["checkpoint_tree_sha256"] == CHECKPOINT_TREE_SHA, "S0 checkpoint tree drift")
    source_summary = s0_summary["source_binding"]
    _require(source_summary["selected_source_files"] == RAW_SOURCE_FILES, "S0 verified file count drift")
    _require(source_summary["selected_source_bytes"] == RAW_SOURCE_BYTES, "S0 verified bytes drift")
    _require(source_summary["content_tree_sha256"] == CONTENT_TREE_SHA, "S0 verified content tree drift")
    _require(source_summary["checkpoint_tree_sha256"] == CHECKPOINT_TREE_SHA, "S0 verified checkpoint tree drift")
    _require(s0_config["orcid_contract"]["natural_person_identity_claim_allowed"] is False, "unsafe ORCID identity claim")
    _require(s0_config["affiliation_contract"]["historical_affiliation_observability_established"] is False, "unsafe affiliation claim")

    expected_binding = {
        "root": str(RAW_AUTHORSHIP_ROOT),
        "root_basename": RAW_AUTHORSHIP_ROOT.parent.name,
        "table_basename": RAW_AUTHORSHIP_ROOT.name,
        "file_glob": "*.parquet",
        "selected_file_count": RAW_SOURCE_FILES,
        "selected_file_bytes": RAW_SOURCE_BYTES,
        "path_size_inventory_sha256": RAW_INVENTORY_SHA,
        "path_size_inventory_canonicalization": "LEXICAL_BASENAME_JSON_SORT_KEYS_COMPACT_UTF8",
        "content_freeze_receipt": str(CONTENT_FREEZE_RECEIPT),
        "content_freeze_receipt_sha256": CONTENT_FREEZE_SHA,
        "content_tree_sha256": CONTENT_TREE_SHA,
        "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA,
        "every_selected_source_file_current_exact_stat_verified": True,
        "first_relative_path": entries[0]["relative_path"],
        "last_relative_path": entries[-1]["relative_path"],
        "source_parquet_files_opened_during_preflight": 0,
        "source_rows_read_during_preflight": 0,
        "mutation_allowed": False,
    }
    _require(binding == expected_binding, "raw authorship binding drift")
    return {
        "selected_file_count": RAW_SOURCE_FILES,
        "selected_file_bytes": RAW_SOURCE_BYTES,
        "path_size_inventory_sha256": inventory,
        "content_tree_sha256": CONTENT_TREE_SHA,
        "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA,
    }


def verify(authority_path: Path) -> dict[str, Any]:
    _require(authority_path.is_file(), f"missing authority: {authority_path}")
    authority = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    _require(authority.get("schema_version") == "pivot_feedback_e_g1_execution_input_preflight.v2", "v2 schema drift")
    _require(
        authority.get("status")
        == "E_G1_EXECUTION_INPUT_PREFLIGHT_V2_COMPLETENESS_PENDING_INDEPENDENT_VALIDATION_NO_G1_RUN",
        "v2 status drift",
    )
    _require(
        authority.get("mode")
        == "ADDITIVE_PRE_FIRST_READ_G1_INPUT_COMPLETENESS_CORRECTION_NO_ANALYTIC_READ",
        "v2 mode drift",
    )

    parent = _load_repository(PARENT)
    parent_summary = _load_repository(PARENT_SUMMARY)
    field_config = _load_repository(FIELD_CONFIG)
    field_summary = _load_repository(FIELD_SUMMARY)
    s0_config = _load_repository(S0_CONFIG)
    s0_summary = _load_repository(S0_SUMMARY)

    additive_parent = dict(authority["additive_parent"])
    parent_verification = additive_parent.pop("parent_independent_verification", None)
    _verify_repository_binding(additive_parent, PARENT)
    _require(parent_verification is not None, "parent verification binding missing")
    _verify_repository_binding(parent_verification, PARENT_SUMMARY)
    _require(additive_parent["parent_remains_immutable_provenance"] is True, "parent provenance mutable")
    _require(additive_parent["parent_execution_authorization_superseded_by_complete_conjunction"] is True, "v1 alone remains authorized")
    _require(parent["lifecycle"]["empirical_g1_status"] == "NOT_RUN", "parent G1 already run")
    _require(parent_summary["fail_closed_state"]["confirmatory_rows_read"] == 0, "parent confirmatory rows read")
    _require(parent_summary["fail_closed_state"]["holdout_rows_read"] == 0, "parent holdout rows read")

    repository_bindings = authority["added_repository_bindings"]
    expected_repository_bindings = {
        "field_episode_taxonomy_contract": FIELD_CONFIG,
        "field_episode_taxonomy_verification": FIELD_SUMMARY,
        "s0_orcid_affiliation_contract": S0_CONFIG,
        "s0_orcid_affiliation_verification": S0_SUMMARY,
    }
    _require(set(repository_bindings) == set(expected_repository_bindings), "added repository binding set drift")
    for name, path in expected_repository_bindings.items():
        _verify_repository_binding(repository_bindings[name], path)
    _require(field_summary["bound_implementation"]["config_sha256"] == REPOSITORY_EXPECTED[FIELD_CONFIG][0], "field verification/config lineage drift")

    diagnosis = authority["completeness_diagnosis"]
    expected_diagnosis = {
        "detected_before_first_confirmatory_read": True,
        "confirmatory_rows_read_before_detection": 0,
        "holdout_rows_read_before_detection": 0,
        "omitted_required_input_roles": OMITTED_ROLES,
        "scientific_rule_or_threshold_changed": False,
        "registered_scope_changed": False,
        "output_extension_required": "IDENTITY_AND_AFFILIATION_SENSITIVITY_ARTIFACT",
        "this_correction_is_new_scientific_gate": False,
    }
    _require(diagnosis == expected_diagnosis, "completeness diagnosis drift")

    inputs = authority["added_exact_non_holdout_inputs"]
    _require(
        set(inputs)
        == {"citation_chronology", "topic_taxonomy_projection", "raw_authorship_orcid_affiliation"},
        "added input set drift",
    )
    replay = {
        "citation_chronology": _replay_citation(inputs["citation_chronology"]),
        "topic_taxonomy_projection": _replay_taxonomy(inputs["topic_taxonomy_projection"], field_config),
        "raw_authorship_orcid_affiliation": _replay_raw_authorship(
            inputs["raw_authorship_orcid_affiliation"], s0_config, s0_summary
        ),
    }

    effective = authority["effective_execution_input_contract"]
    _require(effective["requires_parent_v1_and_this_v2"] is True, "v2 conjunction not required")
    _require(effective["effective_input_group_count"] == 7, "effective input count drift")
    _require(effective["input_group_ids"] == INPUT_GROUP_IDS, "effective input group drift")
    _require(effective["private_g4_holdout_included"] is False, "private G4 holdout included")
    _require(effective["g4_field_or_year_included"] is False, "G4 field or year included")
    _require(effective["all_required_primary_and_mandatory_sensitivity_sources_bound"] is True, "input contract incomplete")
    scope = parent["registered_execution_scope"]
    _require([scope["entry_year_start"], scope["entry_year_end"]] == [2013, 2016], "parent year scope drift")
    _require(scope["selected_field_ids_in_registered_slot_order"] == [16, 32, 11], "parent field scope drift")
    _require(scope["assigned_episode_count"] == 28 and scope["assigned_family_count"] == 13, "parent episode/family scope drift")
    _require(scope["g4_field_allowed"] is False, "parent G4 field allowed")
    _require(scope["g4_private_artifact_root_or_manifest_is_execution_input"] is False, "parent private G4 input included")

    extension = authority["execution_output_contract_extension"]
    _require(len(parent["execution_output_contract"]["artifacts"]) == 7, "parent output schema count drift")
    _require(extension["effective_output_artifact_schema_count"] == 8, "effective output count drift")
    _require(
        extension["added_artifact"]
        == {
            "relative_path": "g1_identity_affiliation_sensitivities.parquet",
            "schema_version": "pivot_feedback_e_g1_identity_affiliation_sensitivities.v1",
            "confidentiality": "AGGREGATE_SUPPRESSION_REQUIRED",
            "exact_columns": SENSITIVITY_COLUMNS,
        },
        "sensitivity output schema drift",
    )
    _require(extension["orcid_linked_is_required_sensitivity_not_primary_replacement"] is True, "ORCID replaced primary analysis")
    _require(extension["affiliation_is_current_snapshot_proxy_not_historical_truth"] is True, "affiliation overclaim")
    _require(extension["raw_authorship_identifiers_operator_private"] is True, "raw identifiers not private")

    expected_access = {
        "parent_repository_yaml_documents_read": 1,
        "repository_json_documents_read": 3,
        "additional_repository_yaml_documents_read": 2,
        "operator_metadata_receipts_read": 2,
        "taxonomy_reference_parquet_footers_read": 1,
        "raw_source_file_stats_read": RAW_SOURCE_FILES,
        "analytic_parquet_or_row_level_files_opened": 0,
        "confirmatory_rows_read": 0,
        "holdout_rows_read": 0,
        "labels_materialized": 0,
        "estimators_run": 0,
        "model_selection_scores_read": 0,
        "seal_open_events": 0,
        "g4_open": False,
    }
    _require(authority["preflight_access_accounting"] == expected_access, "access accounting drift")
    lifecycle = authority["lifecycle"]
    _require(lifecycle["complete_input_contract_authored"] is True, "complete contract not authored")
    _require(lifecycle["independent_v2_verification_complete"] is False, "authority self-asserts verification")
    _require(lifecycle["g1_execution_run"] is False and lifecycle["empirical_g1_status"] == "NOT_RUN", "G1 already run")
    _require(lifecycle["pass_e"] is False and lifecycle["pass_g1"] is False, "E/G1 passed during preflight")
    _require(lifecycle["holdout_seal_opened"] is False and lifecycle["g4_open"] is False, "holdout opened")
    _require(lifecycle["scientific_effect"] == "NOT_ESTABLISHED", "scientific effect asserted")
    authorization = authority["authorization"]
    _require(authorization["exact_g1_execution_successor"] == "AUTHORIZED_ONLY_AFTER_INDEPENDENT_V2_COMPLETENESS_PASS", "unsafe execution authorization")
    _require(authorization["v1_alone_for_execution"] == "NOT_AUTHORIZED_AFTER_PRE_READ_COMPLETENESS_DIAGNOSIS", "v1 alone authorized")
    for key in (
        "g4_artifact_access",
        "g4_seal_open",
        "model_selection_score_read",
        "predictive_or_nabm_model_fitting",
        "causal_claim",
    ):
        _require(authorization[key] == "NOT_AUTHORIZED", f"unsafe authorization: {key}")

    return {
        "schema_version": "pivot_feedback_e_g1_execution_input_preflight_v2_summary.v1",
        "status": "PASS_INDEPENDENT_E_G1_EXECUTION_INPUT_PREFLIGHT_V2_COMPLETE_NO_G1_RUN",
        "repository_artifacts": {
            "preflight_contract": {"relative_path": _relative(authority_path), "sha256": _sha256(authority_path)},
            "builder": {"relative_path": _relative(BUILDER), "sha256": _sha256(BUILDER)},
            "independent_verifier": {"relative_path": _relative(VERIFIER), "sha256": _sha256(VERIFIER)},
        },
        "parent_and_scope_replay": {
            "parent_v1_sha256": REPOSITORY_EXPECTED[PARENT][0],
            "parent_v1_independent_pass_sha256": REPOSITORY_EXPECTED[PARENT_SUMMARY][0],
            "entry_years": [2013, 2016],
            "field_ids_in_registered_slot_order": [16, 32, 11],
            "episode_count": 28,
            "family_count": 13,
            "parent_v1_alone_superseded": True,
            "g4_field_or_year_excluded": True,
            "g4_private_artifact_excluded": True,
        },
        "completeness_correction_replay": {
            "detected_before_first_confirmatory_read": True,
            "omitted_required_input_roles": OMITTED_ROLES,
            "scientific_rule_or_threshold_changed": False,
            "registered_scope_changed": False,
            "this_is_new_scientific_gate": False,
        },
        "added_exact_input_replay": replay,
        "effective_contract_replay": {
            "effective_input_group_count": 7,
            "input_group_ids": INPUT_GROUP_IDS,
            "all_required_primary_and_mandatory_sensitivity_sources_bound": True,
            "effective_output_artifact_schema_count": 8,
            "identity_affiliation_sensitivity_schema_bound": True,
        },
        "access_accounting": expected_access,
        "fail_closed_state": {
            "g1_execution_run": False,
            "empirical_g1_status": "NOT_RUN",
            "pass_e": False,
            "pass_g1": False,
            "confirmatory_rows_read": 0,
            "holdout_rows_read": 0,
            "labels_materialized": 0,
            "estimators_run": 0,
            "model_selection_scores_read": 0,
            "holdout_seal_opened": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": {
            "exact_g1_execution_successor": "AUTHORIZED_NEXT_FROM_PARENT_V1_PLUS_INDEPENDENTLY_VERIFIED_V2",
            "confirmatory_analytic_access": "AUTHORIZED_ONLY_DURING_SEPARATE_EXECUTION_SUCCESSOR_FOR_EXACT_BOUND_G1_SCOPE",
            "scope_or_threshold_change_after_confirmatory_access": "NOT_AUTHORIZED",
            "g4_artifact_access": "NOT_AUTHORIZED",
            "g4_seal_open": "NOT_AUTHORIZED",
            "model_selection_score_read": "NOT_AUTHORIZED",
            "predictive_or_nabm_model_fitting": "NOT_AUTHORIZED",
            "causal_claim": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "The independent pre-read replay verifies the additive citation, full "
            "taxonomy, and raw authorship ORCID/affiliation bindings required by "
            "the frozen G1 analysis and mandatory sensitivities. It changes no "
            "scientific rule or registered scope, reads no analytic, confirmatory, "
            "or holdout row, produces no G1 result, includes no private G4 artifact, "
            "opens no seal, and establishes no scientific effect."
        ),
    }


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise G1InputPreflightV2VerificationError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify(args.authority)
    if args.output is not None:
        _write_json_create_only(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
