#!/usr/bin/env python3
"""Build the additive E/G1 input-completeness correction before first read."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
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

DEFAULT_OUTPUT = (
    ROOT
    / "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v2.yaml"
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

CITATION_SHA = "fbe753c84ceb3858a06c627f318eb756e997f73577498b06a5e88f72cacabe76"
CITATION_SCHEMA = "openalex_g1_citation_edge_chronology_outputs.v3"
CITATION_STATUS = (
    "PASS_CURRENT_SNAPSHOT_CITATION_EDGE_CHRONOLOGY_ACCOUNTING_WITH_CANONICAL_"
    "TEMPORAL_V3_PARENT_AND_BIBLIOGRAPHIC_EVENT_TIME_CONVENTION_LIMITATION"
)
CITATION_PREFIX = "chronology_safe_edges_by_cited_work/"
CITATION_COUNT = 4096
CITATION_BYTES = 22_721_388_007
CITATION_INVENTORY_SHA = (
    "afac933e0fb067d4b69486b8f038cf1cd5fc3d3d1de04fa13f33b165d8dfc98b"
)

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
RAW_PATH_SIZE_INVENTORY_SHA = (
    "5ca11e55066c14751a0fd2dc3d8631ef75621109026e8b5fb7250f6ebdaf90a9"
)
CONTENT_FREEZE_SHA = "082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12"
CONTENT_TREE_SHA = "73bb8b0409c782ce5964d480ac6c6cecb04bab8c09df31999e581000b79ccff6"
CHECKPOINT_TREE_SHA = "e509867989cec9dfd398221ee5d7abd69dc73fa7330acf9302413184d786bfef"


class G1InputPreflightV2BuildError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G1InputPreflightV2BuildError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_inventory(entries: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_repo(path: Path) -> dict[str, Any]:
    expected_sha, schema, status = REPOSITORY_EXPECTED[path]
    _require(path.is_file() and _sha256(path) == expected_sha, f"repository hash: {path}")
    payload = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.suffix == ".yaml"
        else json.loads(path.read_text(encoding="utf-8"))
    )
    _require(payload.get("schema_version") == schema, f"repository schema: {path}")
    if status is not None:
        _require(payload.get("status") == status, f"repository status: {path}")
    return payload


def _repo_binding(path: Path) -> dict[str, Any]:
    sha, schema, status = REPOSITORY_EXPECTED[path]
    value = {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "sha256": sha,
        "required_schema_version": schema,
        "mutation_allowed": False,
    }
    if status is not None:
        value["required_status"] = status
    return value


def _citation_binding() -> dict[str, Any]:
    _require(CITATION_RECEIPT.is_file(), "citation receipt missing")
    _require(_sha256(CITATION_RECEIPT) == CITATION_SHA, "citation receipt hash")
    receipt = json.loads(CITATION_RECEIPT.read_text(encoding="utf-8"))
    _require(receipt.get("schema_version") == CITATION_SCHEMA, "citation schema")
    _require(receipt.get("status") == CITATION_STATUS, "citation status")
    entries = sorted(
        (
            entry
            for entry in receipt["artifacts"]
            if entry["relative_path"].startswith(CITATION_PREFIX)
        ),
        key=lambda entry: entry["relative_path"],
    )
    _require(len(entries) == CITATION_COUNT, "citation artifact count")
    _require(sum(int(entry["bytes"]) for entry in entries) == CITATION_BYTES, "citation bytes")
    _require(_canonical_inventory(entries) == CITATION_INVENTORY_SHA, "citation inventory")
    return {
        "root": str(CITATION_ROOT),
        "root_basename": CITATION_ROOT.name,
        "receipt_basename": CITATION_RECEIPT.name,
        "receipt_sha256": CITATION_SHA,
        "receipt_schema_version": CITATION_SCHEMA,
        "required_status": CITATION_STATUS,
        "receipt_artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "selector": {"kind": "RELATIVE_PATH_PREFIX", "value": CITATION_PREFIX},
        "selected_artifact_count": CITATION_COUNT,
        "selected_artifact_bytes": CITATION_BYTES,
        "selected_artifact_inventory_sha256": CITATION_INVENTORY_SHA,
        "inventory_canonicalization": "SORT_RELATIVE_PATH_JSON_SORT_KEYS_COMPACT_UTF8",
        "first_relative_path": entries[0]["relative_path"],
        "last_relative_path": entries[-1]["relative_path"],
        "analytic_artifact_files_opened_during_preflight": 0,
        "analytic_artifact_bytes_read_during_preflight": 0,
        "mutation_allowed": False,
    }


def _taxonomy_binding() -> dict[str, Any]:
    _require(TAXONOMY.is_file(), "taxonomy missing")
    _require(_sha256(TAXONOMY) == TAXONOMY_SHA, "taxonomy hash")
    parquet = pq.ParquetFile(TAXONOMY)
    _require(parquet.metadata.num_rows == TAXONOMY_ROWS, "taxonomy rows")
    _require(all(column in parquet.schema_arrow.names for column in TAXONOMY_COLUMNS), "taxonomy columns")
    _require(TAXONOMY.stat().st_size == TAXONOMY_BYTES, "taxonomy bytes")
    return {
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


def _raw_authorship_binding(s0_summary: dict[str, Any]) -> dict[str, Any]:
    _require(RAW_AUTHORSHIP_ROOT.is_dir(), "raw authorship root missing")
    entries = [
        {"relative_path": path.name, "bytes": path.stat().st_size}
        for path in sorted(RAW_AUTHORSHIP_ROOT.glob("*.parquet"))
    ]
    _require(len(entries) == RAW_SOURCE_FILES, "raw authorship file count")
    _require(sum(entry["bytes"] for entry in entries) == RAW_SOURCE_BYTES, "raw authorship bytes")
    _require(_canonical_inventory(entries) == RAW_PATH_SIZE_INVENTORY_SHA, "raw authorship inventory")
    _require(CONTENT_FREEZE_RECEIPT.is_file(), "content-freeze receipt missing")
    _require(_sha256(CONTENT_FREEZE_RECEIPT) == CONTENT_FREEZE_SHA, "content-freeze receipt hash")
    receipt = json.loads(CONTENT_FREEZE_RECEIPT.read_text(encoding="utf-8"))
    _require(receipt["content_tree_sha256"] == CONTENT_TREE_SHA, "content tree hash")
    _require(receipt["checkpoint_tree"]["tree_sha256"] == CHECKPOINT_TREE_SHA, "checkpoint tree hash")
    source = s0_summary["source_binding"]
    _require(source["selected_source_files"] == RAW_SOURCE_FILES, "S0 selected source files")
    _require(source["selected_source_bytes"] == RAW_SOURCE_BYTES, "S0 selected source bytes")
    _require(source["content_tree_sha256"] == CONTENT_TREE_SHA, "S0 content tree")
    _require(source["checkpoint_tree_sha256"] == CHECKPOINT_TREE_SHA, "S0 checkpoint tree")
    return {
        "root": str(RAW_AUTHORSHIP_ROOT),
        "root_basename": RAW_AUTHORSHIP_ROOT.parent.name,
        "table_basename": RAW_AUTHORSHIP_ROOT.name,
        "file_glob": "*.parquet",
        "selected_file_count": RAW_SOURCE_FILES,
        "selected_file_bytes": RAW_SOURCE_BYTES,
        "path_size_inventory_sha256": RAW_PATH_SIZE_INVENTORY_SHA,
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


def build_contract() -> dict[str, Any]:
    parent = _load_repo(PARENT)
    parent_summary = _load_repo(PARENT_SUMMARY)
    field_config = _load_repo(FIELD_CONFIG)
    field_summary = _load_repo(FIELD_SUMMARY)
    s0_config = _load_repo(S0_CONFIG)
    s0_summary = _load_repo(S0_SUMMARY)
    _require(parent["lifecycle"]["empirical_g1_status"] == "NOT_RUN", "parent G1 already run")
    _require(parent_summary["fail_closed_state"]["confirmatory_rows_read"] == 0, "parent read rows")
    _require(field_config["normalized_inputs"]["topic_taxonomy_projection"]["sha256"] == TAXONOMY_SHA, "field taxonomy binding")
    _require(
        field_summary["bound_implementation"]["config_sha256"]
        == REPOSITORY_EXPECTED[FIELD_CONFIG][0],
        "field taxonomy contract replay",
    )
    _require(s0_config["source"]["expected_files"] == RAW_SOURCE_FILES, "S0 source count")

    return {
        "schema_version": "pivot_feedback_e_g1_execution_input_preflight.v2",
        "label": "pivot_feedback_e_g1_execution_input_preflight_v2",
        "mode": "ADDITIVE_PRE_FIRST_READ_G1_INPUT_COMPLETENESS_CORRECTION_NO_ANALYTIC_READ",
        "status": "E_G1_EXECUTION_INPUT_PREFLIGHT_V2_COMPLETENESS_PENDING_INDEPENDENT_VALIDATION_NO_G1_RUN",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "additive_parent": {
            **_repo_binding(PARENT),
            "parent_independent_verification": _repo_binding(PARENT_SUMMARY),
            "parent_remains_immutable_provenance": True,
            "parent_execution_authorization_superseded_by_complete_conjunction": True,
        },
        "completeness_diagnosis": {
            "detected_before_first_confirmatory_read": True,
            "confirmatory_rows_read_before_detection": 0,
            "holdout_rows_read_before_detection": 0,
            "omitted_required_input_roles": [
                "AGE3_CITATION_AVAILABILITY_AND_FUTURE_CITATION_NEGATIVE_CONTROL",
                "FULL_TOPIC_TO_SUBFIELD_ALTERNATIVE_PIVOT_AND_UNRELATED_TOPIC_CONTROL",
                "REQUIRED_ORCID_LINKED_AND_AFFILIATION_SENSITIVITIES",
            ],
            "scientific_rule_or_threshold_changed": False,
            "registered_scope_changed": False,
            "output_extension_required": "IDENTITY_AND_AFFILIATION_SENSITIVITY_ARTIFACT",
            "this_correction_is_new_scientific_gate": False,
        },
        "added_repository_bindings": {
            "field_episode_taxonomy_contract": _repo_binding(FIELD_CONFIG),
            "field_episode_taxonomy_verification": _repo_binding(FIELD_SUMMARY),
            "s0_orcid_affiliation_contract": _repo_binding(S0_CONFIG),
            "s0_orcid_affiliation_verification": _repo_binding(S0_SUMMARY),
        },
        "added_exact_non_holdout_inputs": {
            "citation_chronology": _citation_binding(),
            "topic_taxonomy_projection": _taxonomy_binding(),
            "raw_authorship_orcid_affiliation": _raw_authorship_binding(s0_summary),
        },
        "effective_execution_input_contract": {
            "requires_parent_v1_and_this_v2": True,
            "effective_input_group_count": 7,
            "input_group_ids": [
                "normalized_work_metadata",
                "author_history_events",
                "aligned_final_panel_metadata",
                "atomic_partition_metadata",
                "citation_chronology",
                "topic_taxonomy_projection",
                "raw_authorship_orcid_affiliation",
            ],
            "private_g4_holdout_included": False,
            "g4_field_or_year_included": False,
            "all_required_primary_and_mandatory_sensitivity_sources_bound": True,
        },
        "execution_output_contract_extension": {
            "effective_output_artifact_schema_count": 8,
            "added_artifact": {
                "relative_path": "g1_identity_affiliation_sensitivities.parquet",
                "schema_version": "pivot_feedback_e_g1_identity_affiliation_sensitivities.v1",
                "confidentiality": "AGGREGATE_SUPPRESSION_REQUIRED",
                "exact_columns": [
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
                ],
            },
            "orcid_linked_is_required_sensitivity_not_primary_replacement": True,
            "affiliation_is_current_snapshot_proxy_not_historical_truth": True,
            "raw_authorship_identifiers_operator_private": True,
        },
        "preflight_access_accounting": {
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
        },
        "lifecycle": {
            "complete_input_contract_authored": True,
            "independent_v2_verification_complete": False,
            "g1_execution_run": False,
            "empirical_g1_status": "NOT_RUN",
            "pass_e": False,
            "pass_g1": False,
            "holdout_seal_opened": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": {
            "exact_g1_execution_successor": "AUTHORIZED_ONLY_AFTER_INDEPENDENT_V2_COMPLETENESS_PASS",
            "confirmatory_analytic_access": "AUTHORIZED_ONLY_DURING_SEPARATE_EXECUTION_SUCCESSOR_FOR_PARENT_V1_PLUS_V2_BOUND_SCOPE",
            "v1_alone_for_execution": "NOT_AUTHORIZED_AFTER_PRE_READ_COMPLETENESS_DIAGNOSIS",
            "scope_or_threshold_change": "NOT_AUTHORIZED",
            "g4_artifact_access": "NOT_AUTHORIZED",
            "g4_seal_open": "NOT_AUTHORIZED",
            "model_selection_score_read": "NOT_AUTHORIZED",
            "predictive_or_nabm_model_fitting": "NOT_AUTHORIZED",
            "causal_claim": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "validation_contract": {
            "verifier_relative_path": "scripts/verify_pivot_feedback_e_g1_execution_input_preflight_v2.py",
            "summary_json_relative_path": "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v2_20260805.summary.json",
            "summary_markdown_relative_path": "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v2_20260805.summary.md",
            "analytic_or_confirmatory_rows_may_be_read": False,
            "source_file_content_may_be_read_except_taxonomy_reference_footer": False,
        },
        "claim_boundary": (
            "This additive pre-read correction binds the citation chronology, full "
            "topic taxonomy, and raw authorship ORCID/affiliation source required by "
            "already-frozen primary and mandatory sensitivity rules. It changes no "
            "field, episode, year, threshold, control, estimand, or decision rule; "
            "reads no analytic, confirmatory, or holdout row; produces no G1 result; "
            "includes no private G4 artifact; opens no seal; and establishes no "
            "association, null, prediction, transport, causality, or scientific effect."
        ),
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise G1InputPreflightV2BuildError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_contract()
    _write_create_only(args.output, payload)
    print(json.dumps({
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "effective_input_group_count": 7,
        "confirmatory_rows_read": 0,
        "g1_status": "NOT_RUN",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
