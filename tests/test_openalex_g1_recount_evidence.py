from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments/evidence/results"
CONFIGS = REPO_ROOT / "experiments/configs"

BLOCKED_SUMMARY_PATH = (
    RESULTS / "openalex_g1_main_id_recount_20260716.blocked.summary.json"
)
GRAMMAR_SUMMARY_PATH = (
    RESULTS / "openalex_g1_grammar_valid_main_id_recount_20260716.summary.json"
)
SENTINEL_SUMMARY_PATH = RESULTS / "openalex_g1_sentinel_incidence_20260716.summary.json"
MEMBERSHIP_SUMMARY_PATH = (
    RESULTS / "openalex_g1_work_source_membership_20260717.summary.json"
)
MEMBERSHIP_MARKDOWN_PATH = (
    RESULTS / "openalex_g1_work_source_membership_20260717.summary.md"
)
MEMBERSHIP_VERIFICATION_PATH = (
    RESULTS / "openalex_g1_work_source_membership_20260717.verification.json"
)
QUARANTINE_CONFIG_PATH = CONFIGS / "openalex_g1_recount_canonical_quarantine_v1.yaml"
SENTINEL_CONFIG_PATH = CONFIGS / "openalex_g1_sentinel_incidence_v0.yaml"
MEMBERSHIP_CONFIG_PATH = CONFIGS / "openalex_g1_work_source_membership_v0.yaml"
BASE_RECOUNT_TOOL_PATH = REPO_ROOT / "scripts/recount_openalex_g1.py"
SENTINEL_TOOL_PATH = REPO_ROOT / "scripts/audit_openalex_sentinel_incidence.py"
MEMBERSHIP_TOOL_PATH = REPO_ROOT / "scripts/audit_openalex_work_source_membership.py"
AMENDMENT_PATH = (
    REPO_ROOT
    / "docs/case-studies/researcher-pivot/openalex-main-id-quarantine-amendment.md"
)

EXPECTED_SENTINEL_RESULT_FINGERPRINT = (
    "9a5447ac8ec6cbc2b679da104a7ecf67ff9835c4a71e73923602eda5b288a673"
)
EXPECTED_MEMBERSHIP_RESULT_FINGERPRINT = (
    "8b7d0273a04eb7a231a9d8b466151c562ac80eac6557ff7444508cb174862242"
)
EXPECTED_MEMBERSHIP_OUTPUT_MANIFEST_SHA256 = (
    "8fb34f92e3525b58c09a31ae970d913ea6858211a7595b165884fa389863156d"
)
EXPECTED_MEMBERSHIP_VERIFICATION_SHA256 = (
    "591b8fbb32698bd0d9dd8e167527e2fd75996fe0d9bdcff048ac074ec0db6d66"
)
EXPECTED_WARNINGS = {
    "WARN_ZERO_ID_UNRESOLVED",
    "WARN_PARENT_CONTENT_SHA256_PENDING",
}


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plain_markdown(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _labelled_backtick(markdown: str, label: str) -> str:
    match = re.search(rf"- {re.escape(label)}:[^`]*`([^`]+)`", markdown)
    assert match is not None, f"missing Markdown commitment label: {label}"
    return match.group(1)


def _format_count(value: object) -> str:
    assert type(value) is int
    return f"{value:,}"


def _format_percent(value: object, *, places: int) -> str:
    assert type(value) is float
    return f"{value * 100:.{places}f}%"


def _small_number_word(value: object) -> str:
    words = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    )
    assert type(value) is int
    assert 0 <= value < len(words)
    return words[value]


def _markdown_table_block(markdown: str, header: str) -> str:
    lines = markdown.splitlines()
    start = lines.index(header)
    assert start + 2 < len(lines)
    assert re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", lines[start + 1])
    end = start + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return "\n".join(lines[start:end])


def _two_column_markdown_table(markdown: str, header: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in _markdown_table_block(markdown, header).splitlines()[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 2
        key, value = cells
        assert key not in rows, f"duplicate Markdown table row: {key}"
        rows[key] = value
    return rows


def test_tracked_json_evidence_parses_and_preserves_the_raw_failure() -> None:
    blocked = _json(BLOCKED_SUMMARY_PATH)
    grammar = _json(GRAMMAR_SUMMARY_PATH)
    sentinel = _json(SENTINEL_SUMMARY_PATH)
    membership = _json(MEMBERSHIP_SUMMARY_PATH)

    assert blocked["status"] == "BLOCKED"
    assert blocked["gate_status"]["pass_main_id_recount"] is False
    assert blocked["gate_status"]["pass_g1"] is False
    assert blocked["counts"]["invalid_ids"] == 1
    assert blocked["counts"]["distinct_ids"] is None
    assert blocked["counts"]["duplicate_row_excess"] is None

    gate = grammar["gate_status"]
    assert gate["raw_source_id_validity"] == "BLOCKED"
    assert gate["source_key_integrity_clean"] is False
    assert gate["pass_grammar_valid_main_id_recount"] is True
    assert gate["g_minus_1_data_readiness"] == "PENDING"
    assert gate["g1_empirical_signature"] == "NOT_RUN"
    assert gate["pass_g1"] is False

    assert sentinel["status"] == "COMPLETE_TARGETED_INCIDENCE_AUDIT"
    assert sentinel["pass_g1"] is False
    assert sentinel["scope"]["row_conservation_claimed"] is False
    assert sentinel["scope"]["all_non_sentinel_identifiers_validated"] is False

    membership_gate = membership["gate_status"]
    assert membership_gate["work_source_membership_component"] == "PASS"
    assert membership_gate["raw_source_id_validity"] == "BLOCKED"
    assert membership_gate["w_zero_entity_validity"] == "UNRESOLVED"
    assert membership_gate["parent_content_identity"] == "UNVERIFIED"
    assert membership_gate["analysis_view_materialized"] is False
    assert membership_gate["g_minus_1_data_readiness"] == "PENDING"
    assert membership_gate["empirical_g1_signature"] == "NOT_RUN"
    assert membership_gate["pass_g1"] is False


def test_grammar_valid_recount_has_exact_conservation_and_warning_boundary() -> None:
    summary = _json(GRAMMAR_SUMMARY_PATH)
    counts = summary["exact_counts"]

    assert counts["body_rows_scanned"] == 492_293_207
    assert counts["body_rows_scanned"] == (
        counts["grammar_valid_rows"]
        + counts["quarantined_malformed_rows"]
        + counts["null_rows"]
        + counts["unexpected_malformed_rows"]
    )
    assert counts["grammar_valid_rows"] == 492_293_206
    assert counts["grammar_valid_rows"] == counts["grammar_valid_distinct_ids"]
    assert counts["grammar_valid_duplicate_row_excess"] == 0
    assert counts["grammar_valid_duplicate_id_groups"] == 0
    assert counts["grammar_valid_max_multiplicity"] == 1
    assert counts["zero_id_rows_retained"] == 1

    quarantine = summary["quarantine"]
    assert quarantine["value"] == "https://openalex.org/W-1"
    assert quarantine["count"] == 1
    assert quarantine["source_was_modified"] is False
    assert quarantine["numeric_sentinel_used"] is False

    zero = summary["zero_id_diagnostic"]
    assert zero["value"] == "https://openalex.org/W0"
    assert zero["count"] == 1
    assert zero["retained_in_grammar_valid_spool"] is True
    assert zero["entity_validity"] == "UNRESOLVED"
    assert zero["sensitivity_status"] == "PENDING"

    assert set(summary["warnings"]) == EXPECTED_WARNINGS
    assert summary["check_summary"] == {"passes": 13, "blocks": 0, "warnings": 2}
    assert summary["gate_status"]["parent_content_identity_verified"] is False
    assert summary["gate_status"]["pass_g1"] is False


def test_sentinel_totals_equal_nonzero_table_sums_and_targets_remain_zero() -> None:
    sentinel = _json(SENTINEL_SUMMARY_PATH)
    grammar = _json(GRAMMAR_SUMMARY_PATH)
    source = sentinel["source_id_totals"]
    tables = sentinel["nonzero_source_tables"]
    targets = sentinel["work_target_totals"]

    assert len(tables) == 6
    assert source["w_minus_1"] == {
        "value": "https://openalex.org/W-1",
        "exact_row_count": 21,
        "affected_file_count": 6,
    }
    assert source["w_zero"] == {
        "value": "https://openalex.org/W0",
        "exact_row_count": 6,
        "affected_file_count": 6,
    }
    assert sum(table["w_minus_1_rows"] for table in tables) == 21
    assert sum(table["w_zero_rows"] for table in tables) == 6
    assert targets["w_minus_1_rows"] == 0
    assert targets["w_zero_rows"] == 0

    fingerprint = sentinel["provenance"]["result_fingerprint_sha256"]
    assert fingerprint == EXPECTED_SENTINEL_RESULT_FINGERPRINT
    assert (
        sentinel["provenance"][
            "result_fingerprint_reproduced_by_independent_full_rerun"
        ]
        is True
    )
    assert (
        grammar["sentinel_relation_incidence"]["result_fingerprint_sha256"]
        == fingerprint
    )
    assert grammar["sentinel_relation_incidence"]["w_minus_1_source_rows"] == 21
    assert grammar["sentinel_relation_incidence"]["w_zero_source_rows"] == 6
    assert (
        grammar["sentinel_relation_incidence"]["w_minus_1_declared_work_target_rows"]
        == 0
    )
    assert (
        grammar["sentinel_relation_incidence"]["w_zero_declared_work_target_rows"] == 0
    )


def test_membership_summary_conserves_rows_and_bounds_target_claims() -> None:
    summary = _json(MEMBERSHIP_SUMMARY_PATH)
    accounting = summary["row_accounting"]
    global_counts = accounting["global"]
    main_counts = accounting["main"]
    child_counts = accounting["children"]

    assert global_counts == {
        "raw_rows": 20_996_232_817,
        "retained_rows": 20_996_232_796,
        "frozen_w_minus_1_quarantine_rows": 21,
        "w_zero_rows_retained": 6,
        "grammar_valid_orphan_rows": 0,
        "null_source_id_rows": 0,
        "unexpected_malformed_source_id_rows": 0,
    }
    assert main_counts == {
        "raw_rows": 492_293_207,
        "retained_rows": 492_293_206,
        "frozen_w_minus_1_quarantine_rows": 1,
        "w_zero_rows_retained": 1,
        "grammar_valid_orphan_rows": 0,
        "null_source_id_rows": 0,
        "unexpected_malformed_source_id_rows": 0,
    }
    assert child_counts == {
        "tables": 22,
        "raw_rows": 20_503_939_610,
        "retained_rows": 20_503_939_590,
        "frozen_w_minus_1_quarantine_rows": 20,
        "w_zero_rows_retained": 5,
        "grammar_valid_orphan_rows": 0,
        "null_source_id_rows": 0,
        "unexpected_malformed_source_id_rows": 0,
    }
    assert summary["sentinel_accounting"] == {
        "w_minus_1": {
            "source_rows": 21,
            "source_files": 6,
            "disposition": "FROZEN_QUARANTINE",
        },
        "w_zero": {
            "source_rows": 6,
            "source_files": 6,
            "disposition": "RETAIN_WARN_ENTITY_VALIDITY_UNRESOLVED",
        },
        "declared_work_target_rows_for_either_sentinel": 0,
        "locator_ledgers_exact": True,
    }
    assert global_counts["raw_rows"] == (
        global_counts["retained_rows"]
        + global_counts["frozen_w_minus_1_quarantine_rows"]
    )
    assert main_counts["raw_rows"] == (
        main_counts["retained_rows"]
        + main_counts["frozen_w_minus_1_quarantine_rows"]
    )
    assert child_counts["raw_rows"] == (
        child_counts["retained_rows"]
        + child_counts["frozen_w_minus_1_quarantine_rows"]
    )
    assert global_counts["retained_rows"] == (
        main_counts["retained_rows"] + child_counts["retained_rows"]
    )
    assert global_counts["frozen_w_minus_1_quarantine_rows"] == (
        main_counts["frozen_w_minus_1_quarantine_rows"]
        + child_counts["frozen_w_minus_1_quarantine_rows"]
    )
    assert global_counts["grammar_valid_orphan_rows"] == 0
    assert global_counts["null_source_id_rows"] == 0
    assert global_counts["unexpected_malformed_source_id_rows"] == 0
    assert accounting["conservation_pass"] is True

    referenced = summary["target_coverage"]["referenced_works"]
    related = summary["target_coverage"]["related_works"]
    assert referenced == {
        "scanned_rows": 3_042_536_160,
        "in_snapshot_rows": 2_904_520_313,
        "snapshot_external_rows": 138_015_847,
        "row_coverage": 0.9546378942625287,
        "unique_valid_targets": 135_996_024,
        "unique_targets_in_snapshot": 112_406_279,
        "unique_targets_external_to_snapshot": 23_589_745,
        "unique_coverage": 0.8265409215198821,
        "self_loop_rows": 1_072_993,
        "null_rows": 0,
        "malformed_rows": 0,
        "w_minus_1_rows": 0,
        "w_zero_rows": 0,
        "duplicate_source_target_edges": "NOT_MEASURED_EXTENSION_PENDING",
    }
    assert related == {
        "scanned_rows": 2_493_161_419,
        "in_snapshot_rows": 2_366_002_635,
        "snapshot_external_rows": 127_158_784,
        "row_coverage": 0.9489969710621452,
        "unique_valid_targets": 100_065_189,
        "unique_targets_in_snapshot": 90_310_458,
        "unique_targets_external_to_snapshot": 9_754_731,
        "unique_coverage": 0.902516238689161,
        "self_loop_rows": 4_663_862,
        "null_rows": 0,
        "malformed_rows": 0,
        "w_minus_1_rows": 0,
        "w_zero_rows": 0,
        "duplicate_source_target_edges": "NOT_MEASURED_EXTENSION_PENDING",
    }
    for target in (referenced, related):
        assert target["scanned_rows"] == (
            target["in_snapshot_rows"] + target["snapshot_external_rows"]
        )
        assert target["unique_valid_targets"] == (
            target["unique_targets_in_snapshot"]
            + target["unique_targets_external_to_snapshot"]
        )
        assert target["null_rows"] == 0
        assert target["malformed_rows"] == 0
        assert target["w_minus_1_rows"] == 0
        assert target["w_zero_rows"] == 0
        assert (
            target["duplicate_source_target_edges"]
            == "NOT_MEASURED_EXTENSION_PENDING"
        )

    assert summary["target_coverage"]["interpretation"] == (
        "OPEN_WORLD_COVERAGE_NOT_SOURCE_ORPHANHOOD"
    )


def test_membership_summary_author_profile_is_descriptive_only() -> None:
    summary = _json(MEMBERSHIP_SUMMARY_PATH)
    profile = summary["author_profile"]
    author_rows = profile["author_id_rows"]
    work_profile = profile["work_profile"]

    assert profile["retained_authorship_rows"] == 1_361_551_037
    assert author_rows == {
        "grammar_valid_non_null": 781_434_618,
        "null": 580_116_419,
        "invalid_non_null": 0,
        "grammar_valid_non_null_rate": 0.5739297292312958,
        "null_rate": 0.4260702707687042,
    }
    assert profile["retained_authorship_rows"] == (
        author_rows["grammar_valid_non_null"]
        + author_rows["null"]
        + author_rows["invalid_non_null"]
    )
    assert author_rows["grammar_valid_non_null_rate"] == (
        author_rows["grammar_valid_non_null"] / profile["retained_authorship_rows"]
    )
    assert author_rows["null_rate"] == (
        author_rows["null"] / profile["retained_authorship_rows"]
    )
    assert profile["distinct_grammar_valid_non_null_author_ids"] == 113_748_578
    assert profile["orcid_rows"] == {
        "non_null": 458_308_219,
        "null": 903_242_818,
        "non_null_rate": 0.3366074473490339,
    }
    assert work_profile == {
        "master_work_rows": 492_293_206,
        "complete": 263_750_998,
        "partial": 2_109_042,
        "none_usable": 177_217_069,
        "zero_authorship_row": 49_216_097,
    }
    assert work_profile["master_work_rows"] == sum(
        work_profile[key]
        for key in ("complete", "partial", "none_usable", "zero_authorship_row")
    )
    assert profile["declared_authors_count"]["matches_observed_slots"] == (
        work_profile["master_work_rows"]
    )
    assert profile["declared_authors_count"]["mismatches_observed_slots"] == 0
    assert profile["strata_dimensions"] == {
        "publication_year_buckets": 1_050,
        "primary_field_buckets": 27,
        "declared_author_count_bands": 8,
        "observed_team_size_bands": 7,
        "authorship_source_lineage_masks": 5,
        "joint_sparse_cube_cells": 122_472,
    }
    assert profile["author_entity_validity_established"] is False
    assert profile["candidate_eligibility_stratum"] == "NOT_EVALUABLE_G0_PENDING"


def test_membership_summary_commits_the_bounded_full_pass_artifacts() -> None:
    summary = _json(MEMBERSHIP_SUMMARY_PATH)
    grammar = _json(GRAMMAR_SUMMARY_PATH)
    config = _yaml(MEMBERSHIP_CONFIG_PATH)
    provenance = summary["provenance"]
    source = summary["source"]
    commitments = summary["local_artifact_commitments"]

    assert summary["schema_version"] == (
        "openalex_g1_work_source_membership_summary.v1"
    )
    assert summary["status"] == (
        "PASS_WORK_SOURCE_MEMBERSHIP_AND_PROFILE_AUDIT_WITH_FROZEN_QUARANTINE"
    )
    assert summary["scope"]["full_scope"] is True
    assert summary["scope"]["files_scanned"] == 21_019
    assert summary["scope"]["data_page_rows_scanned"] == 20_996_232_817
    assert all(
        summary["contract_checks"][key] is True
        for key in (
            "required_component_states_match",
            "global_source_accounting_pass",
            "target_coverage_contract_pass",
            "author_profile_contract_pass",
            "blocking_identifier_ledger_contract_pass",
        )
    )

    assert provenance["test_sha256"] == _sha256(
        REPO_ROOT / "tests/test_openalex_work_source_membership.py"
    )
    assert provenance["canonical_plan_sha256"] == (
        "7afef25a788a938b9eeb12a1802d346eb02da1d3b369a116c784851e984891cb"
    )
    assert provenance["raw_plan_file_sha256"] == (
        "51abd0e42c407d61b7296f97d432e64d0e3958d61a976555150cd2911a98f950"
    )
    assert provenance["task_fingerprint_sha256"] == (
        "30bae6b5eff2c545cc64ca2384dcc63e4f28a9237122987efc68724c37ade6c7"
    )
    assert provenance["result_fingerprint_sha256"] == (
        EXPECTED_MEMBERSHIP_RESULT_FINGERPRINT
    )
    assert provenance["output_manifest_sha256"] == (
        EXPECTED_MEMBERSHIP_OUTPUT_MANIFEST_SHA256
    )
    assert provenance["verification_receipt_sha256"] == (
        EXPECTED_MEMBERSHIP_VERIFICATION_SHA256
    )
    assert provenance["verification_receipt_relative_path"] == (
        MEMBERSHIP_VERIFICATION_PATH.relative_to(REPO_ROOT).as_posix()
    )
    assert provenance["verification_receipt_sha256"] == _sha256(
        MEMBERSHIP_VERIFICATION_PATH
    )
    assert provenance["audit_git_commit"] == (
        "10dd3e8fd1c7cc1f30088fcdbef510439e172917"
    )
    assert provenance["audit_git_dirty"] is True
    assert provenance["raw_plan_file_sha256"] == commitments["audit_plan"][
        "sha256"
    ]
    assert provenance["output_manifest_sha256"] == commitments[
        "outputs_manifest"
    ]["sha256"]
    assert commitments["report_json"] == {
        "bytes": 22_993_263,
        "sha256": "1c385d42595fc34794116189ce940526e1095ccd2a5da9eccd3da902e8f152ce",
    }
    assert commitments["report_markdown"] == {
        "bytes": 4_514,
        "sha256": "34b658b8fcfd601ec133f7af20d262d5c153abb36c5a33336da0b221dfc88275",
    }
    assert source["parent_manifest_sha256"] == config["parent_inventory"][
        "manifest_sha256"
    ]
    assert source["sorted_master_ids"]["sha256"] == config["master_work_id_set"][
        "sha256"
    ]
    assert source["sorted_master_ids"]["sha256"] == grammar[
        "local_artifact_commitments"
    ]["grammar_valid_sorted_ids"]["sha256"]

    checkpoint_tree = commitments["checkpoint_tree"]
    assert checkpoint_tree == {
        "files": 21_019,
        "bytes": 51_417_147,
        "path_set_sha256": (
            "57f15a75c1e7f32fb34785206b119c4db0313a642e70450cd7369b5eea3312b5"
        ),
        "tree_sha256": (
            "1c311833193a7073a5ca78e6cf86be702f38e69bb5ef2e9388f814eddab9f01c"
        ),
    }
    artifact_tree = commitments["data_artifact_tree"]
    assert artifact_tree == {
        "files": 8_757,
        "bytes": 89_957_407_904,
        "path_set_sha256": (
            "68c28fb31717b53ea48edd5f93689a4ccd2febd0531a8c37c4b11cafa0d7d376"
        ),
        "tree_sha256": (
            "bb0c711f8cd6b678d6ced92d10ff5530cf25d37c3f3126dc25b96ea986a61ba9"
        ),
    }

    resume = summary["reproducibility"]["clean_semantic_resume"]
    assert resume["status"] == "PASS_CLEAN_SEMANTIC_RESUME"
    assert resume["pass"] is True
    assert resume["exit_code"] == 0
    assert resume["projected_scans_replayed"] == 21_019
    assert resume["existing_checkpoints_semantically_replayed_from_source"] is True
    assert resume["replay_status"] == summary["status"]
    assert resume["replay_result_fingerprint_sha256"] == provenance[
        "result_fingerprint_sha256"
    ]
    assert resume["replay_output_manifest_sha256"] == provenance[
        "output_manifest_sha256"
    ]
    assert resume["same_status"] is True
    assert resume["same_result_fingerprint"] is True
    assert resume["same_output_manifest_sha256"] is True

    verification = summary["independent_verification"]
    assert verification == {
        "p0_findings": 0,
        "p1_findings": 0,
        "production_output_validator_pass": True,
        "independent_path_sets_and_tree_hashes_exact": True,
        "missing_paths": 0,
        "extra_paths": 0,
        "symlinks": 0,
        "non_regular_files": 0,
        "empty_directories": 0,
        "checkpoint_files_revalidated": 21_019,
        "checkpoint_self_hash_mismatches": 0,
        "checkpoint_plan_binding_mismatches": 0,
        "checkpoint_manifest_binding_mismatches": 0,
        "checkpoint_live_file_binding_mismatches": 0,
        "sentinel_locators_physically_reread": 27,
        "sentinel_locator_source_files": 12,
        "sentinel_locator_or_value_digest_mismatches": 0,
        "quantitative_reconstruction_exact": True,
    }


def test_membership_verification_receipt_binds_replay_and_independent_audits() -> None:
    summary = _json(MEMBERSHIP_SUMMARY_PATH)
    receipt = _json(MEMBERSHIP_VERIFICATION_PATH)
    subject = receipt["subject"]
    replay = receipt["clean_semantic_resume"]
    output_audit = receipt["independent_read_only_verification"][
        "output_commit_audit"
    ]
    quantitative = receipt["independent_read_only_verification"][
        "quantitative_audit"
    ]

    assert receipt["schema_version"] == (
        "openalex_g1_work_source_membership_verification.v1"
    )
    assert subject["status"] == summary["status"]
    assert subject["result_fingerprint_sha256"] == summary["provenance"][
        "result_fingerprint_sha256"
    ]
    assert subject["output_manifest_sha256"] == summary["provenance"][
        "output_manifest_sha256"
    ]
    assert subject["report_json_sha256"] == summary["local_artifact_commitments"][
        "report_json"
    ]["sha256"]
    assert subject["report_markdown_sha256"] == summary[
        "local_artifact_commitments"
    ]["report_markdown"]["sha256"]

    assert replay["exit_code"] == 0
    assert replay["projected_scans_replayed"] == 21_019
    assert replay["final_stdout_payload"]["status"] == subject["status"]
    assert replay["final_stdout_payload"]["result_fingerprint_sha256"] == subject[
        "result_fingerprint_sha256"
    ]
    assert replay["final_stdout_payload"]["output_manifest_sha256"] == subject[
        "output_manifest_sha256"
    ]
    canonical_stdout = json.dumps(
        replay["final_stdout_payload"],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert hashlib.sha256(canonical_stdout).hexdigest() == (
        replay["final_stdout_payload_canonical_sha256"]
    )
    assert replay["full_terminal_transcript_preserved"] is False
    assert "not a byte-for-byte terminal progress transcript" in replay[
        "transcript_limitation"
    ]

    assert output_audit["production_output_validator_pass"] is True
    assert output_audit["missing_paths"] == 0
    assert output_audit["extra_paths"] == 0
    assert output_audit["p0_findings"] == 0
    assert output_audit["p1_findings"] == 0
    assert quantitative["checkpoint_files_revalidated"] == 21_019
    assert quantitative["checkpoint_binding_mismatches"] == 0
    assert quantitative["sentinel_locators_physically_reread"] == 27
    assert quantitative["artifact_files_rehashed"] == 8_757
    assert quantitative["artifact_hash_mismatches"] == 0
    assert quantitative["p0_findings"] == 0
    assert quantitative["p1_findings"] == 0


def test_membership_markdown_matches_compact_json_commitments() -> None:
    summary = _json(MEMBERSHIP_SUMMARY_PATH)
    markdown = _plain_markdown(MEMBERSHIP_MARKDOWN_PATH)
    provenance = summary["provenance"]
    source = summary["source"]
    commitments = summary["local_artifact_commitments"]
    accounting = summary["row_accounting"]
    target_coverage = summary["target_coverage"]
    author_profile = summary["author_profile"]

    assert _labelled_backtick(markdown, "Component status") == summary["status"]
    assert _labelled_backtick(markdown, "Result fingerprint") == provenance[
        "result_fingerprint_sha256"
    ]
    assert _labelled_backtick(markdown, "Output-manifest SHA-256") == provenance[
        "output_manifest_sha256"
    ]
    assert _labelled_backtick(
        markdown, "Verification-receipt SHA-256"
    ) == provenance["verification_receipt_sha256"]
    assert _labelled_backtick(markdown, "Canonical plan SHA-256") == provenance[
        "canonical_plan_sha256"
    ]
    assert _labelled_backtick(markdown, "Parent manifest SHA-256") == source[
        "parent_manifest_sha256"
    ]
    assert _labelled_backtick(markdown, "Tool SHA-256") == provenance["tool_sha256"]
    assert _labelled_backtick(markdown, "Audit-time Git provenance") == provenance[
        "audit_git_commit"
    ]
    assert _labelled_backtick(markdown, "Checkpoint tree") == commitments[
        "checkpoint_tree"
    ]["tree_sha256"]
    assert _labelled_backtick(markdown, "Data-artifact tree") == commitments[
        "data_artifact_tree"
    ]["tree_sha256"]

    report_match = re.search(
        r"- Operator-local full audit report JSON / Markdown SHA-256 "
        r"\(not these compact summaries\):\s+`([0-9a-f]{64})` /\s+"
        r"`([0-9a-f]{64})`",
        markdown,
    )
    assert report_match is not None
    assert report_match.groups() == (
        commitments["report_json"]["sha256"],
        commitments["report_markdown"]["sha256"],
    )

    for boundary in (
        "Raw source ID validity remains `BLOCKED`",
        "G-1 data readiness is `PENDING`",
        "empirical G1 is `NOT_RUN` (`pass_g1=false`)",
        "Duplicate source-target edges were not measured",
        "does not establish G-1, empirical G1, or any scientific effect",
    ):
        assert boundary in markdown

    assert "clean semantic-resume replay passed with exit code 0" in markdown
    assert "semantically replaying all 21,019 existing checkpoints" in markdown
    assert "still running" not in markdown
    assert "`dirty=true`" in markdown
    assert "byte-for-byte terminal progress transcript was not retained" in markdown

    def accounting_row(label: str, counts: dict[str, object]) -> str:
        malformed = " / ".join(
            _format_count(counts[key])
            for key in (
                "grammar_valid_orphan_rows",
                "null_source_id_rows",
                "unexpected_malformed_source_id_rows",
            )
        )
        return (
            f"| {label} | {_format_count(counts['raw_rows'])} | "
            f"{_format_count(counts['retained_rows'])} | "
            f"{_format_count(counts['frozen_w_minus_1_quarantine_rows'])} | "
            f"{_format_count(counts['w_zero_rows_retained'])} | {malformed} |"
        )

    for label, counts in (
        (f"All {summary['scope']['source_tables']} tables", accounting["global"]),
        ("Main works", accounting["main"]),
        (f"{accounting['children']['tables']} child tables", accounting["children"]),
    ):
        assert accounting_row(label, counts) in markdown

    def target_row(label: str, target: dict[str, object]) -> str:
        return (
            f"| {label} | {_format_count(target['in_snapshot_rows'])} | "
            f"{_format_count(target['snapshot_external_rows'])} | "
            f"{_format_percent(target['row_coverage'], places=6)} | "
            f"{_format_count(target['unique_targets_in_snapshot'])} | "
            f"{_format_count(target['unique_targets_external_to_snapshot'])} | "
            f"{_format_percent(target['unique_coverage'], places=6)} | "
            f"{_format_count(target['self_loop_rows'])} |"
        )

    assert target_row("Referenced works", target_coverage["referenced_works"]) in markdown
    assert target_row("Related works", target_coverage["related_works"]) in markdown

    author_rows = author_profile["author_id_rows"]
    assert (
        f"The retained authorship view contains "
        f"{_format_count(author_profile['retained_authorship_rows'])} rows: "
        f"{_format_count(author_rows['grammar_valid_non_null'])} grammar-valid "
        f"non-null author IDs "
        f"({_format_percent(author_rows['grammar_valid_non_null_rate'], places=3)}), "
        f"{_format_count(author_rows['null'])} null author IDs "
        f"({_format_percent(author_rows['null_rate'], places=3)}), and "
        f"{_small_number_word(author_rows['invalid_non_null'])} malformed non-null "
        "author IDs."
    ) in markdown
    assert (
        f"It contains "
        f"{_format_count(author_profile['distinct_grammar_valid_non_null_author_ids'])} "
        "distinct grammar-valid author IDs."
    ) in markdown
    orcid = author_profile["orcid_rows"]
    assert (
        f"ORCID is non-null in {_format_count(orcid['non_null'])} rows "
        f"({_format_percent(orcid['non_null_rate'], places=3)})."
    ) in markdown

    work_profile = author_profile["work_profile"]
    assert (
        f"Across {_format_count(work_profile['master_work_rows'])} retained works, "
        "the descriptive classes are:"
    ) in markdown
    for key, suffix in (
        ("complete", ";"),
        ("partial", ";"),
        ("none_usable", "; and"),
        ("zero_authorship_row", "."),
    ):
        assert f"- `{key}`: {_format_count(work_profile[key])}{suffix}" in markdown

    strata = author_profile["strata_dimensions"]
    assert (
        f"closes the margins for "
        f"{_format_count(strata['publication_year_buckets'])} publication-year "
        f"buckets, {_format_count(strata['primary_field_buckets'])} primary-field "
        f"buckets, {_small_number_word(strata['declared_author_count_bands'])} "
        f"declared-author bands, "
        f"{_small_number_word(strata['observed_team_size_bands'])} "
        f"observed-team-size bands, "
        f"{_small_number_word(strata['authorship_source_lineage_masks'])} "
        f"authorship-lineage masks, and "
        f"{_format_count(strata['joint_sparse_cube_cells'])} sparse joint cells."
    ) in markdown

    for label, tree in (
        ("Checkpoint tree", commitments["checkpoint_tree"]),
        ("Data-artifact tree", commitments["data_artifact_tree"]),
    ):
        assert (
            f"- {label}: {_format_count(tree['files'])} files, "
            f"{_format_count(tree['bytes'])} bytes, `{tree['tree_sha256']}`."
        ) in markdown


def test_frozen_config_and_compact_summaries_bind_tracked_dependencies() -> None:
    config = _yaml(QUARANTINE_CONFIG_PATH)
    blocked = _json(BLOCKED_SUMMARY_PATH)
    grammar = _json(GRAMMAR_SUMMARY_PATH)
    sentinel = _json(SENTINEL_SUMMARY_PATH)
    membership = _json(MEMBERSHIP_SUMMARY_PATH)

    dependency = config["tool_dependency"]
    parent_evidence = config["parent_evidence"]["blocked_zero_invalid_recount"]
    assert dependency["relative_path"] == "scripts/recount_openalex_g1.py"
    assert dependency["sha256"] == _sha256(BASE_RECOUNT_TOOL_PATH)
    assert (
        parent_evidence["relative_path"]
        == BLOCKED_SUMMARY_PATH.relative_to(REPO_ROOT).as_posix()
    )
    assert parent_evidence["sha256"] == _sha256(BLOCKED_SUMMARY_PATH)
    assert parent_evidence["status"] == "BLOCKED"
    assert (
        parent_evidence["raw_report_sha256"]
        == blocked["provenance"]["raw_report_sha256"]
    )

    provenance = grammar["provenance"]
    assert provenance["config_sha256"] == _sha256(QUARANTINE_CONFIG_PATH)
    assert provenance["base_dependency_sha256"] == dependency["sha256"]
    assert (
        provenance["blocked_zero_invalid_summary_sha256"] == parent_evidence["sha256"]
    )
    assert (
        provenance["blocked_zero_invalid_raw_report_sha256"]
        == parent_evidence["raw_report_sha256"]
    )

    sentinel_provenance = sentinel["provenance"]
    assert sentinel_provenance["config_sha256"] == _sha256(SENTINEL_CONFIG_PATH)
    assert sentinel_provenance["tool_sha256"] == _sha256(SENTINEL_TOOL_PATH)

    membership_provenance = membership["provenance"]
    assert membership_provenance["config_sha256"] == _sha256(
        MEMBERSHIP_CONFIG_PATH
    )
    assert membership_provenance["tool_sha256"] == _sha256(MEMBERSHIP_TOOL_PATH)
    assert (
        membership_provenance["inventory_summary_sha256"]
        == _sha256(RESULTS / "openalex_g1_source_inventory_20260716.summary.json")
    )
    assert (
        membership_provenance["grammar_valid_recount_summary_sha256"]
        == _sha256(GRAMMAR_SUMMARY_PATH)
    )
    assert (
        membership_provenance["sentinel_summary_sha256"]
        == _sha256(SENTINEL_SUMMARY_PATH)
    )
    assert (
        membership_provenance["result_fingerprint_sha256"]
        == EXPECTED_MEMBERSHIP_RESULT_FINGERPRINT
    )


def test_evidence_claim_boundaries_keep_remaining_gates_explicit() -> None:
    config = _yaml(QUARANTINE_CONFIG_PATH)
    blocked = _json(BLOCKED_SUMMARY_PATH)
    grammar = _json(GRAMMAR_SUMMARY_PATH)
    sentinel = _json(SENTINEL_SUMMARY_PATH)
    membership = _json(MEMBERSHIP_SUMMARY_PATH)

    blocked_boundary = blocked["claim_boundary"]
    assert "does not establish PASS_MAIN_ID_RECOUNT" in blocked_boundary
    assert "pass_g1=false means NOT_RUN" in blocked_boundary

    grammar_boundary = grammar["claim_boundary"]
    for phrase in (
        "blocked on raw ID validity",
        "492,293,206-row grammar-valid analysis view",
        "W0 remains in the view and has unresolved entity validity",
        "Parent content SHA-256",
        "relation closure",
        "empirical G1",
        "every scientific effect remain pending",
    ):
        assert phrase in grammar_boundary

    sentinel_boundary = sentinel["claim_boundary"]
    for phrase in (
        "targeted W-1 and W0 incidence only",
        "does not claim row conservation",
        "establish PASS_G1",
        "any scientific effect",
    ):
        assert phrase in sentinel_boundary

    membership_boundary = membership["claim_boundary"]
    for phrase in (
        "23-table source-ID scope",
        "open-world coverage",
        "untouched raw source remains blocked",
        "no filtered analysis view was materialized",
        "duplicate source-target edges",
        "G-1 data readiness",
        "empirical G1",
        "every scientific effect remain unestablished",
    ):
        assert phrase in membership_boundary

    assert "g_minus_1_data_readiness" in membership["non_claims"]
    assert "empirical_g1_execution" in membership["non_claims"]
    assert "any_scientific_effect" in membership["non_claims"]

    config_boundary = config["claim_boundary"]
    assert "does not establish canonical" in config_boundary
    assert "full-root content SHA-256 remains pending" in config_boundary
    assert "PASS_G1" in config_boundary
    assert config["pending_after_grammar_valid_main_id_recount"]


def test_amendment_reports_only_the_bounded_pass_and_links_tracked_evidence() -> None:
    amendment = AMENDMENT_PATH.read_text(encoding="utf-8")

    front_status_match = re.search(
        r"(?ms)^Status:\s*(?P<status>.+?)(?:\n\s*\n)",
        amendment,
    )
    assert front_status_match is not None
    status_table_block = _markdown_table_block(amendment, "| Layer | Status |")
    status_bearing_content = "\n".join(
        (front_status_match.group("status"), status_table_block)
    )
    status_assignments = re.finditer(
        r"(?i)\b(?P<subject>raw (?:source|main-ID) validity|"
        r"G-1 data readiness|empirical G1(?: signature)?)\b\s*"
        r"(?:remains|is|=|\|)\s*`?(?P<status>[A-Z][A-Z0-9_-]*)",
        status_bearing_content,
    )
    expected_status = {
        "raw source validity": "BLOCKED",
        "raw main-id validity": "BLOCKED",
        "g-1 data readiness": "PENDING",
        "empirical g1": "NOT_RUN",
        "empirical g1 signature": "NOT_RUN",
    }
    observed_subjects: set[str] = set()
    for match in status_assignments:
        subject = match.group("subject").casefold()
        observed_subjects.add(subject)
        assert match.group("status").upper() == expected_status[subject]
    assert observed_subjects == set(expected_status)
    assert re.search(
        r"(?i)\b(?:raw (?:source|main-ID)(?: validity)?|"
        r"G-1(?: data readiness)?|empirical G1(?: signature)?)\b\s*"
        r"(?:(?:is|=|:|\|)\s*`?(?:PASS(?:ED)?|VALID|VERIFIED|READY|"
        r"COMPLETE|RUN)\b|(?:has\s+)?(?:passed|passes|ran|completed)\b)",
        status_bearing_content,
    ) is None
    assert re.search(
        r"(?i)\b(?:raw\s+)?source\s+(?:has\s+)?passed\s+validation\b",
        status_bearing_content,
    ) is None

    status_rows = _two_column_markdown_table(amendment, "| Layer | Status |")
    assert status_rows["Raw main-ID validity"] == "`BLOCKED`"
    assert status_rows["G-1 data readiness"] == "`PENDING`"
    assert status_rows["Empirical G1 signature"] == (
        "`NOT_RUN`; `pass_g1=false`"
    )

    for required_boundary in (
        "remains `BLOCKED_RAW_ID_VALIDITY`",
        "`WARN_ZERO_ID_UNRESOLVED`",
        "Full-root content SHA-256 | pending",
        "G-1 data readiness | `PENDING`",
        "Empirical G1 signature | `NOT_RUN`; `pass_g1=false`",
        "Work-source membership and row conservation | passed; audit-only",
        "Materialized filtered analysis view | not created",
        "No manuscript may summarize this amendment",
    ):
        assert required_boundary in amendment

    expected_links = {
        "../../../experiments/evidence/results/"
        "openalex_g1_main_id_recount_20260716.blocked.summary.md",
        "../../../experiments/evidence/results/"
        "openalex_g1_grammar_valid_main_id_recount_20260716.summary.md",
        "../../../experiments/evidence/results/"
        "openalex_g1_sentinel_incidence_20260716.summary.md",
        "../../../experiments/evidence/results/"
        "openalex_g1_work_source_membership_20260717.summary.md",
    }
    for relative_link in expected_links:
        assert f"]({relative_link})" in amendment
        assert (AMENDMENT_PATH.parent / relative_link).resolve().is_file()
