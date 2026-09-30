from __future__ import annotations

import io
import json
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSPECT_SCRIPT = ROOT / "scripts" / "inspect_release_artifacts.py"


def test_release_artifact_inspection_builds_and_reports_package_boundary() -> None:
    completed = subprocess.run(
        [sys.executable, str(INSPECT_SCRIPT), "--build", "--ephemeral", "--json"],
        check=True,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])

    assert payload["status"] == "pass"
    assert payload["blocking_issues"] == []
    assert payload["pyproject"]["name"] == "neural-abm"
    assert payload["pyproject"]["version"] == "0.1.0a6.dev0"
    assert payload["pyproject"]["requires_python"] == ">=3.11"
    assert payload["pyproject"]["default_dependencies"] == ["numpy", "pyyaml"]
    assert payload["pyproject"]["has_authors"] is True
    assert payload["pyproject"]["has_classifiers"] is True
    assert payload["pyproject"]["has_keywords"] is True
    assert payload["pyproject"]["has_license"] is True
    assert payload["pyproject"]["has_urls"] is True
    assert payload["wheel"]["metadata_name"] == "neural-abm"
    assert payload["wheel"]["metadata_version"] == "0.1.0a6.dev0"
    assert payload["wheel"]["runtime_version"] == "0.1.0a6.dev0"
    assert payload["wheel"]["default_requires"] == ["numpy", "pyyaml"]
    assert "neural_abm/scenario_lite.py" in payload["wheel"]["required_modules"]
    assert "neural_abm/workflow_lite.py" in payload["wheel"]["required_modules"]
    assert payload["wheel"]["required_modules_present"] is True
    assert (
        "examples/research_pivot_scenario_lite.py" in payload["sdist"]["required_files"]
    )
    claim_bearing_files = {
        "docs/case-studies/researcher-pivot/README.md",
        "docs/case-studies/researcher-pivot/data/learning_study_results.json",
        "docs/case-studies/researcher-pivot/data/study_results.json",
        "docs/classical-reductions.md",
        "docs/decisions/0015-researcher-scenario-lite-contract.md",
        "docs/figures/nabm_unit_recurrent_block.png",
        "docs/figures/nabm_unit_recurrent_block.svg",
        "examples/classical_reductions.py",
        "examples/research_pivot_learning_study.py",
        "examples/research_pivot_study.py",
        "scripts/inspect_release_artifacts.py",
        "scripts/plot_nabm_unit_schematic.py",
        "scripts/plot_research_pivot_study.py",
    }
    assert claim_bearing_files <= set(payload["sdist"]["required_files"])
    assert payload["sdist"]["required_files_present"] is True
    assert payload["sdist"]["internal_links_checked"] > 0
    assert payload["sdist"]["internal_links_ok"] is True
    assert payload["sdist"]["broken_internal_links"] == []
    assert payload["sdist"]["forbidden_files_present"] is False
    assert payload["sdist"]["forbidden_files"] == []
    assert payload["sdist"]["member_boundary_ok"] is True
    assert payload["sdist"]["member_boundary_issues"] == []
    assert payload["sdist"]["researcher_pivot_boundary_ok"] is True
    assert payload["sdist"]["researcher_pivot_boundary_issues"] == []
    assert not any("project.license is not set" in item for item in payload["warnings"])
    assert not any("project.urls is not set" in item for item in payload["warnings"])

    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_only_paths = {
        "docs/case-studies/researcher-pivot/compute-execution-contract.md",
        "docs/case-studies/researcher-pivot/openalex-canonical-work-time-spine-audit-design.md",
        "docs/case-studies/researcher-pivot/openalex-canonical-work-time-spine-v1-audit-design.md",
        "docs/research/openalex-canonical-work-time-spine-v2-production-design.md",
        "docs/research/openalex-canonical-work-time-spine-capacity-review-v0.md",
        "docs/research/openalex-canonical-work-time-spine-downstream-benchmark-v0.md",
        "docs/research/openalex-canonical-work-time-spine-downstream-stress-v1.md",
        "docs/research/openalex-canonical-work-time-spine-independent-verifier-v1.md",
        "docs/research/openalex-canonical-work-time-spine-ops-readiness-20260720.md",
        "docs/research/openalex-canonical-work-time-spine-pilot-benchmark-tool.md",
        "docs/case-studies/researcher-pivot/openalex-citation-available-at-audit-design.md",
        "docs/case-studies/researcher-pivot/openalex-citation-available-at-v1-audit-design.md",
        "docs/research/openalex-citation-edge-chronology-v3-production.md",
        "docs/case-studies/researcher-pivot/openalex-main-id-quarantine-amendment.md",
        "docs/case-studies/researcher-pivot/openalex-main-temporal-validity-audit.md",
        "docs/case-studies/researcher-pivot/openalex-temporal-author-gate-inventory.md",
        "docs/case-studies/researcher-pivot/openalex-work-side-author-history-audit-design.md",
        "docs/research/peer-exposure-identification-design.md",
        "docs/case-studies/researcher-pivot/pivot-feedback-empirical-nabm-protocol.md",
        "docs/research/openalex-work-side-author-history-v1-audit-design.md",
        "docs/research/openalex-work-side-author-history-v2-synthetic-validator.md",
        "docs/research/openalex-work-side-author-history-v3-production.md",
        "docs/research/openalex-work-topic-relation-pilot-20260721.md",
        "docs/research/pivot-feedback-cutoff-primitives-source-replay-v2.md",
        "docs/research/pivot-feedback-d0a-structural-pilot-freeze-20260724.md",
        "docs/research/pivot-feedback-d0b-author-opportunity-foundation-20260724.md",
        "docs/research/pivot-feedback-d0b-selected-work-author-linkage-audit-20260724.md",
        "docs/research/pivot-feedback-cutoff-primitives-v0.md",
        "docs/research/pivot-feedback-g-minus-1a-and-s0-core-20260723.md",
        "docs/research/pivot-feedback-gate-sequence-v1.md",
        "docs/research/pivot-feedback-post-primitives-critical-review-20260723.md",
        "docs/research/pivot-feedback-researcher-pivot-definition-v1.md",
        "docs/research/pivot-feedback-s0-coauthor-graph-profile-v1.md",
        "docs/research/pivot-feedback-s0-completion-20260724.md",
        "experiments/configs/openalex_g1_cross_field_feasibility_v0.yaml",
        "experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml",
        "experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml",
        "experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml",
        "experiments/configs/openalex_g1_canonical_work_time_spine_capacity_review_v0.yaml",
        "experiments/configs/openalex_g1_citation_available_at_v0.yaml",
        "experiments/configs/openalex_g1_citation_available_at_v1.yaml",
        "experiments/configs/openalex_g1_citation_available_at_v2.yaml",
        "experiments/configs/openalex_g1_content_freeze_v0.yaml",
        "experiments/configs/openalex_g1_field_taxonomy_reference_v0.yaml",
        "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml",
        "experiments/configs/openalex_g1_main_temporal_validity_v1.yaml",
        "experiments/configs/openalex_g1_main_temporal_validity_v2.yaml",
        "experiments/configs/openalex_g1_work_side_author_history_v0.yaml",
        "experiments/configs/openalex_g1_work_side_author_history_v1.yaml",
        "experiments/configs/openalex_g1_work_side_author_history_v2.yaml",
        "experiments/configs/openalex_g1_work_side_author_history_v3.yaml",
        "experiments/configs/openalex_g1_w0_positive_id_sensitivity_v0.yaml",
        "experiments/configs/openalex_g1_work_topic_relation_v0.yaml",
        "experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml",
        "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml",
        "experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml",
        "experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml",
        "experiments/configs/pivot_feedback_cutoff_primitives_v0.yaml",
        "experiments/configs/pivot_feedback_d0_structural_pilot_freeze_v1.yaml",
        "experiments/configs/pivot_feedback_d0b_author_opportunity_foundation_v1.yaml",
        "experiments/configs/pivot_feedback_d0b_selected_work_author_linkage_audit_v1.yaml",
        "experiments/configs/pivot_feedback_g_minus_1a_receipt_v1.yaml",
        "experiments/configs/pivot_feedback_gate_sequence_v1.yaml",
        "experiments/configs/pivot_feedback_gate_state_v2.yaml",
        "experiments/configs/pivot_feedback_gate_state_v3.yaml",
        "experiments/configs/pivot_feedback_gate_state_v4.yaml",
        "experiments/configs/pivot_feedback_gate_state_v5.yaml",
        "experiments/configs/pivot_feedback_gate_state_v6.yaml",
        "experiments/configs/pivot_feedback_s0_authorship_orcid_affiliation_v1.yaml",
        "experiments/configs/pivot_feedback_s0_coauthor_graph_profile_v1.yaml",
        "experiments/configs/pivot_feedback_s0_threshold_free_inventory_v1.yaml",
        "scripts/audit_openalex_field_taxonomy_reference.py",
        "scripts/audit_openalex_canonical_work_time_spine.py",
        "scripts/audit_openalex_canonical_work_time_spine_v2.py",
        "scripts/audit_openalex_canonical_work_time_spine_v3.py",
        "scripts/audit_openalex_citation_available_at.py",
        "scripts/audit_openalex_citation_available_at_v2.py",
        "scripts/audit_openalex_citation_available_at_v3.py",
        "scripts/verify_openalex_citation_available_at_v3.py",
        "scripts/audit_openalex_content_freeze.py",
        "scripts/audit_openalex_main_temporal_validity.py",
        "scripts/audit_openalex_main_temporal_validity_v2.py",
        "scripts/audit_openalex_main_temporal_validity_v3.py",
        "scripts/verify_openalex_work_side_author_history_fixture_v2.py",
        "scripts/audit_openalex_work_side_author_history_v3.py",
        "scripts/verify_openalex_work_side_author_history_v3.py",
        "scripts/audit_openalex_sentinel_incidence.py",
        "scripts/audit_openalex_source_inventory.py",
        "scripts/audit_openalex_work_source_membership.py",
        "scripts/audit_openalex_w0_positive_id_sensitivity.py",
        "scripts/audit_openalex_work_topic_relation.py",
        "scripts/plan_research_pivot_field_matrix.py",
        "scripts/recount_openalex_g1.py",
        "scripts/recount_openalex_g1_canonical_quarantine.py",
        "scripts/benchmark_openalex_canonical_work_time_spine_downstream.py",
        "scripts/benchmark_openalex_canonical_work_time_spine_downstream_stress_v1.py",
        "scripts/review_openalex_canonical_work_time_spine_capacity.py",
        "scripts/summarize_openalex_canonical_work_time_spine_pilot_benchmark.py",
        "scripts/summarize_openalex_cross_field_feasibility.py",
        "scripts/verify_openalex_canonical_work_time_spine_downstream_benchmark_v1.py",
        "scripts/verify_openalex_canonical_work_time_spine_downstream_stress_v1.py",
        "scripts/verify_openalex_canonical_work_time_spine_v1.py",
        "scripts/verify_pivot_feedback_g0_measurement_freeze.py",
        "scripts/verify_pivot_feedback_g_minus_1_readiness.py",
        "scripts/build_pivot_feedback_cutoff_primitives.py",
        "scripts/build_pivot_feedback_d0_structural_pilot_freeze.py",
        "scripts/build_pivot_feedback_d0b_author_opportunity_foundation.py",
        "scripts/build_pivot_feedback_d0b_selected_work_author_linkage_audit.py",
        "scripts/build_pivot_feedback_s0_authorship_orcid_affiliation.py",
        "scripts/build_pivot_feedback_s0_coauthor_graph_profile.py",
        "scripts/build_pivot_feedback_s0_threshold_free_inventory.py",
        "scripts/verify_pivot_feedback_cutoff_primitives_source_replay_v2.py",
        "scripts/verify_pivot_feedback_g_minus_1a_receipt_v1.py",
        "scripts/verify_pivot_feedback_gate_sequence_v1.py",
        "scripts/verify_pivot_feedback_gate_state_v2.py",
        "scripts/verify_pivot_feedback_gate_state_v3.py",
        "scripts/verify_pivot_feedback_gate_state_v4.py",
        "scripts/verify_pivot_feedback_gate_state_v5.py",
        "scripts/verify_pivot_feedback_gate_state_v6.py",
        "scripts/verify_pivot_feedback_d0_structural_pilot_freeze.py",
        "scripts/verify_pivot_feedback_d0b_author_opportunity_foundation.py",
        "scripts/verify_pivot_feedback_d0b_selected_work_author_linkage_audit.py",
        "scripts/verify_pivot_feedback_s0_authorship_orcid_affiliation.py",
        "scripts/verify_pivot_feedback_s0_coauthor_graph_profile.py",
        "scripts/verify_pivot_feedback_s0_completion_v1.py",
        "scripts/verify_pivot_feedback_s0_threshold_free_inventory.py",
        "scripts/verify_pivot_feedback_cutoff_primitives.py",
    }
    assert repository_only_paths <= inspector["FORBIDDEN_SDIST_PATHS"]
    cross_field_paths = {
        path
        for path in repository_only_paths
        if "cross_field" in path or "field_taxonomy" in path or "field_matrix" in path
    }
    assert cross_field_paths == inspector["CROSS_FIELD_REPOSITORY_ONLY_PATHS"]
    assert inspector["PEER_MECHANISM_REPOSITORY_ONLY_PATHS"] == {
        "docs/research/peer-exposure-identification-design.md",
        "experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml",
    }
    assert inspector["PIVOT_NABM_BRIDGE_REPOSITORY_ONLY_PATHS"] == {
        "docs/research/pivot-nabm-case-study-bridge-v1.md",
        "experiments/configs/pivot_nabm_case_study_bridge_v1.yaml",
    }
    assert inspector["PIVOT_GATE_REPOSITORY_ONLY_PATHS"] == {
        "experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml",
        "experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml",
        "scripts/verify_pivot_feedback_g0_measurement_freeze.py",
        "scripts/verify_pivot_feedback_g_minus_1_readiness.py",
    }
    assert inspector["CONTENT_FREEZE_REPOSITORY_ONLY_PATHS"] == {
        "experiments/configs/openalex_g1_content_freeze_v0.yaml",
        "scripts/audit_openalex_content_freeze.py",
    }
    assert inspector["PIVOT_POST_PRIMITIVES_REPOSITORY_ONLY_PATHS"] == {
        "docs/research/pivot-feedback-cutoff-primitives-source-replay-v2.md",
        "docs/research/pivot-feedback-d0-adaptive-candidate-frontier-v1.md",
        "docs/research/pivot-feedback-d0-api-assisted-review-protocol-v2.md",
        "docs/research/pivot-feedback-d0-api-only-topic-routing-v1.md",
        "docs/research/pivot-feedback-d0-candidate-content-dry-run-v1.md",
        "docs/research/pivot-feedback-d0-candidate-screen-scope-audit-v1.md",
        "docs/research/pivot-feedback-d0-content-indexing-audit-20260724.md",
        "docs/research/pivot-feedback-d0-content-review-protocol-v1.md",
        "docs/research/pivot-feedback-d0a-structural-pilot-freeze-20260724.md",
        "docs/research/pivot-feedback-d0b-author-opportunity-foundation-20260724.md",
        "docs/research/pivot-feedback-d0b-selected-work-author-linkage-audit-20260724.md",
        "docs/research/pivot-feedback-g-minus-1a-and-s0-core-20260723.md",
        "docs/research/pivot-feedback-gate-sequence-v1.md",
        "docs/research/pivot-feedback-post-primitives-critical-review-20260723.md",
        "docs/research/pivot-feedback-researcher-pivot-definition-v1.md",
        "docs/research/pivot-feedback-s0-coauthor-graph-profile-v1.md",
        "docs/research/pivot-feedback-s0-completion-20260724.md",
        "experiments/evidence/reference/pivot_feedback_topic_cards_openalex_snapshot_20260729.json",
        "experiments/evidence/results/pivot_feedback_d0_adaptive_candidate_frontier_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_adaptive_candidate_frontier_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_api_only_topic_routing_dry_run_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_api_only_topic_routing_dry_run_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_candidate_content_dry_run_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_candidate_content_dry_run_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_candidate_screen_scope_audit_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_candidate_screen_scope_audit_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_cross_provider_api_review_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_cross_provider_api_review_20260729.summary.md",
        "experiments/configs/pivot_feedback_g_minus_1a_receipt_v1.yaml",
        "experiments/configs/pivot_feedback_d0_adaptive_candidate_frontier_v1.yaml",
        "experiments/configs/pivot_feedback_d0_api_assisted_review_v2.yaml",
        "experiments/configs/pivot_feedback_d0_api_only_topic_routing_v1.yaml",
        "experiments/configs/pivot_feedback_d0_candidate_content_dry_run_v1.yaml",
        "experiments/configs/pivot_feedback_d0_candidate_screen_scope_audit_v1.yaml",
        "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v3.yaml",
        "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v4.yaml",
        "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v5.yaml",
        "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v6.yaml",
        "experiments/configs/pivot_feedback_d0_content_indexing_audit_v1.yaml",
        "experiments/configs/pivot_feedback_d0_content_review_v1.yaml",
        "experiments/configs/pivot_feedback_d0_measurement_abi_v1.yaml",
        "experiments/configs/pivot_feedback_d0_structural_pilot_freeze_v1.yaml",
        "experiments/configs/pivot_feedback_d0b_author_opportunity_foundation_v1.yaml",
        "experiments/configs/pivot_feedback_d0b_selected_work_author_linkage_audit_v1.yaml",
        "experiments/configs/pivot_feedback_gate_sequence_v1.yaml",
        "experiments/configs/pivot_feedback_gate_state_v2.yaml",
        "experiments/configs/pivot_feedback_gate_state_v3.yaml",
        "experiments/configs/pivot_feedback_gate_state_v4.yaml",
        "experiments/configs/pivot_feedback_gate_state_v5.yaml",
        "experiments/configs/pivot_feedback_gate_state_v6.yaml",
        "experiments/configs/pivot_feedback_gate_state_v7.yaml",
        "experiments/configs/pivot_feedback_gate_state_v8.yaml",
        "experiments/configs/pivot_feedback_gate_state_v9.yaml",
        "experiments/configs/pivot_feedback_gate_state_v10.yaml",
        "experiments/configs/pivot_feedback_gate_state_v11.yaml",
        "experiments/configs/pivot_feedback_gate_state_v12.yaml",
        "experiments/configs/pivot_feedback_gate_state_v13.yaml",
        "experiments/configs/pivot_feedback_gate_state_v14.yaml",
        "experiments/configs/pivot_feedback_gate_state_v15.yaml",
        "experiments/configs/pivot_feedback_researcher_pivot_definition_v1.yaml",
        "experiments/configs/pivot_feedback_s0_authorship_orcid_affiliation_v1.yaml",
        "experiments/configs/pivot_feedback_s0_coauthor_graph_profile_v1.yaml",
        "experiments/configs/pivot_feedback_s0_threshold_free_inventory_v1.yaml",
        "scripts/build_pivot_feedback_s0_authorship_orcid_affiliation.py",
        "scripts/build_pivot_feedback_s0_coauthor_graph_profile.py",
        "scripts/build_pivot_feedback_s0_threshold_free_inventory.py",
        "scripts/audit_pivot_feedback_d0_content_indexing.py",
        "scripts/build_pivot_feedback_d0_adaptive_candidate_frontier_v1.py",
        "scripts/build_pivot_feedback_d0_api_only_topic_routing_v1.py",
        "scripts/build_pivot_feedback_d0_candidate_content_dry_run_v1.py",
        "scripts/build_pivot_feedback_d0_candidate_screen_scope_audit_v1.py",
        "scripts/build_pivot_feedback_d0_content_review_packets.py",
        "scripts/build_pivot_feedback_d0_measurement_abi_reselection.py",
        "scripts/build_pivot_feedback_d0_structural_pilot_freeze.py",
        "scripts/build_pivot_feedback_d0b_author_opportunity_foundation.py",
        "scripts/build_pivot_feedback_d0b_selected_work_author_linkage_audit.py",
        "scripts/build_pivot_feedback_d0_cross_provider_results_v6.py",
        "scripts/collect_pivot_feedback_d0_cross_provider_api_review_v4.py",
        "scripts/collect_pivot_feedback_d0_openai_retry_v5.py",
        "scripts/prepare_pivot_feedback_d0_api_review_v2.py",
        "scripts/prepare_pivot_feedback_d0_cross_provider_api_review_v3.py",
        "scripts/prepare_pivot_feedback_d0_cross_provider_api_review_v4.py",
        "scripts/prepare_pivot_feedback_d0_openai_retry_v5.py",
        "scripts/submit_pivot_feedback_d0_cross_provider_api_review_v3.py",
        "scripts/submit_pivot_feedback_d0_cross_provider_api_review_v4.py",
        "scripts/submit_pivot_feedback_d0_openai_retry_v5.py",
        "scripts/verify_pivot_feedback_cutoff_primitives_source_replay_v2.py",
        "scripts/verify_pivot_feedback_g_minus_1a_receipt_v1.py",
        "scripts/verify_pivot_feedback_gate_sequence_v1.py",
        "scripts/verify_pivot_feedback_gate_state_v2.py",
        "scripts/verify_pivot_feedback_gate_state_v3.py",
        "scripts/verify_pivot_feedback_gate_state_v4.py",
        "scripts/verify_pivot_feedback_gate_state_v5.py",
        "scripts/verify_pivot_feedback_gate_state_v6.py",
        "scripts/verify_pivot_feedback_gate_state_v7.py",
        "scripts/verify_pivot_feedback_gate_state_v8.py",
        "scripts/verify_pivot_feedback_gate_state_v9.py",
        "scripts/verify_pivot_feedback_gate_state_v10.py",
        "scripts/verify_pivot_feedback_gate_state_v11.py",
        "scripts/verify_pivot_feedback_gate_state_v12.py",
        "scripts/verify_pivot_feedback_gate_state_v13.py",
        "scripts/verify_pivot_feedback_gate_state_v14.py",
        "scripts/verify_pivot_feedback_gate_state_v15.py",
        "scripts/verify_pivot_feedback_researcher_pivot_definition_v1.py",
        "scripts/verify_pivot_feedback_d0_adaptive_candidate_frontier_v1.py",
        "scripts/verify_pivot_feedback_d0_api_only_topic_routing_v1.py",
        "scripts/verify_pivot_feedback_d0_candidate_content_dry_run_v1.py",
        "scripts/verify_pivot_feedback_d0_candidate_screen_scope_audit_v1.py",
        "scripts/verify_pivot_feedback_d0_api_review_preparation_v2.py",
        "scripts/verify_pivot_feedback_d0_content_review_packets.py",
        "scripts/verify_pivot_feedback_d0_cross_provider_api_review_v3.py",
        "scripts/verify_pivot_feedback_d0_cross_provider_api_review_v4.py",
        "scripts/verify_pivot_feedback_d0_cross_provider_results_v6.py",
        "scripts/verify_pivot_feedback_d0_openai_retry_v5.py",
        "scripts/verify_pivot_feedback_d0_measurement_abi_v1.py",
        "scripts/verify_pivot_feedback_d0_structural_pilot_freeze.py",
        "scripts/verify_pivot_feedback_d0b_author_opportunity_foundation.py",
        "scripts/verify_pivot_feedback_d0b_selected_work_author_linkage_audit.py",
        "scripts/verify_pivot_feedback_s0_authorship_orcid_affiliation.py",
        "scripts/verify_pivot_feedback_s0_coauthor_graph_profile.py",
        "scripts/verify_pivot_feedback_s0_completion_v1.py",
        "scripts/verify_pivot_feedback_s0_threshold_free_inventory.py",
    }
    assert inspector["PIVOT_CANDIDATE_SUCCESSOR_REPOSITORY_ONLY_PATHS"] == {
        "docs/research/pivot-feedback-d0-candidate-rule-v2-replay-v1.md",
        "docs/research/pivot-feedback-d0-production-candidate-rule-v1.md",
        "docs/research/pivot-feedback-d0-production-candidate-rule-v2.md",
        "docs/research/pivot-feedback-d0-rank3-production-disposition-v1.md",
        "docs/research/pivot-feedback-d0-rank4-production-disposition-v1.md",
        "docs/research/pivot-feedback-d0-rank5-terminal-disposition-v1.md",
        "docs/research/pivot-feedback-d0-t10543-local-routing-v1.md",
        "docs/research/pivot-feedback-d0-t10543-rank3-local-routing-v1.md",
        "docs/research/pivot-feedback-d0-t10151-rank5-local-routing-v1.md",
        "docs/research/pivot-feedback-d0-t12505-rank4-local-routing-v1.md",
        "experiments/configs/pivot_feedback_d0_candidate_rule_v2_retrospective_replay_v1.yaml",
        "experiments/configs/pivot_feedback_d0_production_candidate_rule_v1.yaml",
        "experiments/configs/pivot_feedback_d0_production_candidate_rule_v2.yaml",
        "experiments/configs/pivot_feedback_d0_rank3_production_disposition_v1.yaml",
        "experiments/configs/pivot_feedback_d0_rank4_production_disposition_v1.yaml",
        "experiments/configs/pivot_feedback_d0_rank5_terminal_disposition_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_local_routing_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_targeted_title_sample_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10543_local_routing_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10543_rank3_local_routing_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10543_rank3_targeted_title_sample_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10543_targeted_title_sample_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t12505_rank4_local_routing_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t12505_rank4_targeted_title_sample_v1.yaml",
        "experiments/configs/pivot_feedback_gate_state_v16.yaml",
        "experiments/configs/pivot_feedback_gate_state_v17.yaml",
        "experiments/configs/pivot_feedback_gate_state_v18.yaml",
        "experiments/configs/pivot_feedback_gate_state_v19.yaml",
        "experiments/configs/pivot_feedback_gate_state_v20.yaml",
        "experiments/configs/pivot_feedback_gate_state_v21.yaml",
        "experiments/configs/pivot_feedback_gate_state_v22.yaml",
        "experiments/configs/pivot_feedback_gate_state_v23.yaml",
        "experiments/configs/pivot_feedback_gate_state_v24.yaml",
        "experiments/configs/pivot_feedback_gate_state_v25.yaml",
        "experiments/evidence/results/pivot_feedback_d0_candidate_rule_v2_replay_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_candidate_rule_v2_replay_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_production_candidate_rule_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_production_candidate_rule_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_production_candidate_rule_v2_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_production_candidate_rule_v2_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_rank3_production_disposition_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_rank3_production_disposition_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_rank4_production_disposition_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_rank4_production_disposition_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_rank5_terminal_disposition_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_rank5_terminal_disposition_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_t10151_rank5_local_routing_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_t10151_rank5_local_routing_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_t10543_local_routing_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_t10543_local_routing_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_t10543_rank3_local_routing_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_t10543_rank3_local_routing_20260729.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_t12505_rank4_local_routing_20260729.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_t12505_rank4_local_routing_20260729.summary.md",
        "scripts/build_pivot_feedback_d0_candidate_rule_v2_replay_v1.py",
        "scripts/build_pivot_feedback_d0_production_candidate_rule_v1.py",
        "scripts/build_pivot_feedback_d0_production_candidate_rule_v2.py",
        "scripts/build_pivot_feedback_d0_rank3_production_disposition_v1.py",
        "scripts/build_pivot_feedback_d0_rank4_production_disposition_v1.py",
        "scripts/build_pivot_feedback_d0_rank5_terminal_disposition_v1.py",
        "scripts/build_pivot_feedback_d0_t10151_rank5_local_routing_v1.py",
        "scripts/build_pivot_feedback_d0_t10151_rank5_targeted_title_sample_v1.py",
        "scripts/build_pivot_feedback_d0_t10543_local_routing_v1.py",
        "scripts/build_pivot_feedback_d0_t10543_rank3_local_routing_v1.py",
        "scripts/build_pivot_feedback_d0_t10543_rank3_targeted_title_sample_v1.py",
        "scripts/build_pivot_feedback_d0_t10543_targeted_title_sample_v1.py",
        "scripts/build_pivot_feedback_d0_t12505_rank4_local_routing_v1.py",
        "scripts/build_pivot_feedback_d0_t12505_rank4_targeted_title_sample_v1.py",
        "scripts/verify_pivot_feedback_d0_candidate_rule_v2_replay_v1.py",
        "scripts/verify_pivot_feedback_d0_production_candidate_rule_v1.py",
        "scripts/verify_pivot_feedback_d0_production_candidate_rule_v2.py",
        "scripts/verify_pivot_feedback_d0_rank3_production_disposition_v1.py",
        "scripts/verify_pivot_feedback_d0_rank4_production_disposition_v1.py",
        "scripts/verify_pivot_feedback_d0_rank5_terminal_disposition_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_local_routing_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_targeted_title_sample_v1.py",
        "scripts/verify_pivot_feedback_d0_t10543_local_routing_v1.py",
        "scripts/verify_pivot_feedback_d0_t10543_rank3_local_routing_v1.py",
        "scripts/verify_pivot_feedback_d0_t10543_rank3_targeted_title_sample_v1.py",
        "scripts/verify_pivot_feedback_d0_t10543_targeted_title_sample_v1.py",
        "scripts/verify_pivot_feedback_d0_t12505_rank4_local_routing_v1.py",
        "scripts/verify_pivot_feedback_d0_t12505_rank4_targeted_title_sample_v1.py",
        "scripts/verify_pivot_feedback_gate_state_v16.py",
        "scripts/verify_pivot_feedback_gate_state_v17.py",
        "scripts/verify_pivot_feedback_gate_state_v18.py",
        "scripts/verify_pivot_feedback_gate_state_v19.py",
        "scripts/verify_pivot_feedback_gate_state_v20.py",
        "scripts/verify_pivot_feedback_gate_state_v21.py",
        "scripts/verify_pivot_feedback_gate_state_v22.py",
        "scripts/verify_pivot_feedback_gate_state_v23.py",
        "scripts/verify_pivot_feedback_gate_state_v24.py",
        "scripts/verify_pivot_feedback_gate_state_v25.py",
    }
    assert inspector["PIVOT_RANK5_API_CLOSEOUT_REPOSITORY_ONLY_PATHS"] == {
        "docs/research/pivot-feedback-d0-rank5-api-augmented-closeout-v1.md",
        "experiments/configs/pivot_feedback_d0_rank5_api_augmented_closeout_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_result_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_result_v2.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_v2.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_v3.yaml",
        "experiments/configs/pivot_feedback_gate_state_v26.yaml",
        "experiments/configs/pivot_feedback_gate_state_v27.yaml",
        "experiments/configs/pivot_feedback_gate_state_v28.yaml",
        "experiments/configs/pivot_feedback_gate_state_v29.yaml",
        "experiments/evidence/results/pivot_feedback_d0_rank5_api_augmented_closeout_20260730.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_rank5_api_augmented_closeout_20260730.summary.md",
        "scripts/build_pivot_feedback_d0_rank5_api_augmented_closeout_v1.py",
        "scripts/build_pivot_feedback_d0_t10151_rank5_cross_provider_result_v1.py",
        "scripts/execute_pivot_feedback_d0_t10151_rank5_cross_provider_v1.py",
        "scripts/prepare_pivot_feedback_d0_t10151_rank5_cross_provider_v1.py",
        "scripts/verify_pivot_feedback_d0_rank5_api_augmented_closeout_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_cross_provider_result_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_cross_provider_v1.py",
        "scripts/verify_pivot_feedback_gate_state_v26.py",
        "scripts/verify_pivot_feedback_gate_state_v27.py",
        "scripts/verify_pivot_feedback_gate_state_v28.py",
        "scripts/verify_pivot_feedback_gate_state_v29.py",
    }
    assert inspector["PIVOT_RANK5_LOCAL_ABSTRACT_REPOSITORY_ONLY_PATHS"] == {
        "docs/research/pivot-feedback-d0-certified-episode-reselection-v1.md",
        "docs/research/pivot-feedback-d0-rank5-external-evidence-closeout-v1.md",
        "docs/research/pivot-feedback-d0-rank5-local-abstract-closeout-v1.md",
        "docs/research/pivot-feedback-d0-rank5-post-diagnostic-certificate-closeout-v1.md",
        "experiments/configs/pivot_feedback_d0_certified_episode_reselection_v1.yaml",
        "experiments/configs/pivot_feedback_d0_rank5_external_evidence_closeout_v1.yaml",
        "experiments/configs/pivot_feedback_d0_rank5_local_abstract_closeout_v1.yaml",
        "experiments/configs/pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_external_cross_provider_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_external_evidence_v1.yaml",
        "experiments/configs/pivot_feedback_d0_t10151_rank5_local_abstract_recovery_v1.yaml",
        "experiments/configs/pivot_feedback_gate_state_v30.yaml",
        "experiments/configs/pivot_feedback_gate_state_v31.yaml",
        "experiments/configs/pivot_feedback_gate_state_v32.yaml",
        "experiments/configs/pivot_feedback_gate_state_v33.yaml",
        "experiments/configs/pivot_feedback_gate_state_v34.yaml",
        "experiments/configs/pivot_feedback_gate_state_v35.yaml",
        "experiments/configs/pivot_feedback_gate_state_v36.yaml",
        "experiments/configs/pivot_feedback_gate_state_v37.yaml",
        "experiments/evidence/results/pivot_feedback_d0_certified_episode_reselection_20260730.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_certified_episode_reselection_20260730.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_rank5_external_evidence_closeout_20260730.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_rank5_external_evidence_closeout_20260730.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_rank5_local_abstract_closeout_20260730.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_rank5_local_abstract_closeout_20260730.summary.md",
        "experiments/evidence/results/pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_20260730.summary.json",
        "experiments/evidence/results/pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_20260730.summary.md",
        "scripts/build_pivot_feedback_d0_certified_episode_reselection_v1.py",
        "scripts/build_pivot_feedback_d0_rank5_external_evidence_closeout_v1.py",
        "scripts/build_pivot_feedback_d0_rank5_local_abstract_closeout_v1.py",
        "scripts/build_pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_v1.py",
        "scripts/build_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_v1.py",
        "scripts/build_pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_v1.py",
        "scripts/build_pivot_feedback_d0_t10151_rank5_local_abstract_recovery_v1.py",
        "scripts/collect_pivot_feedback_d0_t10151_rank5_external_evidence_v1.py",
        "scripts/execute_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.py",
        "scripts/execute_pivot_feedback_d0_t10151_rank5_external_cross_provider_v1.py",
        "scripts/finalize_pivot_feedback_d0_t10151_rank5_external_cross_provider_v1.py",
        "scripts/prepare_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.py",
        "scripts/verify_pivot_feedback_d0_certified_episode_reselection_v1.py",
        "scripts/verify_pivot_feedback_d0_rank5_external_evidence_closeout_v1.py",
        "scripts/verify_pivot_feedback_d0_rank5_local_abstract_closeout_v1.py",
        "scripts/verify_pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_external_cross_provider_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_external_evidence_v1.py",
        "scripts/verify_pivot_feedback_d0_t10151_rank5_local_abstract_recovery_v1.py",
        "scripts/verify_pivot_feedback_gate_state_v30.py",
        "scripts/verify_pivot_feedback_gate_state_v31.py",
        "scripts/verify_pivot_feedback_gate_state_v32.py",
        "scripts/verify_pivot_feedback_gate_state_v33.py",
        "scripts/verify_pivot_feedback_gate_state_v34.py",
        "scripts/verify_pivot_feedback_gate_state_v35.py",
        "scripts/verify_pivot_feedback_gate_state_v36.py",
        "scripts/verify_pivot_feedback_gate_state_v37.py",
    }
    assert inspector["MAIN_TEMPORAL_VALIDITY_REPOSITORY_ONLY_PATHS"] == {
        "docs/case-studies/researcher-pivot/openalex-main-temporal-validity-audit.md",
        "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml",
        "experiments/configs/openalex_g1_main_temporal_validity_v1.yaml",
        "experiments/configs/openalex_g1_main_temporal_validity_v2.yaml",
        "scripts/audit_openalex_main_temporal_validity.py",
        "scripts/audit_openalex_main_temporal_validity_v2.py",
        "scripts/audit_openalex_main_temporal_validity_v3.py",
    }
    assert inspector["AUTHOR_HISTORY_DESIGN_REPOSITORY_ONLY_PATHS"] == {
        "docs/case-studies/researcher-pivot/openalex-work-side-author-history-audit-design.md",
        "docs/research/openalex-work-side-author-history-v1-audit-design.md",
        "docs/research/openalex-work-side-author-history-v2-synthetic-validator.md",
        "docs/research/openalex-work-side-author-history-v3-production.md",
        "experiments/configs/openalex_g1_work_side_author_history_v0.yaml",
        "experiments/configs/openalex_g1_work_side_author_history_v1.yaml",
        "experiments/configs/openalex_g1_work_side_author_history_v2.yaml",
        "experiments/configs/openalex_g1_work_side_author_history_v3.yaml",
        "scripts/audit_openalex_work_side_author_history_v3.py",
        "scripts/verify_openalex_work_side_author_history_fixture_v2.py",
        "scripts/verify_openalex_work_side_author_history_v3.py",
    }
    assert inspector["AUTHOR_HISTORY_OPERATOR_LOCAL_BASENAMES"] == {
        "audit.lock",
        "audit_plan.json",
        "author_history_audit.json",
        "author_history_audit.md",
        "author_history_audit.outputs.json",
        "coverage_cube.parquet",
        "author_year_history.parquet",
        "author_token_history.parquet",
        "team_hyper_sensitivity_grid.parquet",
        "orcid_cardinality_aggregates.parquet",
        "run_state.json",
        "name_variation_aggregates.parquet",
    }
    assert inspector["AUTHOR_HISTORY_OPERATOR_LOCAL_DIRECTORY_BASENAMES"] == {
        ".author_history_tmp",
        "author_token_history",
        "author_year_history",
        "author_history_staging",
        "author_history_checkpoints",
        "author_history_data",
        "checkpoints",
        "events",
    }
    assert inspector["CITATION_AVAILABLE_AT_DESIGN_REPOSITORY_ONLY_PATHS"] == {
        "docs/case-studies/researcher-pivot/openalex-citation-available-at-audit-design.md",
        "docs/case-studies/researcher-pivot/openalex-citation-available-at-v1-audit-design.md",
        "docs/research/openalex-citation-edge-chronology-v3-production.md",
        "experiments/configs/openalex_g1_citation_available_at_v0.yaml",
        "experiments/configs/openalex_g1_citation_available_at_v1.yaml",
        "experiments/configs/openalex_g1_citation_available_at_v2.yaml",
        "scripts/audit_openalex_citation_available_at.py",
        "scripts/audit_openalex_citation_available_at_v2.py",
        "scripts/audit_openalex_citation_available_at_v3.py",
        "scripts/verify_openalex_citation_available_at_v3.py",
    }
    assert inspector["CITATION_OPERATOR_LOCAL_DATA_BASENAMES"] == {
        "ephemeral_noncanonical_direct_mode_time_index",
        "deduplicated_edge_event_partitions",
        "chronology_safe_edge_partitions_by_cited_work",
        ".citation_tmp",
        "deduplicated_edge_events",
        "chronology_safe_edges_by_cited_work",
        "deduplicated_edge_manifest.json",
        "chronology_safe_edge_manifest.json",
        "citation_edge_chronology_audit.json",
        "citation_edge_chronology_audit.md",
        "citation_edge_chronology_audit.outputs.json",
        "duplicate_multiplicity_aggregates.parquet",
        "chronology_lag_aggregates.parquet",
        "open_world_target_coverage.parquet",
        "cutoff_and_maturation_coverage.parquet",
    }
    assert inspector["WORK_TIME_SPINE_DESIGN_REPOSITORY_ONLY_PATHS"] == {
        "docs/case-studies/researcher-pivot/openalex-canonical-work-time-spine-audit-design.md",
        "docs/case-studies/researcher-pivot/openalex-canonical-work-time-spine-v1-audit-design.md",
        "docs/research/openalex-canonical-work-time-spine-v2-production-design.md",
        "experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml",
        "experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml",
        "experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml",
        "scripts/audit_openalex_canonical_work_time_spine.py",
        "scripts/audit_openalex_canonical_work_time_spine_v2.py",
        "scripts/audit_openalex_canonical_work_time_spine_v3.py",
    }
    assert inspector["WORK_TIME_SPINE_OPERATIONAL_REPOSITORY_ONLY_PATHS"] == {
        "docs/research/openalex-canonical-work-time-spine-capacity-review-v0.md",
        "docs/research/openalex-canonical-work-time-spine-downstream-benchmark-v0.md",
        "docs/research/openalex-canonical-work-time-spine-downstream-stress-v1.md",
        "docs/research/openalex-canonical-work-time-spine-independent-verifier-v1.md",
        "docs/research/openalex-canonical-work-time-spine-ops-readiness-20260720.md",
        "docs/research/openalex-canonical-work-time-spine-pilot-benchmark-tool.md",
        "experiments/configs/openalex_g1_canonical_work_time_spine_capacity_review_v0.yaml",
        "scripts/benchmark_openalex_canonical_work_time_spine_downstream.py",
        "scripts/benchmark_openalex_canonical_work_time_spine_downstream_stress_v1.py",
        "scripts/review_openalex_canonical_work_time_spine_capacity.py",
        "scripts/summarize_openalex_canonical_work_time_spine_pilot_benchmark.py",
        "scripts/verify_openalex_canonical_work_time_spine_downstream_benchmark_v1.py",
        "scripts/verify_openalex_canonical_work_time_spine_downstream_stress_v1.py",
        "scripts/verify_openalex_canonical_work_time_spine_v1.py",
    }
    assert inspector["WORK_TIME_SPINE_OPERATOR_LOCAL_DIRECTORY_BASENAMES"] == {
        ".work_time_spine_tmp",
        "staging",
        "spine",
    }
    assert inspector["WORK_TIME_SPINE_OPERATOR_LOCAL_PATH_FRAGMENTS"] == {
        "checkpoints/scan/main",
        "checkpoints/partition",
    }
    assert inspector["WORK_TIME_SPINE_OPERATOR_LOCAL_BASENAMES"] == {
        "audit.lock",
        "audit_plan.json",
        "openalex_g1_canonical_work_time_spine_downstream_benchmark.json",
        "openalex_g1_canonical_work_time_spine_downstream_stress_v1.json",
        "openalex_g1_canonical_work_time_spine_pilot_benchmark.json",
        "openalex_g1_canonical_work_time_spine_pilot_time.txt",
        "preflight_evidence.json",
        "run_state.json",
        "staging_manifest.json",
        "work_time_spine_audit.json",
        "work_time_spine_audit.md",
        "work_time_spine_audit.outputs.json",
        "work_time_spine_manifest.json",
        "work_time_spine_pilot.json",
        "work_time_spine_pilot.md",
        "work_time_spine_pilot.outputs.json",
        "work_time_spine_test_fixture.json",
        "work_time_spine_test_fixture.md",
        "work_time_spine_test_fixture.outputs.json",
    }
    assert inspector["RELATION_SENSITIVITY_REPOSITORY_ONLY_PATHS"] == {
        "docs/research/openalex-work-topic-relation-pilot-20260721.md",
        "experiments/configs/openalex_g1_w0_positive_id_sensitivity_v0.yaml",
        "experiments/configs/openalex_g1_work_topic_relation_v0.yaml",
        "experiments/configs/openalex_g1_work_topic_relation_primary_adjudication_v1.yaml",
        "scripts/adjudicate_openalex_work_topic_primary_scores.py",
        "scripts/audit_openalex_w0_positive_id_sensitivity.py",
        "scripts/audit_openalex_work_topic_relation.py",
        "scripts/verify_openalex_work_topic_primary_adjudication.py",
    }
    assert inspector["CUTOFF_PRIMITIVES_REPOSITORY_ONLY_PATHS"] == {
        "docs/research/pivot-feedback-cutoff-primitives-v0.md",
        "experiments/configs/pivot_feedback_cutoff_primitives_v0.yaml",
        "scripts/build_pivot_feedback_cutoff_primitives.py",
        "scripts/verify_pivot_feedback_cutoff_primitives.py",
    }
    assert inspector["CUTOFF_PRIMITIVES_OPERATOR_LOCAL_BASENAMES"] == {
        "audit_plan.json",
        "pivot_feedback_cutoff_panel.json",
        "pivot_feedback_cutoff_panel.md",
        "pivot_feedback_cutoff_panel.outputs.json",
        "run_state.json",
        "work_metadata_manifest.json",
    }
    assert inspector["CUTOFF_PRIMITIVES_OPERATOR_LOCAL_DIRECTORY_BASENAMES"] == {
        ".cutoff_primitives_tmp",
        "work_metadata",
        "work_spools",
    }
    assert inspector["PIVOT_TOPIC_ROUTING_OPERATOR_LOCAL_BASENAMES"] == {
        "API_ONLY_TOPIC_ROUTING_DRY_RUN.md",
        "pivot-feedback-d0-api-only-topic-routing-v1-dry-run-20260729.verification-v1.json",
        "topic_routing_decisions.csv",
        "topic_routing_dry_run.outputs.json",
        "topic_routing_summary.csv",
    }
    assert inspector["PIVOT_CANDIDATE_SCREEN_OPERATOR_LOCAL_BASENAMES"] == {
        "ADAPTIVE_CANDIDATE_FRONTIER.md",
        "CANDIDATE_CONTENT_DRY_RUN.md",
        "CANDIDATE_SCREEN_SCOPE_AUDIT.md",
        "adaptive_candidate_frontier.outputs.json",
        "adaptive_frontier_equivalence_proof.json",
        "candidate_content_dry_run.outputs.json",
        "candidate_content_dry_run.parquet",
        "candidate_content_sensitivity.parquet",
        "candidate_frontier_cost_estimate.json",
        "candidate_screen_scope_audit.json",
        "candidate_screen_scope_audit.outputs.json",
        "candidate_topic_scope.parquet",
        "candidate_window_scope.parquet",
        "initial_structural_frontier.parquet",
        "pivot-feedback-d0-adaptive-candidate-frontier-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-candidate-content-dry-run-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-candidate-screen-scope-audit-v1-20260729.verification-v1.json",
    }
    assert inspector["PIVOT_CANDIDATE_SUCCESSOR_OPERATOR_LOCAL_BASENAMES"] == {
        "CANDIDATE_RULE_V2_RETROSPECTIVE_REPLAY.md",
        "PRODUCTION_CANDIDATE_RULE_AND_FRONTIER_V2.md",
        "PRODUCTION_CANDIDATE_RULE_V2_AND_FRONTIER_V3.md",
        "RANK3_PRODUCTION_DISPOSITION_AND_FRONTIER_V4.md",
        "RANK4_PRODUCTION_DISPOSITION_AND_FRONTIER_V5.md",
        "RANK5_TERMINAL_DISPOSITION_AND_CERTIFICATE_STATUS.md",
        "T10151_RANK5_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN.md",
        "T10151_RANK5_TARGETED_TITLE_SAMPLE.md",
        "T10543_LOCAL_ROUTING_AND_CANDIDATE_DRY_RUN.md",
        "T10543_RANK3_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN.md",
        "T10543_RANK3_TARGETED_TITLE_SAMPLE.md",
        "T10543_TARGETED_TITLE_SAMPLE.md",
        "T12505_RANK4_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN.md",
        "T12505_RANK4_TARGETED_TITLE_SAMPLE.md",
        "adaptive_candidate_frontier_v2.parquet",
        "adaptive_candidate_frontier_v3.parquet",
        "adaptive_candidate_frontier_v4.parquet",
        "adaptive_candidate_frontier_v5.parquet",
        "adaptive_candidate_frontier_v6.parquet",
        "candidate_rule_v2_change_summary.json",
        "candidate_rule_v2_replay.outputs.json",
        "candidate_rule_v2_retrospective_replay.parquet",
        "cumulative_production_candidate_dispositions_v3.parquet",
        "cumulative_production_candidate_dispositions_v4.parquet",
        "cumulative_production_candidate_dispositions_v5.parquet",
        "pivot-feedback-d0-candidate-rule-v2-replay-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-production-candidate-rule-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-production-candidate-rule-v2-20260729.verification-v1.json",
        "pivot-feedback-d0-rank3-production-disposition-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-rank4-production-disposition-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-rank5-terminal-disposition-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-local-routing-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-targeted-title-sample-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-t10543-local-routing-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-t10543-rank3-local-routing-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-t10543-rank3-targeted-title-sample-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-t10543-targeted-title-sample-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-t12505-rank4-local-routing-v1-20260729.verification-v1.json",
        "pivot-feedback-d0-t12505-rank4-targeted-title-sample-v1-20260729.verification-v1.json",
        "production_candidate_dispositions.parquet",
        "production_candidate_dispositions_v2.parquet",
        "production_candidate_rule.outputs.json",
        "production_candidate_rule_v2.outputs.json",
        "rank3_production_disposition.outputs.json",
        "rank4_production_disposition.outputs.json",
        "rank5_terminal_disposition.outputs.json",
        "terminal_selection_certificate_status.json",
        "t10151_rank5_candidate_pool_summary.json",
        "t10151_rank5_candidate_rule_v2_dry_run.parquet",
        "t10151_rank5_local_routing.outputs.json",
        "t10151_rank5_local_title_routes.csv",
        "t10151_rank5_targeted_title_sample.outputs.json",
        "t10151_rank5_targeted_title_sample.parquet",
        "t10543_candidate_content_dry_run.parquet",
        "t10543_candidate_content_sensitivity.parquet",
        "t10543_candidate_pool_summary.json",
        "t10543_local_routing.outputs.json",
        "t10543_local_title_routes.csv",
        "t10543_rank3_candidate_pool_summary.json",
        "t10543_rank3_candidate_rule_v2_dry_run.parquet",
        "t10543_rank3_local_routing.outputs.json",
        "t10543_rank3_local_title_routes.csv",
        "t10543_rank3_targeted_title_sample.outputs.json",
        "t10543_rank3_targeted_title_sample.parquet",
        "t10543_targeted_title_sample.outputs.json",
        "t10543_targeted_title_sample.parquet",
        "t12505_rank4_candidate_pool_summary.json",
        "t12505_rank4_candidate_rule_v2_dry_run.parquet",
        "t12505_rank4_local_routing.outputs.json",
        "t12505_rank4_local_title_routes.csv",
        "t12505_rank4_targeted_title_sample.outputs.json",
        "t12505_rank4_targeted_title_sample.parquet",
    }
    assert inspector["PIVOT_RANK5_API_CLOSEOUT_OPERATOR_LOCAL_BASENAMES"] == {
        "API_RESULT_SUMMARY.md",
        "RANK5_API_AUGMENTED_TERMINAL_CLOSEOUT.md",
        "adaptive_candidate_frontier_v7.parquet",
        "claude_request.json",
        "claude_response.json",
        "cross_provider_execution.outputs.json",
        "cross_provider_preparation.outputs.json",
        "cross_provider_result.outputs.json",
        "cumulative_production_candidate_dispositions_v6.parquet",
        "execution_failure.outputs.json",
        "execution_manifest.json",
        "openai_request.json",
        "openai_response.json",
        "pivot-feedback-d0-rank5-api-augmented-closeout-v1-20260730.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-cross-provider-v1-prepared-20260730.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-cross-provider-v2-prepared-20260730.verification-v2.json",
        "pivot-feedback-d0-t10151-rank5-cross-provider-v2-results-20260730.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-cross-provider-v3-prepared-20260730.verification-v3.json",
        "pivot-feedback-d0-t10151-rank5-cross-provider-v3-results-20260730.verification-v2.json",
        "preparation_manifest.json",
        "rank5_api_augmented_closeout.outputs.json",
        "rank5_augmented_candidate.json",
        "rank5_cross_provider_decisions.csv",
        "rank5_cross_provider_result_summary.json",
        "request_key.csv",
        "result_manifest.json",
        "terminal_selection_certificate_status_v2.json",
    }
    assert inspector["PIVOT_RANK5_LOCAL_ABSTRACT_OPERATOR_LOCAL_BASENAMES"] == {
        "ABSTRACT_RESULT_SUMMARY.md",
        "CERTIFIED_EPISODE_RESELECTION.md",
        "EXTERNAL_CONTROLLED_SUBJECT_RULE_REPLAY.md",
        "EXTERNAL_CROSS_PROVIDER_RESULT.md",
        "EXTERNAL_EVIDENCE_COLLECTION.md",
        "LOCAL_ABSTRACT_RECOVERY.md",
        "RANK5_EXTERNAL_EVIDENCE_CLOSEOUT.md",
        "RANK5_LOCAL_ABSTRACT_TERMINAL_CLOSEOUT.md",
        "RANK5_POST_DIAGNOSTIC_CERTIFICATE_CLOSEOUT.md",
        "abstract_cross_provider_execution.outputs.json",
        "abstract_cross_provider_preparation.outputs.json",
        "abstract_cross_provider_result.outputs.json",
        "adaptive_candidate_frontier_v8.parquet",
        "adaptive_candidate_frontier_v9.parquet",
        "adaptive_candidate_frontier_v10.parquet",
        "certified_episode_reselection.json",
        "certified_episode_reselection.outputs.json",
        "certified_episode_selection.parquet",
        "claude_request.json",
        "claude_response.json",
        "cumulative_production_candidate_dispositions_v7.parquet",
        "cumulative_production_candidate_dispositions_v8.parquet",
        "cumulative_production_candidate_dispositions_v9.parquet",
        "execution_failure.outputs.json",
        "execution_manifest.json",
        "external_cross_provider.outputs.json",
        "external_cross_provider_failure.outputs.json",
        "external_controlled_subject_rule.outputs.json",
        "external_evidence.outputs.json",
        "local_abstract_recovery.outputs.json",
        "openai_request.json",
        "openai_response.json",
        "pivot-feedback-d0-certified-episode-reselection-v1-20260730.verification-v1.json",
        "pivot-feedback-d0-rank5-external-evidence-closeout-v1-20260730.verification-v1.json",
        "pivot-feedback-d0-rank5-local-abstract-closeout-v1-20260730.verification-v1.json",
        "pivot-feedback-d0-rank5-post-diagnostic-certificate-closeout-v1-20260730.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-abstract-cross-provider-v1-prepared-20260730.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-abstract-cross-provider-v1-results-20260730.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-external-cross-provider-v1-20260730.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-external-controlled-subject-rule-v1-20260730.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-external-evidence-v1-20260730.verification-v1.json",
        "pivot-feedback-d0-t10151-rank5-local-abstract-recovery-v1-20260730.verification-v1.json",
        "preparation_manifest.json",
        "rank5_external_augmented_candidate.json",
        "rank5_external_controlled_subject_rule_replay.parquet",
        "rank5_external_cross_provider_decisions.csv",
        "rank5_external_evidence.parquet",
        "rank5_external_evidence_availability.csv",
        "rank5_external_evidence_closeout.outputs.json",
        "rank5_external_review_packet.json",
        "rank5_abstract_augmented_candidate.json",
        "rank5_abstract_cross_provider_decisions.csv",
        "rank5_abstract_cross_provider_result_summary.json",
        "rank5_local_abstract_availability.csv",
        "rank5_local_abstract_closeout.outputs.json",
        "rank5_local_abstract_recovery.parquet",
        "rank5_post_diagnostic_candidate.json",
        "rank5_post_diagnostic_certificate_closeout.outputs.json",
        "raw_crossref_responses.json",
        "raw_openalex_response.json",
        "raw_pubmed_response.xml",
        "request_key.csv",
        "terminal_selection_certificate_status_v3.json",
        "terminal_selection_certificate_status_v4.json",
        "terminal_selection_certificate_status_v5.json",
    }
    assert "experiments/" in inspector["FORBIDDEN_SDIST_PREFIXES"]
    assert any(
        "experiments/evidence/results/example.json".startswith(prefix)
        for prefix in inspector["FORBIDDEN_SDIST_PREFIXES"]
    )


def test_release_inspector_rejects_citation_operator_local_output_fixture() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/deduplicated_edge_event_partitions/part-0000.bin",
        "private/chronology_safe_edge_partitions_by_cited_work/part-0000.bin",
        "private/deduplicated_edge_events/part-0000.parquet",
        "private/chronology_safe_edges_by_cited_work/part-0000.parquet",
        "private/deduplicated_edge_manifest.json",
        "private/chronology_safe_edge_manifest.json",
        "private/citation_edge_chronology_audit.outputs.json",
        "private/chronology_lag_aggregates.parquet",
        "private/ephemeral_noncanonical_direct_mode_time_index/part-0000.bin",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_cutoff_primitives_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/.cutoff_primitives_tmp/work_spools/work-0000.bin",
        "private/work_metadata/part-0000.parquet",
        "private/work_metadata_manifest.json",
        "private/pivot_feedback_cutoff_panel.outputs.json",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_topic_routing_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/API_ONLY_TOPIC_ROUTING_DRY_RUN.md",
        "private/pivot-feedback-d0-api-only-topic-routing-v1-dry-run-20260729.verification-v1.json",
        "private/topic_routing_decisions.csv",
        "private/topic_routing_dry_run.outputs.json",
        "private/topic_routing_summary.csv",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_candidate_screen_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/ADAPTIVE_CANDIDATE_FRONTIER.md",
        "private/CANDIDATE_CONTENT_DRY_RUN.md",
        "private/CANDIDATE_SCREEN_SCOPE_AUDIT.md",
        "private/adaptive_candidate_frontier.outputs.json",
        "private/adaptive_frontier_equivalence_proof.json",
        "private/candidate_content_dry_run.outputs.json",
        "private/candidate_content_dry_run.parquet",
        "private/candidate_content_sensitivity.parquet",
        "private/candidate_frontier_cost_estimate.json",
        "private/candidate_screen_scope_audit.json",
        "private/candidate_screen_scope_audit.outputs.json",
        "private/candidate_topic_scope.parquet",
        "private/candidate_window_scope.parquet",
        "private/initial_structural_frontier.parquet",
        "private/pivot-feedback-d0-adaptive-candidate-frontier-v1-20260729.verification-v1.json",
        "private/pivot-feedback-d0-candidate-content-dry-run-v1-20260729.verification-v1.json",
        "private/pivot-feedback-d0-candidate-screen-scope-audit-v1-20260729.verification-v1.json",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_candidate_successor_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/adaptive_candidate_frontier_v4.parquet",
        "private/cumulative_production_candidate_dispositions_v3.parquet",
        "private/rank3_production_disposition.outputs.json",
        "private/t10543_rank3_local_title_routes.csv",
        "private/t10543_rank3_targeted_title_sample.parquet",
        "private/pivot-feedback-d0-rank3-production-disposition-v1-20260729.verification-v1.json",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_rank5_api_closeout_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/openai_response.json",
        "private/claude_response.json",
        "private/rank5_cross_provider_decisions.csv",
        "private/rank5_augmented_candidate.json",
        "private/cumulative_production_candidate_dispositions_v6.parquet",
        "private/adaptive_candidate_frontier_v7.parquet",
        "private/terminal_selection_certificate_status_v2.json",
        "private/pivot-feedback-d0-rank5-api-augmented-closeout-v1-20260730.verification-v1.json",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_rank5_local_abstract_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/rank5_local_abstract_recovery.parquet",
        "private/abstract_cross_provider_execution.outputs.json",
        "private/rank5_abstract_cross_provider_decisions.csv",
        "private/adaptive_candidate_frontier_v8.parquet",
        "private/terminal_selection_certificate_status_v3.json",
        "private/pivot-feedback-d0-rank5-local-abstract-closeout-v1-20260730.verification-v1.json",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_rank5_external_evidence_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/raw_openalex_response.json",
        "private/raw_crossref_responses.json",
        "private/raw_pubmed_response.xml",
        "private/rank5_external_review_packet.json",
        "private/rank5_external_cross_provider_decisions.csv",
        "private/cumulative_production_candidate_dispositions_v8.parquet",
        "private/adaptive_candidate_frontier_v9.parquet",
        "private/terminal_selection_certificate_status_v4.json",
        "private/pivot-feedback-d0-rank5-external-evidence-closeout-v1-20260730.verification-v1.json",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_persistence_source_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    expected = {
        "PERSISTENCE_OUTCOME_SOURCE.md",
        "persistence_outcome_source.json",
        "persistence_outcome_source.outputs.json",
        "persistence_outcome_source.parquet",
        "pivot-feedback-d0-persistence-source-v1-full-20260731.verification-v1.json",
    }
    assert inspector["PIVOT_PERSISTENCE_SOURCE_OPERATOR_LOCAL_BASENAMES"] == expected
    names = {"README.md", *(f"private/{name}" for name in expected)}

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_persistence_calibration_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    expected = {
        "PERSISTENCE_CALIBRATION_READOUT.md",
        "persistence_calibration_readout.json",
        "persistence_calibration_readout.outputs.json",
        "pivot-feedback-d0-persistence-calibration-readout-v1-20260731.verification-v1.json",
    }
    assert (
        inspector["PIVOT_PERSISTENCE_CALIBRATION_OPERATOR_LOCAL_BASENAMES"]
        == expected
    )
    names = {"README.md", *(f"private/{name}" for name in expected)}

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_author_history_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/coverage_cube.parquet",
        "private/author_year_history.parquet",
        "private/author_token_history.parquet",
        "private/team_hyper_sensitivity_grid.parquet",
        "private/orcid_cardinality_aggregates.parquet",
        "private/name_variation_aggregates.parquet",
        "private/author_history_staging/part-0000.bin",
        "private/author_history_checkpoints/part-0000.json",
        "private/author_history_data/part-0000.parquet",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_researcher_pivot_boundary_is_exact_path_and_content_allowlist() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    expected = inspector["RESEARCHER_PIVOT_PUBLIC_SUBTREE_SHA256"]
    check = inspector["_researcher_pivot_boundary_issues"]

    assert check(set(expected), expected, []) == []

    prefix = inspector["RESEARCHER_PIVOT_PUBLIC_SUBTREE_PREFIX"]
    canaries = (
        "coverage_cube.parquet.gz",
        "author_token_history.csv",
        "author_history_data.tar.zst",
        "renamed-author-records.parquet",
    )
    for canary in canaries:
        path = f"{prefix}{canary}"
        issues = check(set(expected) | {path}, expected, [])
        assert any(path in issue and "unexpected member" in issue for issue in issues)

    aliases = (
        "docs/case-studies/other/../researcher-pivot/renamed-author-records.parquet",
        "docs\\case-studies\\researcher-pivot\\renamed-author-records.parquet",
    )
    for alias in aliases:
        issues = check(set(expected) | {alias}, expected, [])
        assert any(repr(alias) in issue and "noncanonical" in issue for issue in issues)

    changed = dict(expected)
    changed_path = next(iter(changed))
    changed[changed_path] = "0" * 64
    issues = check(set(expected), changed, [])
    assert any(
        changed_path in issue and "content SHA-256 mismatch" in issue
        for issue in issues
    )

    issues = check(set(expected), expected, [changed_path])
    assert any(
        changed_path in issue and "non-regular archive member" in issue
        for issue in issues
    )


def test_sdist_member_boundary_rejects_archive_aliases_and_duplicates(
    tmp_path: Path,
) -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    check = inspector["_sdist_member_boundary_issues"]

    def inspect_entries(
        filename: str,
        entries: list[tuple[str, bytes, bytes, str]],
    ) -> list[str]:
        archive_path = tmp_path / filename
        with tarfile.open(archive_path, "w:gz") as archive:
            for name, payload, member_type, linkname in entries:
                member = tarfile.TarInfo(name)
                member.type = member_type
                member.linkname = linkname
                member.size = len(payload) if member_type == tarfile.REGTYPE else 0
                archive.addfile(
                    member,
                    io.BytesIO(payload) if member_type == tarfile.REGTYPE else None,
                )
        with tarfile.open(archive_path, "r:gz") as archive:
            return check(archive.getmembers())

    root = "neural_abm-0.1.0a6.dev0"
    regular = tarfile.REGTYPE
    assert (
        inspect_entries(
            "valid.tar.gz",
            [
                (f"{root}/README.md", b"safe", regular, ""),
                (f"{root}/PKG-INFO", b"metadata", regular, ""),
            ],
        )
        == []
    )

    adversarial = {
        "dotdot.tar.gz": [
            (
                f"{root}/docs/case-studies/other/../researcher-pivot/leak.parquet",
                b"secret",
                regular,
                "",
            )
        ],
        "backslash.tar.gz": [
            (
                f"{root}/docs\\case-studies\\researcher-pivot\\leak.parquet",
                b"secret",
                regular,
                "",
            )
        ],
        "duplicate.tar.gz": [
            (f"{root}/docs/safe.json", b"secret", regular, ""),
            (f"{root}/docs/safe.json", b"safe", regular, ""),
        ],
        "symlink.tar.gz": [
            (f"{root}/docs/link", b"", tarfile.SYMTYPE, "../../secret"),
        ],
        "hardlink.tar.gz": [
            (f"{root}/docs/link", b"", tarfile.LNKTYPE, f"{root}/README.md"),
        ],
        "directory.tar.gz": [
            (f"{root}/docs/unexpected", b"", tarfile.DIRTYPE, ""),
        ],
        "multi-root.tar.gz": [
            (f"{root}/README.md", b"safe", regular, ""),
            ("other-root/PKG-INFO", b"metadata", regular, ""),
        ],
    }
    for filename, entries in adversarial.items():
        assert inspect_entries(filename, entries), filename


def test_release_inspector_rejects_work_time_spine_operator_outputs() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    names = {
        "README.md",
        "private/spine/part-0000.parquet",
        "private/staging/main/abc.bin",
        "private/.work_time_spine_tmp/spool/part-0000.bin",
        "private/checkpoints/scan/main/abc.json",
        "private/checkpoints/partition/part-0000.json",
        "private/audit.lock",
        "private/audit_plan.json",
        "private/openalex_g1_canonical_work_time_spine_downstream_benchmark.json",
        "private/openalex_g1_canonical_work_time_spine_downstream_stress_v1.json",
        "private/openalex_g1_canonical_work_time_spine_pilot_benchmark.json",
        "private/openalex_g1_canonical_work_time_spine_pilot_time.txt",
        "docs/research/openalex-canonical-work-time-spine-downstream-stress-v1.md",
        "scripts/verify_openalex_canonical_work_time_spine_downstream_stress_v1.py",
        "private/preflight_evidence.json",
        "private/run_state.json",
        "private/staging_manifest.json",
        "private/work_time_spine_manifest.json",
        "private/work_time_spine_audit.json",
        "private/work_time_spine_audit.md",
        "private/work_time_spine_audit.outputs.json",
        "private/work_time_spine_pilot.json",
        "private/work_time_spine_pilot.md",
        "private/work_time_spine_pilot.outputs.json",
        "private/work_time_spine_test_fixture.json",
        "private/work_time_spine_test_fixture.md",
        "private/work_time_spine_test_fixture.outputs.json",
    }

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_holdout_seal_repository_and_operator_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_holdout_access_seal_v1.yaml",
        "experiments/configs/pivot_feedback_gate_state_v54.yaml",
        "experiments/configs/pivot_feedback_gate_state_v55.yaml",
        "experiments/configs/pivot_feedback_gate_state_v56.yaml",
        "experiments/configs/pivot_feedback_gate_state_v57.yaml",
        "experiments/configs/pivot_feedback_holdout_production_input_resolution_v1.yaml",
        "docs/research/pivot-feedback-holdout-access-seal-contract-v1.md",
        "docs/research/pivot-feedback-holdout-seal-tooling-v1.md",
        "docs/research/pivot-feedback-holdout-seal-preflight-tooling-v1.md",
        "docs/research/pivot-feedback-holdout-production-input-resolution-v1.md",
        "experiments/evidence/results/pivot_feedback_holdout_access_seal_contract_20260731.summary.json",
        "experiments/evidence/results/pivot_feedback_holdout_access_seal_contract_20260731.summary.md",
        "experiments/evidence/results/pivot_feedback_holdout_seal_tooling_closed_fixture_20260731.summary.json",
        "experiments/evidence/results/pivot_feedback_holdout_seal_tooling_closed_fixture_20260731.summary.md",
        "experiments/evidence/results/pivot_feedback_holdout_seal_preflight_tooling_closed_fixture_20260731.summary.json",
        "experiments/evidence/results/pivot_feedback_holdout_seal_preflight_tooling_closed_fixture_20260731.summary.md",
        "experiments/evidence/results/pivot_feedback_holdout_production_input_resolution_20260731.summary.json",
        "experiments/evidence/results/pivot_feedback_holdout_production_input_resolution_20260731.summary.md",
        "scripts/create_pivot_feedback_holdout_access_seal_v1.py",
        "scripts/prepare_pivot_feedback_holdout_access_seal_preflight_v1.py",
        "scripts/verify_pivot_feedback_holdout_access_seal_v1.py",
        "scripts/verify_pivot_feedback_holdout_access_seal_receipt_v1.py",
        "scripts/verify_pivot_feedback_holdout_production_input_resolution_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]
    operator_paths = {
        "private/pivot-feedback/holdout-seal-v1/manifest.json",
        "private/pivot-feedback/holdout-seal-v1/artifacts/part-0000.parquet",
        "private/pivot-feedback/holdout-seal-v1/staging/manifest.json",
        "private/operator-control/pivot_feedback_holdout_access_seal_preflight_v1.json",
    }
    names = {"README.md", *repository_paths, *operator_paths}

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_release_inspector_rejects_f_assignment_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_field_episode_assignment_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v3.md",
        "experiments/evidence/results/pivot_feedback_f_field_episode_assignment_v1_20260803.summary.json",
        "experiments/evidence/results/pivot_feedback_f_field_episode_assignment_v1_20260803.summary.md",
        "scripts/build_pivot_feedback_f_field_episode_assignment_v1.py",
        "scripts/verify_pivot_feedback_f_field_episode_assignment_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_final_panel_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_final_panel_conformance_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v4.md",
        "experiments/evidence/results/pivot_feedback_f_final_panel_conformance_v1_20260804.summary.json",
        "experiments/evidence/results/pivot_feedback_f_final_panel_conformance_v1_20260804.summary.md",
        "scripts/build_pivot_feedback_f_final_panel_conformance_v1.py",
        "scripts/verify_pivot_feedback_f_final_panel_conformance_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_definition_theory_audit_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_definition_theory_audit_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v5.md",
        "experiments/evidence/results/pivot_feedback_f_definition_theory_audit_v1_20260804.summary.json",
        "experiments/evidence/results/pivot_feedback_f_definition_theory_audit_v1_20260804.summary.md",
        "scripts/build_pivot_feedback_f_definition_theory_audit_v1.py",
        "scripts/verify_pivot_feedback_f_definition_theory_audit_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_aligned_strict_rescreen_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v6.md",
        "experiments/evidence/results/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1_20260804.summary.json",
        "experiments/evidence/results/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1_20260804.summary.md",
        "scripts/build_pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.py",
        "scripts/verify_pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_aligned_final_panel_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_aligned_final_panel_conformance_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v7.md",
        "experiments/evidence/results/pivot_feedback_f_aligned_final_panel_conformance_v1_20260804.summary.json",
        "experiments/evidence/results/pivot_feedback_f_aligned_final_panel_conformance_v1_20260804.summary.md",
        "scripts/build_pivot_feedback_f_aligned_final_panel_conformance_v1.py",
        "scripts/verify_pivot_feedback_f_aligned_final_panel_conformance_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_atomic_partition_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_atomic_partition_assignment_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v8.md",
        "experiments/evidence/results/pivot_feedback_f_atomic_partition_assignment_v1_20260804.summary.json",
        "experiments/evidence/results/pivot_feedback_f_atomic_partition_assignment_v1_20260804.summary.md",
        "scripts/build_pivot_feedback_f_atomic_partition_assignment_v1.py",
        "scripts/verify_pivot_feedback_f_atomic_partition_assignment_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_g4_private_membership_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_g4_private_membership_materialization_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v9.md",
        "experiments/evidence/results/pivot_feedback_f_g4_private_membership_materialization_v1_20260804.summary.json",
        "experiments/evidence/results/pivot_feedback_f_g4_private_membership_materialization_v1_20260804.summary.md",
        "scripts/build_pivot_feedback_f_g4_private_membership_materialization_v1.py",
        "scripts/verify_pivot_feedback_f_g4_private_membership_materialization_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_g4_private_feature_source_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_g4_private_feature_source_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v10.md",
        "experiments/evidence/results/pivot_feedback_f_g4_private_feature_source_v1_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_f_g4_private_feature_source_v1_20260805.summary.md",
        "scripts/build_pivot_feedback_f_g4_private_feature_source_v1.py",
        "scripts/verify_pivot_feedback_f_g4_private_feature_source_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_g4_private_root_binding_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_g4_private_root_artifact_binding_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v11.md",
        "experiments/evidence/results/pivot_feedback_f_g4_private_root_artifact_binding_v1_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_f_g4_private_root_artifact_binding_v1_20260805.summary.md",
        "scripts/bind_pivot_feedback_f_g4_private_root_artifact_list_v1.py",
        "scripts/verify_pivot_feedback_f_g4_private_root_artifact_binding_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_g4_private_preflight_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_f_g4_private_explicit_list_preflight_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v12.md",
        "experiments/evidence/results/pivot_feedback_f_g4_private_explicit_list_preflight_v1_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_f_g4_private_explicit_list_preflight_v1_20260805.summary.md",
        "scripts/prepare_pivot_feedback_f_g4_private_explicit_list_preflight_v1.py",
        "scripts/verify_pivot_feedback_f_g4_private_explicit_list_preflight_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_f_g4_private_seal_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-feedback-current-status-evidence-ledger-v13.md",
        "experiments/evidence/results/pivot_feedback_holdout_access_seal_receipt_v1.summary.json",
        "experiments/evidence/results/pivot_feedback_holdout_access_seal_receipt_verification_v1_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_holdout_access_seal_receipt_verification_v1_20260805.summary.md",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_atomic_f_closure_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_current_state_v2.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v14.md",
        "experiments/evidence/results/pivot_feedback_current_state_v2_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_current_state_v2_20260805.summary.md",
        "scripts/build_pivot_feedback_current_state_v2.py",
        "scripts/verify_pivot_feedback_current_state_v2.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_e_g1_input_preflight_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v15.md",
        "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.md",
        "scripts/build_pivot_feedback_e_g1_execution_input_preflight_v1.py",
        "scripts/verify_pivot_feedback_e_g1_execution_input_preflight_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_e_g1_input_preflight_v2_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v2.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v16.md",
        "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v2_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v2_20260805.summary.md",
        "scripts/build_pivot_feedback_e_g1_execution_input_preflight_v2.py",
        "scripts/verify_pivot_feedback_e_g1_execution_input_preflight_v2.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_terminal_blocked_e_g1_repository_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_e_g1_execution_v1.yaml",
        "experiments/configs/pivot_feedback_current_state_v3.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v17.md",
        "experiments/evidence/results/pivot_feedback_e_g1_execution_v1_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_e_g1_execution_v1_20260805.summary.md",
        "experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.md",
        "scripts/execute_pivot_feedback_e_g1_v1.py",
        "scripts/verify_pivot_feedback_e_g1_execution_v1.py",
        "scripts/build_pivot_feedback_current_state_v3.py",
        "scripts/verify_pivot_feedback_current_state_v3.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_guard_exception_and_support_blocked_successor_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_current_state_v4.yaml",
        "experiments/configs/pivot_feedback_e_g1_guard_only_exception_v1.yaml",
        "experiments/configs/pivot_feedback_e_g1_guarded_risk_label_source_v1.yaml",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v18.md",
        "experiments/evidence/results/pivot_feedback_current_state_v4_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_current_state_v4_20260805.summary.md",
        "experiments/evidence/results/pivot_feedback_e_g1_guard_only_exception_v1_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_e_g1_guard_only_exception_v1_20260805.summary.md",
        "experiments/evidence/results/pivot_feedback_e_g1_guarded_risk_label_source_v1_20260805.summary.json",
        "experiments/evidence/results/pivot_feedback_e_g1_guarded_risk_label_source_v1_20260805.summary.md",
        "scripts/build_pivot_feedback_current_state_v4.py",
        "scripts/verify_pivot_feedback_current_state_v4.py",
        "scripts/build_pivot_feedback_e_g1_guarded_risk_label_source_v1.py",
        "scripts/verify_pivot_feedback_e_g1_guard_only_exception_v1.py",
        "scripts/verify_pivot_feedback_e_g1_guarded_risk_label_source_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_disjoint_successor_routing_closure_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_e_g1_disjoint_successor_availability_v1.yaml",
        "experiments/configs/pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v1.json",
        "docs/research/pivot-feedback-current-status-evidence-ledger-v19.md",
        "docs/research/pivot-feedback-lean-h3-successor-design-v1.md",
        "experiments/evidence/results/pivot_feedback_e_g1_disjoint_successor_availability_v1_20260806.summary.json",
        "experiments/evidence/results/pivot_feedback_e_g1_disjoint_successor_availability_v1_20260806.summary.md",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_source_intake_v1_20260806.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_source_intake_v1_20260806.summary.md",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_footer_preflight_v1_20260806.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_footer_preflight_v1_20260806.summary.md",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_taxonomy_dry_run_v1_20260806.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_taxonomy_dry_run_v1_20260806.summary.md",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_full_core_canary_v1_20260806.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_full_core_canary_v1_20260806.summary.md",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_full_core_launch_v1_20260806.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_openalex_full_core_launch_v1_20260806.summary.md",
        "scripts/inspect_pivot_feedback_lean_h3_openalex_footer_preflight_v1.py",
        "scripts/download_pivot_feedback_lean_h3_openalex_v1.py",
        "scripts/verify_pivot_feedback_lean_h3_openalex_taxonomy_dry_run_v1.py",
        "scripts/download_pivot_feedback_lean_h3_openalex_full_core_v1.py",
        "scripts/verify_pivot_feedback_lean_h3_openalex_full_core_canary_v1.py",
        "scripts/verify_pivot_feedback_e_g1_disjoint_successor_availability_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_lean_h3_result_space_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_lean_h3_result_space_v1.yaml",
        "docs/research/pivot-feedback-lean-h3-successor-design-v1.md",
        "experiments/evidence/results/pivot_feedback_lean_h3_result_space_v1_20260806.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_result_space_v1_20260806.summary.md",
        "scripts/simulate_pivot_feedback_lean_h3_result_space_v1.py",
        "scripts/verify_pivot_feedback_lean_h3_result_space_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_lean_h3_blind_challenge_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_lean_h3_blind_challenge_v1.yaml",
        "docs/research/pivot-feedback-lean-h3-successor-design-v1.md",
        "scripts/generate_pivot_feedback_lean_h3_blind_challenge_v1.py",
        "scripts/estimate_pivot_feedback_lean_h3_blind_challenge_v1.py",
        "scripts/evaluate_pivot_feedback_lean_h3_blind_challenge_v1.py",
        "scripts/verify_pivot_feedback_lean_h3_blind_challenge_v1.py",
        "experiments/evidence/results/pivot_feedback_lean_h3_blind_challenge_v1_20260806.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_blind_challenge_v1_20260806.summary.md",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_lean_h3_peak_set_development_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_lean_h3_peak_set_development_v1.yaml",
        "docs/research/pivot-feedback-lean-h3-successor-design-v1.md",
        "experiments/evidence/results/pivot_feedback_lean_h3_peak_set_development_v1_20260806.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_peak_set_development_v1_20260806.summary.md",
        "scripts/develop_pivot_feedback_lean_h3_peak_set_estimator_v1.py",
        "scripts/verify_pivot_feedback_lean_h3_peak_set_development_v1.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_lean_h3_set_identified_prospectus_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_lean_h3_set_identified_prospectus_v2.yaml",
        "docs/research/pivot-feedback-lean-h3-successor-design-v1.md",
        "experiments/evidence/results/pivot_feedback_lean_h3_set_identified_prospectus_v2_20260807.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_set_identified_prospectus_v2_20260807.summary.md",
        "scripts/prototype_pivot_feedback_lean_h3_set_identified_v2.py",
        "scripts/verify_pivot_feedback_lean_h3_set_identified_v2.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_lean_h3_stochastic_set_development_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_lean_h3_stochastic_set_development_v3.yaml",
        "docs/research/pivot-feedback-lean-h3-successor-design-v1.md",
        "experiments/evidence/results/pivot_feedback_lean_h3_stochastic_set_development_v3_20260807.summary.json",
        "experiments/evidence/results/pivot_feedback_lean_h3_stochastic_set_development_v3_20260807.summary.md",
        "scripts/develop_pivot_feedback_lean_h3_stochastic_set_v3.py",
        "scripts/verify_pivot_feedback_lean_h3_stochastic_set_v3.py",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_pivot_nabm_bridge_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-nabm-case-study-bridge-v1.md",
        "experiments/configs/pivot_nabm_case_study_bridge_v1.yaml",
    }
    assert repository_paths == inspector["PIVOT_NABM_BRIDGE_REPOSITORY_ONLY_PATHS"]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_delayed_feedback_prototype_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-delayed-feedback-learning-prototype-v1.md",
        "experiments/configs/pivot_delayed_feedback_learning_prototype_v1.yaml",
        "scripts/prototype_pivot_delayed_feedback_learning_v1.py",
        "tests/test_pivot_delayed_feedback_learning_prototype_v1.py",
    }
    assert repository_paths == inspector[
        "PIVOT_DELAYED_FEEDBACK_PROTOTYPE_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_persistence_exit_prototype_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-persistence-exit-prototype-v1.md",
        "experiments/configs/pivot_persistence_exit_prototype_v1.yaml",
        "scripts/prototype_pivot_persistence_exit_v1.py",
        "tests/test_pivot_persistence_exit_prototype_v1.py",
    }
    assert repository_paths == inspector[
        "PIVOT_PERSISTENCE_EXIT_PROTOTYPE_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_adaptive_baseline_ladder_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-adaptive-baseline-ladder-v1.md",
        "experiments/configs/pivot_adaptive_baseline_ladder_v1.yaml",
        "scripts/prototype_pivot_adaptive_baseline_ladder_v1.py",
        "tests/test_pivot_adaptive_baseline_ladder_v1.py",
    }
    assert repository_paths == inspector[
        "PIVOT_ADAPTIVE_BASELINE_LADDER_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_matched_marginal_discriminator_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-matched-marginal-discriminator-v1.md",
        "experiments/configs/pivot_matched_marginal_discriminator_v1.yaml",
        "scripts/prototype_pivot_matched_marginal_discriminator_v1.py",
        "tests/test_pivot_matched_marginal_discriminator_v1.py",
        "docs/research/pivot-matched-marginal-discriminator-v2.md",
        "experiments/configs/pivot_matched_marginal_discriminator_v2.yaml",
        "scripts/prototype_pivot_matched_marginal_discriminator_v2.py",
        "tests/test_pivot_matched_marginal_discriminator_v2.py",
        "docs/research/pivot-matched-marginal-discriminator-v3.md",
        "experiments/configs/pivot_matched_marginal_discriminator_v3.yaml",
        "scripts/prototype_pivot_matched_marginal_discriminator_v3.py",
        "tests/test_pivot_matched_marginal_discriminator_v3.py",
    }
    assert repository_paths == inspector[
        "PIVOT_MATCHED_MARGINAL_DISCRIMINATOR_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_nonlinear_policy_abm_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-nonlinear-policy-abm-v1.md",
        "experiments/configs/pivot_nonlinear_policy_abm_v1.yaml",
        "scripts/prototype_pivot_nonlinear_policy_abm_v1.py",
        "tests/test_pivot_nonlinear_policy_abm_v1.py",
        "docs/research/pivot-nonlinear-policy-abm-v2.md",
        "experiments/configs/pivot_nonlinear_policy_abm_v2.yaml",
        "scripts/prototype_pivot_nonlinear_policy_abm_v2.py",
        "tests/test_pivot_nonlinear_policy_abm_v2.py",
    }
    assert repository_paths == inspector[
        "PIVOT_NONLINEAR_POLICY_ABM_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_policy_ladder_consolidation_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-policy-ladder-consolidation-v1.md",
        "experiments/configs/pivot_policy_ladder_consolidation_v1.yaml",
        "tests/test_pivot_policy_ladder_consolidation_v1.py",
        "docs/research/pivot-policy-ladder-consolidation-v2.md",
        "experiments/configs/pivot_policy_ladder_consolidation_v2.yaml",
        "tests/test_pivot_policy_ladder_consolidation_v2.py",
        "docs/research/pivot-policy-ladder-consolidation-v3.md",
        "experiments/configs/pivot_policy_ladder_consolidation_v3.yaml",
        "tests/test_pivot_policy_ladder_consolidation_v3.py",
    }
    assert repository_paths == inspector[
        "PIVOT_POLICY_LADDER_CONSOLIDATION_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_relative_delay_cost_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-relative-delay-cost-v1.md",
        "experiments/configs/pivot_relative_delay_cost_v1.yaml",
        "scripts/prototype_pivot_relative_delay_cost_v1.py",
        "tests/test_pivot_relative_delay_cost_v1.py",
    }
    assert repository_paths == inspector[
        "PIVOT_RELATIVE_DELAY_COST_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_llm_prior_pilot_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "docs/research/pivot-llm-prior-pilot-v1.md",
        "experiments/configs/pivot_llm_prior_pilot_v1.yaml",
        "scripts/prototype_pivot_llm_prior_pilot_v1.py",
        "tests/test_pivot_llm_prior_pilot_v1.py",
        "docs/research/pivot-llm-prior-pilot-v2.md",
        "experiments/configs/pivot_llm_prior_pilot_v2.yaml",
        "scripts/prototype_pivot_llm_prior_pilot_v2.py",
        "tests/test_pivot_llm_prior_pilot_v2.py",
        "docs/research/pivot-llm-prior-pilot-v3.md",
        "experiments/configs/pivot_llm_prior_pilot_v3.yaml",
        "scripts/prototype_pivot_llm_prior_pilot_v3.py",
        "tests/test_pivot_llm_prior_pilot_v3.py",
        "docs/research/pivot-llm-prior-pilot-v4.md",
        "experiments/configs/pivot_llm_prior_pilot_v4.yaml",
        "scripts/prototype_pivot_llm_prior_pilot_v4.py",
        "tests/test_pivot_llm_prior_pilot_v4.py",
        "docs/research/pivot-llm-agent-reference-v1.md",
        "experiments/configs/pivot_llm_agent_reference_v1.yaml",
        "scripts/prototype_pivot_llm_agent_reference_v1.py",
        "tests/test_pivot_llm_agent_reference_v1.py",
        "docs/research/pivot-llm-persona-agent-v1.md",
        "experiments/configs/pivot_llm_persona_agent_v1.yaml",
        "scripts/prototype_pivot_llm_persona_agent_v1.py",
        "tests/test_pivot_llm_persona_agent_v1.py",
    }
    assert repository_paths == inspector[
        "PIVOT_LLM_PRIOR_PILOT_REPOSITORY_ONLY_PATHS"
    ]

    forbidden = inspector["_forbidden_sdist_files"]({"README.md", *repository_paths})

    assert forbidden == sorted(repository_paths)


def test_release_inspector_rejects_g0_measurement_precommitment_paths() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    repository_paths = {
        "experiments/configs/pivot_feedback_g0_measurement_freeze_v3.yaml",
        "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml",
        "experiments/configs/pivot_feedback_g0_measurement_proxy_precommitment_v1.yaml",
        "experiments/configs/pivot_feedback_g0_risk_set_weight_support_precommitment_v1.yaml",
        "experiments/configs/pivot_feedback_gate_state_v58.yaml",
        "experiments/configs/pivot_feedback_gate_state_v59.yaml",
        "experiments/configs/pivot_feedback_gate_state_v60.yaml",
        "experiments/configs/pivot_feedback_gate_state_v61.yaml",
        "docs/research/pivot-feedback-g0-measurement-freeze-v3-binding.md",
        "docs/research/pivot-feedback-g0-measurement-freeze-v4-binding.md",
        "docs/research/pivot-feedback-g0-measurement-proxy-precommitment-v1.md",
        "docs/research/pivot-feedback-g0-risk-set-weight-support-precommitment-v1.md",
        "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v3_binding_20260731.summary.json",
        "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v3_binding_20260731.summary.md",
        "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v4_binding_20260731.summary.json",
        "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v4_binding_20260731.summary.md",
        "experiments/evidence/results/pivot_feedback_g0_measurement_proxy_precommitment_closed_fixture_20260731.summary.json",
        "experiments/evidence/results/pivot_feedback_g0_measurement_proxy_precommitment_closed_fixture_20260731.summary.md",
        "experiments/evidence/results/pivot_feedback_g0_risk_set_weight_support_precommitment_closed_fixture_20260731.summary.json",
        "experiments/evidence/results/pivot_feedback_g0_risk_set_weight_support_precommitment_closed_fixture_20260731.summary.md",
        "scripts/apply_pivot_feedback_g0_measurement_proxy_precommitment_v1.py",
        "scripts/apply_pivot_feedback_g0_risk_set_weight_support_precommitment_v1.py",
        "scripts/verify_pivot_feedback_g0_measurement_freeze_v3.py",
        "scripts/verify_pivot_feedback_g0_measurement_freeze_v4.py",
        "scripts/verify_pivot_feedback_g0_measurement_proxy_precommitment_v1.py",
        "scripts/verify_pivot_feedback_g0_risk_set_weight_support_precommitment_v1.py",
        "tests/fixtures/pivot_feedback_g0_measurement_proxy_discovery_v1.expected.json",
        "tests/fixtures/pivot_feedback_g0_measurement_proxy_discovery_v1.input.json",
        "tests/fixtures/pivot_feedback_g0_risk_set_weight_support_discovery_v1.expected.json",
        "tests/fixtures/pivot_feedback_g0_risk_set_weight_support_discovery_v1.input.json",
    }
    assert repository_paths <= inspector[
        "PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS"
    ]
    names = {"README.md", *repository_paths}

    forbidden = inspector["_forbidden_sdist_files"](names)

    assert forbidden == sorted(names - {"README.md"})


def test_internal_markdown_link_inspection_catches_missing_targets() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    inspect_links = inspector["_inspect_internal_markdown_links"]
    names = {"README.md", "docs/guide.md"}
    markdown_sources = {
        "README.md": "\n".join(
            (
                "[valid](docs/guide.md#present-anchor)",
                "[missing file](docs/missing.md)",
                "[missing anchor](docs/guide.md#absent-anchor)",
                "[external](https://example.com/not-in-the-sdist)",
            )
        ),
        "docs/guide.md": "# Present Anchor\n",
    }

    issues, checked_count = inspect_links(names, markdown_sources)

    assert checked_count == 3
    assert len(issues) == 2
    assert "targets missing file 'docs/missing.md'" in issues[0]
    assert "targets missing anchor 'absent-anchor'" in issues[1]


def test_cross_artifact_metadata_check_rejects_stale_dist() -> None:
    inspector = runpy.run_path(str(INSPECT_SCRIPT))
    cross_check = inspector["_cross_artifact_metadata_issues"]
    pyproject = {
        "name": "neural-abm",
        "version": "0.1.0a6.dev0",
        "requires_python": ">=3.11",
    }
    stale = {
        "metadata_name": "other-name",
        "metadata_version": "0.1.0a7.dev0",
        "requires_python": ">=3.12",
    }

    issues = cross_check(pyproject, stale, stale)

    assert len(issues) == 6
    assert any("wheel version differs from pyproject" in issue for issue in issues)
    assert any(
        "sdist requires_python differs from pyproject" in issue for issue in issues
    )
