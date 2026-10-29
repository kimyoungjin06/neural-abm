"""Inspect built pre-release artifacts for package-boundary checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from collections import Counter
from collections.abc import Mapping
from email.parser import Parser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_DEPENDENCIES = {"numpy", "pyyaml"}
REQUIRED_PYTHON = ">=3.11"
REQUIRED_EXTRAS = {"torch", "research", "full"}
REQUIRED_WHEEL_MODULES = {
    "neural_abm/__init__.py",
    "neural_abm/api.py",
    "neural_abm/api_lite.py",
    "neural_abm/capabilities.py",
    "neural_abm/policy_ladder.py",
    "neural_abm/policy_learners.py",
    "neural_abm/policy_scoring.py",
    "neural_abm/scenario_lite.py",
    "neural_abm/social_core.py",
    "neural_abm/unit_core.py",
    "neural_abm/workflow_lite.py",
}
RESEARCH_PIVOT_SDIST_FILES = {
    "docs/case-studies/researcher-pivot/README.md",
    "docs/case-studies/researcher-pivot/data/learning_study_results.json",
    "docs/case-studies/researcher-pivot/data/study_results.json",
    "docs/case-studies/researcher-pivot/figures/fig1_productive_pivot_distributions.png",
    "docs/case-studies/researcher-pivot/figures/fig2_pivot_composition.png",
    "docs/case-studies/researcher-pivot/figures/fig3_sensitivity.png",
    "docs/case-studies/researcher-pivot/figures/fig4_readiness_trajectories.png",
    "docs/case-studies/researcher-pivot/figures/fig5_learning_failed_trajectories.png",
    "docs/case-studies/researcher-pivot/figures/fig6_learning_attention_weights.png",
    "examples/research_pivot_learning_study.py",
    "examples/research_pivot_study.py",
    "scripts/plot_research_pivot_study.py",
}
RESEARCHER_PIVOT_PUBLIC_SUBTREE_PREFIX = "docs/case-studies/researcher-pivot/"
RESEARCHER_PIVOT_PUBLIC_SUBTREE_SHA256 = {
    "docs/case-studies/researcher-pivot/README.md": (
        "5752f8e1028e242b245fe7be1460fcb9c837ffd8e361051f2b9e3e5ca3fd25a8"
    ),
    "docs/case-studies/researcher-pivot/data/learning_study_results.json": (
        "98c97cac1a4d18b0e3fc0c087d63adc717b470374aae861f87696c7389d314b8"
    ),
    "docs/case-studies/researcher-pivot/data/study_results.json": (
        "3fb32d3cb46de548f2e21f21f9295ce469c58062ca444827b1ed97ea638ea521"
    ),
    "docs/case-studies/researcher-pivot/figures/"
    "fig1_productive_pivot_distributions.png": (
        "e42c9ffff28a3987fc63fcfac1839556fdc42caad676079eb5736c56c583bc59"
    ),
    "docs/case-studies/researcher-pivot/figures/fig2_pivot_composition.png": (
        "fb541bc189ef1fdf5ff0479ce2a27bde71394871a1ddc9745c15917abe0aba6e"
    ),
    "docs/case-studies/researcher-pivot/figures/fig3_sensitivity.png": (
        "a17aa3663b06f58467652641443f3290f8b19f6e655ac05fd96828fdf28f674c"
    ),
    "docs/case-studies/researcher-pivot/figures/fig4_readiness_trajectories.png": (
        "b9fc39aaa2a9ef2dd5afdb73f3576485767fe1e850e7ea8f13ef4711893a4089"
    ),
    "docs/case-studies/researcher-pivot/figures/"
    "fig5_learning_failed_trajectories.png": (
        "68ee22398060b117380a99609faabfc068f289ae459effc0595f2707b61fbb17"
    ),
    "docs/case-studies/researcher-pivot/figures/fig6_learning_attention_weights.png": (
        "d06f49c6d09252a4b8fb029ce52715f36f16b61964a00f17f1b820a516ed148f"
    ),
}
CLASSICAL_REDUCTION_SDIST_FILES = {
    "docs/classical-reductions.md",
    "docs/figures/nabm_unit_recurrent_block.png",
    "docs/figures/nabm_unit_recurrent_block.svg",
    "examples/classical_reductions.py",
    "scripts/plot_nabm_unit_schematic.py",
}
README_LINKED_SDIST_FILES = {
    "docs/api-surface-audit.md",
    "docs/early-git-user-handoff.md",
    "docs/git-distribution-flow.md",
    "docs/package-release-boundary.md",
    "docs/pre-release-artifact-flow.md",
    "docs/release-readiness.md",
    "docs/toy-models/README.md",
    "docs/toy-models/capability-matrix.md",
    "examples/README.md",
    *CLASSICAL_REDUCTION_SDIST_FILES,
    *RESEARCH_PIVOT_SDIST_FILES,
}
REQUIRED_SDIST_FILES = {
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "docs/decisions/0015-researcher-scenario-lite-contract.md",
    "docs/decisions/0016-out-of-sample-policy-scoring-contract.md",
    "docs/decisions/0017-policy-ladder-semantics-extraction.md",
    "docs/toy-models/neural-contagion-adoption.md",
    "docs/toy-models/neural-hk-classification.md",
    "docs/toy-models/neural-opinion-rewiring.md",
    "docs/toy-models/neural-public-goods-commons.md",
    "docs/toy-models/neural-spatial-pd.md",
    "examples/first_run.py",
    "examples/minimal_api_nabm.py",
    "examples/research_pivot_scenario_lite.py",
    "examples/toy_catalog.py",
    "scripts/inspect_release_artifacts.py",
    "scripts/smoke_package_profiles.py",
    *README_LINKED_SDIST_FILES,
}
CROSS_FIELD_REPOSITORY_ONLY_PATHS = {
    "experiments/configs/openalex_g1_cross_field_feasibility_v0.yaml",
    "experiments/configs/openalex_g1_field_taxonomy_reference_v0.yaml",
    "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml",
    "scripts/audit_openalex_field_taxonomy_reference.py",
    "scripts/plan_research_pivot_field_matrix.py",
    "scripts/summarize_openalex_cross_field_feasibility.py",
}
PEER_MECHANISM_REPOSITORY_ONLY_PATHS = {
    "docs/research/peer-exposure-identification-design.md",
    "experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml",
}
PIVOT_NABM_BRIDGE_REPOSITORY_ONLY_PATHS = {
    "docs/research/pivot-nabm-case-study-bridge-v1.md",
    "experiments/configs/pivot_nabm_case_study_bridge_v1.yaml",
}
PIVOT_DELAYED_FEEDBACK_PROTOTYPE_REPOSITORY_ONLY_PATHS = {
    "docs/research/pivot-delayed-feedback-learning-prototype-v1.md",
    "experiments/configs/pivot_delayed_feedback_learning_prototype_v1.yaml",
    "scripts/prototype_pivot_delayed_feedback_learning_v1.py",
    "tests/test_pivot_delayed_feedback_learning_prototype_v1.py",
}
PIVOT_PERSISTENCE_EXIT_PROTOTYPE_REPOSITORY_ONLY_PATHS = {
    "docs/research/pivot-persistence-exit-prototype-v1.md",
    "experiments/configs/pivot_persistence_exit_prototype_v1.yaml",
    "scripts/prototype_pivot_persistence_exit_v1.py",
    "tests/test_pivot_persistence_exit_prototype_v1.py",
}
PIVOT_ADAPTIVE_BASELINE_LADDER_REPOSITORY_ONLY_PATHS = {
    "docs/research/pivot-adaptive-baseline-ladder-v1.md",
    "experiments/configs/pivot_adaptive_baseline_ladder_v1.yaml",
    "scripts/prototype_pivot_adaptive_baseline_ladder_v1.py",
    "tests/test_pivot_adaptive_baseline_ladder_v1.py",
}
PIVOT_MATCHED_MARGINAL_DISCRIMINATOR_REPOSITORY_ONLY_PATHS = {
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
PIVOT_NONLINEAR_POLICY_ABM_REPOSITORY_ONLY_PATHS = {
    "docs/research/pivot-nonlinear-policy-abm-v1.md",
    "experiments/configs/pivot_nonlinear_policy_abm_v1.yaml",
    "scripts/prototype_pivot_nonlinear_policy_abm_v1.py",
    "tests/test_pivot_nonlinear_policy_abm_v1.py",
    "docs/research/pivot-nonlinear-policy-abm-v2.md",
    "experiments/configs/pivot_nonlinear_policy_abm_v2.yaml",
    "scripts/prototype_pivot_nonlinear_policy_abm_v2.py",
    "tests/test_pivot_nonlinear_policy_abm_v2.py",
}
PIVOT_POLICY_LADDER_CONSOLIDATION_REPOSITORY_ONLY_PATHS = {
    "docs/research/pivot-policy-ladder-consolidation-v1.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v1.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v1.py",
    "docs/research/pivot-policy-ladder-consolidation-v2.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v2.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v2.py",
    "docs/research/pivot-policy-ladder-consolidation-v3.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v3.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v3.py",
    "docs/research/pivot-policy-ladder-consolidation-v4.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v4.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v4.py",
    "docs/research/pivot-policy-ladder-consolidation-v5.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v5.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v5.py",
    "docs/research/info-spread-policy-ladder-design-v1.md",
    "scripts/policy_ladder_core_v1.py",
    "scripts/prototype_info_spread_policy_ladder_v1.py",
    "tests/test_policy_ladder_core_v1.py",
    "tests/test_info_spread_policy_ladder_v1.py",
    "tests/test_info_spread_policy_ladder_d1_result.py",
    "scripts/power_info_spread_policy_ladder_v1.py",
    "tests/test_power_info_spread_policy_ladder_v1.py",
    "tests/test_info_spread_policy_ladder_d2_result.py",
    "scripts/diagnose_info_spread_d2_v1.py",
    "tests/test_diagnose_info_spread_d2_v1.py",
    "tests/test_info_spread_policy_ladder_d2_diagnosis_result.py",
    "tests/test_info_spread_policy_ladder_d2_fit_budget_result.py",
    "scripts/diagnose_info_spread_labels_v1.py",
    "tests/test_diagnose_info_spread_labels_v1.py",
    "tests/test_info_spread_policy_ladder_label_selection_result.py",
    "scripts/power_info_spread_policy_ladder_v2.py",
    "tests/test_power_info_spread_policy_ladder_v2.py",
    "tests/test_info_spread_policy_ladder_d2b_result.py",
    "scripts/power_info_spread_policy_ladder_v3.py",
    "tests/test_power_info_spread_policy_ladder_v3.py",
    "tests/test_info_spread_policy_ladder_d2c_result.py",
    "docs/research/info-spread-policy-ladder-study-v1.md",
    "experiments/configs/info_spread_policy_ladder_study_v1.yaml",
    "scripts/run_info_spread_policy_ladder_study_v1.py",
    "tests/test_info_spread_policy_ladder_study_v1.py",
    "tests/test_info_spread_policy_ladder_study_v1_d4_result.py",
    "scripts/diagnose_info_spread_null_leak_v1.py",
    "tests/test_diagnose_info_spread_null_leak_v1.py",
    "tests/test_info_spread_policy_ladder_null_leak_diagnosis_result.py",
    "docs/research/info-spread-policy-ladder-d2d-v1.md",
    "scripts/info_spread_unverified_measure_v1.py",
    "tests/test_info_spread_unverified_measure_v1.py",
    "scripts/power_info_spread_policy_ladder_v4.py",
    "tests/test_power_info_spread_policy_ladder_v4.py",
    "tests/test_info_spread_policy_ladder_d2d_result.py",
    "docs/research/info-spread-policy-ladder-study-v2.md",
    "experiments/configs/info_spread_policy_ladder_study_v2.yaml",
    "scripts/run_info_spread_policy_ladder_study_v2.py",
    "tests/test_info_spread_policy_ladder_study_v2.py",
    "docs/research/info-spread-policy-ladder-llm-arms-v1.md",
    "experiments/configs/info_spread_policy_ladder_llm_arms_v1.yaml",
    "scripts/run_info_spread_policy_ladder_llm_arms_v1.py",
    "tests/test_info_spread_policy_ladder_llm_arms_v1.py",
    "tests/test_info_spread_policy_ladder_study_v2_d4_result.py",
    "tests/test_info_spread_policy_ladder_llm_arms_d5_result.py",
    "docs/research/n14-pivot-parity-v1.md",
    "scripts/reexpress_pivot_llm_prior_v3_through_core_v1.py",
    "tests/test_reexpress_pivot_llm_prior_v3_through_core_v1.py",
    "tests/test_n14_pivot_parity_result.py",
    "docs/research/pivot-policy-ladder-consolidation-v6.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v6.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v6.py",
    "docs/research/n15-package-parity-v1.md",
    "scripts/reexpress_pivot_through_policy_ladder_package_v1.py",
    "tests/test_reexpress_pivot_through_policy_ladder_package_v1.py",
    "scripts/attest_research_pivot_study_reproduction.py",
    "tests/test_n15_package_parity_result.py",
    "scripts/plot_info_spread_policy_ladder_v1.py",
    "docs/research/info-spread-policy-ladder-transfer-v1.md",
    "experiments/configs/info_spread_policy_ladder_transfer_v1.yaml",
    "scripts/run_info_spread_policy_ladder_transfer_v1.py",
    "tests/test_info_spread_policy_ladder_transfer_v1.py",
    "tests/test_info_spread_policy_ladder_transfer_v1_result.py",
    "docs/research/info-spread-policy-ladder-flip-rate-v1.md",
    "experiments/configs/info_spread_policy_ladder_flip_rate_v1.yaml",
    "scripts/run_info_spread_policy_ladder_flip_rate_v1.py",
    "tests/test_info_spread_policy_ladder_flip_rate_v1.py",
    "tests/test_info_spread_policy_ladder_flip_rate_v1_result.py",
    "docs/research/info-spread-policy-ladder-unclipped-prior-v1.md",
    "experiments/configs/info_spread_policy_ladder_unclipped_prior_v1.yaml",
    "scripts/run_info_spread_policy_ladder_unclipped_prior_v1.py",
    "tests/test_info_spread_policy_ladder_unclipped_prior_v1.py",
    "tests/test_info_spread_policy_ladder_unclipped_prior_v1_result.py",
    "docs/research/info-spread-policy-ladder-utility-v1.md",
    "experiments/configs/info_spread_policy_ladder_utility_v1.yaml",
    "scripts/run_info_spread_policy_ladder_utility_v1.py",
    "tests/test_info_spread_policy_ladder_utility_v1.py",
    "scripts/info_spread_utility_measure_v1.py",
    "tests/test_info_spread_utility_measure_v1.py",
    "tests/test_info_spread_policy_ladder_utility_v1_result.py",
    "docs/research/pivot-policy-ladder-consolidation-v7.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v7.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v7.py",
    "scripts/plot_info_spread_policy_ladder_follow_ups_v1.py",
    "docs/research/info-spread-policy-ladder-prior-calibration-v1.md",
    "experiments/configs/info_spread_policy_ladder_prior_calibration_v1.yaml",
    "scripts/run_info_spread_policy_ladder_prior_calibration_v1.py",
    "tests/test_info_spread_policy_ladder_prior_calibration_v1.py",
    "tests/test_info_spread_policy_ladder_prior_calibration_v1_result.py",
    "docs/research/pivot-policy-ladder-consolidation-v8.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v8.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v8.py",
    "docs/research/info-spread-policy-ladder-world-calibration-v1.md",
    "experiments/configs/info_spread_policy_ladder_world_calibration_v1.yaml",
    "scripts/run_info_spread_policy_ladder_world_calibration_v1.py",
    "tests/test_info_spread_policy_ladder_world_calibration_v1.py",
    "tests/test_info_spread_policy_ladder_world_calibration_v1_result.py",
    "docs/research/pivot-policy-ladder-consolidation-v9.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v9.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v9.py",
    "docs/research/pivot-policy-ladder-consolidation-v10.md",
    "experiments/configs/pivot_policy_ladder_consolidation_v10.yaml",
    "tests/test_pivot_policy_ladder_consolidation_v10.py",
}
PIVOT_RELATIVE_DELAY_COST_REPOSITORY_ONLY_PATHS = {
    "docs/research/pivot-relative-delay-cost-v1.md",
    "experiments/configs/pivot_relative_delay_cost_v1.yaml",
    "scripts/prototype_pivot_relative_delay_cost_v1.py",
    "tests/test_pivot_relative_delay_cost_v1.py",
    "docs/research/pivot-observable-discriminant-v1.md",
    "experiments/configs/pivot_observable_discriminant_v1.yaml",
    "scripts/prototype_pivot_observable_discriminant_v1.py",
    "tests/test_pivot_observable_discriminant_v1.py",
    "docs/research/pivot-observable-confirmation-v1.md",
    "experiments/configs/pivot_observable_confirmation_v1.yaml",
    "scripts/prototype_pivot_observable_confirmation_v1.py",
    "tests/test_pivot_observable_confirmation_v1.py",
    "docs/research/pivot-observable-one-sided-v1.md",
    "experiments/configs/pivot_observable_one_sided_v1.yaml",
    "scripts/prototype_pivot_observable_one_sided_v1.py",
    "tests/test_pivot_observable_one_sided_v1.py",
    "docs/research/pivot-empirical-outcome-selection-trend-v1.md",
    "experiments/configs/pivot_empirical_outcome_selection_trend_v1.yaml",
    "scripts/prototype_pivot_empirical_outcome_selection_trend_v1.py",
    "tests/test_pivot_empirical_outcome_selection_trend_v1.py",
    "docs/research/pivot-empirical-outcome-selection-trend-v2.md",
    "experiments/configs/pivot_empirical_outcome_selection_trend_v2.yaml",
    "scripts/prototype_pivot_empirical_outcome_selection_trend_v2.py",
    "tests/test_pivot_empirical_outcome_selection_trend_v2.py",
    "docs/research/pivot-empirical-outcome-selection-trend-s1-build-v1.md",
    "experiments/configs/pivot_empirical_outcome_selection_trend_s1_build_v1.yaml",
    "scripts/build_pivot_empirical_outcome_selection_trend_s1_v1.py",
    "tests/test_pivot_empirical_outcome_selection_trend_s1_build_v1.py",
    "experiments/configs/pivot_empirical_outcome_selection_trend_s1_authorization_v1.json",
    "docs/research/pivot-empirical-model-ladder-design-v1.md",
    "docs/research/pivot-empirical-ladder-recovery-v1.md",
    "experiments/configs/pivot_empirical_ladder_recovery_v1.yaml",
    "scripts/prototype_pivot_empirical_ladder_recovery_v1.py",
    "tests/test_pivot_empirical_ladder_recovery_v1.py",
    "docs/research/pivot-empirical-ladder-recovery-v2.md",
    "experiments/configs/pivot_empirical_ladder_recovery_v2.yaml",
    "scripts/prototype_pivot_empirical_ladder_recovery_v2.py",
    "tests/test_pivot_empirical_ladder_recovery_v2.py",
    "docs/research/pivot-empirical-ladder-risk-sets-v1.md",
    "experiments/configs/pivot_empirical_ladder_risk_sets_v1.yaml",
    "scripts/build_pivot_empirical_ladder_risk_sets_v1.py",
    "tests/test_pivot_empirical_ladder_risk_sets_v1.py",
    "experiments/configs/pivot_empirical_ladder_risk_sets_authorization_v1.json",
    "docs/research/pivot-empirical-ladder-recovery-v3.md",
    "experiments/configs/pivot_empirical_ladder_recovery_v3.yaml",
    "scripts/prototype_pivot_empirical_ladder_recovery_v3.py",
    "tests/test_pivot_empirical_ladder_recovery_v3.py",
    "scripts/explore_pivot_empirical_ladder_power_levers_v1.py",
    "tests/test_explore_pivot_empirical_ladder_power_levers_v1.py",
}
PIVOT_LLM_PRIOR_PILOT_REPOSITORY_ONLY_PATHS = {
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
    "docs/research/pivot-llm-prior-entry-matched-v1.md",
    "experiments/configs/pivot_llm_prior_entry_matched_v1.yaml",
    "scripts/prototype_pivot_llm_prior_entry_matched_v1.py",
    "tests/test_pivot_llm_prior_entry_matched_v1.py",
    "tests/test_pivot_llm_prior_entry_matched_v1_result.py",
}
PIVOT_GATE_REPOSITORY_ONLY_PATHS = {
    "experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml",
    "experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml",
    "scripts/verify_pivot_feedback_g0_measurement_freeze.py",
    "scripts/verify_pivot_feedback_g_minus_1_readiness.py",
}
CONTENT_FREEZE_REPOSITORY_ONLY_PATHS = {
    "experiments/configs/openalex_g1_content_freeze_v0.yaml",
    "scripts/audit_openalex_content_freeze.py",
}
MAIN_TEMPORAL_VALIDITY_REPOSITORY_ONLY_PATHS = {
    "docs/case-studies/researcher-pivot/openalex-main-temporal-validity-audit.md",
    "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml",
    "experiments/configs/openalex_g1_main_temporal_validity_v1.yaml",
    "experiments/configs/openalex_g1_main_temporal_validity_v2.yaml",
    "scripts/audit_openalex_main_temporal_validity.py",
    "scripts/audit_openalex_main_temporal_validity_v2.py",
    "scripts/audit_openalex_main_temporal_validity_v3.py",
}
AUTHOR_HISTORY_DESIGN_REPOSITORY_ONLY_PATHS = {
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
AUTHOR_HISTORY_OPERATOR_LOCAL_BASENAMES = {
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
AUTHOR_HISTORY_OPERATOR_LOCAL_DIRECTORY_BASENAMES = {
    ".author_history_tmp",
    "author_token_history",
    "author_year_history",
    "author_history_staging",
    "author_history_checkpoints",
    "author_history_data",
    "checkpoints",
    "events",
}
CITATION_AVAILABLE_AT_DESIGN_REPOSITORY_ONLY_PATHS = {
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
CITATION_OPERATOR_LOCAL_DATA_BASENAMES = {
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
WORK_TIME_SPINE_DESIGN_REPOSITORY_ONLY_PATHS = {
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
WORK_TIME_SPINE_OPERATIONAL_REPOSITORY_ONLY_PATHS = {
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
WORK_TIME_SPINE_OPERATOR_LOCAL_DIRECTORY_BASENAMES = {
    ".work_time_spine_tmp",
    "staging",
    "spine",
}
WORK_TIME_SPINE_OPERATOR_LOCAL_PATH_FRAGMENTS = {
    "checkpoints/scan/main",
    "checkpoints/partition",
}
WORK_TIME_SPINE_OPERATOR_LOCAL_BASENAMES = {
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
RELATION_SENSITIVITY_REPOSITORY_ONLY_PATHS = {
    "docs/research/openalex-work-topic-relation-pilot-20260721.md",
    "experiments/configs/openalex_g1_w0_positive_id_sensitivity_v0.yaml",
    "experiments/configs/openalex_g1_work_topic_relation_v0.yaml",
    "experiments/configs/openalex_g1_work_topic_relation_primary_adjudication_v1.yaml",
    "scripts/adjudicate_openalex_work_topic_primary_scores.py",
    "scripts/audit_openalex_w0_positive_id_sensitivity.py",
    "scripts/audit_openalex_work_topic_relation.py",
    "scripts/verify_openalex_work_topic_primary_adjudication.py",
}
CUTOFF_PRIMITIVES_REPOSITORY_ONLY_PATHS = {
    "docs/research/pivot-feedback-cutoff-primitives-v0.md",
    "experiments/configs/pivot_feedback_cutoff_primitives_v0.yaml",
    "scripts/build_pivot_feedback_cutoff_primitives.py",
    "scripts/verify_pivot_feedback_cutoff_primitives.py",
}
PIVOT_POST_PRIMITIVES_REPOSITORY_ONLY_PATHS = {
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
    "experiments/configs/pivot_feedback_d0_structural_pilot_freeze_v1.yaml",
    "experiments/configs/pivot_feedback_d0_content_indexing_audit_v1.yaml",
    "experiments/configs/pivot_feedback_d0_content_review_v1.yaml",
    "experiments/configs/pivot_feedback_d0_measurement_abi_v1.yaml",
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
PIVOT_CANDIDATE_SUCCESSOR_REPOSITORY_ONLY_PATHS = {
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
PIVOT_RANK5_API_CLOSEOUT_REPOSITORY_ONLY_PATHS = {
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
PIVOT_RANK5_LOCAL_ABSTRACT_REPOSITORY_ONLY_PATHS = {
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
PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS = {
    "experiments/configs/pivot_feedback_current_state_v1.yaml",
    "experiments/configs/pivot_feedback_current_state_v2.yaml",
    "experiments/configs/pivot_feedback_current_state_v3.yaml",
    "experiments/configs/pivot_feedback_current_state_v4.yaml",
    "experiments/configs/pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v1.json",
    "experiments/configs/pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v2.json",
    "experiments/configs/pivot_feedback_lean_h3_openalex_full_core_transfer_authorization_v3.json",
    "experiments/configs/pivot_feedback_lean_h3_result_space_v1.yaml",
    "experiments/configs/pivot_feedback_lean_h3_blind_challenge_v1.yaml",
    "experiments/configs/pivot_feedback_lean_h3_peak_set_development_v1.yaml",
    "experiments/configs/pivot_feedback_lean_h3_set_identified_prospectus_v2.yaml",
    "experiments/configs/pivot_feedback_lean_h3_stochastic_set_development_v3.yaml",
    "experiments/configs/pivot_feedback_e_g1_disjoint_successor_availability_v1.yaml",
    "experiments/configs/pivot_feedback_e_g1_guard_only_exception_v1.yaml",
    "experiments/configs/pivot_feedback_e_g1_guarded_risk_label_source_v1.yaml",
    "experiments/configs/pivot_feedback_e_g1_execution_v1.yaml",
    "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v1.yaml",
    "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v2.yaml",
    "experiments/configs/pivot_feedback_f_atomic_partition_assignment_v1.yaml",
    "experiments/configs/pivot_feedback_f_g4_private_membership_materialization_v1.yaml",
    "experiments/configs/pivot_feedback_f_g4_private_feature_source_v1.yaml",
    "experiments/configs/pivot_feedback_f_g4_private_root_artifact_binding_v1.yaml",
    "experiments/configs/pivot_feedback_f_g4_private_explicit_list_preflight_v1.yaml",
    "experiments/configs/pivot_feedback_f_aligned_final_panel_conformance_v1.yaml",
    "experiments/configs/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.yaml",
    "experiments/configs/pivot_feedback_f_definition_theory_audit_v1.yaml",
    "experiments/configs/pivot_feedback_f_final_panel_conformance_v1.yaml",
    "experiments/configs/pivot_feedback_f_field_episode_assignment_v1.yaml",
    "experiments/configs/pivot_feedback_f_pre_g1_scientific_sections_v1.yaml",
    "experiments/configs/pivot_feedback_d0_certified_author_opportunity_rebuild_v1.yaml",
    "experiments/configs/pivot_feedback_d0_certified_leaveout_prior_peer_v1.yaml",
    "experiments/configs/pivot_feedback_d0_certified_predecision_author_measurement_v1.yaml",
    "experiments/configs/pivot_feedback_d0_certified_raw_authorship_identity_affiliation_v1.yaml",
    "experiments/configs/pivot_feedback_d0_entry_source_build_v1.yaml",
    "experiments/configs/pivot_feedback_d0_entry_source_classifier_contract_v1.yaml",
    "experiments/configs/pivot_feedback_d0_persistence_calibration_readout_v1.yaml",
    "experiments/configs/pivot_feedback_d0_persistence_outcome_contract_v1.yaml",
    "experiments/configs/pivot_feedback_d0_persistence_source_build_v1.yaml",
    "experiments/configs/pivot_feedback_d0_production_pivot_classification_v1.yaml",
    "experiments/configs/pivot_feedback_g0_measurement_freeze_v1.yaml",
    "experiments/configs/pivot_feedback_g0_measurement_freeze_v2.yaml",
    "experiments/configs/pivot_feedback_g0_measurement_freeze_v3.yaml",
    "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml",
    "experiments/configs/pivot_feedback_g0_measurement_proxy_precommitment_v1.yaml",
    "experiments/configs/pivot_feedback_g0_risk_set_weight_support_precommitment_v1.yaml",
    "experiments/configs/pivot_feedback_g0_g_minus_1a_binding_reconciliation_v1.yaml",
    "experiments/configs/pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml",
    "experiments/configs/pivot_feedback_governance_compaction_v1.yaml",
    "experiments/configs/pivot_feedback_holdout_access_seal_v1.yaml",
    "experiments/configs/pivot_feedback_g0_prerequisite_closure_plan_v1.yaml",
    "experiments/configs/pivot_feedback_g0_readiness_inventory_v1.yaml",
    "experiments/configs/pivot_feedback_researcher_pivot_definition_v2.yaml",
    "experiments/configs/pivot_feedback_gate_state_v38.yaml",
    "experiments/configs/pivot_feedback_gate_state_v39.yaml",
    "experiments/configs/pivot_feedback_gate_state_v40.yaml",
    "experiments/configs/pivot_feedback_gate_state_v41.yaml",
    "experiments/configs/pivot_feedback_gate_state_v42.yaml",
    "experiments/configs/pivot_feedback_gate_state_v43.yaml",
    "experiments/configs/pivot_feedback_gate_state_v44.yaml",
    "experiments/configs/pivot_feedback_gate_state_v45.yaml",
    "experiments/configs/pivot_feedback_gate_state_v46.yaml",
    "experiments/configs/pivot_feedback_gate_state_v47.yaml",
    "experiments/configs/pivot_feedback_gate_state_v48.yaml",
    "experiments/configs/pivot_feedback_gate_state_v49.yaml",
    "experiments/configs/pivot_feedback_gate_state_v50.yaml",
    "experiments/configs/pivot_feedback_gate_state_v51.yaml",
    "experiments/configs/pivot_feedback_gate_state_v52.yaml",
    "experiments/configs/pivot_feedback_gate_state_v53.yaml",
    "experiments/configs/pivot_feedback_gate_state_v54.yaml",
    "experiments/configs/pivot_feedback_gate_state_v55.yaml",
    "experiments/configs/pivot_feedback_gate_state_v56.yaml",
    "experiments/configs/pivot_feedback_gate_state_v57.yaml",
    "experiments/configs/pivot_feedback_gate_state_v58.yaml",
    "experiments/configs/pivot_feedback_gate_state_v59.yaml",
    "experiments/configs/pivot_feedback_gate_state_v60.yaml",
    "experiments/configs/pivot_feedback_gate_state_v61.yaml",
    "experiments/configs/pivot_feedback_holdout_production_input_resolution_v1.yaml",
    "docs/research/pivot-feedback-d0-entry-source-classifier-contract-v1.md",
    "docs/research/pivot-feedback-d0-entry-source-production-v1.md",
    "docs/research/pivot-feedback-d0-persistence-outcome-contract-v1.md",
    "docs/research/pivot-feedback-d0-persistence-calibration-readout-v1.md",
    "docs/research/pivot-feedback-d0-persistence-source-production-v1.md",
    "docs/research/pivot-feedback-d0-production-pivot-classification-v1.md",
    "docs/research/pivot-feedback-researcher-pivot-definition-v2.md",
    "docs/research/pivot-feedback-g0-readiness-inventory-v1.md",
    "docs/research/pivot-feedback-g0-measurement-freeze-v1-skeleton.md",
    "docs/research/pivot-feedback-g0-measurement-freeze-v2-binding.md",
    "docs/research/pivot-feedback-g0-measurement-freeze-v3-binding.md",
    "docs/research/pivot-feedback-g0-measurement-freeze-v4-binding.md",
    "docs/research/pivot-feedback-g0-measurement-proxy-precommitment-v1.md",
    "docs/research/pivot-feedback-g0-risk-set-weight-support-precommitment-v1.md",
    "docs/research/pivot-feedback-g0-g-minus-1a-binding-reconciliation-v1.md",
    "docs/research/pivot-feedback-g0-g1-replication-tolerance-precommitment-v1.md",
    "docs/research/pivot-feedback-governance-compaction-v1.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v1.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v2.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v3.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v4.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v5.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v6.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v7.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v8.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v9.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v10.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v11.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v12.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v13.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v14.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v15.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v16.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v17.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v18.md",
    "docs/research/pivot-feedback-current-status-evidence-ledger-v19.md",
    "docs/research/pivot-feedback-lean-h3-successor-design-v1.md",
    "docs/research/pivot-feedback-g0-prerequisite-closure-plan-v1.md",
    "docs/research/pivot-feedback-holdout-access-seal-contract-v1.md",
    "docs/research/pivot-feedback-holdout-seal-tooling-v1.md",
    "docs/research/pivot-feedback-holdout-seal-preflight-tooling-v1.md",
    "docs/research/pivot-feedback-holdout-production-input-resolution-v1.md",
    "experiments/evidence/results/pivot_feedback_d0_certified_author_opportunity_rebuild_20260730.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_certified_author_opportunity_rebuild_20260730.summary.md",
    "experiments/evidence/results/pivot_feedback_d0_certified_leaveout_prior_peer_20260730.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_certified_leaveout_prior_peer_20260730.summary.md",
    "experiments/evidence/results/pivot_feedback_d0_certified_predecision_author_measurement_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_certified_predecision_author_measurement_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_d0_certified_raw_authorship_identity_affiliation_20260730.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_certified_raw_authorship_identity_affiliation_20260730.summary.md",
    "experiments/evidence/results/pivot_feedback_d0_entry_source_classifier_contract_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_entry_source_classifier_contract_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_d0_entry_source_production_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_entry_source_production_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_d0_persistence_outcome_contract_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_persistence_outcome_contract_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_d0_persistence_calibration_readout_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_persistence_calibration_readout_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_d0_persistence_source_production_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_persistence_source_production_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_d0_production_pivot_classification_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_d0_production_pivot_classification_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_g0_readiness_inventory_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_readiness_inventory_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v1_skeleton_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v1_skeleton_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v2_binding_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v2_binding_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v3_binding_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v3_binding_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v4_binding_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_measurement_freeze_v4_binding_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_g0_measurement_proxy_precommitment_closed_fixture_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_measurement_proxy_precommitment_closed_fixture_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_g0_risk_set_weight_support_precommitment_closed_fixture_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_risk_set_weight_support_precommitment_closed_fixture_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_g0_g_minus_1a_binding_reconciliation_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_g_minus_1a_binding_reconciliation_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_g0_g1_replication_tolerance_precommitment_20260803.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_g1_replication_tolerance_precommitment_20260803.summary.md",
    "experiments/evidence/results/pivot_feedback_governance_compaction_v1_20260803.summary.json",
    "experiments/evidence/results/pivot_feedback_governance_compaction_v1_20260803.summary.md",
    "experiments/evidence/results/pivot_feedback_current_state_v1_20260803.summary.json",
    "experiments/evidence/results/pivot_feedback_current_state_v1_20260803.summary.md",
    "experiments/evidence/results/pivot_feedback_current_state_v2_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_current_state_v2_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_current_state_v4_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_current_state_v4_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_e_g1_disjoint_successor_availability_v1_20260806.summary.json",
    "experiments/evidence/results/pivot_feedback_e_g1_disjoint_successor_availability_v1_20260806.summary.md",
    "experiments/evidence/results/pivot_feedback_e_g1_guard_only_exception_v1_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_e_g1_guard_only_exception_v1_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_e_g1_guarded_risk_label_source_v1_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_e_g1_guarded_risk_label_source_v1_20260805.summary.md",
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
    "experiments/evidence/results/pivot_feedback_lean_h3_openalex_full_core_resume_v2_attempt_20261007.summary.json",
    "experiments/evidence/results/pivot_feedback_lean_h3_openalex_full_core_resume_v2_attempt_20261007.summary.md",
    "experiments/evidence/results/pivot_feedback_lean_h3_openalex_full_core_resume_v3_20261007.summary.json",
    "experiments/evidence/results/pivot_feedback_lean_h3_openalex_full_core_resume_v3_20261007.summary.md",
    "experiments/evidence/results/pivot_feedback_lean_h3_result_space_v1_20260806.summary.json",
    "experiments/evidence/results/pivot_feedback_lean_h3_result_space_v1_20260806.summary.md",
    "experiments/evidence/results/pivot_feedback_lean_h3_blind_challenge_v1_20260806.summary.json",
    "experiments/evidence/results/pivot_feedback_lean_h3_blind_challenge_v1_20260806.summary.md",
    "experiments/evidence/results/pivot_feedback_lean_h3_peak_set_development_v1_20260806.summary.json",
    "experiments/evidence/results/pivot_feedback_lean_h3_peak_set_development_v1_20260806.summary.md",
    "experiments/evidence/results/pivot_feedback_lean_h3_set_identified_prospectus_v2_20260807.summary.json",
    "experiments/evidence/results/pivot_feedback_lean_h3_set_identified_prospectus_v2_20260807.summary.md",
    "experiments/evidence/results/pivot_feedback_lean_h3_stochastic_set_development_v3_20260807.summary.json",
    "experiments/evidence/results/pivot_feedback_lean_h3_stochastic_set_development_v3_20260807.summary.md",
    "experiments/evidence/results/pivot_feedback_e_g1_execution_v1_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_e_g1_execution_v1_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v2_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_e_g1_execution_input_preflight_v2_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_f_field_episode_assignment_v1_20260803.summary.json",
    "experiments/evidence/results/pivot_feedback_f_field_episode_assignment_v1_20260803.summary.md",
    "experiments/evidence/results/pivot_feedback_f_atomic_partition_assignment_v1_20260804.summary.json",
    "experiments/evidence/results/pivot_feedback_f_atomic_partition_assignment_v1_20260804.summary.md",
    "experiments/evidence/results/pivot_feedback_f_g4_private_membership_materialization_v1_20260804.summary.json",
    "experiments/evidence/results/pivot_feedback_f_g4_private_membership_materialization_v1_20260804.summary.md",
    "experiments/evidence/results/pivot_feedback_f_g4_private_feature_source_v1_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_f_g4_private_feature_source_v1_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_f_g4_private_root_artifact_binding_v1_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_f_g4_private_root_artifact_binding_v1_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_f_g4_private_explicit_list_preflight_v1_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_f_g4_private_explicit_list_preflight_v1_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_holdout_access_seal_receipt_v1.summary.json",
    "experiments/evidence/results/pivot_feedback_holdout_access_seal_receipt_verification_v1_20260805.summary.json",
    "experiments/evidence/results/pivot_feedback_holdout_access_seal_receipt_verification_v1_20260805.summary.md",
    "experiments/evidence/results/pivot_feedback_f_aligned_final_panel_conformance_v1_20260804.summary.json",
    "experiments/evidence/results/pivot_feedback_f_aligned_final_panel_conformance_v1_20260804.summary.md",
    "experiments/evidence/results/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1_20260804.summary.json",
    "experiments/evidence/results/pivot_feedback_f_aligned_strict_incumbent_rescreen_v1_20260804.summary.md",
    "experiments/evidence/results/pivot_feedback_f_definition_theory_audit_v1_20260804.summary.json",
    "experiments/evidence/results/pivot_feedback_f_definition_theory_audit_v1_20260804.summary.md",
    "experiments/evidence/results/pivot_feedback_f_final_panel_conformance_v1_20260804.summary.json",
    "experiments/evidence/results/pivot_feedback_f_final_panel_conformance_v1_20260804.summary.md",
    "experiments/evidence/results/pivot_feedback_f_pre_g1_scientific_sections_v1_20260803.summary.json",
    "experiments/evidence/results/pivot_feedback_f_pre_g1_scientific_sections_v1_20260803.summary.md",
    "scripts/apply_pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.py",
    "scripts/verify_pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.py",
    "scripts/verify_pivot_feedback_governance_compaction_v1.py",
    "scripts/verify_pivot_feedback_current_state_v1.py",
    "scripts/build_pivot_feedback_current_state_v2.py",
    "scripts/verify_pivot_feedback_current_state_v2.py",
    "scripts/build_pivot_feedback_current_state_v3.py",
    "scripts/verify_pivot_feedback_current_state_v3.py",
    "scripts/build_pivot_feedback_current_state_v4.py",
    "scripts/verify_pivot_feedback_current_state_v4.py",
    "scripts/verify_pivot_feedback_e_g1_disjoint_successor_availability_v1.py",
    "scripts/inspect_pivot_feedback_lean_h3_openalex_footer_preflight_v1.py",
    "scripts/download_pivot_feedback_lean_h3_openalex_v1.py",
    "scripts/verify_pivot_feedback_lean_h3_openalex_taxonomy_dry_run_v1.py",
    "scripts/download_pivot_feedback_lean_h3_openalex_full_core_v1.py",
    "scripts/resume_pivot_feedback_lean_h3_openalex_full_core_v2.py",
    "tests/test_pivot_feedback_lean_h3_openalex_full_core_resume_v2.py",
    "scripts/resume_pivot_feedback_lean_h3_openalex_full_core_v3.py",
    "tests/test_pivot_feedback_lean_h3_openalex_full_core_resume_v3.py",
    "scripts/verify_pivot_feedback_lean_h3_openalex_full_core_canary_v1.py",
    "scripts/simulate_pivot_feedback_lean_h3_result_space_v1.py",
    "scripts/verify_pivot_feedback_lean_h3_result_space_v1.py",
    "scripts/generate_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/estimate_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/evaluate_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/verify_pivot_feedback_lean_h3_blind_challenge_v1.py",
    "scripts/develop_pivot_feedback_lean_h3_peak_set_estimator_v1.py",
    "scripts/verify_pivot_feedback_lean_h3_peak_set_development_v1.py",
    "scripts/prototype_pivot_feedback_lean_h3_set_identified_v2.py",
    "scripts/verify_pivot_feedback_lean_h3_set_identified_v2.py",
    "scripts/develop_pivot_feedback_lean_h3_stochastic_set_v3.py",
    "scripts/verify_pivot_feedback_lean_h3_stochastic_set_v3.py",
    "scripts/build_pivot_feedback_e_g1_guarded_risk_label_source_v1.py",
    "scripts/verify_pivot_feedback_e_g1_guard_only_exception_v1.py",
    "scripts/verify_pivot_feedback_e_g1_guarded_risk_label_source_v1.py",
    "scripts/execute_pivot_feedback_e_g1_v1.py",
    "scripts/verify_pivot_feedback_e_g1_execution_v1.py",
    "scripts/build_pivot_feedback_e_g1_execution_input_preflight_v1.py",
    "scripts/verify_pivot_feedback_e_g1_execution_input_preflight_v1.py",
    "scripts/build_pivot_feedback_e_g1_execution_input_preflight_v2.py",
    "scripts/verify_pivot_feedback_e_g1_execution_input_preflight_v2.py",
    "scripts/verify_pivot_feedback_f_pre_g1_scientific_sections_v1.py",
    "tests/fixtures/pivot_feedback_g0_g1_replication_tolerance_v1.expected.json",
    "tests/fixtures/pivot_feedback_g0_g1_replication_tolerance_v1.input.json",
    "experiments/evidence/results/pivot_feedback_g0_prerequisite_closure_plan_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_g0_prerequisite_closure_plan_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_holdout_access_seal_contract_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_holdout_access_seal_contract_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_holdout_seal_tooling_closed_fixture_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_holdout_seal_tooling_closed_fixture_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_holdout_seal_preflight_tooling_closed_fixture_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_holdout_seal_preflight_tooling_closed_fixture_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_holdout_production_input_resolution_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_holdout_production_input_resolution_20260731.summary.md",
    "experiments/evidence/results/pivot_feedback_researcher_pivot_definition_v2_20260731.summary.json",
    "experiments/evidence/results/pivot_feedback_researcher_pivot_definition_v2_20260731.summary.md",
    "scripts/build_pivot_feedback_d0_certified_author_opportunity_rebuild_v1.py",
    "scripts/build_pivot_feedback_d0_certified_leaveout_prior_peer_v1.py",
    "scripts/build_pivot_feedback_d0_certified_predecision_author_measurement_v1.py",
    "scripts/build_pivot_feedback_d0_certified_raw_authorship_identity_affiliation_v1.py",
    "scripts/build_pivot_feedback_d0_entry_source_v1.py",
    "scripts/build_pivot_feedback_d0_persistence_calibration_readout_v1.py",
    "scripts/build_pivot_feedback_d0_persistence_source_v1.py",
    "scripts/build_pivot_feedback_d0_production_pivot_classification_v1.py",
    "scripts/build_pivot_feedback_f_field_episode_assignment_v1.py",
    "scripts/build_pivot_feedback_f_atomic_partition_assignment_v1.py",
    "scripts/build_pivot_feedback_f_g4_private_membership_materialization_v1.py",
    "scripts/build_pivot_feedback_f_g4_private_feature_source_v1.py",
    "scripts/bind_pivot_feedback_f_g4_private_root_artifact_list_v1.py",
    "scripts/build_pivot_feedback_f_aligned_final_panel_conformance_v1.py",
    "scripts/build_pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.py",
    "scripts/build_pivot_feedback_f_definition_theory_audit_v1.py",
    "scripts/build_pivot_feedback_f_final_panel_conformance_v1.py",
    "scripts/verify_pivot_feedback_d0_certified_author_opportunity_rebuild_v1.py",
    "scripts/verify_pivot_feedback_d0_certified_leaveout_prior_peer_v1.py",
    "scripts/verify_pivot_feedback_d0_certified_predecision_author_measurement_v1.py",
    "scripts/verify_pivot_feedback_d0_certified_raw_authorship_identity_affiliation_v1.py",
    "scripts/verify_pivot_feedback_d0_entry_source_classifier_contract_v1.py",
    "scripts/verify_pivot_feedback_d0_entry_source_v1.py",
    "scripts/verify_pivot_feedback_d0_persistence_outcome_contract_v1.py",
    "scripts/verify_pivot_feedback_d0_persistence_calibration_readout_v1.py",
    "scripts/verify_pivot_feedback_d0_persistence_source_v1.py",
    "scripts/verify_pivot_feedback_d0_production_pivot_classification_v1.py",
    "scripts/verify_pivot_feedback_f_field_episode_assignment_v1.py",
    "scripts/verify_pivot_feedback_f_atomic_partition_assignment_v1.py",
    "scripts/verify_pivot_feedback_f_g4_private_membership_materialization_v1.py",
    "scripts/verify_pivot_feedback_f_g4_private_feature_source_v1.py",
    "scripts/verify_pivot_feedback_f_g4_private_root_artifact_binding_v1.py",
    "scripts/prepare_pivot_feedback_f_g4_private_explicit_list_preflight_v1.py",
    "scripts/verify_pivot_feedback_f_g4_private_explicit_list_preflight_v1.py",
    "scripts/verify_pivot_feedback_f_aligned_final_panel_conformance_v1.py",
    "scripts/verify_pivot_feedback_f_aligned_strict_incumbent_rescreen_v1.py",
    "scripts/verify_pivot_feedback_f_definition_theory_audit_v1.py",
    "scripts/verify_pivot_feedback_f_final_panel_conformance_v1.py",
    "scripts/verify_pivot_feedback_g0_readiness_inventory_v1.py",
    "scripts/verify_pivot_feedback_g0_measurement_freeze_v1.py",
    "scripts/verify_pivot_feedback_g0_measurement_freeze_v2.py",
    "scripts/verify_pivot_feedback_g0_measurement_freeze_v3.py",
    "scripts/verify_pivot_feedback_g0_measurement_freeze_v4.py",
    "scripts/apply_pivot_feedback_g0_measurement_proxy_precommitment_v1.py",
    "scripts/verify_pivot_feedback_g0_measurement_proxy_precommitment_v1.py",
    "scripts/apply_pivot_feedback_g0_risk_set_weight_support_precommitment_v1.py",
    "scripts/verify_pivot_feedback_g0_risk_set_weight_support_precommitment_v1.py",
    "scripts/verify_pivot_feedback_g0_g_minus_1a_binding_reconciliation_v1.py",
    "scripts/verify_pivot_feedback_g0_prerequisite_closure_plan_v1.py",
    "scripts/create_pivot_feedback_holdout_access_seal_v1.py",
    "scripts/prepare_pivot_feedback_holdout_access_seal_preflight_v1.py",
    "scripts/verify_pivot_feedback_holdout_access_seal_v1.py",
    "scripts/verify_pivot_feedback_holdout_access_seal_receipt_v1.py",
    "scripts/verify_pivot_feedback_holdout_production_input_resolution_v1.py",
    "scripts/verify_pivot_feedback_researcher_pivot_definition_v2.py",
    "tests/fixtures/pivot_feedback_g0_measurement_proxy_discovery_v1.expected.json",
    "tests/fixtures/pivot_feedback_g0_measurement_proxy_discovery_v1.input.json",
    "tests/fixtures/pivot_feedback_g0_risk_set_weight_support_discovery_v1.expected.json",
    "tests/fixtures/pivot_feedback_g0_risk_set_weight_support_discovery_v1.input.json",
}
PIVOT_HOLDOUT_SEAL_OPERATOR_LOCAL_PATH_FRAGMENTS = {
    "pivot-feedback/holdout-seal-v1",
}
PIVOT_HOLDOUT_SEAL_OPERATOR_LOCAL_BASENAMES = {
    "pivot_feedback_holdout_access_seal_preflight_v1.json",
}
PIVOT_ENTRY_SOURCE_OPERATOR_LOCAL_BASENAMES = {
    "PIVOT_ENTRY_SOURCE_BUILD.md",
    "pivot_entry_source_build.json",
    "pivot_entry_source_build.outputs.json",
    "pivot_entry_source_features.parquet",
    "pivot-feedback-d0-entry-source-v1-full-20260731.verification-v1.json",
}
PIVOT_PRODUCTION_CLASSIFICATION_OPERATOR_LOCAL_BASENAMES = {
    "PRODUCTION_PIVOT_CLASSIFICATION.md",
    "production_pivot_classification.json",
    "production_pivot_classification.outputs.json",
    "production_pivot_classification.parquet",
    "pivot-feedback-d0-production-pivot-classification-v1-20260731.verification-v1.json",
}
PIVOT_PERSISTENCE_SOURCE_OPERATOR_LOCAL_BASENAMES = {
    "PERSISTENCE_OUTCOME_SOURCE.md",
    "persistence_outcome_source.json",
    "persistence_outcome_source.outputs.json",
    "persistence_outcome_source.parquet",
    "pivot-feedback-d0-persistence-source-v1-full-20260731.verification-v1.json",
}
PIVOT_PERSISTENCE_CALIBRATION_OPERATOR_LOCAL_BASENAMES = {
    "PERSISTENCE_CALIBRATION_READOUT.md",
    "persistence_calibration_readout.json",
    "persistence_calibration_readout.outputs.json",
    "pivot-feedback-d0-persistence-calibration-readout-v1-20260731.verification-v1.json",
}
CUTOFF_PRIMITIVES_OPERATOR_LOCAL_BASENAMES = {
    "audit_plan.json",
    "pivot_feedback_cutoff_panel.json",
    "pivot_feedback_cutoff_panel.md",
    "pivot_feedback_cutoff_panel.outputs.json",
    "run_state.json",
    "work_metadata_manifest.json",
}
CUTOFF_PRIMITIVES_OPERATOR_LOCAL_DIRECTORY_BASENAMES = {
    ".cutoff_primitives_tmp",
    "work_metadata",
    "work_spools",
}
PIVOT_TOPIC_ROUTING_OPERATOR_LOCAL_BASENAMES = {
    "API_ONLY_TOPIC_ROUTING_DRY_RUN.md",
    "pivot-feedback-d0-api-only-topic-routing-v1-dry-run-20260729.verification-v1.json",
    "topic_routing_decisions.csv",
    "topic_routing_dry_run.outputs.json",
    "topic_routing_summary.csv",
}
PIVOT_CANDIDATE_SCREEN_OPERATOR_LOCAL_BASENAMES = {
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
PIVOT_CANDIDATE_SUCCESSOR_OPERATOR_LOCAL_BASENAMES = {
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
PIVOT_RANK5_API_CLOSEOUT_OPERATOR_LOCAL_BASENAMES = {
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
PIVOT_RANK5_LOCAL_ABSTRACT_OPERATOR_LOCAL_BASENAMES = {
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
FORBIDDEN_SDIST_PATHS = {
    "docs/case-studies/researcher-pivot/compute-execution-contract.md",
    "docs/case-studies/researcher-pivot/openalex-main-id-quarantine-amendment.md",
    "docs/case-studies/researcher-pivot/openalex-temporal-author-gate-inventory.md",
    "docs/case-studies/researcher-pivot/pivot-feedback-empirical-nabm-protocol.md",
    "docs/release-candidate-dry-run.md",
    "scripts/audit_openalex_sentinel_incidence.py",
    "scripts/audit_openalex_source_inventory.py",
    "scripts/audit_openalex_work_source_membership.py",
    "scripts/recount_openalex_g1.py",
    "scripts/recount_openalex_g1_canonical_quarantine.py",
    "scripts/reproduce_early_git.py",
    *CROSS_FIELD_REPOSITORY_ONLY_PATHS,
    *PEER_MECHANISM_REPOSITORY_ONLY_PATHS,
    *PIVOT_NABM_BRIDGE_REPOSITORY_ONLY_PATHS,
    *PIVOT_DELAYED_FEEDBACK_PROTOTYPE_REPOSITORY_ONLY_PATHS,
    *PIVOT_PERSISTENCE_EXIT_PROTOTYPE_REPOSITORY_ONLY_PATHS,
    *PIVOT_ADAPTIVE_BASELINE_LADDER_REPOSITORY_ONLY_PATHS,
    *PIVOT_MATCHED_MARGINAL_DISCRIMINATOR_REPOSITORY_ONLY_PATHS,
    *PIVOT_NONLINEAR_POLICY_ABM_REPOSITORY_ONLY_PATHS,
    *PIVOT_POLICY_LADDER_CONSOLIDATION_REPOSITORY_ONLY_PATHS,
    *PIVOT_RELATIVE_DELAY_COST_REPOSITORY_ONLY_PATHS,
    *PIVOT_LLM_PRIOR_PILOT_REPOSITORY_ONLY_PATHS,
    *PIVOT_GATE_REPOSITORY_ONLY_PATHS,
    *CONTENT_FREEZE_REPOSITORY_ONLY_PATHS,
    *MAIN_TEMPORAL_VALIDITY_REPOSITORY_ONLY_PATHS,
    *AUTHOR_HISTORY_DESIGN_REPOSITORY_ONLY_PATHS,
    *CITATION_AVAILABLE_AT_DESIGN_REPOSITORY_ONLY_PATHS,
    *WORK_TIME_SPINE_DESIGN_REPOSITORY_ONLY_PATHS,
    *WORK_TIME_SPINE_OPERATIONAL_REPOSITORY_ONLY_PATHS,
    *RELATION_SENSITIVITY_REPOSITORY_ONLY_PATHS,
    *CUTOFF_PRIMITIVES_REPOSITORY_ONLY_PATHS,
    *PIVOT_POST_PRIMITIVES_REPOSITORY_ONLY_PATHS,
    *PIVOT_CANDIDATE_SUCCESSOR_REPOSITORY_ONLY_PATHS,
    *PIVOT_RANK5_API_CLOSEOUT_REPOSITORY_ONLY_PATHS,
    *PIVOT_RANK5_LOCAL_ABSTRACT_REPOSITORY_ONLY_PATHS,
    *PIVOT_CERTIFIED_D0_MEASUREMENT_REPOSITORY_ONLY_PATHS,
}
FORBIDDEN_SDIST_PREFIXES = (
    "archive/",
    "experiments/",
    "paper/",
    "ref/",
    "docs/nabm-unit-v1",
)
MARKDOWN_LINK_PATTERN = re.compile(
    r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect wheel/sdist metadata for a pre-release artifact.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build artifacts with uv build into --dist-dir first.",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=ROOT / "dist",
        help="Directory containing built wheel and sdist artifacts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print compact JSON only.",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="With --build, build into a temporary directory instead of dist/.",
    )
    args = parser.parse_args()

    if args.build and args.ephemeral:
        with tempfile.TemporaryDirectory(prefix="neural-abm-release-inspect.") as temp:
            dist_dir = Path(temp)
            _build_dist(dist_dir)
            report = inspect_release_artifacts(dist_dir)
    elif args.build:
        _build_dist(args.dist_dir)
        report = inspect_release_artifacts(args.dist_dir)
    else:
        report = inspect_release_artifacts(args.dist_dir)

    text = json.dumps(report, indent=None if args.json else 2, sort_keys=True)
    print(text)
    if report["blocking_issues"]:
        raise SystemExit(1)


def inspect_release_artifacts(dist_dir: Path) -> dict[str, Any]:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    wheel = _single_artifact(dist_dir, "*.whl")
    sdist = _single_artifact(dist_dir, "*.tar.gz")
    wheel_report = _inspect_wheel(wheel)
    sdist_report = _inspect_sdist(sdist)
    pyproject_report = _inspect_pyproject(pyproject)
    cross_artifact_issues = _cross_artifact_metadata_issues(
        pyproject_report["summary"],
        wheel_report["summary"],
        sdist_report["summary"],
    )
    blocking_issues = [
        *pyproject_report["blocking_issues"],
        *wheel_report["blocking_issues"],
        *sdist_report["blocking_issues"],
        *cross_artifact_issues,
    ]
    warnings = [
        *pyproject_report["warnings"],
        *wheel_report["warnings"],
        *sdist_report["warnings"],
    ]
    return {
        "status": "pass" if not blocking_issues else "fail",
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "pyproject": pyproject_report["summary"],
        "wheel": wheel_report["summary"],
        "sdist": sdist_report["summary"],
    }


def _cross_artifact_metadata_issues(
    pyproject: Mapping[str, Any],
    wheel: Mapping[str, Any],
    sdist: Mapping[str, Any],
) -> list[str]:
    """Reject stale artifacts whose identity differs from the source project."""

    issues: list[str] = []
    expected = {
        "name": pyproject["name"],
        "version": pyproject["version"],
        "requires_python": pyproject["requires_python"],
    }
    for artifact, summary, keys in (
        (
            "wheel",
            wheel,
            {
                "name": "metadata_name",
                "version": "metadata_version",
                "requires_python": "requires_python",
            },
        ),
        (
            "sdist",
            sdist,
            {
                "name": "metadata_name",
                "version": "metadata_version",
                "requires_python": "requires_python",
            },
        ),
    ):
        for field, summary_key in keys.items():
            actual = summary.get(summary_key)
            if actual != expected[field]:
                issues.append(
                    f"{artifact} {field} differs from pyproject: "
                    f"{actual!r} != {expected[field]!r}"
                )
    return issues


def _build_dist(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("neural_abm-*.whl", "neural_abm-*.tar.gz"):
        for artifact in output_dir.glob(pattern):
            artifact.unlink()
    subprocess.run(
        ["uv", "build", "--out-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
    )


def _single_artifact(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {pattern!r} artifact in {dist_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _inspect_pyproject(pyproject: dict[str, Any]) -> dict[str, Any]:
    project = pyproject["project"]
    optional_dependencies = project["optional-dependencies"]
    default_dependency_names = {
        _dependency_name(requirement) for requirement in project["dependencies"]
    }
    missing_extras = sorted(REQUIRED_EXTRAS - set(optional_dependencies))
    blocking_issues: list[str] = []
    warnings: list[str] = []

    if default_dependency_names != DEFAULT_DEPENDENCIES:
        blocking_issues.append(
            "default dependencies must stay limited to numpy and pyyaml"
        )
    if missing_extras:
        blocking_issues.append(f"missing required extras: {', '.join(missing_extras)}")
    if project.get("requires-python") != REQUIRED_PYTHON:
        blocking_issues.append(f"requires-python must stay at {REQUIRED_PYTHON}")
    if "license" not in project:
        warnings.append("project.license is not set; choose a license before release")
    if "urls" not in project:
        warnings.append("project.urls is not set; add project links before release")
    version = project.get("version", "")
    if not _is_pre_release_version(version):
        warnings.append(
            "version is not marked as an alpha, beta, rc, or dev release; "
            "reserve final versions for intentional public releases"
        )

    return {
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "summary": {
            "name": project["name"],
            "version": project["version"],
            "requires_python": project["requires-python"],
            "default_dependencies": sorted(default_dependency_names),
            "extras": sorted(optional_dependencies),
            "has_authors": bool(project.get("authors")),
            "has_classifiers": bool(project.get("classifiers")),
            "has_keywords": bool(project.get("keywords")),
            "has_license": "license" in project,
            "has_urls": "urls" in project,
        },
    }


def _inspect_wheel(wheel: Path) -> dict[str, Any]:
    blocking_issues: list[str] = []
    warnings: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(
            archive.read(metadata_name).decode("utf-8", errors="replace")
        )
        runtime_version = _read_runtime_version(
            archive.read("neural_abm/__init__.py").decode("utf-8", errors="replace")
        )

    missing_modules = sorted(REQUIRED_WHEEL_MODULES - names)
    if missing_modules:
        blocking_issues.append(
            f"wheel missing required modules: {', '.join(missing_modules)}"
        )
    requires_dist = metadata.get_all("Requires-Dist") or []
    default_requires = sorted(
        {
            _dependency_name(requirement)
            for requirement in requires_dist
            if "extra ==" not in requirement
        }
    )
    extras = sorted(set(metadata.get_all("Provides-Extra") or []))
    if set(default_requires) != DEFAULT_DEPENDENCIES:
        blocking_issues.append(
            "wheel default Requires-Dist must stay limited to numpy and pyyaml"
        )
    if REQUIRED_EXTRAS - set(extras):
        blocking_issues.append(
            "wheel metadata missing required extras: "
            f"{', '.join(sorted(REQUIRED_EXTRAS - set(extras)))}"
        )
    if metadata.get("Description-Content-Type") != "text/markdown":
        warnings.append("wheel README metadata is not marked as text/markdown")
    metadata_version = metadata.get("Version")
    if runtime_version is None:
        blocking_issues.append("wheel package does not declare neural_abm.__version__")
    elif runtime_version != metadata_version:
        blocking_issues.append(
            "wheel metadata version and neural_abm.__version__ differ: "
            f"{metadata_version!r} != {runtime_version!r}"
        )

    return {
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "summary": {
            "path": str(wheel),
            "metadata_name": metadata.get("Name"),
            "metadata_version": metadata_version,
            "runtime_version": runtime_version,
            "requires_python": metadata.get("Requires-Python"),
            "default_requires": default_requires,
            "extras": extras,
            "required_modules": sorted(REQUIRED_WHEEL_MODULES),
            "required_modules_present": not missing_modules,
            "description_content_type": metadata.get("Description-Content-Type"),
        },
    }


def _inspect_sdist(sdist: Path) -> dict[str, Any]:
    blocking_issues: list[str] = []
    metadata = None
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        member_boundary_issues = _sdist_member_boundary_issues(members)
        names: set[str] = set()
        markdown_sources: dict[str, str] = {}
        researcher_pivot_sha256: dict[str, str] = {}
        researcher_pivot_nonregular: list[str] = []
        for member in members:
            name = _strip_sdist_root(member.name)
            if name.startswith(RESEARCHER_PIVOT_PUBLIC_SUBTREE_PREFIX):
                if not member.isfile() and not member.isdir():
                    researcher_pivot_nonregular.append(name)
            if not member.isfile():
                continue
            names.add(name)
            payload: bytes | None = None
            if name.startswith(RESEARCHER_PIVOT_PUBLIC_SUBTREE_PREFIX):
                extracted = archive.extractfile(member)
                if extracted is None:
                    researcher_pivot_nonregular.append(name)
                    continue
                payload = extracted.read()
                researcher_pivot_sha256[name] = hashlib.sha256(payload).hexdigest()
            if name == "PKG-INFO":
                extracted = archive.extractfile(member)
                if extracted is not None:
                    metadata = Parser().parsestr(
                        extracted.read().decode("utf-8", errors="replace")
                    )
                continue
            if not name.endswith(".md"):
                continue
            if payload is None:
                extracted = archive.extractfile(member)
                if extracted is not None:
                    payload = extracted.read()
            if payload is not None:
                markdown_sources[name] = payload.decode("utf-8", errors="replace")
    if metadata is None:
        blocking_issues.append("sdist is missing PKG-INFO metadata")
    if member_boundary_issues:
        blocking_issues.append(
            "sdist has noncanonical, duplicate, multi-root, or non-regular members: "
            f"{'; '.join(member_boundary_issues[:10])}"
        )
    missing_files = sorted(REQUIRED_SDIST_FILES - names)
    if missing_files:
        blocking_issues.append(
            f"sdist missing required files: {', '.join(missing_files)}"
        )
    broken_internal_links, checked_link_count = _inspect_internal_markdown_links(
        names,
        markdown_sources,
    )
    if broken_internal_links:
        blocking_issues.append(
            "sdist contains broken internal Markdown links: "
            f"{'; '.join(broken_internal_links[:10])}"
        )
    forbidden_files = _forbidden_sdist_files(names)
    if forbidden_files:
        blocking_issues.append(
            f"sdist includes internal-history files: {', '.join(forbidden_files[:10])}"
        )
    researcher_pivot_issues = _researcher_pivot_boundary_issues(
        names,
        researcher_pivot_sha256,
        researcher_pivot_nonregular,
    )
    if researcher_pivot_issues:
        blocking_issues.append(
            "sdist violates the researcher-pivot exact public boundary: "
            f"{'; '.join(researcher_pivot_issues[:10])}"
        )
    return {
        "blocking_issues": blocking_issues,
        "warnings": [],
        "summary": {
            "path": str(sdist),
            "metadata_name": None if metadata is None else metadata.get("Name"),
            "metadata_version": None if metadata is None else metadata.get("Version"),
            "requires_python": (
                None if metadata is None else metadata.get("Requires-Python")
            ),
            "required_files": sorted(REQUIRED_SDIST_FILES),
            "required_files_present": not missing_files,
            "missing_required_files": missing_files,
            "internal_links_checked": checked_link_count,
            "internal_links_ok": not broken_internal_links,
            "broken_internal_links": broken_internal_links,
            "forbidden_files_present": bool(forbidden_files),
            "forbidden_files": forbidden_files,
            "member_boundary_ok": not member_boundary_issues,
            "member_boundary_issues": member_boundary_issues,
            "researcher_pivot_boundary_ok": not researcher_pivot_issues,
            "researcher_pivot_boundary_issues": researcher_pivot_issues,
        },
    }


def _sdist_member_boundary_issues(members: list[tarfile.TarInfo]) -> list[str]:
    issues: list[str] = []
    raw_names = [member.name for member in members]
    raw_counts = Counter(raw_names)
    stripped_names: list[str] = []
    top_level_roots: set[str] = set()

    for member in members:
        raw_name = member.name
        if not raw_name or raw_name.startswith("/"):
            issues.append(f"absolute or empty raw member name {raw_name!r}")
            continue
        if "\\" in raw_name:
            issues.append(f"backslash in raw member name {raw_name!r}")
        if any(ord(character) < 32 or ord(character) == 127 for character in raw_name):
            issues.append(f"control character in raw member name {raw_name!r}")
        parts = raw_name.split("/")
        if len(parts) < 2:
            issues.append(f"raw member lacks a single sdist root {raw_name!r}")
            continue
        top_level_roots.add(parts[0])
        if any(part in {"", ".", ".."} for part in parts):
            issues.append(f"noncanonical path segment in raw member name {raw_name!r}")
        if posixpath.normpath(raw_name) != raw_name:
            issues.append(f"noncanonical POSIX spelling for raw member {raw_name!r}")
        stripped_name = "/".join(parts[1:])
        stripped_names.append(stripped_name)
        if not stripped_name:
            issues.append(f"root-only archive member {raw_name!r}")
        if not member.isfile():
            issues.append(f"non-regular archive member {raw_name!r}")

    if len(top_level_roots) != 1:
        issues.append(
            "archive must contain exactly one top-level sdist root; found "
            f"{sorted(top_level_roots)!r}"
        )
    for raw_name, count in sorted(raw_counts.items()):
        if count != 1:
            issues.append(f"duplicate raw archive member {raw_name!r} count={count}")
    for stripped_name, count in sorted(Counter(stripped_names).items()):
        if count != 1:
            issues.append(
                f"duplicate root-stripped archive member {stripped_name!r} count={count}"
            )
    return issues


def _researcher_pivot_boundary_issues(
    names: set[str],
    observed_sha256: Mapping[str, str],
    nonregular_members: list[str],
) -> list[str]:
    expected = set(RESEARCHER_PIVOT_PUBLIC_SUBTREE_SHA256)
    noncanonical_names = {
        name
        for name in names
        if "\\" in name
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or posixpath.normpath(name) != name
    }
    observed = {
        name
        for name in names
        if name.startswith(RESEARCHER_PIVOT_PUBLIC_SUBTREE_PREFIX)
    }
    issues: list[str] = []
    for name in sorted(noncanonical_names):
        issues.append(f"noncanonical root-stripped member {name!r}")
    for name in sorted(observed - expected):
        issues.append(f"unexpected member {name!r}")
    for name in sorted(expected - observed):
        issues.append(f"missing allowlisted member {name!r}")
    for name in sorted(expected & observed):
        expected_digest = RESEARCHER_PIVOT_PUBLIC_SUBTREE_SHA256[name]
        actual_digest = observed_sha256.get(name)
        if actual_digest != expected_digest:
            issues.append(
                f"content SHA-256 mismatch for {name!r}: "
                f"{actual_digest!r} != {expected_digest!r}"
            )
    for name in sorted(set(nonregular_members)):
        issues.append(f"non-regular archive member {name!r}")
    return issues


def _forbidden_sdist_files(names: set[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name in FORBIDDEN_SDIST_PATHS
        or any(name.startswith(prefix) for prefix in FORBIDDEN_SDIST_PREFIXES)
        or any(
            part in CITATION_OPERATOR_LOCAL_DATA_BASENAMES for part in name.split("/")
        )
        or any(
            part in AUTHOR_HISTORY_OPERATOR_LOCAL_BASENAMES for part in name.split("/")
        )
        or any(
            part in AUTHOR_HISTORY_OPERATOR_LOCAL_DIRECTORY_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in WORK_TIME_SPINE_OPERATOR_LOCAL_DIRECTORY_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in WORK_TIME_SPINE_OPERATOR_LOCAL_BASENAMES for part in name.split("/")
        )
        or any(
            f"/{fragment}/" in f"/{name.strip('/')}/"
            for fragment in WORK_TIME_SPINE_OPERATOR_LOCAL_PATH_FRAGMENTS
        )
        or any(
            f"/{fragment}/" in f"/{name.strip('/')}/"
            for fragment in PIVOT_HOLDOUT_SEAL_OPERATOR_LOCAL_PATH_FRAGMENTS
        )
        or any(
            part in PIVOT_HOLDOUT_SEAL_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in CUTOFF_PRIMITIVES_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in CUTOFF_PRIMITIVES_OPERATOR_LOCAL_DIRECTORY_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in PIVOT_TOPIC_ROUTING_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in PIVOT_CANDIDATE_SCREEN_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in PIVOT_CANDIDATE_SUCCESSOR_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in PIVOT_RANK5_API_CLOSEOUT_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in PIVOT_RANK5_LOCAL_ABSTRACT_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in PIVOT_ENTRY_SOURCE_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in PIVOT_PRODUCTION_CLASSIFICATION_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in PIVOT_PERSISTENCE_SOURCE_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
        or any(
            part in PIVOT_PERSISTENCE_CALIBRATION_OPERATOR_LOCAL_BASENAMES
            for part in name.split("/")
        )
    )


def _inspect_internal_markdown_links(
    names: set[str],
    markdown_sources: Mapping[str, str],
) -> tuple[list[str], int]:
    """Return broken local Markdown links from the files inside an sdist."""

    issues: list[str] = []
    checked_count = 0
    anchors_by_path = {
        path: _markdown_anchors(source) for path, source in markdown_sources.items()
    }
    for source_path, source in sorted(markdown_sources.items()):
        for match in MARKDOWN_LINK_PATTERN.finditer(source):
            target = (match.group(1) or match.group(2)).strip()
            try:
                parsed = urlsplit(target)
            except ValueError:
                issues.append(f"{source_path}: invalid link target {target!r}")
                checked_count += 1
                continue
            if parsed.scheme or parsed.netloc or target.startswith("//"):
                continue
            checked_count += 1
            link_path = unquote(parsed.path)
            if link_path:
                if link_path.startswith("/"):
                    resolved = posixpath.normpath(link_path.lstrip("/"))
                else:
                    resolved = posixpath.normpath(
                        posixpath.join(posixpath.dirname(source_path), link_path)
                    )
            else:
                resolved = source_path
            if resolved == ".." or resolved.startswith("../"):
                issues.append(
                    f"{source_path}: {target!r} escapes the source distribution"
                )
                continue
            if resolved not in names:
                issues.append(
                    f"{source_path}: {target!r} targets missing file {resolved!r}"
                )
                continue
            fragment = unquote(parsed.fragment)
            if fragment and resolved.endswith(".md"):
                anchors = anchors_by_path.get(resolved, set())
                if fragment not in anchors:
                    issues.append(
                        f"{source_path}: {target!r} targets missing anchor "
                        f"{fragment!r} in {resolved!r}"
                    )
    return issues, checked_count


def _markdown_anchors(source: str) -> set[str]:
    """Build the GitHub-style heading anchors needed by local doc links."""

    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in source.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match is None:
            continue
        heading = re.sub(r"<[^>]+>", "", match.group(1)).strip().lower()
        slug = re.sub(r"[^\w\- ]", "", heading)
        slug = re.sub(r"\s+", "-", slug)
        duplicate_index = counts.get(slug, 0)
        counts[slug] = duplicate_index + 1
        anchors.add(slug if duplicate_index == 0 else f"{slug}-{duplicate_index}")
    return anchors


def _dependency_name(requirement: str) -> str:
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise ValueError(f"could not parse dependency name from {requirement!r}")
    return match.group(1).lower()


def _read_runtime_version(source: str) -> str | None:
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    return match.group(1) if match is not None else None


def _is_pre_release_version(version: str) -> bool:
    return bool(re.search(r"(a|b|rc|dev)\d*$", version))


def _strip_sdist_root(path: str) -> str:
    _, separator, relative = path.partition("/")
    return relative if separator else path


if __name__ == "__main__":
    main()
