from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


def _plain(text: str) -> str:
    return " ".join(text.split())


def test_nabm_unit_readme_records_v1_contract_freeze() -> None:
    readme = (ROOT / "src/neural_abm/README.md").read_text()

    assert "## NABM Unit v1 Contract Freeze" in readme
    assert "lifecycle, typed exchange, backend dispatch" in readme
    assert "explicit contract-gap remediation" in readme
    assert "must not construct" in readme
    for forbidden_domain_semantic in (
        "rewards",
        "payoffs",
        "thresholds",
        "teacher signals",
        "basin credit",
        "readiness meaning",
        "revision pressure meaning",
        "evidence criteria",
    ):
        assert forbidden_domain_semantic in readme
    for boundary_doc in (
        "docs/decisions/0010-nabm-unit-v1-contract.md",
        "docs/decisions/0011-continuous-scalar-unit-contract.md",
        "docs/decisions/0012-existing-toy-migration-parity-consolidation.md",
        "docs/decisions/0013-public-api-v0-contract.md",
        "docs/decisions/0014-package-dependency-policy.md",
        "docs/nabm-unit-v1-boundary-audit.md",
        "docs/api-surface-audit.md",
        "docs/nabm-unit-v1-completeness-checklist.md",
        "docs/nabm-unit-v1-migration-candidate-audit.md",
        "docs/nabm-unit-v1-runner-lifecycle-audit.md",
    ):
        assert boundary_doc in readme
    assert "tests/test_toy8_runner.py" in readme
    assert "tests/test_toy9_runner.py" in readme
    assert "tests/test_toy7_runner.py" in readme
    assert "tests/test_toy10_runner.py" in readme
    assert "tests/test_toy6_runner.py" in readme
    assert "tests/test_domain_toy_artifact_contracts.py" in readme
    assert "unit-backed scalar path" in readme
    assert "SCALAR_PROBABILITY_CHANNEL" in readme
    assert "BOUNDED_SCALAR_CHANNEL" in readme
    assert "mix_bounded_scalars" in readme
    assert "without using probability semantics" in readme
    assert "dynamic rewiring stay" in readme
    assert "probability-distribution path" in readme
    assert "typed social exchange reuse" in readme
    assert "not an upgrade of Toy6-10 to full NABM status" in readme
    assert "DomainToyRunner" in readme
    assert "step(...) phase ordering remains" in readme
    assert "domain_social_diagnostics" in readme
    assert "aggregate_social_diagnostic_fields" in readme
    assert "micro_social_diagnostic_fields" in readme
    assert "make_domain_run_dir" in readme
    assert "write_domain_run_metadata" in readme
    assert "api_lite" in readme


def test_nabm_unit_checklist_records_gate1_completion_and_guards() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    gate1 = _between(
        checklist,
        "### Gate 1: Unit Contract Freeze",
        "### Gate 2: Hard Holdout Expansion",
    )

    assert "Status: first pass complete." in gate1
    assert "tests/test_nabm_unit_docs.py" in gate1
    for guard_test in (
        "tests/test_spatial_binary_runner.py",
        "tests/test_toy2_runner.py",
        "tests/test_toy4_runner.py",
        "tests/test_toy5_runner.py",
        "tests/test_nabm_unit_adapter_holdout.py",
        "tests/test_readiness.py",
    ):
        assert guard_test in gate1
    assert "Any future generic unit API expansion should update Decision 0010" in gate1


def test_paper_claim_matrix_records_gate4_boundaries() -> None:
    claim_matrix = (ROOT / "paper/claim-matrix.md").read_text()
    table_candidates = (
        ROOT / "paper/tables/nabm-unit-v1-manuscript-tables.md"
    ).read_text()
    section_3 = (ROOT / "paper/sections/03-neural-abm-node.md").read_text()
    section_6 = (
        ROOT / "paper/sections/06-calibration-and-analysis.md"
    ).read_text()
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()

    for supported_claim_artifact in (
        "docs/decisions/0010-nabm-unit-v1-contract.md",
        "toy5_neural_threshold_target_threshold_aware_grid_quick.yaml",
        "toy24_gate3_evidence_triage_findings.md",
        "toy24_precommitment_peer_evidence_reputation_fragility_stress_quick.yaml",
        "toy4_resource_threshold_heterogeneous_local_observation_stress_quick.yaml",
    ):
        assert supported_claim_artifact in claim_matrix

    for limitation in (
        "not a claim that NABM policies generally outperform classical ABM rules",
        "targeted baseline-fragility evidence",
        "Clean reputation imitation remains `5/5` and faster",
        "Unsupported",
    ):
        assert limitation in claim_matrix

    for table_anchor in (
        "Table 1",
        "Table 2",
        "Table 3",
        "Table 4",
        "Table 5",
        "Table 6",
        "Table 7",
    ):
        assert f"nabm-unit-v1-manuscript-tables.md` {table_anchor}" in claim_matrix
        assert f"## {table_anchor}:" in table_candidates

    for table_value in (
        "No-seed heterogeneous safety",
        "revision_operator_quick",
        "reputation_imitation_open_sparse_noisy_p0p1_s1p0",
        "rev_local_sustain_obs_noisy_s2p0_hetero",
        "targeted baseline-fragility evidence",
        "adapter_only_congestion_holdout_quick",
        "adapter_only_stochastic_commons_quick",
    ):
        assert table_value in table_candidates

    for architecture_boundary in (
        "neural policy generally outperforms Fermi",
        "Moving Toy-specific rewards",
        "evidence criteria into the generic unit",
        "binary policy, revision, readiness, backend commit, and diagnostic lifecycles",
    ):
        assert architecture_boundary in section_3

    for evidence_boundary in (
        "final ceiling hits",
        "ever ceiling hits",
        "mean time-to-ceiling",
        "targeted baseline-fragility evidence, not a general demonstration",
        "does not yet support general neural dominance",
        "zero capacity error",
        "local-resource main avoids collapse",
    ):
        assert evidence_boundary in section_6

    gate4 = _between(
        checklist,
        "### Gate 4: Manuscript Claim Matrix",
        "## Recommended Next Slice",
    )
    assert "Status: first pass complete." in gate4
    assert "paper/claim-matrix.md" in gate4
    assert "paper/tables/nabm-unit-v1-manuscript-tables.md" in gate4
    assert "paper/sections/03-neural-abm-node.md" in gate4
    assert "paper/sections/06-calibration-and-analysis.md" in gate4
    assert "Deferred claims are explicit" in gate4


def test_nabm_unit_checklist_records_adapter_only_holdout_smoke() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    readme = (ROOT / "src/neural_abm/README.md").read_text()
    gate5 = _between(
        checklist,
        "### Gate 5: Adapter-Only Holdout Evidence",
        "### Gate 6: Stochastic Endogenous Holdout Evidence",
    )

    assert "Status: quick evidence complete." in gate5
    assert "tests/test_nabm_unit_adapter_holdout.py" in gate5
    assert "without changing `src/neural_abm`" in gate5
    assert "adapter-threshold-readiness main reaches full adoption" in gate5
    assert "tiny scripted binary" in gate5
    assert "tests/test_nabm_unit_adapter_holdout.py" in readme


def test_nabm_unit_checklist_records_stochastic_endogenous_holdout() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    gate6 = _between(
        checklist,
        "### Gate 6: Stochastic Endogenous Holdout Evidence",
        "### Gate 7A: Existing-Toy Migration Candidate Audit",
    )

    assert "Status: quick evidence complete." in gate6
    assert "adapter_only_stochastic_commons_quick.yaml" in gate6
    assert "run_adapter_stochastic_commons_holdout_evidence.py" in gate6
    assert "actions deplete local resources" in gate6
    assert "local-resource main avoids collapse" in gate6
    assert "compact scripted binary commons" in gate6


def test_nabm_unit_migration_candidate_audit_selects_toy8() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-migration-candidate-audit.md").read_text()
    gate7 = _between(
        checklist,
        "### Gate 7A: Existing-Toy Migration Candidate Audit",
        "### Gate 7B: Toy8 Async Social-Hazard Parity",
    )

    for required in (
        "Toy8 async social-hazard parity",
        "Toy9 heterogeneous binary probability mixing",
        "Toy7 continuous resource intensity",
        "migration parity plus diagnostics boundary",
    ):
        assert required in gate7

    for audited_path in (
        "src/neural_abm/toy_async.py",
        "src/neural_abm/toy_resource.py",
        "src/neural_abm/toy_heterogeneous.py",
    ):
        assert audited_path in audit

    assert "Proceed with **Toy8 async social-hazard parity**" in audit
    assert "Do not move `ScheduledEvent`" in audit
    assert "Fallback rule:" in audit
    assert "continuous scalar policy lifecycle" in audit


def test_nabm_unit_checklist_records_toy8_social_hazard_parity() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-migration-candidate-audit.md").read_text()
    gate7b = _between(
        checklist,
        "### Gate 7B: Toy8 Async Social-Hazard Parity",
        "### Gate 7C: Toy9 Heterogeneous Probability Parity",
    )

    for required in (
        "Status: parity slice complete.",
        "src/neural_abm/mixers.py::apply_scalar_output_average",
        "activation_propensity",
        "event_hazard_commit",
        "not new Toy8 performance evidence",
    ):
        assert required in gate7b

    assert "## Gate 7B Result" in audit
    assert "not Toy8 performance evidence" in audit
    assert "does not promote Toy8 from" in audit


def test_nabm_unit_checklist_records_toy9_probability_parity() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-migration-candidate-audit.md").read_text()
    gate7c = _between(
        checklist,
        "### Gate 7C: Toy9 Heterogeneous Probability Parity",
        "### Gate 7D: Continuous-Scalar Contract Decision",
    )

    for required in (
        "Status: parity slice complete.",
        "src/neural_abm/toy_heterogeneous.py::apply_output_average",
        "heterogeneous_action_probability",
        "group_gated_probability_commit",
        "not evidence that Toy9 is a full NABM",
    ):
        assert required in gate7c

    assert "## Gate 7C Result" in audit
    assert "two existing-toy parity users" in audit
    assert "Toy7 remains a contract-gap decision" in audit


def test_continuous_scalar_contract_records_toy7_gap() -> None:
    decision = (
        ROOT / "docs/decisions/0011-continuous-scalar-unit-contract.md"
    ).read_text()
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-migration-candidate-audit.md").read_text()
    gate7d = _between(
        checklist,
        "### Gate 7D: Continuous-Scalar Contract Decision",
        "### Gate 7E: Toy7 Bounded-Scalar Intensity Parity",
    )

    for required in (
        "Toy7",
        "continuous extraction intensity",
        "not a probability",
        "SCALAR_PROBABILITY_CHANNEL",
        "BOUNDED_SCALAR_CHANNEL",
        "mix_bounded_scalars",
        "Gate 7E",
        "not promote Toy7",
    ):
        assert required in decision

    for required in (
        "Status: contract decision complete.",
        "bounded continuous scalar",
        "SCALAR_PROBABILITY_CHANNEL",
        "BOUNDED_SCALAR_CHANNEL",
        "mix_bounded_scalars",
        "parity-only",
    ):
        assert required in gate7d

    assert "## Gate 7D Result" in audit
    assert "Toy7 should not be routed through `SCALAR_PROBABILITY_CHANNEL`" in audit
    assert "Toy7 remains deferred until that contract has toy-independent tests" in audit


def test_bounded_scalar_contract_records_toy7_parity_slice() -> None:
    decision = (
        ROOT / "docs/decisions/0011-continuous-scalar-unit-contract.md"
    ).read_text()
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-migration-candidate-audit.md").read_text()
    gate7e = _between(
        checklist,
        "### Gate 7E: Toy7 Bounded-Scalar Intensity Parity",
        "### Gate 7F: Toy10 Market/Ecology Channel Parity",
    )

    for required in (
        "Status: parity slice complete.",
        "src/neural_abm/social.py::BOUNDED_SCALAR_CHANNEL",
        "src/neural_abm/social.py::mix_bounded_scalars",
        "src/neural_abm/social.py::select_bounded_scalar_output_peers",
        "src/neural_abm/mixers.py::apply_bounded_scalar_output_average",
        "src/neural_abm/toy_resource.py::apply_output_average",
        "test_toy7_output_average_matches_unit_bounded_scalar_parity",
        "test_toy7_output_average_routes_through_unit_bounded_scalar_helper",
        "not evidence that Toy7 is a",
        "full NABM claim path",
    ):
        assert required in gate7e

    for required in (
        "Gate 7E implemented the contract surface",
        "BOUNDED_SCALAR_CHANNEL",
        "select_bounded_scalar_output_peers",
        "apply_bounded_scalar_output_average",
        "Toy7 now routes only social extraction-intensity selection and mixing",
    ):
        assert required in decision

    for required in (
        "## Gate 7E Result",
        "SCALAR_PROBABILITY_CHANNEL",
        "apply_bounded_scalar_output_average",
        "Toy7 now uses a bounded scalar unit surface",
        "does not promote Toy7 to full NABM evidence",
    ):
        assert required in audit

    assert "Bounded-scalar unit contract prototype" not in checklist


def test_toy10_market_ecology_channels_record_bounded_scalar_parity() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-migration-candidate-audit.md").read_text()
    gate7f = _between(
        checklist,
        "### Gate 7F: Toy10 Market/Ecology Channel Parity",
        "### Gate 7G: Toy6 Categorical Distribution Parity",
    )

    for required in (
        "Status: parity slice complete.",
        "src/neural_abm/toy_market.py::select_peer_ids",
        "src/neural_abm/toy_market.py::mix_channel",
        "apply_bounded_scalar_output_average",
        "price_expectation",
        "conservation_norm",
        "multi_channel_market_commit",
        "not a general multi-channel NABM claim",
    ):
        assert required in gate7f

    for required in (
        "## Gate 7F Result",
        "does not require a vector-valued multi-channel contract",
        "select_bounded_scalar_output_peers",
        "price_expectation",
        "conservation_norm",
        "dynamic graph rewiring",
        "does not promote Toy10 to full",
        "does not claim a general multi-channel message contract",
    ):
        assert required in audit

    assert "Toy10 multi-channel contract audit" not in checklist


def test_toy6_categorical_distribution_records_unit_parity() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-migration-candidate-audit.md").read_text()
    gate7g = _between(
        checklist,
        "### Gate 7G: Toy6 Categorical Distribution Parity",
        "## Recommended Next Slice",
    )

    for required in (
        "Status: parity slice complete.",
        "src/neural_abm/mixers.py::apply_distribution_output_average",
        "src/neural_abm/toy_categorical.py::apply_output_average",
        "PROBABILITY_DISTRIBUTION_CHANNEL",
        "strategy_distribution",
        "categorical_probability_commit",
        "does not promote Toy6 to full",
    ):
        assert required in gate7g

    for required in (
        "## Gate 7G Result",
        "A new categorical-policy channel is not needed",
        "apply_distribution_output_average",
        "strategy_distribution",
        "cyclic payoff construction",
        "does not claim a general categorical ABM mechanism",
    ):
        assert required in audit

    assert "Existing-Toy Migration Consolidation" in checklist
    assert "Toy6 categorical contract audit" not in checklist


def test_existing_toy_migration_consolidation_records_engineering_boundary() -> None:
    decision = (
        ROOT
        / "docs/decisions/0012-existing-toy-migration-parity-consolidation.md"
    ).read_text()
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-migration-candidate-audit.md").read_text()
    capability_matrix = (ROOT / "docs/toy-models/capability-matrix.md").read_text()
    contract_decision = (
        ROOT / "docs/decisions/0010-nabm-unit-v1-contract.md"
    ).read_text()
    gate7h = _between(
        checklist,
        "### Gate 7H: Existing-Toy Migration Consolidation",
        "### Gate 8A: Runner Lifecycle Consolidation Audit",
    )

    for required in (
        "Toy6-10 can route their primary social exchange slices",
        "Toy6-10 are full NABM evidence cases",
        "scalar probability",
        "bounded scalar",
        "probability distribution",
        "typed social-exchange reuse",
        "not as evidence",
        "Toy6-10 are full NABM models",
    ):
        assert required in decision

    for required in (
        "Status: documentation consolidation complete.",
        "docs/decisions/0012-existing-toy-migration-parity-consolidation.md",
        "capability matrix",
        "Toy6-10 status as `compatible`, not `full`",
        "performance evidence",
    ):
        assert required in gate7h
    assert "Runner lifecycle consolidation audit" in checklist

    for required in (
        "## Gate 7H Consolidation",
        "Toy6",
        "Toy7",
        "Toy8",
        "Toy9",
        "Toy10",
        "typed social-exchange",
        "reuse",
        "parity-only",
    ):
        assert required in audit

    for required in (
        "## Unit-Backed Migration Parity",
        "PROBABILITY_DISTRIBUTION_CHANNEL",
        "BOUNDED_SCALAR_CHANNEL",
        "activation_propensity",
        "heterogeneous_action_probability",
        "price_expectation",
        "conservation_norm",
        "These migrations keep Toy6-10 in the `compatible` category",
        "Parity slice complete; compatible but not evidence-default",
    ):
        assert required in capability_matrix

    assert "bounded scalar, state dict" in contract_decision
    assert "0012 then consolidated" in contract_decision
    assert "Toy6-10 migration parity slices" in contract_decision


def test_runner_lifecycle_audit_records_gate8a_boundary() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    readme = (ROOT / "src/neural_abm/README.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-runner-lifecycle-audit.md").read_text()
    gate8a = _between(
        checklist,
        "### Gate 8A: Runner Lifecycle Consolidation Audit",
        "### Gate 8B: Social Diagnostics Mapper Prototype",
    )

    for required in (
        "DomainToyRunner",
        "DomainToyAdapter",
        "DomainRunSettings",
        "run directory",
        "metadata",
        "aggregate_metrics.csv",
        "micro_state.csv",
        "fallback",
        "DomainToyResult",
        "Toy6",
        "Toy7",
        "Toy8",
        "Toy9",
        "Toy10",
        "Do not unify Toy6-10 step order",
        "Gate 8B: Social Diagnostics Mapper Prototype",
        "peer_count",
        "mean_peer_count",
        "mean_social_loss",
        "mean_social_update_norm",
        "not a full runner rewrite",
    ):
        assert required in audit

    for required in (
        "Status: audit complete.",
        "docs/nabm-unit-v1-runner-lifecycle-audit.md",
        "DomainRunSettings",
        "DomainToyAdapter",
        "DomainToyRunner",
        "Gate 8B: Social",
        "not a full runner rewrite",
        "peer_count",
        "mean_peer_count",
        "mean_social_loss",
        "mean_social_update_norm",
    ):
        assert required in gate8a

    for required in (
        "docs/nabm-unit-v1-runner-lifecycle-audit.md",
        "DomainToyRunner",
        "The first diagnostics mapping slice",
        "not a full runner rewrite",
    ):
        assert required in readme


def test_social_diagnostics_mapper_records_gate8b_boundary() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    readme = (ROOT / "src/neural_abm/README.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-runner-lifecycle-audit.md").read_text()
    gate8b = _between(
        checklist,
        "### Gate 8B: Social Diagnostics Mapper Prototype",
        "### Gate 8C: Compatible-Toy Adapter Thinness",
    )

    for required in (
        "src/neural_abm/domain_social_diagnostics.py",
        "aggregate_social_diagnostic_fields",
        "micro_social_diagnostic_fields",
        "tests/test_domain_social_diagnostics.py",
        "Toy6",
        "Toy7",
        "Toy8",
        "Toy9",
        "Toy10",
        "peer_count",
        "mean_peer_count",
        "mean_social_loss",
        "mean_social_update_norm",
        "not a full runner rewrite",
    ):
        assert required in audit

    for required in (
        "Status: complete for Toy6-10 diagnostics rows.",
        "src/neural_abm/domain_social_diagnostics.py",
        "tests/test_domain_social_diagnostics.py",
        "test_toy6_rows_route_social_diagnostics_through_mapper",
        "test_toy7_rows_route_social_diagnostics_through_mapper",
        "test_toy8_rows_route_social_diagnostics_through_mapper",
        "test_toy9_rows_route_social_diagnostics_through_mapper",
        "test_toy10_rows_route_social_diagnostics_through_mapper",
        "optional toy-supplied `component_id`",
        "DomainToyRunner",
        "DomainToyAdapter",
        "toy `step(...)`",
        "performance evidence",
    ):
        assert required in gate8b

    for required in (
        "domain_social_diagnostics",
        "aggregate_social_diagnostic_fields",
        "micro_social_diagnostic_fields",
        "Toy6-Toy10",
        "mean_social_update_norm",
    ):
        assert required in readme


def test_adapter_thinness_records_gate8c_boundary() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    readme = (ROOT / "src/neural_abm/README.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-runner-lifecycle-audit.md").read_text()
    gate8c = _between(
        checklist,
        "### Gate 8C: Compatible-Toy Adapter Thinness",
        "### Gate 8D: Compatible-Toy Artifact Contracts",
    )

    for required in (
        "Gate 8C Result",
        "make_domain_run_dir",
        "write_domain_run_metadata",
        "DomainRunSettings",
        "Toy6",
        "Toy7",
        "Toy8",
        "Toy9",
        "Toy10",
        "test_domain_run_artifact_helpers_use_settings",
        "rejects moving adapter `step(...)`",
        "domain_metrics(...)",
    ):
        assert required in audit

    for required in (
        "Status: run-artifact helper extraction complete.",
        "src/neural_abm/domain_runner.py::make_domain_run_dir",
        "src/neural_abm/domain_runner.py::write_domain_run_metadata",
        "test_domain_run_artifact_helpers_use_settings",
        "DomainToyRunner.run()",
        "step(...)",
        "aggregate_row(...)",
        "micro_rows(...)",
        "final_epoch(...)",
        "domain_metrics(...)",
        "artifact-contract tests",
    ):
        assert required in gate8c

    for required in (
        "domain_runner",
        "api",
        "make_domain_run_dir",
        "write_domain_run_metadata",
    ):
        assert required in readme


def test_artifact_contracts_record_gate8d_boundary() -> None:
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    readme = (ROOT / "src/neural_abm/README.md").read_text()
    audit = (ROOT / "docs/nabm-unit-v1-runner-lifecycle-audit.md").read_text()
    gate8d = _between(
        checklist,
        "### Gate 8D: Compatible-Toy Artifact Contracts",
        "### Gate 8E: API Surface Audit and v0 Contract",
    )

    for required in (
        "Gate 8D Result",
        "tests/test_domain_toy_artifact_contracts.py",
        "aggregate_metrics.csv",
        "micro_state.csv",
        "Toy6-Toy10",
        "actual CSV headers",
        "no-extract boundary",
    ):
        assert required in audit

    for required in (
        "Status: artifact-contract tests complete.",
        "tests/test_domain_toy_artifact_contracts.py",
        "TOY6_AGGREGATE_FIELDS",
        "TOY10_MICRO_STATE_FIELDS",
        "exact expected field-order tests",
        "tiny-run CSV header checks",
        "does not assert performance",
        "schema guardrail",
        "Optional import cleanup",
    ):
        assert required in gate8d

    assert "tests/test_domain_toy_artifact_contracts.py" in readme


def test_public_api_v0_contract_records_surface_boundary() -> None:
    docs_index = (ROOT / "docs/README.md").read_text()
    readme = (ROOT / "src/neural_abm/README.md").read_text()
    root_readme = (ROOT / "README.md").read_text()
    checklist = (ROOT / "docs/nabm-unit-v1-completeness-checklist.md").read_text()
    audit = (ROOT / "docs/api-surface-audit.md").read_text()
    decision = (
        ROOT / "docs/decisions/0013-public-api-v0-contract.md"
    ).read_text()
    dependency_decision = (
        ROOT / "docs/decisions/0014-package-dependency-policy.md"
    ).read_text()
    plain_dependency_decision = _plain(dependency_decision)
    gate8e = _between(
        checklist,
        "### Gate 8E: API Surface Audit and v0 Contract",
        "### Gate 8F: Stable v0 API Facade",
    )
    gate8f = _between(
        checklist,
        "### Gate 8F: Stable v0 API Facade",
        "### Gate 8G: API Release Smoke",
    )
    gate8g = _between(
        checklist,
        "### Gate 8G: API Release Smoke",
        "### Gate 8H: Package Dependency Policy",
    )
    gate8h = _between(
        checklist,
        "### Gate 8H: Package Dependency Policy",
        "### Gate 8I: Torch-Free Facade Seed",
    )
    gate8i = _between(
        checklist,
        "### Gate 8I: Torch-Free Facade Seed",
        "### Gate 8J: Optional Dependency Profiles",
    )
    gate8j = _between(
        checklist,
        "### Gate 8J: Optional Dependency Profiles",
        "### Gate 8K: Built-Wheel Dependency Profile Smokes",
    )
    gate8k = _between(
        checklist,
        "### Gate 8K: Built-Wheel Dependency Profile Smokes",
        "### Gate 8L: Torch-Free Social Core",
    )
    gate8l = _between(
        checklist,
        "### Gate 8L: Torch-Free Social Core",
        "### Gate 8M: Torch-Free Lifecycle Reports",
    )
    gate8m = _between(
        checklist,
        "### Gate 8M: Torch-Free Lifecycle Reports",
        "### Gate 8N: Product Package Release Boundary",
    )
    gate8n = _between(
        checklist,
        "### Gate 8N: Product Package Release Boundary",
        "### Gate 8O: Pre-Release Artifact Flow",
    )
    gate8o = _between(
        checklist,
        "### Gate 8O: Pre-Release Artifact Flow",
        "## Recommended Next Slice",
    )
    plain_gate8h = _plain(gate8h)
    plain_gate8i = _plain(gate8i)
    plain_gate8j = _plain(gate8j)
    plain_gate8k = _plain(gate8k)
    plain_gate8l = _plain(gate8l)
    plain_gate8m = _plain(gate8m)
    plain_gate8n = _plain(gate8n)
    plain_gate8o = _plain(gate8o)

    for required in (
        "api-surface-audit.md",
        "package-release-boundary.md",
        "pre-release-artifact-flow.md",
        "decisions/0013-public-api-v0-contract.md",
        "decisions/0014-package-dependency-policy.md",
        "stable",
        "experimental",
        "paper evidence tooling",
        "torch-free social-core and unit-core splits",
    ):
        assert required in docs_index

    for required in (
        "## Public API v0 Boundary",
        "neural_abm.__init__",
        "lazy compatibility surface",
        "neural_abm.api",
        "stable lifecycle",
        "typed social exchange",
        "compatible-toy runner",
        "paper-only",
        "src/neural_abm/api.py",
        "intentionally excludes",
        "## Package Dependency Boundary",
        "default package profile is the lightweight no-torch `api_lite` boundary",
        "src/neural_abm/social_core.py",
        "src/neural_abm/unit_core.py",
        "Decision 0014",
    ):
        assert required in readme

    for required in (
        "## Public API and Package Status",
        "from neural_abm.api import NABMUnit, SocialBlock, SocialChannel",
        "lazy compatibility surface",
        "neural_abm.api_lite",
        "NumPy-only social primitives",
        "lightweight lifecycle reports/local-step primitives",
        "scalar/bounded scalar",
        "lightweight torch-free install",
        "neural-abm[torch]",
        "neural-abm[research]",
        "neural-abm[full]",
        "Package release boundary",
        "toy_catalog",
        "uv run python scripts/smoke_package_profiles.py",
        "uv run python scripts/inspect_release_artifacts.py --build",
        "uv run python examples/toy_catalog.py",
        "Decision 0014",
    ):
        assert required in root_readme

    for required in (
        "Stable Core Candidates",
        "Experimental Core Candidates",
        "Internal or Paper-Only Surfaces",
        "Do Not Export as Stable",
        "Current Export Gap",
        "Recommended v0 Facade Shape",
        "neural_abm.api",
        "neural_abm.api_lite",
        "neural_abm.__init__",
        "lazy compatibility surface",
        "not a replacement for the full stable v0 facade",
        "scalar/bounded scalar `SocialChannel` metadata",
        "neural_abm.social_core",
        "neural_abm.unit_core",
        "toy feature-taxonomy helpers",
    ):
        assert required in audit

    for required in (
        "Public API v0 Contract Boundary",
        "The preferred stable import path is `neural_abm.api`",
        "lazy compatibility",
        "Stable v0 Responsibilities",
        "Experimental Responsibilities",
        "Non-API Responsibilities",
        "Toy Runner Policy",
        "Evidence Package Boundary",
        "add a small `neural_abm.api` facade",
    ):
        assert required in decision

    for required in (
        "Package Dependency Policy for Lightweight API Readiness",
        "The default install is the `api_lite` floor",
        "`torch`",
        "`research`",
        "`full`",
        "The dev dependency group also includes the full research stack",
        "built-wheel profile smokes prove",
        "default profile imports the package root",
        "`research` extra imports representative research dependencies",
        "scripts/smoke_package_profiles.py",
        "must be lazy",
        "neural_abm.api_lite",
        "not a replacement for `neural_abm.api`",
        "project.optional-dependencies",
        "torch-free social primitives",
        "`api_lite.SocialChannel` accepts only scalar/bounded",
        "lightweight lifecycle reports/local-step primitives",
        "Amendment: Torch-Free Social Core",
        "Amendment: Torch-Free Lifecycle Reports",
        "`neural_abm.unit` remains a compatibility module",
        "The full stable v0 facade should not be marketed as a no-torch API",
    ):
        assert required in plain_dependency_decision

    for required in (
        "Status: audit and contract complete.",
        "docs/api-surface-audit.md",
        "docs/decisions/0013-public-api-v0-contract.md",
        "current broad `neural_abm.__init__` export list",
        "neural_abm.api",
        "paper evidence package",
        "public Python package",
    ):
        assert required in gate8e

    for required in (
        "Status: first facade slice complete.",
        "src/neural_abm/api.py",
        "tests/test_public_api_v0.py",
        "neural_abm.__init__",
        "lazy compatibility surface",
        "Toy runners",
        "evidence gates",
        "accelerator",
        "excluded from the stable",
    ):
        assert required in gate8f

    for required in (
        "Status: release smoke complete.",
        "examples/minimal_api_nabm.py",
        "tests/test_public_api_examples.py",
        "imports only from",
        "neural_abm.api",
        "wheel-style build",
        "installed-wheel import smoke",
        "separate from toy runners",
    ):
        assert required in gate8g

    for required in (
        "Status: dependency policy recorded.",
        "docs/decisions/0014-package-dependency-policy.md",
        "tests/test_package_dependency_policy.py",
        "default-runtime candidate",
        "torch-backed runtime candidate extra",
        "not yet a lightweight no-torch import boundary",
        "isolated import smokes",
        "`torch` optionalization is explicitly deferred",
        "import-time coupling",
    ):
        assert required in plain_gate8h

    for required in (
        "Status: torch-free facade seed and no-deps wheel smoke complete.",
        "src/neural_abm/api_lite.py",
        "tests/test_public_api_lite.py",
        "no-deps wheel import smoke command output",
        "blocks `torch`",
        "neural_abm.api_lite",
        "only `numpy` and `pyyaml`",
        "NABMUnit",
        "SocialBlock",
        "profile seed, not a replacement",
        "No default dependency has been removed yet",
        "default-runtime floor",
    ):
        assert required in plain_gate8i

    for required in (
        "Status: optional dependency profiles recorded.",
        "pyproject.toml",
        "uv.lock",
        "`numpy` and `pyyaml`",
        "`config`, `torch`, `research`, `plot`, `cli`, and `full`",
        "uv `dev` dependency group",
        "default package install no longer declares `torch`",
        "explicit extras",
        "built-wheel install/import smokes",
    ):
        assert required in plain_gate8j

    for required in (
        "Status: built-wheel profile smokes complete.",
        "scripts/smoke_package_profiles.py",
        "profile smoke command output",
        "`default`, `torch`, `research`, and `full` profiles",
        "without a default `torch` requirement and without loading `torch`",
        "`NABMUnit`, `SocialBlock`, and `SocialChannel`",
        "Toy6 runner symbols",
        "Toy10 runner symbols",
        "validated against built wheels",
        "uv run python scripts/smoke_package_profiles.py",
        "social/lifecycle modules into torch-free code",
    ):
        assert required in plain_gate8k

    for required in (
        "Status: torch-free social core split complete.",
        "src/neural_abm/social_core.py",
        "src/neural_abm/metrics_core.py",
        "api_lite.SocialChannel",
        "api_lite.mix_scalar_probabilities",
        "scalar/bounded scalar mix channel kinds",
        "rejects tensor channel metadata",
        "without installing or loading torch",
        "NABMUnit",
        "SocialBlock",
        "mix_probability_distributions",
        "lifecycle import boundaries",
    ):
        assert required in plain_gate8l

    for required in (
        "Status: torch-free lifecycle report split complete.",
        "src/neural_abm/unit_core.py",
        "src/neural_abm/unit.py",
        "CommitReport",
        "LocalUpdateAdapter",
        "SocialDiagnostics",
        "NABMLocalStep",
        "api_lite",
        "ObservationSpec",
        "SocialMessageSpec",
        "NABMStep",
        "NABMUnit",
        "no-torch full lifecycle claim",
    ):
        assert required in plain_gate8m

    for required in (
        "Status: release boundary and catalog smoke complete.",
        "docs/package-release-boundary.md",
        "examples/toy_catalog.py",
        "toy_catalog()",
        "neural_abm.api_lite",
        "imports only from",
        "api_lite.toy_catalog()",
        "Product-facing docs now point users to `api_lite`",
        "release checklist",
        "alpha artifact validation, distribution metadata",
    ):
        assert required in plain_gate8n

    for required in (
        "Status: pre-release artifact flow and inspector complete.",
        "docs/pre-release-artifact-flow.md",
        "scripts/inspect_release_artifacts.py",
        "tests/test_release_artifact_inspection.py",
        "authors, keywords, and classifiers",
        "Apache-2.0",
        "0.1.0a2",
        "requires-python = \">=3.11\"",
        "inspects wheel/sdist metadata",
        "forbidden internal-history paths",
        "local wheel-install matrix",
    ):
        assert required in plain_gate8o

    assert "persistent artifact validation" in checklist
    assert "Do not expose toy-owned semantics" in checklist
