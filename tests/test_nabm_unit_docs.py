from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


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
        "docs/nabm-unit-v1-boundary-audit.md",
        "docs/nabm-unit-v1-completeness-checklist.md",
    ):
        assert boundary_doc in readme


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
        "## Recommended Next Slice",
    )

    assert "Status: quick evidence complete." in gate5
    assert "tests/test_nabm_unit_adapter_holdout.py" in gate5
    assert "without changing `src/neural_abm`" in gate5
    assert "adapter-threshold-readiness main reaches full adoption" in gate5
    assert "tiny scripted binary" in gate5
    assert "tests/test_nabm_unit_adapter_holdout.py" in readme
