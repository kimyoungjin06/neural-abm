from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "case-studies"
    / "researcher-pivot"
    / "pivot-feedback-empirical-nabm-protocol.md"
)
COMPUTE_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "case-studies"
    / "researcher-pivot"
    / "compute-execution-contract.md"
)


def _plain(text: str) -> str:
    """Collapse Markdown wrapping so prose checks do not depend on line breaks."""

    return " ".join(text.split())


def test_pivot_feedback_protocol_freezes_science_first_gates() -> None:
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    plain_protocol = _plain(protocol)

    for required_heading in (
        "## 1. Plain-Language Research Question",
        "## 6. Data Readiness Gate (G-1)",
        "## 7. Measurement Freeze Gate (G0)",
        "## 8. Empirical Signature Gate (G1)",
        "## 9. Baseline Model Ladder (G2)",
        "## 10. NABM Redesign Gate (G3)",
        "## 11. Frozen Holdout Gate (G4)",
        "## 12. Causal Validation Gate (G5)",
        "## 13. Promotion and Stop Rules (G6)",
    ):
        assert required_heading in protocol

    for claim_boundary in (
        "no substantive empirical claim has passed yet",
        "This is not an executable preregistration until G0",
        "that only a Neural ABM can generate an information cascade",
        "The OpenAlex-only construct is `topic attention acceleration`, not `hype`",
        "incumbent target-topic researchers whose entry predates the signal window",
        "`Failed pivot` is reserved for a design that observes the attempt",
        "If G1 fails, stop the cascade mechanism route",
        "If B1-B3 match B4, adopt the simpler model",
        "Training fit alone cannot promote a claim into the manuscript",
    ):
        assert claim_boundary in plain_protocol


def test_pivot_feedback_protocol_records_current_data_boundary() -> None:
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    plain_protocol = _plain(protocol)

    for current_boundary in (
        "a mounted backup contains the full derived OpenAlex 20260330 merged-works parquet root",
        "21,019 readable parquet footers",
        "returned `INVENTORY_ONLY`, not `PASS_METADATA` or `PASS_G1`",
        "the raw authorship table has 1,361,551,038 footer rows, of which 580,116,420 have null author IDs",
        "1,361,551,037 retained authorship rows, 580,116,419 null author-ID rows",
        "20,996,232,796 rows, and found zero null, other-malformed, or orphan source work IDs",
        "Duplicate source-target edges were not measured",
        "contains raw exception payload fields, not a validated abstract-text source",
        "the corpus is not yet analysis-ready",
        "G-1 is a verification and empirical-pipeline construction gate, not a corpus reacquisition gate",
        "No full substantive analysis or model fitting is authorized by this protocol until G-1 passes",
        "Reading isolated `part-*` or `b000000` files is permitted for schema and pipeline smoke tests",
        "not as a representative scientific pilot or for effect estimation",
        "The next executable step is a resumable CPU/I/O body recount, not model fitting and not a GPU job",
        "rather than relying on footer-only `COUNT(*)` results",
        "The completed 2026-07-19 current-content run separately established `PASS_CURRENT_CONTENT_BASELINE_COMPONENT`",
        "historical inventory-era byte identity remains `UNVERIFIED`",
        "The separate v3 full temporal run established `PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE`",
        "does not establish citation-edge chronology or `available_at`",
        "G-1 remains `PENDING`, empirical G1 remains `NOT_RUN`",
        "`experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml`",
        "`G_MINUS_1_PENDING`, `pass_g_minus_1=false`, and `execution_authorized=false`",
    ):
        assert current_boundary in plain_protocol

    assert (
        "the profiled full 20260330 final parquet root is not mounted"
        not in plain_protocol
    )
    assert "schema/sample slice rather than the full corpus" not in plain_protocol


def test_compute_contract_routes_only_appropriate_work_to_gpu() -> None:
    contract = _plain(COMPUTE_CONTRACT_PATH.read_text(encoding="utf-8"))

    for compute_boundary in (
        "The default plan is therefore **not** to copy the full root to a GPU server",
        "G-1 inventory, recount, work-source, and current-content baseline audits",
        "work-source membership + row conservation: PASS under frozen quarantine (audit only)",
        "filtered child dataset or an eligible researcher panel now exists",
        "G1 event studies, hazards, placebos, bootstrap",
        "108 NABM cells and 216 frozen or explicit-adaptive-baseline cells",
        "authenticates non-interactively when the dedicated identity",
        "the Local_Map_Analysis preflight succeeded",
        "No G4 row or feature is released",
        "**Post-G3 release:** G4 features only",
        "The final G4 package sent to the server is `features_only`",
        "complete checksummed prediction bundles must exist for every registered B0-B4 model and robustness variant",
        "Merely running that script on a machine with GPUs will not make it a GPU workload",
        "at least 2x end-to-end throughput over the best CPU execution",
        "No current step requires copying the approximately 921 GiB root to the GPU server",
    ):
        assert compute_boundary in contract

    assert "/hdd/" not in contract
    assert "`GPU-Server`" not in contract
    assert "non-interactive probe could not authenticate" not in contract


def test_pivot_feedback_protocol_separates_constructs_and_holdouts() -> None:
    protocol = _plain(PROTOCOL_PATH.read_text(encoding="utf-8"))

    for design_boundary in (
        "`neither`, `public-success proxy only`, `capacity proxy only`, and `both`",
        "No focal work used to construct this proxy may satisfy any primary or robustness entry definition",
        "The G1 replication partition is distinct from every G4 final test partition",
        "No G4 final temporal, domain, or shock-family test partition may be opened to decide whether G1 passes",
        "The G1 empirical-replication set cannot be reused as a G4 final test",
        "the incumbent-only attention signal in H1 predicts later entry",
        "| Network-local message diffusion speed | slow, empirical proxy-mapped speed, fast |",
        "the shared quantile is then passed through each arm's mapping rather than forcing identical realized outcomes",
        "neither causal identification nor proof that the process is the unique explanation",
    ):
        assert design_boundary in protocol


def test_cross_field_expansion_is_prospective_outcome_blind_and_gated() -> None:
    protocol = _plain(PROTOCOL_PATH.read_text(encoding="utf-8"))
    contract = _plain(COMPUTE_CONTRACT_PATH.read_text(encoding="utf-8"))

    for boundary in (
        "Final empirical field selection remains `PENDING`",
        "authoritative taxonomy-reference binding",
        "outcome-blind structural coverage audit",
        "`experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml`",
        "`experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml`",
        "A structurally valid pending overlay is not a G0 pass",
        "applies an outcome-blind maximin rule",
        "Ties are broken by stable numeric field ID",
        "deterministic reserve list",
        "at least three independent attention-surge episodes",
        "Authors, related topic families, and all response cohorts belonging to one shock episode are assigned atomically",
        "H3 has one confirmatory primary endpoint",
        "The earliest maximum resolves a frozen flat peak",
        "`R` is a secondary transport endpoint",
        "lead-attention exposure at `t+1` or `t+2`",
        "a three-hypothesis Holm family",
        "two fields by two independent shocks",
        "labeled `DISCOVERY_ONLY` in every artifact",
        "The full run is four fields with at least three independent episodes per field",
        "The primary cross-field test is domain-and-author-cold",
        "No final field names may appear",
    ):
        assert boundary in protocol

    for compute_boundary in (
        "Final empirical field selection is also `PENDING`",
        "G-1 taxonomy-reference, main-work temporal, author, and outcome-blind field-structure audits",
        "Two-field by two-shock `DISCOVERY_ONLY` pilot",
        "Four-field G1 event studies, hazards, equivalence placebos, bootstrap",
        "`experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml`",
        "authoritative taxonomy-reference and cross-field-matrix hashes",
        "The primary cross-field bundle is domain-and-author-cold",
        "Keep final field selection `PENDING`",
        "three-field G1 CPU/RAM analysis with its Holm and equivalence gates",
    ):
        assert compute_boundary in contract


def test_pivot_feedback_protocol_is_routed_without_promoting_current_results() -> None:
    relative_protocol = "pivot-feedback-empirical-nabm-protocol.md"
    relative_compute_contract = "compute-execution-contract.md"
    case_readme = (ROOT / "docs/case-studies/researcher-pivot/README.md").read_text(
        encoding="utf-8"
    )
    docs_readme = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    paper_readme = (ROOT / "paper/README.md").read_text(encoding="utf-8")
    paper_outline = (ROOT / "paper/outline.md").read_text(encoding="utf-8")
    application_section = (
        ROOT / "paper/sections/07-research-application.md"
    ).read_text(encoding="utf-8")
    claim_matrix = (ROOT / "paper/claim-matrix.md").read_text(encoding="utf-8")

    assert relative_protocol in case_readme
    assert relative_compute_contract in case_readme
    assert relative_compute_contract in docs_readme
    assert relative_compute_contract in PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "hypothesis-generating inputs" in case_readme
    for routed_text in (
        docs_readme,
        root_readme,
        paper_readme,
        paper_outline,
        application_section,
        claim_matrix,
    ):
        assert relative_protocol in routed_text

    assert "method-first Neural Social Nodes track" in _plain(paper_readme)
    assert "open a separate `paper/pivot-feedback/` companion track" in _plain(
        paper_outline
    )
    assert "remains stylized lifecycle evidence" in _plain(application_section)
    assert "These rows are planned tests, not supported claims" in claim_matrix
    assert "Untested" in claim_matrix


def test_routed_local_markdown_links_resolve() -> None:
    routed_files = (
        ROOT / "README.md",
        ROOT / "docs/README.md",
        ROOT / "docs/case-studies/researcher-pivot/README.md",
        PROTOCOL_PATH,
        COMPUTE_CONTRACT_PATH,
        ROOT / "paper/README.md",
        ROOT / "paper/outline.md",
        ROOT / "paper/claim-matrix.md",
    )

    for source in routed_files:
        text = source.read_text(encoding="utf-8")
        for raw_target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if raw_target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = raw_target.split("#", maxsplit=1)[0]
            assert (source.parent / target).resolve().exists(), (
                f"broken local Markdown link in {source}: {raw_target}"
            )
