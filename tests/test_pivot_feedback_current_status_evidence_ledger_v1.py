from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = (
    ROOT
    / "docs/research/pivot-feedback-current-status-evidence-ledger-v1.md"
)
PROTOCOL_PATH = (
    ROOT
    / "docs/case-studies/researcher-pivot/"
    "pivot-feedback-empirical-nabm-protocol.md"
)
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v61.yaml"
G0 = ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml"
CLASSIFICATION = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_d0_production_pivot_classification_20260731.summary.json"
)
PERSISTENCE = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_d0_persistence_source_production_20260731.summary.json"
)
HOLDOUT = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_holdout_production_input_resolution_20260731.summary.json"
)
BOUND_EVIDENCE = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_g_minus_1a_receipt_20260723.summary.json",
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_d0_certified_episode_reselection_20260730.summary.json",
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_d0_certified_predecision_author_measurement_20260731."
    "summary.json",
    CLASSIFICATION,
    PERSISTENCE,
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_d0_persistence_calibration_readout_20260731.summary.json",
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_g0_measurement_freeze_v4_binding_20260731.summary.json",
    HOLDOUT,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ledger_binds_current_machine_authority_and_g0() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    g0 = yaml.safe_load(G0.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v61"
    assert _sha256(GATE) in text
    assert _sha256(G0) in text
    assert gate["verified_g0_v4_binding"]["contract"]["sha256"] == _sha256(G0)
    assert g0["lifecycle"]["pass_g0"] is False
    assert "CURRENT_HUMAN_FACING_INDEX_V61_NO_EXECUTION_AUTHORITY" in text
    assert "If this document conflicts with that immutable gate" in text


def test_ledger_binds_every_declared_primary_evidence_artifact() -> None:
    text = LEDGER.read_text(encoding="utf-8")

    for path in BOUND_EVIDENCE:
        assert path.exists()
        assert _sha256(path) in text


def test_ledger_replays_current_g0_boundary() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_g0_v4_binding"]["result"]

    assert result["total_resolved_choice_count"] == 12
    assert result["remaining_quantitative_choice_count"] == 7
    assert result["pending_required_section_count"] == 15
    assert result["freeze_ready_required_section_count"] == 0
    assert result["holdout_receipt_bound"] is False
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert "Twelve choices are resolved" in text
    assert "Seven quantitative choices remain unresolved" in text
    assert "All fifteen required sections remain not freeze-ready" in text

    remaining = yaml.safe_load(G0.read_text(encoding="utf-8"))["resolution"][
        "unresolved_quantitative_choices"
    ]
    assert set(remaining) == {
        "g1_replication_tolerance",
        "g2_score_margin",
        "g2_maximum_calibration_degradation",
        "g2_macro_trajectory_tolerance",
        "simulation_draws_or_mcse_stopping_rule",
        "parameter_budget",
        "tuning_budget",
    }
    for choice_id in remaining:
        assert f"`{choice_id}`" in text


def test_ledger_replays_d0_episode_entry_and_persistence_counts() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    classification = _read_json(CLASSIFICATION)
    persistence = _read_json(PERSISTENCE)
    states = classification["state_replay"]
    totals = states["totals"]

    assert totals == {
        "author_rows": 3_256_271,
        "no_focal_entry_rows": 3_246_984,
        "new_topic_entry_rows": 4_301,
        "return_rows": 125,
        "continuation_rows": 4_861,
    }
    episodes = [
        (row["field_id"], row["topic_id"], row["entry_year"])
        for row in states["by_episode"]
    ]
    assert episodes == [
        (17, 14_430, 2003),
        (17, 10_579, 2012),
        (27, 10_151, 1993),
        (27, 12_421, 1953),
    ]
    outcome = persistence["outcome_replay"]["totals"]
    assert outcome["rows"] == totals["new_topic_entry_rows"]
    assert outcome["matured_persistence_rows"] == 1_487
    assert outcome["matured_no_persistence_rows"] == 2_814
    assert outcome["descriptive_persistence_rate"] == 0.3457335503371309
    for token in (
        "3,256,271",
        "3,246,984",
        "4,301",
        "1,487",
        "2,814",
        "34.573355%",
    ):
        assert token in text


def test_ledger_preserves_holdout_and_next_authorization_boundary() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    holdout = _read_json(HOLDOUT)
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    declared = holdout["declared_inputs"]
    replay = holdout["materialization_replay"]
    assert declared["artifact_count"] == 0
    assert declared["production_seal_created"] is False
    assert declared["production_receipt_written"] is False
    assert replay["g4_domain_selected"] is False
    assert replay["g4_field_selected"] is False
    assert replay["g4_artifact_set_materialized"] is False
    assert _sha256(HOLDOUT) in text

    scope = gate["authorized_next_precommitment_scope"]
    assert scope["exact_choice_ids"] == ["g1_replication_tolerance"]
    assert "G0_G1_REPLICATION_TOLERANCE_PRECOMMITMENT_CLOSED_FIXTURE" in text
    assert "A logical temporal reservation is not" in text
    assert "a byte seal" in text


def test_all_local_ledger_links_resolve() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    destinations = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)

    assert destinations
    for destination in destinations:
        assert not destination.startswith(("http://", "https://"))
        path = (LEDGER.parent / destination.split("#", 1)[0]).resolve()
        assert path.exists(), destination


def test_claim_matrix_routes_operational_status_to_additive_ledger() -> None:
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    claim_matrix = (ROOT / "paper/claim-matrix.md").read_text(encoding="utf-8")
    normalized_claim_matrix = " ".join(claim_matrix.split())

    assert "pivot-feedback-current-status-evidence-ledger-v2.md" in claim_matrix
    assert "pivot_feedback_current_state_v1.yaml" in claim_matrix
    assert "complete thirteen-choice pre-G1 registry" in normalized_claim_matrix
    assert "six G2/G3 choices" in normalized_claim_matrix
    assert "empirical G1 is `NOT_RUN`" in claim_matrix
    assert "no baseline or NABM model has run" in claim_matrix
    assert "protocol v0.4 byte is retained" in claim_matrix

    assert _sha256(PROTOCOL_PATH) == (
        "731e39dec4c7be5c29a0f0404b48992ad0caef152062d1d1d7876acc07064746"
    )
    assert "Protocol version: 0.4" in protocol
    assert "Date: 2026-07-21" in protocol
    assert "Operational status refresh: 2026-08-03" in claim_matrix
    assert "These rows are planned tests, not supported claims" in claim_matrix
