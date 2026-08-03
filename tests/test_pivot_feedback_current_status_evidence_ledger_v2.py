from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v2.md"
AUTHORITY = ROOT / "experiments/configs/pivot_feedback_current_state_v1.yaml"
SUMMARY = (
    ROOT
    / "experiments/evidence/results/pivot_feedback_current_state_v1_20260803.summary.json"
)
F_SECTIONS_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_pre_g1_scientific_sections_v1_20260803.summary.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_ledger_binds_active_authority_and_receipt() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert "CURRENT_HUMAN_FACING_INDEX_CURRENT_STATE_V1_F_IN_PROGRESS" in text
    assert _sha256(AUTHORITY) in text
    assert _sha256(SUMMARY) in text
    assert authority["active_routing"]["active_phase"] == "F"
    assert summary["status"] == (
        "PASS_ACTIVE_SIX_PHASE_AUTHORITY_F_IN_PROGRESS_FAIL_CLOSED"
    )


def test_v2_ledger_states_exact_f_boundary() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))

    assert authority["lifecycle"]["pass_f"] is False
    assert authority["lifecycle"]["pass_g0"] is False
    assert authority["lifecycle"]["pre_g1_pending_choice_count"] == 0
    assert authority["lifecycle"]["pre_g1_pending_section_count"] == 13
    assert "F is one atomic completion milestone" in text
    assert "`pass_f=false` and `pass_g0=false`" in text
    assert "No confirmatory or holdout analytic row has" in text
    assert "empirical G1 is `NOT_RUN`" in text


def test_v2_ledger_distinguishes_section_component_from_atomic_f_pass() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    section_summary = json.loads(F_SECTIONS_SUMMARY.read_text(encoding="utf-8"))

    assert section_summary["status"] == (
        "PASS_THIRTEEN_PRE_G1_SCIENTIFIC_SECTIONS_FROZEN_NO_F_PASS"
    )
    assert "All thirteen pre-G1 scientific sections are now outcome-blind frozen" in text
    assert "not a new micro-gate and not an F pass" in text
    assert "additive authority successor records the atomic transition" in text


def test_v2_ledger_links_resolve() -> None:
    text = LEDGER.read_text(encoding="utf-8")
    destinations = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)

    assert destinations
    for destination in destinations:
        path = (LEDGER.parent / destination.split("#", 1)[0]).resolve()
        assert path.exists(), destination
