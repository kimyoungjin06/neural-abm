from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType
from typing import Callable

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "experiments/configs/pivot_feedback_governance_compaction_v1.yaml"
)
VERIFIER = ROOT / "scripts/verify_pivot_feedback_governance_compaction_v1.py"
SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_governance_compaction_v1_20260803.summary.json"
)
NOTE = ROOT / "docs/research/pivot-feedback-governance-compaction-v1.md"
GATE_V61 = ROOT / "experiments/configs/pivot_feedback_gate_state_v61.yaml"
G0_V4 = ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml"
TOLERANCE_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_g0_g1_replication_tolerance_precommitment_20260803.summary.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_governance_compaction_verifier", VERIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict], None],
) -> str:
    verifier = _load_verifier()
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "mutated_compaction.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(verifier, "DEFAULT_CONTRACT", path.resolve())
    monkeypatch.setattr(verifier, "CONTRACT_SHA256", _sha256(path))
    with pytest.raises(verifier.GovernanceCompactionVerificationError) as error:
        verifier.verify(path.resolve())
    return str(error.value)


def test_shadow_contract_binds_current_authority_without_replacing_it() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    boundary = contract["authority_boundary"]

    assert _sha256(CONTRACT) == (
        "4c2562be842ec30b3fafee5320e88f5949b28fcd89e6091b34e5e43dcdd4c4ee"
    )
    assert boundary["effective_machine_authority"]["sha256"] == _sha256(GATE_V61)
    assert contract["bound_current_sources"]["g0_v4"]["sha256"] == _sha256(
        G0_V4
    )
    assert contract["bound_current_sources"][
        "g1_tolerance_validation_summary"
    ]["sha256"] == _sha256(TOLERANCE_SUMMARY)
    assert boundary["shadow_contract_can_replace_effective_authority"] is False
    assert boundary["shadow_contract_can_authorize_scientific_execution"] is False
    assert boundary["shadow_contract_can_bind_a_g0_successor"] is False


def test_six_phase_mapping_is_exact_complete_and_acyclic() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    phases = contract["proposed_live_phases"]
    optional = contract["optional_lanes"]

    assert [phase["phase_id"] for phase in phases] == ["D", "C", "F", "E", "M", "H"]
    assert [phase["order"] for phase in phases] == [10, 20, 30, 40, 50, 60]
    mapped = [gate for phase in phases for gate in phase["legacy_gate_ids"]]
    mapped += optional[0]["legacy_gate_ids"]
    assert mapped == [
        "G_MINUS_1A",
        "S0",
        "D0",
        "G0",
        "G0_5",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
    ]
    assert len(mapped) == len(set(mapped))

    seen: set[str] = set()
    for phase in phases:
        assert set(phase["requires"]).issubset(seen)
        seen.add(phase["phase_id"])
        assert phase["transition_receipts_required"]
        assert phase["claim_ceiling"]


def test_all_choices_and_sections_are_staged_exactly_once() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    g0 = yaml.safe_load(G0_V4.read_text(encoding="utf-8"))
    resolution = g0["resolution"]
    legacy_choices = set(resolution["resolved_measurements"]) | set(
        resolution["unresolved_quantitative_choices"]
    )
    choice_stages = contract["decision_staging_proposal"]["stages"]
    staged_choices = [choice for stage in choice_stages for choice in stage["choices"]]

    assert len(staged_choices) == len(set(staged_choices)) == 19
    assert set(staged_choices) == legacy_choices
    assert [stage["choice_count"] for stage in choice_stages] == [13, 3, 3]
    assert choice_stages[0]["independently_precommitted_not_bound_count"] == 1
    assert choice_stages[1]["may_depend_on_g1_effect_magnitude"] is False

    legacy_sections = set(resolution["unresolved_required_sections"])
    section_stages = contract["section_staging_proposal"]["stages"]
    staged_sections = [
        section for stage in section_stages for section in stage["section_ids"]
    ]
    assert len(staged_sections) == len(set(staged_sections)) == 15
    assert set(staged_sections) == legacy_sections
    assert [stage["section_count"] for stage in section_stages] == [13, 1, 1]


def test_independent_verifier_matches_frozen_summary() -> None:
    completed = _run(str(VERIFIER))

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    expected = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert observed == expected
    assert observed["status"] == (
        "PASS_SHADOW_SIX_PHASE_SEMANTIC_EQUIVALENCE_NO_AUTHORITY_CHANGE"
    )
    assert observed["phase_mapping_replay"][
        "every_legacy_gate_mapped_exactly_once"
    ]
    assert observed["semantic_equivalence_replay"][
        "authority_relaxation_detected"
    ] is False
    assert observed["decision_staging_replay"][
        "six_later_stage_choices_removed_from_pre_g1_blocker"
    ]
    assert observed["authorization"]["authority_successor"] == "NOT_AUTHORIZED"


def test_shadow_contract_preserves_denials_and_hard_stops() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    equivalence = contract["semantic_equivalence_contract"]

    assert len(equivalence["preserved_current_denials"]) == 16
    assert len(equivalence["preserved_current_claim_boundaries"]) == 8
    assert equivalence["implicit_deny_for_actions_not_explicitly_allowed"] is True
    assert equivalence["shadow_allowed_actions"] == [
        "GOVERNANCE_DESIGN_REVIEW",
        "REPOSITORY_ONLY_SEMANTIC_EQUIVALENCE_VALIDATION",
    ]
    boundaries = contract["irreversible_boundaries"]
    assert {item["boundary_id"] for item in boundaries} == {
        "FIRST_CONFIRMATORY_READ",
        "FIRST_MODEL_SELECTION_SCORE_READ",
        "FIRST_HOLDOUT_OPEN",
        "CLAIM_PROMOTION",
    }
    assert all(item["fail_closed"] is True for item in boundaries)


def test_duplicate_legacy_mapping_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(document: dict) -> None:
        document["proposed_live_phases"][1]["legacy_gate_ids"].append("S0")

    message = _verify_mutation(tmp_path, monkeypatch, mutate)

    assert "legacy mapping" in message


def test_removed_current_denial_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(document: dict) -> None:
        document["semantic_equivalence_contract"]["preserved_current_denials"].pop()

    message = _verify_mutation(tmp_path, monkeypatch, mutate)

    assert "denials" in message


def test_duplicate_decision_choice_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(document: dict) -> None:
        document["decision_staging_proposal"]["stages"][1]["choices"][0] = (
            "g1_replication_tolerance"
        )

    message = _verify_mutation(tmp_path, monkeypatch, mutate)

    assert "staged choice" in message


def test_non_fail_closed_irreversible_boundary_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(document: dict) -> None:
        document["irreversible_boundaries"][0]["fail_closed"] = False

    message = _verify_mutation(tmp_path, monkeypatch, mutate)

    assert "fail closed" in message


def test_premature_authority_successor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(document: dict) -> None:
        document["migration_contract"]["steps"][3]["state"] = "AUTHORIZED"

    message = _verify_mutation(tmp_path, monkeypatch, mutate)

    assert "future migration states" in message


def test_documented_hashes_and_local_links_resolve() -> None:
    text = NOTE.read_text(encoding="utf-8")

    for path in (CONTRACT, VERIFIER, SUMMARY):
        assert _sha256(path) in text
    assert "gate v61" in text
    assert "does not depend on G1 effect magnitude" not in text
    assert "may not depend on G1 effect magnitude" in text
    destinations = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
    assert destinations
    for destination in destinations:
        path = (NOTE.parent / destination.split("#", 1)[0]).resolve()
        assert path.exists(), destination
