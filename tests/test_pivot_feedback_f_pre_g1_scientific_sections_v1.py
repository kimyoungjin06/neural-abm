from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Callable

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "experiments/configs/pivot_feedback_f_pre_g1_scientific_sections_v1.yaml"
)
VERIFIER = ROOT / "scripts/verify_pivot_feedback_f_pre_g1_scientific_sections_v1.py"
SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_pre_g1_scientific_sections_v1_20260803.summary.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("f_sections_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_mutation(tmp_path: Path, mutate: Callable[[dict], None]) -> str:
    verifier = _load_verifier()
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "mutated_f_sections.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(verifier.SectionFreezeVerificationError) as error:
        verifier.verify(path.resolve())
    return str(error.value)


def test_all_thirteen_pre_g1_sections_are_frozen_once() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    assert len(contract["section_order"]) == 13
    assert list(contract["sections"]) == contract["section_order"]
    assert all(
        section["freeze_state"] == "FROZEN_OUTCOME_BLIND"
        for section in contract["sections"].values()
    )
    assert "g2_predictive_gate" not in contract["sections"]
    assert "g3_simulation_gate" not in contract["sections"]
    assert _sha256(CONTRACT) == (
        "cc58a651667b7cecb4cc61d96b6700aa665f1121c7c0b0a254368fb0299041dd"
    )


def test_bound_sources_are_byte_exact() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    bindings = {
        "authorization_parent": contract["authorization_parent"],
        **contract["bound_design_sources"],
    }

    for binding in bindings.values():
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
        assert binding["mutation_allowed"] is False


def test_h3_and_replication_rules_match_bound_policy() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    timing = contract["sections"]["outcome_horizons_and_h3_timing"]

    assert timing["h3"]["event_time_grid"] == [-2, -1, 0, 1, 2]
    assert timing["h3"]["primary_minimum_point_cohort_years"] == 1.0
    assert timing["h3"]["primary_one_sided_95_lower_bound_strictly_greater_than"] == 0.0
    assert timing["replication_noninferiority"]["lower_margin_cohort_years"] == -1.0
    assert timing["replication_noninferiority"]["boundary_counts_as_pass"] is False


def test_section_component_does_not_pass_f_or_open_data() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    completion = contract["completion_accounting"]
    authorization = contract["authorization"]

    assert completion["this_component_is_new_gate"] is False
    assert completion["this_component_alone_may_pass_f"] is False
    assert completion["final_panel_conformance_pending"] is True
    assert completion["atomic_partition_and_holdout_byte_seal_pending"] is True
    assert completion["confirmatory_or_holdout_rows_read"] == 0
    assert authorization["confirmatory_or_holdout_analytic_access"] == "NOT_AUTHORIZED"
    assert authorization["e_g1_execution"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"


def test_verifier_matches_frozen_summary() -> None:
    completed = subprocess.run(
        [sys.executable, str(VERIFIER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    expected = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert observed == expected
    assert observed["status"] == (
        "PASS_THIRTEEN_PRE_G1_SCIENTIFIC_SECTIONS_FROZEN_NO_F_PASS"
    )
    assert observed["section_inventory_replay"]["frozen_section_count"] == 13
    assert observed["fail_closed_replay"]["new_gate_created"] is False


def test_relaxed_attention_threshold_is_rejected(tmp_path: Path) -> None:
    def mutate(document: dict) -> None:
        document["sections"]["topic_attention_and_surge"][
            "surge_threshold_inclusive"
        ] = 2.5

    assert "attention threshold" in _verify_mutation(tmp_path, mutate)


def test_removed_negative_control_is_rejected(tmp_path: Path) -> None:
    def mutate(document: dict) -> None:
        document["sections"]["negative_control_decision_gates"][
            "probability_difference_equivalence"
        ]["control_ids"].pop()

    assert "seven control coverage" in _verify_mutation(tmp_path, mutate)


def test_premature_f_pass_is_rejected(tmp_path: Path) -> None:
    def mutate(document: dict) -> None:
        document["completion_accounting"]["this_component_alone_may_pass_f"] = True

    assert "F pass" in _verify_mutation(tmp_path, mutate)
