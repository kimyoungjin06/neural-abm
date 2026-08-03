from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v40.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v40_binds_verified_raw_component_and_keeps_outcomes_closed() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v40"
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_raw_authorship_identity_affiliation"]
    for key in ("config", "producer", "verifier", "summary_json", "summary_markdown"):
        binding = verified[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
    assert verified["operator_local"]["verification_status"].startswith(
        "PASS_INDEPENDENT_EXACT_FOUR"
    )
    assert verified["result"]["identity_bidirectional_row_difference"] == 0
    assert verified["result"]["institution_bidirectional_row_difference"] == 0
    assert verified["result"]["affiliation_edge_bidirectional_row_difference"] == 0
    assert verified["result"]["author_cap_bidirectional_row_differences"] == [0, 0, 0]
    assert gate["authorized_next_scope"]["single_minimum_history_threshold_selection"] == (
        "NOT_AUTHORIZED"
    )
    assert gate["authorized_next_scope"]["graph_team_cap_selection"] == "NOT_AUTHORIZED"
    authorization = gate["authorization"]
    assert authorization["d0_peer_visible_success"] == "NOT_AUTHORIZED"
    assert authorization["d0_production_pivot_classification"] == "NOT_AUTHORIZED"
    assert authorization["d0_discovery_outcome_opening"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
