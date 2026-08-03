from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments/configs/pivot_feedback_gate_state_v61.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_v61_binds_exact_g0_v4_evidence() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))

    assert gate["schema_version"] == "pivot_feedback_gate_state.v61"
    assert gate["status"] == (
        "VALID_G0_V4_TWELVE_RESOLVED_SEVEN_PENDING_"
        "G1_TOLERANCE_PRECOMMITMENT_AUTHORIZED"
    )
    parent = gate["additive_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    verified = gate["verified_g0_v4_binding"]
    for key in (
        "contract",
        "independent_validator",
        "research_note",
        "summary_json",
        "summary_markdown",
    ):
        binding = verified[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
    summary = json.loads(
        (ROOT / verified["summary_json"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["schema_version"] == verified["summary_json"][
        "required_schema_version"
    ]
    assert summary["status"] == verified["summary_json"]["required_status"]
    assert summary["repository_artifacts"]["g0_v4"]["sha256"] == (
        verified["contract"]["sha256"]
    )
    assert summary["repository_artifacts"]["validator"]["sha256"] == (
        verified["independent_validator"]["sha256"]
    )


def test_gate_v61_records_exact_remaining_boundary() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    result = gate["verified_g0_v4_binding"]["result"]

    assert result["predecessor_preserved_top_level_subtree_count"] == 10
    assert result["inherited_resolved_choice_count"] == 6
    assert result["newly_bound_choice_count"] == 6
    assert result["total_resolved_choice_count"] == 12
    assert result["remaining_quantitative_choice_count"] == 7
    assert result["pending_required_section_count"] == 15
    assert result["freeze_ready_required_section_count"] == 0
    assert result["holdout_receipt_bound"] is False
    assert result["discovery_production_application_run"] is False
    assert result["confirmatory_or_holdout_rows_read"] == 0
    assert gate["binding_decision"]["g0_passed"] is False
    packages = gate["remaining_choice_packages"]
    assert [item["choice_count"] for item in packages] == [1, 3, 3]
    assert sum(item["choice_count"] for item in packages) == 7


def test_gate_v61_authorizes_only_g1_tolerance_precommitment() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scope = gate["authorized_next_precommitment_scope"]

    assert scope["exact_choice_ids"] == ["g1_replication_tolerance"]
    for key in (
        "must_bind_this_gate_byte_for_byte",
        "must_bind_g0_v4_byte_for_byte",
        "must_keep_value_outcome_blind",
        "must_define_estimand_scale_direction_interval_and_failure_rule",
        "must_label_judgment_based_value_as_policy_not_empirical_truth",
        "must_include_independent_closed_fixture_validation",
    ):
        assert scope[key] is True
    for key in (
        "may_write_or_modify_g0_v4",
        "may_bind_value_into_g0_successor",
        "may_resolve_g2_or_g3_choices",
        "may_complete_required_sections",
        "may_apply_rule_to_g1_or_production_data",
        "may_read_g1_confirmatory_or_holdout_rows",
        "may_change_temporal_partitions",
        "may_select_g4_field_or_episode",
        "may_materialize_g4_artifacts",
        "may_create_holdout_preflight_or_seal",
        "may_pass_g0",
        "may_fit_model",
        "may_open_g4",
    ):
        assert scope[key] is False

    assert gate["gate_state"]["G0"]["complete"] is False
    assert gate["gate_state"]["G4"]["state"] == (
        "LOGICALLY_RESERVED_NOT_BYTE_SEALED_NOT_OPENED"
    )
    authorization = gate["authorization"]
    assert authorization["g1_replication_tolerance_precommitment_closed_fixture"] == (
        "AUTHORIZED_EXACT_SCOPE"
    )
    assert authorization["g0_successor_binding_for_g1_tolerance"] == (
        "NOT_AUTHORIZED"
    )
    assert authorization["g2_or_g3_choice_precommitment"] == "NOT_AUTHORIZED"
    assert authorization["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert authorization["model_fitting"] == "NOT_AUTHORIZED"
    assert authorization["g4_open"] == "NOT_AUTHORIZED"
