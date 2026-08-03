from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/verify_pivot_feedback_g0_measurement_freeze.py"
CONTRACT_PATH = (
    ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml"
)
PEER_PATH = ROOT / "experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml"
MATRIX_PATH = ROOT / "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml"

spec = importlib.util.spec_from_file_location(
    "test_pivot_feedback_g0_measurement_freeze_module",
    SCRIPT_PATH,
)
assert spec is not None and spec.loader is not None
verifier = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = verifier
spec.loader.exec_module(verifier)


def _raw_contract() -> dict[str, Any]:
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_contract(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "g0.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _copy_parent_tree(tmp_path: Path) -> Path:
    for source in (PEER_PATH, MATRIX_PATH):
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return tmp_path


def test_production_contract_binds_both_parent_v0_files_byte_exactly() -> None:
    raw = _raw_contract()
    parents = raw["foundation_parents"]

    assert verifier.sha256_file(PEER_PATH) == verifier.PEER_FILE_SHA256
    assert verifier.sha256_file(MATRIX_PATH) == verifier.MATRIX_FILE_SHA256
    assert parents["peer_mechanism_registry"]["required_file_sha256"] == (
        verifier.PEER_FILE_SHA256
    )
    assert parents["cross_field_matrix"]["required_file_sha256"] == (
        verifier.MATRIX_FILE_SHA256
    )

    peer = yaml.safe_load(PEER_PATH.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    assert verifier.canonical_sha256(peer) == verifier.PEER_CANONICAL_SHA256
    assert verifier.canonical_sha256(matrix) == verifier.MATRIX_CANONICAL_SHA256
    assert parents["peer_mechanism_registry"]["required_canonical_sha256"] == (
        verifier.PEER_CANONICAL_SHA256
    )
    assert parents["cross_field_matrix"]["required_canonical_sha256"] == (
        verifier.MATRIX_CANONICAL_SHA256
    )


def test_pending_contract_validates_without_promotion_or_execution() -> None:
    loaded = verifier.load_contract(CONTRACT_PATH)
    contract = loaded.value
    summary = verifier.validation_summary(loaded)

    assert contract.schema_version == "pivot_feedback_g0_measurement_freeze.v1"
    assert contract.lifecycle.freeze_status == "PENDING_G0_MEASUREMENT_FREEZE"
    assert contract.lifecycle.pass_g0 is False
    assert contract.lifecycle.pass_g1 is False
    assert contract.lifecycle.execution_authorized is False
    assert contract.lifecycle.confirmatory_outcomes_inspected is False
    assert contract.lifecycle.g4_holdout_opened is False
    assert contract.g_minus_1_binding.state == "PENDING"
    assert contract.holdout_access_binding.state == "PENDING"
    assert contract.resolution.state == "PENDING"
    assert contract.resolution.resolved_measurements is None
    assert contract.resolution.unresolved_required_sections == (
        verifier.REQUIRED_RESOLUTION_SECTIONS
    )
    unresolved = contract.resolution.unresolved_quantitative_choices.model_dump()
    assert unresolved
    assert set(unresolved.values()) == {None}

    assert summary["status"] == "VALID_PENDING_G0_MEASUREMENT_FREEZE"
    assert summary["gate_state"] == {
        "g_minus_1_data_readiness": "PENDING",
        "holdout_access_seal": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g0": False,
        "pass_g1": False,
        "execution_authorized": False,
    }
    assert summary["sealable"] is False
    assert summary["scientific_effect"] == "NOT_ESTABLISHED"
    assert summary["parent_bindings"]["validated"] is True
    assert not str(summary).startswith(str(ROOT))


def test_documented_fixed_design_choices_are_explicit_and_exact() -> None:
    fixed = verifier.load_contract(CONTRACT_PATH).value.fixed_design_choices
    clocks = fixed.temporal_clocks

    assert clocks.resolution == "ANNUAL"
    assert (
        clocks.portfolio_history.start_offset,
        clocks.portfolio_history.end_offset,
    ) == (-3, -1)
    assert (
        clocks.pre_entry_graph.start_offset,
        clocks.pre_entry_graph.end_offset,
    ) == (-5, -1)
    assert (
        clocks.attention_acceleration.recent_window.start_offset,
        clocks.attention_acceleration.recent_window.end_offset,
    ) == (-3, -1)
    assert (
        clocks.attention_acceleration.reference_window.start_offset,
        clocks.attention_acceleration.reference_window.end_offset,
    ) == (-6, -4)
    assert (
        clocks.persistence_outcome.start_offset,
        clocks.persistence_outcome.end_offset,
    ) == (1, 3)
    assert clocks.response_offsets == (0, 1, 2)
    assert clocks.pretrend_offsets == (-3, -2, -1)
    assert clocks.public_availability_rule == (
        "AVAILABLE_AT_LESS_THAN_OR_EQUAL_TO_T_MINUS_1"
    )

    assert fixed.h3_primary.minimum_point_gap_cohort_years == 1
    assert fixed.h3_primary.endpoint == (
        "EARLIEST_ARGMAX_ADJUSTED_ENTRY_HAZARD_K_0_1_2_MINUS_"
        "EARLIEST_ARGMAX_ADJUSTED_THREE_YEAR_PERSISTENCE_PROBABILITY_K_0_1_2"
    )
    assert fixed.h3_primary.interval_sidedness == "ONE_SIDED"
    assert fixed.h3_primary.confidence_level == 0.95
    assert fixed.h3_primary.lower_bound_rule == "GREATER_THAN_ZERO"
    assert fixed.h3_primary.field_weighting == "EQUAL"
    assert fixed.h3_primary.within_field_episode_weighting == "EQUAL"
    assert fixed.h3_primary.leave_one_field_out_direction_rule == "ALL_POSITIVE"
    assert fixed.h3_primary.alternative_pivot_construction_count == 1
    assert fixed.h3_primary.alternative_pivot_predeclared is True
    assert fixed.h3_primary.alternative_pivot_substantively_different is True
    assert fixed.multiplicity.first_test == "H3"
    assert fixed.multiplicity.family_wise_alpha == 0.05
    assert fixed.multiplicity.conditional_family == ("H1", "H2", "H4")
    assert fixed.multiplicity.conditional_method == "HOLM"

    assert fixed.full_panel.field_count == 4
    assert fixed.full_panel.g1_field_count == 3
    assert fixed.full_panel.sealed_g4_field_count == 1
    assert fixed.full_panel.minimum_independent_episodes_per_field == 3
    assert fixed.discovery_pilot.label == "DISCOVERY_ONLY"
    assert fixed.discovery_pilot.field_count == 2
    assert fixed.discovery_pilot.independent_shocks_per_field == 2
    assert fixed.discovery_pilot.may_pass_g1 is False

    assert fixed.peer_mechanism.negative_control_ids == verifier.NEGATIVE_CONTROL_IDS
    assert fixed.peer_mechanism.capacity_cells == verifier.CAPACITY_CELLS
    assert fixed.peer_mechanism.entry_mode_ids == verifier.ENTRY_MODE_IDS
    assert fixed.engineering_rng.base_seed == 20260718
    assert fixed.engineering_rng.replicate_seeds == (1, 2, 3)
    assert fixed.engineering_rng.role == "ENGINEERING_PAIRING_ONLY"
    assert fixed.engineering_rng.monte_carlo_sufficiency_established is False
    assert fixed.engineering_rng.confirmatory_draw_count_frozen is False

    observed_partitions = tuple(
        (
            partition.partition_id,
            partition.entry_start_year,
            partition.entry_end_year,
            partition.outcome_followup_through_year,
            partition.sealed,
        )
        for partition in fixed.temporal_partitions
    )
    assert observed_partitions == verifier.EXPECTED_TEMPORAL_PARTITIONS


def test_inherited_subtree_digests_preserve_claim_and_partition_boundaries() -> None:
    contract = verifier.load_contract(CONTRACT_PATH).value
    peer = yaml.safe_load(PEER_PATH.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    peer_digests = contract.inherited_subtree_digests.peer_mechanism_registry
    matrix_digests = contract.inherited_subtree_digests.cross_field_matrix

    assert verifier.canonical_sha256(peer["exposure_constructs"]) == (
        peer_digests.exposure_constructs_sha256
    )
    assert verifier.canonical_sha256(peer["entry_mode_strata"]) == (
        peer_digests.entry_mode_strata_sha256
    )
    assert verifier.canonical_sha256(peer["negative_controls"]) == (
        peer_digests.negative_controls_sha256
    )
    assert verifier.canonical_sha256(peer["claim_levels"]) == (
        peer_digests.claim_levels_sha256
    )
    assert verifier.canonical_sha256(peer["claim_downgrade_rules"]) == (
        peer_digests.claim_downgrade_rules_sha256
    )
    assert verifier.canonical_sha256(matrix["candidate_universe"]) == (
        matrix_digests.candidate_universe_sha256
    )
    assert verifier.canonical_sha256(matrix["selection_rules"]) == (
        matrix_digests.selection_rules_sha256
    )
    assert verifier.canonical_sha256(matrix["temporal_partitions"]) == (
        matrix_digests.temporal_partitions_sha256
    )
    assert len(peer["negative_controls"]) == 7
    assert len(peer["claim_downgrade_rules"]) == 10
    assert contract.inherited_subtree_digests.override_allowed is False


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_top_level",
        "authorize_execution",
        "claim_g0_pass",
        "claim_g1_pass",
        "claim_outcomes_inspected",
        "open_g4",
        "claim_frozen",
        "resolve_inline",
        "drop_resolution_section",
        "allow_override",
        "allow_pending_seal",
        "drop_blocked_command",
        "promote_g_minus_1_without_receipt",
        "change_portfolio_clock",
        "allow_current_year_availability",
        "weaken_h3_gap",
        "change_h3_endpoint",
        "change_h3_weighting",
        "weaken_h3_leave_one_field_out",
        "drop_h3_alternative_pivot",
        "change_multiplicity_family",
        "open_pilot_claim",
        "drop_negative_control",
        "drop_capacity_cell",
        "drop_entry_mode",
        "claim_mc_sufficiency",
        "change_engineering_replicate_seeds",
        "fill_unfrozen_threshold",
    ],
)
def test_schema_rejects_pending_contract_promotion_and_incomplete_mutations(
    mutation: str,
) -> None:
    raw = copy.deepcopy(_raw_contract())
    if mutation == "extra_top_level":
        raw["unexpected"] = True
    elif mutation == "authorize_execution":
        raw["lifecycle"]["execution_authorized"] = True
    elif mutation == "claim_g0_pass":
        raw["lifecycle"]["pass_g0"] = True
    elif mutation == "claim_g1_pass":
        raw["lifecycle"]["pass_g1"] = True
    elif mutation == "claim_outcomes_inspected":
        raw["lifecycle"]["confirmatory_outcomes_inspected"] = True
    elif mutation == "open_g4":
        raw["lifecycle"]["g4_holdout_opened"] = True
    elif mutation == "claim_frozen":
        raw["resolution"]["state"] = "FROZEN"
    elif mutation == "resolve_inline":
        raw["resolution"]["resolved_measurements"] = {"h3": "invented"}
    elif mutation == "drop_resolution_section":
        raw["resolution"]["unresolved_required_sections"].pop()
    elif mutation == "allow_override":
        raw["inherited_subtree_digests"]["override_allowed"] = True
    elif mutation == "allow_pending_seal":
        raw["completion_contract"]["pending_contract_sealable"] = True
    elif mutation == "drop_blocked_command":
        raw["failure_policy"]["forbidden_commands"].pop()
    elif mutation == "promote_g_minus_1_without_receipt":
        raw["g_minus_1_binding"]["state"] = "PASS"
    elif mutation == "change_portfolio_clock":
        raw["fixed_design_choices"]["temporal_clocks"]["portfolio_history"][
            "start_offset"
        ] = -4
    elif mutation == "allow_current_year_availability":
        raw["fixed_design_choices"]["temporal_clocks"]["public_availability_rule"] = (
            "AVAILABLE_AT_LESS_THAN_OR_EQUAL_TO_T"
        )
    elif mutation == "weaken_h3_gap":
        raw["fixed_design_choices"]["h3_primary"]["minimum_point_gap_cohort_years"] = 0
    elif mutation == "change_h3_endpoint":
        raw["fixed_design_choices"]["h3_primary"]["endpoint"] = (
            "LATEST_ARGMAX_MINUS_EARLIEST_ARGMAX"
        )
    elif mutation == "change_h3_weighting":
        raw["fixed_design_choices"]["h3_primary"]["within_field_episode_weighting"] = (
            "SIZE_WEIGHTED"
        )
    elif mutation == "weaken_h3_leave_one_field_out":
        raw["fixed_design_choices"]["h3_primary"][
            "leave_one_field_out_direction_rule"
        ] = "MAJORITY_POSITIVE"
    elif mutation == "drop_h3_alternative_pivot":
        raw["fixed_design_choices"]["h3_primary"][
            "alternative_pivot_construction_count"
        ] = 0
    elif mutation == "change_multiplicity_family":
        raw["fixed_design_choices"]["multiplicity"]["conditional_family"] = [
            "H1",
            "H2",
        ]
    elif mutation == "open_pilot_claim":
        raw["fixed_design_choices"]["discovery_pilot"]["may_pass_g1"] = True
    elif mutation == "drop_negative_control":
        raw["fixed_design_choices"]["peer_mechanism"]["negative_control_ids"].pop()
    elif mutation == "drop_capacity_cell":
        raw["fixed_design_choices"]["peer_mechanism"]["capacity_cells"].pop()
    elif mutation == "drop_entry_mode":
        raw["fixed_design_choices"]["peer_mechanism"]["entry_mode_ids"].pop()
    elif mutation == "claim_mc_sufficiency":
        raw["fixed_design_choices"]["engineering_rng"][
            "monte_carlo_sufficiency_established"
        ] = True
    elif mutation == "change_engineering_replicate_seeds":
        raw["fixed_design_choices"]["engineering_rng"]["replicate_seeds"] = [
            1,
            2,
            4,
        ]
    elif mutation == "fill_unfrozen_threshold":
        raw["resolution"]["unresolved_quantitative_choices"][
            "attention_surge_threshold"
        ] = 2.0
    else:  # pragma: no cover - parametrization owns the values
        raise AssertionError(mutation)

    with pytest.raises(ValidationError):
        verifier.G0MeasurementFreezeContract.model_validate(raw)


@pytest.mark.parametrize("mutation", ["false_from_zero", "one_from_true"])
def test_contract_loader_rejects_scalar_type_coercion(
    tmp_path: Path,
    mutation: str,
) -> None:
    raw = copy.deepcopy(_raw_contract())
    if mutation == "false_from_zero":
        raw["lifecycle"]["execution_authorized"] = 0
    elif mutation == "one_from_true":
        raw["fixed_design_choices"]["h3_primary"]["minimum_point_gap_cohort_years"] = (
            True
        )
    else:  # pragma: no cover - parametrization owns the values
        raise AssertionError(mutation)

    with pytest.raises(verifier.G0ContractError, match="scalar type differs"):
        verifier.load_contract(_write_contract(tmp_path, raw))


def test_contract_loader_rejects_oversize_input_before_payload_parse(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "oversized.yaml"
    oversized.write_bytes(b" " * (verifier.MAX_CONTRACT_BYTES + 1))

    with pytest.raises(verifier.G0ContractError, match="1 MiB safety cap"):
        verifier.load_contract(oversized)


def test_contract_loader_rejects_symlink_input(tmp_path: Path) -> None:
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(CONTRACT_PATH)

    with pytest.raises(verifier.G0ContractError, match="regular file"):
        verifier.load_contract(linked)


def test_contract_loader_and_cli_reject_excessive_yaml_nesting(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested.yaml"
    depth = verifier.MAX_CONTRACT_NESTING_DEPTH + 1
    nested.write_text("[" * depth + "null" + "]" * depth, encoding="utf-8")

    with pytest.raises(verifier.G0ContractError, match="nesting safety cap"):
        verifier.load_contract(nested)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "validate", "--contract", str(nested)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload["status"] == "BLOCKED_INVALID_G0_MEASUREMENT_FREEZE_CONTRACT"
    assert payload["pass_g0"] is False
    assert payload["execution_authorized"] is False
    assert "nesting safety cap" in payload["reason"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            "schema_version: first\nschema_version: second\n",
            "duplicate YAML mapping key",
        ),
        (
            "schema_version: &schema version\nlabel: *schema\n",
            "YAML aliases are forbidden",
        ),
        (
            "schema_version: version\n<<: {label: merged}\n",
            "YAML tag is forbidden",
        ),
        (
            "schema_version: !custom version\n",
            "YAML tag is forbidden",
        ),
    ],
)
def test_contract_loader_rejects_ambiguous_or_custom_yaml(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    path = tmp_path / "ambiguous.yaml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(yaml.YAMLError, match=message):
        verifier.load_contract(path)


@pytest.mark.parametrize(
    ("relative_parent", "message"),
    [
        (
            "experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml",
            "peer registry file SHA-256 mismatch",
        ),
        (
            "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml",
            "cross-field matrix file SHA-256 mismatch",
        ),
    ],
)
def test_parent_byte_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_parent: str,
    message: str,
) -> None:
    fake_root = _copy_parent_tree(tmp_path / "repo")
    parent = fake_root / relative_parent
    parent.write_bytes(parent.read_bytes() + b"\n")
    monkeypatch.setattr(verifier, "REPO_ROOT", fake_root)

    with pytest.raises(verifier.G0ContractError, match=message):
        verifier.load_contract(CONTRACT_PATH)


def test_canonical_contract_hash_is_stable_under_yaml_key_reordering(
    tmp_path: Path,
) -> None:
    reordered = tmp_path / "reordered.yaml"
    reordered.write_text(
        yaml.safe_dump(_raw_contract(), sort_keys=True),
        encoding="utf-8",
    )

    original = verifier.load_contract(CONTRACT_PATH)
    resolved = verifier.load_contract(reordered)

    assert original.file_sha256 != resolved.file_sha256
    assert original.canonical_sha256 == resolved.canonical_sha256


def test_cli_validate_reports_valid_pending_contract() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "validate"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == "VALID_PENDING_G0_MEASUREMENT_FREEZE"
    assert payload["gate_state"]["g0_measurement_freeze"] == "PENDING"
    assert payload["gate_state"]["pass_g0"] is False
    assert payload["gate_state"]["execution_authorized"] is False
    assert payload["sealable"] is False


def test_cli_seal_fails_closed_without_writing_a_pass_receipt() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "seal"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload["status"] == "BLOCKED_PENDING_G0_MEASUREMENT_FREEZE"
    assert payload["g0_measurement_freeze"] == "PENDING"
    assert payload["pass_g0"] is False
    assert payload["execution_authorized"] is False
    assert payload["successor_required"] is True


@pytest.mark.parametrize("command", ["run", "execute", "authorize", "promote"])
def test_cli_execution_and_promotion_commands_fail_closed(command: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload["status"] == "BLOCKED_G0_MEASUREMENT_FREEZE_COMMAND"
    assert payload["requested_command"] == command
    assert payload["g0_measurement_freeze"] == "PENDING"
    assert payload["pass_g0"] is False
    assert payload["execution_authorized"] is False
