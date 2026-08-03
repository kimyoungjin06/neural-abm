from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_measurement_proxy_precommitment_v1.yaml"
)
APPLICATION = (
    ROOT
    / "scripts/apply_pivot_feedback_g0_measurement_proxy_precommitment_v1.py"
)
VERIFIER = (
    ROOT
    / "scripts/verify_pivot_feedback_g0_measurement_proxy_precommitment_v1.py"
)
FIXTURE_INPUT = (
    ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_measurement_proxy_discovery_v1.input.json"
)
FIXTURE_EXPECTED = (
    ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_measurement_proxy_discovery_v1.expected.json"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


APPLY_MODULE = _load(APPLICATION, "measurement_proxy_application_test_module")
VERIFY_MODULE = _load(VERIFIER, "measurement_proxy_verifier_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mutated_input(tmp_path: Path, mutate) -> Path:
    value = json.loads(FIXTURE_INPUT.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _mutated_contract(tmp_path: Path, mutate) -> Path:
    value = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    mutate(value)
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_contract_binds_exact_parent_and_design_sources() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    assert contract["schema_version"] == (
        "pivot_feedback_g0_measurement_proxy_precommitment.v1"
    )
    assert contract["status"] == (
        "FROZEN_FOUR_MEASUREMENT_PROXY_CHOICES_PENDING_INDEPENDENT_VALIDATION_"
        "NO_G0_PROMOTION"
    )
    bindings = {
        "authorization_parent": contract["authorization_parent"],
        **contract["bound_design_sources"],
    }
    assert set(bindings) == set(VERIFY_MODULE.EXPECTED_BINDINGS)
    for binding in bindings.values():
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
        assert binding["mutation_allowed"] is False


def test_contract_freezes_four_policy_choices_without_g0_promotion() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    choices = contract["precommitment_choices"]

    assert contract["scope"]["choice_ids"] == VERIFY_MODULE.CHOICE_IDS
    assert choices["attention_surge_threshold"][
        "primary_modified_z_threshold_inclusive"
    ] == 3.0
    assert choices["public_success_threshold"]["target_fraction"] == 0.10
    assert choices["public_success_threshold"]["boundary_tie_rule"] == (
        "INCLUDE_ALL_WORKS_AT_OR_ABOVE_CUTOFF"
    )
    assert choices["citation_maturation_lag_years"]["value"] == 3
    margins = choices["negative_control_equivalence_margins"]
    assert margins["symmetric_practical_null_lower"] == -0.01
    assert margins["symmetric_practical_null_upper"] == 0.01
    assert margins["equivalence_test"] == "TWO_ONE_SIDED_TESTS"
    assert margins["one_sided_alpha"] == 0.05
    assert contract["lifecycle"]["g0_v2_changed"] is False
    assert contract["authorization"]["g0_measurement_freeze"] == "NOT_AUTHORIZED"


def test_application_matches_frozen_expected_fixture_byte_values() -> None:
    result = APPLY_MODULE.run(FIXTURE_INPUT, CONTRACT)
    expected = json.loads(FIXTURE_EXPECTED.read_text(encoding="utf-8"))

    assert result == expected
    attention = result["attention_surge_application"]
    assert attention["primary_surge_count"] == 1
    assert attention["mad_zero_group_count"] == 1
    assert attention["groups"][1]["mad_zero_fail_closed"] is True
    citations = result["public_success_application"]
    assert citations["groups"][0]["realized_public_success_fraction"] == 0.1
    assert citations["groups"][1]["realized_public_success_fraction"] == 0.15
    assert citations["groups"][1]["inclusive_tie_inflation_count"] == 1
    assert citations["groups"][2]["status"] == (
        "UNRESOLVED_INSUFFICIENT_REFERENCE_COHORT"
    )


def test_independent_verifier_replays_sources_choices_and_fixture() -> None:
    result = VERIFY_MODULE.run(CONTRACT, FIXTURE_INPUT, FIXTURE_EXPECTED)

    assert result["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert result["choice_replay"] == {
        "choice_count": 4,
        "attention_modified_z_threshold": 3.0,
        "public_success_target_fraction": 0.1,
        "citation_maturation_lag_years": 3,
        "negative_control_practical_null": [-0.01, 0.01],
        "tost_one_sided_alpha": 0.05,
    }
    assert result["fixture_replay"]["attention_primary_surge_count"] == 1
    assert result["fixture_replay"]["attention_mad_zero_group_count"] == 1
    assert result["fixture_replay"]["citation_unresolved_group_count"] == 1
    assert result["fixture_replay"]["citation_inclusive_tie_inflation_total"] == 1
    assert result["application_tool"]["row_data_library_imports"] == []
    assert result["lifecycle_replay"]["g0_v2_changed"] is False
    assert result["lifecycle_replay"]["remaining_choice_count_after_future_binding"] == 13


def test_application_rejects_post_discovery_attention_year(tmp_path: Path) -> None:
    path = _mutated_input(
        tmp_path,
        lambda value: value["attention_topic_year_rows"][0].__setitem__(
            "decision_year", 2013
        ),
    )
    with pytest.raises(
        APPLY_MODULE.MeasurementProxyApplicationError,
        match="post-discovery decision year",
    ):
        APPLY_MODULE.run(path, CONTRACT)


def test_application_rejects_negative_citation_count(tmp_path: Path) -> None:
    path = _mutated_input(
        tmp_path,
        lambda value: value["citation_work_rows"][0].__setitem__(
            "citation_count_age_3", -1
        ),
    )
    with pytest.raises(
        APPLY_MODULE.MeasurementProxyApplicationError,
        match="citation count below minimum",
    ):
        APPLY_MODULE.run(path, CONTRACT)


def test_application_rejects_duplicate_attention_key(tmp_path: Path) -> None:
    path = _mutated_input(
        tmp_path,
        lambda value: value["attention_topic_year_rows"].append(
            dict(value["attention_topic_year_rows"][0])
        ),
    )
    with pytest.raises(
        APPLY_MODULE.MeasurementProxyApplicationError,
        match="duplicate attention row",
    ):
        APPLY_MODULE.run(path, CONTRACT)


def test_application_output_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    assert APPLY_MODULE.main(
        ["--input", str(FIXTURE_INPUT), "--contract", str(CONTRACT), "--output", str(output)]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        APPLY_MODULE.OUTPUT_STATUS
    )
    with pytest.raises(
        APPLY_MODULE.MeasurementProxyApplicationError, match="output exists"
    ):
        APPLY_MODULE.main(
            [
                "--input",
                str(FIXTURE_INPUT),
                "--contract",
                str(CONTRACT),
                "--output",
                str(output),
            ]
        )


def test_verifier_rejects_threshold_relaxation(tmp_path: Path) -> None:
    path = _mutated_contract(
        tmp_path,
        lambda value: value["precommitment_choices"]["attention_surge_threshold"].__setitem__(
            "primary_modified_z_threshold_inclusive", 2.0
        ),
    )
    with pytest.raises(
        VERIFY_MODULE.MeasurementProxyPrecommitmentError,
        match="z threshold",
    ):
        VERIFY_MODULE.run(path, FIXTURE_INPUT, FIXTURE_EXPECTED)


def test_verifier_rejects_equivalence_margin_drift(tmp_path: Path) -> None:
    path = _mutated_contract(
        tmp_path,
        lambda value: value["precommitment_choices"][
            "negative_control_equivalence_margins"
        ].__setitem__("symmetric_practical_null_upper", 0.02),
    )
    with pytest.raises(
        VERIFY_MODULE.MeasurementProxyPrecommitmentError,
        match="upper margin",
    ):
        VERIFY_MODULE.run(path, FIXTURE_INPUT, FIXTURE_EXPECTED)
