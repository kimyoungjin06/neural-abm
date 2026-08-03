from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_certified_predecision_author_measurement_v1.yaml"
)
BUILDER_PATH = (
    ROOT
    / "scripts/build_pivot_feedback_d0_certified_predecision_author_measurement_v1.py"
)
VERIFIER_PATH = (
    ROOT
    / "scripts/verify_pivot_feedback_d0_certified_predecision_author_measurement_v1.py"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _module(BUILDER_PATH, "predecision_author_measurement_builder")
verifier = _module(VERIFIER_PATH, "predecision_author_measurement_verifier")


def test_contract_registers_full_base_without_history_or_cap_selection() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    membership = config["registered_author_membership"]

    assert membership["rule_id"] == "EXACT_VERIFIED_AUTHOR_OPPORTUNITY_BASE_V1"
    assert membership["upstream_structural_predicate"] == {
        "column": "portfolio_window_focal_field_work_count_all",
        "operator": "GREATER_THAN_OR_EQUAL_TO",
        "value": 1,
    }
    assert membership["single_minimum_history_threshold_selected"] is False
    assert membership["threshold_specific_membership_flags_materialized"] is False
    assert membership["continuous_history_counts_retained"] is True
    assert membership["graph_team_cap_profiles_retained_nonranked"] == [10, 50, 100]
    assert membership["graph_team_cap_selected"] is False
    assert config["join_contract"]["required_rows"] == 3_256_271
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def test_column_classifier_keeps_snapshot_sensitivities_out_of_t_minus_1() -> None:
    assert builder._availability_for("pre_entry_target_topic_work_count") == (
        "PREDECISION_T_MINUS_1",
        "PREDECISION_AUTHOR_MEASUREMENT",
    )
    assert builder._availability_for("orcid_stable_single_valid_token") == (
        "CURRENT_SNAPSHOT_SENSITIVITY",
        "ORCID_SUPPORT_OR_STABILITY_SENSITIVITY",
    )
    assert builder._availability_for(
        "has_institution_comparable_prior_peer_cap_100"
    ) == (
        "CURRENT_SNAPSHOT_WORK_TIME_PROXY",
        "AFFILIATION_COVERAGE_OR_BRIDGE_PROXY",
    )
    assert builder._availability_for("registered_measurement_member") == (
        "REGISTERED_MEMBERSHIP",
        "OUTCOME_BLIND_STRUCTURAL_MEMBERSHIP",
    )
    assert verifier._availability_for("nonnull_orcid_row_count")[0] == (
        "CURRENT_SNAPSHOT_SENSITIVITY"
    )


def test_independent_sorted_unique_check_rejects_duplicate_and_reverse_keys() -> None:
    schema = pa.schema(
        [
            pa.field("certified_episode_index", pa.uint8()),
            pa.field("field_id", pa.uint8()),
            pa.field("topic_id", pa.uint32()),
            pa.field("entry_year", pa.uint16()),
            pa.field("author_numeric_id", pa.uint64()),
        ]
    )
    sorted_table = pa.Table.from_pylist(
        [
            {
                "certified_episode_index": 1,
                "field_id": 17,
                "topic_id": 100,
                "entry_year": 2000,
                "author_numeric_id": 1,
            },
            {
                "certified_episode_index": 1,
                "field_id": 17,
                "topic_id": 100,
                "entry_year": 2000,
                "author_numeric_id": 2,
            },
            {
                "certified_episode_index": 2,
                "field_id": 27,
                "topic_id": 200,
                "entry_year": 2001,
                "author_numeric_id": 1,
            },
        ],
        schema=schema,
    )
    assert verifier._strictly_sorted_unique(sorted_table) is True
    assert (
        verifier._strictly_sorted_unique(
            pa.concat_tables([sorted_table.slice(0, 1), sorted_table.slice(0, 1)])
        )
        is False
    )
    assert (
        verifier._strictly_sorted_unique(
            pa.concat_tables([sorted_table.slice(1, 1), sorted_table.slice(0, 1)])
        )
        is False
    )


def test_monotonicity_audit_counts_violating_rows() -> None:
    table = pa.table(
        {
            "a": pa.array([0, 2, 3], type=pa.uint32()),
            "b": pa.array([1, 1, 3], type=pa.uint32()),
            "c": pa.array([2, 4, 3], type=pa.uint32()),
        }
    )
    assert verifier._count_violations(table, ["a", "b", "c"]) == 1


def test_registry_has_exact_explicit_boundaries() -> None:
    schema = pa.schema(
        [
            pa.field("author_numeric_id", pa.uint64(), nullable=False),
            pa.field(
                "pre_entry_target_topic_work_count",
                pa.uint32(),
                nullable=False,
            ),
            pa.field(
                "registered_measurement_member",
                pa.bool_(),
                nullable=False,
            ),
            pa.field(
                "orcid_stable_single_valid_token",
                pa.bool_(),
                nullable=False,
            ),
            pa.field(
                "has_same_affiliation_bridge_cap_100",
                pa.bool_(),
                nullable=False,
            ),
        ]
    )
    registry = builder.build_column_registry(schema)
    assert registry["availability_counts"] == {
        "IDENTIFIER_OPERATOR_LOCAL": 1,
        "PREDECISION_T_MINUS_1": 1,
        "REGISTERED_MEMBERSHIP": 1,
        "CURRENT_SNAPSHOT_SENSITIVITY": 1,
        "CURRENT_SNAPSHOT_WORK_TIME_PROXY": 1,
    }
    assert registry["primary_model_input_authorized_now"] is False
    assert registry["raw_numeric_ids_repository_export_authorized"] is False
