from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest
import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_citation_available_at_v1.yaml"
)
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_citation_available_at_v2.py"
DOC_PATH = (
    REPO_ROOT / "docs/case-studies/researcher-pivot/"
    "openalex-citation-available-at-v1-audit-design.md"
)
LEGACY_SURFACES = {
    REPO_ROOT / "experiments/configs/openalex_g1_citation_available_at_v0.yaml": (
        "51f5044459514b5150a8e24b2e4daf05575e06fc366d412bc3d046d87ba3c798"
    ),
    REPO_ROOT / "scripts/audit_openalex_citation_available_at.py": (
        "0be5444c9b72aea95251e609543a0da44d1ff95a432c839b674921e639409bde"
    ),
    REPO_ROOT / "tests/test_openalex_citation_available_at.py": (
        "dfab9957856e985fd2609db0c708cc9d8426ecbb68b24f05390d274b093a41eb"
    ),
    REPO_ROOT / "tests/test_openalex_citation_available_at_design.py": (
        "7121cc3058d564256c3b696566c508966b5478a290beefef716e1be956a801b3"
    ),
    REPO_ROOT / "docs/case-studies/researcher-pivot/"
    "openalex-citation-available-at-audit-design.md": (
        "0a143334d59b3cbddd441b0ccf00c5683ef13d2e96b7da771b0abfee8692ae33"
    ),
}


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "citation_v2_design_module", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config() -> dict[str, Any]:
    value = yaml.load(CONFIG_PATH.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    assert isinstance(value, dict)
    return value


def test_legacy_v0_surfaces_are_preserved_byte_exact() -> None:
    assert {path: _sha256(path) for path in LEGACY_SURFACES} == LEGACY_SURFACES


def test_successor_yaml_is_unique_and_runtime_validator_accepts_it() -> None:
    config = _load_config()
    assert config["schema_version"] == audit.CONFIG_SCHEMA
    audit._validate_config(config)
    with pytest.raises(ConstructorError, match="duplicate key 'status'"):
        yaml.load("status: A\nstatus: B\n", Loader=UniqueKeyLoader)


def test_temporal_v3_parent_has_exact_schema_anchor_and_metric_abis() -> None:
    parent = _load_config()["parents"]["main_temporal_validity"]
    assert parent["parent_version_policy"] == (
        "TEMPORAL_V3_PARENT_ONLY_NO_V1_OR_V2_FALLBACK"
    )
    assert {
        "receipt_schema_version": parent["receipt_schema_version"],
        "required_plan_schema_version": parent["required_plan_schema_version"],
        "required_checkpoint_schema_version": parent[
            "required_checkpoint_schema_version"
        ],
        "required_report_schema_version": parent["required_report_schema_version"],
        "required_state_schema_version": parent["required_state_schema_version"],
    } == {
        "receipt_schema_version": audit.TEMPORAL_V3_OUTPUTS_SCHEMA,
        "required_plan_schema_version": audit.TEMPORAL_V3_PLAN_SCHEMA,
        "required_checkpoint_schema_version": audit.TEMPORAL_V3_CHECKPOINT_SCHEMA,
        "required_report_schema_version": audit.TEMPORAL_V3_REPORT_SCHEMA,
        "required_state_schema_version": audit.TEMPORAL_V3_STATE_SCHEMA,
    }
    assert parent["required_status"] == audit.TEMPORAL_V3_STATUS
    assert {
        key: parent[key] for key in audit.TEMPORAL_V3_ANCHORS
    } == audit.TEMPORAL_V3_ANCHORS
    assert tuple(parent["exact_checkpoint_metric_keyset"]) == (
        audit.TEMPORAL_V3_METRIC_KEYS
    )
    assert len(audit.TEMPORAL_V3_METRIC_KEYS) == 35
    assert parent["direct_reconstruction_recomputes_publication_metric_count"] == 24
    assert parent["created_updated_metric_count_inherited_not_recomputed"] == 11
    assert parent["row_level_temporal_parent_product_exists"] is False
    assert parent["direct_row_identity_claim_allowed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("receipt_schema_version", "openalex_g1_main_temporal_validity_outputs.v1"),
        ("receipt_schema_version", "openalex_g1_main_temporal_validity_outputs.v2"),
        ("required_plan_schema_version", "openalex_g1_main_temporal_validity_plan.v1"),
        ("required_plan_schema_version", "openalex_g1_main_temporal_validity_plan.v2"),
        (
            "required_checkpoint_schema_version",
            "openalex_g1_main_temporal_validity_checkpoint.v1",
        ),
        (
            "required_checkpoint_schema_version",
            "openalex_g1_main_temporal_validity_checkpoint.v2",
        ),
        (
            "required_report_schema_version",
            "openalex_g1_main_temporal_validity_report.v1",
        ),
        (
            "required_report_schema_version",
            "openalex_g1_main_temporal_validity_report.v2",
        ),
        (
            "required_state_schema_version",
            "openalex_g1_main_temporal_validity_state.v1",
        ),
        (
            "required_state_schema_version",
            "openalex_g1_main_temporal_validity_state.v2",
        ),
        (
            "required_status",
            "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_VALIDITY_COMPONENT_WITH_FROZEN_EXCLUSIONS",
        ),
        (
            "required_status",
            "PILOT_MAIN_TEMPORAL_VALIDITY_V2_EXACT_UTC_MILLISECOND_ONLY",
        ),
        (
            "required_config_path",
            "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml",
        ),
        (
            "required_config_path",
            "experiments/configs/openalex_g1_main_temporal_validity_v1.yaml",
        ),
        ("required_tool_path", "scripts/audit_openalex_main_temporal_validity.py"),
        ("required_tool_path", "scripts/audit_openalex_main_temporal_validity_v2.py"),
    ],
)
def test_temporal_v1_v2_schema_status_and_paths_are_rejected(
    field: str,
    value: str,
) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["main_temporal_validity"][field] = value
    with pytest.raises(audit.CitationSyntheticAuditError, match="temporal-v3-only"):
        audit._validate_temporal_v3_parent_spec(config)


@pytest.mark.parametrize("field", tuple(audit.TEMPORAL_V3_ANCHORS))
def test_arbitrary_sha_drift_is_rejected(field: str) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["main_temporal_validity"][field] = "1" * 64
    with pytest.raises(audit.CitationSyntheticAuditError, match="exact anchor drift"):
        audit._validate_temporal_v3_parent_spec(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "required_config_sha256",
            "d208b45cc87de577726a8310a899eb66882a519bae29df5f7dbbf7833562b172",
        ),
        (
            "required_config_sha256",
            "11a668b8f2b10c83e2d36b3f5d2cd16f1e5f86b41ef64f58c79f10cabebc30e8",
        ),
        (
            "required_tool_sha256",
            "b4b39c016ba4c6867a3e08366f70f6125230c2a0a18c1e8c45d40826f373f563",
        ),
        (
            "required_tool_sha256",
            "b86f2496bbf7258c4e981f9a7ad8b1618fdcdc34a7385de628b133e0082a89dc",
        ),
        (
            "required_receipt_sha256",
            "f6eec88337fb8482400ad3a25c7b7b7099b1613d575e0c29c6c26575e8612780",
        ),
        (
            "required_receipt_sha256",
            "ec59f3c649baa4a6a5333daace92a5e46f6492fad62e4c1ed42a1702394425f4",
        ),
    ],
)
def test_prior_temporal_hashes_are_rejected_explicitly(field: str, value: str) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["main_temporal_validity"][field] = value
    with pytest.raises(
        audit.CitationSyntheticAuditError,
        match="v1/v2 parent evidence is forbidden",
    ):
        audit._validate_temporal_v3_parent_spec(config)


def test_canonical_successor_is_exactly_bound_but_not_a_production_parent() -> None:
    parent = _load_config()["parents"]["canonical_work_time_spine"]
    assert parent["design_config_path"] == audit.CANONICAL_CONFIG_PATH
    assert parent["design_config_sha256"] == audit.CANONICAL_CONFIG_SHA256
    assert parent["synthetic_execution_script"] == audit.CANONICAL_TOOL_PATH
    assert parent["synthetic_execution_script_sha256"] == (audit.CANONICAL_TOOL_SHA256)
    assert parent["design_schema_version"] == audit.CANONICAL_DESIGN_SCHEMA
    assert parent["production_status"] == audit.PRODUCTION_STATUS
    assert parent["production_result_exists"] is False
    assert parent["production_receipt_exists"] is False
    assert parent["production_parent_eligible"] is False
    assert parent["reuse_mode_status"] == audit.CANONICAL_REUSE_BLOCKED
    assert "production_receipt_sha256" not in parent
    assert "required_future_receipt_schema_version" not in parent
    assert _sha256(REPO_ROOT / audit.CANONICAL_CONFIG_PATH) == (
        audit.CANONICAL_CONFIG_SHA256
    )
    assert _sha256(REPO_ROOT / audit.CANONICAL_TOOL_PATH) == (
        audit.CANONICAL_TOOL_SHA256
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "design_config_path",
            "experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml",
        ),
        (
            "design_config_sha256",
            "192d78e3df094121e63922a6eaf20f57a57af41bc174c0fdbbcc347e3eaed385",
        ),
        (
            "synthetic_execution_script",
            "scripts/audit_openalex_canonical_work_time_spine.py",
        ),
        (
            "synthetic_execution_script_sha256",
            "827bab5a6edcd9a574149203c68c77ae5178a8da210499b81f05a388ace5c0dc",
        ),
        ("design_schema_version", "openalex_g1_canonical_work_time_spine_design.v1"),
        (
            "synthetic_status",
            "PASS_SYNTHETIC_CANONICAL_WORK_TIME_SPINE_CONTRACT_ONLY",
        ),
        (
            "required_future_status",
            "PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_INHERITED_TEMPORAL_EXCLUSIONS",
        ),
    ],
)
def test_old_canonical_surfaces_are_rejected(field: str, value: Any) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["canonical_work_time_spine"][field] = value
    with pytest.raises(audit.CitationSyntheticAuditError, match="blocked-parent drift"):
        audit._validate_canonical_blocked_parent(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("production_receipt_exists", True),
        ("production_parent_eligible", True),
        ("reuse_mode_status", "PASS"),
    ],
)
def test_canonical_production_promotion_is_rejected(field: str, value: Any) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["canonical_work_time_spine"][field] = value
    with pytest.raises(audit.CitationSyntheticAuditError, match="blocked-parent drift"):
        audit._validate_canonical_blocked_parent(config)


def test_fake_canonical_receipt_hash_is_rejected() -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["canonical_work_time_spine"]["production_receipt_sha256"] = (
        "1" * 64
    )
    with pytest.raises(audit.CitationSyntheticAuditError, match="must be absent"):
        audit._validate_canonical_blocked_parent(config)


def test_only_direct_synthetic_mode_and_distinct_v2_surfaces_are_enabled() -> None:
    config = _load_config()
    synthetic = config["synthetic_execution_contract"]
    assert audit.ALLOWED_MODES == (audit.DIRECT_MODE,)
    assert synthetic["allowed_temporal_input_modes"] == [audit.DIRECT_MODE]
    assert synthetic["canonical_reuse_status"] == audit.CANONICAL_REUSE_BLOCKED
    assert synthetic["fixture_schema_version"] == audit.FIXTURE_SCHEMA
    assert synthetic["report_schema_version"] == audit.REPORT_SCHEMA
    assert synthetic["receipt_schema_version"] == audit.OUTPUTS_SCHEMA
    assert all(
        value.endswith(".v2")
        for value in (
            audit.FIXTURE_SCHEMA,
            audit.PLAN_SCHEMA,
            audit.CHECKPOINT_SCHEMA,
            audit.REPORT_SCHEMA,
            audit.STATE_SCHEMA,
            audit.OUTPUTS_SCHEMA,
        )
    )
    assert synthetic["report_status"] == audit.SYNTHETIC_STATUS
    assert synthetic["production_status"] == audit.PRODUCTION_STATUS
    assert synthetic["production_input_access_allowed"] is False
    assert synthetic["full_component_status_allowed"] is False
    blocked_scope = config["mode_scope_contracts"][audit.REUSE_MODE]
    assert blocked_scope["status"] == audit.CANONICAL_REUSE_BLOCKED
    assert blocked_scope["executable"] is False
    recommendation = config["work_time_spine_recommendation"]
    assert recommendation["modify_temporal_v3"] is False
    assert "modify_existing_temporal_v0" not in recommendation


def test_every_declared_canonical_reuse_activation_surface_fails_closed() -> None:
    mutations = (
        lambda value: value["temporal_input_modes"]["allowed_modes"].append(
            audit.REUSE_MODE
        ),
        lambda value: value["temporal_input_modes"][audit.REUSE_MODE].update(
            executable=True
        ),
        lambda value: value["mode_scope_contracts"][audit.REUSE_MODE].update(
            executable=True
        ),
        lambda value: value["mode_conditioned_parent_sets"][audit.REUSE_MODE].update(
            required_in_order=["canonical_work_time_spine"]
        ),
        lambda value: value["partition_plan"][
            "mode_specific_temporal_input_stage"
        ].update({audit.REUSE_MODE: "validate_spine"}),
        lambda value: value["pilot_and_full_rules"]["full"].update(
            canonical_reuse_execution_allowed=True
        ),
    )
    for mutate in mutations:
        config = copy.deepcopy(_load_config())
        mutate(config)
        with pytest.raises(audit.CitationSyntheticAuditError):
            audit._validate_config(config)


def test_cli_exposes_no_production_root_and_reuse_is_not_a_choice() -> None:
    parser = audit.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "synthetic_fixture" in destinations
    assert "source_root" not in destinations
    assert "production_source_root" not in destinations
    mode_action = next(action for action in parser._actions if action.dest == "mode")
    assert tuple(mode_action.choices or ()) == (audit.DIRECT_MODE,)


def test_document_config_and_runner_are_synchronized() -> None:
    config = _load_config()
    text = DOC_PATH.read_text(encoding="utf-8")
    required = (
        "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN",
        audit.TEMPORAL_V3_STATUS,
        audit.CANONICAL_REUSE_BLOCKED,
        audit.CANONICAL_CONFIG_SHA256,
        audit.CANONICAL_TOOL_SHA256,
        audit.SYNTHETIC_STATUS,
        "no production receipt",
        "BIBLIOGRAPHIC_EVENT_TIME_CONVENTION",
    )
    assert all(value in text for value in required)
    assert (
        config["lifecycle"]["execution_script"]
        == SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
    )
    assert config["lifecycle"]["production_result_exists"] is False
    assert "No production run or result exists" in config["claim_boundary"]


def test_no_successor_production_result_is_stored_in_repository() -> None:
    result_root = REPO_ROOT / "experiments/evidence/results"
    assert not list(result_root.glob("openalex_g1_citation_available_at_v1*"))
    assert not list(result_root.glob("citation_edge_chronology_v2*"))
