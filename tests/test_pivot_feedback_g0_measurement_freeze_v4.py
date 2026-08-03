from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml"
)
PREDECESSOR = (
    ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v3.yaml"
)
VERIFIER = ROOT / "scripts/verify_pivot_feedback_g0_measurement_freeze_v4.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load(VERIFIER, "g0_measurement_freeze_v4_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_binds_exact_authorization_predecessor_and_precommitment() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == "pivot_feedback_g0_measurement_freeze.v5"
    assert config["status"] == VERIFY_MODULE.G0_V4_STATUS
    for key in (
        "authorization_parent",
        "immutable_predecessor",
        "risk_set_weight_support_precommitment_binding",
    ):
        binding = config[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
        mutation_key = (
            "predecessor_mutation_allowed"
            if key == "immutable_predecessor"
            else "mutation_allowed"
        )
        assert binding[mutation_key] is False


def test_v4_changes_only_authorized_top_level_subtrees() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    predecessor = yaml.safe_load(PREDECESSOR.read_text(encoding="utf-8"))
    unchanged = sorted(
        set(predecessor) - VERIFY_MODULE.ALLOWED_CHANGED_TOP_LEVEL
    )

    assert set(config) - VERIFY_MODULE.ALLOWED_CHANGED_TOP_LEVEL == set(unchanged)
    for subtree in unchanged:
        assert config[subtree] == predecessor[subtree]


def test_v4_resolves_exactly_six_new_choices_and_leaves_seven_null() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    resolution = config["resolution"]

    assert resolution["resolved_measurement_count"] == 12
    assert list(resolution["resolved_measurements"])[6:] == (
        VERIFY_MODULE.CHOICE_IDS
    )
    assert resolution["unresolved_quantitative_choice_count"] == 7
    assert list(resolution["unresolved_quantitative_choices"]) == (
        VERIFY_MODULE.REMAINING_CHOICE_IDS
    )
    assert all(
        value is None
        for value in resolution["unresolved_quantitative_choices"].values()
    )
    assert resolution["unresolved_required_section_count"] == 15
    assert resolution["freeze_ready_required_section_count"] == 0


def test_verifier_is_metadata_only_and_current_v4_passes() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})

    result = VERIFY_MODULE.run(CONFIG)

    assert result["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert result["preservation_replay"][
        "only_authorized_six_choice_binding_changed"
    ] is True
    assert result["preservation_replay"][
        "unchanged_top_level_subtree_count"
    ] == 10
    assert result["resolution_replay"]["resolved_measurement_count"] == 12
    assert result["resolution_replay"]["remaining_choice_count"] == 7
    assert result["current_boundary_replay"] == {
        "holdout_receipt_bound": False,
        "g0_freeze_ready": False,
        "confirmatory_or_holdout_rows_read": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    assert result["authorization"]["g0_measurement_freeze"] == (
        "NOT_AUTHORIZED"
    )


def test_verifier_rejects_unauthorized_preserved_subtree_drift(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["fixed_design_choices"]["temporal_clocks"]["response_offsets"] = [0]
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0MeasurementFreezeV4Error,
        match="unauthorized subtree drift fixed_design_choices",
    ):
        VERIFY_MODULE.run(mutated)


def test_verifier_rejects_resolution_of_seventh_new_choice(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["resolution"]["unresolved_quantitative_choices"][
        "g1_replication_tolerance"
    ] = 0.1
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0MeasurementFreezeV4Error,
        match="remaining choice values",
    ):
        VERIFY_MODULE.run(mutated)


def test_v4_remains_unsealed_and_cannot_access_holdout() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["holdout_access_binding"]["receipt_sha256"] is None
    assert config["lifecycle"]["pass_g0"] is False
    assert config["lifecycle"]["g4_holdout_opened"] is False
    assert config["readiness"]["g0_freeze_ready"] is False
    assert config["authorization"]["confirmatory_or_holdout_access"] == (
        "NOT_AUTHORIZED"
    )
    assert config["authorization"]["g0_measurement_freeze_seal"] == (
        "NOT_AUTHORIZED"
    )
    assert config["authorization"]["model_fitting"] == "NOT_AUTHORIZED"
    assert config["authorization"]["g4_open"] == "NOT_AUTHORIZED"
