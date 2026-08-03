from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_persistence_calibration_readout_v1.yaml"
)
BUILDER = (
    ROOT
    / "scripts/"
    "build_pivot_feedback_d0_persistence_calibration_readout_v1.py"
)
VERIFIER = (
    ROOT
    / "scripts/"
    "verify_pivot_feedback_d0_persistence_calibration_readout_v1.py"
)
PRODUCTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-persistence-calibration-readout-v1-20260731"
)
PRODUCTION_VERIFICATION = Path(
    "/ssd/"
    "pivot-feedback-d0-persistence-calibration-readout-v1-20260731"
    ".verification-v1.json"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD_MODULE = _load(BUILDER, "persistence_calibration_builder_test_module")
VERIFY_MODULE = _load(VERIFIER, "persistence_calibration_verifier_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_binds_v47_and_only_existing_verified_artifact() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gate = config["repository_bindings"]["gate_state"]
    source = config["operator_input"]
    contract = config["readout_contract"]

    assert config["status_before_execution"] == "AUTHORIZED_BY_GATE_STATE_V47"
    assert gate["sha256"] == _sha256(ROOT / gate["relative_path"])
    assert source["rows"] == 4_301
    assert source["columns"] == 13
    assert source["artifact_sha256"] == (
        "0d347a296dbe08960e8a0fd2d5aaf9ff0afa7c9f55b08b761258e9874ae7f6c5"
    )
    assert contract["grouping_levels"] == ["OVERALL", "CERTIFIED_EPISODE"]
    assert contract["adjusted_comparison"] == "NONE"
    assert contract["inferential_interval"] == "NONE"
    assert contract["ranking_or_selection"] == "FORBIDDEN"
    assert all(
        value is False
        for value in config["forbidden_inputs_and_outputs"].values()
    )


def test_independent_verifier_does_not_import_builder_or_duckdb() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "duckdb" not in imported
    assert not any("build_pivot_feedback" in name for name in imported)


def test_vector_and_rowwise_descriptive_replays_match_exact_fixture() -> None:
    rows = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 2, 0),
        (0, 0, 3),
        (1, 1, 1),
        (5, 0, 0),
    ]
    t1 = np.asarray([row[0] for row in rows], dtype=np.uint32)
    t2 = np.asarray([row[1] for row in rows], dtype=np.uint32)
    t3 = np.asarray([row[2] for row in rows], dtype=np.uint32)

    vector = BUILD_MODULE.summarize_counts(t1, t2, t3)
    rowwise = VERIFY_MODULE.summarize_rows(rows)

    assert vector == rowwise
    assert vector == {
        "rows": 6,
        "matured_rows": 6,
        "matured_persistence_rows": 5,
        "matured_no_persistence_rows": 1,
        "right_censored_rows": 0,
        "descriptive_persistence_rate": 5 / 6,
        "qualifying_author_work_pairs": 14,
        "target_topic_work_count_distribution": {
            "0": 1,
            "1": 1,
            "2": 1,
            "3": 2,
            "4": 0,
            "5_PLUS": 1,
        },
        "first_observed_persistence_offset_distribution": {
            "NONE": 1,
            "T_PLUS_1": 3,
            "T_PLUS_2": 1,
            "T_PLUS_3": 1,
        },
        "yearly_presence_pattern_distribution": {
            "000": 1,
            "001": 1,
            "010": 1,
            "011": 0,
            "100": 2,
            "101": 0,
            "110": 0,
            "111": 1,
        },
    }


def test_current_production_readout_and_independent_replay_are_bound() -> None:
    receipt_path = (
        PRODUCTION_ROOT / "persistence_calibration_readout.outputs.json"
    )
    report_path = PRODUCTION_ROOT / "persistence_calibration_readout.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verification = json.loads(PRODUCTION_VERIFICATION.read_text(encoding="utf-8"))

    assert receipt["status"] == BUILD_MODULE.STATUS
    assert receipt["source_rows_read"] == 4_301
    assert receipt["aggregate_only_no_author_ids"] is True
    assert receipt["matured_persistence_rows"] == 1_487
    assert receipt["matured_no_persistence_rows"] == 2_814
    assert receipt["right_censored_rows"] == 0
    assert receipt["association_or_effect_computed"] is False
    assert receipt["api_requests_created"] == 0
    assert _sha256(receipt_path) == (
        "36c7b64a041070cef494cf731f18d53e42484e53fbdbd5aa2aa4ef7aeb53d722"
    )
    assert _sha256(report_path) == (
        "07b73cbcbb53622011eea0dd9c539d5d484f88607a685e8a2161e8ab30ddbbfa"
    )

    overall = report["calibration_readout"]["overall"]
    assert overall["target_topic_work_count_distribution"] == {
        "0": 2_814,
        "1": 722,
        "2": 325,
        "3": 174,
        "4": 90,
        "5_PLUS": 176,
    }
    assert overall["first_observed_persistence_offset_distribution"] == {
        "NONE": 2_814,
        "T_PLUS_1": 837,
        "T_PLUS_2": 401,
        "T_PLUS_3": 249,
    }
    assert overall["yearly_presence_pattern_distribution"]["111"] == 190

    assert verification["status"] == VERIFY_MODULE.STATUS
    assert verification["readout_replay"] == report["calibration_readout"]
    assert all(value == 0 for value in verification["mismatches"].values())
    assert verification["verifier"]["imports_producer_module"] is False
    assert (
        verification["verifier"]["uses_producer_aggregation_implementation"]
        is False
    )
    assert verification["scope"]["confirmatory_or_holdout_rows_read"] == 0
    assert verification["scope"]["joined_external_rows"] == 0
    assert (
        verification["scope"][
            "adjusted_comparison_or_inferential_effect_computed"
        ]
        is False
    )
