from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "experiments/configs/pivot_feedback_e_g1_disjoint_successor_availability_v1.yaml"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_e_g1_disjoint_successor_availability_v1_20260806.summary.json"
VERIFIER_PATH = ROOT / "scripts/verify_pivot_feedback_e_g1_disjoint_successor_availability_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_PATH, "pivot_feedback_e_g1_disjoint_successor_availability_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authority_closes_every_current_partition_without_new_gate() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    replay = authority["partition_replay"]
    disposition = authority["availability_disposition"]

    assert {name: row["final_panel_episode_count"] for name, row in replay.items()} == {
        "discovery_calibration": 0,
        "g1_empirical_replication": 28,
        "model_selection": 0,
        "g4_temporal_holdout": 37,
    }
    assert all(row["successor_available"] is False for row in replay.values())
    assert disposition["current_registered_disjoint_episode_count"] == 0
    assert disposition["successor_execution_authorized"] is False
    assert disposition["this_is_new_scientific_gate"] is False
    assert disposition["this_changes_current_state_v4"] is False
    assert authority["disjoint_successor_requirements"]["current_g1_support_counts_may_select_new_scope"] is False
    assert authority["authorization_after_independent_pass"]["holdout_or_g4_access"] == "NOT_AUTHORIZED"


def test_independent_summary_replays_metadata_only_and_preserves_state() -> None:
    replay = VERIFIER.verify(AUTHORITY)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert replay == summary
    assert summary["authority"]["sha256"] == _sha(AUTHORITY)
    assert summary["partition_metadata_replay"]["atomic_episode_rows_read"] == 65
    assert summary["partition_metadata_replay"]["current_registered_disjoint_episode_count"] == 0
    assert summary["next_executable_prerequisite"]["status"] == (
        "NEW_IMMUTABLE_MATURED_SOURCE_OR_EXTERNAL_DATASET_REQUIRED"
    )
    assert summary["access_accounting"] == {
        "repository_documents_read": 7,
        "open_partition_metadata_rows_read": 65,
        "current_g1_label_rows_read": 0,
        "model_selection_rows_read": 0,
        "holdout_or_private_g4_rows_read": 0,
        "seal_open_events": 0,
        "g4_open": False,
        "models_fitted": 0,
    }


def test_verifier_has_only_atomic_metadata_ssd_path_and_no_model_logic() -> None:
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    ssd_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("/ssd/")
    }

    assert ssd_literals == {
        "/ssd/pivot-feedback-f-atomic-partition-assignment-v1-20260804/atomic_episode_partition_assignment.parquet"
    }
    assert "g4_author_cold_guard.parquet" not in source
    assert "holdout_outcomes" not in source
    assert "sklearn" not in source
    assert "statsmodels" not in source
