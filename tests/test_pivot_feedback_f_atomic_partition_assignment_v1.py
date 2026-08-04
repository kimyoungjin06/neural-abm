from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_atomic_partition_assignment_v1.yaml"
BUILDER = SCRIPTS / "build_pivot_feedback_f_atomic_partition_assignment_v1.py"
VERIFIER = SCRIPTS / "verify_pivot_feedback_f_atomic_partition_assignment_v1.py"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_f_atomic_partition_assignment_v1_20260804.summary.json"
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v8.md"
FINAL_PANEL = Path("/ssd/pivot-feedback-f-aligned-final-panel-conformance-v1-20260804")
PRODUCTION = Path("/ssd/pivot-feedback-f-atomic-partition-assignment-v1-20260804")
VERIFICATION = Path("/ssd/pivot-feedback-f-atomic-partition-assignment-v1-20260804.verification.json")


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY = _load(VERIFIER, "f_atomic_partition_verifier_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_output(tmp_path: Path) -> Path:
    output = tmp_path / PRODUCTION.name
    shutil.copytree(PRODUCTION, output)
    return output


def _rebind_receipt(output: Path) -> None:
    receipt_path = output / "atomic_partition_assignment.outputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for item in receipt["artifacts"]:
        artifact = output / item["relative_path"]
        item["bytes"] = artifact.stat().st_size
        item["sha256"] = _sha(artifact)
        if artifact.suffix == ".parquet":
            item["rows"] = pq.ParquetFile(artifact).metadata.num_rows
    ordered = sorted(receipt["artifacts"], key=lambda row: row["relative_path"])
    tree = [
        (row["relative_path"], row["bytes"], row["sha256"])
        for row in ordered
    ]
    receipt["artifact_tree_sha256"] = hashlib.sha256(
        VERIFY._canonical_bytes(tree)
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)


def test_contract_freezes_episode_family_atomicity_and_future_row_boundary() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    atomic = contract["atomic_partition_contract"]

    assert contract["schema_version"] == VERIFY.CONFIG_SCHEMA
    assert atomic["assignment_unit"] == "WHOLE_SHOCK_EPISODE_AND_ALL_RESPONSE_COHORTS"
    assert atomic["family_key_columns"] == ["field_id", "subfield_id"]
    assert atomic["response_cohort_offsets"] == [0, 1, 2]
    assert atomic["split_episode_or_response_cohort"] == "FORBIDDEN"
    assert atomic["shock_family_cannot_cross_g1_and_g4"] is True
    assert atomic["selected_field_replacement"] == "FORBIDDEN"
    assert atomic["selected_episode_reselection"] == "FORBIDDEN"
    assert atomic["outcome_based_assignment"] == "FORBIDDEN"
    assert atomic["expected_final_panel_partition_counts"] == {
        "discovery_calibration": 0,
        "g1_empirical_replication": 28,
        "model_selection": 0,
        "g4_temporal_holdout": 37,
    }
    assert atomic["model_selection_folds"]["fold_count"] == 5
    assert atomic["model_selection_folds"]["component_may_be_split"] is False
    assert contract["g4_boundary"]["private_row_membership_materialized"] is False
    assert contract["completion_accounting"]["pass_f"] is False
    assert contract["completion_accounting"]["pass_g0"] is False


def test_production_assigns_whole_final_panel_without_fabricating_model_episodes() -> None:
    report = json.loads(
        (PRODUCTION / "atomic_partition_assignment.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == (
        "PASS_OUTCOME_BLIND_ATOMIC_EPISODE_FAMILY_PARTITION_ASSIGNMENT_"
        "PRIVATE_HOLDOUT_MATERIALIZATION_AUTHORIZED_NO_F_PASS"
    )
    assert report["accounting"] == {
        "source_episode_rows": 65,
        "source_slot_rows": 4,
        "atomic_episode_rows": 65,
        "atomic_family_rows": 29,
        "partition_rows": 4,
        "response_boundary_crossing_episode_rows": 39,
        "g1_episode_rows": 28,
        "g1_family_rows": 13,
        "model_selection_episode_rows": 0,
        "g4_episode_rows": 37,
        "g4_family_rows": 16,
    }
    assert [row["assigned_episode_count"] for row in report["partition_summary"]] == [0, 28, 0, 37]
    assert [row["assigned_family_count"] for row in report["partition_summary"]] == [0, 13, 0, 16]
    assert report["authorization"]["operator_only_private_holdout_materialization"] == "AUTHORIZED_NEXT"
    assert report["scope"]["author_token_membership_rows_read"] == 0
    assert report["scope"]["holdout_paths_enumerated"] == 0
    assert report["scope"]["holdout_bytes_hashed"] == 0
    assert report["scope"]["pass_f"] is False
    assert report["scope"]["pass_g0"] is False


def test_exact_production_passes_independent_atomic_reconstruction(tmp_path: Path) -> None:
    result = VERIFY.verify(
        config_path=CONFIG,
        final_panel_root=FINAL_PANEL,
        output_root=PRODUCTION,
        verification_output=tmp_path / "verification.json",
    )

    assert result["status"] == VERIFY.PASS_STATUS
    assert result["episode_rows_verified"] == 65
    assert result["family_rows_verified"] == 29
    assert result["partition_rows_verified"] == 4
    assert result["partition_episode_counts_verified"] == {
        "discovery_calibration": 0,
        "g1_empirical_replication": 28,
        "model_selection": 0,
        "g4_temporal_holdout": 37,
    }
    assert result["partition_family_counts_verified"] == {
        "discovery_calibration": 0,
        "g1_empirical_replication": 13,
        "model_selection": 0,
        "g4_temporal_holdout": 16,
    }
    assert result["response_boundary_crossing_episode_rows_verified"] == 39
    assert result["whole_episode_and_family_atomicity_verified"] is True
    assert result["g1_g4_family_disjointness_verified"] is True
    assert result["raw_id_columns_absent"] is True
    assert result["pass_f"] is False
    assert result["pass_g0"] is False


def test_verifier_is_independent_of_builder_and_private_rows() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert "build_pivot_feedback_f_atomic_partition_assignment_v1" not in source
    assert "episode assignment differs from independent reconstruction" in source
    assert "family assignment differs from independent reconstruction" in source
    assert "author_token_membership_rows_read" in source
    assert "raw ID in output" in source


def test_verifier_rejects_rebound_episode_partition_tamper(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    path = output / "atomic_episode_partition_assignment.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["registered_partition_id"] = "model_selection"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    path.chmod(0o600)
    _rebind_receipt(output)

    with pytest.raises(
        VERIFY.AtomicPartitionVerificationError,
        match="episode assignment differs from independent reconstruction",
    ):
        VERIFY.verify(
            config_path=CONFIG,
            final_panel_root=FINAL_PANEL,
            output_root=output,
            verification_output=tmp_path / "should-not-exist.json",
        )


def test_verifier_rejects_receipt_that_improperly_passes_f(tmp_path: Path) -> None:
    output = _copy_output(tmp_path)
    receipt_path = output / "atomic_partition_assignment.outputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["pass_f"] = True
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)

    with pytest.raises(
        VERIFY.AtomicPartitionVerificationError,
        match="unsafe receipt state: pass_f",
    ):
        VERIFY.verify(
            config_path=CONFIG,
            final_panel_root=FINAL_PANEL,
            output_root=output,
            verification_output=tmp_path / "should-not-exist.json",
        )


def test_repository_summary_and_ledger_bind_fail_closed_result() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))
    ledger = LEDGER.read_text(encoding="utf-8")

    production = summary["production_artifact"]
    bindings = summary["repository_bindings"]
    assert production["output_receipt_sha256"] == _sha(
        PRODUCTION / "atomic_partition_assignment.outputs.json"
    )
    assert bindings["config_sha256"] == _sha(CONFIG)
    assert bindings["builder_sha256"] == _sha(BUILDER)
    assert bindings["verifier_sha256"] == _sha(VERIFIER)
    assert summary["independent_verification"]["verification_sha256"] == _sha(
        VERIFICATION
    )
    assert verification["status"] == VERIFY.PASS_STATUS
    assert summary["fail_closed_state"]["atomic_episode_family_partition_assignment_complete"] is True
    assert summary["fail_closed_state"]["holdout_paths_enumerated"] == 0
    assert summary["fail_closed_state"]["holdout_bytes_hashed"] == 0
    assert summary["fail_closed_state"]["pass_f"] is False
    assert summary["fail_closed_state"]["pass_g0"] is False
    assert "Thirty-nine episodes" in ledger
    assert "operator-only materialization" in ledger
    assert "`pass_f=false` and `pass_g0=false`" in ledger
