from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_f_field_episode_assignment_v1.py"
CONFIG = ROOT / "experiments/configs/pivot_feedback_f_field_episode_assignment_v1.yaml"
PRODUCTION = Path("/ssd/pivot-feedback-f-field-episode-assignment-v1-20260803")
VERIFICATION = Path(
    "/ssd/pivot-feedback-f-field-episode-assignment-v1-20260803.verification.json"
)
SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_f_field_episode_assignment_v1_20260803.summary.json"
)


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("f_assignment_verifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_verifier()


def _copy_production(tmp_path: Path) -> Path:
    output = tmp_path / PRODUCTION.name
    shutil.copytree(PRODUCTION, output)
    return output


def _rebind_receipt(output: Path) -> None:
    path = output / "field_episode_assignment.outputs.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    for item in receipt["artifacts"]:
        artifact = output / item["relative_path"]
        item["bytes"] = artifact.stat().st_size
        item["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if artifact.suffix == ".parquet":
            item["rows"] = pq.ParquetFile(artifact).metadata.num_rows
    tree = [
        (item["relative_path"], item["bytes"], item["sha256"])
        for item in receipt["artifacts"]
    ]
    receipt["artifact_tree_sha256"] = hashlib.sha256(
        MODULE._canonical_bytes(tree)
    ).hexdigest()
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_table(*, modified_z: float = 3.5, period: str = "g1_empirical_replication") -> pa.Table:
    reference = 30
    recent = 300
    acceleration = math.log1p(recent / 3.0) - math.log1p(reference / 3.0)
    median = 0.1
    mad_value = 0.6744897501960817 * (acceleration - median) / modified_z
    schema = pq.ParquetFile(
        PRODUCTION / "attention_shock_candidates.parquet"
    ).schema_arrow
    return pa.Table.from_pylist(
        [
            {
                "field_id": 11,
                "topic_id": 101,
                "topic_name": "Synthetic topic",
                "subfield_id": 1101,
                "subfield_name": "Synthetic subfield",
                "decision_year": 2013,
                "temporal_partition_id": period,
                "reference_count": reference,
                "recent_count": recent,
                "acceleration": acceleration,
                "field_year_median": median,
                "field_year_mad": mad_value,
                "modified_z": modified_z,
                "selected_field": False,
                "assigned_slot_id": None,
                "retained_for_assigned_panel": False,
            }
        ],
        schema=schema,
    )


def test_verifier_is_independent_of_builder() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_pivot_feedback_f_field_episode_assignment_v1" not in source
    assert "LEXICOGRAPHIC" not in source
    assert "candidate below modified-z threshold" in source
    assert "receipt improperly passes F" in source


def test_verifier_config_check_keeps_component_fail_closed() -> None:
    config, config_sha = MODULE._validate_config(CONFIG)

    assert len(config_sha) == 64
    assert config["completion_accounting"]["this_component_is_new_gate"] is False
    assert config["completion_accounting"]["this_component_alone_may_pass_f"] is False
    assert config["completion_accounting"]["final_panel_conformance_after_assignment"] == "PENDING"
    assert config["completion_accounting"]["atomic_partition_and_holdout_byte_seal"] == "PENDING"


def test_verifier_replays_candidate_formula() -> None:
    rows, families = MODULE._validate_candidates(_candidate_table())

    assert len(rows) == 1
    assert families[(11, "g1_empirical_replication")] == 1


def test_verifier_rejects_candidate_below_registered_threshold() -> None:
    with pytest.raises(
        MODULE.AssignmentVerificationError,
        match="candidate below modified-z threshold",
    ):
        MODULE._validate_candidates(_candidate_table(modified_z=2.999))


def test_verifier_rejects_candidate_in_wrong_temporal_partition() -> None:
    with pytest.raises(
        MODULE.AssignmentVerificationError,
        match="candidate period mismatch",
    ):
        MODULE._validate_candidates(_candidate_table(period="g4_temporal_holdout"))


def test_config_has_no_outcome_based_substitution_escape() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["episode_assignment"]["outcome_based_candidate_or_field_substitution"] is False
    assert config["episode_assignment"]["reserve_activation_after_outcome_read"] is False


def test_exact_production_output_passes_independent_verification() -> None:
    result = MODULE.verify(PRODUCTION)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["artifacts_rehashed"] == 6
    assert result["field_rows_verified"] == 26
    assert result["candidate_rows_verified"] == 1744
    assert [row["selected_field_id"] for row in result["slot_assignment"]] == [
        17,
        32,
        11,
        36,
    ]
    assert sum(row["reserve_activated"] for row in result["slot_assignment"]) == 2
    assert result["pass_f"] is False
    assert result["g4_open"] is False


def test_repository_summary_binds_exact_receipts_and_fail_closed_state() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["production_artifact"]["output_receipt_sha256"] == hashlib.sha256(
        (PRODUCTION / "field_episode_assignment.outputs.json").read_bytes()
    ).hexdigest()
    assert summary["independent_verification"]["verification_sha256"] == hashlib.sha256(
        VERIFICATION.read_bytes()
    ).hexdigest()
    assert summary["assignment"]["reserve_activation_count"] == 2
    assert summary["registered_reserve_reconciliation"]["scientific_threshold_changed"] is False
    assert summary["registered_reserve_reconciliation"]["outcome_read_before_reconciliation"] is False
    assert summary["fail_closed_state"]["final_panel_conformance_complete"] is False
    assert summary["fail_closed_state"]["holdout_byte_seal_complete"] is False
    assert summary["fail_closed_state"]["pass_f"] is False
    assert summary["fail_closed_state"]["pass_g0"] is False
    assert summary["fail_closed_state"]["g4_open"] is False


def test_verifier_rejects_mutated_reserve_activation_marker(tmp_path: Path) -> None:
    output = _copy_production(tmp_path)
    path = output / "slot_assignment.parquet"
    table = pq.read_table(path)
    values = table["reserve_activated"].to_pylist()
    values[0] = False
    table = table.set_column(
        table.column_names.index("reserve_activated"),
        table.schema.field("reserve_activated"),
        pa.array(values, type=pa.bool_()),
    )
    pq.write_table(table, path)
    _rebind_receipt(output)

    with pytest.raises(
        MODULE.AssignmentVerificationError,
        match="reserve activation marker mismatch",
    ):
        MODULE.verify(output)


def test_verifier_rejects_mutated_g4_author_cold_marker(tmp_path: Path) -> None:
    output = _copy_production(tmp_path)
    path = output / "slot_assignment.parquet"
    table = pq.read_table(path)
    values = table["author_cold"].to_pylist()
    values[-1] = False
    table = table.set_column(
        table.column_names.index("author_cold"),
        table.schema.field("author_cold"),
        pa.array(values, type=pa.bool_()),
    )
    pq.write_table(table, path)
    _rebind_receipt(output)

    with pytest.raises(
        MODULE.AssignmentVerificationError,
        match="G4 slot is not author-cold",
    ):
        MODULE.verify(output)


def test_verifier_rejects_rebound_candidate_below_threshold(tmp_path: Path) -> None:
    output = _copy_production(tmp_path)
    path = output / "attention_shock_candidates.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    row = rows[0]
    row["modified_z"] = 2.999
    row["field_year_mad"] = (
        0.6744897501960817
        * (float(row["acceleration"]) - float(row["field_year_median"]))
        / float(row["modified_z"])
    )
    table = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(table, path)
    _rebind_receipt(output)

    with pytest.raises(
        MODULE.AssignmentVerificationError,
        match="candidate below modified-z threshold",
    ):
        MODULE.verify(output)


def test_verifier_rejects_receipt_that_improperly_passes_f(tmp_path: Path) -> None:
    output = _copy_production(tmp_path)
    path = output / "field_episode_assignment.outputs.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["pass_f"] = True
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        MODULE.AssignmentVerificationError,
        match="receipt improperly passes F",
    ):
        MODULE.verify(output)
