from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType
from typing import Any, Callable

import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine.py"
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml"
)
TEMPORAL_TEST_PATH = REPO_ROOT / "tests/test_openalex_main_temporal_validity.py"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module(SCRIPT_PATH, "test_canonical_work_time_spine_module")
temporal_tests = _load_module(TEMPORAL_TEST_PATH, "spine_temporal_fixture_module")
# Atomic rename and all final byte/hash checks remain exercised; per-artifact
# fsync is disabled only for the 8,202-file pytest fixture to keep CI bounded.
audit.DURABLE_FSYNC = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class SyntheticSpineContract:
    def __init__(
        self,
        *,
        temporal_contract: Any,
        temporal_output: Path,
        config_path: Path,
    ) -> None:
        self.temporal_contract = temporal_contract
        self.temporal_output = temporal_output
        self.config_path = config_path

    @property
    def source_root(self) -> Path:
        return self.temporal_contract.source

    @property
    def content_receipt(self) -> Path:
        return self.temporal_contract.receipt

    @property
    def temporal_receipt(self) -> Path:
        return self.temporal_output / "main_temporal_validity_audit.outputs.json"

    @property
    def temporal_config(self) -> Path:
        return self.temporal_contract.config

    @property
    def temporal_repo_root(self) -> Path:
        return self.temporal_contract.repo

    def run(self, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        options: dict[str, Any] = {
            "config_path": self.config_path,
            "synthetic_source_root": self.source_root,
            "content_receipt_path": self.content_receipt,
            "temporal_receipt_path": self.temporal_receipt,
            "temporal_config_path": self.temporal_config,
            "temporal_repo_root": self.temporal_repo_root,
            "output_dir": output_dir,
            "batch_size": 2,
            "partition_sort_chunk_rows": 3,
            "synthetic_fixture_only": True,
        }
        options.update(kwargs)
        return audit.run_audit(**options)


def _make_contract(
    base: Path,
    mutate_rows: Callable[[list[dict[str, Any]]], None] | None = None,
) -> SyntheticSpineContract:
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    base.chmod(0o700)
    fixture_root = base / "temporal-fixture"
    fixture_root.mkdir(mode=0o700)
    fixture_root.chmod(0o700)
    temporal_contract = temporal_tests._make_contract(
        fixture_root,
        mutate_rows,
    )
    temporal_output = base / "temporal-output"
    temporal_report = temporal_contract.run(temporal_output)
    assert temporal_report["status"] == temporal_tests.audit.FULL_STATUS

    temporal_config = yaml.safe_load(
        temporal_contract.config.read_text(encoding="utf-8")
    )
    temporal_receipt = _read(
        temporal_output / "main_temporal_validity_audit.outputs.json"
    )
    temporal_plan = _read(temporal_output / "audit_plan.json")
    content_receipt = _read(temporal_contract.receipt)
    content_plan = _read(temporal_contract.receipt.parent / "audit_plan.json")

    config = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    config["source"] = {
        "root_env": "SYNTHETIC_SPINE_ROOT",
        "root_id": "synthetic-openalex-work-time-spine",
        "snapshot_date": "2026-03-30",
        "lineage_id": "synthetic",
        "historical_content_identity": "UNVERIFIED",
    }
    config["table"].update(
        {
            "name": temporal_config["table"]["name"],
            "expected_files": 2,
            "expected_raw_rows": 6,
            "expected_source_bytes": temporal_plan["table"]["selected_bytes"],
        }
    )
    content = config["parents"]["current_content_baseline"]
    content.update(
        {
            "required_plan_sha256": content_receipt["plan_sha256"],
            "required_task_fingerprint_sha256": content_receipt[
                "task_fingerprint_sha256"
            ],
            "required_file_count": 2,
            "required_source_bytes": content_plan["declared_bytes"],
        }
    )
    temporal = config["parents"]["main_temporal_validity"]
    temporal.update(
        {
            "required_config_sha256": _sha256(temporal_contract.config),
            "required_full_task_count": 2,
            "required_raw_rows": 6,
            "required_source_bytes": temporal_plan["table"]["selected_bytes"],
            "exact_checkpoint_artifact_count": 2,
            "exact_total_receipt_artifact_count": 6,
        }
    )
    assert temporal_receipt["status"] == temporal["required_status"]

    config["identifier_accounting"].update(
        {
            "raw_rows": 6,
            "frozen_w_minus_1_quarantine_rows": 1,
            "retained_rows": 5,
            "retained_distinct_numeric_ids": 5,
            "duplicate_numeric_id_groups": 0,
            "duplicate_numeric_row_excess": 0,
            "minimum_numeric_id": 0,
            "maximum_numeric_id": 4,
            "w_zero_rows_retained": 1,
        }
    )
    config["publication_accounting"].update(
        {
            "fixed_disposition_anchors": {
                "BOTH_NULL_EXCLUDED": 1,
                "VALID_PRE_1900_EXCLUDED": 1,
                "USABLE_1900_BEFORE_SNAPSHOT_YEAR": 1,
                "VALID_POST_SNAPSHOT_YEAR_EXCLUDED": 0,
            },
            "snapshot_year_total_rows": 2,
            "required_publication_year_minimum": 1899,
            "required_publication_year_maximum": 2026,
        }
    )
    config["publication_accounting"]["snapshot_year_split"]["required_sum"] = 2
    totals = config["cross_stage_conservation"]["required_equal_row_totals"]
    totals.update(
        {
            "raw_rows": 6,
            "quarantined_w_minus_1_rows": 1,
            "scan_retained_rows": 5,
            "staging_manifest_rows": 5,
            "partition_input_rows": 5,
            "parquet_manifest_rows": 5,
            "decoded_final_rows": 5,
        }
    )
    config["resource_envelope"].update(
        {
            "declared_input_files": 2,
            "declared_input_rows": 6,
            "declared_source_bytes": temporal_plan["table"]["selected_bytes"],
        }
    )

    config_path = base / "synthetic-spine.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return SyntheticSpineContract(
        temporal_contract=temporal_contract,
        temporal_output=temporal_output,
        config_path=config_path,
    )


@pytest.fixture(scope="module")
def completed_contract(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[SyntheticSpineContract, Path, dict[str, Any]]:
    base = tmp_path_factory.mktemp("canonical-spine")
    contract = _make_contract(base)
    output = base / "spine-output"
    report = contract.run(output)
    return contract, output, report


def _hardlink_clone(source: Path, target: Path) -> None:
    shutil.copytree(source, target, copy_function=os.link)


def test_record_abi_and_all_six_classifier_boundaries() -> None:
    snapshot = audit.date(2026, 3, 30)
    cases = [
        (None, None, 0),
        ("1899-12-31", 1899.0, 1),
        ("1900-01-01", 1900.0, 2),
        ("2026-03-30", 2026.0, 3),
        ("2026-03-31", 2026.0, 4),
        ("2027-01-01", 2027.0, 5),
    ]
    records = []
    for work_id, (raw_date, raw_year, expected) in enumerate(cases):
        disposition, day, year = audit.classify_publication(
            raw_date,
            raw_year,
            snapshot=snapshot,
        )
        assert disposition == expected
        record = audit.Record(work_id, day, year, disposition)
        assert len(record.encode()) == 16
        assert audit.Record.decode(record.encode()) == record
        records.append(record)
    assert audit.record_digest(records) == audit.record_digest(tuple(records))
    with pytest.raises(audit.SpineAuditError, match="exactly-one-null"):
        audit.classify_publication(None, 2020.0, snapshot=snapshot)
    with pytest.raises(audit.SpineAuditError, match="mismatch"):
        audit.classify_publication("2020-01-01", 2021.0, snapshot=snapshot)
    corrupted = bytearray(records[0].encode())
    corrupted[15] = 1
    with pytest.raises(audit.SpineAuditError, match="padding"):
        audit.Record.decode(bytes(corrupted))


def test_synthetic_full_surface_is_exact_and_never_promotes_production(
    completed_contract: tuple[SyntheticSpineContract, Path, dict[str, Any]],
) -> None:
    _, output, report = completed_contract
    receipt = _read(output / "work_time_spine_audit.outputs.json")
    state = _read(output / "run_state.json")
    assert report["schema_version"] == audit.REPORT_SCHEMA
    assert receipt["schema_version"] == audit.OUTPUTS_SCHEMA
    assert report["status"] == receipt["status"] == audit.SYNTHETIC_STATUS
    assert report["scope"]["execution_scope"] == "SYNTHETIC_FIXTURE_ONLY"
    assert report["scope"]["production_status"] == "PRODUCTION_NOT_RUN"
    assert report["production_component_pass"] is False
    assert receipt["production_component_pass"] is False
    assert report["scope"]["production_row_invariant_met"] is False
    assert report["scope"]["partition_count"] == 4096
    assert receipt["synthetic_artifact_count"] == 6 + 2 * 2 + 2 * 4096
    assert receipt["production_artifact_invariant_n_493"] == 9184
    assert len(list((output / "spine").glob("part-*.parquet"))) == 4096
    assert len(list((output / "checkpoints/partition").glob("part-*.json"))) == 4096
    assert len(list((output / "staging/main").glob("*.bin"))) == 2
    assert len(list((output / "checkpoints/scan/main").glob("*.json"))) == 2
    assert state["status"] == "COMPLETE_PENDING_RECEIPT"
    assert state["complete_only_if_receipt_exists"] is True
    assert state["result_fingerprint_sha256"] == receipt["result_fingerprint_sha256"]
    assert _read(output / "work_time_spine_manifest.json")["rows"] == 5
    empty = pq.ParquetFile(output / "spine/part-0005.parquet")
    assert empty.metadata.num_rows == 0
    assert empty.schema_arrow == audit.ARROW_SCHEMA
    assert b"ARROW:schema" not in (empty.metadata.metadata or {})
    assert stat_mode(output) == 0o700
    assert stat_mode(output / "spine/part-0005.parquet") == 0o600


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o7777


def test_terminal_resume_validates_before_any_source_payload_open(
    completed_contract: tuple[SyntheticSpineContract, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, output, original = completed_contract
    receipt_mtime = (output / "work_time_spine_audit.outputs.json").stat().st_mtime_ns

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("terminal resume opened a source payload")

    monkeypatch.setattr(audit, "_source_payload_batches", forbidden)
    observed = contract.run(output, resume=True)
    assert observed == original
    assert (
        output / "work_time_spine_audit.outputs.json"
    ).stat().st_mtime_ns == receipt_mtime


def test_partial_scan_and_partition_resume_preserve_exact_plan(
    completed_contract: tuple[SyntheticSpineContract, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    contract, _, _ = completed_contract
    output = tmp_path / "partial-resume"
    scan_pause = contract.run(output, pause_after_new_scan_files=1)
    assert scan_pause["status"] == "PAUSED_SCAN"
    assert scan_pause["completed_scan_files"] == 1
    assert len(list((output / "staging/main").glob("*.bin"))) == 1
    partition_pause = contract.run(
        output,
        resume=True,
        pause_after_new_partitions=3,
    )
    assert partition_pause["status"] == "PAUSED_PARTITION"
    assert partition_pause["completed_scan_files"] == 2
    assert partition_pause["completed_partitions"] == 3
    plan_before = (output / "audit_plan.json").read_bytes()
    report = contract.run(output, resume=True)
    assert report["status"] == audit.SYNTHETIC_STATUS
    assert (output / "audit_plan.json").read_bytes() == plan_before
    assert len(list((output / "spine").glob("part-*.parquet"))) == 4096


def test_terminal_resume_rejects_same_surface_byte_corruption_without_source_open(
    completed_contract: tuple[SyntheticSpineContract, Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, output, _ = completed_contract
    clone = tmp_path / "corrupt-terminal"
    _hardlink_clone(output, clone)
    target = clone / "spine/part-0005.parquet"
    payload = target.read_bytes()
    target.unlink()
    target.write_bytes(payload + b"x")
    target.chmod(0o600)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("terminal corruption path opened a source payload")

    monkeypatch.setattr(audit, "_source_payload_batches", forbidden)
    with pytest.raises(audit.SpineAuditError, match="bytes/hash mismatch"):
        contract.run(clone, resume=True)
    assert not (clone / "work_time_spine_audit.outputs.json").exists()
    assert not (clone / "work_time_spine_audit.json").exists()
    assert not (clone / "work_time_spine_audit.md").exists()
    state = _read(clone / "run_state.json")
    assert state["status"] == audit.FAILURE_STATUS
    assert "result_fingerprint_sha256" not in state


def test_final_validation_failure_removes_receipt_first_and_failure_state_has_no_result(
    completed_contract: tuple[SyntheticSpineContract, Path, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, output, _ = completed_contract
    clone = tmp_path / "finalization-failure"
    _hardlink_clone(output, clone)
    for name in audit.FINAL_REPORT_NAMES:
        (clone / name).unlink()
    events: list[str] = []
    original_remove = audit._remove_if_exists
    original_write = audit.atomic_write_json

    def injected_validation(*args: Any, **kwargs: Any) -> None:
        raise audit.SpineAuditError("injected final validation failure")

    def observed_remove(path: Path) -> None:
        events.append(f"remove:{path.name}:exists={path.exists()}")
        original_remove(path)

    def observed_write(path: Path, value: Any) -> None:
        if (
            path.name == "run_state.json"
            and value.get("status") == audit.FAILURE_STATUS
        ):
            events.append("write:failure-state")
        original_write(path, value)

    monkeypatch.setattr(audit, "_validate_final_publication", injected_validation)
    monkeypatch.setattr(audit, "_remove_if_exists", observed_remove)
    monkeypatch.setattr(audit, "atomic_write_json", observed_write)
    with pytest.raises(
        audit.SpineAuditError, match="injected final validation failure"
    ):
        contract.run(clone, resume=True)
    receipt_event = "remove:work_time_spine_audit.outputs.json:exists=True"
    assert receipt_event in events
    assert events.index(receipt_event) < events.index("write:failure-state")
    assert events.index("write:failure-state") < events.index(
        "remove:work_time_spine_audit.json:exists=True"
    )
    assert not (clone / "work_time_spine_audit.outputs.json").exists()
    assert not (clone / "work_time_spine_audit.json").exists()
    assert not (clone / "work_time_spine_audit.md").exists()
    state = _read(clone / "run_state.json")
    assert state["status"] == audit.FAILURE_STATUS
    assert "result_fingerprint_sha256" not in state
    assert len(list((clone / "checkpoints/partition").glob("*.json"))) == 4096


def test_duplicate_retained_id_is_rejected_before_any_staging_publication(
    tmp_path: Path,
) -> None:
    def duplicate(rows: list[dict[str, Any]]) -> None:
        rows[4]["id"] = "https://openalex.org/W3"

    contract = _make_contract(tmp_path / "duplicate", duplicate)
    output = tmp_path / "duplicate-output"
    with pytest.raises(audit.SpineAuditError, match="distinct numeric ID mismatch"):
        contract.run(output)
    assert list((output / "staging/main").glob("*.bin")) == []
    assert list((output / "checkpoints/scan/main").glob("*.json")) == []
    assert list((output / "spine").glob("*.parquet")) == []


def test_repository_production_config_and_production_root_are_not_executable(
    tmp_path: Path,
) -> None:
    with pytest.raises(audit.SpineAuditError, match="repository production config"):
        audit.run_audit(
            config_path=CONFIG_PATH,
            synthetic_source_root=tmp_path,
            content_receipt_path=tmp_path / "missing-content.json",
            temporal_receipt_path=tmp_path / "missing-temporal.json",
            temporal_config_path=tmp_path / "missing-temporal.yaml",
            temporal_repo_root=tmp_path,
            output_dir=tmp_path / "output",
            synthetic_fixture_only=True,
        )


def test_cli_exposes_only_explicit_synthetic_source_root(tmp_path: Path) -> None:
    parser = audit._parser()
    destinations = {action.dest for action in parser._actions}
    assert "synthetic_source_root" in destinations
    assert "source_root" not in destinations
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--source-root",
                str(tmp_path / "source"),
                "--content-receipt",
                str(tmp_path / "content.json"),
                "--temporal-receipt",
                str(tmp_path / "temporal.json"),
                "--temporal-config",
                str(tmp_path / "temporal.yaml"),
                "--temporal-repo-root",
                str(tmp_path / "repo"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


def test_production_path_component_is_rejected_before_source_payload_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _make_contract(tmp_path / "path-contract")
    production_root = tmp_path / audit.PRODUCTION_ROOT_ID
    production_root.mkdir()
    opened = False

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("source payload opened")

    monkeypatch.setattr(audit, "_source_payload_batches", forbidden)
    with pytest.raises(audit.SpineAuditError, match="production OpenAlex source path"):
        contract.run(
            tmp_path / "path-output",
            synthetic_source_root=production_root,
        )
    assert opened is False


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "table",
            "expected_files",
            audit.MAX_SYNTHETIC_INPUT_TASKS + 1,
            "input tasks exceeds hard cap",
        ),
        (
            "table",
            "expected_raw_rows",
            audit.MAX_SYNTHETIC_FOOTER_ROWS + 1,
            "footer rows exceeds hard cap",
        ),
        (
            "table",
            "expected_source_bytes",
            audit.MAX_SYNTHETIC_SOURCE_BYTES + 1,
            "source bytes exceeds hard cap",
        ),
        (
            "identifier_accounting",
            "retained_rows",
            audit.MAX_SYNTHETIC_RETAINED_ROWS + 1,
            "retained rows exceeds hard cap",
        ),
    ],
)
def test_synthetic_declared_caps_fail_before_source_payload_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    value: int,
    message: str,
) -> None:
    contract = _make_contract(tmp_path / f"cap-{field}")
    config = yaml.safe_load(contract.config_path.read_text(encoding="utf-8"))
    config[section][field] = value
    oversized = tmp_path / f"oversized-{field}.yaml"
    oversized.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    contract.config_path = oversized
    opened = False

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("source payload opened")

    monkeypatch.setattr(audit, "_source_payload_batches", forbidden)
    with pytest.raises(audit.SpineAuditError, match=message):
        contract.run(tmp_path / f"cap-output-{field}")
    assert opened is False


def test_parent_metadata_caps_are_checked_before_spine_payload_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _make_contract(tmp_path / "cap-order")
    cap_validations = 0
    original_caps = audit._validate_synthetic_parent_caps
    original_open = audit._source_payload_batches

    def observed_caps(*args: Any, **kwargs: Any) -> dict[str, int]:
        nonlocal cap_validations
        cap_validations += 1
        return original_caps(*args, **kwargs)

    def observed_open(*args: Any, **kwargs: Any) -> Any:
        assert cap_validations > 0
        return original_open(*args, **kwargs)

    monkeypatch.setattr(audit, "_validate_synthetic_parent_caps", observed_caps)
    monkeypatch.setattr(audit, "_source_payload_batches", observed_open)
    report = contract.run(tmp_path / "cap-order-output")
    assert report["status"] == audit.SYNTHETIC_STATUS
    assert cap_validations >= 2


def test_parent_task_bytes_cannot_understate_selected_bytes_before_payload_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _make_contract(tmp_path / "cap-task-bytes")
    plan = _read(contract.temporal_output / "audit_plan.json")
    report = _read(contract.temporal_output / "main_temporal_validity_audit.json")
    config = yaml.safe_load(contract.config_path.read_text(encoding="utf-8"))
    plan["table"]["selected_bytes"] -= 1
    opened = False

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("source payload opened")

    monkeypatch.setattr(audit, "_source_payload_batches", forbidden)
    with pytest.raises(audit.SpineAuditError, match="selected/task source-byte"):
        audit._validate_synthetic_parent_caps(
            config,
            plan=plan,
            tasks=plan["tasks"],
            report=report,
            source_root=contract.source_root,
        )
    assert opened is False
