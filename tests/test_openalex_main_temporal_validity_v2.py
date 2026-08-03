from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable

import pyarrow as pa
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_main_temporal_validity_v2.py"
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_main_temporal_validity_v1.yaml"
)
V1_SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_main_temporal_validity.py"
V1_CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml"
)
HELPER_PATH = REPO_ROOT / "tests/test_openalex_main_temporal_validity.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module("test_main_temporal_audit_v2_module", SCRIPT_PATH)
helper = _load_module("test_main_temporal_audit_v2_helper", HELPER_PATH)
_v1_base_rows = helper._base_rows


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_rows_v2() -> list[dict[str, Any]]:
    rows = _v1_base_rows()
    for row in rows:
        for field in ("created_date", "updated_date"):
            if row[field] is not None:
                row[field] = f"{row[field]}T00:00:00.000Z"
    return rows


helper.audit = audit
helper.CONFIG_PATH = CONFIG_PATH
helper._base_rows = _base_rows_v2


def _make_contract(
    base: Path,
    mutate_rows: Callable[[list[dict[str, Any]]], None] | None = None,
    *,
    unresolved_policy: bool = False,
) -> Any:
    return helper._make_contract(
        base,
        mutate_rows,
        unresolved_policy=unresolved_policy,
    )


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_published_json(path: Path, value: Any) -> None:
    path.write_bytes(audit.canonical_json_bytes(value) + b"\n")
    path.chmod(0o600)


def test_v1_tool_and_config_are_preserved_byte_exact() -> None:
    assert _sha(V1_SCRIPT_PATH) == (
        "b4b39c016ba4c6867a3e08366f70f6125230c2a0a18c1e8c45d40826f373f563"
    )
    assert _sha(V1_CONFIG_PATH) == (
        "d208b45cc87de577726a8310a899eb66882a519bae29df5f7dbbf7833562b172"
    )


def test_repository_v2_config_is_the_frozen_production_config() -> None:
    assert audit.PRODUCTION_CONFIG_SHA256 == _sha(CONFIG_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    audit._validate_config_contract(config)


def test_real_observed_timestamps_parse_to_exact_epoch_milliseconds() -> None:
    values = pa.array(
        [
            "2016-06-24T00:00:00.000Z",
            "2025-07-23T07:32:40.638Z",
        ],
        type=pa.string(),
    )
    null, valid, epoch_ms, years = audit._strict_utc_millisecond_timestamp(values)
    assert null.tolist() == [False, False]
    assert valid.tolist() == [True, True]
    assert epoch_ms.tolist() == [1466726400000, 1753255960638]
    assert years.tolist() == [2016, 2025]


@pytest.mark.parametrize("fraction", ["000", "001", "638", "999"])
def test_exact_three_digit_millisecond_fractions_are_accepted(fraction: str) -> None:
    value = pa.array([f"2025-07-23T07:32:40.{fraction}Z"], type=pa.string())
    _, valid, epoch_ms, _ = audit._strict_utc_millisecond_timestamp(value)
    assert valid.tolist() == [True]
    assert epoch_ms[0] % 1000 == int(fraction)


@pytest.mark.parametrize(
    "value",
    [
        "2025-07-23T07:32:40Z",
        "2025-07-23T07:32:40.6Z",
        "2025-07-23T07:32:40.63Z",
        "2025-07-23T07:32:40.6380Z",
        "2025-07-23T07:32:40.638000Z",
        "2025-07-23T07:32:40,638Z",
        "2025-07-23T07:32:40.638z",
        "2025-07-23T07:32:40.638+00:00",
        "2025-07-23T07:32:40.638Z ",
        " 2025-07-23T07:32:40.638Z",
    ],
)
def test_noncanonical_fraction_timezone_and_whitespace_are_rejected(value: str) -> None:
    _, valid, _, _ = audit._strict_utc_millisecond_timestamp(
        pa.array([value], type=pa.string())
    )
    assert valid.tolist() == [False]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-02-29T23:59:59.999Z", True),
        ("2000-02-29T00:00:00.000Z", True),
        ("2023-02-29T00:00:00.000Z", False),
        ("1900-02-29T00:00:00.000Z", False),
        ("2025-13-01T00:00:00.000Z", False),
        ("2025-04-31T00:00:00.000Z", False),
        ("2025-01-01T24:00:00.000Z", False),
        ("2025-01-01T00:00:60.000Z", False),
        ("0000-01-01T00:00:00.000Z", False),
    ],
)
def test_calendar_and_clock_validity_is_exact(value: str, expected: bool) -> None:
    _, valid, _, _ = audit._strict_utc_millisecond_timestamp(
        pa.array([value], type=pa.string())
    )
    assert valid.tolist() == [expected]


def test_publication_and_metadata_lexical_contracts_are_separate() -> None:
    _, publication_valid, _, _ = audit._strict_publication_date(
        pa.array(["2025-07-23", "2025-07-23T00:00:00.000Z"], type=pa.string())
    )
    _, metadata_valid, _, _ = audit._strict_utc_millisecond_timestamp(
        pa.array(["2025-07-23", "2025-07-23T00:00:00.000Z"], type=pa.string())
    )
    assert publication_valid.tolist() == [True, False]
    assert metadata_valid.tolist() == [False, True]


def test_synthetic_full_pass_uses_v2_schemas_and_exact_timestamp_policy(
    tmp_path: Path,
) -> None:
    contract = _make_contract(tmp_path)
    output = tmp_path / "output"
    report = contract.run(output)
    receipt = _read(output / "main_temporal_validity_audit.outputs.json")
    checkpoint = _read(next((output / "checkpoints/main").glob("*.json")))
    assert report["status"] == audit.FULL_STATUS
    assert report["schema_version"] == audit.REPORT_SCHEMA
    assert report["gate_status"] == audit._expected_gates("full")
    assert report["policy"]["metadata_comparison_precision"] == "epoch_millisecond"
    assert receipt["schema_version"] == audit.OUTPUTS_SCHEMA
    assert checkpoint["schema_version"] == audit.CHECKPOINT_SCHEMA


def test_one_millisecond_same_day_reversal_blocks_only_metadata_subcomponent(
    tmp_path: Path,
) -> None:
    def reverse_one_ms(rows: list[dict[str, Any]]) -> None:
        rows[3].update(
            created_date="2020-01-01T00:00:00.001Z",
            updated_date="2020-01-01T00:00:00.000Z",
        )

    contract = _make_contract(tmp_path, reverse_one_ms)
    report = contract.run(tmp_path / "output")
    assert report["status"] == audit.BLOCKED_STATUS
    assert report["violations"]["created_gt_updated_unexpected_rows"] == 1
    assert report["gate_status"]["publication_date_year_validity"] == (
        "PASS_WITH_FROZEN_EXCLUSIONS"
    )
    assert report["gate_status"]["created_updated_chronology"] == "BLOCKED"


@pytest.mark.parametrize(
    ("updated", "expected_status", "future_rows"),
    [
        ("2026-03-30T23:59:59.999Z", audit.FULL_STATUS, 0),
        ("2026-03-31T00:00:00.000Z", audit.BLOCKED_STATUS, 1),
    ],
)
def test_snapshot_boundary_is_exact_to_the_millisecond(
    tmp_path: Path,
    updated: str,
    expected_status: str,
    future_rows: int,
) -> None:
    def set_boundary(rows: list[dict[str, Any]]) -> None:
        rows[3]["updated_date"] = updated

    contract = _make_contract(tmp_path, set_boundary)
    report = contract.run(tmp_path / "output")
    assert report["status"] == expected_status
    assert report["violations"]["future_metadata_unexpected_rows"] == future_rows


def test_clean_pilot_is_explicit_and_still_nonpromotable(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    report = contract.run(
        tmp_path / "pilot",
        max_files=1,
        content_freeze_receipt=None,
    )
    assert report["status"] == audit.PILOT_STATUS
    assert report["pilot_local_contract_clean"] is True
    assert report["pass_main_temporal_validity_component"] is False
    assert report["gate_status"] == audit._expected_gates("pilot")
    args = audit.build_parser().parse_args(
        [
            "--execute",
            "--max-files",
            "1",
            "--require-clean-pilot",
            "--output-dir",
            "/redacted",
        ]
    )
    assert args.require_clean_pilot is True


def test_dirty_pilot_is_reported_and_fails_clean_pilot_contract(tmp_path: Path) -> None:
    def malformed_metadata(rows: list[dict[str, Any]]) -> None:
        rows[0]["created_date"] = "2016-01-01"

    contract = _make_contract(tmp_path, malformed_metadata)
    report = contract.run(
        tmp_path / "pilot",
        max_files=1,
        content_freeze_receipt=None,
    )
    assert report["status"] == audit.PILOT_STATUS
    assert report["pilot_local_contract_clean"] is False
    assert report["violations"]["created_date_parse_error_rows"] == 1


@pytest.mark.parametrize(("clean", "expected_exit"), [(True, 0), (False, 1)])
def test_require_clean_pilot_exit_code_tracks_report_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean: bool,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(
        audit,
        "run_audit",
        lambda **_kwargs: {
            "status": audit.PILOT_STATUS,
            "pilot_local_contract_clean": clean,
        },
    )
    assert (
        audit.main(
            [
                "--execute",
                "--max-files",
                "1",
                "--require-clean-pilot",
                "--output-dir",
                str(tmp_path / "unused"),
            ]
        )
        == expected_exit
    )


def test_v1_config_and_checkpoint_schemas_are_rejected_fail_closed(
    tmp_path: Path,
) -> None:
    v1_config = yaml.safe_load(V1_CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises(audit.TemporalAuditError, match="config schema drift"):
        audit._validate_config_contract(v1_config)

    contract = _make_contract(tmp_path)
    output = tmp_path / "paused"
    contract.run(output, pause_after_new_files=1)
    plan = _read(output / "audit_plan.json")
    checkpoint_path = next((output / "checkpoints/main").glob("*.json"))
    checkpoint = _read(checkpoint_path)
    checkpoint["schema_version"] = "openalex_g1_main_temporal_validity_checkpoint.v1"
    core = dict(checkpoint)
    core.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = audit.sha256_bytes(
        audit.canonical_json_bytes(core)
    )
    task = audit._task_from_mapping(plan["tasks"][0])
    config = yaml.safe_load(contract.config.read_text(encoding="utf-8"))
    with pytest.raises(audit.TemporalAuditError, match="checkpoint schema mismatch"):
        audit.validate_checkpoint(
            checkpoint,
            task,
            source_root=contract.source,
            config=config,
            plan=plan,
        )


def test_v1_temporal_receipt_schema_is_rejected_by_final_validator(
    tmp_path: Path,
) -> None:
    contract = _make_contract(tmp_path)
    output = tmp_path / "output"
    report = contract.run(output)
    plan = _read(output / "audit_plan.json")
    receipt_path = output / "main_temporal_validity_audit.outputs.json"
    receipt = _read(receipt_path)
    receipt["schema_version"] = "openalex_g1_main_temporal_validity_outputs.v1"
    _write_published_json(receipt_path, receipt)
    config = yaml.safe_load(contract.config.read_text(encoding="utf-8"))
    content_context = audit._validate_content_freeze_parent(
        config,
        contract.receipt,
        required=True,
    )
    with pytest.raises(audit.TemporalAuditError, match="final receipt schema mismatch"):
        audit.validate_final_publication(
            output,
            report=report,
            outputs=receipt,
            config=config,
            plan=plan,
            source_root=contract.source,
            config_path=contract.config,
            content_context=content_context,
        )


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (
            ("temporal_policy", "created_updated", "strict_lexical_format"),
            "YYYY-MM-DD",
            "metadata timestamp format drift",
        ),
        (
            ("temporal_policy", "created_updated", "timezone_representation"),
            "offset_allowed",
            "metadata timezone drift",
        ),
        (
            ("temporal_policy", "created_updated", "fractional_second_digits"),
            6,
            "metadata fraction drift",
        ),
        (
            (
                "temporal_policy",
                "created_updated",
                "next_utc_day_start_exclusive_epoch_ms",
            ),
            1774915200001,
            "metadata epoch boundary drift",
        ),
        (
            ("success_contract", "full_success_status"),
            "PASS_DRIFT",
            "full status drift",
        ),
        (
            ("schema_version",),
            "openalex_g1_main_temporal_validity_config.v1",
            "config schema drift",
        ),
    ],
)
def test_v2_contract_mutations_fail_closed(
    path: tuple[str, ...],
    value: Any,
    error: str,
) -> None:
    config = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(audit.TemporalAuditError, match=error):
        audit._validate_config_contract(config)
