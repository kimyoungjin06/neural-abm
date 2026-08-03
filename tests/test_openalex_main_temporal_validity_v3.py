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
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_main_temporal_validity_v3.py"
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_main_temporal_validity_v2.yaml"
)
SUMMARY_PATH = (
    REPO_ROOT / "experiments/evidence/results/"
    "openalex_g1_main_temporal_validity_v3_full_20260719.summary.json"
)
SUMMARY_MARKDOWN_PATH = SUMMARY_PATH.with_suffix(".md")
V2_SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_main_temporal_validity_v2.py"
V2_CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_main_temporal_validity_v1.yaml"
)
V2_TEST_PATH = REPO_ROOT / "tests/test_openalex_main_temporal_validity_v2.py"
HELPER_PATH = REPO_ROOT / "tests/test_openalex_main_temporal_validity.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module("test_main_temporal_audit_v3_module", SCRIPT_PATH)
helper = _load_module("test_main_temporal_audit_v3_helper", HELPER_PATH)
_v1_base_rows = helper._base_rows


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_rows_v3() -> list[dict[str, Any]]:
    rows = _v1_base_rows()
    for row in rows:
        for field in ("created_date", "updated_date"):
            if row[field] is not None:
                row[field] = f"{row[field]}T00:00:00.000Z"
    return rows


helper.audit = audit
helper.CONFIG_PATH = CONFIG_PATH
helper._base_rows = _base_rows_v3


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


def test_v2_tool_config_and_test_are_preserved_byte_exact() -> None:
    assert _sha(V2_SCRIPT_PATH) == (
        "b86f2496bbf7258c4e981f9a7ad8b1618fdcdc34a7385de628b133e0082a89dc"
    )
    assert _sha(V2_CONFIG_PATH) == (
        "11a668b8f2b10c83e2d36b3f5d2cd16f1e5f86b41ef64f58c79f10cabebc30e8"
    )
    assert _sha(V2_TEST_PATH) == (
        "70ac23af07a8e0832e563ae7df5b8716fc09138c4873cfde9c750855d1364501"
    )


def test_repository_v3_config_is_the_frozen_production_config() -> None:
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
        "2025-07-23T07:32:40.638+09:00",
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


def test_timestamp_equality_and_next_second_boundary_preserve_milliseconds() -> None:
    values = pa.array(
        [
            "2025-01-01T00:00:00.999Z",
            "2025-01-01T00:00:01.000Z",
            "2025-01-01T00:00:01.000Z",
        ],
        type=pa.string(),
    )
    _, valid, epoch_ms, _ = audit._strict_utc_millisecond_timestamp(values)
    assert valid.tolist() == [True, True, True]
    assert int(epoch_ms[1] - epoch_ms[0]) == 1
    assert int(epoch_ms[2] - epoch_ms[1]) == 0


def test_synthetic_full_pass_uses_v3_schemas_and_nonblocking_order_policy(
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
    assert report["policy"]["created_updated_order_is_validity_gate"] is False
    assert report["created_updated_relation_diagnostic"]["partition_exact"] is True
    assert receipt["schema_version"] == audit.OUTPUTS_SCHEMA
    assert checkpoint["schema_version"] == audit.CHECKPOINT_SCHEMA


def test_one_millisecond_same_day_reversal_is_descriptive_and_does_not_block(
    tmp_path: Path,
) -> None:
    def reverse_one_ms(rows: list[dict[str, Any]]) -> None:
        rows[3].update(
            created_date="2020-01-01T00:00:00.001Z",
            updated_date="2020-01-01T00:00:00.000Z",
        )

    contract = _make_contract(tmp_path, reverse_one_ms)
    report = contract.run(tmp_path / "output")
    assert report["status"] == audit.FULL_STATUS
    assert "created_gt_updated_unexpected_rows" not in report["violations"]
    diagnostic = report["created_updated_relation_diagnostic"]
    assert diagnostic["validity_gate"] is False
    assert diagnostic["created_gt_updated_rows"] == 1
    assert (
        diagnostic["parsed_pair_rows"]
        == report["metrics"]["created_updated_parsed_pair_rows"]
    )
    assert "created_gt_updated_rate_among_parsed_pairs" not in diagnostic
    assert not any(isinstance(value, float) for value in diagnostic.values())
    assert diagnostic["partition_exact"] is True
    assert report["gate_status"]["created_updated_timestamp_validity"] == (
        "PASS_EXACT_UTC_MILLISECOND_NON_NULL_NOT_AFTER_SNAPSHOT"
    )
    assert report["gate_status"]["created_updated_relation_diagnostic"] == (
        "DESCRIPTIVE_ONLY_NO_VENDOR_ORDER_INVARIANT"
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_status", "future_rows"),
    [
        ("created_date", "2026-03-30T23:59:59.999Z", audit.FULL_STATUS, 0),
        ("created_date", "2026-03-31T00:00:00.000Z", audit.BLOCKED_STATUS, 1),
        ("updated_date", "2026-03-30T23:59:59.999Z", audit.FULL_STATUS, 0),
        ("updated_date", "2026-03-31T00:00:00.000Z", audit.BLOCKED_STATUS, 1),
    ],
)
def test_snapshot_boundary_is_exact_to_the_millisecond(
    tmp_path: Path,
    field: str,
    value: str,
    expected_status: str,
    future_rows: int,
) -> None:
    def set_boundary(rows: list[dict[str, Any]]) -> None:
        rows[3][field] = value

    contract = _make_contract(tmp_path, set_boundary)
    report = contract.run(tmp_path / "output")
    assert report["status"] == expected_status
    assert report["violations"]["future_metadata_field_values"] == future_rows


def test_both_future_timestamps_count_as_two_field_values_in_one_row(
    tmp_path: Path,
) -> None:
    def set_both_future(rows: list[dict[str, Any]]) -> None:
        rows[3].update(
            created_date="2026-03-31T00:00:00.000Z",
            updated_date="2026-03-31T00:00:00.001Z",
        )

    contract = _make_contract(tmp_path, set_both_future)
    report = contract.run(tmp_path / "output")
    assert report["status"] == audit.BLOCKED_STATUS
    assert report["violations"]["future_metadata_field_values"] == 2
    assert report["metrics"]["created_after_snapshot_rows"] == 1
    assert report["metrics"]["updated_after_snapshot_rows"] == 1
    assert (
        report["accounting_contract"]["future_metadata_field_value_count_partition"]
        is True
    )


def test_reversal_does_not_dirty_a_nonpromotable_pilot(tmp_path: Path) -> None:
    def reverse_one_ms(rows: list[dict[str, Any]]) -> None:
        rows[0].update(
            created_date="2020-01-01T00:00:00.001Z",
            updated_date="2020-01-01T00:00:00.000Z",
        )

    contract = _make_contract(tmp_path, reverse_one_ms)
    report = contract.run(
        tmp_path / "pilot",
        max_files=1,
        content_freeze_receipt=None,
    )
    assert report["status"] == audit.PILOT_STATUS
    assert report["pilot_local_contract_clean"] is True
    assert report["created_updated_relation_diagnostic"]["created_gt_updated_rows"] == 1


@pytest.mark.parametrize("null_mode", ["both", "created_only", "updated_only"])
def test_metadata_nulls_remain_blocking(tmp_path: Path, null_mode: str) -> None:
    def set_nulls(rows: list[dict[str, Any]]) -> None:
        if null_mode in {"both", "created_only"}:
            rows[3]["created_date"] = None
        if null_mode in {"both", "updated_only"}:
            rows[3]["updated_date"] = None

    contract = _make_contract(tmp_path, set_nulls)
    report = contract.run(tmp_path / "output")
    assert report["status"] == audit.BLOCKED_STATUS
    assert report["violations"]["created_updated_null_rows"] == 1
    assert any(
        sample["reason"] == "created_updated_null"
        for sample in report["violation_samples"]
    )
    assert report["gate_status"]["publication_date_year_validity"] == (
        "PASS_WITH_FROZEN_EXCLUSIONS"
    )
    assert report["gate_status"]["created_updated_timestamp_validity"] == "BLOCKED"


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


def test_v2_config_and_checkpoint_schemas_are_rejected_fail_closed(
    tmp_path: Path,
) -> None:
    v2_config = yaml.safe_load(V2_CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises(audit.TemporalAuditError, match="config schema drift"):
        audit._validate_config_contract(v2_config)

    contract = _make_contract(tmp_path)
    output = tmp_path / "paused"
    contract.run(output, pause_after_new_files=1)
    plan = _read(output / "audit_plan.json")
    checkpoint_path = next((output / "checkpoints/main").glob("*.json"))
    checkpoint = _read(checkpoint_path)
    checkpoint["schema_version"] = "openalex_g1_main_temporal_validity_checkpoint.v2"
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


def test_v2_temporal_receipt_schema_is_rejected_by_final_validator(
    tmp_path: Path,
) -> None:
    contract = _make_contract(tmp_path)
    output = tmp_path / "output"
    report = contract.run(output)
    plan = _read(output / "audit_plan.json")
    receipt_path = output / "main_temporal_validity_audit.outputs.json"
    receipt = _read(receipt_path)
    receipt["schema_version"] = "openalex_g1_main_temporal_validity_outputs.v2"
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
    "legacy_key",
    [
        "reversal_policy",
        "exceptions",
        "exception_policy_resolution",
        "exception_key_fields",
        "require_every_exception_observed_exactly_once",
        "future_metadata_exceptions",
        "future_exception_policy_resolution",
        "future_exception_key_fields",
        "require_every_future_exception_observed_exactly_once",
    ],
)
def test_legacy_exception_allowlist_keys_are_forbidden(legacy_key: str) -> None:
    config = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    config["temporal_policy"]["created_updated"][legacy_key] = []
    with pytest.raises(
        audit.TemporalAuditError,
        match="metadata exception allowlists are forbidden in v3",
    ):
        audit._validate_config_contract(config)


def test_order_diagnostic_mutation_changes_result_fingerprint_without_blocking(
    tmp_path: Path,
) -> None:
    contract = _make_contract(tmp_path)
    output = tmp_path / "output"
    report = contract.run(output)
    plan = _read(output / "audit_plan.json")
    checkpoints = [
        _read(path) for path in sorted((output / "checkpoints/main").glob("*.json"))
    ]
    mutated = copy.deepcopy(checkpoints)
    assert mutated[0]["metrics"]["created_lte_updated_rows"] > 0
    mutated[0]["metrics"]["created_lte_updated_rows"] -= 1
    mutated[0]["metrics"]["created_gt_updated_rows"] += 1
    mutated_report = audit.build_report(
        yaml.safe_load(contract.config.read_text(encoding="utf-8")),
        plan,
        mutated,
        source_evidence=report["source_identity_evidence"],
        content_commitment=report["content_baseline_parent"],
    )
    assert mutated_report["status"] == audit.FULL_STATUS
    assert mutated_report["violations"] == report["violations"]
    assert mutated_report["created_updated_relation_diagnostic"][
        "created_gt_updated_rows"
    ] == (report["created_updated_relation_diagnostic"]["created_gt_updated_rows"] + 1)
    assert (
        mutated_report["provenance"]["result_fingerprint_sha256"]
        != report["provenance"]["result_fingerprint_sha256"]
    )


def test_claim_boundary_excludes_order_available_at_and_scientific_gates() -> None:
    assert "retained only as a descriptive count" in audit.CLAIM_BOUNDARY
    assert "does not establish record-event order" in audit.CLAIM_BOUNDARY
    assert "available_at" in audit.CLAIM_BOUNDARY
    assert "G-1 or G0" in audit.CLAIM_BOUNDARY
    assert "empirical G1" in audit.CLAIM_BOUNDARY


def test_full_v3_compact_evidence_is_exact_and_claim_bounded() -> None:
    summary = _read(SUMMARY_PATH)
    assert summary["schema_version"] == (
        "openalex_g1_main_temporal_validity_v3_full_summary.v1"
    )
    assert summary["status"] == audit.FULL_STATUS
    assert summary["scope"] == {
        "full_scope": True,
        "selected_files": 493,
        "selected_footer_rows": 492293207,
        "selected_source_bytes": 150062449219,
        "gpu_used": False,
        "component_pass": True,
        "downstream_promoted": False,
    }
    assert summary["row_accounting"]["raw_rows"] == 492293207
    assert summary["row_accounting"]["quarantined_w_minus_1_rows"] == 1
    assert summary["row_accounting"]["retained_rows"] == 492293206
    assert summary["publication_date_year"]["date_year_match_rows"] == 464727674
    assert summary["publication_date_year"]["date_year_mismatch_rows"] == 0
    metadata = summary["metadata_timestamp_validity"]
    assert metadata["created_updated_parsed_pair_rows"] == 492293206
    assert metadata["created_date_parse_error_rows"] == 0
    assert metadata["updated_date_parse_error_rows"] == 0
    assert metadata["future_metadata_field_values"] == 0
    ordering = summary["ordering_profile"]
    assert ordering["validity_gate"] is False
    assert ordering["created_lte_updated_rows"] == 492293203
    assert ordering["created_gt_updated_rows"] == 3
    assert ordering["parsed_pair_rows"] == 492293206
    assert ordering["partition_exact"] is True
    assert ordering["exception_allowlist_used"] is False
    assert summary["gate_status"]["main_temporal_validity_component"] == "PASS"
    assert summary["gate_status"]["g_minus_1_data_readiness"] == "PENDING"
    assert summary["gate_status"]["empirical_g1_signature"] == "NOT_RUN"
    assert summary["gate_status"]["scientific_effect"] == "NOT_ESTABLISHED"
    provenance = summary["provenance"]
    assert provenance["report_sha256"] == (
        "de98be73fb0365672cec7ee770c40de090ae012d45a0422faebd34c0b9d19200"
    )
    assert provenance["receipt_sha256"] == (
        "2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe"
    )
    assert provenance["result_fingerprint_sha256"] == (
        "a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5"
    )
    assert provenance["receipt_artifacts_rehashed_exactly"] == 497
    assert provenance["receipt_artifacts_all_exact"] is True
    assert provenance["extra_or_missing_contract_artifacts"] == 0
    encoded = SUMMARY_PATH.read_text(encoding="utf-8")
    markdown = SUMMARY_MARKDOWN_PATH.read_text(encoding="utf-8")
    for forbidden in ("/ssd/", "/hdd/", "PASS_G1", "effect established"):
        assert forbidden not in encoded
        assert forbidden not in markdown
    assert provenance["receipt_sha256"] in markdown
    assert provenance["result_fingerprint_sha256"] in markdown
    assert "Downstream products require a separate v3-only parent rebind" in markdown


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
            "openalex_g1_main_temporal_validity_config.v2",
            "config schema drift",
        ),
        (
            ("execution", "resource_preflight", "projected_bytes_per_row_upper_bound"),
            160,
            "projected row memory bound drift",
        ),
        (
            (
                "temporal_policy",
                "created_updated",
                "relation_diagnostic",
                "validity_gate",
            ),
            True,
            "created-updated relation diagnostic drift",
        ),
    ],
)
def test_v3_contract_mutations_fail_closed(
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
