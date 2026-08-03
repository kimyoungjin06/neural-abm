from __future__ import annotations

import copy
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import sys
from types import ModuleType
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_citation_available_at.py"
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_citation_available_at_v0.yaml"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_openalex_citation_available_at_module",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _day(value: str) -> int:
    return (date.fromisoformat(value) - date(1970, 1, 1)).days


def _parent(name: str, fingerprint_character: str) -> dict[str, Any]:
    common = {
        "name": name,
        "synthetic_contract_fixture_valid": True,
        "result_fingerprint_sha256": fingerprint_character * 64,
    }
    if name == "current_content_baseline":
        common.update(
            {
                "status": "PASS_CURRENT_CONTENT_BASELINE_COMPONENT",
                "state": "COMPLETE",
                "completion_contract": (
                    "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
                ),
                "artifact_tree_valid": True,
                "full_scope": True,
            }
        )
    elif name == "main_temporal_validity":
        common.update(
            {
                "status": (
                    "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_VALIDITY_COMPONENT_"
                    "WITH_FROZEN_EXCLUSIONS"
                ),
                "state": "COMPLETE_PENDING_RECEIPT",
                "completion_contract": (
                    "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
                ),
                "artifact_tree_valid": True,
                "receipt_written_last": True,
                "final_disk_validated": True,
                "embedded_content_matches": True,
                "full_scope": True,
            }
        )
    elif name == "work_source_membership_regression":
        common.update(
            {
                "status": (
                    "PASS_WORK_SOURCE_MEMBERSHIP_AND_PROFILE_AUDIT_WITH_"
                    "FROZEN_QUARANTINE"
                ),
                "artifact_hash_valid": True,
                "anchors_reproduced_on_synthetic_current_bytes": True,
            }
        )
    elif name == "canonical_work_time_spine":
        common.update(
            {
                "status": (
                    "PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_"
                    "WITH_INHERITED_TEMPORAL_EXCLUSIONS"
                ),
                "state": "COMPLETE_PENDING_RECEIPT",
                "completion_contract": (
                    "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
                ),
                "artifact_tree_valid": True,
                "receipt_written_last": True,
                "final_disk_validated": True,
                "embedded_parents_match": True,
                "partition_files": 4096,
                "rows": 492293206,
                "data_tree_sha256": "e" * 64,
            }
        )
    else:  # pragma: no cover - test helper guard
        raise AssertionError(name)
    return common


def _work_times() -> list[dict[str, Any]]:
    return [
        {
            "work_id_numeric": 1,
            "publication_day": _day("2026-03-20"),
            "publication_year": 2026,
            "temporal_disposition": 3,
        },
        {
            "work_id_numeric": 3,
            "publication_day": None,
            "publication_year": None,
            "temporal_disposition": 0,
        },
        {
            "work_id_numeric": 5,
            "publication_day": None,
            "publication_year": None,
            "temporal_disposition": 0,
        },
        {
            "work_id_numeric": 6,
            "publication_day": _day("2026-03-30"),
            "publication_year": 2026,
            "temporal_disposition": 3,
        },
        {
            "work_id_numeric": 7,
            "publication_day": _day("2026-03-20"),
            "publication_year": 2026,
            "temporal_disposition": 3,
        },
        {
            "work_id_numeric": 8,
            "publication_day": _day("2025-12-31"),
            "publication_year": 2025,
            "temporal_disposition": 2,
        },
    ]


def _pairs() -> list[dict[str, Any]]:
    return [
        {
            "citing_work_id_numeric": 1,
            "cited_work_id_numeric": 1,
            "source_task": "part-a",
        },
        {
            "citing_work_id_numeric": 1,
            "cited_work_id_numeric": 1,
            "source_task": "part-a",
        },
        {
            "citing_work_id_numeric": 3,
            "cited_work_id_numeric": 99,
            "source_task": "part-a",
        },
        {
            "citing_work_id_numeric": 1,
            "cited_work_id_numeric": 99,
            "source_task": "part-a",
        },
        {
            "citing_work_id_numeric": 1,
            "cited_work_id_numeric": 5,
            "source_task": "part-a",
        },
        {
            "citing_work_id_numeric": 1,
            "cited_work_id_numeric": 6,
            "source_task": "part-b",
        },
        {
            "citing_work_id_numeric": 1,
            "cited_work_id_numeric": 7,
            "source_task": "part-b",
        },
        {
            "citing_work_id_numeric": 1,
            "cited_work_id_numeric": 8,
            "source_task": "part-a",
        },
        {
            "citing_work_id_numeric": 1,
            "cited_work_id_numeric": 8,
            "source_task": "part-b",
        },
    ]


def _fixture(mode: str) -> dict[str, Any]:
    parent_names = [
        "current_content_baseline",
        "main_temporal_validity",
        "work_source_membership_regression",
    ]
    if mode == audit.REUSE_MODE:
        parent_names.append("canonical_work_time_spine")
    value: dict[str, Any] = {
        "schema_version": audit.FIXTURE_SCHEMA,
        "surface_kind": audit.SYNTHETIC_SURFACE,
        "mode": mode,
        "production_input": False,
        "parents": [
            _parent(name, character)
            for name, character in zip(
                parent_names, "abcd"[: len(parent_names)], strict=True
            )
        ],
        "reference_pairs": _pairs(),
    }
    value[audit.MODE_TIME_KEY[mode]] = _work_times()
    return value


def _write_fixture(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _run(
    tmp_path: Path,
    mode: str,
    *,
    fixture_value: dict[str, Any] | None = None,
    output_name: str = "output",
    **kwargs: Any,
) -> tuple[dict[str, Any], Path, Path]:
    fixture_path = tmp_path / f"{mode}.json"
    _write_fixture(fixture_path, fixture_value or _fixture(mode))
    output = tmp_path / output_name
    report = audit.run_synthetic_audit(
        config_path=CONFIG_PATH,
        synthetic_fixture=fixture_path,
        output_dir=output,
        mode=mode,
        **kwargs,
    )
    return report, output, fixture_path


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _snapshot_tree(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_direct_and_reuse_modes_are_accounting_equivalent() -> None:
    work_times = audit._parse_work_times(_work_times())
    pairs = audit._parse_pairs(_pairs())
    direct = audit.audit_pairs(work_times, pairs)
    reuse = audit.audit_pairs(dict(work_times), list(pairs))
    assert direct == reuse
    assert direct["exclusive_pair_buckets"] == {
        "self_loop": 1,
        "source_temporal_unusable": 1,
        "target_external_open_world": 1,
        "target_temporal_unusable": 1,
        "negative_lag_target_after_source": 1,
        "same_day": 1,
        "positive_lag": 1,
    }
    assert direct["accounting"] == {
        "raw_pair_rows": 9,
        "sum_pair_multiplicity": 9,
        "unique_pair_rows": 7,
        "duplicate_pair_groups": 2,
        "duplicate_pair_row_excess": 2,
        "maximum_pair_multiplicity": 2,
        "duplicate_groups_crossing_source_tasks": 1,
        "raw_self_loop_rows": 2,
        "unique_self_loop_pairs": 1,
        "duplicate_self_loop_row_excess": 1,
        "raw_in_snapshot_target_rows": 7,
        "raw_external_target_rows": 2,
        "unique_in_snapshot_target_pairs": 5,
        "unique_external_target_pairs": 2,
        "distinct_valid_target_ids": 6,
        "distinct_in_snapshot_target_ids": 5,
        "distinct_external_target_ids": 1,
    }
    assert direct["availability_accounting"] == {
        "resolution": "BIBLIOGRAPHIC_EVENT_TIME_CONVENTION",
        "edge_available_at_field": "citing_work_publication_date",
        "synthetic_physical_representation": "publication_day_int32",
        "raw_pair_rows_with_available_at": 8,
        "raw_pair_rows_without_available_at": 1,
        "unique_pairs_with_available_at": 6,
        "unique_pairs_without_available_at": 1,
        "openalex_observation_time_established": False,
        "historical_openalex_graph_established": False,
    }
    assert all(direct["closures"].values())
    assert all(direct["cross_tab_closures"].values())


@pytest.mark.parametrize("mode", audit.ALLOWED_MODES)
def test_synthetic_runner_emits_only_bounded_receipt_surface(
    tmp_path: Path,
    mode: str,
) -> None:
    report, output, _ = _run(tmp_path, mode)
    assert report["status"] == audit.SYNTHETIC_STATUS
    assert report["status"] != audit.FULL_STATUS
    assert report["surface_kind"] == "SYNTHETIC_FIXTURE_ONLY"
    assert report["production_status"] == "PRODUCTION_NOT_RUN"
    assert report["production_boundary"] == {
        "synthetic_fixture_opened": True,
        "production_source_roots_opened": False,
        "production_execution_supported": False,
        "production_component_status_emitted": False,
        "production_receipt_schema_emitted": False,
        "production_receipt_schema": audit.PRODUCTION_OUTPUTS_SCHEMA,
        "production_planned_operator_outputs_created": False,
        "canonical_spine_partitions_created": 0,
        "canonical_spine_partitions_consumed_from_disk": 0,
        "synthetic_fixture_payload_only": True,
        "synthetic_fixture_hard_byte_cap": audit.MAX_SYNTHETIC_FIXTURE_BYTES,
        "synthetic_work_time_hard_row_cap": audit.MAX_SYNTHETIC_WORK_TIME_ROWS,
        "synthetic_reference_pair_hard_row_cap": (
            audit.MAX_SYNTHETIC_REFERENCE_PAIR_ROWS
        ),
    }
    assert report["mode"] == mode
    assert report["execution_abi"] == audit.EXECUTION_ABI[mode]
    assert report["parent_contract_fixture"]["ordered_names"] == list(
        audit.MODE_PARENT_ORDER[mode]
    )
    receipt = _read(output / audit.RECEIPT_BASENAME)
    assert receipt["schema_version"] == audit.OUTPUTS_SCHEMA
    assert receipt["schema_version"] != audit.PRODUCTION_OUTPUTS_SCHEMA
    assert receipt["completion_contract"] == audit.SYNTHETIC_COMPLETION_CONTRACT
    assert receipt["production_receipt_schema_satisfied"] is False
    assert receipt["production_planned_operator_outputs_created"] is False
    assert receipt["canonical_spine_partition_outputs_created"] == 0
    assert set(receipt["artifacts"]) == set(audit.EXPECTED_RECEIPT_ARTIFACTS)
    assert audit._relative_output_files(output) == audit.EXPECTED_FINAL_FILES
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in output.rglob("*")
        if path.is_file()
    )
    assert not (output / "deduplicated_edge_event_partitions").exists()
    assert not (output / "chronology_safe_edge_partitions_by_cited_work").exists()
    disk_report, _ = audit.validate_terminal_publication(output)
    assert disk_report == report


def test_two_runner_modes_publish_the_same_mode_invariant_result_hash(
    tmp_path: Path,
) -> None:
    direct, _, _ = _run(tmp_path, audit.DIRECT_MODE, output_name="direct")
    reuse, _, _ = _run(tmp_path, audit.REUSE_MODE, output_name="reuse")
    assert (
        direct["mode_invariant_result_sha256"] == reuse["mode_invariant_result_sha256"]
    )
    assert direct["accounting"] == reuse["accounting"]
    assert direct["exclusive_pair_buckets"] == reuse["exclusive_pair_buckets"]


def test_pair_hash_abi_is_exact_stable_directed_and_keeps_duplicates_together() -> None:
    expected = (
        int.from_bytes(hashlib.sha256(struct.pack(">QQ", 1, 8)).digest()[:8], "big")
        % 17
    )
    assert audit.pair_hash_bucket(1, 8, 17) == expected
    assert audit.pair_hash_bucket(1, 8, 17) == expected
    assert struct.pack(">QQ", 1, 8) != struct.pack(">QQ", 8, 1)

    work_times = audit._parse_work_times(_work_times())
    pairs = audit._parse_pairs(_pairs())
    selected_index = audit.pair_hash_bucket(1, 8, 17)
    result = audit.audit_pairs(
        work_times,
        pairs,
        bucket_count=17,
        bucket_index=selected_index,
    )
    expected_raw = sum(
        audit.pair_hash_bucket(source, target, 17) == selected_index
        for source, target, _ in pairs
    )
    assert result["accounting"]["raw_pair_rows"] == expected_raw
    assert (
        result["accounting"]["raw_pair_rows"] > result["accounting"]["unique_pair_rows"]
    )
    assert result["selection"]["global_coverage_claim_allowed"] is False


def test_classifier_is_total_and_obeys_first_true_precedence() -> None:
    work_times = {
        1: (None, None, 0),
        2: (100, 2000, 2),
        3: (None, None, 0),
        4: (110, 2000, 2),
        5: (100, 2000, 3),
        6: (80, 1999, 2),
    }
    cases = {
        (1, 1): "self_loop",
        (1, 99): "source_temporal_unusable",
        (2, 99): "target_external_open_world",
        (2, 3): "target_temporal_unusable",
        (2, 4): "negative_lag_target_after_source",
        (2, 5): "same_day",
        (2, 6): "positive_lag",
    }
    observed = {pair: audit._pair_attributes(*pair, 1, work_times)[0] for pair in cases}
    assert observed == cases
    self_bucket, self_attributes, self_lag = audit._pair_attributes(
        1,
        1,
        1,
        work_times,
    )
    assert self_bucket == "self_loop"
    assert self_attributes["source_temporal_disposition"] == "disposition_0"
    assert self_attributes["signed_lag_class_or_na"] == "NA"
    assert self_lag is None


@pytest.mark.parametrize(
    ("mode", "mutator", "message"),
    [
        (
            audit.DIRECT_MODE,
            lambda value: value["parents"].append(
                _parent("canonical_work_time_spine", "d")
            ),
            "parent order",
        ),
        (
            audit.REUSE_MODE,
            lambda value: value["parents"][1].update(state="COMPLETE"),
            "temporal state",
        ),
        (
            audit.REUSE_MODE,
            lambda value: value["parents"][3].update(artifact_tree_valid=False),
            "spine artifact tree",
        ),
        (
            audit.REUSE_MODE,
            lambda value: value.update(direct_work_times=_work_times()),
            "forbidden other-mode",
        ),
    ],
)
def test_parent_or_mode_substitution_fails_before_publication(
    tmp_path: Path,
    mode: str,
    mutator: Any,
    message: str,
) -> None:
    fixture = _fixture(mode)
    mutator(fixture)
    with pytest.raises(audit.CitationSyntheticAuditError, match=message):
        _run(tmp_path, mode, fixture_value=fixture)
    assert not (tmp_path / "output" / audit.RECEIPT_BASENAME).exists()


def test_reuse_fixture_cannot_fall_back_to_direct_times(tmp_path: Path) -> None:
    fixture = _fixture(audit.REUSE_MODE)
    fixture["direct_work_times"] = fixture.pop("spine_work_times")
    with pytest.raises(audit.CitationSyntheticAuditError, match="forbidden other-mode"):
        _run(tmp_path, audit.REUSE_MODE, fixture_value=fixture)


def test_terminal_resume_rehashes_exact_surface_before_opening_fixture_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, output, _ = _run(tmp_path, audit.DIRECT_MODE)
    before = _snapshot_tree(output)

    def forbidden_fixture_open(path: Path) -> tuple[dict[str, Any], bytes, str]:
        raise AssertionError(f"terminal resume opened fixture payload: {path.name}")

    monkeypatch.setattr(audit, "_load_fixture_payload", forbidden_fixture_open)
    resumed = audit.run_synthetic_audit(
        config_path=CONFIG_PATH,
        synthetic_fixture=tmp_path / "does-not-exist.json",
        output_dir=output,
        mode=audit.DIRECT_MODE,
        resume=True,
    )
    assert resumed == report
    assert _snapshot_tree(output) == before


def test_corrupt_terminal_artifact_is_invalidated_receipt_first_without_fixture_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, output, fixture_path = _run(tmp_path, audit.DIRECT_MODE)
    checkpoint = output / "checkpoints/synthetic_accounting.json"
    payload = checkpoint.read_bytes()
    assert b'"raw_pair_rows":9' in payload
    checkpoint.write_bytes(
        payload.replace(b'"raw_pair_rows":9', b'"raw_pair_rows":8', 1)
    )
    checkpoint.chmod(0o600)

    opened = False

    def forbidden_fixture_open(path: Path) -> tuple[dict[str, Any], bytes, str]:
        nonlocal opened
        opened = True
        raise AssertionError(path)

    removed: list[str] = []
    original_unlink = audit._unlink_regular

    def record_unlink(path: Path) -> None:
        removed.append(path.name)
        original_unlink(path)

    monkeypatch.setattr(audit, "_load_fixture_payload", forbidden_fixture_open)
    monkeypatch.setattr(audit, "_unlink_regular", record_unlink)
    with pytest.raises(audit.CitationSyntheticAuditError, match="artifact hash"):
        audit.run_synthetic_audit(
            config_path=CONFIG_PATH,
            synthetic_fixture=fixture_path,
            output_dir=output,
            mode=audit.DIRECT_MODE,
            resume=True,
        )
    assert opened is False
    assert removed[0] == audit.RECEIPT_BASENAME
    assert not (output / audit.RECEIPT_BASENAME).exists()
    assert not (output / audit.REPORT_BASENAME).exists()
    assert not (output / audit.MARKDOWN_BASENAME).exists()
    assert _read(output / "run_state.json")["status"] == (
        "FINALIZATION_FAILED_NO_SYNTHETIC_RECEIPT"
    )


def test_unexpected_terminal_file_invalidates_publication_before_fixture_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, output, fixture_path = _run(tmp_path, audit.REUSE_MODE)
    unexpected = output / "unexpected-production-like.parquet"
    unexpected.write_bytes(b"not parquet")
    unexpected.chmod(0o600)

    def forbidden_fixture_open(path: Path) -> tuple[dict[str, Any], bytes, str]:
        raise AssertionError(path)

    monkeypatch.setattr(audit, "_load_fixture_payload", forbidden_fixture_open)
    with pytest.raises(audit.CitationSyntheticAuditError, match="file set"):
        audit.run_synthetic_audit(
            config_path=CONFIG_PATH,
            synthetic_fixture=fixture_path,
            output_dir=output,
            mode=audit.REUSE_MODE,
            resume=True,
        )
    assert not (output / audit.RECEIPT_BASENAME).exists()
    assert not (output / audit.REPORT_BASENAME).exists()


def test_duplicate_key_receipt_is_rejected_before_fixture_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, output, fixture_path = _run(tmp_path, audit.DIRECT_MODE)
    receipt_path = output / audit.RECEIPT_BASENAME
    payload = receipt_path.read_bytes()
    assert payload.startswith(b"{")
    receipt_path.write_bytes(b'{"status":"SHADOW",' + payload[1:])
    receipt_path.chmod(0o600)

    def forbidden_fixture_open(path: Path) -> tuple[dict[str, Any], bytes, str]:
        raise AssertionError(path)

    monkeypatch.setattr(audit, "_load_fixture_payload", forbidden_fixture_open)
    with pytest.raises(audit.CitationSyntheticAuditError, match="duplicate JSON key"):
        audit.run_synthetic_audit(
            config_path=CONFIG_PATH,
            synthetic_fixture=fixture_path,
            output_dir=output,
            mode=audit.DIRECT_MODE,
            resume=True,
        )
    assert not receipt_path.exists()
    assert not (output / audit.REPORT_BASENAME).exists()


def test_failure_after_receipt_write_leaves_no_promotable_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = tmp_path / "fixture.json"
    _write_fixture(fixture_path, _fixture(audit.DIRECT_MODE))
    output = tmp_path / "output"
    removed: list[str] = []
    original_unlink = audit._unlink_regular

    def record_unlink(path: Path) -> None:
        removed.append(path.name)
        original_unlink(path)

    def fail_final_validation(
        output_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert (output_dir / audit.RECEIPT_BASENAME).exists()
        raise audit.CitationSyntheticAuditError("injected final disk failure")

    monkeypatch.setattr(audit, "_unlink_regular", record_unlink)
    monkeypatch.setattr(audit, "validate_terminal_publication", fail_final_validation)
    with pytest.raises(audit.CitationSyntheticAuditError, match="injected"):
        audit.run_synthetic_audit(
            config_path=CONFIG_PATH,
            synthetic_fixture=fixture_path,
            output_dir=output,
            mode=audit.DIRECT_MODE,
        )
    assert removed[0] == audit.RECEIPT_BASENAME
    assert not (output / audit.RECEIPT_BASENAME).exists()
    assert not (output / audit.REPORT_BASENAME).exists()
    assert not (output / audit.MARKDOWN_BASENAME).exists()
    assert _read(output / "run_state.json")["status"] == (
        "FINALIZATION_FAILED_NO_SYNTHETIC_RECEIPT"
    )


def test_fixture_drift_during_execution_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = tmp_path / "fixture.json"
    _write_fixture(fixture_path, _fixture(audit.DIRECT_MODE))
    output = tmp_path / "output"
    original_audit_pairs = audit.audit_pairs

    def mutate_fixture_after_accounting(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_audit_pairs(*args, **kwargs)
        fixture_path.write_bytes(fixture_path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(audit, "audit_pairs", mutate_fixture_after_accounting)
    with pytest.raises(audit.CitationSyntheticAuditError, match="fixture changed"):
        audit.run_synthetic_audit(
            config_path=CONFIG_PATH,
            synthetic_fixture=fixture_path,
            output_dir=output,
            mode=audit.DIRECT_MODE,
        )
    assert not (output / audit.RECEIPT_BASENAME).exists()
    assert not (output / audit.REPORT_BASENAME).exists()
    assert _read(output / "run_state.json")["status"] == ("FAILED_SYNTHETIC_NO_RECEIPT")


def test_cli_has_no_production_source_argument_and_requires_explicit_synthetic_execute(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "fixture.json"
    _write_fixture(fixture_path, _fixture(audit.DIRECT_MODE))
    output = tmp_path / "output"
    with pytest.raises(SystemExit):
        audit.build_parser().parse_args(
            [
                "--synthetic-fixture",
                str(fixture_path),
                "--output-dir",
                str(output),
                "--mode",
                audit.DIRECT_MODE,
                "--source-root",
                "/hdd/forbidden",
            ]
        )
    assert (
        audit.main(
            [
                "--synthetic-fixture",
                str(fixture_path),
                "--output-dir",
                str(output),
                "--mode",
                audit.DIRECT_MODE,
            ]
        )
        == 2
    )
    assert not output.exists()


def test_cli_executes_only_with_explicit_synthetic_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture_path = tmp_path / "fixture.json"
    _write_fixture(fixture_path, _fixture(audit.REUSE_MODE))
    output = tmp_path / "output"
    assert (
        audit.main(
            [
                "--synthetic-fixture",
                str(fixture_path),
                "--output-dir",
                str(output),
                "--mode",
                audit.REUSE_MODE,
                "--execute-synthetic",
            ]
        )
        == 0
    )
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == audit.SYNTHETIC_STATUS
    assert emitted["surface_kind"] == audit.SYNTHETIC_SURFACE
    assert emitted["production_status"] == audit.PRODUCTION_STATUS
    assert (output / audit.RECEIPT_BASENAME).is_file()


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("production_execution_supported", True),
        ("production_source_access_supported", True),
        ("synthetic_fixture_only", False),
        ("production_result_exists", True),
    ],
)
def test_config_cannot_enable_production_through_synthetic_runner(
    tmp_path: Path,
    field: str,
    unsafe: Any,
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["lifecycle"][field] = unsafe
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    fixture_path = tmp_path / "fixture.json"
    _write_fixture(fixture_path, _fixture(audit.DIRECT_MODE))
    with pytest.raises(audit.CitationSyntheticAuditError):
        audit.run_synthetic_audit(
            config_path=config_path,
            synthetic_fixture=fixture_path,
            output_dir=tmp_path / "output",
            mode=audit.DIRECT_MODE,
        )
    assert not (tmp_path / "output").exists()


def test_bucket_and_identifier_fail_closed_boundaries() -> None:
    work_times = audit._parse_work_times(_work_times())
    pairs = audit._parse_pairs(_pairs())
    with pytest.raises(audit.CitationSyntheticAuditError, match="supplied together"):
        audit.audit_pairs(work_times, pairs, bucket_count=17)
    with pytest.raises(audit.CitationSyntheticAuditError, match="greater than one"):
        audit.pair_hash_bucket(1, 8, 1)
    with pytest.raises(audit.CitationSyntheticAuditError, match="bucket index"):
        audit.audit_pairs(work_times, pairs, bucket_count=17, bucket_index=True)
    malformed = copy.deepcopy(_pairs())
    malformed[0]["cited_work_id_numeric"] = 0
    with pytest.raises(audit.CitationSyntheticAuditError, match="outside uint64"):
        audit._parse_pairs(malformed)


def test_hard_fixture_and_row_caps_make_full_production_payload_impossible(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (audit.MAX_SYNTHETIC_FIXTURE_BYTES + 1))
    with pytest.raises(audit.CitationSyntheticAuditError, match="hard byte cap"):
        audit._load_fixture_payload(oversized)

    pair = _pairs()[0]
    with pytest.raises(audit.CitationSyntheticAuditError, match="hard row cap"):
        audit._parse_pairs(
            [
                copy.deepcopy(pair)
                for _ in range(audit.MAX_SYNTHETIC_REFERENCE_PAIR_ROWS + 1)
            ]
        )
    work_time = _work_times()[0]
    with pytest.raises(audit.CitationSyntheticAuditError, match="hard row cap"):
        audit._parse_work_times(
            [
                {**work_time, "work_id_numeric": index}
                for index in range(audit.MAX_SYNTHETIC_WORK_TIME_ROWS + 1)
            ]
        )


def test_missing_source_time_and_duplicate_work_time_fail_closed() -> None:
    pairs = audit._parse_pairs(_pairs())
    work_times = audit._parse_work_times(_work_times())
    del work_times[3]
    with pytest.raises(audit.CitationSyntheticAuditError, match="every citing work"):
        audit.audit_pairs(work_times, pairs)

    duplicate = _work_times()
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(audit.CitationSyntheticAuditError, match="globally unique"):
        audit._parse_work_times(duplicate)


def test_work_time_rows_cover_all_exact_snapshot_disposition_boundaries() -> None:
    rows = [
        {
            "work_id_numeric": 1,
            "publication_day": None,
            "publication_year": None,
            "temporal_disposition": 0,
        },
        {
            "work_id_numeric": 2,
            "publication_day": _day("1899-12-31"),
            "publication_year": 1899,
            "temporal_disposition": 1,
        },
        {
            "work_id_numeric": 3,
            "publication_day": _day("1900-01-01"),
            "publication_year": 1900,
            "temporal_disposition": 2,
        },
        {
            "work_id_numeric": 4,
            "publication_day": _day("2026-03-30"),
            "publication_year": 2026,
            "temporal_disposition": 3,
        },
        {
            "work_id_numeric": 5,
            "publication_day": _day("2026-03-31"),
            "publication_year": 2026,
            "temporal_disposition": 4,
        },
        {
            "work_id_numeric": 6,
            "publication_day": _day("2027-01-01"),
            "publication_year": 2027,
            "temporal_disposition": 5,
        },
    ]

    parsed = audit._parse_work_times(rows)

    assert [parsed[index][2] for index in range(1, 7)] == list(range(6))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda row: row.update(publication_year=2025),
            "day/year exact match",
        ),
        (
            lambda row: row.update(temporal_disposition=2),
            "disposition is inconsistent",
        ),
        (
            lambda row: row.update(publication_year=0),
            "calendar range",
        ),
    ],
)
def test_inconsistent_work_time_rows_fail_before_publication(
    tmp_path: Path,
    mutator: Any,
    message: str,
) -> None:
    fixture = _fixture(audit.DIRECT_MODE)
    mutator(fixture["direct_work_times"][0])

    with pytest.raises(audit.CitationSyntheticAuditError, match=message):
        _run(tmp_path, audit.DIRECT_MODE, fixture_value=fixture)

    assert not (tmp_path / "output").exists()


def test_live_config_declares_distinct_synthetic_and_future_production_abis() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    lifecycle = config["lifecycle"]
    assert lifecycle["design_status"] == (
        "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN"
    )
    assert lifecycle["implementation_status"] == "IMPLEMENTED_SYNTHETICALLY"
    assert lifecycle["production_status"] == "PRODUCTION_NOT_RUN"
    assert lifecycle["synthetic_fixture_only"] is True
    assert lifecycle["production_execution_supported"] is False
    assert lifecycle["production_source_access_supported"] is False
    synthetic = config["synthetic_execution_contract"]
    assert synthetic["fixture_schema_version"] == audit.FIXTURE_SCHEMA
    assert synthetic["report_schema_version"] == audit.REPORT_SCHEMA
    assert synthetic["receipt_schema_version"] == audit.OUTPUTS_SCHEMA
    assert synthetic["receipt_completion_contract"] == (
        audit.SYNTHETIC_COMPLETION_CONTRACT
    )
    assert synthetic["future_production_receipt_schema_satisfied"] is False
    assert synthetic["production_planned_outputs_created"] is False
    assert synthetic["canonical_spine_partition_outputs_created"] == 0
    assert synthetic["hard_max_fixture_bytes"] == audit.MAX_SYNTHETIC_FIXTURE_BYTES
    assert synthetic["hard_max_work_time_rows"] == audit.MAX_SYNTHETIC_WORK_TIME_ROWS
    assert synthetic["hard_max_reference_pair_rows"] == (
        audit.MAX_SYNTHETIC_REFERENCE_PAIR_ROWS
    )
    assert synthetic["work_time_publication_day_epoch"] == "1970-01-01"
    assert synthetic["work_time_day_year_exact_match_required"] is True
    assert synthetic["work_time_disposition_must_match_snapshot_classifier"] is True
    assert config["completion_contract"]["future_receipt_schema_version"] == (
        audit.PRODUCTION_OUTPUTS_SCHEMA
    )
    assert audit.OUTPUTS_SCHEMA != audit.PRODUCTION_OUTPUTS_SCHEMA
