from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_w0_positive_id_sensitivity.py"
CONFIG_PATH = (
    REPO_ROOT
    / "experiments/configs/openalex_g1_w0_positive_id_sensitivity_v0.yaml"
)
CROSS_FIELD_PATH = (
    REPO_ROOT
    / "experiments/evidence/results/"
    "openalex_g1_cross_field_feasibility_20260718.summary.json"
)
SUMMARY_PATH = (
    REPO_ROOT
    / "experiments/evidence/results/"
    "openalex_g1_w0_positive_id_sensitivity_20260718.summary.json"
)
MARKDOWN_PATH = SUMMARY_PATH.with_suffix(".md")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_openalex_w0_positive_id_sensitivity_module", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _selected_spec(
    path: Path,
    *,
    root: Path,
    role: str,
    columns: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    info = path.stat()
    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    relative = path.relative_to(root).as_posix()
    selected_schema = {
        column: {
            "nullable": bool(parquet.schema_arrow.field(column).nullable),
            "type": str(parquet.schema_arrow.field(column).type),
        }
        for column in columns
    }
    physical_fingerprint = "a" * 64
    canonical_fingerprint = "b" * 64
    spec = {
        "role": role,
        "relative_path": relative,
        "expected_rows": rows,
        "expected_bytes": info.st_size,
        "expected_mtime_ns": info.st_mtime_ns,
        "physical_schema_fingerprint_sha256": physical_fingerprint,
        "canonical_schema_fingerprint_sha256": canonical_fingerprint,
        "selected_columns": columns,
        "selected_schema": selected_schema,
    }
    inventory = {
        "relative_path": relative,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "footer_rows": rows,
        "physical_schema_fingerprint_sha256": physical_fingerprint,
        "canonical_schema_fingerprint_sha256": canonical_fingerprint,
        "schema_fields": selected_schema,
    }
    return spec, inventory


def _synthetic_scans(tmp_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = tmp_path / "snapshot"
    main = root / "main/part-0.parquet"
    authorship = root / "authorship/part-0.parquet"
    main.parent.mkdir(parents=True)
    authorship.parent.mkdir(parents=True)

    main_columns = [
        "id",
        "ids__openalex",
        "ids__mag",
        "publication_date",
        "publication_year",
        "authors_count",
        "primary_topic__id",
        "primary_topic__field__id",
        "primary_topic__domain__id",
    ]
    pq.write_table(
        pa.table(
            {
                "id": ["https://openalex.org/W1", audit.W0],
                "ids__openalex": ["https://openalex.org/W1", audit.W0],
                "ids__mag": ["1", "0"],
                "publication_date": ["2020-01-01", "2024-01-01"],
                "publication_year": pa.array([2020.0, 2024.0], type=pa.float64()),
                "authors_count": pa.array([2, 1], type=pa.int64()),
                "primary_topic__id": ["https://openalex.org/T1", None],
                "primary_topic__field__id": [
                    "https://openalex.org/fields/11",
                    None,
                ],
                "primary_topic__domain__id": [
                    "https://openalex.org/domains/1",
                    None,
                ],
            }
        ),
        main,
    )
    authorship_columns = [
        "id",
        "authorships__author_position",
        "authorships__author__id",
        "authorships__author__orcid",
    ]
    pq.write_table(
        pa.table(
            {
                "id": ["https://openalex.org/W1", audit.W0],
                "authorships__author_position": ["first", "first"],
                "authorships__author__id": ["https://openalex.org/A1", None],
                "authorships__author__orcid": [None, None],
            }
        ),
        authorship,
    )

    main_spec, main_inventory = _selected_spec(
        main, root=root, role="main_work", columns=main_columns
    )
    author_spec, author_inventory = _selected_spec(
        authorship, root=root, role="authorship", columns=authorship_columns
    )
    inventory = {
        main_inventory["relative_path"]: main_inventory,
        author_inventory["relative_path"]: author_inventory,
    }
    scans = [
        audit.scan_current_w0_projection(root, main_spec, inventory),
        audit.scan_current_w0_projection(root, author_spec, inventory),
    ]
    return scans, _config()["positive_id_sensitivity"]


def test_config_freezes_bounded_positive_id_policy_and_pending_gates() -> None:
    config = audit.read_config(CONFIG_PATH)
    source = config["source"]
    policy = config["positive_id_sensitivity"]
    gates = config["success_contract"]["gate_status"]

    assert source["inventory_manifest"]["sha256"] == (
        "6b5929a0155c033e818a816aad57095380a3907ab558c8d2cd58a89055f22b46"
    )
    assert len(source["current_selected_files"]) == 2
    assert re.fullmatch(policy["retained_work_id_regex"], "https://openalex.org/W1")
    assert not re.fullmatch(policy["retained_work_id_regex"], audit.W0)
    assert policy["require_master_work_delta"] == -1
    assert policy["require_null_field_delta"] == -1
    assert config["success_contract"]["status"] == audit.SUCCESS_STATUS
    assert gates["positive_id_exclusion_sensitivity"] == "PASS"
    assert gates["w_zero_entity_validity"] == "UNRESOLVED"
    assert gates["parent_content_identity"] == "UNVERIFIED"
    assert gates["author_entity_validity"] == "PENDING"
    assert gates["publication_date_validity"] == "PENDING"
    assert gates["g_minus_1_data_readiness"] == "PENDING"
    assert gates["empirical_g1_signature"] == "NOT_RUN"
    assert gates["pass_g1"] is False
    serialized = json.dumps(config, sort_keys=True)
    assert "/home/" not in serialized
    assert "/hdd/" not in serialized
    assert "/ssd/" not in serialized


def test_tracked_summary_is_exact_bounded_w0_sensitivity_evidence() -> None:
    summary = _json(SUMMARY_PATH)
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")

    assert summary["status"] == audit.SUCCESS_STATUS
    assert summary["scope"] == {
        "field_selection_performed": False,
        "full_root_rescan_performed": False,
        "gpu_used": False,
        "positive_id_sensitivity_only": True,
        "scientific_effect_estimation_performed": False,
        "w0_entity_validity_decided": False,
        "work_parquet_files_read": 2,
        "work_parquet_rows_scanned": 2_000_000,
    }
    assert summary["frozen_w0_incidence"]["source_rows"] == 6
    assert summary["frozen_w0_incidence"]["source_files"] == 6
    assert "affected_source_files" not in summary["frozen_w0_incidence"]
    assert summary["current_main_topics_w0"] == {"main_rows": 1, "topic_rows": 0}
    assert summary["sensitivity"]["delta"]["master_work_rows"] == -1
    assert summary["sensitivity"]["delta"]["null_primary_field_works"] == -1
    assert set(
        summary["sensitivity"]["delta"]["nonnull_field_count_deltas"].values()
    ) == {0}
    assert all(summary["sensitivity"]["checks"].values())
    assert summary["provenance"]["tool_sha256"] == audit.sha256_file(SCRIPT_PATH)
    assert summary["provenance"]["config_sha256"] == audit.sha256_file(CONFIG_PATH)
    assert summary["provenance"]["result_fingerprint_sha256"] == (
        "3056af90867d813ef70b9bcf11c67385c3b288d2f21d89de150c4f2ddbc30190"
    )
    serialized = json.dumps(summary, sort_keys=True)
    assert "/home/" not in serialized
    assert "/hdd/" not in serialized
    assert "/ssd/" not in serialized
    assert audit.SUCCESS_STATUS in markdown
    assert summary["provenance"]["result_fingerprint_sha256"] in markdown


def test_selected_file_scan_revalidates_exact_w0_projection(
    tmp_path: Path,
) -> None:
    scans, policy = _synthetic_scans(tmp_path)

    projection = audit.validate_current_rows(scans, policy)

    assert all(len(scan["projected_payload_sha256"]) == 64 for scan in scans)
    assert sum(scan["projected_rows"] for scan in scans) == 4
    assert len(projection["main_selected_values_sha256"]) == 64
    assert len(projection["authorship_selected_values_sha256"]) == 64
    assert audit.validate_current_rows(copy.deepcopy(scans), policy) == projection


def test_current_projection_mutations_fail_closed(tmp_path: Path) -> None:
    scans, policy = _synthetic_scans(tmp_path)
    mutated = copy.deepcopy(scans)
    mutated[0]["w0_rows"][0]["publication_date"] = "2024-01-02"
    with pytest.raises(audit.W0SensitivityError, match="publication_date"):
        audit.validate_current_rows(mutated, policy)

    type_drift = copy.deepcopy(scans)
    type_drift[0]["w0_rows"][0]["authors_count"] = 1.0
    with pytest.raises(audit.W0SensitivityError, match="authors_count"):
        audit.validate_current_rows(type_drift, policy)

    duplicate = copy.deepcopy(scans)
    duplicate[1]["w0_rows"].append(copy.deepcopy(duplicate[1]["w0_rows"][0]))
    with pytest.raises(audit.W0SensitivityError, match="authorship W0 count"):
        audit.validate_current_rows(duplicate, policy)

    invalid_policy = copy.deepcopy(policy)
    invalid_policy["expected_current_main_rows"] = 0
    with pytest.raises(audit.W0SensitivityError, match="must select one row"):
        audit.validate_current_rows(scans, invalid_policy)


def test_compact_sentinel_claims_only_the_proven_table_set() -> None:
    config = _config()
    contract = config["frozen_evidence"]["sentinel_incidence"]
    sentinel = _json(
        REPO_ROOT / contract["relative_path"]
    )

    result = audit.validate_sentinel(sentinel, contract)

    assert result["source_rows"] == 6
    assert result["source_files"] == 6
    assert result["source_tables"] == 6
    assert result["work_target_rows"] == 0
    assert result["affected_source_tables"] == sorted(
        contract["affected_source_tables"]
    )
    assert "affected_source_files" not in result

    drifted = copy.deepcopy(contract)
    drifted["affected_source_tables"][0] = "unproven_table"
    with pytest.raises(audit.W0SensitivityError, match="table set mismatch"):
        audit.validate_sentinel(sentinel, drifted)


def test_live_selected_schema_drift_blocks_before_projection(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    path = root / "main/part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"id": pa.array([audit.W0], type=pa.string())}), path)
    spec, inventory = _selected_spec(
        path,
        root=root,
        role="main_work",
        columns=["id"],
    )
    pq.write_table(pa.table({"id": pa.array([0], type=pa.int64())}), path)
    info = path.stat()
    spec["expected_bytes"] = inventory["bytes"] = info.st_size
    spec["expected_mtime_ns"] = inventory["mtime_ns"] = info.st_mtime_ns

    with pytest.raises(audit.W0SensitivityError, match="live selected schema drift"):
        audit.scan_current_w0_projection(
            root,
            spec,
            {inventory["relative_path"]: inventory},
        )


def test_claim_and_gate_drift_are_rejected(tmp_path: Path) -> None:
    config = _config()
    config["success_contract"]["gate_status"]["g_minus_1_data_readiness"] = "PASS"
    path = tmp_path / "gate-drift.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(audit.W0SensitivityError, match="gate status drift"):
        audit.read_config(path)

    config = _config()
    config["claim_boundary"] = "This component proves G-1."
    path = tmp_path / "claim-drift.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(audit.W0SensitivityError, match="claim boundary drift"):
        audit.read_config(path)

    config = _config()
    config["positive_id_sensitivity"]["retained_work_id_regex"] = (
        r"^https://openalex[.]org/W1$"
    )
    path = tmp_path / "positive-policy-drift.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(audit.W0SensitivityError, match="policy drift"):
        audit.read_config(path)

    config = _config()
    del config["positive_id_sensitivity"]["expected_main_projection"][
        "authors_count"
    ]
    path = tmp_path / "projection-policy-drift.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(audit.W0SensitivityError, match="policy drift"):
        audit.read_config(path)


def test_positive_id_profile_changes_only_the_predeclared_w0_stratum() -> None:
    cross_field = _json(CROSS_FIELD_PATH)
    policy = _config()["positive_id_sensitivity"]
    w0_profile = {
        "joint_stratum_before": {
            "author_id_status": "none_usable",
            "declared_authors_band": "1",
            "primary_field_id": None,
            "publication_year": 2024,
            "works": 4_155_373,
        }
    }

    result = audit.derive_positive_id_profile(cross_field, w0_profile, policy)

    assert result["baseline_grammar_valid_view"]["master_work_rows"] == 492_293_206
    assert result["positive_id_view"]["master_work_rows"] == 492_293_205
    assert result["positive_id_view"]["classified_primary_field_works"] == (
        354_145_248
    )
    assert result["positive_id_view"]["null_primary_field_works"] == 138_147_957
    assert set(result["delta"]["nonnull_field_count_deltas"].values()) == {0}
    assert len(result["delta"]["nonnull_field_count_deltas"]) == 26
    assert result["delta"]["global_authorship_status_counts"] == {
        "complete": 0,
        "none_usable": -1,
        "partial": 0,
        "zero_authorship_row": 0,
    }
    assert result["delta"]["global_temporal_window_counts"][
        "followup_diagnostic_2023_2025"
    ] == -1
    assert all(result["checks"].values())


def test_profile_and_delta_contract_mutations_fail_closed() -> None:
    cross_field = _json(CROSS_FIELD_PATH)
    policy = _config()["positive_id_sensitivity"]
    w0_profile = {"joint_stratum_before": {"works": 1}}

    broken_conservation = copy.deepcopy(cross_field)
    broken_conservation["profile"]["master_work_rows"] += 1
    with pytest.raises(audit.W0SensitivityError, match="field conservation"):
        audit.derive_positive_id_profile(broken_conservation, w0_profile, policy)

    broken_delta = copy.deepcopy(policy)
    broken_delta["require_master_work_delta"] = -2
    with pytest.raises(audit.W0SensitivityError, match="exactly one work"):
        audit.derive_positive_id_profile(cross_field, w0_profile, broken_delta)


def test_output_preflight_rejects_collisions_existing_files_and_symlinks(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output_dir = tmp_path / "output"
    json_path, markdown_path, outputs_path = audit.prepare_output_paths(
        output_dir / "report.json",
        output_dir / "report.md",
        source_root,
    )
    assert json_path.parent == markdown_path.parent == outputs_path.parent
    json_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(audit.W0SensitivityError, match="already exists"):
        audit.prepare_output_paths(json_path, markdown_path, source_root)

    real = tmp_path / "real-output"
    real.mkdir()
    link = tmp_path / "linked-output"
    os.symlink(real, link)
    with pytest.raises(audit.W0SensitivityError, match="symlink component"):
        audit.prepare_output_paths(
            link / "report.json",
            link / "report.md",
            source_root,
        )

    with pytest.raises(audit.W0SensitivityError, match="inside source root"):
        audit.prepare_output_paths(
            source_root / "report.json",
            source_root / "report.md",
            source_root,
        )


def test_atomic_write_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    audit.atomic_write(path, "first\n")
    assert path.read_text(encoding="utf-8") == "first\n"
    with pytest.raises(audit.W0SensitivityError, match="already exists"):
        audit.atomic_write(path, "second\n")
