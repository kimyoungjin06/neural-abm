from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_field_taxonomy_reference.py"
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_field_taxonomy_reference_v0.yaml"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_openalex_field_taxonomy_reference_module", SCRIPT_PATH
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


def _markdown_table(markdown: str, header: str) -> list[list[str]]:
    lines = markdown.splitlines()
    start = lines.index(header)
    rows: list[list[str]] = []
    index = start + 2
    while index < len(lines) and lines[index].startswith("|"):
        rows.append(
            [cell.strip() for cell in lines[index].strip("|").split("|")]
        )
        index += 1
    return rows


def _taxonomy() -> Any:
    rows = {
        "domain": [
            {"id": "https://openalex.org/domains/1", "display_name": "Domain 1"}
        ],
        "field": [
            {
                "id": "https://openalex.org/fields/11",
                "display_name": "Field 11",
            }
        ],
        "subfield": [
            {
                "id": "https://openalex.org/subfields/101",
                "display_name": "Subfield 101",
            }
        ],
        "topic": [
            {
                "id": "https://openalex.org/T1001",
                "display_name": "Topic 1001",
                "domain": {"id": "https://openalex.org/domains/1"},
                "field": {"id": "https://openalex.org/fields/11"},
                "subfield": {"id": "https://openalex.org/subfields/101"},
            }
        ],
    }
    ids = {
        entity: pa.array([row["id"] for row in entity_rows], type=pa.string())
        for entity, entity_rows in rows.items()
    }
    names = {
        entity: np.asarray(
            [row["display_name"] for row in entity_rows], dtype=object
        )
        for entity, entity_rows in rows.items()
    }
    return audit.FrozenTaxonomy(
        public={
            "canonical_sha256": "a" * 64,
            "counts": {"domains": 1, "fields": 1, "subfields": 1, "topics": 1},
            "internal_graph_checks": {
                "unique_entity_ids": True,
                "parent_ids_resolve": True,
                "parent_names_match": True,
                "parent_chains_match": True,
                "declared_child_sets_match": True,
            },
            "file_commitments": [],
        },
        ids=ids,
        names=names,
        rows=rows,
        id_to_index={entity: {row["id"]: 0} for entity, (row,) in rows.items()},
        topic_parent_ids={
            "domain": np.asarray(["https://openalex.org/domains/1"], dtype=object),
            "field": np.asarray(["https://openalex.org/fields/11"], dtype=object),
            "subfield": np.asarray(
                ["https://openalex.org/subfields/101"], dtype=object
            ),
        },
    )


def _hierarchy(prefix: str, rows: list[dict[str, Any] | None]) -> dict[str, pa.Array]:
    suffixes = {
        "id": "string",
        "display_name": "string",
        "score": "double",
        "domain__id": "string",
        "domain__display_name": "string",
        "field__id": "string",
        "field__display_name": "string",
        "subfield__id": "string",
        "subfield__display_name": "string",
    }
    values: dict[str, pa.Array] = {}
    for suffix, kind in suffixes.items():
        raw = [None if row is None else row.get(suffix) for row in rows]
        values[f"{prefix}{suffix}"] = pa.array(
            raw, type=pa.float64() if kind == "double" else pa.string()
        )
    return values


def _valid_hierarchy(**updates: Any) -> dict[str, Any]:
    value = {
        "id": "https://openalex.org/T1001",
        "display_name": "Topic 1001",
        "score": 0.75,
        "domain__id": "https://openalex.org/domains/1",
        "domain__display_name": "Domain 1",
        "field__id": "https://openalex.org/fields/11",
        "field__display_name": "Field 11",
        "subfield__id": "https://openalex.org/subfields/101",
        "subfield__display_name": "Subfield 101",
    }
    value.update(updates)
    return value


def _write_table(
    root: Path,
    *,
    table_name: str,
    prefix: str,
    source_ids: list[str | None],
    hierarchy: list[dict[str, Any] | None],
) -> tuple[Path, Any]:
    path = root / table_name / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(source_ids, type=pa.string()),
                **_hierarchy(prefix, hierarchy),
            }
        ),
        path,
        row_group_size=2,
    )
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    columns = ("id", *(f"{prefix}{suffix}" for suffix in _config()["tables"]["projected_suffixes"]))
    projected_schema = {
        name: {
            "nullable": bool(schema.field(name).nullable),
            "type": str(schema.field(name).type),
        }
        for name in columns
    }
    info = path.stat()
    task = audit.FileTask(
        table_key="main" if table_name == "main" else "topics",
        table_name=table_name,
        hierarchy_prefix=prefix,
        relative_path=path.relative_to(root).as_posix(),
        bytes=info.st_size,
        mtime_ns=info.st_mtime_ns,
        footer_rows=parquet.metadata.num_rows,
        projected_columns=columns,
        projected_schema=projected_schema,
        projected_schema_sha256=hashlib.sha256(
            audit.canonical_json_bytes(projected_schema)
        ).hexdigest(),
    )
    return path, task


def _scan(
    root: Path,
    *,
    table_name: str,
    prefix: str,
    source_ids: list[str | None],
    hierarchy: list[dict[str, Any] | None],
) -> tuple[Path, Any, dict[str, Any]]:
    path, task = _write_table(
        root,
        table_name=table_name,
        prefix=prefix,
        source_ids=source_ids,
        hierarchy=hierarchy,
    )
    checkpoint = audit.scan_file(
        task,
        source_root=root,
        taxonomy=_taxonomy(),
        config=_config(),
        batch_size=2,
        task_fingerprint_sha256="f" * 64,
    )
    return path, task, checkpoint


def test_config_freezes_parent_taxonomy_and_claim_boundaries() -> None:
    config = audit.read_config(CONFIG_PATH)
    assert config["parent_inventory"]["manifest_sha256"] == (
        "6b5929a0155c033e818a816aad57095380a3907ab558c8d2cd58a89055f22b46"
    )
    parent = config["parent_work_source_audit"]
    assert parent["required_status"] == (
        "PASS_WORK_SOURCE_MEMBERSHIP_AND_PROFILE_AUDIT_WITH_FROZEN_QUARANTINE"
    )
    assert parent["result_fingerprint_sha256"] == (
        "8b7d0273a04eb7a231a9d8b466151c562ac80eac6557ff7444508cb174862242"
    )
    profile = config["compact_cross_field_profile"]
    assert profile["sha256"] == (
        "562c3ced372755beb2248998fa541e695c0d74736a6dcf986dfce3b3f207bfd8"
    )
    assert profile["result_fingerprint_sha256"] == (
        "d7dd61e46f2556fe7d97f9ac3d5d5e595bbaa10360bd0bddcf039867b8aa84fe"
    )
    assert len(config["taxonomy"]["files"]) == 7
    assert config["tables"]["expected_total_files"] == 1423
    assert config["tables"]["expected_total_footer_rows"] == 1_420_855_753
    assert config["success_contract"]["full_success_status"] == (
        audit.FULL_SUCCESS_STATUS
    )
    gates = config["success_contract"]["required_gate_status"]
    assert gates["duplicate_work_topic_edges"] == "PENDING"
    assert gates["primary_topic_child_inclusion"] == "PENDING"
    assert gates["publication_date_validity"] == "PENDING"
    assert gates["temporal_available_at"] == "PENDING"
    assert gates["candidate_eligibility"] == "PENDING"
    assert gates["g_minus_1_data_readiness"] == "PENDING"
    assert gates["empirical_g1_signature"] == "NOT_RUN"
    assert gates["pass_g1"] is False
    serialized = str(config)
    assert "/home/" not in serialized
    assert "/hdd/" not in serialized
    assert "/ssd/" not in serialized


def test_synthetic_main_scan_conserves_rows_and_freezes_w_sentinels(
    tmp_path: Path,
) -> None:
    _, _, checkpoint = _scan(
        tmp_path,
        table_name="main",
        prefix="primary_topic__",
        source_ids=["https://openalex.org/W1", audit.W_ZERO, audit.W_MINUS_1],
        hierarchy=[_valid_hierarchy(), None, _valid_hierarchy(score=2.0)],
    )
    metrics = checkpoint["payload"]["metrics"]
    assert metrics["raw_rows"] == 3
    assert metrics["quarantined_rows"] == 1
    assert metrics["retained_rows"] == 2
    assert metrics["source_w_minus_1_rows"] == 1
    assert metrics["source_w_zero_rows"] == 1
    assert metrics["hierarchy_all_null_rows"] == 1
    assert metrics["hierarchy_partial_rows"] == 0
    assert metrics["hierarchy_complete_rows"] == 1
    assert metrics["valid_reference_rows"] == 1
    assert metrics["complete_invalid_rows"] == 0
    assert metrics["main_field_null_rows"] == 1
    assert checkpoint["payload"]["main_field_counts"] == {"11": 1}
    assert len(checkpoint["payload"]["projected_payload_sha256"]) == 64
    assert checkpoint["payload_sha256"] == hashlib.sha256(
        audit.canonical_json_bytes(checkpoint["payload"])
    ).hexdigest()


def test_synthetic_fail_closed_reference_mutations_are_counted(
    tmp_path: Path,
) -> None:
    _, _, checkpoint = _scan(
        tmp_path,
        table_name="topics",
        prefix="topics__",
        source_ids=[
            "https://openalex.org/W1",
            "https://openalex.org/W2",
            "https://openalex.org/W3",
            "https://openalex.org/W4",
            "https://openalex.org/W5junk",
        ],
        hierarchy=[
            _valid_hierarchy(id="https://openalex.org/T9999"),
            _valid_hierarchy(display_name="Wrong topic name"),
            _valid_hierarchy(field__id="https://openalex.org/fields/12"),
            _valid_hierarchy(score=float("nan")),
            {"id": "https://openalex.org/T1001", "score": 2.0},
        ],
    )
    metrics = checkpoint["payload"]["metrics"]
    assert metrics["source_unexpected_malformed_rows"] == 1
    assert metrics["topic_id_unknown_rows"] == 1
    assert metrics["topic_name_mismatch_rows"] == 1
    assert metrics["parent_chain_mismatch_rows"] >= 1
    assert metrics["nonfinite_score_rows"] == 1
    assert metrics["out_of_range_score_rows"] == 1
    assert metrics["hierarchy_partial_rows"] == 1
    assert metrics["complete_invalid_rows"] == 4
    assert metrics["valid_reference_rows"] == 0
    assert metrics["raw_rows"] == (
        metrics["hierarchy_all_null_rows"]
        + metrics["hierarchy_partial_rows"]
        + metrics["hierarchy_complete_rows"]
    )


def test_resume_checkpoint_rejects_payload_and_source_identity_corruption(
    tmp_path: Path,
) -> None:
    path, task, checkpoint = _scan(
        tmp_path,
        table_name="topics",
        prefix="topics__",
        source_ids=["https://openalex.org/W1"],
        hierarchy=[_valid_hierarchy()],
    )
    valid = audit.validate_checkpoint(
        checkpoint,
        task,
        source_root=tmp_path,
        task_fingerprint_sha256="f" * 64,
        batch_size=2,
    )
    assert valid["relative_path"] == task.relative_path

    corrupted = copy.deepcopy(checkpoint)
    corrupted["payload"]["metrics"]["raw_rows"] += 1
    with pytest.raises(audit.TaxonomyReferenceAuditError, match="payload hash"):
        audit.validate_checkpoint(
            corrupted,
            task,
            source_root=tmp_path,
            task_fingerprint_sha256="f" * 64,
            batch_size=2,
        )

    path.touch()
    with pytest.raises(audit.TaxonomyReferenceAuditError, match="identity changed"):
        audit.validate_checkpoint(
            checkpoint,
            task,
            source_root=tmp_path,
            task_fingerprint_sha256="f" * 64,
            batch_size=2,
        )


def test_resume_rehash_rejects_same_size_restored_mtime_payload_change(
    tmp_path: Path,
) -> None:
    path, task, checkpoint = _scan(
        tmp_path / "source",
        table_name="topics",
        prefix="topics__",
        source_ids=["https://openalex.org/W1"],
        hierarchy=[_valid_hierarchy()],
    )
    alternate, _ = _write_table(
        tmp_path / "alternate",
        table_name="topics",
        prefix="topics__",
        source_ids=["https://openalex.org/W2"],
        hierarchy=[_valid_hierarchy()],
    )
    assert alternate.stat().st_size == task.bytes
    shutil.copyfile(alternate, path)
    os.utime(path, ns=(path.stat().st_atime_ns, task.mtime_ns))
    assert path.stat().st_size == task.bytes
    assert path.stat().st_mtime_ns == task.mtime_ns

    with pytest.raises(audit.TaxonomyReferenceAuditError, match="projected payload"):
        audit.validate_checkpoint(
            checkpoint,
            task,
            source_root=tmp_path / "source",
            task_fingerprint_sha256="f" * 64,
            batch_size=2,
        )


def test_resume_rejects_intermediate_source_symlink(tmp_path: Path) -> None:
    root = tmp_path / "source"
    _, task, checkpoint = _scan(
        root,
        table_name="topics",
        prefix="topics__",
        source_ids=["https://openalex.org/W1"],
        hierarchy=[_valid_hierarchy()],
    )
    (root / "topics").rename(root / "real-topics")
    (root / "topics").symlink_to(root / "real-topics", target_is_directory=True)
    with pytest.raises(audit.TaxonomyReferenceAuditError, match="path component"):
        audit.validate_checkpoint(
            checkpoint,
            task,
            source_root=root,
            task_fingerprint_sha256="f" * 64,
            batch_size=2,
        )


def test_pilot_report_cannot_promote_full_component(
    tmp_path: Path,
) -> None:
    _, main_task, main_checkpoint = _scan(
        tmp_path,
        table_name="main",
        prefix="primary_topic__",
        source_ids=["https://openalex.org/W1", audit.W_ZERO],
        hierarchy=[_valid_hierarchy(), None],
    )
    _, topics_task, topics_checkpoint = _scan(
        tmp_path,
        table_name="topics",
        prefix="topics__",
        source_ids=["https://openalex.org/W1"],
        hierarchy=[_valid_hierarchy()],
    )
    config = _config()
    context = {
        "config": config,
        "cross_field_matrix": {
            11: {
                "complete": 1,
                "partial": 0,
                "none_usable": 0,
                "zero_authorship_row": 0,
            },
            None: {
                "complete": 0,
                "partial": 0,
                "none_usable": 1,
                "zero_authorship_row": 0,
            },
        },
        "taxonomy": _taxonomy(),
        "hashes": {"config_sha256": "c" * 64},
    }
    plan = {
        "full_scope": False,
        "max_files_per_table": 1,
        "task_fingerprint_sha256": "f" * 64,
        "tables": {
            "main": {
                "name": "main",
                "selected_files": 1,
                "parent_files": 493,
                "selected_footer_rows": 2,
                "parent_footer_rows": 492_293_207,
                "selected_bytes": main_task.bytes,
                "parent_bytes": 150_062_449_219,
            },
            "topics": {
                "name": "topics",
                "selected_files": 1,
                "parent_files": 930,
                "selected_footer_rows": 1,
                "parent_footer_rows": 928_562_546,
                "selected_bytes": topics_task.bytes,
                "parent_bytes": 16_637_022_028,
            },
        },
    }
    report = audit.build_report(
        context,
        plan,
        [main_checkpoint, topics_checkpoint],
        elapsed_seconds=1.0,
    )
    assert report["status"] == audit.PILOT_STATUS
    assert report["status"] != audit.FULL_SUCCESS_STATUS
    assert report["gate_status"]["current_snapshot_taxonomy_reference_component"] == (
        "NOT_ESTABLISHED_PILOT"
    )
    assert report["field_status_margin_comparison"]["status"] == "NOT_RUN_PILOT"
    assert report["scope"]["scientific_effect_estimation_performed"] is False
    assert report["scope"]["gpu_used"] is False
    markdown = audit.render_markdown(report)
    assert "`PILOT_REFERENCE_ONLY`" in markdown
    gate_rows = _markdown_table(markdown, "| Gate | State |")
    assert len(gate_rows) == len(report["gate_status"])
    assert {row[0]: row[1] for row in gate_rows} == {
        key: str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in report["gate_status"].items()
    }
    pending_rows = _markdown_table(
        markdown, "| Pending item | Gate | Current state |"
    )
    assert len(pending_rows) == len(report["pending"])
    pending_by_item = {row[0]: row[1:] for row in pending_rows}
    assert pending_by_item["g_minus_1_data_readiness"] == [
        "g_minus_1_data_readiness",
        "PENDING",
    ]
    assert pending_by_item["empirical_g1_signature"] == [
        "empirical_g1_signature",
        "NOT_RUN",
    ]

    mutated = copy.deepcopy(report)
    mutated["execution"]["workload_class"] = "SYNTHETIC_CPU_IO"
    mutated["scope"]["gpu_used"] = True
    mutated["scope"]["scientific_effect_estimation_performed"] = True
    mutated["field_status_margin_comparison"][
        "parent_field_status_margins_equal_compact_profile"
    ] = False
    mutated["gate_status"]["empirical_g1_signature"] = "SYNTHETIC_NOT_RUN"
    mutated_markdown = audit.render_markdown(mutated)
    assert "Workload: `SYNTHETIC_CPU_IO`; GPU used: `true`" in mutated_markdown
    assert "Scientific effect estimation: `true`" in mutated_markdown
    assert "profile: `false`" in mutated_markdown
    assert [
        row
        for row in _markdown_table(mutated_markdown, "| Gate | State |")
        if row[0] == "empirical_g1_signature"
    ] == [["empirical_g1_signature", "SYNTHETIC_NOT_RUN"]]
    assert [
        row
        for row in _markdown_table(
            mutated_markdown, "| Pending item | Gate | Current state |"
        )
        if row[0] == "empirical_g1_signature"
    ] == [
        [
            "empirical_g1_signature",
            "empirical_g1_signature",
            "SYNTHETIC_NOT_RUN",
        ]
    ]


def test_read_config_rejects_scientific_gate_promotion(tmp_path: Path) -> None:
    config = _config()
    config["success_contract"]["required_gate_status"]["pass_g1"] = True
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(audit.TaxonomyReferenceAuditError, match="gate status"):
        audit.read_config(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("duplicate_work_topic_edges", "PASS"),
        ("primary_topic_child_inclusion", "PASS"),
        ("publication_date_validity", "PASS"),
        ("temporal_available_at", "PASS"),
        ("author_entity_validity", "PASS"),
        ("candidate_eligibility", "PASS"),
        ("g_minus_1_data_readiness", "PASS"),
        ("g0_measurement_freeze", "PASS"),
        ("empirical_g1_signature", "PASS"),
    ],
)
def test_read_config_rejects_pending_gate_promotion(
    tmp_path: Path, key: str, value: str
) -> None:
    config = _config()
    config["success_contract"]["required_gate_status"][key] = value
    path = tmp_path / f"mutated-{key}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(audit.TaxonomyReferenceAuditError, match="gate status"):
        audit.read_config(path)


@pytest.mark.parametrize(
    "mutation", ["claim", "pending", "score", "grammar", "sentinel", "resume"]
)
def test_read_config_rejects_boundary_contract_drift(
    tmp_path: Path, mutation: str
) -> None:
    config = _config()
    if mutation == "claim":
        config["claim_boundary"] += " This proves an effect."
    elif mutation == "pending":
        config["success_contract"]["pending_after_success"].pop()
    elif mutation == "score":
        config["reference_rules"]["score_range"]["maximum_inclusive"] = 2.0
    elif mutation == "grammar":
        config["reference_rules"]["id_grammar"]["topic"] = ".*"
    elif mutation == "sentinel":
        config["reference_rules"]["zero_sentinels"]["topic"] = (
            "https://openalex.org/T-1"
        )
    else:
        config["execution"]["resume_verification"] = "trust_size_and_mtime"
    path = tmp_path / f"mutated-{mutation}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(audit.TaxonomyReferenceAuditError, match="drift"):
        audit.read_config(path)
