from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_work_topic_relation.py"
CONFIG_PATH = REPO_ROOT / "experiments/configs/openalex_g1_work_topic_relation_v0.yaml"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_openalex_work_topic_relation_module", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_content_json(path: Path, value: Any) -> None:
    _write_json(path, value)
    path.chmod(0o600)


def _content_source_stat(path: Path) -> dict[str, int]:
    info = path.stat()
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode & 0o7777),
        "bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        row_group_size=2,
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
    )


def _write_gzip_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in rows:
                handle.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))


def _schema_fields(path: Path) -> dict[str, dict[str, Any]]:
    schema = pq.ParquetFile(path).schema_arrow
    return {
        field.name: {"nullable": bool(field.nullable), "type": str(field.type)}
        for field in schema
    }


def _inventory_file(path: Path, *, source_root: Path) -> dict[str, Any]:
    info = path.stat()
    parquet = pq.ParquetFile(path)
    return {
        "relative_path": path.relative_to(source_root).as_posix(),
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "footer_rows": parquet.metadata.num_rows,
        "schema_fields": _schema_fields(path),
    }


class SyntheticContract:
    def __init__(
        self,
        *,
        repo_root: Path,
        source_root: Path,
        taxonomy_root: Path,
        config_path: Path,
        main_path: Path,
        source_paths: list[Path],
        content_receipt: Path,
        config_sha256: str,
    ) -> None:
        self.repo_root = repo_root
        self.source_root = source_root
        self.taxonomy_root = taxonomy_root
        self.config_path = config_path
        self.main_path = main_path
        self.source_paths = source_paths
        self.content_receipt = content_receipt
        self.config_sha256 = config_sha256

    def run(self, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        audit.REPO_ROOT = self.repo_root
        audit.FROZEN_CONFIG_SHA256 = self.config_sha256
        options: dict[str, Any] = {
            "config_path": self.config_path,
            "source_root": self.source_root,
            "taxonomy_root": self.taxonomy_root,
            "content_freeze_receipt": self.content_receipt,
            "output_dir": output_dir,
            "batch_size": 2,
            "aggregation_chunk_rows": 3,
            "authorize_full_scope": True,
        }
        options.update(kwargs)
        return audit.run_audit(**options)


def _make_contract(
    base: Path,
    *,
    duplicate_score: float = 0.8,
    omit_primary_child: bool = False,
    overflow_work: bool = False,
    expected_main_w0: int = 1,
    quarantine_file_row: int = 4,
) -> SyntheticContract:
    repo_root = base / "repo"
    source_root = base / "source"
    taxonomy_root = base / "taxonomy"
    content_root = base / "content-freeze"
    repo_root.mkdir(parents=True)
    base.chmod(0o700)
    source_root.mkdir(parents=True)
    taxonomy_root.mkdir(parents=True)
    content_root.mkdir(mode=0o700)
    content_root.chmod(0o700)

    overflow_numeric = (2**64 - 1 >> 14) + 1
    fourth_work = (
        f"https://openalex.org/W{overflow_numeric}"
        if overflow_work
        else "https://openalex.org/W3"
    )
    main = pa.table(
        {
            "id": pa.array(
                [
                    "https://openalex.org/W0",
                    "https://openalex.org/W1",
                    "https://openalex.org/W2",
                    fourth_work,
                    "https://openalex.org/W-1",
                ],
                type=pa.string(),
            ),
            "primary_topic__id": pa.array(
                [
                    None,
                    "https://openalex.org/T10001",
                    "https://openalex.org/T10002",
                    None,
                    None,
                ],
                type=pa.string(),
            ),
        }
    )
    topic_a_ids = [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
        "https://openalex.org/W3",
    ]
    topic_a_topics = [
        "https://openalex.org/T10001",
        "https://openalex.org/T10002",
        "https://openalex.org/T10003",
    ]
    topic_a_scores = [0.8, 0.5, 0.1]
    if omit_primary_child:
        del topic_a_ids[1]
        del topic_a_topics[1]
        del topic_a_scores[1]
    topics_a = pa.table(
        {
            "id": pa.array(topic_a_ids, type=pa.string()),
            "topics__id": pa.array(topic_a_topics, type=pa.string()),
            "topics__score": pa.array(topic_a_scores, type=pa.float64()),
        }
    )
    topics_b = pa.table(
        {
            "id": pa.array(
                ["https://openalex.org/W1", "https://openalex.org/W2"],
                type=pa.string(),
            ),
            "topics__id": pa.array(
                [
                    "https://openalex.org/T10001",
                    "https://openalex.org/T10003",
                ],
                type=pa.string(),
            ),
            "topics__score": pa.array([duplicate_score, 0.2], type=pa.float64()),
        }
    )

    main_path = source_root / "synthetic_main/part-0.parquet"
    topic_a_path = source_root / "synthetic_topics/part-0.parquet"
    topic_b_path = source_root / "synthetic_topics/part-1.parquet"
    _write_parquet(main_path, main)
    _write_parquet(topic_a_path, topics_a)
    _write_parquet(topic_b_path, topics_b)

    main_files = [_inventory_file(main_path, source_root=source_root)]
    topic_files = [
        _inventory_file(topic_a_path, source_root=source_root),
        _inventory_file(topic_b_path, source_root=source_root),
    ]
    inventory_fingerprint = hashlib.sha256(b"synthetic-inventory").hexdigest()
    stability_fingerprint = hashlib.sha256(b"synthetic-stability").hexdigest()
    snapshot_date = "2026-03-30"
    source_id = "synthetic-openalex-work-topic"
    inventory = {
        "schema_version": "openalex_source_inventory.v1",
        "root": {
            "id": source_id,
            "snapshot_date": snapshot_date,
            "inventory_fingerprint_sha256": inventory_fingerprint,
            "stability_fingerprint_sha256": stability_fingerprint,
        },
        "tables": [
            {
                "name": "synthetic_main",
                "file_count": len(main_files),
                "footer_rows": sum(row["footer_rows"] for row in main_files),
                "bytes": sum(row["bytes"] for row in main_files),
                "files": main_files,
            },
            {
                "name": "synthetic_topics",
                "file_count": len(topic_files),
                "footer_rows": sum(row["footer_rows"] for row in topic_files),
                "bytes": sum(row["bytes"] for row in topic_files),
                "files": topic_files,
            },
        ],
    }

    config = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    config["source"].update(
        {
            "root_env": "SYNTHETIC_OPENALEX_ROOT",
            "root_id": source_id,
            "snapshot_date": snapshot_date,
            "lineage_id": "synthetic-lineage",
        }
    )
    config["execution"]["resource_preflight"].update(
        {
            "disk_safety_multiplier_milli": 1000,
            "disk_reserve_bytes": 0,
            "memory_safety_multiplier_milli": 1000,
            "memory_reserve_bytes": 0,
        }
    )
    grammar_exact = {
        "body_rows_scanned": 5,
        "quarantined_malformed_rows": 1,
        "unexpected_malformed_rows": 0,
        "grammar_valid_rows": 4,
        "grammar_valid_distinct_ids": 4,
        "grammar_valid_duplicate_row_excess": 0,
        "grammar_valid_min_numeric_id": 0,
        "grammar_valid_max_numeric_id": overflow_numeric if overflow_work else 3,
        "zero_id_rows_retained": 1,
    }
    config["parents"]["grammar_valid_main_recount"]["required_exact_counts"] = (
        grammar_exact
    )
    config["parents"]["inventory"].update(
        {
            "inventory_fingerprint_sha256": inventory_fingerprint,
            "stability_fingerprint_sha256": stability_fingerprint,
            "content_sha256_complete": False,
        }
    )

    taxonomy_path = taxonomy_root / "topics/part-0.gz"
    _write_gzip_json_lines(
        taxonomy_path,
        [
            {"id": "https://openalex.org/T10001"},
            {"id": "https://openalex.org/T10002"},
            {"id": "https://openalex.org/T10003"},
        ],
    )
    config["taxonomy"].update(
        {
            "root_env": "SYNTHETIC_OPENALEX_TAXONOMY_ROOT",
            "expected_topic_count": 3,
            "expected_min_numeric_id": 10001,
            "expected_max_numeric_id": 10003,
            "files": [
                {
                    "relative_path": "topics/part-0.gz",
                    "sha256": _sha256(taxonomy_path),
                    "expected_rows": 3,
                }
            ],
        }
    )
    config["tables"]["main"].update(
        {
            "name": "synthetic_main",
            "expected_files": 1,
            "expected_footer_rows": 5,
            "expected_bytes": main_files[0]["bytes"],
        }
    )
    config["tables"]["topics"].update(
        {
            "name": "synthetic_topics",
            "expected_files": 2,
            "expected_footer_rows": sum(row["footer_rows"] for row in topic_files),
            "expected_bytes": sum(row["bytes"] for row in topic_files),
        }
    )
    config["quarantine"]["w_minus_1"].update(
        {
            "relative_path": "synthetic_main/part-0.parquet",
            "row_group": 2,
            "file_row": quarantine_file_row,
        }
    )
    config["quarantine"]["w_zero"]["expected_main_rows"] = expected_main_w0

    parents_dir = repo_root / "parents"
    inventory_path = parents_dir / "inventory.json"
    grammar_path = parents_dir / "grammar.json"
    work_source_path = parents_dir / "work-source.json"
    taxonomy_report_path = parents_dir / "taxonomy-report.json"
    _write_json(inventory_path, inventory)

    grammar_contract = config["parents"]["grammar_valid_main_recount"]
    grammar = {
        "schema_version": grammar_contract["schema_version"],
        "status": grammar_contract["required_status"],
        "gate_status": grammar_contract["required_gate_states"],
        "exact_counts": grammar_exact,
    }
    _write_json(grammar_path, grammar)

    work_contract = config["parents"]["work_source_membership"]
    work_fingerprint = hashlib.sha256(b"synthetic-work-source").hexdigest()
    work_source = {
        "schema_version": work_contract["schema_version"],
        "status": work_contract["required_status"],
        "gate_status": work_contract["required_gate_states"],
        "provenance": {"result_fingerprint_sha256": work_fingerprint},
    }
    _write_json(work_source_path, work_source)

    taxonomy_contract = config["parents"]["taxonomy_reference"]
    taxonomy_fingerprint = hashlib.sha256(b"synthetic-taxonomy-report").hexdigest()
    taxonomy_canonical = hashlib.sha256(b"synthetic-taxonomy-canonical").hexdigest()
    taxonomy_report = {
        "schema_version": taxonomy_contract["schema_version"],
        "status": taxonomy_contract["required_status"],
        "gate_status": taxonomy_contract["required_gate_states"],
        "provenance": {"result_fingerprint_sha256": taxonomy_fingerprint},
        "taxonomy": {"canonical_sha256": taxonomy_canonical},
        "accounting": {
            "tables": {
                "main": {
                    "metrics": {
                        "hierarchy_all_null_rows": 2,
                        "hierarchy_complete_rows": 2,
                    }
                }
            }
        },
    }
    _write_json(taxonomy_report_path, taxonomy_report)

    parent_specs = {
        "inventory": inventory_path,
        "grammar_valid_main_recount": grammar_path,
        "work_source_membership": work_source_path,
        "taxonomy_reference": taxonomy_report_path,
    }
    for key, path in parent_specs.items():
        config["parents"][key].update(
            {
                "relative_path": path.relative_to(repo_root).as_posix(),
                "sha256": _sha256(path),
            }
        )
    config["parents"]["work_source_membership"]["result_fingerprint_sha256"] = (
        work_fingerprint
    )
    config["parents"]["taxonomy_reference"].update(
        {
            "result_fingerprint_sha256": taxonomy_fingerprint,
            "taxonomy_canonical_sha256": taxonomy_canonical,
        }
    )

    source_paths = [main_path, topic_a_path, topic_b_path]
    relation_bytes = sum(path.stat().st_size for path in source_paths)
    content_task_fingerprint = hashlib.sha256(b"synthetic-content-task").hexdigest()
    content_plan: dict[str, Any] = {
        "schema_version": "openalex_g1_content_freeze_plan.v1",
        "full_scope": True,
        "max_files": None,
        "parent_inventory": {
            "manifest_sha256": hashlib.sha256(
                b"synthetic-content-parent-manifest"
            ).hexdigest()
        },
        "declared_file_count": len(source_paths),
        "declared_bytes": relation_bytes,
        "selected_file_count": len(source_paths),
        "selected_bytes": relation_bytes,
        "task_fingerprint_sha256": content_task_fingerprint,
        "tasks": [
            {
                "table_name": (
                    "synthetic_main" if path == main_path else "synthetic_topics"
                ),
                "relative_path": path.relative_to(source_root).as_posix(),
                "source_stat": _content_source_stat(path),
            }
            for path in source_paths
        ],
    }
    content_plan["plan_sha256"] = audit.sha256_bytes(
        audit.canonical_json_bytes(content_plan)
    )
    content_plan_path = content_root / "audit_plan.json"
    content_report_path = content_root / "content_freeze_audit.json"
    content_markdown_path = content_root / "content_freeze_audit.md"
    _write_content_json(content_plan_path, content_plan)

    content_tree_sha256 = hashlib.sha256(b"synthetic-content-tree").hexdigest()
    content_checkpoint_tree = {
        "files": len(source_paths),
        "bytes": 777,
        "tree_sha256": hashlib.sha256(b"synthetic-checkpoint-tree").hexdigest(),
    }
    content_gates = {
        "current_content_baseline_component": "PASS",
        "historical_inventory_era_byte_identity": "UNVERIFIED",
        "prior_component_run_content_identity": "UNVERIFIED",
        "raw_source_id_validity": "BLOCKED",
        "w_zero_entity_validity": "UNRESOLVED",
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }
    content_claim = "synthetic current-content claim boundary"
    same_size_limitation = "synthetic same-size restored-mtime limitation"
    checkpoint_limitation = "synthetic unkeyed-checkpoint limitation"
    completion_contract = "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
    content_report: dict[str, Any] = {
        "schema_version": "openalex_g1_content_freeze_report.v1",
        "status": "PASS_CURRENT_CONTENT_BASELINE_COMPONENT",
        "scope": {
            "full_scope": True,
            "max_files": None,
            "declared_files": len(source_paths),
            "declared_bytes": relation_bytes,
            "selected_files": len(source_paths),
            "selected_bytes": relation_bytes,
            "gpu_used": False,
        },
        "parent_inventory": dict(content_plan["parent_inventory"]),
        "pass_current_content_baseline_component": True,
        "content_baseline": {
            "files": len(source_paths),
            "bytes": relation_bytes,
            "content_tree_sha256": content_tree_sha256,
            "historical_identity_established": False,
            "prior_component_run_identity_established": False,
        },
        "gate_status": content_gates,
        "checkpoint_evidence": {
            "checkpoint_files": len(source_paths),
            "checkpoint_bytes": content_checkpoint_tree["bytes"],
            "checkpoint_tree_sha256": content_checkpoint_tree["tree_sha256"],
        },
        "publication": {"completion_contract": completion_contract},
        "execution": {"path_redacted": True},
        "claim_boundary": content_claim,
        "same_size_restored_mtime_limitation": same_size_limitation,
        "checkpoint_trust_limitation": checkpoint_limitation,
    }
    content_result_core = {
        "schema_version": content_report["schema_version"],
        "status": content_report["status"],
        "full_scope": True,
        "max_files": None,
        "parent_manifest_sha256": content_plan["parent_inventory"][
            "manifest_sha256"
        ],
        "files": len(source_paths),
        "bytes": relation_bytes,
        "content_tree_sha256": content_tree_sha256,
        "gate_status": content_gates,
        "claim_boundary": content_claim,
        "same_size_restored_mtime_limitation": same_size_limitation,
        "checkpoint_trust_limitation": checkpoint_limitation,
        "publication_completion_contract": completion_contract,
    }
    content_result_fingerprint = audit.sha256_bytes(
        audit.canonical_json_bytes(content_result_core)
    )
    content_report["provenance"] = {
        "parent_manifest_sha256": content_plan["parent_inventory"][
            "manifest_sha256"
        ],
        "task_fingerprint_sha256": content_task_fingerprint,
        "plan_sha256": content_plan["plan_sha256"],
        "result_fingerprint_sha256": content_result_fingerprint,
        "path_redacted": True,
    }
    _write_content_json(content_report_path, content_report)
    content_markdown_path.write_text(
        "synthetic content freeze\n", encoding="utf-8"
    )
    content_markdown_path.chmod(0o600)
    content_artifacts = [
        {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (content_plan_path, content_report_path, content_markdown_path)
    ]
    content_receipt = content_root / "content_freeze_audit.outputs.json"
    _write_content_json(
        content_receipt,
        {
            "schema_version": "openalex_g1_content_freeze_outputs.v1",
            "status": "PASS_CURRENT_CONTENT_BASELINE_COMPONENT",
            "task_fingerprint_sha256": content_task_fingerprint,
            "plan_sha256": content_plan["plan_sha256"],
            "result_fingerprint_sha256": content_result_fingerprint,
            "content_tree_sha256": content_tree_sha256,
            "checkpoint_tree": content_checkpoint_tree,
            "artifacts": content_artifacts,
            "completion_contract": completion_contract,
            "run_state_bound_separately": True,
            "path_redacted": True,
        },
    )
    _write_content_json(
        content_root / "run_state.json",
        {
            "schema_version": "openalex_g1_content_freeze_state.v1",
            "status": "COMPLETE",
            "plan_sha256": content_plan["plan_sha256"],
            "task_fingerprint_sha256": content_task_fingerprint,
            "completed_files": len(source_paths),
            "completed_bytes": relation_bytes,
            "total_files": len(source_paths),
            "total_bytes": relation_bytes,
            "path_redacted": True,
        },
    )
    content_parent = config["parents"]["current_content_baseline"]
    content_parent.update(
        {
            "required_plan_sha256": content_plan["plan_sha256"],
            "required_task_fingerprint_sha256": content_task_fingerprint,
            "required_file_count": len(source_paths),
            "required_bytes": relation_bytes,
        }
    )
    content_parent["relation_source_subset"].update(
        {
            "required_files": len(source_paths),
            "required_bytes": relation_bytes,
        }
    )

    config_path = repo_root / "synthetic-work-topic.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_sha256 = _sha256(config_path)
    return SyntheticContract(
        repo_root=repo_root,
        source_root=source_root,
        taxonomy_root=taxonomy_root,
        config_path=config_path,
        main_path=main_path,
        source_paths=source_paths,
        content_receipt=content_receipt,
        config_sha256=config_sha256,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _republish_content_plan(
    contract: SyntheticContract, plan: dict[str, Any]
) -> None:
    plan_core = dict(plan)
    plan_core.pop("plan_sha256", None)
    plan["plan_sha256"] = audit.sha256_bytes(audit.canonical_json_bytes(plan_core))
    plan_path = contract.content_receipt.parent / "audit_plan.json"
    _write_content_json(plan_path, plan)
    report_path = contract.content_receipt.parent / "content_freeze_audit.json"
    report = _read_json(report_path)
    report["provenance"]["plan_sha256"] = plan["plan_sha256"]
    _write_content_json(report_path, report)
    receipt = _read_json(contract.content_receipt)
    for artifact in receipt["artifacts"]:
        if artifact["relative_path"] == plan_path.name:
            artifact["bytes"] = plan_path.stat().st_size
            artifact["sha256"] = _sha256(plan_path)
        elif artifact["relative_path"] == report_path.name:
            artifact["bytes"] = report_path.stat().st_size
            artifact["sha256"] = _sha256(report_path)
    receipt["plan_sha256"] = plan["plan_sha256"]
    _write_content_json(contract.content_receipt, receipt)
    state_path = contract.content_receipt.parent / "run_state.json"
    state = _read_json(state_path)
    state["plan_sha256"] = plan["plan_sha256"]
    _write_content_json(state_path, state)
    config = yaml.safe_load(contract.config_path.read_text(encoding="utf-8"))
    config["parents"]["current_content_baseline"]["required_plan_sha256"] = plan[
        "plan_sha256"
    ]
    contract.config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    contract.config_path.chmod(0o600)
    contract.config_sha256 = _sha256(contract.config_path)


def _republish_content_report(
    contract: SyntheticContract, report: dict[str, Any]
) -> None:
    report_path = contract.content_receipt.parent / "content_freeze_audit.json"
    _write_content_json(report_path, report)
    receipt = _read_json(contract.content_receipt)
    for artifact in receipt["artifacts"]:
        if artifact["relative_path"] == report_path.name:
            artifact["bytes"] = report_path.stat().st_size
            artifact["sha256"] = _sha256(report_path)
    _write_content_json(contract.content_receipt, receipt)


def _mutate_frozen_config(config: dict[str, Any], case: str) -> None:
    if case == "source":
        config["source"]["root_id"] = "drifted-root"
    elif case == "parent_gate":
        config["parents"]["grammar_valid_main_recount"]["required_gate_states"][
            "pass_g1"
        ] = True
    elif case == "parent_identity":
        config["parents"]["work_source_membership"]["required_gate_states"][
            "parent_content_identity"
        ] = "VERIFIED"
    elif case == "taxonomy":
        config["taxonomy"]["expected_topic_count"] += 1
    elif case == "tables":
        config["tables"]["topics"]["projected_columns"].append("unsafe_column")
    elif case == "codec":
        config["identifier_codec"]["topic_bits"] = 13
    elif case == "quarantine":
        config["quarantine"]["policy"] = "accept_any_malformed_row"
    elif case == "score_canonicalization":
        config["relation_policy"]["score_canonicalization"] = "round_to_2dp"
    elif case == "score_minimum":
        config["relation_policy"]["score_minimum_inclusive"] = -1.0
    elif case == "execution":
        config["execution"]["gpu_used"] = True
    elif case == "success_gate":
        config["success_contract"]["required_full_parent_boundary"]["pass_g1"] = True
    elif case == "success_status":
        config["success_contract"]["full_success_status"] = "PASS_G1"
    elif case == "claim":
        config["claim_boundary"] = "All scientific claims are established."
    else:
        raise AssertionError(f"unknown mutation case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "source",
        "parent_gate",
        "parent_identity",
        "taxonomy",
        "tables",
        "codec",
        "quarantine",
        "score_canonicalization",
        "score_minimum",
        "execution",
        "success_gate",
        "success_status",
        "claim",
    ],
)
def test_any_frozen_v0_contract_mutation_is_rejected_before_output(
    tmp_path: Path, case: str
) -> None:
    contract = _make_contract(tmp_path / case)
    config = yaml.safe_load(contract.config_path.read_text(encoding="utf-8"))
    _mutate_frozen_config(config, case)
    contract.config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    output = tmp_path / f"output-{case}"

    with pytest.raises(
        audit.RelationAuditError, match="differs from the frozen v0 contract"
    ):
        contract.run(output)
    assert not output.exists()


def test_config_path_swap_after_verified_snapshot_cannot_change_full_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    safe_bytes = contract.config_path.read_bytes()
    safe_claim = yaml.safe_load(safe_bytes)["claim_boundary"]
    original = audit._read_config_snapshot
    swapped = False

    def swap_path_after_single_fd_snapshot(
        path: Path,
    ) -> tuple[dict[str, Any], str, int]:
        nonlocal swapped
        snapshot = original(path)
        unsafe = yaml.safe_load(safe_bytes)
        unsafe["success_contract"]["required_full_parent_boundary"]["pass_g1"] = True
        unsafe["claim_boundary"] = "UNSAFE OVERCLAIM AFTER PATH SWAP"
        replacement = path.with_name("unsafe-replacement.yaml")
        replacement.write_text(
            yaml.safe_dump(unsafe, sort_keys=False), encoding="utf-8"
        )
        os.replace(replacement, path)
        swapped = True
        return snapshot

    monkeypatch.setattr(
        audit, "_read_config_snapshot", swap_path_after_single_fd_snapshot
    )

    report = contract.run(output)

    assert swapped is True
    assert report["status"] == audit.FULL_STATUS
    assert report["gate_status"]["pass_g1"] is False
    assert report["claim_boundary"] == safe_claim
    assert report["claim_boundary"] != "UNSAFE OVERCLAIM AFTER PATH SWAP"
    plan = _read_json(output / "audit_plan.json")
    assert plan["config_sha256"] == contract.config_sha256
    assert plan["config_snapshot_bytes"] == len(safe_bytes)
    assert report["provenance"]["config_sha256"] == contract.config_sha256
    assert report["provenance"]["config_snapshot_bytes"] == len(safe_bytes)


def test_full_synthetic_relation_promotes_only_same_score_dedup(
    tmp_path: Path,
) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"

    report = contract.run(output)

    assert report["status"] == audit.FULL_STATUS
    assert report["scope"]["full_scope"] is True
    assert report["gate_status"]["current_content_baseline_component"] == (
        "PASS_BOUND_TO_EXACT_RELATION_SOURCE_TASK_SET"
    )
    assert report["gate_status"]["historical_content_identity"] == "UNVERIFIED"
    content = report["content_baseline_parent"]
    assert content["status"] == "PASS_BOUND_CURRENT_CONTENT_BASELINE_RECEIPT"
    assert content["relation_source_rich_stat_bridge"] == {
        "status": "PASS_EXACT_RELATION_TASK_SET_AND_RICH_STAT_EQUALITY",
        "task_set_policy": "EXACT_FULL_RELATION_TASK_SET",
        "files": 3,
        "bytes": sum(path.stat().st_size for path in contract.source_paths),
        "table_file_counts": {"main": 1, "topics": 2},
        "table_bytes": {
            "main": contract.main_path.stat().st_size,
            "topics": sum(
                path.stat().st_size
                for path in contract.source_paths
                if path != contract.main_path
            ),
        },
        "tree_sha256": content["relation_source_rich_stat_bridge"]["tree_sha256"],
        "source_stat_fields": [
            "device",
            "inode",
            "mode",
            "bytes",
            "mtime_ns",
            "ctime_ns",
        ],
        "mode_representation": (
            "permission_bits_matching_content_freeze_source_stat"
        ),
    }
    plan = _read_json(output / "audit_plan.json")
    assert plan["current_content_baseline_parent"] == content
    assert plan["task_fingerprint_sha256"] == audit._plan_fingerprint(plan)
    edge = report["accounting"]["edge_groups"]["grammar_valid_including_w0"]
    assert edge == {
        "topic_raw_rows": 5,
        "unique_work_topic_edges": 4,
        "duplicate_row_excess": 1,
        "duplicate_edge_groups": 1,
        "score_conflict_edge_groups": 0,
        "main_primary_nonnull_rows": 2,
        "primary_child_present_rows": 2,
        "primary_child_missing_rows": 0,
        "main_primary_null_rows": 2,
    }
    assert report["gate_status"]["raw_work_topic_edge_uniqueness"] == (
        "FAIL_DUPLICATES_PRESENT"
    )
    assert report["gate_status"]["frozen_exact_score_dedup_view"] == "PASS"
    assert report["gate_status"]["primary_topic_child_inclusion"] == "PASS"
    assert report["deduplicated_relation_artifact"] == {
        "status": "PROMOTED_FROZEN_EXACT_SCORE_SET_VIEW",
        "relative_path": "work_topic_edges.deduplicated.sorted.bin",
        "rows": 4,
        "bytes": 4 * audit.DEDUP_DTYPE.itemsize,
        "sha256": _sha256(output / "work_topic_edges.deduplicated.sorted.bin"),
    }
    checkpoints = sorted((output / "checkpoints").rglob("*.json"))
    assert len(checkpoints) == 3
    main_checkpoint = next(
        _read_json(path)
        for path in checkpoints
        if _read_json(path)["table_key"] == "main"
    )
    assert main_checkpoint["metrics"]["quarantined_w_minus_1_rows"] == 1
    assert main_checkpoint["quarantine_observations"] == [
        {
            "table_key": "main",
            "relative_path": "synthetic_main/part-0.parquet",
            "row_group": 2,
            "file_row": 4,
            "raw_value": "https://openalex.org/W-1",
        }
    ]
    state = _read_json(output / "run_state.json")
    receipt = _read_json(output / "work_topic_relation_audit.outputs.json")
    assert state["status"] == "COMPLETE_PENDING_RECEIPT"
    assert state["complete_only_if_receipt_exists"] is True
    assert receipt["status"] == audit.FULL_STATUS
    checkpoint_artifacts = [
        key for key in receipt["artifacts"] if key.startswith("checkpoints/")
    ]
    assert len(checkpoint_artifacts) == 3


def test_w0_and_null_primary_are_structural_not_entity_claims(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case")

    report = contract.run(tmp_path / "output")

    sensitivity = report["w0_structural_sensitivity"]
    assert sensitivity["ledger_conserved"] is True
    assert sensitivity["w_zero_entity_validity"] == "UNRESOLVED"
    assert sensitivity["scientific_or_panel_sensitivity_established"] is False
    assert sensitivity["w0_work_group"] == {
        "main_work_rows": 1,
        "topic_rows": 0,
        "null_primary_rows": 1,
        "nonnull_primary_rows": 0,
    }
    assert (
        sensitivity["grammar_valid_including_w0_minus_positive_work_ids_only"][
            "edge_metrics"
        ]["main_primary_null_rows"]
        == 1
    )
    assert report["gate_status"]["w_zero_entity_validity"] == "UNRESOLVED"
    assert report["gate_status"]["parent_content_identity"] == "UNVERIFIED"
    assert report["gate_status"]["g_minus_1_data_readiness"] == "PENDING"
    assert report["gate_status"]["g0_measurement_freeze"] == "PENDING"
    assert report["gate_status"]["empirical_g1_signature"] == "NOT_RUN"
    assert report["gate_status"]["pass_g1"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_main_w0": 2},
        {"quarantine_file_row": 3},
    ],
)
def test_w0_component_pass_requires_exact_w0_and_quarantine_prerequisites(
    tmp_path: Path, kwargs: dict[str, int]
) -> None:
    contract = _make_contract(tmp_path / "case", **kwargs)

    report = contract.run(tmp_path / "output")

    assert report["status"] == audit.BLOCKED_STATUS
    assert report["w0_structural_sensitivity"]["ledger_conserved"] is True
    assert report["w0_structural_sensitivity"]["full_prerequisites_pass"] is False
    assert report["w0_structural_sensitivity"]["status"] == "BLOCKED"
    assert (
        report["gate_status"]["w_zero_positive_id_structural_sensitivity"] == "BLOCKED"
    )


def test_cross_file_score_conflict_blocks_without_materializing_choice(
    tmp_path: Path,
) -> None:
    contract = _make_contract(tmp_path / "case", duplicate_score=0.7)
    output = tmp_path / "output"

    report = contract.run(output)

    assert report["status"] == audit.BLOCKED_STATUS
    edge = report["accounting"]["edge_groups"]["grammar_valid_including_w0"]
    assert edge["duplicate_row_excess"] == 1
    assert edge["score_conflict_edge_groups"] == 1
    assert report["gate_status"]["frozen_exact_score_dedup_view"] == (
        "BLOCKED_SCORE_CONFLICT"
    )
    assert report["deduplicated_relation_artifact"] == {
        "status": "NOT_MATERIALIZED_SCORE_CONFLICT",
        "relative_path": None,
        "rows": None,
        "bytes": None,
        "sha256": None,
    }
    assert not (output / "work_topic_edges.deduplicated.sorted.bin").exists()


def test_missing_primary_child_blocks_component(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case", omit_primary_child=True)

    report = contract.run(tmp_path / "output")

    assert report["status"] == audit.BLOCKED_STATUS
    edge = report["accounting"]["edge_groups"]["grammar_valid_including_w0"]
    assert edge["main_primary_nonnull_rows"] == 2
    assert edge["primary_child_present_rows"] == 1
    assert edge["primary_child_missing_rows"] == 1
    assert report["gate_status"]["primary_topic_child_inclusion"] == (
        "BLOCKED_MISSING_CHILD"
    )


def test_packed_codec_rejects_work_shift_overflow(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case", overflow_work=True)

    with pytest.raises(audit.RelationAuditError, match="overflows the 14-bit"):
        contract.run(tmp_path / "output")


def test_score_codec_normalizes_negative_zero() -> None:
    bits = audit._canonical_score_bits(pa.array([-0.0, 0.0], type=pa.float64()))
    assert bits.tolist() == [0, 0]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("checkpoint", "checkpoint payload hash mismatch"),
        ("spool", "event spool segment hash mismatch"),
        (
            "source",
            (
                "current source rich identity differs from content-freeze plan|"
                "resume audit plan mismatch|current projected payload differs from checkpoint"
            ),
        ),
    ],
)
def test_resume_rehash_rejects_checkpoint_source_or_spool_mutation(
    tmp_path: Path, mode: str, expected: str
) -> None:
    contract = _make_contract(tmp_path / mode)
    output = tmp_path / f"output-{mode}"
    paused = contract.run(output, pause_after_new_files=1)
    assert paused["status"] == "PAUSED"
    assert paused["completed_files"] == 1

    checkpoint_path = next((output / "checkpoints/main").glob("*.json"))
    if mode == "checkpoint":
        checkpoint = _read_json(checkpoint_path)
        checkpoint["metrics"]["raw_rows"] += 1
        _write_json(checkpoint_path, checkpoint)
    elif mode == "spool":
        spool = output / "work_topic_relation.events.raw.bin"
        payload = bytearray(spool.read_bytes())
        payload[0] ^= 1
        spool.write_bytes(payload)
    else:
        path = contract.main_path
        before = path.stat()
        table = pq.read_table(path)
        ids = table["id"].to_pylist()
        ids[3] = "https://openalex.org/W4"
        mutated = table.set_column(
            table.schema.get_field_index("id"),
            "id",
            pa.array(ids, type=pa.string()),
        )
        _write_parquet(path, mutated)
        assert path.stat().st_size == before.st_size
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(audit.RelationAuditError, match=expected):
        contract.run(output, resume=True)


def test_pause_then_verified_resume_completes(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"

    paused = contract.run(output, pause_after_new_files=1)
    report = contract.run(output, resume=True)

    assert paused["status"] == "PAUSED"
    assert report["status"] == audit.FULL_STATUS
    assert report["checkpoint_evidence"]["checkpoint_count"] == 3
    assert report["checkpoint_evidence"]["resume_verification"] == (
        "REHASH_CURRENT_PROJECTED_PAYLOAD_AND_SPOOL_SEGMENT"
    )


@pytest.mark.parametrize("resource", ["disk", "memory"])
def test_full_resource_preflight_blocks_before_data_page_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource: str
) -> None:
    contract = _make_contract(tmp_path / resource)
    monkeypatch.setattr(
        audit,
        "scan_task",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("data-page scan must not start")
        ),
    )
    if resource == "disk":
        monkeypatch.setattr(audit, "_available_disk_bytes", lambda path: 0)
        monkeypatch.setattr(audit, "_available_memory_bytes", lambda: 2**63)
    else:
        monkeypatch.setattr(audit, "_available_disk_bytes", lambda path: 2**63)
        monkeypatch.setattr(audit, "_available_memory_bytes", lambda: 0)
    output = tmp_path / f"output-{resource}"

    with pytest.raises(
        audit.RelationAuditError,
        match=f"insufficient {'free disk' if resource == 'disk' else 'available memory'}",
    ):
        contract.run(output)
    assert not (output / "audit_plan.json").exists()
    assert not (output / "work_topic_relation.events.raw.bin").exists()


def test_plan_records_deterministic_resource_upper_bounds(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"

    contract.run(output)

    estimate = _read_json(output / "audit_plan.json")["resource_estimate"]
    assert estimate == {
        "selected_main_footer_rows": 5,
        "selected_topic_footer_rows": 5,
        "event_row_upper_bound": 10,
        "raw_spool_bytes": 240,
        "sorted_event_bytes": 240,
        "deduplicated_edge_bytes_upper_bound": 80,
        "artifact_bytes_upper_bound": 560,
        "peak_in_memory_sort_array_bytes": 240,
        "disk_safety_multiplier_milli": 1000,
        "disk_reserve_bytes": 0,
        "required_free_disk_bytes": 560,
        "memory_safety_multiplier_milli": 1000,
        "memory_reserve_bytes": 0,
        "required_available_memory_bytes": 240,
        "full_scope_only": True,
        "pilot_promotion_allowed": False,
    }


def test_outputs_receipt_failure_removes_pass_report_and_complete_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    original = audit.atomic_write_json

    def fail_receipt(path: Path, value: Any) -> None:
        if path.name == "work_topic_relation_audit.outputs.json":
            raise audit.RelationAuditError("injected receipt failure")
        original(path, value)

    monkeypatch.setattr(audit, "atomic_write_json", fail_receipt)

    with pytest.raises(audit.RelationAuditError, match="injected receipt failure"):
        contract.run(output)

    assert not (output / "work_topic_relation_audit.json").exists()
    assert not (output / "work_topic_relation_audit.md").exists()
    assert not (output / "work_topic_relation_audit.outputs.json").exists()
    state = _read_json(output / "run_state.json")
    assert state["status"] == "FINALIZATION_FAILED_NO_VALID_RECEIPT"
    assert state["completion_receipt_invalidation_attempted_first"] is True
    assert "result_fingerprint_sha256" not in state
    assert state["operator_paths_redacted"] is True


def test_finalization_cleanup_reports_receipt_and_state_failures_without_swallowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    original_unlink = audit._unlink_trusted_output
    original_write = audit.atomic_write_json

    def fail_final_validation(*args: Any, **kwargs: Any) -> None:
        raise audit.RelationAuditError("injected final validation failure")

    def fail_receipt_unlink(path: Path, *, missing_ok: bool = True) -> None:
        if path.name == "work_topic_relation_audit.outputs.json":
            raise audit.RelationAuditError("injected receipt unlink failure")
        original_unlink(path, missing_ok=missing_ok)

    def fail_failure_state(path: Path, value: Any) -> None:
        if (
            path.name == "run_state.json"
            and isinstance(value, dict)
            and value.get("status") == "FINALIZATION_FAILED_NO_VALID_RECEIPT"
        ):
            raise audit.RelationAuditError("injected failure-state write failure")
        original_write(path, value)

    monkeypatch.setattr(
        audit, "validate_final_publication_from_disk", fail_final_validation
    )
    monkeypatch.setattr(audit, "_unlink_trusted_output", fail_receipt_unlink)
    monkeypatch.setattr(audit, "atomic_write_json", fail_failure_state)

    with pytest.raises(
        audit.RelationAuditError,
        match=(
            "finalization cleanup failed after completion invalidation attempts: "
            "completion_receipt,failure_state"
        ),
    ):
        contract.run(output)

    # Both primary invalidation mechanisms failed, but descriptive artifacts
    # were still removed, so the surviving receipt cannot validate as complete.
    assert (output / "work_topic_relation_audit.outputs.json").exists()
    assert not (output / "work_topic_relation_audit.json").exists()
    assert not (output / "work_topic_relation_audit.md").exists()
    assert _read_json(output / "run_state.json")["status"] == (
        "COMPLETE_PENDING_RECEIPT"
    )


@pytest.mark.parametrize("failure", ["receipt_unlink", "failure_state"])
def test_single_cleanup_failure_still_leaves_no_valid_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    contract = _make_contract(tmp_path / failure)
    output = tmp_path / f"output-{failure}"
    original_unlink = audit._unlink_trusted_output
    original_write = audit.atomic_write_json

    def fail_final_validation(*args: Any, **kwargs: Any) -> None:
        raise audit.RelationAuditError("injected final validation failure")

    def maybe_fail_unlink(path: Path, *, missing_ok: bool = True) -> None:
        if (
            failure == "receipt_unlink"
            and path.name == "work_topic_relation_audit.outputs.json"
        ):
            raise audit.RelationAuditError("injected receipt unlink failure")
        original_unlink(path, missing_ok=missing_ok)

    def maybe_fail_state(path: Path, value: Any) -> None:
        if (
            failure == "failure_state"
            and path.name == "run_state.json"
            and isinstance(value, dict)
            and value.get("status") == "FINALIZATION_FAILED_NO_VALID_RECEIPT"
        ):
            raise audit.RelationAuditError("injected failure-state write failure")
        original_write(path, value)

    monkeypatch.setattr(
        audit, "validate_final_publication_from_disk", fail_final_validation
    )
    monkeypatch.setattr(audit, "_unlink_trusted_output", maybe_fail_unlink)
    monkeypatch.setattr(audit, "atomic_write_json", maybe_fail_state)
    expected = "completion_receipt" if failure == "receipt_unlink" else "failure_state"
    with pytest.raises(audit.RelationAuditError, match=expected):
        contract.run(output)

    if failure == "receipt_unlink":
        assert (output / "work_topic_relation_audit.outputs.json").exists()
        state = _read_json(output / "run_state.json")
        assert state["status"] == "FINALIZATION_FAILED_NO_VALID_RECEIPT"
        assert state["completion_receipt_invalidation_attempted_first"] is True
    else:
        assert not (output / "work_topic_relation_audit.outputs.json").exists()
        assert _read_json(output / "run_state.json")["status"] == (
            "COMPLETE_PENDING_RECEIPT"
        )
    assert not (output / "work_topic_relation_audit.json").exists()
    assert not (output / "work_topic_relation_audit.md").exists()


@pytest.mark.parametrize(
    "target_name",
    [
        "work_topic_relation_audit.json",
        "work_topic_relation_audit.outputs.json",
    ],
)
def test_post_write_report_or_receipt_corruption_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_name: str
) -> None:
    contract = _make_contract(tmp_path / target_name)
    output = tmp_path / f"output-{target_name}"
    original = audit.atomic_write_json
    corrupted = False

    def corrupt_after_success(path: Path, value: Any) -> None:
        nonlocal corrupted
        original(path, value)
        if path.name == target_name and not corrupted:
            path.write_text("{}\n", encoding="utf-8")
            corrupted = True

    monkeypatch.setattr(audit, "atomic_write_json", corrupt_after_success)

    with pytest.raises(
        audit.RelationAuditError,
        match="final report schema mismatch|final receipt schema mismatch",
    ):
        contract.run(output)

    assert corrupted is True
    assert not (output / "work_topic_relation_audit.json").exists()
    assert not (output / "work_topic_relation_audit.md").exists()
    assert not (output / "work_topic_relation_audit.outputs.json").exists()
    state = _read_json(output / "run_state.json")
    assert state["status"] == "FINALIZATION_FAILED_NO_VALID_RECEIPT"
    assert state["completion_receipt_invalidation_attempted_first"] is True
    assert "result_fingerprint_sha256" not in state


def test_final_disk_checkpoint_reparse_rejects_post_write_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    original = audit.atomic_write_json
    corrupted = False

    def corrupt_first_checkpoint(path: Path, value: Any) -> None:
        nonlocal corrupted
        original(path, value)
        if not corrupted and "checkpoints" in path.parts:
            path.write_text("{}\n", encoding="utf-8")
            corrupted = True

    monkeypatch.setattr(audit, "atomic_write_json", corrupt_first_checkpoint)

    with pytest.raises(audit.RelationAuditError, match="checkpoint schema mismatch"):
        contract.run(output)

    assert corrupted is True
    assert not (output / "work_topic_relation_audit.outputs.json").exists()


def test_final_rich_stat_tree_rejects_same_size_restored_mtime_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    original = audit.prepare_sorted_events

    def mutate_after_sort(*args: Any, **kwargs: Any) -> dict[str, Any]:
        manifest = original(*args, **kwargs)
        path = contract.main_path
        before = path.stat()
        table = pq.read_table(path)
        ids = table["id"].to_pylist()
        ids[3] = "https://openalex.org/W4"
        mutated = table.set_column(
            table.schema.get_field_index("id"),
            "id",
            pa.array(ids, type=pa.string()),
        )
        _write_parquet(path, mutated)
        assert path.stat().st_size == before.st_size
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return manifest

    monkeypatch.setattr(audit, "prepare_sorted_events", mutate_after_sort)

    with pytest.raises(
        audit.RelationAuditError,
        match="source rich identity differs from frozen plan|final source rich-stat tree",
    ):
        contract.run(output)
    assert not (output / "work_topic_relation_audit.outputs.json").exists()


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("receipt_whitespace", "exact content-freeze published JSON bytes"),
        ("duplicate_artifact", "duplicate artifacts"),
        ("report_fingerprint", "content-freeze report result fingerprint mismatch"),
    ],
)
def test_content_freeze_parent_rejects_noncanonical_or_cross_unbound_publication(
    tmp_path: Path, target: str, expected: str
) -> None:
    contract = _make_contract(tmp_path / target)
    if target == "receipt_whitespace":
        contract.content_receipt.write_bytes(
            contract.content_receipt.read_bytes() + b" "
        )
    elif target == "duplicate_artifact":
        receipt = _read_json(contract.content_receipt)
        receipt["artifacts"].append(copy.deepcopy(receipt["artifacts"][0]))
        _write_content_json(contract.content_receipt, receipt)
    else:
        report_path = contract.content_receipt.parent / "content_freeze_audit.json"
        report = _read_json(report_path)
        report["claim_boundary"] += " tampered"
        _write_content_json(report_path, report)
        receipt = _read_json(contract.content_receipt)
        for artifact in receipt["artifacts"]:
            if artifact["relative_path"] == report_path.name:
                artifact["bytes"] = report_path.stat().st_size
                artifact["sha256"] = _sha256(report_path)
        _write_content_json(contract.content_receipt, receipt)

    with pytest.raises(audit.RelationAuditError, match=expected):
        contract.run(tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        (
            "parent_manifest_sha256",
            "content-freeze report parent manifest provenance mismatch",
        ),
        (
            "task_fingerprint_sha256",
            "content-freeze report task provenance mismatch",
        ),
        ("plan_sha256", "content-freeze report plan provenance mismatch"),
    ],
)
def test_content_report_upstream_provenance_must_equal_anchored_plan(
    tmp_path: Path, field: str, expected: str
) -> None:
    contract = _make_contract(tmp_path / field)
    report_path = contract.content_receipt.parent / "content_freeze_audit.json"
    report = _read_json(report_path)
    report["provenance"][field] = hashlib.sha256(
        f"drift-{field}".encode("utf-8")
    ).hexdigest()
    _republish_content_report(contract, report)

    with pytest.raises(audit.RelationAuditError, match=expected):
        contract.run(tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("receipt_zero", "checkpoint tree bytes must be positive"),
        ("report_zero", "report checkpoint bytes must be positive"),
        ("byte_mismatch", "report/receipt checkpoint byte count mismatch"),
    ],
)
def test_content_checkpoint_byte_count_is_positive_and_cross_bound(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    contract = _make_contract(tmp_path / mutation)
    receipt = _read_json(contract.content_receipt)
    if mutation == "receipt_zero":
        receipt["checkpoint_tree"]["bytes"] = 0
        _write_content_json(contract.content_receipt, receipt)
    else:
        report_path = contract.content_receipt.parent / "content_freeze_audit.json"
        report = _read_json(report_path)
        report["checkpoint_evidence"]["checkpoint_bytes"] = (
            0 if mutation == "report_zero" else receipt["checkpoint_tree"]["bytes"] + 1
        )
        _republish_content_report(contract, report)

    with pytest.raises(audit.RelationAuditError, match=expected):
        contract.run(tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["rich_stat", "duplicate_task_path"])
def test_content_freeze_relation_task_subset_and_rich_stat_bridge_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    contract = _make_contract(tmp_path / mutation)
    plan_path = contract.content_receipt.parent / "audit_plan.json"
    plan = _read_json(plan_path)
    if mutation == "rich_stat":
        plan["tasks"][0]["source_stat"]["ctime_ns"] += 1
        expected = "current source rich identity differs from content-freeze plan"
    else:
        plan["tasks"][1]["relative_path"] = plan["tasks"][0]["relative_path"]
        expected = "relation task paths are duplicated"
    _republish_content_plan(contract, plan)

    with pytest.raises(audit.RelationAuditError, match=expected):
        contract.run(tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_current_source_identity_must_still_match_completed_content_plan(
    tmp_path: Path,
) -> None:
    contract = _make_contract(tmp_path / "case")
    path = contract.source_paths[-1]
    before = path.stat()
    payload = path.read_bytes()
    path.write_bytes(payload)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size

    with pytest.raises(
        audit.RelationAuditError,
        match="current source rich identity differs from content-freeze plan",
    ):
        contract.run(tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("parent_target", "expected"),
    [
        ("receipt", "final content-freeze receipt content changed"),
        (
            "artifact",
            "final content-freeze artifact content_freeze_audit.json content changed",
        ),
        ("state", "final content-freeze artifact run_state.json content changed"),
    ],
)
def test_final_content_parent_drift_removes_relation_pass_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_target: str,
    expected: str,
) -> None:
    contract = _make_contract(tmp_path / parent_target)
    output = tmp_path / f"output-{parent_target}"
    original = audit.atomic_write_json
    changed = False

    def mutate_parent_after_receipt(path: Path, value: Any) -> None:
        nonlocal changed
        original(path, value)
        if path.name == "work_topic_relation_audit.outputs.json" and not changed:
            parent_path = (
                contract.content_receipt
                if parent_target == "receipt"
                else contract.content_receipt.parent / "content_freeze_audit.json"
                if parent_target == "artifact"
                else contract.content_receipt.parent / "run_state.json"
            )
            parent_path.write_bytes(parent_path.read_bytes() + b" ")
            changed = True

    monkeypatch.setattr(audit, "atomic_write_json", mutate_parent_after_receipt)
    with pytest.raises(audit.RelationAuditError, match=expected):
        contract.run(output)
    assert changed is True
    assert not (output / "work_topic_relation_audit.json").exists()
    assert not (output / "work_topic_relation_audit.md").exists()
    assert not (output / "work_topic_relation_audit.outputs.json").exists()
    assert _read_json(output / "run_state.json")["status"] == (
        "FINALIZATION_FAILED_NO_VALID_RECEIPT"
    )


def test_deterministic_replay_is_byte_identical(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case")
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_report = contract.run(first)
    second_report = contract.run(second)

    assert first_report == second_report
    for relative in (
        "audit_plan.json",
        "sort_manifest.json",
        "work_topic_relation.events.raw.bin",
        "work_topic_relation.events.sorted.bin",
        "work_topic_edges.deduplicated.sorted.bin",
        "work_topic_relation_audit.json",
        "work_topic_relation_audit.md",
        "run_state.json",
        "work_topic_relation_audit.outputs.json",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


@pytest.mark.parametrize("duplicate_score", [0.8, 0.7])
def test_completed_output_resume_is_terminal_before_source_or_taxonomy_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate_score: float
) -> None:
    contract = _make_contract(tmp_path / "case", duplicate_score=duplicate_score)
    output = tmp_path / "output"
    contract.run(output)
    touched: list[str] = []

    def forbidden_resolve(*args: Any, **kwargs: Any) -> Path:
        touched.append("root")
        raise AssertionError("source or taxonomy root touched")

    def forbidden_taxonomy(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        touched.append("taxonomy")
        raise AssertionError("taxonomy payload touched")

    monkeypatch.setattr(audit, "resolve_root", forbidden_resolve)
    monkeypatch.setattr(audit, "load_topic_ids", forbidden_taxonomy)
    with pytest.raises(
        audit.RelationAuditError,
        match="completed output is terminal and cannot be resumed",
    ):
        contract.run(output, resume=True)
    assert touched == []


@pytest.mark.parametrize("target", ["raw_event", "sort_manifest", "checkpoint"])
def test_completed_output_corruption_is_not_accepted_as_terminal_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    contract.run(output)
    if target == "raw_event":
        path = output / "work_topic_relation.events.raw.bin"
    elif target == "sort_manifest":
        path = output / "sort_manifest.json"
    else:
        path = next((output / "checkpoints").rglob("*.json"))
    payload = bytearray(path.read_bytes())
    assert payload
    payload[-1] ^= 1
    path.write_bytes(payload)
    touched: list[str] = []

    def forbidden_resolve(*args: Any, **kwargs: Any) -> Path:
        touched.append("root")
        raise AssertionError("source or taxonomy root touched")

    def forbidden_taxonomy(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        touched.append("taxonomy")
        raise AssertionError("taxonomy payload touched")

    monkeypatch.setattr(audit, "resolve_root", forbidden_resolve)
    monkeypatch.setattr(audit, "load_topic_ids", forbidden_taxonomy)
    with pytest.raises(
        audit.RelationAuditError,
        match="terminal receipt artifact SHA-256 mismatch",
    ):
        contract.run(output, resume=True)
    assert touched == []


def test_completed_output_requires_exact_receipt_artifact_set_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    contract.run(output)
    receipt_path = output / "work_topic_relation_audit.outputs.json"
    receipt = _read_json(receipt_path)
    del receipt["artifacts"]["sort_manifest.json"]
    _write_json(receipt_path, receipt)
    touched: list[str] = []

    def forbidden_resolve(*args: Any, **kwargs: Any) -> Path:
        touched.append("root")
        raise AssertionError("source or taxonomy root touched")

    monkeypatch.setattr(audit, "resolve_root", forbidden_resolve)
    with pytest.raises(
        audit.RelationAuditError,
        match="terminal receipt artifact set mismatch",
    ):
        contract.run(output, resume=True)
    assert touched == []


def test_completed_output_checkpoint_tree_is_cross_bound_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    contract.run(output)
    receipt_path = output / "work_topic_relation_audit.outputs.json"
    receipt = _read_json(receipt_path)
    receipt["checkpoint_evidence"]["checkpoint_count"] += 1
    _write_json(receipt_path, receipt)
    touched: list[str] = []

    def forbidden_resolve(*args: Any, **kwargs: Any) -> Path:
        touched.append("root")
        raise AssertionError("source or taxonomy root touched")

    monkeypatch.setattr(audit, "resolve_root", forbidden_resolve)
    with pytest.raises(
        audit.RelationAuditError,
        match="terminal checkpoint evidence mismatch",
    ):
        contract.run(output, resume=True)
    assert touched == []


def test_max_files_and_hash_bucket_runs_are_never_full_pass(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case")

    max_files = contract.run(
        tmp_path / "max-files",
        max_files_per_table=1,
        authorize_full_scope=False,
    )
    hash_bucket = contract.run(
        tmp_path / "hash-bucket",
        hash_bucket_count=2,
        hash_bucket_index=1,
        authorize_full_scope=False,
    )

    for report in (max_files, hash_bucket):
        assert report["status"] == audit.PILOT_STATUS
        assert report["scope"]["full_scope"] is False
        assert report["gate_status"]["work_topic_relation_component"] == (
            "NOT_ESTABLISHED_PILOT"
        )
        assert report["gate_status"]["duplicate_work_topic_audit"] == (
            "NOT_ESTABLISHED_PILOT"
        )
        assert report["gate_status"]["raw_work_topic_edge_uniqueness"] == (
            "NOT_ESTABLISHED_PILOT"
        )
        if report["deduplicated_relation_artifact"]["sha256"] is not None:
            assert report["deduplicated_relation_artifact"]["status"] == (
                "MATERIALIZED_PILOT_ONLY_EXACT_SCORE_SET_VIEW"
            )


def test_unpromotable_pilot_may_omit_content_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path / "case")
    monkeypatch.delenv("OPENALEX_G1_CONTENT_FREEZE_OUTPUTS", raising=False)
    report = contract.run(
        tmp_path / "pilot",
        max_files_per_table=1,
        authorize_full_scope=False,
        content_freeze_receipt=None,
    )
    assert report["status"] == audit.PILOT_STATUS
    assert report["scope"]["full_scope"] is False
    assert report["content_baseline_parent"] == {
        "status": "OMITTED_UNPROMOTABLE_PILOT",
        "current_content_baseline_component": "NOT_REQUIRED_UNPROMOTABLE_PILOT",
        "historical_content_identity": "UNVERIFIED",
        "operator_paths_redacted": True,
        "relation_source_rich_stat_bridge": "OMITTED_PILOT",
    }
    assert report["gate_status"]["current_content_baseline_component"] == (
        "NOT_REQUIRED_UNPROMOTABLE_PILOT"
    )


def test_empty_pilot_bucket_has_defined_zero_row_aggregates(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    sorted_path = output / "work_topic_relation.events.sorted.bin"
    sorted_path.write_bytes(b"")
    plan = {"full_scope": False, "tasks": []}

    edge, artifact = audit.aggregate_edge_groups(
        sorted_path,
        event_rows=0,
        output_dir=output,
        plan=plan,
        topic_bits=14,
        chunk_rows=2,
    )
    work = audit.aggregate_work_groups(
        sorted_path,
        event_rows=0,
        topic_bits=14,
        chunk_rows=2,
    )

    assert edge["grammar_valid_including_w0"] == audit._empty_edge_view()
    assert work["grammar_valid_including_w0"] == audit._empty_work_view()
    assert work["w0_work_group"] is None
    assert artifact["status"] == "MATERIALIZED_PILOT_ONLY_EXACT_SCORE_SET_VIEW"
    assert artifact["rows"] == 0


def test_full_scope_requires_separate_authorization_before_source_or_taxonomy_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path / "case")
    touched: list[str] = []

    def forbidden_resolve(*args: Any, **kwargs: Any) -> Path:
        touched.append("root")
        raise AssertionError("source or taxonomy root touched")

    def forbidden_taxonomy(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        touched.append("taxonomy")
        raise AssertionError("taxonomy payload touched")

    monkeypatch.setattr(audit, "resolve_root", forbidden_resolve)
    monkeypatch.setattr(audit, "load_topic_ids", forbidden_taxonomy)

    with pytest.raises(
        audit.RelationAuditError,
        match="full-scope execution requires explicit authorize_full_scope",
    ):
        contract.run(tmp_path / "output", authorize_full_scope=False)
    assert touched == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("receipt_state", ["MISSING", "PAUSED"])
def test_full_scope_requires_completed_content_parent_before_source_or_taxonomy_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_state: str,
) -> None:
    contract = _make_contract(tmp_path / receipt_state.lower())
    receipt: Path | None = None
    expected = "requires current-content outputs receipt"
    if receipt_state == "PAUSED":
        state_path = contract.content_receipt.parent / "run_state.json"
        state = _read_json(state_path)
        state["status"] = "PAUSED"
        _write_content_json(state_path, state)
        receipt = contract.content_receipt
        expected = "not COMPLETE"
    touched: list[str] = []

    def forbidden_resolve(*args: Any, **kwargs: Any) -> Path:
        touched.append("root")
        raise AssertionError("source or taxonomy root touched")

    def forbidden_taxonomy(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        touched.append("taxonomy")
        raise AssertionError("taxonomy payload touched")

    monkeypatch.setattr(audit, "resolve_root", forbidden_resolve)
    monkeypatch.setattr(audit, "load_topic_ids", forbidden_taxonomy)
    with pytest.raises(audit.RelationAuditError, match=expected):
        contract.run(
            tmp_path / "output",
            content_freeze_receipt=receipt,
        )
    assert touched == []
    assert not (tmp_path / "output").exists()


def test_symlink_inputs_and_output_containment_are_refused(tmp_path: Path) -> None:
    inside_contract = _make_contract(tmp_path / "inside")
    with pytest.raises(audit.RelationAuditError, match="overlaps source root"):
        inside_contract.run(inside_contract.source_root / "audit-output")

    output_contract = _make_contract(tmp_path / "output-link-case")
    output_target = tmp_path / "output-target"
    output_target.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(output_target, target_is_directory=True)
    with pytest.raises(audit.RelationAuditError, match="may not be a symlink"):
        output_contract.run(output_link)

    input_contract = _make_contract(tmp_path / "input-link-case")
    original = input_contract.main_path
    target = tmp_path / "copied-main.parquet"
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(audit.RelationAuditError, match="path component is a symlink"):
        input_contract.run(tmp_path / "input-link-output")


@pytest.mark.parametrize(
    "relative",
    [
        "audit.lock",
        "work_topic_relation.events.raw.bin",
        "work_topic_relation_audit.json",
        ".audit_plan.json.stale.tmp",
    ],
)
def test_internal_output_symlink_cannot_redirect_external_writes(
    tmp_path: Path, relative: str
) -> None:
    contract = _make_contract(tmp_path / relative.replace("/", "-"))
    output = tmp_path / f"output-{relative.replace('/', '-')}"
    output.mkdir(mode=0o700)
    sentinel = tmp_path / f"sentinel-{relative.replace('/', '-')}"
    sentinel.write_bytes(b"must remain unchanged")
    (output / relative).symlink_to(sentinel)

    with pytest.raises(audit.RelationAuditError, match="output tree"):
        contract.run(output, resume=True)
    assert sentinel.read_bytes() == b"must remain unchanged"


def test_unknown_or_stale_managed_output_entry_fails_closed(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    stale = output / ".work_topic_relation.events.raw.bin.123.deadbeef.tmp"
    stale.write_bytes(b"stale")
    stale.chmod(0o600)

    with pytest.raises(
        audit.RelationAuditError, match="unknown or stale managed artifact"
    ):
        contract.run(output, resume=True)


def test_untrusted_output_mode_is_rejected(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"
    output.mkdir(mode=0o777)
    output.chmod(0o777)

    with pytest.raises(audit.RelationAuditError, match="group/world writable"):
        contract.run(output)


def test_published_output_tree_uses_private_regular_files(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path / "case")
    output = tmp_path / "output"

    contract.run(output)

    for path in output.rglob("*"):
        info = path.lstat()
        assert not path.is_symlink()
        if path.is_dir():
            assert info.st_mode & 0o022 == 0
        else:
            assert info.st_mode & 0o777 == 0o600


def test_cli_error_formatter_never_exposes_operator_path() -> None:
    error = audit.RelationAuditError(
        "cannot open /home/private/source.parquet under /tmp/secret"
    )

    rendered = audit.redacted_error_message(error)

    assert rendered.startswith("RelationAuditError:")
    assert "/home/" not in rendered
    assert "/tmp/" not in rendered
    assert "redacted" in rendered


def test_production_config_binds_declared_parent_artifacts() -> None:
    assert audit.PRODUCTION_CONFIG_SHA256 == _sha256(CONFIG_PATH)
    audit.FROZEN_CONFIG_SHA256 = audit.PRODUCTION_CONFIG_SHA256
    config = audit.read_config(CONFIG_PATH)
    for key, contract in config["parents"].items():
        if key == "current_content_baseline":
            continue
        path = REPO_ROOT / contract["relative_path"]
        assert path.is_file()
        assert _sha256(path) == contract["sha256"]
    content = config["parents"]["current_content_baseline"]
    assert content["required_plan_sha256"] == (
        "97294fe1c1fa026f664ca239ebfa9fc6dce593c847a20eb62157e38312b6725e"
    )
    assert content["required_task_fingerprint_sha256"] == (
        "f915cade4ce6581832c46957a2ad0a37ceabe2296099f0faeaac56ffa035693e"
    )
    assert content["required_file_count"] == 21_019
    assert content["required_bytes"] == 988_404_537_807
    assert content["relation_source_subset"] == {
        "required_table_keys": ["main", "topics"],
        "required_files": 1_423,
        "required_bytes": 166_699_471_247,
        "required_source_stat_fields": [
            "device",
            "inode",
            "mode",
            "bytes",
            "mtime_ns",
            "ctime_ns",
        ],
        "full_scope_policy": "exact_relation_table_task_set",
        "pilot_policy": "selected_relation_tasks_may_be_a_strict_subset",
    }
    assert config["execution"]["gpu_used"] is False
    assert config["execution"]["resume_verification"] == (
        "rehash_current_projected_payload_and_spool_segment"
    )
    assert config["success_contract"]["required_full_parent_boundary"] == {
        "raw_source_id_validity": "BLOCKED",
        "w_zero_entity_validity": "UNRESOLVED",
        "parent_content_identity": "UNVERIFIED",
        "publication_date_validity": "PENDING",
        "temporal_available_at": "PENDING",
        "author_entity_validity": "PENDING",
        "candidate_eligibility": "PENDING",
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
    }


def test_production_resource_plan_matches_declared_full_upper_bound() -> None:
    audit.FROZEN_CONFIG_SHA256 = audit.PRODUCTION_CONFIG_SHA256
    config = audit.read_config(CONFIG_PATH)
    estimate = audit._resource_estimate(
        config,
        {
            "main": {
                "selected_footer_rows": config["tables"]["main"]["expected_footer_rows"]
            },
            "topics": {
                "selected_footer_rows": config["tables"]["topics"][
                    "expected_footer_rows"
                ]
            },
        },
    )

    assert estimate["event_row_upper_bound"] == 1_420_855_753
    assert estimate["raw_spool_bytes"] == 34_100_538_072
    assert estimate["sorted_event_bytes"] == 34_100_538_072
    assert estimate["deduplicated_edge_bytes_upper_bound"] == 14_857_000_736
    assert estimate["artifact_bytes_upper_bound"] == 83_058_076_880
    assert estimate["peak_in_memory_sort_array_bytes"] == 34_100_538_072
    assert estimate["required_free_disk_bytes"] == 114_560_014_340
    assert estimate["required_available_memory_bytes"] == 55_445_774_404
    assert estimate["pilot_promotion_allowed"] is False
