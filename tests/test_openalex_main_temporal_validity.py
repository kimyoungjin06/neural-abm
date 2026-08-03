from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_main_temporal_validity.py"
CONFIG_PATH = REPO_ROOT / "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml"
DOC_PATH = REPO_ROOT / "docs/case-studies/researcher-pivot/openalex-main-temporal-validity-audit.md"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_main_temporal_audit_module", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _write_content_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": pa.array([row["id"] for row in rows], type=pa.string()),
            "publication_date": pa.array([row["publication_date"] for row in rows], type=pa.string()),
            "publication_year": pa.array([row["publication_year"] for row in rows], type=pa.float64()),
            "created_date": pa.array([row["created_date"] for row in rows], type=pa.string()),
            "updated_date": pa.array([row["updated_date"] for row in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path, row_group_size=2, compression="NONE", use_dictionary=False, write_statistics=False)


def _rich(path: Path) -> dict[str, int]:
    info = path.stat()
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": info.st_mode & 0o7777,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _schema(path: Path) -> dict[str, dict[str, Any]]:
    schema = pq.ParquetFile(path).schema_arrow
    return {
        field.name: {"nullable": bool(field.nullable), "type": str(field.type)}
        for field in schema
    }


class Contract:
    def __init__(
        self,
        *,
        repo: Path,
        source: Path,
        config: Path,
        receipt: Path,
        config_sha: str,
        files: list[Path],
    ) -> None:
        self.repo = repo
        self.source = source
        self.config = config
        self.receipt = receipt
        self.config_sha = config_sha
        self.files = files

    def run(self, output: Path, **kwargs: Any) -> dict[str, Any]:
        audit.REPO_ROOT = self.repo
        audit.FROZEN_CONFIG_SHA256 = self.config_sha
        options: dict[str, Any] = {
            "config_path": self.config,
            "source_root": self.source,
            "content_freeze_receipt": self.receipt,
            "output_dir": output,
            "batch_size": 2,
            "authorize_full_scope": True,
        }
        options.update(kwargs)
        return audit.run_audit(**options)


def _base_rows() -> list[dict[str, Any]]:
    return [
        {"id": "https://openalex.org/W0", "publication_date": "1899-12-31", "publication_year": 1899.0, "created_date": "2016-01-01", "updated_date": "2016-01-01"},
        {"id": "https://openalex.org/W1", "publication_date": None, "publication_year": None, "created_date": "2016-01-01", "updated_date": "2017-01-01"},
        {"id": "https://openalex.org/W2", "publication_date": "2026-03-30", "publication_year": 2026.0, "created_date": "2020-01-01", "updated_date": "2026-03-30"},
        {"id": "https://openalex.org/W3", "publication_date": "2000-01-01", "publication_year": 2000.0, "created_date": "2019-01-01", "updated_date": "2020-01-01"},
        {"id": "https://openalex.org/W4", "publication_date": "2026-03-31", "publication_year": 2026.0, "created_date": "2020-01-01", "updated_date": "2026-03-30"},
        {"id": "https://openalex.org/W-1", "publication_date": "bad", "publication_year": None, "created_date": None, "updated_date": None},
    ]


def _make_contract(
    base: Path,
    mutate_rows: Callable[[list[dict[str, Any]]], None] | None = None,
    *,
    unresolved_policy: bool = False,
) -> Contract:
    repo = base / "repo"
    source = base / "source"
    content = base / "content-freeze"
    repo.mkdir(parents=True)
    source.mkdir()
    content.mkdir(mode=0o700)
    content.chmod(0o700)
    rows = _base_rows()
    if mutate_rows is not None:
        mutate_rows(rows)
    files = [source / "synthetic_main/part-0.parquet", source / "synthetic_main/part-1.parquet"]
    _write_parquet(files[0], rows[:3])
    _write_parquet(files[1], rows[3:])
    inventory_files = []
    for path in files:
        parquet = pq.ParquetFile(path)
        info = path.stat()
        inventory_files.append(
            {
                "relative_path": path.relative_to(source).as_posix(),
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "footer_rows": parquet.metadata.num_rows,
                "schema_fields": _schema(path),
            }
        )
    total_bytes = sum(item["bytes"] for item in inventory_files)
    config = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    config["source"].update({"root_env": "SYNTHETIC_ROOT", "root_id": "synthetic-main-temporal", "snapshot_date": "2026-03-30", "lineage_id": "synthetic"})
    config["table"].update({"name": "synthetic_main", "expected_files": 2, "expected_footer_rows": 6, "expected_bytes": total_bytes})
    grammar = config["parents"]["grammar_valid_main_recount"]["required_exact_counts"]
    grammar.update(
        {
            "body_rows_scanned": 6,
            "null_rows": 0,
            "quarantined_malformed_rows": 1,
            "unexpected_malformed_rows": 0,
            "grammar_valid_rows": 5,
            "grammar_valid_distinct_ids": 5,
            "grammar_valid_duplicate_id_groups": 0,
            "grammar_valid_duplicate_row_excess": 0,
            "grammar_valid_max_multiplicity": 1,
            "grammar_valid_min_numeric_id": 0,
            "grammar_valid_max_numeric_id": 4,
            "zero_id_rows_retained": 1,
        }
    )
    config["identifier_policy"]["w_minus_1"].update({"relative_path": "synthetic_main/part-1.parquet", "row_group": 1, "file_row": 2})
    config["temporal_policy"]["publication_date_year"]["required_aggregate_regression"] = {
        "publication_year_null_rows": 1,
        "publication_year_pre_1900_rows": 1,
        "publication_year_snapshot_year_rows": 2,
        "publication_year_post_snapshot_year_rows": 0,
        "publication_year_min": 1899,
        "publication_year_max": 2026,
    }
    if unresolved_policy:
        config["temporal_policy"]["resolution"] = "UNRESOLVED"
    resources = config["execution"]["resource_preflight"]
    resources.update({"disk_safety_multiplier_milli": 1000, "disk_reserve_bytes": 0, "memory_safety_multiplier_milli": 1000, "memory_reserve_bytes": 0})

    inventory_fp = hashlib.sha256(b"inventory").hexdigest()
    stability_fp = hashlib.sha256(b"stability").hexdigest()
    inventory = {
        "schema_version": "openalex_source_inventory.v1",
        "root": {"id": "synthetic-main-temporal", "snapshot_date": "2026-03-30", "lineage_id": "synthetic", "inventory_fingerprint_sha256": inventory_fp, "stability_fingerprint_sha256": stability_fp},
        "tables": [{"name": "synthetic_main", "file_count": 2, "footer_rows": 6, "bytes": total_bytes, "files": inventory_files}],
    }
    parents = repo / "parents"
    inventory_path = parents / "inventory.json"
    grammar_path = parents / "grammar.json"
    work_path = parents / "work.json"
    _write_json(inventory_path, inventory)
    grammar_spec = config["parents"]["grammar_valid_main_recount"]
    _write_json(grammar_path, {"schema_version": grammar_spec["schema_version"], "status": grammar_spec["required_status"], "exact_counts": grammar, "gate_status": grammar_spec["required_gate_states"]})
    work_spec = config["parents"]["work_source_membership"]
    work_fp = hashlib.sha256(b"work").hexdigest()
    _write_json(work_path, {"schema_version": work_spec["schema_version"], "status": work_spec["required_status"], "gate_status": work_spec["required_gate_states"], "provenance": {"result_fingerprint_sha256": work_fp}})
    for key, path in (("inventory", inventory_path), ("grammar_valid_main_recount", grammar_path), ("work_source_membership", work_path)):
        config["parents"][key].update({"relative_path": path.relative_to(repo).as_posix(), "sha256": _sha(path)})
    config["parents"]["inventory"].update({"inventory_fingerprint_sha256": inventory_fp, "stability_fingerprint_sha256": stability_fp})
    config["parents"]["work_source_membership"]["result_fingerprint_sha256"] = work_fp

    task_fp = hashlib.sha256(b"content-task").hexdigest()
    content_plan: dict[str, Any] = {
        "schema_version": "openalex_g1_content_freeze_plan.v1",
        "full_scope": True,
        "max_files": None,
        "parent_inventory": {"manifest_sha256": hashlib.sha256(b"content-parent-manifest").hexdigest()},
        "declared_file_count": 2,
        "declared_bytes": total_bytes,
        "selected_file_count": 2,
        "selected_bytes": total_bytes,
        "task_fingerprint_sha256": task_fp,
        "tasks": [
            {"table_name": "synthetic_main", "relative_path": path.relative_to(source).as_posix(), "source_stat": _rich(path)}
            for path in files
        ],
    }
    content_plan["plan_sha256"] = audit.sha256_bytes(audit.canonical_json_bytes(content_plan))
    plan_path = content / "audit_plan.json"
    report_path = content / "content_freeze_audit.json"
    markdown_path = content / "content_freeze_audit.md"
    _write_content_json(plan_path, content_plan)
    tree_fp = hashlib.sha256(b"content-tree").hexdigest()
    manifest_fp = content_plan["parent_inventory"]["manifest_sha256"]
    checkpoint_tree = {"files": 2, "bytes": 777, "tree_sha256": hashlib.sha256(b"checkpoint-tree").hexdigest()}
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
    mutation_limit = "synthetic same-size restored-mtime limitation"
    checkpoint_limit = "synthetic unkeyed-checkpoint limitation"
    completion_contract = "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
    content_report = {
        "schema_version": "openalex_g1_content_freeze_report.v1",
        "status": "PASS_CURRENT_CONTENT_BASELINE_COMPONENT",
        "scope": {
            "full_scope": True,
            "max_files": None,
            "declared_files": 2,
            "declared_bytes": total_bytes,
            "selected_files": 2,
            "selected_bytes": total_bytes,
            "gpu_used": False,
        },
        "parent_inventory": {"manifest_sha256": manifest_fp},
        "pass_current_content_baseline_component": True,
        "content_baseline": {
            "files": 2,
            "bytes": total_bytes,
            "content_tree_sha256": tree_fp,
            "historical_identity_established": False,
            "prior_component_run_identity_established": False,
        },
        "gate_status": content_gates,
        "checkpoint_evidence": {"checkpoint_files": 2, "checkpoint_bytes": 777, "checkpoint_tree_sha256": checkpoint_tree["tree_sha256"]},
        "publication": {"completion_contract": completion_contract},
        "execution": {"path_redacted": True},
        "claim_boundary": content_claim,
        "same_size_restored_mtime_limitation": mutation_limit,
        "checkpoint_trust_limitation": checkpoint_limit,
    }
    result_core = {
        "schema_version": content_report["schema_version"],
        "status": content_report["status"],
        "full_scope": True,
        "max_files": None,
        "parent_manifest_sha256": manifest_fp,
        "files": 2,
        "bytes": total_bytes,
        "content_tree_sha256": tree_fp,
        "gate_status": content_gates,
        "claim_boundary": content_claim,
        "same_size_restored_mtime_limitation": mutation_limit,
        "checkpoint_trust_limitation": checkpoint_limit,
        "publication_completion_contract": completion_contract,
    }
    result_fp = audit.sha256_bytes(audit.canonical_json_bytes(result_core))
    content_report["provenance"] = {
        "result_fingerprint_sha256": result_fp,
        "path_redacted": True,
    }
    _write_content_json(report_path, content_report)
    markdown_path.write_text("synthetic content freeze\n", encoding="utf-8")
    markdown_path.chmod(0o600)
    artifacts = [
        {"relative_path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in (plan_path, report_path, markdown_path)
    ]
    receipt_path = content / "content_freeze_audit.outputs.json"
    _write_content_json(
        receipt_path,
        {
            "schema_version": "openalex_g1_content_freeze_outputs.v1",
            "status": "PASS_CURRENT_CONTENT_BASELINE_COMPONENT",
            "task_fingerprint_sha256": task_fp,
            "plan_sha256": content_plan["plan_sha256"],
            "result_fingerprint_sha256": result_fp,
            "content_tree_sha256": tree_fp,
            "checkpoint_tree": checkpoint_tree,
            "artifacts": artifacts,
            "completion_contract": completion_contract,
            "run_state_bound_separately": True,
            "path_redacted": True,
        },
    )
    _write_content_json(
        content / "run_state.json",
        {
            "schema_version": "openalex_g1_content_freeze_state.v1",
            "status": "COMPLETE",
            "plan_sha256": content_plan["plan_sha256"],
            "task_fingerprint_sha256": task_fp,
            "completed_files": 2,
            "completed_bytes": total_bytes,
            "total_files": 2,
            "total_bytes": total_bytes,
            "path_redacted": True,
        },
    )
    baseline = config["parents"]["current_content_baseline"]
    baseline.update({"required_plan_sha256": content_plan["plan_sha256"], "required_task_fingerprint_sha256": task_fp, "required_file_count": 2, "required_bytes": total_bytes})
    config_path = repo / "synthetic-temporal.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_path.chmod(0o600)
    return Contract(repo=repo, source=source, config=config_path, receipt=receipt_path, config_sha=_sha(config_path), files=files)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_repository_config_and_real_parent_contracts_validate() -> None:
    assert audit.PRODUCTION_CONFIG_SHA256 == _sha(CONFIG_PATH)
    audit.REPO_ROOT = REPO_ROOT
    audit.FROZEN_CONFIG_SHA256 = _sha(CONFIG_PATH)
    config, observed, size, _ = audit._read_config_snapshot(CONFIG_PATH)
    assert observed == _sha(CONFIG_PATH)
    assert size > 0
    context = audit.validate_parents(config)
    assert context["grammar"]["exact_counts"] == config["parents"]["grammar_valid_main_recount"]["required_exact_counts"]


def test_inventory_lineage_must_match_source_contract(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    config = yaml.safe_load(contract.config.read_text(encoding="utf-8"))
    inventory_path = contract.repo / config["parents"]["inventory"]["relative_path"]
    inventory = _read(inventory_path)
    inventory["root"]["lineage_id"] = "different-lineage"
    _write_json(inventory_path, inventory)
    config["parents"]["inventory"]["sha256"] = _sha(inventory_path)
    audit.REPO_ROOT = contract.repo
    with pytest.raises(audit.TemporalAuditError, match="inventory lineage mismatch"):
        audit.validate_parents(config)


def test_synthetic_full_pass_and_frozen_boundaries(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    output = tmp_path / "output"
    report = contract.run(output)
    assert report["status"] == audit.FULL_STATUS
    assert report["pass_main_temporal_validity_component"] is True
    assert report["publication_buckets"] == {
        "publication_both_null_rows": 1,
        "publication_pre_1900_rows": 1,
        "publication_1900_before_snapshot_year_rows": 1,
        "publication_snapshot_year_on_or_before_snapshot_rows": 1,
        "publication_snapshot_year_after_snapshot_rows": 1,
        "publication_post_snapshot_year_rows": 0,
    }
    assert report["quarantine"]["w_minus_1_excluded_from_all_temporal_metrics"] is True
    assert report["gate_status"]["current_content_baseline_component"] == "PASS"
    assert report["gate_status"]["historical_content_identity"] == "UNVERIFIED"
    assert report["gate_status"]["temporal_available_at"] == "PENDING_NOT_IN_THIS_COMPONENT"
    assert report["gate_status"]["pass_g1"] is False
    assert report["content_baseline_parent"]["main_source_rich_stat_bridge"]["status"].startswith("PASS_")
    assert stat_mode(output) == 0o700
    assert stat_mode(output / "main_temporal_validity_audit.outputs.json") == 0o600


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_pilot_never_promotes_and_may_omit_content_receipt(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    report = contract.run(tmp_path / "pilot", max_files=1, content_freeze_receipt=None)
    assert report["status"] == audit.PILOT_STATUS
    assert report["pass_main_temporal_validity_component"] is False
    assert report["gate_status"]["current_content_baseline_component"] == "NOT_REQUIRED_UNPROMOTABLE_PILOT"


def test_full_scope_authorization_precedes_source_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _make_contract(tmp_path)
    called = False

    def forbidden(*args: Any, **kwargs: Any) -> Path:
        nonlocal called
        called = True
        raise AssertionError("source touched")

    monkeypatch.setattr(audit, "resolve_root", forbidden)
    with pytest.raises(audit.TemporalAuditError, match="authorize-full"):
        contract.run(tmp_path / "output", authorize_full_scope=False)
    assert called is False


@pytest.mark.parametrize("receipt_state", ["MISSING", "PAUSED"])
def test_full_scope_rejects_missing_or_paused_content_receipt_before_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_state: str,
) -> None:
    contract = _make_contract(tmp_path)
    receipt: Path | None = None
    expected = "requires current-content outputs receipt"
    if receipt_state == "PAUSED":
        state_path = contract.receipt.parent / "run_state.json"
        state = _read(state_path)
        state["status"] = "PAUSED"
        _write_content_json(state_path, state)
        receipt = contract.receipt
        expected = "not COMPLETE"
    called = False

    def forbidden(*args: Any, **kwargs: Any) -> Path:
        nonlocal called
        called = True
        raise AssertionError("source root touched")

    monkeypatch.setattr(audit, "resolve_root", forbidden)
    with pytest.raises(audit.TemporalAuditError, match=expected):
        contract.run(
            tmp_path / "output",
            content_freeze_receipt=receipt,
        )
    assert called is False


@pytest.mark.parametrize(
    ("mutator", "violation"),
    [
        (lambda rows: rows[3].update(publication_year=2001.0), "publication_date_year_mismatch_rows"),
        (lambda rows: rows[3].update(publication_date="2000-02-30"), "publication_date_parse_error_rows"),
        (lambda rows: rows[1].update(publication_year=2001.0), "publication_pairwise_null_mismatch_rows"),
        (lambda rows: rows[3].update(publication_year=float("nan")), "publication_year_invalid_rows"),
        (lambda rows: rows[3].update(publication_year=2000.5), "publication_year_invalid_rows"),
        (lambda rows: rows[3].update(created_date=None), "created_updated_null_rows"),
        (lambda rows: rows[3].update(created_date="2020-02-30"), "created_date_parse_error_rows"),
        (lambda rows: rows[3].update(updated_date=None), "created_updated_null_rows"),
        (lambda rows: rows[3].update(updated_date="2020-02-30"), "updated_date_parse_error_rows"),
        (lambda rows: rows[3].update(created_date="2021-01-01", updated_date="2020-01-01"), "created_gt_updated_unexpected_rows"),
        (lambda rows: rows[3].update(updated_date="2026-03-31"), "future_metadata_unexpected_rows"),
    ],
)
def test_invalid_temporal_rows_block_full_component(
    tmp_path: Path,
    mutator: Callable[[list[dict[str, Any]]], None],
    violation: str,
) -> None:
    contract = _make_contract(tmp_path, mutator)
    report = contract.run(tmp_path / "output")
    assert report["status"] == audit.BLOCKED_STATUS
    assert report["violations"][violation] > 0
    assert report["pass_main_temporal_validity_component"] is False


def test_strict_date_rejects_year_zero_invalid_leap_and_noncanonical_padding() -> None:
    values = pa.array(
        [
            "0000-01-01",
            "0001-01-01",
            "2024-02-29",
            "2023-02-29",
            "2024-2-09",
            None,
        ],
        type=pa.string(),
    )
    null, valid, _, years = audit._strict_date(values)
    assert null.tolist() == [False, False, False, False, False, True]
    assert valid.tolist() == [False, True, True, False, False, False]
    assert years.tolist()[:5] == [0, 1, 2024, 2023, 2024]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[0].update(id="https://openalex.org/W5"),
        lambda rows: rows[1].update(id="https://openalex.org/W0"),
    ],
)
def test_w0_missing_or_duplicate_fails_full_accounting(
    tmp_path: Path,
    mutator: Callable[[list[dict[str, Any]]], None],
) -> None:
    contract = _make_contract(tmp_path, mutator)
    report = contract.run(tmp_path / "output")
    assert report["status"] == audit.BLOCKED_STATUS
    assert report["accounting_contract"]["w0_equals_frozen_count"] is False
    assert report["violations"]["accounting_contract_failures"] > 0


def test_aggregate_regression_mismatch_blocks_otherwise_valid_rows(
    tmp_path: Path,
) -> None:
    def move_to_post_snapshot(rows: list[dict[str, Any]]) -> None:
        rows[3].update(publication_date="2027-01-01", publication_year=2027.0)

    contract = _make_contract(tmp_path, move_to_post_snapshot)
    report = contract.run(tmp_path / "output")
    assert report["status"] == audit.BLOCKED_STATUS
    assert report["violations"]["aggregate_regression_failures"] == 1
    assert report["aggregate_regression"]["exact"] is False


def test_unresolved_policy_cannot_pass(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path, unresolved_policy=True)
    report = contract.run(tmp_path / "output")
    assert report["status"] == audit.BLOCKED_STATUS
    assert report["violations"]["policy_resolution_failures"] == 1


def test_pause_then_verified_resume(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    output = tmp_path / "output"
    paused = contract.run(output, pause_after_new_files=1)
    assert paused["status"] == "PAUSED"
    report = contract.run(output, resume=True)
    assert report["status"] == audit.FULL_STATUS
    assert report["checkpoint_evidence"]["checkpoint_count"] == 2


@pytest.mark.parametrize("target", ["checkpoint", "source"])
def test_resume_rejects_checkpoint_or_same_size_source_mutation(tmp_path: Path, target: str) -> None:
    contract = _make_contract(tmp_path)
    output = tmp_path / "output"
    contract.run(output, pause_after_new_files=1)
    if target == "checkpoint":
        checkpoint = next((output / "checkpoints/main").glob("*.json"))
        value = _read(checkpoint)
        value["metrics"]["raw_rows"] += 1
        _write_json(checkpoint, value)
    else:
        path = contract.files[0]
        before = path.stat()
        payload = bytearray(path.read_bytes())
        payload[-8] ^= 1
        path.write_bytes(payload)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(audit.TemporalAuditError):
        contract.run(output, resume=True)


def test_config_drift_is_rejected_before_output(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    value = yaml.safe_load(contract.config.read_text(encoding="utf-8"))
    value["claim_boundary"] += " drift"
    contract.config.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(audit.TemporalAuditError, match="SHA-256"):
        contract.run(tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_config_path_swap_after_single_fd_snapshot_fails_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _make_contract(tmp_path)
    original = audit._read_config_snapshot

    def swapped(path: Path) -> tuple[dict[str, Any], str, int, dict[str, int]]:
        snapshot = original(path)
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, path)
        return snapshot

    monkeypatch.setattr(audit, "_read_config_snapshot", swapped)
    with pytest.raises(audit.TemporalAuditError, match="final config identity changed"):
        contract.run(tmp_path / "output")
    assert not (tmp_path / "output/main_temporal_validity_audit.outputs.json").exists()


def test_symlink_source_config_and_output_are_refused(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    source_link = tmp_path / "source-link"
    source_link.symlink_to(contract.source, target_is_directory=True)
    with pytest.raises(audit.TemporalAuditError, match="symlink"):
        contract.run(tmp_path / "out-source", source_root=source_link)
    output_target = tmp_path / "target"
    output_target.mkdir(mode=0o700)
    output_link = tmp_path / "out-link"
    output_link.symlink_to(output_target, target_is_directory=True)
    with pytest.raises(audit.TemporalAuditError, match="symlink"):
        contract.run(output_link)

    config_link = tmp_path / "config-link.yaml"
    config_link.symlink_to(contract.config)
    with pytest.raises(audit.TemporalAuditError, match="symlink"):
        contract.run(tmp_path / "out-config", config_path=config_link)


def test_symlink_source_file_and_internal_checkpoint_are_refused(
    tmp_path: Path,
) -> None:
    source_contract = _make_contract(tmp_path / "source-case")
    (tmp_path / "source-case").chmod(0o700)
    source_file = source_contract.files[0]
    moved_source = tmp_path / "moved-source.parquet"
    source_file.replace(moved_source)
    source_file.symlink_to(moved_source)
    with pytest.raises(audit.TemporalAuditError, match="symlink"):
        source_contract.run(tmp_path / "source-output")

    checkpoint_contract = _make_contract(tmp_path / "checkpoint-case")
    (tmp_path / "checkpoint-case").chmod(0o700)
    checkpoint_output = tmp_path / "checkpoint-output"
    checkpoint_contract.run(checkpoint_output, pause_after_new_files=1)
    checkpoint = next((checkpoint_output / "checkpoints/main").glob("*.json"))
    moved_checkpoint = tmp_path / "moved-checkpoint.json"
    checkpoint.replace(moved_checkpoint)
    checkpoint.symlink_to(moved_checkpoint)
    with pytest.raises(audit.TemporalAuditError, match="checkpoint artifact is unsafe"):
        checkpoint_contract.run(checkpoint_output, resume=True)


def test_resource_preflight_blocks_before_page_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _make_contract(tmp_path)
    monkeypatch.setattr(audit, "_available_disk_bytes", lambda path: 0)
    called = False

    def forbidden(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("data page scanned")

    monkeypatch.setattr(audit, "scan_task", forbidden)
    with pytest.raises(audit.TemporalAuditError, match="RESOURCE_PREFLIGHT"):
        contract.run(tmp_path / "output")
    assert called is False


@pytest.mark.parametrize("target_name", ["main_temporal_validity_audit.json", "main_temporal_validity_audit.outputs.json"])
def test_post_write_report_or_receipt_corruption_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_name: str
) -> None:
    contract = _make_contract(tmp_path)
    original = audit.atomic_write_json
    corrupted = False

    def corrupt(path: Path, value: Any) -> None:
        nonlocal corrupted
        original(path, value)
        if path.name == target_name and not corrupted:
            path.write_bytes(path.read_bytes() + b" ")
            corrupted = True

    monkeypatch.setattr(audit, "atomic_write_json", corrupt)
    with pytest.raises(audit.TemporalAuditError):
        contract.run(tmp_path / "output")
    assert corrupted
    assert not (tmp_path / "output/main_temporal_validity_audit.outputs.json").exists()


@pytest.mark.parametrize("target", ["plan", "checkpoint", "state"])
def test_noncanonical_internal_json_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    contract = _make_contract(tmp_path)
    original = audit.atomic_write_json
    corrupted = False

    def corrupt(path: Path, value: Any) -> None:
        nonlocal corrupted
        original(path, value)
        matches = (
            (target == "plan" and path.name == "audit_plan.json")
            or (target == "checkpoint" and path.parent.name == "main")
            or (
                target == "state"
                and path.name == "run_state.json"
                and isinstance(value, dict)
                and value.get("status") == "COMPLETE_PENDING_RECEIPT"
            )
        )
        if matches and not corrupted:
            path.write_bytes(path.read_bytes() + b" ")
            corrupted = True

    monkeypatch.setattr(audit, "atomic_write_json", corrupt)
    with pytest.raises(audit.TemporalAuditError, match="canonical JSON"):
        contract.run(tmp_path / "output")
    assert corrupted
    assert not (tmp_path / "output/main_temporal_validity_audit.outputs.json").exists()


@pytest.mark.parametrize("entry_kind", ["directory", "symlink"])
def test_undeclared_checkpoint_root_entry_fails_final_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    contract = _make_contract(tmp_path)
    output = tmp_path / "output"
    original = audit.atomic_write_json
    injected = False

    def inject(path: Path, value: Any) -> None:
        nonlocal injected
        original(path, value)
        if path.name == "main_temporal_validity_audit.outputs.json" and not injected:
            extra = output / "checkpoints/undeclared"
            if entry_kind == "directory":
                extra.mkdir(mode=0o700)
            else:
                target = tmp_path / "checkpoint-target"
                target.mkdir(mode=0o700)
                extra.symlink_to(target, target_is_directory=True)
            injected = True

    monkeypatch.setattr(audit, "atomic_write_json", inject)
    with pytest.raises(audit.TemporalAuditError, match="checkpoint root entry set"):
        contract.run(output)
    assert injected
    assert not (output / "main_temporal_validity_audit.outputs.json").exists()


def test_receipt_write_failure_removes_pass_surfaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _make_contract(tmp_path)
    original = audit.atomic_write_json

    def fail(path: Path, value: Any) -> None:
        if path.name == "main_temporal_validity_audit.outputs.json":
            raise audit.TemporalAuditError("injected receipt failure")
        original(path, value)

    monkeypatch.setattr(audit, "atomic_write_json", fail)
    with pytest.raises(audit.TemporalAuditError, match="injected"):
        contract.run(tmp_path / "output")
    for name in ("main_temporal_validity_audit.json", "main_temporal_validity_audit.md", "main_temporal_validity_audit.outputs.json"):
        assert not (tmp_path / "output" / name).exists()
    assert _read(tmp_path / "output/run_state.json")["status"] == (
        "FINALIZATION_FAILED_NO_RECEIPT"
    )


def test_completed_output_is_terminal_not_resumable(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    output = tmp_path / "output"
    contract.run(output)
    with pytest.raises(audit.TemporalAuditError, match="terminal"):
        contract.run(output, resume=True)


def test_final_source_drift_removes_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _make_contract(tmp_path)
    original = audit.atomic_write_json
    changed = False

    def mutate(path: Path, value: Any) -> None:
        nonlocal changed
        original(path, value)
        if path.name == "main_temporal_validity_audit.outputs.json" and not changed:
            source = contract.files[1]
            before = source.stat()
            payload = bytearray(source.read_bytes())
            payload[-8] ^= 1
            source.write_bytes(payload)
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            changed = True

    monkeypatch.setattr(audit, "atomic_write_json", mutate)
    with pytest.raises(audit.TemporalAuditError):
        contract.run(tmp_path / "output")
    assert changed
    assert not (tmp_path / "output/main_temporal_validity_audit.outputs.json").exists()


def test_content_freeze_task_bridge_and_receipt_corruption_are_rejected(tmp_path: Path) -> None:
    contract = _make_contract(tmp_path)
    plan_path = contract.receipt.parent / "audit_plan.json"
    plan = _read(plan_path)
    plan["tasks"][0]["source_stat"]["ctime_ns"] += 1
    core = dict(plan)
    core.pop("plan_sha256")
    plan["plan_sha256"] = audit.sha256_bytes(audit.canonical_json_bytes(core))
    _write_content_json(plan_path, plan)
    receipt = _read(contract.receipt)
    for artifact in receipt["artifacts"]:
        if artifact["relative_path"] == "audit_plan.json":
            artifact["bytes"] = plan_path.stat().st_size
            artifact["sha256"] = _sha(plan_path)
    receipt["plan_sha256"] = plan["plan_sha256"]
    _write_content_json(contract.receipt, receipt)
    state_path = contract.receipt.parent / "run_state.json"
    state = _read(state_path)
    state["plan_sha256"] = plan["plan_sha256"]
    _write_content_json(state_path, state)
    config = yaml.safe_load(contract.config.read_text(encoding="utf-8"))
    config["parents"]["current_content_baseline"]["required_plan_sha256"] = plan["plan_sha256"]
    contract.config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    contract.config_sha = _sha(contract.config)
    with pytest.raises(audit.TemporalAuditError, match="rich identity differs"):
        contract.run(tmp_path / "output")


@pytest.mark.parametrize(
    "target",
    [
        "receipt",
        "plan",
        "report",
        "state",
        "duplicate_artifact",
        "report_fingerprint",
        "report_plan_crosslink",
    ],
)
def test_content_freeze_parent_rejects_noncanonical_or_duplicate_publication(
    tmp_path: Path,
    target: str,
) -> None:
    contract = _make_contract(tmp_path)
    parent = contract.receipt.parent
    if target == "receipt":
        contract.receipt.write_bytes(contract.receipt.read_bytes() + b" ")
        expected = "exact content-freeze published JSON bytes"
    elif target == "state":
        state_path = parent / "run_state.json"
        state_path.write_bytes(state_path.read_bytes() + b" ")
        expected = "exact content-freeze published JSON bytes"
    elif target == "duplicate_artifact":
        receipt = _read(contract.receipt)
        receipt["artifacts"].append(copy.deepcopy(receipt["artifacts"][0]))
        _write_content_json(contract.receipt, receipt)
        expected = "duplicate artifacts"
    elif target == "report_fingerprint":
        report_path = parent / "content_freeze_audit.json"
        report = _read(report_path)
        report["claim_boundary"] += " tampered"
        _write_content_json(report_path, report)
        receipt = _read(contract.receipt)
        for artifact in receipt["artifacts"]:
            if artifact["relative_path"] == report_path.name:
                artifact["bytes"] = report_path.stat().st_size
                artifact["sha256"] = _sha(report_path)
        _write_content_json(contract.receipt, receipt)
        expected = "report result fingerprint mismatch"
    elif target == "report_plan_crosslink":
        report_path = parent / "content_freeze_audit.json"
        report = _read(report_path)
        report["scope"]["max_files"] = 1
        _write_content_json(report_path, report)
        receipt = _read(contract.receipt)
        for artifact in receipt["artifacts"]:
            if artifact["relative_path"] == report_path.name:
                artifact["bytes"] = report_path.stat().st_size
                artifact["sha256"] = _sha(report_path)
        _write_content_json(contract.receipt, receipt)
        expected = "report/plan max-files mismatch"
    else:
        artifact_name = (
            "audit_plan.json" if target == "plan" else "content_freeze_audit.json"
        )
        artifact_path = parent / artifact_name
        artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
        receipt = _read(contract.receipt)
        for artifact in receipt["artifacts"]:
            if artifact["relative_path"] == artifact_name:
                artifact["bytes"] = artifact_path.stat().st_size
                artifact["sha256"] = _sha(artifact_path)
        _write_content_json(contract.receipt, receipt)
        expected = "exact content-freeze published JSON bytes"
    with pytest.raises(audit.TemporalAuditError, match=expected):
        contract.run(tmp_path / "output")


def test_contract_doc_excludes_runtime_outputs_and_available_at() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "runtime checkpoints" in text.lower()
    assert "excluded" in text.lower()
    assert "available_at" in text
    assert "PENDING_NOT_IN_THIS_COMPONENT" in text
