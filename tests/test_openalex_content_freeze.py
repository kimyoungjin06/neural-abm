from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_content_freeze.py"
CONFIG_PATH = REPO_ROOT / "experiments/configs/openalex_g1_content_freeze_v0.yaml"
PARENT_MANIFEST = (
    REPO_ROOT
    / "experiments/runs/20260716_openalex_g1_source_inventory/source_manifest.json"
)

spec = importlib.util.spec_from_file_location(
    "test_openalex_content_freeze_module", SCRIPT_PATH
)
assert spec is not None and spec.loader is not None
freeze = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = freeze
spec.loader.exec_module(freeze)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contains_absolute_path(value: Any, needles: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item, needles) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item, needles) for item in value)
    if isinstance(value, str):
        return any(needle in value for needle in needles)
    return False


def _synthetic_contract(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "synthetic_openalex_root"
    payloads = {
        "works/part-0.parquet": b"first synthetic parquet payload\n",
        "works/part-1.parquet": b"second synthetic parquet payload\n",
        "topics/base-0.parquet": b"topic relation payload\n",
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    tables: list[dict[str, Any]] = []
    physical_rows: list[tuple[str, int, int]] = []
    for table_name in ("works", "topics"):
        files = []
        for path in sorted((root / table_name).iterdir(), key=lambda item: item.name):
            info = path.stat()
            relative = path.relative_to(root).as_posix()
            physical_rows.append((relative, info.st_size, info.st_mtime_ns))
            files.append(
                {
                    "relative_path": relative,
                    "bytes": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                    "content_sha256": None,
                }
            )
        tables.append(
            {
                "name": table_name,
                "file_count": len(files),
                "bytes": sum(item["bytes"] for item in files),
                "files": files,
            }
        )
    inventory_fingerprint = freeze.snapshot_fingerprint(
        physical_rows, include_mtime=False
    )
    stability_fingerprint = freeze.snapshot_fingerprint(
        physical_rows, include_mtime=True
    )
    manifest = {
        "schema_version": "openalex_source_inventory.v1",
        "manifest_id": "synthetic_source_inventory_v0",
        "status": "INVENTORY_ONLY",
        "pass_metadata": False,
        "pass_g1": False,
        "root": {
            "id": root.name,
            "table_count": len(tables),
            "parquet_file_count": len(physical_rows),
            "bytes": sum(item[1] for item in physical_rows),
            "inventory_fingerprint_sha256": inventory_fingerprint,
            "stability_fingerprint_sha256": stability_fingerprint,
        },
        "execution": {
            "content_sha256": False,
            "path_redacted": True,
        },
        "tables": tables,
    }
    manifest_path = tmp_path / "synthetic_parent_manifest.json"
    _write_json(manifest_path, manifest)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = config["parent_inventory"]
    parent.update(
        {
            "manifest_relative_path": manifest_path.name,
            "manifest_sha256": _sha256(manifest_path),
            "required_manifest_id": manifest["manifest_id"],
            "source_root_id": root.name,
            "expected_table_count": len(tables),
            "expected_parquet_file_count": len(physical_rows),
            "expected_bytes": manifest["root"]["bytes"],
            "inventory_fingerprint_sha256": inventory_fingerprint,
            "stability_fingerprint_sha256": stability_fingerprint,
        }
    )
    config["source"].update(
        {"root_env": "SYNTHETIC_OPENALEX_ROOT", "expected_root_id": root.name}
    )
    config["execution"].update(
        {
            "read_chunk_bytes": 7,
            "progress_every_files": 1,
            "progress_every_seconds": 1,
        }
    )
    config_path = tmp_path / "synthetic_content_freeze.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, manifest_path, config_path


def _run(
    tmp_path: Path,
    *,
    output_name: str = "output",
    max_files: int | None = None,
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / output_name
    report, outputs = freeze.run_audit(
        config_path=config,
        inventory_manifest_path=manifest,
        source_root=root,
        output_dir=output,
        max_files=max_files,
        resume=resume,
    )
    return report, outputs, output


def test_production_contract_binds_footer_inventory_without_content_hashes() -> None:
    config = freeze.read_config(CONFIG_PATH)
    parent = config["parent_inventory"]
    assert parent["manifest_sha256"] == _sha256(PARENT_MANIFEST)
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["root"]["parquet_file_count"] == 21_019
    assert manifest["root"]["bytes"] == 988_404_537_807
    assert manifest["root"]["inventory_fingerprint_sha256"] == (
        "1bee200c61e746ee774715de2521550cb2c7a6f2bf08e0f8b06f43a39c9ec473"
    )
    content_hashes = [
        record["content_sha256"]
        for table in manifest["tables"]
        for record in table["files"]
    ]
    assert len(content_hashes) == 21_019
    assert set(content_hashes) == {None}


def test_config_snapshot_hashes_the_exact_bytes_it_parses(tmp_path: Path) -> None:
    _root, _manifest, config = _synthetic_contract(tmp_path)
    raw = config.read_bytes()
    snapshot = freeze.read_config_snapshot(config)

    assert snapshot.sha256 == hashlib.sha256(raw).hexdigest()
    assert snapshot.value == yaml.safe_load(raw.decode("utf-8"))
    assert snapshot.source_stat["bytes"] == len(raw)


def test_json_snapshot_rejects_path_replacement_during_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.json"
    original = {"generation": "original", "value": 1}
    replacement = {"generation": "replacement", "value": 2}
    original_bytes = json.dumps(original, sort_keys=True).encode("utf-8")
    path.write_bytes(original_bytes)
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_text(json.dumps(replacement), encoding="utf-8")
    real_read = os.read
    replaced = False

    def replace_path_then_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        if not replaced:
            replacement_path.replace(path)
            replaced = True
        return real_read(descriptor, size)

    monkeypatch.setattr(freeze.os, "read", replace_path_then_read)
    with pytest.raises(freeze.ContentFreezeError, match="changed while being read"):
        freeze.read_json_snapshot(path, field="test input")

    assert replaced is True
    assert json.loads(path.read_text(encoding="utf-8")) == replacement


def test_full_synthetic_run_creates_current_baseline_with_bounded_claims(
    tmp_path: Path,
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*.parquet")
    }
    output = tmp_path / "full-output"
    report, outputs = freeze.run_audit(
        config_path=config,
        inventory_manifest_path=manifest,
        source_root=root,
        output_dir=output,
    )

    assert report["status"] == freeze.FULL_SUCCESS_STATUS
    assert report["pass_current_content_baseline_component"] is True
    assert report["pass_g1"] is False
    assert report["scope"]["full_scope"] is True
    assert report["gate_status"] == freeze.EXPECTED_FULL_GATES
    assert report["content_baseline"]["historical_identity_established"] is False
    assert report["content_baseline"]["prior_component_run_identity_established"] is False
    assert report["source_snapshot"]["exactly_equal_to_plan_start"] is True
    assert report["checkpoint_evidence"]["checkpoint_files"] == 3
    assert report["checkpoint_evidence"]["resume_rehashes_completed_content"] is False
    assert report["checkpoint_evidence"]["checkpoint_digest_authenticated"] is False
    assert report["checkpoint_evidence"]["checkpoint_integrity_scope"] == (
        "ACCIDENTAL_CORRUPTION_ONLY"
    )
    assert report["checkpoint_evidence"]["trusted_output_directory_required"] is True
    assert report["checkpoint_evidence"]["output_trust_precondition"] == (
        "VERIFIED_OWNER_AND_ANCESTOR_MODE_CHAIN"
    )
    assert report["checkpoint_evidence"]["disk_revalidated_before_report"] is True
    assert (
        report["checkpoint_evidence"][
            "disk_revalidated_in_final_output_validation"
        ]
        is True
    )
    assert "unkeyed self-digest" in report["checkpoint_trust_limitation"]
    assert report["provenance"]["result_fingerprint_scope"] == (
        "PATH_SIZE_CONTENT_AND_CONTRACT"
    )
    assert report["provenance"]["tool_attestation_scope"] == (
        "CURRENT_TOOL_FILE_SNAPSHOT_NOT_LOADED_CODE_ATTESTATION"
    )
    assert report["publication"] == {
        "complete_state_written_last": True,
        "completion_contract": (
            "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
        ),
        "failure_removes_pass_report_markdown_and_manifest": True,
        "run_state_bound_separately_from_outputs_artifacts": True,
    }
    assert outputs["completion_contract"] == (
        "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
    )
    assert outputs["run_state_bound_separately"] is True
    assert outputs["path_redacted"] is True
    assert (output / "content_freeze_audit.json").is_file()
    assert (output / "content_freeze_audit.md").is_file()
    assert (output / "content_freeze_audit.outputs.json").is_file()
    assert outputs["checkpoint_tree"]["tree_sha256"] == report[
        "checkpoint_evidence"
    ]["checkpoint_tree_sha256"]
    assert outputs["content_tree_sha256"] == report["content_baseline"][
        "content_tree_sha256"
    ]
    for artifact in outputs["artifacts"]:
        path = output / artifact["relative_path"]
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["sha256"] == _sha256(path)
    assert "run_state.json" not in {
        artifact["relative_path"] for artifact in outputs["artifacts"]
    }
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "COMPLETE"
    assert (output / "content_freeze_audit.md").read_text(
        encoding="utf-8"
    ) == freeze.render_markdown(report)
    assert before == {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*.parquet")
    }


def test_max_files_is_always_pilot_even_when_it_selects_every_file(
    tmp_path: Path,
) -> None:
    report, _, _ = _run(tmp_path, max_files=99)
    assert report["status"] == freeze.PILOT_STATUS
    assert report["pass_current_content_baseline_component"] is False
    assert report["scope"]["full_scope"] is False
    assert (
        report["gate_status"]["current_content_baseline_component"]
        == "NOT_ESTABLISHED_PILOT"
    )
    assert report["pass_g1"] is False


def test_task_plan_content_and_result_fingerprints_are_deterministic(
    tmp_path: Path,
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output_a = tmp_path / "output-a"
    report_a, _ = freeze.run_audit(
        config_path=config,
        inventory_manifest_path=manifest,
        source_root=root,
        output_dir=output_a,
    )
    output_b = tmp_path / "output-b"
    report_b, _ = freeze.run_audit(
        config_path=config,
        inventory_manifest_path=manifest,
        source_root=root,
        output_dir=output_b,
    )
    plan_a = json.loads((output_a / "audit_plan.json").read_text(encoding="utf-8"))
    plan_b = json.loads((output_b / "audit_plan.json").read_text(encoding="utf-8"))
    assert plan_a == plan_b
    assert plan_a["task_fingerprint_sha256"] == plan_b["task_fingerprint_sha256"]
    assert plan_a["plan_sha256"] == plan_b["plan_sha256"]
    assert report_a["content_baseline"]["content_tree_sha256"] == report_b[
        "content_baseline"
    ]["content_tree_sha256"]
    assert report_a["provenance"]["result_fingerprint_sha256"] == report_b[
        "provenance"
    ]["result_fingerprint_sha256"]


def test_file_error_pauses_and_resume_reuses_completed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "resume-output"
    original = freeze.hash_task
    calls = 0

    def flaky(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise freeze.ContentFreezeError("synthetic read failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(freeze, "hash_task", flaky)
    with pytest.raises(freeze.ContentFreezeError, match="synthetic read failure"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
        )
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == freeze.PAUSED_STATUS
    assert state["completed_files"] == 1
    assert len(list((output / "checkpoints").rglob("*.json"))) == 1

    monkeypatch.setattr(freeze, "hash_task", original)
    report, _ = freeze.run_audit(
        config_path=config,
        inventory_manifest_path=manifest,
        source_root=root,
        output_dir=output,
        resume=True,
    )
    assert report["status"] == freeze.FULL_SUCCESS_STATUS
    assert report["checkpoint_evidence"]["resumed_files"] == 1
    assert report["checkpoint_evidence"]["newly_hashed_files"] == 2


def test_final_reread_rejects_checkpoint_corrupted_after_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "corrupt-after-checkpoint-write"
    original = freeze.atomic_write_json
    corrupted = False

    def corrupt_checkpoint_after_write(path: Path, value: Any) -> None:
        nonlocal corrupted
        original(path, value)
        if not corrupted and path.parent.parent.name == "checkpoints":
            path.write_text("{invalid checkpoint", encoding="utf-8")
            corrupted = True

    monkeypatch.setattr(freeze, "atomic_write_json", corrupt_checkpoint_after_write)
    with pytest.raises(freeze.ContentFreezeError, match="final checkpoint"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            max_files=1,
        )

    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert corrupted is True
    assert state["status"] == freeze.PAUSED_STATUS
    assert not (output / "content_freeze_audit.json").exists()
    assert not (output / "content_freeze_audit.outputs.json").exists()


def test_final_validator_rejects_checkpoint_corrupted_after_outputs_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "corrupt-before-final-validator"
    original = freeze.atomic_write_json
    corrupted = False

    def corrupt_after_outputs(path: Path, value: Any) -> None:
        nonlocal corrupted
        original(path, value)
        if path.name == "content_freeze_audit.outputs.json":
            checkpoint = next((output / "checkpoints").rglob("*.json"))
            checkpoint.write_text("{invalid checkpoint", encoding="utf-8")
            corrupted = True

    monkeypatch.setattr(freeze, "atomic_write_json", corrupt_after_outputs)
    with pytest.raises(freeze.ContentFreezeError, match="final checkpoint"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            max_files=1,
        )

    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert corrupted is True
    assert state["status"] == freeze.PAUSED_STATUS
    assert not (output / "content_freeze_audit.json").exists()
    assert not (output / "content_freeze_audit.md").exists()
    assert not (output / "content_freeze_audit.outputs.json").exists()


def test_paused_error_state_omits_operator_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "redacted-error-output"
    secret = "/hdd/private/operator-only/source.parquet"

    def leaking_error(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise FileNotFoundError(errno.ENOENT, "missing", secret)

    monkeypatch.setattr(freeze, "hash_task", leaking_error)
    with pytest.raises(FileNotFoundError):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            max_files=1,
        )

    state_text = (output / "run_state.json").read_text(encoding="utf-8")
    state = json.loads(state_text)
    assert state["status"] == freeze.PAUSED_STATUS
    assert state["path_redacted"] is True
    assert state["error"] == {
        "detail_omitted": True,
        "message": "audit paused; exception detail omitted to preserve path redaction",
        "type": "FileNotFoundError",
    }
    for forbidden in (secret, "/hdd/private", str(root), str(output), str(config)):
        assert forbidden not in state_text


def test_outputs_write_failure_removes_pass_publication_and_pauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "outputs-enospc"
    original = freeze.atomic_write_json

    def fail_outputs_manifest(path: Path, value: Any) -> None:
        if path.name == "content_freeze_audit.outputs.json":
            raise OSError(errno.ENOSPC, "synthetic no space")
        original(path, value)

    monkeypatch.setattr(freeze, "atomic_write_json", fail_outputs_manifest)
    with pytest.raises(OSError) as captured:
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            max_files=1,
        )
    assert captured.value.errno == errno.ENOSPC

    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == freeze.PAUSED_STATUS
    for name in (
        "content_freeze_audit.json",
        "content_freeze_audit.md",
        "content_freeze_audit.outputs.json",
    ):
        assert not (output / name).exists()


def test_complete_state_write_failure_removes_pass_publication_and_pauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "complete-state-enospc"
    original = freeze.atomic_write_json

    def fail_complete_state(path: Path, value: Any) -> None:
        if path.name == "run_state.json" and value.get("status") == "COMPLETE":
            raise OSError(errno.ENOSPC, "synthetic complete-state failure")
        original(path, value)

    monkeypatch.setattr(freeze, "atomic_write_json", fail_complete_state)
    with pytest.raises(OSError) as captured:
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            max_files=1,
        )
    assert captured.value.errno == errno.ENOSPC

    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == freeze.PAUSED_STATUS
    for name in (
        "content_freeze_audit.json",
        "content_freeze_audit.md",
        "content_freeze_audit.outputs.json",
    ):
        assert not (output / name).exists()


@pytest.mark.parametrize(
    "mutation",
    ["schema", "task", "source_identity", "digest"],
)
def test_resume_rejects_stale_or_corrupted_checkpoint(
    tmp_path: Path, mutation: str
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / f"stale-{mutation}"
    freeze.run_audit(
        config_path=config,
        inventory_manifest_path=manifest,
        source_root=root,
        output_dir=output,
        max_files=1,
    )
    checkpoint_path = next((output / "checkpoints").rglob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        checkpoint["schema_version"] = "stale"
    elif mutation == "task":
        checkpoint["task_fingerprint_sha256"] = "0" * 64
    elif mutation == "source_identity":
        checkpoint["source_identity_after"]["ctime_ns"] += 1
    else:
        checkpoint["content_sha256"] = "0" * 64
    if mutation != "digest":
        core = dict(checkpoint)
        core.pop("checkpoint_sha256")
        checkpoint["checkpoint_sha256"] = freeze.sha256_bytes(
            freeze.canonical_json_bytes(core)
        )
    _write_json(checkpoint_path, checkpoint)

    with pytest.raises(freeze.ContentFreezeError):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            max_files=1,
            resume=True,
        )


def test_resume_rejects_same_size_restored_mtime_mutation(tmp_path: Path) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "restored-mtime-output"
    freeze.run_audit(
        config_path=config,
        inventory_manifest_path=manifest,
        source_root=root,
        output_dir=output,
        max_files=1,
    )
    plan = json.loads((output / "audit_plan.json").read_text(encoding="utf-8"))
    source = root / plan["tasks"][0]["relative_path"]
    before = source.stat()
    payload = source.read_bytes()
    source.write_bytes(payload[::-1])
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    assert source.stat().st_ctime_ns != before.st_ctime_ns

    with pytest.raises(freeze.ContentFreezeError, match="resume plan differs"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            max_files=1,
            resume=True,
        )


def test_pre_post_stat_detects_same_size_mutation_during_hash(
    tmp_path: Path,
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    loaded = freeze.read_config_snapshot(config)
    parent, tasks, snapshot, manifest_snapshot = freeze.validate_parent_inventory(
        config=loaded.value,
        manifest_path=manifest,
        source_root=freeze.validate_source_root(root, expected_id=root.name),
    )
    _, tool_sha256, _ = freeze.read_regular_file_bytes(
        SCRIPT_PATH, field="audit tool"
    )
    plan = freeze.build_plan(
        config=loaded.value,
        config_path=config,
        config_sha256=loaded.sha256,
        manifest_path=manifest,
        manifest_sha256=manifest_snapshot.sha256,
        manifest=parent,
        all_tasks=tasks,
        snapshot=snapshot,
        max_files=1,
        tool_sha256=tool_sha256,
    )
    task = freeze.task_from_mapping(plan["tasks"][0])

    def mutate(path: Path) -> None:
        before = path.stat()
        path.write_bytes(path.read_bytes()[::-1])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(freeze.ContentFreezeError, match="source changed during hash"):
        freeze.hash_task(
            task,
            source_root=root,
            task_fingerprint_sha256=plan["task_fingerprint_sha256"],
            plan_sha256=plan["plan_sha256"],
            chunk_bytes=7,
            after_hash_hook=mutate,
        )


def test_final_full_snapshot_detects_mutation_after_last_checkpoint(
    tmp_path: Path,
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "final-mutation-output"
    mutated = False

    def progress(
        completed: int,
        total: int,
        _completed_bytes: int,
        _task: Any,
        _resumed: bool,
    ) -> None:
        nonlocal mutated
        if completed == total:
            target = root / "works/part-0.parquet"
            before = target.stat()
            target.write_bytes(target.read_bytes()[::-1])
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
            mutated = True

    with pytest.raises(freeze.ContentFreezeError, match="final source stat differs"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            progress=progress,
        )
    assert mutated is True
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == freeze.PAUSED_STATUS


def test_final_full_snapshot_rejects_new_undeclared_source_entry(
    tmp_path: Path,
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "final-extra-entry-output"

    def progress(
        completed: int,
        total: int,
        _completed_bytes: int,
        _task: Any,
        _resumed: bool,
    ) -> None:
        if completed == total:
            (root / "works/unexpected.parquet").write_bytes(b"unexpected\n")

    with pytest.raises(freeze.ContentFreezeError, match="source table entries differ"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            progress=progress,
        )


def test_source_file_and_intermediate_directory_symlinks_are_rejected(
    tmp_path: Path,
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    target = root / "works/part-0.parquet"
    external = tmp_path / "external.parquet"
    external.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(freeze.ContentFreezeError, match="symlink"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=tmp_path / "symlink-file-output",
        )

    root, manifest, config = _synthetic_contract(tmp_path / "second")
    works = root / "works"
    real_works = tmp_path / "real-works"
    works.rename(real_works)
    works.symlink_to(real_works, target_is_directory=True)
    with pytest.raises(freeze.ContentFreezeError, match="symlink"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=tmp_path / "symlink-dir-output",
        )


def test_output_containment_symlink_and_lock_are_rejected(tmp_path: Path) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    contained = root / "forbidden-output"
    with pytest.raises(freeze.ContentFreezeError, match="inside source root"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=contained,
        )
    assert not contained.exists()

    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(freeze.ContentFreezeError, match="symlink"):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=linked_output,
        )

    locked = tmp_path / "locked-output"
    locked.mkdir(mode=0o700)
    lock_path = locked / "audit.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(freeze.ContentFreezeError, match="holds the output lock"):
            freeze.run_audit(
                config_path=config,
                inventory_manifest_path=manifest,
                source_root=root,
                output_dir=locked,
            )


@pytest.mark.parametrize("unsafe_scope", ["output", "ancestor"])
def test_untrusted_output_write_permissions_are_rejected(
    tmp_path: Path, unsafe_scope: str
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    if unsafe_scope == "output":
        output = tmp_path / "shared-output"
        output.mkdir(mode=0o700)
        output.chmod(0o777)
        expected = "output directory is group/world writable"
    else:
        shared_parent = tmp_path / "shared-parent"
        shared_parent.mkdir(mode=0o700)
        shared_parent.chmod(0o777)
        output = shared_parent / "private-leaf"
        expected = "writable output ancestor lacks sticky protection"

    with pytest.raises(freeze.ContentFreezeError, match=expected):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            max_files=1,
        )


@pytest.mark.parametrize("mutation", ["partial", "empty_directory", "nested_symlink"])
def test_resume_rejects_unbound_checkpoint_tree_entries(
    tmp_path: Path, mutation: str
) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / f"checkpoint-tree-{mutation}"
    freeze.run_audit(
        config_path=config,
        inventory_manifest_path=manifest,
        source_root=root,
        output_dir=output,
        max_files=1,
    )
    checkpoint_root = output / "checkpoints"
    if mutation == "partial":
        prefix = next(path for path in checkpoint_root.iterdir() if path.is_dir())
        (prefix / ".stale.partial").write_text("partial", encoding="utf-8")
        expected = "unexpected checkpoint entry"
    elif mutation == "empty_directory":
        (checkpoint_root / "ff").mkdir()
        expected = "unexpected checkpoint directory"
    else:
        external = tmp_path / "external-checkpoints"
        external.mkdir()
        (checkpoint_root / "ff").symlink_to(external, target_is_directory=True)
        expected = "checkpoint directory is not a regular directory"
    with pytest.raises(freeze.ContentFreezeError, match=expected):
        freeze.run_audit(
            config_path=config,
            inventory_manifest_path=manifest,
            source_root=root,
            output_dir=output,
            max_files=1,
            resume=True,
        )


def test_outputs_and_checkpoints_are_path_redacted(tmp_path: Path) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "redacted-output"
    freeze.run_audit(
        config_path=config,
        inventory_manifest_path=manifest,
        source_root=root,
        output_dir=output,
        max_files=2,
    )
    needles = (str(root), str(output), str(tmp_path))
    for path in output.rglob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        assert not _contains_absolute_path(value, needles), path


def test_cli_pilot_returns_nonzero_under_require_pass(tmp_path: Path) -> None:
    root, manifest, config = _synthetic_contract(tmp_path)
    output = tmp_path / "cli-pilot"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config),
            "--inventory-manifest",
            str(manifest),
            "--source-root",
            str(root),
            "--output-dir",
            str(output),
            "--max-files",
            "1",
            "--require-pass",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout.strip())
    assert payload["status"] == freeze.PILOT_STATUS
    assert "content freeze: 1/1 files" in completed.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "claim",
        "limitation",
        "checkpoint_trust",
        "workers",
        "reuse",
        "progress_files",
        "progress_seconds",
    ],
)
def test_config_contract_drift_is_rejected(tmp_path: Path, mutation: str) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if mutation == "claim":
        config["claim_boundary"] += " drift"
    elif mutation == "limitation":
        config["same_size_restored_mtime_limitation"] += " drift"
    elif mutation == "checkpoint_trust":
        config["checkpoint_trust_limitation"] += " drift"
    elif mutation == "workers":
        config["execution"]["workers"] = 2
    elif mutation == "progress_files":
        config["execution"]["progress_every_files"] = 0
    elif mutation == "progress_seconds":
        config["execution"]["progress_every_seconds"] = 0
    else:
        config["execution"]["checkpoint_reuse"] = "trust_size_only"
    path = tmp_path / f"mutated-{mutation}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(freeze.ContentFreezeError):
        freeze.read_config(path)
