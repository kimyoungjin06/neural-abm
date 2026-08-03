from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v3.py"
VERIFIER_PATH = REPO_ROOT / "scripts/verify_openalex_canonical_work_time_spine_v1.py"
DESIGN_PATH = (
    REPO_ROOT
    / "docs/research/openalex-canonical-work-time-spine-independent-verifier-v1.md"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module("canonical_spine_v3_independent_verifier_fixture", RUNNER_PATH)
VERIFIER = _load_module("canonical_spine_independent_verifier_v1_test", VERIFIER_PATH)


def _source_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(
                [
                    "https://openalex.org/W4",
                    "https://openalex.org/W1",
                    "https://openalex.org/W0",
                ],
                type=pa.string(),
            ),
            pa.array([None, "2020-01-01", "2021-02-03"], type=pa.string()),
            pa.array([None, 2020, 2021], type=pa.int64()),
        ],
        names=["id", "publication_date", "publication_year"],
    )
    pq.write_table(table, path, row_group_size=1)


def _closed_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    monkeypatch.setattr(RUNNER, "PARTITION_COUNT", 4)
    monkeypatch.setattr(
        RUNNER,
        "_enforce_resource_preflight",
        lambda *_args, **_kwargs: {
            "observed_free_disk_bytes": 10**12,
            "observed_available_ram_bytes": 10**12,
            "observed_nofile_soft_limit": 16_384,
        },
    )
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    _source_parquet(source / "main.parquet")
    output = tmp_path / "output"
    RUNNER.run_closed_fixture(source_root=source, output_dir=output)
    return source, output


def _surface(root: Path) -> dict[str, tuple[bytes, tuple[int, ...]]]:
    result: dict[str, tuple[bytes, tuple[int, ...]]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            info = path.lstat()
            result[path.relative_to(root).as_posix()] = (
                path.read_bytes(),
                (
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_nlink,
                    info.st_uid,
                    info.st_gid,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                ),
            )
    return result


def _published_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_full_parquet_verification_is_independent_read_only_and_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, output = _closed_output(monkeypatch, tmp_path)
    before = _surface(output)
    real_open = VERIFIER.os.open
    observed_flags: list[int] = []

    def readonly_open(path, flags, mode=0o777, *, dir_fd=None):
        observed_flags.append(flags)
        assert flags & os.O_ACCMODE == os.O_RDONLY
        assert not flags & (os.O_CREAT | os.O_TRUNC)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    decoded: list[int] = []
    real_validate_parquet = VERIFIER._validate_parquet

    def observed_validate_parquet(path, *, partition_id, **kwargs):
        decoded.append(partition_id)
        return real_validate_parquet(path, partition_id=partition_id, **kwargs)

    monkeypatch.setattr(VERIFIER.os, "open", readonly_open)
    monkeypatch.setattr(VERIFIER, "_validate_parquet", observed_validate_parquet)
    result = VERIFIER.verify_output(output)

    assert result["status"] == VERIFIER.VERIFIER_PASS
    assert result["verified_output_scope"] == VERIFIER.CLOSED_FIXTURE_SCOPE
    assert result["production_component_pass"] is False
    assert result["pass_g1"] is False
    assert result["partition_count"] == 4
    assert result["artifact_count"] == 17
    assert result["output_tree_unchanged"] is True
    assert decoded == [0, 1, 2, 3]
    assert observed_flags
    assert _surface(output) == before


def test_verifier_has_no_production_runner_or_frozen_config_dependency() -> None:
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        "audit_openalex_canonical_work_time_spine" in name for name in imported
    )
    assert "yaml" not in imported
    assert "importlib" not in imported
    assert "PRODUCTION_FROZEN_INPUTS" in source
    assert "loaded-code attestation" in source
    assert "source-projection replay" in source


def test_tiny_self_consistent_surface_cannot_be_relabelled_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, output = _closed_output(monkeypatch, tmp_path)
    plan = json.loads((output / "audit_plan.json").read_text(encoding="utf-8"))
    forged = copy.deepcopy(plan)
    forged.update(
        {
            "scope": VERIFIER.FULL_SCOPE,
            "schema_version": VERIFIER.SCHEMAS[VERIFIER.FULL_SCOPE]["plan"],
            "status_if_complete": VERIFIER.FULL_STATUS,
            "parents": copy.deepcopy(VERIFIER.PRODUCTION_PARENTS),
            "own_frozen_input_revalidation_required": True,
            "partition_count": VERIFIER.FULL_PARTITION_COUNT,
            "partition_formula": "work_id_numeric_mod_4096",
            "expected_receipt_artifact_count": 7 + 2 + 2 * 4096,
            "batch_size": VERIFIER.PRODUCTION_BATCH_SIZE,
            "partition_sort_chunk_rows": VERIFIER.PRODUCTION_SORT_CHUNK_ROWS,
            "resource_requirements": copy.deepcopy(VERIFIER.FULL_RESOURCE_REQUIREMENTS),
        }
    )
    identity = copy.deepcopy(plan["tool_rich_identity"])
    for name, (path_key, sha_key, identity_key) in {
        "config": (
            "config_relative_path",
            "config_sha256",
            "config_rich_identity",
        ),
        "tool": ("tool_relative_path", "tool_sha256", "tool_rich_identity"),
        "foundation_config": (
            "foundation_config_relative_path",
            "foundation_config_sha256",
            "foundation_config_rich_identity",
        ),
        "foundation_synthetic_runner": (
            "foundation_synthetic_runner_relative_path",
            "foundation_synthetic_runner_sha256",
            "foundation_synthetic_runner_rich_identity",
        ),
    }.items():
        forged[path_key] = VERIFIER.PRODUCTION_FROZEN_INPUTS[name]["path"]
        forged[sha_key] = VERIFIER.PRODUCTION_FROZEN_INPUTS[name]["sha256"]
        forged[identity_key] = copy.deepcopy(identity)
    forged["tasks"][0]["projected_schema"] = copy.deepcopy(
        VERIFIER.PRODUCTION_PROJECTED_SCHEMA
    )
    forged["tasks"][0]["projected_schema_sha256"] = hashlib.sha256(
        VERIFIER._canonical_json_bytes(VERIFIER.PRODUCTION_PROJECTED_SCHEMA)
    ).hexdigest()
    forged["source_tree_sha256"] = hashlib.sha256(
        VERIFIER._canonical_json_bytes(
            [
                {
                    "relative_path": forged["tasks"][0]["relative_path"],
                    "source_rich_identity": forged["tasks"][0]["source_rich_identity"],
                }
            ]
        )
    ).hexdigest()
    forged.pop("task_fingerprint_sha256")
    forged["task_fingerprint_sha256"] = hashlib.sha256(
        VERIFIER._canonical_json_bytes(forged)
    ).hexdigest()

    with pytest.raises(VERIFIER.VerificationError, match="full plan task count"):
        VERIFIER._validate_plan(forged)


@pytest.mark.parametrize(
    ("publication_date", "year", "forged_disposition"),
    [
        (date(1899, 12, 31), 1899, 2),
        (date(2026, 3, 31), 2026, 3),
        (date(2027, 1, 1), 2027, 4),
    ],
)
def test_frozen_disposition_is_recomputed_not_trusted(
    publication_date: date, year: int, forged_disposition: int
) -> None:
    records = np.zeros(1, dtype=VERIFIER.RECORD_DTYPE)
    records["work_id_numeric"] = 1
    records["publication_day"] = (publication_date - date(1970, 1, 1)).days
    records["publication_year"] = year
    records["temporal_disposition"] = forged_disposition
    with pytest.raises(VERIFIER.VerificationError, match="frozen classifier"):
        VERIFIER._consume_records(VERIFIER._empty_record_stats(1), records)


def test_design_freezes_read_only_independence_and_claim_boundaries() -> None:
    design = DESIGN_PATH.read_text(encoding="utf-8")
    for phrase in (
        "does not import or call the production v3 runner",
        "never starts a source scan, pilot, or production execution",
        "opens output files only with `O_RDONLY | O_NOFOLLOW`",
        "The independence is implementation-level only",
        "trusted-launcher or process-isolation evidence",
        "does not by itself promote a closed fixture or pilot",
    ):
        assert phrase in design
    assert "--parquet-validation hash-only" in design
    assert "distinct success status" in design


def test_hash_only_mode_is_explicitly_weaker_and_does_not_decode_parquet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, output = _closed_output(monkeypatch, tmp_path)
    monkeypatch.setattr(
        VERIFIER,
        "_validate_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hash-only mode decoded Parquet")
        ),
    )
    result = VERIFIER.verify_output(output, parquet_validation="hash-only")
    assert result["status"] == VERIFIER.VERIFIER_HASH_ONLY_PASS
    assert result["parquet_validation"] == "hash-only"


def test_receipt_hash_tamper_fails_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, output = _closed_output(monkeypatch, tmp_path)
    report = output / RUNNER.FIXTURE_BASENAMES["report"]
    report.write_bytes(report.read_bytes() + b" ")
    before = _surface(output)
    with pytest.raises(VERIFIER.VerificationError, match="artifact differs"):
        VERIFIER.verify_output(output)
    assert _surface(output) == before


def test_state_cross_binding_tamper_fails_even_with_updated_receipt_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, output = _closed_output(monkeypatch, tmp_path)
    state_path = output / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["result_fingerprint_sha256"] = "0" * 64
    state_payload = _published_json(state)
    state_path.write_bytes(state_payload)
    state_path.chmod(0o600)

    receipt_path = output / RUNNER.FIXTURE_BASENAMES["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["run_state.json"] = {
        "relative_path": "run_state.json",
        "bytes": len(state_payload),
        "sha256": hashlib.sha256(state_payload).hexdigest(),
    }
    receipt_path.write_bytes(_published_json(receipt))
    receipt_path.chmod(0o600)
    before = _surface(output)

    with pytest.raises(VERIFIER.VerificationError, match="terminal state"):
        VERIFIER.verify_output(output)
    assert _surface(output) == before


def test_noncanonical_json_and_symlink_are_rejected_without_repair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, output = _closed_output(monkeypatch, tmp_path)
    receipt_path = output / RUNNER.FIXTURE_BASENAMES["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    before = _surface(output)
    with pytest.raises(VERIFIER.VerificationError, match="not canonical"):
        VERIFIER.verify_output(output)
    assert _surface(output) == before

    receipt_path.write_bytes(_published_json(receipt))
    markdown = output / RUNNER.FIXTURE_BASENAMES["markdown"]
    payload = markdown.read_bytes()
    target = tmp_path / "outside.md"
    target.write_bytes(payload)
    target.chmod(0o600)
    markdown.unlink()
    markdown.symlink_to(target)
    with pytest.raises(VERIFIER.VerificationError, match="symlink"):
        VERIFIER.verify_output(output)
    assert markdown.is_symlink()


def test_privacy_guard_rejects_absolute_operator_paths() -> None:
    with pytest.raises(VERIFIER.VerificationError, match="operator path"):
        VERIFIER._assert_redacted(
            {"operator_paths_redacted": True, "debug_path": "/home/operator/data"}
        )


def test_cli_emits_result_only_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, output = _closed_output(monkeypatch, tmp_path)
    before = _surface(output)
    assert VERIFIER.main(["--output-root", str(output)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == VERIFIER.VERIFIER_PASS
    assert _surface(output) == before
