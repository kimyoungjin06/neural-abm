from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT / "scripts/prepare_pivot_feedback_holdout_access_seal_preflight_v1.py"
)
CREATOR_PATH = ROOT / "scripts/create_pivot_feedback_holdout_access_seal_v1.py"
VERIFIER_PATH = (
    ROOT / "scripts/verify_pivot_feedback_holdout_access_seal_receipt_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load(PREFLIGHT_PATH, "holdout_seal_preflight_test_module")
CREATOR = _load(CREATOR_PATH, "holdout_seal_creator_preflight_test_module")
VERIFIER = _load(VERIFIER_PATH, "holdout_seal_verifier_preflight_test_module")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "alpha.bin").write_bytes(b"alpha")
    (source / "nested").mkdir()
    (source / "nested/beta.parquet").write_bytes(b"PAR1opaque-fixturePAR1")
    (source / "unlisted.bin").write_bytes(b"must-not-be-discovered")
    controls = tmp_path / "operator-control"
    controls.mkdir()
    output = controls / PREFLIGHT.OUTPUT_BASENAME
    operator = tmp_path / "operator"
    operator.mkdir()
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    return {
        "source": source,
        "output": output,
        "operator": operator,
        "receipt": receipts / "receipt.json",
    }


def _prepare(paths: dict[str, Path], artifacts=None, **overrides):
    arguments = {
        "source_root": paths["source"],
        "artifact_relative_paths": artifacts
        or ["nested/beta.parquet", "alpha.bin"],
        "output_path": paths["output"],
        "authorized_at_utc": "2026-07-31T09:00:00Z",
        "attest_complete": True,
        "attest_private_source_root": True,
        "attest_no_row_parsing": True,
        "prior_open_event_count": 0,
        "confirmatory_or_holdout_rows_read": 0,
    }
    arguments.update(overrides)
    return PREFLIGHT.prepare_preflight(**arguments)


def test_preflight_tool_imports_no_row_data_library() -> None:
    tree = ast.parse(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})


def test_explicit_preflight_to_seal_to_independent_verification(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    preflight = _prepare(paths)

    assert paths["output"].is_file()
    assert [
        item["canonical_relative_path"] for item in preflight["artifact_entries"]
    ] == ["alpha.bin", "nested/beta.parquet"]
    assert preflight["artifact_count"] == 2
    assert preflight["artifact_bytes_total"] == sum(
        (paths["source"] / item["canonical_relative_path"]).stat().st_size
        for item in preflight["artifact_entries"]
    )
    assert all(
        item["expected_sha256"]
        == hashlib.sha256(
            (paths["source"] / item["canonical_relative_path"]).read_bytes()
        ).hexdigest()
        for item in preflight["artifact_entries"]
    )
    assert "unlisted.bin" not in paths["output"].read_text(encoding="utf-8")

    receipt = CREATOR.create_seal(
        preflight_path=paths["output"],
        source_root=paths["source"],
        operator_root=paths["operator"],
        receipt_path=paths["receipt"],
        seal_created_at_utc="2026-07-31T09:00:01Z",
        receipt_written_at_utc="2026-07-31T09:00:02Z",
    )
    result = VERIFIER.verify(
        receipt_path=paths["receipt"], operator_root=paths["operator"]
    )
    assert receipt["artifact_count"] == 2
    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["artifact_replay"]["artifact_count"] == 2
    assert result["access_accounting"]["operator_parquet_files_parsed"] == 0


@pytest.mark.parametrize("artifact", ["*.parquet", "nested/?eta.parquet", "nested"])
def test_preflight_rejects_wildcards_and_directory_inputs(
    tmp_path: Path, artifact: str
) -> None:
    paths = _fixture(tmp_path)
    expected = "wildcard artifact path" if artifact != "nested" else "nonregular artifact"
    with pytest.raises(PREFLIGHT.HoldoutSealPreflightError, match=expected):
        _prepare(paths, artifacts=[artifact])
    assert not paths["output"].exists()


def test_preflight_rejects_symlink_and_duplicate_paths(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["source"] / "link.bin").symlink_to(paths["source"] / "alpha.bin")
    with pytest.raises(PREFLIGHT.HoldoutSealPreflightError, match="symlink artifact"):
        _prepare(paths, artifacts=["link.bin"])

    paths = _fixture(tmp_path / "duplicate")
    with pytest.raises(PREFLIGHT.HoldoutSealPreflightError, match="duplicate artifact"):
        _prepare(paths, artifacts=["alpha.bin", "alpha.bin"])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"attest_complete": False}, "complete artifact-set attestation"),
        ({"attest_private_source_root": False}, "private source-root attestation"),
        ({"attest_no_row_parsing": False}, "no-row-parsing attestation"),
        ({"prior_open_event_count": 1}, "prior open events must be zero"),
        (
            {"confirmatory_or_holdout_rows_read": 1},
            "confirmatory or holdout row reads must be zero",
        ),
    ],
)
def test_preflight_requires_exact_zero_access_attestations(
    tmp_path: Path, override: dict, message: str
) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(PREFLIGHT.HoldoutSealPreflightError, match=message):
        _prepare(paths, **override)
    assert not paths["output"].exists()


def test_preflight_output_is_create_only(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = _prepare(paths)
    first_bytes = paths["output"].read_bytes()
    with pytest.raises(PREFLIGHT.HoldoutSealPreflightError, match="output exists"):
        _prepare(paths)
    assert paths["output"].read_bytes() == first_bytes
    assert json.loads(first_bytes)["artifact_count"] == first["artifact_count"]


def test_preflight_rejects_symlinked_source_root(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source_link = tmp_path / "source-link"
    source_link.symlink_to(paths["source"], target_is_directory=True)
    with pytest.raises(PREFLIGHT.HoldoutSealPreflightError, match="source root symlink"):
        _prepare(paths, source_root=source_link)
    assert not paths["output"].exists()


def test_preflight_output_permissions_are_read_only(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _prepare(paths)
    assert os.stat(paths["output"]).st_mode & 0o777 == 0o400
