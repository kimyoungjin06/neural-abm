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


CREATOR = _load(CREATOR_PATH, "holdout_seal_creator_test_module")
VERIFIER = _load(VERIFIER_PATH, "holdout_seal_receipt_verifier_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "alpha.bin").write_bytes(b"alpha\x00payload")
    (source / "nested").mkdir()
    (source / "nested/beta.parquet").write_bytes(b"PAR1fixture-bytes-onlyPAR1")
    operator = tmp_path / "operator"
    operator.mkdir()
    receipt_parent = tmp_path / "receipts"
    receipt_parent.mkdir()
    receipt = receipt_parent / "holdout_receipt.json"
    preflight = tmp_path / "preflight.json"
    entries = []
    for relative in ("alpha.bin", "nested/beta.parquet"):
        path = source / relative
        entries.append(
            {
                "canonical_relative_path": relative,
                "expected_byte_size": path.stat().st_size,
                "expected_sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": CREATOR.PREFLIGHT_SCHEMA,
        "status": CREATOR.PREFLIGHT_STATUS,
        "partition_contract": CREATOR.EXPECTED_PARTITION,
        **{
            f"{key}_binding": {
                "relative_path": value["relative_path"],
                "sha256": value["sha256"],
            }
            for key, value in CREATOR.REPOSITORY_BINDINGS.items()
        },
        "artifact_count": len(entries),
        "artifact_bytes_total": sum(item["expected_byte_size"] for item in entries),
        "artifact_entries": entries,
        "attestations": {
            "explicit_artifact_set_complete": True,
            "paths_enumerated_under_v54_authorization": True,
            "hashes_computed_under_v54_authorization": True,
            "source_root_private_and_access_controlled": True,
            "final_operator_path_clear": True,
            "receipt_path_clear": True,
            "row_parsing_or_analysis_performed": False,
            "prior_open_event_count": 0,
            "confirmatory_or_holdout_rows_read": 0,
        },
    }
    preflight.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "source": source,
        "operator": operator,
        "receipt": receipt,
        "preflight": preflight,
    }


def _create(paths: dict[str, Path]) -> dict:
    return CREATOR.create_seal(
        preflight_path=paths["preflight"],
        source_root=paths["source"],
        operator_root=paths["operator"],
        receipt_path=paths["receipt"],
        seal_created_at_utc="2026-07-31T08:00:00Z",
        receipt_written_at_utc="2026-07-31T08:00:01Z",
    )


def test_creator_and_verifier_import_no_row_data_libraries() -> None:
    for path in (CREATOR_PATH, VERIFIER_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})


def test_closed_fixture_creation_and_independent_rehash_pass(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = _create(paths)

    assert receipt["status"] == CREATOR.RECEIPT_STATUS
    assert receipt["receipt_written_last"] is True
    assert receipt["seal_opened"] is False
    assert receipt["confirmatory_or_holdout_rows_read"] == 0
    assert receipt["artifact_count"] == 2
    assert paths["receipt"].is_file()
    final = paths["operator"] / CREATOR.FINAL_OPERATOR_RELATIVE_PATH
    assert (final / "manifest.json").is_file()
    assert (final / "artifacts/alpha.bin").read_bytes() == b"alpha\x00payload"
    assert (final / "artifacts/nested/beta.parquet").read_bytes() == (
        b"PAR1fixture-bytes-onlyPAR1"
    )

    result = VERIFIER.verify(
        receipt_path=paths["receipt"], operator_root=paths["operator"]
    )
    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["artifact_replay"] == {
        "artifact_count": 2,
        "artifact_bytes_total": receipt["artifact_bytes_total"],
        "all_artifacts_rehashed": True,
        "sealed_tree_exact": True,
        "manifest_fingerprint_valid": True,
    }
    assert result["access_accounting"]["operator_parquet_files_parsed"] == 0
    assert result["access_accounting"]["confirmatory_or_holdout_rows_read"] == 0
    assert result["authorization"]["seal_opening"] == "NOT_AUTHORIZED"


def test_creator_rejects_hash_drift_and_removes_staging(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["source"] / "alpha.bin").write_bytes(b"changed-after-preflight")

    with pytest.raises(CREATOR.HoldoutSealCreationError, match="artifact size drift"):
        _create(paths)

    assert not paths["receipt"].exists()
    assert not (
        paths["operator"] / CREATOR.FINAL_OPERATOR_RELATIVE_PATH
    ).exists()
    assert not list(paths["operator"].rglob(".holdout-seal-v1.staging-*"))


def test_creator_rejects_symlinked_artifact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    target = paths["source"] / "real.bin"
    target.write_bytes(b"alpha\x00payload")
    (paths["source"] / "alpha.bin").unlink()
    (paths["source"] / "alpha.bin").symlink_to(target)

    with pytest.raises(CREATOR.HoldoutSealCreationError, match="symlink artifact"):
        _create(paths)

    assert not paths["receipt"].exists()


def test_creator_rejects_unsorted_or_duplicate_allowlist(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    payload = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    payload["artifact_entries"] = list(reversed(payload["artifact_entries"]))
    paths["preflight"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        CREATOR.HoldoutSealCreationError, match="artifact allowlist must be sorted"
    ):
        _create(paths)

    paths = _fixture(tmp_path / "duplicate")
    payload = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    payload["artifact_entries"].append(dict(payload["artifact_entries"][0]))
    payload["artifact_count"] += 1
    payload["artifact_bytes_total"] += payload["artifact_entries"][0][
        "expected_byte_size"
    ]
    paths["preflight"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CREATOR.HoldoutSealCreationError, match="duplicate artifact"):
        _create(paths)


def test_creator_is_create_only_for_final_tree_and_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = _create(paths)
    first_receipt_bytes = paths["receipt"].read_bytes()

    with pytest.raises(
        CREATOR.HoldoutSealCreationError, match="final seal path already exists"
    ):
        _create(paths)

    assert paths["receipt"].read_bytes() == first_receipt_bytes
    assert json.loads(first_receipt_bytes)["manifest_sha256"] == first["manifest_sha256"]


def test_creator_rejects_receipt_timestamp_before_seal(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(
        CREATOR.HoldoutSealCreationError,
        match="receipt timestamp precedes seal",
    ):
        CREATOR.create_seal(
            preflight_path=paths["preflight"],
            source_root=paths["source"],
            operator_root=paths["operator"],
            receipt_path=paths["receipt"],
            seal_created_at_utc="2026-07-31T08:00:01Z",
            receipt_written_at_utc="2026-07-31T08:00:00Z",
        )
    assert not paths["receipt"].exists()


def test_verifier_rejects_tampered_sealed_artifact(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _create(paths)
    sealed = (
        paths["operator"]
        / CREATOR.FINAL_OPERATOR_RELATIVE_PATH
        / "artifacts/alpha.bin"
    )
    os.chmod(sealed, 0o600)
    sealed.write_bytes(b"tampered")

    with pytest.raises(VERIFIER.HoldoutSealVerificationError, match="artifact size drift"):
        VERIFIER.verify(
            receipt_path=paths["receipt"], operator_root=paths["operator"]
        )


def test_verifier_rejects_unexpected_file_in_sealed_tree(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _create(paths)
    unexpected = (
        paths["operator"] / CREATOR.FINAL_OPERATOR_RELATIVE_PATH / "unexpected.bin"
    )
    unexpected.write_bytes(b"not-allowlisted")

    with pytest.raises(
        VERIFIER.HoldoutSealVerificationError, match="sealed tree file-set drift"
    ):
        VERIFIER.verify(
            receipt_path=paths["receipt"], operator_root=paths["operator"]
        )


def test_verifier_rejects_receipt_that_claims_opening(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _create(paths)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["seal_opened"] = True
    os.chmod(paths["receipt"], 0o600)
    paths["receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(VERIFIER.HoldoutSealVerificationError, match="seal opened"):
        VERIFIER.verify(
            receipt_path=paths["receipt"], operator_root=paths["operator"]
        )
