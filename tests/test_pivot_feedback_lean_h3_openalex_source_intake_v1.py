from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_lean_h3_openalex_source_intake_v1_20260806.summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_captured_manifests_match_compact_receipt() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    manifest_root = Path(summary["captured_manifest_root"])

    assert summary["status"] == (
        "PASS_MANIFEST_ONLY_NEWER_PUBLIC_SNAPSHOT_CANDIDATE_BOUND_NO_DATA_DOWNLOAD_NO_ROW_ACCESS"
    )
    assert manifest_root.is_dir() and not manifest_root.is_symlink()
    assert stat.S_IMODE(manifest_root.stat().st_mode) == 0o700
    expected_names = {
        "combined.manifest.json",
        "works.manifest.json",
        "authors.manifest.json",
        "topics.manifest.json",
        "subfields.manifest.json",
        "fields.manifest.json",
    }
    assert {path.name for path in manifest_root.iterdir()} == expected_names

    for artifact in summary["captured_manifests"]:
        manifest = manifest_root / artifact["relative_path"]
        assert manifest.is_file() and not manifest.is_symlink()
        assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
        assert manifest.stat().st_size == artifact["bytes"]
        assert _sha256(manifest) == artifact["sha256"]


def test_manifest_values_and_lean_dependencies_replay() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    manifest_root = Path(summary["captured_manifest_root"])
    combined = json.loads((manifest_root / "combined.manifest.json").read_text(encoding="utf-8"))

    assert combined["date"] == summary["snapshot_candidate"]["manifest_date"] == "2026-06-26"
    assert combined["format"] == summary["snapshot_candidate"]["format"] == "parquet"
    assert len(combined["entities"]) == summary["snapshot_candidate"]["entity_count"] == 21
    assert combined["meta"]["record_count"] == 649_096_577
    assert combined["meta"]["content_length"] == 783_555_949_110
    combined_entities = {item["entity"]: item for item in combined["entities"]}

    for entity, expected in summary["entity_inventory"].items():
        if entity == "all_other_entities":
            continue
        payload = json.loads((manifest_root / f"{entity}.manifest.json").read_text(encoding="utf-8"))
        assert payload["date"] == "2026-06-26"
        assert payload["format"] == "parquet"
        assert payload["entity"] == entity
        assert payload["record_count"] == expected["record_count"]
        assert payload["content_length"] == expected["content_length_bytes"]
        assert len(payload["files"]) == expected["file_count"]
        assert combined_entities[entity] == {
            key: payload[key]
            for key in ("entity", "record_count", "content_length", "files")
        }

    assert summary["entity_inventory"]["works"]["role"] == "CORE_REQUIRED_FOR_H3"
    assert summary["entity_inventory"]["authors"]["role"] == (
        "CONDITIONAL_IDENTITY_AND_ORCID_SENSITIVITY_NOT_YET_CORE"
    )
    assert summary["unresolved_before_download"]["snapshot_year_completeness_and_h3_horizon_maturity"] == (
        "NOT_ESTABLISHED_BY_MANIFEST"
    )


def test_receipt_does_not_authorize_data_or_scientific_execution() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    access = summary["access_accounting"]
    authorization = summary["authorization"]

    assert access["remote_source_data_files_read"] == 0
    assert access["remote_source_records_read"] == 0
    assert access["current_g1_label_rows_read"] == 0
    assert access["holdout_or_private_g4_rows_read"] == 0
    assert access["seal_open_events"] == 0
    assert access["g4_open"] is False
    assert access["models_fitted"] == 0
    assert authorization["source_data_download"] == "NOT_AUTHORIZED_BY_THIS_RECEIPT"
    assert authorization["confirmatory_successor_execution"] == "NOT_AUTHORIZED"
    assert authorization["scientific_effect"] == "NOT_ESTABLISHED"
