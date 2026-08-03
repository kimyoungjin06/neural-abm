from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_content_review_packets.py"
CONFIG = ROOT / "experiments/configs/pivot_feedback_d0_content_review_v1.yaml"
SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)
OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-review-packets-v1-20260729"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_content_review_packet_verifier_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _validate(output_root: Path):
    return MODULE.validate(
        config_path=CONFIG,
        source_root=SOURCE_ROOT,
        output_root=output_root,
    )


def test_live_review_packets_pass_independent_verification() -> None:
    result = _validate(OUTPUT_ROOT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["packet_content_exact_match"] is True
    assert result["packet_orders_differ"] is True
    assert result["adjudication_key_exact_match"] is True
    assert result["reviewer_packet_rows_each"] == 384
    assert result["reviewer_labels_filled"] == 0
    assert result["human_review_complete"] is False
    assert result["d0_complete"] is False
    assert result["outcomes_read"] is False
    assert result["g4_open"] is False
    assert result["producer_module_imported"] is False


def test_verifier_rejects_a_filled_reviewer_label(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT_ROOT.name
    shutil.copytree(OUTPUT_ROOT, copied)
    packet = copied / "reviewer_1_packet.csv"
    with packet.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["reviewer_label"] = "ON_TOPIC"
    with packet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(
        MODULE.ReviewPacketVerificationError,
        match=r"artifact (size|SHA) differs",
    ):
        _validate(copied)


def test_verifier_rejects_an_extra_output_file(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT_ROOT.name
    shutil.copytree(OUTPUT_ROOT, copied)
    (copied / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    (copied / "unexpected.txt").chmod(0o600)

    with pytest.raises(
        MODULE.ReviewPacketVerificationError,
        match="output file set differs",
    ):
        _validate(copied)


def test_verifier_does_not_import_packet_builder() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_pivot_feedback_d0_content_review_packets" not in source
    assert '"producer_module_imported": False' in source
