from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_api_review_preparation_v2.py"
)
CONFIG = ROOT / "experiments/configs/pivot_feedback_d0_api_assisted_review_v2.yaml"
SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)
OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-api-review-v2-prepared-20260729"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_api_review_preparation_verifier_test_module",
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


def test_live_api_review_preparation_passes_independent_verification() -> None:
    result = _validate(OUTPUT_ROOT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["request_content_exact_match"] is True
    assert result["request_visible_fields_exact_match"] is True
    assert result["request_bodies_contain_forbidden_fields"] is False
    assert result["calibration_packets_exact_match"] is True
    assert result["calibration_key_exact_match"] is True
    assert result["api_submitted"] is False
    assert result["human_calibration_complete"] is False
    assert result["d0_complete"] is False
    assert result["producer_module_imported"] is False


def test_verifier_rejects_tampered_request(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT_ROOT.name
    shutil.copytree(OUTPUT_ROOT, copied)
    request_path = copied / "judge_a_requests.jsonl"
    request_path.write_text(
        request_path.read_text(encoding="utf-8").replace(
            '"store":false',
            '"store":true',
            1,
        ),
        encoding="utf-8",
    )
    request_path.chmod(0o600)

    with pytest.raises(
        MODULE.ApiReviewVerificationError,
        match="judge_a requests differ",
    ):
        _validate(copied)


def test_verifier_rejects_extra_output_file(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT_ROOT.name
    shutil.copytree(OUTPUT_ROOT, copied)
    unexpected = copied / "unexpected.txt"
    unexpected.write_text("unexpected\n", encoding="utf-8")
    unexpected.chmod(0o600)

    with pytest.raises(
        MODULE.ApiReviewVerificationError,
        match="output file set differs",
    ):
        _validate(copied)


def test_verifier_does_not_import_preparer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "prepare_pivot_feedback_d0_api_review_v2" not in source
    assert '"producer_module_imported": False' in source
