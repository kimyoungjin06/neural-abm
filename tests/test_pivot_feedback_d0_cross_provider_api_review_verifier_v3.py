from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_cross_provider_api_review_v3.py"
)
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v3.yaml"
)
SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)
OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_verifier_v3_test_module",
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


def test_live_cross_provider_package_passes() -> None:
    result = _validate(OUTPUT_ROOT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["cross_provider_request_content_exact_match"] is True
    assert result["request_bodies_contain_forbidden_fields"] is False
    assert result["calibration_packets_exact_match"] is True
    assert result["cost_control_exact_match"] is True
    assert result["api_submitted"] is False
    assert result["producer_module_imported"] is False


def test_verifier_rejects_tampered_claude_request(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT_ROOT.name
    shutil.copytree(OUTPUT_ROOT, copied)
    request = copied / "claude_batch_request.json"
    request.write_text(
        request.read_text(encoding="utf-8").replace(
            "claude-haiku-4-5-20251001",
            "claude-sonnet-5",
            1,
        ),
        encoding="utf-8",
    )
    request.chmod(0o600)

    with pytest.raises(
        MODULE.CrossProviderVerificationError,
        match="Claude requests differ",
    ):
        _validate(copied)


def test_verifier_does_not_import_producer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "prepare_pivot_feedback_d0_cross_provider_api_review_v3" not in source
    assert '"producer_module_imported": False' in source
