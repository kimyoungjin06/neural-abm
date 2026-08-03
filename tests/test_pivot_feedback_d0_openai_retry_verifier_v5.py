from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_openai_retry_v5.py"
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v5.yaml"
)
V4_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
V4_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
    ".verification-v4.json"
)
V4_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-execution-20260729"
)
OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_openai_retry_v5_verifier_test_module",
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
        v4_prepared_root=V4_PREPARED_ROOT,
        v4_report=V4_REPORT,
        v4_execution_root=V4_EXECUTION_ROOT,
        output_root=output_root,
    )


def test_live_v5_package_passes_independent_verification() -> None:
    result = _validate(OUTPUT_ROOT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["retry_request_rows"] == 384
    assert result["unique_expected_custom_ids"] == 384
    assert result["only_reasoning_and_output_cap_changed"] is True
    assert result["api_submitted"] is False
    assert result["producer_module_imported"] is False


def test_v5_verifier_rejects_visible_request_change(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT_ROOT.name
    shutil.copytree(OUTPUT_ROOT, copied)
    path = copied / "openai_retry_requests.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"store":false',
            '"store":true',
            1,
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)

    with pytest.raises(
        MODULE.OpenAIRetryV5VerificationError,
        match="retry body changes differ",
    ):
        _validate(copied)


def test_v5_verifier_does_not_import_v5_producer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "prepare_pivot_feedback_d0_openai_retry_v5" not in source
    assert '"producer_module_imported": False' in source
