from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_cross_provider_api_review_v4.py"
)
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v4.yaml"
)
PARENT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
)
PARENT_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
    ".verification-v3.json"
)
OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_v4_verifier_test_module",
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
        parent_root=PARENT_ROOT,
        parent_report=PARENT_REPORT,
        output_root=output_root,
    )


def test_live_v4_package_passes_independent_verification() -> None:
    result = _validate(OUTPUT_ROOT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["custom_id_characters"] == 64
    assert result["custom_ids_unique_per_provider"] is True
    assert result["only_custom_ids_changed_from_parent_requests"] is True
    assert result["unchanged_private_artifacts_exact_match"] is True
    assert result["api_submitted"] is False
    assert result["producer_module_imported"] is False


def test_v4_verifier_rejects_non_id_request_change(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT_ROOT.name
    shutil.copytree(OUTPUT_ROOT, copied)
    path = copied / "openai_requests.jsonl"
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
        MODULE.CrossProviderV4VerificationError,
        match="OpenAI non-ID content changed",
    ):
        _validate(copied)


def test_v4_verifier_does_not_import_v4_producer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "prepare_pivot_feedback_d0_cross_provider_api_review_v4" not in source
    assert '"producer_module_imported": False' in source
