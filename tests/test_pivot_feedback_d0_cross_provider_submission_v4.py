from __future__ import annotations

from decimal import Decimal
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/submit_pivot_feedback_d0_cross_provider_api_review_v4.py"
)
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v4.yaml"
)
PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
    ".verification-v4.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_submission_v4_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_v4_preflight_passes_at_frozen_cap() -> None:
    result = MODULE._preflight(
        config_path=CONFIG,
        prepared_root=PREPARED_ROOT,
        verification_report=REPORT,
        operator_cap_usd=Decimal("1.00"),
    )

    assert result["custom_id_characters"] == 64
    assert Decimal(result["estimated_maximum_total_usd"]) < Decimal("1.00")
    assert result["api_submitted"] is False


def test_v4_preflight_rejects_lower_cap() -> None:
    with pytest.raises(
        MODULE.CrossProviderV4SubmissionError,
        match="estimated maximum exceeds operator cap",
    ):
        MODULE._preflight(
            config_path=CONFIG,
            prepared_root=PREPARED_ROOT,
            verification_report=REPORT,
            operator_cap_usd=Decimal("0.50"),
        )
