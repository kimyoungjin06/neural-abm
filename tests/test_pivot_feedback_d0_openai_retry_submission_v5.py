from __future__ import annotations

from decimal import Decimal
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/submit_pivot_feedback_d0_openai_retry_v5.py"
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v5.yaml"
)
PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729"
)
REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729.verification-v5.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_openai_retry_submission_v5_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_v5_preflight_passes_at_frozen_cumulative_cap() -> None:
    result = MODULE._preflight(
        config_path=CONFIG,
        prepared_root=PREPARED_ROOT,
        verification_report=REPORT,
        operator_cap_usd=Decimal("1.00"),
    )

    assert result["retry_request_rows"] == 384
    assert result["claude_resubmitted"] is False
    assert Decimal(
        result["cumulative_actual_plus_retry_maximum_usd"]
    ) < Decimal("1.00")
    assert result["api_submitted"] is False


def test_v5_preflight_rejects_lower_cumulative_cap() -> None:
    with pytest.raises(
        MODULE.OpenAIRetryV5SubmissionError,
        match="cumulative maximum exceeds operator cap",
    ):
        MODULE._preflight(
            config_path=CONFIG,
            prepared_root=PREPARED_ROOT,
            verification_report=REPORT,
            operator_cap_usd=Decimal("0.50"),
        )


def test_v5_submitter_has_no_claude_endpoint() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "api.anthropic.com" not in source
    assert '"resubmitted": False' in source
