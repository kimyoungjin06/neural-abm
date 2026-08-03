from __future__ import annotations

from decimal import Decimal
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/submit_pivot_feedback_d0_cross_provider_api_review_v3.py"
)
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v3.yaml"
)
PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
)
REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v3-prepared-20260729"
    ".verification-v3.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_submission_v3_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_preflight_accepts_frozen_one_dollar_cap() -> None:
    result = MODULE._preflight(
        config_path=CONFIG,
        prepared_root=PREPARED_ROOT,
        verification_report=REPORT,
        operator_cap_usd=Decimal("1.00"),
    )

    assert Decimal(result["estimated_maximum_total_usd"]) < Decimal("1.00")
    assert result["operator_cap_usd"] == "1.00"
    assert result["credential_values_persisted"] is False
    assert result["api_submitted"] is False


def test_preflight_rejects_cap_below_estimate() -> None:
    with pytest.raises(
        MODULE.CrossProviderSubmissionError,
        match="estimated maximum exceeds operator cap",
    ):
        MODULE._preflight(
            config_path=CONFIG,
            prepared_root=PREPARED_ROOT,
            verification_report=REPORT,
            operator_cap_usd=Decimal("0.50"),
        )


def test_preflight_rejects_cap_above_frozen_authorization() -> None:
    with pytest.raises(
        MODULE.CrossProviderSubmissionError,
        match="operator cap exceeds frozen authorization",
    ):
        MODULE._preflight(
            config_path=CONFIG,
            prepared_root=PREPARED_ROOT,
            verification_report=REPORT,
            operator_cap_usd=Decimal("1.01"),
        )


def test_credentials_require_private_file_mode(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY=test-openai\nCLAUDE_API_KEY=test-claude\n",
        encoding="utf-8",
    )
    path.chmod(0o644)

    with pytest.raises(
        MODULE.CrossProviderSubmissionError,
        match="permissions too broad",
    ):
        MODULE._read_credentials(path)


def test_credentials_are_parsed_without_persistence(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        'OPENAI_API_KEY="test-openai"\nCLAUDE_API_KEY="test-claude"\n',
        encoding="utf-8",
    )
    path.chmod(0o600)

    result = MODULE._read_credentials(path)

    assert result == {
        "OPENAI_API_KEY": "test-openai",
        "ANTHROPIC_API_KEY": "test-claude",
    }
