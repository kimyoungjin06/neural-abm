from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_pivot_feedback_d0_openai_retry_v5.py"
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v5.yaml"
)
V4_PREPARED = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
V4_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
    ".verification-v4.json"
)
V4_EXECUTION = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-execution-20260729"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_openai_retry_v5_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_prepares_full_openai_retry_within_cumulative_cap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"

    result = MODULE.prepare(
        config_path=CONFIG,
        v4_prepared_root=V4_PREPARED,
        v4_report=V4_REPORT,
        v4_execution_root=V4_EXECUTION,
        output=output,
    )

    source = [
        json.loads(line)
        for line in (V4_PREPARED / "openai_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    retry = [
        json.loads(line)
        for line in (output / "openai_retry_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(retry) == 384
    for before, after in zip(source, retry, strict=True):
        assert before["custom_id"] == after["custom_id"]
        before_body = dict(before["body"])
        after_body = dict(after["body"])
        assert after_body.pop("reasoning") == {"effort": "minimal"}
        assert before_body.pop("max_output_tokens") == 256
        assert after_body.pop("max_output_tokens") == 1024
        assert before_body == after_body
    assert (
        Decimal(
            result["cost_control"][
                "cumulative_actual_plus_retry_maximum_usd"
            ]
        )
        <= Decimal("1.00")
    )
    assert result["v4_claude_valid_structured_rows"] == 384
    assert result["v4_openai_valid_structured_rows"] == 144
    assert result["api_submitted"] is False
