from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/prepare_pivot_feedback_d0_cross_provider_api_review_v3.py"
)
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v3.yaml"
)
SOURCE_ROOT = Path(
    "/ssd/pivot-feedback-d0-content-indexing-audit-v1-full-20260724"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_api_review_v3_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_prepares_cross_provider_requests_with_cost_cap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"

    result = MODULE.prepare(
        config_path=CONFIG,
        source_root=SOURCE_ROOT,
        output=output,
    )

    assert result["status"] == MODULE.STATUS
    assert result["request_rows_per_judge"] == 384
    assert result["total_request_rows"] == 768
    assert result["base_calibration_rows"] == 96
    assert result["api_submitted"] is False
    assert result["human_calibration_complete"] is False
    assert result["d0_complete"] is False
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
    assert (
        Decimal(result["cost_control"]["total_maximum_usd"])
        <= Decimal(result["cost_control"]["authorized_maximum_usd"])
        == Decimal("1.00")
    )

    openai = _jsonl(output / "openai_requests.jsonl")
    claude = json.loads(
        (output / "claude_batch_request.json").read_text(encoding="utf-8")
    )["requests"]
    assert len(openai) == len(claude) == 384
    assert {row["body"]["model"] for row in openai} == {
        "gpt-5-mini-2025-08-07"
    }
    assert {row["params"]["model"] for row in claude} == {
        "claude-haiku-4-5-20251001"
    }
    assert all(row["body"]["store"] is False for row in openai)
    assert all("tools" not in row["body"] for row in openai)
    assert all("tools" not in row["params"] for row in claude)
    assert all(
        row["params"]["output_config"]["format"]["type"] == "json_schema"
        for row in claude
    )


def test_preparation_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.mkdir()

    with pytest.raises(
        MODULE.CrossProviderPreparationError,
        match="output already exists",
    ):
        MODULE.prepare(
            config_path=CONFIG,
            source_root=SOURCE_ROOT,
            output=output,
        )


def test_cleanup_rejects_current_directory(tmp_path: Path) -> None:
    with pytest.raises(MODULE.CrossProviderPreparationError):
        MODULE._cleanup_temporary_directory(
            Path(),
            output=(tmp_path / "prepared").resolve(),
        )
