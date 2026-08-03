from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_d0_cross_provider_results_v6.py"
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v6.yaml"
)
V4_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-prepared-20260729"
)
V4_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v4-execution-20260729"
)
V5_PREPARED_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729"
)
V5_REPORT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "prepared-20260729.verification-v5.json"
)
V5_EXECUTION_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v5-openai-retry-"
    "execution-20260729"
)
TERMINAL_SNAPSHOT = (
    V5_EXECUTION_ROOT / "status_snapshots/status-20260729T050931994847Z.json"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_results_v6_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_v6_builder_creates_valid_blinded_queue(tmp_path: Path) -> None:
    output = tmp_path / "pivot-feedback-d0-cross-provider-api-review-v6-results-20260729"
    result = MODULE.build(
        config_path=CONFIG,
        v4_prepared_root=V4_PREPARED_ROOT,
        v4_execution_root=V4_EXECUTION_ROOT,
        v5_prepared_root=V5_PREPARED_ROOT,
        v5_report=V5_REPORT,
        v5_execution_root=V5_EXECUTION_ROOT,
        terminal_snapshot=TERMINAL_SNAPSHOT,
        output=output,
    )

    assert result["status"] == MODULE.STATUS
    assert result["openai_valid_structured_rows"] == 384
    assert result["claude_valid_structured_rows"] == 384
    assert result["reduction"]["label_agreement_rows"] == 306
    assert result["reduction"]["blinded_human_queue_rows"] == 233
    assert result["reduction"]["dual_high_confidence_on_topic_rows"] == 197
    assert result["human_packets_blinded_from_api_outputs"] is True
    assert result["human_calibration_complete"] is False


def test_human_packet_schema_excludes_api_outputs(tmp_path: Path) -> None:
    output = tmp_path / "pivot-feedback-d0-cross-provider-api-review-v6-results-20260729"
    MODULE.build(
        config_path=CONFIG,
        v4_prepared_root=V4_PREPARED_ROOT,
        v4_execution_root=V4_EXECUTION_ROOT,
        v5_prepared_root=V5_PREPARED_ROOT,
        v5_report=V5_REPORT,
        v5_execution_root=V5_EXECUTION_ROOT,
        terminal_snapshot=TERMINAL_SNAPSHOT,
        output=output,
    )

    header = (output / "human_review_reviewer_1.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert header.split(",") == MODULE.PACKET_FIELDS
    assert "openai" not in header
    assert "claude" not in header
