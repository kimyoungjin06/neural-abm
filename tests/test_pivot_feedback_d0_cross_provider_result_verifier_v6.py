from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_cross_provider_results_v6.py"
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_cross_provider_api_review_v6.yaml"
)
OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-cross-provider-api-review-v6-results-20260729"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_result_v6_verifier_test_module",
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
        v4_prepared_root=MODULE.DEFAULT_V4_PREPARED_ROOT,
        v4_execution_root=MODULE.DEFAULT_V4_EXECUTION_ROOT,
        v5_prepared_root=MODULE.DEFAULT_V5_PREPARED_ROOT,
        v5_report=MODULE.DEFAULT_V5_REPORT,
        v5_execution_root=MODULE.DEFAULT_V5_EXECUTION_ROOT,
        terminal_snapshot=MODULE.DEFAULT_TERMINAL_SNAPSHOT,
        output_root=output_root,
    )


def test_live_v6_results_pass_independent_verification() -> None:
    result = _validate(OUTPUT_ROOT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["openai_valid_structured_rows"] == 384
    assert result["claude_valid_structured_rows"] == 384
    assert result["label_agreement_rows"] == 306
    assert result["dual_high_confidence_on_topic_rows"] == 197
    assert result["blinded_human_queue_rows"] == 233
    assert result["human_calibration_complete"] is False
    assert result["producer_module_imported"] is False


def test_v6_verifier_rejects_populated_human_label(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT_ROOT.name
    shutil.copytree(OUTPUT_ROOT, copied)
    packet = copied / "human_review_reviewer_1.csv"
    packet.write_text(
        packet.read_text(encoding="utf-8").replace(",,\n", ",ON_TOPIC,\n", 1),
        encoding="utf-8",
    )
    packet.chmod(0o600)

    with pytest.raises(
        MODULE.CrossProviderResultVerificationError,
        match="human packet already populated",
    ):
        _validate(copied)


def test_v6_verifier_does_not_import_result_builder() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "build_pivot_feedback_d0_cross_provider_results_v6" not in source
    assert '"producer_module_imported": False' in source
