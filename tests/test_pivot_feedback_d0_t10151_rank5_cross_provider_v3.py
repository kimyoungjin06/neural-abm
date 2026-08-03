from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_v3.yaml"
OUTPUT = Path("/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v3-prepared-20260730")
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_t10151_rank5_cross_provider_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_rank5_cross_provider_v3_test")


def test_correct_hcv_requests_replay() -> None:
    result = VERIFIER.validate(config_path=CONFIG, output=OUTPUT)
    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["request_rows_verified"] == 25
    assert result["provider_calls_verified"] == 2
    assert result["prior_attempt_results_reusable"] is False
    assert float(result["cumulative_cost_upper_bound_usd"]) < 0.10


def test_wrong_target_payload_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    request_path = copied / "openai_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    visible = json.loads(request["input"][0]["content"][0]["text"])
    visible["target_topic_name"] = "Hepatitis B Virus Infection"
    request["input"][0]["content"][0]["text"] = json.dumps(
        visible,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    request_path.chmod(0o600)
    with pytest.raises(
        VERIFIER.Rank5CrossProviderVerificationError,
        match="OpenAI visible payload",
    ):
        VERIFIER.validate(config_path=CONFIG, output=copied)


def test_claude_cardinality_keywords_are_absent() -> None:
    request = json.loads((OUTPUT / "claude_request.json").read_text(encoding="utf-8"))
    decisions = request["output_config"]["format"]["schema"]["properties"]["decisions"]
    assert "minItems" not in decisions
    assert "maxItems" not in decisions
