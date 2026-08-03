from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/prepare_pivot_feedback_d0_cross_provider_api_review_v4.py"
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


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_cross_provider_v4_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_v4_changes_only_provider_custom_ids(tmp_path: Path) -> None:
    output = tmp_path / "prepared"

    result = MODULE.prepare(
        config_path=CONFIG,
        parent_root=PARENT_ROOT,
        parent_report=PARENT_REPORT,
        output=output,
    )

    parent_openai = [
        json.loads(line)
        for line in (PARENT_ROOT / "openai_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    repaired_openai = [
        json.loads(line)
        for line in (output / "openai_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    parent_claude = json.loads(
        (PARENT_ROOT / "claude_batch_request.json").read_text(encoding="utf-8")
    )["requests"]
    repaired_claude = json.loads(
        (output / "claude_batch_request.json").read_text(encoding="utf-8")
    )["requests"]

    assert result["custom_id_characters"] == 64
    assert result["only_custom_ids_changed_from_parent_requests"] is True
    assert all(len(row["custom_id"]) == 64 for row in repaired_openai)
    assert all(len(row["custom_id"]) == 64 for row in repaired_claude)
    for parent, repaired in zip(parent_openai, repaired_openai, strict=True):
        assert parent["custom_id"].removeprefix("openai-") == repaired["custom_id"]
        assert parent["body"] == repaired["body"]
    for parent, repaired in zip(parent_claude, repaired_claude, strict=True):
        assert parent["custom_id"].removeprefix("claude-") == repaired["custom_id"]
        assert parent["params"] == repaired["params"]
    for name in MODULE.UNCHANGED_FILES:
        assert MODULE._sha256(PARENT_ROOT / name) == MODULE._sha256(output / name)
