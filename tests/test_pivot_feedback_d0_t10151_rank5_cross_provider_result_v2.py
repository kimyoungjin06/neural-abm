from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_d0_t10151_rank5_cross_provider_result_v2.yaml"
OUTPUT = Path("/ssd/pivot-feedback-d0-t10151-rank5-cross-provider-v3-results-20260730")
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_t10151_rank5_cross_provider_result_v1.py"
PRODUCER = "build_pivot_feedback_d0_t10151_rank5_cross_provider_result_v1"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_rank5_cross_provider_result_v2_test")


def test_correct_hcv_result_replays() -> None:
    result = VERIFIER.validate(config_path=CONFIG, output=OUTPUT)
    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["exact_label_agreement_rows"] == 17
    assert result["consensus_route_counts"] == {
        "API_DUAL_HIGH_REJECT": 9,
        "API_REMAINS_UNRESOLVED": 16,
    }
    assert result["candidate"]["augmented_reject_rows"] == 21
    assert result["candidate"]["candidate_content_state"] == "CONTENT_UNRESOLVED"


def test_promoted_candidate_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    candidate_path = copied / "rank5_augmented_candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["candidate_content_state"] = "CONTENT_FAIL"
    candidate_path.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate_path.chmod(0o600)
    with pytest.raises(
        VERIFIER.Rank5CrossProviderResultVerificationError,
        match="candidate",
    ):
        VERIFIER.validate(config_path=CONFIG, output=copied)


def test_verifier_does_not_import_producer() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert PRODUCER not in imports
