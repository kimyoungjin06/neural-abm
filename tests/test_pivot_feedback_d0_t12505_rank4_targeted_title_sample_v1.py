from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_t12505_rank4_targeted_title_sample_v1.yaml"
)
OUTPUT = Path(
    "/ssd/pivot-feedback-d0-t12505-rank4-targeted-title-sample-v1-20260729"
)
VERIFIER_SCRIPT = (
    ROOT
    / "scripts/verify_pivot_feedback_d0_t12505_rank4_targeted_title_sample_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_SCRIPT, "t12505_rank4_sample_verifier_test")


def test_current_rank4_sample_fast_replay() -> None:
    result = VERIFIER.validate(
        config_path=CONFIG,
        output_root=OUTPUT,
        workers=4,
        replay_normalized=False,
    )

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["selected_title_rows"] == 96
    assert result["missing_title_rows"] == 0
    assert result["recent_six_year_author_linked_rows"] == 13_766
    assert result["raw_selected_row_replay"]["selected_rows_checked"] == 96
    assert result["normalized_replay_executed"] is False
    assert result["api_requests_submitted"] == 0
    assert result["selection_certificate_issued"] is False


def test_rank4_disposition_cannot_be_opened(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["authorization"]["production_content_disposition"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.T12505Rank4TargetedSampleVerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(
            config_path=mutated,
            output_root=OUTPUT,
            replay_normalized=False,
        )


def test_sample_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    sample = copied / "t12505_rank4_targeted_title_sample.parquet"
    sample.write_bytes(sample.read_bytes() + b"\n")
    sample.chmod(0o600)

    with pytest.raises(
        VERIFIER.T12505Rank4TargetedSampleVerificationError,
        match="artifact",
    ):
        VERIFIER.validate(
            config_path=CONFIG,
            output_root=copied,
            replay_normalized=False,
        )


def test_verifier_does_not_import_rank4_producer() -> None:
    tree = ast.parse(VERIFIER_SCRIPT.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert (
        "build_pivot_feedback_d0_t12505_rank4_targeted_title_sample_v1"
        not in imports
    )
