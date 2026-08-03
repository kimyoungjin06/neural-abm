from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_d0_rank5_api_augmented_closeout_v1.yaml"
OUTPUT = Path("/ssd/pivot-feedback-d0-rank5-api-augmented-closeout-v1-20260730")
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_rank5_api_augmented_closeout_v1.py"
PRODUCER = "build_pivot_feedback_d0_rank5_api_augmented_closeout_v1"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "pivot_feedback_rank5_api_augmented_closeout_test")


def test_current_closeout_replays() -> None:
    result = VERIFIER.validate(config_path=CONFIG, output=OUTPUT)
    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["rank5_content_state"] == "CONTENT_UNRESOLVED"
    assert result["augmented_reject_rows"] == 21
    assert result["wrong_target_api_evidence_used"] is False
    assert result["medicine_field_certificate_issued"] is False
    assert result["global_selection_certificate_issued"] is False


def test_certificate_opening_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    certificate_path = copied / "terminal_selection_certificate_status_v2.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["medicine_field_certificate_issued"] = True
    certificate_path.write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    certificate_path.chmod(0o600)
    with pytest.raises(
        VERIFIER.Rank5ApiAugmentedCloseoutVerificationError,
        match="certificate status",
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
