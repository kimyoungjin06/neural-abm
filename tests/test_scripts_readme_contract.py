from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_nabm_effect_matrix.py"


def import_script() -> object:
    spec = importlib.util.spec_from_file_location("run_nabm_effect_matrix", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_nabm_effect_matrix_help_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_script()

    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--help"])
    with pytest.raises(SystemExit) as exc_info:
        module.parse_args()

    assert exc_info.value.code == 0
