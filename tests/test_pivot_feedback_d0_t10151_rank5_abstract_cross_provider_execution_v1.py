from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/execute_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.py"


def _load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("abstract_cross_provider_execute", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decision_parser_requires_exact_five_complete_ids() -> None:
    module = _load()
    config = module._yaml(module.DEFAULT_CONFIG)
    ids = {f"id-{index}" for index in range(5)}
    text = json.dumps(
        {
            "decisions": [
                {
                    "item_id": item_id,
                    "label": "AMBIGUOUS",
                    "confidence": "HIGH",
                    "rationale": "Bounded abstract evidence.",
                }
                for item_id in sorted(ids)
            ]
        }
    )
    parsed = module.decisions(
        text,
        provider="fixture",
        expected_ids=ids,
        config=config,
    )
    assert len(parsed) == 5
    assert {row["item_id"] for row in parsed} == ids
