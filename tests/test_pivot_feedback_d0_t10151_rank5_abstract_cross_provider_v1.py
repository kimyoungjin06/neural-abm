from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("abstract_cross_provider_prepare", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_five_local_abstract_request_is_bounded() -> None:
    module = _load()
    config = module.validate_config(module.DEFAULT_CONFIG)
    rows = module.selected_rows(config)
    assert len(rows) == 5
    assert len({row["item_id"] for row in rows}) == 5
    openai, claude = module.request_bodies(rows, config)
    assert openai["store"] is False
    assert "tools" not in openai
    assert "tools" not in claude
    assert openai["text"]["format"]["schema"]["properties"]["decisions"]["minItems"] == 5
    assert "minItems" not in claude["output_config"]["format"]["schema"]["properties"]["decisions"]
    cost = module.cost_upper_bound({"openai": openai, "claude": claude}, config)
    assert cost["within_authorized_caps"] is True
    assert float(cost["pair_upper_bound_usd"]) <= 0.04
    assert float(cost["cumulative_upper_bound_usd"]) <= 0.10
