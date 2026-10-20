from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import power_info_spread_policy_ladder_v1 as power  # noqa: E402


def _rows(deltas: list[float], slopes: tuple[float, float] = (0.0, 0.0)) -> list[dict]:
    shocks = np.linspace(0.3, 0.7, len(deltas))
    return [
        {
            "early_shock": float(s),
            "first": {"share_precision": 0.6 + d, "false_reach": 10 + slopes[0] * s + 0.01 * (-1) ** i},
            "second": {"share_precision": 0.6, "false_reach": 10 + slopes[1] * s + 0.01 * (-1) ** (i // 2)},
        }
        for i, (d, s) in enumerate(zip(deltas, shocks, strict=True))
    ]


def test_the_paired_rule_needs_twelve_agreeing_replicates() -> None:
    assert power.paired_rule(_rows([0.03] * 12 + [0.0] * 4))["holds"]
    assert power.paired_rule(_rows([-0.03] * 12 + [0.0] * 4))["holds"]
    assert not power.paired_rule(_rows([0.03] * 11 + [0.0] * 5))["holds"]
    assert not power.paired_rule(_rows([0.03] * 8 + [-0.03] * 8))["holds"]


def test_the_path_rule_detects_a_slope_difference_only() -> None:
    assert power.path_rule(_rows([0.0] * 16, (40.0, 0.0)), (1, 2))["holds"]
    assert not power.path_rule(_rows([0.0] * 16, (0.0, 0.0)), (1, 2))["holds"]


def test_a_cell_runs_end_to_end_at_a_tiny_size() -> None:
    sizes = {"agents": 60, "steps": 16, "items_per_step": 4, "epsilon": 0.02}
    for control in power.CONTROLS:
        cell = power.run_cell(control, "A", "linear", 0, replicates=3, sizes=sizes)
        assert len(cell["rows"]) == 3
        scored = power.score_cell(cell)
        assert set(scored) == {"r1", "r2", "material"}
