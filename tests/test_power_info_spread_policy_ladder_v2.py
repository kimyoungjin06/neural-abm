from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import power_info_spread_policy_ladder_v2 as power2  # noqa: E402


def test_the_group_rule_needs_an_interval_off_zero_and_a_minimum_effect() -> None:
    assert power2.group_rule([0.02] * 10 + [0.03] * 10, (1,))["material"]
    assert not power2.group_rule([0.004] * 20, (1,))["material"]  # off zero but below 0.01
    assert not power2.group_rule([0.05, -0.05] * 10, (1,))["material"]


def test_every_control_runs_at_a_tiny_size() -> None:
    sizes = {"agents": 60, "steps": 24, "items_per_step": 4, "epsilon": 0.02}
    cells = [
        power2.run_cell(control, regime, power2.FAMILIES[regime][0], 0, replicates=2, sizes=sizes)
        for control in power2.CONTROLS for regime in ("A", "B")
    ]
    summary = power2.summarize(cells)
    assert set(summary["reading"]) == {"neg_no_material_pair", "pos_sign_closed_loop_material_and_positive", "adequate"}
    assert any(key.startswith("POS_flip/B/partial_linear/L0_window-L0") for key in summary["table"])
