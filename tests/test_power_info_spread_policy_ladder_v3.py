from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import power_info_spread_policy_ladder_v3 as power3  # noqa: E402


def test_linear_families_stay_on_cpu() -> None:
    assert power3.device_for(0, "linear_exact") == "cpu"
    assert power3.device_for(1, "partial_linear_exact") == "cpu"


def test_every_control_runs_at_a_tiny_size_with_exact_learners() -> None:
    sizes = {"agents": 60, "steps": 24, "items_per_step": 4, "epsilon": 0.02}
    cells = [
        power3.run_cell(control, regime, power3.FAMILIES[regime][0], 0, replicates=2, sizes=sizes)
        for control in power3.CONTROLS for regime in ("A", "B")
    ]
    summary = power3.summarize(cells)
    assert set(summary["reading"]) == {"neg_no_material_pair", "pos_sign_closed_loop_material_and_positive", "adequate"}
    assert "POS_sign/A/linear_exact/L0-T0_1" in summary["table"]
    assert "POS_flip/B/partial_linear_exact/L0_window-L0" in summary["table"]
