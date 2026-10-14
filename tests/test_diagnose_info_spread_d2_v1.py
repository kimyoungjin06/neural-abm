from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_info_spread_d2_v1 as diagnose  # noqa: E402


def test_the_diagnosis_runs_and_separates_both_factors() -> None:
    sizes = {"agents": 60, "steps": 20, "items_per_step": 4, "epsilon": 0.02}
    cells = [diagnose.run_cell("POS_flip", "A", "linear", 0, replicates=2, sizes=sizes)]
    summary = diagnose.summarize(cells)["POS_flip/A/linear"]
    assert set(summary["mean_precision"]) == {"L0_full", "L0_window", "T0_1", "T0_4"}
    assert "T0_1 - T0_4 (ensembling)" in summary["contrasts"]
    assert "L0_window - L0_full (forgetting)" in summary["contrasts"]
    assert summary["credibility_weight_by_steps_from_shift"]["T0_4"]
