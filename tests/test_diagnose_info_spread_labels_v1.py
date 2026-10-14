from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_info_spread_labels_v1 as labels  # noqa: E402


def test_the_label_diagnosis_runs_all_three_modes() -> None:
    sizes = {"agents": 60, "steps": 24, "items_per_step": 4, "epsilon": 0.02}
    cell = labels.run_cell("A", "linear", 0, replicates=2, sizes=sizes)
    summary = labels.summarize([cell])["A/linear"]
    assert set(summary["arms"]) == {"S_shares", "U_matched", "F_all"}
    # U matches the share count of its own run, which can differ from S's run
    # once learning diverges; F keeps every decided exposure.
    rows = {name: arm["rows_at_last_refit_before_shift"] for name, arm in summary["arms"].items()}
    assert 0 < rows["U_matched"] < rows["F_all"] and 0 < rows["S_shares"] < rows["F_all"]
