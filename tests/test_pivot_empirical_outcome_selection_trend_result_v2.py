from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_empirical_outcome_selection_trend_v2_20261007.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_stage_t_v2_is_precommitted_reproduced_and_reads_no_empirical_row() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "58c87f4bf86b84bc756b9c406400b7ba66d732c7"
    assert result["precommit"]["amends"] == "49f5345"
    execution = result["execution"]
    assert (execution["empirical_rows_read"], execution["empirical_entry_levels"]) == (0, 12)
    assert execution["byte_identical"] is True
    assert execution["reconstruction_failures"] == 0


def test_the_fine_clock_threshold_carries_to_twelve_levels() -> None:
    registered = _result()["registered_result"]
    assert registered["disposition"] == "THRESHOLD_TRANSFERRED"
    assert registered["coarse_threshold"] < registered["fine_clock_threshold"] == registered["data_threshold"] == 0.098417
    assert registered["maximum_control"] < 0
    assert registered["cells_undefined_on_the_twelve_level_clock"] == 0


def test_twelve_levels_recover_the_fine_clock_sensitivity() -> None:
    reported = _result()["reported_not_decided"]
    assert reported["twelve_level_active_cells_above_the_data_threshold"] == reported["fine_clock_active_cells_above_0_098417"] == 29
    assert reported["v1_three_level_active_cells_above_its_threshold"] == 12
    assert "N09:hazard_q0p15:failure_lag_4" in reported["missed_active_cells"]
    assert set(_result()["authorization"].values()) == {"NOT_AUTHORIZED"}
