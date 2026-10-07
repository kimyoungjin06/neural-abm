from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_empirical_outcome_selection_trend_v1_20261007.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_stage_t_is_precommitted_reproduced_and_reads_no_empirical_row() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "49f5345583e29fc7edc838b697180ee29d8f0921"
    execution = result["execution"]
    assert execution["empirical_rows_read"] == 0
    assert execution["byte_identical"] is True
    assert execution["runs_read"] == {"N05": 4608, "N08": 4608, "N09": 4608}
    assert execution["reconstruction_failures"] == 0
    assert (execution["geometry"]["first"], execution["geometry"]["last"]) == (3, 20)


def test_the_threshold_transferred_above_the_fine_clock_one() -> None:
    registered = _result()["registered_result"]
    assert registered["disposition"] == "THRESHOLD_TRANSFERRED"
    assert registered["data_threshold"] == registered["coarse_threshold"] > registered["fine_clock_threshold"] == 0.098417
    assert registered["data_threshold_rounded"] == 0.105635
    assert registered["maximum_control"] < 0
    assert registered["control_cells_above_threshold"] == []
    assert registered["undefined_cell_statuses"]["ENTRY_CHANNEL_ACTIVE"] == 0
    assert registered["active_cells"] + registered["inactive_cells_with_a_value"] + registered["cells_undefined_on_the_coarse_clock"] == 72


def test_the_coarse_clock_loses_sensitivity_and_stages_stay_unauthorized() -> None:
    result = _result()
    reported = result["reported_not_decided"]
    assert (reported["coarse_active_cells_above_the_data_threshold"], reported["fine_clock_active_cells_above_0_098417"]) == (12, 29)
    assert reported["coarse_separation"] < 0
    assert len(reported["passing_active_cells"]) == 12
    assert set(result["authorization"].values()) == {"NOT_AUTHORIZED"}
