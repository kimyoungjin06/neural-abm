from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "n15_package_parity_20261020.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_the_package_reproduces_the_v3_record_byte_for_byte() -> None:
    run = _result()["runs"]["v3_release"]
    assert run["whole_file_identical"] is True
    assert run["runs_jsonl_sha256"] == _result()["targets"]["v3"]["recorded_runs_jsonl_sha256"]
    assert run["rows"] == run["identical_rows"] == 1152


def test_both_insertion_modes_reproduce_every_delayed_field_row() -> None:
    result = _result()
    for name in ("abm2_release", "abm2_event"):
        run = result["runs"][name]
        assert run["rows"] == run["identical_rows"] == 432
        assert {group: v["identical"] for group, v in run["by_group"].items()} == {
            f"{regime}/{arm}": 48
            for regime in ("failure_lag_2", "failure_lag_4", "immediate")
            for arm in ("linear_batch", "mlp_batch", "tree_batch")
        }
    assert result["ordering_rule"]["event_insertion_reproduces_pivot_under_delays"] is True
    assert result["boundaries"]["frozen_scripts_changed"] is False
