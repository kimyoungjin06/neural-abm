from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_empirical_ladder_risk_sets_v1_20261012.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_n11b_passed_and_reproduced_byte_for_byte() -> None:
    result = _result()
    assert result["registered_result"]["status"] == "N11B_PASS"
    assert all(result["execution"]["byte_identical"].values())
    assert set(result["registered_result"]["invariants"].values()) == {0}


def test_every_s1_entrant_has_one_event() -> None:
    result = _result()
    events = result["events"]
    assert events["precise_event_rows"] + events["imprecise_entrants"] == result["sampling"]["entrants_total"] == 19_132
    assert sum(events["precise_events_by_quarter"]) == events["precise_event_rows"]
    assert sum(events["imprecise_entrants_by_response_year"]) == events["imprecise_entrants"]


def test_no_outcome_was_computed_or_authorized() -> None:
    result = _result()
    assert result["persistence_labels_computed"] == result["visible_evidence_computed"] == result["follow_up_counts_computed"] == 0
    assert set(result["authorization"].values()) == {"NOT_AUTHORIZED"}
