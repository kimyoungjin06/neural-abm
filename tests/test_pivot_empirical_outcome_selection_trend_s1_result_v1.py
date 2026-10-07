from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_empirical_outcome_selection_trend_s1_20261007.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_s1_passed_and_reproduced_byte_for_byte() -> None:
    result = _result()
    assert result["registered_result"]["status"] == "S1_PASS"
    assert all(result["execution"]["byte_identical"].values())
    assert set(result["registered_result"]["failure_metrics"].values()) == {0}
    assert result["source_accounting"]["rows"] == 510_372_821


def test_every_field_has_four_confirmatory_episodes_meeting_the_floors() -> None:
    by_field = _result()["registered_result"]["disposition_by_field"]
    assert set(by_field) == {"11", "16", "32"}
    assert all(cell["confirmatory_meeting_floors"] >= 4 for cell in by_field.values())


def test_entrant_accounting_and_no_persistence() -> None:
    result = _result()
    entrants = result["entrants"]
    assert entrants["pivot_eligible_entrants"] - entrants["g4_guard_removed"] - entrants["discovery_author_removed"] == entrants["retained"]
    assert result["authorization"]["persistence_labels"] == "NOT_AUTHORIZED"
    assert result["post_hoc_structural_diagnostics"]["xpac_share_of_entry_works"] == 0.0
