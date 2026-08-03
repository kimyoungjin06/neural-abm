from pathlib import Path

from scripts.verify_pivot_feedback_gate_state_v36 import verify


def test_pivot_feedback_gate_state_v36() -> None:
    result = verify(Path("experiments/configs/pivot_feedback_gate_state_v36.yaml"))
    assert result["status"] == "PASS_PIVOT_FEEDBACK_GATE_STATE_V36"
    assert result["global_selection_certificate_verified"] is True
    assert result["authorized_field_candidate_ranks"] == {
        "computer_science": [1, 45],
        "medicine": [5, 6],
    }
    assert result["deterministic_reselection_successor_authorized"] is True
    assert result["deterministic_reselection_executed"] is False
    assert result["author_opportunity_panel_rebuild_authorized"] is False
