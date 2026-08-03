from pathlib import Path

from scripts.verify_pivot_feedback_gate_state_v35 import verify


def test_pivot_feedback_gate_state_v35() -> None:
    result = verify(Path("experiments/configs/pivot_feedback_gate_state_v35.yaml"))
    assert result["status"] == "PASS_PIVOT_FEEDBACK_GATE_STATE_V35"
    assert result["rank5_content_state"] == "CONTENT_PASS"
    assert result["medicine_selected_candidate_ranks"] == [5, 6]
    assert result["global_selection_certificate_authorized"] is True
    assert result["deterministic_reselection_authorized"] is False
