from pathlib import Path

from scripts.verify_pivot_feedback_gate_state_v37 import verify


def test_pivot_feedback_gate_state_v37() -> None:
    result = verify(Path("experiments/configs/pivot_feedback_gate_state_v37.yaml"))
    assert result["status"] == "PASS_PIVOT_FEEDBACK_GATE_STATE_V37"
    assert result["certified_episode_reselection_complete"] is True
    assert result["selected_field_candidate_ranks"] == {
        "computer_science": [1, 45],
        "medicine": [5, 6],
    }
    assert result["exact_selected_work_manifest_rebuild_authorized"] is True
    assert result["author_opportunity_panel_rebuild_authorized"] is True
    assert result["leaveout_network_bridge_build_authorized"] is False
    assert result["outcomes_read"] is False
