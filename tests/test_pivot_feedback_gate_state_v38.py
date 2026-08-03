from pathlib import Path

from scripts.verify_pivot_feedback_gate_state_v38 import verify


def test_pivot_feedback_gate_state_v38() -> None:
    result = verify(Path("experiments/configs/pivot_feedback_gate_state_v38.yaml"))

    assert result["status"] == "PASS_PIVOT_FEEDBACK_GATE_STATE_V38"
    assert result["author_opportunity_rebuild_complete"] is True
    assert result["independent_replay_complete"] is True
    assert result["selected_work_rows"] == 30765
    assert result["author_opportunity_rows"] == 3256271
    assert result["leaveout_network_bridge_build_authorized"] is True
    assert result["production_pivot_classification_authorized"] is False
    assert result["discovery_outcome_opening_authorized"] is False
    assert result["outcomes_read"] is False
    assert result["api_requests_created"] == 0
    assert result["g4_open"] is False
