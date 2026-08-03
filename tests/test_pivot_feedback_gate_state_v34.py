from pathlib import Path

from scripts.verify_pivot_feedback_gate_state_v34 import verify


def test_pivot_feedback_gate_state_v34() -> None:
    result = verify(
        Path("experiments/configs/pivot_feedback_gate_state_v34.yaml")
    )
    assert result["status"] == "PASS_PIVOT_FEEDBACK_GATE_STATE_V34"
    assert result["replay_rows"] == 9
    assert result["expected_new_accept_rows"] == 2
    assert result["production_disposition_authorized"] is False
    assert result["deterministic_reselection_authorized"] is False
