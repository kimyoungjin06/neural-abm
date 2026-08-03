from pathlib import Path

from scripts.verify_pivot_feedback_d0_certified_episode_reselection_v1 import (
    verify,
)


CONFIG = Path(
    "experiments/configs/pivot_feedback_d0_certified_episode_reselection_v1.yaml"
)
ROOT = Path("/ssd/pivot-feedback-d0-certified-episode-reselection-v1-20260730")


def test_certified_episode_reselection_independent_verification() -> None:
    result = verify(CONFIG, ROOT)
    assert result["status"] == "PASS_INDEPENDENT_EXACT_FOUR_CERTIFIED_EPISODE_RESELECTION"
    assert result["selected_field_candidate_ranks"] == {
        "computer_science": [1, 45],
        "medicine": [5, 6],
    }
    assert result["retained_episode_rows"] == 3
    assert result["newly_selected_episode_rows"] == 1
    assert result["deterministic_reselection_executed"] is True
    assert result["author_opportunity_panel_rebuild_executed"] is False
    assert result["outcomes_read"] is False
