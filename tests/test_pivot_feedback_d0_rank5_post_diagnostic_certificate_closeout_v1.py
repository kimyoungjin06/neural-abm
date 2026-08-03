from pathlib import Path

from scripts.build_pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_v1 import (
    _select_pair,
)
from scripts.verify_pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_v1 import (
    verify,
)


CONFIG = Path(
    "experiments/configs/pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_v1.yaml"
)
ROOT = Path(
    "/ssd/pivot-feedback-d0-rank5-post-diagnostic-certificate-closeout-v1-20260730"
)


def test_post_diagnostic_certificate_closeout_independent_verification() -> None:
    result = verify(CONFIG, ROOT)
    assert result["status"] == "PASS_INDEPENDENT_RANK5_POST_DIAGNOSTIC_CONTENT_PASS_GLOBAL_CERTIFICATES"
    assert result["rank5_content_state"] == "CONTENT_PASS"
    assert result["selected_candidate_ranks"] == {
        "computer_science": [1, 45],
        "medicine": [5, 6],
    }
    assert result["global_selection_certificate_issued"] is True
    assert result["deterministic_reselection_executed"] is False


def test_selection_blocks_on_earlier_compatible_unresolved() -> None:
    rows = [
        {
            "field_id": 27,
            "field_candidate_rank": 1,
            "subfield_id": 1,
            "entry_year": 2000,
            "content_state": "CONTENT_PASS",
        },
        {
            "field_id": 27,
            "field_candidate_rank": 2,
            "subfield_id": 2,
            "entry_year": 1980,
            "content_state": "CONTENT_UNRESOLVED",
        },
        {
            "field_id": 27,
            "field_candidate_rank": 3,
            "subfield_id": 3,
            "entry_year": 1970,
            "content_state": "CONTENT_PASS",
        },
    ]
    try:
        _select_pair(rows, 27)
    except RuntimeError as error:
        assert "unresolved compatible candidate" in str(error)
    else:
        raise AssertionError("earlier compatible unresolved row did not block")
