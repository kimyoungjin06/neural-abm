from pathlib import Path

from scripts.build_pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_v1 import (
    load_config,
    load_inputs,
    replay,
)
from scripts.verify_pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_v1 import (
    verify,
)


CONFIG = Path(
    "experiments/configs/pivot_feedback_d0_t10151_rank5_external_controlled_subject_rule_v1.yaml"
)
ROOT = Path(
    "/ssd/pivot-feedback-d0-t10151-rank5-external-controlled-subject-rule-v1-20260730"
)


def test_external_controlled_subject_rule_replay_is_exact_accept_only() -> None:
    config = load_config(CONFIG)
    evidence, decisions, _, pubmed, _ = load_inputs(config)
    rows = replay(evidence, decisions, pubmed, config)
    accepted = [row for row in rows if row["deterministic_accept"]]
    assert len(rows) == 9
    assert len(accepted) == 2
    assert all(row["replay_state"] == "REMAIN_UNRESOLVED" for row in rows if not row["deterministic_accept"])


def test_external_controlled_subject_rule_independent_verification() -> None:
    result = verify(CONFIG, ROOT)
    assert result["status"] == "PASS_INDEPENDENT_EXACT_NINE_EXTERNAL_CONTROLLED_SUBJECT_RULE_REPLAY"
    assert result["candidate_counts"] == {"accept": 32, "reject": 25, "unresolved": 7}
    assert result["candidate_content_state"] == "CONTENT_PASS"
    assert result["production_disposition_created"] is False
