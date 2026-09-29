from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "configs" / "pivot_policy_ladder_consolidation_v2.yaml"
DOCUMENT = ROOT / "docs" / "research" / "pivot-policy-ladder-consolidation-v2.md"
OPERATORS = ("equals", "contains", "greater_than", "less_than")


def _ledger() -> dict[str, Any]:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


def _inherited() -> dict[str, Any]:
    path = ROOT / _ledger()["inherits"]["ledger"]["path"]
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, Any]:
    return {
        row["id"]: json.loads((ROOT / row["path"]).read_text(encoding="utf-8"))
        for row in _ledger()["sources"]
    }


def _resolve(document: Any, pointer: str) -> Any:
    value = document
    for raw in pointer.split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _holds(check: dict[str, Any], sources: dict[str, Any]) -> bool:
    value = _resolve(sources[check["source"]], check["pointer"])
    (operator,) = [name for name in OPERATORS if name in check]
    expected = check[operator]
    if operator == "equals":
        return value == expected
    if operator == "contains":
        return expected in value
    if operator == "greater_than":
        return value > expected
    return value < expected


def _all_checks(ledger: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    checks = []
    for status in ("confirmed", "rejected", "open"):
        for claim in ledger["claims"][status]:
            checks.extend((claim["id"], check) for check in claim.get("evidence", []))
    for update in ledger["status_updates"]:
        checks.extend((update["id"], check) for check in update.get("evidence", []))
    for tier in ledger["positioning"]["tiers"]:
        checks.extend((tier["id"], check) for check in tier["evidence"])
    checks.extend(("positioning", check) for check in ledger["positioning"]["evidence"])
    for decision in ledger["framework_decisions"]["decisions"]:
        checks.extend((decision["id"], check) for check in decision.get("evidence", []))
    return checks


def test_ledger_is_a_no_execution_consolidation() -> None:
    ledger = _ledger()
    assert ledger["status"] == "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION"
    boundaries = ledger["boundaries"]
    assert boundaries["new_execution"] == "NONE"
    assert boundaries["empirical_rows_read"] == 0
    assert boundaries["llm_calls"] == 0
    assert boundaries["generic_nabm_api_changed"] is False


def test_the_inherited_ledger_and_every_source_are_bound_by_hash() -> None:
    ledger = _ledger()
    inherits = ledger["inherits"]
    for row in (inherits["ledger"], inherits["document"], *ledger["sources"]):
        assert _sha256(ROOT / row["path"]) == row["sha256"], row["path"]
    assert (ROOT / inherits["inherited_claims_remain_audited_by"]).exists()


def test_every_evidence_check_holds_against_the_recorded_results() -> None:
    ledger = _ledger()
    sources = _sources()
    checks = _all_checks(ledger)
    assert len(checks) >= 50
    failures = [
        (claim_id, check) for claim_id, check in checks if not _holds(check, sources)
    ]
    assert failures == []


def test_new_claims_extend_the_inherited_ids_without_collision() -> None:
    inherited = _inherited()["claims"]
    claims = _ledger()["claims"]
    old_ids = {claim["id"] for status in inherited for claim in inherited[status]}
    new_ids = [claim["id"] for status in claims for claim in claims[status]]
    assert len(new_ids) == len(set(new_ids))
    assert not old_ids & set(new_ids)

    def number(claim_id: str) -> int:
        return int(re.match(r"[CRO](\d+)_", claim_id).group(1))

    for status, prefix in (("confirmed", "C"), ("rejected", "R"), ("open", "O")):
        previous = max(number(claim["id"]) for claim in inherited[status])
        numbers = [number(claim["id"]) for claim in claims[status]]
        assert all(claim["id"].startswith(prefix) for claim in claims[status])
        assert numbers == list(range(previous + 1, previous + 1 + len(numbers)))
    for claim in claims["open"]:
        assert claim["kind"] in {"EMPIRICAL", "SYNTHETIC"}
        assert claim["priority"] in {1, 2, 3}


def test_post_hoc_claims_are_labelled() -> None:
    by_id = {claim["id"]: claim for claim in _ledger()["claims"]["confirmed"]}
    for claim_id in (
        "C23_the_exploratory_share_traces_a_frontier",
        "C24_escaping_the_hand_prior_explains_most_local_rescue",
    ):
        assert by_id[claim_id]["evidence_type"] == "POST_HOC_DESCRIPTIVE"
        for check in by_id[claim_id]["evidence"]:
            assert "post_hoc" in check["pointer"] or check["source"] == "llm_prior_v2"


def test_status_updates_point_at_inherited_items_and_new_claims() -> None:
    ledger = _ledger()
    inherited = _inherited()
    inherited_ids = {
        claim["id"] for status in inherited["claims"] for claim in inherited["claims"][status]
    }
    inherited_ids |= {row["id"] for row in inherited["next_steps"]}
    inherited_ids |= {row["id"] for row in inherited["empirical_targets"]}
    new_ids = {claim["id"] for status in ledger["claims"] for claim in ledger["claims"][status]}
    for update in ledger["status_updates"]:
        assert update["id"] in inherited_ids, update["id"]
        for answer in update.get("answered_by", []):
            assert answer in new_ids
        if "remaining" in update:
            assert update["remaining"] in new_ids
    t01 = next(row for row in inherited["empirical_targets"] if row["id"].startswith("T01"))
    update = next(row for row in ledger["status_updates"] if row["id"] == t01["id"])
    assert update["requirement_replaced"] in t01["requirements_before_comparison"]


def test_positioning_orders_the_tiers_by_cost_and_marks_the_llm_tier_estimated() -> None:
    positioning = _ledger()["positioning"]
    tiers = [tier["id"] for tier in positioning["tiers"]]
    assert tiers == [
        "RULE_BASED_ABM",
        "NABM_POOLED_LEARNED_POLICY",
        "NABM_PER_AGENT_LOCAL_LEARNER",
        "NABM_WITH_ONCE_ELICITED_LLM_PRIOR",
        "LLM_AGENT_SIMULATION",
    ]
    by_id = {tier["id"]: tier for tier in positioning["tiers"]}
    pooled = by_id["NABM_POOLED_LEARNED_POLICY"]["measured_cost_relative_to_rule"]
    local = by_id["NABM_PER_AGENT_LOCAL_LEARNER"]["measured_cost_relative_to_rule"]
    assert max(pooled) < min(local)
    assert by_id["LLM_AGENT_SIMULATION"]["behavior"] == "NOT_MEASURED_SEE_O09"
    assert "ESTIMATED_NOT_MEASURED" in positioning["claim_ceiling"]
    cost = _sources()["relative_delay_cost"]["cost_ledger"]["records"]
    field = [row for row in cost if row["evidence_regime"] == "field_public"]
    feature = [row for row in field if row["arm_id"] != "scalar_batch"]
    ratios = sorted(row["cost_relative_to_fixed_rule"] for row in feature)
    assert [round(ratios[0], 1), round(ratios[-1], 1)] == pooled


def test_framework_decisions_keep_the_second_consumer_rule() -> None:
    decisions = _ledger()["framework_decisions"]
    for row in decisions["decisions"]:
        if row["decision"] in {"DOMAIN_OWNED", "REPOSITORY_ONLY_RESEARCH_INSTRUMENT"}:
            assert row["second_consumer"] == "NONE", row["id"]
        assert not row["decision"].startswith("EXTRACTION_CANDIDATE"), row["id"]
    inherited = {row["id"] for row in _inherited()["framework_decisions"]["decisions"]}
    assert set(decisions["extraction_candidates_unchanged"]) <= inherited
    llm = next(row for row in decisions["decisions"] if row["id"].startswith("F11"))
    assert llm["package_llm_dependency"] is False


def test_next_steps_answer_open_questions_and_keep_the_transfer_gated() -> None:
    ledger = _ledger()
    open_ids = {claim["id"] for claim in ledger["claims"]["open"]}
    steps = {row["id"]: row for row in ledger["next_steps"]}
    assert "N01" not in steps
    assert steps["N02"]["authorization"] == "USER_DECISION_REQUIRED"
    for row in steps.values():
        if "answers" in row:
            assert row["answers"] in open_ids
    assert steps["N06"]["serves"] == "NABM_POSITIONING"


def test_document_states_the_registered_headline_numbers() -> None:
    text = " ".join(DOCUMENT.read_text(encoding="utf-8").split())
    for required in (
        "7 of 9",
        "5 of 9",
        "1.9 to 6.7 times",
        "under 10 GPU minutes",
        "38 to 125 GPU hours",
        "`NEITHER_SIMPLE_RULE`",
        "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION",
    ):
        assert required in text, required
