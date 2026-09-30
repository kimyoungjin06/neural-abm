from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "configs" / "pivot_policy_ladder_consolidation_v3.yaml"
DOCUMENT = ROOT / "docs" / "research" / "pivot-policy-ladder-consolidation-v3.md"
README = ROOT / "README.md"
OPERATORS = ("equals", "contains", "greater_than", "less_than")


def _ledger() -> dict[str, Any]:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


def _inherited() -> dict[str, Any]:
    path = ROOT / _ledger()["inherits"]["ledger"]["path"]
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _all_inherited_ids() -> set[str]:
    ids: set[str] = set()
    v2 = _inherited()
    v1 = yaml.safe_load((ROOT / v2["inherits"]["ledger"]["path"]).read_text(encoding="utf-8"))
    for ledger in (v1, v2):
        ids |= {claim["id"] for status in ledger["claims"] for claim in ledger["claims"][status]}
        ids |= {row["id"] for row in ledger.get("next_steps", [])}
    return ids


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
    checks.extend(("positioning_update", check) for check in ledger["positioning_update"]["evidence"])
    for decision in ledger["framework_decisions"]["decisions"]:
        checks.extend((decision["id"], check) for check in decision.get("evidence", []))
    return checks


def test_ledger_is_a_no_execution_consolidation() -> None:
    boundaries = _ledger()["boundaries"]
    assert _ledger()["status"] == "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION"
    assert boundaries["new_execution"] == "NONE"
    assert boundaries["llm_calls"] == 0
    assert boundaries["generic_nabm_api_changed"] is False


def test_the_inherited_ledger_and_every_source_are_bound_by_hash() -> None:
    ledger = _ledger()
    inherits = ledger["inherits"]
    for row in (inherits["ledger"], inherits["document"], *ledger["sources"]):
        assert _sha256(ROOT / row["path"]) == row["sha256"], row["path"]
    assert (ROOT / inherits["inherited_claims_remain_audited_by"]).exists()


def test_every_evidence_check_holds_and_mutations_are_caught() -> None:
    ledger = _ledger()
    sources = _sources()
    checks = _all_checks(ledger)
    assert len(checks) >= 30
    assert [claim_id for claim_id, check in checks if not _holds(check, sources)] == []
    for _claim_id, check in checks:
        mutated = dict(check)
        (operator,) = [name for name in OPERATORS if name in check]
        value = _resolve(sources[check["source"]], check["pointer"])
        if operator == "equals":
            mutated[operator] = (not value) if isinstance(value, bool) else f"{value}_X"
        elif operator == "greater_than":
            mutated[operator] = value + 1e-9
        elif operator == "less_than":
            mutated[operator] = value - 1e-9
        else:
            continue
        assert not _holds(mutated, sources), check["pointer"]


def test_new_claims_continue_the_inherited_numbering() -> None:
    inherited = _inherited()["claims"]
    claims = _ledger()["claims"]
    new_ids = [claim["id"] for status in claims for claim in claims[status]]
    assert len(new_ids) == len(set(new_ids))
    assert not set(new_ids) & _all_inherited_ids()

    def number(claim_id: str) -> int:
        return int(re.match(r"[CRO](\d+)_", claim_id).group(1))

    for status in ("confirmed", "rejected", "open"):
        previous = max(number(claim["id"]) for claim in inherited[status])
        numbers = [number(claim["id"]) for claim in claims[status]]
        assert numbers == list(range(previous + 1, previous + 1 + len(numbers)))


def test_post_hoc_claims_are_labelled() -> None:
    by_id = {claim["id"]: claim for claim in _ledger()["claims"]["confirmed"]}
    for claim_id in ("C31_phase_a_auc_tracks_abm_selection", "C34_the_llm_agent_fails_on_calibration_not_ranking"):
        assert by_id[claim_id]["evidence_type"] == "POST_HOC_DESCRIPTIVE"
        assert all("post_hoc" in check["pointer"] for check in by_id[claim_id]["evidence"])


def test_status_updates_close_o09_and_n06_against_inherited_items() -> None:
    ledger = _ledger()
    new_ids = {claim["id"] for status in ledger["claims"] for claim in ledger["claims"][status]}
    inherited = _all_inherited_ids()
    updates = {row["id"]: row for row in ledger["status_updates"]}
    assert set(updates) == {"O09_llm_agent_reference", "N06"}
    for update in updates.values():
        assert update["id"] in inherited
        assert set(update.get("answered_by", [])) <= new_ids
    assert updates["O09_llm_agent_reference"]["remaining"] in new_ids
    v2_tier = next(
        tier for tier in _inherited()["positioning"]["tiers"] if tier["id"] == "LLM_AGENT_SIMULATION"
    )
    assert v2_tier["behavior"] == _ledger()["positioning_update"]["previous_behavior"]


def test_next_steps_keep_the_transfer_gated_and_answer_open_questions() -> None:
    ledger = _ledger()
    open_ids = {claim["id"] for claim in ledger["claims"]["open"]} | _all_inherited_ids()
    steps = {row["id"]: row for row in ledger["next_steps"]}
    assert "N06" not in steps
    assert steps["N02"]["authorization"] == "USER_DECISION_REQUIRED"
    assert all(row["answers"] in open_ids for row in steps.values() if "answers" in row)


def test_document_and_readme_state_the_measured_llm_agent_tier() -> None:
    document = " ".join(DOCUMENT.read_text(encoding="utf-8").split())
    for required in ("9 of 9", "110 GPU-seconds", "7 of 9", "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION"):
        assert required in document, required
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "LLM-agent cost is estimated rather than measured" not in readme
    assert "docs/research/pivot-policy-ladder-consolidation-v3.md" in readme
    assert "110 GPU-seconds" in readme
