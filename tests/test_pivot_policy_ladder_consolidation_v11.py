from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "configs" / "pivot_policy_ladder_consolidation_v11.yaml"
DOCUMENT = ROOT / "docs" / "research" / "pivot-policy-ladder-consolidation-v11.md"
README = ROOT / "README.md"
OPERATORS = ("equals", "contains", "greater_than", "less_than")


def _ledger() -> dict[str, Any]:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


def _inherited() -> dict[str, Any]:
    return yaml.safe_load((ROOT / _ledger()["inherits"]["ledger"]["path"]).read_text(encoding="utf-8"))


def _all_inherited_ids() -> set[str]:
    ids: set[str] = set()
    path = ROOT / _ledger()["inherits"]["ledger"]["path"]
    while True:
        ledger = yaml.safe_load(path.read_text(encoding="utf-8"))
        ids |= {claim["id"] for status in ledger["claims"] for claim in ledger["claims"][status]}
        ids |= {row["id"] for row in ledger.get("next_steps", [])}
        inherits = ledger.get("inherits")
        if not inherits:
            return ids
        path = ROOT / inherits["ledger"]["path"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, Any]:
    return {row["id"]: json.loads((ROOT / row["path"]).read_text(encoding="utf-8")) for row in _ledger()["sources"]}


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
    checks.extend(("paper_direction", check) for check in ledger["paper_direction"]["evidence"])
    return checks


def test_ledger_is_a_no_execution_consolidation() -> None:
    ledger = _ledger()
    boundaries = ledger["boundaries"]
    assert ledger["status"] == "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION"
    assert boundaries["new_execution"] == "NONE"
    assert boundaries["llm_calls"] == 0
    assert boundaries["generic_nabm_api_changed"] is False
    assert boundaries["package_promotion"] == "ACCEPTED_IN_DECISION_0017_RELEASED_IN_V0_1_0A6"


def test_the_inherited_ledger_and_every_source_are_bound_by_hash() -> None:
    ledger = _ledger()
    inherits = ledger["inherits"]
    for row in (inherits["ledger"], inherits["document"], *ledger["sources"]):
        assert _sha256(ROOT / row["path"]) == row["sha256"], row["path"]
    assert (ROOT / inherits["inherited_claims_remain_audited_by"]).exists()


def test_every_source_was_recorded_after_its_freeze() -> None:
    for row in _ledger()["sources"]:
        source = json.loads((ROOT / row["path"]).read_text(encoding="utf-8"))
        frozen = source.get("precommit_commit") or source.get("code_commit")
        assert frozen.startswith(row["freeze_commit"]), row["id"]
        assert row["record_commit"] != row["freeze_commit"]


def test_every_evidence_check_holds_and_mutations_are_caught() -> None:
    sources = _sources()
    checks = _all_checks(_ledger())
    assert len(checks) >= 20
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
            mutated[operator] = f"{check[operator]}_X"
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


def test_status_updates_close_or_carry_the_inherited_next_steps() -> None:
    ledger = _ledger()
    new_ids = {claim["id"] for status in ledger["claims"] for claim in ledger["claims"][status]}
    updates = {row["id"]: row for row in ledger["status_updates"]}
    inherited_steps = {row["id"]: row for row in _inherited()["next_steps"]}
    assert set(updates) == set(inherited_steps) == {"N31", "N32", "N33", "N34"}
    for update in updates.values():
        assert update["previous"] == inherited_steps[update["id"]]["authorization"]
        assert set(update.get("answered_by", [])) <= new_ids
    assert updates["N32"]["status"].startswith("DONE")
    steps = {row["id"]: row for row in ledger["next_steps"]}
    for step_id in ("N31", "N33", "N34"):
        carried = re.fullmatch(r"CARRIED_FORWARD_AS_(N\d+)", updates[step_id]["status"]).group(1)
        assert carried in steps
    assert steps["N36"]["kind"] == inherited_steps["N33"]["kind"] == "REPOSITORY"
    assert "O36" in steps["N37"]["action"]


def test_inherited_open_items_are_answered_or_carried() -> None:
    ledger = _ledger()
    inherited = _inherited()
    new_ids = {claim["id"] for status in ledger["claims"] for claim in ledger["claims"][status]}
    answered = {row["id"]: row["answered_by"] for row in ledger["answered_open_items"]}
    inherited_open = {claim["id"] for claim in inherited["claims"]["open"]} | set(inherited["carried_open_items"])
    assert set(answered) == {"O35_why_the_linear_learner_loses_in_the_linear_world"}
    assert set(answered) | set(ledger["carried_open_items"]) == inherited_open
    assert not set(answered) & set(ledger["carried_open_items"])
    for claim_ids in answered.values():
        assert claim_ids and set(claim_ids) <= new_ids


def _new_confirmed() -> list[str]:
    return [claim["id"] for claim in _ledger()["claims"]["confirmed"]]


def test_no_claim_is_narrowed_or_explained() -> None:
    ledger = _ledger()
    assert ledger["narrowed_claims"] == [] and ledger["explained_claims"] == []


def test_the_new_open_item_follows_from_the_diagnosis() -> None:
    (opened,) = _ledger()["claims"]["open"]
    assert opened["id"].startswith("O36_")
    cited = set(re.findall(r"\((C\d+)\)", opened["motivation"]))
    assert cited == {"C80", "C81"}


def test_the_reporting_rules_carry_v10_and_add_one() -> None:
    rules = {row["id"]: row for row in _ledger()["reporting_rules"]}
    v10 = {row["id"] for row in _inherited()["reporting_rules"]}
    carried = {row["id"] for row in rules.values() if row.get("carried_from") == "v10"}
    added = {row["id"] for row in rules.values() if "carried_from" not in row and "replaces" not in row}
    assert carried == v10
    assert added == {"CONVERGED_CAPACITY_COMPARISON"}
    assert set(rules) == carried | added
    cited = set(re.findall(r"\((C\d+)\)", rules["CONVERGED_CAPACITY_COMPARISON"]["rule"]))
    assert cited and all(any(c.startswith(f"{tag}_") for c in _new_confirmed()) for tag in cited)


def test_the_next_steps_keep_outward_actions_behind_the_user() -> None:
    steps = {row["id"]: row for row in _ledger()["next_steps"]}
    assert set(steps) == {"N35", "N36", "N37"}
    assert not set(steps) & _all_inherited_ids()
    assert "OUTSIDE_THE_REPOSITORY" in steps["N35"]["action"]
    assert steps["N36"]["authorization"] == "USER_CONFIRMATION_REQUIRED"
    assert steps["N37"]["authorization"] == "USER_CHOICE_THEN_SEPARATE_PRECOMMIT"


def test_document_and_readme_state_the_results_narrowly() -> None:
    document = " ".join(DOCUMENT.read_text(encoding="utf-8").split())
    for required in (
        "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION",
        "O35_LEARNER_PARTLY_OPTIMIZATION",
        "training, not representation",
        "Converging the registered linear objective closes 71% of the gap",
        "No earlier claim is narrowed",
    ):
        assert required in document, required
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "docs/research/pivot-policy-ladder-consolidation-v11.md" in readme
    assert "the linear learner's deficit against the MLP is mostly its training budget" in readme
    assert "no claim about real information spread" in readme