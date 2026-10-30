from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "configs" / "pivot_policy_ladder_consolidation_v14.yaml"
DOCUMENT = ROOT / "docs" / "research" / "pivot-policy-ladder-consolidation-v14.md"
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
    checks.extend(("data_access_state", check) for check in ledger["data_access_state"]["evidence"])
    for row in ledger["invalidated_executions"]:
        checks.extend((row["id"], check) for check in row["evidence"])
    return checks


def test_ledger_is_a_no_execution_consolidation() -> None:
    ledger = _ledger()
    boundaries = ledger["boundaries"]
    assert ledger["status"] == "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION"
    assert boundaries["new_execution"] == "NONE"
    assert boundaries["llm_calls"] == 0
    assert boundaries["generic_nabm_api_changed"] is False
    assert boundaries["package_promotion"] == "ACCEPTED_IN_DECISION_0017_RELEASED_IN_V0_1_0A6"
    assert boundaries["empirical_rows_read_by_its_source"] == 0
    assert boundaries["calibrates_study2"] is False


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


def _chain_numbers(status: str) -> list[int]:
    numbers: list[int] = []
    path = ROOT / _ledger()["inherits"]["ledger"]["path"]
    while True:
        ledger = yaml.safe_load(path.read_text(encoding="utf-8"))
        numbers += [int(re.match(r"[CRO](\d+)_", claim["id"]).group(1)) for claim in ledger["claims"].get(status, [])]
        inherits = ledger.get("inherits")
        if not inherits:
            return numbers
        path = ROOT / inherits["ledger"]["path"]


def test_new_claims_continue_the_inherited_numbering() -> None:
    claims = _ledger()["claims"]
    new_ids = [claim["id"] for status in claims for claim in claims[status]]
    assert len(new_ids) == len(set(new_ids))
    assert not set(new_ids) & _all_inherited_ids()
    for status in ("confirmed", "rejected", "open"):
        previous = max(_chain_numbers(status))
        numbers = [int(re.match(r"[CRO](\d+)_", claim["id"]).group(1)) for claim in claims[status]]
        assert numbers == list(range(previous + 1, previous + 1 + len(numbers))), status
    assert claims["rejected"] == []


def test_status_updates_close_or_carry_the_inherited_next_steps() -> None:
    ledger = _ledger()
    new_ids = {claim["id"] for status in ledger["claims"] for claim in ledger["claims"][status]}
    updates = {row["id"]: row for row in ledger["status_updates"]}
    inherited_steps = {row["id"]: row for row in _inherited()["next_steps"]}
    assert set(updates) == set(inherited_steps) == {"N41", "N42", "N43"}
    for update in updates.values():
        assert update["previous"] == inherited_steps[update["id"]]["authorization"]
        assert update["status"].startswith("DONE")
        assert set(update.get("answered_by", [])) <= new_ids
    assert "PULL_REQUEST_15" in updates["N42"]["status"]
    assert updates["N43"]["status"].endswith(ledger["sources"][0]["record_commit"])
    assert ledger["sources"][0]["freeze_commit"] in updates["N43"]["status"] and "O39" in updates["N43"]["status"]
    steps = {row["id"]: row for row in ledger["next_steps"]}
    assert steps["N45"]["kind"] == inherited_steps["N42"]["kind"] == "REPOSITORY"
    assert {"O37", "O38", "O40"} <= set(re.findall(r"O\d+", steps["N46"]["action"]))
    assert "O39" not in steps["N46"]["action"]


def test_o39_is_answered_and_every_other_open_item_is_carried() -> None:
    ledger = _ledger()
    inherited = _inherited()
    inherited_open = {claim["id"] for claim in inherited["claims"]["open"]} | set(inherited["carried_open_items"])
    answered = {row["id"]: row for row in ledger["answered_open_items"]}
    assert set(answered) == {"O39_study2_at_a_learning_strength_the_data_allow"}
    assert set(ledger["carried_open_items"]) == inherited_open - set(answered)
    confirmed = {claim["id"] for claim in ledger["claims"]["confirmed"]}
    for row in answered.values():
        assert set(row["answered_by"]) <= confirmed


def _new_confirmed() -> list[str]:
    return [claim["id"] for claim in _ledger()["claims"]["confirmed"]]


def test_every_expectation_held_and_only_case_study_results_are_narrowed() -> None:
    ledger = _ledger()
    assert ledger["narrowed_claims"] == [] and ledger["explained_claims"] == [] and ledger["invalidated_executions"] == []
    source = _sources()["learning_strength"]
    assert {key: row["held"] for key, row in source["decision"]["expectations"].items()} == {
        "X1": True, "X2": True, "X3": True, "X4": True}
    narrowed = {row["id"]: row for row in ledger["narrowed_case_study_results"]}
    assert set(narrowed) == {"CASE_STUDY_R5_IMITATIVE_LEARNING_AMPLIFIES_BASELINE_AND_GRANT_PIVOTING",
                             "CASE_STUDY_R6_FAILURE_ONLY_LEARNING_REDUCES_HYPE_FAILURES"}
    for row in narrowed.values():
        assert (ROOT / row["source"]).exists()
        cited = set(re.findall(r"\((C\d+)\)", row["scope"]))
        assert cited and all(any(c.startswith(f"{label}_") for c in _new_confirmed()) for label in cited)


def test_the_post_hoc_claim_says_so_and_no_data_row_is_read() -> None:
    ledger = _ledger()
    (post_hoc,) = [claim for claim in ledger["claims"]["confirmed"] if "post_hoc" in claim["id"]]
    assert post_hoc["statement"].startswith("Post hoc and descriptive.")
    assert {check["pointer"] for check in post_hoc["evidence"]} >= {"post_hoc/status"}
    state = ledger["data_access_state"]
    assert state["confirmatory_outcomes"] == state["g4_outcomes"] == "SEALED"
    assert ledger["boundaries"]["identifies_learning"] is False
    assert ledger["boundaries"]["empirical_rows_read_by_this_consolidation"] == 0
    assert _sources()["learning_strength"]["data_access"]["empirical_rows_read"] == 0


def test_the_new_open_item_follows_from_the_calibration() -> None:
    opened = {claim["id"][:3]: claim for claim in _ledger()["claims"]["open"]}
    assert set(opened) == {"O40"}
    assert set(re.findall(r"\((C\d+)\)", opened["O40"]["motivation"])) == {"C96"}


def test_the_reporting_rules_carry_v13_and_add_one() -> None:
    rules = {row["id"]: row for row in _ledger()["reporting_rules"]}
    v13 = {row["id"]: row for row in _inherited()["reporting_rules"]}
    carried = {row["id"] for row in rules.values() if row.get("carried_from") == "v13"}
    new = {row["id"] for row in rules.values() if row.get("new")}
    assert carried == set(v13) and new == {"LEARNING_STRENGTH_WITH_STUDY2"}
    assert set(rules) == carried | new
    assert all(rules[key]["rule"] == v13[key]["rule"] for key in carried)
    cited = set(re.findall(r"\((C\d+)\)", rules["LEARNING_STRENGTH_WITH_STUDY2"]["rule"]))
    assert cited == {"C95", "C99"}
    assert all(any(c.startswith(f"{label}_") for c in _new_confirmed()) for label in cited)


def test_the_next_steps_keep_outward_actions_behind_the_user() -> None:
    steps = {row["id"]: row for row in _ledger()["next_steps"]}
    assert set(steps) == {"N44", "N45", "N46"}
    assert not set(steps) & _all_inherited_ids()
    assert "OUTSIDE_THE_REPOSITORY" in steps["N44"]["action"]
    assert steps["N45"]["authorization"] == "USER_CONFIRMATION_REQUIRED"
    assert steps["N46"]["authorization"] == "USER_CHOICE_THEN_SEPARATE_PRECOMMIT"


def test_document_and_readme_state_the_results_narrowly() -> None:
    document = " ".join(DOCUMENT.read_text(encoding="utf-8").split())
    for required in (
        "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION",
        "O39_THE_LOOP_FALLS_BELOW_MATERIAL_AT_THE_DATA_ALLOWED_STRENGTH",
        "+0.0161 [+0.0117, +0.0208] under grants",
        "No ledger claim is narrowed.",
        "Nothing is rejected.",
        "The confirmatory and G4 outcomes stay sealed.",
        "puts Study 2's agent steps and OpenAlex topic quarters on one log-odds scale",
    ):
        assert required in document, required
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "docs/research/pivot-policy-ladder-consolidation-v14.md" in readme
    assert "Its imitative loop of failed pivots fell below material size" in readme
    assert "puts agent steps and topic quarters on one log-odds scale" in readme
    assert "no claim about real information spread" in readme
