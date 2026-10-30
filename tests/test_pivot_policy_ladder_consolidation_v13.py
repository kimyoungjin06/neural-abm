from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "configs" / "pivot_policy_ladder_consolidation_v13.yaml"
DOCUMENT = ROOT / "docs" / "research" / "pivot-policy-ladder-consolidation-v13.md"
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
    assert boundaries["empirical_rows_read_by_its_source"] == "DISCOVERY_PARTITION_ONLY"


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


def test_status_updates_close_or_carry_the_inherited_next_steps() -> None:
    ledger = _ledger()
    new_ids = {claim["id"] for status in ledger["claims"] for claim in ledger["claims"][status]}
    updates = {row["id"]: row for row in ledger["status_updates"]}
    inherited_steps = {row["id"]: row for row in _inherited()["next_steps"]}
    assert set(updates) == set(inherited_steps) == {"N38", "N39", "N40"}
    for update in updates.values():
        assert update["previous"] == inherited_steps[update["id"]]["authorization"]
        assert update["status"].startswith("DONE")
        assert set(update.get("answered_by", [])) <= new_ids
    assert "PULL_REQUEST_14" in updates["N39"]["status"]
    assert updates["N40"]["status"].endswith(ledger["sources"][0]["record_commit"])
    assert ledger["sources"][0]["freeze_commit"] in updates["N40"]["status"]
    steps = {row["id"]: row for row in ledger["next_steps"]}
    assert steps["N42"]["kind"] == inherited_steps["N39"]["kind"] == "REPOSITORY"
    assert {"O37", "O38", "O39"} <= set(re.findall(r"O\d+", steps["N43"]["action"]))


def test_inherited_open_items_are_carried_and_none_is_answered() -> None:
    ledger = _ledger()
    inherited = _inherited()
    inherited_open = {claim["id"] for claim in inherited["claims"]["open"]} | set(inherited["carried_open_items"])
    assert ledger["answered_open_items"] == []
    assert set(ledger["carried_open_items"]) == inherited_open


def _new_confirmed() -> list[str]:
    return [claim["id"] for claim in _ledger()["claims"]["confirmed"]]


def test_the_failed_expectations_are_rejected_and_nothing_is_narrowed() -> None:
    ledger = _ledger()
    rejected = {claim["id"][:3]: claim for claim in ledger["claims"]["rejected"]}
    assert set(rejected) == {"R24", "R25", "R26"}
    pointers = {check["pointer"] for claim in rejected.values() for check in claim["evidence"]}
    assert {"decision/expectations/ES2/held", "decision/expectations/ES3/held", "decision/expectations/EE2/held"} <= pointers
    assert ledger["narrowed_claims"] == [] and ledger["explained_claims"] == [] and ledger["invalidated_executions"] == []
    source = _sources()["peer_outcome_contrast"]
    assert {key: row["held"] for key, row in source["decision"]["expectations"].items()} == {
        "ES2": False, "ES3": False, "EE1": True, "EE2": False}


def test_the_post_hoc_claim_says_so_and_the_data_state_is_recorded() -> None:
    ledger = _ledger()
    (post_hoc,) = [claim for claim in ledger["claims"]["confirmed"] if "post_hoc" in claim["id"]]
    assert post_hoc["statement"].startswith("Post hoc and descriptive.")
    assert {check["pointer"] for check in post_hoc["evidence"]} >= {"post_hoc/status"}
    state = ledger["data_access_state"]
    assert state["confirmatory_outcomes"] == state["g4_outcomes"] == "SEALED"
    assert ledger["boundaries"]["identifies_learning"] is False
    assert ledger["boundaries"]["empirical_rows_read_by_this_consolidation"] == 0


def test_the_new_open_items_follow_from_the_contrast() -> None:
    opened = {claim["id"][:3]: claim for claim in _ledger()["claims"]["open"]}
    assert set(opened) == {"O38", "O39"}
    assert set(re.findall(r"\((C\d+)\)", opened["O38"]["motivation"])) == {"C90"}
    assert set(re.findall(r"\((C\d+)\)", opened["O39"]["motivation"])) == {"C92", "C93"}


def test_the_reporting_rules_carry_v12_and_add_one() -> None:
    rules = {row["id"]: row for row in _ledger()["reporting_rules"]}
    v12 = {row["id"] for row in _inherited()["reporting_rules"]}
    carried = {row["id"] for row in rules.values() if row.get("carried_from") == "v12"}
    new = {row["id"] for row in rules.values() if row.get("new")}
    assert carried == v12 and new == {"GENERATOR_FIRST_PLACEBO"}
    assert set(rules) == carried | new
    cited = set(re.findall(r"\((C\d+)\)", rules["GENERATOR_FIRST_PLACEBO"]["rule"]))
    assert cited == {"C90"} and any(c.startswith("C90_") for c in _new_confirmed())


def test_the_next_steps_keep_outward_actions_behind_the_user() -> None:
    steps = {row["id"]: row for row in _ledger()["next_steps"]}
    assert set(steps) == {"N41", "N42", "N43"}
    assert not set(steps) & _all_inherited_ids()
    assert "OUTSIDE_THE_REPOSITORY" in steps["N41"]["action"]
    assert steps["N42"]["authorization"] == "USER_CONFIRMATION_REQUIRED"
    assert steps["N43"]["authorization"] == "USER_CHOICE_THEN_SEPARATE_PRECOMMIT"


def test_document_and_readme_state_the_results_narrowly() -> None:
    document = " ".join(DOCUMENT.read_text(encoding="utf-8").split())
    for required in (
        "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION",
        "E4_NOT_INTERPRETABLE_THE_ESTIMATOR_FLAGS_STUDY2_FROZEN_ARM",
        "the data DiD (−0.062 [−0.492, +0.538]) is not interpreted",
        "These are associations, not effects.",
        "No earlier claim is narrowed.",
        "The confirmatory and G4 outcomes stay sealed.",
    ):
        assert required in document, required
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "docs/research/pivot-policy-ladder-consolidation-v13.md" in readme
    assert "flagged learning in Study 2's own no-learning arm" in readme
    assert "no claim about real information spread" in readme
