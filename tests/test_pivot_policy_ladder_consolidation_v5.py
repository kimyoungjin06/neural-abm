from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "configs" / "pivot_policy_ladder_consolidation_v5.yaml"
DOCUMENT = ROOT / "docs" / "research" / "pivot-policy-ladder-consolidation-v5.md"
README = ROOT / "README.md"
OPERATORS = ("equals", "contains", "greater_than", "less_than")


def _ledger() -> dict[str, Any]:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


def _inherited() -> dict[str, Any]:
    path = ROOT / _ledger()["inherits"]["ledger"]["path"]
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
    checks.extend(("empirical_stage", check) for check in ledger["empirical_stage"]["evidence"])
    checks.extend(("paper_direction", check) for check in ledger["paper_direction"]["evidence"])
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


def test_status_updates_close_inherited_items_only() -> None:
    ledger = _ledger()
    new_ids = {claim["id"] for status in ledger["claims"] for claim in ledger["claims"][status]}
    inherited = _all_inherited_ids()
    updates = {row["id"]: row for row in ledger["status_updates"]}
    assert set(updates) == {
        "O17_empirical_persistence_definition_and_threshold_transfer",
        "O18_empirical_source_choice",
        "N02",
        "N10",
        "N04",
    }
    for update in updates.values():
        assert update["id"] in inherited
        assert set(update.get("answered_by", [])) <= new_ids


def test_the_current_n11_design_is_closed_without_overclaiming() -> None:
    stage = _ledger()["empirical_stage"]
    assert stage["current_design"] == "CLOSED_BEFORE_CONFIRMATORY_SCORING"
    assert stage["N11c"] == stage["N11d"] == "NOT_PROPOSED"
    assert stage["closure"].endswith("This does not imply the absence of learning.")
    assert "THE_ABSENCE_OF_LEARNING_IN_THE_DATA" in stage["not_implied"]
    assert "THAT_NO_METHOD_COULD_DISTINGUISH_LEARNING_FROM_ANTICIPATION_IN_THIS_DATA" in stage["not_implied"]
    assert "N11_CONFIRMATORY_OUTCOME_LABELS" in stage["data_access"]["not_read"]
    assert "N11B_THE_G4_GUARD_AUTHOR_ID_COLUMN" in stage["data_access"]["read"]


def test_zero_of_eight_is_reported_with_its_upper_bound() -> None:
    rules = {row["id"]: row["rule"] for row in _ledger()["reporting_rules"]}
    assert set(rules) == {"ZERO_OF_N", "MATCHES", "LLM_SCOPE", "NEURAL_NECESSITY"}
    upper = 1 - 0.05 ** (1 / 8)
    assert round(upper, 2) == 0.31
    assert "31%" in rules["ZERO_OF_N"]
    assert "not a test of statistical equivalence" in rules["MATCHES"]


def test_paper_direction_keeps_performance_conditional() -> None:
    direction = _ledger()["paper_direction"]
    assert direction["paper_1"] == "METHOD_AND_FRAMEWORK"
    assert direction["performance_comparisons"] == "CONDITIONAL_CASES_NOT_THE_CENTRAL_CLAIM"
    steps = {row["id"]: row for row in _ledger()["next_steps"]}
    assert set(steps) == {"N12", "N13", "N14"}
    assert not set(steps) & _all_inherited_ids()
    assert steps["N13"]["authorization"] == "SEPARATE_PRECOMMIT_AFTER_N12"


def test_document_and_readme_state_the_closure() -> None:
    lines = DOCUMENT.read_text(encoding="utf-8").splitlines()
    document = " ".join(" ".join(line.removeprefix(">") for line in lines).split())
    for required in (
        "This does not imply the absence of learning.",
        "about 31%",
        "0.02",
        "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION",
        "W1 1 of 8 and the confounded W2 3 of 8",
    ):
        assert required in document, required
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "docs/research/pivot-policy-ladder-consolidation-v5.md" in readme
    assert "registered 0.02 margin" in readme
