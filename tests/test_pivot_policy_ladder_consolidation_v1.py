from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "configs" / "pivot_policy_ladder_consolidation_v1.yaml"
OPERATORS = ("equals", "contains", "greater_than", "less_than")


def _ledger() -> dict[str, Any]:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict[str, Any]:
    return {
        row["id"]: json.loads((ROOT / row["path"]).read_text(encoding="utf-8"))
        for row in _ledger()["sources"]
    }


def _resolve(document: Any, pointer: str) -> Any:
    value = document
    for key in pointer.split("/"):
        value = value[key]
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
    for target in ledger["empirical_targets"]:
        checks.extend((target["id"], check) for check in target.get("current_blockers", []))
    return checks


def test_ledger_is_a_no_execution_consolidation() -> None:
    ledger = _ledger()
    assert ledger["status"] == "CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION"
    assert ledger["boundaries"]["new_execution"] == "NONE"
    assert ledger["boundaries"]["empirical_rows_read"] == 0
    assert ledger["boundaries"]["generic_nabm_api_changed"] is False


def test_every_source_and_reference_is_bound_by_hash() -> None:
    ledger = _ledger()
    for row in [*ledger["sources"], *ledger["reference_documents"]]:
        assert _sha256(ROOT / row["path"]) == row["sha256"], row["path"]


def test_every_evidence_check_holds_against_the_recorded_results() -> None:
    ledger = _ledger()
    sources = _sources()
    failures = [
        (claim_id, check)
        for claim_id, check in _all_checks(ledger)
        if not _holds(check, sources)
    ]
    assert failures == []


def test_claims_are_unique_and_complete() -> None:
    claims = _ledger()["claims"]
    ids = [claim["id"] for status in claims for claim in claims[status]]
    assert len(ids) == len(set(ids))
    for claim in claims["confirmed"]:
        assert claim["statement"] and claim["evidence"]
    for claim in claims["rejected"]:
        assert claim["statement"] and claim["reason"] and claim["evidence"]
    for claim in claims["open"]:
        assert claim["kind"] in {"EMPIRICAL", "SYNTHETIC"}
        assert claim["priority"] in {1, 2, 3}


def test_each_rejected_registered_claim_cites_the_result_that_overturned_it() -> None:
    by_id = {claim["id"]: claim for claim in _ledger()["claims"]["rejected"]}
    for claim_id in (
        "R02_v2_neural_specific_advantage",
        "R04_abm_v1_tree_dominance",
        "R05_abm_v1_learning_survives_local_evidence",
        "R06_abm_v1_behavior_unchanged",
    ):
        cited = {check["source"] for check in by_id[claim_id]["evidence"]}
        assert len(cited) >= 2, claim_id


def test_empirical_targets_answer_open_questions_in_priority_order() -> None:
    ledger = _ledger()
    open_ids = {claim["id"] for claim in ledger["claims"]["open"]}
    targets = ledger["empirical_targets"]
    assert [target["priority"] for target in targets] == [1, 2, 3]
    assert all(target["answers"] in open_ids for target in targets)
    first = targets[0]
    assert first["id"] == "T01_relative_failure_visibility_delay"
    assert "O04_RESOLVED_SO_BANDS_DO_NOT_DEPEND_ON_THE_STYLIZED_ENTRY_WAVE" in first[
        "requirements_before_comparison"
    ]
    assert first["source_transfer_observed_2026_09_29"][
        "resume_requires_user_authorization"
    ] is True


def test_framework_decisions_follow_the_second_consumer_rule() -> None:
    decisions = {row["id"]: row for row in _ledger()["framework_decisions"]["decisions"]}
    candidates = [
        row_id
        for row_id, row in decisions.items()
        if row["decision"].startswith("EXTRACTION_CANDIDATE")
    ]
    assert candidates == ["F03_out_of_sample_agent_policy_scoring"]
    toy = (ROOT / "src" / "neural_abm" / "toy_classification.py").read_text(
        encoding="utf-8"
    )
    assert "def evaluate_agents(" in toy
    for row_id, row in decisions.items():
        if row["decision"] in {"DOMAIN_OWNED", "CASE_OWNED", "CASE_OWNED_RESEARCH_INSTRUMENTS"}:
            assert row["second_consumer"] == "NONE", row_id


def test_flagship_map_cites_known_sources() -> None:
    ledger = _ledger()
    source_ids = {row["id"] for row in ledger["sources"]}
    for row in ledger["flagship_output_map"]:
        supplied = row["supplied_by"]
        if isinstance(supplied, list) and supplied != ["THIS_CONSOLIDATION"]:
            assert set(supplied) <= source_ids, row["output"]


def test_derived_raw_row_facts_reproduce_when_rows_are_present() -> None:
    derived = _ledger()["derived_raw_row_facts"]
    path = Path(derived["source_path"])
    if not path.exists():
        pytest.skip("machine-local ABM v2 raw rows are not available")
    assert _sha256(path) == derived["source_sha256"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    facts = {fact["id"]: fact for fact in derived["facts"]}

    def step_reaching(trajectory: list[float], fraction: float) -> int:
        final = trajectory[-1]
        return next(
            index + 1
            for index, value in enumerate(trajectory)
            if value >= fraction * final - 1e-12
        )

    half_steps = {
        step_reaching(row["cumulative_entry_rate_trajectory"], 0.5)
        for row in rows
        if row["cumulative_entry_rate_trajectory"][-1] > 0
    }
    assert half_steps == {facts["entry_wave_is_compressed"]["step_of_half_final_entry_in_every_run"]}
    first_failure = facts["first_public_failure_step_by_regime"]
    for row in rows:
        failures = row["public_failure_per_step"]
        observed = next(index + 1 for index, value in enumerate(failures) if value > 0)
        assert observed == first_failure[row["visibility_id"]]
    ninety = facts["field_arms_mean_step_of_ninety_percent_entry"]
    grouped: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        if row["evidence_regime"] == "field_public" and row["arm_id"] in ninety["arms"]:
            grouped.setdefault((row["visibility_id"], row["arm_id"]), []).append(
                step_reaching(row["cumulative_entry_rate_trajectory"], 0.9)
            )
    for (visibility, _arm), steps in grouped.items():
        assert sum(steps) / len(steps) <= ninety[f"{visibility}_at_most"]
