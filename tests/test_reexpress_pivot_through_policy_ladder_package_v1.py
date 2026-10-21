from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import reexpress_pivot_through_policy_ladder_package_v1 as rx  # noqa: E402


# A stand-in for the hash-bound LLM prior, which lives outside the repository.
SYNTHETIC_LLM_PRIOR = {"weights": [0.8, -0.3, 0.5, -0.6, -0.4, 0.2, 0.3, 0.1, 0.4], "bias": -1.2}
WORLD = "smooth_interaction_fit"


def _dump(rows: list[dict]) -> list[str]:
    return [json.dumps(row, sort_keys=True) for row in rows]


@pytest.fixture(autouse=True)
def _single_thread() -> None:
    torch.set_num_threads(1)


def _pivot_field_rows(visibility_id: str) -> list[dict]:
    contract = rx.abm2.load_contract()
    config = rx.abm2.study_config(contract, "smoke")
    probe = rx.abm2.probe_cohort(contract, 1, WORLD)
    with threadpool_limits(limits=1):
        return [
            {"replicate": 1, "visibility_id": visibility_id, "world_id": WORLD, "arm_id": arm,
             "evidence_regime": "field_public",
             **rx.abm2.run_arm(contract, config, replicate=1, visibility_id=visibility_id, world_id=WORLD,
                               arm_id=arm, evidence_regime="field_public", probe=probe)}
            for arm in rx.FIELD_ARMS
        ]


def _package_field_rows(visibility_id: str, insertion: str) -> list[dict]:
    with threadpool_limits(limits=1):
        return rx.run_abm2_task(rx.abm2.load_contract(), "smoke", 1, visibility_id, WORLD, insertion)


@pytest.mark.parametrize("condition_id", ["hand", "hybrid_20"])
def test_a_v3_smoke_task_through_the_package_makes_pivots_rows(condition_id: str) -> None:
    contract = rx.pilot3.load_contract()
    with threadpool_limits(limits=1):
        pivot = rx.pilot3.run_task(contract, "smoke", 0, WORLD, condition_id, SYNTHETIC_LLM_PRIOR)
        package = rx.run_v3_task(contract, "smoke", 0, WORLD, condition_id, SYNTHETIC_LLM_PRIOR)
    assert _dump(package) == _dump(pivot)


@pytest.mark.parametrize("insertion", ["release", "event"])
def test_delayed_field_arms_through_the_package_make_pivots_rows(insertion: str) -> None:
    assert _dump(_package_field_rows("failure_lag_4", insertion)) == _dump(_pivot_field_rows("failure_lag_4"))
    assert rx.abm2.ProbedLearner.__name__ == "ProbedLearner"
    assert rx.delayed.DelayedPivotEnvironment.ingest_private_events is rx.ORIGINAL_INGEST


def test_event_insertion_needs_the_ordering_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the rule (the frozen core's insertion order), delayed rows differ."""

    pivot = _dump(_pivot_field_rows("failure_lag_4"))
    monkeypatch.setattr(rx.ladder, "OutcomeRecord", core.OutcomeRecord)
    insertion_order = _dump(_package_field_rows("failure_lag_4", "event"))
    assert all(ours != theirs for ours, theirs in zip(insertion_order, pivot, strict=True))


def test_compare_reports_each_mismatched_field() -> None:
    recorded = [{"replicate": 0, "visibility_id": "immediate", "world_id": "w", "arm_id": "a",
                 "evidence_regime": "field_public", "x": 1.0}]
    report = rx.compare([dict(recorded[0], x=2.0)], recorded)
    assert report["identical_rows"] == 0 and report["mismatched"] == {"0/immediate/w/a/field_public": ["x"]}
    assert rx.compare(recorded, recorded)["by_group"] == {"immediate/a": {"rows": 1, "identical": 1}}
