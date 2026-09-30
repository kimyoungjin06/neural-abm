from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_observable_confirmation_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_observable_confirmation_v1 as n08  # noqa: E402


_RAW = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _raw() -> dict:
    return copy.deepcopy(_RAW)


def test_contract_plan_binds_sources_and_builds_new_schedules() -> None:
    contract = n08.load_contract(CONFIG)
    plan = n08._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert plan["cells"] == 24
    assert len(plan["source_bindings"]) == 17
    built = n08.schedules(contract, "development")
    assert set(built["hazard_q0p35"]) == {0.35}
    assert max(built["surge_peak8_width2"]) == pytest.approx(1.0)


def test_thresholds_are_n05_midpoints_schedules_are_new_and_seeds_fresh() -> None:
    contract = _raw()
    contract["thresholds"]["outcome_selection_trend"]["threshold"] = 0.2
    with pytest.raises(ValueError, match="N05 midpoint"):
        n08.validate_contract(contract)
    contract = _raw()
    contract["entry_schedules"][0] = {"id": "again", "type": "hazard", "q": 0.5}
    with pytest.raises(ValueError, match="repeats an opened"):
        n08.validate_contract(contract)
    contract = _raw()
    contract["profiles"]["development"]["base_seed"] = 20261031
    with pytest.raises(ValueError, match="fresh seed"):
        n08.validate_contract(contract)


def test_prediction_directions_follow_the_registered_thresholds() -> None:
    thresholds = n08.load_contract(CONFIG)["thresholds"]
    assert n08.predict("cohort_persistence_drop", -0.1, thresholds) is True
    assert n08.predict("cohort_persistence_drop", 0.05, thresholds) is False
    assert n08.predict("outcome_selection_trend", 0.2, thresholds) is True
    assert n08.predict("outcome_selection_trend", 0.05, thresholds) is False
    assert n08.predict("outcome_selection_trend", None, thresholds) is False


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = n08.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    kwargs = {"schedule_ids": ["hazard_q0p7"], "visibilities": ["immediate", "failure_lag_2"]}
    serial = n08.run_study(reduced, "smoke", workers=1, **kwargs)
    parallel = n08.run_study(reduced, "smoke", workers=3, **kwargs)
    assert len(serial) == 2 * 3 * len(n08.MAIN_ARMS)
    assert serial == parallel


def _rows(active_cells: set[tuple[str, str]], fixed_drop: float = 0.08) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for schedule in _RAW["entry_schedules"]:
            for visibility in n08.VISIBILITY_IDS:
                active = (schedule["id"], visibility) in active_cells
                for world in n08.FIT_WORLDS:
                    for arm, evidence in n08.MAIN_ARMS:
                        edge = 0.05 if active and arm in ("tree_batch", "mlp_batch") and world != "linear_fit" else 0.0
                        fixed = arm == "fixed_rule"
                        rows.append(
                            {
                                "replicate": replicate, "clock_id": schedule["id"], "visibility_id": visibility,
                                "world_id": world, "arm_id": arm, "evidence_regime": evidence,
                                "productive_share_of_entrants": 0.30 + edge + wobble,
                                "cumulative_entry_rate": 0.9,
                                "post_failure_entry_share": 0.4 if active else 0.1,
                                "interior_entry_peak": True,
                                "cohort_persistence_drop": fixed_drop if fixed else (-0.1 if active else 0.1),
                                "outcome_selection_trend": -0.08 if fixed else (0.2 if active else -0.1),
                            }
                        )
    return rows


def test_decision_confirms_observables_and_checks_the_no_learning_control() -> None:
    contract = n08.load_contract(CONFIG)
    active = {("hazard_q0p35", "immediate"), ("surge_peak8_width4", "failure_lag_1")}
    decision = n08.decide(_rows(active), contract)
    assert decision["status_mix"] == {"active_cells": 2, "inactive_cells": 22, "discriminating": True}
    for name in n08.CONFIRMED:
        assert decision["observable_tests"][name]["disposition"] == "CONFIRMED_IN_ALL_CELLS"
    assert decision["joint_rule_both_observables"]["disposition"] == "CONFIRMED_IN_ALL_CELLS"
    assert decision["no_learning_control"]["disposition"] == "CONTROL_READS_INACTIVE_EVERYWHERE"
    leaky = n08.decide(_rows(active, fixed_drop=-0.2), contract)
    assert leaky["no_learning_control"]["disposition"] == "CONTROL_READS_ACTIVE_SOMEWHERE_OBSERVABLES_MAY_TRACK_TIMING"


def test_summary_and_outputs_write_from_full_shaped_rows(tmp_path: Path) -> None:
    contract = n08.load_contract(CONFIG)
    rows = _rows({("hazard_q0p7", "immediate")})
    summary = n08.summarize(rows, contract, "development")
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    receipt = n08.write_outputs(
        output_dir=tmp_path / "out", rows=rows, summary=summary, precommit={"commit_sha": "0" * 40}, contract_path=CONFIG
    )
    assert receipt["row_count"] == 16 * 24 * 3 * 4


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = n08.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        n08.delayed.verify_precommit("short", contract)
