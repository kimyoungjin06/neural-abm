from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_observable_one_sided_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_observable_one_sided_v1 as n09  # noqa: E402


_RAW = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _raw() -> dict:
    return copy.deepcopy(_RAW)


def test_contract_plan_binds_sources_and_builds_new_shapes() -> None:
    contract = n09.load_contract(CONFIG)
    plan = n09._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert plan["cells"] == 24
    built = n09.schedules(contract, "development")
    assert built["decay_tau3"][0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(built["decay_tau3"], built["decay_tau3"][1:], strict=False))
    assert built["ramp_8"][7] == pytest.approx(1.0) and built["ramp_8"][0] < 0.2


def test_thresholds_are_unchanged_schedules_new_and_seeds_fresh() -> None:
    contract = _raw()
    contract["thresholds"]["outcome_selection_trend"]["threshold"] = 0.05
    with pytest.raises(ValueError, match="unchanged"):
        n09.validate_contract(contract)
    contract = _raw()
    contract["entry_schedules"][0] = {"id": "again", "type": "hazard", "q": 0.35}
    with pytest.raises(ValueError, match="repeats an opened"):
        n09.validate_contract(contract)
    contract = _raw()
    contract["profiles"]["development"]["base_seed"] = 20261033
    with pytest.raises(ValueError, match="fresh seed"):
        n09.validate_contract(contract)


def test_one_sided_rule_counts_false_actives_and_needs_enough_cells() -> None:
    rule = _RAW["decision_contract"]["one_sided"]
    active = [True] * 6 + [False] * 18
    good = n09.one_sided([True] * 5 + [False] * 19, active, rule)
    assert good["disposition"] == "SUFFICIENT_SIGN_CONFIRMED"
    assert good["sensitivity"] == pytest.approx(5 / 6)
    bad = n09.one_sided([True] * 5 + [True] + [False] * 18, [True] * 5 + [False] * 19, rule)
    assert bad["disposition"] == "SUFFICIENT_SIGN_FAILS"
    thin = n09.one_sided([True] * 2 + [False] * 22, active, rule)
    assert thin["disposition"] == "NOT_INFORMATIVE"


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = n09.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    kwargs = {"schedule_ids": ["decay_tau3"], "visibilities": ["immediate", "failure_lag_2"]}
    serial = n09.run_study(reduced, "smoke", workers=1, **kwargs)
    parallel = n09.run_study(reduced, "smoke", workers=3, **kwargs)
    assert len(serial) == 2 * 3 * len(n09.MAIN_ARMS)
    assert serial == parallel


def _rows(active_cells: set[tuple[str, str]], false_flag: tuple[str, str] | None = None) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for schedule in _RAW["entry_schedules"]:
            for visibility in n09.VISIBILITY_IDS:
                cell = (schedule["id"], visibility)
                active = cell in active_cells
                flagged = active or cell == false_flag
                for world in n09.FIT_WORLDS:
                    for arm, evidence in n09.MAIN_ARMS:
                        edge = 0.05 if active and arm in ("tree_batch", "mlp_batch") and world != "linear_fit" else 0.0
                        fixed = arm == "fixed_rule"
                        rows.append(
                            {
                                "replicate": replicate, "clock_id": schedule["id"], "visibility_id": visibility,
                                "world_id": world, "arm_id": arm, "evidence_regime": evidence,
                                "productive_share_of_entrants": 0.30 + edge + wobble,
                                "cumulative_entry_rate": 0.9, "post_failure_entry_share": 0.4 if active else 0.1,
                                "interior_entry_peak": True,
                                "cohort_persistence_drop": 0.08 if fixed else (-0.1 if flagged else 0.1),
                                "outcome_selection_trend": -0.08 if fixed else (0.2 if flagged else -0.1),
                            }
                        )
    return rows


def test_decision_confirms_or_fails_the_sufficient_sign() -> None:
    contract = n09.load_contract(CONFIG)
    ids = [row["id"] for row in _RAW["entry_schedules"]]
    active = {(ids[0], "immediate"), (ids[1], "immediate"), (ids[2], "immediate"), (ids[3], "immediate"), (ids[4], "immediate")}
    decision = n09.decide(_rows(active), contract)
    for entry in decision["one_sided_tests"].values():
        assert entry["disposition"] == "SUFFICIENT_SIGN_CONFIRMED"
    assert decision["no_learning_control"]["disposition"] == "CONTROL_READS_INACTIVE_EVERYWHERE"
    failing = n09.decide(_rows(active, false_flag=(ids[5], "failure_lag_4")), contract)
    assert failing["one_sided_tests"]["outcome_selection_trend"]["disposition"] == "SUFFICIENT_SIGN_FAILS"
    assert failing["one_sided_either_observable"]["false_active"] == 1


def test_summary_and_outputs_write_from_full_shaped_rows(tmp_path: Path) -> None:
    contract = n09.load_contract(CONFIG)
    summary = n09.summarize(_rows({("decay_tau3", "immediate")}), contract, "development")
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    receipt = n09.write_outputs(
        output_dir=tmp_path / "out", rows=_rows(set()), summary=summary, precommit={"commit_sha": "0" * 40}, contract_path=CONFIG
    )
    assert receipt["row_count"] == 16 * 24 * 3 * 4
    assert "One-sided tests" in (tmp_path / "out" / "scorecard.md").read_text(encoding="utf-8")


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = n09.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        n09.delayed.verify_precommit("short", contract)
