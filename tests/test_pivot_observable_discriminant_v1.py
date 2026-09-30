from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_observable_discriminant_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_observable_discriminant_v1 as n05  # noqa: E402


_RAW = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _raw() -> dict:
    return copy.deepcopy(_RAW)


def test_contract_plan_binds_sources_and_builds_six_clocks() -> None:
    contract = n05.load_contract(CONFIG)
    plan = n05._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 14
    assert plan["cells"] == 24
    assert len(plan["clocks"]) == 6


def test_thresholds_must_be_the_n01_midpoints_and_seeds_fresh() -> None:
    contract = _raw()
    contract["discriminants"]["post_failure_entry_share"]["threshold"] = 0.2
    with pytest.raises(ValueError, match="N01 midpoint"):
        n05.validate_contract(contract)
    contract = _raw()
    contract["discriminants"]["feature_arm_relative_delay"]["threshold"] = 0.8
    with pytest.raises(ValueError, match="N01 midpoint"):
        n05.validate_contract(contract)
    contract = _raw()
    contract["profiles"]["development"]["base_seed"] = 20261029
    with pytest.raises(ValueError, match="fresh seed"):
        n05.validate_contract(contract)
    contract = _raw()
    contract["observables"]["registered"] = contract["observables"]["registered"][:-1]
    with pytest.raises(ValueError, match="observable candidates"):
        n05.validate_contract(contract)


def test_surge_schedule_peaks_at_one_and_decays_to_the_floor() -> None:
    schedule = n05.surge_schedule(6, 1.5, 0.05, 22)
    assert schedule[5] == pytest.approx(1.0)
    assert schedule[-1] == pytest.approx(0.05, abs=1e-6)
    assert all(a <= b for a, b in zip(schedule[:5], schedule[1:6], strict=True))
    assert all(a >= b for a, b in zip(schedule[5:-1], schedule[6:], strict=True))


def test_a_flat_clock_nests_the_n01_arms_and_the_slot_is_restored() -> None:
    contract = n05.load_contract(CONFIG)
    n05.check_nesting(contract, "smoke", 0)
    assert n05.rdc.HazardCommitAdapter is n05.ORIGINAL_ADAPTER


def test_run_observables_compute_every_candidate_from_one_run() -> None:
    run = {
        "entry_incidence": [0, 0, 2, 4, 2, 1, 1, 0],
        "public_failure_per_step": [0, 0, 0, 0, 1, 0, 0, 0],
        "decline_onset_step": 4,
        "cohort_persistence": [[3, 2, 2], [4, 4, 2]],
        "entry_peak_steps": [4],
    }
    fits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    entries = [3, 3, 4, 4, 4, 4, 5, 5, 6, 7]
    exits = [None, None, None, 6, None, 7, None, None, None, None]
    agents = [SimpleNamespace(base=SimpleNamespace(fit=fit)) for fit in fits]
    adapter = SimpleNamespace(agents=agents, entry_steps=entries, exit_steps=exits)
    original = n05.study1.Researcher.productive_fit
    # The attribute fit is read through the linear-fit property on agent.base.
    n05.study1.Researcher.productive_fit = property(lambda base: base.fit)
    try:
        values = n05.run_observables(run, adapter, horizon=2, steps=8)
    finally:
        n05.study1.Researcher.productive_fit = original
    assert values["post_failure_entry_share"] == pytest.approx(4 / 10)
    assert values["post_decline_entry_share"] == pytest.approx(2 / 10)
    assert values["post_first_cohort_entry_share"] == pytest.approx(4 / 10)
    assert values["post_peak_entry_share"] == pytest.approx(4 / 10)
    assert values["cohort_persistence_drop"] == pytest.approx(1.0 - 0.5)
    assert values["attribute_selection_trend"] > 0.9
    assert values["interior_entry_peak"] is True


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = n05.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    kwargs = {"clocks": ["peak6_width1p5"], "visibilities": ["immediate", "failure_lag_2"]}
    serial = n05.run_study(reduced, "smoke", workers=1, **kwargs)
    parallel = n05.run_study(reduced, "smoke", workers=3, **kwargs)
    assert len(serial) == 2 * 3 * len(n05.MAIN_ARMS)
    assert serial == parallel
    assert all("post_failure_entry_share" in row and "attribute_selection_trend" in row for row in serial)


def _rows(active_cells: set[tuple[str, str]]) -> list[dict]:
    rows = []
    clocks = [row["id"] for row in _RAW["surge_clock"]["clocks"]]
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for clock in clocks:
            for visibility in n05.VISIBILITY_IDS:
                active = (clock, visibility) in active_cells
                for world in n05.FIT_WORLDS:
                    for arm, evidence in n05.MAIN_ARMS:
                        edge = 0.05 if active and arm in n05.NONLINEAR_ARMS and world != "linear_fit" else 0.0
                        rows.append(
                            {
                                "replicate": replicate, "clock_id": clock, "visibility_id": visibility,
                                "world_id": world, "arm_id": arm, "evidence_regime": evidence,
                                "productive_share_of_entrants": 0.30 + edge + wobble,
                                "cumulative_entry_rate": 0.9,
                                "entry_wave_width": 4.0,
                                "post_failure_entry_share": 0.4 if active else 0.1,
                                "interior_entry_peak": True,
                                "model_h3_gap": -1,
                                "post_decline_entry_share": 0.2 if active else 0.0,
                                "post_first_cohort_entry_share": 0.3,
                                "cohort_persistence_drop": None,
                                "post_peak_entry_share": 0.5 if active else 0.6,
                                "attribute_selection_trend": 0.1,
                                "outcome_selection_trend": 0.1 if active else 0.2,
                            }
                        )
    return rows


def test_decision_tests_the_discriminants_and_screens_the_observables() -> None:
    contract = n05.load_contract(CONFIG)
    active = {("peak3_width1p5", "immediate"), ("peak9_width3p0", "failure_lag_4")}
    decision = n05.decide(_rows(active), contract)
    statuses = {(c["clock_id"], c["visibility_id"]): c["status"] for c in decision["cells"]}
    assert {key for key, value in statuses.items() if value == "ENTRY_CHANNEL_ACTIVE"} == active
    assert decision["status_mix"] == {"active_cells": 2, "inactive_cells": 22, "discriminating": True}
    assert decision["post_failure_entry_share_test"]["disposition"] == "CONFIRMED_IN_ALL_CELLS"
    assert decision["design_check"]["disposition"] == "SURGE_CLOCK_PRODUCES_INTERIOR_PEAKS"
    screen = decision["observable_screen"]
    assert screen["post_decline_entry_share"]["disposition"] == "SEPARATES"
    assert screen["post_peak_entry_share"]["direction"] == "ACTIVE_LOWER"
    assert screen["post_first_cohort_entry_share"]["disposition"] == "DOES_NOT_SEPARATE"
    assert screen["cohort_persistence_drop"]["disposition"] == "NOT_DISCRIMINATED"
    assert "post_decline_entry_share" in decision["separating_observables"]


def test_summary_and_outputs_write_from_full_shaped_rows(tmp_path: Path) -> None:
    contract = n05.load_contract(CONFIG)
    rows = _rows({("peak3_width1p5", "immediate")})
    for row in rows:
        row["probe_brier"] = 0.1
    summary = n05.summarize(rows, contract, "development")
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    receipt = n05.write_outputs(
        output_dir=tmp_path / "out", rows=rows, summary=summary, precommit={"commit_sha": "0" * 40}, contract_path=CONFIG
    )
    assert receipt["row_count"] == len(rows) == 16 * 24 * 3 * 4


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = n05.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        n05.delayed.verify_precommit("short", contract)
    assert np.isfinite(n05.surge_schedule(3, 1.5, 0.05, 22)).all()
