from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_relative_delay_cost_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_relative_delay_cost_v1 as study  # noqa: E402


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_contract_plan_binds_sources_and_the_design() -> None:
    contract = study.load_contract(CONFIG)
    plan = study._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 11
    assert plan["q_values"] == [1.0, 0.5, 0.25]
    assert plan["visibility_regimes"] == list(study.VISIBILITY_IDS)
    assert contract["purpose"]["answers_open_question"] == (
        "O04_saturation_boundary_versus_entry_wave_width"
    )


def test_reused_seed_and_changed_learners_are_rejected() -> None:
    contract = _raw()
    contract["profiles"]["development"]["base_seed"] = 20261010
    with pytest.raises(ValueError, match="fresh seed"):
        study.validate_contract(contract)
    contract = _raw()
    contract["batch_learners"]["mlp_batch"]["weight_decay"] = 0.01
    with pytest.raises(ValueError, match="differs from v2"):
        study.validate_contract(contract)


def test_v2_visibility_components_are_preserved() -> None:
    for regime, component in study.abm2.VISIBILITY_COMPONENT.items():
        assert study.VISIBILITY_COMPONENT[regime] == component
    assert len(set(study.VISIBILITY_COMPONENT.values())) == len(study.VISIBILITY_IDS)


def test_timing_observables_on_a_hand_built_run() -> None:
    # Steps 1-10, first open step 3, horizon 2.
    entry = [3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, None]
    exits = [None, None, None, 6, None, 5, None, 6, 7, 7, 7, None]
    timing = study.timing_observables(
        entry, exits, steps=10, first_open_step=3, horizon=2
    )
    assert timing["entry_incidence"] == [0, 0, 4, 2, 5, 0, 0, 0, 0, 0]
    assert timing["entry_peak_steps"] == [5]
    # Cohort 3 keeps 4 of 4 through step 5; cohort 4 keeps 1 of 2 through 6;
    # cohort 5 keeps 1 of 5 through step 7.
    assert timing["cohort_persistence"] == [[3, 4, 4], [4, 2, 1], [5, 5, 1]]
    assert timing["decline_onset_step"] == 4
    assert timing["model_h3_gap"] == 1
    # 90% of 11 entries is reached at step 5, the third open step.
    assert timing["entry_wave_width"] == 3


def test_timing_observables_with_no_entry() -> None:
    timing = study.timing_observables(
        [None, None], [None, None], steps=5, first_open_step=3, horizon=2
    )
    assert timing["entry_peak_steps"] == []
    assert timing["model_h3_gap"] is None
    assert timing["entry_wave_width"] is None


def _cell(q: float, delay: int, width: float, active: bool) -> dict:
    return {
        "q": q,
        "delay": delay,
        "relative_delay": (delay + 1) / width,
        "status": "ENTRY_CHANNEL_ACTIVE" if active else "ENTRY_SATURATED",
    }


def test_relative_delay_disposition_cases() -> None:
    relative = [
        _cell(1.0, 0, 2.0, True),
        _cell(1.0, 1, 2.0, False),
        _cell(0.25, 0, 8.0, True),
        _cell(0.25, 2, 8.0, True),
        _cell(0.25, 4, 8.0, False),
    ]
    result = study.relative_delay_disposition(relative)
    assert result["disposition"] == "RELATIVE_DELAY_SUPPORTED"
    assert result["threshold_interval"] == [0.5, 0.625]
    absolute = [
        _cell(1.0, 0, 2.0, True),
        _cell(1.0, 2, 2.0, False),
        _cell(0.25, 0, 0.5, True),
        _cell(0.25, 2, 0.5, False),
    ]
    assert study.relative_delay_disposition(absolute)["disposition"] == (
        "ABSOLUTE_DELAY_SUPPORTED"
    )
    same_pattern = [
        _cell(1.0, 0, 2.0, True),
        _cell(1.0, 2, 2.0, False),
        _cell(0.25, 0, 2.5, True),
        _cell(0.25, 2, 2.5, False),
    ]
    assert study.relative_delay_disposition(same_pattern)["disposition"] == (
        "NOT_DISCRIMINATED"
    )
    all_active = [_cell(1.0, 0, 2.0, True), _cell(0.25, 4, 8.0, True)]
    assert study.relative_delay_disposition(all_active)["disposition"] == (
        "NOT_DISCRIMINATED"
    )
    neither = [
        _cell(1.0, 0, 2.0, True),
        _cell(1.0, 1, 2.0, False),
        _cell(0.25, 1, 1.0, True),
        _cell(0.25, 0, 1.0, False),
    ]
    assert study.relative_delay_disposition(neither)["disposition"] == (
        "NEITHER_SIMPLE_RULE"
    )


def test_hazard_at_one_admits_everyone_and_below_one_is_deterministic() -> None:
    contract = study.load_contract(CONFIG)
    config = study.study_config(contract, "smoke")
    probe = study.abm2.probe_cohort(contract, 0, "linear_fit")
    runs = {
        q: study.run_arm(
            contract,
            config,
            replicate=0,
            visibility_id="immediate",
            world_id="linear_fit",
            arm_id="fixed_rule",
            evidence_regime="none",
            probe=probe,
            q=q,
        )
        for q in (1.0, 0.25)
    }
    again = study.run_arm(
        contract,
        config,
        replicate=0,
        visibility_id="immediate",
        world_id="linear_fit",
        arm_id="fixed_rule",
        evidence_regime="none",
        probe=probe,
        q=0.25,
    )
    assert again == runs[0.25]
    first_open = config.burn_in_steps
    assert runs[0.25]["cumulative_entry_rate_trajectory"][first_open] < runs[1.0][
        "cumulative_entry_rate_trajectory"
    ][first_open]
    assert runs[1.0]["policy_queries"] == config.agent_count * config.steps
    assert 0 < runs[1.0]["pre_entry_decisions"] <= runs[1.0]["policy_queries"]


def test_smoke_task_nests_the_persistence_and_v2_arms_at_q_one() -> None:
    contract = study.load_contract(CONFIG)
    rows = study.run_task(contract, "smoke", 0, "immediate", "linear_fit", 1.0)
    assert [(row["arm_id"], row["evidence_regime"]) for row in rows] == list(
        study.MAIN_ARMS
    )


def test_nesting_check_detects_a_changed_arm() -> None:
    contract = study.load_contract(CONFIG)
    config = study.study_config(contract, "smoke")
    probe = study.abm2.probe_cohort(contract, 0, "linear_fit")
    runs = {
        arm: study.run_arm(
            contract,
            config,
            replicate=0,
            visibility_id="failure_lag_2",
            world_id="linear_fit",
            arm_id=arm,
            evidence_regime=evidence,
            probe=probe,
            q=1.0,
        )
        for arm, evidence in study.MAIN_ARMS[:1]
    }
    runs["fixed_rule"] = {**runs["fixed_rule"], "active_rate": -1.0}
    with pytest.raises(RuntimeError, match="does not nest"):
        study.check_nesting(
            contract,
            config,
            replicate=0,
            visibility_id="failure_lag_2",
            runs=runs,
            probe=probe,
        )


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = study.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    reduced["profiles"]["smoke"]["steps"] = 6
    reduced["entry_hazard"]["q_values"] = [1.0, 0.25]
    serial = study.run_study(reduced, "smoke", workers=1)
    parallel = study.run_study(reduced, "smoke", workers=3)
    assert len(serial) == 2 * 4 * 3 * len(study.MAIN_ARMS)
    assert serial == parallel


def _rows(entry_share: dict, full_entry: set) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for q in study.Q_VALUES:
            for visibility in study.VISIBILITY_IDS:
                for world in study.FIT_WORLDS:
                    for arm, evidence in study.MAIN_ARMS:
                        rows.append(
                            {
                                "replicate": replicate,
                                "q": q,
                                "visibility_id": visibility,
                                "world_id": world,
                                "arm_id": arm,
                                "evidence_regime": evidence,
                                "productive_share_of_entrants": entry_share.get(
                                    (q, visibility, world, arm), 0.30
                                )
                                + wobble,
                                "cumulative_entry_rate": (
                                    1.0 if (q, visibility) in full_entry else 0.8
                                ),
                                "entry_wave_width": {1.0: 2, 0.5: 4, 0.25: 8}[q],
                            }
                        )
    return rows


def test_decision_classifies_cells_and_tests_relative_delay() -> None:
    contract = study.load_contract(CONFIG)
    share = {}
    # The tree improves axis-world entrant precision while failure evidence
    # arrives within the wave: delay 0 at q = 1, delays up to 2 at q = 0.25.
    for q, delays in ((1.0, (0,)), (0.5, (0, 1)), (0.25, (0, 1, 2))):
        for delay in delays:
            visibility = {0: "immediate", 1: "failure_lag_1", 2: "failure_lag_2"}[delay]
            share[(q, visibility, "axis_interaction_fit", "tree_batch")] = 0.36
    full_entry = {(1.0, "failure_lag_4"), (0.5, "failure_lag_4"), (0.25, "failure_lag_4")}
    decision = study.decide(_rows(share, full_entry), contract)
    by_cell = {(cell["q"], cell["visibility_id"]): cell for cell in decision["cells"]}
    assert by_cell[(1.0, "immediate")]["status"] == "ENTRY_CHANNEL_ACTIVE"
    assert by_cell[(1.0, "failure_lag_2")]["status"] == "ENTRY_CHANNEL_ABSENT"
    assert by_cell[(0.25, "failure_lag_4")]["status"] == "ENTRY_SATURATED"
    assert by_cell[(0.25, "failure_lag_2")]["relative_delay"] == pytest.approx(3 / 8)
    test = decision["relative_delay_test"]
    assert test["disposition"] == "RELATIVE_DELAY_SUPPORTED"
    assert test["active_delays_by_q"] == {"1.0": [0], "0.5": [0, 1], "0.25": [0, 1, 2]}


def test_development_summary_and_outputs_write_from_full_shaped_rows(
    tmp_path: Path,
) -> None:
    contract = study.load_contract(CONFIG)
    template = study.run_task(contract, "smoke", 0, "immediate", "linear_fit", 1.0)
    rows = [
        {**row, "replicate": replicate, "q": q, "visibility_id": visibility, "world_id": world}
        for replicate in range(16)
        for q in study.Q_VALUES
        for visibility in study.VISIBILITY_IDS
        for world in study.FIT_WORLDS
        for row in template
    ]
    summary = study.summarize(rows, contract, "development")
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert len(summary["dispositions"]["cells"]) == 12
    assert summary["llm_reference_counts"]["runs"] == 2304
    receipt = study.write_outputs(
        output_dir=tmp_path / "out",
        rows=rows,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert receipt["row_count"] == 2304
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == sorted(
        [*receipt["outputs"], "receipt.json"]
    )


def test_benchmark_records_every_arm_with_positive_cost() -> None:
    contract = study.load_contract(CONFIG)
    ledger = study.benchmark(contract, profile="smoke", repeats=1)
    assert ledger["status"] == "TIMING_NOT_BYTE_REPRODUCIBLE"
    assert [(r["arm_id"], r["evidence_regime"]) for r in ledger["records"]] == list(
        study.BENCHMARK_ARMS
    )
    assert all(r["median_cpu_seconds_per_run"] > 0 for r in ledger["records"])
    fixed = ledger["records"][0]
    assert fixed["cost_relative_to_fixed_rule"] == pytest.approx(1.0)
    assert np.isfinite([r["median_cpu_ms_per_agent_step"] for r in ledger["records"]]).all()


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = study.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        study.delayed.verify_precommit("short", contract)
