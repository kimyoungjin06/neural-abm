from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_entry_matched_v1.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task functions.
import prototype_pivot_llm_prior_entry_matched_v1 as study  # noqa: E402


PRIOR_PATH = Path(
    yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["v2_calibrated_priors"]["path"]
)
needs_v2 = pytest.mark.skipif(not PRIOR_PATH.exists(), reason="v2 priors are machine-local")
FAKE_PRIOR = {"weights": [0.5, 2.0, 1.5, -2.0, -1.0, 0.5, 0.0, 0.0, 0.0], "bias": -1.5}


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _quick(contract: dict, *, replicates: int = 2, iterations: int = 12) -> dict:
    """A smoke-sized calibration whose entry resolution needs a looser tolerance."""

    quick = copy.deepcopy(contract)
    quick["entry_calibration"]["replicates"] = replicates
    quick["entry_calibration"]["bisection_iterations"] = iterations
    quick["entry_calibration"]["in_sample_relative_tolerance"] = 0.25
    return quick


def test_contract_plan_binds_sources_and_the_exploration_floor() -> None:
    contract = study.load_contract(CONFIG)
    plan = study._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 15
    for condition in contract["conditions"]:
        epsilon = plan["epsilon_by_condition"][condition["id"]]
        assert 1.0 - (1.0 - epsilon) ** 16 == pytest.approx(condition["exploratory_share"])
    assert plan["epsilon_by_condition"]["hand_at_llm_entry_20"] == plan["epsilon_by_condition"]["hybrid_20"]


def test_reused_seeds_llm_calls_and_self_matching_are_rejected() -> None:
    contract = _raw()
    contract["entry_calibration"]["base_seed"] = 20261024
    with pytest.raises(ValueError, match="fresh seed"):
        study.validate_contract(contract)
    contract = _raw()
    contract["entry_calibration"]["probe_base_seed"] = contract["probe_cohort"]["base_seed"]
    with pytest.raises(ValueError, match="fresh seed"):
        study.validate_contract(contract)
    contract = _raw()
    contract["boundaries"]["new_llm_calls"] = 1
    with pytest.raises(ValueError, match="no LLM calls"):
        study.validate_contract(contract)
    contract = _raw()
    contract["conditions"][3]["level"] = "hand_entry"
    with pytest.raises(ValueError, match="other direction"):
        study.validate_contract(contract)


@needs_v2
def test_llm_prior_is_the_hash_bound_v2_base_rate_prior(tmp_path: Path) -> None:
    contract = study.load_contract(CONFIG)
    prior = study.load_llm_prior(contract)
    stored = json.loads(PRIOR_PATH.read_text(encoding="utf-8"))["priors"]["llm_base_rate"]
    assert prior == {"weights": stored["weights"], "bias": stored["bias"]}
    copy_path = tmp_path / "calibrated_priors.json"
    shutil.copy(PRIOR_PATH, copy_path)
    copy_path.write_text(copy_path.read_text(encoding="utf-8").replace("1", "2", 1), encoding="utf-8")
    tampered = copy.deepcopy(contract)
    tampered["v2_calibrated_priors"]["path"] = str(copy_path)
    with pytest.raises(RuntimeError, match="changed"):
        study.load_llm_prior(tampered)


def test_matched_priors_keep_their_weights_and_shift_only_the_intercept() -> None:
    contract = study.load_contract(CONFIG)
    shifts = {"llm_entry": -1.25, "hand_entry": 0.75}
    rows = {row["id"]: row for row in contract["conditions"]}
    assert study.condition_prior(rows["hand"], FAKE_PRIOR, shifts) is None
    assert study.condition_prior(rows["llm_selective"], FAKE_PRIOR, shifts) == FAKE_PRIOR
    hand = study.hand_prior()
    matched_hand = study.condition_prior(rows["hand_at_llm_entry"], FAKE_PRIOR, shifts)
    assert matched_hand == {"weights": hand["weights"], "bias": hand["bias"] - 1.25}
    matched_llm = study.condition_prior(rows["llm_at_hand_entry"], FAKE_PRIOR, shifts)
    assert matched_llm == {"weights": FAKE_PRIOR["weights"], "bias": FAKE_PRIOR["bias"] + 0.75}


def test_hand_prior_through_the_slot_reproduces_the_hand_rule() -> None:
    contract = study._calibration_contract(study.load_contract(CONFIG))
    for world in study.FIT_WORLDS:
        assert study.hand_slot_nesting_task((contract, "smoke", 0, world)) is True
    shifted = study.run_arms(
        contract, "smoke", 0, "linear_fit", study.shifted(study.hand_prior(), 3.0), 0.0, ("fixed_rule",)
    )
    native = study.run_arms(contract, "smoke", 0, "linear_fit", None, 0.0, ("fixed_rule",))
    assert shifted != native


def test_calibration_matches_entry_deterministically_in_serial_and_parallel() -> None:
    contract = _quick(study.load_contract(CONFIG))
    with study.TaskMap(1) as serial_map:
        serial = study.calibrate_entry(contract, "smoke", FAKE_PRIOR, serial_map)
    with study.TaskMap(3) as parallel_map:
        parallel = study.calibrate_entry(contract, "smoke", FAKE_PRIOR, parallel_map)
    assert serial == parallel
    assert serial["status"] == "ENTRY_CALIBRATION_COMPLETE"
    assert len(serial["records"]) == 2 * len(study.FIT_WORLDS)
    for record in serial["records"]:
        assert abs(record["in_sample_ratio"] - 1.0) <= 0.25
        assert record["matched_direction"] != record["source_direction"]
        assert serial["shifts"][record["world_id"]][record["level"]] == record["intercept_shift"]


def test_calibration_stops_when_the_target_is_not_bracketed() -> None:
    contract = _quick(study.load_contract(CONFIG), iterations=1)
    contract["entry_calibration"]["bracket"] = [5.0, 5.001]
    with study.TaskMap(1) as task_map, pytest.raises(study.CalibrationError, match="not bracketed"):
        study.calibrate_entry(contract, "smoke", FAKE_PRIOR, task_map)


def test_study_tasks_nest_the_predecessor_and_record_the_matched_bias() -> None:
    contract = study.load_contract(CONFIG)
    shifts = {world: {"llm_entry": -0.5, "hand_entry": 0.5} for world in study.FIT_WORLDS}
    hand = study.study_task((contract, "smoke", 0, "linear_fit", "hand", FAKE_PRIOR, shifts))
    assert [row["arm_id"] for row in hand] == list(study.ALL_ARMS)
    assert all(row["prior_bias"] is None and row["exploratory_entry_rate"] == 0.0 for row in hand)
    matched = study.study_task(
        (contract, "smoke", 0, "axis_interaction_fit", "hand_at_llm_entry_20", FAKE_PRIOR, shifts)
    )
    assert {row["prior_bias"] for row in matched} == {study.hand_prior()["bias"] - 0.5}
    assert matched[0]["epsilon"] > 0.0
    assert matched == study.study_task(
        (contract, "smoke", 0, "axis_interaction_fit", "hand_at_llm_entry_20", FAKE_PRIOR, shifts)
    )


def _rows(precision: dict, brier: dict, entry: dict | None = None) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for world in study.FIT_WORLDS:
            for condition in study.CONDITION_IDS:
                for arm in study.ALL_ARMS:
                    key = (world, condition, arm)
                    rows.append(
                        {
                            "replicate": replicate,
                            "world_id": world,
                            "condition_id": condition,
                            "arm_id": arm,
                            "cumulative_entry_rate": (entry or {}).get((world, condition), 0.5),
                            "productive_share_of_entrants": precision.get(key, 0.30) + wobble,
                            # Only the hybrid varies, so its Brier gaps have a spread.
                            "probe_brier": brier.get(key, 0.10)
                            + (3 * wobble if condition == "hybrid_20" else 0.0),
                        }
                    )
    return rows


def test_decisions_separate_ranking_from_selectivity() -> None:
    contract = study.load_contract(CONFIG)
    precision, brier = {}, {}
    for arm in study.FIELD_ARMS:
        # The hybrid beats the native hand rule everywhere, as C22 did...
        for world in study.FIT_WORLDS:
            precision[(world, "hybrid_20", arm)] = 0.40
            precision[(world, "hand_at_llm_entry_20", arm)] = 0.40
        # ...but beats the entry-matched hand rule only in the linear world.
        precision[("linear_fit", "hand_at_llm_entry_20", arm)] = 0.35
    brier[("axis_interaction_fit", "hybrid_20", "mlp_batch")] = 0.1035
    precision[("linear_fit", "llm_selective", "fixed_rule")] = 0.45
    precision[("smooth_interaction_fit", "hand_at_llm_entry", "fixed_rule")] = 0.45
    decisions = study.decide(_rows(precision, brier), contract)
    assert decisions["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    fixed = decisions["ranking_at_llm_selectivity_fixed_rule"]
    assert fixed["cells"]["linear_fit/fixed_rule"]["outcome"] == "FIRST_BETTER"
    assert fixed["cells"]["smooth_interaction_fit/fixed_rule"]["outcome"] == "SECOND_BETTER"
    assert fixed["counts"] == {"FIRST_BETTER": 1, "SECOND_BETTER": 1, "NO_CLEAR_DIFFERENCE": 1}
    matched = decisions["c22_against_entry_matched_hand"]
    assert matched["status"] == "C22_PRECISION_SURVIVES_IN_MINORITY"
    assert matched["precision_better_cells"] == [f"linear_fit/{arm}" for arm in sorted(study.FIELD_ARMS)]
    replication = decisions["c22_replication_against_hand"]
    assert len(replication["precision_better_cells"]) == 9
    # A mean gap inside the margin passes the mean-only rule but not the interval rule.
    cell = replication["cells"]["axis_interaction_fit/mlp_batch"]
    assert cell["learning"]["mean_only_rule_pass"] is True
    assert cell["learning"]["pass"] is False
    assert cell["disposition"] == "SELECTION_ONLY"
    assert cell["mean_only_disposition"] == "LLM_DOMINATES"
    assert replication["dominance"] == "MAJORITY"


def test_a_missed_out_of_sample_entry_match_stops_the_decisions() -> None:
    contract = study.load_contract(CONFIG)
    entry = {("smooth_interaction_fit", "hand_at_llm_entry"): 0.45}
    decisions = study.decide(_rows({}, {}, entry), contract)
    assert decisions["status"] == "ENTRY_MATCH_FAILED_STOP"
    failed = [r for r in decisions["entry_match_check"]["records"] if not r["within_bounds"]]
    assert [(r["matched"], r["world_id"]) for r in failed] == [("hand_at_llm_entry", "smooth_interaction_fit")]
    # The exploration pair is descriptive only and cannot stop the study.
    entry = {("linear_fit", "hand_at_llm_entry_20"): 0.40}
    assert study.decide(_rows({}, {}, entry), contract)["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"


def test_development_summary_and_outputs_write_from_full_shaped_rows(tmp_path: Path) -> None:
    contract = study.load_contract(CONFIG)
    shifts = {world: {"llm_entry": -0.5, "hand_entry": 0.5} for world in study.FIT_WORLDS}
    template = study.study_task((contract, "smoke", 0, "linear_fit", "hand", FAKE_PRIOR, shifts))
    rows = [
        {**row, "replicate": replicate, "world_id": world, "condition_id": condition}
        for replicate in range(16)
        for world in study.FIT_WORLDS
        for condition in study.CONDITION_IDS
        for row in template
    ]
    calibration = {
        "status": "ENTRY_CALIBRATION_COMPLETE",
        "records": [
            {
                "world_id": world,
                "level": level,
                "matched_direction": "hand" if level == "llm_entry" else "llm",
                "intercept_shift": 0.0,
                "target_entry": 0.5,
                "achieved_entry": 0.5,
            }
            for world in study.FIT_WORLDS
            for level in study.MATCH_SOURCES
        ],
        "shifts": shifts,
    }
    summary = study.summarize(rows, calibration, contract, "development")
    assert summary["decisions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert "shifts" not in summary["calibration"]
    assert len(summary["cells"]) == 3 * len(study.CONDITION_IDS) * len(study.ALL_ARMS)
    receipt = study.write_outputs(
        output_dir=tmp_path / "out",
        rows=rows,
        calibration=calibration,
        summary=summary,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert set(receipt["outputs"]) == {"calibration.json", "runs.jsonl", "scorecard.json", "scorecard.md"}
    assert "C22 against the entry-matched hand rule" in (tmp_path / "out" / "scorecard.md").read_text()


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = study.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        study.delayed.verify_precommit("short", contract)
