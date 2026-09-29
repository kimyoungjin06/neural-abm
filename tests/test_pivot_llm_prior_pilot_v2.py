from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "experiments" / "configs" / "pivot_llm_prior_pilot_v2.yaml"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Imported by module name so spawned workers can resolve the task function.
import prototype_pivot_llm_prior_pilot_v2 as pilot  # noqa: E402


V1_DIR = Path(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["v1_elicitation"]["directory"])
needs_v1 = pytest.mark.skipif(not V1_DIR.exists(), reason="v1 elicitation artifacts are machine-local")


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_contract_plan_makes_no_llm_calls_and_binds_sources() -> None:
    contract = pilot.load_contract(CONFIG)
    plan = pilot._plan_payload(contract, CONFIG)
    assert plan["status"] == "PRECOMMIT_PLAN_VALID"
    assert len(plan["source_bindings"]) == 11
    assert contract["boundaries"]["new_llm_calls"] == 0
    assert contract["calibration"]["information_used"] == "AGGREGATE_ONLY_NO_INDIVIDUAL_LABELS"
    assert plan["variants"][0] == "llm_base_rate"


def test_reused_v1_seed_and_llm_calls_are_rejected() -> None:
    contract = _raw()
    contract["profiles"]["development"]["base_seed"] = 20261018
    with pytest.raises(ValueError, match="fresh seed"):
        pilot.validate_contract(contract)
    contract = _raw()
    contract["boundaries"]["new_llm_calls"] = 1024
    with pytest.raises(ValueError, match="no LLM calls"):
        pilot.validate_contract(contract)


def test_intercept_calibration_hits_the_target_and_keeps_weights() -> None:
    rng = np.random.default_rng(3)
    features = rng.random((500, 9))
    weights = [1.0, -2.0, 0.5, 0.0, 3.0, -1.0, 0.0, 0.25, 0.0]
    bias = pilot.calibrate_intercept(
        weights, -7.0, features, 0.30, interval=[-40.0, 40.0], tolerance=1e-10
    )
    assert pilot.mean_probability(weights, bias, features) == pytest.approx(0.30, abs=1e-10)
    with pytest.raises(RuntimeError, match="outside the bisection interval"):
        pilot.calibrate_intercept(
            weights, -7.0, features, 0.30, interval=[-1.0, 1.0], tolerance=1e-10
        )


@needs_v1
def test_priors_build_from_the_hash_bound_v1_elicitation() -> None:
    contract = pilot.load_contract(CONFIG)
    records, llm_prior = pilot.load_v1_elicitation(contract)
    built = pilot.build_priors(contract, records, llm_prior)
    pilot.check_priors(built, llm_prior, contract)
    diagnostics = built["diagnostics"]
    assert diagnostics["llm_base_rate"]["achieved_mean"] == pytest.approx(0.30, abs=1e-9)
    assert diagnostics["llm_hand_level"]["target_mean"] == pytest.approx(
        diagnostics["hand_mean_over_design_cohort"]
    )
    assert diagnostics["llm_base_rate"]["intercept_shift"] > 0
    assert set(built["oracle_linear"]) == set(pilot.FIT_WORLDS)
    tampered = copy.deepcopy(built)
    tampered["priors"]["llm_base_rate"]["weights"][0] += 1.0
    with pytest.raises(RuntimeError, match="changed the v1 weights"):
        pilot.check_priors(tampered, llm_prior, contract)


@needs_v1
def test_a_changed_v1_artifact_is_rejected(tmp_path: Path) -> None:
    copy_dir = tmp_path / "v1"
    shutil.copytree(V1_DIR, copy_dir)
    (copy_dir / "llm_prior.json").write_text("{}", encoding="utf-8")
    contract = copy.deepcopy(pilot.load_contract(CONFIG))
    contract["v1_elicitation"]["directory"] = str(copy_dir)
    with pytest.raises(RuntimeError, match="changed"):
        pilot.load_v1_elicitation(contract)


def _fake_built() -> dict:
    prior = {"weights": [0.5, 2.0, 1.5, -2.0, -1.0, 0.5, 0.0, 0.0, 0.0], "bias": -0.5}
    return {
        "priors": {prior_id: prior for prior_id in pilot.VARIANTS},
        "oracle_linear": {world: prior for world in pilot.FIT_WORLDS},
        "flat": {"weights": [0.0] * 9, "bias": 0.0},
        "diagnostics": {},
    }


def test_parallel_workers_reproduce_the_serial_rows() -> None:
    contract = pilot.load_contract(CONFIG)
    reduced = copy.deepcopy(contract)
    reduced["profiles"]["smoke"]["replicates"] = 1
    reduced["profiles"]["smoke"]["steps"] = 6
    serial = pilot.run_study(reduced, "smoke", _fake_built(), workers=1)
    parallel = pilot.run_study(reduced, "smoke", _fake_built(), workers=3)
    assert len(serial) == 3 * len(pilot.PRIOR_IDS) * len(pilot.ABM_ARMS)
    assert serial == parallel


def _rows(overrides: dict) -> list[dict]:
    rows = []
    for replicate in range(16):
        wobble = 0.001 * ((replicate % 4) - 1.5)
        for world in pilot.FIT_WORLDS:
            for prior in pilot.PRIOR_IDS:
                for arm, evidence in pilot.ABM_ARMS:
                    base = overrides.get((world, prior, arm), {})
                    rows.append(
                        {
                            "replicate": replicate,
                            "world_id": world,
                            "prior_id": prior,
                            "arm_id": arm,
                            "evidence_regime": evidence,
                            "probe_brier": base.get("probe_brier", 0.20) + wobble,
                            "productive_share_of_entrants": base.get("share", 0.30) + wobble,
                            "cumulative_entry_rate": base.get("entry", 0.80),
                        }
                    )
    return rows


def test_decisions_compare_every_variant_with_hand_and_flag_self_sealing() -> None:
    contract = pilot.load_contract(CONFIG)
    overrides = {
        ("linear_fit", "llm_base_rate", "fixed_rule"): {"probe_brier": 0.15, "share": 0.36},
        ("linear_fit", "llm_base_rate", "mlp_batch"): {"probe_brier": 0.23},
        ("linear_fit", "llm_uncalibrated", "fixed_rule"): {"probe_brier": 0.31, "entry": 0.0},
        ("smooth_interaction_fit", "llm_uncalibrated", "fixed_rule"): {"entry": 0.0},
        ("axis_interaction_fit", "llm_uncalibrated", "fixed_rule"): {"entry": 0.0},
    }
    decision = pilot.decide(_rows(overrides), contract)
    primary = decision["by_variant"]["llm_base_rate"]["linear_fit"]
    assert primary["prior_quality"] == "VARIANT_BETTER"
    assert primary["entry_precision"] == "VARIANT_BETTER"
    assert primary["prior_persists_for_arms"] == {"mlp_batch": "HAND_BETTER"}
    assert decision["by_variant"]["llm_uncalibrated"]["linear_fit"]["prior_quality"] == (
        "HAND_BETTER"
    )
    assert decision["by_variant"]["llm_verbalized"]["axis_interaction_fit"][
        "pooled_washout"
    ] == "PRIOR_WASHES_OUT_UNDER_POOLING"
    assert decision["self_sealing_priors"] == [
        "llm_uncalibrated/axis_interaction_fit",
        "llm_uncalibrated/linear_fit",
        "llm_uncalibrated/smooth_interaction_fit",
    ]


def test_development_summary_and_outputs_write_from_full_shaped_rows(tmp_path: Path) -> None:
    contract = pilot.load_contract(CONFIG)
    template = pilot.pilot1.run_task(contract, "smoke", 0, "linear_fit", "hand", None)
    rows = [
        {**row, "replicate": replicate, "world_id": world, "prior_id": prior}
        for replicate in range(16)
        for world in pilot.FIT_WORLDS
        for prior in pilot.PRIOR_IDS
        for row in template
    ]
    built = _fake_built()
    summary = pilot.summarize(rows, contract, "development", built)
    assert summary["dispositions"]["status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert len(summary["cells"]) == 3 * len(pilot.PRIOR_IDS) * len(pilot.ABM_ARMS)
    receipt = pilot.write_outputs(
        output_dir=tmp_path / "out",
        rows=rows,
        summary=summary,
        built=built,
        precommit={"commit_sha": "0" * 40},
        contract_path=CONFIG,
    )
    assert receipt["row_count"] == 16 * 3 * len(pilot.PRIOR_IDS) * len(pilot.ABM_ARMS)
    assert json.loads((tmp_path / "out" / "calibrated_priors.json").read_text())["flat"]["bias"] == 0.0


def test_run_preflight_rejects_a_nonfull_sha_before_git_access() -> None:
    contract = pilot.load_contract(CONFIG)
    with pytest.raises(RuntimeError, match="full 40-character lowercase Git SHA"):
        pilot.delayed.verify_precommit("short", contract)
