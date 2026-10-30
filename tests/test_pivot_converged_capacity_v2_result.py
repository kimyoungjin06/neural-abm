from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
RESULT = RESULTS / "pivot_converged_capacity_v2_20261030.summary.json"
SUMMARY = RESULTS / "pivot_converged_capacity_v2_20261030.summary.md"
REGIMES = ("registered", "converged", "tuned")


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _text() -> str:
    return " ".join(SUMMARY.read_text(encoding="utf-8").split())


def _r(value: float, places: str = "0.001", sign: bool = True) -> str:
    text = str(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))
    if text.startswith("-"):
        return text.replace("-", "−")
    return f"+{text}" if sign else text


def test_reproduced_under_its_freeze_with_every_check_passing() -> None:
    result = _result()
    assert result["precommit_commit"] == "1cfd1abb4ea9f7261209d61bbe48a59d5d6d9995"
    assert result["supersedes"]["freeze_commit"] == "4062c8c"
    assert result["supersedes"]["status"] == "STOPPED_INVALID_IMPLEMENTATION_NO_OUTPUTS_WRITTEN"
    identical = result["reproduction"]["byte_identical"]
    assert all(identical.values()) and len(identical) == 5
    checks = result["checks"]
    assert checks["parity"] == {
        "abm_rows": 1008,
        "policy_layer_rows": 480,
        "reference": "THE_RECORDED_ROWS",
        "row_for_row": True,
    }
    assert checks["registered"] == {"dispositions_equal_both_records": True, "every_claim_holds": True}
    assert checks["convergence"]["within_ceiling"] is True
    assert result["tuned_neighbor_rows_equal_registered"] is True
    assert result["decision_status"] == "FROZEN_SYNTHETIC_DECISION_COMPLETE"
    assert (result["execution"]["abm_rows"], result["execution"]["policy_layer_rows"]) == (2448, 1440)


def test_every_capacity_claim_holds_under_both_retrained_regimes() -> None:
    result = _result()
    assert result["status"] == "O36_HOLD_UNDER_BOTH"
    for regime in REGIMES:
        claims = result["decisions"]["by_regime"][regime]["claims"]
        assert claims["outcome"] == "CLAIMS_HOLD" and claims["changed"] == []
        assert len(claims["claims"]) == 9
    assert {key: row["held"] for key, row in result["expectations"].items()} == {
        "E1": True,
        "E2": True,
        "E3": True,
        "E4": True,
    }


def test_the_linear_world_edge_vanishes_and_the_mlp_behavior_dids_pass() -> None:
    result = _result()
    quantities = result["decisions"]["registered_quantities"]
    edges = quantities["linear_world_mlp_edge"]
    assert edges["registered"]["low"] > 0.04
    assert edges["tuned"]["low"] < 0.0 < edges["tuned"]["high"]
    assert quantities["mlp_budget_effect"]["low"] > 0.0
    for regime in ("converged", "tuned"):
        immediate = quantities["behavior_passing_lists"][regime]["immediate"]
        assert {"mlp_batch/smooth_interaction_fit", "mlp_batch/axis_interaction_fit"} <= set(immediate["entry"])
        assert "mlp_batch/smooth_interaction_fit" in immediate["exit"]
    registered = quantities["behavior_passing_lists"]["registered"]["immediate"]
    assert not any(item.startswith("mlp_batch/") for item in registered["entry"] + registered["exit"])
    convergence = result["checks"]["convergence"]["by_study_and_regime"]
    assert convergence["abm/tuned"]["unconverged_inner_fits"] == 1
    assert all(row["unconverged_acting_fits"] == 0 for row in convergence.values())


def test_the_summary_quotes_the_record() -> None:
    result = _result()
    text = _text()
    field = result["descriptive"]["field_means"]
    edges = result["decisions"]["registered_quantities"]["linear_world_mlp_edge"]
    for regime in REGIMES:
        linear = field[regime]["immediate"]["linear_fit"]
        edge = edges[regime]
        quoted = (
            f"| {regime} | {_r(linear['linear_batch']['probe_brier'], '0.0001', False)} | "
            f"{_r(linear['mlp_batch']['probe_brier'], '0.0001', False)} | "
            f"{_r(linear['tree_batch']['probe_brier'], '0.0001', False)} | "
            f"{_r(edge['mean'])} [{_r(edge['low'])}, {_r(edge['high'])}] |"
        )
        assert quoted in text, quoted
        for world, label in (("smooth_interaction_fit", "smooth"), ("axis_interaction_fit", "axis")):
            row = field[regime]["immediate"][world]
            quoted = f"| {label} | {regime} | " + " | ".join(
                _r(row[arm]["probe_brier"], "0.0001", False) for arm in ("linear_batch", "mlp_batch", "tree_batch")
            ) + " |"
            assert quoted in text, quoted
    budget = result["decisions"]["registered_quantities"]["mlp_budget_effect"]
    assert f"rises by {_r(budget['mean'], sign=False)} [{_r(budget['low'], sign=False)}, {_r(budget['high'], sign=False)}]" in text
    brier = result["descriptive"]["policy_layer_brier"]
    gaps = {
        regime: [
            _r(brier[regime][world]["matched_linear"] - brier[regime][world]["residual_mlp"])
            for world in ("linear_support_aligned", "linear_support_reversed")
        ]
        for regime in REGIMES
    }
    assert f"is {gaps['registered'][0]} and {gaps['registered'][1]} registered" in text
    assert f"{gaps['converged'][0]} and {gaps['converged'][1]} converged" in text
    assert f"{gaps['tuned'][0]} and {gaps['tuned'][1]} tuned" in text
    abm = result["reports"]["abm"]
    learning = [abm[r]["immediate"]["field_nonlinear_learning"]["mlp_batch/smooth_interaction_fit"]["mean"] for r in REGIMES]
    assert f"{_r(learning[0], sign=False)} registered, {_r(learning[1], sign=False)} converged, {_r(learning[2], sign=False)} tuned" in text
    smooth = [abm[r]["immediate"]["field_geometry_crossover"]["mlp_batch_over_tree_batch/smooth_interaction_fit"] for r in REGIMES]
    axis = [abm[r]["immediate"]["field_geometry_crossover"]["tree_batch_over_mlp_batch/axis_interaction_fit"] for r in REGIMES]
    assert "by {}, {}, and {}".format(*(_r(row["mean"], sign=False) for row in smooth)) in text
    assert "by {}, {}, and {}".format(*(_r(row["mean"], sign=False) for row in axis)) in text
    assert all(row["positive_replicates"] == 16 for row in smooth + axis)
    tuned = abm["tuned"]["immediate"]
    entry_smooth, entry_axis = (tuned["entry_behavior"][f"mlp_batch/{w}"] for w in ("smooth_interaction_fit", "axis_interaction_fit"))
    assert f"tuned: {_r(entry_smooth['mean'], sign=False)} smooth, {_r(entry_axis['mean'], sign=False)} axis, 16 of 16" in text
    exit_smooth, exit_axis = (tuned["exit_channel"][f"mlp_batch/{w}"] for w in ("smooth_interaction_fit", "axis_interaction_fit"))
    assert f"({_r(exit_smooth['mean'], sign=False)}, {exit_smooth['positive_replicates']} of 16)" in text
    assert f"it gains {_r(exit_axis['mean'], sign=False)}, in only {exit_axis['positive_replicates']} of 16" in text
    policy = result["reports"]["policy_layer"]
    need = {r: policy[r]["nonlinear_representation_need"] for r in REGIMES}
    assert "{} and {} registered, against {} and {} tuned".format(
        *(_r(need[r][w]["mean"], sign=False) for r in ("registered",) for w in ("smooth_interaction_aligned", "smooth_interaction_reversed")),
        *(_r(need[r][w]["mean"], sign=False) for r in ("tuned",) for w in ("smooth_interaction_aligned", "smooth_interaction_reversed")),
    ) in text
    walls = result["execution"]["wall_seconds"]
    assert f"{round(walls['development']):,} s for the development run" in text
    assert f"{round(walls['reproduction']):,} s for the reproduction run" in text
    training = result["checks"]["convergence"]["by_study_and_regime"]
    assert f"All {training['abm/converged']['lbfgs_acting_fits']:,} acting fits" in text
    assert f"all {training['abm/tuned']['lbfgs_acting_fits']:,} acting L-BFGS fits" in text
    assert f"Of {training['abm/tuned']['lbfgs_inner_fits']:,} inner L-BFGS fits" in text
