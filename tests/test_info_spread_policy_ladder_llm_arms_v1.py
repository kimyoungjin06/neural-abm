from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_ladder_core_v1 as core  # noqa: E402
import prototype_info_spread_policy_ladder_v1 as spread  # noqa: E402
import run_info_spread_policy_ladder_llm_arms_v1 as llm  # noqa: E402
import run_info_spread_policy_ladder_study_v2 as d4  # noqa: E402


def _contract() -> dict:
    return llm.load_contract()


def _stub_log_odds(contract: dict) -> dict[str, np.ndarray]:
    """Log-odds that are a known function of what each prompt shows."""

    out = {}
    for regime in ("A", "B"):
        features, _ = llm.p0_cohort(contract, regime)
        out[f"P0/{regime}"] = 3.0 * features[:, 0] - 2.0 * features[:, 2] + 0.5
        keys = llm.x0_keys(contract, regime)
        values = []
        for archetype, item_class, (kind, n, value) in keys:
            rate = 0.5 if kind == 0 else (value / n if kind == 1 else value / 100.0)
            values.append(np.log((rate + 0.05) / (1.05 - rate)) + 0.1 * archetype + 0.2 * item_class)
        out[f"X0/{regime}"] = np.asarray(values)
    return out


def _write_stub_elicitation(directory: Path, contract: dict) -> Path:
    log_odds = _stub_log_odds(contract)
    records = [
        {"set": name, "index": i, "prompt_sha256": llm._text_sha256(text), "log_odds": float(log_odds[name][i]),
         "answer_token_mass": 1.0, "top_token_is_answer": True}
        for name, texts in llm.prompt_sets(contract).items() for i, text in enumerate(texts)
    ]
    result = {"records": records, "receipt": {"validity": {}, "determinism": {}, "timing_seconds": {}}}
    llm.write_elicitation(directory, result, contract, llm.DEFAULT_CONTRACT, None)
    return directory


def test_the_contract_is_a_precommit_bound_to_its_sources() -> None:
    contract = _contract()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert len(llm.verify_source_bindings(contract)) == len(contract["source_bindings"])
    assert set(contract["precommit"]["precommit_paths"]) == {
        "docs/research/info-spread-policy-ladder-llm-arms-v1.md",
        "experiments/configs/info_spread_policy_ladder_llm_arms_v1.yaml",
        "scripts/run_info_spread_policy_ladder_llm_arms_v1.py",
        "tests/test_info_spread_policy_ladder_llm_arms_v1.py",
    }
    assert contract["llm"]["paid_api_calls"] == 0
    assert contract["rule"]["minimum_effect"] == power_min_effect()


def power_min_effect() -> float:
    import power_info_spread_policy_ladder_v2 as power2

    return power2.MIN_GROUP_EFFECT


def test_the_aggregate_display_is_exact_then_rounded_half_up() -> None:
    spec = _contract()["x0"]["aggregates"]
    assert llm.aggregate_state(0, 0, spec) == (0, 0, 0)
    assert llm.aggregate_state(7, 3, spec) == (1, 7, 3)
    assert llm.aggregate_state(10, 10, spec) == (1, 10, 10)
    assert llm.aggregate_state(14, 7, spec) == (2, 10, 50)
    assert llm.aggregate_state(15, 3, spec) == (2, 20, 20)
    assert llm.aggregate_state(140, 49, spec) == (2, 140, 35)  # 35.0% exactly
    assert llm.aggregate_state(173, 60, spec) == (2, 170, 35)  # 34.68%
    states = llm.aggregate_states(spec)
    assert len(states) == 1 + 65 + 20 * 21
    assert len(set(states)) == len(states)


def test_the_prompt_sets_have_the_registered_sizes_and_read_as_intended() -> None:
    contract = _contract()
    sets = llm.prompt_sets(contract)
    states = len(llm.aggregate_states(contract["x0"]["aggregates"]))
    assert {k: len(v) for k, v in sets.items()} == {
        "P0/A": 1024, "P0/B": 1024, "X0/A": 2 * 4 * states, "X0/B": 6 * 4 * states,
    }
    for name in ("X0/A", "X0/B"):
        assert len(set(sets[name])) == len(sets[name])
    first_a = sets["P0/A"][0]
    assert "Track record" not in first_a and "Track record" in sets["P0/B"][0]
    assert first_a.endswith("Is this item true? Answer with Yes or No.")
    assert "- Emotional intensity of the wording: " in first_a
    # Regime B archetype 5 is the cautious threshold bin with the highest cost; class 3 is credible and novel.
    keys = llm.x0_keys(contract, "B")
    index = keys.index((5, 3, (2, 140, 35)))
    text = sets["X0/B"][index]
    assert "They are cautious" in text and "twice as much" in text
    assert "looks more credible than most, and its claim is more novel" in text
    assert "about 35% true out of about 140" in text


def test_the_class_aggregates_count_what_is_visible_at_each_step() -> None:
    world = spread.build_world(spread.WorldConfig(agents=60, steps=20, items_per_step=4, world_seed=11))
    stream = spread.build_stream(world, 3)
    classify = spread.item_classes(2, 2)
    n, k = llm.class_aggregates(world, stream, classify, 4)
    classes = classify(stream.features)
    for step in (0, 4, 9, 19):
        for c in range(4):
            seen = [i for i in range(len(classes))
                    if classes[i] == c and stream.revealed[i] and stream.born[i] + world.config.delay <= step]
            assert n[step, c] == len(seen)
            assert k[step, c] == sum(bool(stream.veracity[i]) for i in seen)
    assert n[3].sum() == 0  # nothing is visible before the first delay


def test_x0_looks_up_its_archetype_class_and_step_state() -> None:
    contract = _contract()
    world = spread.build_world(spread.WorldConfig(agents=60, steps=20, items_per_step=4, regime="B", world_seed=5))
    stream = spread.build_stream(world, 0)
    states = llm.aggregate_states(contract["x0"]["aggregates"])
    table = np.zeros((6, 4, len(states)))
    for a in range(6):
        for c in range(4):
            table[a, c] = 0.01 * np.arange(len(states)) + a - 2.0 * c
    x0 = {"table": table, "states": {s: i for i, s in enumerate(states)}}
    log: list[int] = []
    policy = llm.x0_policy(contract, world, stream, x0, log)
    ids = np.asarray([0, 7, 33])
    x = np.hstack([stream.features[[1, 5, 9]], np.zeros((3, 3))])
    step = 12
    probability = policy.probability(ids, x, step)
    archetype = spread.archetypes(world, 2, cost_bins=True)[ids]
    item_class = spread.item_classes(2, 2)(x)
    n, k = llm.class_aggregates(world, stream, spread.item_classes(2, 2), 4)
    for j in range(3):
        state = llm.aggregate_state(int(n[step, item_class[j]]), int(k[step, item_class[j]]), contract["x0"]["aggregates"])
        expected = core.sigmoid(np.asarray([table[archetype[j], item_class[j], x0["states"][state]]]))[0]
        assert probability[j] == expected
    assert log == [len(set(zip(archetype, item_class, strict=True)))]


def test_the_stub_elicitation_builds_calibrated_arms(tmp_path: Path) -> None:
    contract = _contract()
    directory = _write_stub_elicitation(tmp_path / "elicitation", contract)
    arms = llm.build_llm_arms(contract, llm.load_elicitation(directory, contract))
    for regime in ("A", "B"):
        features, _ = llm.p0_cohort(contract, regime)
        prior = arms["P0"][regime]
        assert prior.probability(features).mean() == pytest.approx(0.4, abs=1e-9)
        assert arms["report"][regime]["p0_distillation_r_squared"] > 0.999
        weights = np.asarray(arms["report"][regime]["p0_weights"])
        assert weights[0] == pytest.approx(3.0, abs=0.05) and weights[2] == pytest.approx(-2.0, abs=0.05)
        x0 = arms["X0"][regime]
        before = x0["table"][:, :, x0["states"][(0, 0, 0)]]
        assert core.sigmoid(before).mean() == pytest.approx(0.4, abs=1e-9)


def test_an_elicitation_of_other_prompts_is_rejected(tmp_path: Path) -> None:
    contract = _contract()
    directory = _write_stub_elicitation(tmp_path / "elicitation", contract)
    path = directory / "elicitation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["prompt_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    with pytest.raises(RuntimeError):
        llm.load_elicitation(directory, contract)


def test_p0_with_the_hand_prior_is_f1_and_l1_with_it_is_d4s_l0() -> None:
    contract = _contract()
    d4_contract = llm.load_d4_contract(contract)
    d4.apply_learner_settings(d4_contract)
    for regime, family in (("A", "linear_exact"), ("A", "mlp"), ("B", "partial_linear_exact"), ("B", "partial_mlp")):
        world = llm.build_world(d4_contract, "smoke", regime, "smooth", 0)
        stream = spread.build_stream(world, 1)
        hand = spread.hand_prior(world.config)
        f1 = spread.simulate(world, stream, spread.prior_policy("F1", hand), 1)
        p0 = spread.simulate(world, stream, spread.prior_policy("P0", hand), 1)
        assert p0.digest == f1.digest
        ours = llm._learner(d4_contract, world, family, 1, 0, 1, "cpu", hand)
        theirs = d4._learner(d4_contract, world, family, 1, 0, 1, "cpu")
        assert spread.simulate(world, stream, ours, 1).digest == spread.simulate(world, stream, theirs, 1).digest


def test_a_smoke_study_runs_pairs_with_d4_and_passes_parity(tmp_path: Path) -> None:
    contract = _contract()
    d4_contract = llm.load_d4_contract(contract)
    d4.apply_learner_settings(d4_contract)
    directory = _write_stub_elicitation(tmp_path / "elicitation", contract)
    arms = llm.build_llm_arms(contract, llm.load_elicitation(directory, contract))
    d4_cells, cells = {}, []
    for regime in d4_contract["design"]["regimes"]:
        for geometry in ("linear", "null"):
            first = llm.run_fixed_cell(contract, d4_contract, arms, "smoke", regime, geometry, 0)
            assert json.dumps(first, sort_keys=True) == json.dumps(
                llm.run_fixed_cell(contract, d4_contract, arms, "smoke", regime, geometry, 0), sort_keys=True)
            assert all(r["arms"]["X0"]["queries_without_cache"] > 0 for r in first["rows"])
            cells.append(first)
            for family in d4_contract["design"]["families"][regime]:
                d4_cells[(regime, family, geometry, 0)] = d4.run_cell(
                    d4_contract, "smoke", regime, family, geometry, 0, "cpu")
                cells.append(llm.run_learner_cell(contract, d4_contract, arms, "smoke", regime, family, geometry, 0, "cpu"))
    assert "seconds" not in json.dumps(cells)  # timing never enters cells.jsonl
    summary = llm.summarize(contract, d4_contract, d4_cells, cells)
    assert summary["parity_failures"] == []
    entry = summary["table"]["A/linear_exact/linear"]
    assert set(entry["mean_unverified_share_precision"]) == {*d4.ARMS, *llm.ARMS}
    for name in d4.MEASURES:
        assert set(entry[name]) == {*llm.COMPARISONS, *llm.DESCRIPTIVE}
    primary = {key: value[d4.PRIMARY] for key, value in summary["table"].items()}
    assert summary["dispositions"] == llm.dispositions(contract, d4_contract, primary)


def test_the_d4_record_must_come_from_this_precommit_and_contract(tmp_path: Path) -> None:
    contract = _contract()
    cells = tmp_path / "cells.jsonl"
    cells.write_text('{"regime": "A", "family": "mlp", "geometry": "null", "cell": 0, "rows": []}\n')
    record = {
        "precommit_commit": "a" * 40, "contract_sha256": llm._sha256(ROOT / contract["d4"]["contract"]),
        "reproduction": {"byte_identical": {"cells.jsonl": True, "summary.json": True},
                         "outputs": {"development": {"cells.jsonl": llm._sha256(cells)}}},
    }
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    assert list(llm.load_d4_cells(contract, cells, path, "a" * 40)) == [("A", "mlp", "null", 0)]
    with pytest.raises(RuntimeError):
        llm.load_d4_cells(contract, cells, path, "b" * 40)
    record["contract_sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError):
        llm.load_d4_cells(contract, cells, path, "a" * 40)


def _table(positive: dict[str, int], negative: dict[str, int], null_positive: bool = False) -> dict:
    d4_contract = llm.load_d4_contract(_contract())
    design = d4_contract["design"]
    table = {}
    for regime in design["regimes"]:
        study = [f"{regime}/{f}/{g}" for f in design["families"][regime] for g in design["study_geometries"]]
        for index, key in enumerate(study):
            sign = 1 if index < positive[regime] else (-1 if index < positive[regime] + negative[regime] else 0)
            rule = {"material": sign != 0, "mean_cell_delta": 0.05 * sign}
            table[key] = {name: rule for name in llm.COMPARISONS}
        for family in design["families"][regime]:
            null = {"material": null_positive, "mean_cell_delta": 0.05 if null_positive else 0.0}
            table[f"{regime}/{family}/null"] = {name: null for name in llm.COMPARISONS}
    return table


@pytest.mark.parametrize(
    ("positive", "negative", "expected"),
    [({"A": 4, "B": 6}, {"A": 0, "B": 0}, {"A": "MOSTLY_POSITIVE", "B": "MOSTLY_POSITIVE"}),
     ({"A": 0, "B": 3}, {"A": 4, "B": 0}, {"A": "MOSTLY_NEGATIVE", "B": "MIXED_OR_NONE"}),
     ({"A": 2, "B": 0}, {"A": 2, "B": 0}, {"A": "MIXED_OR_NONE", "B": "MIXED_OR_NONE"})],
)
def test_the_registered_dispositions(positive: dict, negative: dict, expected: dict) -> None:
    contract = _contract()
    result = llm.dispositions(contract, llm.load_d4_contract(contract), _table(positive, negative))
    assert all(result["by_comparison"][name] == expected for name in llm.COMPARISONS)
    assert result["design_check"] == "NULL_WORLD_PASSES"


def test_a_closed_loop_gain_on_the_llm_prior_in_the_null_world_is_a_defect() -> None:
    contract = _contract()
    table = _table({"A": 0, "B": 0}, {"A": 0, "B": 0}, null_positive=True)
    result = llm.dispositions(contract, llm.load_d4_contract(contract), table)
    assert result["design_check"] == "DESIGN_DEFECT_IN_NULL_WORLD"
