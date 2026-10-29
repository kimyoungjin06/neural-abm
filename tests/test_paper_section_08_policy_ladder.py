"""Paper Section 8 states the recorded results, and only with their boundaries."""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "paper" / "sections" / "08-policy-ladder.md"
FIGURE = ROOT / "paper" / "figures" / "info_spread_policy_ladder_comparisons.png"
FOLLOW_UPS = ROOT / "paper" / "figures" / "info_spread_policy_ladder_follow_ups.png"
RESULTS = ROOT / "experiments" / "evidence" / "results"
MEASURE = "unverified_share_precision"


def _text() -> str:
    return " ".join(SECTION.read_text(encoding="utf-8").split())


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _study_groups(table: dict) -> dict:
    return {key: value for key, value in table.items() if not key.endswith("/null")}


def _r(value: float) -> str:
    """Three decimals, rounded half up from the recorded value, as the documents do."""

    return str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _span(values: list[float], signed: bool = False) -> str:
    low, high = _r(min(values)), _r(max(values))
    return f"+{low} to +{high}" if signed else f"{low} to {high}"


def test_study_v2_numbers_come_from_the_record() -> None:
    table = _study_groups(_load("info_spread_policy_ladder_study_v2_d4_20261019.summary.json")["table"])
    text = _text()
    assert _span([v[MEASURE]["hand_rule"]["mean_cell_delta"] for v in table.values()], signed=True) in text
    assert _span([-v[MEASURE]["best_table"]["mean_cell_delta"] for v in table.values()]) in text
    exception = table["A/mlp/smooth"][MEASURE]["closed_loop"]
    low, high = exception["interval_95"]
    assert f"+{_r(exception['mean_cell_delta'])} [+{_r(low)}, +{_r(high)}]" in text
    assert "TABLE_SUFFICES_EVEN_IN_B" in text


def test_control_and_leak_numbers_come_from_the_records() -> None:
    d2d = _load("info_spread_policy_ladder_d2d_controls_20261018.summary.json")["table"]
    text = _text()
    for pair in ("L0-T0_1", "L0-T0_4"):
        values = [v[MEASURE]["mean_cell_delta"] for k, v in d2d.items() if k.startswith("POS_sign/") and k.endswith(pair)]
        assert _span(values) in text, pair
    assert f"+{_r(d2d['NULL/B/partial_mlp/L0-F1']['share_precision']['mean_cell_delta'])}" in text
    leak = _load("info_spread_policy_ladder_study_v1_d4_20261016.summary.json")["table"]["B/partial_mlp/null"]["hand_rule"]
    low, high = leak["interval_95"]
    assert f"+{_r(leak['mean_cell_delta'])} [+{_r(low)}, +{_r(high)}]" in text


def test_llm_arm_numbers_come_from_the_record() -> None:
    d5 = _load("info_spread_policy_ladder_llm_arms_d5_20261019.summary.json")
    table = _study_groups(d5["table"])
    text = _text()
    assert _span([v[MEASURE]["nabm_vs_archetypes"]["mean_cell_delta"] for v in table.values()]) in text
    regime_b = [v[MEASURE]["llm_prior_vs_hand"]["mean_cell_delta"] for k, v in table.items() if k.startswith("B/")]
    assert _span(regime_b, signed=True) in text
    counts = d5["registered_reading"]["counts"]["llm_prior_in_learner"]
    assert counts["B"]["INDISTINGUISHABLE"] == 6 and counts["A"]["INDISTINGUISHABLE"] == 4
    assert "not material in any Regime B group and in four of six Regime A groups" in text


def test_reuse_numbers_come_from_the_parity_records() -> None:
    n14 = _load("n14_pivot_parity_20261019.summary.json")
    n15 = _load("n15_package_parity_20261020.summary.json")
    assert n14["parity"]["identical_rows"] == 1152 and n15["runs"]["v3_release"]["whole_file_identical"]
    assert n15["runs"]["abm2_event"]["identical_rows"] == 432
    assert "1,152 of 1,152" in _text()


def test_follow_up_numbers_come_from_the_records() -> None:
    text = _text()
    o23 = _study_groups(_load("info_spread_policy_ladder_transfer_v1_20261021.summary.json")["table"])
    assert _span([-v[MEASURE]["transfer_best_table"]["mean_cell_delta"] for v in o23.values()]) in text
    assert not any(v[MEASURE]["transfer_cost"]["material"] for v in o23.values())
    o26 = _load("info_spread_policy_ladder_flip_rate_v1_20261021.summary.json")["table"]
    for rate, key in (("0.1", "closed_loop"), ("0.25", "best_table")):
        values = [v[MEASURE][key]["mean_cell_delta"] for k, v in o26.items() if k.endswith(f"/p={rate}")]
        assert _span(values, signed=True) in text, rate
    o24 = _study_groups(_load("info_spread_policy_ladder_unclipped_prior_v1_20261021.summary.json")["table"])
    assert _span([v[MEASURE]["unclipped_prior_vs_hand"]["mean_cell_delta"] for v in o24.values()], signed=True) in text
    gains = [v[MEASURE]["unclipped_prior_in_learner"] for v in o24.values()]
    material = [g["mean_cell_delta"] for g in gains if g["material"]]
    assert len(material) == 5 and min(material) > 0
    assert f"positive in 5 of 12 groups, by {_span(material, signed=True)}, and negative in none" in text
    o25 = _load("info_spread_policy_ladder_utility_v1_20261022.summary.json")["dispositions"]
    assert len(o25["disagreements"]) == 5 and len(o25["by_comparison"]) == 20
    assert "Five of 20 summaries change" in text and "disagree in 5 of 20 summaries" in text
    counts = o25["by_comparison"]
    assert counts["llm_prior_in_learner/A"]["utility_counts"]["NEGATIVE"] == 6
    assert counts["llm_prior_in_learner/B"]["utility_counts"]["NEGATIVE"] == 4
    assert "lowers utility in 6 of 6 Regime A groups and 4 of 6 Regime B" in text


def _two(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def test_calibration_numbers_come_from_the_records() -> None:
    text = _text()
    o28 = _load("info_spread_policy_ladder_prior_calibration_v1_20261023.summary.json")
    o33 = _load("info_spread_policy_ladder_world_calibration_v1_20261024.summary.json")
    costed = ("P0/A", "P0/B", "P0u/B")
    assert [o28["dispositions"]["by_case"][c]["attribution"] for c in costed] == [
        "CALIBRATION", "CALIBRATION", "NEITHER_CONSISTENTLY"]
    cases = o33["dispositions"]["by_case"]
    assert [cases[c]["attribution"] for c in costed] == ["CALIBRATION", "BOTH", "ANSWERS"]

    def negative(case: str, comparison: str) -> int:
        return cases[case]["counts"][comparison]["NEGATIVE"]

    a, b, u = costed
    assert negative(a, "calibration_given_llm_answers") == negative(a, "calibration_given_hand_answers") == 6
    assert (negative(a, "answers_given_hand_calibration"), negative(a, "answers_given_llm_calibration")) == (5, 0)
    assert negative(b, "calibration_given_llm_answers") == negative(b, "calibration_given_hand_answers") == 4
    for case in (b, u):
        assert (negative(case, "answers_given_hand_calibration"), negative(case, "answers_given_llm_calibration")) == (6, 5)
    assert negative(u, "calibration_given_llm_answers") == 4
    assert cases[u]["by_comparison"]["calibration_given_hand_answers"] == "MIXED_OR_NONE"
    for row in ("| P0, Regime A | 6 and 6 of 6 negative | 5 of 6 negative; 0 of 6 | Calibration | Calibration |",
                "| P0, Regime B | 4 and 4 of 6 negative | 6 and 5 of 6 negative | Both | Calibration |",
                "| P0u, Regime B | 4 of 6 negative; mixed | 6 and 5 of 6 negative | Answers | Neither alone |"):
        assert row in text, row
    check = o28["calibration_check"]
    ratios = [check[f"{p}/{r}"]["mean_unverified_shares"] / check[f"{p}/{r}"]["calibration_source"]
              for p in ("P0_hcal", "P0u_hcal") for r in ("A", "B")]
    assert f"{_two(min(ratios))} to {_two(max(ratios))} times as much as F1" in text
    world = [row["ratio_to_calibration_source"] for row in o33["world_match_check"].values()]
    assert f"{_r(min(world))} to {_r(max(world))} times as much as their sources" in text
    rows = [v for k, v in o33["learner_description"].items() if k.startswith("B/")]
    pairs = (("L1_hcalw", "L0"), ("L1", "L0_pcalw"), ("L1u_hcalw", "L0"), ("L1u", "L0_ucalw"))
    shares = [row[x]["mean_unverified_shares"] / row[y]["mean_unverified_shares"] - 1 for row in rows for x, y in pairs]
    precision = [row[x]["mean_unverified_share_precision"] - row[y]["mean_unverified_share_precision"]
                 for row in rows for x, y in pairs]
    assert f"between {round(-100 * min(shares))}% fewer and {Decimal(str(100 * max(shares))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}% more" in text
    assert f"from {_r(-min(precision))} lower to {_r(max(precision))} higher" in text


def test_the_section_keeps_its_boundaries() -> None:
    text = _text()
    for required in (
        "This does not imply the absence of learning.",
        "it is not statistical equivalence",
        "do not establish that a neural policy is necessary",
        "donors ran in the same world",
        "not in the released `v0.1.0a5` tag",
        "is not the agents' utility",
        "not evidence about the model's knowledge",
        "about 31%",
        "other worlds of the same generator",
        "Transfer across veracity relations or domains is not tested.",
        "not the reach of false items",
        "on precision only",
        "not to the model's answers",
        "Only the location was matched in the world.",
    ):
        assert required in text, required
    for figure in (FIGURE, FOLLOW_UPS):
        assert figure.is_file()
        assert f"paper/figures/{figure.name}" in text
