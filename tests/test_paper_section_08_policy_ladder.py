"""Paper Section 8 states the recorded results, and only with their boundaries."""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "paper" / "sections" / "08-policy-ladder.md"
FIGURE = ROOT / "paper" / "figures" / "info_spread_policy_ladder_comparisons.png"
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
    ):
        assert required in text, required
    assert FIGURE.is_file()
    assert "paper/figures/info_spread_policy_ladder_comparisons.png" in text
