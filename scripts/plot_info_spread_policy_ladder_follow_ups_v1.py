"""Second figure for paper Section 8: the information-spread follow-ups O23, O26, and O25.

Reads only recorded, hash-audited results (consolidation v7 sources, and D5 and O24 for precision):
- O23 transfer: L0 - T0x-4 and T0-4 - T0x-4 by study group;
- O26 flip rate: L0 - T0-1 and L0 - T0-4 against the flip rate, by family;
- O25 utility: L1 - L0 and L1u - L0 by study group, on precision (D5, O24) against utility (O25).
In (a) and (b), each point is a group's mean cell difference with its cell-resampled 95% interval, filled when
material under the registered group rule. In (c), each point is a study group, filled when material on utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "evidence" / "results"
O23 = RESULTS / "info_spread_policy_ladder_transfer_v1_20261021.summary.json"
O26 = RESULTS / "info_spread_policy_ladder_flip_rate_v1_20261021.summary.json"
O24 = RESULTS / "info_spread_policy_ladder_unclipped_prior_v1_20261021.summary.json"
O25 = RESULTS / "info_spread_policy_ladder_utility_v1_20261022.summary.json"
D5 = RESULTS / "info_spread_policy_ladder_llm_arms_d5_20261019.summary.json"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "info_spread_policy_ladder_follow_ups.png"
FAMILIES = (("A", "linear_exact"), ("A", "mlp"), ("B", "partial_linear_exact"), ("B", "partial_mlp"))
GROUPS = [f"{regime}/{family}/{geometry}" for regime, family in FAMILIES for geometry in ("linear", "smooth", "axis")]
SHORT = {"linear_exact": "lin", "mlp": "mlp", "partial_linear_exact": "lin", "partial_mlp": "mlp"}
MEASURE = "unverified_share_precision"
FLIP_RATES = ("0", "0.1", "0.25", "0.5")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _label(group: str) -> str:
    regime, family, geometry = group.split("/")
    return f"{regime} {SHORT[family]} {geometry}"


def _interval(ax, x: float, y: float, rule: dict, color: str, horizontal: bool = True) -> None:
    low, high = rule["interval_95"]
    if horizontal:
        ax.plot([low, high], [y, y], color=color, linewidth=1.2)
        ax.plot(rule["mean_cell_delta"], y, marker="o", markersize=4.5, color=color,
                markerfacecolor=color if rule["material"] else "white")
    else:
        ax.plot([x, x], [low, high], color=color, linewidth=1.2)
        ax.plot(x, rule["mean_cell_delta"], marker="o", markersize=4.5, color=color,
                markerfacecolor=color if rule["material"] else "white")


def _transfer(ax, o23: dict) -> None:
    series = (("transfer_best_table", "L0 − T0x-4 (table from other worlds)", "#55a868", -0.15),
              ("transfer_cost", "T0-4 − T0x-4 (same-world donors' worth)", "#8c8c8c", 0.15))
    for row, group in enumerate(GROUPS):
        for key, _name, color, offset in series:
            _interval(ax, 0.0, row + offset, o23[group][MEASURE][key], color)
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([_label(g) for g in GROUPS], fontsize=7.5)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Mean cell Δ, precision on unverified items", fontsize=8)
    ax.set_title("(a) O23: tables from other worlds", fontsize=9)
    handles = [plt.Line2D([], [], color=color, marker="o", markersize=4.5) for _k, _n, color, _o in series]
    ax.legend(handles, [name for _k, name, _c, _o in series], fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), frameon=False)


def _flip_rate(ax, o26: dict) -> None:
    colors = {"closed_loop": "#dd8452", "best_table": "#55a868"}
    styles = dict(zip(FAMILIES, ("-", "--", "-.", ":"), strict=True))
    for f_index, (regime, family) in enumerate(FAMILIES):
        for key, color in colors.items():
            xs, ys = [], []
            for p_index, rate in enumerate(FLIP_RATES):
                rule = o26[f"{regime}/{family}/p={rate}"][MEASURE][key]
                x = p_index + (f_index - 1.5) * 0.06
                _interval(ax, x, 0.0, rule, color, horizontal=False)
                xs.append(x)
                ys.append(rule["mean_cell_delta"])
            ax.plot(xs, ys, color=color, linewidth=0.9, linestyle=styles[(regime, family)], alpha=0.8)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    shares = [o26[f"A/linear_exact/p={rate}"]["flipped_run_share"] for rate in FLIP_RATES]
    ax.set_xticks(range(len(FLIP_RATES)))
    ax.set_xticklabels([f"p = {rate}\n({share:.0%} reversed)" for rate, share in zip(FLIP_RATES, shares, strict=True)],
                       fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Mean cell Δ, precision on unverified items", fontsize=8)
    ax.set_title("(b) O26: runs whose relation is reversed", fontsize=9)
    handles = [plt.Line2D([], [], color=colors["closed_loop"], marker="o", markersize=4.5),
               plt.Line2D([], [], color=colors["best_table"], marker="o", markersize=4.5)]
    handles += [plt.Line2D([], [], color="0.3", linestyle=styles[f]) for f in FAMILIES]
    names = ["L0 − T0-1 (one donor)", "L0 − T0-4 (four donors)"] + [f"{r} {SHORT[f]}" for r, f in FAMILIES]
    ax.legend(handles, names, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)


def _utility(ax, o25: dict, d5: dict, o24: dict) -> None:
    series = (("llm_prior_in_learner", d5, "L1 − L0 (D5's prior in the learner)", "#937860"),
              ("unclipped_prior_in_learner", o24, "L1u − L0 (unclipped prior)", "#da8bc3"))
    markers = {"A": "o", "B": "s"}
    for key, precision_table, _name, color in series:
        for group in GROUPS:
            utility = o25[group][key]
            precision = precision_table[group][MEASURE][key]["mean_cell_delta"]
            ax.plot(precision, utility["mean_cell_delta"], marker=markers[group[0]], markersize=5, color=color,
                    markerfacecolor=color if utility["material"] else "white", linestyle="none")
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Mean cell Δ, precision on unverified items", fontsize=8)
    ax.set_ylabel("Mean cell Δ, utility per agent on unverified items", fontsize=8)
    ax.set_title("(c) O25: the LLM prior inside the learner", fontsize=9)
    handles = [plt.Line2D([], [], color=color, marker="o", linestyle="none", markersize=5)
               for _k, _t, _n, color in series]
    handles += [plt.Line2D([], [], color="0.3", marker=m, linestyle="none", markerfacecolor="white", markersize=5)
                for m in markers.values()]
    names = [name for _k, _t, name, _c in series] + ["Regime A", "Regime B"]
    ax.legend(handles, names, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1, 1.05, 1]})
    _transfer(axes[0], _load(O23)["table"])
    _flip_rate(axes[1], _load(O26)["table"])
    _utility(axes[2], _load(O25)["table"], _load(D5)["table"], _load(O24)["table"])
    figure.text(0.01, 0.005, "Filled: material under the registered group rule ((a), (b): on precision; (c): on "
                "utility, minimum effect 1% of F1's mean |utility| per regime). Open: not material. "
                "20 cells × 16 replicates per group.", fontsize=7, color="0.3")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
