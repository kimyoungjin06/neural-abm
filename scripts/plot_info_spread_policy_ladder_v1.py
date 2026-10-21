"""Figure for paper Section 8: the information-spread policy ladder.

Reads only recorded, hash-audited results (consolidation v6 sources):
- study v2 (D4 v2): L0 against the hand rule and the two tables;
- the LLM arms (D5): L0 - X0, L1 - L0, and L1 - P0;
- the D2d controls: POS-sign, and NULL on the old and the unverified measure.
Each point is a group's mean cell difference with its cell-resampled 95%
interval; filled markers are material under the registered group rule.
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
D4V2 = RESULTS / "info_spread_policy_ladder_study_v2_d4_20261019.summary.json"
D5 = RESULTS / "info_spread_policy_ladder_llm_arms_d5_20261019.summary.json"
D2D = RESULTS / "info_spread_policy_ladder_d2d_controls_20261018.summary.json"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "info_spread_policy_ladder_comparisons.png"
GROUPS = [f"{regime}/{family}/{geometry}"
          for regime, families in (("A", ("linear_exact", "mlp")), ("B", ("partial_linear_exact", "partial_mlp")))
          for family in families for geometry in ("linear", "smooth", "axis")]
SHORT = {"linear_exact": "lin", "mlp": "mlp", "partial_linear_exact": "lin", "partial_mlp": "mlp"}
MEASURE = "unverified_share_precision"


def _label(group: str) -> str:
    regime, family, geometry = group.split("/")
    return f"{regime} {SHORT[family]} {geometry}"


def _points(ax, rows: list[tuple[str, dict]], series: list[tuple[str, str, str]], offset_step: float = 0.22) -> None:
    for s_index, (key, name, color) in enumerate(series):
        offset = (s_index - (len(series) - 1) / 2) * offset_step
        for r_index, (_label_text, entry) in enumerate(rows):
            rule = entry[key]
            low, high = rule["interval_95"]
            y = r_index + offset
            ax.plot([low, high], [y, y], color=color, linewidth=1.2)
            ax.plot(rule["mean_cell_delta"], y, marker="o", markersize=4.5, color=color,
                    markerfacecolor=color if rule["material"] else "white")
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _entry in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8)
    if series:
        handles = [plt.Line2D([], [], color=color, marker="o", markersize=4.5) for _key, _name, color in series]
        ax.legend(handles, [name for _key, name, _color in series], fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13), frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    d4 = json.loads(D4V2.read_text(encoding="utf-8"))["table"]
    d5 = json.loads(D5.read_text(encoding="utf-8"))["table"]
    d2d = json.loads(D2D.read_text(encoding="utf-8"))["table"]

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    _points(axes[0], [(_label(g), d4[g][MEASURE]) for g in GROUPS],
            [("hand_rule", "L0 − F1 (hand rule)", "#4c72b0"),
             ("closed_loop", "L0 − T0-1 (one donor)", "#dd8452"),
             ("best_table", "L0 − T0-4 (four donors)", "#55a868")])
    axes[0].set_title("(a) Study v2: learner against rule and tables", fontsize=9)
    axes[0].set_xlabel("Mean cell Δ, precision on unverified items", fontsize=8)

    _points(axes[1], [(_label(g), d5[g][MEASURE]) for g in GROUPS],
            [("nabm_vs_archetypes", "L0 − X0 (archetypes)", "#8172b3"),
             ("llm_prior_in_learner", "L1 − L0 (LLM prior in learner)", "#937860"),
             ("closed_loop_on_llm_prior", "L1 − P0 (closed loop on P0)", "#da8bc3")])
    axes[1].set_title("(b) LLM arms", fontsize=9)
    axes[1].set_xlabel("Mean cell Δ, precision on unverified items", fontsize=8)

    families = (("A", "linear_exact"), ("A", "mlp"), ("B", "partial_linear_exact"), ("B", "partial_mlp"))
    rows = []
    for regime, family in families:
        name = f"{regime} {SHORT[family]}"
        sign1 = d2d[f"POS_sign/{regime}/{family}/L0-T0_1"]
        sign4 = d2d[f"POS_sign/{regime}/{family}/L0-T0_4"]
        null = d2d[f"NULL/{regime}/{family}/L0-F1"]
        rows.append((f"{name} POS-sign", {"t1": sign1[MEASURE], "t4": sign4[MEASURE]}))
        rows.append((f"{name} NULL", {"old": null["share_precision"], "new": null[MEASURE]}))
    sign_rows = [(label, entry) for label, entry in rows if "t1" in entry]
    null_rows = [(label, entry) for label, entry in rows if "old" in entry]
    _points(axes[2], sign_rows + null_rows, [])
    for r_index, (_label_text, entry) in enumerate(sign_rows + null_rows):
        pairs = (("t1", "#dd8452", -0.15), ("t4", "#55a868", 0.15)) if "t1" in entry else \
            (("old", "#c44e52", -0.15), ("new", "#4c72b0", 0.15))
        for key, color, offset in pairs:
            rule = entry[key]
            low, high = rule["interval_95"]
            axes[2].plot([low, high], [r_index + offset] * 2, color=color, linewidth=1.2)
            axes[2].plot(rule["mean_cell_delta"], r_index + offset, marker="o", markersize=4.5, color=color,
                         markerfacecolor=color if rule["material"] else "white")
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="-", markersize=4.5) for c in
               ("#dd8452", "#55a868", "#c44e52", "#4c72b0")]
    axes[2].legend(handles, ["POS-sign: L0 − T0-1", "POS-sign: L0 − T0-4", "NULL: L0 − F1, all items",
                             "NULL: L0 − F1, unverified"], fontsize=7, loc="upper center",
                   bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    axes[2].set_title("(c) Controls (D2d)", fontsize=9)
    axes[2].set_xlabel("Mean cell Δ in share precision", fontsize=8)

    figure.text(0.01, 0.005, "Filled: material under the registered group rule (cell-bootstrap 95% interval "
                "excludes 0 and |mean| ≥ 0.01). Open: not material. 20 cells × 16 replicates per group.",
                fontsize=7, color="0.3")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
