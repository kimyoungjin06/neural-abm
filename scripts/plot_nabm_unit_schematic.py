#!/usr/bin/env python
"""Render the canonical NABM Unit recurrent-block schematic.

The public documentation asset is the source-of-truth output.  When the local
``research/`` workspace is present, the same render also refreshes the Route B
manuscript Figure 1 copy so the two surfaces cannot silently diverge.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DOC_FIGURES = ROOT / "docs" / "figures"
RESEARCH_FIGURES = ROOT / "research" / "figures"

DOC_PNG = DOC_FIGURES / "nabm_unit_recurrent_block.png"
DOC_SVG = DOC_FIGURES / "nabm_unit_recurrent_block.svg"
RESEARCH_PNG = RESEARCH_FIGURES / "figure1_nabm_lifecycle.png"
RESEARCH_SVG = RESEARCH_FIGURES / "figure1_nabm_lifecycle.svg"


COLORS = {
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#64748b",
    "panel": "#ffffff",
    "panel_edge": "#cbd5e1",
    "unit_panel": "#eff6ff",
    "unit_edge": "#60a5fa",
    "unit_fill": "#dbeafe",
    "unit_strong": "#2563eb",
    "domain_fill": "#fff7ed",
    "domain_edge": "#fb923c",
    "domain_strong": "#c2410c",
    "typed_fill": "#ecfdf5",
    "typed_edge": "#34d399",
    "typed_strong": "#047857",
    "audit_fill": "#f5f3ff",
    "audit_edge": "#a78bfa",
    "audit_strong": "#6d28d9",
    "variant_fill": "#fffbeb",
    "variant_edge": "#f59e0b",
    "variant_strong": "#b45309",
    "soft": "#f8fafc",
}


def rounded_rect(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str,
    ec: str,
    lw: float = 1.2,
    radius: float = 0.012,
    linestyle: str = "-",
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        facecolor=fc,
        edgecolor=ec,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    lw: float = 1.35,
    mutation_scale: float = 10,
    connectionstyle: str = "arc3",
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
            linestyle=linestyle,
            alpha=alpha,
            shrinkA=3,
            shrinkB=3,
            zorder=zorder,
        )
    )


def label_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    detail: str = "",
    *,
    fc: str,
    ec: str,
    title_color: str = COLORS["ink"],
    detail_color: str = COLORS["muted"],
    title_size: float = 8.0,
    detail_size: float = 6.7,
    linestyle: str = "-",
    title_fraction: float = 0.64,
    zorder: int = 2,
) -> None:
    rounded_rect(
        ax,
        x,
        y,
        w,
        h,
        fc=fc,
        ec=ec,
        lw=1.15,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.text(
        x + w / 2,
        y + h * title_fraction,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        color=title_color,
        fontweight="bold",
        linespacing=1.02,
        zorder=zorder + 1,
    )
    if detail:
        ax.text(
            x + w / 2,
            y + h * 0.27,
            detail,
            ha="center",
            va="center",
            fontsize=detail_size,
            color=detail_color,
            linespacing=1.03,
            zorder=zorder + 1,
        )


def panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    letter: str,
    title: str,
    subtitle: str,
) -> None:
    rounded_rect(
        ax,
        x,
        y,
        w,
        h,
        fc=COLORS["panel"],
        ec=COLORS["panel_edge"],
        lw=1.25,
        radius=0.014,
        zorder=0,
    )
    ax.text(
        x + 0.018,
        y + h - 0.032,
        letter,
        ha="left",
        va="center",
        fontsize=12.5,
        color=COLORS["unit_strong"],
        fontweight="bold",
    )
    ax.text(
        x + 0.052,
        y + h - 0.032,
        title,
        ha="left",
        va="center",
        fontsize=12.0,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        x + 0.018,
        y + h - 0.064,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.7,
        color=COLORS["muted"],
    )


def chip(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fc: str,
    ec: str,
    color: str,
    width: float,
    height: float = 0.028,
    size: float = 6.8,
) -> None:
    rounded_rect(ax, x, y, width, height, fc=fc, ec=ec, lw=0.9, radius=0.008)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        fontweight="bold",
        zorder=3,
    )


def draw_full_unit_panel(ax: plt.Axes) -> None:
    x0, y0, w, h = 0.028, 0.075, 0.632, 0.815
    panel(
        ax,
        x0,
        y0,
        w,
        h,
        "A",
        "NABM Unit as an auditable recurrent block",
        "One call to NABMUnit.run; injected semantics remain outside the reusable contract",
    )

    label_box(
        ax,
        0.095,
        0.765,
        0.495,
        0.062,
        r"Domain / runner supplies  $E_t,\ G_t,$ observations, and objectives",
        "caller owns environment meaning, topology semantics, and temporal repetition",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
        title_size=8.5,
        detail_size=6.7,
    )

    unit_x, unit_y, unit_w, unit_h = 0.075, 0.255, 0.535, 0.475
    rounded_rect(
        ax,
        unit_x,
        unit_y,
        unit_w,
        unit_h,
        fc=COLORS["unit_panel"],
        ec=COLORS["unit_edge"],
        lw=1.45,
        radius=0.014,
    )
    ax.text(
        unit_x + 0.018,
        unit_y + unit_h - 0.026,
        "NABMUnit.run — one synchronous step",
        fontsize=10.2,
        color=COLORS["unit_strong"],
        fontweight="bold",
        va="center",
    )
    chip(
        ax,
        unit_x + unit_w - 0.102,
        unit_y + unit_h - 0.042,
        "× N agents",
        fc="#ffffff",
        ec=COLORS["unit_edge"],
        color=COLORS["unit_strong"],
        width=0.082,
    )

    local = (0.105, 0.605, 0.205, 0.072)
    message = (0.376, 0.605, 0.198, 0.072)
    label_box(
        ax,
        *local,
        "① Optional local update",
        "agent.local_update(...) · injected adapter",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
        linestyle="--",
    )
    label_box(
        ax,
        *message,
        "Validated social message",
        r"$m_i$ · SocialMessageSpec",
        fc=COLORS["typed_fill"],
        ec=COLORS["typed_edge"],
        title_color=COLORS["typed_strong"],
    )
    arrow(
        ax,
        (local[0] + local[2], local[1] + local[3] / 2),
        (message[0], message[1] + message[3] / 2),
        color=COLORS["unit_strong"],
    )

    selector = (0.105, 0.495, 0.205, 0.072)
    values = (0.376, 0.495, 0.198, 0.072)
    label_box(
        ax,
        *selector,
        "② Peer selector",
        r"$P_i=P(G_t,\{m_j\};\tau)$ · injected rule",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
        linestyle="--",
    )
    label_box(
        ax,
        *values,
        "Typed value builder",
        r"$v_i=B(agents,\ messages)$",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
        linestyle="--",
    )
    arrow(
        ax,
        (message[0] + message[2] * 0.35, message[1]),
        (selector[0] + selector[2] / 2, selector[1] + selector[3]),
        color=COLORS["typed_strong"],
        connectionstyle="arc3,rad=0.10",
    )
    arrow(
        ax,
        (message[0] + message[2] * 0.72, message[1]),
        (values[0] + values[2] / 2, values[1] + values[3]),
        color=COLORS["typed_strong"],
    )

    barrier_y = 0.468
    ax.plot(
        [0.105, 0.574],
        [barrier_y, barrier_y],
        color=COLORS["line"],
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        alpha=0.8,
        zorder=2,
    )
    ax.text(
        0.339,
        barrier_y + 0.007,
        "synchronous pre-commit snapshot",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=COLORS["muted"],
        fontstyle="italic",
    )

    step_x, step_y, step_w, step_h = 0.097, 0.325, 0.485, 0.125
    rounded_rect(
        ax,
        step_x,
        step_y,
        step_w,
        step_h,
        fc="#ffffff",
        ec=COLORS["typed_edge"],
        lw=1.25,
        radius=0.010,
    )
    ax.text(
        step_x + 0.010,
        step_y + step_h - 0.016,
        "NABMStep.run",
        fontsize=7.2,
        color=COLORS["typed_strong"],
        fontweight="bold",
        va="center",
    )

    channel = (0.112, 0.348, 0.112, 0.065)
    social = (0.245, 0.342, 0.205, 0.078)
    commit = (0.469, 0.348, 0.098, 0.065)
    label_box(
        ax,
        *channel,
        "③ SocialChannel",
        "kind · bounds\ncommit mode · alignment",
        fc=COLORS["typed_fill"],
        ec=COLORS["typed_edge"],
        title_color=COLORS["typed_strong"],
        title_size=7.0,
        detail_size=5.8,
    )
    label_box(
        ax,
        *social,
        "SocialBlock.mix  (α)",
        r"$\hat v_i=(1-\alpha)v_i+\alpha\,\mathrm{mean}_{j\in P_i}v_j$"
        "\nempty peer set: keep self",
        fc=COLORS["typed_fill"],
        ec=COLORS["typed_edge"],
        title_color=COLORS["typed_strong"],
        title_size=7.5,
        detail_size=5.8,
        title_fraction=0.69,
    )
    label_box(
        ax,
        *commit,
        "④ Commit",
        "adapter\n(optional)",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
        title_size=7.0,
        detail_size=5.7,
        linestyle="--",
    )
    arrow(
        ax,
        (channel[0] + channel[2], channel[1] + channel[3] / 2),
        (social[0], social[1] + social[3] / 2),
        color=COLORS["typed_strong"],
    )
    arrow(
        ax,
        (social[0] + social[2], social[1] + social[3] / 2),
        (commit[0], commit[1] + commit[3] / 2),
        color=COLORS["typed_strong"],
    )
    arrow(
        ax,
        (selector[0] + selector[2] / 2, selector[1]),
        (social[0] + social[2] * 0.28, social[1] + social[3]),
        color=COLORS["line"],
        connectionstyle="arc3,rad=-0.10",
    )
    arrow(
        ax,
        (values[0] + values[2] / 2, values[1]),
        (social[0] + social[2] * 0.72, social[1] + social[3]),
        color=COLORS["line"],
        connectionstyle="arc3,rad=0.08",
    )

    rail_x, rail_y, rail_w, rail_h = 0.104, 0.275, 0.470, 0.035
    rounded_rect(
        ax,
        rail_x,
        rail_y,
        rail_w,
        rail_h,
        fc=COLORS["audit_fill"],
        ec=COLORS["audit_edge"],
        lw=1.0,
        radius=0.008,
    )
    ax.text(
        rail_x + rail_w / 2,
        rail_y + rail_h / 2,
        "Audit rail  ·  local losses  ·  messages / peers  ·  update norms  ·  commit losses  ·  post-commit logs",
        ha="center",
        va="center",
        fontsize=6.35,
        color=COLORS["audit_strong"],
        fontweight="bold",
        zorder=3,
    )
    for tap_x in (
        local[0] + local[2] / 2,
        message[0] + message[2] / 2,
        social[0] + social[2] / 2,
        commit[0] + commit[2] / 2,
    ):
        arrow(
            ax,
            (tap_x, 0.325),
            (tap_x, rail_y + rail_h),
            color=COLORS["audit_edge"],
            lw=0.8,
            mutation_scale=7,
            linestyle="--",
            alpha=0.8,
        )

    arrow(
        ax,
        (0.342, 0.765),
        (0.342, unit_y + unit_h),
        color=COLORS["line"],
        lw=1.5,
    )
    arrow(
        ax,
        (0.083, unit_y),
        (0.105, 0.197),
        color=COLORS["line"],
        lw=1.45,
    )

    label_box(
        ax,
        0.095,
        0.125,
        0.495,
        0.066,
        r"Domain transition  $E_t \rightarrow E_{t+1}$",
        "actions, resources, payoffs, topology, adoption · caller-owned",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
        title_size=8.5,
        detail_size=6.5,
    )
    arrow(
        ax,
        (0.598, 0.158),
        (0.598, 0.796),
        color=COLORS["line"],
        lw=1.1,
        mutation_scale=9,
        connectionstyle="arc3,rad=0.08",
        alpha=0.75,
    )
    ax.text(
        0.625,
        0.478,
        "caller loop  × T steps",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["muted"],
        fontweight="bold",
        rotation=90,
    )

    chip(
        ax,
        0.102,
        0.218,
        "scalar probability",
        fc="#ffffff",
        ec=COLORS["typed_edge"],
        color=COLORS["typed_strong"],
        width=0.092,
        size=5.7,
    )
    chip(
        ax,
        0.202,
        0.218,
        "bounded scalar",
        fc="#ffffff",
        ec=COLORS["typed_edge"],
        color=COLORS["typed_strong"],
        width=0.083,
        size=5.7,
    )
    chip(
        ax,
        0.293,
        0.218,
        "distribution",
        fc="#ffffff",
        ec=COLORS["typed_edge"],
        color=COLORS["typed_strong"],
        width=0.071,
        size=5.7,
    )
    chip(
        ax,
        0.372,
        0.218,
        "tensor",
        fc="#ffffff",
        ec=COLORS["typed_edge"],
        color=COLORS["typed_strong"],
        width=0.052,
        size=5.7,
    )
    chip(
        ax,
        0.432,
        0.218,
        "state dict",
        fc="#ffffff",
        ec=COLORS["typed_edge"],
        color=COLORS["typed_strong"],
        width=0.066,
        size=5.7,
    )
    ax.text(
        0.510,
        0.232,
        "supported channel kinds",
        ha="left",
        va="center",
        fontsize=5.9,
        color=COLORS["muted"],
        fontstyle="italic",
    )


def draw_lite_panel(ax: plt.Axes) -> None:
    x0, y0, w, h = 0.682, 0.075, 0.290, 0.815
    panel(
        ax,
        x0,
        y0,
        w,
        h,
        "B",
        "Classical-example instantiation",
        "The torch-free path used by examples/classical_reductions.py",
    )

    label_box(
        ax,
        0.710,
        0.770,
        0.236,
        0.057,
        "api_lite  →  scenario_lite",
        "workflow_lite  →  social_core",
        fc=COLORS["soft"],
        ec=COLORS["panel_edge"],
        title_size=7.8,
        detail_size=6.5,
    )

    boxes = [
        (
            0.710,
            0.682,
            "Scenario + agents",
            "state, anchors, thresholds",
            COLORS["domain_fill"],
            COLORS["domain_edge"],
            COLORS["domain_strong"],
        ),
        (
            0.710,
            0.595,
            "① Local adaptation",
            "off / pre-mix anchor / custom callback",
            COLORS["unit_fill"],
            COLORS["unit_edge"],
            COLORS["unit_strong"],
        ),
        (
            0.710,
            0.508,
            "② Neighbor + peer filter",
            r"topology · similarity threshold $\tau$",
            COLORS["typed_fill"],
            COLORS["typed_edge"],
            COLORS["typed_strong"],
        ),
        (
            0.710,
            0.421,
            "③ Bounded-scalar mix",
            r"typed channel · social strength $\alpha$",
            COLORS["typed_fill"],
            COLORS["typed_edge"],
            COLORS["typed_strong"],
        ),
        (
            0.710,
            0.334,
            "④ Domain transition",
            "overwrite / absorbing threshold",
            COLORS["domain_fill"],
            COLORS["domain_edge"],
            COLORS["domain_strong"],
        ),
        (
            0.710,
            0.247,
            "Aggregate + micro audit",
            "actual post-transition state",
            COLORS["audit_fill"],
            COLORS["audit_edge"],
            COLORS["audit_strong"],
        ),
    ]
    box_w, box_h = 0.236, 0.061
    for x, y, title, detail, fc, ec, title_color in boxes:
        label_box(
            ax,
            x,
            y,
            box_w,
            box_h,
            title,
            detail,
            fc=fc,
            ec=ec,
            title_color=title_color,
            title_size=7.5,
            detail_size=6.0,
        )
    for upper, lower in zip(boxes, boxes[1:]):
        arrow(
            ax,
            (upper[0] + box_w / 2, upper[1]),
            (lower[0] + box_w / 2, lower[1] + box_h),
            color=COLORS["line"],
            lw=1.15,
            mutation_scale=8,
        )

    rounded_rect(
        ax,
        0.702,
        0.107,
        0.252,
        0.105,
        fc=COLORS["soft"],
        ec=COLORS["panel_edge"],
        lw=1.0,
        radius=0.010,
    )
    ax.text(
        0.715,
        0.193,
        "Current evidence labels",
        ha="left",
        va="center",
        fontsize=7.0,
        color=COLORS["ink"],
        fontweight="bold",
    )
    chip(
        ax,
        0.714,
        0.156,
        "DeGroot · exact instance",
        fc=COLORS["typed_fill"],
        ec=COLORS["typed_edge"],
        color=COLORS["typed_strong"],
        width=0.108,
        height=0.026,
        size=5.8,
    )
    chip(
        ax,
        0.830,
        0.156,
        "Granovetter · knife edge",
        fc=COLORS["typed_fill"],
        ec=COLORS["typed_edge"],
        color=COLORS["typed_strong"],
        width=0.112,
        height=0.026,
        size=5.8,
    )
    chip(
        ax,
        0.714,
        0.120,
        "FJ-like · pre-mix anchor",
        fc=COLORS["variant_fill"],
        ec=COLORS["variant_edge"],
        color=COLORS["variant_strong"],
        width=0.108,
        height=0.026,
        size=5.8,
    )
    chip(
        ax,
        0.830,
        0.120,
        "HK · self-excluding variant",
        fc=COLORS["variant_fill"],
        ec=COLORS["variant_edge"],
        color=COLORS["variant_strong"],
        width=0.112,
        height=0.026,
        size=5.8,
    )


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "neural-abm-nabm-unit-v1",
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(16.2, 9.1))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "NABM Unit: an auditable recurrent block",
        fontsize=17.5,
        fontweight="bold",
        color=COLORS["ink"],
        y=0.975,
    )
    ax.text(
        0.5,
        0.928,
        (
            "Full typed unit contract (A) and the lightweight bounded-scalar "
            "instantiation used by the classical examples (B)"
        ),
        ha="center",
        va="center",
        fontsize=9.0,
        color=COLORS["muted"],
    )

    draw_full_unit_panel(ax)
    draw_lite_panel(ax)

    ax.text(
        0.5,
        0.025,
        (
            "Blue: unit orchestration   ·   Orange dashed: injected domain/backend adapter   ·   "
            "Green: typed exchange contract   ·   Purple: audit-only output"
        ),
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["muted"],
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="Write only the tracked public documentation assets.",
    )
    return parser.parse_args()


def normalize_svg_whitespace(path: Path) -> None:
    """Remove renderer-added line-end spaces from the tracked SVG."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    for target in (DOC_PNG, DOC_SVG):
        kwargs = (
            {"dpi": 260} if target.suffix == ".png" else {"metadata": {"Date": None}}
        )
        fig.savefig(target, bbox_inches="tight", **kwargs)
        if target.suffix == ".svg":
            normalize_svg_whitespace(target)
        print(target.relative_to(ROOT))
    plt.close(fig)

    if not args.docs_only and RESEARCH_FIGURES.parent.exists():
        RESEARCH_FIGURES.mkdir(parents=True, exist_ok=True)
        for source, target in (
            (DOC_PNG, RESEARCH_PNG),
            (DOC_SVG, RESEARCH_SVG),
        ):
            shutil.copy2(source, target)
            print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
