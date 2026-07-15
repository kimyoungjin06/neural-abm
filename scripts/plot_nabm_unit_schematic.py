#!/usr/bin/env python
"""Render the canonical NABM Unit recurrent-block schematic.

The public documentation asset is the source-of-truth output.  When the local
``research/`` workspace is present, the same render also refreshes the Route B
manuscript Figure 1 copy so the two surfaces cannot silently diverge.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DOC_FIGURES = ROOT / "docs" / "figures"
RESEARCH_FIGURES = ROOT / "research" / "figures"

DOC_PNG = DOC_FIGURES / "nabm_unit_recurrent_block.png"
DOC_SVG = DOC_FIGURES / "nabm_unit_recurrent_block.svg"
RESEARCH_PNG = RESEARCH_FIGURES / "figure1_nabm_lifecycle.png"
RESEARCH_SVG = RESEARCH_FIGURES / "figure1_nabm_lifecycle.svg"

# At the manuscript's 7.2-inch full-width placement, every source font remains
# at least 7.2 pt.  Keep these constants public so the regression test can guard
# the publication typography budget.
PUBLICATION_WIDTH_IN = 7.2
FIGURE_SIZE = (12.0, 10.4)
MIN_TEXT_SIZE = 12.0


COLORS = {
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#64748b",
    "panel": "#ffffff",
    "panel_edge": "#cbd5e1",
    "unit_panel": "#eff6ff",
    "unit_edge": "#60a5fa",
    "unit_fill": "#dbeafe",
    "unit_strong": "#1d4ed8",
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
    "variant_strong": "#92400e",
    "soft": "#f8fafc",
}


def _configure_matplotlib() -> None:
    """Use stable, accessible text output across PNG and SVG renders."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": MIN_TEXT_SIZE,
            "svg.fonttype": "none",
            "svg.hashsalt": "nabm-unit-recurrent-block-v2",
        }
    )


def rounded_rect(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str,
    ec: str,
    lw: float = 1.35,
    radius: float = 0.012,
    linestyle: str = "-",
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.007,rounding_size={radius}",
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
    lw: float = 1.55,
    mutation_scale: float = 13,
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


def panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
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
        lw=1.45,
        radius=0.016,
        zorder=0,
    )
    ax.text(
        x + 0.022,
        y + h - 0.034,
        title,
        ha="left",
        va="center",
        fontsize=16.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + w - 0.022,
        y + h - 0.034,
        subtitle,
        ha="right",
        va="center",
        fontsize=MIN_TEXT_SIZE,
        color=COLORS["muted"],
    )


def label_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    detail: str,
    fc: str,
    ec: str,
    title_color: str = COLORS["ink"],
    detail_color: str = COLORS["muted"],
    title_size: float = 13.0,
    detail_size: float = MIN_TEXT_SIZE,
    linestyle: str = "-",
    title_y: float = 0.66,
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
        lw=1.35,
        radius=0.010,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.text(
        x + w / 2,
        y + h * title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        linespacing=1.04,
        zorder=zorder + 1,
    )
    ax.text(
        x + w / 2,
        y + h * 0.27,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color=detail_color,
        linespacing=1.04,
        zorder=zorder + 1,
    )


def step_badge(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fc: str,
    color: str,
) -> None:
    rounded_rect(
        ax,
        x,
        y,
        0.029,
        0.024,
        fc=fc,
        ec=fc,
        lw=0,
        radius=0.012,
        zorder=8,
    )
    ax.text(
        x + 0.0145,
        y + 0.012,
        text,
        ha="center",
        va="center",
        fontsize=MIN_TEXT_SIZE,
        fontweight="bold",
        color=color,
        zorder=9,
    )


def _draw_full_unit_panel(ax: plt.Axes) -> None:
    x, y, w, h = 0.035, 0.475, 0.93, 0.435
    panel(
        ax,
        x,
        y,
        w,
        h,
        title="A   NABMUnit — generic typed execution path",
        subtitle="one synchronous population step",
    )

    ax.text(
        0.50,
        0.850,
        "Post-local, pre-commit population snapshot",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["typed_strong"],
    )
    arrow(
        ax,
        (0.282, 0.836),
        (0.718, 0.836),
        color=COLORS["typed_edge"],
        lw=1.2,
        mutation_scale=11,
    )

    top_y, top_h, box_w = 0.724, 0.096, 0.185
    xs = [0.065, 0.292, 0.519, 0.746]
    label_box(
        ax,
        xs[0],
        top_y,
        box_w,
        top_h,
        title="Local update ×N",
        detail="agent.local_update(...)\ndomain-supplied method",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
    )
    label_box(
        ax,
        xs[1],
        top_y,
        box_w,
        top_h,
        title="Validated messages ×N",
        detail="social_message → validate",
        fc=COLORS["typed_fill"],
        ec=COLORS["typed_edge"],
        title_color=COLORS["typed_strong"],
    )
    label_box(
        ax,
        xs[2],
        top_y,
        box_w,
        top_h,
        title="PeerSelector(messages)",
        detail="population callback → Pᵢ",
        fc=COLORS["unit_fill"],
        ec=COLORS["unit_edge"],
        title_color=COLORS["unit_strong"],
    )
    label_box(
        ax,
        xs[3],
        top_y,
        box_w,
        top_h,
        title="SocialValueBuilder",
        detail="population callback → vᵢ",
        fc=COLORS["unit_fill"],
        ec=COLORS["unit_edge"],
        title_color=COLORS["unit_strong"],
    )
    for idx, box_x in enumerate(xs, start=1):
        badge_fc = COLORS["domain_strong"] if idx == 1 else COLORS["unit_strong"]
        step_badge(
            ax,
            box_x - 0.012,
            top_y + top_h - 0.012,
            f"{idx}",
            fc=badge_fc,
            color="white",
        )

    for left, right in zip(xs[:-1], xs[1:], strict=True):
        arrow(
            ax,
            (left + box_w, top_y + top_h / 2),
            (right, top_y + top_h / 2),
            color=COLORS["unit_strong"],
        )

    # Step execution container: selection and value construction are separate
    # ordered callbacks; their P_i and v_i products meet only at the mixer.
    step_x, step_y, step_w, step_h = 0.100, 0.558, 0.620, 0.124
    rounded_rect(
        ax,
        step_x,
        step_y,
        step_w,
        step_h,
        fc=COLORS["unit_panel"],
        ec=COLORS["unit_edge"],
        lw=1.6,
        radius=0.014,
        zorder=1,
    )
    ax.text(
        step_x + 0.018,
        step_y + step_h - 0.020,
        "5   NABMStep.run",
        ha="left",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=COLORS["unit_strong"],
    )

    inner_y, inner_h = 0.575, 0.070
    inner_xs = [0.125, 0.305, 0.535]
    inner_ws = [0.145, 0.195, 0.155]
    label_box(
        ax,
        inner_xs[0],
        inner_y,
        inner_ws[0],
        inner_h,
        title="SocialChannel",
        detail="kind · bounds\ncommit mode",
        fc=COLORS["typed_fill"],
        ec=COLORS["typed_edge"],
        title_color=COLORS["typed_strong"],
        title_size=12.5,
        title_y=0.67,
    )
    label_box(
        ax,
        inner_xs[1],
        inner_y,
        inner_ws[1],
        inner_h,
        title="SocialBlock.mix(Pᵢ, vᵢ)",
        detail="v̂ᵢ = (1−α)vᵢ + α·mᵢ\nmᵢ = mean(vⱼ : j∈Pᵢ)",
        fc=COLORS["typed_fill"],
        ec=COLORS["typed_edge"],
        title_color=COLORS["typed_strong"],
        title_size=12.5,
        title_y=0.67,
    )
    label_box(
        ax,
        inner_xs[2],
        inner_y,
        inner_ws[2],
        inner_h,
        title="CommitAdapter",
        detail="optional · injected",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
        title_size=12.5,
        linestyle="--",
        title_y=0.67,
    )
    arrow(ax, (0.270, 0.610), (0.305, 0.610), color=COLORS["typed_strong"])
    arrow(ax, (0.500, 0.610), (0.535, 0.610), color=COLORS["domain_strong"])

    # Ordered-control arrow plus the two actual mixer inputs.
    arrow(
        ax,
        (xs[3] + box_w / 2, top_y),
        (0.402, step_y + step_h),
        color=COLORS["unit_strong"],
        connectionstyle="arc3,rad=-0.12",
    )
    arrow(
        ax,
        (xs[2] + box_w / 2, top_y),
        (0.355, step_y + step_h),
        color=COLORS["unit_strong"],
        connectionstyle="arc3,rad=0.08",
    )
    ax.text(
        0.464,
        0.697,
        "Pᵢ + vᵢ",
        ha="center",
        va="center",
        fontsize=MIN_TEXT_SIZE,
        fontweight="bold",
        color=COLORS["unit_strong"],
    )

    audit_x, audit_y, audit_w, audit_h = 0.775, 0.558, 0.155, 0.124
    label_box(
        ax,
        audit_x,
        audit_y,
        audit_w,
        audit_h,
        title="6   Audit + report",
        detail="NABMStepResult\n→ NABMUnitReport",
        fc=COLORS["audit_fill"],
        ec=COLORS["audit_edge"],
        title_color=COLORS["audit_strong"],
        title_y=0.68,
    )
    arrow(
        ax,
        (step_x + step_w, step_y + step_h / 2),
        (audit_x, audit_y + audit_h / 2),
        color=COLORS["audit_strong"],
    )

    domain_x, domain_y, domain_w, domain_h = 0.180, 0.488, 0.640, 0.052
    rounded_rect(
        ax,
        domain_x,
        domain_y,
        domain_w,
        domain_h,
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        lw=1.35,
        radius=0.010,
        zorder=2,
    )
    ax.text(
        domain_x + domain_w / 2,
        domain_y + domain_h * 0.67,
        "Domain owns the state advance",
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color=COLORS["domain_strong"],
        zorder=3,
    )
    ax.text(
        domain_x + domain_w / 2,
        domain_y + domain_h * 0.27,
        "execution site: injected CommitAdapter inside step  OR  external caller after report",
        ha="center",
        va="center",
        fontsize=MIN_TEXT_SIZE,
        color=COLORS["domain_strong"],
        zorder=3,
    )
    arrow(
        ax,
        (0.612, inner_y),
        (0.570, domain_y + domain_h),
        color=COLORS["domain_strong"],
        linestyle="--",
        connectionstyle="arc3,rad=0.10",
    )
    arrow(
        ax,
        (audit_x + audit_w / 2, audit_y),
        (0.735, domain_y + domain_h),
        color=COLORS["domain_strong"],
        linestyle="--",
        connectionstyle="arc3,rad=-0.10",
    )
    ax.plot(
        [domain_x, 0.053, 0.053],
        [domain_y + domain_h / 2, domain_y + domain_h / 2, top_y + top_h / 2],
        color=COLORS["domain_strong"],
        linewidth=1.45,
        solid_capstyle="round",
        zorder=4,
    )
    arrow(
        ax,
        (0.053, top_y + top_h / 2),
        (xs[0], top_y + top_h / 2),
        color=COLORS["domain_strong"],
        lw=1.45,
    )
    ax.text(
        0.053,
        0.625,
        "next timestep  t → t+1",
        ha="center",
        va="center",
        fontsize=MIN_TEXT_SIZE,
        fontweight="bold",
        color=COLORS["domain_strong"],
        rotation=90,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"},
        zorder=6,
    )


def _draw_lite_panel(ax: plt.Axes) -> None:
    x, y, w, h = 0.035, 0.080, 0.93, 0.350
    panel(
        ax,
        x,
        y,
        w,
        h,
        title="B   Torch-free bounded-scalar instantiation",
        subtitle="exact runtime order in scenario_lite",
    )

    rounded_rect(
        ax,
        0.185,
        0.346,
        0.630,
        0.034,
        fc=COLORS["soft"],
        ec=COLORS["panel_edge"],
        lw=1.0,
        radius=0.010,
        zorder=1,
    )
    ax.text(
        0.50,
        0.363,
        "api_lite  →  scenario_lite  →  workflow_lite  →  social_core",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["unit_strong"],
    )

    top_y, node_h, node_w = 0.242, 0.082, 0.185
    top_xs = [0.065, 0.292, 0.519, 0.746]
    label_box(
        ax,
        top_xs[0],
        top_y,
        node_w,
        node_h,
        title="Scenario + agents",
        detail="initialise once",
        fc=COLORS["soft"],
        ec=COLORS["panel_edge"],
        title_color=COLORS["muted"],
    )
    label_box(
        ax,
        top_xs[1],
        top_y,
        node_w,
        node_h,
        title="Candidate neighbors",
        detail="build before workflow",
        fc=COLORS["unit_fill"],
        ec=COLORS["unit_edge"],
        title_color=COLORS["unit_strong"],
    )
    label_box(
        ax,
        top_xs[2],
        top_y,
        node_w,
        node_h,
        title="Local adaptation",
        detail="then read bounded xᵢ",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
    )
    label_box(
        ax,
        top_xs[3],
        top_y,
        node_w,
        node_h,
        title="Peer filter",
        detail="uses post-local xᵢ",
        fc=COLORS["unit_fill"],
        ec=COLORS["unit_edge"],
        title_color=COLORS["unit_strong"],
    )
    for idx, box_x in enumerate(top_xs, start=0):
        badge_fc = COLORS["muted"] if idx == 0 else COLORS["unit_strong"]
        step_badge(
            ax,
            box_x - 0.012,
            top_y + node_h - 0.012,
            f"{idx}",
            fc=badge_fc,
            color="white",
        )
    for left, right in zip(top_xs[:-1], top_xs[1:], strict=True):
        arrow(
            ax,
            (left + node_w, top_y + node_h / 2),
            (right, top_y + node_h / 2),
            color=COLORS["unit_strong"],
        )

    bottom_y = 0.127
    bottom_xs = [0.746, 0.452, 0.158]
    label_box(
        ax,
        bottom_xs[0],
        bottom_y,
        node_w,
        node_h,
        title="Bounded-scalar mix",
        detail="(1−α)xᵢ + α mean(xⱼ : j∈Pᵢ)",
        fc=COLORS["typed_fill"],
        ec=COLORS["typed_edge"],
        title_color=COLORS["typed_strong"],
    )
    label_box(
        ax,
        bottom_xs[1],
        bottom_y,
        node_w,
        node_h,
        title="Domain transition",
        detail="actual state mutation",
        fc=COLORS["domain_fill"],
        ec=COLORS["domain_edge"],
        title_color=COLORS["domain_strong"],
    )
    label_box(
        ax,
        bottom_xs[2],
        bottom_y,
        node_w,
        node_h,
        title="Aggregate + micro audit",
        detail="step row + agent rows",
        fc=COLORS["audit_fill"],
        ec=COLORS["audit_edge"],
        title_color=COLORS["audit_strong"],
        title_size=12.5,
    )
    for idx, box_x in zip([4, 5, 6], bottom_xs, strict=True):
        step_badge(
            ax,
            box_x - 0.012,
            bottom_y + node_h - 0.012,
            f"{idx}",
            fc=COLORS["unit_strong"],
            color="white",
        )

    arrow(
        ax,
        (top_xs[3] + node_w / 2, top_y),
        (bottom_xs[0] + node_w / 2, bottom_y + node_h),
        color=COLORS["typed_strong"],
    )
    arrow(
        ax,
        (bottom_xs[0], bottom_y + node_h / 2),
        (bottom_xs[1] + node_w, bottom_y + node_h / 2),
        color=COLORS["domain_strong"],
    )
    arrow(
        ax,
        (bottom_xs[1], bottom_y + node_h / 2),
        (bottom_xs[2] + node_w, bottom_y + node_h / 2),
        color=COLORS["audit_strong"],
    )

    # The loop deliberately returns to neighbor construction, not to scenario
    # initialization: scenario_lite rebuilds candidates before every workflow.
    return_x = top_xs[1] + node_w / 2
    return_y = 0.223
    ax.plot(
        [bottom_xs[2], 0.085, 0.085, return_x],
        [bottom_y + node_h / 2, bottom_y + node_h / 2, return_y, return_y],
        color=COLORS["unit_strong"],
        linewidth=1.55,
        solid_capstyle="round",
        zorder=4,
    )
    arrow(
        ax,
        (return_x, return_y),
        (return_x, top_y),
        color=COLORS["unit_strong"],
        lw=1.55,
    )
    ax.text(
        0.185,
        return_y,
        "repeat each timestep  t → t+1",
        ha="center",
        va="center",
        fontsize=MIN_TEXT_SIZE,
        fontweight="bold",
        color=COLORS["unit_strong"],
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none"},
        zorder=6,
    )

    ax.text(
        0.50,
        0.100,
        "Exact here: DeGroot averaging · Granovetter threshold cascade     |     Near variants: FJ-like anchor · self-excluding HK",
        ha="center",
        va="center",
        fontsize=MIN_TEXT_SIZE,
        color=COLORS["muted"],
    )


def build_figure() -> Figure:
    """Build the full-width, two-panel execution schematic."""

    _configure_matplotlib()
    final_min_size = MIN_TEXT_SIZE * PUBLICATION_WIDTH_IN / FIGURE_SIZE[0]
    if final_min_size < 7.0:
        raise ValueError(
            f"publication text would shrink below 7 pt: {final_min_size:.2f}"
        )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.035,
        0.965,
        "NABM Unit: recurrent execution without hidden state transitions",
        ha="left",
        va="center",
        fontsize=20.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.035,
        0.936,
        "Typed exchange, explicit execution ownership, and audit outputs at every timestep",
        ha="left",
        va="center",
        fontsize=13.0,
        color=COLORS["muted"],
    )

    _draw_full_unit_panel(ax)
    _draw_lite_panel(ax)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def _normalise_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8"
    )


def render_targets(png_path: Path, svg_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    try:
        fig.savefig(
            png_path,
            dpi=260,
            facecolor=fig.get_facecolor(),
            metadata={"Software": "Neural ABM schematic generator"},
        )
        fig.savefig(
            svg_path,
            facecolor=fig.get_facecolor(),
            metadata={"Date": None, "Creator": "Neural ABM schematic generator"},
        )
    finally:
        plt.close(fig)
    _normalise_svg(svg_path)


def check_docs_assets() -> None:
    """Fail when tracked documentation assets are stale."""

    with tempfile.TemporaryDirectory(prefix="nabm-schematic-") as temp_dir:
        temp_root = Path(temp_dir)
        png_path = temp_root / DOC_PNG.name
        svg_path = temp_root / DOC_SVG.name
        render_targets(png_path, svg_path)
        stale = [
            tracked.name
            for tracked, rendered in ((DOC_PNG, png_path), (DOC_SVG, svg_path))
            if not tracked.exists() or tracked.read_bytes() != rendered.read_bytes()
        ]
    if stale:
        raise SystemExit(f"stale schematic asset(s): {', '.join(stale)}")
    print("schematic assets are current")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="Refresh only the tracked documentation assets.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Render in a temporary directory and fail if tracked assets differ.",
    )
    args = parser.parse_args()

    if args.check:
        check_docs_assets()
        return

    render_targets(DOC_PNG, DOC_SVG)
    if not args.docs_only and RESEARCH_FIGURES.exists():
        RESEARCH_FIGURES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DOC_PNG, RESEARCH_PNG)
        shutil.copyfile(DOC_SVG, RESEARCH_SVG)

    print(f"wrote {DOC_PNG.relative_to(ROOT)}")
    print(f"wrote {DOC_SVG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
