#!/usr/bin/env python3
"""
Generate the multi-rung sensitivity figure for the relative score gain
of the facial modality (unstratified; manuscript Figure 2).

    figures/marginal_value_rungs.png
        Primary metric — ``pct_of_baseline`` (Δ_face = Z / μ_cell × 100,
        manuscript Eq. 3, using the common per-cell yardstick μ). Gated by
        the normalisation rule (n ≥ 5 paired comparisons, recognised
        instrument), so it covers the primary synthesis set only.

The PROBAST+AI-stratified variant was removed: the credible core (k = 3)
is too small to support a stratum and is reported in text instead.

Usage:
    python scripts/generate_marginal_value_figure.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from marginal_metric import (
    BANDS_PCT,
    BAND_COLOURS,
    BAND_LABELS_SHORT,
    load_extraction_data,
)


# ── Figure variants ──────────────────────────────────────────────────────────
# Single variant: the primary metric (Eq. 3).
@dataclass(frozen=True)
class FigureSpec:
    metric_col: str      # column in the prepared dataframe
    out_name: str        # PNG filename written under figures/
    xlabel: str          # x-axis label


FIGURE_SPECS = [
    FigureSpec(
        metric_col="pct_of_baseline",
        out_name="marginal_value_rungs.png",
        xlabel="Relative gain of facial modality",
    ),
]


# ── Data pipeline ────────────────────────────────────────────────────────────


def load_analysis(data, metric_col):
    """Return the analysis dataframe (rows with a valid *metric_col*)."""
    return data.dropna(subset=[metric_col, "metric_family"]).copy()


# ── Rung collapsing ──────────────────────────────────────────────────────────

RUNG_DEFS = [
    ("All comparisons", "comparisons"),
    ("Per model–dataset", "pairs"),
    ("Per study", "studies"),
]


def collapse_to_rungs(analysis, metric_col):
    """Return list of (label, unit_label, DataFrame) for rungs 0–2."""
    rungs = []
    rungs.append((*RUNG_DEFS[0], analysis))

    r1 = analysis.groupby(
        [
            "title_key",
            "Dataset",
            "Clinical target",
            "Non-facial modalities",
            "Facial modality",
        ],
        as_index=False,
    ).agg(**{metric_col: (metric_col, "median")})
    rungs.append((*RUNG_DEFS[1], r1))

    r2 = analysis.groupby(["title_key"], as_index=False).agg(
        **{metric_col: (metric_col, "median")}
    )
    rungs.append((*RUNG_DEFS[2], r2))

    return rungs


# ── Figure ───────────────────────────────────────────────────────────────────

BANDS = BANDS_PCT
POINT_COLOUR = "#2B6CB0"


def make_rung_figure(analysis, out_dir, metric_col, out_name, xlabel):
    """Build a 3-panel sensitivity figure, one row per aggregation rung."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rungs = collapse_to_rungs(analysis, metric_col)
    n_rungs = len(rungs)
    fig, axes = plt.subplots(
        n_rungs, 1, figsize=(10.0, 1.9 * n_rungs), sharex=True
    )
    if n_rungs == 1:
        axes = [axes]

    all_vals = pd.concat(
        [r[2][metric_col] for r in rungs], ignore_index=True
    )
    xmin = min(all_vals.min(), -20.0)
    xmax = max(all_vals.max(), 20.0)
    pad = (xmax - xmin) * 0.04
    xleft, xright = xmin - pad, xmax + pad

    np.random.seed(42)

    for panel_idx, ((label, unit, rung_df), ax) in enumerate(zip(rungs, axes)):
        n_total = len(rung_df)

        for (lo, hi), colour in zip(BANDS, BAND_COLOURS):
            left = max(lo, xleft)
            right = min(hi, xright)
            if left < right:
                ax.axvspan(
                    left, right, color=colour, alpha=0.55, zorder=0,
                    linewidth=0,
                )

        # Band edges (pre-registered ±1/5/15 % thresholds)
        for thresh in [-15, -5, -1, 1, 5, 15]:
            if xleft <= thresh <= xright:
                ax.axvline(
                    thresh, color="#999999", linewidth=0.3, linestyle=":",
                    zorder=1,
                )

        ax.axvline(0, color="black", linewidth=1.2, zorder=2)

        vals = rung_df[metric_col]
        jitter = np.random.uniform(-0.32, 0.32, size=n_total)
        ax.scatter(
            vals,
            jitter,
            marker="o",
            s=22,
            alpha=0.55,
            edgecolors=POINT_COLOUR,
            facecolors=POINT_COLOUR,
            linewidths=0.3,
            zorder=3,
        )

        med = vals.median()
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        ax.plot(
            [q1, q3],
            [0, 0],
            color=POINT_COLOUR,
            linewidth=2.5,
            solid_capstyle="round",
            alpha=0.85,
            zorder=5,
        )
        ax.scatter(
            [med],
            [0],
            marker="D",
            s=40,
            color="white",
            edgecolors=POINT_COLOUR,
            linewidths=1.3,
            zorder=6,
        )
        # Dark, high-contrast text on a semi-transparent white plate so the
        # readout stays legible over the dark "large positive" band (never
        # coloured — blue/green wash out against the background).
        ax.text(
            0.985,
            0.90,
            f"n={n_total}   Mdn={med:.1f}%   IQR=[{q1:.1f}, {q3:.1f}]",
            transform=ax.transAxes,
            fontsize=11.5,
            color="#1a1a1a",
            va="top",
            ha="right",
            zorder=8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#cccccc",
                linewidth=0.5,
                alpha=0.82,
            ),
        )

        ax.set_xlim(xleft, xright)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(
            f"{label}  (N = {n_total} {unit})",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        # Panel label (a, b, c) required by the journal; sits just left of
        # the left-aligned panel title.
        ax.text(
            -0.012,
            1.0,
            "abcdefghij"[panel_idx],
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    axes[-1].set_xlabel(xlabel, fontsize=13)
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(10))
    axes[-1].xaxis.set_minor_locator(mticker.MultipleLocator(2))
    axes[-1].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:g}%")
    )

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="D",
            color="white",
            markeredgecolor="grey",
            markersize=5,
            linewidth=0,
            linestyle="None",
            label="Median",
        ),
        Line2D(
            [],
            [],
            color="grey",
            linewidth=2.5,
            solid_capstyle="round",
            label="IQR",
        ),
    ]
    fig.tight_layout(pad=0.8)
    fig.subplots_adjust(bottom=0.14)

    # Figure-level legend, top-right above the first panel (its title is
    # left-aligned, so this space is free and no data points are covered).
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 1.005),
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.0,
        ncol=2,
    )

    from matplotlib.patches import FancyBboxPatch

    BAND_KEY = list(zip(BAND_LABELS_SHORT, BAND_COLOURS))
    n_bands = len(BAND_KEY)
    swatch_w = 0.12
    swatch_h = 0.022
    gap = 0.02
    total_w = n_bands * swatch_w + (n_bands - 1) * gap
    x_start = 0.5 - total_w / 2
    swatch_y = 0.005

    for i, (blabel, colour) in enumerate(BAND_KEY):
        x = x_start + i * (swatch_w + gap)
        rect = FancyBboxPatch(
            (x, swatch_y),
            swatch_w,
            swatch_h,
            boxstyle="round,pad=0.002",
            facecolor=colour,
            edgecolor="#bbbbbb",
            linewidth=0.4,
            alpha=0.75,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.patches.append(rect)
        fig.text(
            x + swatch_w / 2,
            swatch_y + swatch_h / 2,
            blabel,
            ha="center",
            va="center",
            fontsize=8,
            color="#444444",
            transform=fig.transFigure,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / out_name,
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved: {out_dir / out_name}", file=sys.stderr)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    out_dir = project_dir / "figures"

    print("Loading data...", file=sys.stderr)
    data, _norm_table = load_extraction_data(project_dir)

    for spec in FIGURE_SPECS:
        analysis = load_analysis(data, spec.metric_col)
        print(
            f"[{spec.metric_col}]  "
            f"Comparisons: {len(analysis)}  |  "
            f"Studies: {analysis['title_key'].nunique()}",
            file=sys.stderr,
        )
        print(
            f"Rendering multi-rung sensitivity figure -> {spec.out_name}...",
            file=sys.stderr,
        )
        make_rung_figure(
            analysis, out_dir, spec.metric_col, spec.out_name, spec.xlabel
        )

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
