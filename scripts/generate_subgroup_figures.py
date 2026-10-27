#!/usr/bin/env python3
"""
Generate separate beeswarm figures for the pre-registered subgroup
analyses (outcome type, baseline modalities).

Primary metric: Δ_face = Z / μ_cell × 100 (``pct_of_baseline``, manuscript
Eq. 3), matching the main synthesis figure. Not the pre-registered
secondary Δ% (``marginal_pct``, Eq. 4).

Outputs (per model–dataset collapse — one median per model–dataset pair):
    figures/outcome_type_beeswarm.png
    figures/baseline_modalities_beeswarm.png

Usage:
    python scripts/generate_subgroup_figures.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from marginal_metric import (
    BANDS_PCT,
    BAND_COLOURS,
    BAND_LABELS_SHORT,
    load_extraction_data,
)

_METRIC = "pct_of_baseline"

# ── Collapse ─────────────────────────────────────────────────────────────────


def _collapse_per_model_dataset(df, group_col):
    """Collapse to one value per model–dataset pair within each moderator level."""
    return (
        df.groupby(
            [
                "title_key",
                "Dataset",
                "Clinical target",
                "Non-facial modalities",
                "Facial modality",
                group_col,
            ],
            as_index=False,
        ).agg(**{_METRIC: (_METRIC, "median")})
    )


# ── Shared visual constants ──────────────────────────────────────────────────

BANDS = BANDS_PCT
BAND_KEY = list(zip(BAND_LABELS_SHORT, BAND_COLOURS))

SUBGROUP_COLOURS = ["#2B6CB0", "#C53030", "#2F855A", "#B7791F"]
MUTED_COLOUR = "#999999"
IQR_MIN = 3


# ── Panel rendering ──────────────────────────────────────────────────────────


def _draw_panel(
    ax, df, group_col, group_order, xleft, xright, title,
    is_collapsed=False,
):
    for (lo, hi), colour in zip(BANDS, BAND_COLOURS):
        left, right = max(lo, xleft), min(hi, xright)
        if left < right:
            ax.axvspan(
                left, right, color=colour, alpha=0.55, zorder=0, linewidth=0
            )
    # Pre-registered ±1/5/15 % band edges (data are in percent-of-baseline)
    for thresh in [-15, -5, -1, 1, 5, 15]:
        if xleft <= thresh <= xright:
            ax.axvline(
                thresh, color="#999999", linewidth=0.3, linestyle=":", zorder=1
            )
    ax.axvline(0, color="black", linewidth=1.2, zorder=2)

    y_positions = {g: i for i, g in enumerate(group_order)}

    # Faint divider lines *between* rows only (not above the top row, which
    # would read as an empty pane). Each group's readout sits on the divider
    # just above its cloud; the top group's sits in a small headroom below the
    # title (see set_ylim).
    for i in range(len(group_order) - 1):
        ax.axhline(i + 0.5, color="#d9d9d9", linewidth=0.8, zorder=1.5)

    np.random.seed(42)

    for group in group_order:
        sub = df[df[group_col] == group]
        y_base = y_positions[group]
        n = len(sub)

        n_pairs = (
            n
            if is_collapsed
            else sub.groupby(["title_key", "Dataset"]).ngroups
        )
        muted = n_pairs < IQR_MIN

        colour = (
            MUTED_COLOUR
            if muted
            else SUBGROUP_COLOURS[y_base % len(SUBGROUP_COLOURS)]
        )

        jitter = np.random.uniform(-0.30, 0.30, size=n)
        ax.scatter(
            sub[_METRIC],
            y_base + jitter,
            marker="o",
            s=18,
            alpha=0.35 if muted else 0.50,
            edgecolors=colour,
            facecolors=colour,
            linewidths=0.3,
            zorder=3,
        )

        vals = sub[_METRIC]
        med = vals.median()
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)

        if not muted:
            ax.plot(
                [q1, q3],
                [y_base, y_base],
                color=colour,
                linewidth=2.5,
                solid_capstyle="round",
                alpha=0.85,
                zorder=5,
            )

        ax.scatter(
            [med],
            [y_base],
            marker="D",
            s=40,
            color="white",
            edgecolors=colour,
            linewidths=1.3,
            zorder=6,
        )

        if is_collapsed:
            ann = (
                f"n={n} pairs   Mdn={med:.2f}   "
                + (
                    f"IQR=[{q1:.2f}, {q3:.2f}]"
                    if not muted
                    else "(descriptive only)"
                )
            )
        else:
            ann = (
                f"n={n} ({n_pairs} pairs)   Mdn={med:.2f}   "
                + (
                    f"IQR=[{q1:.2f}, {q3:.2f}]"
                    if not muted
                    else "(descriptive only)"
                )
            )

        # Dark, high-contrast text on a semi-transparent white plate so the
        # median/IQR readout stays legible over the dark "large positive"
        # band. Never coloured (blue/green wash out against the background).
        ax.text(
            0.985,
            y_base + 0.5,
            ann,
            transform=ax.get_yaxis_transform(),
            fontsize=11.5,
            color="#1a1a1a",
            va="center",
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
    # Just enough top headroom for the top group's readout to sit below the
    # title (no full empty row above the top cloud).
    ax.set_ylim(-0.50, len(group_order) - 0.22)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(group_order, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=6)


# ── Single-figure builder ────────────────────────────────────────────────────


def _make_single_figure(
    df, group_col, group_order, title, filename, out_dir, xleft, xright,
    is_collapsed=False,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    n_groups = len(group_order)
    fig, ax = plt.subplots(figsize=(10.0, 1.6 * n_groups + 0.8))

    _draw_panel(
        ax, df, group_col, group_order, xleft, xright, title,
        is_collapsed=is_collapsed,
    )

    ax.set_xlabel(
        "Relative gain of facial modality",
        fontsize=13,
    )

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
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.0,
        ncol=2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=0.6)

    fig.subplots_adjust(bottom=0.22)

    n_bands = len(BAND_KEY)
    swatch_w = 0.12
    swatch_h = 0.025
    gap = 0.02
    total_w = n_bands * swatch_w + (n_bands - 1) * gap
    x_start = 0.5 - total_w / 2
    swatch_y = 0.01

    for i, (label, colour) in enumerate(BAND_KEY):
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
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="#444444",
            transform=fig.transFigure,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / filename, bbox_inches="tight", dpi=300, facecolor="white"
    )
    plt.close(fig)
    print(f"Saved: {out_dir / filename}", file=sys.stderr)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    out_dir = project_dir / "figures"

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

    print("Loading data...", file=sys.stderr)
    data, _norm = load_extraction_data(project_dir)
    analysis = data.dropna(subset=[_METRIC, "metric_family"]).copy()

    # ── Subgroup 1: Outcome type ─────────────────────────────────────────
    analysis["outcome_group"] = (
        analysis["Clinical target"]
        .str.strip()
        .apply(lambda x: "Depression" if x == "Depression" else "Other")
    )

    # ── Subgroup 2: Facial feature extraction method ─────────────────────
    _face_map = {
        "OpenFace": "AU-based (OpenFace)",
        "Facial frames": "Pixel-based (frames)",
        "Facial landmarks": "Landmarks",
        "Facial expression": "Other",
        "Facial action units (Proprietary)": "AU-based (OpenFace)",
    }
    analysis["facial_method_group"] = (
        analysis["Facial modality"].str.strip().map(_face_map).fillna("Other")
    )

    # ── Subgroup 3: Baseline modalities ──────────────────────────────────
    def _baseline_group(cfg):
        cfg = str(cfg).strip()
        if cfg in ("A",):
            return "Audio only"
        if cfg in ("A+T", "T+A"):
            return "Audio + Text"
        return "Other"

    analysis["baseline_group"] = (
        analysis["Baseline config"].str.strip().apply(_baseline_group)
    )

    for col, label in [
        ("outcome_group", "Outcome"),
        ("baseline_group", "Baseline"),
    ]:
        counts = analysis.groupby(col)["title_key"].nunique()
        print(f"  {label}: {dict(counts)}", file=sys.stderr)

    # Shared x bounds
    xmin = min(analysis[_METRIC].min(), -20.0)
    xmax = max(analysis[_METRIC].max(), 20.0)
    pad = (xmax - xmin) * 0.04
    xleft, xright = xmin - pad, xmax + pad

    # ── Outcome type ─────────────────────────────────────────────────────
    collapsed_outcome = _collapse_per_model_dataset(analysis, "outcome_group")
    outcome_order = (
        collapsed_outcome.groupby("outcome_group")[_METRIC]
        .median()
        .sort_values()
        .index.tolist()
    )
    outcome_order = [
        g for g in outcome_order if (analysis["outcome_group"] == g).any()
    ]
    print(f"  Outcome order (by median): {outcome_order}", file=sys.stderr)

    _make_single_figure(
        collapsed_outcome,
        "outcome_group",
        outcome_order,
        "Outcome type (per model\u2013dataset median)",
        "outcome_type_beeswarm.png",
        out_dir,
        xleft,
        xright,
        is_collapsed=True,
    )

    # ── Baseline modalities ──────────────────────────────────────────────
    baseline_order = ["Audio only", "Audio + Text", "Other"]
    baseline_order = [
        g for g in baseline_order if (analysis["baseline_group"] == g).any()
    ]

    collapsed_baseline = _collapse_per_model_dataset(
        analysis, "baseline_group"
    )

    _make_single_figure(
        collapsed_baseline,
        "baseline_group",
        baseline_order,
        None,  # no in-figure title; the LaTeX caption describes the figure
        "baseline_modalities_beeswarm.png",
        out_dir,
        xleft,
        xright,
        is_collapsed=True,
    )

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
