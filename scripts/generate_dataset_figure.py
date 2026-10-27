#!/usr/bin/env python3
"""
Generate beeswarm plots of the relative score gain stratified by dataset.

Primary metric: Δ_face = Z / μ_cell × 100 (``pct_of_baseline``, manuscript
Eq. 3), matching the main synthesis figure. Not the pre-registered
secondary Δ% (``marginal_pct``, Eq. 4).

Output (per model–dataset collapse — one median per model–dataset pair):
    figures/dataset_beeswarm.png

Usage:
    python scripts/generate_dataset_figure.py
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
    """Collapse to one value per model–dataset pair (median across metrics)."""
    return (
        df.groupby(
            [
                "title_key",
                "Clinical target",
                "Non-facial modalities",
                "Facial modality",
                group_col,
            ],
            as_index=False,
        ).agg(**{_METRIC: (_METRIC, "median")})
    )


# ── Visual constants ─────────────────────────────────────────────────────────

BANDS = BANDS_PCT
BAND_KEY = list(zip(BAND_LABELS_SHORT, BAND_COLOURS))

DATASET_COLOURS = [
    "#2563EB",
    "#DC2626",
    "#059669",
    "#D97706",
    "#7C3AED",
    "#DB2777",
    "#0891B2",
    "#6366F1",
]
MUTED_COLOUR = "#999999"
PROPRIETARY_COLOUR = "#6B7280"
IQR_MIN = 3

PROPRIETARY_LABEL = "Proprietary / single-cohort"


# ── Figure rendering ─────────────────────────────────────────────────────────


def _make_figure(
    df, ds_order, title, filename, out_dir, xleft, xright,
    proprietary_idx, is_collapsed=False,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    n_datasets = len(ds_order)
    fig, ax = plt.subplots(figsize=(10.0, 0.9 * n_datasets + 0.8))

    # Magnitude bands
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

    # Faint divider lines between rows; each dataset's readout sits on the
    # divider just above its cloud, clear of the points. The proprietary
    # separator (drawn below) supplies its own line, so skip it here.
    for i in range(len(ds_order) - 1):
        if proprietary_idx is not None and i == proprietary_idx:
            continue
        ax.axhline(i + 0.5, color="#d9d9d9", linewidth=0.8, zorder=1.5)

    if proprietary_idx is not None:
        sep_y = proprietary_idx + 0.5
        ax.axhline(
            sep_y, color="#888888", linewidth=0.8, linestyle="--", zorder=1
        )

    y_positions = {ds: i for i, ds in enumerate(ds_order)}
    np.random.seed(42)

    colour_idx = 0

    for ds in ds_order:
        sub = df[df["Dataset"] == ds]
        y_base = y_positions[ds]
        n = len(sub)

        n_pairs = (
            n
            if is_collapsed
            else sub.groupby(["title_key", "Dataset"]).ngroups
        )
        is_proprietary = ds == PROPRIETARY_LABEL
        muted = n_pairs < IQR_MIN

        if is_proprietary:
            colour = PROPRIETARY_COLOUR
        elif muted:
            colour = MUTED_COLOUR
        else:
            colour = DATASET_COLOURS[colour_idx % len(DATASET_COLOURS)]

        if not is_proprietary:
            colour_idx += 1

        jitter = np.random.uniform(-0.30, 0.30, size=n)
        ax.scatter(
            sub[_METRIC],
            y_base + jitter,
            marker="o",
            s=18,
            alpha=0.30 if (muted or is_proprietary) else 0.50,
            edgecolors=colour,
            facecolors=colour,
            linewidths=0.3,
            zorder=3,
        )

        vals = sub[_METRIC]
        med = vals.median()
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)

        show_iqr = (not muted) and (not is_proprietary)
        if show_iqr:
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

        if is_proprietary:
            ann = f"n={n} pairs   Mdn={med:.2f}   (heterogeneous)"
        elif is_collapsed:
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

        # Dark, high-contrast text on a semi-transparent white plate, seated on
        # the divider above the cloud (never coloured — washes out on the bands).
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
    # Small top headroom for the top dataset's readout (sits below the title).
    ax.set_ylim(-0.55, n_datasets - 0.22)
    ax.set_yticks(list(range(n_datasets)))

    labels = list(ds_order)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    if proprietary_idx is not None:
        ax.get_yticklabels()[proprietary_idx].set_style("italic")
        ax.get_yticklabels()[proprietary_idx].set_fontweight("normal")

    ax.set_xlabel(
        "Relative gain of facial modality",
        fontsize=13,
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=6)

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

    fig.subplots_adjust(bottom=0.12)

    n_bands = len(BAND_KEY)
    swatch_w = 0.12
    swatch_h = 0.022
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
    analysis = data.dropna(subset=[_METRIC]).copy()

    analysis["Dataset"] = (
        analysis["Dataset"].str.strip().replace("Proprietary", PROPRIETARY_LABEL)
    )

    ds_counts = analysis.groupby("Dataset")["title_key"].nunique()
    for ds in sorted(ds_counts.index):
        n_st = ds_counts[ds]
        n_comp = (analysis["Dataset"] == ds).sum()
        print(
            f"  {ds}: {n_st} studies, {n_comp} comparisons", file=sys.stderr
        )

    # Shared x bounds
    xmin = min(analysis[_METRIC].min(), -20.0)
    xmax = max(analysis[_METRIC].max(), 20.0)
    pad = (xmax - xmin) * 0.04
    xleft, xright = xmin - pad, xmax + pad

    collapsed = _collapse_per_model_dataset(analysis, "Dataset")

    nonprop = collapsed[collapsed["Dataset"] != PROPRIETARY_LABEL]
    nonprop_order = (
        nonprop.groupby("Dataset")[_METRIC]
        .median()
        .sort_values()
        .index.tolist()
    )
    nonprop_order = [
        d for d in nonprop_order if (analysis["Dataset"] == d).any()
    ]

    has_proprietary = (analysis["Dataset"] == PROPRIETARY_LABEL).any()
    if has_proprietary:
        ds_order = [PROPRIETARY_LABEL] + nonprop_order
        proprietary_idx = 0
    else:
        ds_order = nonprop_order
        proprietary_idx = None

    print(f"  Dataset order: {ds_order}", file=sys.stderr)

    _make_figure(
        collapsed,
        ds_order,
        "Dataset (per model\u2013dataset median)",
        "dataset_beeswarm.png",
        out_dir,
        xleft,
        xright,
        proprietary_idx,
        is_collapsed=True,
    )

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
