#!/usr/bin/env python3
"""
Generate the LaTeX table for the relative-score-gain analysis,
segmented by Risk of Bias (RoB) and Applicability Concern.

A single table is produced: rungs 0--2 of the aggregation ladder,
each with an unstratified "All" row plus RoB and Applicability strata.
(The previously separate unstratified-rungs and overall-stratified
tables were strict subsets of this one and were removed.)

Outputs:
    tables/marginal_rungs_stratified.tex
    tables/normalization_table.csv

Usage:
    python scripts/generate_marginal_tables.py
"""

import sys
from pathlib import Path

import pandas as pd

from marginal_metric import (
    DATA_DIR,
    load_extraction_data,
    title_key,
    write_normalization_table,
)

# ── Data pipeline ────────────────────────────────────────────────────────────


def load_analysis(project_dir):
    """Return the analysis dataframe with the primary metric, RoB, and
    applicability."""
    data, norm_table = load_extraction_data(project_dir)

    prob_dev = pd.read_excel(
        project_dir / DATA_DIR / "PROBAST_assessment.xlsx", sheet_name="Model Development"
    )
    prob_eval = pd.read_excel(
        project_dir / DATA_DIR / "PROBAST_assessment.xlsx", sheet_name="Model Evaluation"
    )

    for df in (prob_dev, prob_eval):
        df["title_key"] = df["Title"].map(title_key)
        df["year_key"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    dev_k = prob_dev[
        [
            "title_key",
            "year_key",
            "Overall Quality Concern",
            "Overall Applicability Concern",
        ]
    ].drop_duplicates()
    eval_k = prob_eval[
        [
            "title_key",
            "year_key",
            "Overall RoB",
            "Overall Applicability Concern",
        ]
    ].drop_duplicates()

    merged = data.merge(dev_k, on=["title_key", "year_key"], how="left").merge(
        eval_k,
        on=["title_key", "year_key"],
        how="left",
        suffixes=("_dev", "_eval"),
    )

    def _worst(a, b):
        if a == "High" or b == "High":
            return "High"
        if a == "Low" or b == "Low":
            return "Low"
        return a

    merged["rob"] = merged.apply(
        lambda r: _worst(r["Overall Quality Concern"], r["Overall RoB"]),
        axis=1,
    )
    merged["applicability_concern"] = merged.apply(
        lambda r: _worst(
            r["Overall Applicability Concern_dev"],
            r["Overall Applicability Concern_eval"],
        ),
        axis=1,
    )

    _unmatched = merged[
        merged["rob"].isna() | merged["applicability_concern"].isna()
    ]
    if not _unmatched.empty:
        print(
            f"WARNING: {len(_unmatched)} rows with unmatched PROBAST data:",
            file=sys.stderr,
        )
        for _, _r in (
            _unmatched[["Title", "Year"]].drop_duplicates().iterrows()
        ):
            print(f"  - {_r['Title']} ({_r['Year']})", file=sys.stderr)

    # Keep rows that have a valid primary metric and metric_family
    analysis = merged.dropna(subset=[_METRIC, "metric_family"]).copy()
    return analysis, norm_table


# ── Aggregation ──────────────────────────────────────────────────────────────

_METRIC = "pct_of_baseline"


# ── Rung collapsing ─────────────────────────────────────────────────────────

RUNG_DEFS = [
    ("All comparisons", "comparisons"),
    ("Per model–dataset", "pairs"),
    ("Per study", "studies"),
]


def collapse_to_rungs(analysis):
    """Return list of (label, unit_label, DataFrame) for rungs 0–2."""
    rungs = []
    rungs.append((*RUNG_DEFS[0], analysis))

    r1 = (
        analysis.groupby(
            [
                "title_key",
                "Dataset",
                "Clinical target",
                "Non-facial modalities",
                "Facial modality",
                "rob",
                "applicability_concern",
            ],
            as_index=False,
        ).agg(**{_METRIC: (_METRIC, "median")})
    )
    rungs.append((*RUNG_DEFS[1], r1))

    r2 = (
        analysis.groupby(
            ["title_key", "rob", "applicability_concern"], as_index=False
        ).agg(**{_METRIC: (_METRIC, "median")})
    )
    rungs.append((*RUNG_DEFS[2], r2))

    return rungs


# ── LaTeX rendering ──────────────────────────────────────────────────────────


def _fmt(val, decimals=2):
    return f"{val:.{decimals}f}"


def _agg_simple(df, group_col):
    return (
        df.groupby(group_col)
        .agg(
            n=(_METRIC, "size"),
            median_gain=(_METRIC, "median"),
            q1=(_METRIC, lambda s: s.quantile(0.25)),
            q3=(_METRIC, lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )


def _build_rung_stratified_table(analysis):
    """LaTeX table: rungs x PROBAST strata."""
    rungs = collapse_to_rungs(analysis)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Relative score gain of the facial modality at each "
        r"aggregation rung, stratified by PROBAST+AI assessment.}"
        r"\label{tab:marginal_rungs_stratified}"
    )
    lines.append(r"\begin{tabular}{l l l r r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Aggregation} & \textbf{Stratification} & \textbf{Level} "
        r"& \textbf{N} "
        r"& \textbf{Median} & \textbf{Q1} & \textbf{Q3} \\"
    )
    lines.append(r"\midrule")

    for rung_idx, (label, _unit, rung_df) in enumerate(rungs):
        label_tex = label.replace("–", "--")
        n = len(rung_df)
        med = rung_df[_METRIC].median()
        q1 = rung_df[_METRIC].quantile(0.25)
        q3 = rung_df[_METRIC].quantile(0.75)
        lines.append(
            f"{label_tex} & All & --- & {n} "
            f"& {_fmt(med)} & {_fmt(q1)} & {_fmt(q3)} \\\\"
        )

        by_rob = _agg_simple(rung_df, "rob")
        for level in ["High", "Low"]:
            row = by_rob[by_rob["rob"] == level]
            if row.empty:
                continue
            r = row.iloc[0]
            strat_label = "Risk of Bias" if level == "High" else ""
            lines.append(
                f" & {strat_label} & {level} & {int(r['n'])} "
                f"& {_fmt(r['median_gain'])} & {_fmt(r['q1'])} & {_fmt(r['q3'])} \\\\"
            )

        by_appl = _agg_simple(rung_df, "applicability_concern")
        for level in ["High", "Low"]:
            row = by_appl[by_appl["applicability_concern"] == level]
            if row.empty:
                continue
            r = row.iloc[0]
            strat_label = "Applicability" if level == "High" else ""
            lines.append(
                f" & {strat_label} & {level} & {int(r['n'])} "
                f"& {_fmt(r['median_gain'])} & {_fmt(r['q1'])} & {_fmt(r['q3'])} \\\\"
            )

        if rung_idx < len(rungs) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    out_dir = project_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...", file=sys.stderr)
    analysis, norm_table = load_analysis(project_dir)
    print(
        f"  {len(analysis)} comparisons (metric-included), "
        f"{analysis['title_key'].nunique()} studies",
        file=sys.stderr,
    )

    # Export normalisation table
    write_normalization_table(norm_table, out_dir / "normalization_table.csv")
    print(f"Saved: {out_dir / 'normalization_table.csv'}", file=sys.stderr)

    tex_strat = _build_rung_stratified_table(analysis)
    (out_dir / "marginal_rungs_stratified.tex").write_text(
        tex_strat, encoding="utf-8"
    )
    print(
        f"Saved: {out_dir / 'marginal_rungs_stratified.tex'}", file=sys.stderr
    )

    print("\n" + "=" * 80, file=sys.stderr)
    print(tex_strat)
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
