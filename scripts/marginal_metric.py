#!/usr/bin/env python3
"""
Single source of truth for the relative-score-gain metric used in the
systematic review of facial modality in multimodal mental-health ML.

Primary metric (manuscript Eqs. 1–3)
    Z      = s · (P_with − P_without)      signed performance gain   (Eq. 1)
    μ_cell = mean(P_without) over the cell  mean baseline performance (Eq. 2)
    Δ_face = Z / μ_cell × 100              relative gain, in %        (Eq. 3)
    s = +1 higher-is-better, −1 lower-is-better
    Cells with n < 5 paired comparisons are excluded.

Auxiliary (not reported in the manuscript)
    κ_cell = sqrt(mean(Z²)) — root-mean-square gain, and Δz = Z / κ_cell.
    Kept only for the κ column of normalization_table.csv and for
    generate_metric_sensitivity_figure.py. Δ_face does NOT depend on κ.

Secondary metric (manuscript Eq. 4, pre-registered)
    Δ% = s · (P_with − P_without) / P_without × 100

Normalisation cells
    unit-bearing metrics (RMSE, MAE): metric × clinical instrument
    dimensionless metrics (everything else): metric pooled globally

Exposes
-------
LOWER_IS_BETTER           set of normalised metric names where ↓ = better
DIRECTION / FAMILY        legacy dicts (exact-cased keys from XLSX)
build_normalization_table  μ (mean baseline) and κ (RMS gain) per cell
delta_face                 primary row-level metric (Eq. 3)
delta_percent              secondary row-level metric (Eq. 4)
delta_z / to_raw_units     auxiliary κ-scale helpers (not reported)
write_normalization_table  CSV export for OSF
load_extraction_data       convenience loader
prepare_extraction_data    add all derived columns to a raw DataFrame
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Folder (relative to the repository root) holding the source workbooks.
DATA_DIR = "csv_files"

# ═══════════════════════════════════════════════════════════════════════════════
# Lower-is-better set (canonical, normalised to lowercase)
# ═══════════════════════════════════════════════════════════════════════════════

LOWER_IS_BETTER: set = {"rmse", "mae"}


def _norm_metric(name):
    """Strip whitespace + lowercase for case-insensitive metric matching."""
    return str(name).strip().lower()


def is_lower_better(metric):
    """True when a decrease in *metric* means improvement."""
    return _norm_metric(metric) in LOWER_IS_BETTER


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy direction / family dicts (exact-cased keys from data_extraction.xlsx)
# ═══════════════════════════════════════════════════════════════════════════════

DIRECTION = {
    "AUC": "higher_better",
    "Accuracy": "higher_better",
    "Average": "higher_better",
    "Average (Acc, Prec, Recall, F1)": "higher_better",  # legacy alias of "Average"
    "Correlation": "higher_better",
    "F1-Score": "higher_better",
    "F1-score": "higher_better",
    "MAE": "lower_better",
    "RMSE": "lower_better",
    "Recall": "higher_better",
    "Unweighted Average Recall": "higher_better",  # legacy alias of "Recall"
}

FAMILY = {
    "AUC": "classification",
    "Accuracy": "classification",
    "Average": "classification",
    "Average (Acc, Prec, Recall, F1)": "classification",  # legacy alias of "Average"
    "Correlation": "regression",
    "F1-Score": "classification",
    "F1-score": "classification",
    "MAE": "regression",
    "RMSE": "regression",
    "Recall": "classification",
    "Unweighted Average Recall": "classification",  # legacy alias of "Recall"
}

FAMILY_DISPLAY = {"classification": "Classification", "regression": "Regression"}

# ═══════════════════════════════════════════════════════════════════════════════
# Unit-bearing metrics & dataset → clinical instrument
# ═══════════════════════════════════════════════════════════════════════════════

_UNIT_BEARING = {"rmse", "mae"}

DATASET_INSTRUMENT = {
    "DAIC-WOZ": "PHQ-8",
    "E-DAIC": "PHQ-8",
    "AViD (AVEC 2013)": "BDI-II",
    "AViD (AVEC 2014)": "BDI-II",
    "CMDeep": "BDI-II",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Magnitude-band definitions (shared across figures)
# ═══════════════════════════════════════════════════════════════════════════════

# Δz bands. NOTE: Δz is now expressed in units of κ (the cell's
# root-mean-square gain), so these cut-points are descriptive reference
# lines, NOT Cohen's benchmarks — those are defined against a standard
# deviation and do not transfer to an uncentred normaliser. On the κ
# scale the natural anchor is |Δz| = 1, i.e. a gain of typical size for
# that cell.
BANDS_Z = [
    (-np.inf, -0.8),
    (-0.8, -0.5),
    (-0.5, -0.2),
    (-0.2, 0.2),
    (0.2, 0.5),
    (0.5, 0.8),
    (0.8, np.inf),
]

# Legacy Δ% bands — pre-registered thresholds
BANDS_PCT = [
    (-np.inf, -15),
    (-15, -5),
    (-5, -1),
    (-1, 1),
    (1, 5),
    (5, 15),
    (15, np.inf),
]

BAND_COLOURS = [
    "#F4CCCC",  # large neg
    "#FCE5CD",  # moderate neg
    "#FFF2CC",  # small neg
    "#EFEFEF",  # negligible
    "#D9EAD3",  # small pos
    "#B6D7A8",  # moderate pos
    "#93C47D",  # large pos
]

BAND_LABELS_SHORT = [
    "Large neg.",
    "Moderate neg.",
    "Small neg.",
    "Negligible",
    "Small pos.",
    "Moderate pos.",
    "Large pos.",
]

BAND_KEY_Z = list(zip(BAND_LABELS_SHORT, BAND_COLOURS))
BAND_KEY_PCT = list(zip(BAND_LABELS_SHORT, BAND_COLOURS))

# ═══════════════════════════════════════════════════════════════════════════════
# Text helpers (de-duplicated from scripts)
# ═══════════════════════════════════════════════════════════════════════════════


def to_num(s):
    """Column → numeric, handling European comma decimals."""
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )


_TYPO = str.maketrans(
    {
        " ": " ",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "‘": "'",
        "’": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
    }
)


def title_key(v):
    """Normalise a study title for deduplication and matching."""
    s = unicodedata.normalize("NFKC", str(v)).translate(_TYPO).lower().strip()
    s = re.sub(r"\.+$", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Normalisation table
# ═══════════════════════════════════════════════════════════════════════════════


# Clinical scales usable as normalisation instruments for unit-bearing
# metrics. Anything else in the workbook's Instrument column (diagnostic
# labels, "Unverifiable", ...) does not define a measurement scale.
_SCALE_CANON = {
    "phq-8": "PHQ-8",
    "phq-9": "PHQ-9",
    "bdi-ii": "BDI-II",
    "hamd": "HAMD",
    "ham-d": "HAMD",
    "mmse": "MMSE",
    "casi": "CASI",
}


def _canon_instrument(value):
    """Map a workbook Instrument value to a canonical scale name, or None."""
    if value is None:
        return None
    return _SCALE_CANON.get(str(value).strip().lower())


def _cell_for_row(metric_norm, dataset, instrument=None):
    """Return ``(cell_id, instrument)`` for one row; ``(None, None)`` when
    the instrument cannot be determined (unit-bearing metric with no
    recognised scale). The workbook's per-row Instrument column takes
    precedence; DATASET_INSTRUMENT is the fallback."""
    dataset = str(dataset).strip()
    if metric_norm in _UNIT_BEARING:
        inst = _canon_instrument(instrument) or DATASET_INSTRUMENT.get(dataset)
        if inst is None:
            return None, None
        return f"{metric_norm}|{inst}", inst
    return f"{metric_norm}|global", "global"


def build_normalization_table(df):
    """Build the normalisation reference for each cell.

    Two quantities are computed per cell, both over the *paired*
    comparisons (rows where both ``baseline`` and ``with_face`` are
    present):

    ``mu``
        Mean baseline performance — the average performance achieved
        without the facial modality (manuscript Eq. 2).
    ``kappa``
        Root-mean-square signed gain, sqrt(mean(Z²)) with
        Z = s·(with_face − baseline). Auxiliary: not used by the primary
        metric and not reported in the manuscript. This is an
        UNCENTRED second moment about the origin Z = 0, not a standard
        deviation: no mean-centring is applied, because Z = 0 is the
        substantively meaningful reference (the facial modality adding
        nothing). Since Z² is symmetric in sign, the direction factor s
        does not affect kappa, but it is applied for clarity.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``Metric``, ``Dataset``, ``baseline``,
        ``with_face``.

    Returns
    -------
    DataFrame with columns ``cell_id  metric  instrument  n  mu  kappa``.
    """
    cols = ["cell_id", "metric", "instrument", "n", "mu", "kappa"]
    if len(df) == 0:
        return pd.DataFrame(columns=cols)

    mn = df["Metric"].apply(_norm_metric)
    inst_col = (
        df["Instrument"] if "Instrument" in df.columns else [None] * len(df)
    )
    cids, insts = zip(
        *(
            _cell_for_row(m, d, i)
            for m, d, i in zip(
                mn, df["Dataset"].astype(str).str.strip(), inst_col
            )
        )
    )

    tmp = df[["baseline", "with_face"]].copy()
    tmp["_cid"] = list(cids)
    tmp["_inst"] = list(insts)
    tmp["_mn"] = mn.values
    tmp["_s"] = np.where(mn.isin(LOWER_IS_BETTER).values, -1, 1)
    # Paired comparisons only: both values are required for Z, and mu is
    # taken over the same rows so that the two cell statistics describe
    # one and the same set of comparisons.
    tmp = tmp.dropna(subset=["_cid", "baseline", "with_face"])
    tmp["_z"] = tmp["_s"] * (tmp["with_face"] - tmp["baseline"])

    records = []
    for cid, grp in tmp.groupby("_cid"):
        records.append(
            {
                "cell_id": cid,
                "metric": grp["_mn"].iloc[0],
                "instrument": grp["_inst"].iloc[0],
                "n": len(grp),
                "mu": grp["baseline"].mean(),
                "kappa": float(np.sqrt((grp["_z"] ** 2).mean())),
            }
        )
    return pd.DataFrame(records, columns=cols)


# ═══════════════════════════════════════════════════════════════════════════════
# Primary metric: Δ_face (manuscript Eq. 3)
# ═══════════════════════════════════════════════════════════════════════════════


def delta_face(row, norm_table):
    r"""Δ_face = Z / μ_cell × 100, with Z = s · (P_with − P_without).

    Returns ``NaN`` when either value is missing, the instrument is
    unknown, the row's cell has *n* < 5 paired comparisons, or μ is
    zero/missing.
    """
    bl, wf = row["baseline"], row["with_face"]
    if pd.isna(bl) or pd.isna(wf):
        return np.nan

    mn = _norm_metric(row["Metric"])
    cid, _ = _cell_for_row(mn, row["Dataset"], row.get("Instrument"))
    if cid is None:
        return np.nan

    cell = norm_table.loc[norm_table["cell_id"] == cid]
    if cell.empty:
        return np.nan
    c = cell.iloc[0]

    if c["n"] < 5 or pd.isna(c["mu"]) or c["mu"] == 0:
        return np.nan

    s = -1 if mn in LOWER_IS_BETTER else 1
    return s * (wf - bl) / c["mu"] * 100


# ═══════════════════════════════════════════════════════════════════════════════
# Auxiliary: Δz = Z / κ (not reported in the manuscript)
# ═══════════════════════════════════════════════════════════════════════════════


def delta_z(row, norm_table):
    r"""Δz = Z / κ_cell, with Z = s · (P_with − P_without).

    Returns ``NaN`` when the row's cell has *n* < 5 paired comparisons,
    the instrument is unknown, or κ is zero/missing.
    """
    bl, wf = row["baseline"], row["with_face"]
    if pd.isna(bl) or pd.isna(wf):
        return np.nan

    mn = _norm_metric(row["Metric"])
    cid, _ = _cell_for_row(mn, row["Dataset"], row.get("Instrument"))
    if cid is None:
        return np.nan

    cell = norm_table.loc[norm_table["cell_id"] == cid]
    if cell.empty:
        return np.nan
    c = cell.iloc[0]

    if c["n"] < 5 or pd.isna(c["kappa"]) or c["kappa"] == 0:
        return np.nan

    s = -1 if mn in LOWER_IS_BETTER else 1
    return s * (wf - bl) / c["kappa"]


def excluded_reason_for(row, norm_table):
    """Return a human-readable exclusion reason, or ``None`` if included."""
    bl, wf = row.get("baseline"), row.get("with_face")
    if pd.isna(bl) or pd.isna(wf):
        return "missing performance value"

    mn = _norm_metric(row["Metric"])
    cid, _ = _cell_for_row(mn, row["Dataset"], row.get("Instrument"))
    if cid is None:
        ds = str(row["Dataset"]).strip()
        inst = str(row.get("Instrument") or "not recorded").strip()
        return (
            f"no normalisable scale for {mn} "
            f"(dataset: {ds}; instrument: {inst})"
        )

    cell = norm_table.loc[norm_table["cell_id"] == cid]
    if cell.empty:
        return f"no paired comparisons for cell {cid}"
    c = cell.iloc[0]

    if c["n"] < 5:
        return f"cell {cid} has n={int(c['n'])} < 5 paired comparisons"
    if pd.isna(c["mu"]) or c["mu"] == 0:
        return f"cell {cid} has zero/missing μ"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Secondary metric: Δ% (manuscript Eq. 4, pre-registered)
# ═══════════════════════════════════════════════════════════════════════════════


def delta_percent(row):
    """Relative-% change, sign-flipped for lower-is-better metrics.

    Preserves the existing ``marginal_pct`` behaviour exactly.
    """
    bl, wf = row["baseline"], row["with_face"]
    if pd.isna(bl) or pd.isna(wf) or bl == 0:
        return np.nan
    mn = _norm_metric(row["Metric"])
    if mn in LOWER_IS_BETTER:
        return (bl - wf) / bl * 100
    return (wf - bl) / bl * 100


# ═══════════════════════════════════════════════════════════════════════════════
# Auxiliary κ-scale helper
# ═══════════════════════════════════════════════════════════════════════════════


def to_raw_units(delta_val, cell_id, norm_table):
    """Δz · κ  — back-transform to the original measurement scale."""
    cell = norm_table.loc[norm_table["cell_id"] == cell_id]
    if cell.empty:
        return np.nan
    return delta_val * cell.iloc[0]["kappa"]


# ═══════════════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════════════


def write_normalization_table(norm_table, path):
    """Export the normalisation table to CSV (for OSF publication)."""
    norm_table.to_csv(path, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: prepare a DataFrame with all derived columns
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_extraction_data(df):
    """Add every derived column to a raw extraction DataFrame.

    Returns ``(df_with_columns, norm_table)``.
    """
    df = df.copy()
    df["title_key"] = df["Title"].map(title_key)
    df["year_key"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    for col in (
        "Metric",
        "Clinical target",
        "Non-facial modalities",
        "Facial modality",
        "Baseline config",
        "Dataset",
    ):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["baseline"] = to_num(df["Baseline"])
    df["with_face"] = to_num(df["Multimodal (with face)"])

    # Canonicalise proportion-type metrics to the 0–1 scale.
    # The workbook mixes 0–1 and 0–100 conventions (e.g. F1 of 0.56 vs 56.0),
    # which corrupts κ and μ within a normalisation cell. Unit-bearing metrics
    # (RMSE/MAE) are left untouched. Applied per row so baseline and
    # with-face values always stay on the same scale.
    _proportion = ~df["Metric"].apply(_norm_metric).isin(_UNIT_BEARING)
    _on_100_scale = _proportion & (
        df[["baseline", "with_face"]].max(axis=1) > 1.5
    )
    df.loc[_on_100_scale, ["baseline", "with_face"]] /= 100.0
    _bad = _proportion & (df[["baseline", "with_face"]].max(axis=1) > 1.0)
    if _bad.any():
        raise ValueError(
            f"{int(_bad.sum())} proportion-metric rows remain outside [0, 1] "
            "after canonicalisation — inspect the workbook."
        )
    df["direction"] = df["Metric"].map(DIRECTION)
    df["metric_family"] = df["Metric"].map(FAMILY)

    norm_table = build_normalization_table(df)

    df["marginal_pct"] = df.apply(delta_percent, axis=1)
    df["delta_z"] = df.apply(lambda r: delta_z(r, norm_table), axis=1)
    df["excluded_reason"] = df.apply(
        lambda r: excluded_reason_for(r, norm_table), axis=1
    )
    # Store cell_id for back-transforms
    mn_series = df["Metric"].apply(_norm_metric)
    inst_series = (
        df["Instrument"] if "Instrument" in df.columns else [None] * len(df)
    )
    df["cell_id"] = [
        _cell_for_row(m, d, i)[0]
        for m, d, i in zip(
            mn_series, df["Dataset"].astype(str).str.strip(), inst_series
        )
    ]

    # Primary metric (manuscript Eq. 3): Δ_face = Z / μ_cell × 100.
    # Uses the common per-cell yardstick (μ of the cell's baselines) rather
    # than each study's own baseline (the secondary marginal_pct / Eq. 4).
    df["pct_of_baseline"] = df.apply(
        lambda r: delta_face(r, norm_table), axis=1
    )

    return df, norm_table


def load_extraction_data(project_dir):
    """Load ``data_extraction.xlsx`` and prepare all columns.

    Returns ``(df, norm_table)``.
    """
    raw = pd.read_excel(
        Path(project_dir) / DATA_DIR / "data_extraction.xlsx", sheet_name="Data Extraction"
    )
    return prepare_extraction_data(raw)
