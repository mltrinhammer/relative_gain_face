#!/usr/bin/env python3
"""
Generate a combined landscape longtable of study characteristics,
performance estimates, marginal contribution, and PROBAST+AI
risk-of-bias assessment, grouped by dataset.

Output:
    tables/study_characteristics.tex   (fragment for Overleaf)

Usage:
    python scripts/generate_study_table.py
"""

import sys
from pathlib import Path

import pandas as pd

from marginal_metric import (
    DATA_DIR,
    load_extraction_data,
    title_key as _title_key,
)

# ── Citation-key mapping ─────────────────────────────────────────────────────

TITLE_TO_CITEKEY = {
    "Detecting Distress Changes Using Multimodal Data During Interaction with A Smart Speaker": "lin2024detecting", 
    "Diagnosis of depression by behavioural signals: a multimodal approach" : "cummins2013diagnosis",
    "End-to-end multimodal clinical depression recognition using deep neural networks: A comparative analysis" : "muzammel2021end",
    "A Model of Normality Inspired Deep Learning Framework for Depression Relapse Prediction Using Audiovisual Data.": "Othmani_Zeghina_Muzammel_2022",
    "A multi-modal feature layer fusion model for assessment of depression based on attention mechanisms": "wang2022multi",
    "A Multimodal Artificial Intelligence Model for Depression Severity Detection Based on Audio and Video Signals": "Zhang_Zhang_Zhang_Zhao_2025",
    "A multimodal cross-transformer-based model to predict mild cognitive impairment using speech, language and vision.": "Poor_Dodge_Mahoor_2024",
    "A multimodal Framework for exploring behavioural cues for automatic Stress Detection": "Valerio_Mahmoud_2025",
    "A multimodal fusion model with multi-level attention mechanism for depression detection": "fang2023multimodal",
    "A Novel Audio-Visual Information Fusion System for Mental Disorders Detection": "Y.Li_S.Li_S.M.Naqvi_2024",
    "A novel audio-visual model with real-time gradient modulation for mental disorder detection": "li2026novel",
    "Acoustic and Facial Features From Clinical Interviews for Machine Learning-Based Psychiatric Diagnosis: Algorithm Development.": "Birnbaum_Abrami_Heisig_Ali_Arenare_Agurto_Lu_Kane_Cecchi_2022",
    "Artificial intelligent system for automatic depression level analysis through visual and vocal expressions": "jan2017artificial",
    "Audiface: Multimodal deep learning for depression screening": "flores2022audiface",
    "Audio-visual Feature Disentanglement and Fusion Network for Automatic Depression Severity Prediction": "li2025audio",
    "Automated Video Analysis of Audio-Visual Approaches to Predict and Detect Mild Cognitive Impairment and Dementia in Older Adults.": "Chu_Wang_Liang_Chou_Hsu_Wang_Liao_Chu_Lin_2023",
    "Automatic detection of self-adaptors for psychological distress": "lin2020automatic",
    "AVEC 2018 workshop and challenge: Bipolar disorder and cross-cultural affect recognition": "ringeval2018avec",
    "Bi-CFNet: A Multi-Task Audiovisual Framework with Visual Enhancement and Cross-Feedback for Depression Detection": "chan2026bi",
    "D-vlog: Multimodal Vlog Dataset for Depression Detection": "Yoon_Kang_Kim_Han_2022",
    "Deep learning-based detection of depression by fusing auditory, visual and textual clues.": "Xu_Chen_Tao_Xie_Liu_Lin_Liang_Du_Lin_Shi_2025",
    "Deep Multi-Modal Network Based Automated Depression Severity Estimation": "M.A.Uddin_J.",
    "Depmamba: Progressive fusion mamba for multimodal depression detection": "J.Ye_J.Zhang_H.Shan_2025",
    "Depression assessment by fusing high and low level features from audio, video, and text": "pampouchidou2016depression",
    "Depression recognition based on dynamic facial and vocal expression features using partial least square regression": "meng2013depression",
    "Depression scale dictionary decomposition framework for multimodal automatic depression level prediction": "niu2025depression",
    "Depression scale recognition from audio, visual and text analysis": "dham2017depression",
    "Diagnosis Framework for Probable Alzheimer's Disease and Mild Cognitive Impairment Based on Multi-Dimensional Emotion Features.": "Zhang_Lei_Ma_Long_Long_Chen_Luo_Tao_2024",
    "Diagnosis of depression based on facial multimodal data.": "Jin_Ye_Li_2025",
    "Digital detection of Alzheimer's disease using smiles and conversations with a chatbot.": "Takeshige-Amano_Oyama_Ogawa_Fusegi_Kambe_Shiina_Ueno_Okuzumi_Hatano_Motoi_etal._2024",
    "Disease and personality information enhanced depression detection based on the TransGCL framework": "Zhao_Zhang_Sun_Zhou_Li_Lian_Tian_Wang_Fu_2025",
    "Disentangled-multimodal privileged knowledge distillation for depression recognition with incomplete multimodal data": "Pan_Jiang_Jiang_Liu_2024",
    "DPD (DePression Detection) Net: a deep neural network for multimodal depression detection.": "He_Bakker_Lew_2024",
    "Dynamic multimodal measurement of depression severity using deep autoencoding": "dibekliouglu2017dynamic",
    "End-to-end multimodal system for depression detection from online recordings.": "Kowalewski_Stroinski_Kwarciak_Laptiev_Hemmerling_2023",
    "FC-AEN: Fully Convolutional Adaptive Ensemble Network for Automatic Depression Detection": "Y.Wang_ZLin_Y.Teng_Y.Cheng_H.Jiang_Y.Yang_2025",
    "FPT-Former: A Flexible Parallel Transformer of Recognizing Depression by Using Audiovisual Expert-Knowledge-Based Multimodal Measures": "li2024fpt",
    "Fusing Affective Dimensions and Audio-Visual Features from Segmented Video for Depression Recognition: INAOE-BUAP's Participation at AVEC'14 Challenge": "perez2014fusing",
    "Fusion of audio-visual features using hierarchical classifier systems for the recognition of affective states and the state of depression": "kachele2014fusion",
    "Harnessing multimodal approaches for depression detection using large language models and facial expressions": "sadeghi2024harnessing",
    "Harnessing multimodal emotion features in depression detection across gender: Integrating large language model, acoustic fusion and facial expression recognition": "jin2026harnessing",
    "Hique: Hierarchical question embedding network for multimodal depression detection": "jung2024hique",
    "Hybrid depression classification and estimation from audio video and text information": "yang2017hybrid",
    "Image encoding and fusion of multi-modal data enhance depression diagnosis in Parkinson's disease patients": "J.Li_Y.Zhao_H.",
    "Integrating Deep and Shallow Models for Multi-Modal Depression Analysis\u2014Hybrid Architectures": "L.Yang_D.Jiang_H.",
    "MDD-MARF: A multimodal depression detection model based on multi-level attention mechanism and residual fusion": "Zhou_Ge_Chen_Tan_Li_2026",
    "MDD-Net: Multimodal depression detection through mutual transformer": "M.R.Haque_M.M.Islam_S.M.TaslimUddinRaju_H.Altaheri_L.Nassar_F.Karray_2025",
    "Measurement of neuropsychiatric symptoms in the older adults with mild cognitive impairment based on speech and facial expressions: a cross-sectional observational study.": "Zhou_Yao_Han_Li_Xue_Li_2024",
    "Modeling Multimodal Depression Diagnosis From the Perspective of Local Depressive Representation": "J.Liang_P.Cao_C.",
    "MPCF: Multi-stage Progressive Cross-modal Fusion for Depression Severity Prediction Using Audio-Visual Modalities": "li2026mpcf",
    "Multi-level attention network using text, audio and video for depression prediction": "ray2019multi",
    "Multi-level spatiotemporal graph attention fusion for multimodal depression detection": "Yang_Zheng_2025",
    "Multi-modal depression estimation based on sub-attentional fusion": "wei2022multi",
    "Multi-modal human behaviour graph representation learning for automatic depression assessment": "H.Shen_S.Song_H.",
    "Multimodal affect analysis of psychodynamic play therapy": "halfon2021multimodal",
    "Multimodal and multiresolution depression detection from speech and facial landmark features": "nasir2016multimodal",
    "Multimodal approach for assessing neuromotor coordination in schizophrenia using convolutional neural networks": "siriwardena2021multimodal",
    "Multimodal depression detection using task-oriented transformer-based embedding": "rasipuram2022multimodal",
    "Multimodal depression recognition and analysis: Facial expression and body posture changes via emotional stimuli.": "Liu_Li_Wang_Bi_Lin_Wang_Yu_Ye_Zheng_2025",
    "Multimodal Depression Recognition via Mutual Information Maximization Joint With Multi-Task Learning.": "Liu_Liu_Ma_Wang_Yang_Mo_Alqahtani_Chen_2026",
    "Multimodal detection of depression in clinical interviews": "dibekliouglu2015multimodal",
    "Multimodal Fusion for Dementia Detection using Voice and Facial Features": "chen2025multimodal",
    "Multimodal interpretable depression analysis using visual, physiological, audio and textual data": "kumar2025multimodal",
    "Multimodal Sensing for Depression Risk Detection: Integrating Audio, Video, and Text Data.": "Zhang_Zhang_Ni_Wei_Yang_Jin_Huang_Liang_Zhang_Li_etal._2024",
    "Multimodal Spatiotemporal Representation for Automatic Depression Level Detection": "M.Niu_J",
    "Parallel Multiscale Bridge Fusion Network for Audio\u2013Visual Automatic Depression Assessment": "M.Hu_L.Liu_X.",
    "Predicting depression by using a novel deep learning model and video-audio-text multimodal data.": "Li_Yang_Zhao_Wang_Yao_Qian_Qi_2025",
    "Reading between the frames: Multi-modal depression detection in videos from non-verbal cues": "gimeno2024reading",
    "Real-life Depression, and Affect Recognition Workshop and Challenge": "ringeval2017avec",
    "Self-Esteem Assessment Based on Self-Introduction: A Multimodal Approach to Personality Computing.": "Zang_Yang_2025",
    "Semi-Structural Interview-Based Chinese Multimodal Depression Corpus Towards Automatic Preliminary Screening of Depressive Disorders": "B.Zou_J.Han_Y.",
    "SIMMA: Multimodal Automatic Depression Detection via Spatio-Temporal Ensemble and Cross-Modal Alignment": "wang2025simma",
    "STCM-Mamba: Multimodal Spatio-Temporal Cross-Modal Mamba for Depression Detection": "B.Zhou_M.A.",
    "Taking all the factors we need: A multimodal depression classification with uncertainty approximation": "ahmed2023taking",
    "Towards remote differential diagnosis of mental and neurological disorders using automatically extracted speech and facial features.": "Richter_Neumann_Ramanarayanan_2024",
    "Towards stable cross-domain depression recognition under missing modalities": "Chen_Tan_Lu_Xu_Wang_Zeng_Hu_2026",
    "Towards the automatic detection of social biomarkers in autism spectrum disorder: introducing the simulated interaction task (SIT).": "Drimalla_Scheffer_Landwehr_Baskow_Roepke_Behnia_Dziobek_2020",
    "Tracking depression severity from audio and video based on speech articulatory coordination": "williamson2019tracking",
    "Transformer-based multimodal feature enhancement networks for multimodal depression detection integrating video, audio and remote photoplethysmograph signals": "fan2024transformer",
    "TSFFM: Depression detection based on latent association of facial and body expressions.": "Li_Yi_Lu_Wang_Zheng_Han_Wang_2024",
    "Using convolutional neural networks for the assessment research of mental health": "liu2022using",
}


# ── Text normalisation helpers ────────────────────────────────────────────────

def _normalise_title(t):
    """Lowercase, strip trailing period, collapse whitespace, normalise dashes/quotes."""
    t = str(t).strip().rstrip(".")
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2012", "-")
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(t.lower().split())


_NORMALISED_LOOKUP = {}
for _title, _key in TITLE_TO_CITEKEY.items():
    _NORMALISED_LOOKUP[_normalise_title(_title)] = _key


def _get_citekey(title):
    """Return the BibTeX citation key for *title*, or None."""
    if pd.isna(title):
        return None
    return _NORMALISED_LOOKUP.get(_normalise_title(title))


def _escape(text):
    """Escape special LaTeX characters."""
    if pd.isna(text):
        return "---"
    text = str(text).strip()
    for old, new in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                     ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                     ("}", r"\}")]:
        text = text.replace(old, new)
    return text


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(project_dir):
    """Load data_extraction.xlsx via shared module, compute marginal_pct."""
    df, _norm = load_extraction_data(project_dir)
    # Alias columns expected by table rendering
    df["year"] = df["year_key"]
    df["baseline_val"] = df["baseline"]
    df["face_val"] = df["with_face"]
    return df.dropna(subset=["marginal_pct"]).copy()


def load_probast(project_dir):
    """Load PROBAST sheets, return per-study DataFrame with domain-level RoB."""
    xlsx = project_dir / DATA_DIR / "PROBAST_assessment.xlsx"
    dev = pd.read_excel(xlsx, sheet_name="Model Development")
    evl = pd.read_excel(xlsx, sheet_name="Model Evaluation")

    dev["title_key"] = dev["Title"].map(_title_key)
    evl["title_key"] = evl["Title"].map(_title_key)

    def _norm(v):
        if pd.isna(v):
            return None
        v = str(v).strip().lower()
        if v == "high":
            return "High"
        if v == "low":
            return "Low"
        return None

    def _worst(a, b):
        if a == "High" or b == "High":
            return "High"
        if a == "Low" or b == "Low":
            return "Low"
        return None

    records = []
    for _, d in dev.iterrows():
        tk = d["title_key"]
        e_row = evl[evl["title_key"] == tk]
        e = e_row.iloc[0] if not e_row.empty else pd.Series(dtype=object)

        records.append({
            "title_key": tk,
            "D1": _norm(d.get("Participants: Quality Concern")),
            "D2": _norm(d.get("Predictors: Quality Concern")),
            "D3": _norm(d.get("Outcome: Quality Concern")),
            "D4": _norm(d.get("Analysis: Quality Concern")),
            "E4": _norm(e.get("Analysis: RoB")) if not e_row.empty else None,
            "overall_quality": _worst(
                _norm(d.get("Overall Quality Concern")),
                _norm(e.get("Overall RoB")) if not e_row.empty else None,
            ),
            "overall_applicability": _worst(
                _norm(d.get("Overall Applicability Concern")),
                _norm(e.get("Overall Applicability Concern")) if not e_row.empty else None,
            ),
        })

    return pd.DataFrame(records).drop_duplicates(subset="title_key")


# ── LaTeX rendering ──────────────────────────────────────────────────────────

def _rob_cell(level):
    r"""Traffic-light cell: green + or red -."""
    if level == "Low":
        return r"\cellcolor{roblow}\textbf{+}"
    if level == "High":
        return r"\cellcolor{robhigh}\textbf{$-$}"
    return r"\cellcolor{robunclear}\textbf{?}"


def _delta_cell(val):
    """Format marginal-pct with colour: green for positive, red for negative."""
    if pd.isna(val):
        return "---"
    sign = "+" if val >= 0 else ""
    if val >= 0:
        return f"\\textcolor{{posgreen}}{{{sign}{val:.1f}}}"
    return f"\\textcolor{{negred}}{{{val:.1f}}}"


def _fmt_val(val):
    """Format a numeric performance value."""
    if pd.isna(val):
        return "---"
    # Show up to 2 decimal places, strip trailing zeros
    return f"{val:.2f}".rstrip("0").rstrip(".")


# Compact display names for long metric labels (otherwise the auto-width
# Metric column stretches to fit the longest label on every page).
METRIC_DISPLAY = {
    "Average (Acc, Prec, Recall, F1)": "Avg.\\ (A/P/R/F1)",
    "Unweighted Average Recall": "UAR",
    "F1-score": "F1-Score",
}


def _metric_label(metric):
    m = str(metric).strip()
    return _escape(METRIC_DISPLAY.get(m, m))


def _cite(title):
    """Return \\citeauthor{key}~(\\citeyear{key})\\cite{key} or escaped title.

    The trailing \\cite{key} prints the study's number from the main
    manuscript's reference list (the supplement shares that bibliography).
    """
    ck = _get_citekey(title)
    if ck:
        return f"\\citeauthor{{{ck}}}~(\\citeyear{{{ck}}})\\cite{{{ck}}}"
    return _escape(title)


NCOLS = 16


def generate_table(data, probast):
    """Generate the landscape longtable LaTeX string."""

    merged = data.merge(probast, on="title_key", how="left")

    # Dataset ordering: Proprietary first, then public datasets by study count desc
    ds_counts = (
        merged.groupby("Dataset")["title_key"]
        .nunique().sort_values(ascending=False)
    )
    public_ds = [ds for ds in ds_counts.index if ds != "Proprietary"]
    ds_order = (["Proprietary"] if "Proprietary" in ds_counts.index else []) + public_ds

    n_studies = merged["title_key"].nunique()
    n_comp = len(merged)

    lines = []

    # ── Preamble ──────────────────────────────────────────────────────────
    lines.append(r"\begin{landscape}")
    lines.append(r"\pagestyle{empty}  % suppress page numbers on landscape pages")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(
        # Column widths are tuned to the widest unbreakable item in each
        # column at \scriptsize (measured, in cm):
        #   Target      "Schizophrenia"   1.80
        #   Instrument  "ADOS/ADIR"       1.78
        #   Multimodal  header            1.77   ("A+V+rPPG" 1.67)
        #   Metric      "Correlation"     1.48
        # The Metric column carried ~0.2 cm of slack, which is redistributed
        # to Target / Instrument / Multimodal so that nothing overflows into
        # its right-hand neighbour. Total p-column width is unchanged
        # (11.6 cm), so the table's overall width is identical.
        # \hspace{0pt} re-enables hyphenation inside \raggedright columns, so
        # an over-long value in a future data revision wraps instead of
        # bleeding into the next column.
        r"\begin{longtable}{"
        r">{\raggedright\hspace{0pt}\arraybackslash}p{2.75cm} "  # Study
        r">{\raggedright\hspace{0pt}\arraybackslash}p{1.95cm} "  # Target
        r">{\raggedright\hspace{0pt}\arraybackslash}p{1.90cm} "  # Instrument
        r">{\raggedright\hspace{0pt}\arraybackslash}p{1.35cm} "  # Baseline config
        r">{\raggedright\hspace{0pt}\arraybackslash}p{1.90cm} "  # Multimodal config
        r">{\raggedright\hspace{0pt}\arraybackslash}p{1.60cm} "  # Metric
        r"r "                              # Base
        r"r "                              # +Face
        r"r "                              # Primary delta%
        r">{\centering}p{0.55cm} "        # D1
        r">{\centering}p{0.55cm} "        # D2
        r">{\centering}p{0.55cm} "        # D3
        r">{\centering}p{0.55cm} "        # D4
        r">{\centering}p{0.55cm} "        # E4
        # "RoB" is the one wide header among the PROBAST columns (0.67 cm at
        # \scriptsize); 0.55 cm made it bleed into the "A" column.
        r">{\centering}p{0.70cm} "        # RoB
        r">{\centering\arraybackslash}p{0.55cm}"  # A
        r"}"
    )
    lines.append(
        r"\caption{Characteristics and performance of included studies "
        r"(N\,=\," + str(n_studies) + r" studies, "
        + str(n_comp) + r" comparisons), grouped by dataset. "
        r"Modality abbreviations: "
        r"A\,=\,Audio, V\,=\,Visual (face), T\,=\,Text, "
        r"G\,=\,Gaze, H\,=\,Head, B\,=\,Body, "
        r"P\,=\,Physiological, U\,=\,Motion, SAM\,=\,Self-Assessment, "
        r"rPPG\,=\,remote photoplethysmography. "
        r"Reference numbers refer to the reference list of the main "
        r"manuscript. "
        r"\textbf{Instrument} is the clinical measurement scale for the target. "
        r"\textbf{Rel.\ gain} is the relative gain defined in "
        r"\autoref{sec:marginal-share-gain} (in \%, positive favours the "
        r"facial modality); it is left blank (---) for cells with fewer "
        r"than five paired comparisons or no recognised instrument. "
        r"PROBAST+AI domains: "
        r"\textbf{D1}\,Participants, "
        r"\textbf{D2}\,Predictors, "
        r"\textbf{D3}\,Outcome, "
        r"\textbf{D4}\,Analysis (dev.), "
        r"\textbf{E4}\,Analysis (eval.), "
        r"\textbf{RoB}\,Overall risk of bias, "
        r"\textbf{A}\,Overall applicability concern. "
        r"\colorbox{roblow}{\textbf{+}}\,Low risk; "
        r"\colorbox{robhigh}{\textbf{$-$}}\,High risk.}"
        r"\label{tab:study_characteristics} \\"
    )

    # ── Header ────────────────────────────────────────────────────────────
    # Grouped top row: the two relative-score-gain columns (9-10) sit under a
    # shared "Rel. gain (%)" spanner, and the seven risk-of-bias columns
    # (11-17) under a "PROBAST+AI" spanner.
    group_row = (
        " & ".join(
            [""] * 8
            + [""]
            + [r"\multicolumn{7}{c}{\textbf{PROBAST+AI}}"]
        )
        + r" \\"
    )
    header = (
        group_row + "\n"
        + r"\cmidrule(lr){10-16}" + "\n"
        + r"\textbf{Study}"
        r" & \textbf{Target}"
        r" & \textbf{Instr.}"
        r" & \textbf{Baseline}"
        r" & \textbf{Multimodal}"
        r" & \textbf{Metric}"
        r" & \textbf{Base}"
        r" & \textbf{+Face}"
        r" & \textbf{Rel.\ gain (\%)}"
        r" & \textbf{D1}"
        r" & \textbf{D2}"
        r" & \textbf{D3}"
        r" & \textbf{D4}"
        r" & \textbf{E4}"
        r" & \textbf{RoB}"
        r" & \textbf{A} \\"
    )

    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append("")
    lines.append(
        f"\\multicolumn{{{NCOLS}}}{{c}}"
        r"{{\textit{(continued from previous page)}}} \\"
    )
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append("")
    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{{NCOLS}}}{{r}}"
        r"{{\textit{(continued on next page)}}} \\"
    )
    lines.append(r"\endfoot")
    lines.append("")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    # ── Body: iterate over dataset groups ─────────────────────────────────
    for ds_idx, ds in enumerate(ds_order):
        ds_sub = merged[merged["Dataset"] == ds].copy()
        ds_sub.sort_values(["year", "Title", "Metric"], inplace=True)

        n_ds_studies = ds_sub["title_key"].nunique()
        n_ds_comp = len(ds_sub)

        # Dataset group header
        ds_label = _escape(ds) if ds != "Proprietary" else "Proprietary datasets"
        lines.append("")
        if ds_idx > 0:
            lines.append(r"\midrule")
        lines.append(
            f"\\multicolumn{{{NCOLS}}}{{l}}"
            f"{{\\textbf{{{ds_label}}} "
            f"\\textnormal{{({n_ds_studies} studies, {n_ds_comp} comparisons)}}"
            f"}} \\\\"
        )
        lines.append(r"\midrule")

        # Track previous study to suppress repeated citation
        prev_tk = None

        for _, row in ds_sub.iterrows():
            tk = row["title_key"]
            first_row_for_study = (tk != prev_tk)
            prev_tk = tk

            # Study citation (only on first row for this study in this group)
            study_cell = _cite(row["Title"]) if first_row_for_study else ""

            # Modality / performance cells (always shown)
            target_cell = _escape(row.get("Clinical target"))
            instrument_cell = _escape(row.get("Instrument"))
            baseline_cfg = _escape(row.get("Baseline config"))
            multimodal_cfg = _escape(row.get("Multimodal config"))
            metric_cell = _metric_label(row.get("Metric"))
            base_cell = _fmt_val(row.get("baseline_val"))
            face_val_cell = _fmt_val(row.get("face_val"))
            # Primary = Eq. 3 (pct_of_baseline)
            primary_cell = _delta_cell(row.get("pct_of_baseline"))

            # RoB cells (shown on every row — PROBAST is per-study,
            # propagated to all comparison rows via the merge)
            rob_cells = [
                _rob_cell(row.get("D1")),
                _rob_cell(row.get("D2")),
                _rob_cell(row.get("D3")),
                _rob_cell(row.get("D4")),
                _rob_cell(row.get("E4")),
                _rob_cell(row.get("overall_quality")),
                _rob_cell(row.get("overall_applicability")),
            ]

            cells = [
                study_cell,
                target_cell,
                instrument_cell,
                baseline_cfg,
                multimodal_cfg,
                metric_cell,
                base_cell,
                face_val_cell,
                primary_cell,
            ] + rob_cells

            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\end{longtable}")
    lines.append(r"\normalsize")
    lines.append(r"\end{landscape}")
    lines.append(r"\pagestyle{plain}  % restore page numbers")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    out_dir = project_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data_extraction.xlsx...", file=sys.stderr)
    data = load_data(project_dir)
    print(f"  {len(data)} comparisons, {data['title_key'].nunique()} studies",
          file=sys.stderr)

    print("Loading PROBAST_assessment.xlsx...", file=sys.stderr)
    probast = load_probast(project_dir)
    print(f"  {len(probast)} studies with RoB data", file=sys.stderr)

    # Check for unmatched studies
    data_keys = set(data["title_key"].unique())
    prob_keys = set(probast["title_key"].unique())
    unmatched = data_keys - prob_keys
    if unmatched:
        print(f"  WARNING: {len(unmatched)} studies in data but not in PROBAST:",
              file=sys.stderr)
        for tk in sorted(unmatched):
            t = data[data["title_key"] == tk]["Title"].iloc[0]
            print(f"    - {t}", file=sys.stderr)

    # Check for unmapped citation keys
    unmapped = [t for t in data["Title"].unique() if _get_citekey(t) is None]
    if unmapped:
        print(f"  WARNING: {len(unmapped)} title(s) have no citation key:",
              file=sys.stderr)
        for t in sorted(unmapped):
            print(f"    - {t}", file=sys.stderr)

    print("Generating table...", file=sys.stderr)
    tex = generate_table(data, probast)

    out_path = out_dir / "study_characteristics.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"Saved: {out_path}", file=sys.stderr)

    # Also print required preamble
    print("", file=sys.stderr)
    print("Required in your preamble:", file=sys.stderr)
    print(r"  \usepackage{booktabs}", file=sys.stderr)
    print(r"  \usepackage{longtable}", file=sys.stderr)
    print(r"  \usepackage{array}", file=sys.stderr)
    print(r"  \usepackage[table]{xcolor}", file=sys.stderr)
    print(r"  \usepackage{pdflscape}  % rotates the PDF page itself", file=sys.stderr)
    print(r"  \usepackage{natbib}", file=sys.stderr)
    print(r"  \definecolor{roblow}{HTML}{A8D5A2}", file=sys.stderr)
    print(r"  \definecolor{robhigh}{HTML}{E8A09A}", file=sys.stderr)
    print(r"  \definecolor{robunclear}{HTML}{F5E6A3}", file=sys.stderr)
    print(r"  \definecolor{posgreen}{HTML}{2E7D32}", file=sys.stderr)
    print(r"  \definecolor{negred}{HTML}{C62828}", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
