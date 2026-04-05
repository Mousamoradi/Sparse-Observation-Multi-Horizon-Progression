"""
data_preparation.py
====================
Data preparation pipeline for multi-horizon glaucoma progression prediction.

Loads paired cpRNFL OCT and visual field (VF) data, computes total deviation
(TD) values, assigns MD-slope-based progression labels, and categorizes eyes
into progression groups for downstream sequence generation.

Reference:
    Moradi et al., "Multi-Horizon Glaucoma Progression Prediction from Minimal
    Longitudinal Data: A Reliability-Aware Multimodal Deep Learning Framework"
    IEEE TBME, 2025.

Usage:
    python data_preparation.py --vf_path /path/to/vf_data.csv \
                                --rnfl_path /path/to/rnfl_data.csv \
                                --output_path /path/to/output.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# Column name constants (adjust to your dataset)
# ─────────────────────────────────────────────
ID_COL   = "patient_id"
EYE_COL  = "eye"          # e.g., "OD" / "OS"
DATE_COL = "exam_date"
MD_COL   = "md"           # Mean Deviation (dB)
RNFL_COL = "avg_rnfl"     # Average cpRNFL thickness (µm)


# ─────────────────────────────────────────────
# 1. Loading utilities
# ─────────────────────────────────────────────

def load_csv(path: str, date_col: str = DATE_COL) -> pd.DataFrame:
    """Load a CSV file and standardize the date column to YYYY-MM-DD."""
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True).dt.strftime("%Y-%m-%d")
    return df


def match_vf_rnfl(df_vf: pd.DataFrame,
                  df_rnfl: pd.DataFrame,
                  window_days: int = 180) -> pd.DataFrame:
    """
    Match VF and cpRNFL records by patient ID, eye laterality, and exam date
    within ±window_days tolerance.

    Parameters
    ----------
    df_vf       : VF dataframe (must contain ID_COL, EYE_COL, DATE_COL).
    df_rnfl     : cpRNFL dataframe (must contain ID_COL, EYE_COL, DATE_COL).
    window_days : Maximum allowable date difference in days (default 180).

    Returns
    -------
    Merged dataframe of matched VF–cpRNFL pairs.
    """
    df_vf   = df_vf.copy()
    df_rnfl = df_rnfl.copy()

    df_vf[DATE_COL]   = pd.to_datetime(df_vf[DATE_COL])
    df_rnfl[DATE_COL] = pd.to_datetime(df_rnfl[DATE_COL])

    merged_rows = []
    for (pid, eye), vf_group in df_vf.groupby([ID_COL, EYE_COL]):
        rnfl_group = df_rnfl[(df_rnfl[ID_COL] == pid) & (df_rnfl[EYE_COL] == eye)]
        if rnfl_group.empty:
            continue
        for _, vf_row in vf_group.iterrows():
            diffs = (rnfl_group[DATE_COL] - vf_row[DATE_COL]).abs()
            idx   = diffs.idxmin()
            if diffs[idx].days <= window_days:
                row = pd.concat([vf_row, rnfl_group.loc[idx].add_suffix("_rnfl")], axis=0)
                merged_rows.append(row)

    return pd.DataFrame(merged_rows).reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. Quality control
# ─────────────────────────────────────────────

def apply_quality_filters(df: pd.DataFrame,
                           fp_col: str   = "false_positive_rate",
                           fp_thresh: float = 0.33,
                           ss_col: str   = "signal_strength",
                           ss_thresh: int = 7) -> pd.DataFrame:
    """
    Retain only reliable exams:
      - VF false-positive rate < fp_thresh (default 33 %)
      - cpRNFL OCT signal strength ≥ ss_thresh (default 7)
    """
    n_before = len(df)
    if fp_col in df.columns:
        df = df[df[fp_col] < fp_thresh]
    if ss_col in df.columns:
        df = df[df[ss_col] >= ss_thresh]
    print(f"Quality filter: {n_before:,} → {len(df):,} records retained.")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. MD slope computation
# ─────────────────────────────────────────────

def compute_md_slope(df: pd.DataFrame,
                     min_visits: int = 5) -> pd.DataFrame:
    """
    Estimate per-eye MD slope (dB/year) using ordinary least squares regression
    on all available longitudinal VF records.

    Eyes with fewer than min_visits reliable VFs are excluded.
    """
    results = []
    for (pid, eye), grp in df.groupby([ID_COL, EYE_COL]):
        grp = grp.sort_values(DATE_COL).dropna(subset=[MD_COL, DATE_COL])
        if len(grp) < min_visits:
            continue
        t0    = pd.to_datetime(grp[DATE_COL].iloc[0])
        years = (pd.to_datetime(grp[DATE_COL]) - t0).dt.days / 365.25
        md    = grp[MD_COL].values
        if years.std() < 1e-6:
            continue
        slope = np.polyfit(years, md, 1)[0]
        results.append({ID_COL: pid, EYE_COL: eye, "md_slope": slope})

    slope_df = pd.DataFrame(results)
    return df.merge(slope_df, on=[ID_COL, EYE_COL], how="left")


# ─────────────────────────────────────────────
# 4. Progression categorization
# ─────────────────────────────────────────────

def categorize_progression(slope: float) -> str:
    """
    Assign a progression category based on MD slope (dB/year).

    Categories
    ----------
    Fast          : md_slope ≤ −1.0
    Moderate      : −1.0 < md_slope < −0.5
    Slow          : −0.5 ≤ md_slope < 0
    Non-progressor: md_slope ≥ 0
    """
    if pd.isna(slope):
        return pd.NA
    if slope <= -1.0:
        return "Fast"
    elif slope < -0.5:
        return "Moderate"
    elif slope < 0.0:
        return "Slow"
    else:
        return "Non-progressor"


def assign_progression_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply progression categorization to the dataframe."""
    df = df.copy()
    df["Progression_Category"] = df["md_slope"].apply(categorize_progression)
    return df


# ─────────────────────────────────────────────
# 5. Summary reporting
# ─────────────────────────────────────────────

def print_progression_summary(df: pd.DataFrame) -> None:
    """Print progression category distribution for unique (ID, Eye) pairs."""
    print("\n" + "=" * 70)
    print("PROGRESSION CATEGORY SUMMARY  (unique patient-eyes)")
    print("=" * 70)

    unique = df[[ID_COL, EYE_COL, "md_slope", "Progression_Category"]].drop_duplicates()
    total  = len(unique)
    valid  = unique["md_slope"].notna().sum()

    print(f"Total unique patient-eyes : {total:,}")
    print(f"  With valid md_slope     : {valid:,}")
    print(f"  Missing md_slope        : {total - valid:,}\n")

    cats = ["Fast", "Moderate", "Slow", "Non-progressor"]
    for cat in cats:
        sub  = unique[unique["Progression_Category"] == cat]["md_slope"]
        n    = len(sub)
        pct  = n / valid * 100 if valid > 0 else 0.0
        mean = sub.mean() if n > 0 else float("nan")
        print(f"  {cat:<18s}: {n:5,}  ({pct:5.1f}%)   mean slope = {mean:.3f} dB/year")

    print()

    # Threshold breakdown
    valid_df = unique[unique["md_slope"].notna()]
    fast     = (valid_df["md_slope"] <= -1.0).sum()
    moderate = ((valid_df["md_slope"] > -1.0) & (valid_df["md_slope"] < -0.5)).sum()
    slow     = ((valid_df["md_slope"] >= -0.5) & (valid_df["md_slope"] < 0.0)).sum()
    stable   = (valid_df["md_slope"] >= 0.0).sum()

    print("-" * 70)
    print(f"{'Fast progression (≤ −1.0)':<40}: {fast:,}  ({fast/valid*100:.1f}%)")
    print(f"{'Moderate (−1.0 to −0.5)':<40}: {moderate:,}  ({moderate/valid*100:.1f}%)")
    print(f"{'Slow (−0.5 to 0)':<40}: {slow:,}  ({slow/valid*100:.1f}%)")
    print(f"{'Stable / Improving (≥ 0)':<40}: {stable:,}  ({stable/valid*100:.1f}%)")
    print(f"{'Total categorized':<40}: {fast+moderate+slow+stable:,}  (should equal {valid:,})")

    # Per-severity breakdown if available
    if "Severity" in df.columns:
        print("\n" + "-" * 70)
        print("MD Slope by Severity Stage")
        print("-" * 70)
        sev_df = df[[ID_COL, EYE_COL, "Severity", "md_slope"]].drop_duplicates()
        sev_df = sev_df[sev_df["md_slope"].notna()]
        for sev in sorted(sev_df["Severity"].dropna().unique()):
            sub = sev_df[sev_df["Severity"] == sev]
            n   = len(sub)
            print(f"\n  {sev} (n={n:,})")
            print(f"    Mean slope   : {sub['md_slope'].mean():.4f} dB/year")
            print(f"    Median slope : {sub['md_slope'].median():.4f} dB/year")
            for label, mask in [
                ("Fast (≤ −1)",        sub["md_slope"] <= -1.0),
                ("Moderate",           (sub["md_slope"] > -1.0) & (sub["md_slope"] < -0.5)),
                ("Slow",               (sub["md_slope"] >= -0.5) & (sub["md_slope"] < 0.0)),
                ("Stable (≥ 0)",       sub["md_slope"] >= 0.0),
            ]:
                cnt = mask.sum()
                print(f"    {label:<25}: {cnt:,}  ({cnt/n*100:.1f}%)")


# ─────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(vf_path: str, rnfl_path: str, output_path: str) -> pd.DataFrame:
    """
    End-to-end data preparation pipeline.

    Steps
    -----
    1. Load VF and cpRNFL CSV files.
    2. Match records within ±6-month window.
    3. Apply quality control filters.
    4. Compute per-eye MD slope.
    5. Assign progression categories.
    6. Save output.
    """
    print("Loading VF data ...")
    df_vf = load_csv(vf_path)

    print("Loading cpRNFL data ...")
    df_rnfl = load_csv(rnfl_path)

    print("Matching VF–cpRNFL pairs (±6 months) ...")
    df_matched = match_vf_rnfl(df_vf, df_rnfl, window_days=180)

    print("Applying quality filters ...")
    df_filtered = apply_quality_filters(df_matched)

    print("Computing MD slopes ...")
    df_slopes = compute_md_slope(df_filtered, min_visits=5)

    print("Assigning progression labels ...")
    df_final = assign_progression_labels(df_slopes)

    print_progression_summary(df_final)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\nSaved prepared dataset → {output_path}")
    return df_final


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data preparation for multi-horizon glaucoma progression prediction."
    )
    parser.add_argument("--vf_path",   required=True, help="Path to VF CSV file.")
    parser.add_argument("--rnfl_path", required=True, help="Path to cpRNFL CSV file.")
    parser.add_argument("--output_path", default="prepared_data.csv",
                        help="Path to save the prepared dataset (default: prepared_data.csv).")
    args = parser.parse_args()

    run_pipeline(args.vf_path, args.rnfl_path, args.output_path)
