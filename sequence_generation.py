"""
sequence_generation.py
=======================
Sparse-observation sequence generation for multi-horizon glaucoma progression
prediction.

For each eligible eye, a sliding window extracts two-visit observation inputs
(T0, T1) and assigns binary progression labels at prediction horizons of 2, 3,
and 4 years using the vfprogression package. Horizon labels are masked (rather
than excluded) when a valid visit is unavailable within the tolerance window.

Reference:
    Moradi et al., "Multi-Horizon Glaucoma Progression Prediction from Minimal
    Longitudinal Data: A Reliability-Aware Multimodal Deep Learning Framework"
    IEEE TBME, 2025.

Usage:
    python sequence_generation.py --input_path prepared_data.csv \
                                   --output_path sequences.csv
"""

import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Column name constants (adjust to your dataset)
# ─────────────────────────────────────────────
ID_COL          = "patient_id"
EYE_COL         = "eye"
DATE_COL        = "exam_date"
MD_COL          = "md"
RNFL_COL        = "avg_rnfl"
VFI_COL         = "vfi"               # Visual Field Index (for progression labeling)
PROGRESSION_COL = "vf_progression"    # Binary progression flag from vfprogression package
SEVERITY_COL    = "Severity"
SUBTYPE_COL     = "gl_subtype"

# ─────────────────────────────────────────────
# Temporal tolerances (years)
# ─────────────────────────────────────────────
T0_TARGET        = 0.0   # Baseline
T1_TARGET        = 1.0   # ~1 year follow-up
OBS_TOLERANCE    = 0.25  # ±0.25 yr for observation visits
PRED_HORIZONS    = [2, 3, 4]        # Prediction years
PRED_TOLERANCE   = 0.75  # ±0.75 yr for prediction visits
MIN_FOLLOWUP_YRS = 3.0   # Minimum total follow-up required


# ─────────────────────────────────────────────
# 1. Nearest-neighbor visit matching
# ─────────────────────────────────────────────

def find_nearest_visit(visits: pd.DataFrame,
                        anchor_date: pd.Timestamp,
                        target_years: float,
                        tolerance_years: float) -> Optional[pd.Series]:
    """
    Return the visit row closest to (anchor_date + target_years),
    provided it falls within ±tolerance_years. Returns None if no
    valid visit exists.
    """
    target_date = anchor_date + pd.DateOffset(days=int(target_years * 365.25))
    diffs_days  = (visits[DATE_COL] - target_date).abs().dt.days
    tol_days    = int(tolerance_years * 365.25)

    valid = visits[diffs_days <= tol_days]
    if valid.empty:
        return None
    return valid.loc[diffs_days[valid.index].idxmin()]


# ─────────────────────────────────────────────
# 2. Progression label assignment
# ─────────────────────────────────────────────

def get_horizon_label(visits: pd.DataFrame,
                       anchor_date: pd.Timestamp,
                       horizon_years: float,
                       tolerance_years: float = PRED_TOLERANCE) -> Optional[int]:
    """
    Assign a binary progression label (0/1) for a given prediction horizon.

    Progression is determined from VFI-based trend analysis over all visits
    from baseline (anchor_date) through the horizon window. Returns None if no
    valid visit is available within the tolerance window (masked label).

    Parameters
    ----------
    visits         : All visits for this (ID, Eye) pair, sorted by date.
    anchor_date    : Sequence baseline date (T0).
    horizon_years  : Target prediction year (2, 3, or 4).
    tolerance_years: Allowable date offset for matching (default ±0.75 yr).

    Returns
    -------
    1 if progressor, 0 if non-progressor, None if no valid visit (masked).
    """
    horizon_visit = find_nearest_visit(visits, anchor_date, horizon_years, tolerance_years)
    if horizon_visit is None:
        return None  # Masked label — horizon excluded from loss computation

    # Use all visits from T0 up to the horizon visit for progression assessment
    horizon_date = horizon_visit[DATE_COL]
    window_visits = visits[
        (visits[DATE_COL] >= anchor_date) &
        (visits[DATE_COL] <= horizon_date)
    ]

    if PROGRESSION_COL in window_visits.columns:
        # Use precomputed vfprogression label if available
        prog_flags = window_visits[PROGRESSION_COL].dropna()
        if prog_flags.empty:
            return None
        return int(prog_flags.iloc[-1])

    # Fallback: simple OLS trend on VFI
    if VFI_COL not in window_visits.columns or len(window_visits) < 2:
        return None
    t0    = pd.to_datetime(window_visits[DATE_COL].iloc[0])
    years = (pd.to_datetime(window_visits[DATE_COL]) - t0).dt.days / 365.25
    vfi   = window_visits[VFI_COL].values
    if years.std() < 1e-6:
        return None
    slope = np.polyfit(years, vfi, 1)[0]
    return int(slope < -0.5)  # Threshold: VFI declining > 0.5%/year


# ─────────────────────────────────────────────
# 3. Sequence extraction for one eye
# ─────────────────────────────────────────────

def extract_sequences_for_eye(visits: pd.DataFrame,
                               pid: str,
                               eye: str) -> list:
    """
    Apply a sliding window over all eligible baseline visits for one eye.

    For each candidate T0:
      - Match T1 at ~1 year (±OBS_TOLERANCE).
      - Require ≥ MIN_FOLLOWUP_YRS total follow-up.
      - Assign labels at prediction horizons 2, 3, 4 years.
      - Exclude the sequence only if ALL horizon labels are missing.

    Returns a list of sequence dictionaries.
    """
    visits = visits.sort_values(DATE_COL).reset_index(drop=True)
    visits[DATE_COL] = pd.to_datetime(visits[DATE_COL])

    sequences = []

    for i, t0_row in visits.iterrows():
        t0_date = t0_row[DATE_COL]

        # Check minimum follow-up
        last_date = visits[DATE_COL].iloc[-1]
        if (last_date - t0_date).days / 365.25 < MIN_FOLLOWUP_YRS:
            continue

        # Match T1 (observation visit ~1 year after T0)
        future_visits = visits[visits[DATE_COL] > t0_date]
        t1_row = find_nearest_visit(future_visits, t0_date, T1_TARGET, OBS_TOLERANCE)
        if t1_row is None:
            continue

        # Assign horizon labels (masked if visit unavailable)
        labels = {}
        for h in PRED_HORIZONS:
            labels[f"label_y{h}"] = get_horizon_label(visits, t0_date, h)

        # Exclude sequence only if ALL prediction labels are missing
        if all(v is None for v in labels.values()):
            continue

        # Build sequence record
        seq = {
            ID_COL:      pid,
            EYE_COL:     eye,
            "t0_date":   t0_date.strftime("%Y-%m-%d"),
            "t1_date":   t1_row[DATE_COL].strftime("%Y-%m-%d"),
            "t0_md":     t0_row.get(MD_COL),
            "t1_md":     t1_row.get(MD_COL),
            "t0_rnfl":   t0_row.get(RNFL_COL),
            "t1_rnfl":   t1_row.get(RNFL_COL),
            "t0_age":    t0_row.get("age"),
            "t0_sex":    t0_row.get("sex"),
            "t0_race":   t0_row.get("race"),
            "subtype":   t0_row.get(SUBTYPE_COL),
            "severity":  t0_row.get(SEVERITY_COL),
            "md_slope":  t0_row.get("md_slope"),
            "prog_cat":  t0_row.get("Progression_Category"),
        }
        seq.update(labels)
        sequences.append(seq)

    return sequences


# ─────────────────────────────────────────────
# 4. Full cohort sequence generation
# ─────────────────────────────────────────────

def generate_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate multi-horizon sequences for all eligible eyes in the dataset.

    Parameters
    ----------
    df : Prepared dataframe from data_preparation.py.

    Returns
    -------
    DataFrame of sequences with T0/T1 inputs and Y2/Y3/Y4 binary labels.
    """
    all_sequences = []
    groups = df.groupby([ID_COL, EYE_COL])
    total  = len(groups)

    for i, ((pid, eye), grp) in enumerate(groups, 1):
        if i % 500 == 0 or i == total:
            print(f"  Processing eye {i:,}/{total:,} ...", end="\r")
        seqs = extract_sequences_for_eye(grp, pid, eye)
        all_sequences.extend(seqs)

    print()
    return pd.DataFrame(all_sequences).reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. Summary reporting
# ─────────────────────────────────────────────

def print_sequence_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for the generated sequence dataset."""
    print("\n" + "=" * 70)
    print("SEQUENCE GENERATION SUMMARY")
    print("=" * 70)

    total_seq   = len(df)
    unique_eyes = df[[ID_COL, EYE_COL]].drop_duplicates()
    unique_pts  = df[ID_COL].nunique()

    print(f"Total sequences generated : {total_seq:,}")
    print(f"Unique patients           : {unique_pts:,}")
    print(f"Unique patient-eyes       : {len(unique_eyes):,}")
    avg_seq = total_seq / len(unique_eyes) if len(unique_eyes) > 0 else 0
    print(f"Mean sequences per eye    : {avg_seq:.2f}")

    print("\n--- Horizon Label Availability ---")
    for h in PRED_HORIZONS:
        col    = f"label_y{h}"
        if col not in df.columns:
            continue
        avail  = df[col].notna().sum()
        prog   = (df[col] == 1).sum()
        nprog  = (df[col] == 0).sum()
        masked = df[col].isna().sum()
        ratio  = nprog / prog if prog > 0 else float("nan")
        print(f"  Year {h}: {avail:,} labeled  "
              f"| progressors {prog:,}  non-progressors {nprog:,}  "
              f"| ratio {ratio:.1f}:1  | masked {masked:,}")

    if "subtype" in df.columns:
        print("\n--- Sequences by Glaucoma Subtype ---")
        sub_counts = df["subtype"].value_counts()
        for sub, cnt in sub_counts.items():
            pct = cnt / total_seq * 100
            print(f"  {sub:<20}: {cnt:,}  ({pct:.1f}%)")

    if "prog_cat" in df.columns:
        print("\n--- Fast Progressor Sequences ---")
        fast_eyes = df[df["prog_cat"] == "Fast"][[ID_COL, EYE_COL]].drop_duplicates()
        print(f"  Fast progressor eyes: {len(fast_eyes):,}  "
              f"({len(fast_eyes)/len(unique_eyes)*100:.1f}% of all eyes)")
        fast_seqs = df[df["prog_cat"] == "Fast"]
        print(f"  Fast progressor sequences: {len(fast_seqs):,}")


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

def run(input_path: str, output_path: str) -> pd.DataFrame:
    """Load prepared data, generate sequences, and save output."""
    print(f"Loading prepared data from: {input_path}")
    df = pd.read_csv(input_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"\nGenerating multi-horizon sequences ...")
    seq_df = generate_sequences(df)

    print_sequence_summary(seq_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(output_path, index=False)
    print(f"\nSaved sequences → {output_path}")
    return seq_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sparse-observation multi-horizon sequences for glaucoma prediction."
    )
    parser.add_argument("--input_path",  required=True,
                        help="Path to prepared_data.csv from data_preparation.py.")
    parser.add_argument("--output_path", default="sequences.csv",
                        help="Path to save generated sequences (default: sequences.csv).")
    args = parser.parse_args()

    run(args.input_path, args.output_path)
