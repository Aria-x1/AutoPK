"""
AutoPK NCA parameter calculator (manifest-based)
--------------------------------------------------
Fills in the 4 empty parameter columns in pk_curve_level_master_index.csv:
    Half life (hours), CMAX, AUC, TMAX

This is the version that reads from the split per-curve CSVs produced by
05_curve_processing (curve_manifest.csv's sub_csv_path column), and assumes
the curve-identification/labeling work (which curve is which drug/dose/
matrix, which figures are even PK curves) has already been done upstream by
the curve-splitting + manifest pipeline. It only computes NCA parameters
from already-clean time/concentration pairs.

(See also nca_from_raw_csvs_with_type_classifier.py, which is a separate,
self-contained approach that works directly from the raw digitized CSVs and
does its own data-type classification. Both are kept in active use --
see file docstring there for how the two differ.)

Method (matches pipeline conventions already established in AutoPK):
  - CMAX / TMAX: simple max of observed (non-missing) concentration and its time
  - AUC: linear trapezoidal over all valid points (AUC_last),
         then extrapolated to infinity: AUC_inf = AUC_last + C_last / ke
         (ke from the terminal log-linear regression)
  - Half-life: log-linear regression on the terminal window (last up to 5
         valid, positive-concentration points), t1/2 = ln(2) / ke

Known caveats (already documented for this project, kept here for
transparency / flagging, not "fixed" silently):
  - Fixed 5-point terminal window can UNDERESTIMATE t1/2 for biphasic
    IV-bolus curves (true terminal phase may need fewer/later points)
  - No C0 back-extrapolation for curves that start declining from an
    off-scale first point -> AUC can be UNDERESTIMATED for such curves
  - If the terminal-window regression slope is not clearly negative
    (R^2 low, or slope >= 0), t1/2/AUC_inf are flagged for manual review
    rather than silently reported

Usage:
    python nca_from_split_curves.py --master pk_curve_level_master_index.csv \
                        --base-dir /path/to/AutoPK \
                        --out pk_curve_level_master_index_filled.csv

    # --curve-root and --master/--out can each be overridden individually;
    # --base-dir (or $AUTOPK_BASE_DIR) just supplies defaults for them.
"""
import argparse
import os
import re
import numpy as np
import pandas as pd

DEFAULT_BASE_DIR = os.environ.get("AUTOPK_BASE_DIR", ".")

_LEADING_NUM = re.compile(r"^\s*([+-]?\d+\.?\d*)")


def parse_conc_cell(v):
    """Returns (numeric_value_or_nan, parse_flag_or_None).

    Handles known messy patterns found in AutoPK Branch-A digitized CSVs:
      - clean numeric -> used as-is
      - '-' or blank  -> missing (matches pre-existing convention for empty cells)
      - '~X'          -> approximate reading, use X, flag 'approx_reading'
      - 'X (some annotation/leaked reasoning text)' -> extract leading X,
                         flag 'annotated_estimate'
      - pure text, no leading number ('(N/A)', 'BIC', 'weeks from injection', ...)
                      -> missing, flag 'text_discarded'
    """
    if pd.isna(v):
        return np.nan, None
    if isinstance(v, (int, float)):
        return float(v), None
    s = str(v).strip()
    if s == "" or s == "-":
        return np.nan, None
    try:
        return float(s), None
    except ValueError:
        pass
    if s.startswith("~"):
        m = _LEADING_NUM.match(s[1:])
        if m:
            return float(m.group(1)), "approx_reading"
        return np.nan, "text_discarded"
    m = _LEADING_NUM.match(s)
    if m:
        return float(m.group(1)), "annotated_estimate"
    return np.nan, "text_discarded"


def compute_nca(df_curve, time_col, conc_col, terminal_n=5, min_r2=0.7):
    """df_curve: DataFrame with time_col, conc_col (may contain messy strings)."""
    parsed_conc = df_curve[conc_col].apply(parse_conc_cell)
    parsed_time = df_curve[time_col].apply(parse_conc_cell)  # same parser works for time text
    df_curve = df_curve.copy()
    df_curve[conc_col] = parsed_conc.apply(lambda x: x[0])
    df_curve[time_col] = parsed_time.apply(lambda x: x[0])
    cell_flags = [f for _, f in parsed_conc if f is not None]
    cell_flags += [f for _, f in parsed_time if f is not None]

    d = df_curve[[time_col, conc_col]].dropna()
    d = d[d[conc_col] > 0].sort_values(time_col)

    cell_flag_summary = ""
    if cell_flags:
        counts = pd.Series(cell_flags).value_counts()
        cell_flag_summary = ";".join(f"{k}={v}" for k, v in counts.items())

    if len(d) < 2:
        flag = "insufficient_points"
        if cell_flag_summary:
            flag = f"{flag}|{cell_flag_summary}"
        return dict(cmax=np.nan, tmax=np.nan, half_life=np.nan, auc=np.nan,
                     n_valid=len(d), flag=flag)

    t = d[time_col].values.astype(float)
    c = d[conc_col].values.astype(float)

    # CMAX / TMAX
    cmax_idx = np.argmax(c)
    cmax = c[cmax_idx]
    tmax = t[cmax_idx]

    # AUC_last: linear trapezoidal over available (non-missing) points
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    auc_last = _trapz(c, t)

    # Terminal phase regression (last up to `terminal_n` points)
    n_term = min(terminal_n, len(t))
    t_term = t[-n_term:]
    c_term = c[-n_term:]

    flag = ""
    half_life = np.nan
    auc_inf = auc_last

    if n_term >= 3:
        ln_c = np.log(c_term)
        slope, intercept = np.polyfit(t_term, ln_c, 1)
        pred = slope * t_term + intercept
        ss_res = np.sum((ln_c - pred) ** 2)
        ss_tot = np.sum((ln_c - ln_c.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if slope < 0:
            ke = -slope
            half_life = np.log(2) / ke
            c_last = c[-1]
            auc_inf = auc_last + c_last / ke
            if (r2 is not None and not np.isnan(r2) and r2 < min_r2):
                flag = f"low_terminal_R2={r2:.2f}"
        else:
            flag = "non_negative_terminal_slope"
    else:
        flag = "fewer_than_3_terminal_points"

    if cell_flag_summary:
        flag = f"{flag}|{cell_flag_summary}" if flag else cell_flag_summary

    return dict(cmax=cmax, tmax=tmax, half_life=half_life, auc=auc_inf,
                 n_valid=len(d), flag=flag)


def load_curve_file(path):
    d = pd.read_csv(path)
    # first column = time, second column = concentration (pipeline convention)
    time_col, conc_col = d.columns[0], d.columns[1]
    return d, time_col, conc_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                     help="AutoPK root dir (default: $AUTOPK_BASE_DIR or '.'), "
                          "used to fill in --master/--curve-root/--out when not given")
    ap.add_argument("--master", default=None,
                     help="Input master index CSV (default: <base-dir>/pk_curve_level_master_index.csv)")
    ap.add_argument("--curve-root", default=None,
                     help="Local dir such that curve_root + sub_csv_path = full path "
                          "(default: <base-dir>, since sub_csv_path already starts with "
                          "AutoPK/extracted_curves/... -- pass the parent of your AutoPK "
                          "checkout if that convention doesn't match your layout)")
    ap.add_argument("--out", default=None,
                     help="Output CSV (default: <base-dir>/pk_curve_level_master_index_filled.csv)")
    args = ap.parse_args()

    base_dir = args.base_dir
    master_path = args.master or os.path.join(base_dir, "pk_curve_level_master_index.csv")
    curve_root = args.curve_root or base_dir
    out_path = args.out or os.path.join(base_dir, "pk_curve_level_master_index_filled.csv")

    master = pd.read_csv(master_path)
    results = []
    for i, row in master.iterrows():
        rel_path = row["sub_csv_path"]
        full_path = os.path.join(curve_root, rel_path)
        if not os.path.exists(full_path):
            results.append(dict(cmax=np.nan, tmax=np.nan, half_life=np.nan,
                                 auc=np.nan, n_valid=0, flag="file_not_found"))
            continue
        try:
            d, time_col, conc_col = load_curve_file(full_path)
            res = compute_nca(d, time_col, conc_col)
        except Exception as e:
            res = dict(cmax=np.nan, tmax=np.nan, half_life=np.nan,
                        auc=np.nan, n_valid=0, flag=f"error:{e}")
        results.append(res)

    res_df = pd.DataFrame(results)
    master["CMAX"] = res_df["cmax"]
    master["TMAX"] = res_df["tmax"]
    master["Half life (hours)"] = res_df["half_life"]
    master["AUC"] = res_df["auc"]
    master["nca_flag"] = res_df["flag"]  # extra QC column, not in original schema

    master.to_csv(out_path, index=False)

    n_total = len(master)
    n_flagged = (res_df["flag"] != "").sum()
    print(f"Processed {n_total} curves. {n_flagged} flagged for review.")
    print(res_df["flag"].value_counts())


if __name__ == "__main__":
    main()
