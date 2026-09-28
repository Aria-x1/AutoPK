#!/usr/bin/env python3
"""
NCA parameter calculator (raw-CSV + self-contained classifier version)
------------------------------------------------------------------------
Calculates PK parameters directly from the raw digitized CSVs in
extracted_images/, without going through the curve-splitting/manifest
pipeline (05_curve_processing). Unlike nca_from_split_curves.py, this
script does its own data-type classification (is this figure actually a
concentration-time curve? is it a scatter plot, a correlation plot, viral
load data, etc.?) and its own time/concentration column identification,
rather than relying on the curve_manifest.csv labeling.

Strategy:
- Calculate every PK parameter that can be calculated
- No unit conversion (units are kept as-is, unconverted)
- No data filtering beyond the type classification below (everything that
  passes classification is kept)
- For individual-subject curves, only the mean across curves is computed

Usage:
    python nca_from_raw_csvs_with_type_classifier.py --base-dir /path/to/AutoPK
    python nca_from_raw_csvs_with_type_classifier.py --base-dir /path/to/AutoPK --test
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid

warnings.filterwarnings('ignore')

DEFAULT_BASE_DIR = os.environ.get("AUTOPK_BASE_DIR", ".")


class SmartDataIdentifier:
    """Determines what kind of PK-relevant data (if any) a CSV contains."""

    def __init__(self):
        self.time_keywords = ['time', 'hour', 'hr', 'h)', 'day', 'week', 'month']
        self.conc_keywords = ['concentration', 'conc', 'level', 'mg/l', 'ng/ml', 'µg', 'ug', 'mg/ml']
        self.exclude_keywords = [
            'hiv-1 rna', 'viral rna', 'viral load', 'log10 hiv',
            'cd4', 'cd8', 'lymphocyte',
            'schema', 'study design', 'flowchart',
            'goodness of fit', 'residual', 'pred vs obs',
            'median change', 'decrease in log10', 'change in',
            'distribution', 'histogram'
        ]

    def identify_data_type(self, df, metadata_row=None, caption=""):
        """Classify what kind of data this CSV holds."""

        # Step 1: check metadata figure_type first
        if metadata_row is not None:
            figure_type = str(metadata_row.get('figure_type', '')).lower()
            has_pk_curve = metadata_row.get('has_pk_curve', False)

            # Metadata explicitly says this is a concentration-time curve
            if 'concentration-time' in figure_type or has_pk_curve == True:
                # still verify it's actually a time series
                if not self._is_valid_time_series(df):
                    return {
                        'data_type': 'scatter_plot_or_correlation',
                        'should_process': False,
                        'confidence': 'high',
                        'reason': 'Metadata says PK curve but data structure suggests scatter/correlation plot'
                    }
                return {
                    'data_type': 'concentration_time_curve',
                    'should_process': True,
                    'confidence': 'high',
                    'reason': 'Metadata indicates concentration-time curve'
                }

        # Step 2: check exclusion keywords
        caption_lower = caption.lower()

        # check for correlation/relationship plots
        correlation_keywords = ['vs', 'versus', 'relationship', 'correlation',
                               'clearance', 'weight', 'creatinine', 'body weight']
        if any(kw in caption_lower for kw in correlation_keywords):
            return {
                'data_type': 'correlation_or_relationship',
                'should_process': False,
                'confidence': 'high',
                'reason': 'Correlation or relationship plot (not time series)'
            }

        if any(kw in caption_lower for kw in self.exclude_keywords):
            return {
                'data_type': 'excluded',
                'should_process': False,
                'confidence': 'high',
                'reason': 'Excluded based on caption keywords'
            }

        # Step 3: analyze column structure
        columns = df.columns.tolist()
        if len(columns) < 2:
            return {
                'data_type': 'insufficient_columns',
                'should_process': False,
                'confidence': 'high',
                'reason': 'Less than 2 columns'
            }

        # classify column types
        col_types = self._classify_columns(df)

        # Step 4: check whether the column headers indicate a non-time X axis
        first_col_lower = columns[0].lower()
        non_time_x_keywords = ['weight', 'clearance', 'creatinine', 'dose', 'bmi']
        if any(kw in first_col_lower for kw in non_time_x_keywords):
            return {
                'data_type': 'pk_parameter_relationship',
                'should_process': False,
                'confidence': 'high',
                'reason': f'X-axis is {columns[0]}, not time'
            }

        # Step 5: time + any numeric column -> check whether it's a valid time series
        if col_types['time'] and len(col_types['numeric']) >= 2:

            # check whether this is really a time series (not a scatter plot)
            if not self._is_valid_time_series(df):
                return {
                    'data_type': 'scatter_plot_at_time_points',
                    'should_process': False,
                    'confidence': 'medium',
                    'reason': 'Multiple measurements at same time points (scatter plot)'
                }

            # caption mentions concentration/plasma/serum
            if any(kw in caption_lower for kw in ['concentration', 'plasma', 'serum', 'level']):
                return {
                    'data_type': 'concentration_time_curve',
                    'should_process': True,
                    'confidence': 'medium',
                    'reason': 'Has time column and caption mentions concentration'
                }

            # caption mentions trough/Cmin
            if any(kw in caption_lower for kw in ['trough', 'cmin', 'c min']):
                return {
                    'data_type': 'trough_over_time',
                    'should_process': True,
                    'confidence': 'medium',
                    'reason': 'Time series of trough concentrations'
                }

            # has time + numeric columns, but the specific meaning is unclear
            # conservative default: assume it's concentration
            return {
                'data_type': 'possible_concentration_time',
                'should_process': True,
                'confidence': 'low',
                'reason': 'Has time and numeric data, assuming concentration'
            }

        # Step 6: no time column, but has numeric data
        if len(col_types['numeric']) >= 2:
            return {
                'data_type': 'unknown_numeric',
                'should_process': False,
                'confidence': 'low',
                'reason': 'Numeric data without time axis'
            }

        # Default
        return {
            'data_type': 'not_processable',
            'should_process': False,
            'confidence': 'high',
            'reason': 'No recognizable PK data structure'
        }

    def _is_valid_time_series(self, df):
        """
        Checks whether this is a valid time series (as opposed to a scatter plot).

        Valid time series: each time point has only 1-2 measurements.
        Scatter plot: some time points have many repeated values.

        Returns:
        --------
        bool: True if valid time series, False if scatter plot
        """
        time_col = None
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in self.time_keywords):
                time_col = col
                break

        if time_col is None:
            return True  # no time column, can't judge -- default to True

        try:
            time_data = pd.to_numeric(df[time_col], errors='coerce').dropna()

            if len(time_data) == 0:
                return False

            # count how many times each time value appears
            time_counts = time_data.value_counts()

            # if any time point appears more than 5 times, it's likely a scatter plot
            if (time_counts > 5).any():
                return False

            # if most time points appear only 1-2 times, it's a time series;
            # if many time points repeat a lot, it's a scatter plot
            median_count = time_counts.median()
            if median_count > 3:
                return False

            return True

        except Exception:
            return True  # parsing failed, default to True (conservative)

    def _classify_columns(self, df):
        """Classify each column's type."""
        result = {
            'time': [],
            'concentration': [],
            'numeric': []
        }

        for col in df.columns:
            col_lower = col.lower()

            # check whether it's a numeric column
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > 0:
                    result['numeric'].append(col)
            except Exception:
                pass

            # time column
            if any(kw in col_lower for kw in self.time_keywords):
                result['time'].append(col)

            # concentration column
            if any(kw in col_lower for kw in self.conc_keywords):
                result['concentration'].append(col)

        return result


class PKCalculator:
    """PK parameter calculator -- kept intentionally simple."""

    def __init__(self, min_terminal_points=3):
        self.min_terminal_points = min_terminal_points

    def calculate_half_life(self, time, conc):
        """Calculate half-life using the terminal-phase method."""
        time = np.array(time)
        conc = np.array(conc)

        # remove zero and negative values
        valid_idx = conc > 0
        time = time[valid_idx]
        conc = conc[valid_idx]

        if len(time) < self.min_terminal_points:
            return {'t_half': np.nan, 'r_squared': np.nan, 'ke': np.nan,
                    'n_points': len(time), 'error': 'insufficient_points'}

        # check that time points aren't all identical (would break the regression)
        if len(np.unique(time)) < 2:
            return {'t_half': np.nan, 'r_squared': np.nan, 'ke': np.nan,
                    'n_points': len(time), 'error': 'identical_time_points'}

        # use the last 4-5 points
        n_points = min(5, len(time))
        terminal_time = time[-n_points:]
        terminal_conc = conc[-n_points:]

        # re-check that the terminal-phase time points actually vary
        if len(np.unique(terminal_time)) < 2:
            return {'t_half': np.nan, 'r_squared': np.nan, 'ke': np.nan,
                    'n_points': n_points, 'error': 'identical_terminal_time_points'}

        # log transformation
        ln_conc = np.log(terminal_conc)

        # linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(terminal_time, ln_conc)
        except ValueError:
            # catch any linregress error
            return {'t_half': np.nan, 'r_squared': np.nan, 'ke': np.nan,
                    'n_points': n_points, 'error': 'linregress_failed'}

        ke = -slope
        r_squared = r_value ** 2

        if ke <= 0:
            return {'t_half': np.nan, 'r_squared': r_squared, 'ke': ke,
                    'n_points': n_points, 'error': 'negative_ke'}

        t_half = 0.693 / ke

        # quality tier
        if r_squared >= 0.9:
            quality = 'high'
        elif r_squared >= 0.7:
            quality = 'medium'
        else:
            quality = 'low'

        return {
            't_half': t_half,
            'r_squared': r_squared,
            'ke': ke,
            'quality': quality,
            'n_points': n_points,
            'error': None
        }

    def calculate_auc(self, time, conc):
        """Calculate AUC using the trapezoidal rule."""
        time = np.array(time)
        conc = np.array(conc)

        # sort by time
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        conc = conc[sort_idx]

        # AUC 0-last
        auc_0_last = trapezoid(conc, time)

        # try extrapolating to infinity
        try:
            t_half_result = self.calculate_half_life(time, conc)
            if not np.isnan(t_half_result['ke']) and t_half_result['ke'] > 0:
                ke = t_half_result['ke']
                c_last = conc[-1]
                auc_extrapolated = c_last / ke
                auc_0_inf = auc_0_last + auc_extrapolated
                extrap_pct = (auc_extrapolated / auc_0_inf * 100) if auc_0_inf > 0 else np.nan
            else:
                auc_0_inf = np.nan
                extrap_pct = np.nan
        except Exception:
            auc_0_inf = np.nan
            extrap_pct = np.nan

        return {
            'auc_0_last': auc_0_last,
            'auc_0_inf': auc_0_inf,
            'auc_extrap_pct': extrap_pct
        }

    def calculate_cmax_tmax(self, time, conc):
        """Calculate Cmax and Tmax."""
        cmax_idx = np.argmax(conc)
        return {
            'cmax': conc[cmax_idx],
            'tmax': time[cmax_idx]
        }

    def calculate_cmin(self, time, conc, steady_state=False):
        """Calculate Cmin."""
        if steady_state:
            return conc[-1]  # steady state: last point
        else:
            return np.min(conc)  # single dose: minimum value


class DataProcessor:
    """Processes the extracted datapoint CSVs."""

    def __init__(self, extracted_images_dir, metadata_file):
        self.extracted_dir = Path(extracted_images_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.calculator = PKCalculator()
        self.identifier = SmartDataIdentifier()

        # build image_filename -> caption map
        self.caption_map = {}
        self.metadata_map = {}  # full metadata row map
        for _, row in self.metadata.iterrows():
            img_fn = row['image_filename']
            self.caption_map[img_fn] = row.get('caption', '')
            self.metadata_map[img_fn] = row

    def identify_time_column(self, df):
        """Identify the time column."""
        columns = df.columns.tolist()
        time_keywords = ['time', 'hour', 'h)', 'hr', 'day']

        for col in columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in time_keywords):
                return col
        return None

    def identify_concentration_columns(self, df, time_col):
        """Identify concentration columns (excluding the time column)."""
        conc_cols = []
        exclude_keywords = ['rna', 'hiv', 'viral', 'log', 'change']

        for col in df.columns:
            if col == time_col:
                continue

            col_lower = col.lower()

            # exclude non-concentration columns
            if any(ex in col_lower for ex in exclude_keywords):
                continue

            # check whether it's numeric
            try:
                pd.to_numeric(df[col], errors='coerce')
                conc_cols.append(col)
            except Exception:
                continue

        return conc_cols

    def identify_data_level(self, caption, n_curves):
        """Determine whether this is individual-subject or mean data."""
        caption_lower = caption.lower()

        # mean indicators
        mean_keywords = ['mean', 'median', 'average', '±', 'error bars']
        has_mean = any(kw in caption_lower for kw in mean_keywords)

        # individual indicators
        individual_keywords = ['individual', 'subject', 'each subject']
        has_individual = any(kw in caption_lower for kw in individual_keywords)

        if has_mean:
            return 'mean'
        elif has_individual or n_curves > 5:
            return 'individual'
        else:
            return 'unknown'

    def extract_n_subjects(self, caption):
        """Extract the number of subjects from the caption."""
        # matches "n = 23", "N=8", "(eight or nine)", etc.
        patterns = [
            r'n\s*=\s*(\d+)',
            r'N\s*=\s*(\d+)',
            r'\(n\s*=\s*(\d+)\)',
            r'(\d+)\s+subjects?',
            r'(\d+)\s+patients?'
        ]

        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def is_steady_state(self, caption):
        """Determine whether this is steady-state data."""
        caption_lower = caption.lower()
        ss_keywords = ['steady state', 'steady-state', 'day 14', 'day 7', 'multiple dose']
        return any(kw in caption_lower for kw in ss_keywords)

    def process_csv(self, csv_path, pmid):
        """Process a single CSV file."""
        results = []

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return [{
                'pmid': pmid,
                'csv_filename': csv_path.name,
                'error': f'failed_to_read: {e}',
                'is_pk_curve': False
            }]

        # infer image_filename from the CSV filename (strip the hash suffix
        # convention doesn't apply here -- just swap the extension)
        # e.g.: 9736567_p03_01_8080dec4e402360b.csv -> 9736567_p03_01_8080dec4e402360b.png
        image_filename = csv_path.stem + '.png'

        # get caption from metadata
        caption = self.caption_map.get(image_filename, "")

        # fall back to the captions file if metadata doesn't have it
        if not caption:
            # try captions.csv first
            caption_csv = csv_path.parent / 'captions.csv'
            caption_txt = csv_path.parent / 'captions.txt'

            if caption_csv.exists():
                try:
                    captions_df = pd.read_csv(caption_csv)
                    # assume CSV format: filename,caption
                    matching_row = captions_df[captions_df.iloc[:, 0].str.contains(csv_path.stem, na=False)]
                    if len(matching_row) > 0:
                        caption = str(matching_row.iloc[0, 1])  # second column is the caption
                except Exception:
                    pass

            # fall back to captions.txt
            if not caption and caption_txt.exists():
                try:
                    with open(caption_txt, 'r', encoding='utf-8') as f:
                        for line in f:
                            if csv_path.name in line:
                                caption = line.strip()
                                break
                except Exception:
                    pass

        # classify the data type using SmartDataIdentifier
        metadata_row = self.metadata_map.get(image_filename)

        data_type_result = self.identifier.identify_data_type(
            df,
            metadata_row=metadata_row,
            caption=caption
        )

        # check whether this should be processed at all
        if not data_type_result['should_process']:
            return [{
                'pmid': pmid,
                'csv_filename': csv_path.name,
                'image_filename': image_filename,
                'is_pk_curve': False,
                'data_type': data_type_result['data_type'],
                'reason': data_type_result['reason'],
                'caption': caption
            }]

        # identify columns
        time_col = self.identify_time_column(df)
        if time_col is None:
            return [{
                'pmid': pmid,
                'csv_filename': csv_path.name,
                'is_pk_curve': False,
                'error': 'no_time_column',
                'caption': caption
            }]

        conc_cols = self.identify_concentration_columns(df, time_col)
        if not conc_cols:
            return [{
                'pmid': pmid,
                'csv_filename': csv_path.name,
                'is_pk_curve': False,
                'error': 'no_concentration_column',
                'caption': caption
            }]

        # extract metadata
        data_level = self.identify_data_level(caption, len(conc_cols))
        n_subjects = self.extract_n_subjects(caption)
        steady_state = self.is_steady_state(caption)

        # extract time data
        time_data = pd.to_numeric(df[time_col], errors='coerce')

        # process each concentration column
        curve_count = 0  # used for subfigure numbering

        for conc_col in conc_cols:
            conc_data = pd.to_numeric(df[conc_col], errors='coerce')

            # drop NaN
            valid_idx = ~(time_data.isna() | conc_data.isna())
            time_vals = time_data[valid_idx].values
            conc_vals = conc_data[valid_idx].values

            if len(time_vals) < 3:
                continue

            # if this is individual-subject data, compute the mean instead
            if data_level == 'individual' and len(conc_cols) > 1:
                # this is a case of multiple individual curves --
                # only compute the mean once, on the first column
                if conc_col != conc_cols[0]:
                    continue  # skip the other individual curves

                # compute the mean across all curves
                all_conc_data = []
                for col in conc_cols:
                    col_data = pd.to_numeric(df[col], errors='coerce')
                    valid = ~(time_data.isna() | col_data.isna())
                    if valid.sum() > 0:
                        all_conc_data.append(col_data[valid].values)

                # make sure all curves share the same time points
                if len(set(len(c) for c in all_conc_data)) == 1:
                    conc_vals = np.mean(all_conc_data, axis=0)
                    dose_group = 'mean_of_individual_subjects'
                    subfigure_id = 'mean'
                else:
                    # time points don't match, just use the first curve
                    dose_group = conc_col
                    curve_count += 1
                    subfigure_id = f'curve_{curve_count}'
            else:
                dose_group = conc_col
                curve_count += 1
                subfigure_id = f'curve_{curve_count}'

            # build the subfigure-tagged image_filename
            # if there's only one curve, no suffix is added
            # if there are multiple curves, add a _curve_1, _curve_2, etc. suffix
            if len(conc_cols) == 1:
                image_filename_with_suffix = image_filename
            else:
                # e.g.: 9736567_p03_01_xxx.png -> 9736567_p03_01_xxx_curve_1.png
                base_name = image_filename.rsplit('.', 1)[0]
                ext = image_filename.rsplit('.', 1)[1] if '.' in image_filename else 'png'
                image_filename_with_suffix = f"{base_name}_{subfigure_id}.{ext}"

            # calculate PK parameters
            t_half_result = self.calculator.calculate_half_life(time_vals, conc_vals)
            auc_result = self.calculator.calculate_auc(time_vals, conc_vals)
            cmax_result = self.calculator.calculate_cmax_tmax(time_vals, conc_vals)
            cmin_val = self.calculator.calculate_cmin(time_vals, conc_vals, steady_state)

            # assemble the result
            result = {
                'pmid': pmid,
                'csv_filename': csv_path.name,
                'image_filename': image_filename,  # original image filename
                'image_filename_with_subfigure': image_filename_with_suffix,  # with subfigure suffix
                'subfigure_id': subfigure_id,  # curve_1, curve_2, mean, etc.
                'is_pk_curve': True,
                'data_type': data_type_result['data_type'],
                'data_confidence': data_type_result['confidence'],
                'dose_group': dose_group,
                'data_level': data_level,
                'n_subjects': n_subjects,
                'n_datapoints': len(time_vals),
                'time_range_min': time_vals.min(),
                'time_range_max': time_vals.max(),
                'steady_state': steady_state,

                # half-life
                'calc_t_half': t_half_result['t_half'],
                'calc_t_half_r_squared': t_half_result['r_squared'],
                'calc_t_half_quality': t_half_result.get('quality', 'unknown'),
                'calc_t_half_n_points': t_half_result['n_points'],
                'calc_ke': t_half_result['ke'],

                # AUC
                'calc_auc_0_last': auc_result['auc_0_last'],
                'calc_auc_0_inf': auc_result['auc_0_inf'],
                'calc_auc_extrap_pct': auc_result['auc_extrap_pct'],

                # Cmax, Tmax
                'calc_cmax': cmax_result['cmax'],
                'calc_tmax': cmax_result['tmax'],

                # Cmin
                'calc_cmin': cmin_val,

                # caption from metadata
                'caption': caption
            }

            results.append(result)

            # if this is individual curves collapsed to a mean, only output once
            if data_level == 'individual' and len(conc_cols) > 1:
                break

        return results

    def process_pmid_list(self, pmid_list):
        """Process a given list of PMIDs."""
        all_results = []

        print(f"\n{'='*80}")
        print(f"Processing {len(pmid_list)} PMIDs")
        print(f"{'='*80}\n")

        for i, pmid in enumerate(pmid_list, 1):
            pmid_str = str(pmid)
            pmid_dir = self.extracted_dir / pmid_str

            print(f"[{i}/{len(pmid_list)}] PMID {pmid_str}")

            if not pmid_dir.exists():
                print(f"  [WARN] Directory not found: {pmid_dir}")
                continue

            # find all CSV files
            csv_files = list(pmid_dir.glob('*.csv'))
            csv_files = [f for f in csv_files if 'caption' not in f.name.lower()]

            print(f"  Found {len(csv_files)} CSV files")

            for csv_file in csv_files:
                results = self.process_csv(csv_file, pmid)
                all_results.extend(results)

                # print results
                for result in results:
                    if result['is_pk_curve']:
                        print(f"    [OK] {csv_file.name}: {result['dose_group']}")
                        if not np.isnan(result['calc_t_half']):
                            print(f"      t1/2={result['calc_t_half']:.2f}h (R^2={result['calc_t_half_r_squared']:.3f})")
                    else:
                        print(f"    [SKIP] {csv_file.name}: {result.get('reason', result.get('error', 'unknown'))}")

            print()

        return pd.DataFrame(all_results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                     help="AutoPK root dir (default: $AUTOPK_BASE_DIR or '.')")
    ap.add_argument("--images-dir", default=None,
                     help="Directory of extracted digitized-CSV images (default: <base-dir>/extracted_images)")
    ap.add_argument("--metadata-file", default=None,
                     help="ML-ready metadata CSV with a pmid column (default: <base-dir>/ml_ready_data/all_pk_data_ml_ready.csv)")
    ap.add_argument("--output-dir", default=None,
                     help="Output directory (default: <base-dir>/pk_calculation_results)")
    ap.add_argument("--test", action="store_true",
                     help="Test mode: only process the first 5 PMIDs")
    args = ap.parse_args()

    base_dir = args.base_dir
    extracted_images_dir = args.images_dir or os.path.join(base_dir, "extracted_images")
    metadata_file = args.metadata_file or os.path.join(base_dir, "ml_ready_data", "all_pk_data_ml_ready.csv")
    output_dir = args.output_dir or os.path.join(base_dir, "pk_calculation_results")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PK Parameter Calculation (raw-CSV + type-classifier version)")
    print("=" * 80)

    # read metadata to get the PMID list
    metadata = pd.read_csv(metadata_file)
    unique_pmids = metadata['pmid'].unique()

    print(f"\nTotal PMIDs in metadata: {len(unique_pmids)}")

    if args.test:
        # test mode: first 5 PMIDs
        pmids_to_process = unique_pmids[:5]
        output_filename = "step1_calculated_pk_parameters_test5.csv"
        print("\n[TEST MODE] Processing first 5 PMIDs")
    else:
        # full run: all PMIDs
        pmids_to_process = unique_pmids
        output_filename = "step1_calculated_pk_parameters_all.csv"
        print(f"\n[FULL MODE] Processing all {len(unique_pmids)} PMIDs")

    print(f"PMIDs to process: {len(pmids_to_process)}")

    # initialize processor
    processor = DataProcessor(extracted_images_dir, metadata_file)

    # process data
    results_df = processor.process_pmid_list(pmids_to_process)

    # save results
    output_file = Path(output_dir) / output_filename
    results_df.to_csv(output_file, index=False)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal rows processed: {len(results_df)}")
    print(f"Valid PK curves: {results_df['is_pk_curve'].sum()}")

    pk_data = results_df[results_df['is_pk_curve'] == True]
    if len(pk_data) > 0:
        print("\nPK parameters calculated:")
        print(f"  t1/2 values: {pk_data['calc_t_half'].notna().sum()}")
        print(f"  AUC values: {pk_data['calc_auc_0_last'].notna().sum()}")
        print(f"  Cmax values: {pk_data['calc_cmax'].notna().sum()}")

        print("\nQuality distribution:")
        quality_counts = pk_data['calc_t_half_quality'].value_counts()
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count}")

        print("\nData types:")
        type_counts = results_df['data_type'].value_counts()
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count}")

    print(f"\nResults saved to: {output_file}")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    results = main()
