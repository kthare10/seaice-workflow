#!/usr/bin/env python3

"""
Harmonize pre-labeled ATL03 segment CSVs into the workflow's schema.

Takes a directory of already-labeled, already-segmented ATL03 CSV files (the
"IS2_Corrected_data" style products produced by the co-registration / labeling
step described in Iqrah et al., IPDPSW 2025) and rewrites them into the column
schema the rest of this workflow expects, so the download, preprocess, and
auto-label stages can be skipped entirely.

Input columns are mapped as follows:

    lat              <- lat
    lon              <- lon
    along_track_dist <- x_atc
    mean_h           <- h_cor_mean   (geoid/tide/DAC/MSS-corrected height)
    median_h         <- h_cor_med
    std_h            <- height_sd
    photon_count     <- N
    bg_rate          <- brate_mean
    beam             <- parsed from the file name (gt1l/gt1r/.../gt3r)
    granule          <- parsed from the file name
    label            <- label

The corrected heights (h_cor_*) are used rather than the raw ellipsoidal
heights (height_*), because the freeboard stage expects sea-surface-referenced
elevations. Raw heights are used as a fallback, with a warning, if the
corrected columns are absent.

The ``label`` column is carried through into the preprocessed output as well as
the training output, so it survives inference as ground truth and lets
visualize_results.py emit a confusion matrix and per-class accuracy.

Usage:
    python prepare_labeled_csv.py --input-dir . \
                                  --output atl03_preprocessed.csv \
                                  --labeled-output labeled_data.csv
"""

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Workflow schema -> candidate source columns, in order of preference
COLUMN_MAP = {
    'lat': ['lat'],
    'lon': ['lon'],
    'along_track_dist': ['x_atc', 'along_track_dist'],
    'mean_h': ['h_cor_mean', 'height_mean'],
    'median_h': ['h_cor_med', 'height_med'],
    'std_h': ['height_sd', 'h_cor_sd'],
    'photon_count': ['N', 'photon_count'],
    'bg_rate': ['brate_mean', 'bg_rate'],
    'label': ['label'],
}

# Columns whose preferred source is a corrected height; warn when falling back
CORRECTED_HEIGHT_COLUMNS = {'mean_h', 'median_h'}

OUTPUT_COLUMNS = ['lat', 'lon', 'along_track_dist', 'mean_h', 'median_h',
                  'std_h', 'photon_count', 'bg_rate', 'beam', 'granule', 'label']

BEAM_RE = re.compile(r'(gt[1-3][lr])', re.IGNORECASE)


def granule_key(path):
    """
    Derive a granule key from a labeled CSV file name.

    Strips the trailing ``_labeled...`` product suffix, so
    ``ATL03_20191104195311_05940510_T02CNA_gt1r_labeled_10m_done.csv`` becomes
    ``ATL03_20191104195311_05940510_T02CNA_gt1r``.

    Args:
        path: Path to the labeled CSV file

    Returns:
        Granule key string
    """
    return re.sub(r'_labeled.*$', '', Path(path).stem)


def beam_name(path):
    """
    Extract the beam name (e.g. gt1r) from a labeled CSV file name.

    Args:
        path: Path to the labeled CSV file

    Returns:
        Lowercase beam name, or 'unknown' if the file name has no beam token
    """
    match = BEAM_RE.search(Path(path).stem)
    return match.group(1).lower() if match else 'unknown'


def harmonize_file(path):
    """
    Read one labeled CSV and rewrite it into the workflow column schema.

    Args:
        path: Path to the labeled CSV file

    Returns:
        DataFrame with OUTPUT_COLUMNS

    Raises:
        ValueError: If a required source column is missing
    """
    import pandas as pd

    df = pd.read_csv(path)
    logger.info(f"  {Path(path).name}: {len(df):,} rows, {len(df.columns)} columns")

    out = pd.DataFrame()
    for target, candidates in COLUMN_MAP.items():
        source = next((c for c in candidates if c in df.columns), None)
        if source is None:
            raise ValueError(
                f"{Path(path).name}: no source column for '{target}' "
                f"(looked for {candidates})"
            )
        if target in CORRECTED_HEIGHT_COLUMNS and source != candidates[0]:
            logger.warning(
                f"  {Path(path).name}: '{candidates[0]}' not found, falling back "
                f"to '{source}' for {target}. Heights may not be "
                f"sea-surface-referenced, which will skew freeboard."
            )
        out[target] = df[source]

    out['beam'] = beam_name(path)
    out['granule'] = granule_key(path)

    return out[OUTPUT_COLUMNS]


def prepare_labeled_csv(input_dir, output_file, labeled_output):
    """
    Harmonize every labeled CSV in a directory into the workflow schema.

    Args:
        input_dir: Directory containing the labeled CSV files
        output_file: Output preprocessed CSV (features + truth label)
        labeled_output: Output training CSV (features + label)
    """
    import pandas as pd

    out_names = {Path(output_file).name, Path(labeled_output).name}
    inputs = sorted(
        p for p in Path(input_dir).glob("*.csv")
        if p.name not in out_names
    )
    if not inputs:
        logger.error(f"No labeled .csv files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Harmonizing {len(inputs)} labeled CSV file(s) from {input_dir}")

    frames = []
    for path in inputs:
        try:
            frames.append(harmonize_file(path))
        except Exception as e:
            logger.error(f"Failed to harmonize {path}: {e}")
            sys.exit(1)

    df = pd.concat(frames, ignore_index=True)

    # Drop rows the labeling step could not assign (label -1) or with no height
    before = len(df)
    df = df[df['label'] >= 0]
    df = df.dropna(subset=['mean_h', 'median_h', 'std_h', 'photon_count', 'bg_rate'])
    df = df.reset_index(drop=True)
    if len(df) < before:
        logger.info(f"Dropped {before - len(df):,} unlabeled/incomplete segments")

    df['label'] = df['label'].astype(int)

    # Sort within each granule so LSTM sequences follow the along-track order
    df = df.sort_values(['granule', 'beam', 'along_track_dist']).reset_index(drop=True)

    df.to_csv(output_file, index=False)
    df.to_csv(labeled_output, index=False)

    class_counts = df['label'].value_counts().sort_index()
    class_names = {0: 'thick_ice', 1: 'thin_ice', 2: 'open_water'}

    logger.info(f"Wrote {len(df):,} segments to {output_file} and {labeled_output}")

    print(f"\n{'='*70}")
    print("LABELED CSV PREPARATION COMPLETE")
    print(f"{'='*70}")
    print(f"Input files: {len(inputs)}")
    print(f"Granules: {df['granule'].nunique()}")
    print(f"Beams: {sorted(df['beam'].unique())}")
    print(f"Total segments: {len(df):,}")
    print("Class distribution:")
    for cls, count in class_counts.items():
        name = class_names.get(cls, f"class_{cls}")
        print(f"  {cls} ({name}): {count:,} ({100.0 * count / len(df):.1f}%)")
    print(f"Output: {output_file}, {labeled_output}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Harmonize pre-labeled ATL03 segment CSVs into the workflow schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Harmonize every labeled CSV staged into the job working directory
  %(prog)s --input-dir .

  # Harmonize a directory of labeled CSVs locally
  %(prog)s --input-dir data/IS2_Corrected_data --output atl03_preprocessed.csv
        """
    )

    parser.add_argument("--input-dir", type=str, default=".",
                        help="Directory of labeled ATL03 .csv files (default: .)")
    parser.add_argument("--output", type=str, default="atl03_preprocessed.csv",
                        help="Output preprocessed CSV (default: atl03_preprocessed.csv)")
    parser.add_argument("--labeled-output", type=str, default="labeled_data.csv",
                        help="Output training CSV (default: labeled_data.csv)")

    args = parser.parse_args()

    try:
        prepare_labeled_csv(args.input_dir, args.output, args.labeled_output)
        logger.info("Labeled CSV preparation completed successfully")
    except Exception as e:
        logger.error(f"Failed to prepare labeled CSV data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
